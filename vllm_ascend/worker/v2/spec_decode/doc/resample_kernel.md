# ResampleKernel

> 源码：`vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py:82`
> （kernel `_resample_kernel`，host 入口 `rejection_sample`）。
> 本算子位于 `worker/v2/spec_decode/` 而非 `vllm_ascend/ops/triton/`，故文档随源码目录落在
> `vllm_ascend/worker/v2/spec_decode/doc/`，与 `worker/v2/sample/doc/` 的处理方式一致。
>
> 本 kernel 内部调用 device function `_npu_gumbel_block_argmax`（`:165`），
> 其算子文档见 [npu_gumbel_block_argmax.md](npu_gumbel_block_argmax.md)。
>
> **不要与 `vllm_ascend/ops/triton/reject_sample.py` 混淆**：那是 model_runner **v1** 的拒绝采样，
> nightly 里的 `test_rejection_sample.py` 测的是那一套，与本算子无任何关系。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列加速卡产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

> 无 device adaptor 分支，A2/A3/950 走完全相同的一份 kernel，无下发参数差异。
> 310P 上 `HAS_TRITON` 为 False，且 model_runner v2 未在 310P 上启用，不经过本算子。
>
> **NPU 相对上游的能力缺口**（影响本 kernel 的下发与入口校验）：
> - `resampled_local_max` 上游是 FLOAT64，NPU 无 float64，改用 FLOAT32；
> - `rejection_sample` 入口对 `use_fp64=True` 与 `synthetic_conditional_rates is not None`
>   直接 `NotImplementedError`，**不做静默降级**（两者都依赖 `tl_rand64`）；
> - 噪声侧的差异见 [npu_gumbel_block_argmax.md](npu_gumbel_block_argmax.md)。

## 功能说明

- API 功能：投机推理（speculative decoding）拒绝采样的**最后一步**。
  `_probabilistic_rejection_kernel` 判定出每个请求在第几步被拒（`num_sampled`），
  本 kernel 负责为那个位置**重新采一个 token**：

  - 若被拒位置正好是请求的最后一个 logit（bonus token），从 target 分布本身采；
  - 否则从**残差分布** $\max(0,\ p - q)$ 采（$p$ = target，$q$ = draft），
    这是标准拒绝采样保持 target 分布无偏的必要条件；
  - 贪婪请求（`temperature == 0`）且非 bonus 时**直接返回不写**，
    因为 `_probabilistic_rejection_kernel` 已经把 target argmax 写进 `sampled` 了。

  本 kernel 只产出**块内**的 (max, argmax)，跨块归约由上游实现的
  `_insert_resampled_kernel` 完成。

- 计算公式：

    设 $\ell^{t}$ 为 target logits，$\ell^{d}$ 为 draft logits，
    $Z_t,\ Z_d$ 为两侧的 logsumexp（由 `_probabilistic_rejection_kernel` 预先算好并传入）。
    残差 logits 为：

    $$
    \text{residual}_v =
    \begin{cases}
      \ell^{t}_v, & \text{is\_bonus} \\[4pt]
      (\ell^{t}_v - Z_t) + \log\!\left(1 - \text{ratio}_v\right), & \text{HAS\_DRAFT\_LOGITS},\ \text{ratio}_v < 1 \\[4pt]
      -\infty, & \text{HAS\_DRAFT\_LOGITS},\ \text{ratio}_v \ge 1 \\[4pt]
      \ell^{t}_v \cdot [\,v \ne \text{rejected}\,] + (-\infty)\cdot[\,v = \text{rejected}\,], & \text{otherwise（one-hot draft）}
    \end{cases}
    $$

    其中 $\text{ratio}_v = \exp\big((\ell^{d}_v - Z_d) - (\ell^{t}_v - Z_t)\big) = q_v / p_v$。

    随后交给 `_npu_gumbel_block_argmax` 做块内 Gumbel-max 采样（`APPLY_TEMPERATURE=False`），
    并把块内相对下标补成全局 token id：

    $$\text{token\_id} = \text{block\_idx} \times \text{BLOCK\_SIZE} + \text{idx}$$

- 调用链：

    ```
    RejectionSampler.__call__                    vllm/v1/worker/gpu/spec_decode/rejection_sampler.py（上游）
      └─ rejection_sampler.rejection_sample      被 patch 替换为本文件的实现
           └─ vllm_ascend/patch/worker/patch_v2/patch_triton.py:37-38
                └─ rejection_sample              vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py:325
                     ├─ _compute_block_stats_kernel      上游 _compute_local_logits_stats_kernel（原样复用）
                     ├─ _probabilistic_rejection_kernel  同文件:181，产出 num_sampled / 两个 logsumexp
                     ├─ _resample_kernel                 同文件:82  ← 本算子
                     │    └─ _npu_gumbel_block_argmax    同文件:34
                     └─ _insert_resampled_kernel         上游（原样复用），跨块归约并写回 sampled
    ```

- 任务划分：二维 grid `(num_reqs, resample_num_blocks)`。
  - `axis=0` 一个 program 对应一个请求（注意是 `req_idx`，不是 request state 的行号 `req_state_idx`）；
  - `axis=1` 沿词表切块，块宽 `RESAMPLE_BLOCK_SIZE = 1024`，
    `resample_num_blocks = cdiv(vocab_size, 1024)`。

  不做核间 round-robin，不读 AI Core 数量，不依赖 `init_device_properties_triton()`。

## 参数说明

### Python 接口 `rejection_sample`

只列与本算子直接相关的参数；其余参数服务于前两个 kernel。

| 参数名 |输入/输出/属性| 描述 | 数据类型 |数据格式|
|-------|------------|------|---------|-----|
|target_logits|输入|target 模型的 logits，shape 为 [num_logits, V]。已经过 `apply_sampling_params`（penalty / top-k / 温度等），本 kernel 不再做温度缩放。|FLOAT32|ND|
|draft_logits|输入|draft 模型的 logits，shape 为 [max_num_reqs, num_speculative_steps, V]，可为 None。为 None 时 wrapper 造一个 `new_empty(1,1,1)` 的哑张量占位，并置 `HAS_DRAFT_LOGITS=False`，kernel 不会读它。|FLOAT32|ND|
|draft_sampled|输入|draft 采出的 token 序列，shape 为 [num_logits]。**与 logits 错开一位**，见约束说明。|INT32|ND|
|cu_num_logits|输入|每个请求 logits 的前缀和，shape 为 [num_reqs + 1]。|INT32|ND|
|pos|输入|每个 logit 的**全局**位置，shape 为 [num_logits]，用作 Gumbel 噪声的 philox key。|INT64|ND|
|idx_mapping|输入|`req_idx -> req_state_idx`，shape 为 [num_reqs]。本 kernel 不用，由 `_probabilistic_rejection_kernel` 使用。|INT32|ND|
|expanded_idx_mapping|输入|`token_idx -> req_state_idx`，shape 为 [num_logits]，同一请求的多个 logit 映射到同一行，**含重复**。|INT32|ND|
|temperature|输入|按 request state 行号索引的温度，shape 为 [max_num_reqs]。0 表示贪婪。|FLOAT32|ND|
|seed|输入|按 request state 行号索引的随机种子，shape 为 [max_num_reqs]。|INT64|ND|
|num_speculative_steps|属性|投机步数，决定输出 `sampled` 的列数。host 标量。|INT32|-|
|use_fp64|属性|必须为 False，否则抛 `NotImplementedError`。|BOOL|-|
|synthetic_conditional_rates|属性|必须为 None，否则抛 `NotImplementedError`。|FLOAT32|ND|
|use_block_verification|属性|**接受但未实现**，传 True 不报错也不生效。|BOOL|-|
|sampled|输出|采样结果，shape 为 [num_reqs, num_speculative_steps + 1]。只有 `[:num_sampled]` 有效。|INT64|ND|
|num_sampled|输出|每个请求实际产出的 token 数，shape 为 [num_reqs]。**注意它在 `_insert_resampled_kernel` 里被 +1 覆盖写**，返回给调用方的值 = 接受的 draft token 数 + 1。|INT32|ND|

### Kernel 接口 `_resample_kernel`

| 参数名 |输入/输出/属性| 描述 | 数据类型 |
|-------|------------|------|---------|
|resampled_local_argmax_ptr|输出|块内 argmax 的**全局** token id，shape 为 [num_reqs, num_blocks]。由 wrapper `new_empty` 分配，**未初始化**。|INT64|
|resampled_local_argmax_stride|属性|运行时参数，取 `.stride(0)`。|INT32|
|resampled_local_max_ptr|输出|块内最大值（含 Gumbel 噪声），shape 为 [num_reqs, num_blocks]。NPU 上是 FLOAT32（上游为 FLOAT64）。同样未初始化。|FLOAT32|
|resampled_local_max_stride|属性|运行时参数。|INT32|
|target_logits_ptr / target_logits_stride|输入|同 wrapper。|FLOAT32 / INT32|
|target_rejected_logsumexp_ptr|输入|按 `req_idx` 索引，shape 为 [num_reqs]。由 `_probabilistic_rejection_kernel` 写出，**只在 `HAS_DRAFT_LOGITS` 且非 bonus 时被读**。|FLOAT32|
|draft_logits_ptr / draft_logits_stride_0 / draft_logits_stride_1|输入|按 `[req_state_idx, resample_idx, :]` 索引——**行号是 request state 行号，不是 `req_idx`**。|FLOAT32 / INT32|
|draft_rejected_logsumexp_ptr|输入|按 `req_idx` 索引，shape 为 [num_reqs]。|FLOAT32|
|rejected_step_ptr|输入|即 wrapper 里的 `num_sampled`，请求内被拒的步号，shape 为 [num_reqs]。|INT32|
|cu_num_logits_ptr|输入|同 wrapper。|INT32|
|expanded_idx_mapping_ptr|输入|同 wrapper。|INT32|
|draft_sampled_ptr|输入|同 wrapper；访问下标是 `resample_token_idx + 1`。|INT32|
|temp_ptr / seed_ptr|输入|按 `req_state_idx` 索引。|FLOAT32 / INT64|
|pos_ptr|输入|按 `resample_token_idx` 索引。|INT64|
|vocab_size|属性|运行时参数（无 `do_not_specialize`），词表大小。|INT32|
|BLOCK_SIZE|属性|`tl.constexpr`，词表分块宽度，wrapper 硬编码为 1024。|INT32|
|HAS_DRAFT_LOGITS|属性|`tl.constexpr`，选择残差分支。|BOOL|

## 约束说明

- 该接口仅用于推理采样路径，无反向。

- **`draft_sampled` 与 logits 错开一位。** kernel 读的是
  `draft_sampled_ptr + resample_token_idx + 1`，因为 `draft_sampled` 取自
  `input_batch.input_ids[logits_indices]`：第 $i$ 个 logit 对应的 draft token
  存在第 $i+1$ 槽。写基准时按 $i$ 取会静默排除掉错误的 token——被拒的那个仍然可被采回，
  且不会报任何错。
  相应地，请求最后一个 logit（`end_idx - 1`）的 `+1` 会越界读到下一个请求的槽，
  **该分支被 `is_bonus` 短路掉，永远不会执行**；若将来把 bonus 判定改坏，
  最后一个请求会读到张量尾部之外。

- **`temp == 0 且非 bonus` 时 kernel 直接返回，两个输出保持未初始化。**
  这不是优化，是与下游 `_insert_resampled_kernel` 的**契约**：后者在完全相同的条件下
  同样提前返回（`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:839`），
  改用 `_probabilistic_rejection_kernel` 已经写进 `sampled` 的 target argmax。
  两边的条件必须同时改，只改一边会让 `_insert_resampled_kernel` 读到
  `new_empty` 的脏数据，表现为贪婪请求偶发采出随机 token。

- **三套下标不能混。**
  `req_idx`（grid 维，索引两个 logsumexp / `rejected_step` / 输出行）、
  `req_state_idx`（`expanded_idx_mapping` 的取值，索引 `temp` / `seed` / **`draft_logits` 的第 0 维**）、
  `resample_token_idx`（`cu_num_logits[req_idx] + rejected_step[req_idx]`，索引
  `target_logits` / `pos` / `draft_sampled`）。
  三者在单请求、且 `idx_mapping` 恰为 `arange` 时数值相同，用例必须刻意打乱才能测出写反。

- **`_npu_gumbel_block_argmax` 返回块内相对下标，本 kernel 负责补 `block_idx * BLOCK_SIZE`**
  （`:172`）。漏掉会让所有请求都采到词表前 1024 个 token。

- **`-inf` 是排除机制。** 三条残差分支都用 `-inf` 标记不可采的 token，
  依赖 device function 内 `-inf + 有限噪声 == -inf`。
  若换成一个很小的有限值，一旦某个 block 内**全部**是被排除的 token，
  行为就会从"返回 -inf"变成"返回一个具体 token"。

- **`HAS_DRAFT_LOGITS` 分支里 `ratio` 可能是 `nan`。** 词表尾块的填充位置
  target 与 draft 都是 `-inf`，`(-inf) - (-inf) = nan`，`exp(nan) = nan`，
  而 `nan < 1.0` 为 False，因此落到 `-inf` 分支——**结果恰好正确，但这是巧合而非设计**。
  改写这段比较（例如反转判断方向）需要重新确认 nan 的走向。

- **`vocab_size` 是运行时参数，`BLOCK_SIZE` 是 `tl.constexpr`。**
  换词表不会触发重编译，改块宽会。wrapper 里 `RESAMPLE_BLOCK_SIZE = 1024`
  与 `VOCAB_BLOCK_SIZE = 8192` 是两个不同的块宽，分别服务 resample 与 block-stats，不要合并。

- **入口的两处 `NotImplementedError` 是刻意的。** `use_fp64=True` 与
  `synthetic_conditional_rates is not None` 都直接抛错而非降级：#13470 之前
  `_probabilistic_rejection_kernel` 就是静默用 `u = 0.0` 全盘接受，
  导致 acceptance rate 恒为 1.0 且无人察觉。

- **噪声侧的约束**（`pos` 必须放得进 int32、`(seed, pos)` 是唯一随机源、
  同请求内不同 logit 必须有不同 `pos`、尾部量化差异）见
  [npu_gumbel_block_argmax.md](npu_gumbel_block_argmax.md) 的「约束说明」。

## 测试说明

数值精度用例：`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_resample_kernel.py`

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_resample_kernel.py
```

kernel 内部调用的 device function 会引入 Gumbel 噪声，PyTorch 没有等价的 philox，
故用例内提供一个**测试专用**探针 kernel `_gumbel_noise_probe_kernel`：逐行复刻
`_npu_gumbel_block_argmax` 的 RNG 段，**重放同一条噪声流**供基准消费。
残差公式、mask、`-inf`、块内归约、提前返回等**本 kernel 自己拥有的语义**，
全部由 PyTorch fp32 独立算出。

> 该探针在 `test_npu_gumbel_block_argmax.py` 里有一份**同样的副本**，
> 两份必须与 `_npu_gumbel_block_argmax` 的 RNG 段逐字一致；
> 改动 device function 的噪声实现时，两个用例文件都要同步。
> 噪声流本身的正确性（分布、可复现性）由那个文件的用例负责，本文件不重复验证。

| 输出 | 比对方式 | 容差 |
|---|---|---|
| `resampled_local_max` | `torch.testing.assert_close` | rtol = atol = 1e-5 |
| `resampled_local_argmax` | 逐元素相等；仅当两个候选分数在上述容差内相等时允许换位 | 见左 |
| 提前返回的输出槽 | 与投毒哨兵值 `==` | 精确 |
| `rejection_sample` 端到端 token id | `==`（整型精确） | 无 |

索引比对允许"精确并列时换位"，是因为 kernel 与基准的 fp32 归约顺序不同；
换位只有在被选中位置的分数**严格劣于**基准最大值时才判失败，
并且额外断言下标没有越出所属 block、没有落进词表填充区。

覆盖范围：

- 三条残差分支：bonus（`temp == 0` 与 `temp != 0` 各一条）、
  `HAS_DRAFT_LOGITS=True` 的 $\log(1-\text{ratio})$、one-hot draft 的单点排除；
  后两条都加了"守护守护者"断言（前者要求 `ratio < 1` 与 `ratio >= 1` 同时出现，
  后者把被排除的 token 抬成本 block 的绝对最大值，并断言它没有被采回）；
- `temp == 0 且 is_bonus`：唯一无噪声路径，与 target logits 的逐块 argmax **精确**比对；
- `temp == 0 且非 bonus` 的提前返回：输出先投毒再断言原样，
  覆盖与 `_insert_resampled_kernel` 的契约；
- 混合批：5 个请求 × 3 个词表块（均非 2 的幂且互不相等），贪婪/采样、bonus/非 bonus 交错，
  每请求 logits 数不等，`req_state_idx` 打乱且不连续——防止三套下标写反，
  同时验证提前返回的请求不影响相邻请求的写入；
- 词表尾块非对齐（`V` 取 `1024k + 137 / 3 / 91 / 233 / 401 / 17 / 7`），
  并断言采出的 token id 恒小于 `vocab_size`；
- `HAS_DRAFT_LOGITS` 的真假两侧；
- 可复现性：同一批次两次 launch 逐位一致；
- 端到端：贪婪批经 `rejection_sample` 全流程（含上游的 block-stats 与 insert 两个 kernel），
  用生产 launch 配置（`BLOCK_SIZE=1024`、`V = 2 * 8192 + 37`）跑一遍，
  刻意构造出接受长度 0..num_spec 全谱，同时覆盖"提前返回"与"bonus 重采"两条分支，
  并断言批次里两条分支都确实出现。

未覆盖及原因：

- **`_probabilistic_rejection_kernel` 的概率接受判定（#13470 的主体改动）**：
  它是本文件的另一个 kernel，不在本算子范围内；端到端用例只跑 `temp == 0` 的贪婪路径，
  不触发它的随机分支。概率路径需要一条独立的统计用例，建议随该 kernel 的整改一并补。
- **Gumbel 噪声的分布与可复现性**：属于 device function 的职责，
  在 `test_npu_gumbel_block_argmax.py` 中覆盖，本文件不重复。
- **`use_fp64=True` / `synthetic_conditional_rates` 的 `NotImplementedError`**：
  纯 host 侧参数校验，不涉及数值，且 `tests/ut/` 的 CPU runner 会 mock 掉 `torch_npu`，
  放在 nightly 里跑一条只为触发 `raise` 的用例不划算。
- **`use_block_verification=True`**：参数被接受但未实现，无可比对的语义。
- **性能特征**：nightly 用例不做性能门禁。

## 变更记录

| PR | 说明 |
|---|---|
|[#9155](https://github.com/vllm-project/vllm-ascend/pull/9155)|main2main 0514 批量同步，把 `rejection_sampler_utils.py` 整个文件带入，本 kernel 自此存在；未附带任何数值用例|
|[#9238](https://github.com/vllm-project/vllm-ascend/pull/9238) / [#9399](https://github.com/vllm-project/vllm-ascend/pull/9399) / [#10454](https://github.com/vllm-project/vllm-ascend/pull/10454) / [#11227](https://github.com/vllm-project/vllm-ascend/pull/11227) / [#11709](https://github.com/vllm-project/vllm-ascend/pull/11709)|后续 main2main 同步，随上游改动跟进签名与调用方式|
|[#13470](https://github.com/vllm-project/vllm-ascend/pull/13470)|启用 NPU 上的概率拒绝采样。本 kernel 未改动，但它的上游 `_probabilistic_rejection_kernel` 不再恒 `u = 0.0`，本 kernel 的 `rejected_step` 输入自此才有真正的随机性；同时在 `rejection_sample` 入口对 `use_fp64` / `synthetic_conditional_rates` 显式抛错|
