# NpuGumbelBlockArgmax

> 源码：`vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py:34`
> （`@triton.jit` **device function** `_npu_gumbel_block_argmax`）。
> 本算子位于 `worker/v2/spec_decode/` 而非 `vllm_ascend/ops/triton/`，故文档随源码目录落在
> `vllm_ascend/worker/v2/spec_decode/doc/`，与 `worker/v2/sample/doc/` 的处理方式一致。
>
> 唯一调用方是同文件的 `_resample_kernel`（`:165`），其算子文档见
> [resample_kernel.md](resample_kernel.md)。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列加速卡产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

> 无 device adaptor 分支，A2/A3/950 走完全相同的一份实现。
> 310P 上 `HAS_TRITON` 为 False，且 model_runner v2 未在 310P 上启用，不经过本算子。
>
> **本算子是上游 `vllm/v1/worker/gpu/sample/gumbel.py:gumbel_block_argmax` 的 NPU 改写版**，
> 差异由以下 NPU 能力缺口造成：
>
> | 上游 | NPU 版 | 原因 |
> |---|---|---|
> | `tl_rand64`（float64 philox） | `tl.rand`（fp32） | NPU Triton 无 float64 |
> | `pos` 原样传入 | `pos.to(tl.int32)` | Ascend vector core 的 `umulhi` 不支持 uint64 |
> | `USE_FP64` constexpr | 无该参数 | 同上 |
> | `PER_TOKEN_COL` constexpr | 无该参数 | 上游后加，NPU 侧未跟进 |
> | `is_valid_req = req_state_idx >= 0` 掩码 | **无** | 见「约束说明」 |
> | `-log(-log1p(-u))` | `-log(-log(u + 1e-20) + 1e-20)` | 见「约束说明」 |

## 功能说明

- API 功能：Gumbel-max 采样的**块内**部分。给定一个已经 load 好的 logits 数据块，
  按请求的温度决定是否加 Gumbel 噪声，返回该块内的最大值与下标。
  跨块归约由调用方（或下游 kernel）负责。

  它同时承担一个副作用：把**温度缩放之后、加噪之前**的 logits 写入可选的
  `processed_logits` 缓冲，供 logprobs 侧使用。

- 计算公式：

    温度缩放（仅当 `APPLY_TEMPERATURE` 且 $T \ne 0$）：

    $$\ell'_v = \ell_v / T$$

    Gumbel 噪声（仅当 $T \ne 0$）：

    $$
    g_v = -\log\big(-\log(u_v + 10^{-20}) + 10^{-20}\big),\quad
    u_v = \text{tl.rand}\big(\text{randint}(\text{seed},\ \text{pos}),\ v\big)
    $$

    块内归约：

    $$
    \text{value},\ \text{idx} = \max_{v \in \text{block}} \tilde{\ell}_v,\ \arg\max_{v \in \text{block}} \tilde{\ell}_v,
    \qquad
    \tilde{\ell}_v =
    \begin{cases}
      \ell'_v + g_v, & T \ne 0 \wedge \text{mask}_v \\
      -\infty, & T \ne 0 \wedge \neg\text{mask}_v \\
      \ell'_v, & T = 0
    \end{cases}
    $$

    $T = 0$ 时噪声整体关闭，退化为纯 argmax（贪婪采样）。

- 调用链：

    ```
    rejection_sample                       vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py:325
      └─ _resample_kernel                  同文件:82（唯一调用方）
           └─ _npu_gumbel_block_argmax     同文件:34，调用点在:165
    ```

    上游 sampler 侧的 `gumbel_block_argmax` 调用点**未被 patch 替换**，
    所以本算子只在 MRV2 投机推理的重采样路径上生效。

- 任务划分：**本算子自身不划分任务**。它没有 `tl.program_id`，
  `logits` / `block` / `mask` / `token_idx` 全部由调用方计算好传入，
  grid 由调用方决定（`_resample_kernel` 用 `(num_reqs, cdiv(vocab_size, 1024))`）。

## 参数说明

### Device function 接口 `_npu_gumbel_block_argmax`

**不是 kernel，不能从 host launch**：无 `tl.program_id`，`kernel[(grid,)](...)` 语法对它不成立。

| 参数名 |输入/输出/属性| 描述 | 数据类型 |
|-------|------------|------|---------|
|logits|输入|一个 block 的 logits **值**（不是指针），由调用方 `tl.load` 好。|FLOAT32|
|block|输入|该 block 的全局词表下标向量。**同时充当 philox 的 offset**，所以不同 block 的噪声天然不同。|INT32|
|mask|输入|`block < vocab_size`，标记块内哪些位置是有效词表位置。|BOOL|
|token_idx|输入|logit 的全局下标，用于索引 `expanded_idx_mapping_ptr` 与 `pos_ptr`。|INT32|
|expanded_idx_mapping_ptr|输入|`token_idx -> req_state_idx`，shape 为 [num_logits]。|INT32|
|temp_ptr|输入|按 `req_state_idx` 索引的温度，shape 为 [max_num_reqs]。0 表示贪婪。|FLOAT32|
|seeds_ptr|输入|按 `req_state_idx` 索引的随机种子，shape 为 [max_num_reqs]。|INT64|
|pos_ptr|输入|按 `token_idx` 索引的全局位置，shape 为 [num_logits]，作为 philox 的 key。|INT64|
|processed_logits_ptr|输入/输出|可选侧输出，写入温度缩放**之后、加噪之前**的 logits；传 `None` 关闭该分支。**行号是 `req_state_idx`，不是 `token_idx`**。|FLOAT32|
|processed_logits_stride|属性|`processed_logits` 的行 stride；`processed_logits_ptr` 为 `None` 时被忽略（`_resample_kernel` 传 0）。|INT32|
|processed_logits_col_ptr|输入|可选，指向一个**标量**列号；为 `None` 时列号取 0。写入偏移为 `req_state_idx * stride + col * vocab_size + block`。|INT32|
|vocab_size|属性|运行时参数，**仅**参与 `processed_logits` 的列偏移计算，不参与 mask（mask 由调用方给）。|INT32|
|APPLY_TEMPERATURE|属性|`tl.constexpr`。True 且 `temp != 0` 时把 logits 除以温度。`_resample_kernel` **恒传 False**（温度已在上游 `apply_sampling_params` 施加）。|BOOL|
|返回值 value|输出|块内最大值（含噪声）。|FLOAT32|
|返回值 idx|输出|**块内相对下标**，调用方需自行加 `block_idx * BLOCK_SIZE`。|INT32|

> 三个 `None` 敏感参数（`processed_logits_ptr` / `processed_logits_col_ptr`）走的是
> Triton 的编译期 `is not None` 判断，传 `None` 时对应分支根本不会生成代码。
> 因此"传 None"与"传指针"是两份不同的编译产物，测试必须两种形态都跑。

## 约束说明

- 该接口仅用于推理采样路径，无反向。

- **不能独立下发。** 无 `tl.program_id`，`logits` 是值不是指针。要单独验证它，
  必须现写一个 `@triton.jit` kernel 薄壳把 `program_id` / `tl.load` 补上——
  用例里的 `_gumbel_probe_kernel` 就是干这个的。

- **返回的是块内相对下标。** 调用方必须自己补 `block_idx * BLOCK_SIZE`。
  `_resample_kernel:172` 做了这件事；漏掉会让所有请求都采到词表前 `BLOCK_SIZE` 个 token，
  且不会报任何错。

- **`processed_logits` 的行号是 `req_state_idx`，不是 `token_idx`。**
  与它相邻的 `pos_ptr` / `expanded_idx_mapping_ptr` 用的是 `token_idx`，
  两者在单请求场景下数值相同，写反测不出来。

- **`vocab_size` 只影响 `processed_logits` 的列偏移。** 屏蔽越界位置靠调用方传进来的
  `mask`；本函数不会自己再算一次 `block < vocab_size`。若调用方给的 mask 与
  `vocab_size` 不一致，`processed_logits` 会写到错误的列而块内归约仍然"正确"。

- **`temp != 0` 判断出现两次，语义不同。**
  第一次（`temp != 0.0 and APPLY_TEMPERATURE`）决定要不要除温度，
  第二次（`temp != 0.0`）决定要不要加噪声。**第二次没有 `APPLY_TEMPERATURE` 保护**，
  所以 `APPLY_TEMPERATURE=False` 只关掉缩放、不关掉噪声——`_resample_kernel` 依赖的正是这一点。

- **`-inf` 是排除机制，依赖 `-inf + 有限噪声 == -inf`。**
  `tl.where(mask, logits + gumbel_noise, -inf)` 把无效位置压成 `-inf`；
  调用方也用 `-inf` 标记"不可采"的 token（见 `_resample_kernel` 的残差分支）。
  若把 `-inf` 换成一个很小的有限值，一旦某个 block 内**全部**是被排除的 token，
  行为就会从"返回 -inf"变成"返回一个具体 token"。

- **上游有、NPU 版没有的 `req_state_idx < 0` 保护。**
  上游 `gumbel_block_argmax` 用 `is_valid_req = req_state_idx >= 0` 给
  `temp` / `seed` 的加载和 `processed_logits` 的写入加了 mask，用于处理 padding 请求；
  NPU 版**没有移植这段**，同时也没有 `.to(tl.int64)`。
  当前唯一调用方不会产生负行号，但若将来接入会填负值的 padding 路径，会变成负偏移访存。

- **Gumbel 噪声的实现与上游不同，尾部分辨率更差。**
  NPU 版用 `-log(-log(u + 1e-20) + 1e-20)`，上游 fp32 路径用 `-log(-log1p(-u))`。
  上游注释明确指出：决定 argmax 胜负的是噪声的**大值尾部**，朴素写法把它压在 $u \to 1$ 一侧，
  而 fp32 在该处的间隔约 $2^{-24}$，导致噪声被硬顶在 $\approx 16.6$ 并被粗量化；
  改用 `log1p(-u)` 才能把尾部挪到 $u \to 0$ 这个分辨率充足的区间。
  NPU 版正是那个朴素写法。实测 argmax 频率仍与 softmax 相符（见「测试说明」），
  但极低概率 token 的采样保真度弱于上游。**这是已知差异，不是 bug。**
  两处 `+ 1e-20` 在 fp32 下对 `tl.rand` 的最小非零输出（$\approx 2.3\times10^{-10}$）
  是无效操作，只在 `u == 0` 时起兜底作用。

- **`pos` 必须能放进 int32。** 函数显式 `.to(tl.int32)`。
  序列长度超过 $2^{31}$ 时噪声流会回绕，实际不可达。

- **`(seed, pos)` 之外没有任何随机源。** 同一对输入必然给出同一结果，
  这是 `SamplingParams.seed` 可复现的前提；也意味着**同一请求内不同 logit 必须有不同的 `pos`**，
  否则它们会共享噪声。`block` 参与 offset，所以同一 token 的不同块噪声不同。

## 测试说明

数值精度用例：`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_npu_gumbel_block_argmax.py`

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_npu_gumbel_block_argmax.py
```

因为本算子无法从 host 下发，用例内提供了两个**测试专用**探针 kernel：

- `_gumbel_probe_kernel`：可 launch 的薄壳，只做 `tl.program_id` + `tl.load`，
  按 `PROCESSED_MODE` 复刻三种调用形态（不带 `processed_logits` / 隐式列 0 / 显式列号），
  并把 `APPLY_TEMPERATURE` 的真假两侧都跑到；
- `_gumbel_noise_probe_kernel`：逐行复刻 RNG 段，**重放同一条 philox 噪声流**，供基准消费。

基准分工：噪声本身由 Triton 侧提供（PyTorch 无等价 philox），
其余（温度缩放、`processed_logits` 写入、mask、`-inf`、块内归约）全部由 PyTorch fp32 独立算出。
噪声流本身的正确性由一条**独立的统计用例**兜底，不依赖上述重放。

| 输出 | 比对方式 | 容差 |
|---|---|---|
| value | `torch.testing.assert_close` | rtol = atol = 1e-5 |
| idx | 逐元素相等；仅当两个候选分数在上述容差内相等时允许换位 | 见左 |
| `processed_logits` | `torch.testing.assert_close` | rtol = atol = 1e-5 |
| argmax 频率分布 | 与 `softmax(logits)` 比 | atol = 0.02（16384 抽样下约 5σ） |

索引比对允许"精确并列时换位"，是因为 kernel 与基准的 fp32 归约顺序不同；
换位只有在被选中位置的分数**严格劣于**基准最大值时才判失败，
并且额外断言下标没有越出所属 block、没有落进词表填充区。

覆盖范围：

- `temp == 0`（噪声整体关闭，纯 argmax，精确比对）与 `temp != 0`（Gumbel-max）两条路径；
- `APPLY_TEMPERATURE` 的真假两侧——True 侧从 `_resample_kernel` **不可达**，
  只有薄壳能测到，但它是函数契约的一部分，也是上游 sampler 的用法；
- `processed_logits` 三种形态：关闭（`None`，与生产调用点一致）、隐式列 0、显式列号；
  并断言未被写的行/列保持 `nan`，即 mask 没有越界写、列偏移没有算错；
- `processed_logits` 的行号确实是 `req_state_idx`：`expanded_idx_mapping` 取随机排列，
  与 `token_idx` 刻意不等；
- 词表尾块非对齐（`V` 取 `1024k + 37 / 11 / 5`），断言 idx 恒小于 `vocab_size`；
- 噪声确实改变了至少一个块的胜者（守护守护者），否则该用例会退化成贪婪用例；
- 噪声分布：16384 次抽样，8 类，比对 argmax 频率与 `softmax(logits)`；
- 可复现性：同一 `(seed, pos)` 两次 launch 逐位一致，`pos` 变化则结果变化。

未覆盖及原因：

- **Gumbel 噪声尾部的量化差异**：属于与上游的已知实现差异（见「约束说明」），
  把当前行为写进断言等于把它固化为期望值；统计用例只保证一阶分布正确。
- **`req_state_idx < 0` 的 padding 请求**：NPU 版未移植上游保护，当前调用方也不会产生负行号；
  写成用例等于把越界访存固化为期望行为，故只在「约束说明」中记录。
- **`PER_TOKEN_COL`**：上游有、NPU 版无此参数，无可测对象。
- **性能特征**：nightly 用例不做性能门禁。

## 变更记录

| PR | 说明 |
|---|---|
|[#9155](https://github.com/vllm-project/vllm-ascend/pull/9155)|main2main 0514 批量同步，把 `rejection_sampler_utils.py` 整个文件带入，本函数自此存在；未附带任何数值用例|
|[#9238](https://github.com/vllm-project/vllm-ascend/pull/9238) / [#9399](https://github.com/vllm-project/vllm-ascend/pull/9399) / [#10454](https://github.com/vllm-project/vllm-ascend/pull/10454) / [#11227](https://github.com/vllm-project/vllm-ascend/pull/11227) / [#11709](https://github.com/vllm-project/vllm-ascend/pull/11709)|后续 main2main 同步，随上游改动跟进签名|
|[#13470](https://github.com/vllm-project/vllm-ascend/pull/13470)|本函数未改动，但其 NPU 化模式（int32 `pos`、用 1 元素 block 取 `tl.rand` 替代标量随机数）被该 PR 复用到 `_probabilistic_rejection_kernel`|
