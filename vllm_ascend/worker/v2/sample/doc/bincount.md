# Bincount

> 源码：`vllm_ascend/worker/v2/sample/penalties.py`（kernel `_bincount_kernel`，wrapper `bincount`）。
> 本算子位于 `worker/v2/sample/` 而非 `vllm_ascend/ops/triton/`，故文档随源码目录落在
> `vllm_ascend/worker/v2/sample/doc/`，不放 `ops/triton/doc/`——后者已有一个同名但完全无关的
> `bincount.py`（`token_bin_counts_and_mask_kernel`，服务 model_runner v1 的 penalty 路径）。
> 两者不要混淆。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列加速卡产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

> 本算子无 device adaptor 分支，A2/A3/950 走完全相同的一份 kernel，无下发参数差异。
> 310P 上 `HAS_TRITON` 为 False，且 model_runner v2 未在 310P 上启用，不经过本算子。
> **CANN 版本要求**：kernel 同时使用 `tl.atomic_or` 与 `tl.atomic_add`。CANN 8.5.1 上这两个
> 原子操作存在死锁问题（见 #7757 的源码注释），CANN 9.0.0 起修复；仓库当前基线为 9.1.0
> （`Dockerfile:18`）。

## 功能说明

- API 功能：`Bincount` 为 model_runner v2 采样侧 penalty（repetition / frequency / presence
  penalty）准备两份统计量。对每个**新加入**的、开启了 penalty 的请求，一次性扫描其
  prompt + 已生成部分的全部 token：

  - prompt 段（`[0, prompt_len)`）压成 **bitmap**：只关心某 token 是否出现过，供
    repetition penalty 判定；
  - 输出段（`[prompt_len, prefill_len)`）累加成 **直方图**：frequency penalty 需要出现次数。

  下游 `_penalties_kernel` 直接读这两份统计量。它们随请求驻留，后续解码步产生的新 token 由
  `_penalties_kernel` 自己增量累加，**不再回调本算子**。

- 计算公式：

    $$
    \text{prompt\_bin\_mask}[r,\ \lfloor t/32 \rfloor] \mathrel{|}= 1 \ll (t \bmod 32),
    \quad t = \text{all\_token\_ids}[r, p],\ \ 0 \le p < \text{prompt\_len}[r]
    $$

    $$
    \text{output\_bin\_counts}[r,\ t] \mathrel{+}= 1,
    \quad t = \text{all\_token\_ids}[r, p],\ \ \text{prompt\_len}[r] \le p < \text{prefill\_len}[r]
    $$

    其中 $r$ 取自 `expanded_idx_mapping`。两个输出张量在写入前，其 $r$ 所在行先被整行置零。

- 调用链：

    ```
    PenaltiesState.apply_staged_writes            vllm/v1/worker/gpu/sample/penalties.py（上游）
      └─ penalties.bincount                       被 patch 替换为本算子的 wrapper
           └─ vllm_ascend/patch/worker/patch_v2/patch_triton.py:28  penalties.bincount = bincount
                └─ bincount                       vllm_ascend/worker/v2/sample/penalties.py:200
                     └─ _bincount_kernel          vllm_ascend/worker/v2/sample/penalties.py:153
    ```

    上游 `PenaltiesState` 未被继承或改写，替换只发生在 `bincount` 这个模块级函数名上，
    因此**只有 model_runner v2 路径会走到本算子**；v1 采样走
    `vllm_ascend/ops/triton/penalty.py` 的另一套实现。

- 任务划分：二维 grid `(num_tokens, num_blocks)`。
  - `axis=0` 一个 program 对应 `expanded_idx_mapping` 的一个下标，即一个请求；
  - `axis=1` 沿 token 位置切块，块宽 `BLOCK_SIZE = 1024`，
    `num_blocks = cdiv(max_prefill_len, BLOCK_SIZE)` 由全批次最长的 `prefill_len` 决定。

  block 内不做核间 round-robin，也不涉及 AI Core 数，无需 `init_device_properties_triton()`。

## 参数说明

### Python 接口 `bincount`

| 参数名 |输入/输出/属性| 描述 | 数据类型 |数据格式|
|-------|------------|------|---------|-----|
|expanded_idx_mapping|输入|要统计的请求在 request state 中的行号，shape 为 [num_tokens]。**名字有误导性**，详见约束说明：实际传入的是本次新增 penalty 请求的 `idx_mapping`，元素互不相同。|INT32|ND|
|all_token_ids|输入|request state 的常驻 token 缓冲，shape 为 [max_num_reqs, max_model_len]。仅 `[0, prefill_len)` 区间有效，其后为历史残留。|INT32|ND|
|prompt_len|输入|每个请求的 prompt 长度，shape 为 [max_num_reqs]。按 `expanded_idx_mapping` 的元素取值索引，不是按 token 下标。|INT32|ND|
|prefill_len|输入|每个请求 prefill 阶段的总 token 数（prompt + 已生成），shape 为 [max_num_reqs]。|INT32|ND|
|prompt_bin_mask|输入/输出|prompt token 的位图，shape 为 [max_num_reqs, ceil(vocab_size/32)]。原地更新；进入 kernel 前由 wrapper 将 `expanded_idx_mapping` 指定的行整行置零。|INT32|ND|
|output_bin_counts|输入/输出|输出 token 的直方图，shape 为 [max_num_reqs, vocab_size]。原地更新；置零规则同上。|INT32|ND|
|max_prefill_len|属性|本批请求 `prefill_len` 的最大值，仅用于推导 `num_blocks`。**不是** tensor，是 host 标量。|INT32|-|

### Kernel 接口 `_bincount_kernel`

| 参数名 |输入/输出/属性| 描述 | 数据类型 |
|-------|------------|------|---------|
|expanded_idx_mapping_ptr|输入|同 wrapper，按 `tl.program_id(0)` 索引。|INT32|
|all_token_ids_ptr|输入|同 wrapper。|INT32|
|all_token_ids_stride|属性|运行时参数（无 `do_not_specialize`），取 `all_token_ids.stride(0)`。|INT32|
|prompt_len_ptr|输入|同 wrapper。|INT32|
|prefill_len_ptr|输入|同 wrapper。|INT32|
|prompt_bin_mask_ptr|输出|`tl.atomic_or` 目标。|INT32|
|prompt_bin_mask_stride|属性|运行时参数，取 `prompt_bin_mask.stride(0)`。|INT32|
|output_bin_counts_ptr|输出|`tl.atomic_add` 目标。|INT32|
|output_bin_counts_stride|属性|运行时参数，取 `output_bin_counts.stride(0)`。|INT32|
|BLOCK_SIZE|属性|`tl.constexpr`，位置维分块宽度，wrapper 硬编码为 1024。|INT32|

> 本 kernel **没有任何 `do_not_specialize` 标注**：三个 stride 是普通运行时参数，
> Triton 会按 16 对齐性做特化，但不按取值特化，因此换 shape 不会重编译。
> 唯一的 `tl.constexpr` 是 `BLOCK_SIZE`，取值恒为 1024。

## 约束说明

- 该接口仅用于推理采样路径，无反向。

- **`prompt_len[r] <= prefill_len[r]` 是硬前提。** kernel 先按
  `block_idx * BLOCK_SIZE >= prefill_len` 提前返回，再用
  `block_idx * BLOCK_SIZE < prompt_len` 与 `(block_idx + 1) * BLOCK_SIZE >= prompt_len`
  两个条件把一个 block 拆给 prompt 位图和输出直方图。若 `prompt_len > prefill_len`，
  提前返回会先于位图写入生效，结果是 prompt 位图被静默截断，**不报错**。
  #7757 引入的原始用例正是独立随机生成这两个长度，因而落在契约外。

- **`prefill_len[r] <= max_prefill_len` 是硬前提。** `num_blocks` 只由 host 侧的
  `max_prefill_len` 决定，超出部分的 block 根本不会被 launch，超长请求的尾部被静默丢弃。
  调用方 `PenaltiesState.apply_staged_writes` 取的正是本批 `prefill_len` 的最大值。

- **`expanded_idx_mapping` 名不副实，且必须元素互不相同。** 它与
  `_penalties_kernel` / `apply_min_p` 等同名参数不是一回事：那些是 spec decode 展开后的
  「token → 请求」映射，含重复；本算子的实参是 `PenaltiesState._new_penalties_reqs`，
  即本次新增的 penalty 请求行号，天然去重。
  若将来误传含重复元素的映射，`prompt_bin_mask` 因 `atomic_or` 幂等而正确，
  但 `output_bin_counts` 走 `atomic_add`，会按重复次数**成倍计数**，且不会报错——
  表现为 frequency penalty 被悄悄放大。

- **`prompt_bin_mask` 是 int32 存的位模式，不是数值。** `token_id % 32 == 31` 时
  `1 << 31` 溢出到符号位，该 bin 呈负数。任何比对基准都必须在 uint32 下打包再重解释为
  int32；用更宽的整型打包会在这些 token 上与 kernel 不一致。
  #7757 曾把移位改成 `pow(2.0, bit_idx)`（浮点）以绕过当时的 npu_ir 限制，
  #9726 已改回移位，当前实现与上游 vLLM 一致。

- **`all_token_ids` 在 `[prefill_len, max_model_len)` 区间是历史残留数据。**
  kernel 靠 `mask` 屏蔽，不读该区间；不得依赖调用方把尾部清零。

- **只有 `expanded_idx_mapping` 指定的行会被写。** 其余行属于其他在跑的请求，
  其 penalty 统计量必须保持不变。置零由 wrapper 的
  `prompt_bin_mask[expanded_idx_mapping] = 0` 完成，不在 kernel 内，
  改动 wrapper 时容易连带破坏这一点。

- **对同一行重复调用是覆盖而非累加。** 请求槽位会被复用，若 wrapper 的置零回退，
  `atomic_add` 会把新请求的计数加到上一个占用者的残值上，同样不会报错。

- CANN 版本要求见「产品支持情况」的引言块。`tl.atomic_or` 在 CANN 8.5.1 上会挂死，
  这是本算子用例长期被 `skip` 的原因。

## 测试说明

数值精度用例：`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_bincount.py`

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_bincount.py
```

用例以 NumPy 循环实现为基准做**逐元素精确比对**（整型输出，`torch.equal`，无容差）：

| 输出 | 比对方式 | 容差 |
|---|---|---|
| prompt_bin_mask | `torch.equal`（uint32 打包后重解释为 int32） | 精确 |
| output_bin_counts | `torch.equal` | 精确 |

基准的等价性已离线验证：把 kernel 的分块结构（`BLOCK_SIZE` 分块 + 两个分支条件）逐行
翻译成模拟器，与用例里的「直观语义」基准在 `prompt_len <= prefill_len <= max_prefill_len`
的全部小规模组合上穷举比对，结果一致。

覆盖范围：

- prompt / 输出两段在 `BLOCK_SIZE=1024` 边界上的交接（`prompt_len` 取 1023 / 1024 / 1025 三档）；
- prompt 跨多个 block（2500）、输出跨多个 block、`prefill_len` 正好块对齐；
- 退化长度：空 prompt（`prompt_len=0`）、空输出段（`prompt_len == prefill_len`）、
  `prefill_len=0` 的提前返回分支（放在多请求场景内）；
- 多请求：6 行、行号不连续且乱序、长度各异，防止「请求行号」与「token 下标」写反；
- 符号位打包：全部 prompt token 取 `id % 32 == 31`，并断言确实产生了负数 bin（守护守护者）；
- 重复 token 的 `atomic_add` 累加（断言最大计数 > 1 且总数守恒）；
- `prefill_len` 之后的残留 token 不被计入（用与有效 token 池不相交的哨兵 id 检测）；
- 未在 `expanded_idx_mapping` 中的行保持不变（两个输出张量调用前均被投毒）；
- 同一行连续两次调用是覆盖而非累加；
- 真实词表规模 151936（行 stride 151936 / 4748，均非 2 的幂）跑一次。

未覆盖及原因：

- **#7757 声称的 10% 性能提升**：属于性能特征而非计算正确性，nightly 用例不做性能门禁。
- **含重复元素的 `expanded_idx_mapping` 会成倍计数**：这是契约外输入，当前调用方不会产生。
  写成用例等于把错误行为固化为期望值，故只在「约束说明」中记录。
- **`prompt_len > prefill_len` 的静默截断**：同上，契约外输入，只记录不断言。
- **`prefill_len` 单请求为 0 导致 `num_blocks == 0`**：会 launch 一个含零维的 grid，
  属于调用方契约外的场景（真实请求 `prefill_len >= 1`）。该分支通过多请求场景中的
  `(0, 0)` 行覆盖，此时 `max_prefill_len > 0`，grid 合法。

## 变更记录

| PR | 说明 |
|---|---|
|[#7757](https://github.com/vllm-project/vllm-ascend/pull/7757)|引入 `_bincount_kernel` 与 `bincount`（对齐上游 vLLM 实现），并新增 `test_bincount.py`——该用例自引入起即被 `skip`|
|[#9085](https://github.com/vllm-project/vllm-ascend/pull/9085)|CANN 升至 9.0.0、triton-ascend 升至 3.2.1，`atomic_or` 死锁问题的修复版本|
|[#9726](https://github.com/vllm-project/vllm-ascend/pull/9726)|main2main 同步，位打包由 `pow(2.0, bit_idx)` 改回 `1 << bit_idx`，与上游一致|
