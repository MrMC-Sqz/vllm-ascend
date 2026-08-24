# GumbelSample

本文档覆盖 `vllm_ascend/worker/v2/sample/gumbel.py` 中的 `_gumbel_sample_kernel`
及其 Python 入口 `gumbel_sample`。

> **源码位置说明**：该 kernel 不在 `vllm_ascend/ops/triton/` 下，而在
> `vllm_ascend/worker/v2/sample/` 下（model runner v2 的采样路径），但它是标准的
> Triton kernel，其同文件的兄弟算子 `_temperature_kernel` 的用例也已落在
> `tests/e2e/nightly/.../triton/test_temperature.py`。为便于集中检索，
> 本文档仍与其它 Triton 算子文档一起放在本目录。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列加速卡产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

> 本算子不经过 `DeviceOperator`，A2/A3 与 Ascend 950 下发参数完全一致，无 adaptor 分支。
> 310P 上 `vllm.triton_utils.HAS_TRITON` 为 False，`vllm_ascend/patch/worker/__init__.py`
> 不会导入 `patch_v2/patch_triton.py`，因此 `gumbel_sample` / `apply_temperature` 的
> NPU 版本根本不会被注册（model runner v2 亦不在 310P 上启用）。

## 功能说明

- API 功能：model runner v2 的**采样算子**。给定每个 token 的 logits、每个请求的
  temperature 与随机种子，用 Gumbel-max trick 一次性完成「按 softmax 分布采样」，
  返回每个 token 采到的 token id；temperature 为 0 的请求退化为 greedy（argmax）。
  同时可选地把「加噪前、已按 temperature 缩放」的 logits 旁路输出到一块
  `[max_num_reqs, (num_steps,) vocab_size]` 的 buffer，供 EAGLE 投机推理与拒绝采样
  复用（省掉一次全量 logits 重算）。
  它替换上游 `vllm.v1.worker.gpu.sample.gumbel.gumbel_sample`，被 `sampler`、
  EAGLE `speculator`、`rejection_sampler` 三处 monkey-patch 引用。

- 计算公式：Gumbel-max trick——若 $g_i$ 独立同分布于标准 Gumbel 分布，则

    $$
    \arg\max_i \left( \frac{z_i}{T} + g_i \right) \sim \mathrm{Softmax}\left(\frac{z}{T}\right)
    $$

    kernel 中 $g_i$ 由均匀分布 $u_i \in [0, 1)$ 变换得到（**取的是最大值型 Gumbel**）：

    $$
    g_i = -\log\left(-\log(u_i + 10^{-20}) + 10^{-20}\right),\quad
    u = \mathrm{tl.rand}(\mathrm{randint}(\mathrm{seed}_{req},\ \mathrm{pos}_{token}),\ i)
    $$

    $T = 0$ 时**不加噪**，直接取 $\arg\max_i z_i$。
    两个 $10^{-20}$ 是防 $\log(0)$ 的下钳位，其代价是噪声上界被截到约 46
    （fp32 下 $-\log(10^{-20}) \approx 46$），下文「约束说明」有用到这个界。

- 调用链：

    ```text
    vllm.v1.worker.gpu.sample.sampler.Sampler.sample                vllm 上游（已被 patch）
    vllm.v1.worker.gpu.spec_decode.eagle.speculator                 vllm 上游（已被 patch）
    vllm.v1.worker.gpu.spec_decode.rejection_sampler                vllm 上游（已被 patch）
      └─ gumbel_sample                                              vllm_ascend/patch/worker/patch_v2/patch_triton.py 绑定
           └─ gumbel_sample                                         vllm_ascend/worker/v2/sample/gumbel.py
                ├─ _gumbel_sample_kernel                            Triton kernel（本算子）
                ├─ local_max.argmax(dim=-1)                         torch，块间归约
                └─ local_argmax.gather(...)                         torch，取回 token id
    ```

    同文件另有 `apply_temperature` / `_temperature_kernel`（原地缩放 logits），
    与本算子是**互斥关系**：调用方若已经用 `apply_temperature` 缩放过，就以
    `apply_temperature=False` 下发本算子，此时 kernel 只加噪、不再除 $T$。

- 任务划分：grid 为 `(num_tokens,)`，**一个 token 一个 program**。
  词表方向不并行，由 program 内 `for block_idx in range(num_blocks)` 串行遍历，
  `BLOCK_SIZE = 1024`，`num_blocks = cdiv(vocab_size, 1024)`。
  每块把块内 `(argmax, max)` 写进 `local_argmax/local_max[token, block_idx]`，
  块间的最终归约放在 host 侧用 torch 完成（`argmax` + `gather`）。

## 参数说明

### Python 接口 `gumbel_sample`

| 参数名 |输入/输出/属性| 描述 | 数据类型 |数据格式|
|-------|------------|------|---------|-----|
|logits|输入|shape [num_tokens, vocab_size]，**不被原地修改**；只按 `stride(0)` 寻址，无需连续。|FLOAT32（生产路径，见约束）|ND|
|expanded_idx_mapping|输入|shape [num_tokens]，token → 请求槽位（`req_state_idx`）的映射，可非连续、可多对一。|INT32/INT64|ND|
|temperature|输入|shape [max_num_reqs]，按 `req_state_idx` 索引。0 表示 greedy。|FLOAT32|ND|
|seed|输入|shape [max_num_reqs]，每请求随机种子，按 `req_state_idx` 索引。|INT64|ND|
|pos|输入|shape [num_tokens]，每 token 的位置，参与种子派生，**kernel 内被截断为 int32**。|INT32/INT64|ND|
|apply_temperature|属性|`tl.constexpr`。True 时 kernel 内部除以 $T$；False 表示调用方已缩放过。|BOOL|-|
|output_processed_logits|可选输出|shape [max_num_reqs, vocab_size] 或 [max_num_reqs, num_steps, vocab_size]，写入**加噪前**的 logits；为 None 时该分支被 constexpr 折叠掉。|FLOAT32|ND|
|output_processed_logits_col|可选输入|选择写入哪一列（draft step）。0 维张量 = 所有 token 同一列（`PER_TOKEN_COL=False`）；1 维张量 [num_tokens] = 每 token 各自一列（`PER_TOKEN_COL=True`）。|INT32|ND|
|use_fp64|属性|**必须为 False**，为 True 直接抛 `NotImplementedError`。|BOOL|-|
|输出|输出|shape [num_tokens] 的采样 token id。|INT64|ND|

### Kernel 接口 `_gumbel_sample_kernel`

`do_not_specialize` 覆盖 `local_argmax_stride`、`local_max_stride`、
`processed_logits_stride`、`logits_stride`、`vocab_size`、`num_blocks`——
即所有随 batch/词表变化的量都是**运行时参数**，只有下表标注为 `tl.constexpr` 的三个
参数参与 JIT 签名，因此 shape 变化不触发重编译。

| 参数名 |输入/输出/属性| 描述 | 数据类型 |
|-------|------------|------|---------|
|local_argmax_ptr / local_argmax_stride|输出|[num_tokens, num_blocks]，每块的块内 argmax（已还原为全局 token id）。stride 为运行时参数。|INT64|
|local_max_ptr / local_max_stride|输出|[num_tokens, num_blocks]，每块的块内最大值（含噪）。stride 为运行时参数。|FLOAT32|
|processed_logits_ptr / processed_logits_stride|可选输出|旁路 buffer 与其 `stride(0)`；wrapper 在为 None 时传 stride 0，kernel 用 `is not None` 做**编译期**分支。|FLOAT32|
|processed_logits_col_ptr|可选输入|列号张量指针，同样是编译期 `is not None` 分支。|INT32|
|logits_ptr / logits_stride|输入|输入 logits 与其 `stride(0)`，运行时参数。|FLOAT32|
|expanded_idx_mapping_ptr|输入|token → 请求槽位映射。|INT32/INT64|
|seeds_ptr|输入|每请求种子，按 `req_state_idx` 取。|INT64|
|pos_ptr|输入|每 token 位置，`.to(tl.int32)` 后使用。|INT32/INT64|
|temp_ptr|输入|每请求 temperature，按 `req_state_idx` 取。|FLOAT32|
|vocab_size|属性|运行时参数（`do_not_specialize`），用于 mask 与列偏移。|INT32|
|num_blocks|属性|运行时参数（`do_not_specialize`），串行块循环次数。|INT32|
|BLOCK_SIZE|属性|`tl.constexpr`，wrapper 固定为 1024。|INT32|
|APPLY_TEMPERATURE|属性|`tl.constexpr`，是否在 kernel 内除以 temperature。|BOOL|
|PER_TOKEN_COL|属性|`tl.constexpr`，列号是每 token 一个还是全局一个。|BOOL|

## 约束说明

- **`apply_temperature=False` 不等于「temperature 不生效」**：该开关只关掉
  `logits / T` 这一步，**噪声仍然只在 $T \ne 0$ 时才加**。也就是说 temperature
  在 kernel 里有两个作用（缩放、greedy 开关），开关只关掉前者。调用方若已用
  `apply_temperature()` 预缩放，必须继续把**原始 temperature 张量**传进来，
  否则 $T$ 被置 0 会静默退化成 greedy。
- **temperature / seed 按 `req_state_idx` 索引，logits / pos 按 `token_idx` 索引。**
  两套下标混用过一次（#9173：原实现用 `batch_idx` 取 temperature），
  在「一个请求多 token」（投机推理、chunked prefill）时结果错误。用例中所有
  `expanded_idx_mapping` 都刻意取非恒等映射来锁这条。
- **`pos` 被截断为 int32**：triton-ascend 的 philox 只支持 int32/uint32
  （`umulhi`），kernel 内 `tl.load(pos_ptr + token_idx).to(tl.int32)`。
  位置超过 $2^{31}-1$ 时种子会回绕——现网 `max_model_len` 远小于该值，不构成问题，
  但移植到长上下文场景需重新评估。
- **噪声是 fp32 而非上游的 fp64**：上游 vLLM 用 `tl_rand64` + float64，
  triton-ascend 无 float64。因此本实现与 GPU 版**逐位不可比**，
  同 seed 同 pos 在 NPU/GPU 上会采到不同 token（分布仍相同）。
  `use_fp64=True` 因此直接抛 `NotImplementedError`，而不是悄悄降级为 fp32。
- **噪声上界约 46**：`-log(-log(u + 1e-20) + 1e-20)` 在 fp32 下不超过约 46。
  这既是「logit 差距 > 60 即可确定性获胜」的依据，也意味着**极端稀疏的 logits
  （如 bad-words 掩码用 -1e4）不会被噪声翻盘**。
- **`output_processed_logits` 按 `req_state_idx` 写**：若多个 token 映射到同一请求，
  这些 token 会写同一行，**写入顺序未定义**。现网只有 EAGLE 使用该旁路输出，
  且恒为 1:1 映射；破坏该前提会得到不确定结果，且 kernel 不做任何检查。
- **旁路 buffer 存的是加噪前的值**，且当 `apply_temperature=False` 时存的是**原始
  logits**（未除 $T$）。下游把它当 draft 分布使用，语义依赖这一点。
- `PER_TOKEN_COL` 由 wrapper 按 `output_processed_logits_col.dim() > 0` 判定：
  传 0 维张量（`torch.tensor(1)`）与传 1 维张量（`torch.tensor([1])`）走**不同分支**，
  前者所有 token 共用一列，后者按 token 取列。传 `int` 而非张量会在 kernel 内
  `tl.load` 时报错。
- **词表方向串行**：grid 只有 `num_tokens` 一维，`vocab_size = 151936` 时每个 program
  串行 148 次块循环。这是 #13470 为支持拒绝采样复用 kernel 而做的改动
  （原为二维 grid），属于已知的性能取舍，不是正确性问题。
- **greedy 路径的并列处理**：块内 `tl.argmax` 与 host 侧 `argmax` 都取**最小下标**，
  因此与 `torch.argmax` 的并列语义一致；但这依赖两侧实现，随机 fp32 logits 下不会触发。
- **输入 dtype**：现网路径（vllm main / v0.27.1）传入的是 fp32 logits，kernel 内
  统一 `.to(tl.float32)`。bf16/fp16 输入在语义上可跑，但 argmax 会因大量并列值
  而与 torch 结果不一致，用例中未覆盖（见「测试说明」）。
- 调用前置条件：需先 `init_device_properties_triton()`（nightly 的
  `tests/e2e/nightly/single_node/ops/conftest.py` 已自动执行）。

## 测试说明

数值精度用例：`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gumbel_sample.py`

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_gumbel_sample.py
```

另有已存在的 `tests/ut/sample/a2/test_gumbel_sampling.py`（A2 NPU 目录，
PR 触发，由 `test_config.yaml` 的 `sample` module 挂载），它锁的是 greedy 路径、
`apply_temperature` 的等价性与旁路输出的基本正确性；本次新增的用例与之互补，
锁的是**它锁不住的部分**：Gumbel 分支的数值语义。

**为什么原有覆盖不够**：原用例对 $T > 0$ 的路径只断言了「同 seed 可复现」
「不同 seed 结果不同」「token id 在合法区间」以及一条粗粒度的熵比较。
噪声公式写错（符号翻转变成 min-Gumbel、漏掉 temperature 除法、每块复用同一份噪声）
时，上述断言**全部仍然通过**。新用例改为直接比对**采样分布**与 softmax。

基准（golden）：

| 路径 | 基准 |
|---|---|
|greedy（$T = 0$）|`torch.argmax`，逐元素**精确相等** |
|旁路 logits|PyTorch fp32 循环实现 `_ref_processed_logits`，`assert_close` |
|Gumbel 分支|`torch.softmax(z/T)`，与 8192 次独立抽样的经验频率比对 |

容差：

| 比较对象 | rtol | atol |
|---|---|---|
|greedy token id|—|逐位相等（`torch.equal`）|
|旁路 logits（fp32）|1e-6|1e-6|
|采样频率 vs softmax|—|$6\sqrt{p(1-p)/N} + 0.005$（$N$ 为抽样数）|

频率容差取 6σ 二项置信带加 5e-3 松弛：足以避免统计抖动导致的偶发失败，
又远小于「符号翻转 / 漏乘 temperature」造成的偏差（后者是分布级的量级变化）。
抽样通过「一次下发 8192 个 token 行、每行不同 `pos`」实现，
因此一条分布用例只有一次 kernel 下发，不会触发 nightly conftest 的 120s 熔断。

覆盖范围：

- greedy 路径与 `torch.argmax` 逐元素相等：词表 1 / 512 / 1024 / 3072 / 3000 / 151936，
  覆盖单块、块整数倍、非对齐尾块、真实模型词表；`num_tokens` 与 `num_reqs`
  取不相等且非 2 的幂，防 `//` 与 `%` 写反时碰巧算对
- 获胜 token 被强制放进**最后一块**，锁 `block_idx * BLOCK_SIZE + idx` 的还原
  与 host 侧块间归约（随机 logits 下获胜块是随机的，锁不住这条）
- `pos` 的 int32 / int64 两种 dtype
- 同一次下发内混合 $T = 0$ 与 $T > 0$ 的请求，greedy 行仍逐位精确
- Gumbel 分支的采样分布 == softmax：`APPLY_TEMPERATURE` 真假两侧，
  temperature 取「变尖（0.7）/ 变平（1.8）」两档；
  `APPLY_TEMPERATURE=False` 那条专门锁 #9173 的语义（不缩放但仍加噪）
- 噪声在词表块之间相互独立：全平 logits 下 4 个块的胜率均为 25%±，
  16 等分桶均匀——直接锁死「每块复用同一份噪声」这类退化
- 非对齐尾块：既不越界采到 padding，尾块胜率也精确等于 `952/3000`
- Gumbel-max 的两条结构不变量：logits 整体平移不改变采样结果（逐位相等）、
  logit 高出 200 的 token 必胜
- 种子派生：同请求同 `pos` 得到同一噪声；不同 `pos` 必须解相关（守护守护者断言）
- 旁路输出：`[max_num_reqs, vocab]` 与 `[max_num_reqs, num_steps, vocab]` 两种布局，
  `PER_TOKEN_COL` 真假两侧（**1 维列张量分支此前无任何用例**），
  非连续 EAGLE 式映射写到正确的请求槽位、未使用槽位保持为零、
  $T = 0$ 的行不做缩放、旁路值不含噪声（换 seed 后逐位相同）
- `use_fp64=True` 抛 `NotImplementedError`

未覆盖及原因：

- **bf16/fp16 logits**：现网路径是 fp32（见约束说明）。低精度下 randn 词表里
  存在大量并列最大值，`torch.argmax` 与 kernel 的并列选择无法保证一致，
  用例会变成对并列规则的测试而非对算子的测试。
- **与 GPU 版逐位对齐**：噪声是 fp32（上游为 fp64），设计上就不逐位可比，
  只能在分布层面对齐，已由分布用例覆盖。
- **多 token 映射到同一请求时的旁路输出**：写入顺序未定义（见约束说明），
  给它上锁等于给未定义行为上锁。
- **`pos` 超过 int32 范围的回绕**：需要 $2^{31}$ 量级的位置，现网不可达，
  构造用例的代价与收益不成比例，只在约束说明中记录。
- **词表方向串行带来的性能特征**（`vocab_size = 151936` 时 148 次块循环）：
  属性能问题，且检测手段依赖 Triton 内部结构，按规范不做成用例。

## 变更记录

| PR | 说明 |
|---|---|
|[#5210](https://github.com/vllm-project/vllm-ascend/pull/5210)|引入 `_gumbel_sample_kernel` 与 `gumbel_sample`，适配 model runner v2 eager 模式；相对上游改动：`pos` 降为 int32、噪声由 fp64 降为 fp32|
|[#7885](https://github.com/vllm-project/vllm-ascend/pull/7885)|适配 EAGLE，`speculator` 侧补 patch|
|[#8083](https://github.com/vllm-project/vllm-ascend/pull/8083)|同文件新增 `_temperature_kernel` / `apply_temperature`（NPU 版温度缩放）|
|[#9173](https://github.com/vllm-project/vllm-ascend/pull/9173)|**修 temperature 取错下标**：`idx_mapping` 更名为 `expanded_idx_mapping`，temperature/seed 改为按 `req_state_idx` 取，logits/pos 仍按 `token_idx` 取；同时补 `do_not_specialize` 消重编译|
|[#12859](https://github.com/vllm-project/vllm-ascend/pull/12859)|`apply_temperature` 的 `BLOCK_SIZE` 随 vLLM 0.26 的 bf16 logits 下调至 32768（UB 上限）|
|[#13470](https://github.com/vllm-project/vllm-ascend/pull/13470)|支持投机推理的概率拒绝采样：grid 由 `(num_tokens, num_blocks)` 改为 `(num_tokens,)` + 块内串行循环，新增 `num_blocks` 运行时参数与 `PER_TOKEN_COL`|
|[#14027](https://github.com/vllm-project/vllm-ascend/pull/14027)|上游恢复 fp32 logits 上采，`apply_temperature` 的 `BLOCK_SIZE` 固定回 44032|
