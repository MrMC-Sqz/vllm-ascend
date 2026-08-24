# LightningAttention

本文档覆盖 `vllm_ascend/ops/triton/mamba/lightning_attn.py` 中的四个 Triton kernel：
`_fwd_diag_kernel`、`_fwd_kv_parallel`、`_fwd_kv_reduce`、`_fwd_none_diag_kernel`。
四者只作为一条流水线整体下发（`_attention.forward`），没有独立的对外入口，故合并成一篇文档。

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
> 310P 上 `HAS_TRITON` 为 False，BailingMoE 线性注意力不走本路径。
> `_fwd_diag_kernel` 额外传入 `multibuffer` / `set_workspace_multibuffer` /
> `tile_mix_vector_loop` / `tile_mix_cube_loop` 等 Triton-Ascend 私有编译选项，
> 这些选项是 Ascend 后端专有的，非 Ascend 后端上无法直接复用该下发代码。

## 功能说明

- API 功能：BailingMoE 线性注意力（lightning attention）prefill 阶段的前向算子，
  是 GPU 侧 `MiniMaxText01LinearKernel` 的 NPU 替代。给定 $Q, K, V$ 与每个 head 的
  衰减率 $s$，按指数衰减权重做因果线性注意力，并把序列结束时的 KV 状态写回
  `kv_history`（即 mamba cache），供后续 decode 步继续递推。
  上游为 `AscendBailingMoELinearAttention._prefill_and_mix_infer`（`vllm_ascend/ops/bailing_moe_linear_attn.py`），
  该路径会先把 QKV 提升到 float32 再调用本算子。

- 计算公式：算法把序列按 `BLOCK = 256` 分块，块内（对角块）与块间（非对角块）分别计算：

    $$
    O_t = \underbrace{\sum_{j \le t,\ \lfloor j/B \rfloor = \lfloor t/B \rfloor} e^{-s (t-j)} (Q_t \cdot K_j) V_j}_{\text{\_fwd\_diag\_kernel}}
        + \underbrace{\sum_{j < t,\ \lfloor j/B \rfloor \ne \lfloor t/B \rfloor} e^{-s (t-j-1)} (Q_t \cdot K_j) V_j
        + e^{-s t} \, Q_t \cdot KV_{\text{hist}}}_{\text{\_fwd\_kv\_parallel} \to \text{\_fwd\_kv\_reduce} \to \text{\_fwd\_none\_diag\_kernel}}
    $$

    KV 状态的更新为：

    $$
    KV_{\text{hist}}' = e^{-s n} KV_{\text{hist}} + \sum_{j=0}^{n-1} e^{-s (n-1-j)} K_j \otimes V_j
    $$

    其中 $B$ 为 `BLOCK`，$n$ 为序列长度。**注意块间指数是 $t-j-1$ 而不是 $t-j$**，
    详见「约束说明」。

- 分步说明：

    | 步骤 | kernel | 作用 |
    |---|---|---|
    | 1 | `_fwd_diag_kernel` | 块内因果注意力，直接写 `Out` |
    | 2 | `_fwd_kv_parallel` | 每个块独立计算 $\sum_j e^{-s(B-1-j)} K_j \otimes V_j$，写入 `KV[b,h,块号]` |
    | 3 | `_fwd_kv_reduce` | 对 `KV` 沿块维做**排他前缀扫描**：`KV[i]` 被改写为「进入第 i 块前的状态」，同时把最终状态写回 `KV_HISTORY` |
    | 4 | `_fwd_none_diag_kernel` | $Q_t \cdot KV[\text{块号}] \cdot e^{-s\,t_{\text{local}}}$，**累加**到步骤 1 的 `Out` 上 |

- 调用链：

    ```
    AscendBailingMoELinearAttention._prefill_and_mix_infer   vllm_ascend/ops/bailing_moe_linear_attn.py
      └─ linear_attention_prefill_and_mix(prefix_fn=...)     vllm 上游
           └─ AscendLightningAttentionKernel.jit_linear_forward_prefix   本文件
                └─ lightning_attention_npu                   本文件（d 维分块）
                     └─ lightning_attention_npu_ = _attention.apply      本文件
                          ├─ _fwd_diag_kernel                Triton kernel
                          ├─ _fwd_kv_parallel                Triton kernel
                          ├─ _fwd_kv_reduce                  Triton kernel
                          └─ _fwd_none_diag_kernel           Triton kernel
    ```

- 任务划分：

    | kernel | grid | 说明 |
    |---|---|---|
    | `_fwd_diag_kernel` | `(b*h*NUM_BLOCK, BLOCK//32)` | dim-0 展平了 batch-head 与块号（`off // NUM_BLOCK` / `off % NUM_BLOCK`），dim-1 是块内 32 行一组的子块 |
    | `_fwd_kv_parallel` | `(b*h, NUM_BLOCK, 2)` | dim-2 把 `e` 切成两个 `E_FBLOCK = e/2` 的列块，防 UB 溢出 |
    | `_fwd_kv_reduce` | `(b*h, 2)` | 块维必须串行（前缀扫描），只在 batch-head 与 `e` 列块上并行 |
    | `_fwd_none_diag_kernel` | `(b*h, NUM_BLOCK*(BLOCK//64), 2)` | dim-1 展平了块号与块内 64 行子块 |

    `NUM_BLOCK = cdiv(n, 256)`。UB 占用估算见 `_attention.forward` 中的注释。

## 参数说明

### Python 接口 `AscendLightningAttentionKernel.jit_linear_forward_prefix`

| 参数名 |输入/输出/属性| 描述 | 数据类型 |数据格式|
|-------|------------|------|---------|-----|
|q|输入|查询，shape 为 [h, n, d]（3 维时内部 `unsqueeze(0)`）或 [1, h, n, d]。|FLOAT32/BFLOAT16/FLOAT16|ND|
|k|输入|键，shape 同 `q`。|FLOAT32/BFLOAT16/FLOAT16|ND|
|v|输入|值，shape 为 [h, n, e]。|FLOAT32/BFLOAT16/FLOAT16|ND|
|kv_caches|输入/输出|KV 状态（mamba cache），shape 为 [h, d, e]，**原地更新**为序列结束后的状态。|FLOAT32|ND|
|slope_rate|输入|每个 head 的衰减率 $s$，shape 为 [h] 或 [1, h, 1, 1]，内部转 float32。|FLOAT32|ND|
|block_size|属性|**当前被忽略**，实际分块长度固定为 256，详见约束说明。|INT32|-|
|layer_idx|可选属性|未使用，仅为与上游 `prefix_fn` 签名对齐。|INT32|-|
|输出|输出|shape 为 [n, h*e]，由 `rearrange(o, "h n d -> n (h d)")` 得到。|与 `q` 同|ND|

### Python 接口 `lightning_attention_npu`

| 参数名 |输入/输出/属性| 描述 | 数据类型 |数据格式|
|-------|------------|------|---------|-----|
|q / k|输入|shape [b, h, n, d]。|FLOAT32/BFLOAT16/FLOAT16|ND|
|v|输入|shape [b, h, n, e]。|FLOAT32/BFLOAT16/FLOAT16|ND|
|ed|输入|衰减率，1 维时内部 `view(1, -1, 1, 1)`。|FLOAT32|ND|
|block_size|属性|**被忽略**。|INT32|-|
|kv_history|可选输入|shape [b, h, d, e]，为 None 时内部建零张量；非 None 时先 `clone()`，不改写调用方张量。|FLOAT32|ND|
|输出 0|输出|注意力输出 [b, h, n, e]。|与 `q` 同|ND|
|输出 1|输出|`kv`，shape [b, h, NUM_BLOCK+1, d, e]，前 `NUM_BLOCK` 项为各块的排他前缀状态，最后一项为最终状态。|FLOAT32|ND|

### Kernel 接口

四个 kernel 的公共属性：`b`、`h`、`d`、`e`、`BLOCK` 均为 `tl.constexpr`；
`n`、`NUM_BLOCK` 在 `_fwd_diag_kernel` / `_fwd_kv_parallel` 中是 `tl.constexpr`，
在 `_fwd_kv_reduce` / `_fwd_none_diag_kernel` 中是运行时参数。
**没有任何参数带 `do_not_specialize`**，因此每个新的 `(b, h, n, d, e)` 组合都会触发一次重编译。

#### `_fwd_diag_kernel`

| 参数名 |输入/输出/属性| 描述 | 数据类型 |
|-------|------------|------|---------|
|Q / K / V|输入|连续张量，按 `off_bh * n * d`（或 `* n * e`）寻址。|FLOAT32/BFLOAT16/FLOAT16|
|Out|输出|[b, h, n, e]，本 kernel **写**（非累加），必须先于 `_fwd_none_diag_kernel` 执行。|与 Q 同|
|S|输入|每 head 衰减率，按 `off_h = off_bh % h` 取标量。|FLOAT32|
|CBLOCK|属性|`tl.constexpr`，块内子块行数，wrapper 固定为 32。|INT32|
|NUM_BLOCK|属性|`tl.constexpr`，块数，同时用于从 `program_id(0)` 还原 `off_bh` / `off_block`。|INT32|

#### `_fwd_kv_parallel`

| 参数名 |输入/输出/属性| 描述 | 数据类型 |
|-------|------------|------|---------|
|K / V|输入|同上。|FLOAT32/BFLOAT16/FLOAT16|
|K_decay|输入|[h, BLOCK]，由 wrapper 预计算为 `exp(-s * (BLOCK - (arange+1)))`。|FLOAT32|
|KV|输出|[b, h, NUM_BLOCK, d, e]，每个块写自己的那一片。|FLOAT32|
|D_FBLOCK|属性|`tl.constexpr`，wrapper 固定传 `d`（d 方向不切分）。|INT32|
|E_FBLOCK|属性|`tl.constexpr`，wrapper 固定传 `e // 2`。|INT32|
|NUM_FBLOCK|属性|`tl.constexpr`，传入但 kernel 内**未使用**。|INT32|
|CBLOCK / NUM_CBLOCK|属性|`tl.constexpr`，64 与 `BLOCK // 64 = 4`。|INT32|

#### `_fwd_kv_reduce`

| 参数名 |输入/输出/属性| 描述 | 数据类型 |
|-------|------------|------|---------|
|S|输入|每 head 衰减率。|FLOAT32|
|KV|输入/输出|[b, h, NUM_BLOCK, d, e]，**原地**改写为排他前缀扫描结果。|FLOAT32|
|KV_HISTORY|输入/输出|[b, h, d, e]，**原地**改写为最终状态。|FLOAT32|
|n / NUM_BLOCK|属性|运行时参数（非 `constexpr`），用于计算尾块实际长度 `min(n - i*BLOCK, BLOCK)`。|INT32|

#### `_fwd_none_diag_kernel`

| 参数名 |输入/输出/属性| 描述 | 数据类型 |
|-------|------------|------|---------|
|Q|输入|同上。|FLOAT32/BFLOAT16/FLOAT16|
|Out|输入/输出|读出 `_fwd_diag_kernel` 的结果并累加后写回，因此中间结果经历一次 `q.dtype` 舍入。|与 Q 同|
|S|输入|每 head 衰减率。|FLOAT32|
|KV|输入|`_fwd_kv_reduce` 输出的前缀状态。|FLOAT32|
|E_FBLOCK / CBLOCK / NUM_CBLOCK|属性|`tl.constexpr`，`e/2`、64、4。|INT32|

## 约束说明

- 该接口只支持推理前向。`_attention` 虽继承 `torch.autograd.Function` 并 `save_for_backward`，
  但**没有实现 `backward`**，反向传播会抛异常。
- **块间衰减比精确递推多一个 $e^{s}$ 因子**：`_fwd_kv_parallel` 的 `k_decay` 把块状态衰减到
  该块**最后一个 token**（`exp(-s*(BLOCK-1-j))`），而 `_fwd_none_diag_kernel` 只按
  `exp(-s*t_local)` 回放，于是跨块 token 对的权重是 $e^{-s(t-j-1)}$，块内则是 $e^{-s(t-j)}$。
  同样的 token 间距，落在同块与跨块权重不一致。该行为与上游 vLLM / MiniMax 的 GPU 实现一致，
  属于移植保真而非本仓引入，故未改动；用例中以此为基准，并另有两条不依赖该约定的语义用例
  （`n <= BLOCK`、`s == 0`）作为兜底。修改衰减约定前请先同步上游。
- **`d` 必须 ≤ 128**：`lightning_attention_npu` 在 `d > 128` 时按 `m=128` 把 `d` 切块循环，
  但每次循环把**完整的** `kv_history`（[b, h, d, e]）传给 kernel，而 kernel 内部按
  `d = 128` 计算 `off_bh * d * e` 偏移，读写位置错位；且函数只返回最后一个 d 块的 `kv`。
  当前 BailingMoE 的 `head_dim` 为 128，走单块路径，不受影响。
- `d` 与 `e` 必须是 2 的幂（`tl.arange(0, d)` / `tl.arange(0, E_FBLOCK)` 的要求），
  且 `e` 必须能被 2 整除（`E_FBLOCK = e // 2`，wrapper 中有 assert）。
- **`block_size` 参数被忽略**：`jit_linear_forward_prefix` 与 `lightning_attention_npu` 都接收
  `block_size`，但 `_attention.forward` 内硬编码 `BLOCK = 256`。传 64 与传 256 结果完全相同。
  由于块边界会影响上面那条衰减约定，这个"静默忽略"值得注意。
- `kv_history` / `kv_caches` 必须是 float32 且连续；`_fwd_kv_reduce` 对其**原地写**。
  用例比对时必须先 `clone()` 再下发，否则基准拿到的是被改写后的值。
- `lightning_attention_npu_`（即 `_attention.apply`）也会**原地改写**传入的 `kv_history`；
  只有外层 `lightning_attention_npu` 做了 `clone()`。
- 分块前缀语义：返回的 `kv[:, :, i]` 是「进入第 i 块**之前**」的状态（排他扫描），
  因此 `kv[:, :, 0]` 恒等于传入的 `kv_history`，最终状态在 `kv[:, :, -1]`。
- 分段 prefill 只有在 **256 对齐**的切分点上才与一次性 prefill 等价。
  在非对齐点切分会改变哪些 token 对属于"同块"，结果与一次性计算不同（差异量级为 $e^{s}$）。
- `_fwd_diag_kernel` 对尾块 padding 行在 `tl.load(..., other=0.0)` 之外**额外做了 `tl.where` 重置**
  （#10276）：Ascend 上向量到 cube 的搬运可能不清零越界数据，残留值进 `tl.dot` 会出 NaN。
  `_fwd_kv_parallel` 的越界行目前**只有 mask、没有 `tl.where` 兜底**，
  若同类问题在该 kernel 复现，需按 #10276 的方式补齐。
- `_fwd_kv_parallel` 处理尾块时用 `left_shift` 把子块整体左移对齐到块尾，
  首个子块会读到本块起始位置**之前**的地址（由 mask 屏蔽）。序列首块 + `n < 64` 时，
  该地址在张量首地址之前，依赖 masked load 不触发非法访问。
- 变长/多序列场景不由本算子处理：`jit_linear_forward_prefix` 内 `assert output.shape[0] == 1`，
  batch 维必须为 1，多序列由上游 `linear_attention_prefill_and_mix` 逐序列切分后调用。
- 无 `do_not_specialize`，`(b, h, n, d, e)` 每变一次触发 4 次重编译。这是首 token 时延特征，
  不影响正确性；写用例时需注意 shape 网格规模。

## 测试说明

数值精度用例：`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_lightning_attn.py`

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_lightning_attn.py
```

基准为 PyTorch fp32 朴素实现。基准**不复刻 kernel 的分块结构**（不模拟 CBLOCK 循环、
`left_shift`、decay 指针偏移），而是按上面「计算公式」一节的闭式权重矩阵直接实现，
避免把 kernel 的实现错误一并抄进基准。另有 `_ref_exact_recurrence`
（逐 token 顺序递推）作为不依赖分块约定的第二基准。

容差：

| dtype | rtol | atol |
|---|---|---|
| float32 | 1e-3 | 1e-3 |
| float16 | 1e-2 | 1e-2 |
| bfloat16 | 3e-2 | 3e-2 |

bf16/fp16 容差较宽，原因是 `Out` 在两个 kernel 之间被舍入到输入 dtype 一次
（对角结果先落盘，再由非对角 kernel 读回累加）。输入按 $Q,K \sim N(0, 1/d)$、$V \sim N(0,1)$
构造，使输出量级为 O(1)，避免"输出接近 0 时任何容差都能过"的假通过。

四个 kernel 没有各自的下发入口，用例只能经 `_attention.apply` 整体触发。
下表给出每个 kernel 由哪些用例锁定，便于按 kernel 追溯：

| kernel | 锁定它的用例 |
|---|---|
|`_fwd_diag_kernel`|`test_causal_property`（因果掩码，逐位相等）、`test_single_block_matches_exact_recurrence`（`n == BLOCK` 时输出全部来自本 kernel）、`test_..._matches_reference` 的尾块用例（#10276 的 padding 重置）|
|`_fwd_kv_parallel`|`test_..._matches_reference` 中对 `kv` 全部前缀项的比对——该比对不经过 Q，与注意力输出解耦；`partial-block-left-shift` / `one-token-tail-block` / `tiny-single-cblock` 三个用例专打 `left_shift` 分支|
|`_fwd_kv_reduce`|同上的前缀项比对（扫描结果逐块校验）、`kv[:, :, 0]` 恒等于入参的排他扫描不变量、`three-blocks`（扫描多于一步）、非零 `kv_history` 用例、256 对齐分段等价|
|`_fwd_none_diag_kernel`|所有 `n > BLOCK` 的用例（跨块项只由它产生）、`test_zero_decay_multi_block_matches_exact_recurrence`、非零 `kv_history` 用例（历史项经由它回放）、256 对齐分段等价|

覆盖范围：

- 四个 kernel 的联合数值比对（`n` 小于/等于/大于 `BLOCK`，尾块只剩 1 个 token，
  `n` 非 `CBLOCK` 整数倍触发 `left_shift`）
- 返回的 `kv` 全部 `NUM_BLOCK+1` 项与前缀状态基准比对——单独锁定
  `_fwd_kv_parallel` 与 `_fwd_kv_reduce`，与输出正确性解耦
- 非零 `kv_history` 路径，含排他扫描首项恒等于入参的不变量
- `b`、`h` 取不相等且非 2 的幂（3 × 5），防 `//` 与 `%` 写反时碰巧算对
- `e != d`、`b > 1`、三个块以上的多块序列
- 不依赖分块约定的语义用例：`n == BLOCK`（无跨块对）与 `s == 0`（跨块因子退化为 1）
  分别与逐 token 精确递推比对
- 256 对齐的分段 prefill 与一次性 prefill 等价（串起 `kv_history` 的写与读）
- 因果性：改写 t 之后的 `V` 不得影响 t 之前的输出
- `block_size` 参数被忽略这一行为（传 64 与传 256 结果逐位相同）
- `jit_linear_forward_prefix` 的 layout 变换与 `kv_caches` 原地更新
- #10276 回归：所有用例都保留尾块并显式断言输出无 NaN

未覆盖及原因：

- **`d > 128` 的 d 维分块路径**：如「约束说明」所述，该路径当前语义就是错的
  （`kv_history` 偏移错位 + 只返回最后一块的 `kv`）。补用例等于给错误行为上锁，
  故只在文档记录；现网 `head_dim = 128` 不触及该路径。修复应作为独立 PR，
  届时同步补回归用例。
- **重编译行为**：无 `do_not_specialize` 导致的每 shape 重编译属性能特征，
  且检测手段依赖 Triton 内部 JIT 缓存结构、跨版本不稳定，不做成用例。
- **反向传播**：算子未实现 `backward`，推理场景不涉及。
- shape 网格刻意控制在 12 个左右的编译签名内：nightly conftest 对单文件累计 5 条
  超 120s 的用例会跳过该文件剩余全部用例，组合爆炸会导致"看起来全过了"。

## 变更记录

| PR | 说明 |
|---|---|
|[#8657](https://github.com/vllm-project/vllm-ascend/pull/8657)|引入四个 kernel，适配 BailingMoE 线性注意力|
|[#8702](https://github.com/vllm-project/vllm-ascend/pull/8702)|由 monkey-patch 改为 `PluggableLayer` 注册，调用方换成 `AscendBailingMoELinearAttention`|
|[#10276](https://github.com/vllm-project/vllm-ascend/pull/10276)|修复尾块 padding 未清零导致的 NaN：`_fwd_diag_kernel` 中对 `q` / `k` 在 `tl.load` 之后补 `tl.where` 重置|
