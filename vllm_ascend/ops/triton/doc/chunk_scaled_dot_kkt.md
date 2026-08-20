# ChunkScaledDotKkt

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列加速卡产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

> 310P 上 `HAS_TRITON` 为 False，GDN 走 `vllm_ascend/_310p/ops/fla/` 下的 PyTorch 实现，
> 不经过本算子。
> Ascend 950 与 A2/A3 的差异仅在下发参数：`A5DeviceAdaptor` 额外传入
> `disable_tightly_coupled_buffer_reuse=True`，计算逻辑一致。

## 功能说明

- API 功能：`ChunkScaledDotKkt` 是 GDN（Gated Delta Net）WY 表示的第一级算子，
  按 `(batch, head, chunk)` 分块计算带门控衰减与 beta 缩放的 $KK^T$ 严格下三角矩阵。
  其输出 $A$ 交由下游 `solve_tril` 对 $(I - A)$ 求逆。

- 计算公式：

    $$
    A_{ij} = \begin{cases}
    \beta_i \cdot \exp(g_i - g_j) \cdot (K_i \cdot K_j^T), & i > j \ \text{且}\ g_i - g_j \le 0 \\
    0, & \text{其他}
    \end{cases}
    $$

    其中 $i, j$ 为 chunk 内的行列下标，$\beta$ 为 delta rule 的写入强度，
    $g$ 为门控的 chunk 内累加和（`chunk_local_cumsum` 的输出）。

- 调用链：

    ```
    AscendGatedDeltaNetAttention          vllm_ascend/ops/gdn.py
      └─ chunk_gated_delta_rule_fwd       vllm_ascend/ops/triton/fla/chunk.py
           └─ chunk_scaled_dot_kkt_fwd    本算子 Python wrapper
                └─ DeviceOperator.chunk_scaled_dot_kkt_fwd
                     ├─ BaseDeviceAdaptor   vllm_ascend/device/device_op.py  (A2/A3)
                     └─ A5DeviceAdaptor     vllm_ascend/device/device_op.py  (Ascend 950)
                          └─ chunk_scaled_dot_kkt_fwd_kernel   Triton kernel
    ```

- 任务划分：kernel 采用 `(num_core,)` 一维 grid，将 `NT * B * H` 个任务展平后
  以 `tl.range(core_id, task_num, num_core)` 的步长方式 round-robin 分配到各 AI Core，
  chunk 维与 batch/head 维同时并行。

## 参数说明

### Python 接口 `chunk_scaled_dot_kkt_fwd`

| 参数名 |输入/输出/属性| 描述 | 数据类型 |数据格式|
|-------|------------|------|---------|-----|
|k|输入|公式中的 $K$，shape 为 [B, T, Hg, K]。|BFLOAT16/FLOAT16|ND|
|beta|输入|公式中的 $\beta$，shape 为 [B, T, H]。内部会 permute 为 [H, B, T] 并 contiguous 后下发。|BFLOAT16/FLOAT16|ND|
|g_cumsum|可选输入|公式中的 $g$，门控的 chunk 内累加和，shape 为 [B, T, H]。内部 permute 为 [H, B, T]。签名标注为可选，但当前**不可传 None**，详见约束说明。|BFLOAT16/FLOAT16/FLOAT32|ND|
|cu_seqlens|可选输入|变长场景下各 batch 的 token 数累加和，维度为 N+1。为 None 时按定长 [B, T] 处理。|INT32|ND|
|chunk_indices|可选输入|变长场景下的 chunk 索引表，shape 为 [NT, 2]，每行为 (序列号, 序列内 chunk 号)。为 None 且 `cu_seqlens` 非 None 时由 `prepare_chunk_indices` 内部生成。|INT32|ND|
|chunk_size|可选属性|分块长度 $BT$，默认值 64，当前调用方固定传 64。|INT32|-|
|output_dtype|可选属性|输出 $A$ 的数据类型，默认 `torch.float32`。|-|-|
|A|输出|公式中的 $A$，shape 为 [B, T, H, BT]。每个 token 行保存其所在 chunk 内的一行严格下三角值。|FLOAT32|ND|

### Kernel 接口 `chunk_scaled_dot_kkt_fwd_kernel`

| 参数名 |输入/输出/属性| 描述 | 数据类型 |
|-------|------------|------|---------|
|k|输入|shape [B, T, Hg, K]。|BFLOAT16/FLOAT16|
|beta|输入|已 permute 为 [H, B, T]。|BFLOAT16/FLOAT16|
|g_cumsum|可选输入|已 permute 为 [H, B, T]，为 None 时 `USE_G` 置 False，跳过门控项。|BFLOAT16/FLOAT16/FLOAT32|
|A|输出|shape [B, T, H, BT]，由调用方预分配。|FLOAT32|
|cu_seqlens|可选输入|同上，为 None 时 `IS_VARLEN` 置 False。|INT32|
|chunk_indices|可选输入|同上，仅 `IS_VARLEN` 为 True 时读取。|INT32|
|T|属性|定长场景为单 batch 序列长度；变长场景为总 token 数，kernel 内按 `cu_seqlens` 重新赋值为当前序列长度。`do_not_specialize`。|INT32|
|B|属性|batch 数，变长场景固定为 1。`do_not_specialize`。|INT32|
|bh_step|属性|`B * H`，用于 `task_id` 到 `(chunk, batch, head)` 的拆分。`do_not_specialize`。|INT32|
|task_num|属性|`NT * B * H`，任务总数。`do_not_specialize`。|INT32|
|num_core|属性|参与计算的 AI Core 数，取自 `get_aicore_num()`，同时作为 grid 大小与 round-robin 步长。`do_not_specialize`。|INT32|
|H|属性|`tl.constexpr`，query/gate 的 head 数。|INT32|
|Hg|属性|`tl.constexpr`，$K$ 的 head 数，GQA 场景下 `H` 为其整数倍。|INT32|
|K|属性|`tl.constexpr`，head dim。|INT32|
|BT|属性|`tl.constexpr`，分块长度，当前为 64。|INT32|
|BK|属性|`tl.constexpr`，K 方向分块宽度，wrapper 硬编码为 128。|INT32|
|IS_VARLEN|属性|`tl.constexpr`，由 heuristic 按 `cu_seqlens is not None` 推导。|BOOL|
|USE_G|属性|`tl.constexpr`，由 heuristic 按 `g_cumsum is not None` 推导。|BOOL|

## 约束说明

- 该接口支持推理场景下使用，仅前向，无反向。
- `H` 必须能被 `Hg` 整除，head 映射关系为 `i_h // (H // Hg)`。
- 变长场景（`cu_seqlens` 非 None）要求 `B` 为 1，`k` 的 shape 为 [1, total_tokens, Hg, K]。
  kernel 内 `bt_stride = B * T` 在进入任务循环前计算，随后 `T` 才被按序列重新赋值，二者不可混淆。
- `cu_seqlens` 维度为 N+1，要求后一元素不小于前一元素，取值为当前及前序 batch 有效 token 数的累加和。
- `BT` 当前仅验证过 64；`BK` 由 wrapper 硬编码为 128，`K` 大于 128 时通过多次循环累加。
- **`g_cumsum` 当前不可传 None**：wrapper 中
  `g_cumsum=torch.permute(g_cumsum, (2, 0, 1)).contiguous()` 为无条件执行，
  传 None 会在 kernel 下发前抛 `AttributeError`。kernel 侧的 `USE_G=False` 分支本身可用，
  但经 wrapper 不可达。当前唯一调用方 `chunk.py` 始终传入 `g`，故不影响现网。
- 门控项使用 `safe_exp(x) = exp(x) if x <= 0 else 0`，**不是**普通 `exp`。
  在门控单调非增时 $g_i - g_j \le 0$ 恒成立，该置零分支不会触发；
  但若上游门控语义变更，该分支的行为与普通 `exp` 存在差异。
- 输出 $A$ 为**严格下三角**：对角线及上三角恒为 0。下游 `solve_tril` 依赖该性质。
- 长度不足 `BT` 的尾块，输出中列号大于等于该块实际长度的部分恒为 0
  （由 `boundary_check` 的零填充加载保证）。
- 调用前必须先执行 `init_device_properties_triton()`，否则 `get_aicore_num()` 断言失败。
- `bh_step` / `task_num` / `num_core` 必须保持为运行时参数并带 `do_not_specialize`，
  若改回 `tl.constexpr` 会导致每个新 batch shape 触发一次重编译。

## 测试说明

数值精度用例：`tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_chunk_scaled_dot_kkt.py`

```bash
pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_chunk_scaled_dot_kkt.py
```

用例以 PyTorch fp32 朴素实现为基准做逐元素比对，容差：

| dtype | rtol | atol |
|---|---|---|
| bfloat16 | 1e-2 | 1e-2 |
| float16 | 2e-3 | 2e-3 |

覆盖范围：定长与变长两条路径、`chunk_indices` 内部生成与外部预建、GQA head 映射、
非对齐尾块、`K > BK`、`safe_exp` 置零分支、`USE_G=False` 分支、严格下三角结构不变量，
以及针对 #10033 的 `task_num` 相对 `num_core` 不整除、多 batch/多 head 任务分解。

未覆盖：#11577 的重编译行为属于性能特征而非计算正确性，且检测手段依赖 Triton
内部的 JIT 缓存结构、跨版本不稳定，故未纳入用例。该约束以文档形式记录在上方
「约束说明」中。

## 变更记录

| PR | 说明 |
|---|---|
|[#10033](https://github.com/vllm-project/vllm-ascend/pull/10033)|grid 由 `(NT, 1)` + 串行 `for i_bh in range(B*H)` 改为 `(num_core,)` + `tl.range` 全核 round-robin|
|[#11577](https://github.com/vllm-project/vllm-ascend/pull/11577)|`bh_step` / `task_num` / `num_core` 由 `tl.constexpr` 降级为运行时参数，消除重编译|
