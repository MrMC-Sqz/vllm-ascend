# SPDX-License-Identifier: Apache-2.0
# Numerical test for vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt (Triton-Ascend)
# against a plain PyTorch fp32 reference.
# Requires NPU and Triton-Ascend.
#
# See vllm_ascend/ops/triton/doc/chunk_scaled_dot_kkt.md for the operator spec.
#
# Regression scope:
#   * #10033 -- grid moved from (NT, 1) with a serial `for i_bh in range(B*H)`
#     loop to (num_core,) with `tl.range(core_id, task_num, num_core)`.
#   * #11577 -- bh_step/task_num/num_core demoted from tl.constexpr to runtime
#     args to stop recompilation.

import gc

import pytest
import torch
import torch_npu  # noqa: F401  # registers the npu backend / torch.npu namespace

from vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
    chunk_scaled_dot_kkt_fwd_kernel,
)
from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices
from vllm_ascend.ops.triton.triton_utils import get_aicore_num, init_device_properties_triton

DEVICE = "npu"
CHUNK_SIZE = 64
# chunk_scaled_dot_kkt_fwd hardcodes the K-loop block width.
BLOCK_K = 128

# bf16/fp16 inputs are accumulated in fp32 inside tl.dot; the reference upcasts
# first, so a small relative gap over a K-length reduction is expected.
_TOLERANCE = {
    torch.bfloat16: (1e-2, 1e-2),
    torch.float16: (2e-3, 2e-3),
}

SHAPE_CASES = [
    pytest.param(1, 64, 1, 1, 64, id="single-chunk"),
    pytest.param(2, 256, 4, 4, 128, id="multi-chunk-mha"),
    pytest.param(2, 200, 4, 4, 128, id="ragged-tail"),
    pytest.param(1, 128, 8, 2, 64, id="gqa-group4"),
    pytest.param(1, 128, 2, 2, 256, id="k-gt-block-k"),
    pytest.param(1, 33, 2, 2, 64, id="t-lt-chunk"),
]

VARLEN_CASES = [
    pytest.param([64], 2, 2, 64, id="one-seq-aligned"),
    pytest.param([37, 91, 128], 4, 4, 128, id="three-seqs-ragged"),
    pytest.param([1, 200, 63, 65], 2, 1, 64, id="degenerate-and-gqa"),
]


@pytest.fixture(autouse=True)
def _npu_env():
    init_device_properties_triton()
    yield
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def _ref_chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_size: int = CHUNK_SIZE,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Straightforward torch reference, fp32 throughout.

    Deliberately loop-based rather than vectorised: it is the oracle, so being
    obviously correct matters more than being fast.  Test shapes are tiny.

    Mirrors two kernel behaviours that are easy to miss:
      * ``safe_exp(x) = exp(x) if x <= 0 else 0`` -- not a plain ``exp``.
      * a chunk shorter than ``BT`` leaves columns ``>= n`` at zero, because the
        kernel's block loads are zero-padded by ``boundary_check``.
    """
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size
    group = H // Hg

    A = torch.zeros(B, T, H, BT, dtype=torch.float32, device=k.device)
    if cu_seqlens is None:
        spans = [(b, 0, T) for b in range(B)]
    else:
        bounds = cu_seqlens.tolist()
        spans = [(0, bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    k_f32 = k.float()
    beta_f32 = beta.float()
    g_f32 = None if g_cumsum is None else g_cumsum.float()

    for b, bos, eos in spans:
        seq_len = eos - bos
        for h in range(H):
            k_h = k_f32[b, bos:eos, h // group, :]
            beta_h = beta_f32[b, bos:eos, h]
            g_h = None if g_f32 is None else g_f32[b, bos:eos, h]
            for start in range(0, seq_len, BT):
                end = min(start + BT, seq_len)
                n = end - start
                block = k_h[start:end]
                a = block @ block.transpose(0, 1)
                if g_h is not None:
                    diff = g_h[start:end, None] - g_h[None, start:end]
                    a = a * torch.where(diff <= 0, torch.exp(diff), torch.zeros_like(diff))
                a = a * beta_h[start:end, None]
                pos = torch.arange(n, device=a.device)
                a = torch.where(pos[:, None] > pos[None, :], a, torch.zeros_like(a))
                A[b, bos + start : bos + end, h, :n] = a

    return A.to(output_dtype)


def _make_inputs(B, T, H, Hg, K, dtype, seed=0, monotonic_g=True):
    """Build GDN-shaped inputs: beta in (0, 1), g_cumsum non-positive."""
    torch.manual_seed(seed)
    # Keep |k| modest so the K-length fp32 accumulation stays well conditioned.
    k = (torch.randn(B, T, Hg, K, device=DEVICE, dtype=torch.float32) * 0.25).to(dtype)
    beta = torch.sigmoid(torch.randn(B, T, H, device=DEVICE, dtype=torch.float32)).to(dtype)
    if monotonic_g:
        # log-sigmoid style decay, cumulated along T -> non-increasing
        decay = -torch.nn.functional.softplus(torch.randn(B, T, H, device=DEVICE, dtype=torch.float32)) * 0.05
        g_cumsum = decay.cumsum(dim=1).to(dtype)
    else:
        # Non-monotonic on purpose: exercises the ``x > 0 -> 0`` half of safe_exp.
        g_cumsum = (torch.randn(B, T, H, device=DEVICE, dtype=torch.float32) * 0.5).to(dtype)
    return k, beta, g_cumsum


def _assert_close(actual, expected, dtype):
    rtol, atol = _TOLERANCE[dtype]
    torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(("B", "T", "H", "Hg", "K"), SHAPE_CASES)
@torch.inference_mode()
def test_fixed_length_matches_reference(B, T, H, Hg, K, dtype):
    """Non-varlen path, through the public wrapper.

    ``ragged-tail`` (T=200 -> chunks 64/64/64/8) and ``t-lt-chunk`` cover the
    zero-padded boundary_check loads; ``k-gt-block-k`` (K=256 > BK=128) forces
    more than one iteration of the K accumulation loop; ``gqa-group4`` covers
    the ``i_h // (H // Hg)`` head mapping.
    """
    k, beta, g_cumsum = _make_inputs(B, T, H, Hg, K, dtype)

    actual = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g_cumsum, cu_seqlens=None, chunk_size=CHUNK_SIZE, output_dtype=torch.float32
    )
    expected = _ref_chunk_scaled_dot_kkt_fwd(k, beta, g_cumsum, cu_seqlens=None)

    assert actual.shape == (B, T, H, CHUNK_SIZE)
    assert actual.dtype == torch.float32
    _assert_close(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(("seqlens", "H", "Hg", "K"), VARLEN_CASES)
@pytest.mark.parametrize("prebuilt_indices", [False, True], ids=["indices-derived", "indices-prebuilt"])
@torch.inference_mode()
def test_varlen_matches_reference(seqlens, H, Hg, K, dtype, prebuilt_indices):
    """Varlen path (IS_VARLEN=True), with and without prebuilt chunk_indices.

    ``indices-derived`` exercises the wrapper's internal
    ``prepare_chunk_indices`` call; ``indices-prebuilt`` mirrors how
    ``chunk_gated_delta_rule_fwd`` actually calls it (metadata prebuilt by the
    attention builder).  A ``seqlen`` of 1 checks a chunk whose only row is on
    the diagonal, i.e. the output must be all zeros for that sequence.
    """
    total = sum(seqlens)
    cu_seqlens = torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], device=DEVICE, dtype=torch.int32)
    k, beta, g_cumsum = _make_inputs(1, total, H, Hg, K, dtype, seed=1)

    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE) if prebuilt_indices else None

    actual = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g_cumsum=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=CHUNK_SIZE,
        output_dtype=torch.float32,
    )
    expected = _ref_chunk_scaled_dot_kkt_fwd(k, beta, g_cumsum, cu_seqlens=cu_seqlens)

    _assert_close(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_non_monotonic_gate_zeroes_positive_diff(dtype):
    """``safe_exp`` must return 0 -- not ``exp(x)`` -- when ``g_i - g_j > 0``.

    With a monotonically decaying gate this branch is unreachable, so a plain
    ``exp`` would pass every other test in this file.
    """
    B, T, H, K = 1, 128, 2, 64
    k, beta, g_cumsum = _make_inputs(B, T, H, H, K, dtype, seed=2, monotonic_g=False)

    actual = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g_cumsum, cu_seqlens=None, chunk_size=CHUNK_SIZE, output_dtype=torch.float32
    )
    expected = _ref_chunk_scaled_dot_kkt_fwd(k, beta, g_cumsum, cu_seqlens=None)

    # Guard the guard: the random gate must actually produce a positive diff
    # inside the strict lower triangle, otherwise this test proves nothing.
    g = g_cumsum.float()[0, :CHUNK_SIZE, 0]
    diff = g[:, None] - g[None, :]
    pos = torch.arange(CHUNK_SIZE, device=diff.device)
    assert (diff > 0)[pos[:, None] > pos[None, :]].any(), "fixture no longer exercises the safe_exp zero branch"

    _assert_close(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_use_g_false_branch_via_device_operator(dtype):
    """USE_G=False, reached by launching through ``DeviceOperator`` directly.

    The wrapper cannot reach this branch: ``chunk_scaled_dot_kkt_fwd`` calls
    ``torch.permute(g_cumsum, ...)`` unconditionally, so passing the documented
    ``g_cumsum=None`` raises AttributeError before the kernel is launched.  The
    heuristic and the ``if USE_G:`` block exist regardless, so cover the branch
    at the launch layer and keep the wrapper gap documented rather than
    untested.
    """
    from vllm_ascend.device.device_op import DeviceOperator

    B, T, H, K = 2, 192, 2, 64
    k, beta, _ = _make_inputs(B, T, H, H, K, dtype, seed=7)
    num_tasks = (T // CHUNK_SIZE) * B * H
    A = torch.empty(B, T, H, CHUNK_SIZE, device=k.device, dtype=torch.float32)

    actual = DeviceOperator.chunk_scaled_dot_kkt_fwd(
        num_core=get_aicore_num(),
        bh_step=B * H,
        task_num=num_tasks,
        k=k,
        beta=torch.permute(beta, (2, 0, 1)).contiguous(),
        g_cumsum=None,
        A=A,
        cu_seqlens=None,
        chunk_indices=None,
        T=T,
        B=B,
        H=H,
        Hg=H,
        K=K,
        BT=CHUNK_SIZE,
        BK=BLOCK_K,
    )
    expected = _ref_chunk_scaled_dot_kkt_fwd(k, beta, g_cumsum=None, cu_seqlens=None)

    _assert_close(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_output_is_strictly_lower_triangular(dtype):
    """Structural invariant: diagonal and upper triangle are exactly zero.

    Downstream ``solve_tril`` inverts (I - A) assuming strict lower triangular
    input, so a non-zero diagonal is silently wrong rather than loudly wrong.
    """
    B, T, H, K = 2, 192, 2, 64
    k, beta, g_cumsum = _make_inputs(B, T, H, H, K, dtype, seed=3)

    actual = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g_cumsum, cu_seqlens=None, chunk_size=CHUNK_SIZE, output_dtype=torch.float32
    )

    row_in_chunk = torch.arange(T, device=actual.device) % CHUNK_SIZE
    col = torch.arange(CHUNK_SIZE, device=actual.device)
    upper = col[None, :] >= row_in_chunk[:, None]  # [T, BT], includes diagonal
    masked = actual.masked_select(upper[None, :, None, :].expand_as(actual))
    assert torch.count_nonzero(masked) == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("task_offset", [-1, 0, 1], ids=["under-core", "exact-core", "over-core"])
@torch.inference_mode()
def test_task_num_not_divisible_by_num_core(dtype, task_offset):
    """Regression for #10033: round-robin ``tl.range(core_id, task_num, num_core)``.

    Shapes are derived from the live core count so that ``task_num`` lands just
    below / exactly on / just above a multiple of ``num_core``.  Under the old
    ``(NT, 1)`` grid this dimension was not parallelised at all, so an
    off-by-one in the new task->(chunk, batch, head) decomposition would only
    show up when the division is uneven.
    """
    num_core = get_aicore_num()
    B, H, K = 1, 1, 64
    # task_num = NT * B * H = NT, so drive NT directly via T.
    num_tasks = max(1, num_core * 2 + task_offset)
    T = num_tasks * CHUNK_SIZE
    k, beta, g_cumsum = _make_inputs(B, T, H, H, K, dtype, seed=4)

    actual = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g_cumsum, cu_seqlens=None, chunk_size=CHUNK_SIZE, output_dtype=torch.float32
    )
    expected = _ref_chunk_scaled_dot_kkt_fwd(k, beta, g_cumsum, cu_seqlens=None)

    _assert_close(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_multi_head_task_decomposition(dtype):
    """Regression for #10033, batch/head axis.

    ``task_id -> (i_t_i, i_bh) -> (i_b, i_h)`` is the part the PR rewrote.
    B and H are chosen unequal and non-power-of-two so a swapped ``//``/``%``
    or a transposed (i_b, i_h) split cannot coincidentally produce the right
    answer.
    """
    B, T, H, Hg, K = 3, 320, 5, 5, 64
    k, beta, g_cumsum = _make_inputs(B, T, H, Hg, K, dtype, seed=5)

    actual = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g_cumsum, cu_seqlens=None, chunk_size=CHUNK_SIZE, output_dtype=torch.float32
    )
    expected = _ref_chunk_scaled_dot_kkt_fwd(k, beta, g_cumsum, cu_seqlens=None)

    _assert_close(actual, expected, dtype)


def _kernel_cache_size() -> int | None:
    """Total number of compiled variants held by the JIT function, or None.

    Triton has moved this around between releases; return None so the caller
    can skip rather than assert against an attribute that no longer exists.
    """
    for attr in ("device_caches", "cache"):
        cache = getattr(chunk_scaled_dot_kkt_fwd_kernel, attr, None)
        if isinstance(cache, dict):
            total = 0
            for entry in cache.values():
                # device_caches values are tuples whose first element is the dict
                target = entry[0] if isinstance(entry, tuple) and entry and isinstance(entry[0], dict) else entry
                if isinstance(target, dict):
                    total += len(target)
            return total
    return None


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_varying_task_num_does_not_recompile(dtype):
    """Regression for #11577: ``bh_step``/``task_num``/``num_core`` are runtime args.

    If they regress to ``tl.constexpr`` (or lose ``do_not_specialize``), every
    new batch shape becomes a fresh compilation -- a latency cliff on the first
    token of each new shape, not a wrong answer, so no numerical test catches it.
    """
    if _kernel_cache_size() is None:
        pytest.skip("triton JITFunction cache layout not recognised on this version")

    B, H, K = 1, 1, 64
    shapes = [4, 7, 11, 23]

    # Warm up on the first shape so the baseline excludes the initial compile.
    k, beta, g_cumsum = _make_inputs(B, shapes[0] * CHUNK_SIZE, H, H, K, dtype, seed=6)
    chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g_cumsum=g_cumsum, cu_seqlens=None, chunk_size=CHUNK_SIZE)
    baseline = _kernel_cache_size()

    for nt in shapes[1:]:
        k, beta, g_cumsum = _make_inputs(B, nt * CHUNK_SIZE, H, H, K, dtype, seed=6)
        chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g_cumsum=g_cumsum, cu_seqlens=None, chunk_size=CHUNK_SIZE)

    assert _kernel_cache_size() == baseline, (
        "kernel recompiled when task_num/bh_step/num_core changed; "
        "these must stay runtime args with do_not_specialize (see #11577)"
    )
