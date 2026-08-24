# SPDX-License-Identifier: Apache-2.0
# Numerical test for vllm_ascend.ops.triton.mamba.lightning_attn (Triton-Ascend)
# against a plain PyTorch fp32 reference.
# Requires NPU and Triton-Ascend.
#
# Covers the four kernels of the prefill pipeline, all of which are only
# reachable through _attention.apply:
#   _fwd_diag_kernel      intra-block causal attention
#   _fwd_kv_parallel      per-block decayed K^T V outer product
#   _fwd_kv_reduce        exclusive prefix scan of the block states
#   _fwd_none_diag_kernel cross-block contribution, accumulated onto Out
#
# See vllm_ascend/ops/triton/doc/lightning_attn.md for the operator spec.
#
# Regression scope: #10276 -- padding rows of the last, partially filled BLOCK
# were fed into tl.dot unmasked and produced NaN, so every case below keeps a
# tail block and the tail-only case (n = 257) is exercised explicitly.

import gc

import pytest
import torch
import torch_npu  # noqa: F401  # registers the npu backend / torch.npu namespace
from einops import rearrange

from vllm_ascend.ops.triton.mamba.lightning_attn import (
    AscendLightningAttentionKernel,
    lightning_attention_npu,
    lightning_attention_npu_,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

DEVICE = "npu"

# _attention.forward hardcodes this; the block_size argument of
# lightning_attention_npu / jit_linear_forward_prefix is ignored.
BLOCK = 256

# Inputs are drawn so that q @ k^T is O(1) and the output is O(1) as well:
# with q, k ~ N(0, 1/d) the dot product over d dims stays around 1, and the
# decay keeps the effective summation window short.  This matters -- with
# O(1e-3) outputs an atol of 1e-2 would pass on an all-zero kernel.
_DECAY_LO, _DECAY_HI = 0.01, 0.1

# bf16/fp16 inputs are accumulated in fp32 inside tl.dot, and the diagonal
# result is rounded to the output dtype once before _fwd_none_diag_kernel adds
# the cross-block part on top; the reference stays in fp32 throughout, so a
# small relative gap over an n-length reduction is expected.
_TOLERANCE = {
    torch.bfloat16: (3e-2, 3e-2),
    torch.float16: (1e-2, 1e-2),
    torch.float32: (1e-3, 1e-3),
}

# Every one of b, h, n, d, e is a tl.constexpr in all four kernels, so each
# distinct shape triggers four fresh Triton compilations.  The grid below is
# deliberately small and shared between tests: the nightly conftest kills the
# remaining cases of a file after five cases over 120s, which would silently
# skip the tail of this file.
SHAPE_CASES = [
    # n below one CBLOCK of _fwd_kv_parallel (64): num_blocks == 1 and the
    # left-shifted load reaches in front of the block.
    pytest.param(1, 4, 32, 64, 64, torch.bfloat16, id="tiny-single-cblock"),
    # n not a multiple of CBLOCK: exercises left_shift != 0 in _fwd_kv_parallel.
    pytest.param(1, 4, 100, 128, 128, torch.bfloat16, id="partial-block-left-shift"),
    # n == BLOCK exactly: no padding anywhere, single block.
    pytest.param(1, 4, 256, 128, 128, torch.float16, id="exactly-one-block-fp16"),
    # Tail block holding a single token -- the #10276 padding case at its worst.
    pytest.param(1, 4, 257, 64, 64, torch.bfloat16, id="one-token-tail-block"),
    # Two blocks with a ragged tail: cross-block decay plus padding together.
    pytest.param(1, 4, 300, 128, 128, torch.float32, id="ragged-two-blocks"),
    # b and h unequal and neither a power of two: catches off_bh // NUM_BLOCK
    # and off_bh % h being swapped, which powers of two can hide.
    pytest.param(3, 5, 70, 64, 64, torch.bfloat16, id="bh-non-power-of-two"),
    # Block-aligned multi-block, batch > 1, e != d.
    pytest.param(2, 4, 512, 64, 128, torch.float32, id="aligned-multi-block-e-ne-d"),
    # Three blocks: the prefix scan of _fwd_kv_reduce runs more than one step.
    pytest.param(1, 4, 768, 64, 64, torch.float32, id="three-blocks"),
]

# Reuses signatures from SHAPE_CASES so no extra kernel compilation is needed.
HISTORY_CASES = [
    pytest.param(1, 4, 100, 128, 128, torch.bfloat16, id="partial-block-left-shift"),
    pytest.param(1, 4, 300, 128, 128, torch.float32, id="ragged-two-blocks"),
]

PREFIX_CASES = [
    pytest.param(4, 100, 128, 128, torch.bfloat16, id="partial-block-left-shift"),
    pytest.param(4, 256, 128, 128, torch.float32, id="exactly-one-block"),
]


@pytest.fixture(autouse=True)
def _npu_env():
    init_device_properties_triton()
    yield
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def _randn(*shape, seed):
    """Draw on CPU, then move: NPU generators are not available everywhere."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=gen, dtype=torch.float32).to(DEVICE)


def _inputs(b, h, n, d, e, dtype, seed=0):
    """q, k scaled by 1/sqrt(d) so that q @ k^T -- and hence the output -- is O(1)."""
    q = _randn(b, h, n, d, seed=seed) / d**0.5
    k = _randn(b, h, n, d, seed=seed + 100) / d**0.5
    v = _randn(b, h, n, e, seed=seed + 200)
    return q.to(dtype), k.to(dtype), v.to(dtype)


def _decay(h, seed=0, lo=_DECAY_LO, hi=_DECAY_HI):
    gen = torch.Generator().manual_seed(seed + 300)
    return (lo + (hi - lo) * torch.rand(h, generator=gen, dtype=torch.float32)).to(DEVICE)


def _ref_lightning_attention(q, k, v, s, kv_history, block=BLOCK):
    """Straightforward torch reference, fp32 throughout.

    Deliberately written as a dense per-(batch, head) weight matrix rather than
    a re-implementation of the kernel's tiling: it is the oracle, so being
    obviously correct matters more than being fast, and mirroring the tiling
    (CBLOCK loop, ``left_shift`` boundary trick, decay pointer offsets) would
    make any bug in it invisible.  Test shapes are tiny.

    The four kernels together collapse to the closed form below.  Writing
    ``t``/``j`` for token positions and ``blk(x) = x // BLOCK``:

        o[t] = sum_{j <= t} w(t, j) * (q[t] . k[j]) * v[j]
               + exp(-s * t) * (q[t] @ kv_history)

        w(t, j) = exp(-s * (t - j))       if blk(t) == blk(j)
                  exp(-s * (t - j - 1))   otherwise

        kv_out = exp(-s * n) * kv_history
                 + sum_j exp(-s * (n - 1 - j)) * k[j] (x) v[j]

    Two kernel behaviours that are easy to miss are encoded here:

      * the cross-block exponent is ``t - j - 1``, not ``t - j``.  The block
        state built by _fwd_kv_parallel decays to the *last token* of its block
        (``k_decay = exp(-s * (BLOCK - 1 - j_local))``) while
        _fwd_none_diag_kernel re-inflates it by ``exp(-s * t_local)``, so a pair
        straddling a block boundary is weighted ``exp(s)`` higher than the same
        pair inside one block.  See the operator doc, 约束说明.
      * the exponent above is independent of where the block boundaries fall
        and of the length of the ragged tail block, because _fwd_kv_reduce
        advances the scan by the *actual* block length ``min(n - i*BLOCK, BLOCK)``.
    """
    b, h, n, d = q.shape
    e = v.shape[-1]
    q32, k32, v32 = q.float(), k.float(), v.float()
    hist32 = kv_history.float()
    s32 = s.float().reshape(-1)

    pos = torch.arange(n, device=q.device, dtype=torch.float32)
    gap = pos[:, None] - pos[None, :]
    causal = gap >= 0
    cross = (pos[:, None] // block) != (pos[None, :] // block)

    exponent = gap.clamp_min(0) - cross.float()

    o = torch.zeros(b, h, n, e, dtype=torch.float32, device=q.device)
    kv_out = torch.zeros(b, h, d, e, dtype=torch.float32, device=q.device)
    for bi in range(b):
        for hi in range(h):
            si = s32[hi]
            w = torch.where(causal, torch.exp(-si * exponent), torch.zeros_like(gap))

            scores = q32[bi, hi] @ k32[bi, hi].transpose(0, 1)
            o[bi, hi] = (scores * w) @ v32[bi, hi]
            o[bi, hi] += torch.exp(-si * pos)[:, None] * (q32[bi, hi] @ hist32[bi, hi])

            k_decay = torch.exp(-si * (n - 1 - pos))[:, None]
            kv_out[bi, hi] = torch.exp(-si * n) * hist32[bi, hi] + (k32[bi, hi] * k_decay).transpose(0, 1) @ v32[bi, hi]
    return o, kv_out


def _ref_block_states(k, v, s, kv_history, n, block=BLOCK):
    """Reference for the ``kv`` tensor returned alongside the output.

    Entry ``i`` is the state _fwd_kv_reduce leaves in front of block ``i`` --
    an *exclusive* scan, so entry 0 is the incoming history untouched -- and the
    trailing entry is the updated history.  Checking all of them pins down
    _fwd_kv_parallel and _fwd_kv_reduce separately from the attention output.
    """
    b, h, _, d = k.shape
    e = v.shape[-1]
    k32, v32 = k.float(), v.float()
    hist32 = kv_history.float()
    s32 = s.float().reshape(-1)
    num_blocks = (n + block - 1) // block

    states = torch.zeros(b, h, num_blocks + 1, d, e, dtype=torch.float32, device=k.device)
    for i in range(num_blocks + 1):
        # Tokens already consumed in front of block i; the trailing entry sees
        # the whole sequence.
        t_i = min(i * block, n)
        pos = torch.arange(t_i, device=k.device, dtype=torch.float32)
        for bi in range(b):
            for hi in range(h):
                si = s32[hi]
                acc = torch.exp(-si * t_i) * hist32[bi, hi]
                if t_i > 0:
                    decay = torch.exp(-si * (t_i - 1 - pos))[:, None]
                    acc = acc + (k32[bi, hi, :t_i] * decay).transpose(0, 1) @ v32[bi, hi, :t_i]
                states[bi, hi, i] = acc
    return states


def _ref_exact_recurrence(q, k, v, s):
    """Sequential linear-attention recurrence, with no notion of blocks.

    ``state[t] = exp(-s) * state[t-1] + k[t] (x) v[t]``, ``o[t] = q[t] @ state[t]``.

    This is the textbook semantics the operator claims to implement.  It agrees
    with the tiled algorithm only where no token pair straddles a BLOCK
    boundary (n <= BLOCK) or where the decay is disabled (s == 0), which is
    exactly how the two tests using it are set up -- it is a check of the
    algorithm's meaning that is independent of the kernel's tiling convention.
    Starts from a zero state, so the caller must pass a zero kv_history.
    """
    b, h, n, d = q.shape
    e = v.shape[-1]
    q32, k32, v32 = q.float(), k.float(), v.float()
    s32 = s.float().reshape(-1)

    decay = torch.exp(-s32).reshape(1, h, 1, 1)
    o = torch.zeros(b, h, n, e, dtype=torch.float32, device=q.device)
    state = torch.zeros(b, h, d, e, dtype=torch.float32, device=q.device)
    for t in range(n):
        state = decay * state + k32[:, :, t].unsqueeze(-1) @ v32[:, :, t].unsqueeze(-2)
        o[:, :, t] = (q32[:, :, t].unsqueeze(-2) @ state).squeeze(-2)
    return o, state


def _assert_close(actual, expected, dtype):
    rtol, atol = _TOLERANCE[dtype]
    torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize(("b", "h", "n", "d", "e", "dtype"), SHAPE_CASES)
@torch.inference_mode()
def test_lightning_attention_matches_reference(b, h, n, d, e, dtype):
    """Output and block states of the full four-kernel pipeline, zero history.

    This is the case that fails if any single kernel is wrong: the diagonal
    part is only correct if _fwd_diag_kernel is, the tail rows only if its
    padding mask is, and the cross-block part only if _fwd_kv_parallel and
    _fwd_kv_reduce agree on the decay convention.
    """
    q, k, v = _inputs(b, h, n, d, e, dtype)
    s = _decay(h).view(1, h, 1, 1)
    kv_history = torch.zeros(b, h, d, e, dtype=torch.float32, device=DEVICE)

    # _fwd_kv_reduce writes the updated history in place, so the reference has
    # to be given the pre-call value.
    o, kv = lightning_attention_npu_(q, k, v, s, kv_history.clone())
    o_ref, kv_ref = _ref_lightning_attention(q, k, v, s, kv_history)

    assert not torch.isnan(o).any(), "output contains NaN -- see #10276 (unmasked padding rows)"
    _assert_close(o, o_ref, dtype)
    _assert_close(kv[:, :, -1], kv_ref, dtype)

    states_ref = _ref_block_states(k, v, s, kv_history, n)
    assert kv.shape == states_ref.shape
    _assert_close(kv, states_ref, dtype)


@pytest.mark.parametrize(("b", "h", "n", "d", "e", "dtype"), HISTORY_CASES)
@torch.inference_mode()
def test_lightning_attention_matches_reference_with_history(b, h, n, d, e, dtype):
    """Same, with a non-zero incoming KV state (the decode-after-prefill path).

    Without this case the ``kv_pre`` load of _fwd_kv_reduce and the history term
    of the output are both dead code as far as the tests are concerned.
    """
    q, k, v = _inputs(b, h, n, d, e, dtype, seed=1)
    s = _decay(h, seed=1).view(1, h, 1, 1)
    kv_history = _randn(b, h, d, e, seed=2)

    o, kv = lightning_attention_npu_(q, k, v, s, kv_history.clone())
    o_ref, kv_ref = _ref_lightning_attention(q, k, v, s, kv_history)

    _assert_close(o, o_ref, dtype)
    _assert_close(kv[:, :, -1], kv_ref, dtype)
    # Entry 0 of the scan is exclusive: it must be the untouched input history.
    _assert_close(kv[:, :, 0], kv_history, dtype)


@torch.inference_mode()
def test_single_block_matches_exact_recurrence():
    """n == BLOCK: the tiled result must equal the textbook recurrence exactly.

    The main test compares against a reference that encodes the kernel's own
    cross-block decay convention.  Here no pair straddles a block boundary, so
    the convention drops out and this checks the operator actually computes
    decayed linear attention rather than something merely self-consistent.
    """
    b, h, n, d, e = 1, 4, 256, 128, 128
    dtype = torch.float32
    q, k, v = _inputs(b, h, n, d, e, dtype, seed=3)
    s = _decay(h, seed=3).view(1, h, 1, 1)
    kv_history = torch.zeros(b, h, d, e, dtype=torch.float32, device=DEVICE)

    o, kv = lightning_attention_npu_(q, k, v, s, kv_history.clone())
    o_ref, state_ref = _ref_exact_recurrence(q, k, v, s)

    _assert_close(o, o_ref, dtype)
    _assert_close(kv[:, :, -1], state_ref, dtype)


@torch.inference_mode()
def test_zero_decay_multi_block_matches_exact_recurrence():
    """s == 0 over three blocks: cross-block weights lose the exp(s) offset.

    With the decay disabled the tiled convention and the exact recurrence
    coincide for every pair, so this is the multi-block counterpart of the test
    above -- it would catch a block state that is scanned in the wrong
    direction or dropped, which a self-consistent reference cannot.
    """
    b, h, n, d, e = 1, 4, 768, 64, 64
    dtype = torch.float32
    q, k, v = _inputs(b, h, n, d, e, dtype, seed=4)
    s = torch.zeros(1, h, 1, 1, dtype=torch.float32, device=DEVICE)
    kv_history = torch.zeros(b, h, d, e, dtype=torch.float32, device=DEVICE)

    o, kv = lightning_attention_npu_(q, k, v, s, kv_history.clone())
    o_ref, state_ref = _ref_exact_recurrence(q, k, v, s)

    _assert_close(o, o_ref, dtype)
    _assert_close(kv[:, :, -1], state_ref, dtype)


@torch.inference_mode()
def test_block_aligned_split_is_equivalent_to_one_shot():
    """Feeding 2 x BLOCK tokens in two calls must equal one call of 2 * BLOCK.

    This is the chunked-prefill invariant the KV history exists for.  It holds
    only on a BLOCK-aligned split -- see the operator doc, 约束说明 -- and it
    ties the history written by _fwd_kv_reduce to the history consumed on the
    next call, an end-to-end property no single-call test covers.
    """
    b, h, n, d, e = 2, 4, 512, 64, 128
    dtype = torch.float32
    q, k, v = _inputs(b, h, n, d, e, dtype, seed=5)
    s = _decay(h, seed=5).view(1, h, 1, 1)
    kv_history = torch.zeros(b, h, d, e, dtype=torch.float32, device=DEVICE)

    o_full, kv_full = lightning_attention_npu_(q, k, v, s, kv_history.clone())

    o_head, kv_head = lightning_attention_npu_(
        q[:, :, :BLOCK], k[:, :, :BLOCK], v[:, :, :BLOCK], s, kv_history.clone()
    )
    o_tail, kv_tail = lightning_attention_npu_(
        q[:, :, BLOCK:],
        k[:, :, BLOCK:],
        v[:, :, BLOCK:],
        s,
        kv_head[:, :, -1].contiguous(),
    )

    _assert_close(torch.cat([o_head, o_tail], dim=2), o_full, dtype)
    _assert_close(kv_tail[:, :, -1], kv_full[:, :, -1], dtype)


@torch.inference_mode()
def test_causal_property():
    """Rewriting v at positions >= t must not move the output before t.

    Guards the causal mask of _fwd_diag_kernel directly; a mask that leaks one
    position would still match a reference that made the same mistake.
    """
    b, h, n, d, e = 1, 4, 256, 128, 128
    cut = 50
    q, k, v = _inputs(b, h, n, d, e, torch.float32, seed=6)
    s = _decay(h, seed=6).view(1, h, 1, 1)
    kv_history = torch.zeros(b, h, d, e, dtype=torch.float32, device=DEVICE)

    o_orig, _ = lightning_attention_npu_(q, k, v, s, kv_history.clone())

    v_scrambled = v.clone()
    v_scrambled[:, :, cut:] = _randn(b, h, n - cut, e, seed=7)
    # Guard the guard: if the rewrite happened to reproduce v, the assertion
    # below would hold for the wrong reason.
    assert not torch.allclose(v_scrambled[:, :, cut:], v[:, :, cut:]), "fixture no longer perturbs the future"

    o_scrambled, _ = lightning_attention_npu_(q, k, v_scrambled, s, kv_history.clone())
    torch.testing.assert_close(o_orig[:, :, :cut], o_scrambled[:, :, :cut], rtol=0, atol=0)


@torch.inference_mode()
def test_block_size_argument_is_ignored():
    """lightning_attention_npu ignores block_size; _attention.forward pins 256.

    Recorded as a test because the argument is part of the public signature and
    a caller passing 64 would reasonably expect 64-token blocks, which changes
    the cross-block weights.  If the argument is ever honoured, this fails and
    the operator doc has to be updated with it.
    """
    b, h, n, d, e = 1, 4, 100, 128, 128
    dtype = torch.bfloat16
    q, k, v = _inputs(b, h, n, d, e, dtype, seed=8)
    ed = _decay(h, seed=8)

    o_256, kv_256 = lightning_attention_npu(q, k, v, ed, block_size=256, kv_history=None)
    o_64, kv_64 = lightning_attention_npu(q, k, v, ed, block_size=64, kv_history=None)

    torch.testing.assert_close(o_64.float(), o_256.float(), rtol=0, atol=0)
    torch.testing.assert_close(kv_64.float(), kv_256.float(), rtol=0, atol=0)


@pytest.mark.parametrize(("h", "n", "d", "e", "dtype"), PREFIX_CASES)
@torch.inference_mode()
def test_jit_linear_forward_prefix_matches_reference(h, n, d, e, dtype):
    """The production entry point: [h, n, d] in, [n, h*e] out, cache in place.

    Covers the two things the wrapper adds on top of the kernels -- the
    unsqueeze/rearrange layout change and the copy of the final KV state back
    into kv_caches -- which a test calling _attention.apply directly cannot.
    """
    q, k, v = _inputs(1, h, n, d, e, dtype, seed=9)
    q, k, v = q.squeeze(0), k.squeeze(0), v.squeeze(0)
    slope_rate = _decay(h, seed=9)
    kv_caches = _randn(h, d, e, seed=10)

    kv_caches_out = kv_caches.clone()
    out = AscendLightningAttentionKernel.jit_linear_forward_prefix(
        q.clone(), k.clone(), v.clone(), kv_caches_out, slope_rate.clone(), block_size=BLOCK
    )

    o_ref, kv_ref = _ref_lightning_attention(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        slope_rate,
        kv_caches.reshape(1, h, d, e),
    )
    out_ref = rearrange(o_ref.squeeze(0), "h n d -> n (h d)")

    assert out.shape == (n, h * e)
    _assert_close(out, out_ref, dtype)
    _assert_close(kv_caches_out, kv_ref.reshape(h, d, e), dtype)
