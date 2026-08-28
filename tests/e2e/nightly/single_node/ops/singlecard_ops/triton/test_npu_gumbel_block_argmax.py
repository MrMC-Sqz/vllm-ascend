# SPDX-License-Identifier: Apache-2.0
# Numerical test for the `_npu_gumbel_block_argmax` device function in
# vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils, against a plain
# PyTorch fp32 reference.
# Requires NPU and Triton-Ascend.
#
# See vllm_ascend/worker/v2/spec_decode/doc/npu_gumbel_block_argmax.md for the
# operator spec.
#
# Regression scope: #9155 -- the main2main import that brought the whole MRV2
# rejection sampler in, with no numerical coverage for this function.
#
# `_npu_gumbel_block_argmax` is a `@triton.jit` *device* function: it has no
# `tl.program_id` and receives `logits` / `block` / `mask` as already-loaded
# values, so it cannot be launched from host.  `_gumbel_probe_kernel` below is a
# test-only shell that supplies exactly what the sole production caller
# (`_resample_kernel`) supplies, which makes the function reachable on its own --
# including the two branches that caller can never reach (`processed_logits` and
# `APPLY_TEMPERATURE=True`).
#
# The Gumbel noise comes from Triton's philox and has no PyTorch equivalent, so
# `_gumbel_noise_probe_kernel` replays the *same* stream and the reference
# consumes it.  That pins everything except the RNG itself (temperature scaling,
# the processed-logits store, masking, the -inf branch, the block-local argmax).
# The RNG is covered independently and statistically by
# `test_gumbel_argmax_follows_softmax_distribution`.

import gc

import pytest
import torch
import torch_npu  # noqa: F401  # registers the npu backend / torch.npu namespace
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import _npu_gumbel_block_argmax

DEVICE = "npu"

# Everything is fp32 end to end; the slack is only for reduction-order drift.
_RTOL = 1e-5
_ATOL = 1e-5

# The vocabulary block width the only production caller uses -- `rejection_sample`
# launches `_resample_kernel` with 1024 -- mirrored so the probe exercises the
# same tiling the device function really sees.
PROBE_BLOCK_SIZE = 1024


@pytest.fixture(autouse=True)
def _npu_env():
    init_device_properties_triton()
    yield
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Probe kernels (test-only harness)
# ---------------------------------------------------------------------------


@triton.jit
def _gumbel_probe_kernel(
    out_value_ptr,
    out_value_stride,
    out_idx_ptr,
    out_idx_stride,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    processed_logits_ptr,
    processed_logits_stride,
    processed_logits_col_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    # 0 = no processed_logits (mirrors the real call site in _resample_kernel)
    # 1 = processed_logits, implicit column 0
    # 2 = processed_logits, column read from processed_logits_col_ptr
    PROCESSED_MODE: tl.constexpr,
):
    """Thin host-launchable wrapper around the `_npu_gumbel_block_argmax` device function.

    The three `PROCESSED_MODE` variants pass literal `None`s rather than relying
    on `None` surviving a kernel-argument boundary, so each variant compiles the
    same way the production call site does.
    """
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)

    if PROCESSED_MODE == 0:
        value, idx = _npu_gumbel_block_argmax(
            logits,
            block,
            mask,
            token_idx,
            expanded_idx_mapping_ptr,
            temp_ptr,
            seeds_ptr,
            pos_ptr,
            None,
            0,
            None,
            vocab_size,
            APPLY_TEMPERATURE=APPLY_TEMPERATURE,
        )
    elif PROCESSED_MODE == 1:
        value, idx = _npu_gumbel_block_argmax(
            logits,
            block,
            mask,
            token_idx,
            expanded_idx_mapping_ptr,
            temp_ptr,
            seeds_ptr,
            pos_ptr,
            processed_logits_ptr,
            processed_logits_stride,
            None,
            vocab_size,
            APPLY_TEMPERATURE=APPLY_TEMPERATURE,
        )
    else:
        value, idx = _npu_gumbel_block_argmax(
            logits,
            block,
            mask,
            token_idx,
            expanded_idx_mapping_ptr,
            temp_ptr,
            seeds_ptr,
            pos_ptr,
            processed_logits_ptr,
            processed_logits_stride,
            processed_logits_col_ptr,
            vocab_size,
            APPLY_TEMPERATURE=APPLY_TEMPERATURE,
        )

    tl.store(out_value_ptr + token_idx * out_value_stride + block_idx, value)
    tl.store(out_idx_ptr + token_idx * out_idx_stride + block_idx, block_idx * BLOCK_SIZE + idx)


@triton.jit
def _gumbel_noise_probe_kernel(
    noise_ptr,
    noise_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Re-draw the exact Gumbel noise `_npu_gumbel_block_argmax` uses.

    Line-for-line copy of the RNG block of the device function.  It only
    reproduces the noise; it says nothing about how the noise is combined with
    the logits, which is what the tests using it check.
    """
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    seed = tl.load(seeds_ptr + req_state_idx)
    pos = tl.load(pos_ptr + token_idx).to(tl.int32)
    gumbel_seed = tl.randint(seed, pos)
    r = tl.rand(gumbel_seed, block).to(tl.float32)
    gumbel_noise = -tl.log(-tl.log(r + 1e-20) + 1e-20)
    tl.store(noise_ptr + token_idx * noise_stride + block, gumbel_noise, mask=mask)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _draw_noise(num_tokens, vocab_size, expanded_idx_mapping, seeds, pos, block_size):
    """Materialise [num_tokens, vocab_size] of the kernel's own Gumbel noise."""
    num_blocks = triton.cdiv(vocab_size, block_size)
    noise = torch.zeros(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    _gumbel_noise_probe_kernel[(num_tokens, num_blocks)](
        noise,
        noise.stride(0),
        expanded_idx_mapping,
        seeds,
        pos,
        vocab_size,
        BLOCK_SIZE=block_size,
    )
    torch.npu.synchronize()
    return noise


def _ref_block_argmax(logits, vocab_size, block_size, noise=None):
    """Per-block max/argmax over `logits`, fp32, deliberately loop-free but
    written straight from the kernel's semantics:

      * positions >= vocab_size are -inf (the kernel's `other=-inf` load plus
        the `tl.where(mask, ...)` re-mask), so they can never win a block;
      * noise is added only when the caller says so, and -inf + noise stays -inf,
        which is what keeps excluded tokens out of the argmax;
      * the returned index is *global* (`block_idx * BLOCK_SIZE + idx`).

    Returns (values [T, num_blocks] fp32, indices [T, num_blocks] int64).
    """
    num_tokens = logits.shape[0]
    num_blocks = triton.cdiv(vocab_size, block_size)
    padded = torch.full(
        (num_tokens, num_blocks * block_size),
        float("-inf"),
        dtype=torch.float32,
        device=logits.device,
    )
    scored = logits.float() if noise is None else logits.float() + noise.float()
    # -inf entries (masked-out / excluded tokens) must stay -inf even after the
    # noise add; float arithmetic already does that, but nan would not, so guard.
    scored = torch.where(torch.isneginf(logits.float()), logits.float(), scored)
    padded[:, :vocab_size] = scored
    padded = padded.view(num_tokens, num_blocks, block_size)
    values, idx = padded.max(dim=-1)
    offsets = torch.arange(num_blocks, device=logits.device, dtype=torch.int64) * block_size
    return values, idx.to(torch.int64) + offsets


def _assert_block_argmax_close(actual_idx, actual_val, ref_idx, ref_val, ref_scores, block_size):
    """Compare (value, index) pairs, tolerating an exact-tie index swap.

    The kernel and the reference both reduce in fp32 but not necessarily in the
    same order, so two near-equal candidates inside one block can swap.  A wrong
    index is only a real failure when the value it points at is *worse* than the
    reference maximum.
    """
    torch.testing.assert_close(actual_val.float(), ref_val.float(), rtol=_RTOL, atol=_ATOL)
    mismatch = actual_idx != ref_idx
    if not bool(mismatch.any()):
        return
    rows, blocks = torch.nonzero(mismatch, as_tuple=True)
    for r, b in zip(rows.tolist(), blocks.tolist()):
        chosen = int(actual_idx[r, b])
        assert b * block_size <= chosen < (b + 1) * block_size, (
            f"token {r} block {b}: index {chosen} escaped its own block"
        )
        assert chosen < ref_scores.shape[1], (
            f"token {r} block {b}: index {chosen} points into the padded tail past the vocabulary"
        )
        got = float(ref_scores[r, chosen])
        want = float(ref_val[r, b])
        assert abs(got - want) <= _ATOL + _RTOL * abs(want), (
            f"token {r} block {b}: kernel picked index {chosen} (score {got}) "
            f"but the reference maximum is {want} at index {int(ref_idx[r, b])}"
        )


def _gumbel_setup(num_tokens, vocab_size, max_num_reqs, temps, seed=1234, shuffle_rows=True):
    torch.manual_seed(seed)
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    if shuffle_rows:
        # req_state rows deliberately not equal to the token index, so mixing up
        # `token_idx` and `req_state_idx` cannot pass by accident.
        rows = torch.randperm(max_num_reqs)[:num_tokens].to(torch.int32)
    else:
        rows = torch.zeros(num_tokens, dtype=torch.int32)
    expanded_idx_mapping = rows.to(DEVICE)
    temperature = torch.zeros(max_num_reqs, dtype=torch.float32, device=DEVICE)
    for i, t in enumerate(temps):
        temperature[int(rows[i])] = t
    seeds = torch.randint(1, 2**30, (max_num_reqs,), dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE) * 7 + 3
    return logits, expanded_idx_mapping, temperature, seeds, pos


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@torch.inference_mode()
def test_gumbel_greedy_is_plain_block_argmax():
    """temp == 0 disables the noise entirely -- the only fully deterministic path.

    This is the branch `_resample_kernel` takes for greedy bonus tokens, and the
    one the whole greedy spec-decode path depends on, so it is checked exactly
    rather than through the noise probe.  `vocab_size` is deliberately not a
    multiple of BLOCK_SIZE so the padded tail (`other=-inf`) is exercised.
    """
    num_tokens, vocab_size, max_num_reqs = 6, 3 * PROBE_BLOCK_SIZE + 37, 11
    logits, mapping, temperature, seeds, pos = _gumbel_setup(
        num_tokens, vocab_size, max_num_reqs, [0.0] * num_tokens
    )
    num_blocks = triton.cdiv(vocab_size, PROBE_BLOCK_SIZE)

    values = torch.empty(num_tokens, num_blocks, dtype=torch.float32, device=DEVICE)
    idxs = torch.empty(num_tokens, num_blocks, dtype=torch.int64, device=DEVICE)
    dummy = torch.empty(1, dtype=torch.float32, device=DEVICE)
    dummy_col = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    _gumbel_probe_kernel[(num_tokens, num_blocks)](
        values,
        values.stride(0),
        idxs,
        idxs.stride(0),
        logits,
        logits.stride(0),
        mapping,
        temperature,
        seeds,
        pos,
        dummy,
        0,
        dummy_col,
        vocab_size,
        BLOCK_SIZE=PROBE_BLOCK_SIZE,
        APPLY_TEMPERATURE=False,
        PROCESSED_MODE=0,
    )
    torch.npu.synchronize()

    ref_val, ref_idx = _ref_block_argmax(logits, vocab_size, PROBE_BLOCK_SIZE)
    torch.testing.assert_close(values, ref_val, rtol=_RTOL, atol=_ATOL)
    assert torch.equal(idxs, ref_idx)
    # The tail block must never point past the vocabulary.
    assert int(idxs.max()) < vocab_size


@pytest.mark.parametrize("apply_temperature", [True, False])
@torch.inference_mode()
def test_gumbel_matches_reference_with_noise(apply_temperature):
    """temp != 0: noise on, and `APPLY_TEMPERATURE` toggled on both sides.

    The `APPLY_TEMPERATURE=True` half is unreachable from `_resample_kernel`
    (which hardcodes False) but is part of the device function's contract and is
    what the upstream sampler uses, so both sides of the constexpr are pinned.
    """
    num_tokens, vocab_size, max_num_reqs = 5, 2 * PROBE_BLOCK_SIZE + 11, 9
    temps = [0.5, 1.0, 2.0, 0.7, 1.3]
    logits, mapping, temperature, seeds, pos = _gumbel_setup(num_tokens, vocab_size, max_num_reqs, temps)
    num_blocks = triton.cdiv(vocab_size, PROBE_BLOCK_SIZE)

    values = torch.empty(num_tokens, num_blocks, dtype=torch.float32, device=DEVICE)
    idxs = torch.empty(num_tokens, num_blocks, dtype=torch.int64, device=DEVICE)
    dummy = torch.empty(1, dtype=torch.float32, device=DEVICE)
    dummy_col = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    _gumbel_probe_kernel[(num_tokens, num_blocks)](
        values,
        values.stride(0),
        idxs,
        idxs.stride(0),
        logits,
        logits.stride(0),
        mapping,
        temperature,
        seeds,
        pos,
        dummy,
        0,
        dummy_col,
        vocab_size,
        BLOCK_SIZE=PROBE_BLOCK_SIZE,
        APPLY_TEMPERATURE=apply_temperature,
        PROCESSED_MODE=0,
    )
    torch.npu.synchronize()

    noise = _draw_noise(num_tokens, vocab_size, mapping, seeds, pos, PROBE_BLOCK_SIZE)
    # Guard the guard: a degenerate (constant / all-zero) noise draw would make
    # this test collapse into the greedy one above.
    assert float(noise.std()) > 0.1, "gumbel noise probe no longer produces a spread of values"

    scaled = logits.float()
    if apply_temperature:
        per_token_temp = temperature[mapping.long()].unsqueeze(1)
        scaled = scaled / per_token_temp
    ref_val, ref_idx = _ref_block_argmax(scaled, vocab_size, PROBE_BLOCK_SIZE, noise=noise)

    _assert_block_argmax_close(idxs, values, ref_idx, ref_val, scaled + noise, PROBE_BLOCK_SIZE)

    # Guard the guard: the noise must actually move at least one winner, else
    # this case proves nothing beyond the greedy path.
    _, greedy_idx = _ref_block_argmax(scaled, vocab_size, PROBE_BLOCK_SIZE)
    assert bool((greedy_idx != ref_idx).any()), "noise no longer changes any block winner"


@pytest.mark.parametrize("processed_mode", [1, 2], ids=["implicit-col-0", "explicit-col"])
@torch.inference_mode()
def test_gumbel_stores_processed_logits(processed_mode):
    """The `processed_logits` side output: written before the noise, after the temperature.

    Both column modes are covered: `processed_logits_col_ptr is None` (column 0)
    and an explicit column, which is the branch that makes the write land at
    `req_state_idx * stride + col * vocab_size`.  Note the row index is
    `req_state_idx`, *not* `token_idx` -- getting that wrong is silent.
    """
    num_tokens, vocab_size, max_num_reqs = 4, PROBE_BLOCK_SIZE + 5, 7
    num_cols = 3
    col = 2 if processed_mode == 2 else 0
    temps = [0.0, 0.5, 2.0, 1.0]
    logits, mapping, temperature, seeds, pos = _gumbel_setup(num_tokens, vocab_size, max_num_reqs, temps)
    num_blocks = triton.cdiv(vocab_size, PROBE_BLOCK_SIZE)

    processed = torch.full(
        (max_num_reqs, num_cols * vocab_size), float("nan"), dtype=torch.float32, device=DEVICE
    )
    col_tensor = torch.tensor([col], dtype=torch.int32, device=DEVICE)
    values = torch.empty(num_tokens, num_blocks, dtype=torch.float32, device=DEVICE)
    idxs = torch.empty(num_tokens, num_blocks, dtype=torch.int64, device=DEVICE)
    _gumbel_probe_kernel[(num_tokens, num_blocks)](
        values,
        values.stride(0),
        idxs,
        idxs.stride(0),
        logits,
        logits.stride(0),
        mapping,
        temperature,
        seeds,
        pos,
        processed,
        processed.stride(0),
        col_tensor,
        vocab_size,
        BLOCK_SIZE=PROBE_BLOCK_SIZE,
        APPLY_TEMPERATURE=True,
        PROCESSED_MODE=processed_mode,
    )
    torch.npu.synchronize()

    for token_idx in range(num_tokens):
        row = int(mapping[token_idx])
        temp = float(temperature[row])
        expected = logits[token_idx].float()
        if temp != 0.0:
            expected = expected / temp
        actual = processed[row, col * vocab_size : col * vocab_size + vocab_size]
        torch.testing.assert_close(actual, expected, rtol=_RTOL, atol=_ATOL)

    # Rows/columns nobody wrote must still be untouched: the store is masked to
    # `block < vocab_size`, so no neighbouring column may be clobbered.
    written_rows = {int(r) for r in mapping}
    for row in range(max_num_reqs):
        for c in range(num_cols):
            if row in written_rows and c == col:
                continue
            chunk = processed[row, c * vocab_size : c * vocab_size + vocab_size]
            assert bool(torch.isnan(chunk).all()), f"row {row} col {c} was overwritten"


@torch.inference_mode()
def test_gumbel_argmax_follows_softmax_distribution():
    """Gumbel-max must sample proportionally to softmax(logits).

    This is the one check that does *not* reuse the kernel's own RNG, so it is
    the only thing standing between a broken philox call (wrong seed mixing,
    a constant draw, a sign error in `-log(-log(u))`) and a silently biased
    sampler.  8 categories in a single block, 16384 draws, one distinct `pos`
    per draw.
    """
    num_tokens, vocab_size, max_num_reqs = 16384, 8, 1
    torch.manual_seed(7)
    row_logits = torch.tensor([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0], dtype=torch.float32)
    logits = row_logits.to(DEVICE).repeat(num_tokens, 1).contiguous()
    mapping = torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE)
    temperature = torch.ones(max_num_reqs, dtype=torch.float32, device=DEVICE)
    seeds = torch.full((max_num_reqs,), 20260827, dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)

    values = torch.empty(num_tokens, 1, dtype=torch.float32, device=DEVICE)
    idxs = torch.empty(num_tokens, 1, dtype=torch.int64, device=DEVICE)
    dummy = torch.empty(1, dtype=torch.float32, device=DEVICE)
    dummy_col = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    _gumbel_probe_kernel[(num_tokens, 1)](
        values,
        values.stride(0),
        idxs,
        idxs.stride(0),
        logits,
        logits.stride(0),
        mapping,
        temperature,
        seeds,
        pos,
        dummy,
        0,
        dummy_col,
        vocab_size,
        BLOCK_SIZE=vocab_size,
        APPLY_TEMPERATURE=False,
        PROCESSED_MODE=0,
    )
    torch.npu.synchronize()

    counts = torch.bincount(idxs.flatten().cpu(), minlength=vocab_size).float()
    empirical = counts / num_tokens
    expected = torch.softmax(row_logits, dim=0)
    # 16384 draws => per-category std <= 0.004; 0.02 is ~5 sigma and leaves room
    # for the coarse fp32 tail of `-log(-log(u))` (see the operator doc).
    assert torch.allclose(empirical, expected, atol=0.02), (
        f"argmax frequencies {empirical.tolist()} deviate from softmax {expected.tolist()}"
    )


@torch.inference_mode()
def test_gumbel_is_deterministic_in_seed_and_pos():
    """Same (seed, pos) must give the same token; a different pos must not.

    Reproducibility across two runs of the same request is a user-visible
    property (`SamplingParams.seed`), and it is what makes the noise-probe
    oracle in the tests above legitimate.
    """
    num_tokens, vocab_size, max_num_reqs = 8, PROBE_BLOCK_SIZE, 8
    logits, mapping, temperature, seeds, pos = _gumbel_setup(
        num_tokens, vocab_size, max_num_reqs, [1.0] * num_tokens
    )

    def _run(pos_tensor):
        values = torch.empty(num_tokens, 1, dtype=torch.float32, device=DEVICE)
        idxs = torch.empty(num_tokens, 1, dtype=torch.int64, device=DEVICE)
        dummy = torch.empty(1, dtype=torch.float32, device=DEVICE)
        dummy_col = torch.zeros(1, dtype=torch.int32, device=DEVICE)
        _gumbel_probe_kernel[(num_tokens, 1)](
            values,
            values.stride(0),
            idxs,
            idxs.stride(0),
            logits,
            logits.stride(0),
            mapping,
            temperature,
            seeds,
            pos_tensor,
            dummy,
            0,
            dummy_col,
            vocab_size,
            BLOCK_SIZE=PROBE_BLOCK_SIZE,
            APPLY_TEMPERATURE=False,
            PROCESSED_MODE=0,
        )
        torch.npu.synchronize()
        return values, idxs

    v1, i1 = _run(pos)
    v2, i2 = _run(pos)
    assert torch.equal(i1, i2)
    torch.testing.assert_close(v1, v2, rtol=0.0, atol=0.0)

    v3, i3 = _run(pos + 1)
    assert bool((i3 != i1).any()), "shifting pos no longer changes the draw"
