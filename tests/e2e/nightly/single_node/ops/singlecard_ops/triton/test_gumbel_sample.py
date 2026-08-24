# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Numerical test for vllm_ascend.worker.v2.sample.gumbel._gumbel_sample_kernel
# (Triton-Ascend) against plain PyTorch fp32 references.
# Requires NPU and Triton-Ascend.
#
# See vllm_ascend/ops/triton/doc/gumbel_sample.md for the operator spec.
#
# Why this file exists next to tests/ut/sample/a2/test_gumbel_sampling.py:
# that file pins the greedy (temperature == 0) path and the processed-logits
# side output, but the Gumbel branch itself -- the noise formula, its
# interaction with temperature, and the per-block reduction under noise -- is
# only checked there for determinism, "different seeds differ" and one coarse
# entropy comparison.  Every one of those assertions still passes if the noise
# is wrong (a sign flip, a missing temperature division, or the same noise
# reused in every vocab block).  The tests below close that gap by comparing
# the *sampling distribution* against softmax, which is what the Gumbel-max
# trick is supposed to reproduce exactly.

import gc

import pytest
import torch
import torch_npu  # noqa: F401  # registers the npu backend / torch.npu namespace

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.worker.v2.sample.gumbel import gumbel_sample

DEVICE = "npu"

# Mirrors the BLOCK_SIZE hard-coded in gumbel_sample(); shapes below are chosen
# relative to it (single block / exact multiple / ragged tail).
KERNEL_BLOCK_SIZE = 1024

# Logits, the temperature division and the noise are all fp32 inside the
# kernel, and the references below run in fp32 too, so the processed-logits
# comparison is a near-exact one; the slack only absorbs the reciprocal
# rounding of the division.
_RTOL, _ATOL = 1e-6, 1e-6

# Statistical tests: number of independent draws.  Each draw is one token row,
# so a whole distribution check costs a single kernel launch.
_NUM_DRAWS = 8192


@pytest.fixture(autouse=True)
def _npu_env():
    init_device_properties_triton()
    yield
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def _tol_for(p: float, n: int) -> float:
    """6-sigma binomial band plus a small slack, as an absolute frequency."""
    return 6.0 * (p * (1.0 - p) / n) ** 0.5 + 5e-3


def _ref_processed_logits(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    apply_temperature: bool,
) -> list[torch.Tensor]:
    """Per-token reference for the processed-logits side output, fp32.

    Deliberately loop-based rather than vectorised: it is the oracle, so being
    obviously correct matters more than being fast.  Test shapes are tiny.

    Mirrors kernel behaviours that are easy to miss:
      * the division is skipped when temperature == 0 (greedy request), even
        with APPLY_TEMPERATURE set -- the row is stored unscaled;
      * with apply_temperature=False the row is stored raw, because the caller
        is expected to have scaled the logits already (apply_temperature()).
    """
    out = []
    for token_idx in range(logits.shape[0]):
        req = int(expanded_idx_mapping[token_idx].item())
        temp = float(temperature[req].item())
        row = logits[token_idx].float()
        if apply_temperature and temp != 0.0:
            row = row / temp
        out.append(row)
    return out


def _build_spread_logits(vocab_size: int, active: list[int], values: list[float]) -> torch.Tensor:
    """One logits row: a handful of live tokens, everything else far below.

    The filler value is chosen so that it can never win: Gumbel noise is
    -log(-log(u)) with u in [0, 1), which is bounded above by ~46 in fp32
    (the +1e-20 clamps inside the kernel), so a 60-wide gap is unreachable.
    """
    row = torch.full((vocab_size,), -60.0, dtype=torch.float32)
    for idx, val in zip(active, values):
        row[idx] = val
    return row


def _draw(
    logits_row: torch.Tensor,
    temperature_value: float,
    num_reqs: int,
    apply_temperature: bool,
    num_draws: int = _NUM_DRAWS,
    seed_base: int = 1234,
) -> torch.Tensor:
    """num_draws independent samples of the same categorical distribution.

    Tokens are spread over num_reqs requests with distinct seeds and distinct
    positions, so every draw uses a different Gumbel seed
    (randint(seed, pos)) while the underlying distribution stays identical.
    """
    logits = logits_row.to(DEVICE).unsqueeze(0).repeat(num_draws, 1).contiguous()
    expanded_idx_mapping = (torch.arange(num_draws, dtype=torch.int32) % num_reqs).to(DEVICE)
    temperature = torch.full((num_reqs,), temperature_value, dtype=torch.float32, device=DEVICE)
    seed = (torch.arange(num_reqs, dtype=torch.int64) * 7919 + seed_base).to(DEVICE)
    pos = torch.arange(num_draws, dtype=torch.int32, device=DEVICE)
    return gumbel_sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=apply_temperature,
    )


# ---------------------------------------------------------------------------
# Greedy path: exact, elementwise
# ---------------------------------------------------------------------------

GREEDY_CASES = [
    pytest.param(1, 1, 1, id="vocab-1"),
    pytest.param(3, 3, 512, id="single-partial-block"),
    pytest.param(4, 4, KERNEL_BLOCK_SIZE, id="exactly-one-block"),
    pytest.param(6, 3, 3 * KERNEL_BLOCK_SIZE, id="exact-multiple-3-blocks"),
    pytest.param(5, 3, 3000, id="ragged-tail"),
    pytest.param(4, 4, 151936, id="qwen2-vocab"),
]


@pytest.mark.parametrize(("num_tokens", "num_reqs", "vocab_size"), GREEDY_CASES)
@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_greedy_matches_argmax(num_tokens, num_reqs, vocab_size, pos_dtype):
    """temperature == 0 must return the exact argmax of every row.

    This is the only path where the kernel output is bit-reproducible, so it is
    the one that locks the block decomposition: BLOCK_SIZE is 1024 and the
    per-block (argmax, max) pairs are reduced on the host side, so a wrong
    `block_idx * BLOCK_SIZE + idx` offset or a wrong tail mask only shows up
    when the winner does not sit in block 0.  The shape list therefore walks
    single-partial-block / exact-multiple / ragged-tail / real-model vocab.

    num_tokens and num_reqs are kept unequal and non-power-of-two so that a
    swapped `//` vs `%` in the token-to-request mapping cannot pass by luck.
    pos is drawn in both int32 and int64 because the kernel casts it down to
    int32 (triton-ascend has no uint64 philox) and callers pass int64.
    """
    torch.manual_seed(42)
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    expanded_idx_mapping = (torch.arange(num_tokens, dtype=torch.int32) % num_reqs).to(DEVICE)
    temperature = torch.zeros(num_reqs, dtype=torch.float32, device=DEVICE)
    seed = torch.arange(num_reqs, dtype=torch.int64, device=DEVICE) + 11
    pos = torch.arange(num_tokens, dtype=pos_dtype, device=DEVICE)

    sampled = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False)

    expected = logits.argmax(dim=-1)
    assert torch.equal(sampled, expected), f"greedy mismatch: {sampled.tolist()} vs {expected.tolist()}"
    assert sampled.dtype == torch.int64


@torch.inference_mode()
def test_greedy_winner_outside_first_block():
    """Force the winner into the last block of every row.

    test_greedy_matches_argmax uses random logits, where the winning block is
    whatever the RNG picked; this case pins it, so a reduction that silently
    prefers block 0 (for example a tie-break bug in the host-side argmax over
    local_max) cannot survive.
    """
    torch.manual_seed(7)
    num_tokens, vocab_size = 4, 4 * KERNEL_BLOCK_SIZE
    winners = torch.tensor([vocab_size - 1, 3 * KERNEL_BLOCK_SIZE, vocab_size - 2, 3 * KERNEL_BLOCK_SIZE + 5])
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32)
    logits[torch.arange(num_tokens), winners] = 100.0
    logits = logits.to(DEVICE)

    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
    temperature = torch.zeros(num_tokens, dtype=torch.float32, device=DEVICE)
    seed = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

    sampled = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False)
    assert torch.equal(sampled.cpu(), winners), f"{sampled.tolist()} vs {winners.tolist()}"


@torch.inference_mode()
def test_mixed_temperature_keeps_greedy_rows_exact():
    """Greedy and sampled requests in one launch: the greedy rows stay exact.

    temp is read per request inside the kernel, so a batch that mixes 0 and
    non-zero temperatures is the case where a hoisted / mis-broadcast `temp`
    would show up.
    """
    torch.manual_seed(11)
    num_tokens, vocab_size = 8, 2 * KERNEL_BLOCK_SIZE
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
    temperature = torch.tensor([0.0, 0.8, 0.0, 1.0, 0.0, 1.7, 0.0, 0.3], dtype=torch.float32, device=DEVICE)
    seed = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE) + 5
    pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

    sampled = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=True)

    greedy = logits.argmax(dim=-1)
    greedy_rows = torch.arange(0, num_tokens, 2)
    assert torch.equal(sampled[greedy_rows], greedy[greedy_rows])
    # Guard the guard: the sampled rows must not be trivially greedy as well,
    # otherwise this case would also pass with the noise wired to zero.
    assert not torch.equal(sampled, greedy), "noise never moved any row off the argmax"


# ---------------------------------------------------------------------------
# Gumbel branch: the sampling distribution must be softmax
# ---------------------------------------------------------------------------

# Live tokens placed at both ends of both vocab blocks, so the distribution
# check also exercises the cross-block reduction under noise.
_ACTIVE = [0, 7, KERNEL_BLOCK_SIZE - 1, KERNEL_BLOCK_SIZE, 1500, 2 * KERNEL_BLOCK_SIZE - 1]
_VALUES = [2.0, 1.0, 0.5, 0.0, -0.5, -1.5]


@pytest.mark.parametrize(
    ("apply_temperature", "temperature"),
    [
        pytest.param(True, 0.7, id="apply-temp-sharpen"),
        pytest.param(True, 1.8, id="apply-temp-flatten"),
        pytest.param(False, 1.3, id="no-apply-temp-raw-logits"),
    ],
)
@torch.inference_mode()
def test_sampling_distribution_matches_softmax(apply_temperature, temperature):
    """The Gumbel-max trick must reproduce softmax exactly, in distribution.

    This is the case the existing UT does not have.  P(argmax_i(logit_i + g_i))
    == softmax(logits)_i holds only for the *max*-Gumbel -log(-log(u)); a sign
    flip, a missing +1e-20 clamp, dropping the temperature division, or reusing
    one noise draw across the vocab all break this equality while leaving
    "deterministic for a fixed seed" and "different seeds differ" intact.

    The APPLY_TEMPERATURE=False row is what pins the #9173 semantics: with the
    flag off the kernel must *not* divide by temperature (the caller already
    did), yet must still add noise -- so the reference is softmax of the raw
    logits, not of logits/temp.
    """
    row = _build_spread_logits(2 * KERNEL_BLOCK_SIZE, _ACTIVE, _VALUES)
    scaled = row / temperature if apply_temperature else row
    expected = torch.softmax(scaled.float(), dim=-1)

    # Guard the guard: a near-uniform target would be matched by broken noise
    # too.  Keep the fixture peaked.
    assert expected.max().item() > 0.25, "fixture is too flat to discriminate"

    sampled = _draw(row, temperature, num_reqs=4, apply_temperature=apply_temperature).cpu()

    active = torch.tensor(_ACTIVE)
    assert torch.isin(sampled, active).all(), "sampled a filler token: masking or noise bound is wrong"

    n = sampled.numel()
    freq = torch.bincount(sampled, minlength=row.numel()).float() / n
    for idx in _ACTIVE:
        p = expected[idx].item()
        assert abs(freq[idx].item() - p) <= _tol_for(p, n), (
            f"token {idx}: empirical {freq[idx].item():.4f} vs softmax {p:.4f} (tol {_tol_for(p, n):.4f}, {n} draws)"
        )


@torch.inference_mode()
def test_noise_is_independent_across_vocab_blocks():
    """Flat logits must sample uniformly over the whole vocabulary.

    The noise is drawn as tl.rand(gumbel_seed, block) with `block` the *global*
    vocab offsets, so every 1024-wide block gets a different draw.  If it were
    ever re-seeded per block (or the offsets were made block-local), each block
    would produce the same noise vector, every block maximum would tie, and the
    host-side argmax over local_max would always return block 0.  With flat
    logits that failure is a 4x bias, which this case measures directly.
    """
    vocab_size = 4 * KERNEL_BLOCK_SIZE
    row = torch.zeros(vocab_size, dtype=torch.float32)
    sampled = _draw(row, 1.0, num_reqs=4, apply_temperature=False, num_draws=4096).cpu()
    n = sampled.numel()

    block_win = torch.bincount(sampled // KERNEL_BLOCK_SIZE, minlength=4).float() / n
    tol = _tol_for(0.25, n)
    assert (block_win - 0.25).abs().max().item() <= tol, f"per-block win rates {block_win.tolist()} (tol {tol:.4f})"

    # Finer check: 16 equal bins across the vocabulary.
    bins = torch.bincount(sampled // (vocab_size // 16), minlength=16).float() / n
    tol16 = _tol_for(1 / 16, n)
    assert (bins - 1 / 16).abs().max().item() <= tol16, f"per-bin win rates {bins.tolist()} (tol {tol16:.4f})"


@torch.inference_mode()
def test_ragged_tail_block_is_sampled_but_never_overruns():
    """A vocab that is not a multiple of BLOCK_SIZE: the tail must be exact.

    Two failure modes at once: padding lanes leaking into the argmax (would
    return token ids >= vocab_size) and the short tail block being weighted as
    if it were full (would inflate its win rate).  With flat logits the tail
    block must win exactly (vocab_size % BLOCK_SIZE) / vocab_size of the time.
    """
    vocab_size = 3000  # 1024 + 1024 + 952
    tail_len = vocab_size % KERNEL_BLOCK_SIZE
    row = torch.zeros(vocab_size, dtype=torch.float32)
    sampled = _draw(row, 0.9, num_reqs=4, apply_temperature=True, num_draws=4096).cpu()
    n = sampled.numel()

    assert int(sampled.max().item()) < vocab_size, "sampled a padding lane past the end of the vocabulary"
    assert int(sampled.min().item()) >= 0

    p_tail = tail_len / vocab_size
    freq_tail = (sampled >= vocab_size - tail_len).float().mean().item()
    assert abs(freq_tail - p_tail) <= _tol_for(p_tail, n), (
        f"tail-block win rate {freq_tail:.4f} vs expected {p_tail:.4f}"
    )


@torch.inference_mode()
def test_sampling_is_shift_invariant():
    """Adding a constant to a row must not change the sample, bit for bit.

    Gumbel-max depends on logit *differences* only.  A shift changing the
    outcome means the noise was scaled by, or derived from, the logits.
    """
    torch.manual_seed(21)
    num_tokens, vocab_size = 64, 2 * KERNEL_BLOCK_SIZE
    base = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE)
    temperature = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
    seed = torch.tensor([98765], dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

    a = gumbel_sample(base.clone(), expanded_idx_mapping, temperature, seed, pos, apply_temperature=False)
    b = gumbel_sample(base + 7.5, expanded_idx_mapping, temperature, seed, pos, apply_temperature=False)
    assert torch.equal(a, b), "sample changed under a constant logit shift"

    # Guard the guard: the fixture must actually be sampling, not collapsing
    # onto the argmax for every row.
    assert not torch.equal(a, base.argmax(dim=-1)), "fixture degenerated into greedy"


@torch.inference_mode()
def test_dominant_logit_always_wins():
    """A logit far above the Gumbel noise range must be sampled every time.

    Bounds the noise from above: -log(-log(u)) cannot exceed ~46 in fp32 with
    the kernel's 1e-20 clamps, so a +200 gap is decisive.  If the noise were
    scaled up (for example a missing temperature division on the noise side)
    this case would start to flake.
    """
    torch.manual_seed(31)
    num_tokens, vocab_size = 32, 3 * KERNEL_BLOCK_SIZE
    winners = torch.randint(0, vocab_size, (num_tokens,))
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32)
    logits[torch.arange(num_tokens), winners] = 200.0
    logits = logits.to(DEVICE)

    expanded_idx_mapping = torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE)
    temperature = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
    seed = torch.tensor([4242], dtype=torch.int64, device=DEVICE)
    pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

    sampled = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=True)
    assert torch.equal(sampled.cpu(), winners), f"{sampled.tolist()} vs {winners.tolist()}"


@torch.inference_mode()
def test_same_seed_and_pos_give_the_same_noise():
    """Two tokens of one request at the same position must sample identically.

    The noise seed is randint(seed[req], pos[token]); this pins that it depends
    on nothing else (not on token_idx, not on the row contents beyond the
    logits themselves).
    """
    torch.manual_seed(63)
    vocab_size = 2 * KERNEL_BLOCK_SIZE
    row = torch.randn(1, vocab_size, dtype=torch.float32, device=DEVICE)
    logits = row.repeat(2, 1).contiguous()
    expanded_idx_mapping = torch.zeros(2, dtype=torch.int32, device=DEVICE)
    temperature = torch.tensor([0.8], dtype=torch.float32, device=DEVICE)
    seed = torch.tensor([777], dtype=torch.int64, device=DEVICE)
    pos = torch.tensor([5, 5], dtype=torch.int32, device=DEVICE)

    sampled = gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos, apply_temperature=True)
    assert sampled[0].item() == sampled[1].item()

    # Guard the guard: distinct positions of the same request must decorrelate,
    # otherwise the assertion above would also hold with pos ignored entirely.
    many = gumbel_sample(
        row.repeat(32, 1).contiguous(),
        torch.zeros(32, dtype=torch.int32, device=DEVICE),
        temperature,
        seed,
        torch.arange(32, dtype=torch.int32, device=DEVICE),
        apply_temperature=True,
    )
    assert many.unique().numel() > 1, "position is not feeding the Gumbel seed"


# ---------------------------------------------------------------------------
# processed-logits side output: exact, elementwise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("apply_temperature", [True, False], ids=["apply-temp", "no-apply-temp"])
@torch.inference_mode()
def test_processed_logits_matches_reference(apply_temperature):
    """The [max_num_reqs, vocab] side buffer, row by row.

    Covers the three semantics that are easy to get wrong: the row is written
    at req_state_idx (not token_idx) -- exercised here with a non-contiguous
    EAGLE-style mapping, the division is skipped for temperature == 0 even with
    the flag on, and nothing is written for request slots that no token maps to.
    """
    torch.manual_seed(200)
    num_tokens, max_num_reqs, vocab_size = 4, 8, 2 * KERNEL_BLOCK_SIZE + 300
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    expanded_idx_mapping = torch.tensor([2, 5, 7, 0], dtype=torch.int32, device=DEVICE)
    temperature = torch.zeros(max_num_reqs, dtype=torch.float32, device=DEVICE)
    temperature[2], temperature[5], temperature[7], temperature[0] = 0.0, 0.6, 1.0, 1.9
    seed = torch.arange(max_num_reqs, dtype=torch.int64, device=DEVICE) + 3
    pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

    processed = torch.zeros(max_num_reqs, vocab_size, dtype=torch.float32, device=DEVICE)
    gumbel_sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=apply_temperature,
        output_processed_logits=processed,
    )

    expected_rows = _ref_processed_logits(logits, expanded_idx_mapping, temperature, apply_temperature)
    for token_idx, expected in enumerate(expected_rows):
        req = int(expanded_idx_mapping[token_idx].item())
        torch.testing.assert_close(processed[req].float(), expected, rtol=_RTOL, atol=_ATOL)

    used = set(expanded_idx_mapping.tolist())
    for req in range(max_num_reqs):
        if req not in used:
            assert (processed[req] == 0).all(), f"untouched request slot {req} was written"


@torch.inference_mode()
def test_processed_logits_scalar_column():
    """PER_TOKEN_COL == False: a 0-dim column tensor selects one draft step.

    The buffer is [max_num_reqs, num_steps, vocab]; the kernel addresses it as
    req * stride(0) + col * vocab + block, so a wrong stride or a wrong col
    lands in a neighbouring step, which the untouched-column assertions catch.
    """
    torch.manual_seed(201)
    num_tokens, max_num_reqs, num_steps, vocab_size = 3, 4, 3, 2 * KERNEL_BLOCK_SIZE
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
    temperature = torch.full((max_num_reqs,), 0.9, dtype=torch.float32, device=DEVICE)
    seed = torch.arange(max_num_reqs, dtype=torch.int64, device=DEVICE) + 9
    pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

    draft = torch.zeros(max_num_reqs, num_steps, vocab_size, dtype=torch.float32, device=DEVICE)
    col = torch.tensor(1, dtype=torch.int32, device=DEVICE)  # 0-dim -> PER_TOKEN_COL False
    gumbel_sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=True,
        output_processed_logits=draft,
        output_processed_logits_col=col,
    )

    for token_idx in range(num_tokens):
        req = int(expanded_idx_mapping[token_idx].item())
        expected = logits[token_idx].float() / float(temperature[req].item())
        torch.testing.assert_close(draft[req, 1].float(), expected, rtol=_RTOL, atol=_ATOL)
        assert (draft[req, 0] == 0).all() and (draft[req, 2] == 0).all()


@torch.inference_mode()
def test_processed_logits_per_token_column():
    """PER_TOKEN_COL == True: a 1-D column tensor, one step per token.

    This branch is selected purely by `output_processed_logits_col.dim() > 0`
    in the wrapper and is not exercised anywhere else in the test suite; the
    per-token load lives on a different code path from the scalar one, so a
    swapped `token_idx` / `req_state_idx` there is invisible to the case above.
    Columns are deliberately not the identity permutation of the tokens.
    """
    torch.manual_seed(202)
    num_tokens, max_num_reqs, num_steps, vocab_size = 4, 4, 4, KERNEL_BLOCK_SIZE + 512
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    expanded_idx_mapping = torch.tensor([3, 1, 0, 2], dtype=torch.int32, device=DEVICE)
    temperature = torch.tensor([0.5, 1.0, 1.4, 0.0], dtype=torch.float32, device=DEVICE)
    seed = torch.arange(max_num_reqs, dtype=torch.int64, device=DEVICE) + 17
    pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
    cols = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device=DEVICE)

    draft = torch.zeros(max_num_reqs, num_steps, vocab_size, dtype=torch.float32, device=DEVICE)
    gumbel_sample(
        logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        apply_temperature=True,
        output_processed_logits=draft,
        output_processed_logits_col=cols,
    )

    expected_rows = _ref_processed_logits(logits, expanded_idx_mapping, temperature, apply_temperature=True)
    written = set()
    for token_idx, expected in enumerate(expected_rows):
        req = int(expanded_idx_mapping[token_idx].item())
        col = int(cols[token_idx].item())
        written.add((req, col))
        torch.testing.assert_close(draft[req, col].float(), expected, rtol=_RTOL, atol=_ATOL)

    for req in range(max_num_reqs):
        for col in range(num_steps):
            if (req, col) not in written:
                assert (draft[req, col] == 0).all(), f"slot ({req}, {col}) was written but should not be"


@torch.inference_mode()
def test_processed_logits_does_not_include_noise():
    """The side buffer must hold the pre-noise logits.

    Downstream (EAGLE / rejection sampling) treats it as the draft
    distribution, so leaking the Gumbel perturbation into it would silently
    corrupt the acceptance test.  Two launches with different seeds must write
    the same buffer.
    """
    torch.manual_seed(203)
    num_tokens, vocab_size = 4, 2 * KERNEL_BLOCK_SIZE
    logits = torch.randn(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
    expanded_idx_mapping = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)
    temperature = torch.full((num_tokens,), 0.75, dtype=torch.float32, device=DEVICE)
    pos = torch.arange(num_tokens, dtype=torch.int32, device=DEVICE)

    buffers = []
    for seed_value in (1, 2**20 + 7):
        buf = torch.zeros(num_tokens, vocab_size, dtype=torch.float32, device=DEVICE)
        seed = torch.full((num_tokens,), seed_value, dtype=torch.int64, device=DEVICE)
        gumbel_sample(
            logits,
            expanded_idx_mapping,
            temperature,
            seed,
            pos,
            apply_temperature=True,
            output_processed_logits=buf,
        )
        buffers.append(buf)

    torch.testing.assert_close(buffers[0], buffers[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(buffers[0].float(), logits.float() / 0.75, rtol=_RTOL, atol=_ATOL)


@torch.inference_mode()
def test_fp64_is_rejected():
    """use_fp64 must fail loudly rather than silently sampling in fp32.

    triton-ascend has no float64, so the wrapper raises; upstream vLLM offers
    the fp64 Gumbel path and a caller that asks for it must not be handed a
    quietly different distribution.
    """
    logits = torch.randn(1, 128, dtype=torch.float32, device=DEVICE)
    expanded_idx_mapping = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    temperature = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
    seed = torch.tensor([1], dtype=torch.int64, device=DEVICE)
    pos = torch.zeros(1, dtype=torch.int32, device=DEVICE)

    with pytest.raises(NotImplementedError):
        gumbel_sample(logits, expanded_idx_mapping, temperature, seed, pos, False, use_fp64=True)
