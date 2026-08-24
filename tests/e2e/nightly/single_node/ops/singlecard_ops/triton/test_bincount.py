# SPDX-License-Identifier: Apache-2.0
# Numerical test for vllm_ascend.worker.v2.sample.penalties._bincount_kernel
# (Triton-Ascend) against a plain PyTorch/NumPy integer reference.
# Requires NPU and Triton-Ascend.
#
# See vllm_ascend/worker/v2/sample/doc/bincount.md for the operator spec.
#
# Regression scope: #7757 -- the kernel was introduced with a test that was
# marked `skip` from the very first commit ("atomic_or operator hangs in
# current npu_ir version", i.e. the CANN 8.5.1 atomic deadlock).  The repo now
# builds on CANN 9.1.0, so the skip is dropped and the case is exercised for
# real; the old case also fed the kernel `prompt_len` and `prefill_len` drawn
# independently at random, which violates the `prompt_len <= prefill_len`
# contract and made the reference disagree with the kernel by construction.

import gc

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401  # registers the npu backend / torch.npu namespace

from vllm_ascend.worker.v2.sample.penalties import bincount

DEVICE = "npu"

# `bincount()` hardcodes the position-block width.
BLOCK_SIZE = 1024

# Small vocabulary for the bulk of the grid: output_bin_counts is
# [max_num_reqs, vocab_size], so a realistic vocab is only worth paying for in
# the one case that specifically checks the real-shape strides.
VOCAB_SIZE = 4096
MAX_NUM_REQS = 8

# Sentinel written past prefill_len; if the kernel ever counted it the bin for
# this token id would be non-zero, and the reference keeps it at zero.
PAST_PREFILL_TOKEN = 7


@pytest.fixture(autouse=True)
def _npu_env():
    yield
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def _ref_bincount(
    idx_mapping: np.ndarray,
    all_token_ids: np.ndarray,
    prompt_len: np.ndarray,
    prefill_len: np.ndarray,
    prompt_bin_mask: np.ndarray,
    output_bin_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Straightforward NumPy reference, integer-exact throughout.

    Deliberately loop-based rather than vectorised: it is the oracle, so being
    obviously correct matters more than being fast.  Test lengths are tiny.

    Mirrors kernel/wrapper behaviours that are easy to miss:
      * only the rows named by ``idx_mapping`` are reset, and they are *all*
        reset before any counting happens -- the wrapper zeroes both output
        tensors with a fancy-index assignment ahead of the launch, so rows
        outside the mapping must survive untouched;
      * ``prompt_bin_mask`` is int32 but holds a *bit pattern*: token ids with
        ``id % 32 == 31`` set the sign bit, so the packing is done in uint32
        and reinterpreted, not computed as a signed shift;
      * positions in ``[prompt_len, prefill_len)`` are counted, positions past
        ``prefill_len`` are not read at all, so whatever garbage lives there
        must not leak into the counts.
    """
    mask = prompt_bin_mask.copy().view(np.uint32)
    counts = output_bin_counts.copy()

    rows = idx_mapping.tolist()
    for req in rows:
        mask[req] = 0
        counts[req] = 0

    for req in rows:
        p_len = int(prompt_len[req])
        f_len = int(prefill_len[req])
        row = all_token_ids[req]
        for pos in range(p_len):
            token = int(row[pos])
            mask[req, token // 32] |= np.uint32(1) << np.uint32(token % 32)
        for pos in range(p_len, f_len):
            counts[req, int(row[pos])] += 1

    return mask.view(np.int32), counts


def _build_inputs(
    rows: list[int],
    lens: list[tuple[int, int]],
    *,
    vocab_size: int = VOCAB_SIZE,
    max_num_reqs: int = MAX_NUM_REQS,
    seed: int = 0,
    token_pool: np.ndarray | None = None,
):
    """Build one bincount scenario.

    ``rows`` are the request-state slots named by ``idx_mapping``; ``lens`` are
    the matching ``(prompt_len, prefill_len)`` pairs.  Rows outside ``rows`` are
    poisoned with non-zero state so that "the wrapper only touches the mapped
    rows" is actually observable.
    """
    assert len(rows) == len(lens)
    rng = np.random.default_rng(seed)

    prompt_len_np = np.zeros(max_num_reqs, dtype=np.int32)
    prefill_len_np = np.zeros(max_num_reqs, dtype=np.int32)
    for req, (p_len, f_len) in zip(rows, lens):
        assert p_len <= f_len, "prompt_len must not exceed prefill_len"
        prompt_len_np[req] = p_len
        prefill_len_np[req] = f_len

    max_prefill_len = int(prefill_len_np[rows].max()) if rows else 0
    # One extra block of slack so the padding past prefill_len is real memory
    # the kernel could have read had the masks been wrong.
    max_model_len = max(max_prefill_len + BLOCK_SIZE, BLOCK_SIZE)

    if token_pool is None:
        token_pool = np.arange(vocab_size, dtype=np.int32)
    all_token_ids_np = np.full((max_num_reqs, max_model_len), PAST_PREFILL_TOKEN, dtype=np.int32)
    for req, (_, f_len) in zip(rows, lens):
        if f_len:
            all_token_ids_np[req, :f_len] = rng.choice(token_pool, size=f_len).astype(np.int32)

    num_bins = (vocab_size + 31) // 32
    # Poison every row: mapped rows must be cleared, unmapped rows preserved.
    prompt_bin_mask_np = rng.integers(
        np.iinfo(np.int32).min, np.iinfo(np.int32).max, size=(max_num_reqs, num_bins), dtype=np.int32
    )
    output_bin_counts_np = rng.integers(1, 9, size=(max_num_reqs, vocab_size)).astype(np.int32)

    idx_mapping_np = np.asarray(rows, dtype=np.int32)
    return dict(
        idx_mapping_np=idx_mapping_np,
        all_token_ids_np=all_token_ids_np,
        prompt_len_np=prompt_len_np,
        prefill_len_np=prefill_len_np,
        prompt_bin_mask_np=prompt_bin_mask_np,
        output_bin_counts_np=output_bin_counts_np,
        max_prefill_len=max_prefill_len,
    )


def _run_and_compare(inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the wrapper on device, compare both outputs against the oracle."""
    idx_mapping = torch.from_numpy(inputs["idx_mapping_np"]).to(DEVICE)
    all_token_ids = torch.from_numpy(inputs["all_token_ids_np"]).to(DEVICE)
    prompt_len = torch.from_numpy(inputs["prompt_len_np"]).to(DEVICE)
    prefill_len = torch.from_numpy(inputs["prefill_len_np"]).to(DEVICE)
    prompt_bin_mask = torch.from_numpy(inputs["prompt_bin_mask_np"]).to(DEVICE)
    output_bin_counts = torch.from_numpy(inputs["output_bin_counts_np"]).to(DEVICE)

    bincount(
        idx_mapping,
        all_token_ids,
        prompt_len,
        prefill_len,
        prompt_bin_mask,
        output_bin_counts,
        inputs["max_prefill_len"],
    )

    ref_mask_np, ref_counts_np = _ref_bincount(
        inputs["idx_mapping_np"],
        inputs["all_token_ids_np"],
        inputs["prompt_len_np"],
        inputs["prefill_len_np"],
        inputs["prompt_bin_mask_np"],
        inputs["output_bin_counts_np"],
    )
    ref_mask = torch.from_numpy(ref_mask_np).to(DEVICE)
    ref_counts = torch.from_numpy(ref_counts_np).to(DEVICE)

    _assert_equal(prompt_bin_mask, ref_mask, "prompt_bin_mask")
    _assert_equal(output_bin_counts, ref_counts, "output_bin_counts")
    return prompt_bin_mask, output_bin_counts


def _assert_equal(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    """Integer outputs: exact equality, with the offending rows in the message."""
    if torch.equal(actual, expected):
        return
    bad = (actual != expected).any(dim=-1).nonzero().flatten().tolist()
    raise AssertionError(
        f"{name} differs from the reference on request rows {bad}; "
        f"first mismatch at {(actual != expected).nonzero()[0].tolist()}: "
        f"got {actual[actual != expected][0].item()}, expected {expected[actual != expected][0].item()}"
    )


# (prompt_len, prefill_len) pairs, chosen around the BLOCK_SIZE=1024 boundary
# that splits the kernel's prompt branch from its output branch.
SHAPE_CASES = [
    pytest.param([3], [(10, 40)], id="single-req-single-block"),
    pytest.param([3], [(0, 40)], id="no-prompt"),
    pytest.param([3], [(40, 40)], id="no-output-tokens"),
    pytest.param([3], [(1024, 1500)], id="prompt-len-block-aligned"),
    pytest.param([3], [(1023, 1500)], id="prompt-len-block-aligned-minus-one"),
    pytest.param([3], [(1025, 1500)], id="prompt-len-block-aligned-plus-one"),
    pytest.param([3], [(2500, 3000)], id="prompt-spans-three-blocks"),
    pytest.param([3], [(100, 2048)], id="output-spans-blocks"),
    pytest.param([3], [(2048, 2048)], id="prefill-len-block-aligned"),
    # Six rows, non-contiguous and out of order: the kernel indexes every
    # tensor through `expanded_idx_mapping`, so a row-vs-token-index mix-up
    # would survive a batch of 1 or a contiguous 0..n-1 mapping.
    pytest.param(
        [5, 0, 7, 2, 6, 1],
        [(10, 40), (0, 0), (1024, 1500), (900, 2200), (33, 33), (1500, 1501)],
        id="six-reqs-mixed-lengths",
    ),
]


@pytest.mark.parametrize(("rows", "lens"), SHAPE_CASES)
@torch.inference_mode()
def test_bincount_matches_reference(rows, lens):
    """Every prompt/prefill length arrangement around the block boundary.

    ``BLOCK_SIZE`` splits the kernel into two guarded branches --
    ``block_idx * BLOCK_SIZE < prompt_len`` for the prompt bitmask and
    ``(block_idx + 1) * BLOCK_SIZE >= prompt_len`` for the output counts.  The
    aligned / +-1 triple pins the block on which the two branches hand over,
    which is exactly where an off-by-one would double-count or drop the tokens
    straddling the boundary.  The degenerate lengths (empty prompt, empty output
    range) cover the two branches firing alone.  A request with
    ``prefill_len == 0`` is covered inside "six-reqs-mixed-lengths" rather than
    on its own: alone it would make ``max_prefill_len`` zero and so launch a
    grid with a zero-sized dimension, which is outside the caller's contract.
    """
    _run_and_compare(_build_inputs(rows, lens))


@torch.inference_mode()
def test_bincount_sets_sign_bit_for_high_tokens():
    """Token ids with ``id % 32 == 31`` must land on the int32 sign bit.

    ``bit = tl.full((BLOCK_SIZE,), 1, tl.int32) << bit_idx`` overflows into the
    sign bit for ``bit_idx == 31``.  A reference that packs the mask in a wider
    integer, or a kernel that ever switched back to the float ``pow(2.0, ...)``
    form used briefly in #7757, disagrees precisely on those tokens.
    """
    # Only ids congruent to 31 mod 32, so every prompt token sets a sign bit.
    token_pool = np.arange(31, VOCAB_SIZE, 32, dtype=np.int32)
    inputs = _build_inputs([4], [(300, 900)], seed=7, token_pool=token_pool)

    # Guard the guard: the pool must actually exercise the sign bit.
    prompt_tokens = inputs["all_token_ids_np"][4, :300]
    assert (prompt_tokens % 32 == 31).all(), "fixture no longer exercises the sign-bit packing"

    prompt_bin_mask, _ = _run_and_compare(inputs)
    assert (prompt_bin_mask[4] < 0).any(), "no negative int32 bin produced; sign bit was never set"


@torch.inference_mode()
def test_bincount_accumulates_repeated_output_tokens():
    """Repeated output tokens must accumulate, not saturate at one.

    ``output_bin_counts`` is an ``atomic_add`` histogram while
    ``prompt_bin_mask`` is an idempotent ``atomic_or``.  Drawing the output
    tokens from a tiny pool forces counts well above 1, so a kernel that
    confused the two -- or lost adds to an atomics race across blocks -- fails
    here even though every shape-only assertion still passes.
    """
    token_pool = np.arange(0, 8, dtype=np.int32)
    inputs = _build_inputs([2], [(0, 2500)], seed=11, token_pool=token_pool)

    _, output_bin_counts = _run_and_compare(inputs)

    # Guard the guard: the fixture must produce a genuinely multi-count bin.
    assert output_bin_counts[2].max().item() > 1, "fixture no longer exercises repeated-token accumulation"
    assert output_bin_counts[2].sum().item() == 2500


@torch.inference_mode()
def test_bincount_ignores_tokens_past_prefill_len():
    """Positions at or beyond ``prefill_len`` must never be read.

    ``all_token_ids`` is a persistent ``max_model_len``-wide buffer, so the tail
    past ``prefill_len`` holds stale tokens from earlier requests.  Here the
    tail is filled with a single sentinel id, and the live range is drawn from a
    disjoint pool, so any leak shows up as a non-zero bin for the sentinel.
    """
    token_pool = np.arange(64, 128, dtype=np.int32)  # disjoint from PAST_PREFILL_TOKEN
    inputs = _build_inputs([1], [(200, 1200)], seed=3, token_pool=token_pool)
    assert PAST_PREFILL_TOKEN not in token_pool, "sentinel must not be a live token id"

    prompt_bin_mask, output_bin_counts = _run_and_compare(inputs)

    assert output_bin_counts[1, PAST_PREFILL_TOKEN].item() == 0
    sentinel_bit = 1 << (PAST_PREFILL_TOKEN % 32)
    bin_value = int(prompt_bin_mask[1, PAST_PREFILL_TOKEN // 32].item()) & 0xFFFFFFFF
    assert (bin_value & sentinel_bit) == 0


@torch.inference_mode()
def test_bincount_leaves_unmapped_rows_untouched():
    """Only the rows named by ``idx_mapping`` may be written.

    The wrapper resets exactly ``prompt_bin_mask[idx_mapping]`` and
    ``output_bin_counts[idx_mapping]``; the surrounding rows belong to other
    live requests whose penalty statistics must not be disturbed.  Both output
    tensors are poisoned before the call, so a stray full-tensor reset or a
    row-stride error is visible.
    """
    rows = [5, 0, 7, 2, 6, 1]
    lens = [(10, 40), (0, 0), (1024, 1500), (900, 2200), (33, 33), (1500, 1501)]
    inputs = _build_inputs(rows, lens, seed=5)
    untouched = sorted(set(range(MAX_NUM_REQS)) - set(rows))
    assert untouched, "scenario must leave at least one row unmapped"

    prompt_bin_mask, output_bin_counts = _run_and_compare(inputs)

    for req in untouched:
        assert torch.equal(
            prompt_bin_mask[req].cpu(), torch.from_numpy(inputs["prompt_bin_mask_np"][req])
        ), f"prompt_bin_mask row {req} was modified but is not in idx_mapping"
        assert torch.equal(
            output_bin_counts[req].cpu(), torch.from_numpy(inputs["output_bin_counts_np"][req])
        ), f"output_bin_counts row {req} was modified but is not in idx_mapping"


@torch.inference_mode()
def test_bincount_is_idempotent_across_calls():
    """A second call on the same rows must overwrite, not accumulate.

    ``PenaltiesState.apply_staged_writes`` calls ``bincount`` once per batch of
    newly admitted requests, and a request slot is reused as requests finish.
    If the wrapper's reset ever regressed, the ``atomic_add`` histogram would
    keep adding into the previous occupant's counts -- silently doubling the
    frequency penalty rather than crashing.
    """
    rows = [3, 6]
    lens = [(100, 800), (1024, 2000)]
    first = _build_inputs(rows, lens, seed=13)
    second = _build_inputs(rows, lens, seed=29)

    idx_mapping = torch.from_numpy(first["idx_mapping_np"]).to(DEVICE)
    prompt_len = torch.from_numpy(first["prompt_len_np"]).to(DEVICE)
    prefill_len = torch.from_numpy(first["prefill_len_np"]).to(DEVICE)
    prompt_bin_mask = torch.from_numpy(first["prompt_bin_mask_np"]).to(DEVICE)
    output_bin_counts = torch.from_numpy(first["output_bin_counts_np"]).to(DEVICE)

    for inputs in (first, second):
        bincount(
            idx_mapping,
            torch.from_numpy(inputs["all_token_ids_np"]).to(DEVICE),
            prompt_len,
            prefill_len,
            prompt_bin_mask,
            output_bin_counts,
            inputs["max_prefill_len"],
        )

    # Expected state is the second call's result computed from a clean slate.
    ref_mask_np, ref_counts_np = _ref_bincount(
        second["idx_mapping_np"],
        second["all_token_ids_np"],
        second["prompt_len_np"],
        second["prefill_len_np"],
        first["prompt_bin_mask_np"],
        first["output_bin_counts_np"],
    )
    _assert_equal(prompt_bin_mask, torch.from_numpy(ref_mask_np).to(DEVICE), "prompt_bin_mask")
    _assert_equal(output_bin_counts, torch.from_numpy(ref_counts_np).to(DEVICE), "output_bin_counts")


@torch.inference_mode()
def test_bincount_with_production_vocab_size():
    """One case at a realistic vocabulary, to exercise the real row strides.

    Every other case runs at ``VOCAB_SIZE=4096`` to keep ``output_bin_counts``
    small.  Qwen-class models use 151936, which makes ``output_bin_counts``
    stride 151936 and ``prompt_bin_mask`` stride 4748 -- neither a power of two,
    so a stride computed rather than passed in would go wrong here first.
    """
    vocab_size = 151936
    inputs = _build_inputs([4, 1], [(1500, 2600), (300, 300)], vocab_size=vocab_size, seed=17)
    assert (vocab_size + 31) // 32 == 4748

    _run_and_compare(inputs)
