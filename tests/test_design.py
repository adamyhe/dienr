"""Bin geometry, masking, and exact-equivalence collapsing."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from dienr.annotation import Annotation
from dienr.design import DesignConfig, build_design


def annotation(rows: list[tuple[str, str, int, int, str]]) -> Annotation:
    return Annotation.from_frame(
        pl.DataFrame(
            rows,
            schema=["transcript_id", "contig", "start", "end", "strand"],
            orient="row",
        )
    )


# --- bin geometry -----------------------------------------------------------


def test_bins_are_anchored_at_locus_start():
    """Bins start at the locus, not on a global multiple of bin_size."""
    design = build_design(annotation([("a", "chr1", 130, 630, "+")]), bin_size=250)
    bins = design.bins.sort("bin_index")
    assert bins["start"].to_list() == [130, 380]
    assert bins["end"].to_list() == [380, 630]


def test_bin_count_is_ceiling_of_locus_width():
    # 500 bases at width 250 needs exactly 2 bins; 501 needs 3.
    assert build_design(annotation([("a", "chr1", 0, 500, "+")]), bin_size=250).bin_count(0) == 2
    assert build_design(annotation([("a", "chr1", 0, 501, "+")]), bin_size=250).bin_count(0) == 3


def test_terminal_bin_records_true_effective_length():
    """The last bin overhangs the locus; its usable extent is shorter."""
    design = build_design(annotation([("a", "chr1", 0, 600, "+")]), bin_size=250)
    assert design.effective_lengths(0).tolist() == [250.0, 250.0, 100.0]


def test_single_base_transcript_yields_one_bin():
    design = build_design(annotation([("a", "chr1", 100, 101, "+")]), bin_size=250)
    assert design.bin_count(0) == 1
    assert design.effective_lengths(0).tolist() == [1.0]


def test_subbin_transcript_rounds_to_an_unsupported_model():
    """A transcript occupying under half a bin rounds away entirely.

    1/250 rounds to 0, so the model column is all zeros and no data can
    constrain its abundance. That must be reported, not silently fitted.
    """
    design = build_design(annotation([("a", "chr1", 100, 101, "+")]), bin_size=250)
    assert design.models[0][0, 0] == 0.0
    assert design.unsupported_models.height == 1
    assert not design.transcript_map["has_support"][0]


def test_subbin_transcript_is_supported_under_ceiling():
    """`ceiling` keeps it visible, at the cost of overstating occupancy."""
    design = build_design(
        annotation([("a", "chr1", 100, 101, "+")]), bin_size=250, bin_operation="ceiling"
    )
    assert design.models[0][0, 0] == 1.0
    assert design.unsupported_models.height == 0


def test_supported_models_are_flagged():
    design = build_design(annotation([("a", "chr1", 0, 1000, "+")]), bin_size=250)
    assert design.model_table["has_support"].to_list() == [True]
    assert design.unsupported_models.height == 0


# --- occupancy matrix -------------------------------------------------------


def test_full_bin_occupancy_is_one():
    design = build_design(annotation([("a", "chr1", 0, 750, "+")]), bin_size=250)
    assert design.models[0][:, 0].tolist() == [1.0, 1.0, 1.0]


def test_partial_occupancy_rounds_by_bin_operation():
    """A transcript covering 60% of its final bin rounds differently per mode."""
    rows = [("a", "chr1", 0, 400, "+")]
    # Bins: [0,250) fully covered, [250,500) covered 150/250 = 0.6.
    assert (
        build_design(annotation(rows), bin_size=250, bin_operation="round").models[0][1, 0] == 1.0
    )
    assert (
        build_design(annotation(rows), bin_size=250, bin_operation="floor").models[0][1, 0] == 0.0
    )
    assert (
        build_design(annotation(rows), bin_size=250, bin_operation="ceiling").models[0][1, 0] == 1.0
    )


def test_offset_transcripts_occupy_distinct_bins():
    design = build_design(
        annotation([("a", "chr1", 0, 1000, "+"), ("b", "chr1", 500, 1000, "+")]),
        bin_size=250,
        bin_operation="round",
    )
    matrix = design.models[0]
    assert matrix.shape == (4, 2)
    # tx b starts at bin 2, so its first two bins are empty.
    columns = {tuple(matrix[:, i]) for i in range(matrix.shape[1])}
    assert (1.0, 1.0, 1.0, 1.0) in columns
    assert (0.0, 0.0, 1.0, 1.0) in columns


# --- masking ----------------------------------------------------------------


def test_no_masking_by_default():
    design = build_design(annotation([("a", "chr1", 0, 2500, "+")]), bin_size=250)
    assert not design.mask(0).any()


def test_plus_strand_start_mask_covers_first_bins():
    design = build_design(
        annotation([("a", "chr1", 0, 2500, "+")]), bin_size=250, mask_start_bins=(2, 0)
    )
    assert design.mask(0).tolist() == [True, True] + [False] * 8


def test_plus_strand_end_mask_covers_last_bins():
    design = build_design(
        annotation([("a", "chr1", 0, 2500, "+")]), bin_size=250, mask_end_bins=(2, 0)
    )
    assert design.mask(0).tolist() == [False] * 8 + [True, True]


def test_minus_strand_mask_is_mirrored():
    """The 5' end of a minus-strand transcript is its highest coordinate."""
    plus = build_design(
        annotation([("a", "chr1", 0, 2500, "+")]), bin_size=250, mask_start_bins=(3, 0)
    )
    minus = build_design(
        annotation([("a", "chr1", 0, 2500, "-")]), bin_size=250, mask_start_bins=(3, 0)
    )
    assert plus.mask(0).tolist() == minus.mask(0)[::-1].tolist()


def test_outside_mask_extends_beyond_the_transcript():
    """`outside` bins run away from the transcript, into flanking sequence."""
    design = build_design(
        annotation([("a", "chr1", 0, 1000, "+"), ("b", "chr1", 1000, 2000, "+")]),
        bin_size=250,
        distance=250,
        mask_start_bins=(0, 2),
    )
    # tx b begins at bin 4; two bins outside its 5' end are bins 3 and 2.
    assert design.mask(0)[2:4].tolist() == [True, True]


def test_mask_is_clipped_at_locus_edges():
    """Masking more bins than exist must not wrap or raise."""
    design = build_design(
        annotation([("a", "chr1", 0, 500, "+")]),
        bin_size=250,
        mask_start_bins=(0, 50),
        mask_end_bins=(0, 50),
    )
    assert design.mask(0).size == 2


def test_masked_bins_are_zeroed_in_the_model_matrix():
    design = build_design(
        annotation([("a", "chr1", 0, 2500, "+")]), bin_size=250, mask_start_bins=(2, 0)
    )
    assert design.models[0][:2, :].sum() == 0.0


def test_mask_is_the_union_over_transcripts_in_a_locus():
    design = build_design(
        annotation([("a", "chr1", 0, 2500, "+"), ("b", "chr1", 1000, 2500, "+")]),
        bin_size=250,
        mask_start_bins=(1, 0),
    )
    # Bin 0 masked for tx a, bin 4 masked for tx b.
    masked = np.flatnonzero(design.mask(0)).tolist()
    assert masked == [0, 4]


# --- equivalence collapsing -------------------------------------------------


def test_identical_transcripts_collapse_to_one_model():
    design = build_design(
        annotation([("a", "chr1", 0, 1000, "+"), ("b", "chr1", 0, 1000, "+")]),
        bin_size=250,
    )
    assert design.models[0].shape[1] == 1
    assert design.transcript_map["equivalence_class_size"].to_list() == [2, 2]
    assert design.transcript_map["exactly_identifiable"].to_list() == [False, False]


def test_transcripts_indistinguishable_at_bin_scale_collapse():
    """Ends differing by less than a bin are not resolvable, and must collapse."""
    design = build_design(
        annotation([("a", "chr1", 0, 1000, "+"), ("b", "chr1", 0, 1010, "+")]),
        bin_size=250,
        bin_operation="round",
    )
    assert design.models[0].shape[1] == 1


def test_distinguishable_transcripts_stay_separate():
    design = build_design(
        annotation([("a", "chr1", 0, 1000, "+"), ("b", "chr1", 0, 2000, "+")]),
        bin_size=250,
    )
    assert design.models[0].shape[1] == 2
    assert design.transcript_map["exactly_identifiable"].to_list() == [True, True]


def test_masking_can_make_transcripts_indistinguishable():
    """Masking the 5' ends removes the only feature separating these two."""
    rows = [("a", "chr1", 0, 2000, "+"), ("b", "chr1", 250, 2000, "+")]
    unmasked = build_design(annotation(rows), bin_size=250)
    masked = build_design(annotation(rows), bin_size=250, mask_start_bins=(0, 1))
    assert unmasked.models[0].shape[1] == 2
    assert masked.models[0].shape[1] == 1


def test_every_transcript_maps_to_exactly_one_model():
    design = build_design(
        annotation(
            [
                ("a", "chr1", 0, 1000, "+"),
                ("b", "chr1", 0, 1000, "+"),
                ("c", "chr1", 0, 2000, "+"),
                ("d", "chr2", 0, 500, "-"),
            ]
        ),
        bin_size=250,
    )
    assert design.transcript_map.height == 4
    assert design.transcript_map["transcript_id"].n_unique() == 4


def test_model_ids_are_stable_across_builds():
    rows = [("a", "chr1", 0, 1000, "+"), ("b", "chr1", 0, 2000, "+")]
    first = build_design(annotation(rows), bin_size=250)
    second = build_design(annotation(rows), bin_size=250)
    assert first.model_table["model_id"].to_list() == second.model_table["model_id"].to_list()
    assert first.digest == second.digest


def test_design_digest_changes_with_configuration():
    rows = [("a", "chr1", 0, 2000, "+")]
    base = build_design(annotation(rows), bin_size=250)
    assert build_design(annotation(rows), bin_size=100).digest != base.digest
    assert build_design(annotation(rows), bin_size=250, mask_start_bins=(1, 0)).digest != (
        base.digest
    )


# --- configuration ----------------------------------------------------------


def test_distance_defaults_to_bin_size():
    assert DesignConfig(bin_size=250).effective_distance == 250
    assert DesignConfig(bin_size=250, distance=0).effective_distance == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"bin_size": 0}, "bin_size must be positive"),
        ({"bin_operation": "trunc"}, "bin_operation must be one of"),
        ({"mask_start_bins": (1,)}, "mask_start_bins must be two"),
        ({"mask_end_bins": (-1, 0)}, "mask_end_bins must be two"),
        ({"distance": -5}, "distance must be non-negative"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DesignConfig(**kwargs)


def test_multiple_loci_produce_independent_matrices():
    design = build_design(
        annotation([("a", "chr1", 0, 1000, "+"), ("b", "chr1", 500_000, 501_000, "+")]),
        bin_size=250,
    )
    assert design.n_loci == 2
    assert [m.shape for m in design.models] == [(4, 1), (4, 1)]


def test_mask_anchors_on_unrounded_occupancy():
    """Masks must be derived before `bin_operation` rounds occupancy away.

    A transcript starting 210 bases into a 250bp bin occupies 40/250 = 0.16 of
    it. Rounding first would zero that bin, moving the transcript's 5' anchor
    one bin later and shifting every masked bin with it. The reference computes
    masks from the fractional matrix and rounds afterwards.
    """
    design = build_design(
        annotation([("anchor", "chr1", 0, 5000, "+"), ("late", "chr1", 210, 4090, "+")]),
        bin_size=250,
        mask_start_bins=(1, 0),
    )
    masked = np.flatnonzero(design.mask(0)).tolist()
    # 'late' starts in bin 0 (210 of 250 bases in, so 0.16 occupancy), so both
    # transcripts anchor at bin 0 and only bin 0 is masked. Rounding first
    # would have anchored 'late' at bin 1 and masked bin 1 too.
    assert masked == [0]


def test_subbin_start_does_not_shift_the_mask_on_minus_strand():
    design = build_design(
        annotation([("anchor", "chr1", 0, 5000, "-"), ("early", "chr1", 910, 4790, "-")]),
        bin_size=250,
        mask_start_bins=(1, 0),
    )
    n_bins = design.bin_count(0)
    masked = np.flatnonzero(design.mask(0)).tolist()
    # On the minus strand the 5' end is the high coordinate. 'early' ends at
    # 4790, which is 40 bases into bin 19, so it anchors there rather than 18.
    assert n_bins - 1 in masked
