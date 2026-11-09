"""The one BigWig writer for test fixtures."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
from figwig import write_bigwig


def write_intervals(
    path: str | Path,
    contig_sizes: Mapping[str, int],
    intervals: Iterable[tuple[str, int, int, float]],
) -> Path:
    """Write ``(contig, start, end, value)`` intervals to a BigWig at `path`.

    Intervals may come in any order; they are sorted into header order, which
    figwig requires. Two choices matter for what a reader sees:

    - ``missing=nan`` on write, so an explicit ``0.0`` interval is *stored*.
      `dienr.counts` counts a zero-valued interval as covered (it enters a
      ``mean`` denominator), so dropping it would change the fixture.
    - Every contig in `contig_sizes` is declared in the header whether or not it
      carries an interval, and an empty file is readable. (`pybigtools`, the
      previous writer, dropped quiet contigs and could not write an empty file,
      which forced zero-valued placeholder bases into every fixture.)
    """
    order = {contig: rank for rank, contig in enumerate(contig_sizes)}
    rows = sorted(intervals, key=lambda row: (order[row[0]], row[1]))
    write_bigwig(
        str(path),
        dict(contig_sizes),
        np.array([row[0] for row in rows], dtype=str),
        np.array([row[1] for row in rows], dtype=np.int64),
        np.array([row[3] for row in rows], dtype=np.float32),
        ends=np.array([row[2] for row in rows], dtype=np.int64),
        missing=np.nan,
        zooms=0,  # nothing here draws the file in a browser
        n_jobs=1,
    )
    return Path(path)
