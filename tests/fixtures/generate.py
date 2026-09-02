"""Write parity scenario inputs to disk for the R export script.

Run from the repository root::

    uv run python -m tests.fixtures.generate --output build/parity-inputs

This writes, per scenario, a GTF and two stranded BigWigs, plus a single
``scenarios.json`` describing the design parameters so the R side does not
duplicate them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .scenarios import SCENARIOS, write_inputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    spec = []
    for scenario in SCENARIOS:
        write_inputs(scenario, args.output)
        spec.append(
            {
                "name": scenario.name,
                "bin_size": scenario.bin_size,
                "distance": scenario.distance,
                "mask_start_bins": list(scenario.mask_start_bins),
                "mask_end_bins": list(scenario.mask_end_bins),
                "bin_operation": scenario.bin_operation,
                "transform": scenario.transform,
            }
        )
    (args.output / "scenarios.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(f"wrote {len(spec)} scenario(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
