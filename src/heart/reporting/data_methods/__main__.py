"""CLI entry point: ``python -m heart.reporting.data_methods`` (M005/S01/T02).

Mirrors :func:`heart.data.quality.main` and
:func:`heart.reporting.leaderboard`'s CLI style: load the pinned dataset
through the integrity-checked loader, run the four-stage pre-split EDA, and
write the provenance-locked artifacts + manifest + part A report.

Defaults (overridable):

* artifacts & manifest → ``reports/data-methods/``
* rendered report      → ``reports/data-methods.md``

Failure behaviour: named :class:`DataMethodsError`-family exceptions are
logged and turned into a non-zero exit; the report is written atomically at
the very end, so a failing run never leaves a partial report behind.
All stage writes go through the same atomic pattern, so an interrupted CLI
run leaves the previous committed report intact.
"""

from __future__ import annotations

import argparse
import logging

from heart.config import ensure_project_dirs
from heart.data.load import load_dataset
from heart.reporting.data_methods.eda import DataMethodsError
from heart.reporting.data_methods.report import (
    DATAMETHODS_DIR,
    REPORT_PATH,
    generate_report,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (mirrors ``python -m heart.data.quality``)."""
    parser = argparse.ArgumentParser(
        prog="python -m heart.reporting.data_methods",
        description=(
            "Run the four-stage pre-split EDA on the pinned dataset, lock the "
            "artifacts with a SHA-256 manifest, and render reports/data-methods.md "
            "parts A-F (stage tables, per-model treatment audit, and the "
            "pre-declared treatment plan)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Artifact directory (default: {DATAMETHODS_DIR}).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help=f"Report output path (default: {REPORT_PATH}).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run the EDA and print counts without writing any artifact or report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load the pinned dataset, run the EDA, and write the report."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ensure_project_dirs()

    dataset = load_dataset()
    if args.no_write:
        from heart.reporting.data_methods.eda import run_eda

        result = run_eda(dataset.frame, artifact_dir=None)
        stage1 = result["stage1_hygiene"]
        missing = stage1["missingness"]
        target = result["stage2_univariate"]["target"]
        print(
            f"pre-split EDA: {result['meta']['row_count']} rows x "
            f"{result['meta']['column_count']} columns"
        )
        print(
            f"missingness: random-NaN cells {missing['total_random_nan_cells']}, "
            f"declared sentinel-zero cells {missing['total_sentinel_zero_cells']}"
        )
        print(
            f"target prevalence: {target['prevalence']:.4f} "
            f"(imbalance {target['imbalance_ratio']:.4f})"
        )
        return 0

    try:
        written = generate_report(
            dataset,
            output_dir=args.output_dir if args.output_dir is not None else DATAMETHODS_DIR,
            report_path=args.report if args.report is not None else REPORT_PATH,
        )
    except DataMethodsError as exc:
        logger.error("data-methods report failed: %s", exc)
        return 2

    print(f"wrote report: {written}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    raise SystemExit(main())
