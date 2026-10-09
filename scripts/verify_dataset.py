#!/usr/bin/env python3
"""Dataset integrity gate for the reproduction pipeline (S08/T03).

Run first in every reproduction so the pipeline can never silently train on
different data. Verifies, in order:

1. the pinned raw CSV under ``data/raw/`` matches its pinned SHA-256 digest
   and the declared row count and schema (``heart.data.load.load_dataset``);
2. the committed split artifacts under ``data/processed/splits/v1/`` match
   the digests recorded in their manifest (``heart.data.split.load_split_frames``,
   which raises :class:`SplitIntegrityError` on drift).

Exit codes (machine-readable for `make` / CI):

* ``0`` — every check passed;
* ``2`` — raw dataset missing, checksum mismatch, or schema failure;
* ``3`` — split artifacts missing or drifted from their manifest;
* ``10`` — unexpected internal error (bug, not data).

The raw file is never downloaded here: a clean checkout must carry the
committed copy. A missing raw file is an abort with instructions, not a silent
fetch that could mask substitution of a different dataset.
"""

from __future__ import annotations

import argparse
import sys

HEART_EXIT = {
    "raw": 2,
    "split": 3,
    "unexpected": 10,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/verify_dataset.py",
        description=(
            "Verify the pinned dataset and committed split artifacts before "
            "any training run. Aborts the pipeline with a clear message on "
            "any digest, row-count, or schema drift."
        ),
    )
    parser.add_argument(
        "--split-version",
        default="v1",
        help="Split version to integrity-check (default: v1).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Imported lazily so a clean tree prints the failure message instead of a
    # traceback when the package itself is fine but the data is not.
    try:
        from heart.data.load import load_dataset
        from heart.data.split import load_split_frames
    except ImportError as exc:  # pragma: no cover - environment, not logic
        print(f"dataset check could not import the heart package: {exc}")
        print("Create the environment first: make reproduce (or pip install -e .)")
        return HEART_EXIT["unexpected"]

    try:
        dataset = load_dataset(allow_download=False)
    except Exception as exc:  # named errors upstream; report uniformly here
        print("FAIL: raw dataset integrity check did not pass.")
        print(f"  {type(exc).__name__}: {exc}")
        print(
            "  data/raw/heart.csv is missing, modified, or no longer matches "
            "the pinned digest. Restore the committed copy before training."
        )
        return HEART_EXIT["raw"]
    report = dataset.report().strip() if hasattr(dataset, "report") else str(dataset)
    print("OK: raw dataset matches the pinned digest, row count, and schema.")
    print("  " + "\n  ".join(report.splitlines()[-3:]))

    try:
        train, test = load_split_frames(version=args.split_version)
    except Exception as exc:
        print(
            f"FAIL: committed split {args.split_version!r} does not match its "
            "manifest."
        )
        print(f"  {type(exc).__name__}: {exc}")
        print(
            "  Restore the committed split artifacts (regeneration changes "
            "published numbers and must never happen silently)."
        )
        return HEART_EXIT["split"]
    print(
        f"OK: split {args.split_version!r} verified against its manifest "
        f"(train={len(train)} rows, test={len(test)} rows)."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    sys.exit(main())
