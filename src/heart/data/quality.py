"""Data-quality audit for the fedesoriano heart-failure dataset.

This module owns the one quality question that can silently corrupt every
downstream benchmark: **impossible zeros**. Two columns use ``0`` as a
sentinel for "not recorded":

* ``RestingBP`` — resting blood pressure of 0 mm Hg is incompatible with life.
* ``Cholesterol`` — a serum cholesterol of 0 mg/dl is not physiologically
  possible.

Both must be treated as missing and handled by an **explicit, logged,
declared policy**. A zero that reaches a model as a real measurement is a data
bug, not a feature.

Policy (chosen with the counts in hand)
---------------------------------------
On the pinned 918-row dataset:

* ``RestingBP`` has **1** zero (0.11 %).
* ``Cholesterol`` has **172** zeros (18.74 %), and those rows are heavily
  skewed toward the positive class (152 positive vs. 20 negative), so
  row-dropping would bias the target distribution.

Dropping every affected row would remove 18.7 % of the dataset and skew
prevalence, so the declared action is **median imputation**: zeros become
missing, then each column is filled with its **training-fold median**. The
median is robust to the remaining skew and keeps the full row count.

Leakage safety
--------------
Imputation is data-dependent, so the statistic must never be fit on the whole
dataset before splitting. :class:`ZeroMedianImputer` is an sklearn-compatible
transformer whose ``fit`` computes medians from whatever rows it is given.
S01/T05 wires it into the pipeline and fits it on **training rows only**; the
leakage test proves a test row cannot influence a fitted transform.

Observability
-------------
:func:`audit_quality` returns a :class:`QualityAudit` (machine-readable via
``to_dict``); :func:`render_quality_report` / :func:`write_quality_report`
produce ``reports/data_quality.md``; ``python -m heart.data.quality`` prints
the audit and writes the report. Applying the policy logs the action and the
number of cells replaced at ``INFO`` — the policy is never silent.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from heart.config import PROJECT_ROOT, REPORTS_DIR, ensure_project_dirs
from heart.data.load import LoadedDataset, load_dataset
from heart.data.schema import TARGET_COLUMN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Declared policy constants
# ---------------------------------------------------------------------------

#: Columns in which a numeric zero is physiologically impossible.
ZERO_AS_MISSING_COLUMNS: tuple[str, ...] = ("RestingBP", "Cholesterol")

#: The sentinel value that means "missing" in those columns.
ZERO_SENTINEL: int = 0

#: Declared policy actions.
MEDIAN_IMPUTE: str = "median_impute"
DROP_ROWS: str = "drop_rows"

#: Name of the generated report artifact (relative to the reports directory).
QUALITY_REPORT_NAME: str = "data_quality.md"


# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class QualityError(Exception):
    """Base class for data-quality policy failures."""


class MissingQualityColumnError(QualityError):
    """A column named by the policy is absent from the frame."""


class NoValidValuesError(QualityError):
    """A column has no non-zero values from which to compute a statistic."""


class UnknownPolicyError(QualityError):
    """The declared policy action is not implemented."""


# ---------------------------------------------------------------------------
# Policy declaration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZeroPolicy:
    """An explicit, inspectable zero-as-missing contract.

    ``action`` must be one of :data:`MEDIAN_IMPUTE` or :data:`DROP_ROWS`.
    ``rationale`` is free text recorded in the report so a reviewer can see
    *why* the policy was chosen, not just *what* it is.
    """

    action: str
    columns: tuple[str, ...] = ZERO_AS_MISSING_COLUMNS
    sentinel: int = ZERO_SENTINEL
    rationale: str = ""

    def describe(self) -> str:
        return (
            f"action={self.action}; columns={list(self.columns)}; "
            f"sentinel={self.sentinel}"
        )


#: The declared default policy for this dataset.
DEFAULT_ZERO_POLICY = ZeroPolicy(
    action=MEDIAN_IMPUTE,
    rationale=(
        "Cholesterol carries 172 impossible zeros (18.7% of rows), skewed toward "
        "the positive class (152 pos / 20 neg), so dropping them would bias "
        "prevalence; RestingBP carries 1. Median imputation keeps every row and "
        "is fit on training rows only, so no test row influences the statistic."
    ),
)


# ---------------------------------------------------------------------------
# Audit value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZeroCount:
    """Impossible-zero tally for one column."""

    column: str
    count: int
    share: float
    by_target: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "column": self.column,
            "count": self.count,
            "share": self.share,
            "by_target": {str(k): v for k, v in self.by_target.items()},
        }


@dataclass(frozen=True)
class ClassBalance:
    """Target distribution for ``HeartDisease``."""

    counts: dict[int, int]
    total: int

    @property
    def negative_count(self) -> int:
        return int(self.counts.get(0, 0))

    @property
    def positive_count(self) -> int:
        return int(self.counts.get(1, 0))

    @property
    def prevalence(self) -> float:
        """Share of positive (disease-present) rows."""
        return self.positive_count / self.total if self.total else 0.0

    @property
    def imbalance_ratio(self) -> float:
        """Majority:minority ratio (1.0 == perfectly balanced)."""
        if not self.counts:
            return 0.0
        smallest = min(self.counts.values())
        return max(self.counts.values()) / smallest if smallest else float("inf")

    def to_dict(self) -> dict[str, object]:
        return {
            "counts": {str(k): v for k, v in sorted(self.counts.items())},
            "total": self.total,
            "prevalence": self.prevalence,
            "imbalance_ratio": self.imbalance_ratio,
        }


@dataclass(frozen=True)
class QualityAudit:
    """Aggregate result of the zero/class-balance audit."""

    row_count: int
    zero_counts: dict[str, ZeroCount]
    class_balance: ClassBalance
    policy: ZeroPolicy
    affected_rows: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "row_count": self.row_count,
            "affected_rows": self.affected_rows,
            "zero_counts": {c: z.to_dict() for c, z in self.zero_counts.items()},
            "class_balance": self.class_balance.to_dict(),
            "policy": {
                "action": self.policy.action,
                "columns": list(self.policy.columns),
                "sentinel": self.policy.sentinel,
                "rationale": self.policy.rationale,
            },
        }


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise MissingQualityColumnError(
            f"Frame is missing policy column(s) {missing}. Present columns: "
            f"{list(frame.columns)}. Load the dataset through "
            "heart.data.load.load_dataset before auditing quality."
        )


def invalid_zero_counts(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] = ZERO_AS_MISSING_COLUMNS,
    sentinel: int = ZERO_SENTINEL,
    target: str = TARGET_COLUMN,
) -> dict[str, ZeroCount]:
    """Count physiologically impossible zeros per policy column.

    Also breaks the count down by target class so the report can show whether
    the missingness is associated with the outcome (it is, for Cholesterol).
    """
    _require_columns(frame, columns)
    total = len(frame)
    result: dict[str, ZeroCount] = {}
    for column in columns:
        mask = frame[column] == sentinel
        count = int(mask.sum())
        by_target: dict[int, int] = {}
        if target in frame.columns and count:
            by_target = {
                int(k): int(v)
                for k, v in frame.loc[mask, target].value_counts().items()
            }
        result[column] = ZeroCount(
            column=column,
            count=count,
            share=(count / total) if total else 0.0,
            by_target=by_target,
        )
    return result


def compute_class_balance(
    frame: pd.DataFrame, *, target: str = TARGET_COLUMN
) -> ClassBalance:
    """Return the ``HeartDisease`` class distribution."""
    if target not in frame.columns:
        raise MissingQualityColumnError(
            f"Target column {target!r} is absent; cannot compute class balance."
        )
    counts = {int(k): int(v) for k, v in frame[target].value_counts().items()}
    return ClassBalance(counts=counts, total=int(len(frame)))


def count_affected_rows(
    frame: pd.DataFrame, *, columns: tuple[str, ...] = ZERO_AS_MISSING_COLUMNS
) -> int:
    """Rows carrying at least one impossible zero (the cost of row-dropping)."""
    _require_columns(frame, columns)
    return int(frame[list(columns)].eq(ZERO_SENTINEL).any(axis=1).sum())


def audit_quality(
    frame: pd.DataFrame, *, policy: ZeroPolicy = DEFAULT_ZERO_POLICY
) -> QualityAudit:
    """Measure impossible zeros and class balance under the declared policy."""
    if policy.action not in (MEDIAN_IMPUTE, DROP_ROWS):
        raise UnknownPolicyError(
            f"Unknown zero policy action {policy.action!r}; expected one of "
            f"{MEDIAN_IMPUTE!r} or {DROP_ROWS!r}."
        )
    return QualityAudit(
        row_count=int(len(frame)),
        zero_counts=invalid_zero_counts(
            frame, columns=policy.columns, sentinel=policy.sentinel
        ),
        class_balance=compute_class_balance(frame),
        policy=policy,
        affected_rows=count_affected_rows(frame, columns=policy.columns),
    )


# ---------------------------------------------------------------------------
# Leakage-safe transformer
# ---------------------------------------------------------------------------


def _as_frame(X: object) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise QualityError(
            f"Expected a pandas.DataFrame, got {type(X).__name__}. The zero "
            "policy addresses named columns, so a bare array is not sufficient."
        )
    return X


class ZeroMedianImputer(BaseEstimator, TransformerMixin):
    """Flag impossible zeros as missing and impute column medians.

    sklearn-compatible so it can sit inside a :class:`~sklearn.pipeline.Pipeline`.
    ``fit`` learns the median from the rows it is given; call it on training
    rows only. ``transform`` never recomputes statistics, so a value present in
    the transformed frame cannot influence the imputation of any other frame.

    Parameters
    ----------
    columns:
        Columns in which ``sentinel`` means missing.
    sentinel:
        The value that signals "not recorded" (default ``0``).
    """

    def __init__(
        self,
        columns: tuple[str, ...] = ZERO_AS_MISSING_COLUMNS,
        sentinel: int = ZERO_SENTINEL,
    ) -> None:
        self.columns = columns
        self.sentinel = sentinel

    # -- fitting ----------------------------------------------------------
    def fit(self, X: object, y: object = None) -> "ZeroMedianImputer":
        frame = _as_frame(X)
        _require_columns(frame, tuple(self.columns))

        medians: dict[str, float] = {}
        for column in self.columns:
            observed = frame[column].where(frame[column] != self.sentinel)
            median = observed.median()
            if pd.isna(median):
                raise NoValidValuesError(
                    f"Column {column!r} has no non-zero values in the fit frame, "
                    "so no median can be computed. Inspect the data or drop the "
                    "column from the policy before imputing."
                )
            medians[column] = float(median)

        self.medians_ = medians
        self.n_features_in_ = int(frame.shape[1])
        if all(isinstance(c, str) for c in frame.columns):
            self.feature_names_in_ = pd.Index([str(c) for c in frame.columns])
        logger.info(
            "Fitted zero-as-missing policy on %d rows: medians=%s",
            len(frame),
            {c: round(v, 3) for c, v in medians.items()},
        )
        return self

    # -- applying ---------------------------------------------------------
    def transform(self, X: object) -> pd.DataFrame:
        frame = _as_frame(X)
        _require_columns(frame, tuple(self.columns))
        if not hasattr(self, "medians_"):
            raise QualityError(
                "ZeroMedianImputer.transform called before fit; call fit on the "
                "training rows first."
            )

        out = frame.copy()
        for column in self.columns:
            before = int((out[column] == self.sentinel).sum())
            flagged = out[column].where(out[column] != self.sentinel)
            out[column] = flagged.fillna(self.medians_[column])
            remaining = int((out[column] == self.sentinel).sum())
            if remaining:
                # A genuine measured value equal to the sentinel cannot exist
                # after fillna; this guards against a sentinel that survived as
                # a median, which would silently defeat the policy.
                raise QualityError(
                    f"Column {column!r} still holds {remaining} sentinel value(s) "
                    "after imputation; refusing to emit a frame that presents "
                    "impossible zeros as measurements."
                )
            logger.info(
                "Zero policy applied to %r: %d sentinel value(s) imputed with "
                "median %.3f",
                column,
                before,
                self.medians_[column],
            )
        return out

    def get_feature_names_out(self, input_features: object = None) -> pd.Index:
        if input_features is not None:
            if isinstance(input_features, pd.DataFrame):
                return pd.Index([str(c) for c in input_features.columns])
            return pd.Index([str(c) for c in input_features])
        if hasattr(self, "feature_names_in_"):
            return pd.Index(self.feature_names_in_)
        return pd.Index([])


def fit_zero_policy(
    fit_frame: pd.DataFrame, *, policy: ZeroPolicy = DEFAULT_ZERO_POLICY
) -> ZeroMedianImputer:
    """Fit the declared policy on ``fit_frame`` (training rows only)."""
    if policy.action != MEDIAN_IMPUTE:
        raise UnknownPolicyError(
            f"fit_zero_policy implements {MEDIAN_IMPUTE!r}; got {policy.action!r}."
        )
    imputer = ZeroMedianImputer(columns=policy.columns, sentinel=policy.sentinel)
    return imputer.fit(fit_frame)


def apply_zero_policy(
    frame: pd.DataFrame,
    imputer: ZeroMedianImputer,
    *,
    policy: ZeroPolicy = DEFAULT_ZERO_POLICY,
) -> pd.DataFrame:
    """Transform ``frame`` with a previously fitted policy.

    Passing a fitted ``imputer`` guarantees the statistics came from the frame
    it was fit on (training rows), which is what keeps the split leakage-safe.
    """
    del policy  # the fitted imputer carries the effective policy
    return imputer.transform(frame)


def clean_with_policy(
    frame: pd.DataFrame,
    *,
    fit_frame: pd.DataFrame | None = None,
    policy: ZeroPolicy = DEFAULT_ZERO_POLICY,
) -> pd.DataFrame:
    """Convenience: fit on ``fit_frame`` (default ``frame``) and transform.

    When ``fit_frame`` is ``None`` the statistics are learned from ``frame``
    itself. That is correct for an audit or a final full-data fit, but it is
    **not** safe for producing held-out evaluation data — pass ``fit_frame``
    (or use :class:`ZeroMedianImputer` inside a split pipeline) instead.
    """
    if policy.action != MEDIAN_IMPUTE:
        raise UnknownPolicyError(
            f"clean_with_policy implements {MEDIAN_IMPUTE!r}; got {policy.action!r}."
        )
    source = frame if fit_frame is None else fit_frame
    if fit_frame is None:
        logger.info(
            "Fitting zero policy on the same %d row(s) being transformed; "
            "audit/full-fit path only, not leakage-safe for evaluation.",
            len(frame),
        )
    imputer = fit_zero_policy(source, policy=policy)
    return apply_zero_policy(frame, imputer, policy=policy)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_share(share: float) -> str:
    return f"{share * 100:.2f}%"


def _display_path(path: str | Path) -> str:
    """Render ``path`` relative to the repo root when it lives underneath it.

    Keeps the committed report portable across checkouts (no absolute paths).
    """
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(candidate)


def render_quality_report(
    audit: QualityAudit,
    *,
    affected_rows: int | None = None,
    dataset: LoadedDataset | None = None,
    report_name: str = QUALITY_REPORT_NAME,
) -> str:
    """Render the audit as the ``reports/data_quality.md`` markdown document."""
    policy = audit.policy
    balance = audit.class_balance
    if affected_rows is None:
        affected_rows = audit.affected_rows
    lines: list[str] = [
        "# Data-Quality Audit — fedesoriano Heart Failure Dataset",
        "",
        f"Generated by `python -m heart.data.quality` into `reports/{report_name}`.",
        "Regenerate rather than hand-edit; every number below is computed from",
        "the pinned dataset.",
        "",
        "## Dataset",
        "",
        f"- rows: **{audit.row_count}**",
    ]
    if dataset is not None:
        lines.append(f"- source: `{_display_path(dataset.source_path)}`")
        lines.append(f"- sha256: `{dataset.sha256}`")
    lines.extend(
        [
            "",
            "## Impossible zeros (zero-as-missing)",
            "",
            "A resting blood pressure or serum cholesterol of 0 is not a real",
            "measurement; it is the source dataset's sentinel for \"not recorded\".",
            "",
            "| Column | Zero count | Share of rows | Of which positive class |",
            "|--------|-----------:|--------------:|------------------------:|",
        ]
    )
    for column, zero in audit.zero_counts.items():
        positive = zero.by_target.get(1, 0)
        lines.append(
            f"| `{column}` | {zero.count} | {_fmt_share(zero.share)} | {positive} |"
        )

    if affected_rows is not None:
        lines.extend(
            [
                "",
                f"Rows carrying at least one impossible zero: **{affected_rows}** "
                f"({_fmt_share(affected_rows / audit.row_count if audit.row_count else 0)}).",
                "That is the cost of a naive row-drop policy.",
            ]
        )

    lines.extend(
        [
            "",
            "## Applied policy",
            "",
            f"- **action:** `{policy.action}`",
            f"- **columns:** {', '.join(f'`{c}`' for c in policy.columns)}",
            f"- **sentinel treated as missing:** `{policy.sentinel}`",
            "- **fit scope:** training rows only (the statistic is learned per",
            "  split, never from the full dataset before splitting).",
            "",
            "**Rationale (chosen with the counts above in hand):** "
            + (policy.rationale or "(none recorded)"),
            "",
            "The policy is explicit and logged: applying it emits one `INFO` log",
            "line per column with the number of cells replaced. No zero survives",
            "as a real measurement.",
            "",
            "## Class balance (HeartDisease)",
            "",
            f"- total rows: {balance.total}",
            f"- `0` (no disease): **{balance.negative_count}** "
            f"({_fmt_share(balance.negative_count / balance.total if balance.total else 0)})",
            f"- `1` (disease): **{balance.positive_count}** "
            f"({_fmt_share(balance.prevalence)})",
            f"- positive prevalence: **{balance.prevalence:.4f}**",
            f"- majority:minority imbalance ratio: **{balance.imbalance_ratio:.4f}**",
            "",
            "The target is mildly imbalanced (roughly 1.24:1); downstream slices",
            "should score with stratified cross-validation and report per-class",
            "metrics rather than raw accuracy.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python -m heart.data.quality            # print audit, write report",
            "pytest tests/test_quality.py -v         # policy + leakage-safety tests",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def build_quality_audit(
    *, path: str | Path | None = None, allow_download: bool = True
) -> tuple[QualityAudit, pd.DataFrame, LoadedDataset]:
    """Load the pinned dataset and audit it under the declared policy."""
    dataset = load_dataset(path=path, allow_download=allow_download)
    frame = dataset.frame
    audit = audit_quality(frame)
    return audit, frame, dataset


def write_quality_report(
    audit: QualityAudit,
    *,
    affected_rows: int | None = None,
    dataset: LoadedDataset | None = None,
    path: str | Path | None = None,
) -> Path:
    """Write the rendered quality report and return its path."""
    target = Path(path) if path is not None else Path(REPORTS_DIR) / QUALITY_REPORT_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        render_quality_report(audit, affected_rows=affected_rows, dataset=dataset),
        encoding="utf-8",
    )
    return target


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m heart.data.quality",
        description="Audit impossible zeros and class balance, then write the report.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help=f"Report output path (default: reports/{QUALITY_REPORT_NAME}).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the audit without writing the report artifact.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ensure_project_dirs()

    audit, frame, dataset = build_quality_audit()
    affected = count_affected_rows(frame)

    for column, zero in audit.zero_counts.items():
        print(
            f"{column}: {zero.count} impossible zero(s) "
            f"({_fmt_share(zero.share)}); by target {zero.by_target}"
        )
    balance = audit.class_balance
    print(
        f"class balance: 0={balance.negative_count} 1={balance.positive_count} "
        f"(prevalence {balance.prevalence:.4f}, imbalance {balance.imbalance_ratio:.4f})"
    )
    print(f"policy: {audit.policy.describe()}")

    if not args.no_write:
        written = write_quality_report(
            audit, affected_rows=affected, dataset=dataset, path=args.report
        )
        print(f"wrote report: {written}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    sys.exit(main())
