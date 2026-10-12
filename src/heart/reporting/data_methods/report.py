"""Provenance-locked report assembly for ``reports/data-methods.md`` (M005/S01/T02).

Why this module exists
----------------------
Decision D025 and the M005 vision forbid any hand-assembled numbers in
``reports/data-methods.md`` part A. This module is the only sanctioned way a
report comes into existence: :func:`generate_report` runs the four-stage
pre-split EDA engine (:func:`heart.reporting.data_methods.run_eda`) over the
raw dataset, writes every stage result as a JSON artifact and every figure as
a PNG under ``reports/data-methods/``, locks all artifacts with the SHA-256
manifest from :mod:`heart.reporting.data_methods.provenance`, and only then
renders the markdown — with every table and figure citing the artifact it came
from and the manifest index embedded at the bottom.

Rendering is deliberately deterministic: fixed section order, sorted iteration
over stage results, and fixed-precision float formatting (the engine's
``_jsonable`` already rounds floats to six decimals). Two runs over the same
inputs produce byte-identical report and manifest bytes, which is exactly what
makes "a hand-edit to any number fails the provenance test" a runnable
negative-path test (M005/S01/T03) instead of an aspiration.

Atomicity
---------
The report and manifest are written with a write-tmp-then-:func:`os.replace`
pattern (established by :mod:`heart.data.load` for dataset caching), so a
failed render leaves no half-written report behind.

Observability
-------------
:func:`generate_report` logs one ``INFO`` line per phase (artifacts, manifest,
report) in the style of :mod:`heart.data.quality` and
:mod:`heart.reporting.leaderboard`. Failures surface as named exceptions from
the :class:`DataMethodsError` family; the CLI in
:mod:`heart.reporting.data_methods.__main__` turns those into a non-zero exit
without ever having written a partial report.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from heart.config import PROJECT_ROOT, REPORTS_DIR
from heart.data.load import PINNED_SHA256
from heart.data.schema import TARGET_COLUMN
from heart.data.split import SPLIT_VERSION, SplitError, load_split_artifacts
from heart.reporting.data_methods.audit import (
    AUDIT_FILENAME,
    PLAN_FILENAME,
    build_declared_plan,
    build_treatment_audit,
    write_treatment_artifacts,
)
from heart.reporting.data_methods.eda import (
    EMPIRICAL_LOGIT_BINS,
    HIGH_CORRELATION_THRESHOLD,
    HIST_BINS,
    IQR_FENCE_MULTIPLIER,
    LOGIT_SMOOTHING,
    RARE_CATEGORY_FLOOR,
    DataMethodsError,
    run_eda,
)
from heart.reporting.data_methods.narrative import (
    render_part_b,
    render_part_c,
    render_part_d,
    render_part_e,
    render_part_f,
    selection_carries_calibration_flag,
)
from heart.reporting.data_methods.provenance import (
    MANIFEST_NAME,
    manifest_markdown,
    write_manifest,
)

logger = logging.getLogger(__name__)

#: Default artifact directory (stage JSON tables, PNG figures, manifest.json).
DATAMETHODS_DIR: Path = REPORTS_DIR / "data-methods"

#: Default rendered report path (part A of the data-methods report).
REPORT_PATH: Path = REPORTS_DIR / "data-methods.md"

#: Stage key in the run_eda result -> JSON artifact filename under output_dir.
STAGE_ARTIFACTS: dict[str, str] = {
    "meta": "meta.json",
    "stage1_hygiene": "stage1_hygiene.json",
    "stage2_univariate": "stage2_univariate.json",
    "stage3_bivariate": "stage3_bivariate.json",
    "stage4_assumptions": "stage4_assumptions.json",
}

#: The two treatment artifacts (M006/S01) join the stage artifacts under the
#: same directory; ``write_manifest`` scans the directory, so they join the
#: provenance manifest automatically. ``treatment_audit.json`` is written only
#: when the committed split is present (part B degrades to the benign n/a
#: notice otherwise); ``treatment_plan.json`` is always written.
TREATMENT_ARTIFACT_NAMES: tuple[str, str] = (AUDIT_FILENAME, PLAN_FILENAME)

#: Significance floor quoted in the chi-square table and takeaways.
CHI_SQUARE_ALPHA: float = 0.05

#: Basename of the MLflow-derived gate-verdicts artifact (canonical constant:
#: ``heart.reporting.data_methods.gate_verdicts.GATE_VERDICTS_ARTIFACT``,
#: repeated here as a literal so the report generator never imports the
#: gate-verdicts store machinery at render time).
GATE_VERDICTS_FILENAME: str = "treatment_gate_verdicts.json"


def load_gate_verdicts(artifact_dir: str | Path) -> dict[str, Any] | None:
    """Load the MLflow-derived gate-verdicts artifact when it exists.

    Pure read of ``<artifact_dir>/treatment_gate_verdicts.json`` (T04):
    returns the parsed dict when the file exists, else None so the
    pre-declared Part D/E wording stands. Invalid JSON fails loudly as a
    named :class:`DataMethodsError` (never a silent empty disposition).
    """
    path = Path(artifact_dir) / GATE_VERDICTS_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataMethodsError(
            f"gate verdicts artifact is not valid JSON: {path}: {exc}."
        ) from exc
    if not isinstance(payload, dict):
        raise DataMethodsError(
            f"gate verdicts artifact must be a mapping, got "
            f"{type(payload).__name__}: {path}."
        )
    return payload


# ---------------------------------------------------------------------------
# Deterministic rendering helpers
# ---------------------------------------------------------------------------


def _fmt(value: Any, *, nd: int = 4, pct: bool = False) -> str:
    """Format a numeric value deterministically; ``None`` renders as ``n/a``.

    With ``pct=True`` the input is a fractional share in ``[0, 1]`` and is
    scaled by 100 so a share of ``0.5`` renders as ``50.0000%`` rather than
    the misleading ``0.5000%``.
    """
    if value is None:
        return "n/a"
    scaled = float(value) * (100.0 if pct else 1.0)
    rendered = f"{scaled:.{nd}f}"
    return f"{rendered}%" if pct else rendered


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a fixed markdown table from pre-formatted cells."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _cite(filename: str) -> str:
    """Citation line tying a rendered table/figure to its generated artifact."""
    return f"*Artifact:* `{filename}`"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` via write-tmp then :func:`os.replace` (no partial file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _display_path(path: str | Path) -> str:
    """Render ``path`` relative to the repo root when it lives underneath it."""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(candidate)


# ---------------------------------------------------------------------------
# Treatment plan / audit emission (M006/S01)
# ---------------------------------------------------------------------------


def load_train_rows(version: str | None = None) -> pd.DataFrame | None:
    """Load the committed split's training rows, or ``None`` if it is absent.

    This is the seam between the report generator and the committed split
    artifacts (``data/processed/splits/<version>/train.csv``), re-verified
    against the split manifest by :func:`heart.data.split.load_split_artifacts`.
    A missing split is *not* an error for the report: parts D and F render
    from the declared plan regardless, and part B degrades to the benign
    n/a notice naming the populate command. Only split-level failures
    (:class:`heart.data.split.SplitError` family) degrade; anything else
    (e.g. an unreadable verified CSV) propagates.
    """
    version = SPLIT_VERSION if version is None else version
    try:
        artifacts = load_split_artifacts(version)
    except SplitError as exc:
        logger.warning("report: no committed split v%r for the audit: %s", version, exc)
        return None
    return pd.read_csv(artifacts.train_path)


def write_treatment_plan_artifact(plan_payload: dict[str, Any], output_dir: Path) -> str:
    """Serialize-then-atomically write ``treatment_plan.json`` (plan-only path).

    Used when the committed split is absent, so ``treatment_audit.json`` is
    not written: the plan mirrors the declarations module (D032) and needs no
    frame. Mirrors ``write_treatment_artifacts``' atomic contract.
    """
    try:
        text = (
            json.dumps(plan_payload, indent=2, sort_keys=True, ensure_ascii=False)
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise DataMethodsError(
            f"treatment_plan payload is not JSON-serializable: {exc}"
        ) from exc
    _atomic_write_text(output_dir / PLAN_FILENAME, text)
    return PLAN_FILENAME


# ---------------------------------------------------------------------------
# Stage artifact emission
# ---------------------------------------------------------------------------


def write_stage_artifacts(eda_result: dict[str, Any], output_dir: Path) -> list[str]:
    """Write each stage result as a deterministic JSON artifact under output_dir.

    Returns the artifact basenames written (the report cites these paths).
    """
    written: list[str] = []
    for stage_key, filename in STAGE_ARTIFACTS.items():
        payload = eda_result[stage_key]
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        target = output_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
        written.append(filename)
    return written


# ---------------------------------------------------------------------------
# Stage section renderers (fixed order, every table cites its artifact)
# ---------------------------------------------------------------------------


def _render_stage1(eda: dict[str, Any]) -> list[str]:
    stage = eda["stage1_hygiene"]
    missing = stage["missingness"]
    sentinels = missing["systematic_sentinel"]
    duplicates = stage["exact_duplicates"]
    columns = duplicates["columns"]
    key_display = (
        "`" + ", ".join(columns[:4]) + f"` (+{len(columns) - 4} more)"
        if len(columns) > 4
        else "`" + ", ".join(columns) + "`"
    )

    lines = [
        "## Stage 1 — Hygiene",
        "",
        "The frame shape is checked, then missingness is split into *random* "
        "(true NaN cells) and *systematic* (declared sentinel zeros in the "
        "zero-as-missing columns), exact duplicate rows are grouped, and "
        "impossible values are audited under the declared policy.",
        "",
        "### Missingness division (random NaN vs declared sentinel zeros)",
        "",
        _md_table(
            ["column", "random NaN cells", "declared sentinel-zero cells"],
            [
                [f"`{column}`", str(missing["random_nan"].get(column, 0)), str(count)]
                for column, count in sorted(sentinels.items())
            ],
        ),
        "",
        (
            f"Total: {missing['total_random_nan_cells']} random-NaN cell(s) and "
            f"{missing['total_sentinel_zero_cells']} declared sentinel-zero cell(s); "
            f"{missing['rows_with_any_missingness']} row(s) carry any missingness "
            f"across a {stage['shape']['rows']}-row frame."
        ),
        "",
        _cite("stage1_hygiene.json"),
        "",
        "### Exact duplicate groups",
        "",
        _md_table(
            ["duplicate rows", "duplicate groups", "grouped columns"],
            [[str(duplicates["duplicate_rows"]), str(duplicates["duplicate_groups"]), key_display]],
        ),
        "",
        _cite("stage1_hygiene.json"),
        "",
        "### Impossible values under the declared policy",
        "",
        _md_table(
            ["column", "impossible zeros", "row share"],
            [
                [f"`{info['column']}`", str(info["count"]), _fmt(info["share"], pct=True)]
                for info in sorted(
                    stage["impossible_values"]["zero_counts"].values(),
                    key=lambda info: info["column"],
                )
            ],
        ),
        "",
        _cite("stage1_hygiene.json"),
    ]
    if stage["figures"]:
        lines += [
            "",
            f"*Figure:* `{stage['figures'][0]}` — missingness patterns, NaN cells vs declared zeros.",
        ]
    return lines


def _render_stage2(eda: dict[str, Any]) -> list[str]:
    stage = eda["stage2_univariate"]
    numeric = stage["numeric"]
    categorical = stage["categorical"]
    target = stage["target"]
    total_rows = eda["meta"]["row_count"]

    numeric_rows = [
        [
            f"`{column}`",
            _fmt(numeric[column]["skew"]),
            _fmt(numeric[column]["kurtosis"]),
            str(numeric[column]["box_outliers"]["count"]),
            _fmt(numeric[column]["box_outliers"]["share"], pct=True),
        ]
        for column in sorted(numeric)
    ]

    rare_rows = [
        [f"`{column}`", level, str(categorical[column]["frequencies"][level])]
        for column in sorted(categorical)
        for level in sorted(categorical[column]["rare_levels"])
    ]

    target_rows = [
        ["`" + label + "`", str(count), _fmt(count / total_rows, pct=True)]
        for label, count in sorted(target["counts"].items())
    ]

    lines = [
        "## Stage 2 — Univariate distributions",
        "",
        "### Numeric features: skew, kurtosis, IQR box outliers",
        "",
        _md_table(
            ["column", "skew", "kurtosis", "IQR outliers", "outlier share"],
            numeric_rows,
        ),
        "",
        _cite("stage2_univariate.json"),
        "",
        (
            "### Rare categorical levels (below the "
            f"{RARE_CATEGORY_FLOOR:.0%} prevalence floor)"
            if rare_rows
            else "### Rare categorical levels"
        ),
        "",
        (
            _md_table(["column", "rare level", "count"], rare_rows)
            if rare_rows
            else "No categorical level falls below the rare-level prevalence floor."
        ),
        "",
        _cite("stage2_univariate.json"),
        "",
        "### Target class distribution",
        "",
        _md_table(["class", "count", "share"], target_rows),
        "",
        (
            f"Prevalence {_fmt(target['prevalence'])} — imbalance ratio "
            f"{_fmt(target['imbalance_ratio'])}."
        ),
        "",
        _cite("stage2_univariate.json"),
    ]
    figure_lines = [f"*Figure:* `{figure}`" for figure in stage["figures"]]
    if figure_lines:
        lines += ["", "*Figures:*", ""] + ["- " + line for line in figure_lines]
    return lines


def _render_stage3(eda: dict[str, Any]) -> list[str]:
    stage = eda["stage3_bivariate"]
    contingency = stage["contingency"]
    high_correlations = stage["correlation"]["high_correlations"]
    vif = stage["vif"]

    chi_rows = [
        [
            f"`{column}`",
            _fmt(contingency[column]["chi_square"]["statistic"]),
            str(contingency[column]["chi_square"]["dof"])
            if contingency[column]["chi_square"]["dof"] is not None
            else "n/a",
            _fmt(contingency[column]["chi_square"]["p_value"], nd=6),
            "yes"
            if (contingency[column]["chi_square"]["p_value"] or 1.0) < CHI_SQUARE_ALPHA
            and contingency[column]["chi_square"]["p_value"] is not None
            else "no",
        ]
        for column in sorted(contingency)
    ]

    corr_rows = [
        [f"`{pair['a']}`", f"`{pair['b']}`", _fmt(pair["abs_r"])]
        for pair in high_correlations
    ]

    vif_rows = [
        [
            f"`{column}`",
            _fmt(value),
            "yes" if value >= HIGH_CORRELATION_THRESHOLD else "no",
        ]
        for column, value in sorted(vif.items(), key=lambda item: -item[1])
    ]

    max_vif_column = max(vif, key=lambda column: vif[column]) if vif else None
    significant = sum(1 for row in chi_rows if row[-1] == "yes")

    lines = [
        "## Stage 3 — Bivariate and multivariate structure",
        "",
        "### Categorical features vs target: chi-square",
        "",
        _md_table(
            ["column", "chi-square", "dof", "p value", f"p < {CHI_SQUARE_ALPHA:.2f}"],
            chi_rows,
        ),
        "",
        (
            f"{significant} of {len(chi_rows)} categorical features pass the "
            f"chi-square test at alpha {CHI_SQUARE_ALPHA:.2f}."
        ),
        "",
        _cite("stage3_bivariate.json"),
        "",
        "### Pairs with |r| above the collinearity threshold",
        "",
        (
            _md_table(["feature a", "feature b", "|r|"], corr_rows)
            if corr_rows
            else f"No numeric pair exceeds |r| > {HIGH_CORRELATION_THRESHOLD}."
        ),
        "",
        _cite("stage3_bivariate.json"),
        "",
        "### Variance inflation factors (numeric features)",
        "",
        _md_table(
            ["column", "VIF", f"flagged (VIF >= {HIGH_CORRELATION_THRESHOLD})"],
            vif_rows,
        ),
        "",
        (
            f"Highest VIF {_fmt(vif.get(max_vif_column))} on `{max_vif_column}`."
            if max_vif_column is not None
            else "No VIF columns."
        ),
        "",
        _cite("stage3_bivariate.json"),
    ]
    figure_lines = [f"*Figure:* `{figure}`" for figure in stage["figures"]]
    if figure_lines:
        lines += ["", "*Figures:*", ""] + ["- " + line for line in figure_lines]
    return lines


def _render_stage4(eda: dict[str, Any]) -> list[str]:
    stage = eda["stage4_assumptions"]
    impact = stage["outlier_impact"]
    logit = stage["empirical_logit"]

    impact_rows = [
        [
            f"`{row['column']}`",
            row["policy"],
            str(row["n"]),
            _fmt(row["mean"]),
            _fmt(row["delta_mean"]),
            _fmt(row["std"]),
            _fmt(row["delta_std"]),
        ]
        for row in impact["table"]
    ]

    def _logit_range(entry: dict[str, Any]) -> str:
        logits = [bin_["logit"] for bin_ in entry["bins"] if bin_["logit"] is not None]
        return _fmt(min(logits)) if logits else "n/a"

    logit_rows = [
        [
            f"`{column}`",
            str(len(entry["bins"])),
            _logit_range(entry),
        ]
        for column, entry in sorted(logit.items())
    ]

    bin_rows = [
        [
            f"`{column}`",
            str(max((bin_["n"] for bin_ in entry["bins"]), default=0)),
            str(max((bin_["positives"] for bin_ in entry["bins"]), default=0)),
        ]
        for column, entry in sorted(logit.items())
    ]

    declared_shifts = [
        abs(row["delta_mean"] or 0.0)
        for row in impact["table"]
        if row["policy"] == "median_impute"
    ]
    biggest_declared_shift = max(declared_shifts, default=0.0)

    lines = [
        "## Stage 4 — Assumption checks",
        "",
        "### Outlier handling: raw vs trim vs Winsorize vs the declared median-impute policy",
        "",
        "Every numeric column is measured on the raw frame under all four "
        "policies; Δ columns are relative to the raw baseline, so the declared "
        "policy's shift can be compared against the discard-based baselines.",
        "",
        _md_table(
            ["column", "policy", "n", "mean", "Δ mean", "std", "Δ std"],
            impact_rows,
        ),
        "",
        (
            f"Largest absolute mean shift under the declared median-impute policy: "
            f"{biggest_declared_shift:.4f}."
        ),
        "",
        _cite("stage4_assumptions.json"),
        "",
        "### Empirical logit summaries (decile-binned numeric features)",
        "",
        _md_table(["column", "bins", "min logit"], logit_rows),
        "",
        _md_table(["column", "max bin n", "max bin positives"], bin_rows),
        "",
        _cite("stage4_assumptions.json"),
    ]
    figure_lines = [f"*Figure:* `{figure}`" for figure in stage["figures"]]
    if figure_lines:
        lines += ["", "*Figures:*", ""] + ["- " + line for line in figure_lines]
    return lines


# ---------------------------------------------------------------------------
# Takeaways checklist (generated purely from stage findings)
# ---------------------------------------------------------------------------


def _takeaways(eda: dict[str, Any]) -> list[str]:
    stage1 = eda["stage1_hygiene"]
    stage2 = eda["stage2_univariate"]
    stage3 = eda["stage3_bivariate"]
    stage4 = eda["stage4_assumptions"]
    missing = stage1["missingness"]
    duplicates = stage1["exact_duplicates"]
    target = stage2["target"]
    contingency = stage3["contingency"]
    high_correlations = stage3["correlation"]["high_correlations"]
    vif = stage3["vif"]

    sentinel_columns = ", ".join(sorted(missing["systematic_sentinel"])) or "n/a"
    significant = sorted(
        column
        for column, entry in contingency.items()
        if entry["chi_square"]["p_value"] is not None
        and entry["chi_square"]["p_value"] < CHI_SQUARE_ALPHA
    )
    max_vif_column = max(vif, key=lambda column: vif[column]) if vif else "n/a"
    max_vif_value = vif.get(max_vif_column) if vif else None
    rare_pairs = sorted(
        (column, level)
        for column, entry in stage2["categorical"].items()
        for level in entry["rare_levels"]
    )
    declared_shifts = [
        abs(row["delta_mean"] or 0.0)
        for row in stage4["outlier_impact"]["table"]
        if row["policy"] == "median_impute"
    ]
    biggest_declared_shift = max(declared_shifts, default=0.0)

    return [
        (
            f"- [ ] **Hygiene:** the raw frame is {stage1['shape']['rows']} rows × "
            f"{stage1['shape']['columns']} columns; random-NaN cells "
            f"{missing['total_random_nan_cells']}, declared sentinel-zero cells "
            f"{missing['total_sentinel_zero_cells']} "
            f"({sentinel_columns}); exact duplicate rows "
            f"{duplicates['duplicate_rows']} (artifact: `stage1_hygiene.json`)."
        ),
        (
            f"- [ ] **Policy:** sentinel zeros in {sentinel_columns} are treated as "
            "*systematic* missingness and median-imputed per the declared "
            "zero-as-missing policy; NaN cells would be *random* missingness "
            "(artifact: `stage1_hygiene.json`)."
        ),
        (
            f"- [ ] **Univariate:** target prevalence {_fmt(target['prevalence'])} "
            f"(imbalance ratio {_fmt(target['imbalance_ratio'])}); rare levels "
            + (
                ", ".join(f"`{column}:{level}`" for column, level in rare_pairs)
                if rare_pairs
                else "none"
            )
            + f" at the {RARE_CATEGORY_FLOOR:.0%} floor "
            "(artifact: `stage2_univariate.json`)."
        ),
        (
            f"- [ ] **Bivariate:** chi-square-significant categorical features "
            + (", ".join(f"`{column}`" for column in significant) if significant else "none")
            + f" (p < {CHI_SQUARE_ALPHA:.2f}); {len(high_correlations)} pair(s) with "
            f"|r| > {HIGH_CORRELATION_THRESHOLD}; highest VIF {_fmt(max_vif_value)} "
            f"on `{max_vif_column}` (artifact: `stage3_bivariate.json`)."
        ),
        (
            f"- [ ] **Assumptions:** empirical logit over {EMPIRICAL_LOGIT_BINS} "
            f"decile bins per numeric feature (add-{LOGIT_SMOOTHING} smoothing); "
            "the declared median-impute policy's largest absolute mean shift is "
            f"{biggest_declared_shift:.4f}, benchmarked against trim and "
            "Winsorize (artifact: `stage4_assumptions.json`)."
        ),
    ]


# ---------------------------------------------------------------------------
# Full report rendering
# ---------------------------------------------------------------------------


def render_report(
    eda: dict[str, Any],
    *,
    dataset_sha256: str | None,
    dataset_source: str | None,
    output_dir: Path,
    manifest: dict[str, Any],
    audit: dict[str, Any] | None,
    declared_plan: dict[str, Any],
    verdicts: dict[str, Any] | None = None,
) -> str:
    """Render the data-methods report as deterministic markdown.

    Part A (the four-stage pre-split EDA, plus its takeaways checklist) is
    rendered exactly as before; parts B-F are appended after it and before
    the Technical Appendix, in the fixed order B, C, D, E, F. When the
    MLflow-derived gate-verdicts artifact is supplied, parts D/E gain their
    final gate-measured dispositions; when absent the pre-declared wording
    stands (T04).
    """
    meta = eda["meta"]
    if dataset_sha256 is None:
        digest_line = "n/a (fixture run — no committed dataset digest to verify)"
    elif dataset_sha256 == PINNED_SHA256:
        digest_line = f"`{dataset_sha256}` matches the pinned digest (verified)"
    else:
        digest_line = (
            f"`{dataset_sha256}` does NOT match the pinned `{PINNED_SHA256}`"
        )
    return "\n".join(
        [
            "# Data Methods — Parts A-F",
            "",
            (
                "This report is rendered by `heart.reporting.data_methods`. "
                "Part A is the four-stage pre-split EDA over the raw dataset — "
                "every stage runs entirely before any train/test split. Parts "
                "B-F document the per-model treatment audit, the logistic-"
                "regression preparation methodology, the pre-declared class-"
                "imbalance ladder, the validation/calibration strategy, and "
                "the pre-declared severity gate. Regenerate deterministically "
                "with `PYTHONPATH=src python -m heart.reporting.data_methods`."
            ),
            "",
            "## Dataset provenance",
            "",
            (
                f"- source: `{_display_path(dataset_source)}`"
                if dataset_source is not None
                else "- source: n/a (fixture run — no committed dataset path)"
            ),
            f"- SHA-256: {digest_line}",
            f"- shape: {meta['row_count']} rows × {meta['column_count']} columns",
            f"- target: `{meta['target_column']}`",
            f"- random seed: {meta['seed']}",
            f"- declared zero-as-missing columns: "
            f"`{', '.join(meta['declared_zero_as_missing_columns'])}`",
            f"- artifact directory: `{_display_path(output_dir)}`",
            "",
            *_render_stage1(eda),
            "",
            *_render_stage2(eda),
            "",
            *_render_stage3(eda),
            "",
            *_render_stage4(eda),
            "",
            "## Takeaways checklist",
            "",
            *_takeaways(eda),
            "",
            *render_part_b(audit),
            *render_part_c(eda, audit),
            *render_part_d(declared_plan, verdicts),
            *render_part_e(
                selection_carries_flag=selection_carries_calibration_flag(),
                verdicts=verdicts,
            ),
            *render_part_f(declared_plan),
            "## Technical appendix",
            "",
            "### Parameters",
            "",
            f"- random seed: {meta['seed']}",
            f"- rare-category floor: {RARE_CATEGORY_FLOOR}",
            f"- high-correlation threshold: |r| > {HIGH_CORRELATION_THRESHOLD}",
            f"- histogram bins: {HIST_BINS}",
            f"- empirical logit bins: {EMPIRICAL_LOGIT_BINS} "
            f"(add-{LOGIT_SMOOTHING} smoothing)",
            f"- IQR fence multiplier: {IQR_FENCE_MULTIPLIER}",
            f"- outlier fence definition: "
            f"{eda['stage4_assumptions']['outlier_impact']['fence_definition']}",
            f"- chi-square alpha: {CHI_SQUARE_ALPHA}",
            "",
            "### Regeneration command",
            "",
            "```",
            "PYTHONPATH=src python -m heart.reporting.data_methods",
            "```",
            "",
            "### Artifact index (provenance manifest)",
            "",
            "All tables and figures above are traceable to these generated "
            "artifacts; digests are recomputed by the provenance tests.",
            "",
            _cite(MANIFEST_NAME),
            "",
            manifest_markdown(manifest),
            "",
        ]
    )


# ---------------------------------------------------------------------------
# Public seam
# ---------------------------------------------------------------------------


def generate_report(
    dataset: Any,
    *,
    output_dir: str | Path | None = None,
    report_path: str | Path | None = None,
    target: str = TARGET_COLUMN,
    train_frame: pd.DataFrame | None = None,
) -> Path:
    """Run the four-stage EDA, lock artifacts, and render the full report.

    Parameters
    ----------
    dataset:
        The **raw** dataset to analyse — a :class:`heart.data.load.LoadedDataset`
        (preferred; contributes its verified SHA-256 digest) or a bare raw
        pandas DataFrame (fixture runs carry no dataset digest, and the report
        says so explicitly rather than inventing one).
    output_dir:
        Directory to write stage JSON tables and PNG figures into
        (default ``reports/data-methods``).
    report_path:
        Path of the rendered markdown report
        (default ``reports/data-methods.md``).
    target:
        Target column name (default :data:`heart.data.schema.TARGET_COLUMN`).
    train_frame:
        Optional override for the training rows the treatment audit fits
        over (M006/S01). When ``None`` the committed split's train rows are
        loaded through :func:`load_train_rows`; if the committed split is
        absent, part B renders the benign n/a notice naming the populate
        command and ``treatment_audit.json`` is not written, while parts
        D/F still render from the declared plan.

    Returns
    -------
    Path
        The rendered report path.

    Raises
    ------
    DataMethodsError
        For a non-frame dataset or any engine-stage failure; the report file is
        never partially written (atomic replace), so callers may retry safely.
    TreatmentAuditError
        When ``train_frame`` is given but unusable (non-frame argument, or a
        frame that fails the preprocessing fit). Nothing has been written at
        that point — the audit is built before any artifact lands on disk.
    """
    artifact_dir = Path(output_dir) if output_dir is not None else DATAMETHODS_DIR
    report_target = Path(report_path) if report_path is not None else REPORT_PATH
    frame = getattr(dataset, "frame", dataset)
    dataset_sha256 = getattr(dataset, "sha256", None)
    dataset_source = getattr(dataset, "source_path", None)
    if not hasattr(frame, "columns"):
        raise DataMethodsError(
            "generate_report needs a raw pandas DataFrame or a LoadedDataset; "
            f"got {type(dataset).__name__}"
        )

    # Treatment plan + audit are built *before* anything is written, so an
    # unusable training frame fails with zero bytes on disk.
    declared_plan = build_declared_plan()
    rows = train_frame if train_frame is not None else load_train_rows()
    audit = None if rows is None else build_treatment_audit(rows)
    if audit is not None:
        logger.info(
            "report: treatment audit built over %d training row(s)",
            audit["preprocessing"]["fitted"]["row_count"],
        )
    else:
        logger.info(
            "report: no committed split; rendering part B as n/a "
            "(populate with `python -m heart.data.split`)"
        )

    logger.info(
        "report: writing pre-split EDA artifacts for %d row(s) into %s",
        len(frame),
        artifact_dir,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    eda = run_eda(frame, artifact_dir=artifact_dir, target=target)
    stage_files = write_stage_artifacts(eda, artifact_dir)
    if audit is not None:
        treatment_files = write_treatment_artifacts(audit, declared_plan, artifact_dir)
    else:
        treatment_files = [write_treatment_plan_artifact(declared_plan, artifact_dir)]
    manifest = write_manifest(artifact_dir)
    verdicts = load_gate_verdicts(artifact_dir)
    if verdicts is not None:
        logger.info(
            "report: gate verdicts loaded with %d outcome(s); "
            "rendering Parts D/E final dispositions",
            len(verdicts.get("outcomes", [])),
        )
    else:
        logger.info("report: no gate verdicts artifact; Parts D/E pre-declared")
    rendered = render_report(
        eda,
        dataset_sha256=dataset_sha256,
        dataset_source=str(dataset_source) if dataset_source is not None else None,
        output_dir=artifact_dir,
        manifest=manifest,
        audit=audit,
        declared_plan=declared_plan,
        verdicts=verdicts,
    )
    _atomic_write_text(report_target, rendered + "\n")
    logger.info(
        "report: wrote %s with %d stage artifact(s), %d treatment "
        "artifact(s) and %d manifested artifact(s)",
        report_target,
        len(stage_files),
        len(treatment_files),
        manifest["artifact_count"],
    )
    return report_target
