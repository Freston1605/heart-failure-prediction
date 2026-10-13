"""Prose-heavy renderers for ``reports/data-methods.md`` parts B-F (M006/S01/T03).

Why this module exists
----------------------
Decision D025 forbids hand-assembled numbers anywhere in the generated
report, and decisions D032/D033 fix *what* parts B-F must say and *from
where*. This module holds the five section renderers the report generator
(:func:`heart.reporting.data_methods.report.generate_report`) appends after
part A's stage sections and before the Technical Appendix, in fixed order:

- **Part B** — per-model treatment audit, framed as the Model-Specific
  Pipeline Pattern (D033), rendered strictly from ``treatment_audit.json``:
  a summary table over the whole battery, a *deep* sub-section for the linear
  family (D028) with the actual sklearn pipeline parameters and the fitted
  train-side statistics, and a *light* sub-section for the robust families
  each carrying its declared robustness justification.
- **Part C** — the logistic-regression preparation methodology as the fixed
  six-checklist narrative. The prose is module CONSTANTS (regenerated prose,
  constant wording); every *number* comes from an artifact citation of a
  part A stage (skew table ``stage2_univariate.json``, empirical-logit
  ``stage4_assumptions.json``) or from an exact code reference.
- **Part D** — the pre-declared class-imbalance ladder, rendered strictly
  from ``treatment_plan.json`` (the 1:1 mirror of
  :mod:`heart.eval.treatment_declared`, D032).
- **Part E** — the validation/calibration strategy narrative, including the
  recorded ``calibration_did_not_improve_brier`` honesty flag and a plain
  *pointer* citation of ``reports/selection.md`` that copies none of its
  numbers (D025).
- **Part F** — the pre-declared severity gate, rendered from the declared
  constants through the artifact with a defense-in-depth assertion: the
  rendered gate sentence must equal :func:`heart.eval.treatment_declared.
  gate_text` exactly, or a named :class:`DataMethodsError` is raised.

Determinism: fixed section order, sorted iteration, fixed-precision float
formatting via the report helpers — two runs over the same inputs produce
byte-identical markdown, which is what keeps the provenance lock runnable.
"""

from __future__ import annotations

from typing import Any

from heart.data.split import N_SPLITS, SPLIT_VERSION
from heart.eval.repeated_cv import DEFAULT_N_REPEATS
from heart.eval.treatment_declared import (
    GATE_METRIC,
    GATE_MIN_DELTA_ROC_AUC,
    GATE_WINNER_STANDING_RULE,
    PAIRED_SIGNIFICANCE_ALPHA,
    gate_text,
)
from heart.features.engineering import ENGINEERED_TRANSFORMS
from heart.reporting.data_methods.eda import DataMethodsError

__all__ = [
    "PART_C_CHECKLIST_HEADINGS",
    "CALIBRATION_FLAG_NAME",
    "SELECTION_REPORT_POINTER",
    "SPLIT_POPULATE_COMMAND",
    "SKEW_LISTING_COUNT",
    "GATE_VERDICTS_FILENAME",
    "render_part_b",
    "render_part_c",
    "render_part_d",
    "render_part_e",
    "render_part_f",
    "selection_carries_calibration_flag",
    "index_verdicts_by_variant",
    "format_gate_disposition",
    "summarize_gate_dispositions",
]

#: Basename of the MLflow-derived gate-verdicts artifact (canonical constant:
#: ``heart.reporting.data_methods.gate_verdicts.GATE_VERDICTS_ARTIFACT``,
#: repeated here as a literal so this module never imports the gate-verdicts
#: store machinery at render time).
#: (matches the actionable message of
#: :class:`heart.data.split.SplitNotFoundError`, never a stack trace).
SPLIT_POPULATE_COMMAND: str = "python -m heart.data.split"

#: Basename of the MLflow-derived gate-verdicts artifact (canonical constant:
#: ``heart.reporting.data_methods.gate_verdicts.GATE_VERDICTS_ARTIFACT``,
#: repeated here as a literal so this module never imports the gate-verdicts
#: store machinery at render time).
GATE_VERDICTS_FILENAME: str = "treatment_gate_verdicts.json"

#: Part C's six fixed checklist headings, in the fixed render order.
PART_C_CHECKLIST_HEADINGS: tuple[str, ...] = (
    "Categorical encoding: one-hot keep-all with handle_unknown=ignore, and "
    "the drop-first candidate",
    "Standardization under regularization",
    "Skew and polynomial handling reconciled with already-engineered features",
    "Imputation toolbox incl. missingness flags",
    "Class imbalance runs through the part D ladder",
    "Validation and calibration honesty",
)

#: The honesty flag the pipeline records when calibration fails to improve
#: the Brier score (emitted by ``heart.eval.selection``; part E states the
#: mechanism and points at the committed selection report textually).
CALIBRATION_FLAG_NAME: str = "calibration_did_not_improve_brier"

#: Textual pointer (never a number copy) to the committed selection report.
SELECTION_REPORT_POINTER: str = "reports/selection.md"

#: How many of the largest absolute skews part C item (3) renders from the
#: stage-2 artifact.
SKEW_LISTING_COUNT: int = 3


# ---------------------------------------------------------------------------
# Shared formatting helpers (mirrors report.py's deterministic style)
# ---------------------------------------------------------------------------


def _param_cell(value: Any) -> str:
    """Render one JSON-normalized parameter value for a markdown cell."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return f"'{value}'"
    return str(value)


def _params_line(params: dict[str, Any], *, skip: set[str] | None = None) -> str:
    """Render a parameter dict as ``key=value`` pairs in stored (sorted) order."""
    omitted = skip or set()
    pairs = [
        f"{key}={_param_cell(value)}"
        for key, value in params.items()
        if key not in omitted
    ]
    return ", ".join(pairs) if pairs else "none"


def _keys_cell(keys: list[str]) -> str:
    return ", ".join(f"`{key}`" for key in keys) if keys else "none (fixed)"


def _split_required_notice() -> str:
    """The benign not-a-number notice part B renders without the split."""
    return (
        f"n/a (audit requires the committed split — run `{SPLIT_POPULATE_COMMAND}` "
        f"to populate `data/processed/splits/{SPLIT_VERSION}/`)"
    )


# ---------------------------------------------------------------------------
# Part B — Per-Model Treatment Audit
# ---------------------------------------------------------------------------


def render_part_b(audit: dict[str, Any] | None) -> list[str]:
    """Render part B from ``treatment_audit.json`` (or the benign n/a notice).

    The missing-split branch must never render a stack trace: it states that
    the audit requires the committed split, names the exact populate command,
    and notes that parts D and F still render because they do not depend on
    the split.
    """
    lines = ["## Part B — Per-Model Treatment Audit", ""]
    if audit is None:
        lines += [
            _split_required_notice(),
            "",
            (
                "Parts D and F below still render in this run: they come from "
                "the declared plan, which does not depend on the split."
            ),
            "",
        ]
        return lines

    pattern = audit["pattern"]
    split = audit["split"]
    loader = audit["loader"]
    preprocessing = audit["preprocessing"]
    fitted = preprocessing["fitted"]
    zero_policy = audit["zero_policy"]

    lines += [
        (
            f"The audit follows the **{pattern['name']}** "
            f"(decision {pattern['decision']}): {pattern['description']}. "
            f"Depth policy is decision {pattern['depth_policy_decision']}: the "
            "linear family is audited deep, the robust families light with "
            "written justifications."
        ),
        "",
        (
            f"Provenance: the loader pins SHA-256 `{loader['pinned_sha256']}`; "
            f"the committed split is version `{split['version']}` "
            f"({split['n_splits']} stratified folds, test fold "
            f"{split['test_fold']}); {split['group_straddle_policy']}."
        ),
        "",
        (
            "Preprocessing chain (fit on "
            f"{preprocessing['fit_on']}): declared zero policy "
            f"`{zero_policy['action']}` on "
            f"{', '.join(f'`{column}`' for column in zero_policy['columns'])} "
            f"(sentinel {zero_policy['sentinel']}), then a ColumnTransformer "
            "with StandardScaler on the numeric columns and OneHotEncoder "
            "with handle_unknown='ignore' on the categorical columns."
        ),
        "",
        "### Summary table (whole battery)",
        "",
        _battery_summary_table(audit),
        "",
        f"*Artifact:* `treatment_audit.json`",
        "",
    ]
    lines += _render_deep_section(audit)
    lines += _render_light_section(audit)
    return lines


def _battery_summary_table(audit: dict[str, Any]) -> str:
    """One row per battery member: model, family, chain, tuned keys, params."""
    preprocessing = audit["preprocessing"]
    chain = " -> ".join(str(step["step"]) for step in preprocessing["pipeline_steps"])
    rows = []
    for model in audit["models"]:
        rows.append(
            [
                f"`{model['model_type']}`",
                model["family"],
                f"`{chain}` -> `{model['model_type']}`",
                _keys_cell(model["tuned_space_keys"]),
                _params_line(model["fixed_params"]),
            ]
        )
    return _md_table(
        ["model", "family", "pipeline treatment chain", "tuned keys", "fixed params"],
        rows,
    )


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Local deterministic markdown table (same shape as report.py's)."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _render_deep_section(audit: dict[str, Any]) -> list[str]:
    """The deep block: shared chain parameters + fitted train-side statistics."""
    preprocessing = audit["preprocessing"]
    fitted = preprocessing["fitted"]
    zero_policy = audit["zero_policy"]

    lines = [
        "### Deep audit — the linear family (decision D028)",
        "",
        (
            "The linear-family members "
            + ", ".join(f"`{member}`" for member in audit["depth_tiers"]["deep"])
            + " are audited at deep depth: the shared preprocessing chain is "
            "rendered step by step from the actual sklearn objects "
            f"(`{preprocessing['module']}`), followed by the fitted "
            "train-side statistics."
        ),
        "",
        "#### Shared preprocessing chain (actual sklearn objects)",
        "",
        _pipeline_steps_table(preprocessing["pipeline_steps"]),
        "",
        "*Artifact:* `treatment_audit.json`",
        "",
        (
            "#### Fitted train-side statistics "
            f"(fit on the {fitted['row_count']} training rows only)"
        ),
        "",
        "Declared zero policy and fitted medians:",
        "",
        _md_table(
            ["column", "declared sentinel", "fitted median"],
            [
                [f"`{column}`", str(zero_policy["sentinel"]), str(value)]
                for column, value in fitted["zero_medians"].items()
            ],
        ),
        "",
        "*Artifact:* `treatment_audit.json`",
        "",
        "StandardScaler on the numeric columns (fitted mean/spread):",
        "",
        _md_table(
            ["numeric column", "fitted mean", "fitted scale"],
            [
                [f"`{column}`", str(mean), str(scale)]
                for column, mean, scale in zip(
                    preprocessing["numeric_columns"],
                    fitted["scaler_mean"],
                    fitted["scaler_scale"],
                    strict=True,
                )
            ],
        ),
        "",
        "*Artifact:* `treatment_audit.json`",
        "",
        "OneHotEncoder on the categorical columns (observed categories):",
        "",
        _md_table(
            ["categorical column", "observed categories"],
            [
                [f"`{column}`", ", ".join(f"`{level}`" for level in categories)]
                for column, categories in zip(
                    preprocessing["categorical_columns"],
                    fitted["encoder_categories"],
                    strict=True,
                )
            ],
        ),
        "",
        (
            f"Expanded-feature count after the chain: "
            f"**{fitted['expanded_feature_count']}** — the count the declared "
            "`one-hot-drop-first` candidate would shrink (part C)."
        ),
        "",
        "*Artifact:* `treatment_audit.json`",
        "",
    ]
    for model in audit["models"]:
        if model["depth"] != "deep":
            continue
        lines += [
            f"#### Deep member: {model['model_name']} ({model['model_type']})",
            "",
            f"- family: {model['family']}",
            f"- estimator: {model['estimator_description']}",
            f"- tuned space keys: {_keys_cell(model['tuned_space_keys'])}",
            f"- fixed params: {_params_line(model['fixed_params'])}",
            (
                "- expanded-feature count this member trains on: "
                f"{model['expanded_feature_count']}"
            ),
            "",
            "*Artifact:* `treatment_audit.json`",
            "",
        ]
    return lines


def _pipeline_steps_table(steps: list[dict[str, Any]]) -> str:
    """Render every pipeline step (and ColumnTransformer sub-transformer)."""
    rows: list[list[str]] = []
    for step in steps:
        ct_keys = {"transformers", "transformer_weights"}
        rows.append(
            [
                f"`{step['step']}`",
                step["transformer"],
                "n/a (sub-transformers below)"
                if "sub_transformers" in step
                else "n/a",
                _params_line(step["params"], skip=ct_keys),
            ]
        )
        for sub in step.get("sub_transformers", []):
            rows.append(
                [
                    f"`{step['step']}:{sub['name']}`",
                    sub["transformer"],
                    ", ".join(f"`{column}`" for column in sub["columns"]),
                    _params_line(sub["params"]),
                ]
            )
    return _md_table(["step", "transformer", "columns", "parameters"], rows)


def _render_light_section(audit: dict[str, Any]) -> list[str]:
    """The light block: one robust member per sub-section, justification first."""
    lines = [
        "### Light audit — robust families (decision D028)",
        "",
        (
            "The robust-family members "
            + ", ".join(f"`{member}`" for member in audit["depth_tiers"]["light"])
            + " are audited at light depth. Each carries a declared "
            "robustness justification — a written declaration of why the "
            "transform/imputation/encoding candidates do not bind for the "
            "family, not a measured claim; a battery change that adds an "
            "unjustified robust member fails the audit build loudly."
        ),
        "",
    ]
    for model in audit["models"]:
        if model["depth"] != "light":
            continue
        lines += [
            f"#### Light member: {model['model_name']} ({model['model_type']})",
            "",
            f"- family: {model['family']}",
            f"- estimator: {model['estimator_description']}",
            f"- tuned space keys: {_keys_cell(model['tuned_space_keys'])}",
            f"- fixed params: {_params_line(model['fixed_params'])}",
            "",
            f"Robustness justification: {model['robustness_justification']}",
            "",
            "*Artifact:* `treatment_audit.json`",
            "",
        ]
    return lines


# ---------------------------------------------------------------------------
# Part C — Logistic-Regression Preparation Methodology (six-checklist)
# ---------------------------------------------------------------------------


def render_part_c(eda: dict[str, Any], audit: dict[str, Any] | None) -> list[str]:
    """Render part C: the six fixed checklist headings with constant prose.

    Numbers come only from artifact citations of the part A stages (the
    stage-2 skew table and the stage-4 empirical-logit block); the fitted
    sklearn parameters behind items (1), (2) and (4) live in part B.
    """
    numeric = eda["stage2_univariate"]["numeric"]
    top_skews = sorted(
        ((column, stats["skew"]) for column, stats in numeric.items()),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:SKEW_LISTING_COUNT]
    empirical_logit = eda["stage4_assumptions"]["empirical_logit"]
    transform_names = ", ".join(f"`{name}`" for name in sorted(ENGINEERED_TRANSFORMS))
    part_b_pointer = (
        "recorded in part B (`treatment_audit.json`)"
        if audit is not None
        else "recorded in part B once the committed split is populated"
    )

    lines = [
        "## Part C — Logistic-Regression Preparation Methodology (six-checklist)",
        "",
        (
            "The logistic-regression family is prepared through a fixed "
            "six-point checklist. Every number in this part is cited from a "
            "generated part A artifact or from code constants — nothing is "
            "hand-assembled (D025)."
        ),
        "",
        f"### {PART_C_CHECKLIST_HEADINGS[0]}",
        "",
        (
            "The implemented encoding is one-hot *keep-all*: "
            "`heart.data.pipeline.build_preprocessing_pipeline` builds "
            '`OneHotEncoder(handle_unknown="ignore", sparse_output=False, '
            'dtype=float)` over the categorical columns, so an unseen level at '
            "predict time degrades to an all-zero row instead of crashing and "
            "no observed category is dropped. The classic dummy-trap argument "
            "(a redundant level collinear with the intercept) targets "
            "unpenalized OLS; the linear family here is regularized, so the "
            "redundant column's coefficient is shrunk by the same penalty as "
            "every other feature. The drop-first alternative is not "
            "discarded, though: it is pre-declared as the `one-hot-drop-first` "
            "candidate (part D's register) and will be measured, not assumed."
        ),
        "",
        f"### {PART_C_CHECKLIST_HEADINGS[1]}",
        "",
        (
            "Standardization sits *inside* the pipeline (the `features` step "
            "of the same module), so the regularization penalty applies "
            "equally to every feature regardless of raw scale, and the "
            "StandardScaler is fit on training rows only "
            "(`heart.data.pipeline.fit_preprocessing`); the fitted mean and "
            f"spread per numeric column are {part_b_pointer}, not re-derived "
            "here."
        ),
        "",
        f"### {PART_C_CHECKLIST_HEADINGS[2]}",
        "",
        (
            "Part of the skew/shape story is already answered by the "
            "engineered features from M001/S04 "
            "(`heart/features/engineering.py`): " + transform_names + ". The "
            "remaining raw skew is not hand-corrected either: the power-"
            "transform rungs are pre-declared as the "
            "`yeo-johnson-power-transform` and `box-cox-power-transform` "
            "candidates and measured through the part F gate. The stage-2 "
            "skew table records the raw skews; the largest absolute skews are:"
        ),
        "",
        _md_table(
            ["column", "skew (raw, stage 2)"],
            [[f"`{column}`", f"{skew:.4f}"] for column, skew in top_skews],
        ),
        "",
        "*Artifact:* `stage2_univariate.json`",
        "",
        (
            f"The empirical-logit decile check over {len(empirical_logit)} "
            "numeric feature(s) is the linearity-of-logit diagnostic behind "
            "the same transform candidates."
        ),
        "",
        "*Artifact:* `stage4_assumptions.json`",
        "",
        f"### {PART_C_CHECKLIST_HEADINGS[3]}",
        "",
        (
            "Imputation is the committed `ZeroMedianImputer` "
            "(`heart/data/quality.py`): the declared sentinel zeros become "
            "missing and are filled with the fit-frame median (the declared "
            "policy is recorded in part A stage 1; the fitted medians are in "
            f"part B). The declared toolbox next rungs are `knn-imputer` (a "
            "K-nearest-neighbours imputer fit on training rows only) and "
            "`missingness-flag-columns` (one binary missing-indicator column "
            "per feature with any observed missing value) — both pre-declared "
            "in part D's register, measured rather than assumed."
        ),
        "",
        f"### {PART_C_CHECKLIST_HEADINGS[4]}",
        "",
        (
            "Class imbalance is not handled here: every imbalance action runs "
            "through the ordered pre-declared ladder in part D."
        ),
        "",
        f"### {PART_C_CHECKLIST_HEADINGS[5]}",
        "",
        (
            "Validation and calibration honesty live in part E, including "
            f"the recorded `{CALIBRATION_FLAG_NAME}` honesty flag."
        ),
        "",
    ]
    return lines


def index_verdicts_by_variant(
    verdicts: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Group verdicts outcomes by declared variant key (deterministic order).

    Each group's outcomes are sorted by ``treated_key`` so the disposition
    lines render deterministically. A non-dict outcome entry fails loudly —
    the verdicts artifact is a store projection with a fixed shape, never
    free-form text.
    """
    by_variant: dict[str, list[dict[str, Any]]] = {}
    outcomes = verdicts.get("outcomes")
    if not isinstance(outcomes, list):
        raise DataMethodsError(
            "gate verdicts verdicts artifact carries no outcomes list; "
            "parts D/E dispositions need the measured outcomes."
        )
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            raise DataMethodsError(
                "gate verdicts outcome entry must be a mapping, got "
                f"{type(outcome).__name__}."
            )
        key = outcome.get("candidate_key")
        if not isinstance(key, str) or not key:
            raise DataMethodsError(
                "gate verdicts outcome entry lacks a candidate_key; the "
                "verdicts projection always records one."
            )
        by_variant.setdefault(key, []).append(outcome)
    for group in by_variant.values():
        group.sort(key=lambda entry: str(entry.get("treated_key", "")))
    return by_variant


def format_gate_disposition(outcome: dict[str, Any]) -> str:
    """Render one measured pair as a gate-measured disposition line.

    The line carries the delta, the significance reading at the declared
    alpha, the standing reading, and the verdict — the clause-exact triple
    the gate measured, read from the artifact, never re-derived here.
    """
    treated = outcome.get("treated_key")
    missing = [
        key
        for key in ("treated_key", "delta", "significant", "standing_ok", "verdict")
        if key not in outcome
    ]
    if missing or not isinstance(treated, str) or not treated:
        raise DataMethodsError(
            f"gate verdicts outcome entry is missing field(s) {missing}; "
            "the verdicts projection always records the full field vector."
        )
    try:
        delta = float(outcome["delta"])
    except (TypeError, ValueError) as exc:
        raise DataMethodsError(
            f"gate verdicts outcome {treated!r} delta "
            f"{outcome['delta']!r} is not a number."
        ) from exc
    significance = "significant" if outcome["significant"] else "not significant"
    standing = "ok" if outcome["standing_ok"] else "worsened"
    return (
        f"gate-measured: `{treated}` — delta {delta:+.4f} on {GATE_METRIC}, "
        f"significance {significance} at alpha={PAIRED_SIGNIFICANCE_ALPHA:g}, "
        f"standing {standing}, verdict {outcome['verdict']}."
    )


def summarize_gate_dispositions(verdicts: dict[str, Any]) -> dict[str, Any]:
    """Summarize the verdicts artifact into counts the part E block cites.

    Pure over the artifact dict (no store access): measured/accepted/rejected
    counts, the accepted treated keys, and the split version. Part E renders
    these as prose with an explicit artifact citation — never a number copy
    without provenance.
    """
    outcomes = verdicts.get("outcomes")
    if not isinstance(outcomes, list):
        raise DataMethodsError(
            "gate verdicts artifact carries no outcomes list; the part E "
            "disposition block needs the measured outcomes."
        )
    accepted = sorted(
        str(entry.get("treated_key", ""))
        for entry in outcomes
        if isinstance(entry, dict) and entry.get("verdict") == "accepted"
    )
    measured = len(outcomes)
    return {
        "measured": measured,
        "accepted": len(accepted),
        "rejected": measured - len(accepted),
        "accepted_treated_keys": accepted,
        "split_version": verdicts.get("split_version", ""),
    }


def _render_gate_dispositions_block(
    plan: dict[str, Any], verdicts: dict[str, Any] | None
) -> list[str]:
    """Final dispositions per plan candidate from the verdicts artifact.

    Returns [] when the artifact is absent (pre-declared wording stands).
    Rung 0 keeps documented-not-re-measured; every plan candidate variant
    with measured outcomes renders one gate-measured line per pair via
    :func:`format_gate_disposition` (line format byte-stable); any plan
    variant with no measured outcome renders N/A-by-evidence. Covers the
    transform/imputation/encoding candidates from the plan register, not
    just the imbalance ladder. An outcome whose candidate key is not
    declared in the plan raises loudly (never a silent skip).
    Robust-family N/A-by-evidence justifications already live verbatim in
    part B's light section and are untouched here.
    """
    if verdicts is None:
        return []
    by_variant = index_verdicts_by_variant(verdicts)
    declared = {
        str(entry.get("variant_key", ""))
        for entry in plan.get("candidates", [])
        if isinstance(entry, dict) and entry.get("variant_key")
    }
    for key in by_variant:
        if key not in declared:
            raise DataMethodsError(
                f"gate verdicts outcome references undeclared candidate {key!r}; "
                "the rendered dispositions must never invent a variant."
            )
    rung_by_variant: dict[str, Any] = {}
    for rung in plan.get("imbalance_ladder", []):
        variant = rung.get("variant_key") if isinstance(rung, dict) else None
        if isinstance(variant, str) and variant:
            rung_by_variant[variant] = rung.get("rung")
    lines = [
        "### Gate dispositions (measured)",
        "",
        (
            "Gate-measured dispositions from "
            f"`{GATE_VERDICTS_FILENAME}` (one line per measured pair; the "
            "incumbent rung keeps its documented-not-re-measured status):"
        ),
        "",
        ("- rung 0 (incumbent): documented-not-re-measured — no gate "
         "outcome recorded."),
    ]
    ordered = sorted(
        (c for c in plan.get("candidates", []) if isinstance(c, dict)),
        key=lambda c: c.get("order", 0),
    )
    for candidate in ordered:
        if candidate.get("is_incumbent"):
            continue
        variant = candidate.get("variant_key")
        if not isinstance(variant, str) or not variant:
            continue
        measured = by_variant.get(variant, [])
        if not measured:
            rung = rung_by_variant.get(variant)
            if rung is not None:
                lines.append(
                    f"- rung {rung} (`{variant}`): N/A-by-evidence — "
                    "no measured outcome in the verdicts artifact."
                )
            else:
                lines.append(
                    f"- `{variant}`: N/A-by-evidence — "
                    "no measured outcome in the verdicts artifact."
                )
        else:
            lines.extend(f"- {format_gate_disposition(outcome)}" for outcome in measured)
    lines.append("")
    return lines


def _render_part_e_dispositions_block(verdicts: dict[str, Any] | None) -> list[str]:
    """Part E gate-measured summary block (prose only, never a table).

    Returns [] when the artifact is absent. Counts are cited with an
    explicit artifact citation, never copied bare.
    """
    if verdicts is None:
        return []
    summary = summarize_gate_dispositions(verdicts)
    accepted_keys = summary["accepted_treated_keys"]
    if accepted_keys:
        tail = (
            "Accepted: "
            + ", ".join(f"`{key}`" for key in accepted_keys)
            + "; the acceptance branch record in the artifact governs the "
            "treated generation."
        )
    else:
        tail = (
            "The committed selection report's winner standing is unchanged "
            "by the gate (no acceptance re-ran the battery)."
        )
    return [
        (
            f"Gate-measured dispositions (artifact `{GATE_VERDICTS_FILENAME}`): "
            f"{summary['measured']} measured pair(s), "
            f"{summary['accepted']} accepted, {summary['rejected']} rejected "
            f"at split_version `{summary['split_version']}`; every rejection "
            "keeps the incumbent pipeline. " + tail
        ),
        "",
    ]


# ---------------------------------------------------------------------------
# Part D — Pre-Declared Class-Imbalance Ladder
# ---------------------------------------------------------------------------


def render_part_d(
    plan: dict[str, Any], verdicts: dict[str, Any] | None = None
) -> list[str]:
    """Render part D strictly from ``treatment_plan.json`` (D032 mirror).

    When the MLflow-derived gate-verdicts artifact is supplied, every plan
    candidate variant with measured outcomes gains its final disposition:
    the incumbent rung keeps documented-not-re-measured, each measured
    variant lists one gate-measured line per pair (delta + significance
    reading + verdict) via format_gate_disposition, and a plan variant with
    no measured outcome in the artifact is marked N/A-by-evidence. Covers
    the transform/imputation/encoding candidates from the plan register,
    not just the imbalance ladder. When the artifact is absent the renderer
    keeps today's pre-declared wording (never a stack trace) so upstream
    page rendering never breaks.
    """
    candidates = {entry["variant_key"]: entry for entry in plan["candidates"]}
    rows = []
    for rung in plan["imbalance_ladder"]:
        key = rung.get("variant_key")
        if key is None:
            mechanism = "none (documented, not re-measured)"
            applies = "whole battery (incumbent pipeline with its tuned class-weight policy)"
            key_cell = "—"
        else:
            candidate = candidates.get(key)
            if candidate is None:
                raise DataMethodsError(
                    f"ladder rung {rung['rung']} references candidate {key!r}, "
                    "which the declared plan does not contain; the rendered "
                    "ladder must never invent a rung."
                )
            mechanism = candidate["mechanism"]
            applies = ", ".join(f"`{model}`" for model in candidate["applies_to"])
            key_cell = f"`{key}`"
        rows.append([str(rung["rung"]), rung["label"], key_cell, mechanism, applies])
    return [
        "## Part D — Pre-Declared Class-Imbalance Ladder",
        "",
        (
            "The ladder is rendered strictly from the declared plan artifact "
            f"(`treatment_plan.json`), which mirrors "
            "`heart/eval/treatment_declared.py` 1:1 (decision D032). Rung 0 "
            "is the documented incumbent and is not re-measured; every later "
            "rung is one declared imbalance candidate, applied inside the "
            "training fold only (never before the split)."
        ),
        "",
        _md_table(
            ["rung", "ladder label", "candidate key", "mechanism", "applies to"],
            rows,
        ),
        "",
        "*Artifact:* `treatment_plan.json`",
        "",
    ] + _render_gate_dispositions_block(plan, verdicts)


# ---------------------------------------------------------------------------
# Part E — Validation and Calibration Strategy
# ---------------------------------------------------------------------------


def selection_carries_calibration_flag(
    flag: str = CALIBRATION_FLAG_NAME,
) -> bool:
    """Whether the committed selection report currently carries the flag.

    A repo-level fact probed at render time so the committed report's claim
    stays honest: if the selection report is absent (or does not carry the
    flag), part E says so instead of asserting it. The report copies none of
    the selection report's numbers either way (D025).
    """
    from heart.config import REPORTS_DIR

    path = REPORTS_DIR / "selection.md"
    if not path.is_file():
        return False
    return flag in path.read_text(encoding="utf-8")


def render_part_e(
    *, selection_carries_flag: bool, verdicts: dict[str, Any] | None = None
) -> list[str]:
    """Render part E: repeated CV, paired significance, calibration honesty.

    When the gate-verdicts artifact is supplied, a gate-measured disposition
    block (prose only, never a table) records the measured/accepted/rejected
    counts with an explicit artifact citation. When absent the renderer keeps
    today's wording so upstream page rendering never breaks.
    """
    """Render part E: repeated CV, paired significance, calibration honesty."""
    pointer_sentence = (
        (
            f"The current committed selection report (`{SELECTION_REPORT_POINTER}`) "
            "carries that flag; this part cites it textually and copies none of "
            "its numbers (D025) — MLflow-derived numbers stay in their own "
            "reports."
        )
        if selection_carries_flag
        else (
            f"The textual pointer for that flag is the committed selection "
            f"report `{SELECTION_REPORT_POINTER}` (not found at render time; "
            "regenerate it through its own pipeline) — this part copies none "
            "of its numbers (D025)."
        )
    )
    return [
        "## Part E — Validation and Calibration Strategy",
        "",
        (
            f"Every declared candidate is scored with repeated stratified "
            "cross-validation (`heart/eval/repeated_cv.py`): the committed "
            f"split's {N_SPLITS} folds re-drawn across {DEFAULT_N_REPEATS} "
            "differently-seeded repeats, every fold model fit on its training "
            "fold only, so a candidate's repeated-CV fold deltas stay paired "
            "across the same fold structure."
        ),
        "",
        (
            "The significance layer reads those paired deltas with the "
            "Nadeau-Bengio corrected resampled t-test "
            "(`heart/eval/significance.py`, `corrected_resampled_t_test`) at "
            f"alpha={PAIRED_SIGNIFICANCE_ALPHA:g}; the same alpha is fixed in "
            "the part F gate constants."
        ),
        "",
        (
            "Probability calibration is measured, not assumed: when the "
            "calibrated model fails to improve the Brier score, the pipeline "
            f"records the honesty flag `{CALIBRATION_FLAG_NAME}` instead of "
            "shipping the calibration silently."
        ),
        "",
        pointer_sentence,
        "",
    ] + _render_part_e_dispositions_block(verdicts)


# ---------------------------------------------------------------------------
# Part F — Pre-Declared Severity Gate
# ---------------------------------------------------------------------------


def _gate_sentence(gate: dict[str, Any]) -> str:
    """Cross-check the rendered gate constants against the declared module.

    Defense-in-depth (D032): the artifact text must equal
    :func:`heart.eval.treatment_declared.gate_text` exactly, and the
    threshold/alpha/metric constants must appear in it as rendered. Any
    disagreement between the declarations module and the rendered constants
    fails loudly as a named :class:`DataMethodsError`.
    """
    text = gate.get("text")
    expected = gate_text()
    if not isinstance(text, str) or text != expected:
        raise DataMethodsError(
            "part F gate text drifted from heart.eval.treatment_declared: "
            f"artifact text was {text!r}, declared module says {expected!r}."
        )
    delta = f"+{GATE_MIN_DELTA_ROC_AUC:.4f}"
    if delta not in text:
        raise DataMethodsError(
            f"part F gate text must contain the {delta} threshold formatted "
            f"from GATE_MIN_DELTA_ROC_AUC; rendered text was {text!r}."
        )
    if GATE_METRIC not in text or f"alpha={PAIRED_SIGNIFICANCE_ALPHA:g}" not in text:
        raise DataMethodsError(
            "part F gate text must name the declared metric "
            f"{GATE_METRIC!r} and alpha={PAIRED_SIGNIFICANCE_ALPHA:g}; "
            f"rendered text was {text!r}."
        )
    return text


def render_part_f(plan: dict[str, Any]) -> list[str]:
    """Render part F from the declared plan artifact, with the D032 assertion."""
    gate = plan.get("gate")
    if not isinstance(gate, dict):
        raise DataMethodsError(
            "part F needs the declared gate block in treatment_plan.json; "
            f"got {type(gate).__name__}."
        )
    if plan.get("declared_before_any_candidate_number") is not True:
        raise DataMethodsError(
            "part F renders the gate as pre-declared, but "
            "treatment_plan.json does not record "
            "declared_before_any_candidate_number: true."
        )
    sentence = _gate_sentence(gate)
    return [
        "## Part F — Pre-Declared Severity Gate (fixed before any candidate "
        "number exists)",
        "",
        sentence,
        "",
        (
            "The significance layer reads the delta through the paired "
            "significance layer at "
            f"alpha={PAIRED_SIGNIFICANCE_ALPHA:g}: a delta the corrected "
            "resampled t-test does not read as significant keeps the "
            "incumbent, regardless of size."
        ),
        "",
        _md_table(
            ["declared constant", "value"],
            [
                ["gate metric", f"`{GATE_METRIC}`"],
                ["minimum repeated-CV ROC-AUC delta", f"+{GATE_MIN_DELTA_ROC_AUC:.4f}"],
                ["winner-standing rule", GATE_WINNER_STANDING_RULE],
                ["paired significance alpha", f"{PAIRED_SIGNIFICANCE_ALPHA:g}"],
            ],
        ),
        "",
        "*Artifact:* `treatment_plan.json`",
        "",
        (
            "Declaration of antecedence: the gate metric, the delta "
            "threshold, the significance alpha, and the winner-standing rule "
            "above were fixed in `heart/eval/treatment_declared.py` (decision "
            "D032) before any candidate was measured; the declared plan "
            "artifact records `declared_before_any_candidate_number: true`."
        ),
        "",
    ]
