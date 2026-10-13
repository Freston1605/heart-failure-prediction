"""Treatment-audit engine for ``reports/data-methods.md`` parts B, D, F (M006/S01/T02).

Why this module exists
----------------------
Decision D025 forbids any hand-assembled number in the generated report.
This module is the sanctioned builder of the two treatment artifacts
(``treatment_audit.json``, ``treatment_plan.json``): every value it emits is
extracted either from the *same code constants the actual pipeline runs*
(loader, split, preprocessing chain, battery specs) or from a preprocessing
pipeline *fitted over training rows only*. The report renderer (T03) consumes
these artifacts; it never recomputes or invents a number.

Model-Specific Pipeline Pattern (decision D033)
----------------------------------------------
The audit is framed explicitly as the pattern the whole battery follows:

1. **One pinned raw dataset** — :data:`heart.data.load.PINNED_SHA256` is the
   digest everything downstream is verified against.
2. **Split first** — the committed split (version :data:`SPLIT_VERSION`,
   ``N_SPLITS`` stratified folds, fixed ``TEST_FOLD``) is resolved before any
   transform is fit, and the group-straddle failure policy is fail-closed:
   :func:`heart.data.split.assert_no_group_straddle` raises
   ``SplitLeakageError`` rather than letting near-duplicate rows straddle the
   boundary.
3. **Per-model pipelines fit on training rows only** — every battery member
   prepends the shared leakage-safe preprocessing chain (declared zero policy
   -> ColumnTransformer with StandardScaler on the numeric columns and
   OneHotEncoder with ``handle_unknown="ignore"`` on the categorical
   columns), fit on the training side of the split and never on the full
   frame. The fitted statistics recorded here (imputer medians, scaler
   centre/spread, encoder categories, expanded-feature naming) are exactly
   the numbers a leaky fit would contaminate.
4. **Unified evaluate() scoring** — every member is scored by the same
   contract, so the audit compares treatment chains, not scoring dialects.

Depth tiers (decision D028)
---------------------------
The linear family (logistic-regression L1/L2/elasticnet variants, LDA, QDA —
:data:`heart.eval.treatment_declared.LINEAR_FAMILY_SCOPE`) is audited at
**deep** depth: each pipeline step with its actual sklearn parameters plus
the fitted train-side statistics. Robust families are audited at **light**
depth with a written ``robustness_justification`` naming why the
transform/imputation/encoding candidates do not bind for them. These
justifications are *declared reviewer-facing constants* of this module —
not measured claims — and a battery change that adds an unjustified robust
member fails loudly at build time.

Seam (decision D030)
--------------------
Like ``generate_report``/``run_eda``, the engine takes a plain
:class:`pandas.DataFrame` argument (the training rows) and writes no files
itself; :func:`write_treatment_artifacts` mirrors
``write_stage_artifacts`` in :mod:`heart.reporting.data_methods.report`
(serialize everything first, then write-tmp-then-:func:`os.replace`), so a
failed write leaves no partial artifact behind. Rendering stays
deterministic: sorted keys, fixed six-decimal float rounding, no timestamps.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from heart.data.load import PINNED_SHA256
from heart.data.pipeline import (
    PIPELINE_CATEGORICAL_COLUMNS,
    PIPELINE_NUMERIC_COLUMNS,
    FeatureFrameError,
    build_preprocessing_pipeline,
    fit_preprocessing,
    fitted_statistics,
)
from heart.data.quality import (
    DEFAULT_ZERO_POLICY,
    NoValidValuesError,
)
from heart.data.split import (
    N_SPLITS,
    SPLIT_VERSION,
    TEST_FOLD,
    SplitLeakageError,
    assert_no_group_straddle,
)
from heart.eval.treatment_declared import (
    DECLARED_CANDIDATES,
    GATE_METRIC,
    GATE_MIN_DELTA_ROC_AUC,
    GATE_WINNER_STANDING_RULE,
    IMBALANCE_LADDER,
    LINEAR_FAMILY_SCOPE,
    MLFLOW_RUN_KIND_TREATMENT_CANDIDATE,
    MLFLOW_TAG_TREATMENT_VARIANT,
    PAIRED_SIGNIFICANCE_ALPHA,
    gate_text,
)
from heart.models.registry import BATTERY_SPECS, BATTERY_MODEL_TYPES

__all__ = [
    "TREATMENT_ARTIFACTS",
    "AUDIT_FILENAME",
    "PLAN_FILENAME",
    "FLOAT_DECIMALS",
    "TreatmentAuditError",
    "ROBUSTNESS_JUSTIFICATIONS",
    "build_declared_plan",
    "build_treatment_audit",
    "write_treatment_artifacts",
]

#: Artifact filename per payload (mirrors ``STAGE_ARTIFACTS`` in report.py).
TREATMENT_ARTIFACTS: dict[str, str] = {
    "treatment_audit": "treatment_audit.json",
    "treatment_plan": "treatment_plan.json",
}

#: Shorthand for the two artifact basenames (T03 cites these).
AUDIT_FILENAME: str = TREATMENT_ARTIFACTS["treatment_audit"]
PLAN_FILENAME: str = TREATMENT_ARTIFACTS["treatment_plan"]

#: Fixed rounding for every emitted float (part A's engine rounds to six
#: decimals; the audit matches so cross-artifact comparisons stay exact).
FLOAT_DECIMALS: int = 6


# ---------------------------------------------------------------------------
# Named error
# ---------------------------------------------------------------------------


class TreatmentAuditError(Exception):
    """A treatment-audit payload could not be built or serialized.

    Raised for a non-frame training argument, a frame that fails the
    preprocessing fit (missing feature columns, all-sentinel policy column),
    or a payload that is not JSON-serializable. Mirrors the named-failure
    style of :class:`heart.reporting.data_methods.eda.DataMethodsError`.
    """


# ---------------------------------------------------------------------------
# Deterministic JSON helpers
# ---------------------------------------------------------------------------


def _param_json(value: Any) -> Any:
    """Normalize one parameter value to JSON-native, deterministic form.

    Tuples become lists, numpy scalars unwrap, dtypes and callables render
    by name, and everything exotic falls back to ``str``. Used both for
    sklearn transformer parameters and for registry fixed parameters.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            return str(value)
        return round(value, FLOAT_DECIMALS)
    if isinstance(value, np.generic):
        return _param_json(value.item())
    if isinstance(value, np.dtype):
        return str(value)
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_param_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _param_json(item) for key, item in sorted(value.items())}
    if callable(value):
        return getattr(value, "__name__", repr(value))
    return str(value)


def _round_float(value: float) -> float:
    """Round a finite float to :data:`FLOAT_DECIMALS` (deterministic output)."""
    if not np.isfinite(value):
        return str(value)
    return round(float(value), FLOAT_DECIMALS)


# ---------------------------------------------------------------------------
# Declared plan (parts D and F) — mirrored 1:1 from the T01 module
# ---------------------------------------------------------------------------


def build_declared_plan() -> dict[str, Any]:
    """Build the declared-plan payload from :mod:`heart.eval.treatment_declared`.

    The module (decision D032) is the single source of truth shared with the
    S02 gate engine; this builder only *mirrors* its constants into a
    JSON-shaped dict — it re-declares nothing and invents no number. The
    ``declared_before_any_candidate_number`` marker records the antecedence
    contract: the gate text below was fixed in code before any candidate
    was measured.
    """
    return {
        "source_module": "heart.eval.treatment_declared",
        "declared_before_any_candidate_number": True,
        "candidates": [
            {
                "variant_key": candidate.variant_key,
                "mechanism": candidate.mechanism,
                "group": candidate.group,
                "applies_to": list(candidate.applies_to),
                "description": candidate.description,
                "mlflow_treatment_variant": candidate.mlflow_treatment_variant,
                "order": candidate.order,
                "is_incumbent": candidate.is_incumbent,
            }
            for candidate in DECLARED_CANDIDATES
        ],
        "imbalance_ladder": [
            {
                "rung": rung.rung,
                "label": rung.label,
                "variant_key": rung.variant_key,
            }
            for rung in IMBALANCE_LADDER
        ],
        "gate": {
            "metric": GATE_METRIC,
            "min_delta_roc_auc": GATE_MIN_DELTA_ROC_AUC,
            "winner_standing_rule": GATE_WINNER_STANDING_RULE,
            "paired_significance_alpha": PAIRED_SIGNIFICANCE_ALPHA,
            "text": gate_text(),
        },
        "mlflow": {
            "tag_treatment_variant": MLFLOW_TAG_TREATMENT_VARIANT,
            "run_kind_treatment_candidate": MLFLOW_RUN_KIND_TREATMENT_CANDIDATE,
        },
    }


# ---------------------------------------------------------------------------
# Pipeline-step extraction (deep block material)
# ---------------------------------------------------------------------------


def _pipeline_steps_block() -> list[dict[str, Any]]:
    """Extract the shared preprocessing chain from the *actual* sklearn objects.

    Built from a fresh :func:`heart.data.pipeline.build_preprocessing_pipeline`
    call, so the recorded step names and parameters are the ones the battery
    really runs — never a transcription.
    """
    pipeline = build_preprocessing_pipeline()
    steps: list[dict[str, Any]] = []
    for name, transformer in pipeline.steps:
        if hasattr(transformer, "transformers_") or hasattr(transformer, "transformers"):
            # ColumnTransformer: record its own declared params, then each
            # sub-transformer with its columns and full parameter dict.
            sub_transformers = [
                {
                    "name": sub_name,
                    "transformer": type(sub_transformer).__name__,
                    "columns": [str(column) for column in columns],
                    "params": {
                        key: _param_json(value)
                        for key, value in sorted(sub_transformer.get_params().items())
                    },
                }
                for sub_name, sub_transformer, columns in transformer.transformers
            ]
            steps.append(
                {
                    "step": str(name),
                    "transformer": type(transformer).__name__,
                    "params": {
                        key: _param_json(value)
                        for key, value in sorted(
                            transformer.get_params(deep=False).items()
                        )
                    },
                    "sub_transformers": sub_transformers,
                }
            )
        else:
            steps.append(
                {
                    "step": str(name),
                    "transformer": type(transformer).__name__,
                    "params": {
                        key: _param_json(value)
                        for key, value in sorted(transformer.get_params().items())
                    },
                }
            )
    return steps


# ---------------------------------------------------------------------------
# Fitted train-side statistics
# ---------------------------------------------------------------------------


def _fit_train_statistics(train_frame: pd.DataFrame) -> dict[str, Any]:
    """Fit the declared chain on ``train_frame`` rows only and extract statistics.

    This is the only fitting this module performs, and it mirrors the actual
    usage contract: the caller hands over *training rows*, the pipeline is fit
    on exactly those rows, and the fitted statistics (imputer medians, scaler
    centre/spread, encoder categories, expanded-feature naming) are recorded
    as they are — rounded to :data:`FLOAT_DECIMALS` for deterministic output.
    """
    pipeline = fit_preprocessing(train_frame)
    stats = fitted_statistics(pipeline)
    feature_names = [str(name) for name in pipeline.get_feature_names_out()]
    return {
        "row_count": int(len(train_frame)),
        "zero_medians": {
            str(column): _round_float(value)
            for column, value in sorted(stats["zero_medians"].items())
        },
        "scaler_mean": [_round_float(value) for value in stats["scaler_mean"]],
        "scaler_scale": [_round_float(value) for value in stats["scaler_scale"]],
        "encoder_categories": stats["encoder_categories"],
        "expanded_feature_count": len(feature_names),
        "expanded_feature_names": feature_names,
    }


# ---------------------------------------------------------------------------
# Depth-tier robustness justifications (declared constants, D028)
# ---------------------------------------------------------------------------

#: Reviewer-facing reasons why the transform/imputation/encoding candidates
#: from the declared plan do not bind for the robust battery families. These
#: are written declarations, not measured claims; every robust member must
#: carry one or the build fails loudly (a new battery member cannot sneak in
#: un-audited).
ROBUSTNESS_JUSTIFICATIONS: dict[str, str] = {
    "naive-bayes": (
        "Gaussian naive Bayes estimates one mean and one variance per feature, "
        "so a monotone power transform would shift those estimates but cannot "
        "change the model's shape; the committed median imputation already "
        "keeps every row usable, and the one-hot keep-all encoding matches the "
        "independent-feature assumption the model declares, so the transform, "
        "imputation, and encoding rungs are not re-measured for this family."
    ),
    "knn": (
        "The k-nearest-neighbour vote is taken on distances computed in the "
        "standardized space the shared chain produces, so the metric the "
        "model uses is already the scaled one; a monotone transform preserves "
        "neighbour orderings up to scale and the committed median imputation "
        "is what the distances actually see, so the transform and imputation "
        "rungs are not re-measured for this family."
    ),
    "svm": (
        "Kernel evaluations happen in the standardized space the shared chain "
        "produces, and the tuned C/gamma budget already absorbs any feature-"
        "shape change a transform would introduce; the committed median "
        "imputation feeds the same standardized features regardless, so the "
        "transform, imputation, and encoding rungs are not re-measured for "
        "this family."
    ),
    "random-forest": (
        "Axis-aligned leaf splits are invariant to any monotone transform of "
        "a single feature, so the power-transform rungs cannot change the "
        "trees this family grows; the trees never scale features and the "
        "committed median imputation is the only missing-value policy the "
        "chain applies, so the transform and encoding rungs are not "
        "re-measured for this family."
    ),
    "xgboost": (
        "Split-gain trees are invariant to monotone per-feature transforms and "
        "the booster handles whatever the shared imputer hands it, so the "
        "power-transform rungs cannot change the learned structure; the "
        "committed one-hot keep-all encoding already feeds the booster "
        "numeric columns only, so the transform and encoding rungs are not "
        "re-measured for this family."
    ),
}


def _model_entry(
    model_type: str,
    *,
    fitted: dict[str, Any],
    steps_block: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one battery member's audit block from the registry spec.

    Linear-family members (D028 scope) get the deep block: every pipeline
    step with its actual sklearn parameters plus the fitted expanded-feature
    count that feeds the drop-first narrative. Robust members get the light
    block plus their declared robustness justification.
    """
    spec = BATTERY_SPECS[model_type]
    deep = model_type in LINEAR_FAMILY_SCOPE
    entry: dict[str, Any] = {
        "model_type": spec.model_type,
        "model_name": spec.model_name,
        "family": spec.family,
        "preprocessing": spec.preprocessing,
        "tuned_space_keys": sorted(spec.declared_hyperparameters),
        "fixed_params": {
            str(key): _param_json(value)
            for key, value in sorted(spec.fixed_params.items())
        },
        "estimator_description": spec.description,
        "depth": "deep" if deep else "light",
    }
    if deep:
        entry["pipeline_steps"] = steps_block
        entry["expanded_feature_count"] = fitted["expanded_feature_count"]
        entry["expanded_feature_names"] = fitted["expanded_feature_names"]
    else:
        justification = ROBUSTNESS_JUSTIFICATIONS.get(model_type)
        if not justification:
            raise TreatmentAuditError(
                f"battery member {model_type!r} is not in the declared "
                "linear-family scope and has no robustness justification; "
                "add one to ROBUSTNESS_JUSTIFICATIONS before the audit can "
                "render it at light depth (D028)."
            )
        entry["robustness_justification"] = justification
    return entry


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_treatment_audit(train_frame: pd.DataFrame) -> dict[str, Any]:
    """Build the per-model treatment-audit payload for report part B.

    Parameters
    ----------
    train_frame:
        The **training rows** of the committed split, as a plain
        :class:`pandas.DataFrame` (the D030 seam). The preprocessing pipeline
        is fit on exactly these rows; nothing else is read.

    Returns
    -------
    dict
        Deterministic JSON-shaped payload: pattern framing, loader/split
        provenance, the declared zero policy and preprocessing chain, the
        fitted train-side statistics, and one block per battery member
        sorted by ``model_type``.

    Raises
    ------
    TreatmentAuditError
        For a non-frame argument or a frame that fails the preprocessing fit
        (missing feature columns, a policy column with no non-sentinel value).
        The underlying actionable message is preserved in the chain.
    """
    if not isinstance(train_frame, pd.DataFrame):
        raise TreatmentAuditError(
            "build_treatment_audit needs the training rows as a "
            f"pandas.DataFrame, got {type(train_frame).__name__}."
        )
    try:
        fitted = _fit_train_statistics(train_frame)
    except (FeatureFrameError, NoValidValuesError) as exc:
        raise TreatmentAuditError(
            f"train frame failed the preprocessing fit: {exc}"
        ) from exc

    steps_block = _pipeline_steps_block()
    models = [
        _model_entry(model_type, fitted=fitted, steps_block=steps_block)
        for model_type in sorted(BATTERY_MODEL_TYPES)
    ]
    deep_members = [entry["model_type"] for entry in models if entry["depth"] == "deep"]
    light_members = [entry["model_type"] for entry in models if entry["depth"] == "light"]
    return {
        "pattern": {
            "name": "Model-Specific Pipeline Pattern",
            "decision": "D033",
            "description": (
                "one pinned raw dataset, split first, per-model pipelines fit "
                "on training rows only, unified evaluate() scoring"
            ),
            "depth_policy_decision": "D028",
            "deep_scope": list(LINEAR_FAMILY_SCOPE),
        },
        "loader": {
            "module": "heart.data.load",
            "pinned_sha256": PINNED_SHA256,
        },
        "split": {
            "module": "heart.data.split",
            "version": SPLIT_VERSION,
            "n_splits": N_SPLITS,
            "test_fold": TEST_FOLD,
            "group_straddle_policy": (
                "fail-closed: heart.data.split."
                f"{assert_no_group_straddle.__name__} raises "
                f"{SplitLeakageError.__name__} if any near-duplicate group "
                "spans the train/test boundary"
            ),
        },
        "zero_policy": {
            "module": "heart.data.quality.DEFAULT_ZERO_POLICY",
            "action": DEFAULT_ZERO_POLICY.action,
            "columns": list(DEFAULT_ZERO_POLICY.columns),
            "sentinel": DEFAULT_ZERO_POLICY.sentinel,
        },
        "preprocessing": {
            "module": "heart.data.pipeline",
            "numeric_columns": list(PIPELINE_NUMERIC_COLUMNS),
            "categorical_columns": list(PIPELINE_CATEGORICAL_COLUMNS),
            "pipeline_steps": steps_block,
            "fit_on": "training rows only",
            "fitted": fitted,
        },
        "depth_tiers": {"deep": deep_members, "light": light_members},
        "models": models,
    }


# ---------------------------------------------------------------------------
# Atomic artifact writer (mirrors write_stage_artifacts in report.py)
# ---------------------------------------------------------------------------


def write_treatment_artifacts(
    audit_payload: dict[str, Any],
    declared_plan: dict[str, Any],
    output_dir: str | Path,
) -> list[str]:
    """Write ``treatment_audit.json`` and ``treatment_plan.json`` atomically.

    Both payloads are serialized **before** anything is written, so a
    non-serializable value fails before a single byte lands on disk. Each
    file is then written write-tmp-then-:func:`os.replace`; a failed replace
    removes its tmp file so no partial artifact is ever left behind.

    Returns
    -------
    list[str]
        The artifact basenames written, in the fixed declaration order.

    Raises
    ------
    TreatmentAuditError
        If either payload is not a dict or fails JSON serialization.
    """
    if not isinstance(audit_payload, dict):
        raise TreatmentAuditError(
            "audit payload must be a dict, got "
            f"{type(audit_payload).__name__}."
        )
    if not isinstance(declared_plan, dict):
        raise TreatmentAuditError(
            "declared-plan payload must be a dict, got "
            f"{type(declared_plan).__name__}."
        )
    payloads: dict[str, str] = {}
    for payload, artifact_key in (
        (audit_payload, "treatment_audit"),
        (declared_plan, "treatment_plan"),
    ):
        try:
            payloads[TREATMENT_ARTIFACTS[artifact_key]] = json.dumps(
                payload, indent=2, sort_keys=True, ensure_ascii=False
            ) + "\n"
        except (TypeError, ValueError) as exc:
            raise TreatmentAuditError(
                f"{artifact_key} payload is not JSON-serializable: {exc}"
            ) from exc

    directory = Path(output_dir)
    written: list[str] = []
    for filename, text in payloads.items():
        target = directory / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, target)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
        written.append(filename)
    return written
