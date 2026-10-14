"""Pre-declared treatment plan (decisions D032, D028): candidates, ladder, gate.

This module is the single source of truth for the M006 S01 -> S02 contract,
"pre-registered gate text and the named candidate list" (M006 roadmap,
Contracts Between Slices). Two consumers import it and are forbidden from
re-declaring anything:

- the S01 report generator, which renders
  ``reports/data-methods.md`` parts D (class-imbalance ladder) and
  F (pre-declared severity gate) strictly from the constants below; and
- the S02 gate runner, which measures the declared candidates through
  :func:`heart.eval.contract.evaluate` using the winner-standing rule and
  thresholds fixed here.

Design rules pinned by the slice plan:

1. **Declarative only.** Nothing in this module measures, fits, or computes a
   number. Every field is a written declaration; the incumbent median policy
   is *documented, not re-measured*. No field pretends to carry a measurement.
2. **Cheap import.** Only stdlib modules are imported (``dataclasses``,
   ``typing``). The S01 report CLI startup must never pay for sklearn,
   pandas, or numpy just to render declared prose.
3. **Fail loudly at import time.** Like :mod:`heart.models.registry` specs,
   every candidate is validated when the module loads; a malformed
   declaration raises :class:`TreatmentDeclarationError` before any consumer
   can render or measure half of a broken plan.
4. **Antecedence.** The gate text (part F) is fixed in code *before any
   candidate number exists* -- :data:`GATE_MIN_DELTA_ROC_AUC`,
   :data:`PAIRED_SIGNIFICANCE_ALPHA`, :data:`GATE_METRIC`, and the
   winner-standing rule are constants, not parameters of a measurement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final, Literal, Optional

__all__ = [
    "MechanismKind",
    "CandidateGroup",
    "TreatmentDeclarationError",
    "TreatmentCandidate",
    "LadderRung",
    "LINEAR_FAMILY_SCOPE",
    "BATTERY_MEMBERS",
    "DECLARED_CANDIDATES",
    "DECLARED_CANDIDATE_KEYS",
    "INCUMBENT_CANDIDATE",
    "IMBALANCE_LADDER",
    "GATE_MIN_DELTA_ROC_AUC",
    "GATE_METRIC",
    "GATE_WINNER_STANDING_RULE",
    "PAIRED_SIGNIFICANCE_ALPHA",
    "MLFLOW_TAG_TREATMENT_VARIANT",
    "MLFLOW_RUN_KIND_TREATMENT_CANDIDATE",
    "validate_candidates",
    "validate_ladder",
    "validate_all",
    "gate_text",
]

# --------------------------------------------------------------------------
# Kinds and groups
# --------------------------------------------------------------------------

#: How a candidate changes the pipeline/estimator.
#:
#: - ``estimator-param``: a kwarg changed on the model itself.
#: - ``pipeline-variant``: a mutation of the preprocessing chain, fit on
#:   training rows only inside the same Pipeline.
#: - ``sampling``: a resampling step applied inside the training fold only
#:   (never before the split -- the M006 leakage golden rule).
MechanismKind = Literal["estimator-param", "pipeline-variant", "sampling"]

#: Which part of the data treatment a candidate touches.
CandidateGroup = Literal["transform", "imputation", "encoding", "imbalance"]

_MECHANISM_KINDS: Final[frozenset[str]] = frozenset(
    {
        "estimator-param",
        "pipeline-variant",
        "sampling",
    }
)
_CANDIDATE_GROUPS: Final[frozenset[str]] = frozenset(
    {
        "transform",
        "imputation",
        "encoding",
        "imbalance",
    }
)

_VARIANT_KEY_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9-]*$")

#: Model-type scope of the transform/imputation/encoding/scaling candidates
#: per decision D028 (gate candidate scope): the LR variants plus LDA/QDA.
#: Robust families are audited at lighter depth with written robustness
#: justifications, not re-measured.
LINEAR_FAMILY_SCOPE: Final[tuple[str, ...]] = (
    "logistic-regression-l2",
    "logistic-regression-l1",
    "logistic-regression-elasticnet",
    "lda",
    "qda",
)

#: The committed battery members the documented incumbent median policy
#: applies to (declared literally here; tests cross-check against
#: ``heart.models.registry.BATTERY_MODEL_TYPES`` so this copy can never
#: silently drift from the battery).
BATTERY_MEMBERS: Final[tuple[str, ...]] = (
    "logistic-regression-l2",
    "logistic-regression-l1",
    "logistic-regression-elasticnet",
    "naive-bayes",
    "lda",
    "qda",
    "knn",
    "svm",
    "random-forest",
    "xgboost",
)

# --------------------------------------------------------------------------
# Error type
# --------------------------------------------------------------------------


class TreatmentDeclarationError(Exception):
    """A declared treatment candidate or ladder rung is malformed.

    Raised at import time by the module guard (mirroring how
    :class:`heart.models.registry.RegistryError` fails loudly) and by
    :func:`validate_candidates` / :func:`validate_ladder` for arbitrary
    inputs, so the negative-test surface can drive the same checks.
    """


# --------------------------------------------------------------------------
# Candidate record
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TreatmentCandidate:
    """One declared, re-measurable treatment candidate.

    Fields are declarations, not measurements:

    - ``variant_key``: slug with no spaces; also the MLflow
      ``treatment_variant`` tag value (see D024 extension).
    - ``mechanism``: one of :data:`MechanismKind`'s three values.
    - ``group``: one of :data:`CandidateGroup`'s four values.
    - ``applies_to``: tuple of battery ``model_type`` slugs. Per D028 the
      transform/imputation/encoding/imbalance rungs bind the linear family;
      the documented incumbent applies to the whole battery.
    - ``description``: one declarative sentence naming the exact change.
    - ``mlflow_treatment_variant``: the exact ``treatment_variant`` tag value
      the S02 runner must attach under ``run_kind=treatment-candidate``.
    - ``order``: unique integer fixing the deterministic list order.
    - ``is_incumbent``: True only for the documented incumbent (rung 0 of the
      honest ladder story); incumbents are described, not re-measured.
    """

    variant_key: str
    mechanism: MechanismKind
    group: CandidateGroup
    applies_to: tuple[str, ...]
    description: str
    mlflow_treatment_variant: str
    order: int
    is_incumbent: bool = field(default=False)

    def validate(self, battery: tuple[str, ...] = BATTERY_MEMBERS) -> None:
        """Validate this candidate, raising :class:`TreatmentDeclarationError`."""
        _check_candidate(self, battery)


def _check_candidate(
    candidate: object, battery: tuple[str, ...], *, index: Optional[int] = None
) -> None:
    """Shared validation for :class:`TreatmentCandidate` instances."""

    where = (
        f"candidate[{index}]" if index is not None else "TreatmentCandidate"
    )
    if not isinstance(candidate, TreatmentCandidate):
        raise TreatmentDeclarationError(
            f"{where} must be a TreatmentCandidate, got {type(candidate).__name__}."
        )
    key = candidate.variant_key
    if not isinstance(key, str) or not key.strip():
        raise TreatmentDeclarationError(
            f"{where} variant_key must be a non-empty string."
        )
    if not _VARIANT_KEY_RE.fullmatch(key):
        raise TreatmentDeclarationError(
            f"{where} variant_key {key!r} must be a lowercase slug "
            "(letters/digits, hyphens inside) with no spaces."
        )
    if candidate.mechanism not in _MECHANISM_KINDS:
        raise TreatmentDeclarationError(
            f"{where} ({key}): mechanism {candidate.mechanism!r} is not one of "
            f"{sorted(_MECHANISM_KINDS)}."
        )
    if candidate.group not in _CANDIDATE_GROUPS:
        raise TreatmentDeclarationError(
            f"{where} ({key}): group {candidate.group!r} is not one of "
            f"{sorted(_CANDIDATE_GROUPS)}."
        )
    applies_to = candidate.applies_to
    if not isinstance(applies_to, tuple) or not applies_to:
        raise TreatmentDeclarationError(
            f"{where} ({key}): applies_to must be a non-empty tuple of model types."
        )
    bad_or_dupe = [
        model
        for model in applies_to
        if not isinstance(model, str)
        or model not in battery
        or applies_to.count(model) > 1
    ]
    if bad_or_dupe:
        raise TreatmentDeclarationError(
            f"{where} ({key}): applies_to refers to models outside the battery "
            f"{battery} or repeats members; offenders: {bad_or_dupe}."
        )
    description = candidate.description
    if (
        not isinstance(description, str)
        or not description.strip()
        or not description.strip().endswith(".")
    ):
        raise TreatmentDeclarationError(
            f"{where} ({key}): description must be one non-empty declarative "
            "sentence ending with a period."
        )
    if description.count(".") != 1:
        raise TreatmentDeclarationError(
            f"{where} ({key}): description must be exactly one sentence "
            "(one terminating period)."
        )
    if candidate.mlflow_treatment_variant != key:
        raise TreatmentDeclarationError(
            f"{where}: mlflow_treatment_variant "
            f"{candidate.mlflow_treatment_variant!r} must equal the variant key "
            f"{key!r} (the tag value the S02 runner attaches must never drift)."
        )
    if not isinstance(candidate.order, int) or isinstance(candidate.order, bool):
        raise TreatmentDeclarationError(
            f"{where} ({key}): order must be an int."
        )
    if candidate.is_incumbent and candidate.group != "imputation":
        raise TreatmentDeclarationError(
            f"{where} ({key}): only the documented incumbent (imputation "
            "group) may be flagged is_incumbent."
        )


# --------------------------------------------------------------------------
# Declared candidates (order field fixes the deterministic list order)
# --------------------------------------------------------------------------


def _build_declared_candidates() -> tuple[TreatmentCandidate, ...]:
    """Build the declared candidate list in ``order`` sequence."""

    return (
        TreatmentCandidate(
            variant_key="yeo-johnson-power-transform",
            mechanism="pipeline-variant",
            group="transform",
            applies_to=LINEAR_FAMILY_SCOPE,
            description=(
                "Add a Yeo-Johnson power transform on the skewed numeric "
                "columns inside the preprocessing pipeline, fit on training "
                "rows only."
            ),
            mlflow_treatment_variant="yeo-johnson-power-transform",
            order=1,
        ),
        TreatmentCandidate(
            variant_key="box-cox-power-transform",
            mechanism="pipeline-variant",
            group="transform",
            applies_to=LINEAR_FAMILY_SCOPE,
            description=(
                "Add a Box-Cox power transform restricted to the strictly "
                "positive numeric columns inside the preprocessing pipeline, "
                "fit on training rows only."
            ),
            mlflow_treatment_variant="box-cox-power-transform",
            order=2,
        ),
        TreatmentCandidate(
            variant_key="median-imputer-incumbent",
            mechanism="pipeline-variant",
            group="imputation",
            applies_to=BATTERY_MEMBERS,
            description=(
                "Keep the committed median imputation policy as the incumbent "
                "baseline; it is documented here for ladder honesty and is "
                "not re-measured by the gate."
            ),
            mlflow_treatment_variant="median-imputer-incumbent",
            order=3,
            is_incumbent=True,
        ),
        TreatmentCandidate(
            variant_key="knn-imputer",
            mechanism="pipeline-variant",
            group="imputation",
            applies_to=LINEAR_FAMILY_SCOPE,
            description=(
                "Replace the median imputer with a K-nearest-neighbours "
                "imputer inside the preprocessing pipeline, fit on training "
                "rows only."
            ),
            mlflow_treatment_variant="knn-imputer",
            order=4,
        ),
        TreatmentCandidate(
            variant_key="missingness-flag-columns",
            mechanism="pipeline-variant",
            group="imputation",
            applies_to=LINEAR_FAMILY_SCOPE,
            description=(
                "Add one binary missing-indicator column per feature with any "
                "observed missing value before the committed imputation, so "
                "the model can see the missingness itself."
            ),
            mlflow_treatment_variant="missingness-flag-columns",
            order=5,
        ),
        TreatmentCandidate(
            variant_key="one-hot-drop-first",
            mechanism="pipeline-variant",
            group="encoding",
            applies_to=LINEAR_FAMILY_SCOPE,
            description=(
                "Replace the committed one-hot keep-all encoding with "
                "drop-first encoding to test the classic dummy-trap "
                "alternative against the already-regularised linear family."
            ),
            mlflow_treatment_variant="one-hot-drop-first",
            order=6,
        ),
        TreatmentCandidate(
            variant_key="class-weight-balanced",
            mechanism="estimator-param",
            group="imbalance",
            applies_to=(
                "logistic-regression-l2",
                "logistic-regression-l1",
                "logistic-regression-elasticnet",
            ),
            description=(
                "Pin class_weight='balanced' on the logistic-regression "
                "variants instead of the tuned class-weight policy."
            ),
            mlflow_treatment_variant="class-weight-balanced",
            order=7,
        ),
        TreatmentCandidate(
            variant_key="random-undersampling",
            mechanism="sampling",
            group="imbalance",
            applies_to=LINEAR_FAMILY_SCOPE,
            description=(
                "Apply random undersampling of the majority class inside the "
                "training fold only, after the split and before the fit."
            ),
            mlflow_treatment_variant="random-undersampling",
            order=8,
        ),
        TreatmentCandidate(
            variant_key="random-oversampling",
            mechanism="sampling",
            group="imbalance",
            applies_to=LINEAR_FAMILY_SCOPE,
            description=(
                "Apply random oversampling of the minority class inside the "
                "training fold only, after the split and before the fit."
            ),
            mlflow_treatment_variant="random-oversampling",
            order=9,
        ),
        TreatmentCandidate(
            variant_key="smote-oversampling",
            mechanism="sampling",
            group="imbalance",
            applies_to=LINEAR_FAMILY_SCOPE,
            description=(
                "Apply SMOTE synthetic minority oversampling inside the "
                "training fold only, after the split and before the fit."
            ),
            mlflow_treatment_variant="smote-oversampling",
            order=10,
        ),
        TreatmentCandidate(
            variant_key="adasyn-oversampling",
            mechanism="sampling",
            group="imbalance",
            applies_to=LINEAR_FAMILY_SCOPE,
            description=(
                "Apply ADASYN adaptive synthetic oversampling inside the "
                "training fold only, after the split and before the fit."
            ),
            mlflow_treatment_variant="adasyn-oversampling",
            order=11,
        ),
    )


# --------------------------------------------------------------------------
# Class-imbalance ladder (part D)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LadderRung:
    """One ordered rung of the pre-declared class-imbalance ladder.

    ``rung 0`` carries ``candidate=None`` and names the incumbent
    no-resampling pipeline, so part D can show rung 0 honestly (documented,
    not re-measured). Every later rung references a declared imbalance
    candidate by key.
    """

    rung: int
    label: str
    variant_key: Optional[str] = None


def _build_imbalance_ladder() -> tuple[LadderRung, ...]:
    """Build the ladder: rung 0 (none/incumbent) through ADASYN."""

    return (
        LadderRung(
            rung=0,
            label=(
                "none -- incumbent pipeline with its tuned class-weight "
                "policy (documented, not re-measured)"
            ),
        ),
        LadderRung(rung=1, label="class weights pinned", variant_key="class-weight-balanced"),
        LadderRung(rung=2, label="random undersampling", variant_key="random-undersampling"),
        LadderRung(rung=3, label="random oversampling", variant_key="random-oversampling"),
        LadderRung(rung=4, label="SMOTE", variant_key="smote-oversampling"),
        LadderRung(rung=5, label="ADASYN", variant_key="adasyn-oversampling"),
    )


# --------------------------------------------------------------------------
# Module guard
# --------------------------------------------------------------------------


def validate_candidates(
    candidates: tuple[TreatmentCandidate, ...],
    battery: tuple[str, ...],
) -> None:
    """Validate an ordered candidate list; raise :class:`TreatmentDeclarationError`.

    Checks every candidate individually (slug key, allowed mechanism/group,
    battery-only applies_to, one-sentence description, matched MLflow tag,
    incumbent consistency) plus list-level order uniqueness and contiguity.
    """

    if not isinstance(candidates, tuple) or not candidates:
        raise TreatmentDeclarationError(
            "declared candidates must be a non-empty tuple of TreatmentCandidate."
        )
    for index, candidate in enumerate(candidates):
        _check_candidate(candidate, battery, index=index)
    orders = [candidate.order for candidate in candidates]
    if sorted(orders) != list(range(1, len(candidates) + 1)):
        raise TreatmentDeclarationError(
            f"candidate order values {sorted(orders)} must be the unique "
            "integers 1..N in the declared list."
        )


def validate_ladder(
    ladder: tuple[LadderRung, ...],
    candidates: tuple[TreatmentCandidate, ...],
) -> None:
    """Validate a ladder against a candidate list; raise on any drift.

    Checks contiguous rungs starting at 0, exactly one incumbent-former
    rung 0 with no candidate, and that every later rung references an
    existing imbalance-group candidate at most once.
    """

    if not isinstance(ladder, tuple) or not ladder:
        raise TreatmentDeclarationError(
            "the imbalance ladder must be a non-empty tuple of LadderRung."
        )
    by_key = {candidate.variant_key: candidate for candidate in candidates}
    used: set[str] = set()
    for position, rung in enumerate(ladder):
        if not isinstance(rung, LadderRung):
            raise TreatmentDeclarationError(
                f"ladder[{position}] must be a LadderRung, got "
                f"{type(rung).__name__}."
            )
        if rung.rung != position:
            raise TreatmentDeclarationError(
                f"ladder rungs must be contiguous starting at 0; position "
                f"{position} carries rung {rung.rung}."
            )
        if rung.rung == 0:
            if rung.variant_key is not None:
                raise TreatmentDeclarationError(
                    "ladder rung 0 must name the incumbent (no candidate key)."
                )
            continue
        if rung.variant_key is None:
            raise TreatmentDeclarationError(
                f"ladder rung {rung.rung} must reference a declared candidate."
            )
        candidate = by_key.get(rung.variant_key)
        if candidate is None:
            raise TreatmentDeclarationError(
                f"ladder rung {rung.rung} references unknown candidate "
                f"{rung.variant_key!r}."
            )
        if candidate.group != "imbalance":
            raise TreatmentDeclarationError(
                f"ladder rung {rung.rung} references {rung.variant_key!r}, "
                "which is not an imbalance candidate."
            )
        if rung.variant_key in used:
            raise TreatmentDeclarationError(
                f"ladder rung {rung.rung} repeats candidate "
                f"{rung.variant_key!r}; each rung binds one distinct rung."
            )
        used.add(rung.variant_key)


def validate_all() -> None:
    """Run the full import-time guard over the declared module constants."""

    validate_candidates(DECLARED_CANDIDATES, BATTERY_MEMBERS)
    validate_ladder(IMBALANCE_LADDER, DECLARED_CANDIDATES)


# --------------------------------------------------------------------------
# Pre-declared severity gate constants (part F -- fixed before any number)
# --------------------------------------------------------------------------

#: Minimum improvement of a candidate's repeated-CV ROC-AUC delta over the
#: incumbent before the gate even looks at the standing rule (M006: +0.01).
GATE_MIN_DELTA_ROC_AUC: Final[float] = 0.01

#: The comparison metric. Matches ``heart.eval.metrics.PRIMARY_METRIC``
#: (decision D012: roc_auc is the primary metric); declared here as a copy
#: so this module stays numpy/sklearn-free.
GATE_METRIC: Final[str] = "roc_auc"

#: The winner-standing rule, as one precise sentence: a candidate is accepted
#: only if the treated model's standing versus the baseline and the runner-up
#: on the leaderboard does not worsen.
GATE_WINNER_STANDING_RULE: Final[str] = (
    "the treated model's standing versus the baseline and the runner-up on "
    "the leaderboard does not worsen"
)

#: Significance-layer alpha (paired tests), mirroring
#: ``heart.eval.significance.DEFAULT_ALPHA``; declared as a copy so this
#: module stays numpy-free. Tests assert the two never disagree.
PAIRED_SIGNIFICANCE_ALPHA: Final[float] = 0.05

#: MLflow tag that identifies a candidate measurement run (D024 extension).
MLFLOW_TAG_TREATMENT_VARIANT: Final[str] = "treatment_variant"

#: MLflow ``run_kind`` value used for gate-candidate measurement runs.
MLFLOW_RUN_KIND_TREATMENT_CANDIDATE: Final[str] = "treatment-candidate"

_DECLARED_CANDIDATES: Final[tuple[TreatmentCandidate, ...]] = (
    _build_declared_candidates()
)
_IMBALANCE_LADDER: Final[tuple[LadderRung, ...]] = _build_imbalance_ladder()

# Import-time guard: a malformed declaration above fails the import loudly
# (registry-spec style), before any consumer can render or measure half of
# a broken plan.
validate_candidates(_DECLARED_CANDIDATES, BATTERY_MEMBERS)
validate_ladder(_IMBALANCE_LADDER, _DECLARED_CANDIDATES)

#: The complete declared treatment-candidate register (D032): the S01 report
#: renders parts D/F strictly from these constants and the S02 gate runner
#: imports this list instead of re-declaring anything.
DECLARED_CANDIDATES: Final[tuple[TreatmentCandidate, ...]] = _DECLARED_CANDIDATES

#: Deterministic variant-key view of :data:`DECLARED_CANDIDATES`.
DECLARED_CANDIDATE_KEYS: Final[tuple[str, ...]] = tuple(
    candidate.variant_key for candidate in DECLARED_CANDIDATES
)

#: The one documented incumbent (median imputation policy) -- rung 0 honesty.
INCUMBENT_CANDIDATE: Final[TreatmentCandidate] = next(
    candidate
    for candidate in DECLARED_CANDIDATES
    if candidate.is_incumbent
)

#: The ordered class-imbalance ladder for report part D (D032).
IMBALANCE_LADDER: Final[tuple[LadderRung, ...]] = _IMBALANCE_LADDER


def gate_text() -> str:
    """Return the exact pre-declared severity-gate verdict sentence.

    Parameterized only by the two threshold constants
    (:data:`GATE_MIN_DELTA_ROC_AUC`, :data:`PAIRED_SIGNIFICANCE_ALPHA`);
    the winner-standing rule and metric come from their own constants. The
    S02 verdict must match this sentence's logic, and report part F renders
    it verbatim -- one source, never drift (D032).
    """

    delta = f"+{GATE_MIN_DELTA_ROC_AUC:.4f}"
    alpha = f"{PAIRED_SIGNIFICANCE_ALPHA:g}"
    return (
        "Severity gate, pre-declared before any candidate number existed: a "
        "candidate treatment is accepted only if (a) its "
        f"delta on repeated-CV {GATE_METRIC} against the incumbent is at "
        f"least {delta}, (b) the paired significance layer reads that delta "
        f"as significant at alpha={alpha}, and (c) "
        f"{GATE_WINNER_STANDING_RULE}; every candidate that fails any of the "
        "three keeps the incumbent pipeline and is recorded as rejected."
    )
