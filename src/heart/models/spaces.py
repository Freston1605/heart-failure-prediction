"""Declarative Optuna search spaces for the classical battery (S03).

This module owns **what** each model's hyperparameters can be. It is purposely
Optuna-free at import time: search spaces are plain, validated data structures
(:class:`ParamSpec` + :class:`SearchSpace`), and Optuna is only touched by
:func:`sample_params`, which speaks to the *Optuna Trial protocol*
(``suggest_float`` / ``suggest_int`` / ``suggest_categorical``) without
importing Optuna. Keeping the spaces declarative means:

* the registry (:mod:`heart.models.registry`) can describe the battery without
  requiring Optuna to be installed;
* a trial object can be swapped for a deterministic stand-in in tests or
  smoke mode;
* the *keys* of a space are testable metadata: every spec pins its tuned
  hyperparameter names, and ``tests/test_registry.py`` asserts the sampled
  keys and the declared hyperparameters are identical.

Each named model family gets one builder function (for example
:func:`logistic_regression_l2_space`) so the registry composes the battery from
named, documented spaces rather than inline dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Named exceptions
# ---------------------------------------------------------------------------


class SearchSpaceError(Exception):
    """Base class for every search-space definition or sampling failure."""


class InvalidParameterSpecError(SearchSpaceError):
    """A :class:`ParamSpec` violates the declarative value rules."""


class UnknownParameterError(SearchSpaceError):
    """A requested parameter name is not part of the space."""


class DuplicateParameterError(SearchSpaceError):
    """Two parameters in one space share a name."""


# ---------------------------------------------------------------------------
# Supported parameter kinds
# ---------------------------------------------------------------------------

#: A continuous parameter sampled uniformly (or log-uniformly) in [low, high].
FLOAT_KIND: str = "float"

#: An integer parameter sampled uniformly in [low, high] (inclusive).
INT_KIND: str = "int"

#: A categorical parameter sampled from an explicit choice tuple.
CATEGORICAL_KIND: str = "categorical"

#: Every kind :class:`ParamSpec` accepts.
PARAM_KINDS: tuple[str, ...] = (FLOAT_KIND, INT_KIND, CATEGORICAL_KIND)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    )


# ---------------------------------------------------------------------------
# Parameter and space value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    """One declarable hyperparameter and its allowed region.

    Parameters
    ----------
    name:
        Stable key; must match the estimator constructor keyword.
    kind:
        One of :data:`FLOAT_KIND`, :data:`INT_KIND`, :data:`CATEGORICAL_KIND`.
    low, high:
        Inclusive bounds for ``float``/``int`` kinds (required there).
    step:
        Optional sampling step for ``float``/``int``. ``None`` means
        continuous (float) or unit step (int).
    log:
        ``True`` samples uniformly in log space (``float``/``int`` only).
    choices:
        The permitted values for ``CATEGORICAL_KIND`` (required there).
    default:
        The deterministic fallback used by :func:`SearchSpace.default_params`
        and smoke configurations. May be ``None`` where that is itself a
        meaningful category (for example ``class_weight=None``); a ``None``
        default is only valid for a categorical parameter that includes
        ``None`` among its choices.
    """

    name: str
    kind: str
    low: float | None = None
    high: float | None = None
    step: float | None = None
    log: bool = False
    choices: tuple[object, ...] = field(default_factory=tuple)
    default: object = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise InvalidParameterSpecError(
                f"Parameter name must be a non-empty string, got {self.name!r}."
            )
        if self.kind not in PARAM_KINDS:
            raise InvalidParameterSpecError(
                f"Parameter {self.name!r} has unknown kind {self.kind!r}; "
                f"expected one of {list(PARAM_KINDS)}."
            )
        if self.kind == CATEGORICAL_KIND:
            if not self.choices:
                raise InvalidParameterSpecError(
                    f"Categorical parameter {self.name!r} must declare at least "
                    "one choice."
                )
            if self.default is not None and self.default not in self.choices:
                raise InvalidParameterSpecError(
                    f"Default {self.default!r} for parameter {self.name!r} is "
                    f"not among its choices {list(self.choices)}."
                )
            return

        # float / int bounds
        if not _is_number(self.low) or not _is_number(self.high):
            raise InvalidParameterSpecError(
                f"{self.kind} parameter {self.name!r} needs numeric low and "
                "high bounds."
            )
        if float(self.low) >= float(self.high):
            raise InvalidParameterSpecError(
                f"Parameter {self.name!r} must satisfy low < high, got "
                f"low={self.low!r}, high={self.high!r}."
            )
        if self.step is not None:
            if not _is_number(self.step) or float(self.step) <= 0:
                raise InvalidParameterSpecError(
                    f"Parameter {self.name!r} has invalid step {self.step!r}; "
                    "step must be a positive number."
                )
            if self.kind == INT_KIND and int(float(self.step)) != float(self.step):
                raise InvalidParameterSpecError(
                    f"Integer parameter {self.name!r} must have an integer "
                    f"step, got {self.step!r}."
                )
        if self.kind == INT_KIND and self.log and int(self.low) < 1:
            raise InvalidParameterSpecError(
                f"Log-scaled integer parameter {self.name!r} needs low >= 1, "
                f"got {self.low!r}."
            )
        if self.default is not None:
            if self.kind == INT_KIND and (
                not _is_number(self.default)
                or int(self.default) != float(self.default)
                or not float(self.low) <= float(self.default) <= float(self.high)
            ):
                raise InvalidParameterSpecError(
                    f"Default {self.default!r} for int parameter {self.name!r} "
                    f"is outside [{self.low}, {self.high}] or not integral."
                )
            if self.kind == FLOAT_KIND and (
                not _is_number(self.default)
                or not float(self.low) <= float(self.default) <= float(self.high)
            ):
                raise InvalidParameterSpecError(
                    f"Default {self.default!r} for float parameter {self.name!r} "
                    f"is outside [{self.low}, {self.high}]."
                )

    def to_dict(self) -> dict[str, object]:
        """Serialise the spec (used by registry reports)."""
        return {
            "name": self.name,
            "kind": self.kind,
            "low": self.low,
            "high": self.high,
            "step": self.step,
            "log": self.log,
            "choices": list(self.choices),
            "default": self.default,
        }


@dataclass(frozen=True)
class SearchSpace:
    """An ordered, name-unique collection of :class:`ParamSpec` entries.

    The ``keys`` property is the *declared hyperparameter set* that registry
    tests assert against the estimator's tunable parameters.
    """

    params: tuple[ParamSpec, ...]

    def __post_init__(self) -> None:
        names = [spec.name for spec in self.params]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise DuplicateParameterError(
                f"Search space declares duplicate parameter name(s) "
                f"{duplicates}; parameter names must be unique."
            )

    # -- introspection -----------------------------------------------------

    @property
    def keys(self) -> tuple[str, ...]:
        """The declared parameter names, in declaration order."""
        return tuple(spec.name for spec in self.params)

    def __len__(self) -> int:
        return len(self.params)

    def __iter__(self):
        return iter(self.params)

    def get(self, name: str) -> ParamSpec:
        """Return the :class:`ParamSpec` for ``name`` or raise loudly."""
        for spec in self.params:
            if spec.name == name:
                return spec
        raise UnknownParameterError(
            f"Parameter {name!r} is not part of this search space; declared "
            f"parameters are {list(self.keys)}."
        )

    def default_params(self) -> dict[str, object]:
        """Return every parameter at its declared default value."""
        return {spec.name: spec.default for spec in self.params}

    def to_dict(self) -> dict[str, object]:
        return {"params": [spec.to_dict() for spec in self.params]}

    # -- sampling ----------------------------------------------------------

    def sample(self, trial: object) -> dict[str, object]:
        """Sample one full parameter dict through an Optuna-style trial.

        ``trial`` must expose ``suggest_float(name, low, high, *, step, log)``,
        ``suggest_int(name, low, high, *, step, log)`` and
        ``suggest_categorical(name, choices)`` — exactly the Optuna
        :class:`~optuna.trial.Trial` surface. No Optuna import is required.
        """
        if trial is None:
            raise SearchSpaceError(
                "sample() needs a trial object implementing the Optuna "
                "suggest protocol; got None."
            )
        params: dict[str, object] = {}
        for spec in self.params:
            if spec.kind == CATEGORICAL_KIND:
                params[spec.name] = trial.suggest_categorical(
                    spec.name, list(spec.choices)
                )
                continue
            float_step = float(spec.step) if spec.step is not None else None
            if spec.kind == INT_KIND:
                int_step = int(float_step) if float_step is not None else None
                kwargs: dict[str, object] = {"log": spec.log}
                if int_step is not None:
                    kwargs["step"] = int_step
                params[spec.name] = trial.suggest_int(
                    spec.name,
                    int(spec.low),
                    int(spec.high),
                    **kwargs,  # type: ignore[arg-type]
                )
            else:
                float_kwargs: dict[str, object] = {"log": spec.log}
                if float_step is not None:
                    float_kwargs["step"] = float_step
                params[spec.name] = trial.suggest_float(
                    spec.name,
                    float(spec.low),
                    float(spec.high),
                    **float_kwargs,  # type: ignore[arg-type]
                )
        return params

    def sample_defaults(self) -> dict[str, object]:
        """Alias of :func:`default_params` for call sites that sample."""
        return self.default_params()


# ---------------------------------------------------------------------------
# The named per-model search spaces
# ---------------------------------------------------------------------------


def _space(*specs: ParamSpec) -> SearchSpace:
    """Build a :class:`SearchSpace` from :class:`ParamSpec` entries."""
    return SearchSpace(params=tuple(specs))


def logistic_regression_l2_space() -> SearchSpace:
    """Ridge-regularised logistic regression (``solver='lbfgs'``)."""
    return _space(
        ParamSpec(
            name="C",
            kind=FLOAT_KIND,
            low=1e-4,
            high=1e2,
            log=True,
            default=1.0,
        ),
        ParamSpec(
            name="class_weight",
            kind=CATEGORICAL_KIND,
            choices=(None, "balanced"),
            default=None,
        ),
    )


def logistic_regression_l1_space() -> SearchSpace:
    """Lasso-regularised logistic regression (``solver='liblinear'``)."""
    return _space(
        ParamSpec(
            name="C",
            kind=FLOAT_KIND,
            low=1e-4,
            high=1e2,
            log=True,
            default=1.0,
        ),
        ParamSpec(
            name="class_weight",
            kind=CATEGORICAL_KIND,
            choices=(None, "balanced"),
            default=None,
        ),
    )


def logistic_regression_elasticnet_space() -> SearchSpace:
    """Elastic-net logistic regression (``solver='saga'``)."""
    return _space(
        ParamSpec(
            name="C",
            kind=FLOAT_KIND,
            low=1e-4,
            high=1e2,
            log=True,
            default=1.0,
        ),
        ParamSpec(
            name="l1_ratio",
            kind=FLOAT_KIND,
            low=0.0,
            high=1.0,
            default=0.5,
        ),
        ParamSpec(
            name="class_weight",
            kind=CATEGORICAL_KIND,
            choices=(None, "balanced"),
            default=None,
        ),
    )


def naive_bayes_space() -> SearchSpace:
    """Gaussian Naive Bayes; only the variance-smoothing floor is tuned."""
    return _space(
        ParamSpec(
            name="var_smoothing",
            kind=FLOAT_KIND,
            low=1e-12,
            high=1e-1,
            log=True,
            default=1e-9,
        ),
    )


def lda_space() -> SearchSpace:
    """Linear discriminant analysis (``solver`` + optional shrinkage)."""
    return _space(
        ParamSpec(
            name="solver",
            kind=CATEGORICAL_KIND,
            choices=("svd", "lsqr"),
            default="svd",
        ),
        ParamSpec(
            name="shrinkage",
            kind=FLOAT_KIND,
            low=0.0,
            high=0.9,
            default=0.1,
        ),
    )


def qda_space() -> SearchSpace:
    """Quadratic discriminant analysis (regularisation on the covariance)."""
    return _space(
        ParamSpec(
            name="reg_param",
            kind=FLOAT_KIND,
            low=0.0,
            high=0.9,
            default=0.0,
        ),
    )


def knn_space() -> SearchSpace:
    """k-nearest neighbours: neighbourhood size, weighting, and metric power."""
    return _space(
        ParamSpec(
            name="n_neighbors",
            kind=INT_KIND,
            low=3,
            high=50,
            default=5,
        ),
        ParamSpec(
            name="weights",
            kind=CATEGORICAL_KIND,
            choices=("uniform", "distance"),
            default="uniform",
        ),
        ParamSpec(
            name="p",
            kind=INT_KIND,
            low=1,
            high=2,
            default=2,
        ),
    )


def svm_space() -> SearchSpace:
    """Support-vector classifier with probabilistic output and conditional gamma."""
    return _space(
        ParamSpec(
            name="C",
            kind=FLOAT_KIND,
            low=1e-3,
            high=1e3,
            log=True,
            default=1.0,
        ),
        ParamSpec(
            name="kernel",
            kind=CATEGORICAL_KIND,
            choices=("rbf", "linear"),
            default="rbf",
        ),
        ParamSpec(
            name="gamma",
            kind=FLOAT_KIND,
            low=1e-5,
            high=1e1,
            log=True,
            default=0.1,
        ),
        ParamSpec(
            name="class_weight",
            kind=CATEGORICAL_KIND,
            choices=(None, "balanced"),
            default=None,
        ),
    )


def random_forest_space() -> SearchSpace:
    """Random forest: tree size, split regularity, and feature sampling."""
    return _space(
        ParamSpec(
            name="n_estimators",
            kind=INT_KIND,
            low=50,
            high=500,
            default=200,
        ),
        ParamSpec(
            name="max_depth",
            kind=INT_KIND,
            low=2,
            high=32,
            default=16,
        ),
        ParamSpec(
            name="min_samples_split",
            kind=INT_KIND,
            low=2,
            high=20,
            default=2,
        ),
        ParamSpec(
            name="min_samples_leaf",
            kind=INT_KIND,
            low=1,
            high=10,
            default=1,
        ),
        ParamSpec(
            name="max_features",
            kind=CATEGORICAL_KIND,
            choices=("sqrt", "log2", None),
            default="sqrt",
        ),
        ParamSpec(
            name="class_weight",
            kind=CATEGORICAL_KIND,
            choices=(None, "balanced"),
            default=None,
        ),
    )


def xgboost_space() -> SearchSpace:
    """XGBoost: boosting rounds, tree depth, learning rate, and regularisation."""
    return _space(
        ParamSpec(
            name="n_estimators",
            kind=INT_KIND,
            low=50,
            high=500,
            default=200,
        ),
        ParamSpec(
            name="max_depth",
            kind=INT_KIND,
            low=2,
            high=10,
            default=6,
        ),
        ParamSpec(
            name="learning_rate",
            kind=FLOAT_KIND,
            low=1e-3,
            high=0.3,
            log=True,
            default=0.1,
        ),
        ParamSpec(
            name="subsample",
            kind=FLOAT_KIND,
            low=0.5,
            high=1.0,
            default=0.9,
        ),
        ParamSpec(
            name="colsample_bytree",
            kind=FLOAT_KIND,
            low=0.5,
            high=1.0,
            default=0.9,
        ),
        ParamSpec(
            name="reg_lambda",
            kind=FLOAT_KIND,
            low=1e-3,
            high=1e1,
            log=True,
            default=1.0,
        ),
    )


def sample_params(space: SearchSpace, trial: object) -> dict[str, object]:
    """Module-level helper delegating to :meth:`SearchSpace.sample`."""
    return space.sample(trial)


def default_params(space: SearchSpace) -> dict[str, object]:
    """Module-level helper delegating to :meth:`SearchSpace.default_params`."""
    return space.default_params()


def all_defined_spaces() -> dict[str, SearchSpace]:
    """Every named per-model space, keyed by its canonical name.

    This exists so the registry report and the battery smoke tests can list the
    declared spaces without importing the registry (avoids any import cycle).
    """
    return {
        "logistic-regression-l2": logistic_regression_l2_space(),
        "logistic-regression-l1": logistic_regression_l1_space(),
        "logistic-regression-elasticnet": logistic_regression_elasticnet_space(),
        "naive-bayes": naive_bayes_space(),
        "lda": lda_space(),
        "qda": qda_space(),
        "knn": knn_space(),
        "svm": svm_space(),
        "random-forest": random_forest_space(),
        "xgboost": xgboost_space(),
    }