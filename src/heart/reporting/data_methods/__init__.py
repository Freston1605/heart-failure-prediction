"""Pre-split EDA data methods (M005).

This package owns the milestone's headline deliverable: a pre-split
exploratory-data-analysis engine (:mod:`heart.reporting.data_methods.eda`)
whose four stage functions run entirely before any train/test split, plus the
provenance-locked report machinery (S01/T02: :mod:`...provenance`,
:mod:`...report`, ``python -m heart.reporting.data_methods``) that renders
``reports/data-methods.md`` part A from the engine's outputs.

Every table and figure published in the report must trace to a generated
artifact under ``reports/data-methods/``; nothing here is ever hand-assembled.
"""

from heart.reporting.data_methods.eda import (
    EMPIRICAL_LOGIT_BINS,
    HIGH_CORRELATION_THRESHOLD,
    HIST_BINS,
    IQR_FENCE_MULTIPLIER,
    LOGIT_SMOOTHING,
    MAX_VIF,
    RARE_CATEGORY_FLOOR,
    DataMethodsError,
    EmptyDatasetError,
    MissingTargetColumnError,
    NonDataFrameError,
    StageComputationError,
    run_eda,
    run_stage1_hygiene,
    run_stage2_univariate,
    run_stage3_bivariate,
    run_stage4_assumptions,
)
from heart.reporting.data_methods.provenance import (
    MANIFEST_NAME,
    ProvenanceDigestMismatchError,
    ProvenanceManifestError,
    build_manifest,
    load_manifest,
    manifest_markdown,
    verify_manifest,
    write_manifest,
)
from heart.reporting.data_methods.report import (
    CHI_SQUARE_ALPHA,
    DATAMETHODS_DIR,
    REPORT_PATH,
    STAGE_ARTIFACTS,
    generate_report,
    render_report,
    write_stage_artifacts,
)

__all__ = [
    "RARE_CATEGORY_FLOOR",
    "HIGH_CORRELATION_THRESHOLD",
    "HIST_BINS",
    "EMPIRICAL_LOGIT_BINS",
    "LOGIT_SMOOTHING",
    "IQR_FENCE_MULTIPLIER",
    "MAX_VIF",
    "MANIFEST_NAME",
    "CHI_SQUARE_ALPHA",
    "DATAMETHODS_DIR",
    "REPORT_PATH",
    "STAGE_ARTIFACTS",
    "DataMethodsError",
    "NonDataFrameError",
    "EmptyDatasetError",
    "MissingTargetColumnError",
    "StageComputationError",
    "ProvenanceManifestError",
    "ProvenanceDigestMismatchError",
    "build_manifest",
    "load_manifest",
    "write_manifest",
    "verify_manifest",
    "manifest_markdown",
    "run_stage1_hygiene",
    "run_stage2_univariate",
    "run_stage3_bivariate",
    "run_stage4_assumptions",
    "run_eda",
    "write_stage_artifacts",
    "render_report",
    "generate_report",
]

# Re-exported so `python -m heart.reporting.data_methods --help` stays the
# single documented CLI seam (module instance, not a copied function).
from heart.reporting.data_methods.__main__ import main  # noqa: E402
