"""Portfolio reporting: reports derived from recorded run data (S03+).

The first report is the S03 leaderboard
(:mod:`heart.reporting.leaderboard`), which reads the final per-model MLflow
runs written by :mod:`heart.models.run_battery` and renders the ranked
comparison table. Reports in this package are *generated*, never hand-written:
regenerating them from the store is always the documented path.
"""

from heart.reporting.leaderboard import (
    DEFAULT_REPORT_PATH,
    DEFAULT_RUN_KIND,
    LEADERBOARD_FILENAME,
    REQUIRED_FLAT_KEYS,
    Leaderboard,
    LeaderboardConfig,
    LeaderboardConfigError,
    LeaderboardDataError,
    LeaderboardError,
    LeaderboardExperimentNotFoundError,
    LeaderboardReportError,
    LeaderboardRow,
    NoBenchmarkedModelsError,
    build_leaderboard,
    build_parser,
    describe_leaderboard,
    generate_leaderboard,
    main,
    render_leaderboard,
    select_leaderboard_rows,
    write_leaderboard,
)

__all__ = [
    "LEADERBOARD_FILENAME",
    "DEFAULT_REPORT_PATH",
    "DEFAULT_RUN_KIND",
    "REQUIRED_FLAT_KEYS",
    "LeaderboardError",
    "LeaderboardConfigError",
    "LeaderboardDataError",
    "LeaderboardExperimentNotFoundError",
    "LeaderboardReportError",
    "NoBenchmarkedModelsError",
    "LeaderboardConfig",
    "LeaderboardRow",
    "Leaderboard",
    "select_leaderboard_rows",
    "build_leaderboard",
    "render_leaderboard",
    "write_leaderboard",
    "generate_leaderboard",
    "describe_leaderboard",
    "build_parser",
    "main",
]
