"""Command-line entry point: ``python -m heart.tracking``.

Logs a smoke run through the fixed convention and prints the read-back. Using
``python -m heart.tracking`` (rather than ``python -m heart.tracking.run``)
avoids the runpy re-import warning, because the package init imports the
``run`` module normally rather than executing it as ``__main__``.
"""

from __future__ import annotations

import sys

from heart.tracking.run import main

if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    sys.exit(main())
