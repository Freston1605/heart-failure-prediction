"""GPU / ROCm support for the neural slice (S05).

The neural battery is the only part of the portfolio that may run on a GPU
(decision D003). This package makes the GPU question answerable with evidence
rather than assumption:

* :mod:`heart.gpu.device_check` probes the runtime for the target GPU
  architecture (``gfx1201``, the Radeon RX 9070 XT) and the torch device torch
  actually resolved, and records the finding as a report artifact.

Importing this package never imports torch; torch is imported lazily inside the
probe so the module (and its tests) work on a CPU-only host.
"""

from __future__ import annotations

__all__: list[str] = []
