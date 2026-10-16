"""Compatibility doorbell for scripts/narrative_reconstruction.py."""

from __future__ import annotations

import writing_rag.plan.narrative_reconstruction as _impl

_name: str
for _name in dir(_impl):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_impl, _name)

del _name, _impl
