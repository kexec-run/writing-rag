"""Compatibility doorbell for scripts/source_hydration.py."""

from __future__ import annotations

import writing_rag.plan.source_hydration as _impl

_name: str
for _name in dir(_impl):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_impl, _name)

del _name, _impl
