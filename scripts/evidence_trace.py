"""Compatibility doorbell for scripts/evidence_trace.py."""

from __future__ import annotations

import writing_rag.verify.evidence_trace as _impl

_name: str
for _name in dir(_impl):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_impl, _name)

del _name, _impl
