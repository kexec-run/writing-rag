"""Compatibility doorbell for scripts/paragraph_stages.py."""

from __future__ import annotations

import writing_rag.generate.paragraph_stages as _impl

_name: str
for _name in dir(_impl):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_impl, _name)

del _name, _impl
