"""Compatibility doorbell for scripts/facet_retrieve.py."""

from __future__ import annotations

import writing_rag.retrieve.facet_retrieve as _impl

_name: str
for _name in dir(_impl):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_impl, _name)

del _name, _impl
