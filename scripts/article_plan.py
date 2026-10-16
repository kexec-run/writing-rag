"""Compatibility doorbell for scripts/article_plan.py."""

from __future__ import annotations

import writing_rag.plan.article_plan as _impl

_name: str
for _name in dir(_impl):
    if _name.startswith("__"):
            continue
    globals()[_name] = getattr(_impl, _name)

del _name, _impl
