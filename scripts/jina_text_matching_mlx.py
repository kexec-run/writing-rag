"""Compatibility doorbell for scripts/jina_text_matching_mlx.py."""

from __future__ import annotations

import writing_rag.select.jina_text_matching_mlx as _impl

_name: str
for _name in dir(_impl):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_impl, _name)

del _name, _impl
