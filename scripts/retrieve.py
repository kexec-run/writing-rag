"""Compatibility doorbell for scripts/retrieve.py."""

from __future__ import annotations

import sys

import writing_rag.retrieve  # noqa: F401 - load house package and submodule

_impl = sys.modules["writing_rag.retrieve.retrieve"]

_name: str
for _name in dir(_impl):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_impl, _name)

del _name, _impl

if __name__ == "__main__":
    sys.modules["writing_rag.retrieve.retrieve"].main()
