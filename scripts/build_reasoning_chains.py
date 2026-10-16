"""Compatibility doorbell for scripts/build_reasoning_chains.py."""

from __future__ import annotations

import writing_rag.ingest.build_reasoning_chains as _impl

_name: str
for _name in dir(_impl):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_impl, _name)

del _name, _impl

if __name__ == "__main__":
    import writing_rag.ingest.build_reasoning_chains as _main

    _main.main()
