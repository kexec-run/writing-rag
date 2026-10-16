"""Schemas and fail-closed validators owned by boundary packages, not this package.

Plan and scoped-draft validators live in ``writing_rag.plan``. Claim and
transition verification live in ``writing_rag.verify``. Jina contract checks
live in ``writing_rag.select``. Pipeline structural checks remain CLI
doorbells under ``scripts/`` (for example ``validate_outputs.py``).

This package stays empty until a pure standalone schema module appears.
"""

from __future__ import annotations

__all__: list[str] = []
