from __future__ import annotations

import ast
from pathlib import Path

import article_plan
import generate as generate_shim
import paragraph_generation as paragraph_generation_shim
import pipeline_common
import pytest
import retrieve as retrieve_shim
import sys
import writing_rag.generate.generate as house_generate
import writing_rag.generate.paragraph_generation as house_paragraph_generation
import writing_rag.ingest.build_markdown_content as house_build_markdown_content
import writing_rag.ingest.causal_probability as house_causal_probability
import writing_rag.plan.article_plan as house_article_plan
import writing_rag.runtime.pipeline_common as house_pipeline_common
import writing_rag.verify.claims as house_claims

import build_markdown_content as build_markdown_shim
import causal_probability as causal_probability_shim
import jina_text_matching_mlx as jina_matching_shim
import jina_text_selection as jina_selection_shim
import writing_rag.retrieve  # noqa: F401

house_retrieve = sys.modules["writing_rag.retrieve.retrieve"]

from writing_rag import generate, ingest, plan, retrieve as retrieve_pkg, runtime, select, verify

ROOT = Path(__file__).resolve().parents[1]
WRITING_RAG = ROOT / "writing_rag"

HOUSE_PACKAGES: dict[str, dict[str, object]] = {
    "plan": {
        "dir": WRITING_RAG / "plan",
        "shims": (
            ROOT / "scripts/article_plan.py",
            ROOT / "scripts/source_hydration.py",
            ROOT / "scripts/narrative_reconstruction.py",
        ),
        "modules": {
            "article_plan.py",
            "axis0_validation.py",
            "composition_plan.py",
            "composition_projection.py",
            "composition_scheduler.py",
            "composition_state.py",
            "composition_unit_loop.py",
            "g0_freeze.py",
            "load_bearing_compiler.py",
            "obligation_set.py",
            "oflat_compiler.py",
            "source_hydration.py",
            "satisfiability.py",
            "shape_beat_compiler.py",
            "narrative_reconstruction.py",
            "unit_verification.py",
        },
    },
    "runtime": {
        "dir": WRITING_RAG / "runtime",
        "shims": (
            ROOT / "scripts/pipeline_common.py",
            ROOT / "scripts/model_paths.py",
            ROOT / "scripts/model_tokenizers.py",
        ),
        "modules": {
            "pipeline_common.py",
            "model_paths.py",
            "model_tokenizers.py",
        },
    },
    "retrieve": {
        "dir": WRITING_RAG / "retrieve",
        "shims": (
            ROOT / "scripts/embed_chunks.py",
            ROOT / "scripts/retrieve.py",
            ROOT / "scripts/facet_retrieve.py",
            ROOT / "scripts/graph_retrieve.py",
        ),
        "modules": {
            "embed_chunks.py",
            "retrieve.py",
            "facet_retrieve.py",
            "graph_retrieve.py",
        },
    },
    "select": {
        "dir": WRITING_RAG / "select",
        "shims": (
            ROOT / "scripts/jina_text_selection.py",
            ROOT / "scripts/jina_text_matching_mlx.py",
        ),
        "modules": {
            "jina_text_selection.py",
            "jina_text_matching_mlx.py",
        },
    },
    "generate": {
        "dir": WRITING_RAG / "generate",
        "shims": (
            ROOT / "scripts/generation_calls.py",
            ROOT / "scripts/paragraph_stages.py",
            ROOT / "scripts/paragraph_generation.py",
            ROOT / "scripts/generate.py",
        ),
        "modules": {
            "generation_calls.py",
            "paragraph_stages.py",
            "paragraph_generation.py",
            "generate.py",
        },
    },
    "verify": {
        "dir": WRITING_RAG / "verify",
        "shims": (ROOT / "scripts/evidence_trace.py",),
        "modules": {
            "claims.py",
            "evidence_trace.py",
        },
    },
    "ingest": {
        "dir": WRITING_RAG / "ingest",
        "shims": (
            ROOT / "scripts/extract_whisper.py",
            ROOT / "scripts/create_canonical_messages.py",
            ROOT / "scripts/create_candidate_windows.py",
            ROOT / "scripts/chunk_messages.py",
            ROOT / "scripts/build_retrieval_documents.py",
            ROOT / "scripts/build_chroma_documents.py",
            ROOT / "scripts/merge_messages.py",
            ROOT / "scripts/build_markdown_content.py",
            ROOT / "scripts/causal_probability.py",
            ROOT / "scripts/build_reasoning_chains.py",
        ),
        "modules": {
            "extract_whisper.py",
            "create_canonical_messages.py",
            "create_candidate_windows.py",
            "chunk_messages.py",
            "build_retrieval_documents.py",
            "build_chroma_documents.py",
            "merge_messages.py",
            "build_markdown_content.py",
            "causal_probability.py",
            "build_reasoning_chains.py",
        },
    },
}

REEXPORT_ONLY_PACKAGES = tuple(
    name for name in ("contracts",)
    if name not in HOUSE_PACKAGES
)

PLAN_DRAFT_EXPORTS = (
    "format_scoped_draft_packet",
    "format_scoped_draft_packets",
    "validate_scoped_draft_packet",
)


def test_positive_plan_reexports_same_objects_as_scripts() -> None:
    assert plan.build_article_plan is article_plan.build_article_plan
    assert plan.plan_sha256 is article_plan.plan_sha256
    assert plan.scope_article_plan is article_plan.scope_article_plan
    assert plan.build_article_plan is house_article_plan.build_article_plan


def test_positive_runtime_reexports_same_objects_as_scripts() -> None:
    assert runtime.read_json is pipeline_common.read_json
    assert runtime.read_json is house_pipeline_common.read_json
    assert pipeline_common.read_json is house_pipeline_common.read_json


def test_positive_select_reexports_same_objects_as_scripts() -> None:
    assert select.JinaTextMatchingMLX is jina_matching_shim.JinaTextMatchingMLX
    assert select.select_from_qwen_pool is jina_selection_shim.select_from_qwen_pool


def test_positive_generate_reexports_same_objects_as_scripts() -> None:
    assert generate.GenerationRuntime is generate_shim.GenerationRuntime
    assert generate.GenerationRuntime is house_generate.GenerationRuntime
    assert generate.generate_paragraph_article is paragraph_generation_shim.generate_paragraph_article
    assert (
        generate.generate_paragraph_article
        is house_paragraph_generation.generate_paragraph_article
    )


def test_positive_verify_reexports_same_objects_as_scripts() -> None:
    assert verify.claim_units is paragraph_generation_shim.claim_units
    assert verify.claim_units is house_claims.claim_units
    assert verify.parse_claim_verification is house_claims.parse_claim_verification


def test_positive_ingest_reexports_same_objects_as_scripts() -> None:
    assert ingest.narrative_roles is build_markdown_shim.narrative_roles
    assert ingest.narrative_roles is house_build_markdown_content.narrative_roles
    assert ingest.build_edge_calibration is causal_probability_shim.build_edge_calibration
    assert ingest.build_edge_calibration is house_causal_probability.build_edge_calibration


def test_positive_retrieve_reexports_same_objects_as_scripts() -> None:
    assert retrieve_pkg.retrieve is retrieve_shim.retrieve
    assert retrieve_pkg.retrieve is house_retrieve.retrieve
    assert retrieve_shim.retrieve is house_retrieve.retrieve


def test_positive_plan_exports_draft_packet_symbols() -> None:
    for name in PLAN_DRAFT_EXPORTS:
        assert name in plan.__all__
        assert hasattr(plan, name)


def test_positive_article_plan_has_episode_rank_helpers() -> None:
    assert hasattr(article_plan, "_episode_candidate_rank")
    assert hasattr(article_plan, "_episode_c3_neighbor_candidate_rank")
    assert article_plan._episode_candidate_rank is house_article_plan._episode_candidate_rank


def test_positive_select_does_not_reexport_retrieve_symbols() -> None:
    assert "retrieve" not in select.__all__
    assert "EmbeddingSession" not in select.__all__
    assert "QwenEmbeddingBackend" not in select.__all__


def test_positive_retrieve_does_not_reexport_jina_selectors() -> None:
    assert "select_from_qwen_pool" not in retrieve_pkg.__all__
    assert "JinaTextMatchingMLX" not in retrieve_pkg.__all__
    assert "validate_jina_contract" not in retrieve_pkg.__all__


@pytest.mark.parametrize("package_name", list(HOUSE_PACKAGES))
def test_positive_house_package_contains_implementation_modules(package_name: str) -> None:
    spec = HOUSE_PACKAGES[package_name]
    package_dir = spec["dir"]
    implementation_modules = {
        path.name
        for path in package_dir.glob("*.py")  # type: ignore[union-attr]
        if path.name != "__init__.py"
    }
    assert implementation_modules == spec["modules"]


@pytest.mark.parametrize("subpackage", REEXPORT_ONLY_PACKAGES)
def test_negative_non_house_subpackages_have_no_implementation_modules(subpackage: str) -> None:
    package_dir = WRITING_RAG / subpackage
    py_files = [
        path
        for path in package_dir.glob("*.py")
        if path.name != "__init__.py"
    ]
    assert py_files == []


def _implementation_nodes(path: Path) -> list[ast.AST]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]


@pytest.mark.parametrize(
    "shim_path",
    [shim for spec in HOUSE_PACKAGES.values() for shim in spec["shims"]],  # type: ignore[misc]
)
def test_negative_house_shims_have_no_implementation_bodies(shim_path: Path) -> None:
    assert _implementation_nodes(shim_path) == []


def test_negative_plan_house_has_no_jina_imports() -> None:
    for path in (WRITING_RAG / "plan").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "jina" not in text.lower()
        assert "writing_rag.select" not in text


def test_negative_retrieve_house_has_no_jina_imports() -> None:
    for path in (WRITING_RAG / "retrieve").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "writing_rag.select" not in text
        assert "from jina" not in text.lower()
        assert "import jina" not in text.lower()


def test_negative_verify_house_does_not_import_generate() -> None:
    for path in (WRITING_RAG / "verify").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "writing_rag.generate" not in text
        assert "paragraph_generation" not in text
