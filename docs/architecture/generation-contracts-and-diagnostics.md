# Generation contracts and diagnostics

The public implementation includes source hydration, author-assertion
authority, scoped drafting, paragraph repair and execution traces. These
mechanisms make generation inspectable; they do not establish that generated
articles consistently preserve the author's voice or pass acceptance checks.
The project remains a work in progress.

## Execution boundaries

Whole-query Qwen retrieval is the default content-selection path.
`--paragraph-mode` and `--article-plan` are separate optional generation
features. The paragraph coordinator also supports source-text-bearing draft
packets when its draft-plan-context option is enabled. Their presence in the
code does not mean every generation call uses them.

```text
author intent + Qwen-selected content
                 │
       ┌─────────┴─────────────────────┐
       ▼                               ▼
optional article plan          optional paragraph mode
hydrate source packets         derive paragraph intents
record authority and hashes    retrieve evidence per intent
       │                               │
       └──── scoped packets, when ──────┤
             explicitly enabled         ▼
                              draft paragraph group
                                       │
                                       ▼
                              verify claims and scope
                                       │
                           bounded retrieval/repair
                                       │
                                       ▼
                              assemble and audit article
```

These paths retain the [Qwen/Jina model-role boundary](qwen-jina-reranker-tree.md).
They do not turn Jina into a corpus retriever or make an experimental selector
the default.

## Authority and provenance

### Author assertions are distinct from retrieved evidence

The article-plan contract separates `source_backed` and `author_asserted`
claims. Source-backed claims reference allowed documents or source spans.
Author-asserted claims reference IDs supplied by the author outside the model;
the optional `--author-assertions` input is a list of assertion IDs and text.
Unknown or missing assertion IDs fail validation.

This records who supplied a statement. It does not independently verify its
truth or let the model promote scaffold text into author authority. The claim
trace validator still requires a permitted causal edge for a claim marked as
causal, including an author-asserted one.

Implementation: [`build_article_plan`, `validate_claim_trace`](../../writing_rag/plan/article_plan.py).
Positive and negative fixtures: [article-plan tests](../../tests/test_article_plan.py).

### Retrieval results can be hydrated into source packets

The hydration layer resolves message IDs and recorded thread/time bounds to
chronological canonical messages. It records original source text, span IDs
and hashes. MDX and other scoped non-message documents can instead become
document-backed packets; they do not need invented chat-message IDs.

Unresolved message IDs, missing source scope and duplicate document IDs produce
errors. Style documents cannot enter the article plan as factual evidence.
Hash checks detect changed bytes, not whether the source statement is true.
Hydration restores the recorded source neighborhood, not a proven complete
causal episode.

Implementation: [`hydrate_retrieval_documents`](../../writing_rag/plan/source_hydration.py).
Fixtures: [source-hydration tests](../../tests/test_source_hydration.py).

### Drafting scope is narrower than the article's evidence union

An intent can use only its allowed documents, spans, edges and author
assertions. The same source may belong to several intents when independently
included in their evidence sets; sources do not require exclusive ownership.

The scoped-packet formatter can deliver actual source text to drafting rather
than IDs alone. Grouped formatting renders shared source text once while
retaining each intent's references and scope. Packet validation checks that
displayed evidence has not changed or escaped its allowlist. A planning
reference or high retrieval score is not itself proof of a claim.

Implementation: [`scope_article_plan`, `format_scoped_draft_packet`,
`format_scoped_draft_packets`](../../writing_rag/plan/article_plan.py).
The [article-plan study](../experiments/studies/study-001.md) records the
experimental status and reproduction limits; no variant is promoted as the
default on the basis of these interfaces alone.

## Composition and repair

### Paragraph intents represent reasoning flow

The paragraph contract carries a narrative role, reader question, intended
claim, permitted claim types, what the paragraph opens and closes,
dependencies, transition mode and required qualifiers. Transition modes include
answer, extend, redirect, contrast and return to narration.

These fields represent more than surface phrasing. They let the generator
express a question, develop a distinction and connect it to the next point.
Some missing fields receive inferred defaults, however, so the resulting
contract is not a learned or verified reconstruction of the author's style.
An intent dependency remains a composition relation, not factual causal proof.

Implementation: [`apply_intent_contract_defaults`, `format_intent_contract`,
`dependency_evidence_audit`](../../writing_rag/generate/paragraph_generation.py).
See also [conversation context and causal style](conversation-context-and-style.md).

### Repairs target failed paragraphs

Paragraph mode separates drafting, verification and repair. Support-related
failures can trigger a bounded evidence-retrieval retry. A local repair targets
failed paragraphs and verifies the replacements before merging them, leaving
unrelated paragraphs outside that repair. The coordinator defaults to one
retrieval retry and one repair opportunity per group, rather than an unbounded
rewrite loop.

Verification distinguishes source support, contradictions, insufficient
evidence, interpretations and non-claims. Qualifier checks aim to stop a
tentative statement from becoming certain. The article audit also compares
declared intents with generated text for order, references, key terms and
qualifiers. These checks can still reject an article or make mistakes; their
existence is not a writing-quality result.

Implementation: [paragraph coordinator](../../writing_rag/generate/paragraph_generation.py),
[draft/verify/repair stages](../../writing_rag/generate/paragraph_stages.py),
and [claim verification](../../writing_rag/verify/claims.py).
The [paragraph-generation study](../experiments/studies/study-022.md) does not
establish a repeatable accepted-article advantage.

## Diagnostics and reproducibility

### Retrieval traces help locate the failure

Candidate traces distinguish selected evidence from below-threshold results,
missing source spans, duplicates, malformed documents and usable but
unselected adjacent material. Generation metadata retains evidence ledgers,
verifier outputs, protocol diagnostics, local repairs and retrieval retries.

This supports a more specific investigation: did the corpus lack the material,
did retrieval rank it too low, did selection omit it, or did the verifier fail
to interpret the response? The trace does not automatically answer every one
of those questions, and its interpretation may remain pending. Private traces
are not published with the code.

Implementation: [evidence trace](../../writing_rag/verify/evidence_trace.py)
and [generation metadata](../../writing_rag/generate/generate.py).
Historical scope: [provenance diagnostics](../experiments/studies/study-035.md).

### Embedding identity includes more than dimensions

The retrieval session checks stored vector dimensions and Qwen normalization.
When metadata records them, it also checks the resolved model snapshot,
pooling configuration and tokenizer-file hashes. Query encoding follows the
stored policy: plain, model-card query prompt or custom query instruction.
Checks that depend on optional metadata are not a guarantee for an index that
never recorded those fields.

Reusable embedding sessions cache queries by text, policy and prompt hash.
Supporting an instructed query policy does not mean it won the experiment:
the [Q0/Q1 pilot](../experiments/studies/study-027.md) did not establish a Q1
improvement.

Implementation: [`EmbeddingSession`](../../writing_rag/retrieve/retrieve.py).
Fixtures: [query-policy tests](../../tests/test_qwen_query_encoding.py).

### Local generation measures the formatted prompt

`GenerationRuntime` loads one MLX model for reuse across outline, draft and
verification calls. Prompt-token counting uses the runtime tokenizer after
applying the chat template. The coordinator records prompt counts and checks
context limits; call results expose attempts, runtime failures and fallback
use rather than treating fallback text as an ordinary successful model call.

Implementation: [generation runtime](../../writing_rag/generate/generate.py)
and [generation-call boundary](../../writing_rag/generate/generation_calls.py).
This describes execution behavior, not a newly measured speedup or model-quality
benchmark. The [synthetic demos](../../README.md#try-the-synthetic-demo) do not
run the semantic generation models.

## Optional reasoning-chain exploration

The ingestion pipeline has an opt-in `--with-reasoning` stage. It constructs
reasoning points, connector records and candidate chains, and writes enriched
retrieval views separately from the core content/style documents. It is not
part of the default build.

The scoring module compares candidate narrative links with corpus-derived
calibration. Its scores are risk proxies, not entailment probabilities or
proof that one event caused another. Retaining this implementation does not
promote its chain construction or make it a validated causal-style model.

Implementation: [pipeline switch](../../scripts/run_pipeline.py),
[reasoning-chain builder](../../writing_rag/ingest/build_reasoning_chains.py)
and [causal scoring](../../writing_rag/ingest/causal_probability.py).

## Validation scope

The linked code and fixtures document implemented contracts. They do not
replace the [experiment decisions](experiment-ledger.md), demonstrate universal
article acceptance or make private-corpus experiments publicly reproducible.
The [current problem brief](article-generation-problem-brief.md) describes the
remaining generation and narrative limitations.
