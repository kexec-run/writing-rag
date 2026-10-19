# Conversation context, style and causal flow

The Instagram experiments used an asymmetric representation: other speakers
provided context, while self-authored messages remained the style material.
This was implemented and exercised on the private corpus, not just proposed.
It has not been established as a reliable end-to-end causal-style preservation
method for generated articles.

## What the episode experiments used

```text
chronological conversation + self-authored anchor
                       │
                       ▼
       overlapping candidate episode windows
                       │
             ┌─────────┴──────────┐
             ▼                    ▼
     other-speaker turns     self-authored turns
     provisional roles      original message text
     short context excerpts self-only style view
             │                    │
             └─────────┬──────────┘
                       ▼
       context-to-self links + self-message edges
                       │
                       ▼
       structural checks and Qwen retrieval tests
```

The episode builder assigned provisional roles such as question, challenge,
clarification, correction, agreement, closure and contribution. Each context
record retained its source message IDs. The field named `summary` contained a
whitespace-normalized excerpt capped at 280 characters, not a model-generated
semantic summary of the conversation.

The style view joined the original self-authored message contents in order;
it did not paraphrase them or include the other speaker's words. Candidate
windows overlapped, and the builder retained alternative boundaries. Time,
lexical overlap, dialogue-role cues and causal-language markers contributed
to heuristic continuity scores. These links were not verified causal truth:
the export had no native reply-to metadata, and no human-labeled boundary set
was used.

## Evidence that this ran

The retained private confirmation artifact, `episode-results.json` from the
Instagram episode structural run, contains 60 primary episodes. An inspection
of that saved artifact found:

| Stored representation | Primary-episode records |
|---|---:|
| Other-speaker context roles and excerpts | 520 across 59 episodes |
| Context-to-self links | 386 across 54 episodes |
| Consecutive self-message edges | 376 across 60 episodes |
| Nonempty self-only style views | 60 of 60 |

Every style view's message IDs matched its episode's self-message IDs and
were disjoint from its context-message IDs. Every context excerpt had source
message IDs. The saved validation reported zero invalid episodes; its
contamination and boundary-perturbation fixtures passed. These are structural
checks, not independent labels of conversational meaning or writing quality.

The [episode study](../experiments/studies/study-007.md) also ran Qwen
context-to-response retrieval. The fixed query used raw other-speaker messages
from a method-independent neighborhood; candidate documents used the
preserved self-message text. The stored context excerpts were therefore
constructed and checked, but this was **not** an ablation proving that
summarizing context improves retrieval.

| Split | Queries | Baseline MRR | Episode MRR | Baseline R@10 | Episode R@10 |
|---|---:|---:|---:|---:|---:|
| Earlier screening | 34 | 0.628344 | 0.641140 | 0.911765 | 0.882353 |
| Later time-disjoint evaluation | 53 | 0.395403 | 0.491661 | 0.584906 | 0.698113 |

The later split had 60 structural anchors, but seven lacked nonempty context
in the fixed query window. Retrieval covered the remaining 53 common anchors.
The decision remained `structural_only`: the earlier recall regression blocked
promotion despite the later gain. No article-generation improvement was
established by this run.

The [Jina chain-selection follow-up](../experiments/studies/study-008.md)
also built episodes through the same episode builder before selecting among
local candidates. Qwen still performed retrieval. That follow-up remained
`structural_only` and was archived; it did not promote this construction into
production or authorize an article holdout.

These observations come from the retained implementation and saved private
artifacts, not a new model run. The corpus, excerpts and episode-level traces
remain private. The public episode folder retains contracts and validators,
not the complete builder and runnable corpus experiment.

## What the current pipeline preserves

The [current ingestion path](single-path.md) separates content and self-only
style documents. Its default content renderer groups raw context under
`Context:` and self text under `Self:`; it does not use the experimental
context-summary representation or display the original dialogue interleaving.
Message IDs and timestamps remain available in metadata.

Generation distinguishes factual support from style examples. Its default
style context is an aggregate profile of features such as sentence length,
questions, first-person use and qualification. Optional causal scaffolds and
article-plan narrative packets carry additional structure. They are not a
mandatory dialogue-to-style stage in every generation call.

## What remains unproven

Preserving self text makes the author's reasoning available; it does not by
itself demonstrate that an article preserves the same reasoning moves. The
remaining questions are whether context can be compressed without losing a
challenge or clarification, whether inferred response links are correct, and
whether generation retains the author's causal ordering and transitions
without importing unrelated factual claims from style examples.

Causal style means how the author expresses a supported relationship. It
cannot establish that a new factual relationship is true. Existing causal
checks address source support; they do not establish stylistic fidelity.
