# Proposed Shape–Beat article composition

Status: proposed architecture. This document describes a research direction, not the canonical generation path.

## Intent and evidence

The author supplies what the article should say and how it should unfold. The corpus supplies evidence for source-backed claims and relations. The writing-memory system supplies style examples from the author's own messages.

```text
AUTHOR INTENT                         SOURCE EVIDENCE
facts, emphasis, desired flow         messages and documents
          │                                   │
          └──────────────┬────────────────────┘
                         ▼
               typed composition plan
                         │
             check requested shape is feasible
                         │
                         ▼
           retrieve and hydrate source spans
                         │
                         ▼
        generate one Shape or Beat-sized unit
                         │
             verify claims and relations
                         │
                         ▼
              compose verified units
                         ▲
                         │
             self-authored style examples
```

## Proposed boundaries

- A scaffold expresses author intent; it is not retrieval evidence.
- Qwen remains the corpus retriever.
- Jina may be evaluated only in its documented auxiliary roles over a supplied Qwen pool.
- Hydrated messages and exact source spans remain the evidence substrate.
- Explicit causal edges require evidence for the relation, not merely evidence for both endpoints.
- Inferred edges remain visibly uncertain and cannot silently become factual claims.
- Style selection cannot change the evidence set or factual authority.

## Research sequence

The historical work first tested schema and obligation checks, then bounded Shape/Beat execution on two pilot cases. Those pilots are not confirmation and did not authorize a production change. Any continuation must preserve frozen inputs, separate article acceptance from obligation coverage, and keep positive and negative checks executable.

See the [Shape–Beat study summary](../experiments/studies/study-033.md) and the [complete decision catalog](../experiments/README.md). IDs in the study link are stable within this release catalog; later releases may add records without changing the meaning of this pilot.
