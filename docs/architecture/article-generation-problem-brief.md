# Article generation: current problem

Retrieval can find relevant passages while the generated article still fails. The difficult part is preserving the author's intended sequence of observations, explanations, challenges, revisions and decisions without inventing unsupported causal links.

## Current flow

```text
Qwen selects evidence chunks
        │
        ├── author supplies topic and sometimes a scaffold
        └── self-authored examples supply style
                    │
                    ▼
             article generation
                    │
                    ▼
       claim and format validation
```

The evidence chunks may not contain every transition in the original episode. A scaffold can request a narrative beat, but it does not prove that the beat is factually supported. The generator may omit the transition, write a weakly supported relation, or produce awkward prose; strict verification can then reject the article.

## What the experiments show

- Changing the selector can improve ranking while the article remains rejected.
- Whole-query and per-slot retrieval have not yet produced an accepted article across the bounded comparison.
- Source hydration and narrative-map components exist, but they are not a mandatory input to every canonical generation call.
- The Shape–Beat pilots showed that stronger structure alone does not resolve citation, qualifier and verifier interactions.
- The near-black two-pass run preserved a validated section fallback when the continuity rewrite failed.

These are observed limits in the tested targets, not proof that retrieval, scaffolds or narrative maps cannot help.

## Working design boundary

Author-supplied facts and desired flow describe intent. Retrieved source material constrains factual claims and causal relations. Style examples describe voice. The system should report whether the requested shape is representable from available evidence, then generate and verify under that boundary.

The [future Shape–Beat design](future-shape-beat-architecture.md) is a proposed research direction. It is not the default runtime. Study-by-study evidence and decisions are in the [experiment catalog](../experiments/README.md).
