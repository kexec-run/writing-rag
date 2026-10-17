# Instagram conversation episodes

Study ID: `study-007`

Disposition: `pending sanitization`
Decision: exploratory; no production ingestion change.

## Question

How should a chat be divided into overlapping, time-aware episodes that preserve reply and causal context?

## Method and controls

This study record consolidates a research line from the private experiment history. It covers **15 held artifact directories**; that count is an inventory count, not an execution count, a sample size or a count of independent cases. Individual payloads were not copied into the public release.

## Findings

The study family contains screening, structural, repeatability and gated variants. The aggregate decision record supports continued caution about overmerging and proxy validity; no human-labeled truth set is available.

## Evidence and limits

Several runner and graph dependencies are absent from the public seed. The held corpus and episode-level payloads remain private. See the [public decision ledger](../../architecture/experiment-ledger.md); exact source-line provenance is retained in the private inventory.

## Reproduction

Status: `blocked_pending_sanitization`.

Public code related to this study:

- [`experiments/active/instagram-conversation-episodes/validators.py`](../../../experiments/active/instagram-conversation-episodes/validators.py)
- [`experiments/active/instagram-conversation-episodes/decide_episode_experiment.py`](../../../experiments/active/instagram-conversation-episodes/decide_episode_experiment.py)

## Revisit condition

Restore a complete runner and synthetic structural fixtures, then report overlap and retrieval metrics with clear proxy-label provenance.
