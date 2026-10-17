# Experiments

Experiment entrypoints and provenance live here, separate from the reusable
RAG runtime in `scripts/`. The public project overview and import guide are
linked from the [repository README](../README.md).

- `active/` contains experiments with an open question or a planned next run.
- `archived/` contains completed experiments with a frozen report and
  reproducibility metadata.

Shared runtime modules, canonical messages, model weights, retrieval indexes,
and generated personal-data artifacts remain in their existing locations.
Each experiment should record its entrypoint, inputs, model/runtime, output
report, production-impact boundary, and next action in a local manifest.

The public historical inventory is in the [experiment catalog](../docs/experiments/README.md).
It includes anonymous records for every reviewed experiment directory, excluded
Python support/experiment file, and held generated-artifact directory. It does
not include the private paths or payloads needed to reproduce corpus runs.
