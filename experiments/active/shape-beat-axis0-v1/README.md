# Shape–Beat Axis 0 pilot

This directory contains the public synthetic schema fixtures from the
Shape–Beat composition pilot and a CPU-only validator for two invented cases.
The historical pilot's generation runner and private oracle cases are not
included.

## Public availability

- Catalog disposition: `results-only` ([study-033](../../../docs/experiments/studies/study-033.md)).
- Present in this folder: `manifest.json`, `run_public_validation.py`, `fixtures/synthetic-minimal/composition-plan.json`, `fixtures/synthetic-minimal/g0-baseline.json`, `fixtures/synthetic-minimal/g0-pool.json`, and this README.
- Omitted: the historical generation runner, private oracle inputs, model cache, and generated pilot reports. The manifest records historical study settings; the public runner uses only its fixed synthetic cases.
- Reproduction limit: the public commands validate composition contracts and witness-ID availability. They do not reproduce the pilot, verify factual support, or establish generation quality.

## Public synthetic validation

Run these commands from the repository root. They use only repository fixtures,
standard-library Python, and the existing `writing_rag.plan` code. They do not
load models or access the network.

```bash
set -eu
OUT="$(mktemp -d)"
python3 experiments/active/shape-beat-axis0-v1/run_public_validation.py --case minimal --output "$OUT/minimal"
python3 experiments/active/shape-beat-axis0-v1/run_public_validation.py --case multi-beat --output "$OUT/multi-beat"
```

The minimal case compiles 2 Shapes and 2 Beats; the multi-beat case compiles 2
Shapes and 4 Beats. Each reports 22 passing checks, including all 15 registered
negative mutations. Outputs are `obligation-set.json`, `oflat.txt`,
`compiled-units.json`, `schedules.json`, and `validation-report.json`. The
runner refuses an existing output directory and fails before writing a report
if an input, hash, obligation, schedule, or negative check is invalid.
