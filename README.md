# Diorama Cognition

A reconsolidation-based memory architecture for embodied LLM agents.

Combining the Embodied Cognition Platform (Phase 1/2) with insights from Kurzgesagt's memory model — where every recall rewrites the trace.

## Project Structure

```
├── PLAN.md              — Master plan (read this first)
├── README.md            — This file
│
├── src/
│   ├── core/            — Shared modules
│   │   ├── instrumentation.py  — TurnLogger, JSONL schema, chemistry defaults
│   │   ├── body_state.py       — 7 NTs + 6 body vars, decay, interactions, gates
│   │   └── diorama.py          — (NEW) Diorama Store: pattern memory with reconsolidation
│   ├── phases/          — Experiment runners
│   │   ├── phase1_baseline.py      — Single-model control
│   │   ├── phase2_architecture.py  — Multi-node with static memory
│   │   ├── phase3_diorama.py       — Multi-node with diorama memory
│   │   ├── phase4_sleep.py         — Consolidation cycles
│   │   └── phase5_experiments.py   — Systematic A/B runner
│   └── analysis/        — Post-hoc analysis tools
│       ├── consistency_score.py    — Chemistry-narrative alignment metric
│       └── report.py               — Cross-phase comparison reports
│
├── data/
│   ├── experiments/      — JSONL logs per run
│   ├── comparisons/      — A/B analysis reports
│   └── lineages/         — Memory lineage traces (Phase 3+)
│
├── prompts/
│   ├── test_set.txt      — 100 test prompts
│   └── nodes.py          — System prompts for each processing node
│
├── modelfiles/           — 75 Ollama Modelfile variants (15 models × 5 tiers)
├── scripts/              — Runner scripts
├── phase1-evaluation-report.md  — Old Phase 1 results (reference)
└── phase1_prompt.txt            — Old Phase 1 spec (reference)
```

## Quick Start

```bash
source .venv/bin/activate
python3 -m src.phases.phase1_baseline --turns 5 --log data/experiments/baseline_5.jsonl
python3 -m src.analysis.consistency_score data/experiments/baseline_5.jsonl
```

See PLAN.md for the full architecture, phased experiment plan, and DeepSeek Pro integration strategy.
