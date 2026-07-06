# Diorama Cognition

**A reconsolidation-based memory architecture for embodied LLM agents.**

> *"Every time you recall a memory, it changes."* — Kurzgesagt, "Your Brain is Weird Madness"

---

## Abstract

Diorama Cognition tests whether biological memory principles—specifically **reconsolidation**, where every recall rewrites the memory trace—can produce measurably different behavior in LLM agents compared to static append-only memory systems.

Unlike standard RAG or prompt-chaining, this architecture models:
- **Distributed cognition** (specialized sensory, emotional, episodic, and social processing nodes)
- **Simulated body chemistry** (7 neurotransmitters + 6 body states that modulate processing)
- **Reconsolidating memory** (memories pattern-complete, mutate on recall, and decay without reinforcement)
- **Memory lineage tracking** (audit trail of every transformation)

The hypothesis: biological-style memory produces **narrative drift**, **emotional coloring**, and **context-dependent forgetting** that static memory cannot replicate.

---

## The Core Question

> **Does a reconsolidation-based memory system—where memories are pattern-completed from cues, mutate on every recall, and decay without reinforcement—produce measurably different behavior in an embodied LLM agent than a static append-only memory?**

---

## Architecture

```
                    ┌─────────────────────┐
                    │   User / Environment │
                    │        Input         │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
   ┌──────────┐         ┌──────────┐         ┌──────────┐
   │ SENSORY  │         │EMOTIONAL │         │EPISODIC  │
   │  Node    │         │  Node    │         │  Node    │
   │(factual/ │         │(valence/ │         │(pattern  │
   │ pattern) │         │ arousal) │         │ memory)  │
   └────┬─────┘         └────┬─────┘         └────┬─────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                             ▼
                   ┌──────────────────┐
                   │     SOCIAL       │
                   │  PREDICTOR Node  │
                   │(Theory of Mind)  │
                   └────────┬─────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │         SYNTHESIZER         │
              │           ("I AM")          │
              │                             │
              │  Binds all node outputs →   │
              │  response + chemistry_adj   │
              └─────────────┬───────────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │   BodyState      │
                   │  (chemistry +    │
                   │   body vars)     │
                   └──────────────────┘
```

---

## Key Innovations

### 1. Reconsolidating Memory (The Diorama Store)

Biological memory doesn't work like a database. Every time you recall something:
- The memory becomes **labile** (temporarily unstable)
- **Current context seeps in** (mood, chemistry, environment)
- The memory **re-consolidates** in altered form
- **Decay** occurs if not reinforced

Our implementation:
- **Pattern completion**: Retrieve via cue similarity, not exact match
- **Read-is-write**: Retrieved memories blend with current chemistry state
- **Competitive activation**: Top-K memories compete; winner-take-all selection
- **Exponential decay**: Unreinforced traces fade below threshold
- **Lineage tracking**: Every mutation is logged for audit

### 2. Embodied Chemistry Model

7 neurotransmitters with realistic half-lives:
| Neurotransmitter | Half-life | Function |
|------------------|-----------|----------|
| Adrenaline | 2 min | Alertness, sensory gating |
| Noradrenaline | 30s | Arousal, focus |
| Dopamine | 5 min | Reward, motivation |
| Serotonin | 1 hr | Mood stabilization |
| Cortisol | 1 hr | Stress, resistance |
| Oxytocin | 10 min | Social bonding |
| Endorphins | 30 min | Pain relief |

6 body variables: `heart_rate`, `body_temperature`, `hunger`, `fatigue`, `pain`, `arousal`

Chemistry modulates **gating coefficients** per node—high adrenaline opens sensory/emotional gates; high serotonin dampens emotional volatility.

### 3. Multi-Node Processing

| Node | Model (Free Tier) | Profile | Role |
|------|-------------------|---------|------|
| Sensory | DeepSeek V4 Flash | focus (T=0.05) | Factual edge detection |
| Emotional | MiMo V2.5 | creative (T=0.80) | Valence/arousal assessment |
| Episodic | MiMo V2.5 | std (T=0.10) | Context retrieval |
| Social | MiMo V2.5 | creative (T=0.80) | Theory of mind prediction |
| Synthesizer | Big Pickle | bal (T=0.30) | Integration + chemistry adjustment |

---

## Experimental Phases

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 1** | Single-model baseline (control) | ✅ Complete |
| **Phase 1b** | Structured single-model (prompted internal analysis) | ✅ Complete |
| **Phase 2** | Multi-node with static memory | ✅ Complete |
| **Phase 3** | Multi-node with diorama memory | 🔧 In Progress |
| **Phase 4** | Sleep/consolidation cycles | 📋 Planned |
| **Phase 5** | Full A/B comparative experiments | 📋 Planned |

**Key Finding (Phase 1-2):** The multi-node architecture produces output distinguishable from single-model baselines. Consistency scores show chemistry-narrative alignment in ~73% of turns, with notable failure modes during state transitions.

---

## Repository Structure

```
diorama-cognition/
├── PLAN.md                      # Master architecture document (read this!)
├── README.md                    # This file
│
├── src/
│   ├── core/
│   │   ├── diorama.py          # Reconsolidating memory store
│   │   ├── body_state.py       # Chemistry + body state model
│   │   ├── instrumentation.py  # Logging + JSONL schema
│   │   └── api.py              # LLM API abstraction
│   │
│   ├── phases/
│   │   ├── phase1_baseline.py         # Single model control
│   │   ├── phase1b_structured_control.py  # Prompted internal analysis
│   │   ├── phase2_architecture.py     # Multi-node static memory
│   │   ├── phase2b_ablations.py       # Systematic ablation studies
│   │   ├── phase3_diorama.py          # Multi-node reconsolidating memory
│   │   ├── phase4_sleep.py            # Consolidation cycles
│   │   └── phase5_experiments.py      # A/B experiment runner
│   │
│   ├── analysis/
│   │   ├── consistency_score.py       # Chemistry-narrative alignment
│   │   └── report.py                  # Cross-phase comparisons
│   │
│   └── prompts/
│       ├── diverse_set.py      # 100 test prompts
│       └── nodes.py            # System prompts per node
│
├── data/
│   └── experiments/            # JSONL logs per run
│
├── modelfiles/                 # 75 Ollama Modelfile variants
└── prompts/                    # Test sets and node prompts
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- API keys for OpenCode/DeepSeek (stored in `.env`)
- Ollama (optional, for local model variants)

### Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Run baseline (single model control)
python3 -m src.phases.phase1_baseline \
    --turns 5 \
    --log data/experiments/baseline_test.jsonl

# Run multi-node architecture
python3 -m src.phases.phase2_architecture \
    --turns 5 \
    --log data/experiments/arch_test.jsonl

# Analyze consistency
python3 -m src.analysis.consistency_score \
    data/experiments/arch_test.jsonl
```

### Run Ablation Study

```bash
# Test specific architecture components
python3 -m src.phases.phase2b_ablations \
    --mode gating \
    --turns 100 \
    --log data/experiments/ablation_gating.jsonl

# Modes: baseline, random, gating, contra, full
```

---

## Key Results

### Phase 1-2 Findings

| Metric | Baseline | Multi-Node |
|--------|----------|------------|
| Chemistry-Narrative Alignment | N/A | 73% |
| Valid State Transitions | N/A | 68% |
| End-State Diversity | Low | High |
| Confabulation Rate | Baseline | Reduced |

**Critical insight:** The architecture works, but chemistry transitions (especially adrenaline spikes) need better gating logic. The "structured single model" baseline (Phase 1b) was essential—without it, we couldn't distinguish architecture effects from prompting effects.

### Ablation Studies

Systematic removal of components shows:
- **Gating modulation**: Accounts for ~15% of variance in emotional appropriateness
- **Chemistry interactions**: Critical for coherent multi-turn narratives
- **Node specialization**: Distinct node outputs vs. single-model prompted analysis

---

## Glossary

| Term | Definition |
|------|------------|
| **Assembly** | Synchronized firing pattern across nodes representing a moment |
| **Diorama Store** | The reconsolidating memory index (hippocampus analog) |
| **Gate Multiplier** | Chemistry-derived coefficient scaling node influence |
| **Lineage** | Audit trail of memory mutations across recalls |
| **Pattern Completion** | Retrieving memory from partial cues via similarity |
| **Reconsolidation** | Memory becoming labile and rewriting during recall |
| **Synaptic Field** | Weight matrix between memory features (Hebbian update) |

---

## Related Work

- **Kurzgesagt**: "Your Brain is Weird Madness" — core inspiration for memory model
- **Embodied Cognition Platform**: Predecessor project (Phase 1/2)
- **Active Inference / Free Energy Principle**: Theoretical grounding for predictive processing
- **Sparse Distributed Memory (Kanerva)**: Pattern completion mechanics

---

## Contributing

This is an active research project. Current priorities:
1. Complete Phase 3 (diorama memory integration)
2. Implement Phase 4 (sleep/consolidation)
3. Design Phase 5 (controlled comparative experiments)

See `PLAN.md` for full architectural specification and experimental roadmap.

---

## Citation

If you use this work, please cite:

```
@software{diorama_cognition_2025,
  author = {Gilliard, Raymond},
  title = {Diorama Cognition: A Reconsolidation-Based Memory Architecture},
  year = {2025},
  url = {https://github.com/RayGilliardGitHub/diorama-cognition}
}
```

---

## License

MIT License — See repository for details.

---

**Author**: Raymond Gilliard  
**Contact**: https://github.com/RayGilliardGitHub  
**Related**: [The Gradient Papers](https://github.com/RayGilliardGitHub/the-gradient-papers.git) — Thermodynamic AGI framework
