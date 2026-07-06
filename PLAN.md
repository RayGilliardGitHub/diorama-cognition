# Diorama Cognition

**A reconsolidation-based memory architecture for embodied LLM agents.**

*Inspired by Kurzgesagt — "Your Brain is Weird Madness"*
*Built on the Embodied Cognition Platform (Phase 1/2 prototype)*

---

## Table of Contents

1. [The Big Idea](#1-the-big-idea)
2. [What the Video Contributes](#2-what-the-video-contributes)
3. [Architecture Overview](#3-architecture-overview)
4. [Phase 0 — Foundation](#4-phase-0--foundation)
5. [Phase 1 — Baseline (Single Model)](#5-phase-1--baseline-single-model)
6. [Phase 2 — Architecture Baseline (Multi-Node)](#6-phase-2--architecture-baseline-multi-node)
7. [Phase 3 — Diorama Memory](#7-phase-3--diorama-memory)
8. [Phase 4 — Sleep & Consolidation](#8-phase-4--sleep--consolidation)
9. [Phase 5 — Full Integration & Comparative Experiments](#9-phase-5--full-integration--comparative-experiments)
10. [DeepSeek Pro Role](#10-deepseek-pro-role)
11. [Data & Analysis Plan](#11-data--analysis-plan)
12. [Glossary](#12-glossary)

---

## 1. The Big Idea

The old project asked: *"Can a multi-node architecture (specialized models for sensory, emotional, episodic, social processing → synthesizer) produce output distinguishable from a single model alone?"*

The video adds a second question: *"What if the memory itself behaved like biological memory — where every recall rewrites the trace?"*

Combined, the question becomes:

> **Does a reconsolidation-based memory system — where memories are pattern-completed from cues, mutate on every recall, and decay without reinforcement — produce measurably different behavior in an embodied LLM agent than a static append-only memory?**

The hypothesis is that reconsolidation creates:
- **Narrative drift** — the agent's "life story" subtly shifts over time as context seeps into each recall
- **Emotional memory coloring** — high-arousal chemistry states increase plasticity, making emotionally charged memories change more
- **Forgetting as a feature** — unreinforced traces fade, preventing saturation and keeping the agent responsive to the present
- **Identity coherence** — the LINEAGE of each memory (tracking every rewrite) provides an audit trail that static memory cannot

---

## 2. What the Video Contributes

The Kurzgesagt video on memory describes a specific functional architecture. Here's what maps directly to the project:

| Video Concept | What It Means | Where It Goes |
|---|---|---|
| **Cortical columns** as basic processing units | Each node (sensory, emotional, episodic, social) functions as a specialized column | Already in Phase 2 — the 4 processing nodes |
| **Assembly** = synchronized firing across regions | A moment is represented as a distributed pattern across all active columns | This is the synthesizer's output + the chemistry vector at that turn |
| **Hippocampus** creates a blueprint | A separate index structure that maps cues → stored assemblies | NEW: the Diorama Store index (separate from the synaptic field) |
| **"Neurons that fire together wire together"** | Hebbian update — co-activated features strengthen their connection | NEW: the synaptic weight field between memory features |
| **Wax melting on recall** | Every retrieval is a reconsolidation — the memory becomes labile, context seeps in, then it re-hardens | NEW: read-is-write — memory output is different from storage |
| **Competition between assemblies** | Multiple candidate memories compete; one wins the attention spotlight | NEW: top-K retrieval with winner-take-all |
| **Novelty strengthens encoding** | High surprise → more resources allocated to memory formation | Already implicit in chemistry (adrenaline spikes) |
| **Emotion = plasticity** | Emotional arousal increases the "wax temperature" — memories change more | Already partially in body_state.py; needs to modulate diorama learning rate |
| **Sleep replays assemblies** | Offline replay strengthens patterns without new input | NEW: batch consolidation cycle between sessions |
| **Forgetting via decay** | Unreinforced synapses weaken; memories below threshold fade | NEW: exponential decay of memory strength |
| **Context seepage** | The retrieval context (mood, time, place) blends into the memory | NEW: during reconsolidation, current chemistry blends into the stored pattern |

The old project's episodic memory was a simple list of `(user_msg, response)` tuples, passing the last 2 turns to the Episodic Retriever. The video makes clear this misses the entire insight — memory isn't an append log, it's a *process* that changes what it touches.

---

## 3. Architecture Overview

```
                         ┌─────────────────────┐
                         │   User / Environment │
                         │        Input         │
                         └──────────┬──────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
         ▼                          ▼                          ▼
   ┌──────────┐              ┌──────────┐              ┌──────────┐
   │ SENSORY  │              │EMOTIONAL │              │EPISODIC  │
   │ Edge/    │              │Valuator  │              │Retriever │
   │ Pattern  │              │          │              │          │
   │Detector  │              │          │              │          │
   └────┬─────┘              └────┬─────┘              └────┬─────┘
        │                         │                         │
        └────────────────┬────────┼─────────────────────────┘
                         │        │
                         ▼        ▼
                   ┌──────────────────┐
                   │   SOCIAL         │
                   │   PREDICTOR      │
                   │  (Theory of Mind)│
                   └────────┬─────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │         SYNTHESIZER         │
              │         ("I AM")            │
              │                             │
              │  Output: response text +    │
              │  chemistry_adjustment +     │
              │  weighting                  │
              └──────────────┬──────────────┘
                             │
                             ▼
                    ┌──────────────────┐      ┌──────────────────┐
                    │  Body State      │◄────►│  Diorama Store   │
                    │  (Chemistry)     │      │  (Memory)        │
                    │                  │      │                  │
                    │  7 NTs + 6 body  │      │  Pattern store + │
                    │  vars. Decay,    │      │  index +         │
                    │  interactions.   │      │  reconsolidation │
                    │  wax_temp from   │      │  on every read.  │
                    │  arousal levels  │      │  Lineage tracked.│
                    └──────────────────┘      └──────────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Sleep /        │
                    │   Consolidation  │
                    │                  │
                    │  Batch replay +  │
                    │  strengthen +    │
                    │  prune decayed   │
                    └──────────────────┘
```

### Module Responsibilities

| Module | File | Role |
|---|---|---|
| `instrumentation.py` | `src/core/instrumentation.py` | TurnLogger, JSONL schema, chemistry defaults, node registry |
| `body_state.py` | `src/core/body_state.py` | 13-vector chemistry simulation, half-life decay, gate modulation |
| `diorama.py` | `src/core/diorama.py` | **NEW** — Diorama Store: pattern memory with reconsolidation |
| `phase1_baseline.py` | `src/phases/phase1_baseline.py` | Single-model control runs |
| `phase2_architecture.py` | `src/phases/phase2_architecture.py` | Multi-node with static memory (evolved from old phase2.py) |
| `phase3_diorama.py` | `src/phases/phase3_diorama.py` | Multi-node with diorama memory |
| `phase4_sleep.py` | `src/phases/phase4_sleep.py` | Consolidation cycle runner |
| `phase5_experiments.py` | `src/phases/phase5_experiments.py` | Systematic A/B runner |
| `consistency_score.py` | `src/analysis/consistency_score.py` | Chemistry-narrative alignment metric |
| `report.py` | `src/analysis/report.py` | Cross-phase comparison reports |

---

## 4. Phase 0 — Foundation

### Goal
Working API access, verified toolchain, known starting state.

### What Changed From the Old Project

The old project used **local ollama models** (llama3.2:1b, qwen2.5:7b, lfm2.5:8b).  
**This project uses the OpenCode Go API** (DeepSeek V4 Flash, MiMo-V2.5).

**Why the change:**
- No local model storage needed (no pulling 8GB+ models)
- Consistent API response quality vs. variable local inference
- Cost is negligible — ~$0.002 per 5-call turn, ~30,000 turns/month within the Go subscription
- Two models (Flash and MiMo) at identical pricing ($0.14/$0.28 per M tokens) let us test model effects without confounding by cost

**Node-to-model mapping:**

| Node | Model | Reason |
|------|-------|--------|
| Sensory | DeepSeek V4 Flash | Fast, analytical, factual |
| Emotional | MiMo-V2.5 | Different behavioral profile |
| Episodic | MiMo-V2.5 | Same as emotional — symmetry |
| Social | MiMo-V2.5 | Same as emotional — symmetry |
| Synthesizer | DeepSeek V4 Flash | Strong binding capability |

The entire architecture can run with a single model across all nodes, or with the split above. The PLAN.NODE_MODEL_MAP in `src.core.api` makes this trivially adjustable.

### Inventory (What We Have)

| Asset | Location | Status |
|---|---|---|
| 75 Modelfile variants (15 models × 5 tiers) | `modelfiles/*.txt` | Copied from backup |
| instrumentation.py (TurnLogger, schema, self-test) | `src/core/instrumentation.py` | Copied from old project |
| body_state.py (chemistry, gates, tick) | `src/core/body_state.py` | Copied from old project |
| consistency_score.py (analysis tool) | `src/analysis/consistency_score.py` | Copied from old project |
| 100 test prompts | `prompts/test_set.txt` | Copied from old project |
| phase1.py (old prototype) | `src/phases/phase1.py` | Reference only |
| phase2.py (evolved prototype) | `src/phases/phase2.py` | Reference only |

### Setup Steps

1. **Python environment**
   ```bash
   cd /home/rlg/diorama-cognition
   source .venv/bin/activate  # already done
   ```

2. **Set your OpenCode Go API key**
   ```bash
   export OPENCODE_GO_API_KEY="golive_xxxxx"
   ```
   Or pass `--api-key golive_xxxxx` to each phase runner.

3. **Verify API access**
   ```bash
   python3 -c "
   from src.core.api import set_api_key, opencode_chat
   set_api_key('golive_xxxxx')
   r = opencode_chat('deepseek-v4-flash', [
       {'role': 'user', 'content': 'Say hello in one word.'}
   ])
   print(f'API OK: {r}')
   "
   ```

4. **Run instrumentation self-test**
   ```bash
   python3 -m src.core.instrumentation
   ```
   Should produce 3-turn synthetic data + schema verification (all checks pass).

5. **Verify body_state.py standalone**
   ```bash
   python3 -c "from src.core.body_state import BodyState; bs = BodyState(); bs.tick(elapsed=60.0); print(bs.to_prompt_string())"
   ```

6. **Verify diorama.py self-test**
   ```bash
   python3 -m src.core.diorama
   ```

No ollama models need to be pulled. The 75 Modelfiles in `modelfiles/` are kept as reference documentation of the old project.

### DeepSeek Pro Role in Phase 0
**None.** This phase is entirely local tooling verification.

---

## 5. Phase 1 — Baseline (Single Model)

### Goal
Establish the "control" — what a single LLM (no nodes, no chemistry, no memory architecture) produces on the 100 test prompts.

### Design

```
User Prompt
    │
    ▼
┌─────────────────────┐
│ Single API call     │
│ DeepSeek V4 Flash   │
│ (or MiMo-V2.5)      │
│                     │
│ System:             │
│ "You are a helpful  │
│  assistant."        │
└──────────┬──────────┘
           │
           ▼
    Response text
    (logged via TurnLogger)
```

### Implementation

File: `src/phases/phase1_baseline.py`

- Takes `--model`, `--turns`, `--log`, `--api-key` arguments
- Each turn: one API call to OpenCode Go, one TurnLogger record
- Chemistry state is logged but never changes (static DEFAULT_CHEMISTRY)
- No nodes, no memory, no modulation
- Cost tracking: prints per-turn and total API cost

### Outputs

| File | Contents |
|---|---|
| `data/experiments/baseline_100.jsonl` | 100 turns, single model |
| `data/experiments/baseline_5x3.jsonl` | 5 test prompts × 3 repeats (for consistency check) |

### What We Measure

- **Latency** per turn (ms) — baseline cost
- **Response length** (chars/tokens)
- **Vocabulary diversity** (unique tokens across all responses)
- **Chemistry-consistency score** (should be ~0% — baseline model doesn't know about chemistry)

### Key Question Answered
> *"What does a single model produce when asked these 100 questions?"*

### DeepSeek Pro Role in Phase 1
**Evaluation only.** Run the same 100 prompts through deepseek-v4 (or Pro model) and compare:
- Is the local baseline distinguishable from the Pro baseline?
- This gives a "capability ceiling" for later comparisons.

---

## 6. Phase 2 — Architecture Baseline (Multi-Node)

### Goal
Replicate the old Phase 1/2 architecture in clean code: 4 specialized nodes + synthesizer + static memory. This is the "old system" comparison point.

### Design

```
User Prompt
    │
    ├──→ SENSORY (llama3.2:1b-focus)     ──→ output
    ├──→ EMOTIONAL (qwen2.5:7b-creative)  ──→ output
    ├──→ EPISODIC (lfm2.5:8b-std)         ──→ output   ← static history
    ├──→ SOCIAL (lfm2.5:8b-creative)      ──→ output
    │
    └──→ SYNTHESIZER (lfm2.5:8b-bal)      ──→ response + chem_adjust + weighting
              │
              ▼
         BodyState.tick()
         Chemistry updated
         History appended
```

### What's Different From Old Phase 2

1. **Cleaner module separation** — phase script imports from `src.core.*`
2. **CLI unification** — all switches consistent across phases
3. **New ablation flags inherited from old phase2.py:**
   - `--gate-modulation-off`
   - `--chemistry-injection-off`
   - `--hide-labels`
   - `--permute-variables`
   - `--disable` (per-node)
4. **Memory is still append-only** — list of `(user_msg, response)` tuples, last 2 turns to Episodic Retriever

### Outputs

| File | Contents |
|---|---|
| `data/experiments/arch_full_100.jsonl` | 100 turns, all nodes active |
| `data/experiments/arch_nosensory_100.jsonl` | 100 turns, sensory disabled |
| `data/experiments/arch_noemotional_100.jsonl` | 100 turns, emotional disabled |
| `data/experiments/arch_nosocial_100.jsonl` | 100 turns, social disabled |
| `data/experiments/arch_noepisodic_100.jsonl` | 100 turns, episodic disabled |
| `data/experiments/arch_boss_1.jsonl` | Single boss-context turn |
| `data/experiments/arch_friend_1.jsonl` | Single friend-context turn |

### What We Measure
- **Per-node latency** (sensory ~3s, emotional ~5s, episodic ~7s, social ~7s, synth ~11s)
- **Chemistry trajectory** — does the synthesizer produce plausible adjustments?
- **Social modulation** — boss vs friend output divergence (from old project: clearly distinguishable)
- **Ablation impact** — which node changes output the most when disabled?
- **Chemistry-consistency score** — how often does the response text match the chemistry values?

### Key Question Answered
> *"Does the multi-node architecture produce output distinguishable from the single-model baseline?"*

(Shadowing the old project's question — this establishes the comparison.)

### DeepSeek Pro Role in Phase 2
**Synthesizer swap test.** Run the same pipeline but replace the local synthesizer model with deepseek-v4 (via API). Compare:
- `lfm2.5:8b-bal` synthesizer vs `deepseek-v4` synthesizer
- Does the architecture add the same value regardless of synthesizer capability?

---

## 7. Phase 3 — Diorama Memory

### Goal
Replace the append-only history list with the Diorama Store — a pattern-completion memory system where every recall rewrites the trace.

### The Diorama Store (src/core/diorama.py)

This is the new module. Core abstractions:

```
DioramaStore:
  ├── pattern_dim: int            # dimensionality of the sparse pattern space
  ├── sparsity: float             # what fraction of dims are active per pattern
  │
  ├── dioramas: List[Diorama]     # all stored memories
  ├── weights: np.ndarray[N][N]   # synaptic field (Hebbian connections)
  ├── index: AnnoyIndex           # approximate nearest neighbor for cue retrieval
  │
  ├── wax_temp: float             # global plasticity (modulated by chemistry)
  ├── decay_rate: float           # forgetting — strength decay per tick
  │
  └── operations:
       ├── encode(pattern, context) → DioramaID
       │     Store a new pattern. Apply Hebbian update to weights.
       │     Check for near-duplicates first (novelty gate).
       │
       ├── retrieve(cue) → (Diorama, DioramaID)
       │     Pattern completion → competition → winner.
       │     RECONSOLIDATION: blend current context into pattern.
       │     Hebbian update with new pattern.
       │     Increment recall_count. Append to lineage.
       │     Return the (now-modified) memory.
       │
       ├── consolidate()
       │     Batch replay. For each strong diorama:
       │       replay pattern K times (Hebbian update).
       │       Increase strength.
       │
       ├── tick(dt)
       │     Apply decay to all diorama strengths.
       │     Prune dioramas below threshold.
       │     Decay weights toward 0.
       │
       └── set_wax_temp(chemistry)
             Derive wax_temp from the body_state chemistry vector.
             High adrenaline/cortisol → high plasticity.
             High serotonin → low plasticity (stability).
```

### Diorama Data Structure

```python
@dataclass
class Diorama:
    pattern: np.ndarray           # sparse binary vector — the "assembly"
    strength: float               # 0.0-1.0 — how consolidated
    
    # Metadata
    created_at: int               # turn number of first encoding
    recall_count: int             # how many times retrieved
    last_retrieved: int           # turn number of most recent recall
    
    # Reconsolidation audit trail
    lineage: List[LineageEntry]   # every rewrite recorded
    # LineageEntry = (turn, context_blend_weight, was_winner)

    # Associative tags (from the hippocampus index)
    tags: Set[str]                # "boss", "emotional", "crow_squirrel"
```

### Pattern Creation

The big design question: *what IS a pattern?* Options from most to least literal:

1. **Full-text embedding** — embed the synthesizer's output + node outputs + chemistry vector into a dense vector → binarize to sparse pattern
   - Pro: captures semantic content
   - Con: expensive, depends on an embedding model

2. **Node-output hash** — hash the concatenation of all node outputs into a fixed-dim binary vector
   - Pro: simple, deterministic, no external dependency
   - Con: hash collisions, no semantic similarity

3. **Composite pattern** — split the pattern space into regions:
   - `[0:1000]` = sensory output features
   - `[1000:2000]` = emotional output features
   - `[2000:3000]` = episodic output features
   - `[3000:4000]` = social output features
   - `[4000:5000]` = chemistry state
   - Each region encodes the output of that node as a sparse binary code
   - Pro: interpretable, modular, allows partial-cue retrieval
   - Con: requires a fixed encoding scheme per node

4. **DeepSeek embedding** — use DeepSeek Pro API to generate embeddings from the combined turn data
   - Pro: high-quality semantic patterns
   - Con: requires API, not reproducible offline

**Recommended approach for initial prototype (Phase 3.0):** Option 3 (composite pattern). It's interpretable, offline, and maps directly to the architecture. Option 4 can be added as a Phase 3.1 enhancement.

### Integration into the Turn Loop

```
For each turn:
  1. Process nodes (sensory, emotional, episodic, social) → node_outputs
  2. Encode node_outputs + chemistry into a composite pattern
  3. RETRIEVE from DioramaStore using this pattern as cue:
       ← (previous_episodic_context, enrichment)
     Side effect: the retrieved memory is now mutated
     (context seeps in, lineage updated, strength incremented)
  4. Pass node_outputs + retrieved_context to Synthesizer
  5. Synthesizer produces response + chemistry_adjustment
  6. ENCODE response + adjusted chemistry into new Diorama
  7. Apply chemistry_adjustment to BodyState
  8. DioramaStore.tick() — decay all unreferenced memories
```

### Reconsolidation Mechanics (The Critical Part)

```python
def retrieve(self, cue: np.ndarray) -> Diorama:
    # 1. Pattern completion via synaptic field
    completed = self._attractor_dynamics(cue)
    
    # 2. Competition — find nearest stored diorama
    winner = self._nearest(completed)
    
    # 3. RECONSOLIDATION
    #    Winner is now labile ("wax melts")
    current_context = self._get_context()  # current mood, time, recent tags
    
    #    Blend current context into the stored pattern
    blend_ratio = self.wax_temp * 0.1
    winner.pattern = self._blend(winner.pattern, current_context, blend_ratio)
    
    #    Re-apply Hebbian update (wax re-hardens in new shape)
    self.weights += self.learning_rate * np.outer(winner.pattern, winner.pattern)
    np.clip(self.weights, 0, 1, out=self.weights)
    
    #    Track the mutation
    winner.lineage.append(LineageEntry(
        turn=self.current_turn,
        blend=blend_ratio,
        wax_temp=self.wax_temp
    ))
    winner.recall_count += 1
    
    return winner
```

### Wax Temperature Modulation

The `wax_temp` parameter is how the chemistry system influences memory plasticity:

```python
def derive_wax_temp(chemistry: dict) -> float:
    """Map chemistry vector to global plasticity.
    
    High arousal (adrenaline + noradrenaline) → more plasticity (meltable)
    High serotonin + oxytocin → stability (hardened wax)
    High cortisol → moderate plasticity (stress = changeable)
    """
    arousal = chemistry["adrenaline"] + chemistry["noradrenaline"]
    stability = chemistry["serotonin"] + chemistry["oxytocin"]
    stress = chemistry["cortisol"]
    
    # Core plasticity from arousal, dampened by stability
    base = 0.2 + 0.5 * arousal - 0.3 * stability
    base = max(0.1, min(1.0, base))
    
    # Stress adds noise — memories become less reliable
    noise = 0.1 * stress
    
    return max(0.1, min(1.0, base + noise))
```

### Outputs

| File | Contents |
|---|---|
| `data/experiments/diorama_100.jsonl` | 100 turns with diorama memory |
| `data/experiments/diorama_nosleep_100.jsonl` | 100 turns, consolidation OFF |
| `data/experiments/diorama_higharousal_100.jsonl` | 100 turns, --adrenaline 0.8 |
| `data/experiments/diorama_lowplasticity_100.jsonl` | 100 turns, all gates locked |

### What We Measure
- **Drift rate** — how much does the same memory change across successive recalls?
- **Context bleed** — does a memory recalled under "boss" context show boss-like features when later recalled under "friend" context?
- **Forgetting curve** — how many turns until an unreinforced memory drops below threshold?
- **Chemistry-consistency score comparison** — does reconsolidation improve or degrade chemistry alignment?

### Key Question Answered
> *"Does a reconsolidation-based memory produce different behavior than a static append-only memory?"*

### DeepSeek Pro Role in Phase 3
**Pattern embedding service.** If we use option 4 (DeepSeek embeddings), the Pro model generates the semantic pattern vectors from node outputs. Otherwise, DeepSeek Pro can serve as the **blind drift detector** — given two response outputs from the same prompt at different turns, does the model rate them as coherent or drifted?

---

## 8. Phase 4 — Sleep & Consolidation

### Goal
Implement offline consolidation cycles — the "sleep" phase where memories are replayed, strengthened, and pruned without new input.

### Design

```
Between experiment sessions (or every N turns during a long run):

SLEEP_CYCLE():
  1. For each diorama with strength > threshold:
       replay pattern K times through Hebbian update
       strength += 0.1 (capped at 1.0)
  2. For each diorama with strength < 0.05:
       remove from index (forgotten)
  3. Merge near-duplicate dioramas (overlap > 0.8)
  4. Rebuild nearest-neighbor index
  5. Log consolidation report:
       - Number of memories consolidated
       - Number pruned
       - Number merged
       - Average strength before/after
```

### The Chemistry-Sleep Connection

From the video: *"Your hippocampus replays the assembly over and over, making it more solid"* and *"if you don't sleep enough, you literally forget more of your life."*

This maps to:
- **High cortisol (stress)** → poor sleep → less consolidation → more forgetting
- **High serotonin** → good sleep → better consolidation
- **High adrenaline** → fragmented sleep → selective consolidation of threat memories

In code:
```python
def sleep_quality(chemistry: dict) -> float:
    """0.0 (worst) to 1.0 (best) sleep quality derived from chemistry."""
    stress_penalty = chemistry["cortisol"] * 0.5
    arousal_penalty = chemistry["adrenaline"] * 0.3
    stability_bonus = chemistry["serotonin"] * 0.3
    
    quality = 0.5 + stability_bonus - stress_penalty - arousal_penalty
    return max(0.1, min(1.0, quality))

def sleep_cycle(store, chemistry):
    quality = sleep_quality(chemistry)
    replay_count = int(5 * quality)  # more replays with better sleep
    
    for diorama in store.dioramas:
        if diorama.strength > 0.2:
            for _ in range(replay_count):
                store.weights += 0.02 * np.outer(diorama.pattern, diorama.pattern)
            diorama.strength = min(1.0, diorama.strength + 0.1 * quality)
    
    # Worse sleep means more forgetting
    forget_threshold = 0.05 if quality > 0.5 else 0.15
    store.prune(threshold=forget_threshold)
```

### Outputs

| File | Contents |
|---|---|
| `data/experiments/sleep_5_cycles.jsonl` | 50 turns × 5 sleep cycles |
| `data/experiments/sleep_stress_5.jsonl` | Same but with high cortisol during sleep |
| `data/experiments/sleep_none.jsonl` | 50 turns, no consolidation (control) |

### Key Question Answered
> *"Does offline consolidation improve memory coherence and chemistry alignment compared to no consolidation?"*

### DeepSeek Pro Role in Phase 4
**None.** Sleep/consolidation is entirely local algorithmic work. DeepSeek could evaluate the output coherence before vs. after sleep cycles, but the consolidation itself is a local computation.

---

## 9. Phase 5 — Full Integration & Comparative Experiments

### Goal
All features active. Systematic comparison across all architectures and ablations. This is where we answer the project's core question.

### Experiment Matrix

| Run ID | Architecture | Memory | Chemistry | Sleep | Turns | Purpose |
|---|---|---|---|---|---|---|
| `E01` | Single model | None | Static | No | 100 | Ultimate baseline |
| `E02` | Multi-node | Append | Active | No | 100 | Old system baseline |
| `E03` | Multi-node | Append | Off | No | 100 | Is chemistry doing anything? |
| `E04` | Multi-node | Diorama | Active | No | 100 | Diorama vs append |
| `E05` | Multi-node | Diorama | Off | No | 100 | Diorama without chemistry |
| `E06` | Multi-node | Diorama | Active | Yes | 100 | Everything on |
| `E07` | Multi-node | Diorama | Active | Yes | 500 | Long-run drift test |
| `E08` | Multi-node | Append | Active | No | 500 | Long-run static control |
| `E09` | Single model (Pro) | None | Static | No | 100 | DeepSeek baseline |
| `E10` | Multi-node (Pro synth) | Diorama | Active | Yes | 100 | Pro synthesizer + diorama |

### Ablation Flags (all phases)

| Flag | Effect |
|---|---|
| `--disable sensory` | Remove sensory node |
| `--disable emotional` | Remove emotional node |
| `--disable episodic` | Remove episodic node |
| `--disable social` | Remove social node |
| `--gate-modulation-off` | All gates = 1.0 (ignore chemistry) |
| `--chemistry-injection-off` | Don't inject chemistry into prompts |
| `--hide-labels` | state_N instead of chemical names |
| `--permute-variables` | Shuffle which state_N maps to which chemical (blind test) |
| `--adrenaline FLOAT` | Override initial adrenaline |
| `--cortisol FLOAT` | Override initial cortisol |
| `--diorama-off` | Use append-only memory instead of diorama |
| `--sleep-off` | Disable offline consolidation |
| `--wax-temp FLOAT` | Override wax temperature (1.0 = max plasticity) |

### Evaluation Metrics

1. **Chemistry Consistency Score** (from old project) — how often does the model's narrative match actual chemistry values?
2. **Response Differentiation** — pairwise semantic similarity between responses to the same prompt under different configurations
3. **Social Modulation Score** — boss vs friend output divergence (embedding distance)
4. **Drift Rate** — how much does the response to prompt N change between turn 50 and turn 500?
5. **Memory Lineage Inspection** — for diorama runs, trace the lineage of a specific memory across rewrites
6. **Blind A/B Preference** — human (or DeepSeek) judge prefers configuration X over configuration Y for coherence, naturalness, emotional appropriateness

### Key Questions Answered

> 1. Does the full multi-node + diorama + chemistry system produce output distinguishable from a single model?
> 2. Does reconsolidation-based memory produce measurably different behavior from append-only memory?
> 3. Does the chemistry feedback loop actually influence the agent's responses over time?
> 4. Does offline consolidation improve or degrade response quality?
> 5. Which ablation has the largest impact on output quality?
> 6. Does a DeepSeek Pro synthesizer change the answers to any of the above?

---

## 10. DeepSeek Pro Role

**Updated context:** All models now come from the same OpenCode Go API.
DeepSeek V4 Pro is available at `$1.74/$3.48` per M tokens — ~12× the cost of Flash.

### Where Pro Adds Value

| Role | Phase | What It Does | Cost per 100 turns |
|---|---|---|---|
| **Synthesizer upgrade** | 2, 3, 5 | Use DeepSeek V4 Pro as "I AM" instead of Flash | ~$0.28 (vs $0.02 for Flash) |
| **All-node upgrade** | 2, 3, 5 | Run entire architecture on Pro | ~$1.40 (vs $0.20 for Flash/MiMo) |
| **Blind evaluator** | 5 | Judge A/B output pairs for coherence | ~$0.05 per 100 comparisons |
| **Pattern embedding** | 3 (opt) | Generate semantic embeddings for Diorama Store | Variable |

### The Tradeoff

Pro is 12× the cost per token but presumably more capable. The interesting question is: **does the architecture add value beyond the model capability?** If a Pro synthesizer with the 4-node architecture produces the same quality as a Pro baseline without the nodes, then the architecture isn't doing anything. If the architecture adds value *even with Flash*, that's a stronger result.

**My recommendation:** Start with Flash/MiMo (the Go subscription baseline). Run the full experiment matrix. If results are ambiguous (can't distinguish architecture from baseline), *then* try Pro for the synthesizer as a follow-up test. This keeps costs near-zero during development and reserves the $12/5hr budget for the experiments that need it.

---

## 11. Data & Analysis Plan

### Directory Structure

```
data/
├── experiments/          # JSONL output files
│   ├── baseline_100.jsonl
│   ├── arch_full_100.jsonl
│   ├── diorama_100.jsonl
│   ├── sleep_5_cycles.jsonl
│   ├── e01_baseline_100.jsonl
│   ├── e07_longrun_500.jsonl
│   └── ...
├── comparisons/          # A/B analysis reports
│   ├── e01_vs_e02.md
│   ├── e04_vs_e02.md
│   └── ...
├── lineages/             # Memory lineage traces (Phase 3+)
│   └── turn_47_lineage.json
└── reports/              # Aggregate reports
    ├── phase1_report.md
    ├── phase2_report.md
    └── final_comparison.md
```

### Analysis Pipeline

```
TurnLogger output (JSONL)
    │
    ▼
consistency_score.py ──→ Chemistry Consistency Score (per-term, aggregate)
    │
    ▼
report.py:
  ├── Latency statistics (mean, median, p95 per node)
  ├── Response length distribution
  ├── Chemistry trajectory plots (NTs over turns)
  ├── Social modulation distance (boss vs friend embeddings)
  ├── Drift measurement (early vs late same-prompt responses)
  └── Ablation impact ranking (which node matters most)
```

### Across-Phase Comparisons

```python
# Pseudo-code for comparison runner
comparisons = [
    ("E01 vs E02",  "single_model", "multi_node_append"),
    ("E02 vs E04",  "multi_node_append", "multi_node_diorama"),
    ("E04 vs E05",  "diorama_normal", "diorama_no_chem"),
    ("E04 vs E06",  "diorama_nosleep", "diorama_sleep"),
    ("E07 vs E08",  "diorama_500turns", "append_500turns"),
    ("E01 vs E09",  "local_baseline", "pro_baseline"),
]
for name, a_key, b_key in comparisons:
    report = compare_experiments(data[a_key], data[b_key])
    report.write(f"data/comparisons/{name}.md")
```

---

## 12. Glossary

| Term | Definition |
|---|---|
| **Assembly** | A distributed pattern of synchronized neural/simulated activity that constitutes a moment of experience |
| **Diorama** | A stored memory trace — the pattern + its metadata + its lineage of rewrites |
| **Reconsolidation** | The process by which a retrieved memory becomes labile, incorporates current context, and re-stabilizes |
| **Wax Temperature** | Global plasticity parameter — high = memories change more on recall; low = memories stay stable |
| **Gate Multiplier** | Chemistry-modulated weight per processing node (0.0-1.0) that controls how much of that node's signal reaches the synthesizer |
| **Synaptic Field** | The Hebbian weight matrix connecting all pattern dimensions — encodes learned associations |
| **Hippocampus Index** | Approximate nearest-neighbor index for cue-based memory retrieval |
| **Lineage** | The complete audit trail of every rewrite a diorama has undergone |
| **Consolidation** | Offline replay of memories (analogous to sleep) that strengthens patterns without new input |
| **Chemistry Vector** | 13-variable state (7 NTs + 6 body vars) that modulates plasticity, gating, and prompt injection |
| **Wax_temp** | Derived from chemistry; controls how much a memory changes during reconsolidation |

---

## Quick Start

```bash
# 1. Make sure the environment is active
cd /home/rlg/diorama-cognition
source .venv/bin/activate

# 2. Set your API key once
export OPENCODE_GO_API_KEY="golive_xxxxx"

# 3. Verify core modules
python3 -m src.core.instrumentation    # synthetic self-test
python3 -m src.core.diorama            # diorama store self-test

# 4. Phase 1 — Baseline (single model, 5 turns, ~$0.002)
python3 -m src.phases.phase1_baseline \
    --turns 5 \
    --log data/experiments/baseline_5.jsonl \
    --model deepseek-v4-flash

# 5. Phase 2 — Architecture baseline (4 nodes, 5 turns, ~$0.01)
python3 -m src.phases.phase2_architecture \
    --turns 5 \
    --log data/experiments/arch_5.jsonl

# 6. Phase 3 — Diorama memory (5 turns, ~$0.01)
python3 -m src.phases.phase3_diorama \
    --turns 5 \
    --log data/experiments/diorama_5.jsonl

# 7. Analyze results
python3 -m src.analysis.consistency_score data/experiments/*.jsonl --verbose
python3 -m src.analysis.report --compare data/experiments/*.jsonl
```

---

*"Your memories are the lore of your life and the basis for your future. But it turns out, with some help you may be able to rewrite your lore and be who you want to be."* — Kurzgesagt
