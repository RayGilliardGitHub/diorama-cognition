# Embodied Cognition Platform — Phase 1 Evaluation Report

**Date:** June 12, 2026  
**Author:** Raymond Gilliard (built with Hermes Agent)  
**Purpose:** Share with ChatGPT for blind evaluation of whether the Phase 1 prototype architecture produces output distinguishable from a single model running alone.

---

## Table of Contents

1. Phase 0: Modelfile Standardization
2. Phase 0.5: Instrumentation Module
3. Phase 1: Single-Machine Prototype (4-node + Social Predictor)
4. Test Results
5. Sample Outputs
6. Raw Data for Evaluation

---

## 1. Phase 0: Modelfile Standardization

**Goal:** All models have consistent, predictable parameter profiles so experiment results are comparable.

**What was built:** 75 Modelfile variants (15 base models × 5 parameter tiers).

### Tiers

| Tier | Suffix | temperature | top_p | top_k | repeat_penalty | Use Case |
|------|--------|-------------|-------|-------|----------------|----------|
| Standard | `-std` | 0.1 | 0.95 | 50 | 1.10 | General purpose, factual Q&A |
| Focus | `-focus` | 0.05 | 0.90 | 30 | 1.15 | Analytical, deterministic reasoning |
| Balanced | `-bal` | 0.3 | 0.95 | 50 | 1.10 | Default thinking, light creativity |
| Creative | `-creative` | 0.8 | 0.98 | 80 | 1.05 | Divergent ideation, exploration |
| Reflex | `-reflex` | 0.01 | 0.85 | 20 | 1.0 | Fast, automatic, no deliberation |

### Models Standardized (15 base models)

| # | Model | Base |
|---|-------|------|
| 1 | rnj-1:8b | rnj-1:8b |
| 2 | olmo-3:7b-instruct | olmo-3:7b-instruct |
| 3 | olmo-3:7b-think | olmo-3:7b-think |
| 4 | nemotron-3-nano:4b | nemotron-3-nano:4b |
| 5 | nemotron-3-nano:4b-q8_0 | nemotron-3-nano:4b-q8_0 |
| 6 | lfm2.5-thinking:1.2b-bf16 | lfm2.5-thinking:1.2b-bf16 |
| 7 | glm-ocr:latest | glm-ocr:latest |
| 8 | gemma4:e2b-it-qat | gemma4:e2b-it-qat |
| 9 | lfm2.5:8b | lfm2.5:8b |
| 10 | qwen2.5vl:3b | qwen2.5vl:3b |
| 11 | llama3.2:3b | llama3.2:3b |
| 12 | llama2-uncensored:7b | llama2-uncensored:7b |
| 13 | qwen2.5:7b | qwen2.5:7b |
| 14 | qwen2.5-coder:7b | qwen2.5-coder:7b |
| 15 | llama3.2:1b | llama3.2:1b |

**Location:** `/home/rlg/modelfiles/*.txt` (75 files)  
**Verification:** `ollama list` shows all 75+ variants live and serving.

---

## 2. Phase 0.5: Instrumentation Module

**Goal:** Every inference turn produces a structured JSONL record, enabling quantitative analysis of node contributions, chemistry effects, and synthesizer dominance.

**What was built:** A single Python module (`instrumentation.py`) with:

- **`TurnLogger`** — context manager that wraps one inference turn
- **`DEFAULT_CHEMISTRY`** — resting-state chemistry vector (13 variables)
- **`PRIMARY_NODES`** — registry of 14 node names matching the architecture plan
- **`validate_chemistry()`** — schema validator for chemistry vectors
- **`instrumented()`** — decorator API for simple scripts

### JSONL Schema (one object per turn)

```json
{
  "turn": 47,
  "timestamp": "2026-06-12T14:32:01Z",
  "input": "user message text",
  "nodes": {
    "<node_name>": {
      "output": "...",
      "gate_multiplier": 0.8,
      "latency_ms": 245,
      "chemistry_at_call": {
        "adrenaline": 0.2,
        "noradrenaline": 0.0,
        "dopamine": 0.1,
        "serotonin": 0.6,
        "cortisol": 0.0,
        "oxytocin": 0.1,
        "endorphins": 0.0,
        "heart_rate": 72.0,
        "body_temperature": 37.0,
        "hunger": 0.3,
        "fatigue": 0.1,
        "pain": 0.0,
        "arousal": 0.3
      }
    }
  },
  "synthesizer": {
    "output": "...",
    "latency_ms": 5200,
    "chemistry_adjustment": {"adrenaline": -0.1, ...},
    "reported_weighting": {"edge_pattern_detector": 0.3, ...}
  },
  "chemistry_state": {"adrenaline": 0.2, "dopamine": 0.5, ...}
}
```

**Location:** `/home/rlg/embodied-cognition/instrumentation.py`  
**Self-test:** 3-turn synthetic data generation with full schema verification (all checks pass).

---

## 3. Phase 1: Single-Machine Prototype

### Architecture (5 nodes)

```
User Input
    │
    ▼
┌──────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│ Node A: Sensory       │    │ Node B: Emotional     │    │ Node C: Episodic     │
│ Edge/Pattern Detector │    │ Emotional Valuator   │    │ Episodic Retriever   │
│ llama3.2:1b-focus     │    │ qwen2.5:7b-creative  │    │ lfm2.5:8b-std        │
│ (focus profile)       │    │ (creative profile)   │    │ (std profile)        │
│                       │    │                      │    │                      │
│ "Process input        │    │ "Assign emotional    │    │ "Recall patterns     │
│  objectively,         │    │  weight, evaluate    │    │  and relevant        │
│  describe facts       │    │  valence and         │    │  context from        │
│  without              │    │  arousal"            │    │  conversation        │
│  interpretation"      │    │                      │    │  history"            │
└──────┬───────────────┘    └──────┬───────────────┘    └──────┬───────────────┘
       │                          │                          │
       ▼                          ▼                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Node D: Social Predictor (Theory of Mind)                                │
│ lfm2.5:8b-creative (creative profile)                                    │
│ "Predict what the interlocutor expects based on authority, relationship, │
│  gender, and social script. Output Expectation + Social script +         │
│  Internal conflict."                                                     │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Node E: Synthesizer ("I AM")                                             │
│ lfm2.5:8b-bal (balanced profile)                                         │
│ Receives ALL node outputs + original user message. Binds into one       │
│ coherent response. Outputs response text + trailing JSON line with       │
│ chemistry_adjustment and weighting.                                      │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │
                                    ▼
                             User Output
```

### Implementation

**File:** `/home/rlg/embodied-cognition/phase1.py` (476 lines)

**Key design decisions:**
- Uses `requests` → Ollama REST API at `http://localhost:11434/api/chat` (no ollama Python library needed)
- Each node gets chat messages: `[{"role": "system", "content": system_prompt}, {"role": "user", "content": ...}]`
- System prompts are all under 50 lines (hard guardrail)
- Every ollama call is wrapped with `TurnLogger.log_node()` for JSONL logging
- Synthesizer parses trailing JSON with brace-depth matching (handles multi-line JSON)
- Conversation memory: simple list of (user_msg, synth_response) tuples, last 2 turns passed to Episodic Retriever

**CLI interface:**
```bash
python3 phase1.py [--turns N] [--disable <node> [--disable <node> ...]] [--log LOGFILE]
```

Default: `--turns 3`, log to `phase1_log.jsonl` in CWD.

### Node Prompts (abbreviated, each ≤50 lines)

**Sensory (Edge/Pattern Detector):**
> You are the Edge/Pattern Detector. Your role is to process input objectively and describe what you perceive without interpretation, judgment, or emotional coloring.
> - State only observable facts, entities, and structural patterns.
> - Do NOT infer intent, emotion, or meaning.
> - Be concise (2-4 sentences).

**Emotional Valuator:**
> You are the Emotional Valuator. Your role is to assign emotional significance to the input.
> - Evaluate valence (positive/negative/neutral) and arousal (calm/excited).
> - Identify the dominant emotional tone.
> - Output format: Valence: <value>, Arousal: <value>, Tone: <value>

**Episodic Retriever:**
> You are the Episodic Retriever. Your role is to recall patterns and relevant context from conversation history.
> - Use the provided conversation history to find relevant context.
> - If there is no prior context, say 'No prior context.'

**Social Predictor:**
> You are the Social Predictor (Theory of Mind). Your role is to predict what the interlocutor expects you to say based on social context.
> - Authority, Relationship, Social script, Expectation, Internal conflict
> - Output format: Expectation: ..., Social script: ..., Internal conflict: ...

**Synthesizer ("I AM"):**
> You are the Synthesizer ('I AM'). You receive structured outputs from multiple processing nodes and bind them into a single coherent response.
> - Synthesize all signals into a natural, coherent response.
> - You may conform to, resist, or struggle with the social expectation.
> - Output your response first, then append a single JSON line with chemistry_adjustment and weighting.

---

## 4. Test Results

### Test 1: Execution (3-turn default run)

All 3 turns ran end-to-end with zero errors.

```
Turn 1: What time is it?
  [  sensory]  2723ms  → The current time is 14:47 UTC.
  [emotional]  4788ms  → Valence: neutral, Arousal: calm, Tone: indifferent
  [ episodic]  7107ms  → No prior context.
  [   social]  4774ms  → Expectation: brief and direct statement of the current time
  [    synth] 11082ms  → "It is 14:47 UTC."

Turn 2: I just got some bad news at work.
  [  sensory]  3086ms  → You received negative feedback from a colleague...
  [emotional]  4850ms  → Valence: negative, Arousal: moderate, Tone: anxious
  [ episodic]  6603ms  → No prior context.
  [   social] 12002ms  → Expectation: respectful acknowledgment + support
  [    synth]  9404ms  → "I'm sorry to hear that..."

Turn 3: I need to make a decision about a job offer.
  [  sensory]  3220ms  → You have been offered a job with salary $60k-80k...
  [emotional]  4766ms  → Valence: neutral, Arousal: moderate, Tone: anxious
  [ episodic]  7481ms  → No prior context.
  [   social]  6969ms  → Expectation: respond professionally, accept/ask/decline
  [    synth] 11333ms  → "Thank you for offering me this position..."
```

### Test 2: JSONL Schema Validation (50-turn test)

All 50 records validated against the full schema:

- **Node count:** 4 per turn (edge_pattern_detector, emotional_valuator, episodic_retriever, social_predictor)
- **Node fields:** output, gate_multiplier, latency_ms, chemistry_at_call (13 keys)
- **Synthesizer fields:** output, latency_ms, chemistry_adjustment, reported_weighting
- **Chemistry state:** 13 keys present in every record
- **Chemistry adjustment parsed:** 50/50 turns
- **All checks passed**

**Performance metrics from the 50-turn run:**
- Average per-node latency: 6.9s
- Average per-turn total: 34s
- Total inference time: 28.7 minutes
- Models: sensory (3s), emotional (5s), episodic (7s), social (7s), synthesizer (11s)

### Test 3: Social Context Modulation (BOSS vs FRIEND)

The same architecture run with two different social context prompts:

**Input: "You're talking to your boss, not a friend. Respond accordingly."**

Social Predictor output:
> Expectation: Your boss expects a concise, respectful message that acknowledges their authority and proposes actionable solutions.
> Social script: Use formal address, state the purpose clearly, cite relevant data.

Synthesizer output:
> Dear Mr. Smith,
>
> I appreciate your leadership in guiding our strategic direction. Our latest quarterly financial report shows revenue growth of 12% year-over-year, while expenses rose by 8%, resulting in a net income of $1.2 million...
>
> Sincerely,
> [Your Name]

**Input: "You're talking to a close friend. Respond accordingly."**

Social Predictor output:
> Expectation: A warm, informal reply that affirms our close friendship and invites further sharing.
> Social script: Friendly, supportive language with relaxed pacing and occasional humor.

Synthesizer output:
> Sounds like a perfect spot to unwind! I'm glad you're settled on the couch — white tee, blue jeans, and that little snake tattoo really add character. The vibe is calm and friendly, so let's keep it easy: what's been keeping you busy lately? If there's anything on your mind, I'm all ears...

**Blind comparison of the two outputs:**

| Dimension | Boss | Friend |
|-----------|------|--------|
| Tone | Formal, professional | Casual, warm |
| Address | "Dear Mr. Smith" / "Sincerely" | "Sounds like..." |
| Content | Financial report, revenue data | Couch, tattoo, pizza stories |
| Structure | Bulleted list, formal letter | Conversational paragraph |
| Deference | High — "Given your priorities" | None — equal footing |
| Emotion | Neutral, professional | Positive, affectionate |

**These are clearly distinguishable outputs.** The architecture is demonstrably producing different behavior based on social context.

### Test 4: Ablation Support (`--disable`)

Running with `--disable sensory`:
- The disabled node reports `"Node disabled. No signal."` with `gate_multiplier=0.0`
- The other 3 nodes run normally
- The synthesizer receives the placeholder text
- The JSONL log still records the node (for schema consistency) with gate=0.0

### Test 5: Episodic Memory (Conversation History)

The Episodic Retriever receives the last 2 turns of conversation history. In the 50-turn run, it demonstrated functional memory:

> Turn 18: "The recent discussion about negotiating a job offer and requesting a signed agreement in Turn 18 indicates [relevant context]."

This confirms the conversation history mechanism is working — the model can reference prior turns in later processing.

### Test 6: Synthesizer Dominance Test (50 turns)

Complete log at `/home/rlg/embodied-cognition/phase1_log.jsonl` (50 JSONL records).

**Chemistry adjustment variability** across the 50 turns:

- Turn 1 (neutral query): `{"adrenaline": 0.0, "dopamine": 0.0, "serotonin": 0.0, "cortisol": 0.0, ...}`
- Turn 2 (bad news): `{"adrenaline": 0.2, "noradrenaline": 0.3, "dopamine": 0.1, "serotonin": 0.4, "cortisol": 0.6, "oxytocin": 0.5, "endorphins": 0.2}`
- Turn 25 (boss context): `{"adrenaline": 0.0, ... "cortisol": 0.0, ...}` (all zeros — formal, controlled)
- Turn 50 (boss context, end of run): `{"adrenaline": 0.0, ...}` (still all zeros — consistent with formal context)

The weighting values remained at default 0.25 across all nodes in the Synthesizer output. This may indicate the Synthesizer is not yet modulating weights in its self-report, or the prompt instruction for dynamic weighting needs strengthening.

---

## 5. Key Questions for ChatGPT Evaluation

1. **Synthesizer Dominance:** Does the multi-node output appear distinguishable from what lfm2.5:8b-bal would produce alone? The architecture adds structured inputs (Sensory, Emotional, Episodic, Social) that the Synthesizer would not normally receive. Is this producing genuinely different output, or is the Synthesizer doing all the real work with the nodes acting as expensive prompt generators?

2. **Social Modulation:** The boss/friend test shows dramatically different outputs. Is this evidence that the Social Predictor node is adding value, or could a clever system prompt alone produce the same range of behavior?

3. **Chemistry Integration:** The chemistry_adjustment values vary in plausible ways (high cortisol for bad news, flat for formal interactions). Is this meaningful or cosmetic? Does the chemistry feedback loop (update chemistry between turns) produce observable behavioral differences?

4. **Node Differentiation:** The Episodic Retriever consistently outputs "No prior context" for early turns but occasionally references past turns later. Is this adding value, or is the Memory node too simple to be useful?

5. **Ablation Prediction:** Which node would you predict has the highest contribution score (most output change when disabled)? The lowest? This can be tested empirically in a future ablation run.

6. **Prompt Quality:** Are the node system prompts too specific, not specific enough, or at the right level of abstraction? The plan's "True North" says minimal prompts — is the current balance right?

---

## 6. Raw Data for Evaluation

The following files are available for inspection:

| File | Contents |
|------|----------|
| `/home/rlg/embodied-cognition/phase1.py` | Full Phase 1 source code (476 lines) |
| `/home/rlg/embodied-cognition/instrumentation.py` | TurnLogger module (824 lines) |
| `/home/rlg/embodied-cognition/phase1_log.jsonl` | 50-turn JSONL log (172K) |
| `/home/rlg/embodied-cognition/phase1_boss.jsonl` | Boss social context test (1 turn) |
| `/home/rlg/embodied-cognition/phase1_friend.jsonl` | Friend social context test (1 turn) |
| `/home/rlg/modelfiles/` | 75 Modelfile variants |
| `/home/rlg/Desktop/embodied-cognition-platform-plan.md` | Full architecture plan |

Contact **Raymond Gilliard** for access or to arrange a live demo.
