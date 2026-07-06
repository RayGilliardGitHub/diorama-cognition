#!/usr/bin/env python3
"""
Phase 3 — Diorama Memory: Reconsolidation via OpenCode Go API

Multi-node architecture (4 nodes + synthesizer) with the Diorama Store
replacing the append-only conversation history.

Key differences from Phase 2:
  - Memory is retrieved by cue (current node outputs + chemistry),
    not by indexing the last 2 turns from a list
  - Every retrieval mutates the stored memory (reconsolidation)
  - Unreinforced memories decay over time
  - Wax temperature modulated by chemistry controls plasticity
  - Memory lineage is tracked for drift analysis

Usage:
    python3 -m src.phases.phase3_diorama --turns 50 --log data/experiments/diorama_50.jsonl
    python3 -m src.phases.phase3_diorama --wax-temp 0.8 --turns 10

Cost: ~$0.002 per turn (5 API calls + local diorama processing). 100 turns = ~$0.20.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.core.instrumentation import TurnLogger, DEFAULT_CHEMISTRY
from src.core.body_state import BodyState
from src.core.diorama import DioramaStore
from src.core.api import chat_for_node, set_api_key, estimate_cost
from src.prompts import nodes


def parse_synth_output(text):
    stripped = text.strip()
    chem_adj = {}
    weighting = {}
    last_close = stripped.rfind("}")
    if last_close >= 0:
        depth = 0
        for i in range(last_close, -1, -1):
            if stripped[i] == "}": depth += 1
            elif stripped[i] == "{": depth -= 1
            if depth == 0:
                candidate = stripped[i:last_close + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        chem_adj = parsed.get("chemistry_adjustment", {})
                        weighting = parsed.get("weighting", {})
                        return stripped[:i].strip(), chem_adj, weighting
                except (json.JSONDecodeError, ValueError):
                    pass
                break
    return stripped, {}, {}


def load_prompts(path, turns_needed):
    prompts = nodes.load_test_prompts(path)
    if not prompts:
        prompts = list(nodes.TEST_PROMPTS)
    result = []
    for i in range(turns_needed):
        result.append(prompts[i % len(prompts)])
    return result


def build_synth_messages(system_prompt, node_outputs, user_message,
                         gate_weights=None, memory_context=""):
    """Build messages for the Synthesizer.

    Phase 3 adds memory_context from the Diorama Store retrieval.
    """
    parts = ["--- NODE INPUTS ---\n"]
    for node_key, output in node_outputs.items():
        label = node_key.upper()
        gate = gate_weights.get(node_key, 1.0) if gate_weights else None
        if gate is not None and gate < 0.2:
            parts.append(f"[{label} (gate: {gate:.2f} — SUPPRESSED)]")
            parts.append("(signal below threshold)")
            parts.append("")
        elif gate is not None and gate < 0.5:
            max_chars = max(50, int(len(output) * gate * 2))
            parts.append(f"[{label} (gate: {gate:.2f} — truncated)]")
            parts.append(output[:max_chars].strip())
            parts.append("")
        else:
            parts.append(f"[{label}]" + (f" (gate: {gate:.2f})" if gate is not None else ""))
            parts.append(output.strip())
            parts.append("")

    if memory_context:
        parts.append("--- RETRIEVED MEMORY CONTEXT ---")
        parts.append(str(memory_context)[:500])
        parts.append("")

    parts.append("--- ORIGINAL USER MESSAGE ---")
    parts.append(user_message)
    parts.append("")
    parts.append(
        "--- END INPUTS ---\n"
        "Synthesize your response from all of the above signals.\n\n"
        "IMPORTANT: Your very last line MUST be a JSON object containing "
        "chemistry_adjustment and weighting keys, like this:\n"
        '{"chemistry_adjustment": {"adrenaline": 0.0, ...}, '
        '"weighting": {"sensory": 0.25, ...}}\n'
        "Do NOT put the JSON inside a code block."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(parts)},
    ]


def build_episodic_messages(sys_prompt, user_message, diorama_context):
    """Build messages for the Episodic Retriever with diorama-derived context."""
    context_parts = []
    if diorama_context:
        context_parts.append("Relevant prior context from memory:")
        context_parts.append(str(diorama_context)[:500])
        context_parts.append("")
    context_parts.append(f"Current user message: {user_message}")
    context_parts.append("")
    context_parts.append(
        "What prior context from the conversation history is relevant? "
        "If none, say 'No prior context.'"
    )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": "\n".join(context_parts)},
    ]


def run_phase3(args):
    """Run Phase 3 — multi-node with diorama memory."""
    disabled = set(args.disable or [])
    log_path = os.path.abspath(args.log)
    turns_needed = args.turns
    prompts = load_prompts(args.prompts, turns_needed)

    valid_names = {key for key, _, _, _ in nodes.PROCESSING_NODES}
    for d in disabled:
        if d not in valid_names:
            print(f"Warning: unknown node '{d}'.")

    chemistry = dict(DEFAULT_CHEMISTRY)
    if args.adrenaline is not None:
        chemistry["adrenaline"] = max(0.0, min(1.0, args.adrenaline))
    if args.cortisol is not None:
        chemistry["cortisol"] = max(0.0, min(1.0, args.cortisol))
    body_state = BodyState(chemistry)

    store = DioramaStore(n_dims=2000, sparsity=0.05, seed=42)
    store.wax_temp = args.wax_temp
    total_cost = 0.0

    print(f"Phase 3 — Diorama Memory (Reconsolidation via OpenCode Go)")
    print(f"  Sensory:     DeepSeek V4 Flash")
    print(f"  Emotional:   MiMo-V2.5")
    print(f"  Episodic:    MiMo-V2.5  (retrieves from Diorama Store)")
    print(f"  Social:      MiMo-V2.5")
    print(f"  Synthesizer: DeepSeek V4 Flash")
    print(f"  Wax temp:   {store.wax_temp}")
    print(f"  Chemistry:  {'active' if not args.chemistry_injection_off else 'OFF'}")
    print(f"  Disabled:   {disabled or '(none)'}")
    print(f"  Log: {log_path}")
    print(f"  Turns: {turns_needed}")
    print()

    for turn_idx, user_msg in enumerate(prompts, start=1):
        store.current_turn = turn_idx
        print(f"─" * 60)
        print(f"Turn {turn_idx}: {user_msg[:80]}")
        print(f"─" * 60)

        body_state.tick(elapsed=2.0)
        store.set_wax_temp_from_chemistry(chemistry)
        print(f"  wax_temp={store.wax_temp:.3f}")

        with TurnLogger(
            log_path, turn_number=turn_idx,
            user_input=user_msg, chemistry_state=chemistry,
        ) as logger:

            node_outputs = {}
            diorama_context = None

            for node_key, model, inst_name, sys_prompt in nodes.PROCESSING_NODES:
                if node_key in disabled:
                    print(f"  [{node_key}] DISABLED")
                    node_outputs[node_key] = "Node disabled. No signal."
                    logger.log_node(inst_name, output="Node disabled.",
                                    gate_multiplier=0.0, latency_ms=0,
                                    chemistry_at_call=dict(chemistry))
                    continue

                start_time = time.perf_counter()
                try:
                    if node_key == "episodic":
                        # Phase 3: retrieve from diorama store first
                        cue = store.make_pattern({
                            "user_input": user_msg,
                            "chemistry": str(chemistry),
                        })
                        retrieved = store.retrieve(cue, tags=[node_key])
                        diorama_context = retrieved if retrieved else "No prior context."

                        messages = build_episodic_messages(
                            sys_prompt, user_msg,
                            retrieved.pattern.tolist()[:200] if retrieved else "",
                        )
                    else:
                        messages = [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_msg},
                        ]

                    output = chat_for_node(
                        node_key, messages
                    )
                    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                    node_outputs[node_key] = output.strip()

                    logger.log_node(
                        inst_name, output=node_outputs[node_key],
                        gate_multiplier=body_state.get_gate_multiplier(node_key),
                        latency_ms=elapsed_ms,
                        chemistry_at_call=dict(chemistry),
                    )

                    cost = estimate_cost(model, 500, 150, 68000)
                    total_cost += cost
                    preview = node_outputs[node_key][:100].replace("\n", " ")
                    print(f"  [{node_key:>9}] {elapsed_ms:>5}ms ~${cost:.6f}  {preview}")

                except Exception as e:
                    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                    node_outputs[node_key] = f"(node_error: {e})"
                    logger.log_node(inst_name, output=node_outputs[node_key],
                                    gate_multiplier=0.0, latency_ms=elapsed_ms,
                                    chemistry_at_call=dict(chemistry))
                    print(f"  [{node_key:>9}] ERROR: {e}")

            # Synthesizer
            synth_model = nodes.SYNTHESIZER_NODE[1]
            synth_prompt = nodes.SYNTHESIZER_NODE[2]

            print(f"  [{ 'synth':>9}] running…", end="", flush=True)
            start_time = time.perf_counter()
            try:
                synth_messages = build_synth_messages(
                    synth_prompt, node_outputs, user_msg,
                    gate_weights={k: body_state.get_gate_multiplier(k) for k in node_outputs},
                    memory_context=diorama_context,
                )
                synth_text = chat_for_node(
                    "synthesizer", synth_messages
                )
                synth_elapsed = int((time.perf_counter() - start_time) * 1000)

                response_text, chem_adjust, weighting = parse_synth_output(synth_text)

                logger.finish(
                    output=response_text, latency_ms=synth_elapsed,
                    chemistry_adjustment=chem_adjust, reported_weighting=weighting,
                )

                cost = estimate_cost(synth_model, 800, 280, 68000)
                total_cost += cost
                print(f" {synth_elapsed}ms ~${cost:.6f}")
                print()
                print(response_text)
                print()

                if chem_adjust:
                    body_state.apply_adjustment(chem_adjust)

                # Encode this turn into the Diorama Store
                new_pattern = store.make_pattern_from_components(
                    node_outputs, chemistry, response_text
                )
                store.encode(new_pattern, tags=[
                    k for k in node_outputs if k not in disabled
                ], turn=turn_idx)

                store.tick(dt=2.0)

            except Exception as e:
                synth_elapsed = int((time.perf_counter() - start_time) * 1000)
                logger.finish(output=f"(synthesizer_error: {e})",
                              latency_ms=synth_elapsed,
                              chemistry_adjustment={}, reported_weighting={})
                print(f" ERROR: {e}\n")

    stats = store.stats()
    print(f"\nDiorama Store Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"Done. Log: {log_path}")
    print(f"Estimated total API cost: ${total_cost:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 — Diorama Memory (reconsolidation via API)",
    )
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--disable", action="append", default=[])
    parser.add_argument("--log", default="data/experiments/diorama.jsonl")
    parser.add_argument("--prompts", default="")
    parser.add_argument("--adrenaline", type=float, default=None)
    parser.add_argument("--cortisol", type=float, default=None)
    parser.add_argument("--chemistry-injection-off", action="store_true")
    parser.add_argument("--wax-temp", type=float, default=0.3)
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()
    set_api_key(args.api_key)
    run_phase3(args)


if __name__ == "__main__":
    main()
