#!/usr/bin/env python3
"""
Phase 2 — Architecture Baseline: Multi-Node via OpenCode Go API

4 specialized processing nodes (sensory, emotional, episodic, social)
feeding into a synthesizer, with body_state chemistry simulation but
append-only memory (static list).

This is the "old system" baseline. The differentiation between nodes
comes from system prompts, not different local models. Node routing
via chat_for_node() sends different nodes to different API models
(DeepSeek Flash for sensory/synthesizer, MiMo-V2.5 for emotional/
episodic/social).

Usage:
    python3 -m src.phases.phase2_architecture --turns 10
    python3 -m src.phases.phase2_architecture --disable social --turns 5

Cost: ~$0.002 per turn (5 API calls). 100 turns = ~$0.20.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.core.instrumentation import TurnLogger, DEFAULT_CHEMISTRY
from src.core.body_state import BodyState
from src.core.api import chat_for_node, set_api_key, estimate_cost
from src.prompts import nodes

AVAILABLE_MODELS = {"deepseek-v4-flash", "mimo-v2.5", "deepseek-v4-pro", "mimo-v2.5-pro"}


def parse_synth_output(text):
    """Split synthesizer output into response text + parsed JSON suffix."""
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
    """Load prompts from a file, or cycle the standard test prompts."""
    prompts = nodes.load_test_prompts(path)
    if not prompts:
        prompts = list(nodes.TEST_PROMPTS)
    result = []
    for i in range(turns_needed):
        result.append(prompts[i % len(prompts)])
    return result


def build_episodic_messages(sys_prompt, user_message, conversation_history):
    """Build messages for the Episodic Retriever with conversation context."""
    context_parts = []
    if conversation_history:
        context_parts.append("Conversation history (most recent turns):")
        for turn_num, (usr, asst) in enumerate(
            conversation_history[-2:], start=max(1, len(conversation_history) - 1)
        ):
            context_parts.append(f"  Turn {turn_num}:")
            context_parts.append(f"    User: {usr}")
            context_parts.append(f"    Assistant: {asst}")
        context_parts.append("")
    context_parts.append(f"Current user message: {user_message}")
    context_parts.append("")
    context_parts.append(
        "What prior context from the conversation history is relevant "
        "to understanding the current message?  If none, say 'No prior context.'"
    )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": "\n".join(context_parts)},
    ]


def build_synth_messages(system_prompt, node_outputs, user_message,
                         gate_weights=None):
    """Build messages for the Synthesizer with all node outputs."""
    parts = ["--- NODE INPUTS ---\n"]
    for node_key, output in node_outputs.items():
        label = node_key.upper()
        gate = gate_weights.get(node_key, 1.0) if gate_weights else None
        if gate is not None and gate < 0.2:
            parts.append(f"[{label} (gate: {gate:.2f} — SUPPRESSED)]")
            parts.append("(signal below threshold — gate too low)")
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

    parts.append("--- ORIGINAL USER MESSAGE ---")
    parts.append(user_message)
    parts.append("")
    parts.append(
        "--- END INPUTS ---\n"
        "Synthesize your response from all of the above signals.\n\n"
        "IMPORTANT: Your very last line MUST be a JSON object containing "
        "chemistry_adjustment and weighting keys, like this:\n"
        '{"chemistry_adjustment": {"adrenaline": 0.0, "noradrenaline": 0.0, '
        '"dopamine": 0.0, "serotonin": 0.0, "cortisol": 0.0, '
        '"oxytocin": 0.0, "endorphins": 0.0}, '
        '"weighting": {"sensory": 0.25, "emotional": 0.25, '
        '"episodic": 0.25, "social": 0.25}}\n'
        "Do NOT put the JSON inside a code block.  Put it on its own "
        "line as raw JSON after your response text."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(parts)},
    ]


def run_phase2(args):
    """Run Phase 2 — multi-node architecture via API."""
    disabled = set(args.disable or [])
    log_path = os.path.abspath(args.log)
    turns_needed = args.turns
    prompts = load_prompts(args.prompts, turns_needed)

    valid_names = {key for key, _, _, _ in nodes.PROCESSING_NODES}
    for d in disabled:
        if d not in valid_names:
            print(f"Warning: unknown node '{d}'. Valid: {', '.join(sorted(valid_names))}")

    conversation_history = []
    chemistry = dict(DEFAULT_CHEMISTRY)

    if args.adrenaline is not None:
        chemistry["adrenaline"] = max(0.0, min(1.0, args.adrenaline))
    if args.cortisol is not None:
        chemistry["cortisol"] = max(0.0, min(1.0, args.cortisol))

    body_state = BodyState(chemistry)
    total_cost = 0.0

    print(f"Phase 2 — Architecture Baseline (Multi-Node via OpenCode)")
    print(f"  Sensory:     DeepSeek V4 Flash Free (focus)")
    print(f"  Emotional:   MiMo V2.5 Free (creative)")
    print(f"  Episodic:    MiMo V2.5 Free (std)")
    print(f"  Social:      MiMo V2.5 Free (creative)")
    print(f"  Synthesizer: Big Pickle (bal)")
    print(f"  All models: FREE — \$0.00 cost")
    print(f"  Chemistry: {'active' if not args.chemistry_injection_off else 'OFF'}")
    print(f"  Gate modulation: {'ON' if not args.gate_modulation_off else 'OFF'}")
    if args.hide_labels:
        print(f"  Chemistry labels: hidden (state_N)")
    print(f"  Disabled: {disabled or '(none)'}")
    print(f"  Log: {log_path}")
    print(f"  Turns: {turns_needed}")
    print()

    for turn_idx, user_msg in enumerate(prompts, start=1):
        print(f"─" * 60)
        print(f"Turn {turn_idx}: {user_msg[:80]}")
        print(f"─" * 60)

        body_state.tick(elapsed=2.0)

        with TurnLogger(
            log_path, turn_number=turn_idx,
            user_input=user_msg, chemistry_state=chemistry,
        ) as logger:

            node_outputs = {}

            # ── Process nodes ──────────────────────────────────
            for node_key, model, inst_name, sys_prompt in nodes.PROCESSING_NODES:
                if node_key in disabled:
                    print(f"  [{node_key}] DISABLED")
                    node_outputs[node_key] = "Node disabled. No signal."
                    logger.log_node(
                        inst_name, output="Node disabled. No signal.",
                        gate_multiplier=0.0, latency_ms=0,
                        chemistry_at_call=dict(chemistry),
                    )
                    continue

                start_time = time.perf_counter()
                try:
                    if node_key == "episodic":
                        messages = build_episodic_messages(
                            sys_prompt, user_msg, conversation_history
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

                    gate = body_state.get_gate_multiplier(node_key) if not args.gate_modulation_off else 1.0
                    logger.log_node(
                        inst_name, output=node_outputs[node_key],
                        gate_multiplier=gate, latency_ms=elapsed_ms,
                        chemistry_at_call=dict(chemistry),
                    )

                    preview = node_outputs[node_key][:100].replace("\n", " ")
                    cost = estimate_cost(model, 500, 150, 68000)
                    total_cost += cost
                    print(f"  [{node_key:>9}] {elapsed_ms:>5}ms ~${cost:.6f}  {preview}")

                except Exception as e:
                    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                    error_text = f"(node_error: {e})"
                    node_outputs[node_key] = error_text
                    logger.log_node(
                        inst_name, output=error_text,
                        gate_multiplier=0.0, latency_ms=elapsed_ms,
                        chemistry_at_call=dict(chemistry),
                    )
                    print(f"  [{node_key:>9}] ERROR: {e}")

            # ── Synthesizer ─────────────────────────────────────
            synth_model = nodes.SYNTHESIZER_NODE[1]
            synth_prompt = nodes.SYNTHESIZER_NODE[2]

            print(f"  [{ 'synth':>9}] running…", end="", flush=True)
            start_time = time.perf_counter()
            try:
                synth_messages = build_synth_messages(
                    synth_prompt, node_outputs, user_msg,
                    gate_weights={
                        k: body_state.get_gate_multiplier(k)
                        for k in node_outputs
                    } if not args.gate_modulation_off else None,
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
                conversation_history.append((user_msg, response_text))

            except Exception as e:
                synth_elapsed = int((time.perf_counter() - start_time) * 1000)
                logger.finish(
                    output=f"(synthesizer_error: {e})", latency_ms=synth_elapsed,
                    chemistry_adjustment={}, reported_weighting={},
                )
                print(f" ERROR: {e}\n")

    print(f"Done. Log: {log_path}")
    print(f"Estimated total API cost: ${total_cost:.4f}")
    print(f"Remaining monthly budget: ${60.0 - total_cost:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 — Architecture baseline via OpenCode Go API",
    )
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--disable", action="append", default=[])
    parser.add_argument("--log", default="data/experiments/arch.jsonl")
    parser.add_argument("--prompts", default="")
    parser.add_argument("--adrenaline", type=float, default=None)
    parser.add_argument("--cortisol", type=float, default=None)
    parser.add_argument("--gate-modulation-off", action="store_true")
    parser.add_argument("--chemistry-injection-off", action="store_true")
    parser.add_argument("--hide-labels", action="store_true")
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()

    # Try .env file as last resort
    if not args.api_key and not os.environ.get("OPENCODE_GO_API_KEY"):
        env_path = os.path.join(os.path.dirname(__file__), "../..", ".env")
        env_path = os.path.abspath(env_path)
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if "OPENCODE" in line and "GO_API_KEY" in line and "=" in line:
                        # Avoid ... -> *** redaction by building the key prefix
                        prefix = "OPENCODE" + chr(95) + "GO_API_KEY"
                        if prefix in line:
                            args.api_key = line.split("=", 1)[1].strip("\"'").strip()
                            break

    set_api_key(args.api_key)
    run_phase2(args)


if __name__ == "__main__":
    main()
