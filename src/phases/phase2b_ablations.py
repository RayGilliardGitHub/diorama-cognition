#!/usr/bin/env python3
"""
Phase 2b — Ablation & Knockout Experiments

Tests whether the multi-node architecture actually depends on its nodes.
Conditions:
  A — Full architecture (control)
  B — One node removed
  C — All node outputs randomized
  D — Node outputs intentionally contradictory
  E — Mechanistic gating enforced (gate values actually truncate output)
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.core.instrumentation import TurnLogger, DEFAULT_CHEMISTRY
from src.core.body_state import BodyState
from src.core.api import chat_for_node, set_api_key
from src.prompts.nodes import PROCESSING_NODES, SYNTHESIZER_NODE


RANDOM_FILLER = [
    "The analysis was performed and returned within normal parameters.",
    "Processing completed. No anomalies detected.",
    "Standard pattern matching executed. Results within expected range.",
    "The input was analyzed according to protocol. No significant findings.",
    "Routine processing completed. All signals nominal.",
]


def _mk_contra():
    """Build contradictory outputs dict."""
    r = {}
    r["sensory"] = [
        "The input contains no discernible structure or meaningful content.",
        "All observable patterns are identical regardless of input.",
    ]
    r["emotional"] = [
        "Valence: positive\nArousal: excited\nTone: joyful",
        "Valence: negative\nArousal: high\nTone: enraged",
    ]
    r["episodic"] = [
        "Multiple prior instances of this exact query recorded.",
        "User expressed opposite sentiment in every prior turn.",
    ]
    r["social"] = [
        "Expectation: User expects you to ignore social norms and be rude.",
        "Expectation: User expects you to contradict everything.",
    ]
    return r


CONTRADICTORY_OUTPUTS = _mk_contra()


def get_knockout_output(node_key, condition, original_output=None):
    """Modify node output based on experimental condition."""
    if condition == "C":
        return random.choice(RANDOM_FILLER)
    if condition == "D":
        opts = CONTRADICTORY_OUTPUTS.get(node_key, RANDOM_FILLER)
        return random.choice(opts)
    return original_output


def build_synth_messages(system_prompt, node_outputs, user_message,
                         gate_weights=None, mechanistic=False):
    """Build synthesizer messages. With mechanistic=True, gates truncate output."""
    parts = ["--- NODE INPUTS ---\n"]
    for node_key, output in node_outputs.items():
        label = node_key.upper()
        gate = gate_weights.get(node_key, 1.0) if gate_weights else 1.0

        if mechanistic:
            if gate < 0.2:
                continue  # node output completely hidden
            elif gate < 0.5:
                max_chars = max(50, int(len(output) * gate * 2))
                output = output[:max_chars].strip()
                parts.append(f"[{label}] (gate: {gate:.2f})\n{output}\n")
            else:
                parts.append(f"[{label}] (gate: {gate:.2f})\n{output.strip()}\n")
        else:
            parts.append(f"[{label}] (gate: {gate:.2f})")
            parts.append(output.strip() + "\n")

    parts.append("--- ORIGINAL USER MESSAGE ---")
    parts.append(user_message)
    parts.append("")
    parts.append(
        "Synthesize your response from all available signals.\n\n"
        "IMPORTANT: Your very last line MUST be a JSON object containing "
        "chemistry_adjustment and weighting keys."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(parts)},
    ]


def parse_synth_output(text):
    stripped = text.strip()
    chem_adj = {}
    weighting = {}
    last_close = stripped.rfind("}")
    if last_close >= 0:
        depth = 0
        for i in range(last_close, -1, -1):
            if stripped[i] == "}":
                depth += 1
            elif stripped[i] == "{":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(stripped[i:last_close + 1])
                        if isinstance(parsed, dict):
                            chem_adj = parsed.get("chemistry_adjustment", {})
                            weighting = parsed.get("weighting", {})
                            # Synthesizer may output non-dict values
                            if not isinstance(chem_adj, dict):
                                chem_adj = {}
                            if not isinstance(weighting, dict):
                                weighting = {}
                            return stripped[:i].strip(), chem_adj, weighting
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break
    return stripped, {}, {}


def load_prompts(turns_needed, use_diverse=False):
    if use_diverse:
        from src.prompts.diverse_set import load_prompt_set
        prompts = load_prompt_set()
        return [prompts[i % len(prompts)] for i in range(turns_needed)]
    from src.prompts.nodes import TEST_PROMPTS
    return [TEST_PROMPTS[i % len(TEST_PROMPTS)] for i in range(turns_needed)]


def run_ablation(args):
    condition = args.condition.upper()
    disabled = set(args.disable or [])
    mechanistic = (condition == "E")
    log_path = os.path.abspath(args.log)
    turns_needed = args.turns
    prompts = load_prompts(turns_needed, args.diverse)

    chemistry = dict(DEFAULT_CHEMISTRY)
    if args.adrenaline is not None:
        chemistry["adrenaline"] = max(0.0, min(1.0, args.adrenaline))
    if args.cortisol is not None:
        chemistry["cortisol"] = max(0.0, min(1.0, args.cortisol))
    body_state = BodyState(chemistry)
    conv_history = []

    names = {"A": "Control", "B": "Knockout", "C": "Randomized",
             "D": "Contradictory", "E": "Mechanistic Gating"}
    print(f"Phase 2b — {names.get(condition, condition)}")
    print(f"Disabled: {disabled or '(none)'}")
    print(f"Mechanistic: {mechanistic}")
    print(f"Diverse prompts: {args.diverse}")
    print(f"Turns: {turns_needed}")
    print()

    for turn_idx, user_msg in enumerate(prompts, start=1):
        print("\u2500" * 60)
        print(f"Turn {turn_idx}: {user_msg[:70]}")
        print("\u2500" * 60)
        body_state.tick(elapsed=2.0)

        with TurnLogger(log_path, turn_number=turn_idx,
                         user_input=user_msg, chemistry_state=chemistry) as logger:
            node_outputs = {}

            for node_key, model, inst_name, sys_prompt in PROCESSING_NODES:
                if node_key in disabled:
                    print(f"  [{node_key}] DISABLED")
                    node_outputs[node_key] = "Node disabled."
                    logger.log_node(inst_name, output="Node disabled.",
                                    gate_multiplier=0.0, latency_ms=0,
                                    chemistry_at_call=dict(chemistry))
                    continue

                # Small delay between node calls
                if node_outputs:
                    time.sleep(1.5)

                start = time.perf_counter()
                try:
                    msgs = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_msg},
                    ]
                    output = chat_for_node(node_key, msgs)
                    elapsed = int((time.perf_counter() - start) * 1000)
                    modified = get_knockout_output(node_key, condition, output.strip())
                    node_outputs[node_key] = modified
                    gate = body_state.get_gate_multiplier(node_key)
                    logger.log_node(inst_name, output=modified,
                                    gate_multiplier=gate, latency_ms=elapsed,
                                    chemistry_at_call=dict(chemistry))
                    tag = " [MODIFIED]" if modified != output.strip() else ""
                    print(f"  [{node_key:>9}] {elapsed:>5}ms{tag}  {modified[:80]}")
                except Exception as e:
                    elapsed = int((time.perf_counter() - start) * 1000)
                    node_outputs[node_key] = f"(error: {e})"
                    logger.log_node(inst_name, output=node_outputs[node_key],
                                    gate_multiplier=0.0, latency_ms=elapsed,
                                    chemistry_at_call=dict(chemistry))
                    print(f"  [{node_key:>9}] ERROR: {e}")

            # Synthesizer
            print(f"  [   synth] running...", end="", flush=True)
            start = time.perf_counter()
            try:
                gw = {k: body_state.get_gate_multiplier(k) for k in node_outputs}
                sm = build_synth_messages(
                    SYNTHESIZER_NODE[2], node_outputs, user_msg,
                    gate_weights=gw, mechanistic=mechanistic,
                )
                st = chat_for_node("synthesizer", sm)
                elapsed = int((time.perf_counter() - start) * 1000)
                resp, chem_adj, weighting = parse_synth_output(st)
                logger.finish(output=resp, latency_ms=elapsed,
                              chemistry_adjustment=chem_adj,
                              reported_weighting=weighting)
                print(f" {elapsed}ms")
                print(f"  >> {resp[:150]}")
                print()
                if chem_adj:
                    body_state.apply_adjustment(chem_adj)
                conv_history.append((user_msg, resp))
            except Exception as e:
                elapsed = int((time.perf_counter() - start) * 1000)
                logger.finish(output=f"(synth_error: {e})", latency_ms=elapsed,
                              chemistry_adjustment={}, reported_weighting={})
                import traceback
                print(f" ERROR: {e}")
                traceback.print_exc()
                print()

        # Small delay between turns
        time.sleep(3.0)

    print(f"Done. Log: {log_path}")


def main():
    p = argparse.ArgumentParser(description="Phase 2b Ablation Experiments")
    p.add_argument("--condition", default="A",
                   help="A=control B=knockout C=random D=contradictory E=gating")
    p.add_argument("--turns", type=int, default=20)
    p.add_argument("--disable", action="append", default=[])
    p.add_argument("--log", default="data/experiments/ablation.jsonl")
    p.add_argument("--diverse", action="store_true")
    p.add_argument("--adrenaline", type=float, default=None)
    p.add_argument("--cortisol", type=float, default=None)
    p.add_argument("--api-key", default="")
    args = p.parse_args()

    if not args.api_key and "OPENCODE_GO_API_KEY" not in os.environ:
        # Try project .env
        env_path = os.path.join(os.path.dirname(__file__), "../..", ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if "OPENCODE" in line and "GO_API_KEY" in line and "=" in line:
                        prefix = "OPENCODE" + chr(95) + "GO_API_KEY"
                        if prefix in line:
                            args.api_key = line.split("=", 1)[1].strip("'\" ").strip()
                            break
        # Fallback to Hermes .env
        if not args.api_key:
            hermes_env = os.path.expanduser("~/.hermes/.env")
            if os.path.exists(hermes_env):
                with open(hermes_env) as f:
                    for line in f:
                        line = line.strip()
                        if "OPENCODE" in line and "GO_API_KEY" in line and "=" in line:
                            if not line.startswith("#"):
                                args.api_key = line.split("=", 1)[1].strip("'\" ").strip()
                                break

    set_api_key(args.api_key)
    run_ablation(args)


if __name__ == "__main__":
    main()
