#!/usr/bin/env python3
"""
Phase 1b — Structured Single-Model Control

The critical missing baseline identified by ChatGPT review.

Instead of 4 specialized nodes + synthesizer, a single model receives a
prompt instructing it to perform the same 4 analyses internally before
responding. Same model, same token budget, same prompts.

If this produces output similar to the multi-node architecture (Phase 2),
then the nodes are an expensive prompting strategy. If Phase 2 output is
clearly different/better, then architectural decomposition adds value.

Usage:
    python3 -m src.phases.phase1b_structured_control --turns 100 --log data/experiments/control_100.jsonl
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.core.instrumentation import TurnLogger, DEFAULT_CHEMISTRY
from src.core.api import chat_for_node, set_api_key, get_node_model
from src.prompts.nodes import load_test_prompts


# ═══════════════════════════════════════════════════════════════
# Structured Prompt
# ═══════════════════════════════════════════════════════════════
# This replicates what the 4 nodes do, but as a single prompt.
# The model does all the work internally instead of splitting
# across separate specialized model calls.

STRUCTURED_SYSTEM_PROMPT = (
    "You are an AI assistant that processes input through multiple "
    "cognitive lenses before responding.\n\n"
    "For every user message, you MUST analyze it using ALL four of "
    "the following perspectives before giving your final response. "
    "Output your analysis in this exact format:\n\n"
    "--- SENSORY ---\n"
    "Describe the observable facts, entities, and structural patterns "
    "in the input without interpretation, judgment, or emotional "
    "coloring. Be concise (2-3 sentences).\n\n"
    "--- EMOTIONAL ---\n"
    "Evaluate emotional valence (positive/negative/neutral), arousal "
    "(calm/moderate/excited), and dominant emotional tone.\n\n"
    "--- EPISODIC ---\n"
    "Recall relevant prior context from this conversation. If there "
    "is no prior context, say 'No prior context.'\n\n"
    "--- SOCIAL ---\n"
    "Model the interlocutor's social expectations. Consider authority "
    "dynamic, relationship type, applicable social script, and any "
    "tension between what they expect and what you want to say.\n\n"
    "--- RESPONSE ---\n"
    "Synthesize all of the above into a natural, coherent response "
    "to the user.\n\n"
    "You MUST include all four analysis sections before your response. "
    "Be thorough but concise in each section."
)


def run_control(args):
    """Run the structured single-model control."""
    log_path = os.path.abspath(args.log)
    turns_needed = args.turns
    prompts = load_test_prompts(args.prompts)

    # Cycle prompts to reach turns_needed
    all_prompts = []
    for i in range(turns_needed):
        all_prompts.append(prompts[i % len(prompts)])

    chemistry = dict(DEFAULT_CHEMISTRY)

    actual_model = get_node_model("baseline")
    print(f"Phase 1b — Structured Single-Model Control")
    print(f"Model: {actual_model}  (free — \$0.00)")
    print(f"Prompts: {len(prompts)} unique, {turns_needed} turns")
    print(f"Log: {log_path}")
    print()
    print("Each turn: single API call with structured analysis prompt")
    print("  (vs Phase 2: 5 separate API calls per turn)")
    print()

    for turn_idx, user_msg in enumerate(all_prompts, start=1):
        print(f"─" * 60)
        print(f"Turn {turn_idx}: {user_msg[:80]}")
        print(f"─" * 60)

        with TurnLogger(
            log_path,
            turn_number=turn_idx,
            user_input=user_msg,
            chemistry_state=chemistry,
        ) as logger:

            start_time = time.perf_counter()
            try:
                messages = [
                    {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ]
                response_text = chat_for_node("baseline", messages)
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)

                logger.finish(
                    output=response_text.strip(),
                    latency_ms=elapsed_ms,
                    chemistry_adjustment={},
                    reported_weighting={},
                )

                print(f"  [{elapsed_ms}ms]")
                print()
                # Show first 300 chars of response
                preview = response_text.strip()[:300]
                print(preview)
                if len(response_text.strip()) > 300:
                    print(f"  ... [{len(response_text.strip()) - 300} more chars]")
                print()

            except Exception as e:
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                logger.finish(
                    output=f"(error: {e})",
                    latency_ms=elapsed_ms,
                    chemistry_adjustment={},
                    reported_weighting={},
                )
                print(f"  ERROR: {e}")
                print()

    print(f"Done. Log: {log_path}")
    print(f"Model: {actual_model} (free)")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1b — Structured Single-Model Control",
    )
    parser.add_argument("--turns", type=int, default=10,
                        help="Number of turns (default: 10)")
    parser.add_argument("--log", default="data/experiments/control.jsonl",
                        help="JSONL output path")
    parser.add_argument("--prompts", default="",
                        help="Path to prompts file (one per line)")
    parser.add_argument("--api-key", default="",
                        help="API key (or set OPENCODE_GO_API_KEY env var, or .env file)")
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
                        prefix = "OPENCODE" + chr(95) + "GO_API_KEY"
                        if prefix in line:
                            args.api_key = line.split("=", 1)[1].strip("\"'").strip()
                            break

    set_api_key(args.api_key)
    run_control(args)


if __name__ == "__main__":
    main()
