#!/usr/bin/env python3
"""
Phase 1 — Baseline: Single-Model Control via OpenCode Go API

Single DeepSeek V4 Flash (or MiMo-V2.5) answering prompts directly.
No nodes, no chemistry modulation, no memory architecture.
This is the "control" for every later phase comparison.

Usage:
    python3 -m src.phases.phase1_baseline --turns 100 --log data/experiments/baseline_100.jsonl
    python3 -m src.phases.phase1_baseline --turns 5 --model mimo-v2.5

Models:
  deepseek-v4-flash  ($0.14/$0.28 per M tokens, ~32K req/5hr limit)
  mimo-v2.5          ($0.14/$0.28 per M tokens, ~30K req/5hr limit)

Cost: ~$0.0004 per turn. 100 turns = ~$0.04.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.core.instrumentation import TurnLogger, DEFAULT_CHEMISTRY
from src.core.api import (
    chat_for_node, set_api_key, estimate_cost, get_node_model,
    MODEL_DEEPSEEK_V4_FLASH, MODEL_MIMO_V2_5,
)
from src.prompts.nodes import BASELINE_PROMPT, load_test_prompts


def run_phase1(args):
    """Run Phase 1 — single model baseline via API."""
    log_path = os.path.abspath(args.log)
    turns_needed = args.turns
    model = args.model
    prompts = load_test_prompts(args.prompts)

    # Cycle prompts to reach turns_needed
    all_prompts = []
    for i in range(turns_needed):
        all_prompts.append(prompts[i % len(prompts)])

    chemistry = dict(DEFAULT_CHEMISTRY)
    total_cost = 0.0

    # Get the actual model from node config (not the --model arg display)
    actual_model = get_node_model("baseline")
    print(f"Phase 1 — Baseline (Single Model via OpenCode)")
    print(f"Model: {actual_model}  (free — \$0.00/turn)")
    print(f"All costs are \$0 — unlimited runs")
    print(f"Prompts: {len(prompts)} unique, {turns_needed} turns")
    print(f"Log: {log_path}")
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
                    {"role": "system", "content": BASELINE_PROMPT},
                    {"role": "user", "content": user_msg},
                ]
                response_text = chat_for_node("baseline", messages)
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)

                # Rough cost tracking
                in_tok = len(messages[0]["content"]) + len(messages[1]["content"])
                out_tok = len(response_text)
                turn_cost = estimate_cost(model, in_tok // 2, out_tok // 2, 68000)
                total_cost += turn_cost

                logger.finish(
                    output=response_text.strip(),
                    latency_ms=elapsed_ms,
                    chemistry_adjustment={},
                    reported_weighting={},
                )

                print(f"  [{elapsed_ms}ms]")
                print()
                print(response_text.strip())
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
    print(f"Model: Big Pickle (free — \$0.00 total)")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 — Baseline (single model via OpenCode Go API)",
    )
    parser.add_argument("--turns", type=int, default=10,
                        help="Number of turns (default: 10)")
    parser.add_argument("--model", default=MODEL_DEEPSEEK_V4_FLASH,
                        help=f"Model (default: {MODEL_DEEPSEEK_V4_FLASH})")
    parser.add_argument("--log", default="data/experiments/baseline.jsonl",
                        help="JSONL output path")
    parser.add_argument("--prompts", default="",
                        help="Path to prompts file (one per line)")
    parser.add_argument("--api-key", default="",
                        help="OpenCode Go API key (or set OPENCODE_GO_API_KEY env var, or .env file)")
    args = parser.parse_args()

    # Try .env file as last resort
    if not args.api_key and not os.environ.get("OPENCODE_GO_API_KEY"):
        env_path = os.path.join(os.path.dirname(__file__), "../..", ".env")
        env_path = os.path.abspath(env_path)
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENCODE_GO_API_KEY="):
                        args.api_key = line.split("=", 1)[1].strip("\"'")
                        break

    set_api_key(args.api_key)
    run_phase1(args)


if __name__ == "__main__":
    main()
