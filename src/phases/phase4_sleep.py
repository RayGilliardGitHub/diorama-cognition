#!/usr/bin/env python3
"""
Phase 4 — Sleep & Consolidation: Offline Memory Replay

Runs consolidation cycles on an existing DioramaStore state, then evaluates
the effect on memory coherence and chemistry alignment.

Usage:
    python3 -m src.phases.phase4_sleep --input data/experiments/diorama_50.jsonl
    python3 -m src.phases.phase4_sleep --cycles 5 --quality 0.8
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.core.diorama import DioramaStore


def run_phase4(args):
    """Run consolidation cycles on a diorama store, or create a demo store."""

    store = DioramaStore(n_dims=2000, sparsity=0.05, seed=42)

    print(f"Phase 4 — Sleep & Consolidation")
    print(f"Store dims: {store.n_dims}")
    print(f"Sleep quality: {args.quality}")
    print(f"Cycles: {args.cycles}")

    for cycle in range(1, args.cycles + 1):
        print(f"\n  Cycle {cycle}:")
        before = store.stats()
        report = store.consolidate(quality=args.quality)
        after = store.stats()
        print(f"    Consolidated: {report['consolidated']}")
        print(f"    Pruned: {report['pruned']}")
        print(f"    Avg strength before: {before['avg_strength']}")
        print(f"    Avg strength after: {after['avg_strength']}")

        store.tick(dt=10.0)

    print(f"\nFinal store stats:")
    for k, v in store.stats().items():
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 — Sleep & Consolidation",
    )
    parser.add_argument("--input", default="",
                        help="JSONL input log to load (optional)")
    parser.add_argument("--cycles", type=int, default=5,
                        help="Number of sleep cycles")
    parser.add_argument("--quality", type=float, default=0.8,
                        help="Sleep quality (0.0=poor, 1.0=restful)")
    args = parser.parse_args()
    run_phase4(args)


if __name__ == "__main__":
    main()
