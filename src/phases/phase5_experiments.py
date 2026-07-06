#!/usr/bin/env python3
"""
Phase 5 — Full Integration & Comparative Experiments

Systematic A/B comparison across all architectures and ablations.
Runs experiment matrix from PLAN.md and produces comparison reports.

Usage:
    python3 -m src.phases.phase5_experiments --run E01,E02,E04
    python3 -m src.phases.phase5_experiments --list       # show experiment matrix
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Experiment configuration — matches PLAN.md §9
EXPERIMENTS = {
    "E01": {
        "name": "Single Model Baseline",
        "phase": "phase1_baseline",
        "flags": ["--turns", "100", "--log", "data/experiments/e01_baseline_100.jsonl"],
        "purpose": "Ultimate baseline — single model, no architecture",
    },
    "E02": {
        "name": "Multi-Node Append Memory",
        "phase": "phase2_architecture",
        "flags": ["--turns", "100", "--log", "data/experiments/e02_arch_full_100.jsonl"],
        "purpose": "Old system baseline — multi-node with static memory",
    },
    "E04": {
        "name": "Multi-Node Diorama Memory",
        "phase": "phase3_diorama",
        "flags": ["--turns", "100", "--log", "data/experiments/e04_diorama_100.jsonl"],
        "purpose": "Diorama vs append memory comparison",
    },
    "E06": {
        "name": "Full: Diorama + Chemistry + Sleep",
        "phase": "phase3_diorama",
        "flags": ["--turns", "100", "--log", "data/experiments/e06_full_100.jsonl"],
        "purpose": "Everything active",
    },
    "E07": {
        "name": "Long-Run Diorama (500 turns)",
        "phase": "phase3_diorama",
        "flags": ["--turns", "500", "--log", "data/experiments/e07_longrun_500.jsonl"],
        "purpose": "Long-run drift test",
    },
}


def run_experiment(exp_id: str, dry_run: bool = False):
    """Run a single experiment by ID."""
    if exp_id not in EXPERIMENTS:
        print(f"Unknown experiment: {exp_id}")
        print(f"Available: {', '.join(sorted(EXPERIMENTS.keys()))}")
        return

    exp = EXPERIMENTS[exp_id]
    phase_module = exp["phase"]
    flags = " ".join(exp["flags"])

    cmd = f"python3 -m src.phases.{phase_module} {flags}"
    print(f"\n{'='*60}")
    print(f"  Running {exp_id}: {exp['name']}")
    print(f"  Purpose: {exp['purpose']}")
    print(f"  Command: {cmd}")
    print(f"{'='*60}\n")

    if dry_run:
        print("  (dry run — not executing)")
        return

    os.system(cmd)


def list_experiments():
    """Print the experiment matrix."""
    print(f"\n{'ID':<6} {'Name':<40} {'Phase':<25} {'Purpose':<55}")
    print(f"{'─'*6} {'─'*40} {'─'*25} {'─'*55}")
    for eid in sorted(EXPERIMENTS.keys()):
        exp = EXPERIMENTS[eid]
        print(f"{eid:<6} {exp['name']:<40} {exp['phase']:<25} {exp['purpose']:<55}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 — Full Integration & Comparative Experiments",
    )
    parser.add_argument("--run", default="",
                        help="Comma-separated experiment IDs to run (e.g. E01,E02)")
    parser.add_argument("--list", action="store_true",
                        help="List available experiments")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show commands without executing")
    args = parser.parse_args()

    if args.list:
        list_experiments()
        return

    if args.run:
        for eid in args.run.split(","):
            eid = eid.strip()
            run_experiment(eid, dry_run=args.dry_run)
        return

    list_experiments()


if __name__ == "__main__":
    main()
