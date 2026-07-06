#!/usr/bin/env python3
"""
Analysis & Reporting for Diorama Cognition experiments.

Usage:
    python3 -m src.analysis.consistency_score data/experiments/*.jsonl --verbose
    python3 -m src.analysis.report --compare data/experiments/*.jsonl
    python3 -m src.analysis.report --drift data/experiments/diorama_100.jsonl
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def load_log(path: str) -> List[dict]:
    """Load a JSONL experiment log."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def latency_stats(records: List[dict]) -> Dict:
    """Compute latency statistics across all nodes and synthesizer."""
    latencies = defaultdict(list)

    for r in records:
        for node_name, node_data in r.get("nodes", {}).items():
            latencies[node_name].append(node_data.get("latency_ms", 0))

        synth = r.get("synthesizer", {})
        latencies["synthesizer"].append(synth.get("latency_ms", 0))

    stats = {}
    for name, vals in latencies.items():
        if not vals:
            continue
        vals = sorted(vals)
        stats[name] = {
            "mean_ms": round(sum(vals) / len(vals), 1),
            "min_ms": min(vals),
            "max_ms": max(vals),
            "p50_ms": vals[len(vals) // 2],
            "p95_ms": vals[int(len(vals) * 0.95)],
            "calls": len(vals),
        }
    return stats


def response_lengths(records: List[dict]) -> Dict:
    """Compute response length statistics."""
    lengths = [len(r.get("synthesizer", {}).get("output", "")) for r in records]
    if not lengths:
        return {}
    lengths = sorted(lengths)
    return {
        "mean_chars": round(sum(lengths) / len(lengths), 1),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "total_chars": sum(lengths),
    }


def chemistry_trajectory(records: List[dict]) -> Dict:
    """Track chemistry values over time."""
    if not records:
        return {}

    trajectory = {}
    for r in records:
        turn = r.get("turn", 0)
        chem = r.get("chemistry_state", {})
        for k, v in chem.items():
            if k not in trajectory:
                trajectory[k] = []
            trajectory[k].append((turn, v))

    summary = {}
    for k, vals in trajectory.items():
        values = [v for _, v in vals]
        summary[k] = {
            "start": values[0] if values else None,
            "end": values[-1] if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "diff": (values[-1] - values[0]) if len(values) > 1 else 0,
        }
    return summary


def drift_analysis(records: List[dict]) -> Dict:
    """Analyze response drift for repeated prompts.

    When the same prompt appears multiple times, measure how much the
    response changes across occurrences.
    """
    prompt_responses = defaultdict(list)

    for r in records:
        inp = r.get("input", "")
        out = r.get("synthesizer", {}).get("output", "")
        turn = r.get("turn", 0)
        prompt_responses[inp].append((turn, out))

    drift_results = {}
    for prompt, responses in prompt_responses.items():
        if len(responses) < 2:
            continue

        # Sort by turn
        responses.sort(key=lambda x: x[0])
        first_text = responses[0][1]

        diffs = []
        for turn, text in responses[1:]:
            # Simple length-based drift measure
            length_diff = abs(len(text) - len(first_text))
            diffs.append({
                "turn": turn,
                "char_diff": length_diff,
                "char_diff_pct": round(length_diff / max(len(first_text), 1) * 100, 1),
            })

        drift_results[prompt[:60]] = {
            "occurrences": len(responses),
            "first_turn": responses[0][0],
            "first_length": len(first_text),
            "drifts": diffs,
        }

    return drift_results


def generate_report(log_path: str):
    """Generate a full report for a single experiment log."""
    records = load_log(log_path)
    name = os.path.basename(log_path)

    print(f"\n{'='*60}")
    print(f"  Report: {name}")
    print(f"  Records: {len(records)}")
    print(f"{'='*60}")

    if not records:
        print("  (empty)")
        return

    # Turn range
    turns = [r.get("turn", 0) for r in records]
    print(f"  Turn range: {min(turns)}–{max(turns)}")

    # Latency
    ls = latency_stats(records)
    print(f"\n  --- Latency ---")
    for name, s in sorted(ls.items()):
        print(f"  {name:>25s}: {s['mean_ms']:>7.1f}ms avg  "
              f"({s['min_ms']}–{s['max_ms']}ms, p95={s['p95_ms']}ms, "
              f"n={s['calls']})")

    # Response lengths
    rl = response_lengths(records)
    print(f"\n  --- Response Lengths ---")
    for k, v in rl.items():
        print(f"  {k}: {v}")

    # Chemistry
    ct = chemistry_trajectory(records)
    print(f"\n  --- Chemistry Trajectory ---")
    for nt in ["adrenaline", "dopamine", "serotonin", "cortisol", "oxytocin"]:
        if nt in ct:
            s = ct[nt]
            print(f"  {nt:>15s}: {s['start']:.2f} → {s['end']:.2f}  "
                  f"(range {s['min']:.2f}–{s['max']:.2f}, Δ={s['diff']:+.2f})")

    # Node coverage
    node_coverage = defaultdict(int)
    for r in records:
        for node in r.get("nodes", {}):
            node_coverage[node] += 1
    print(f"\n  --- Node Coverage ---")
    for node, count in sorted(node_coverage.items()):
        print(f"  {node:>30s}: {count}/{len(records)} turns")

    # Drift
    drift = drift_analysis(records)
    if drift:
        print(f"\n  --- Drift (repeated prompts) ---")
        for prompt, d in list(drift.items())[:5]:
            print(f"  \"{prompt}…\" — {d['occurrences']} occurrences")
            for dd in d["drifts"]:
                print(f"      Turn {dd['turn']}: char diff {dd['char_diff_pct']}%")

    print()


def compare_logs(paths: List[str]):
    """Compare multiple experiment logs side by side."""
    datasets = {}
    for p in paths:
        name = os.path.basename(p).replace(".jsonl", "")
        records = load_log(p)
        datasets[name] = records

    print(f"\n{'='*60}")
    print(f"  Cross-Experiment Comparison ({len(datasets)} logs)")
    print(f"{'='*60}")

    if not datasets:
        return

    # Compare basic stats
    headers = list(datasets.keys())
    rows = {
        "Records": [len(datasets[h]) for h in headers],
        "Avg response chars": [
            round(sum(len(r.get("synthesizer", {}).get("output", ""))
                  for r in datasets[h]) / max(len(datasets[h]), 1), 1)
            for h in headers
        ],
        "Avg synth latency (ms)": [
            round(sum(r.get("synthesizer", {}).get("latency_ms", 0)
                  for r in datasets[h]) / max(len(datasets[h]), 1), 1)
            for h in headers
        ],
        "Avg chem adjustments": [
            round(sum(1 for r in datasets[h]
                  if r.get("synthesizer", {}).get("chemistry_adjustment", {}))
                  / max(len(datasets[h]), 1) * 100, 1)
            for h in headers
        ],
    }

    print(f"\n{'Metric':<30s}", end="")
    for h in headers:
        print(f"{h[:20]:>20s}", end="")
    print()

    print(f"{'─'*30}", end="")
    for _ in headers:
        print(f"{'─'*20}", end="")
    print()

    for metric, values in rows.items():
        print(f"{metric:<30s}", end="")
        for v in values:
            print(f"{str(v):>20s}", end="")
        print()

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Diorama Cognition — Analysis & Reporting",
    )
    parser.add_argument("--report", nargs="+", default=[],
                        help="Generate report for one or more JSONL logs")
    parser.add_argument("--compare", nargs="+", default=[],
                        help="Compare multiple JSONL logs")
    parser.add_argument("--drift", default="",
                        help="Analyze memory drift in a diorama run")
    args = parser.parse_args()

    if args.report:
        for path in args.report:
            generate_report(path)

    if args.compare:
        compare_logs(args.compare)

    if args.drift:
        records = load_log(args.drift)
        drift = drift_analysis(records)
        print(f"\n--- Drift Analysis: {args.drift} ---")
        for prompt, d in sorted(drift.items(), key=lambda x: -x[1]["occurrences"]):
            print(f"\nPrompt: \"{prompt}…\"")
            print(f"  Occurrences: {d['occurrences']} "
                  f"(turns {d['first_turn']}–{d['drifts'][-1]['turn'] if d['drifts'] else d['first_turn']})")
            if d['drifts']:
                avg_drift = sum(dd['char_diff_pct'] for dd in d['drifts']) / len(d['drifts'])
                print(f"  Average drift: {avg_drift:.1f}%")

    if not any([args.report, args.compare, args.drift]):
        print("Usage: python3 -m src.analysis.report --report <log> [--compare <logs>]")
        sys.exit(1)


if __name__ == "__main__":
    main()
