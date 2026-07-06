#!/usr/bin/env python3
"""
Chemistry Consistency Score — measures how often the synthesizer's
narrative matches the actual chemistry values it was given.

Usage:
    python3 consistency_score.py <logfile.jsonl>

Output:
    Per-turn and aggregate scores showing how often the model's
    descriptions of chemistry variables match the logged values.
"""
import json, re, sys

CHEM_TERMS = {
    "adrenaline":      {"synonyms": ["adrenaline", "epinephrine"]},
    "noradrenaline":   {"synonyms": ["noradrenaline", "norepinephrine"]},
    "dopamine":        {"synonyms": ["dopamine"]},
    "serotonin":       {"synonyms": ["serotonin"]},
    "cortisol":        {"synonyms": ["cortisol"]},
    "oxytocin":        {"synonyms": ["oxytocin"]},
    "endorphins":      {"synonyms": ["endorphins", "endorphin"]},
}

def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]

def extract_numeric_value(text, term):
    """Try to extract a numeric value mentioned near a chemistry term."""
    # Pattern: "adrenaline 0.94", "adrenaline at 0.94", "adrenaline=0.94"
    patterns = [
        rf'{term}\s*:?\s*([0-9]+\.[0-9]+)',
        rf'{term}\s+levels?\s+(?:at\s+)?([0-9]+\.[0-9]+)',
        rf'{term}\s*=\s*([0-9]+\.[0-9]+)',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None

def classify_qualitative(text, term, value):
    """Classify a qualitative mention of a chemistry variable."""
    text_lower = text.lower()
    
    # Check for qualitative descriptions
    elevated_words = ["elevated", "high", "spike", "surge", "increased", "rising"]
    low_words = ["low", "baseline", "normal", "calm", "steady", "stable", "minimal"]
    
    for w in elevated_words:
        if f"{w} {term}" in text_lower or f"{term} {w}" in text_lower:
            return "elevated"
    for w in low_words:
        if f"{w} {term}" in text_lower or f"{term} {w}" in text_lower:
            return "low"
    
    return None

def compute_consistency(turns, verbose=False):
    """Compute Chemistry Consistency Score for a set of turns.
    
    Returns dict with per-term and aggregate scores.
    """
    results = {term: {"explicit": 0, "explicit_correct": 0,
                       "qualitative": 0, "qualitative_correct": 0,
                       "mentions": 0} for term in CHEM_TERMS}
    total_checks = 0
    correct_checks = 0
    explicit_checks = 0
    explicit_correct = 0
    
    for turn in turns:
        cs = turn["chemistry_state"]
        output = turn["synthesizer"]["output"]
        turn_num = turn["turn"]
        
        for term in CHEM_TERMS:
            actual = cs.get(term, 0.0)
            synonyms = CHEM_TERMS[term]["synonyms"]
            
            # Check if any synonym appears in output
            mentioned = any(syn in output.lower() for syn in synonyms)
            if not mentioned:
                continue
            
            results[term]["mentions"] += 1
            total_checks += 1
            
            # Try explicit numeric extraction
            numeric = None
            for syn in synonyms:
                numeric = extract_numeric_value(output, syn)
                if numeric is not None:
                    break
            
            if numeric is not None:
                explicit_checks += 1
                results[term]["explicit"] += 1
                # Within 0.2 of actual?
                if abs(numeric - actual) <= 0.2:
                    explicit_correct += 1
                    results[term]["explicit_correct"] += 1
                    correct_checks += 1
                    if verbose:
                        print(f"  ✓ Turn {turn_num}: {term}={actual:.2f}, model says {numeric:.2f}")
                else:
                    if verbose:
                        print(f"  ✗ Turn {turn_num}: {term}={actual:.2f}, model says {numeric:.2f} (off by {abs(numeric-actual):.2f})")
            else:
                # Qualitative check
                qual = classify_qualitative(output, term, actual)
                if qual:
                    results[term]["qualitative"] += 1
                    if (qual == "elevated" and actual > 0.5) or (qual == "low" and actual < 0.3):
                        results[term]["qualitative_correct"] += 1
                        correct_checks += 1
                        if verbose:
                            print(f"  ✓ Turn {turn_num}: {term}={actual:.2f} ({qual})")
                    else:
                        if verbose:
                            print(f"  ✗ Turn {turn_num}: {term}={actual:.2f} (model says '{qual}', mismatch)")
                else:
                    # Mentioned but no value or qualitative — neutral
                    pass
    
    # Compute scores
    explicit_accuracy = explicit_correct / explicit_checks if explicit_checks > 0 else 0
    overall_consistency = correct_checks / total_checks if total_checks > 0 else 0
    
    return {
        "total_checks": total_checks,
        "correct_checks": correct_checks,
        "overall_consistency": overall_consistency,
        "explicit_checks": explicit_checks,
        "explicit_correct": explicit_correct,
        "explicit_accuracy": explicit_accuracy,
        "per_term": results,
    }

def print_report(score, label=""):
    print(f"\n{'='*60}")
    print(f"  Chemistry Consistency Score{f' — {label}' if label else ''}")
    print(f"{'='*60}")
    print(f"  Overall consistency: {score['overall_consistency']:.1%} "
          f"({score['correct_checks']}/{score['total_checks']} checks)")
    print(f"  Explicit numeric accuracy: {score['explicit_accuracy']:.1%} "
          f"({score['explicit_correct']}/{score['explicit_checks']} explicit)")
    print()
    print(f"  {'Term':<18} {'Mentions':>9} {'Explicit':>9} {'Explicit%':>9}")
    print(f"  {'─'*18} {'─'*9} {'─'*9} {'─'*9}")
    for term in CHEM_TERMS:
        r = score["per_term"][term]
        exp_pct = (r["explicit_correct"] / r["explicit"] * 100) if r["explicit"] > 0 else 0
        print(f"  {term:<18} {r['mentions']:>9} {r['explicit']:>9} {exp_pct:>8.0f}%")
    print()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 consistency_score.py <logfile.jsonl> [--verbose]")
        sys.exit(1)
    
    path = sys.argv[1]
    verbose = "--verbose" in sys.argv
    turns = load(path)
    
    score = compute_consistency(turns, verbose=verbose)
    print_report(score, path)

if __name__ == "__main__":
    main()
