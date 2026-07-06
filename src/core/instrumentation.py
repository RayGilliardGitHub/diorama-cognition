#!/usr/bin/env python3
"""
Instrumentation module for the Embodied Cognition Platform.

Logs every inference turn to JSONL for interpretability, debugging,
and post-hoc analysis. Used by all phases (Phase 1+) to produce
structured, queryable logs for node contribution analysis, chemistry
effect measurement, and synthesizer dominance detection.

Dependencies: Python stdlib only (json, datetime, time, os).

Schema
------
Each turn produces one JSONL object with this structure:

{
    "turn": 47,                          # Turn number, 1-indexed
    "timestamp": "2026-06-12T14:32:01Z", # ISO-8601 UTC timestamp
    "input": "user message text",        # The user/environment input

    "nodes": {
        "<node_name>": {
            "output": "...",             # Full output text from the node
            "gate_multiplier": 0.8,      # 0.0-1.0, chemistry-modulated gate weight
            "latency_ms": 245,           # Wall-clock inference time in ms
            "chemistry_at_call": {       # Chemistry snapshot when this node ran
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
        # ... one entry per processing node
    },

    "synthesizer": {
        "output": "...",                 # Final synthesized response
        "latency_ms": 5200,             # Wall-clock time for synthesizer inference
        "chemistry_adjustment": {        # Delta applied to chemistry next turn
            "adrenaline": -0.1,
            "dopamine": 0.05,
            "serotonin": 0.0,
            ...
        },
        "reported_weighting": {          # Synthesizer's self-reported importance
            "edge_pattern_detector": 0.3,
            "emotional_valuator": 0.5,
            ...
        }
    },

    "chemistry_state": {                 # Full chemistry vector at turn end
        "adrenaline": 0.2,
        "noradrenaline": 0.0,
        "dopamine": 0.5,
        "serotonin": 0.6,
        "cortisol": 0.1,
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

Usage
-----
Recommended for Phase 1 orchestrator -- context manager pattern:

    from instrumentation import TurnLogger, DEFAULT_CHEMISTRY

    chemistry = dict(DEFAULT_CHEMISTRY)
    chemistry["adrenaline"] = 0.2

    with TurnLogger("logs/experiment.jsonl", turn=1,
                    user_input="Hello", chemistry_state=chemistry) as log:

        result = ollama.chat(model="llama3.2:1b-focus", messages=[...])
        log.log_node(
            "edge_pattern_detector",
            output=result["message"]["content"],
            gate_multiplier=0.8,
            latency_ms=245,
            chemistry_at_call=chemistry
        )

        # ... more nodes ...

        synth_result = ollama.chat(model="lfm2.5:8b-bal", messages=[...])
        log.finish(
            output=synth_result["message"]["content"],
            latency_ms=5200,
            chemistry_adjustment={"adrenaline": -0.1, "dopamine": 0.05},
            reported_weighting={"edge_pattern_detector": 0.3, ...}
        )

    # Record is written automatically on exit.
"""

import json
import os
import time
import datetime
from functools import wraps


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

# Seven neurotransmitters from §6.1 of the platform plan.
NEUROTRANSMITTERS = [
    "adrenaline",
    "noradrenaline",
    "dopamine",
    "serotonin",
    "cortisol",
    "oxytocin",
    "endorphins",
]

# Six body state variables from §6.2 of the platform plan.
BODY_STATE_VARS = [
    "heart_rate",
    "body_temperature",
    "hunger",
    "fatigue",
    "pain",
    "arousal",
]

# All chemistry vector keys.
ALL_CHEMISTRY_KEYS = NEUROTRANSMITTERS + BODY_STATE_VARS

# The 14 primary processing nodes from §5.1 of the platform plan,
# snake_cased, in registry order.
PRIMARY_NODES = [
    "edge_pattern_detector",
    "threat_router",
    "emotional_valuator",
    "episodic_retriever",
    "planner",
    "conflict_monitor",
    "body_regulator",
    "motor_selector",
    "fine_coordinator",
    "interoceptive_evaluator",
    "idle_reflector",
    "language_comprehender",
    "language_producer",
    "social_predictor",
]

# Default resting-state chemistry (mid-range serotonin, everything else low).
DEFAULT_CHEMISTRY = {
    "adrenaline": 0.0,
    "noradrenaline": 0.0,
    "dopamine": 0.0,
    "serotonin": 0.5,
    "cortisol": 0.0,
    "oxytocin": 0.0,
    "endorphins": 0.0,
    "heart_rate": 70.0,
    "body_temperature": 37.0,
    "hunger": 0.3,
    "fatigue": 0.1,
    "pain": 0.0,
    "arousal": 0.3,
}


def validate_chemistry(chemistry, label="chemistry"):
    """Validate that a chemistry dict has all required keys and valid ranges.

    Raises ValueError on first violation. Returns the dict unchanged
    if valid (for chaining).

    Parameters
    ----------
    chemistry : dict
        Chemistry state dict to validate.
    label : str
        Human-readable label for error messages.
    """
    if not isinstance(chemistry, dict):
        raise TypeError(f"{label} must be a dict, got {type(chemistry).__name__}")

    for key in ALL_CHEMISTRY_KEYS:
        if key not in chemistry:
            raise ValueError(f"{label} missing required key: {key!r}")

    for key in chemistry:
        if key not in ALL_CHEMISTRY_KEYS:
            raise ValueError(
                f"{label} contains unknown key: {key!r}. "
                f"Known keys: {ALL_CHEMISTRY_KEYS}"
            )

    for nt in NEUROTRANSMITTERS:
        val = chemistry[nt]
        if not isinstance(val, (int, float)):
            raise TypeError(f"{label}.{nt} must be numeric, got {type(val).__name__}")
        if val < 0.0 or val > 1.0:
            raise ValueError(f"{label}.{nt} must be in [0.0, 1.0], got {val}")

    return chemistry


# ═══════════════════════════════════════════════════════════════
# TurnLogger
# ═══════════════════════════════════════════════════════════════


class TurnLogger:
    """Context manager that records one inference turn and appends it to JSONL.

    Parameters
    ----------
    log_path : str
        Path to the JSONL log file. Created/opened in append mode.
        Intermediate directories are created automatically.
    turn_number : int
        Turn number (1-indexed). Used as the ``turn`` field.
    user_input : str
        The user message or environmental input for this turn.
    chemistry_state : dict or None
        Full chemistry vector at turn start. If None, uses
        ``DEFAULT_CHEMISTRY``. A shallow copy is stored so the
        caller's dict is not modified.

    Examples
    --------
    >>> with TurnLogger("/tmp/test.jsonl", turn=1, user_input="Hello",
    ...                 chemistry_state=DEFAULT_CHEMISTRY) as log:
    ...     log.log_node(
    ...         "edge_pattern_detector",
    ...         output="metallic object, raised arm",
    ...         gate_multiplier=0.8,
    ...         latency_ms=245,
    ...         chemistry_at_call=DEFAULT_CHEMISTRY
    ...     )
    ...     log.finish(
    ...         output="I see you raised an object.",
    ...         latency_ms=3400,
    ...         chemistry_adjustment={"adrenaline": -0.05},
    ...         reported_weighting={"edge_pattern_detector": 1.0}
    ...     )
    """

    def __init__(
        self,
        log_path,
        turn_number,
        user_input,
        chemistry_state=None,
    ):
        self.log_path = os.path.abspath(log_path)
        self.turn_number = int(turn_number)
        self.user_input = str(user_input)
        self.chemistry_state = dict(chemistry_state or DEFAULT_CHEMISTRY)

        # Accumulated node entries (logged via log_node)
        self.nodes = {}

        # Synthesizer entry (set via finish(), or auto-synthesized on exit)
        self.synthesizer_entry = {
            "output": "",
            "latency_ms": 0,
            "chemistry_adjustment": {},
            "reported_weighting": {},
        }

        self._start_time = time.perf_counter()
        self._exited = False

    # ── Public API ──────────────────────────────────────────────

    def log_node(
        self,
        node_name,
        output,
        gate_multiplier=1.0,
        latency_ms=0,
        chemistry_at_call=None,
    ):
        """Record a processing node's output for this turn.

        Parameters
        ----------
        node_name : str
            Unique identifier for the node (e.g. ``"threat_router"``).
            Must match the plan's Node Registry names.
        output : str
            Full text output from the node's inference.
        gate_multiplier : float
            Effective gate weight (0.0-1.0) after chemistry modulation.
        latency_ms : int
            Wall-clock inference time in milliseconds.
        chemistry_at_call : dict or None
            Chemistry snapshot at the moment this node was invoked.
            If None, uses the turn's initial chemistry_state.
        """
        self.nodes[node_name] = {
            "output": str(output),
            "gate_multiplier": float(gate_multiplier),
            "latency_ms": int(latency_ms),
            "chemistry_at_call": dict(
                chemistry_at_call or self.chemistry_state
            ),
        }

    # ── Synthesizer record ──────────────────────────────────────

    def finish(
        self,
        output,
        latency_ms=0,
        chemistry_adjustment=None,
        reported_weighting=None,
    ):
        """Record the synthesizer's output and finalise the turn.

        Must be called exactly once per turn, before the context manager
        exits. If not called before ``__exit__``, an empty synthesizer
        record is written as a placeholder.

        Parameters
        ----------
        output : str
            The final synthesizer response text.
        latency_ms : int
            Wall-clock synthesizer inference time in milliseconds.
        chemistry_adjustment : dict or None
            Delta vector for chemistry changes to apply next turn.
            Only keys that changed need be present.
        reported_weighting : dict or None
            Self-reported per-node importance weights from the
            synthesizer (e.g., ``{"threat_router": 0.8, ...}``).
        """
        chemistry_adjustment = chemistry_adjustment or {}
        reported_weighting = reported_weighting or {}
        if not isinstance(chemistry_adjustment, dict):
            chemistry_adjustment = {}
        if not isinstance(reported_weighting, dict):
            reported_weighting = {}
        self.synthesizer_entry = {
            "output": str(output),
            "latency_ms": int(latency_ms),
            "chemistry_adjustment": dict(chemistry_adjustment),
            "reported_weighting": dict(reported_weighting),
        }

    # ── Context manager protocol ────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._exited:
            self._exited = True
            self._write_record()
        return False  # do not suppress exceptions

    # ── Internal helpers ────────────────────────────────────────

    def _build_record(self):
        """Assemble the full turn record dict from accumulated state."""
        return {
            "turn": self.turn_number,
            "timestamp": (
                datetime.datetime.now(datetime.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            ),
            "input": self.user_input,
            "nodes": dict(self.nodes),
            "synthesizer": dict(self.synthesizer_entry),
            "chemistry_state": dict(self.chemistry_state),
        }

    def _write_record(self):
        """Serialize the turn record and append it to the JSONL file."""
        record = self._build_record()
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════════
# Decorator interface (alternative API for simple scripts)
# ═══════════════════════════════════════════════════════════════


def instrumented(log_path="instrumentation.jsonl"):
    """Decorator that wraps a function as an instrumented turn.

    The wrapped function must accept ``(turn, user_input, chemistry_state)``
    plus any keyword arguments, and must return a dict with keys:
        - ``"nodes"``: list of node result dicts, each with keys:
            ``node_name``, ``output``, ``gate_multiplier``,
            ``latency_ms``, ``chemistry_at_call``
        - ``"synthesizer_output"``: str
        - ``"synthesizer_latency_ms"``: int
        - ``"chemistry_adjustment"``: dict
        - ``"reported_weighting"``: dict

    The decorated function's return value is the same dict (passed through).

    Parameters
    ----------
    log_path : str
        Path to the JSONL output file. Defaults to
        ``instrumentation.jsonl`` in the current directory.

    Examples
    --------
    >>> @instrumented("my_log.jsonl")
    ... def my_turn(turn, user_input, chemistry_state, **kw):
    ...     # ... run nodes ...
    ...     return {
    ...         "nodes": [
    ...             {
    ...                 "node_name": "edge_pattern_detector",
    ...                 "output": "detected X",
    ...                 "gate_multiplier": 0.8,
    ...                 "latency_ms": 245,
    ...                 "chemistry_at_call": chemistry_state,
    ...             }
    ...         ],
    ...         "synthesizer_output": "final response",
    ...         "synthesizer_latency_ms": 3400,
    ...         "chemistry_adjustment": {"adrenaline": -0.05},
    ...         "reported_weighting": {"edge_pattern_detector": 1.0},
    ...     }
    """

    def decorator(func):
        @wraps(func)
        def wrapper(turn, user_input, chemistry_state=None, **kwargs):
            chemistry = dict(chemistry_state or DEFAULT_CHEMISTRY)
            with TurnLogger(
                log_path,
                turn_number=turn,
                user_input=user_input,
                chemistry_state=chemistry,
            ) as logger:
                result = func(turn, user_input, chemistry, **kwargs)

                # Log nodes from the returned list
                for node in result.get("nodes", []):
                    logger.log_node(
                        node_name=node["node_name"],
                        output=node.get("output", ""),
                        gate_multiplier=node.get("gate_multiplier", 1.0),
                        latency_ms=node.get("latency_ms", 0),
                        chemistry_at_call=node.get(
                            "chemistry_at_call", chemistry
                        ),
                    )

                # Record synthesizer
                logger.finish(
                    output=result.get("synthesizer_output", ""),
                    latency_ms=result.get("synthesizer_latency_ms", 0),
                    chemistry_adjustment=result.get(
                        "chemistry_adjustment", {}
                    ),
                    reported_weighting=result.get("reported_weighting", {}),
                )

            return result

        return wrapper

    return decorator


# ═══════════════════════════════════════════════════════════════
# Self-test (run as script)
# ═══════════════════════════════════════════════════════════════

# Schema keys that must exist in every turn record.
REQUIRED_TOP_LEVEL = {"turn", "timestamp", "input", "nodes", "synthesizer", "chemistry_state"}
REQUIRED_NODE_FIELDS = {"output", "gate_multiplier", "latency_ms", "chemistry_at_call"}
REQUIRED_SYNTHESIZER_FIELDS = {"output", "latency_ms", "chemistry_adjustment", "reported_weighting"}
REQUIRED_CHEMISTRY_FIELDS = set(ALL_CHEMISTRY_KEYS)
REQUIRED_CHEM_AT_CALL_FIELDS = set(ALL_CHEMISTRY_KEYS)


def _make_synthetic_nodes(chemistry, base_text="Synthetic output"):
    """Generate realistic-looking synthetic node outputs for all 14 nodes."""
    import random

    random.seed(42)  # deterministic for reproducibility

    synthetic_outputs = {
        "edge_pattern_detector": "Edge/object detection: human figure, metallic reflection, raised angle pattern detected.",
        "threat_router": "Threat assessment: object at 45deg angle, approaching vector. Threat level: MODERATE.",
        "emotional_valuator": "Emotional valence: NEGATIVE. Fear response activated. Signal urgency: HIGH.",
        "episodic_retriever": "Pattern completion: recalling previous instances of raised-object encounters. 3 episodes match.",
        "planner": "Action plan: (1) assess intent, (2) prepare defensive posture, (3) evaluate escape routes.",
        "conflict_monitor": "Error monitoring: detected conflict between threat assessment (HIGH) and emotional state (FEAR).",
        "body_regulator": "Body regulation: increasing heart rate, releasing adrenaline. Preparing stress response.",
        "motor_selector": "Motor selection: tense shoulder girdle, shift weight to rear foot, prepare evasion.",
        "fine_coordinator": "Fine coordination: maintaining gaze fixation, adjusting hand position for readiness.",
        "interoceptive_evaluator": "Interoception: elevated heart rate, shallow breathing, increased muscle tension detected.",
        "idle_reflector": "Default mode: comparing current scenario to past similar experiences. Self-referential processing active.",
        "language_comprehender": "Language comprehension: user query parsed. Intent: request for analysis. Tone: neutral.",
        "language_producer": "Language production: formulating response with modulated tone. Choosing words for clarity.",
        "social_predictor": "Social prediction: interlocutor likely expects a measured, informative response. Power dynamic: equal.",
    }

    nodes = {}
    for node_name in PRIMARY_NODES:
        raw_text = synthetic_outputs.get(
            node_name, f"{node_name} output for '{base_text}'"
        )
        gating = random.uniform(0.3, 1.0)
        nodes[node_name] = {
            "node_name": node_name,
            "output": raw_text,
            "gate_multiplier": round(gating, 2),
            "latency_ms": random.randint(100, 5000),
            "chemistry_at_call": dict(chemistry),
        }
    return nodes


def _make_synthetic_weighting():
    """Generate realistic synthetic reported weighting for all nodes."""
    import random

    random.seed(42)
    weights = {}
    remaining = 1.0
    for i, node in enumerate(PRIMARY_NODES):
        if i == len(PRIMARY_NODES) - 1:
            weights[node] = round(remaining, 3)
        else:
            w = round(random.uniform(0.01, remaining / (len(PRIMARY_NODES) - i)), 3)
            weights[node] = w
            remaining -= w
    # Normalize to ensure sum = 1.0
    total = sum(weights.values())
    if total > 0:
        for k in weights:
            weights[k] = round(weights[k] / total, 3)
    return weights


def run_synthetic_test(log_path="/tmp/instrumentation_test.jsonl"):
    """Generate 3 turns of synthetic data, verify schema, print report."""
    import random

    random.seed(42)
    print("=" * 68)
    print("  INSTRUMENTATION MODULE — SYNTHETIC SELF-TEST")
    print("=" * 68)

    # Chemistry profiles from §7.2 of the platform plan
    chem_profiles = [
        {
            "name": "resting",
            "chemistry": {
                "adrenaline": 0.05,
                "noradrenaline": 0.1,
                "dopamine": 0.1,
                "serotonin": 0.6,
                "cortisol": 0.05,
                "oxytocin": 0.2,
                "endorphins": 0.1,
                "heart_rate": 68.0,
                "body_temperature": 36.9,
                "hunger": 0.25,
                "fatigue": 0.15,
                "pain": 0.0,
                "arousal": 0.25,
            },
            "input": "What's the weather like today?",
        },
        {
            "name": "alert",
            "chemistry": {
                "adrenaline": 0.5,
                "noradrenaline": 0.6,
                "dopamine": 0.3,
                "serotonin": 0.4,
                "cortisol": 0.2,
                "oxytocin": 0.1,
                "endorphins": 0.0,
                "heart_rate": 92.0,
                "body_temperature": 37.2,
                "hunger": 0.2,
                "fatigue": 0.1,
                "pain": 0.0,
                "arousal": 0.7,
            },
            "input": "Did you hear that sound? What was it?",
        },
        {
            "name": "content",
            "chemistry": {
                "adrenaline": 0.02,
                "noradrenaline": 0.05,
                "dopamine": 0.5,
                "serotonin": 0.85,
                "cortisol": 0.02,
                "oxytocin": 0.6,
                "endorphins": 0.3,
                "heart_rate": 65.0,
                "body_temperature": 36.8,
                "hunger": 0.4,
                "fatigue": 0.2,
                "pain": 0.0,
                "arousal": 0.15,
            },
            "input": "Tell me a story about something beautiful.",
        },
    ]

    # ── Generate turns ──────────────────────────────────────────
    print("\n  Generating 3 synthetic turns...")
    for i, profile in enumerate(chem_profiles):
        turn_num = i + 1
        chem = profile["chemistry"]
        nodes_data = _make_synthetic_nodes(chem, profile["input"])
        weighting = _make_synthetic_weighting()

        with TurnLogger(
            log_path,
            turn_number=turn_num,
            user_input=profile["input"],
            chemistry_state=chem,
        ) as logger:
            for node_result in nodes_data.values():
                logger.log_node(
                    node_name=node_result["node_name"],
                    output=node_result["output"],
                    gate_multiplier=node_result["gate_multiplier"],
                    latency_ms=node_result["latency_ms"],
                    chemistry_at_call=node_result["chemistry_at_call"],
                )
            logger.finish(
                output=(
                    f"Synthesized response for '{profile['input']}' "
                    f"under {profile['name']} chemistry."
                ),
                latency_ms=random.randint(3000, 8000),
                chemistry_adjustment={
                    "adrenaline": -0.02,
                    "dopamine": 0.01,
                    "serotonin": 0.0,
                },
                reported_weighting=weighting,
            )

        print(
            f"    Turn {turn_num:>2} ({profile['name']:>8}): "
            f"14 nodes + synthesizer → {log_path}"
        )

    # ── Verify ──────────────────────────────────────────────────
    print("\n  Verifying JSONL output...")
    with open(log_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    total_checks = 0
    passed_checks = 0
    failures = []

    def check(condition, description):
        nonlocal total_checks, passed_checks
        total_checks += 1
        if condition:
            passed_checks += 1
        else:
            failures.append(description)

    for record in records:
        turn = record.get("turn")

        # Top-level keys
        for key in REQUIRED_TOP_LEVEL:
            check(
                key in record,
                f"Turn {turn}: missing top-level key '{key}'",
            )

        # Timestamp format
        ts = record.get("timestamp", "")
        check(
            ts.endswith("Z") and "T" in ts,
            f"Turn {turn}: timestamp missing or not ISO-8601: {ts!r}",
        )

        # Node entries
        nodes = record.get("nodes", {})
        for node_name in PRIMARY_NODES:
            check(
                node_name in nodes,
                f"Turn {turn}: missing node '{node_name}'",
            )
            if node_name in nodes:
                node = nodes[node_name]
                for field in REQUIRED_NODE_FIELDS:
                    check(
                        field in node,
                        f"Turn {turn}, node '{node_name}': "
                        f"missing field '{field}'",
                    )
                # Chemistry at call has all required keys
                chem_at_call = node.get("chemistry_at_call", {})
                for k in REQUIRED_CHEM_AT_CALL_FIELDS:
                    check(
                        k in chem_at_call,
                        f"Turn {turn}, node '{node_name}': "
                        f"missing chemistry key '{k}'",
                    )

        # Synthesizer entry
        synth = record.get("synthesizer", {})
        for field in REQUIRED_SYNTHESIZER_FIELDS:
            check(
                field in synth,
                f"Turn {turn}: synthesizer missing field '{field}'",
            )

        # Reported weighting should cover nodes that were logged
        weighting = synth.get("reported_weighting", {})
        for node_name in PRIMARY_NODES:
            check(
                node_name in weighting,
                f"Turn {turn}: weighting missing node '{node_name}'",
            )

        # Chemistry state has all required keys
        chem_state = record.get("chemistry_state", {})
        for k in REQUIRED_CHEMISTRY_FIELDS:
            check(
                k in chem_state,
                f"Turn {turn}: chemistry_state missing key '{k}'",
            )

        # Chemistry values in range [0, 1] for NTs
        for chem_name in NEUROTRANSMITTERS:
            val = chem_state.get(chem_name, -1)
            check(
                0.0 <= val <= 1.0,
                f"Turn {turn}: {chem_name}={val} out of range [0,1]",
            )

        # Input is a string
        check(
            isinstance(record.get("input"), str),
            f"Turn {turn}: input is not a string",
        )

        # Turn number is integer
        check(
            isinstance(turn, int),
            f"Turn {turn}: turn number is not int",
        )

    # ── Completeness Report ─────────────────────────────────────
    print(f"\n  {'─' * 54}")
    print(f"  COMPLETENESS REPORT")
    print(f"  {'─' * 54}")
    print(f"    Records written:          {len(records)}")
    print(f"    Expected turns:           3")
    print(f"    Total schema checks:      {total_checks}")
    print(f"    Passed:                   {passed_checks}")
    print(f"    Failed:                   {total_checks - passed_checks}")
    print(f"    Log file:                 {os.path.abspath(log_path)}")

    # Node coverage
    if records:
        node_coverage = {}
        for node_name in PRIMARY_NODES:
            present_count = sum(
                1 for r in records if node_name in r.get("nodes", {})
            )
            node_coverage[node_name] = f"{present_count}/{len(records)}"

        print(f"\n    Node coverage (present/total turns):")
        for node_name in PRIMARY_NODES:
            print(f"      {node_name:>30s}: {node_coverage[node_name]}")

    # Failures
    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    ✗ {f}")
    else:
        print(f"\n  ✓ ALL CHECKS PASSED")

    print(f"\n  {'─' * 54}")
    print(f"  Sample record (Turn 1, abbreviated):")
    print(f"  {'─' * 54}")

    if records:
        r1 = records[0]
        print(f"    turn:         {r1['turn']}")
        print(f"    timestamp:    {r1['timestamp']}")
        print(f"    input:        {r1['input'][:60]}")
        print(f"    nodes:        {len(r1['nodes'])} nodes")
        print(f"    synthesizer output length: {len(r1['synthesizer']['output'])} chars")
        print(f"    chemistry_state keys: {len(r1['chemistry_state'])}")

    print("\n" + "=" * 68)

    return passed_checks == total_checks


if __name__ == "__main__":
    import sys

    success = run_synthetic_test()
    sys.exit(0 if success else 1)
