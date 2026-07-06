#!/usr/bin/env python3
"""
Diorama Store — a reconsolidation-based memory system.

Inspired by the Kurzgesagt video on memory ("Your Brain is Weird Madness").
Every memory is a sparse pattern; every recall rewrites the trace.

Core concepts:
  - Diorama: a stored memory pattern + strength + lineage of rewrites
  - SynapticField: Hebbian weight matrix between pattern dimensions
  - Hippocampus: approximate nearest-neighbor index for cue-based retrieval
  - Reconsolidation: every read blends current context into the stored pattern
  - Wax Temperature: global plasticity (modulated by chemistry)
  - Decay: unreinforced memories fade over time

Usage:
    from src.core.diorama import DioramaStore

    store = DioramaStore(n_dims=2000, sparsity=0.05)
    cue = store.make_pattern({"sensory": "It is 14:47 UTC.", ...})
    memory = store.retrieve(cue)   # returns Diorama, mutates it
    store.tick(dt=10.0)            # decay unreferenced memories
    store.consolidate()            # "sleep" — replay + strengthen

Dependencies: numpy
"""
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

DEFAULT_N_DIMS = 2000
"""Pattern dimensionality. Higher = more capacity, more compute."""

DEFAULT_SPARSITY = 0.05
"""Fraction of dimensions active in a typical pattern (~100 active dims at 2000)."""

DEFAULT_LEARNING_RATE = 0.01
"""Hebbian update strength per encode/retrieve."""

DEFAULT_DECAY_RATE = 0.9995
"""Per-tick multiplicative weight decay. Higher = slower forgetting."""

DEFAULT_STRENGTH_DECAY = 0.98
"""Per-tick multiplicative strength decay for unrecalled dioramas."""

CONSOLIDATE_STRENGTH_GAIN = 0.05
"""Strength increase per consolidation replay."""

CONSOLIDATE_PATTERN_REPLAYS = 5
"""Number of Hebbian replays per consolidation."""

MERGE_OVERLAP_THRESHOLD = 0.85
"""If two dioramas overlap this much, they may be merged."""

FORGET_STRENGTH_THRESHOLD = 0.03
"""Dioramas below this strength are pruned."""


# ═══════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════


@dataclass
class LineageEntry:
    """Record of a single reconsolidation event (one read that mutated)."""

    turn: int
    """Turn number when this reconsolidation occurred."""

    wax_temp: float
    """Wax temperature at time of mutation (0.0-1.0)."""

    blend_ratio: float
    """How much current context was blended in (derived from wax_temp)."""

    context_tags: List[str] = field(default_factory=list)
    """Tags from the retrieval context (e.g. ['boss', 'formal'])."""


@dataclass
class Diorama:
    """A stored memory — the distributed pattern plus metadata.

    The ``pattern`` IS the memory. Every reconsolidation mutates it.
    The ``lineage`` is an append-only audit trail of those mutations.
    """

    pattern: np.ndarray
    """Sparse binary vector — the 'assembly'."""

    strength: float = 0.3
    """Consolidation level. 0.0 = forgotten, 1.0 = maximally strong."""

    # -- Metadata --
    created_at: int = 0
    """Turn number when this diorama was first encoded."""

    recall_count: int = 0
    """How many times this diorama has been retrieved."""

    last_retrieved: int = 0
    """Turn number of most recent retrieval."""

    tags: set = field(default_factory=set)
    """Associative tags for index lookup (e.g. {'boss', 'bad_news'})."""

    # -- Reconsolidation audit trail --
    lineage: List[LineageEntry] = field(default_factory=list)
    """Every mutation recorded here. Max 50 entries (oldest dropped)."""

    def __post_init__(self):
        if isinstance(self.tags, list):
            self.tags = set(self.tags)

    def add_lineage(self, turn: int, wax_temp: float,
                    blend_ratio: float, context_tags: Optional[List[str]] = None):
        """Append a reconsolidation event to this diorama's history."""
        entry = LineageEntry(
            turn=turn,
            wax_temp=round(wax_temp, 3),
            blend_ratio=round(blend_ratio, 4),
            context_tags=context_tags or [],
        )
        self.lineage.append(entry)
        if len(self.lineage) > 50:
            self.lineage = self.lineage[-50:]

    def drift_distance(self) -> float:
        """Measure how far the pattern has drifted from the original encoding.

        Returns euclidean distance between the first and current pattern.
        Returns 0.0 if no lineage exists (single pattern).
        """
        if len(self.lineage) < 1:
            return 0.0
        # We don't store the original pattern separately, but the drift
        # is implicit in the count and depth of reconsolidations.
        # Proxy: blend_ratio * recall_count
        total_blend = sum(e.blend_ratio for e in self.lineage)
        return min(1.0, total_blend)

    def __repr__(self):
        return (
            f"Diorama(tags={self.tags}, strength={self.strength:.2f}, "
            f"recalls={self.recall_count}, lineage={len(self.lineage)})"
        )


# ═══════════════════════════════════════════════════════════════
# DioramaStore
# ═══════════════════════════════════════════════════════════════


class DioramaStore:
    """Pattern-completion memory store with reconsolidation.

    Parameters
    ----------
    n_dims : int
        Dimensionality of the pattern space (default 2000).
    sparsity : float
        Fraction of active bits in a typical pattern (default 0.05).
    learning_rate : float
        Hebbian update strength (default 0.01).
    decay_rate : float
        Per-tick weight decay multiplier (default 0.9995).
    strength_decay : float
        Per-tick strength decay for unrecalled dioramas (default 0.98).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_dims: int = DEFAULT_N_DIMS,
        sparsity: float = DEFAULT_SPARSITY,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        decay_rate: float = DEFAULT_DECAY_RATE,
        strength_decay: float = DEFAULT_STRENGTH_DECAY,
        seed: Optional[int] = None,
    ):
        self.n_dims = n_dims
        self.sparsity = sparsity
        self.lr = learning_rate
        self.decay_rate = decay_rate
        self.strength_decay = strength_decay

        # Synaptic field — Hebbian weight matrix (sparse upper-triangle)
        # Keeping full matrix for simplicity; in production this would be sparse.
        self.weights = np.zeros((n_dims, n_dims), dtype=np.float32)

        # Stored dioramas
        self.dioramas: List[Diorama] = []

        # Hippocampus index: tag -> set of diorama indices
        self.tag_index: Dict[str, set] = defaultdict(set)

        # Global plasticity
        self.wax_temp: float = 0.3  # default mid-range

        # Runtime state
        self.current_turn: int = 0
        self.rng = np.random.default_rng(seed)

        # Diaroma ID allocator
        self._next_id: int = 0

    # ── Pattern creation ───────────────────────────────────────

    def make_pattern(self, features: Dict[str, str]) -> np.ndarray:
        """Create a sparse binary pattern from a dictionary of text features.

        Each key corresponds to a region in the pattern space
        (e.g. sensory, emotional, episodic, social, chemistry).
        The text from each feature is hashed to set bits in its region.

        Parameters
        ----------
        features : dict of str -> str
            Named features to encode. Keys determine the region mapping.

        Returns
        -------
        np.ndarray
            Binary sparse vector of shape (n_dims,).
        """
        pattern = np.zeros(self.n_dims, dtype=np.float32)
        n_regions = max(len(features), 1)
        region_size = self.n_dims // n_regions

        for i, (key, text) in enumerate(features.items()):
            start = i * region_size
            end = start + region_size if i < n_regions - 1 else self.n_dims
            region_len = end - start
            bits = max(1, int(region_len * self.sparsity))

            # Deterministic hashing of text into region bits
            h = hash(f"{key}:{text}")
            for j in range(bits):
                idx = (h + j * 10007) % region_len
                pattern[start + idx] = 1.0

        return pattern

    def make_pattern_from_components(
        self,
        node_outputs: Dict[str, str],
        chemistry: Optional[Dict[str, float]] = None,
        response: str = "",
    ) -> np.ndarray:
        """Convenience: create pattern from the full turn state.

        Maps directly to the Phase 2/3 architecture: each processing
        node's output occupies a region of the pattern space, plus
        the chemistry state and synthesizer response.
        """
        features = dict(node_outputs)
        if chemistry:
            chem_str = ", ".join(f"{k}={v:.2f}" for k, v in chemistry.items())
            features["chemistry"] = chem_str
        if response:
            features["response"] = response
        return self.make_pattern(features)

    # ── Core operations ────────────────────────────────────────

    def encode(self, pattern: np.ndarray, tags: Optional[List[str]] = None,
               turn: Optional[int] = None) -> Diorama:
        """Store a new pattern as a diorama.

        Parameters
        ----------
        pattern : np.ndarray
            Binary sparse vector to store.
        tags : list of str or None
            Tags for the hippocampus index.
        turn : int or None
            Current turn number (defaults to internal counter).

        Returns
        -------
        Diorama
            The newly created diorama.
        """
        if turn is not None:
            self.current_turn = turn

        # Check for near-duplicate — novelty gate from the video
        dup = self._find_best_match(pattern)
        if dup is not None and self._overlap(dup.pattern, pattern) > 0.8:
            # Strengthen existing instead of creating new
            dup.strength = min(1.0, dup.strength + 0.1)
            dup.tags.update(tags or [])
            return dup

        # Create new diorama
        d = Diorama(
            pattern=pattern.copy(),
            strength=0.3,
            created_at=self.current_turn,
            tags=set(tags or []),
        )

        # Hebbian update: co-active bits strengthen their connection
        self._hebbian_update(pattern)

        self.dioramas.append(d)
        idx = len(self.dioramas) - 1

        # Update tag index
        for tag in (tags or []):
            self.tag_index[tag].add(idx)

        return d

    def retrieve(self, cue: np.ndarray, tags: Optional[List[str]] = None,
                 turn: Optional[int] = None) -> Optional[Diorama]:
        """Retrieve a memory by pattern completion.

        Finds the best matching diorama, applies reconsolidation
        (mutates the stored pattern), and returns the memory.

        Parameters
        ----------
        cue : np.ndarray
            Partial pattern — the retrieval cue.
        tags : list of str or None
            Optional tag filter (only search indexed dioramas).
        turn : int or None
            Current turn number (defaults to internal counter).

        Returns
        -------
        Diorama or None
            The matched (and mutated) diorama, or None if no match found.
        """
        if turn is not None:
            self.current_turn = turn
        if not self.dioramas:
            return None

        # 1. Pattern completion via attractor dynamics
        completed = self._attractor_dynamics(cue)

        # 2. Competition — find nearest stored diorama
        candidate_indices = self._candidate_indices(tags)
        if not candidate_indices:
            return None

        winner_idx = max(
            candidate_indices,
            key=lambda i: self._overlap(self.dioramas[i].pattern, completed)
                          * self.dioramas[i].strength
        )
        winner = self.dioramas[winner_idx]

        best_overlap = self._overlap(winner.pattern, completed)
        if best_overlap < 0.1:
            return None  # No good match

        # 3. RECONSOLIDATION — every read mutates the memory
        blend_ratio = self.wax_temp * 0.15
        if blend_ratio > 0.001:
            # Get current context as a pattern (temperature reading)
            context = self._current_context_pattern()
            # Blend: new_pattern = (1 - blend) * original + blend * context
            winner.pattern = (
                (1.0 - blend_ratio) * winner.pattern
                + blend_ratio * context
            )
            # Binarize back to sparse
            winner.pattern = (winner.pattern > 0.5).astype(np.float32)

            # Re-Hebbian (wax re-hardens in new shape)
            self._hebbian_update(winner.pattern, scale=blend_ratio)

        # 4. Update metadata
        context_tags = tags or []
        winner.add_lineage(
            turn=self.current_turn,
            wax_temp=self.wax_temp,
            blend_ratio=blend_ratio,
            context_tags=context_tags,
        )
        winner.recall_count += 1
        winner.last_retrieved = self.current_turn
        winner.tags.update(context_tags)

        return winner

    def consolidate(self, quality: float = 1.0):
        """Offline consolidation — 'sleep' mode.

        Replays strong memories to strengthen them, without new input.
        The quality parameter simulates sleep quality (0.0 = poor, 1.0 = restful).

        Parameters
        ----------
        quality : float
            Sleep quality 0.0-1.0. Affects replay count and pruning strictness.
        """
        n_replays = max(1, int(CONSOLIDATE_PATTERN_REPLAYS * quality))
        strength_gain = CONSOLIDATE_STRENGTH_GAIN * quality

        consolidated = 0
        for d in self.dioramas:
            if d.strength > 0.2:
                for _ in range(n_replays):
                    self._hebbian_update(d.pattern, scale=0.02)
                d.strength = min(1.0, d.strength + strength_gain)
                consolidated += 1

        # Pruning — worse sleep means more forgetting
        threshold = FORGET_STRENGTH_THRESHOLD
        if quality < 0.5:
            threshold = 0.15  # More aggressive forgetting with poor sleep
        pruned = self._prune(threshold=threshold)

        return {"consolidated": consolidated, "pruned": pruned}

    def tick(self, dt: float = 1.0):
        """Advance time — apply decay and forgetting.

        Parameters
        ----------
        dt : float
            Simulated seconds since last tick.
        """
        # Decay synaptic weights
        self.weights *= self.decay_rate ** dt

        # Decay unreferenced diorama strengths
        for d in self.dioramas:
            if d.last_retrieved < self.current_turn:
                d.strength *= self.strength_decay ** dt

        # Prune forgotten dioramas
        self._prune(threshold=FORGET_STRENGTH_THRESHOLD)

    # ── Wax temperature ────────────────────────────────────────

    def set_wax_temp_from_chemistry(self, chemistry: Dict[str, float]):
        """Derive wax_temp from the body_state chemistry vector.

        Follows the video: high arousal = more plasticity (meltable wax),
        high serotonin/stability = hardened wax.
        """
        arousal = chemistry.get("adrenaline", 0.0) + chemistry.get("noradrenaline", 0.0)
        stability = chemistry.get("serotonin", 0.5) + chemistry.get("oxytocin", 0.0)
        stress = chemistry.get("cortisol", 0.0)

        # Core plasticity from arousal, dampened by stability
        base = 0.2 + 0.5 * arousal - 0.3 * stability
        base = max(0.1, min(1.0, base))

        # Stress adds noise — memories become less reliable
        noise = 0.1 * stress

        self.wax_temp = max(0.1, min(1.0, base + noise))

    # ── Inspection ─────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics about the store."""
        if not self.dioramas:
            return {
                "n_dioramas": 0,
                "n_dims": self.n_dims,
                "wax_temp": round(self.wax_temp, 3),
                "avg_strength": 0,
                "total_recalls": 0,
                "forgotten": 0,
            }

        strengths = [d.strength for d in self.dioramas]
        total_recalls = sum(d.recall_count for d in self.dioramas)
        forgotten = sum(1 for d in self.dioramas if d.strength < FORGET_STRENGTH_THRESHOLD)

        return {
            "n_dioramas": len(self.dioramas),
            "n_dims": self.n_dims,
            "wax_temp": round(self.wax_temp, 3),
            "avg_strength": round(float(np.mean(strengths)), 3),
            "avg_strength_active": round(
                float(np.mean([s for s in strengths if s > FORGET_STRENGTH_THRESHOLD])), 3
            ) if any(s > FORGET_STRENGTH_THRESHOLD for s in strengths) else 0,
            "total_recalls": total_recalls,
            "forgotten": forgotten,
            "tags": dict(self.tag_index),
        }

    def get_lineage(self, diorama: Diorama) -> List[Dict[str, Any]]:
        """Get a serializable lineage trace for a diorama."""
        return [
            {
                "turn": e.turn,
                "wax_temp": e.wax_temp,
                "blend_ratio": e.blend_ratio,
                "context_tags": e.context_tags,
            }
            for e in diorama.lineage
        ]

    # ── Internal helpers ───────────────────────────────────────

    def _hebbian_update(self, pattern: np.ndarray, scale: Optional[float] = None):
        """Apply Hebbian update to synaptic weights.

        ``weights[i][j] += lr * pattern[i] * pattern[j]``
        """
        s = scale if scale is not None else self.lr
        outer = np.outer(pattern, pattern)
        self.weights += s * outer
        # Clamp to [0, 1]
        np.clip(self.weights, 0.0, 1.0, out=self.weights)

    def _overlap(self, a: np.ndarray, b: np.ndarray) -> float:
        """Fraction of active bits in common (Jaccard-like)."""
        intersection = float(np.dot(a, b))
        union = float(np.sum(a) + np.sum(b) - intersection)
        if union < 0.5:
            return 0.0
        return intersection / union

    def _attractor_dynamics(self, cue: np.ndarray, steps: int = 10) -> np.ndarray:
        """Run pattern completion through the synaptic field.

        Each step: state = sign(weights @ state), thresholded to binary.
        This is the 'assembly completion' — partial cue fills in the rest.
        """
        state = cue.copy().astype(np.float32)
        for _ in range(steps):
            # Weighted sum from all connected dims
            next_state = self.weights @ state
            # Threshold: dims above median activation fire
            threshold = float(np.median(next_state[next_state > 0])) if np.any(next_state > 0) else 0.0
            state = (next_state > threshold).astype(np.float32)
            # Keep the cue bits always active (the 'stimulus')
            state = np.maximum(state, cue)
        return state

    def _find_best_match(self, pattern: np.ndarray) -> Optional[Diorama]:
        """Find the stored diorama with highest overlap to the given pattern."""
        if not self.dioramas:
            return None
        best_idx = max(
            range(len(self.dioramas)),
            key=lambda i: self._overlap(self.dioramas[i].pattern, pattern)
                          * self.dioramas[i].strength
        )
        best = self.dioramas[best_idx]
        if self._overlap(best.pattern, pattern) > 0.2:
            return best
        return None

    def _candidate_indices(self, tags: Optional[List[str]] = None) -> List[int]:
        """Get diorama indices filtered by tags, or all if no tags given."""
        if not tags:
            return list(range(len(self.dioramas)))

        # Union of all tagged sets
        candidates = set()
        for tag in tags:
            candidates.update(self.tag_index.get(tag, set()))
        return sorted(candidates)

    def _current_context_pattern(self) -> np.ndarray:
        """Generate a context pattern from current state (wax_temp, time, turn).

        This is what seeps into memories during reconsolidation.
        """
        pattern = np.zeros(self.n_dims, dtype=np.float32)

        # Encode wax_temp into first 100 dims
        temp_bits = max(1, int(100 * self.wax_temp * 0.5))
        for j in range(temp_bits):
            pattern[j] = 1.0

        # Encode current_turn into next 100 dims
        for j in range(100, 100 + (self.current_turn % 100)):
            pattern[j] = 1.0

        return pattern

    def _prune(self, threshold: float = FORGET_STRENGTH_THRESHOLD) -> int:
        """Remove dioramas below the strength threshold.

        Returns the number removed.
        """
        before = len(self.dioramas)
        surviving = [d for d in self.dioramas if d.strength >= threshold]
        pruned = before - len(surviving)
        self.dioramas = surviving
        # Rebuild tag index
        self.tag_index.clear()
        for idx, d in enumerate(self.dioramas):
            for tag in d.tags:
                self.tag_index[tag].add(idx)
        return pruned


# ═══════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════


def run_self_test():
    """Run a short self-test demonstrating all operations."""
    print("=" * 60)
    print("  DIORAMA STORE — SELF-TEST")
    print("=" * 60)

    store = DioramaStore(n_dims=500, sparsity=0.05, seed=42)

    # ── Encode 3 memories ────────────────────────────────────
    print("\n  Encoding 3 memories...")
    m1 = store.make_pattern({"sensory": "coffee cup on desk",
                             "emotional": "calm"})
    d1 = store.encode(m1, tags=["morning", "coffee"])
    print(f"    Memory 1: {d1}")

    m2 = store.make_pattern({"sensory": "angry email from boss",
                             "emotional": "anxious"})
    d2 = store.encode(m2, tags=["work", "stress"])
    print(f"    Memory 2: {d2}")

    m3 = store.make_pattern({"sensory": "sunset at the park",
                             "emotional": "content"})
    d3 = store.encode(m3, tags=["evening", "nature"])
    print(f"    Memory 3: {d3}")

    # ── Retrieve with a partial cue ──────────────────────────
    print("\n  Retrieving with cue 'anxious work email'...")
    cue = store.make_pattern({"emotional": "anxious"})
    store.wax_temp = 0.5
    result = store.retrieve(cue, tags=["work"])

    if result:
        print(f"    Found: {result}")
        print(f"    Recalls: {result.recall_count}")
        print(f"    Lineage entries: {len(result.lineage)}")
        if result.lineage:
            le = result.lineage[-1]
            print(f"    Last reconsolidation: turn={le.turn}, "
                  f"blend={le.blend_ratio:.4f}")
    else:
        print("    No match found")

    # ── Re-retrieve to show drift ────────────────────────────
    print("\n  Retrieving again (same cue) to show drift...")
    store.set_wax_temp_from_chemistry({"adrenaline": 0.5, "noradrenaline": 0.3,
                                        "serotonin": 0.2, "oxytocin": 0.0,
                                        "cortisol": 0.1})
    print(f"    Wax temp from chemistry: {store.wax_temp:.3f}")
    result2 = store.retrieve(cue, tags=["work"], turn=2)
    if result2:
        print(f"    Recalls: {result2.recall_count}")
        print(f"    Lineage entries: {len(result2.lineage)}")
        drift = result2.drift_distance()
        print(f"    Drift distance: {drift:.4f}")

    # ── Consolidation ────────────────────────────────────────
    print("\n  Consolidation (sleep quality = 0.8)...")
    report = store.consolidate(quality=0.8)
    print(f"    {report}")

    # ── Tick (decay) ─────────────────────────────────────────
    print("\n  Tick (dt=100.0 — simulate time passing)...")
    store.tick(dt=100.0)
    stats = store.stats()
    print(f"    Dioramas remaining: {stats['n_dioramas']}")
    print(f"    Forgotten: {stats['forgotten']}")
    print(f"    Avg strength: {stats['avg_strength']}")

    # ── Stats ────────────────────────────────────────────────
    print(f"\n  Final store stats:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    print("\n" + "=" * 60)
    print("  SELF-TEST COMPLETE")
    print("=" * 60)
    return True


if __name__ == "__main__":
    run_self_test()
