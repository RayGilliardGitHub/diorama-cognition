#!/usr/bin/env python3
"""
Body State & Chemistry Module (Phase 2)
Embodied Cognition Platform

Maintains a simulated body chemistry state with:
- 7 neurotransmitters (adrenaline, noradrenaline, dopamine, serotonin,
  cortisol, oxytocin, endorphins) — each with half-life decay
- 6 body state variables (heart rate, body temperature, hunger,
  fatigue, pain, arousal)
- Interactions: adrenaline -> heart rate, noradrenaline -> arousal, etc.
- Gate multiplier computation per processing node
- Serialization to a string for prompt injection
"""

import math
import time

from src.core.instrumentation import (
    DEFAULT_CHEMISTRY,
    NEUROTRANSMITTERS,
    BODY_STATE_VARS,
    ALL_CHEMISTRY_KEYS,
)

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

# Half-lives in seconds (from §6.1 of the platform plan)
HALF_LIVES = {
    "adrenaline": 120,       # 2 minutes
    "noradrenaline": 30,     # 30 seconds
    "dopamine": 300,         # 5 minutes
    "serotonin": 3600,       # 1 hour
    "cortisol": 3600,        # 1 hour
    "oxytocin": 600,         # 10 minutes
    "endorphins": 1800,      # 30 minutes
}

# Default resting values for body variables (they drift toward these)
BODY_RESTING = {
    "heart_rate": 70.0,
    "body_temperature": 37.0,
    "hunger": 0.5,
    "fatigue": 0.0,
    "pain": 0.0,
    "arousal": 0.3,
}

# Recovery rate per second toward resting values
BODY_RECOVERY = {
    "heart_rate": 0.005,       # returns to baseline in ~3.3 minutes
    "body_temperature": 0.002,
    "hunger": 0.0005,          # hunger increases slowly
    "fatigue": 0.001,          # recovers slowly
    "pain": 0.002,
    "arousal": 0.01,
}

# Defaults for chemistry-state interaction rates
ADRENALINE_HEART_RATE_GAIN = 0.05     # per unit adrenaline per second
ADRENALINE_TEMP_GAIN = 0.01           # per unit adrenaline per second
ADRENALINE_AROUSAL_GAIN = 0.005       # per unit adrenaline per second
NORADRENALINE_AROUSAL_GAIN = 0.02     # per unit noradrenaline per second
CORTISOL_HUNGER_GAIN = 0.001          # per unit cortisol per second
ADRENALINE_FATIGUE_GAIN = 0.002       # per unit adrenaline per second


# ═══════════════════════════════════════════════════════════════
# BodyState
# ═══════════════════════════════════════════════════════════════


class BodyState:
    """Simulated body chemistry state with decay, interactions, and gating.

    Manages the same 13-vector chemistry state as DEFAULT_CHEMISTRY.
    Maintains a shared reference to the caller's chemistry dict so that
    ``body_state.state`` **is** the same dict the orchestrator reads/writes.

    Parameters
    ----------
    initial_chemistry : dict or None
        Initial state dict.  If None, starts from ``DEFAULT_CHEMISTRY``.
        The dict is stored by reference (not copied) so external callers
        see updates from tick/adjustment immediately.
    """

    def __init__(self, initial_chemistry=None):
        if initial_chemistry is not None:
            self.state = initial_chemistry  # store reference, not copy
            # Ensure all keys exist
            for k in ALL_CHEMISTRY_KEYS:
                self.state.setdefault(k, DEFAULT_CHEMISTRY[k])
        else:
            self.state = dict(DEFAULT_CHEMISTRY)
        self._last_tick_time = time.monotonic()

    def tick(self, elapsed=None):
        """Advance chemistry by the given number of simulated seconds.

        Parameters
        ----------
        elapsed : float or None
            Seconds to advance.  If None, use wall-clock time since the
            last call to ``tick()`` or object creation.
        """
        if elapsed is None:
            now = time.monotonic()
            elapsed = now - self._last_tick_time
            self._last_tick_time = now

        if elapsed is None or elapsed <= 0:
            return

        # ── 1. Decay neurotransmitters by half-life ──────────────
        for nt in NEUROTRANSMITTERS:
            half_life = HALF_LIVES.get(nt)
            if half_life and half_life > 0:
                self.state[nt] *= 0.5 ** (elapsed / half_life)

        # ── 1b. Baseline drift toward moderate levels ──────────
        # Prevents saturation by pulling NTs toward 0.3 at 2%/s
        for nt in NEUROTRANSMITTERS:
            target = 0.3
            diff = self.state[nt] - target
            if abs(diff) > 0.01:
                self.state[nt] -= diff * 0.02 * elapsed

        # ── 1c. Antagonistic interactions ─────────────────────
        # Cortisol > 0.5 mildly suppresses dopamine
        cort = self.state["cortisol"]
        if cort > 0.5:
            self.state["dopamine"] *= 1.0 - (cort - 0.5) * 0.05 * elapsed

        # High serotonin (>0.7) mildly suppresses noradrenaline
        ser = self.state["serotonin"]
        if ser > 0.7:
            self.state["noradrenaline"] *= 1.0 - (ser - 0.7) * 0.1 * elapsed

        # High adrenaline + cortisol → extra dopamine suppression
        a = self.state["adrenaline"]
        if a > 0.5 and cort > 0.5:
            self.state["dopamine"] *= 1.0 - 0.03 * elapsed

        # Mutual cortisol-serotonin antagonism (asymmetric)
        # Cortisol suppresses serotonin 3x harder than the reverse
        # Creates competition rather than synchronization
        if cort > 0.6 and ser > 0.6:
            # Both elevated — cortisol dominates, serotonin drops faster
            self.state["serotonin"] -= (cort - 0.6) * 0.12 * elapsed
            self.state["cortisol"] -= (ser - 0.6) * 0.04 * elapsed
        elif cort > 0.6:
            # Cortisol dominance — suppresses serotonin
            self.state["serotonin"] *= 1.0 - (cort - 0.6) * 0.10 * elapsed
        elif ser > 0.6:
            # Serotonin dominance — mildly suppresses cortisol
            self.state["cortisol"] *= 1.0 - (ser - 0.6) * 0.04 * elapsed

        # ── 1d. Overdrive decay (receptor downregulation) ─────
        # NTs above 0.8 decay aggressively toward 0.5
        for nt in NEUROTRANSMITTERS:
            if self.state[nt] > 0.8:
                excess = self.state[nt] - 0.5
                self.state[nt] -= excess * 0.025 * elapsed

        # ── 2. Drift body vars toward resting values ────────────
        for var in BODY_STATE_VARS:
            resting = BODY_RESTING.get(var, 0.0)
            rate = BODY_RECOVERY.get(var, 0.001)
            current = self.state[var]
            diff = current - resting
            if abs(diff) > 0.001:
                self.state[var] -= diff * rate * elapsed
                # Snap to resting if close enough
                if abs(self.state[var] - resting) < 0.01:
                    self.state[var] = resting

        # ── 3. Apply chemistry-body interactions ────────────────
        a = self.state["adrenaline"]
        n = self.state["noradrenaline"]
        cort = self.state["cortisol"]

        # Adrenaline pushes heart rate up (target = 70 + 50 * a)
        hr_target = 70.0 + 50.0 * a
        hr_diff = hr_target - self.state["heart_rate"]
        self.state["heart_rate"] += hr_diff * ADRENALINE_HEART_RATE_GAIN * elapsed

        # Noradrenaline → arousal
        ar_target = min(1.0, 0.3 + 0.7 * (n + 0.3 * a))
        ar_diff = ar_target - self.state["arousal"]
        self.state["arousal"] += ar_diff * NORADRENALINE_AROUSAL_GAIN * elapsed

        # Adrenaline → slight temperature increase
        temp_target = 37.0 + 0.5 * a
        temp_diff = temp_target - self.state["body_temperature"]
        self.state["body_temperature"] += temp_diff * ADRENALINE_TEMP_GAIN * elapsed

        # Cortisol → stress eating (hunger creeps up)
        self.state["hunger"] += cort * CORTISOL_HUNGER_GAIN * elapsed

        # Sustained adrenaline → fatigue
        self.state["fatigue"] += a * ADRENALINE_FATIGUE_GAIN * elapsed
        # Constant upward drift — you get tired from existing
        self.state["fatigue"] += 0.005 * elapsed
        # Recovery when fatigue is high (prevents permanent saturation)
        # Slowly drifts toward 0.3 when above 0.7
        if self.state["fatigue"] > 0.7:
            excess = self.state["fatigue"] - 0.3
            self.state["fatigue"] -= excess * 0.008 * elapsed

        # ── 4. Clamp to sensible ranges ─────────────────────────
        for nt in NEUROTRANSMITTERS:
            self.state[nt] = max(0.0, min(1.0, self.state[nt]))

        self.state["heart_rate"] = max(30.0, min(200.0, self.state["heart_rate"]))
        self.state["body_temperature"] = max(34.0, min(42.0,
                                                self.state["body_temperature"]))
        self.state["hunger"] = max(0.0, min(1.0, self.state["hunger"]))
        self.state["fatigue"] = max(0.0, min(1.0, self.state["fatigue"]))
        self.state["pain"] = max(0.0, min(1.0, self.state["pain"]))
        self.state["arousal"] = max(0.0, min(1.0, self.state["arousal"]))

    def apply_adjustment(self, delta_dict):
        """Apply a delta vector (chemistry_adjustment) from the Synthesizer.

        Parameters
        ----------
        delta_dict : dict
            Dictionary of key -> delta to add.  Neurotransmitters are
            clamped to [0.0, 1.0]; body vars are not clamped here
            (the next tick handles it).
        """
        if not delta_dict:
            return
        for k, v in delta_dict.items():
            if k not in self.state:
                continue
            if k in NEUROTRANSMITTERS:
                self.state[k] = max(0.0, min(1.0, self.state[k] + v))
            else:
                self.state[k] = self.state[k] + v

    # ── Output helpers ──────────────────────────────────────────

    def to_prompt_string(self, hide_labels=False, permutation=None,
                         body_permutation=None):
        """Serialize the current chemistry state for prompt injection.

        Parameters
        ----------
        hide_labels : bool
            If True, replace chemical names with abstract labels
            (state_1..state_7, body_1..body_6) for blind testing.
        permutation : dict or None
            Dict mapping 'state_N' -> internal chemistry key to permute
            which variable appears under each abstract label.
            Example: {'state_1': 'cortisol', 'state_2': 'serotonin', ...}
        body_permutation : dict or None
            Same as permutation but for body_1..body_6.

        Returns
        -------
        str
            A formatted block.
        """
        s = self.state
        if hide_labels:
            # Default identity mapping for NTs
            nt_order = ['adrenaline','noradrenaline','dopamine',
                        'serotonin','cortisol','oxytocin','endorphins']
            body_order = ['heart_rate','body_temperature','hunger',
                          'fatigue','pain','arousal']

            lines = ["[Chemistry]"]
            for i in range(7):
                key = f'state_{i+1}'
                actual_key = (permutation or {}).get(key, nt_order[i])
                lines.append(f"{key}: {s[actual_key]:.2f}")
            for i in range(6):
                key = f'body_{i+1}'
                actual_key = (body_permutation or {}).get(key, body_order[i])
                if actual_key == 'heart_rate':
                    lines.append(f"{key}: {s[actual_key]:.0f} bpm")
                elif actual_key == 'body_temperature':
                    lines.append(f"{key}: {s[actual_key]:.1f} C")
                else:
                    lines.append(f"{key}: {s[actual_key]:.2f}")
            lines.extend(["",
                          "Factor this body state into your response."])
        else:
            lines = [
                "[Chemistry]",
                f"Adrenaline: {s['adrenaline']:.2f}",
                f"Noradrenaline: {s['noradrenaline']:.2f}",
                f"Dopamine: {s['dopamine']:.2f}",
                f"Serotonin: {s['serotonin']:.2f}",
                f"Cortisol: {s['cortisol']:.2f}",
                f"Oxytocin: {s['oxytocin']:.2f}",
                f"Endorphins: {s['endorphins']:.2f}",
                f"Heart Rate: {s['heart_rate']:.0f} bpm",
                f"Temperature: {s['body_temperature']:.1f} C",
                f"Hunger: {s['hunger']:.2f}",
                f"Fatigue: {s['fatigue']:.2f}",
                f"Pain: {s['pain']:.2f}",
                f"Arousal: {s['arousal']:.2f}",
                "",
                "Factor this body state into your response.",
            ]
        return "\n".join(lines)

    def get_gate_multiplier(self, node_name, base_weight=1.0):
        """Compute chemistry-modulated gate multiplier for a processing node.

        Gate multiplier = base_weight * chemistry_modulation * fatigue_penalty

        Chemistry modulation follows §7.2 of the platform plan:
        - High adrenaline → open all gates (modulation near 1.0)
        - High serotonin → dampen emotional gate, sustain social gate
        - High cortisol → resist modulation (gates stay near base)
        - Fatigue → lower all gate multipliers

        Parameters
        ----------
        node_name : str
            One of ``'sensory'``, ``'emotional'``, ``'episodic'``,
            ``'social'``, ``'synthesizer'``.
        base_weight : float
            Base gate weight (default 1.0).

        Returns
        -------
        float
            Effective gate multiplier in [0.0, 1.0].
        """
        s = self.state
        a = s["adrenaline"]
        ser = s["serotonin"]
        cort = s["cortisol"]
        fat = s["fatigue"]

        # Base modulation per node (before cortisol/fatigue)
        if node_name == "sensory":
            # Adrenaline opens sensory gate for threat detection
            mod = 0.6 + 0.4 * a                          # [0.6, 1.0]
        elif node_name == "emotional":
            # Adrenaline amplifies, serotonin dampens negative signals
            mod = 0.5 + 0.5 * a - 0.2 * ser              # [0.3, 1.0]
        elif node_name == "episodic":
            # Adrenaline opens memory retrieval
            mod = 0.5 + 0.5 * a                           # [0.5, 1.0]
        elif node_name == "social":
            # Serotonin sustains social attunement; adrenaline reduces it
            mod = 0.3 + 0.7 * ser * (1.0 - a * 0.6)      # [0.3, 1.0]
        elif node_name == "synthesizer":
            return 1.0                                     # always fully open
        else:
            mod = 0.7                                      # default mid-range

        mod = max(0.1, min(1.0, mod))

        # Cortisol resistance: high cortisol = gates resist modulation
        #   When cort=0: full modulation effect
        #   When cort=1: modulation is halved (gate stays near base)
        resistance = 1.0 - cort * 0.5                      # [0.5, 1.0]
        mod = 1.0 - resistance + resistance * mod           # lerp toward 1.0
        #   When cort=0: mod stays as computed above
        #   When cort=1: mod = 0.5 + 0.5 * computed_mod
        #   Example: emotional with a=0.9, ser=0.0, cort=0.0:
        #     mod = 0.5 + 0.5*0.9 = 0.95 (fully open)
        #   Same inputs with cort=1.0:
        #     resistance = 0.5
        #     mod = 1.0 - 0.5 + 0.5*0.95 = 0.975

        # Fatigue penalty reduces everything
        fat_factor = 1.0 - fat * 0.5                       # [0.5, 1.0]

        gate = base_weight * mod * fat_factor
        return max(0.0, min(1.0, gate))

    # ── Convenience ─────────────────────────────────────────────

    def __repr__(self):
        a = self.state["adrenaline"]
        ser = self.state["serotonin"]
        cort = self.state["cortisol"]
        hr = self.state["heart_rate"]
        return (
            f"BodyState(adrenaline={a:.2f}, serotonin={ser:.2f}, "
            f"cortisol={cort:.2f}, heart_rate={hr:.0f})"
        )
