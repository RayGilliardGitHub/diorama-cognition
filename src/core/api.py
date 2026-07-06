#!/usr/bin/env python3
"""
OpenCode Go API client for the Diorama Cognition project.

Wraps the OpenCode Go OpenAI-compatible endpoint so every phase runner
can call API models (DeepSeek Flash, MiMo-V2.5, etc.) instead of local ollama.

Endpoint: https://opencode.ai/zen/go/v1/chat/completions
Models:
 - deepseek-v4-flash  ($0.14/$0.28 per M tokens, ~31K req/5hr)
 - mimo-v2.5          ($0.14/$0.28 per M tokens, ~30K req/5hr)
 - deepseek-v4-pro    ($1.74/$3.48 per M tokens, ~3.4K req/5hr)
 - mimo-v2.5-pro      ($1.74/$3.48 per M tokens, ~3.2K req/5hr)

Inference Parameter Profiles (matching the 75 Modelfile tiers):
 focus     — temp=0.05, top_p=0.90, freq_penalty~0.30  (deterministic, analytical)
 std       — temp=0.10, top_p=0.95, freq_penalty~0.20  (factual, consistent)
 bal       — temp=0.30, top_p=0.95, freq_penalty~0.10  (balanced default)
 creative  — temp=0.80, top_p=0.98, freq_penalty~0.05  (exploratory, divergent)
 reflex    — temp=0.01, top_p=0.85, freq_penalty~0.00  (fast, automatic)

Node-to-profile mapping (from old project's architecture):
 sensory     → focus     (analytical, pattern detection)
 emotional   → creative  (divergent emotional assessment)
 episodic    → std       (factual recall)
 social      → creative  (creative social reasoning)
 synthesizer → bal       (balanced binding)

Note: top_k and repeat_penalty are Ollama-specific parameters not available
in the standard OpenAI chat completions API. We approximate:
 repeat_penalty 1.0  → frequency_penalty 0.00
 repeat_penalty 1.05 → frequency_penalty 0.10
 repeat_penalty 1.10 → frequency_penalty 0.20
 repeat_penalty 1.15 → frequency_penalty 0.30
"""
import json
import os
import time
from typing import Optional

try:
   import requests
except ImportError:
   print("Error: 'requests' library required.  Install: pip install requests",
         file=sys.stderr)
   raise


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

OPCODE_GO_ENDPOINT = "https://opencode.ai/zen/go/v1/chat/completions"
"""OpenCode Go subscription endpoint (paid, ~$0.002/turn)."""

OPCODE_ZEN_ENDPOINT = "https://opencode.ai/zen/v1/chat/completions"
"""OpenCode Zen endpoint — free models and pay-as-you-go."""

DEFAULT_TIMEOUT = 120
"""Default per-request timeout in seconds."""

# Model IDs (from the Go subscription page)
MODEL_DEEPSEEK_V4_FLASH = "deepseek-v4-flash"
MODEL_MIMO_V2_5 = "mimo-v2.5"
MODEL_DEEPSEEK_V4_PRO = "deepseek-v4-pro"
MODEL_MIMO_V2_5_PRO = "mimo-v2.5-pro"

# Free models via Zen endpoint (cost = $0)
MODEL_BIG_PICKLE = "big-pickle"
MODEL_DEEPSEEK_V4_FLASH_FREE = "deepseek-v4-flash-free"
MODEL_MIMO_V2_5_FREE = "mimo-v2.5-free"
MODEL_NORTH_MINI_CODE_FREE = "north-mini-code-free"
MODEL_NEMOTRON_3_ULTRA_FREE = "nemotron-3-ultra-free"

# ═══════════════════════════════════════════════════════════════
# Inference Parameter Profiles
# ═══════════════════════════════════════════════════════════════
#
# These replicate the 5 Modelfile tiers from the old project.
# The OpenAI API doesn't support top_k or a direct repeat_penalty,
# so we map them approximately:
#
#   Ollama parameter    →  OpenAI API parameter
#   temperature         →  temperature  (direct)
#   top_p               →  top_p        (direct)
#   top_k               →  (not supported — omitted)
#   repeat_penalty 1.0  →  frequency_penalty 0.00
#   repeat_penalty 1.05 →  frequency_penalty 0.10
#   repeat_penalty 1.10 →  frequency_penalty 0.20
#   repeat_penalty 1.15 →  frequency_penalty 0.30
#
# Each profile is a dict passed as **kwargs to opencode_chat().

PROFILE_FOCUS = {
   "temperature": 0.05,
   "top_p": 0.90,
   "frequency_penalty": 0.30,
   "description": "Analytical, deterministic — minimal variation",
}

PROFILE_STD = {
   "temperature": 0.10,
   "top_p": 0.95,
   "frequency_penalty": 0.20,
   "description": "Factual, consistent — low variation",
}

PROFILE_BAL = {
   "temperature": 0.30,
   "top_p": 0.95,
   "frequency_penalty": 0.10,
   "description": "Balanced default — moderate variation",
}

PROFILE_CREATIVE = {
   "temperature": 0.80,
   "top_p": 0.98,
   "frequency_penalty": 0.05,
   "description": "Exploratory, divergent — high variation",
}

PROFILE_REFLEX = {
   "temperature": 0.01,
   "top_p": 0.85,
   "frequency_penalty": 0.00,
   "description": "Fast, automatic — near-deterministic",
}

ALL_PROFILES = {
   "focus": PROFILE_FOCUS,
   "std": PROFILE_STD,
   "bal": PROFILE_BAL,
   "creative": PROFILE_CREATIVE,
   "reflex": PROFILE_REFLEX,
}


# ═══════════════════════════════════════════════════════════════
# Node Configuration
# ═══════════════════════════════════════════════════════════════
#
# Each node gets: (model_id, profile_name)
# This replicates the old project's node-to-Modelfile mapping.

NODE_CONFIG = {
   "sensory":     (MODEL_MIMO_V2_5, "focus"),
   "emotional":   (MODEL_MIMO_V2_5,    "creative"),
   "episodic":    (MODEL_MIMO_V2_5,    "std"),
   "social":      (MODEL_MIMO_V2_5,    "creative"),
   "synthesizer": (MODEL_MIMO_V2_5, "bal"),
}


def get_node_model(node_key: str) -> str:
   """Return the model ID for a given node name."""
   return NODE_CONFIG.get(node_key, (MODEL_DEEPSEEK_V4_FLASH, "bal"))[0]


def get_node_profile(node_key: str) -> dict:
   """Return the inference parameter profile for a given node name."""
   profile_name = NODE_CONFIG.get(node_key, (MODEL_DEEPSEEK_V4_FLASH, "bal"))[1]
   return dict(ALL_PROFILES.get(profile_name, PROFILE_BAL))


# ═══════════════════════════════════════════════════════════════
# API Key Handling
# ═══════════════════════════════════════════════════════════════

_api_key: str = ""


def set_api_key(key: str):
   """Set the OpenCode Go API key.

   Call once at startup. Looks in order:
   1. Explicit call to set_api_key()
   2. OPENCODE_GO_API_KEY environment variable
   3. OpenCode config via ~/.opencode/config
   """
   global _api_key
   if key:
       _api_key = key
       return

   env_key = os.environ.get("OPENCODE_GO_API_KEY")
   if env_key:
       _api_key = env_key
       return

   # Check OpenCode config locations
   config_paths = [
       os.path.expanduser("~/.opencode/config"),
       os.path.expanduser("~/.config/opencode/config"),
   ]
   for cp in config_paths:
       if os.path.exists(cp):
           try:
               with open(cp) as f:
                   for line in f:
                       line = line.strip()
                       if "go" in line.lower() and ("key" in line.lower() or "token" in line.lower()):
                           parts = line.replace("=", " ").split()
                           for p in parts:
                               if p.startswith("golive_") or p.startswith("go_"):
                                   _api_key = p
                                   return
           except (IOError, OSError):
               pass

   # Check .env file in the project root
   _try_dotenv()

   if _api_key:
       return

   raise ValueError(
       "No OpenCode Go API key found. "
       "Call set_api_key('golive_xxxxx') or set the OPENCODE_GO_API_KEY env var."
   )


def _try_dotenv():
    """Try to load the API key from a .env file in the project root."""
    import os
    global _api_key
    if _api_key:
        return
    for d in [os.getcwd()] + [os.path.abspath(os.path.join(os.getcwd(), p)) for p in ['..', '../..']]:
        env_path = os.path.join(d, '.env')
        if os.path.exists(env_path):
            try:
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        prefix = 'OPENCODE' + chr(95) + 'GO_API_KEY'
                        if prefix in line and '=' in line:
                            val = line.split('=', 1)[1].strip().strip(chr(39)).strip(chr(34)).strip()
                            if val:
                                _api_key = val
                                return
            except Exception:
                pass

def get_api_key() -> str:
   """Return the current API key, or raise if not set."""
   global _api_key
   if not _api_key:
       set_api_key("")
   return _api_key


# ═══════════════════════════════════════════════════════════════


def opencode_chat(
   model: str,
   messages: list,
   temperature: float = 0.3,
   top_p: float = 0.95,
   frequency_penalty: float = 0.0,
   presence_penalty: float = 0.0,
   max_tokens: int = 1024,
   timeout: int = DEFAULT_TIMEOUT,
) -> str:
   """Call the OpenCode Go chat completions API.

   Parameters
   ----------
   model : str
       Model ID (e.g. 'deepseek-v4-flash', 'mimo-v2.5').
   messages : list of dict
       Messages in OpenAI format.
   temperature : float
       Sampling temperature (default 0.3).
   top_p : float
       Nucleus sampling parameter (default 0.95).
   frequency_penalty : float
       Penalizes repeated tokens. Positive = less repetition.
       Maps approximately from Ollama's repeat_penalty:
         1.0 → 0.00, 1.05 → 0.10, 1.10 → 0.20, 1.15 → 0.30
   presence_penalty : float
       Penalizes tokens that have appeared (default 0.0).
   max_tokens : int
       Max output tokens (default 1024).
   timeout : int
       Request timeout in seconds.

   Returns
   -------
   str
       The assistant's response content.
   """
   if not _api_key:
       set_api_key("")

   headers = {
       "Authorization": f"Bearer {_api_key}",
       "Content-Type": "application/json",
   }

   payload = {
       "model": model,
       "messages": messages,
       "temperature": temperature,
       "top_p": top_p,
       "frequency_penalty": frequency_penalty,
       "presence_penalty": presence_penalty,
       "max_tokens": max_tokens,
       "stream": False,
   }

   # Route all models through Go endpoint — Zen free endpoint is unusably rate-limited
   endpoint = OPCODE_GO_ENDPOINT

   import time as _time
   import random as _random
   _max_retries = 10
   for _attempt in range(_max_retries):
       resp = requests.post(
           endpoint,
           headers=headers,
           json=payload,
           timeout=timeout,
       )
       if resp.status_code == 429:
           _wait = min(2 ** _attempt + _random.uniform(0, 1), 60.0)
           print(f"[rate-limit] 429 on {model}, retrying in {_wait:.0f}s (attempt {_attempt+1}/{_max_retries})", flush=True)
           _time.sleep(_wait)
           continue
       resp.raise_for_status()
       break
   else:
       resp.raise_for_status()
   data = resp.json()

   if "choices" not in data or not data["choices"]:
       raise ValueError(f"API returned no choices: {data}")

   return data["choices"][0]["message"]["content"]


def chat_for_node(
   node_key: str,
   messages: list,
   max_tokens: Optional[int] = None,
   profile_override: Optional[dict] = None,
) -> str:
   """Route a node's chat call to the right model with the right profile.

   Looks up the model and inference profile for the given node,
   then calls opencode_chat() with those parameters.

   Parameters
   ----------
   node_key : str
       One of: sensory, emotional, episodic, social, synthesizer, baseline.
   messages : list of dict
       OpenAI-format messages.
   max_tokens : int or None
       Max output tokens. If None, uses node-specific default:
         sensory/emotional/episodic/social: 2048
         synthesizer/baseline: 4096
       Higher values for synthesizer because DeepSeek Flash spends
       tokens on reasoning before producing visible output.
   profile_override : dict or None
       If provided, use these parameter values instead of the node's profile.

   Returns
   -------
   str
       Model response text.
   """
   model = get_node_model(node_key)
   params = profile_override if profile_override else get_node_profile(node_key)

   if max_tokens is None:
       # Node-specific defaults accounting for reasoning tokens
       # DeepSeek models burn tokens on reasoning before visible output
       _TOKEN_BUDGETS = {
           "emotional": 2048,
           "episodic": 2048,
           "social": 2048,
           "synthesizer": 4096,
           "baseline": 4096,
       }
       max_tokens = _TOKEN_BUDGETS.get(node_key, 2048)

   return opencode_chat(
       model=model,
       messages=messages,
       temperature=params.get("temperature", 0.3),
       top_p=params.get("top_p", 0.95),
       frequency_penalty=params.get("frequency_penalty", 0.0),
       max_tokens=max_tokens,
   )


# ═══════════════════════════════════════════════════════════════
# Cost Estimation
# ═══════════════════════════════════════════════════════════════

MODEL_PRICING = {
   "deepseek-v4-flash": {"input": 0.14, "output": 0.28, "cached_read": 0.0028},
   "mimo-v2.5":         {"input": 0.14, "output": 0.28, "cached_read": 0.0028},
   "deepseek-v4-pro":   {"input": 1.74, "output": 3.48, "cached_read": 0.0145},
   "mimo-v2.5-pro":     {"input": 1.74, "output": 3.48, "cached_read": 0.0145},
   # Zen free models (cost = $0)
   "big-pickle":              {"input": 0.00, "output": 0.00, "cached_read": 0.00},
   "deepseek-v4-flash-free":  {"input": 0.00, "output": 0.00, "cached_read": 0.00},
   "mimo-v2.5-free":          {"input": 0.00, "output": 0.00, "cached_read": 0.00},
   "north-mini-code-free":    {"input": 0.00, "output": 0.00, "cached_read": 0.00},
   "nemotron-3-ultra-free":   {"input": 0.00, "output": 0.00, "cached_read": 0.00},
}

TYPICAL_TOKENS = {
   "deepseek-v4-flash": {"input": 790, "cached": 68000, "output": 280,
   # Free models
   "big-pickle":              {"input": 800, "cached": 70000, "output": 300},
   "deepseek-v4-flash-free":  {"input": 790, "cached": 68000, "output": 280},
   "mimo-v2.5-free":          {"input": 830, "cached": 71500, "output": 295},
   "north-mini-code-free":    {"input": 500, "cached": 50000, "output": 200},
   "nemotron-3-ultra-free":   {"input": 500, "cached": 50000, "output": 200},
},
   "mimo-v2.5":         {"input": 830, "cached": 71500, "output": 295},
   "deepseek-v4-pro":   {"input": 750, "cached": 82000, "output": 290},
   "mimo-v2.5-pro":     {"input": 790, "cached": 86000, "output": 305},
}

def estimate_cost(model: str, input_tokens: int, output_tokens: int,
                 cached_input: int = 0) -> float:
   """Estimate the cost of a single request in USD."""
   pricing = MODEL_PRICING.get(model, MODEL_PRICING["deepseek-v4-flash"])
   cost = (
       input_tokens / 1_000_000 * pricing["input"]
       + output_tokens / 1_000_000 * pricing["output"]
       + cached_input / 1_000_000 * pricing["cached_read"]
   )
   return cost


# ═══════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
   print("=" * 68)
   print("  OpenCode Go API Client — Self-Test")
   print("=" * 68)
   print(f"\n  Endpoint: {OPCODE_GO_ENDPOINT}")
   print(f"  API key set: {bool(_api_key)}")

   print(f"\n  Inference Parameter Profiles (from Modelfile tiers):")
   for name, profile in sorted(ALL_PROFILES.items()):
       desc = profile.pop("description", "")
       print(f"    {name:>10s}: temp={profile['temperature']:.2f}, "
             f"top_p={profile['top_p']:.2f}, "
             f"freq_penalty={profile['frequency_penalty']:.2f}  ({desc})")
       profile["description"] = desc  # restore

   print(f"\n  Node → Model + Profile:")
   for node, (model, prof) in sorted(NODE_CONFIG.items()):
       print(f"    {node:>15s} → {model:>25s} + {prof:>10s}")

   print(f"\n  Model pricing (per 1M tokens):")
   for model, prices in sorted(MODEL_PRICING.items()):
       print(f"    {model:>25s}: ${prices['input']:.2f} in / ${prices['output']:.2f} out")

   print(f"\n  Estimated cost per typical turn (5 node calls):")
   turn_cost = 0
   for node in ["sensory", "emotional", "episodic", "social", "synthesizer"]:
       model = get_node_model(node)
       typical = TYPICAL_TOKENS.get(model, TYPICAL_TOKENS["deepseek-v4-flash"])
       cost = estimate_cost(model, typical["input"], typical["output"], typical["cached"])
       profile_name = NODE_CONFIG.get(node, (None, "bal"))[1]
       print(f"    {node:>15s} via {model:>25s} ({profile_name}): ~${cost:.6f}")
       turn_cost += cost
   print(f"    {'TOTAL':>15s}: ~${turn_cost:.6f}")
   if turn_cost > 0:
       print(f"    Turns per $1.00: {1.0 / turn_cost:.0f}")
       print(f"    Turns within $60/mo budget: ~{60.0 / turn_cost:.0f}")
   else:
       print(f"    Turns per $1.00: unlimited (free models)")
       print(f"    Turns within $60/mo budget: unlimited (free models)")

   print(f"\n  Note: top_k not available in OpenAI API (Ollama-specific).")
   print(f"  Note: repeat_penalty approximated via frequency_penalty (see docstring).")
   print(f"\n  To use: set_api_key('golive_xxxxx') before calling opencode_chat()")
   print("=" * 68)
