from __future__ import annotations

from dialog_state.state_machine import SessionState

BASE_WEIGHTS = {"keyword": 0.35, "category": 0.20, "vector": 0.20, "slot": 0.20, "tag": 0.05}


def adaptive_weights(state: SessionState, intent: str) -> dict[str, float]:
    """Runtime workflow re-orchestration: reshapes route weights per turn from
    the intent track, how much the candidate pool is narrowing, and the
    buyer's long-term rating behavior — the 'self-evolution' pillar."""
    weights = dict(BASE_WEIGHTS)

    if intent == "buying":
        weights["slot"] += 0.15
        weights["category"] += 0.05
        weights["vector"] -= 0.10
        weights["keyword"] -= 0.10
    else:
        weights["vector"] += 0.10
        weights["category"] -= 0.05
        weights["slot"] -= 0.05

    if state.broaden:
        weights["category"] = max(0.0, weights["category"] - 0.15)
        weights["vector"] += 0.10
        weights["keyword"] += 0.05

    if "critical" in str(state.profile.get("rating_style", "")).lower():
        weights["slot"] += 0.05

    total = sum(max(0.0, value) for value in weights.values()) or 1.0
    return {key: max(0.0, value) / total for key, value in weights.items()}
