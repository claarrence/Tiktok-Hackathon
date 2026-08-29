"""Context & Personalization pillar (Pillar III).

Public surface:
    ContextDistiller / ContextVector  -- per-session distillation of dialog history
    plan_turn / TurnPolicy            -- runtime orchestration decision
    adaptive_weights                  -- distilled context -> route-weight vector
    PARAMS                            -- single tuning surface
"""

from __future__ import annotations

from context_engine.distiller import ContextDistiller, ContextVector
from context_engine.orchestrator import TurnPolicy, plan_turn
from context_engine.params import PARAMS
from context_engine.profile import BASE_WEIGHTS, adaptive_weights

__all__ = [
    "ContextDistiller",
    "ContextVector",
    "TurnPolicy",
    "plan_turn",
    "adaptive_weights",
    "BASE_WEIGHTS",
    "PARAMS",
]
