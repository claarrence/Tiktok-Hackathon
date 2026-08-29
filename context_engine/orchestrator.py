"""Adaptive Orchestration (Pillar III).

The pipeline shape is not fixed. Each turn, :func:`plan_turn` reads the distilled
:class:`~context_engine.distiller.ContextVector` and emits a :class:`TurnPolicy`
that programs how the rest of that turn runs — whether the agent is still allowed
to spend a turn on a clarification question, how broad "the pool is too general"
should be considered right now, and how hard to bias route weights toward
precision. This is the "re-orchestrate the workflow at runtime" half of the
self-evolution pillar; the weight redistribution itself is applied in
:func:`context_engine.profile.adaptive_weights`.

To keep the change surface small and the baseline safe, the agent currently
*applies* only ``force_answer``. The remaining fields are computed and carried on
the policy for the before/after write-up and for A/B sweeps, and are documented
as advisory until measured.
"""

from __future__ import annotations

from dataclasses import dataclass

MAX_TURNS = 10


@dataclass
class TurnPolicy:
    force_answer: bool          # applied: never ask on this turn, answer only
    allow_ask: bool             # applied: convenience negation of force_answer
    stop_asking_advised: bool   # advisory: trajectory says stop spending turns on questions
    over_general_pool: int      # advisory: pool size above which to treat as "too broad"
    confidence_margin: float    # advisory: top-vs-second gap below which to treat as "too broad"
    precision_bias: float       # mirror of ContextVector.precision_bias, for logging
    broaden: bool               # mirror of ContextVector.broaden, for logging
    rationale: str


def plan_turn(vector) -> TurnPolicy:  # vector: ContextVector (pre-policy)
    turns_left = vector.turns_left
    force_answer = turns_left <= 0 or vector.turn >= MAX_TURNS

    # Advisory: once the session is deep and at least two constraints are locked,
    # further questions rarely pay for their turn. Not yet applied by the agent.
    stop_asking_advised = (
        not force_answer
        and turns_left <= 1
        and vector.confirmed_constraints >= 2
    )

    # Advisory: widen the "too broad" bar as constraints accumulate (asking again
    # helps less), tighten it while still wide open early on.
    over_general_pool = 12
    if vector.confirmed_constraints >= 3:
        over_general_pool = 16
    elif vector.turn <= 2:
        over_general_pool = 10

    # Advisory: demand a cleaner top-1 separation from a critical buyer.
    confidence_margin = 0.10 if vector.rating_style_critical else 0.06

    rationale = (
        f"t{vector.turn} left{turns_left} intent={vector.intent} "
        f"pool={vector.pool_size}/{vector.pool_trend} "
        f"slots={vector.confirmed_constraints} ovr={vector.override_count} "
        f"pb={vector.precision_bias:.2f}"
        + (" FORCE_ANSWER" if force_answer else "")
    )

    return TurnPolicy(
        force_answer=force_answer,
        allow_ask=not force_answer,
        stop_asking_advised=stop_asking_advised,
        over_general_pool=over_general_pool,
        confidence_margin=confidence_margin,
        precision_bias=round(vector.precision_bias, 4),
        broaden=vector.broaden,
        rationale=rationale,
    )
