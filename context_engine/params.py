"""Single tuning surface for the context & personalization pillar.

Everything the distiller and the adaptive-weight shaper read lives here so the
whole weight logic can be swept from one place (see project-plan.md, "Search the
ranking weights instead of hand-picking them"). Setting every ``pb_*`` / ``w_*``
entry to 0 reduces :func:`context_engine.profile.adaptive_weights` to the
original intent-only rule table (the ``legacy_*`` deltas below) exactly — the
pillar's safety net.

Values below were grid-searched against the 200 public dev sessions
(``python3 -m evaluator.local_evaluator``), optimising TechnicalScore. The key
finding: precision_bias should be driven by *disclosed information*
(confirmed slots, disclosed phrases, override events) rather than by raw turn
count. See docs/context_distillation_example.md.
"""

from __future__ import annotations

PARAMS: dict[str, float] = {
    # --- precision_bias: how strongly the disclosed information so far has
    #     earned a shift from recall routes toward precision routes (0..pb_max)
    "pb_constraint": 0.16,        # per confirmed slot attribute
    "pb_constraint_cap": 4.0,     # ...counted up to this many
    "pb_override": 0.12,          # per detected intent-override event
    "pb_phrase": 0.06,            # per disclosed constraint phrase
    "pb_phrase_cap": 4.0,
    "pb_stagnant_penalty": 0.20,  # subtracted while the candidate pool will not shrink
    "pb_browsing_scale": 0.5,     # browsing keeps more breadth than buying
    "pb_max": 0.60,

    # --- how precision_bias (pb, 0..pb_max) redistributes route weight
    "w_slot": 0.34,
    "w_phrase": 0.26,
    "w_vector": 0.30,
    "w_keyword": 0.17,
    "w_category": 0.09,

    # --- demandingness fine-tune, applied on top of the legacy "critical" bump,
    #     scaled by (demandingness - 0.5), only when > 0.5
    "w_demand_slot": 0.10,
    "w_demand_phrase": 0.08,

    # --- personalization modulation, centred so 0.5 == no change
    "w_tag": 0.04,               # scaled by (tag_focus - 0.5)

    # --- legacy intent-only rule table: the original hand-picked deltas, moved
    #     here so a grid search can reach the whole weight logic. NOT zeroed by
    #     the "disable the pillar" path (only pb_*/w_* are), so this stays the
    #     safe fallback.
    "legacy_buying_slot": 0.15,
    "legacy_buying_category": 0.05,
    "legacy_buying_vector": -0.10,
    "legacy_buying_keyword": -0.10,
    "legacy_browsing_vector": 0.10,
    "legacy_browsing_category": -0.05,
    "legacy_browsing_slot": -0.05,
    "legacy_broaden_category": -0.15,
    "legacy_broaden_vector": 0.10,
    "legacy_broaden_keyword": 0.05,
    "legacy_critical_slot": 0.05,

    # --- pool-trend classification: "converging" needs the pool to drop to at
    #     most this fraction of the previous turn's size
    "pool_converging_ratio": 0.9,
}
