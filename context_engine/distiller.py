"""Personalized Context Distillation (Pillar III).

Turns the raw dialog history plus the anonymized ``user_profile`` into a single
compact :class:`ContextVector` every turn. The rest of the agent never reads the
transcript directly — it reads this distilled object, so the "what has this
conversation become so far" judgement lives in exactly one place and can evolve
turn over turn instead of being recomputed ad hoc by each pillar.

One :class:`ContextDistiller` is held per session (created in ``Agent.reset``);
it accumulates the cross-turn memory — intent stability, pool trajectory,
constraint growth, override events — that a single turn cannot see on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from context_engine.params import PARAMS
from context_engine.orchestrator import TurnPolicy, plan_turn
from dialog_state.state_machine import SessionState
from retrieval.engine import tokenize

MAX_TURNS = 10

# Cues that the shopper is discarding an earlier preference rather than adding
# to it. The evaluator's intent-override script uses "ignore my earlier
# preference" / "What I need is:"; real phrasings vary, so keep this broad.
OVERRIDE_RE = re.compile(
    r"\b(actually|instead|ignore my earlier|ignore my previous|forget (?:the|my|about)|"
    r"scratch that|never ?mind|changed my mind|on second thought|rather than)\b",
    re.I,
)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _distill_demandingness(profile: dict) -> tuple[float, bool]:
    """How exacting is this buyer? Drives how hard the ranker leans on exact
    slot / phrase matches versus broad similarity. Returns ``(score01, is_critical)``
    where ``is_critical`` reproduces the legacy binary check verbatim."""
    rating_style = str(profile.get("rating_style", "")).lower()
    is_critical = "critical" in rating_style

    score = 0.5
    if is_critical:
        score += 0.30
    elif "mixed" in rating_style:
        score += 0.10
    elif "positive" in rating_style:
        score -= 0.10

    prior = profile.get("average_prior_rating")
    if isinstance(prior, (int, float)):
        if prior <= 2.0:
            score += 0.15
        elif prior <= 3.0:
            score += 0.05
        elif prior >= 5.0:
            score -= 0.05

    return _clip(score, 0.0, 1.0), is_critical


@dataclass
class ContextVector:
    """The distilled, per-turn view of the session. Plain fields only so it can
    be dumped straight to JSON for the before/after write-up."""

    turn: int
    turns_left: int

    intent: str
    intent_stable_turns: int

    pool_size: int
    pool_trend: str  # "converging" | "stagnant" | "expanding" | "unknown"
    stagnant_turns: int
    broaden: bool

    confirmed_constraints: int
    disclosed_phrase_count: int
    override_count: int
    turns_since_override: int | None
    has_budget: bool

    demandingness: float
    rating_style_critical: bool
    price_sensitivity: float
    tag_focus: float

    maturity: float
    precision_bias: float

    policy: TurnPolicy = field(repr=False, default=None)  # type: ignore[assignment]

    def as_dict(self) -> dict:
        data = {key: getattr(self, key) for key in self.__dataclass_fields__ if key != "policy"}
        data["policy"] = None if self.policy is None else vars(self.policy)
        return data


class ContextDistiller:
    """Per-session running memory + the distillation step itself."""

    def __init__(self, profile: dict | None) -> None:
        self.profile = profile or {}
        self.demandingness, self.rating_style_critical = _distill_demandingness(self.profile)
        self.price_sensitivity = 0.5  # wired but neutral: no informative price signal in this dataset
        preference_tags = self.profile.get("preference_tags") or []
        self.preference_terms = set(tokenize(" ".join(str(tag) for tag in preference_tags)))

        self._intent_history: list[str] = []
        self._pool_history: list[int] = []
        self._constraint_history: list[int] = []
        self._override_turns: list[int] = []

    # -- helpers -----------------------------------------------------------
    def _pool_trend(self, pool_size: int) -> str:
        if not self._pool_history:
            return "unknown"
        previous = self._pool_history[-1]
        if pool_size <= previous * 0.9:
            return "converging"
        if pool_size >= previous:
            return "stagnant"
        return "expanding"

    def _intent_stable_turns(self, intent: str) -> int:
        stable = 1
        for past in reversed(self._intent_history):
            if past == intent:
                stable += 1
            else:
                break
        return stable

    def _tag_focus(self, state: SessionState) -> float:
        """How much of the buyer's long-term preference vocabulary the current
        session has actually surfaced. 0.5 (neutral) when there is nothing to
        compare against, so it never perturbs a profile-less session."""
        if not self.preference_terms:
            return 0.5
        surfaced = state.slot_tokens() | set(state.generic_terms)
        hit = len(self.preference_terms & surfaced) / len(self.preference_terms)
        return _clip(0.5 + hit / 2.0, 0.0, 1.0)

    def _precision_bias(
        self,
        *,
        turn: int,
        intent: str,
        confirmed_constraints: int,
        disclosed_phrase_count: int,
        override_count: int,
        pool_trend: str,
        broaden: bool,
    ) -> float:
        p = PARAMS
        bias = (
            p["pb_constraint"] * min(confirmed_constraints, p["pb_constraint_cap"])
            + p["pb_turn"] * min(turn - 1, p["pb_turn_cap"])
            + p["pb_override"] * override_count
            + p["pb_phrase"] * min(disclosed_phrase_count, p["pb_phrase_cap"])
        )
        if broaden or pool_trend == "stagnant":
            bias -= p["pb_stagnant_penalty"]
        if intent == "browsing":
            bias *= p["pb_browsing_scale"]
        return _clip(bias, 0.0, p["pb_max"])

    # -- the distillation step ------------------------------------------------
    def distill(
        self,
        state: SessionState,
        intent: str,
        pool_size: int,
        turn: int,
        user_message: str = "",
    ) -> ContextVector:
        if OVERRIDE_RE.search(user_message or ""):
            if not self._override_turns or self._override_turns[-1] != turn:
                self._override_turns.append(turn)

        confirmed_constraints = len(state.slots)
        disclosed_phrase_count = len(state.disclosed_phrases)
        override_count = len(self._override_turns)
        turns_since_override = (
            turn - self._override_turns[-1] if self._override_turns else None
        )
        pool_trend = self._pool_trend(pool_size)
        intent_stable_turns = self._intent_stable_turns(intent)

        maturity = _clip(
            0.5 * min((turn - 1) / 5.0, 1.0)
            + 0.5 * min(confirmed_constraints / 3.0, 1.0),
            0.0,
            1.0,
        )
        precision_bias = self._precision_bias(
            turn=turn,
            intent=intent,
            confirmed_constraints=confirmed_constraints,
            disclosed_phrase_count=disclosed_phrase_count,
            override_count=override_count,
            pool_trend=pool_trend,
            broaden=state.broaden,
        )

        vector = ContextVector(
            turn=turn,
            turns_left=max(0, MAX_TURNS - turn),
            intent=intent,
            intent_stable_turns=intent_stable_turns,
            pool_size=pool_size,
            pool_trend=pool_trend,
            stagnant_turns=state.stagnant_turns,
            broaden=state.broaden,
            confirmed_constraints=confirmed_constraints,
            disclosed_phrase_count=disclosed_phrase_count,
            override_count=override_count,
            turns_since_override=turns_since_override,
            has_budget=state.budget_target is not None,
            demandingness=self.demandingness,
            rating_style_critical=self.rating_style_critical,
            price_sensitivity=self.price_sensitivity,
            tag_focus=self._tag_focus(state),
            maturity=maturity,
            precision_bias=precision_bias,
        )
        vector.policy = plan_turn(vector)

        self._intent_history.append(intent)
        self._pool_history.append(pool_size)
        self._constraint_history.append(confirmed_constraints)
        return vector
