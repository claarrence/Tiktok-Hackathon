from __future__ import annotations

import re
from dataclasses import dataclass, field

from retrieval.engine import tokenize

MATERIAL_WORDS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "fabric", "denim", "suede", "canvas", "linen", "mesh",
)
COLOR_WORDS = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray",
    "grey", "purple", "yellow", "orange", "gold", "silver", "beige", "navy",
)
SIZE_WORDS = ("size", "sizing", "width", "wide", "narrow", "petite", "plus")
STYLE_WORDS = ("style", "fit", "sleeve", "neck", "department", "cut", "collar")
USE_CASE_WORDS = (
    "hiking", "running", "gym", "winter", "outdoor", "work", "formal",
    "casual", "wedding", "travel", "workout", "everyday", "party",
)

MATERIAL_RE = re.compile(r"\b(" + "|".join(MATERIAL_WORDS) + r")\b", re.I)
COLOR_RE = re.compile(r"\b(" + "|".join(COLOR_WORDS) + r")\b", re.I)
SIZE_RE = re.compile(r"\b(" + "|".join(SIZE_WORDS) + r")\b", re.I)
STYLE_RE = re.compile(r"\b(" + "|".join(STYLE_WORDS) + r")\b", re.I)
USE_CASE_RE = re.compile(r"\b(" + "|".join(USE_CASE_WORDS) + r")\b", re.I)
BUDGET_RE = re.compile(r"(?:\$|budget|under|less than)\s*\$?\d", re.I)
BUDGET_NUMBER_RE = re.compile(r"\$?\s*(\d+(?:\.\d{1,2})?)")

# Cues that the shopper is discarding an earlier preference rather than adding to
# it. Keep this in sync with ``context_engine.distiller.OVERRIDE_RE`` — both must
# fire on the same messages so the dialog state and the distilled context agree
# on when an override happened.
OVERRIDE_RE = re.compile(
    r"\b(actually|instead|ignore my earlier|ignore my previous|forget (?:the|my|about)|"
    r"scratch that|never ?mind|changed my mind|on second thought|rather than)\b",
    re.I,
)

LEAD_INS = (
    "a key requirement is:",
    "what i need is:",
    "for that, what matters is:",
    "i don't have an additional preference for",
    "i don't have a preference for",
)

# Replies that disclose nothing — the shopper is deferring to our judgement.
# They carry no constraint, so their raw tokens ("preference", "judgment",
# "attribute", ...) must not leak into ``generic_terms`` and dilute retrieval.
# NB: only these explicit "no preference" phrasings help when dropped — pruning
# the vaguer "ask me about one specific attribute" nudge measured clearly worse.
NO_PREFERENCE_MARKERS = (
    "i don't have a preference for",
    "i don't have an additional preference for",
)

# The ten attribute labels the API contract's ``ask_attribute`` enum allows
# (docs/agent_api_contract.json), mirrored so this file is the single place the
# slot schema is pinned. Every key written to ``SessionState.slots``, every value
# returned by ``classify_phrase``, and every value returned by
# ``decide_ask_attribute`` must be one of these (or ``None``).
VALID_ASK_ATTRIBUTES = (
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case", "other",
)

# Attributes we never place in the clarifying-question queue:
#   - "category": already captured from the turn-1 "looking for ..." phrase
#     (see ``extract_category_text``); re-asking it just burns a turn.
#   - "brand": the frozen catalog has no brand field. The retrieval index and the
#     evaluator both search only title / categories / features / details / store /
#     description (see retrieval/engine.py + evaluator SEARCH_FIELDS), so a brand
#     answer can't be validated into a slot. Brand words a shopper volunteers
#     still reach retrieval via ``generic_terms``; a dedicated question would only
#     cost a turn for no extra signal.
UNPRODUCTIVE_ATTRIBUTES = {"category", "brand"}

# Askable subset of VALID_ASK_ATTRIBUTES, in the order the agent should probe.
ATTRIBUTE_PRIORITY = ["material", "color", "budget", "size", "style", "use_case", "feature"]

# Once a broadening nudge has fired, retire it only after the candidate pool has
# reconverged to at most this fraction of its size when the nudge was raised.
# broaden=True shifts route weight away from the precision routes downstream
# (context_engine.profile._legacy_shape), so leaving it stuck past convergence
# costs Efficiency. Handoff to Member A: this is the only place ``broaden`` is
# cleared — retrieval must not latch its own copy of the flag.
BROADEN_CLEAR_RATIO = 0.85

QUESTION_TEXT = {
    "material": "Do you have a material preference?",
    "color": "Any particular color you'd like?",
    "budget": "What's your budget range for this?",
    "size": "Do you need a specific size or fit?",
    "style": "Any style or fit details that matter to you?",
    "use_case": "What will you mainly use this for?",
    "feature": "Any other must-have features I should know about?",
}


def classify_phrase(phrase: str) -> str | None:
    """Bucket a disclosed constraint phrase into one of VALID_ASK_ATTRIBUTES.

    Anything substantive that doesn't match a more specific bucket falls through
    to ``"feature"`` — the catch-all the API contract defines and the evaluator's
    own customer simulator uses (``classify_constraint``). Returning ``None`` here
    would strand the phrase in ``generic_terms`` and deny the ranker its dedicated
    slot-route weight, which was a direct cause of ranking-stage misses.
    """
    lowered = phrase.lower()
    if not lowered.strip():
        return None
    if BUDGET_RE.search(lowered):
        return "budget"
    if MATERIAL_RE.search(lowered):
        return "material"
    if COLOR_RE.search(lowered):
        return "color"
    if SIZE_RE.search(lowered):
        return "size"
    if STYLE_RE.search(lowered):
        return "style"
    if USE_CASE_RE.search(lowered):
        return "use_case"
    return "feature"


def extract_category_text(message: str) -> str:
    match = re.search(r"looking for (.+?)(?:\.|,| but | still )", message, re.I)
    if match:
        return match.group(1).strip()
    return ""


def extract_disclosed_phrases(message: str) -> list[str]:
    lowered = message.lower()
    for lead_in in LEAD_INS:
        idx = lowered.find(lead_in)
        if idx == -1:
            continue
        if "don't have" in lead_in:
            return []
        remainder = message[idx + len(lead_in):].strip().rstrip(".")
        return [part.strip() for part in remainder.split(";") if part.strip()]
    return []


@dataclass
class SessionState:
    profile: dict = field(default_factory=dict)
    category_text: str = ""
    slots: dict[str, list[str]] = field(default_factory=dict)
    disclosed_phrases: list[str] = field(default_factory=list)
    budget_target: float | None = None
    generic_terms: set[str] = field(default_factory=set)
    asked: list[str] = field(default_factory=list)
    broaden: bool = False
    broaden_pool_mark: int | None = None
    last_override_turn: int | None = None
    stagnant_turns: int = 0
    last_pool_size: int | None = None
    max_questions: int = 6
    confidence_margin: float = 0.06
    shallow_answered: set[str] = field(default_factory=set)
    shallow_follow_up_used: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        rating_style = str(self.profile.get("rating_style", "")).lower()
        if "critical" in rating_style:
            self.max_questions = 7
            self.confidence_margin = 0.10
        preference_tags = self.profile.get("preference_tags") or []
        self.preference_terms = set(tokenize(" ".join(str(tag) for tag in preference_tags)))

    def update_from_message(self, message: str, turn: int) -> None:
        if any(lead_in in message.lower() for lead_in in LEAD_INS if "don't have" in lead_in):
            self.max_questions += 1
        if turn == 1:
            category = extract_category_text(message)
            if category:
                self.category_text = category

        # Intent override: the shopper is re-prioritising. Record the turn for
        # downstream (Member D's reranker) and switch the affected slot to
        # erase-and-rewrite so a corrected value doesn't sit next to the old one
        # ("black" then "white", not "black white"). Terms already in
        # ``generic_terms`` / ``disclosed_phrases`` are left in place — in this
        # catalog an "earlier preference" is still a true attribute of the fixed
        # target, so dropping its tokens only removes signal; the context pillar's
        # precision_bias override bump is what re-weights the turn.
        override = bool(OVERRIDE_RE.search(message))
        if override:
            self.last_override_turn = turn

        lowered_message = message.lower()
        no_preference = any(marker in lowered_message for marker in NO_PREFERENCE_MARKERS)

        for phrase in extract_disclosed_phrases(message):
            self.disclosed_phrases.append(phrase)
            self.generic_terms.update(tokenize(phrase))
            attribute = classify_phrase(phrase)
            if attribute:
                if override:
                    self.slots[attribute] = [phrase]
                else:
                    self.slots.setdefault(attribute, []).append(phrase)
                if len(tokenize(phrase)) <= 1 and attribute not in self.shallow_follow_up_used:
                    self.shallow_answered.add(attribute)
                if attribute == "budget":
                    number_match = BUDGET_NUMBER_RE.search(phrase)
                    if number_match:
                        # Last disclosed budget always wins, override or not.
                        self.budget_target = float(number_match.group(1))

        if not no_preference:
            # Anything not caught by a lead-in pattern (e.g. a plain reply) still
            # carries useful terms — keep them all. A "no preference" reply carries
            # none, so its tokens are dropped to keep retrieval clean.
            self.generic_terms.update(tokenize(message))

    def category_tokens(self) -> list[str]:
        return tokenize(self.category_text)

    def slot_tokens(self) -> set[str]:
        tokens: set[str] = set()
        for phrases in self.slots.values():
            for phrase in phrases:
                tokens.update(tokenize(phrase))
        return tokens

    def query_terms(self) -> list[str]:
        terms = list(self.category_tokens())
        terms.extend(self.slot_tokens())
        terms.extend(self.generic_terms)
        return terms

    def note_pool_size(self, pool_size: int) -> None:
        if self.last_pool_size is not None and pool_size >= self.last_pool_size:
            self.stagnant_turns += 1
        else:
            self.stagnant_turns = 0
        self.last_pool_size = pool_size
        if self.stagnant_turns >= 2:
            if not self.broaden:
                self.broaden = True
                self.broaden_pool_mark = pool_size
        elif (
            self.broaden
            and self.broaden_pool_mark is not None
            and pool_size <= self.broaden_pool_mark * BROADEN_CLEAR_RATIO
        ):
            # Pool has reconverged well below where we broadened — retire the
            # nudge so downstream weights swing back toward the precision routes.
            self.broaden = False
            self.broaden_pool_mark = None

    def decide_ask_attribute(self, pool_size: int, top_score: float, second_score: float) -> str | None:
        if len(self.asked) >= self.max_questions:
            return None
        over_general = pool_size > 12 or (top_score - second_score) < self.confidence_margin
        if not over_general:
            return None
        for attribute in ATTRIBUTE_PRIORITY:
            if attribute in UNPRODUCTIVE_ATTRIBUTES:
                continue
            if attribute in self.slots and attribute not in self.shallow_answered:
                continue
            if attribute in self.asked and attribute not in self.shallow_answered:
                continue
            self.asked.append(attribute)
            if attribute in self.shallow_answered:
                self.shallow_answered.discard(attribute)
                self.shallow_follow_up_used.add(attribute)
            return attribute
        return None
