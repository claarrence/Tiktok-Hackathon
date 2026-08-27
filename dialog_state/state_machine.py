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

LEAD_INS = (
    "a key requirement is:",
    "what i need is:",
    "for that, what matters is:",
    "i don't have an additional preference for",
    "i don't have a preference for",
)

# Attributes never surfaced by real product data in this catalog's taxonomy —
# asking them wastes a turn, so they're excluded from the question queue.
UNPRODUCTIVE_ATTRIBUTES = {"category", "brand"}

ATTRIBUTE_PRIORITY = ["material", "color", "budget", "size", "style", "use_case", "feature"]

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
    lowered = phrase.lower()
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
    return None


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
    generic_terms: set[str] = field(default_factory=set)
    asked: list[str] = field(default_factory=list)
    broaden: bool = False
    stagnant_turns: int = 0
    last_pool_size: int | None = None
    max_questions: int = 6
    confidence_margin: float = 0.06

    def __post_init__(self) -> None:
        rating_style = str(self.profile.get("rating_style", "")).lower()
        if "critical" in rating_style:
            self.max_questions = 7
            self.confidence_margin = 0.10
        preference_tags = self.profile.get("preference_tags") or []
        self.preference_terms = set(tokenize(" ".join(str(tag) for tag in preference_tags)))

    def update_from_message(self, message: str, turn: int) -> None:
        if turn == 1:
            category = extract_category_text(message)
            if category:
                self.category_text = category
        for phrase in extract_disclosed_phrases(message):
            self.generic_terms.update(tokenize(phrase))
            attribute = classify_phrase(phrase)
            if attribute:
                self.slots.setdefault(attribute, []).append(phrase)
        # Anything not caught by a lead-in pattern (e.g. the override sentence
        # itself, or a plain reply) still carries useful terms — keep them all.
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
            self.broaden = True

    def decide_ask_attribute(self, pool_size: int, top_score: float, second_score: float) -> str | None:
        if len(self.asked) >= self.max_questions:
            return None
        over_general = pool_size > 12 or (top_score - second_score) < self.confidence_margin
        if not over_general:
            return None
        for attribute in ATTRIBUTE_PRIORITY:
            if attribute in UNPRODUCTIVE_ATTRIBUTES:
                continue
            if attribute in self.slots:
                continue
            if attribute in self.asked:
                continue
            self.asked.append(attribute)
            return attribute
        return None
