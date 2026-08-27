from __future__ import annotations

import re

BUYING_RE = re.compile(
    r"\b(requirement is|what i need is|must have|need it|has to be|looking to buy)\b", re.I
)
BROWSING_RE = re.compile(
    r"\b(still exploring|not sure|just browsing|maybe|open to|any suggestions)\b", re.I
)


def classify_intent(message: str, filled_slots: int) -> str:
    """Dual-track routing: 'buying' locks a high-precision filter track,
    'browsing' keeps the diverse cross-category track open. Falls back to
    slot density once the lexical cues run out (e.g. mid-conversation turns)."""
    if BUYING_RE.search(message):
        return "buying"
    if BROWSING_RE.search(message):
        return "browsing"
    return "buying" if filled_slots >= 2 else "browsing"
