from __future__ import annotations

import math

from retrieval.engine import tokenize

# Second-pass disambiguation: candidates within this margin of the leading
# score are treated as a tie cluster rather than a settled ranking. Global
# IDF/phrase-match already fed into the first pass but doesn't distinguish
# terms that are common *within this specific cluster of near-duplicates*
# even when they're globally rare (e.g. "Imported" or "alloy" said once,
# shared by every candidate in an already-filtered-down necklace/jacket
# cluster) — see project-plan.md notes on Buying's rank-1 rate.
TIE_BAND = 0.08
TIE_CLUSTER_CAP = 20
# Local document frequency over a cluster this small is noisy — one
# incidental co-occurrence (e.g. a disclosed "color: green" contributing the
# label word "color" itself, which happens to match an unrelated product's
# "Color" details key) can swing a token's local weight sharply. Nudging the
# original score by at most the tie band itself keeps that noise from ever
# overturning a lead the first pass earned outside the band, while still
# letting it settle ties genuinely inside it.
LOCAL_BONUS_SCALE = TIE_BAND / 2


def _rating_prior(product: dict) -> float:
    rating = product.get("average_rating")
    if not isinstance(rating, (int, float)):
        return 0.0
    return max(0.0, min(1.0, (rating - 3.0) / 2.0))


def _price_score(product: dict, budget_target: float | None) -> float:
    if budget_target is None:
        return 0.0
    price = product.get("price")
    if not isinstance(price, (int, float)):
        return 0.0
    diff_ratio = abs(price - budget_target) / max(budget_target, 1e-6)
    return max(0.0, 1.0 - diff_ratio)


def _phrase_score(raw_text: str, disclosed_phrases: list[str]) -> float:
    # Disclosed constraint phrases are lifted near-verbatim from the target
    # product's own text — a substring hit is a much stronger precision
    # signal than bag-of-words overlap for telling near-duplicates apart.
    meaningful = [phrase.strip().lower() for phrase in disclosed_phrases if len(phrase.strip()) >= 3]
    if not meaningful:
        return 0.0
    hits = sum(1 for phrase in meaningful if phrase in raw_text)
    return hits / len(meaningful)


def _disambiguate_ties(
    scored: list[tuple[str, float]],
    raw_text: dict[str, str],
    disclosed_phrases: list[str],
) -> list[tuple[str, float]]:
    """Re-rank the leading tie cluster using phrase rarity computed within
    that cluster, instead of catalog-wide. A phrase can be globally rare
    (surviving the first pass's IDF weighting) and still be worthless for
    telling two near-duplicate finalists apart if both of them contain it.

    Deliberately phrase-substring only, not bag-of-words over single tokens:
    a lone word like "color" or "green" shows up incidentally all over long
    marketing/spec text for reasons that have nothing to do with the
    disclosed constraint (a stray "Color:" details key, an unrelated "green
    initiative" blurb), and with a cluster this small (single-digit to ~20
    candidates) one coincidental hit is enough to swing its whole local
    weight. A multi-word phrase match doesn't have that failure mode.
    """
    if len(scored) < 2:
        return scored
    top_score = scored[0][1]
    cluster = [item for item in scored[: TIE_CLUSTER_CAP] if top_score - item[1] <= TIE_BAND]
    if len(cluster) < 2:
        return scored

    meaningful_phrases = [phrase.strip().lower() for phrase in disclosed_phrases if len(phrase.strip()) >= 3]
    if not meaningful_phrases:
        return scored

    cluster_ids = [asin for asin, _ in cluster]
    n = len(cluster_ids)

    def local_idf(document_frequency: int) -> float:
        # No +1 floor: a phrase every candidate in the cluster shares (the
        # common case for catalog boilerplate like "Imported" once you're
        # down to near-duplicates) must weight to ~0, not a flat minimum —
        # that's the whole point of scoring rarity *within the cluster*.
        return math.log((n + 1) / (document_frequency + 1))

    phrase_weight = {
        phrase: local_idf(sum(1 for asin in cluster_ids if phrase in raw_text.get(asin, "")))
        for phrase in meaningful_phrases
    }
    phrase_weight_total = sum(phrase_weight.values())
    if phrase_weight_total <= 0:
        return scored

    def local_bonus(asin: str) -> float:
        text = raw_text.get(asin, "")
        matched = sum(weight for phrase, weight in phrase_weight.items() if phrase in text)
        return matched / phrase_weight_total

    reordered = sorted(
        cluster,
        key=lambda item: (-(item[1] + LOCAL_BONUS_SCALE * local_bonus(item[0])), item[0]),
    )
    return reordered + scored[len(cluster):]


def rank(
    candidate_ids: set[str],
    keyword_scores: dict[str, float],
    category_scores: dict[str, float],
    vector_scores: dict[str, float],
    products: dict[str, dict],
    raw_text: dict[str, str],
    idf: dict[str, float],
    default_idf: float,
    slot_tokens: set[str],
    preference_terms: set[str],
    disclosed_phrases: list[str],
    budget_target: float | None,
    weights: dict[str, float],
    intent: str | None = None,
) -> list[tuple[str, float]]:
    """Local semantic-ranking stage: fuses the three retrieval routes with a
    slot-match precision signal and a personalization boost from the buyer's
    profile, standing in for an LLM reranker without needing a model API.

    ``idf``/``default_idf`` come from the same catalog-wide IDF the vector
    route uses (``RetrievalEngine.idf``) — one IDF computation shared across
    both, instead of two slightly different ones over different field sets.

    Route fusion uses each route's raw normalized score directly rather than
    Reciprocal Rank Fusion (RRF) -- RRF was tried (see project-plan.md) and,
    once its weights were properly tuned rather than reusing the raw-score
    weights, landed statistically tied with this simpler approach while
    losing MRR (especially on the Boundary scenario). Not worth the added
    RRF_K tuning surface for a tie.
    """
    hard_price_filter = intent == "buying" and budget_target is not None
    scored: list[tuple[str, float]] = []
    for asin in candidate_ids:
        product = products.get(asin)
        if not product:
            continue
        if hard_price_filter:
            price = product.get("price")
            if isinstance(price, (int, float)) and not (0.75 * budget_target <= price <= 1.25 * budget_target):
                continue
        doc_tokens = set(tokenize(" ".join(str(product.get(f, "")) for f in ("title", "features", "details"))))

        slot_match = 0.0
        if slot_tokens:
            total_weight = sum(idf.get(token, default_idf) for token in slot_tokens)
            if total_weight > 0:
                matched_weight = sum(idf.get(token, default_idf) for token in (slot_tokens & doc_tokens))
                slot_match = matched_weight / total_weight
        tag_match = 0.0
        if preference_terms:
            tag_match = len(preference_terms & doc_tokens) / len(preference_terms)

        score = (
            weights["keyword"] * keyword_scores.get(asin, 0.0)
            + weights["category"] * category_scores.get(asin, 0.0)
            + weights["vector"] * vector_scores.get(asin, 0.0)
            + weights["slot"] * slot_match
            + weights["tag"] * tag_match
            + weights["phrase"] * _phrase_score(raw_text.get(asin, ""), disclosed_phrases)
            + weights["price"] * _price_score(product, budget_target)
            + 0.02 * _rating_prior(product)
        )
        scored.append((asin, score))

    scored.sort(key=lambda item: (-item[1], item[0]))
    return _disambiguate_ties(scored, raw_text, disclosed_phrases)
