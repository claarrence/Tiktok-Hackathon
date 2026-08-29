from __future__ import annotations

import math

from retrieval.engine import tokenize


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


_IDF_CACHE: dict[int, tuple[dict[str, float], float]] = {}


def _catalog_idf(products: dict[str, dict]) -> tuple[dict[str, float], float]:
    cached = _IDF_CACHE.get(id(products))
    if cached is not None:
        return cached
    doc_freq: dict[str, int] = {}
    for product in products.values():
        tokens = set(tokenize(" ".join(str(product.get(f, "")) for f in ("title", "features", "details"))))
        for token in tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1
    total = len(products)
    idf = {token: max(0.0, math.log(total / (1 + df))) for token, df in doc_freq.items()}
    default_idf = max(0.0, math.log(total)) if total > 0 else 0.0
    result = (idf, default_idf)
    _IDF_CACHE[id(products)] = result
    return result


def rank(
    candidate_ids: set[str],
    keyword_scores: dict[str, float],
    category_scores: dict[str, float],
    vector_scores: dict[str, float],
    products: dict[str, dict],
    raw_text: dict[str, str],
    slot_tokens: set[str],
    preference_terms: set[str],
    disclosed_phrases: list[str],
    budget_target: float | None,
    weights: dict[str, float],
    intent: str | None = None,
) -> list[tuple[str, float]]:
    """Local semantic-ranking stage: fuses the three retrieval routes with a
    slot-match precision signal and a personalization boost from the buyer's
    profile, standing in for an LLM reranker without needing a model API."""
    hard_price_filter = intent == "buying" and budget_target is not None
    slot_idf, slot_idf_default = _catalog_idf(products)
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
            total_weight = sum(slot_idf.get(token, slot_idf_default) for token in slot_tokens)
            if total_weight > 0:
                matched_weight = sum(slot_idf.get(token, slot_idf_default) for token in (slot_tokens & doc_tokens))
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
    return scored
