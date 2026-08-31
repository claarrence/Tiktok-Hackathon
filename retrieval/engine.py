from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "still", "that", "the", "this", "to", "want", "would", "you", "your",
}


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text or "")
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _flatten_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


class RetrievalEngine:
    """Multi-route candidate generation over the frozen catalog.

    Three routes feed the ranking stage: keyword (BM25 via SQLite FTS5),
    category (inverted-index overlap on the catalog's own category path),
    and a TF-IDF cosine route standing in for dense vector similarity —
    kept dependency-free and fully in-memory per the challenge's constraints.
    IDF down-weights generic catalog-wide words ("clothing", "comfortable")
    so the route actually discriminates between near-duplicate products,
    which plain token-Jaccard overlap cannot do.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.products: dict[str, dict] = {}
        self.raw_text: dict[str, str] = {}
        self.category_inverted: dict[str, set[str]] = {}
        self.doc_tfidf: dict[str, dict[str, float]] = {}
        self.doc_norm: dict[str, float] = {}
        self.idf: dict[str, float] = {}
        self.default_idf: float = 0.0
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        doc_term_counts: dict[str, Counter[str]] = {}
        doc_freq: Counter[str] = Counter()
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                asin = str(product["parent_asin"])
                title = _flatten_text(product.get("title"))
                categories = _flatten_text(product.get("categories"))
                features = _flatten_text(product.get("features"))
                details = _flatten_text(product.get("details"))
                store = _flatten_text(product.get("store"))
                description = _flatten_text(product.get("description"))

                self.products[asin] = product
                self.raw_text[asin] = " ".join(
                    [title, categories, features, details, store, description]
                ).lower()
                for token in set(tokenize(categories)):
                    self.category_inverted.setdefault(token, set()).add(asin)

                counts = Counter(tokenize(" ".join([title, categories, features, details, store])))
                doc_term_counts[asin] = counts
                doc_freq.update(counts.keys())

                batch.append((asin, title, categories, features, details, store, description))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
        self.connection.commit()

        total_docs = len(self.products)
        self.default_idf = math.log(total_docs) if total_docs else 0.0
        self.idf = {
            token: max(0.0, math.log(total_docs / (1 + df))) for token, df in doc_freq.items()
        }
        for asin, counts in doc_term_counts.items():
            vector = {
                token: (1.0 + math.log(count)) * self.idf.get(token, self.default_idf)
                for token, count in counts.items()
            }
            self.doc_tfidf[asin] = vector
            self.doc_norm[asin] = math.sqrt(sum(weight * weight for weight in vector.values()))

    def keyword_route(self, terms: list[str], limit: int = 200) -> dict[str, float]:
        unique = list(dict.fromkeys(terms))[:40]
        if not unique:
            return {}
        expression = " OR ".join(f'"{term}"' for term in unique)
        rows = self.connection.execute(
            "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) AS score "
            "FROM products WHERE products MATCH ? ORDER BY score LIMIT ?",
            (expression, limit),
        ).fetchall()
        if not rows:
            return {}
        scores = [row[1] for row in rows]
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        # sqlite bm25() is "lower is better" — invert and normalize to 0..1
        return {str(asin): 1.0 - ((score - lo) / span) for asin, score in rows}

    def category_route(self, category_terms: list[str], limit: int = 200) -> dict[str, float]:
        wanted = list(dict.fromkeys(category_terms))
        if not wanted:
            return {}
        overlap: Counter[str] = Counter()
        for token in wanted:
            for asin in self.category_inverted.get(token, ()):
                overlap[asin] += 1
        if not overlap:
            return {}
        denom = len(wanted)
        best_count = max(overlap.values())
        # A broad, widely-shared category path (e.g. hundreds of products all
        # under "Wallets") can produce far more than `limit` candidates tied
        # for the best overlap score. A hard `most_common(limit)` slice would
        # silently drop some of those ties based on arbitrary insertion
        # order, which can exclude the true target while an unrelated
        # tie-mate survives purely by luck. Every candidate tied for the best
        # score is kept, however many there are; `limit` only bounds the
        # lower, partial-match tiers where being cut is harmless.
        best_tier = {asin: best_count / denom for asin, count in overlap.items() if count == best_count}
        remaining = limit - len(best_tier)
        if remaining <= 0:
            return best_tier
        lower_tier = sorted(
            ((asin, count) for asin, count in overlap.items() if count != best_count),
            key=lambda item: (-item[1], item[0]),
        )[:remaining]
        best_tier.update({asin: count / denom for asin, count in lower_tier})
        return best_tier

    def vector_route(
        self, query_terms: list[str], candidate_ids: set[str], limit: int = 200
    ) -> dict[str, float]:
        """TF-IDF cosine similarity — a lightweight, offline stand-in for dense
        embedding similarity, scoped to the candidate pool for speed. Unlike
        plain token-Jaccard overlap, IDF down-weights catalog-wide generic
        words so the score actually separates near-duplicate products."""
        if not query_terms or not candidate_ids:
            return {}
        query_counts = Counter(query_terms)
        query_vec = {
            token: (1.0 + math.log(count)) * self.idf.get(token, self.default_idf)
            for token, count in query_counts.items()
        }
        query_norm = math.sqrt(sum(weight * weight for weight in query_vec.values()))
        if query_norm == 0:
            return {}
        scored: list[tuple[str, float]] = []
        for asin in candidate_ids:
            doc_norm = self.doc_norm.get(asin, 0.0)
            if not doc_norm:
                continue
            doc_vec = self.doc_tfidf[asin]
            dot = sum(weight * doc_vec[token] for token, weight in query_vec.items() if token in doc_vec)
            if dot <= 0:
                continue
            scored.append((asin, dot / (query_norm * doc_norm)))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return dict(scored[:limit])
