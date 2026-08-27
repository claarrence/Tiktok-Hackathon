from __future__ import annotations

import json
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
    and a token-Jaccard route standing in for dense vector similarity —
    kept dependency-free and fully in-memory per the challenge's constraints.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.products: dict[str, dict] = {}
        self.doc_tokens: dict[str, set[str]] = {}
        self.category_inverted: dict[str, set[str]] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
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
                self.doc_tokens[asin] = set(
                    tokenize(" ".join([title, categories, features, details, store]))
                )
                for token in set(tokenize(categories)):
                    self.category_inverted.setdefault(token, set()).add(asin)

                batch.append((asin, title, categories, features, details, store, description))
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
        self.connection.commit()

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
        top = overlap.most_common(limit)
        denom = len(wanted)
        return {asin: count / denom for asin, count in top}

    def vector_route(
        self, query_terms: list[str], candidate_ids: set[str], limit: int = 200
    ) -> dict[str, float]:
        """Jaccard token-overlap similarity — a lightweight, offline stand-in for
        dense embedding similarity, scoped to the candidate pool for speed."""
        query_set = set(query_terms)
        if not query_set or not candidate_ids:
            return {}
        scored: list[tuple[str, float]] = []
        for asin in candidate_ids:
            doc = self.doc_tokens.get(asin)
            if not doc:
                continue
            union = len(query_set | doc)
            if not union:
                continue
            jaccard = len(query_set & doc) / union
            if jaccard > 0:
                scored.append((asin, jaccard))
        scored.sort(key=lambda item: item[1], reverse=True)
        return dict(scored[:limit])
