# Retrieval & Intent Routing Design (Member A)

**Pillar I** of the Shopping Copilot system, handling candidate generation and user intent classification.

## Overview

The retrieval pillar enables **93.5% Hit Rate@10** on the public dev set by:
1. **Dual-track intent routing** — detecting whether the user is in "buying" (high-precision, constraint-locked) or "browsing" (diverse, exploratory) mode
2. **Multi-route hybrid retrieval** — combining keyword (BM25), category (inverted-index), and vector (Jaccard similarity) routes to avoid over-reliance on any single signal
3. **Deterministic candidate fusion** — reliably merging route results with consistent ordering to ensure stable evaluation across runs

## Design Decisions

### Intent Classification

**Approach:** Lightweight regex + slot-density fallback

- **Buying signals** (`BUYING_RE`): Keywords like "must have", "need it", "looking to buy" → locks high-precision track
- **Browsing signals** (`BROWSING_RE`): "still exploring", "just browsing", "open to" → keeps diverse pool
- **Fallback heuristic**: If neither pattern matches, check if the shopper has filed 2+ constraints (very likely buying)

**Why this works:**
- Conversational signals are noisy, but repeated patterns cover ~90% of intent shifts
- Slot count provides a fallback that adapts mid-session (e.g., buyer starts vague, becomes specific after turn 2)
- Regexes execute in milliseconds with zero dependencies

**Trade-offs:**
- Won't catch all nuance (e.g., "I want to look at blue shirts" is actually browsing but lacks explicit keyword)
- Mitigated by the ranking and context pillars, which re-weight routes per-turn based on pool convergence

### Keyword Route (BM25)

**Indexing:** SQLite FTS5 virtual table, pre-indexed at agent startup on:
- `title`, `categories`, `features`, `details`, `store`, `description` fields
- Uses standard `unicode61` tokenization with diacritics removal
- BM25 weights: `(0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)` for (unused, title, categories, features, details, store, description)

**Scoring & Normalization:**
- SQLite FTS5's native BM25 ("lower is better") is inverted and normalized to [0, 1] per query
- Top 200 results returned to downstream ranking

**Why BM25:**
- **Industry standard**: proven on millions of e-commerce products, handles term frequency saturation gracefully
- **In-memory SQLite**: no external index, zero latency, fully reproducible
- **Weights calibrated for Clothing+Shoes**: title (6.0) heavily weighted because "blue shirt" in title >> in description; categories (4.0) next because customers search by type; features (2.5) and details (2.5) equal weight for specification queries

**Limitations:**
- Requires all query terms to match the search expression (via OR) — sparse queries may miss relevant niche products
- No semantic understanding of synonymy (e.g., "trousers" vs "pants")
- Boundary scenario failures (3/10 misses) likely involve rare product names or typos

### Category Route (Inverted-Index Overlap)

**Indexing:** Built at startup by iterating the catalog and tokenizing the `categories` field for each product, storing inverted mappings (token → {ASIN set})

**Scoring:**
- For each candidate, count how many of the user's category tokens it contains
- Divide by the total number of unique category tokens requested → overlap ratio in [0, 1]
- Top 200 results returned

**Why inverted-index:**
- **Fast**: O(1) lookup per token, O(k) to merge k category terms
- **Interpretable**: "Found 2 out of 3 category keywords in this product" is easy to debug
- **Complementary to BM25**: catches products that are perfect categorical fits even if title keywords don't match exactly

**Limitations:**
- Only captures category terms, not all product text
- Requires tokenization consistency (mitigated by shared `tokenize()` function)

### Vector Route (Jaccard Similarity)

**Approach:** Token-level Jaccard similarity scoped to a candidate pool, standing in for dense embeddings

**Algorithm:**
1. Receive candidate pool from keyword + category routes (to avoid O(n) cost over entire catalog)
2. For each candidate, compute Jaccard(query_tokens, product_tokens) = |intersection| / |union|
3. Sort by Jaccard descending, return top 200

**Why Jaccard:**
- **No dependencies**: no embedding model, no external service, fully in-memory
- **Interpretable**: Jaccard directly measures token overlap as a similarity metric
- **Scoped to candidates**: avoids O(n) computation by only reranking the union of keyword + category results
- **Complementary**: catches products with high keyword overlap that BM25 might score lower

**Trade-offs:**
- Not as expressive as dense embeddings (no semantic knowledge of synonymy)
- Bounded by tokenization quality
- Best used as a re-ranker, not primary retrieval (hence scoped to candidate pool)

### Candidate Pool Fusion

**Process** (in `Agent.respond`, pillar I + III):
1. Compute keyword scores → set of ASINs
2. Compute category scores → set of ASINs
3. Merge (set union) and sort deterministically
4. Recompute vector scores scoped to merged pool
5. Merge all three routes via set union again
6. Pass sorted, deduplicated candidate list to ranking

**Why union, not intersection:**
- Intersection would be too restrictive (require keyword AND category AND vector match)
- Union ensures low recall doesn't cascade
- Ranking stage applies the hard filters (price, slot match) downstream

**Determinism:**
- Set results are sorted by ASIN string to ensure consistent ordering regardless of PYTHONHASHSEED
- Critical for reproducible eval scores (without this, small hash seed changes swing ~0.02 TechnicalScore)

## Testing

See `tests/test_retrieval.py` (31 unit tests):
- **Intent classification** (5 tests): keyword detection, slot-based fallback, override behavior
- **Keyword route** (6 tests): normalization, limits, determinism, score ranges
- **Category route** (5 tests): overlap counting, limit enforcement, determinism
- **Vector route** (6 tests): Jaccard range, scoping to candidates, determinism
- **Integration** (5 tests): multi-route combination, consistency, index initialization
- **Tokenization** (5 tests): stopword removal, case handling, special characters

**All tests pass** ✅ Confirms routes are stable and deterministic.

## Performance

**Public Dev Set (200 sessions):**
- **Overall Hit Rate@10:** 93.5% (target: >70%)
- **Scenarios:**
  - Buying: 95% (80 sessions) — high precision works well
  - Browsing: 97.5% (80 sessions) — diverse pool strategy effective
  - Intent Override: 86.7% (30 sessions) — handles mid-session goal changes
  - Boundary: 70% (10 sessions) — edge cases with rare/niche products

**Technical Score:** 0.7875 (78.75%)
- Hit Rate@10: 0.935 × 0.50 = 0.4675
- MRR: 0.6298 × 0.30 = 0.1889
- Efficiency: 0.6555 × 0.20 = 0.1311
- **Total: 0.7875** ← Solid baseline; Pillar II/III/IV add incremental gains

## Known Limitations & Future Work

1. **Boundary scenarios (70% hit rate):**
   - Typically "medium" difficulty sessions with niche product requirements
   - Mitigation: deeper context analysis (Pillar III) and semantic ranking (Pillar IV)

2. **Rare product names:**
   - BM25 can miss obscure SKUs or typo-prone names
   - Future: ASR/spell-correction preprocessing (out of scope for hackathon)

3. **Synonym handling:**
   - "Trousers" vs "pants", "jacket" vs "coat" — not resolved at retrieval layer
   - Mitigated by: (a) dense embeddings in ranking, (b) user clarification prompts (Pillar II)

4. **Category token collision:**
   - Some products appear in many categories, inflating category route scores
   - Mitigation: adaptive weight scaling based on pool convergence (Pillar III)

5. **No dynamic re-indexing:**
   - Catalog is frozen; live catalog updates would require index rebuild
   - Acceptable for static e-commerce snapshot; real systems need incremental updates

## Integration with Other Pillars

- **Pillar II (Dialog State):** Provides session state (slots, preferences) → query construction
- **Pillar III (Context Engine):** Reads candidate pool size, trajectory → adjusts route weights per turn
- **Pillar IV (Ranking):** Receives candidates + route scores → applies semantic re-ranking + slot/phrase matching

This design prioritizes **stability, interpretability, and speed** over maximum accuracy, reflecting the hackathon's time constraint and the need for an end-to-end working system.

---

*Test suite:* `tests/test_retrieval.py`  
*Source code:* `retrieval/engine.py`, `retrieval/intent_router.py`  
*Public set baseline:* Hit Rate@10 = 93.5%, Technical Score = 0.7875
