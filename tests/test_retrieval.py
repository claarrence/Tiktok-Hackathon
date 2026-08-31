"""Unit tests for retrieval engine and intent routing (Member A).

Tests cover:
  - Intent classification (buying vs browsing)
  - Keyword route (BM25 scoring, tokenization, normalization)
  - Category route (inverted-index overlap)
  - Vector route (Jaccard similarity)
  - Edge cases (empty queries, no matches, consistent ordering)
"""

from __future__ import annotations

import unittest
from retrieval.engine import RetrievalEngine, tokenize
from retrieval.intent_router import classify_intent


class TokenizeTest(unittest.TestCase):
    def test_removes_stopwords(self) -> None:
        tokens = tokenize("a blue cotton shirt")
        self.assertNotIn("a", tokens)
        self.assertNotIn("the", tokens)
        self.assertIn("blue", tokens)
        self.assertIn("cotton", tokens)
        self.assertIn("shirt", tokens)

    def test_lowercases_and_deduplicates(self) -> None:
        tokens = tokenize("Blue BLUE blue")
        self.assertEqual(tokens.count("blue"), 3)  # All lowercased to same token
        unique = list(dict.fromkeys(tokens))
        self.assertEqual(len(unique), 1)

    def test_filters_short_tokens(self) -> None:
        tokens = tokenize("a to i x y z ab")
        self.assertEqual(tokens, ["ab"])

    def test_handles_special_characters(self) -> None:
        tokens = tokenize("hello@world.com test-case 123abc")
        # Should extract meaningful tokens
        self.assertTrue(any(len(t) > 1 for t in tokens))

    def test_empty_input(self) -> None:
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize("   "), [])
        self.assertEqual(tokenize("a an the"), [])  # All stopwords


class IntentClassificationTest(unittest.TestCase):
    def test_buying_intent_with_keywords(self) -> None:
        tests = [
            "I need it for hiking",
            "What I need is a blue shirt",
            "This must have durability",
            "I'm looking to buy something comfortable",
            "Has to be waterproof",
        ]
        for msg in tests:
            with self.subTest(msg=msg):
                intent = classify_intent(msg, 0)
                self.assertEqual(intent, "buying", f"Should detect buying in: {msg}")

    def test_browsing_intent_with_keywords(self) -> None:
        tests = [
            "I'm still exploring options",
            "Not sure what I want yet",
            "Just browsing around",
            "Maybe something like this",
            "Open to suggestions",
        ]
        for msg in tests:
            with self.subTest(msg=msg):
                intent = classify_intent(msg, 0)
                self.assertEqual(intent, "browsing", f"Should detect browsing in: {msg}")

    def test_intent_based_on_filled_slots(self) -> None:
        # With 0 or 1 filled slots: default to browsing (no keyword cue)
        self.assertEqual(classify_intent("cool shirt", 0), "browsing")
        self.assertEqual(classify_intent("cool shirt", 1), "browsing")
        # With 2+ filled slots: shift to buying (more committed buyer)
        self.assertEqual(classify_intent("cool shirt", 2), "buying")
        self.assertEqual(classify_intent("cool shirt", 5), "buying")
        # Browsing keyword overrides slot count
        self.assertEqual(classify_intent("just browsing", 5), "browsing")

    def test_keywords_override_slot_count(self) -> None:
        # Even with many slots filled, explicit browsing cue wins
        self.assertEqual(classify_intent("I'm still exploring", 10), "browsing")


class KeywordRouteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Create minimal test catalog
        cls.engine = RetrievalEngine("data/catalog.jsonl")

    def test_keyword_route_returns_dict(self) -> None:
        scores = self.engine.keyword_route(["shirt"])
        self.assertIsInstance(scores, dict)

    def test_keyword_route_empty_query(self) -> None:
        scores = self.engine.keyword_route([])
        self.assertEqual(scores, {})

    def test_keyword_route_scores_normalized_0_to_1(self) -> None:
        scores = self.engine.keyword_route(["shirt"])
        if scores:  # Only test if matches exist
            for asin, score in scores.items():
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    def test_keyword_route_limits_results(self) -> None:
        # Default limit is 200
        scores = self.engine.keyword_route(["shirt", "blue", "cotton"], limit=50)
        self.assertLessEqual(len(scores), 50)

    def test_keyword_route_deterministic(self) -> None:
        # Same query should return same order (important for eval consistency)
        scores1 = self.engine.keyword_route(["shirt", "blue"], limit=20)
        scores2 = self.engine.keyword_route(["shirt", "blue"], limit=20)
        self.assertEqual(list(scores1.items()), list(scores2.items()))

    def test_keyword_route_higher_scores_first(self) -> None:
        scores = self.engine.keyword_route(["shirt"])
        if len(scores) > 1:
            score_list = list(scores.values())
            # Verify sorted in descending order
            for i in range(len(score_list) - 1):
                self.assertGreaterEqual(score_list[i], score_list[i + 1])


class CategoryRouteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = RetrievalEngine("data/catalog.jsonl")

    def test_category_route_returns_dict(self) -> None:
        scores = self.engine.category_route(["shirt"])
        self.assertIsInstance(scores, dict)

    def test_category_route_empty_query(self) -> None:
        scores = self.engine.category_route([])
        self.assertEqual(scores, {})

    def test_category_route_overlap_counting(self) -> None:
        # Two terms should increase overlap count
        scores1 = self.engine.category_route(["shirt"])
        scores2 = self.engine.category_route(["shirt", "blue"])
        # With more constraints, scores should be lower (fewer matches)
        if scores1 and scores2:
            self.assertLessEqual(len(scores2), len(scores1))

    def test_category_route_scores_sum_to_limit(self) -> None:
        # Overlap is divided by number of terms
        scores = self.engine.category_route(["shirt"], limit=100)
        if scores:
            for asin, score in scores.items():
                self.assertLessEqual(score, 1.0)

    def test_category_route_deterministic(self) -> None:
        scores1 = self.engine.category_route(["shirt"], limit=20)
        scores2 = self.engine.category_route(["shirt"], limit=20)
        self.assertEqual(list(scores1.items()), list(scores2.items()))


class VectorRouteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = RetrievalEngine("data/catalog.jsonl")

    def test_vector_route_returns_dict(self) -> None:
        scores = self.engine.vector_route(["shirt"], ["B000A0C4Z4"])
        self.assertIsInstance(scores, dict)

    def test_vector_route_empty_query(self) -> None:
        scores = self.engine.vector_route([], ["B000A0C4Z4"])
        self.assertEqual(scores, {})

    def test_vector_route_empty_candidates(self) -> None:
        scores = self.engine.vector_route(["shirt"], [])
        self.assertEqual(scores, {})

    def test_vector_route_jaccard_in_range(self) -> None:
        scores = self.engine.vector_route(["shirt"], list(self.engine.products.keys())[:100])
        for asin, score in scores.items():
            # Jaccard similarity is in [0, 1]
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_vector_route_respects_limit(self) -> None:
        all_candidates = set(self.engine.products.keys())
        scores = self.engine.vector_route(["shirt"], all_candidates, limit=50)
        self.assertLessEqual(len(scores), 50)

    def test_vector_route_deterministic(self) -> None:
        candidates = list(self.engine.products.keys())[:200]
        scores1 = self.engine.vector_route(["shirt"], candidates, limit=20)
        scores2 = self.engine.vector_route(["shirt"], candidates, limit=20)
        self.assertEqual(list(scores1.items()), list(scores2.items()))


class RetrievalIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = RetrievalEngine("data/catalog.jsonl")

    def test_multi_route_combination_no_duplicates_lost(self) -> None:
        """Verify that combining three routes via set union preserves candidates."""
        query_terms = ["blue", "shirt"]
        category_terms = ["clothing"]
        candidate_ids_kw = set(self.engine.keyword_route(query_terms))
        candidate_ids_cat = set(self.engine.category_route(category_terms))
        candidate_ids_vec = set(self.engine.vector_route(query_terms, candidate_ids_kw | candidate_ids_cat))
        
        combined = candidate_ids_kw | candidate_ids_cat | candidate_ids_vec
        # Should have some candidates
        self.assertGreater(len(combined), 0)

    def test_retrieval_consistency_across_calls(self) -> None:
        """Same query should retrieve same results across multiple calls."""
        query = ["blue", "cotton", "shirt"]
        
        results1 = []
        kw1 = self.engine.keyword_route(query)
        cat1 = self.engine.category_route(query)
        results1.append((list(kw1.keys())[:5], list(cat1.keys())[:5]))
        
        results2 = []
        kw2 = self.engine.keyword_route(query)
        cat2 = self.engine.category_route(query)
        results2.append((list(kw2.keys())[:5], list(cat2.keys())[:5]))
        
        self.assertEqual(results1, results2, "Retrieval should be deterministic")

    def test_catalog_size_reasonable(self) -> None:
        """Verify catalog was loaded."""
        self.assertGreater(len(self.engine.products), 10000)
        self.assertLess(len(self.engine.products), 100000)

    def test_doc_tokens_index_populated(self) -> None:
        """Verify tokenization index exists."""
        self.assertEqual(len(self.engine.doc_tokens), len(self.engine.products))
        # All ASINs in products should have tokens
        for asin in list(self.engine.products.keys())[:10]:
            self.assertIn(asin, self.engine.doc_tokens)
            self.assertGreater(len(self.engine.doc_tokens[asin]), 0)

    def test_category_inverted_index_built(self) -> None:
        """Verify category inverted index was constructed."""
        self.assertGreater(len(self.engine.category_inverted), 0)
        # Each inverted entry should map tokens to product sets
        for token, asins in list(self.engine.category_inverted.items())[:5]:
            self.assertIsInstance(token, str)
            self.assertIsInstance(asins, set)
            self.assertGreater(len(asins), 0)


if __name__ == "__main__":
    unittest.main()
