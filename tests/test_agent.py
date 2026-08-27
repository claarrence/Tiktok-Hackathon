from __future__ import annotations

import unittest

from starter.agent import Agent


class AgentSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = Agent("data/catalog.jsonl")

    def test_respond_matches_contract_shape(self) -> None:
        self.agent.reset(
            "test-session",
            {
                "purchase_frequency": "3-4 prior purchases",
                "average_prior_rating": 4.5,
                "rating_style": "usually positive",
                "preference_tags": ["comfort", "fit"],
                "summary": "Prior purchases emphasize comfort and fit.",
            },
        )
        response = self.agent.respond(
            "test-session", "I'm looking for earrings, but I'm still exploring.", 1, 10
        )
        self.assertIsInstance(response["message"], str)
        self.assertIn(response["ask_attribute"], (None, "material", "color", "budget", "size", "style", "use_case", "feature"))
        self.assertLessEqual(len(response["recommendations"]), 10)
        for item in response["recommendations"]:
            self.assertIn("parent_asin", item)

    def test_second_call_without_reset_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            self.agent.respond("never-reset", "hello", 1, 10)


if __name__ == "__main__":
    unittest.main()
