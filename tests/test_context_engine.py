from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest

from context_engine.distiller import ContextDistiller, ContextVector, _distill_demandingness
from context_engine.orchestrator import plan_turn
from context_engine.params import PARAMS
from context_engine.profile import BASE_WEIGHTS, adaptive_weights
from dialog_state.state_machine import SessionState


def legacy_adaptive_weights(intent: str, broaden: bool, critical: bool) -> dict[str, float]:
    """Oracle: the pre-pillar intent-only rule table, verbatim."""
    weights = dict(BASE_WEIGHTS)
    if intent == "buying":
        weights["slot"] += 0.15
        weights["category"] += 0.05
        weights["vector"] -= 0.10
        weights["keyword"] -= 0.10
    else:
        weights["vector"] += 0.10
        weights["category"] -= 0.05
        weights["slot"] -= 0.05
    if broaden:
        weights["category"] = max(0.0, weights["category"] - 0.15)
        weights["vector"] += 0.10
        weights["keyword"] += 0.05
    if critical:
        weights["slot"] += 0.05
    total = sum(max(0.0, v) for v in weights.values()) or 1.0
    return {k: max(0.0, v) / total for k, v in weights.items()}


def neutral_vector(intent="browsing", broaden=False, critical=False) -> ContextVector:
    """A ContextVector with every trajectory/personalization signal at its
    no-op value, so adaptive_weights must fall back to the legacy table."""
    return ContextVector(
        turn=1, turns_left=9, intent=intent,
        pool_size=0, pool_trend="unknown", stagnant_turns=0, broaden=broaden,
        confirmed_constraints=0, disclosed_phrase_count=0, override_count=0,
        turns_since_override=None, has_budget=False,
        demandingness=0.5, rating_style_critical=critical,
        tag_focus=0.5, precision_bias=0.0,
    )


class BackwardCompatTest(unittest.TestCase):
    def test_neutral_vector_reproduces_legacy_weights(self) -> None:
        for intent in ("buying", "browsing"):
            for broaden in (False, True):
                for critical in (False, True):
                    with self.subTest(intent=intent, broaden=broaden, critical=critical):
                        got = adaptive_weights(neutral_vector(intent, broaden, critical))
                        want = legacy_adaptive_weights(intent, broaden, critical)
                        for key in BASE_WEIGHTS:
                            self.assertAlmostEqual(got[key], want[key], places=9)

    def test_zeroed_params_disable_all_trajectory_shaping(self) -> None:
        original = dict(PARAMS)
        try:
            for key in list(PARAMS):
                if key.startswith(("pb_", "w_")) and not key.endswith(("_cap", "_scale", "_max")):
                    PARAMS[key] = 0.0
            vec = neutral_vector("buying")
            vec.precision_bias = 0.9
            vec.demandingness = 1.0
            vec.tag_focus = 1.0
            got = adaptive_weights(vec)
            want = legacy_adaptive_weights("buying", False, False)
            for key in BASE_WEIGHTS:
                self.assertAlmostEqual(got[key], want[key], places=9)
        finally:
            PARAMS.clear()
            PARAMS.update(original)


class WeightInvariantTest(unittest.TestCase):
    def test_weights_normalized_and_non_negative(self) -> None:
        vec = neutral_vector("buying")
        for pb in (0.0, 0.2, 0.4, 0.6):
            for dem in (0.3, 0.5, 0.9):
                vec.precision_bias = pb
                vec.demandingness = dem
                weights = adaptive_weights(vec)
                self.assertAlmostEqual(sum(weights.values()), 1.0, places=9)
                self.assertTrue(all(v >= 0.0 for v in weights.values()))

    def test_precision_bias_moves_weight_from_recall_to_precision(self) -> None:
        base = adaptive_weights(neutral_vector("buying"))
        hot = neutral_vector("buying")
        hot.precision_bias = 0.5
        shifted = adaptive_weights(hot)
        self.assertGreater(shifted["slot"], base["slot"])
        self.assertGreater(shifted["phrase"], base["phrase"])
        self.assertLess(shifted["vector"], base["vector"])
        self.assertLess(shifted["keyword"], base["keyword"])


class DemandingnessTest(unittest.TestCase):
    def test_rating_style_and_prior_rating_map_to_score(self) -> None:
        crit, is_crit = _distill_demandingness({"rating_style": "critical", "average_prior_rating": 1.0})
        pos, _ = _distill_demandingness({"rating_style": "usually positive", "average_prior_rating": 5.0})
        mixed, _ = _distill_demandingness({"rating_style": "mixed", "average_prior_rating": 3.0})
        self.assertTrue(is_crit)
        self.assertGreater(crit, mixed)
        self.assertGreater(mixed, pos)
        self.assertGreaterEqual(pos, 0.0)
        self.assertLessEqual(crit, 1.0)

    def test_missing_profile_is_neutral(self) -> None:
        score, is_crit = _distill_demandingness({})
        self.assertEqual(score, 0.5)
        self.assertFalse(is_crit)


class DistillerTrajectoryTest(unittest.TestCase):
    def _state(self, profile=None):
        return SessionState(profile=profile or {"rating_style": "usually positive", "preference_tags": []})

    def test_accumulating_constraints_raises_precision_bias(self) -> None:
        distiller = ContextDistiller({"rating_style": "usually positive", "preference_tags": []})
        state = self._state()
        state.update_from_message("I'm looking for hiking boots. A key requirement is: waterproof leather.", 1)
        state.note_pool_size(200)
        v1 = distiller.distill(state, "buying", 200, 1, "A key requirement is: waterproof leather.")

        state.update_from_message("For that, what matters is: ankle support; vibram sole.", 2)
        state.note_pool_size(120)
        v2 = distiller.distill(state, "buying", 120, 2, "For that, what matters is: ankle support; vibram sole.")

        self.assertGreaterEqual(v2.confirmed_constraints, v1.confirmed_constraints)
        self.assertGreater(v2.precision_bias, v1.precision_bias)
        self.assertEqual(v2.pool_trend, "converging")

    def test_override_phrase_is_detected_and_counted_once(self) -> None:
        distiller = ContextDistiller({"preference_tags": []})
        state = self._state()
        state.update_from_message("I'm looking for a jacket. I prefer a slim fit.", 1)
        state.note_pool_size(300)
        distiller.distill(state, "browsing", 300, 1, "I prefer a slim fit.")

        msg = "Actually, ignore my earlier preference. What I need is: insulated for winter."
        state.update_from_message(msg, 2)
        state.note_pool_size(280)
        v2 = distiller.distill(state, "buying", 280, 2, msg)
        # a second distill on the same turn must not double count
        v2b = distiller.distill(state, "buying", 280, 2, msg)

        self.assertEqual(v2.override_count, 1)
        self.assertEqual(v2b.override_count, 1)
        self.assertEqual(v2.turns_since_override, 0)

    def test_stagnant_pool_penalizes_precision_bias(self) -> None:
        # Same disclosed constraints on turn 2; only the pool trajectory differs.
        def bias_for(second_pool: int) -> float:
            distiller = ContextDistiller({"preference_tags": []})
            state = self._state()
            state.update_from_message(
                "I'm looking for trail shoes. A key requirement is: gore-tex waterproof upper.", 1
            )
            state.note_pool_size(400)
            distiller.distill(state, "buying", 400, 1, "gore-tex waterproof upper")
            state.update_from_message("For that, what matters is: rock plate; lug outsole.", 2)
            state.note_pool_size(second_pool)
            return distiller.distill(state, "buying", second_pool, 2, "rock plate; lug outsole").precision_bias

        converging = bias_for(150)
        stagnant = bias_for(500)
        self.assertGreater(converging, 0.0)
        self.assertLess(stagnant, converging)

    def test_tag_focus_neutral_without_preference_tags(self) -> None:
        distiller = ContextDistiller({"preference_tags": []})
        state = self._state()
        state.update_from_message("I'm looking for socks.", 1)
        state.note_pool_size(100)
        vec = distiller.distill(state, "browsing", 100, 1, "socks")
        self.assertEqual(vec.tag_focus, 0.5)

    def test_context_vector_is_json_serializable(self) -> None:
        distiller = ContextDistiller({"rating_style": "critical", "preference_tags": ["fit", "warmth"]})
        state = self._state({"rating_style": "critical", "preference_tags": ["fit", "warmth"]})
        state.update_from_message("I'm looking for gloves. A key requirement is: touchscreen fingertips.", 1)
        state.note_pool_size(90)
        vec = distiller.distill(state, "buying", 90, 1, "touchscreen fingertips")
        json.dumps(vec.as_dict())  # must not raise


class OrchestratorTest(unittest.TestCase):
    def test_force_answer_only_on_final_turn(self) -> None:
        early = plan_turn(neutral_vector("buying"))
        self.assertFalse(early.force_answer)
        self.assertTrue(early.allow_ask)

        last = neutral_vector("buying")
        last.turn = 10
        last.turns_left = 0
        policy = plan_turn(last)
        self.assertTrue(policy.force_answer)
        self.assertFalse(policy.allow_ask)

    def test_over_general_pool_widens_with_constraints(self) -> None:
        v = neutral_vector("buying")
        v.turn = 5
        v.confirmed_constraints = 3
        self.assertEqual(plan_turn(v).over_general_pool, 16)


class AgentPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from starter.agent import Agent
        cls.agent = Agent("data/catalog.jsonl")

    def test_final_turn_never_asks(self) -> None:
        self.agent.reset("ctx-final", {
            "purchase_frequency": "3-4 prior purchases", "average_prior_rating": 4.0,
            "rating_style": "mixed", "preference_tags": ["fit"], "summary": "x",
        })
        response = None
        for turn in range(1, 11):
            response = self.agent.respond("ctx-final", "I'm looking for a scarf, still exploring.", turn, 10)
        self.assertEqual(response["ask_attribute"], None)
        self.assertLessEqual(len(response["recommendations"]), 10)


_SLOW = unittest.skipUnless(
    os.environ.get("CTX_SLOW_TESTS"),
    "slow (loads the catalog / runs the evaluator); set CTX_SLOW_TESTS=1",
)


@_SLOW
class SlowIntegrationTest(unittest.TestCase):
    """Opt-in checks that actually exercise the full pipeline."""

    _ONE_TURN = (
        "from starter.agent import Agent; "
        "a = Agent('data/catalog.jsonl'); a.reset('s', {}); "
        "r = a.respond('s', \"I'm looking for hiking boots. A key requirement is: waterproof.\", 1, 10); "
        "print(','.join(x['parent_asin'] for x in r['recommendations']))"
    )

    def _run_one_turn(self, hash_seed: str) -> str:
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        out = subprocess.run(
            [sys.executable, "-c", self._ONE_TURN],
            capture_output=True, text=True, env=env, check=True,
        )
        return out.stdout.strip()

    def test_recommendations_are_hash_seed_independent(self) -> None:
        # Two *separate processes* with different PYTHONHASHSEED — the only way
        # to actually catch set-iteration order leaking into the ranking.
        self.assertEqual(self._run_one_turn("0"), self._run_one_turn("1"))

    def test_technicalscore_stays_above_floor(self) -> None:
        # Guards against a bad params.py retune silently regressing the pillar.
        from evaluator.local_evaluator import load_jsonl, catalog_index, evaluate
        from starter.agent import Agent

        samples = load_jsonl("data/public_set.jsonl")
        cids, cats, prods = catalog_index("data/catalog.jsonl")
        result = evaluate(Agent("data/catalog.jsonl"), samples, cids, cats, prods)
        self.assertGreaterEqual(result["recommended_technical_score"], 0.61)


if __name__ == "__main__":
    unittest.main()
