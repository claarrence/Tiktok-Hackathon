"""Dialog-strategy unit tests (Pillar II): slot accumulation, intent override,
over-generality detection, and the broaden-flag lifecycle.

None of these load the catalog — they exercise ``SessionState`` directly.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from dialog_state.state_machine import (
    ATTRIBUTE_PRIORITY,
    BROADEN_CLEAR_RATIO,
    UNPRODUCTIVE_ATTRIBUTES,
    VALID_ASK_ATTRIBUTES,
    SessionState,
    classify_phrase,
)

CONTRACT_PATH = Path(__file__).resolve().parent.parent / "docs" / "agent_api_contract.json"


def _state(profile: dict | None = None) -> SessionState:
    return SessionState(profile=profile or {"preference_tags": []})


class ClassifyPhraseTest(unittest.TestCase):
    def test_specific_buckets(self) -> None:
        self.assertEqual(classify_phrase("100% Leather"), "material")
        self.assertEqual(classify_phrase("color: navy"), "color")
        self.assertEqual(classify_phrase("budget around $40"), "budget")
        self.assertEqual(classify_phrase("size: wide"), "size")
        self.assertEqual(classify_phrase("crew neck"), "style")
        self.assertEqual(classify_phrase("great for hiking"), "use_case")

    def test_unmatched_substantive_phrase_falls_back_to_feature(self) -> None:
        # Priority 1: previously returned None, stranding the phrase and starving
        # the ranker of its slot-route signal.
        self.assertEqual(classify_phrase("Buckle closure"), "feature")
        self.assertEqual(classify_phrase("moisture wicking"), "feature")
        self.assertEqual(classify_phrase("Imported"), "feature")

    def test_empty_phrase_is_none(self) -> None:
        self.assertIsNone(classify_phrase("   "))
        self.assertIsNone(classify_phrase(""))

    def test_every_return_value_is_a_valid_attribute(self) -> None:
        for phrase in ("100% Leather", "navy", "$40 budget", "wide", "crew neck",
                       "hiking", "reinforced toe box"):
            result = classify_phrase(phrase)
            self.assertIn(result, VALID_ASK_ATTRIBUTES)


class SlotAccumulationTest(unittest.TestCase):
    def test_slots_accumulate_incrementally_across_turns(self) -> None:
        state = _state()
        state.update_from_message(
            "I'm looking for running shoes. A key requirement is: mesh upper.", 1
        )
        self.assertEqual(state.slots["material"], ["mesh upper"])

        state.update_from_message("For that, what matters is: color: black.", 2)
        self.assertEqual(state.slots["material"], ["mesh upper"])  # untouched
        self.assertEqual(state.slots["color"], ["color: black"])
        self.assertIn("black", state.slot_tokens())

    def test_same_attribute_without_a_cue_appends(self) -> None:
        state = _state()
        state.update_from_message("A key requirement is: cotton.", 1)
        state.update_from_message("For that, what matters is: linen.", 2)
        self.assertEqual(state.slots["material"], ["cotton", "linen"])

    def test_category_captured_only_on_turn_one(self) -> None:
        state = _state()
        state.update_from_message("I'm looking for wool socks, but I'm still exploring.", 1)
        self.assertEqual(state.category_text, "wool socks")
        state.update_from_message("I'm looking for leather boots now.", 2)
        self.assertEqual(state.category_text, "wool socks")


class IntentOverrideTest(unittest.TestCase):
    def test_override_erases_and_rewrites_the_affected_slot(self) -> None:
        state = _state()
        state.update_from_message("A key requirement is: cotton.", 1)
        self.assertEqual(state.slots["material"], ["cotton"])

        state.update_from_message(
            "Actually, ignore my earlier preference. What I need is: linen.", 2
        )
        # erase + rewrite, not append -> no "cotton linen"
        self.assertEqual(state.slots["material"], ["linen"])
        self.assertNotIn("cotton", state.slot_tokens())

    def test_override_erases_earlier_value_from_the_affected_slot(self) -> None:
        state = _state()
        state.update_from_message("For that, what matters is: color: black.", 1)
        state.update_from_message(
            "Actually, scratch that. What I need is: color: red.", 2
        )
        self.assertEqual(state.slots["color"], ["color: red"])
        self.assertNotIn("black", state.slot_tokens())

    def test_override_leaves_the_free_text_bag_intact(self) -> None:
        # In this catalog an "earlier preference" is still a true attribute of the
        # fixed target, so generic_terms / disclosed_phrases are deliberately NOT
        # pruned on override — only the affected slot is rewritten. (Measured:
        # pruning them regressed intent_override MRR hard.)
        state = _state()
        state.update_from_message(
            "I'm looking for a belt. I prefer a woven elastic style.", 1
        )
        state.update_from_message(
            "Actually, ignore my earlier preference. What I need is: full grain leather.", 2
        )
        self.assertIn("woven", state.generic_terms)      # earlier terms retained
        self.assertIn("leather", state.generic_terms)    # new constraint added
        self.assertEqual(state.slots["material"], ["full grain leather"])

    def test_override_keeps_unrelated_slots(self) -> None:
        state = _state()
        state.update_from_message("A key requirement is: mesh upper.", 1)
        state.update_from_message("For that, what matters is: color: black.", 2)
        state.update_from_message(
            "Actually, scratch that. What I need is: color: red.", 3
        )
        self.assertEqual(state.slots["material"], ["mesh upper"])  # survived
        self.assertEqual(state.slots["color"], ["color: red"])

    def test_override_turn_is_recorded_for_downstream(self) -> None:
        state = _state()
        state.update_from_message("A key requirement is: cotton.", 1)
        self.assertIsNone(state.last_override_turn)
        state.update_from_message(
            "Actually, ignore my earlier preference. What I need is: linen.", 3
        )
        self.assertEqual(state.last_override_turn, 3)


class OverGeneralityTest(unittest.TestCase):
    def test_broad_pool_triggers_a_clarifying_question_for_a_missing_slot(self) -> None:
        state = _state()
        attribute = state.decide_ask_attribute(pool_size=200, top_score=0.9, second_score=0.9)
        self.assertIn(attribute, ATTRIBUTE_PRIORITY)
        self.assertIn(attribute, state.asked)

    def test_tight_pool_and_confident_top_hit_asks_nothing(self) -> None:
        state = _state()
        self.assertIsNone(
            state.decide_ask_attribute(pool_size=5, top_score=0.9, second_score=0.1)
        )

    def test_shallow_disclosure_earns_one_followup_then_locks(self) -> None:
        state = _state()
        state.update_from_message("A key requirement is: cotton.", 1)  # single-token, shallow
        followup = state.decide_ask_attribute(pool_size=200, top_score=0.5, second_score=0.5)
        self.assertEqual(followup, "material")  # exactly one follow-up granted

        state.update_from_message("I don't have an additional preference for material.", 2)
        third_ask = state.decide_ask_attribute(pool_size=200, top_score=0.5, second_score=0.5)
        self.assertNotEqual(third_ask, "material")  # follow-up consumed, never asked again

    def test_specific_disclosure_is_not_re_asked(self) -> None:
        state = _state()
        state.update_from_message("A key requirement is: 96% Cotton, 4% Spandex.", 1)  # multi-token, specific
        asked = []
        for _ in range(len(ATTRIBUTE_PRIORITY)):
            got = state.decide_ask_attribute(pool_size=200, top_score=0.5, second_score=0.5)
            if got is None:
                break
            asked.append(got)
        self.assertNotIn("material", asked)

    def test_brand_and_category_are_never_asked(self) -> None:
        # Priority 2: brand has no catalog field; category is captured on turn 1.
        state = _state()
        asked = []
        for _ in range(20):
            got = state.decide_ask_attribute(pool_size=500, top_score=0.5, second_score=0.5)
            if got is None:
                break
            asked.append(got)
        self.assertNotIn("brand", asked)
        self.assertNotIn("category", asked)
        self.assertTrue({"brand", "category"}.issubset(UNPRODUCTIVE_ATTRIBUTES))


class BroadenLifecycleTest(unittest.TestCase):
    def test_broaden_raises_after_two_stagnant_turns(self) -> None:
        state = _state()
        state.note_pool_size(300)
        self.assertFalse(state.broaden)
        state.note_pool_size(300)   # stagnant #1
        self.assertFalse(state.broaden)
        state.note_pool_size(305)   # stagnant #2
        self.assertTrue(state.broaden)
        self.assertEqual(state.broaden_pool_mark, 305)

    def test_broaden_clears_once_pool_reconverges(self) -> None:
        # Priority 3: the flag must not latch forever — it costs Efficiency.
        state = _state()
        for size in (300, 300, 305):
            state.note_pool_size(size)
        self.assertTrue(state.broaden)

        state.note_pool_size(260)  # a drop, but < 15% -> still broadened
        self.assertTrue(state.broaden)

        state.note_pool_size(int(305 * BROADEN_CLEAR_RATIO) - 1)  # well below the mark
        self.assertFalse(state.broaden)
        self.assertIsNone(state.broaden_pool_mark)

    def test_broaden_can_re_trigger_after_being_cleared(self) -> None:
        state = _state()
        for size in (300, 300, 305):
            state.note_pool_size(size)
        state.note_pool_size(100)          # clears
        self.assertFalse(state.broaden)
        state.note_pool_size(100)          # stagnant #1
        state.note_pool_size(100)          # stagnant #2
        self.assertTrue(state.broaden)
        self.assertEqual(state.broaden_pool_mark, 100)


class SlotSchemaLockTest(unittest.TestCase):
    def test_valid_ask_attributes_mirror_the_api_contract_enum(self) -> None:
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        enum = contract["turn_response"]["properties"]["ask_attribute"]["enum"]
        contract_labels = {value for value in enum if value is not None}
        self.assertEqual(set(VALID_ASK_ATTRIBUTES), contract_labels)

    def test_askable_priority_is_a_subset_of_the_schema(self) -> None:
        self.assertTrue(set(ATTRIBUTE_PRIORITY).issubset(set(VALID_ASK_ATTRIBUTES)))
        self.assertEqual(set(ATTRIBUTE_PRIORITY) & UNPRODUCTIVE_ATTRIBUTES, set())


if __name__ == "__main__":
    unittest.main()
