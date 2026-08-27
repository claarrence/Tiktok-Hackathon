from __future__ import annotations

from pathlib import Path

from context_engine.profile import adaptive_weights
from dialog_state.state_machine import QUESTION_TEXT, SessionState
from retrieval.engine import RetrievalEngine
from retrieval.intent_router import classify_intent
from ranking.ranker import rank


class Agent:
    """Shopping Copilot agent: dual-track intent routing -> multi-route hybrid
    retrieval -> local semantic ranking, driven by a per-session dialog state
    machine and profile-adaptive context weighting. See docs in retrieval/,
    dialog_state/, context_engine/, and ranking/ for each pillar."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.engine = RetrievalEngine(catalog_path)
        self._sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(profile=user_profile or {})

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")

        state.update_from_message(user_message, turn)
        intent = classify_intent(user_message, filled_slots=len(state.slots))

        query_terms = state.query_terms()
        category_terms = state.category_tokens()
        keyword_scores = self.engine.keyword_route(query_terms)
        category_scores = self.engine.category_route(category_terms)
        candidate_ids = set(keyword_scores) | set(category_scores)
        if not candidate_ids:
            candidate_ids = set(category_scores) or set(keyword_scores)
        vector_scores = self.engine.vector_route(query_terms, candidate_ids)
        candidate_ids |= set(vector_scores)

        state.note_pool_size(len(candidate_ids))
        weights = adaptive_weights(state, intent)

        ranked = rank(
            candidate_ids,
            keyword_scores,
            category_scores,
            vector_scores,
            self.engine.products,
            self.engine.raw_text,
            state.slot_tokens(),
            state.preference_terms,
            state.disclosed_phrases,
            state.budget_target,
            weights,
        )
        top = ranked[:top_k]
        recommendations = [{"parent_asin": asin, "score": round(score, 4)} for asin, score in top]

        top_score = top[0][1] if top else 0.0
        second_score = top[1][1] if len(top) > 1 else 0.0
        ask_attribute = state.decide_ask_attribute(len(candidate_ids), top_score, second_score)
        message = QUESTION_TEXT.get(ask_attribute, "Here are the closest matches I found so far.")

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
