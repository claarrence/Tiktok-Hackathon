# Devpost Written Project Description — DRAFT

Status: day-1 draft. Architecture section reflects the team's actual design; Tools/APIs/Libraries are placeholders until we decide during build. Update as we go rather than writing this fresh at the end.

## How our solution addresses the problem statement

Traditional keyword search can't tell "browsing for ideas" from "ready to buy," and it can't adapt mid-conversation as a shopper's intent sharpens or changes. Our agent is built around four pieces that map directly to that gap:

- **Dual-track intent routing** — every turn is classified as Buying (hard constraints, high-precision filter track) or Browsing (open-ended, diverse dense retrieval track), so the retrieval strategy matches the shopper's actual mode instead of one-size-fits-all search.
- **Hybrid retrieval → LLM ranking pipeline** — a multi-route retrieval stage (keyword + category + vector similarity) surfaces a wide, in-memory candidate pool, which an LLM semantic ranking stage then narrows to a precise Top-10.
- **Dynamic dialog state machine** — tracks two distinct conversational events: incremental slot accumulation ("also needs to be waterproof") versus intent override ("actually, forget shoes, show me bags"). When the candidate pool is still too broad (over-generality), the agent cuts off retrieval and asks one structured clarification question instead of returning a low-confidence guess.
- **Context & personalization engine** — distills dialog history into a running session state and profile (using the buyer's `purchase_frequency`, `rating_style`, `preference_tags`, `summary`) that continuously reshapes routing weights and ranking, so the agent's own guidance strategy adapts turn over turn rather than staying static.

This design is scored directly against the challenge's own metrics: retrieval breadth drives Hit Rate@10, LLM ranking precision drives MRR, and the clarification logic that avoids wasted turns drives MTTC — all within the hard 10-turn session cap.

*(To fill in once we have real numbers: our local Hit Rate@10 / MRR / MTTC vs. the published weak-BM25 baseline of 0.125 / 0.068 / 9.81, and where our biggest lift came from — the baseline's weakest scenario is Browsing at 0.025 Hit Rate@10, so that's likely to be our headline improvement.)*

## Development tools used
- Python 3.10+ *(confirm team's actual version/IDE — e.g. VSCode, PyCharm, Jupyter)*
- Git / GitHub for version control
- *(add: any notebook/experiment tooling, if used)*

## APIs used
- *TBD.* The organizer provides no hosted model access or API keys; a paid LLM is not required. Decide as a team whether the LLM ranking stage uses an external API (state which one, e.g. OpenAI/Anthropic) or a local/rule-based scorer, and document token usage, latency, and estimated cost either way (required by `docs/submission_rules.md`).
- If any external API is used, also state here whether the system has an offline fallback — required for final scoring, since network access may be disabled.

## Libraries and frameworks used
- *TBD.* The starter agent is Python-stdlib-only. List whatever gets added for retrieval/ranking (e.g. an embeddings library for the dense/vector track, a BM25 implementation, etc.) once Pillar I/IV owners decide.

## Datasets and assets used
- Amazon Reviews 2023 (McAuley Lab, UCSD) — `Clothing_Shoes_and_Jewelry` category, frozen by the organizer to a 50,000-product catalog (SHA256-verified).
- 200 labeled public development sessions provided by the organizer (Buying, Browsing, Intent Override, and Boundary scenarios).
- 800 private evaluation sessions held by the organizer for final scoring (not accessible to us).
- No manually labeled data of our own.

---
*Reminder: also need — a short method/limitations report, and a latency/token-usage/cost disclosure, per `docs/submission_rules.md`. Those can reuse content from this draft.*
