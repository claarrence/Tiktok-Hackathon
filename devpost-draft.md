# Devpost Written Project Description — DRAFT

Status: day-1 draft. Architecture section reflects the team's actual design; Tools/APIs/Libraries are placeholders until we decide during build. Update as we go rather than writing this fresh at the end.

## How our solution addresses the problem statement

Traditional keyword search can't tell "browsing for ideas" from "ready to buy," and it can't adapt mid-conversation as a shopper's intent sharpens or changes. Our agent is built around four pieces that map directly to that gap:

- **Dual-track intent routing** — every turn is classified as Buying (hard constraints, high-precision filter track) or Browsing (open-ended, diverse dense retrieval track), so the retrieval strategy matches the shopper's actual mode instead of one-size-fits-all search.
- **Hybrid retrieval → LLM ranking pipeline** — a multi-route retrieval stage (keyword + category + vector similarity) surfaces a wide, in-memory candidate pool, which an LLM semantic ranking stage then narrows to a precise Top-10.
- **Dynamic dialog state machine** — tracks two distinct conversational events: incremental slot accumulation ("also needs to be waterproof") versus intent override ("actually, forget shoes, show me bags"). When the candidate pool is still too broad (over-generality), the agent cuts off retrieval and asks one structured clarification question instead of returning a low-confidence guess.
- **Context & personalization engine** — distills dialog history into a running session state and profile (using the buyer's `purchase_frequency`, `rating_style`, `preference_tags`, `summary`) that continuously reshapes routing weights and ranking, so the agent's own guidance strategy adapts turn over turn rather than staying static.

This design is scored directly against the challenge's own metrics: retrieval breadth drives Hit Rate@10, LLM ranking precision drives MRR, and the clarification logic that avoids wasted turns drives MTTC — all within the hard 10-turn session cap.

**Local results on the 200 public dev sessions** (reproducible via `python3 -m evaluator.local_evaluator`):

| Metric | Weak BM25 baseline | Our agent | Change |
|---|---|---|---|
| Hit Rate@10 | 0.125 | 0.68 | +0.555 |
| MRR | 0.068 | 0.407 | +0.339 |
| MTTC | 9.81 | 5.53 | −4.28 turns |
| TechnicalScore | 0.107 | 0.571 | ~5.3x |

The single biggest lever turned out to be architectural rather than algorithmic: the baseline never asks a clarification question, so it can never unlock the additional product detail the (deterministic, rule-based) evaluator simulator only discloses in response to a targeted `ask_attribute`. Adding the dialog state machine's proactive clarification loop — on top of the same hybrid retrieval idea — is most of the lift. *(Update this table if further tuning changes the numbers before submission.)*

## Development tools used
- Python 3.10+ *(confirm team's actual version/IDE — e.g. VSCode, PyCharm, Jupyter)*
- Git / GitHub for version control
- *(add: any notebook/experiment tooling, if used)*

## APIs used
- **None, currently.** The "LLM Semantic Ranking" stage is implemented as a local, fully-offline scoring function (BM25 + category-overlap + Jaccard token-similarity + slot-match + profile-tag boost) rather than a hosted LLM call — this is explicitly allowed ("local scoring logic for the LLM ranking stage" is in-scope per the brief) and means the agent has zero network dependency and zero per-call cost. *Open decision for the team: swap the top-N reranking step for a real LLM API call (state which one here) if it beats the local scorer on the dev set — worth an A/B before committing, since it adds cost/latency/network-dependency.*

## Libraries and frameworks used
- **Python standard library only**: `sqlite3` (in-memory FTS5 full-text index for the keyword retrieval route), `re`, `json`, `dataclasses`. No external dependencies, no `requirements.txt` needed yet — deliberate, since it keeps setup to a single `python3` command and satisfies the "in-memory, no heavy vector DB" constraint for free. *Update this if a teammate adds something (e.g. an embeddings library) for a specific pillar.*

## Datasets and assets used
- Amazon Reviews 2023 (McAuley Lab, UCSD) — `Clothing_Shoes_and_Jewelry` category, frozen by the organizer to a 50,000-product catalog (SHA256-verified).
- 200 labeled public development sessions provided by the organizer (Buying, Browsing, Intent Override, and Boundary scenarios).
- 800 private evaluation sessions held by the organizer for final scoring (not accessible to us).
- No manually labeled data of our own.

---
*Reminder: also need — a short method/limitations report, and a latency/token-usage/cost disclosure, per `docs/submission_rules.md`. Those can reuse content from this draft.*
