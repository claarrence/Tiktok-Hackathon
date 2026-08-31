# Devpost Written Project Description — DRAFT

Status: day-1 draft. Architecture section reflects the team's actual design; Tools/APIs/Libraries are placeholders until we decide during build. Update as we go rather than writing this fresh at the end.

## How our solution addresses the problem statement

Traditional keyword search can't tell "browsing for ideas" from "ready to buy," and it can't adapt mid-conversation as a shopper's intent sharpens or changes. Our agent is built around four pieces that map directly to that gap:

- **Dual-track intent routing** — every turn is classified as Buying (hard constraints, high-precision filter track) or Browsing (open-ended, diverse dense retrieval track), so the retrieval strategy matches the shopper's actual mode instead of one-size-fits-all search.
- **Hybrid retrieval → LLM ranking pipeline** — a multi-route retrieval stage (keyword + category + vector similarity) surfaces a wide, in-memory candidate pool, which an LLM semantic ranking stage then narrows to a precise Top-10.
- **Dynamic dialog state machine** — tracks two distinct conversational events: incremental slot accumulation ("also needs to be waterproof") versus intent override ("actually, forget shoes, show me bags"). When the candidate pool is still too broad (over-generality), the agent cuts off retrieval and asks one structured clarification question instead of returning a low-confidence guess.
- **Context & personalization engine** — every turn, the whole dialog so far is distilled into a single `ContextVector` (constraints confirmed, phrases disclosed, intent-override events, candidate-pool trajectory, plus a `demandingness` read from the buyer's `rating_style` / `average_prior_rating`). An orchestration step then *re-programs that turn's pipeline* from the vector: it computes a `precision_bias` — earned by disclosed information, not by turn count — that continuously redistributes the retrieval-route weights from recall routes toward exact slot/phrase matching, and backs off again whenever the pool stops converging. The agent's guidance strategy is rewritten turn over turn rather than following a fixed pipeline.

This design is scored directly against the challenge's own metrics: retrieval breadth drives Hit Rate@10, LLM ranking precision drives MRR, and the clarification logic that avoids wasted turns drives MTTC — all within the hard 10-turn session cap.

**Local results on the 200 public dev sessions** (reproducible via `python3 -m evaluator.local_evaluator`):

| Metric         | Weak BM25 baseline | Our agent | Change       |
| -------------- | ------------------ | --------- | ------------ |
| Hit Rate@10    | 0.125              | 0.950     | +0.825       |
| MRR            | 0.068              | 0.635     | +0.567       |
| MTTC           | 9.81               | 4.18      | −5.63 turns |
| TechnicalScore | 0.107              | 0.802     | ~7.5x        |

(These figures are now run-to-run stable: we removed a set-iteration-order dependency that was leaking `PYTHONHASHSEED` into score ties and swinging TechnicalScore by ~0.02 between runs of identical code.)

The single biggest lever turned out to be architectural rather than algorithmic: the baseline never asks a clarification question, so it can never unlock the additional product detail the (deterministic, rule-based) evaluator simulator only discloses in response to a targeted `ask_attribute`. Adding the dialog state machine's proactive clarification loop — on top of the same hybrid retrieval idea — was most of the initial lift.

A follow-up diagnosis of the remaining misses found ~86% were ranking failures, not recall failures — the correct product was almost always somewhere in the retrieval candidate pool, just not pushed into the top 10. Two ranking fixes closed a chunk of that gap: (1) disclosed budget amounts are now compared numerically against the catalog's actual `price` field instead of being (uselessly) token-matched against product text, and (2) disclosed constraint phrases — which are lifted near-verbatim from the target product's own listing text — now get an exact-substring match bonus in the ranker, which is a much stronger disambiguation signal than bag-of-words overlap for telling near-duplicate products apart.

The context-programming pillar targets the same ranking gap from a different angle. Rather than one static fusion of the retrieval routes, the distilled `ContextVector` drives a per-turn `precision_bias` that shifts weight toward exact slot/phrase matching in proportion to how much the shopper has actually disclosed — and detected intent-override turns give it an extra push, since the post-override constraint is the sharpest signal in the session. Isolated on the 200 dev sessions (against an otherwise identical agent with static weights), this lifts TechnicalScore 0.603 → 0.615, Hit Rate 0.715 → 0.730, and MRR 0.443 → 0.451, concentrated in the intent-override scenario (Hit Rate 0.867 → 0.900). A worked before/after walk-through is in `docs/context_distillation_example.md`.

**Correction:** an earlier version of this draft attributed a "0.615 → 0.790, single largest jump in the project" swing to replacing the vector route's scoring function (Jaccard → TF-IDF cosine). That number was confounded — it compared against a stale 0.615 baseline that predated a separate, larger round of dialog-strategy fixes (shallow-disclosure follow-ups, question-budget exhaustion handling, slot override/broaden-reset behavior) landing in parallel on the same branch. Measured properly in isolation, holding the (already-improved) dialog-strategy code fixed and changing only the vector route: **TechnicalScore 0.7875 → 0.7899, a small, real gain (+0.003), not a dramatic one.** The dialog-strategy fixes are the actual largest single contributor to the project's score — roughly 0.615 → 0.7875 across those commits, though that comparison isn't as cleanly isolated as the vector-route A/B since other small changes landed in the same span.

TF-IDF is still worth keeping over Jaccard: it down-weights generic catalog-wide words ("clothing," "comfortable") relative to the specific, rare details that actually separate near-duplicate products, which is the theoretically correct behavior even where the 200-session dev set doesn't show a large effect from it. (Reciprocal Rank Fusion for the same route-combination step was also tried and measured — see `project-plan.md` — and reverted after tuning landed it at a tie with the simpler raw-score approach while costing MRR.)

A real, generalizable bug surfaced from tracing an actual failing session rather than guessing: `category_route`'s old top-200 cutoff could silently drop legitimately-matching products off the candidate list whenever more than 200 products tied for the best category-overlap score — common for broad categories (482 products fully matched "Card Cases & Money Organizers Wallets" in one session), since the tie-break fell out of arbitrary insertion order rather than any meaningful signal. The target in that session landed at alphabetical position 395 among the ties and was excluded entirely, scoring `category=0.0` despite a perfect category match, while an unrelated product at position 20 got full credit purely by luck. Fixed by guaranteeing every candidate tied for the best score is kept regardless of count, and only truncating the lower, partial-match tiers — this generalizes to any category width without a new heuristic or magic constant. This lifted TechnicalScore 0.7899 → 0.8020 (Hit Rate 0.940 → 0.950, MRR 0.628 → 0.635). Notably, it mostly helped hit rate and MRR broadly rather than fixing rank-1 precision on the specific scenario that motivated the investigation (Buying, still ~43% rank-1 among its hits) — that remains an open problem, not something this fix closed.

## Development tools used

- Python 3.10+ *(confirm team's actual version/IDE — e.g. VSCode, PyCharm, Jupyter)*
- Git / GitHub for version control
- *(add: any notebook/experiment tooling, if used)*

## APIs used

- **None, currently.** The "LLM Semantic Ranking" stage is implemented as a local, fully-offline scoring function (BM25 + category-overlap + TF-IDF cosine similarity + slot-match + profile-tag boost) rather than a hosted LLM call — this is explicitly allowed ("local scoring logic for the LLM ranking stage" is in-scope per the brief) and means the agent has zero network dependency and zero per-call cost. *Open decision for the team: swap the top-N reranking step for a real LLM API call (state which one here) if it beats the local scorer on the dev set — worth an A/B before committing, since it adds cost/latency/network-dependency.*

## Libraries and frameworks used

- **Python standard library only**: `sqlite3` (in-memory FTS5 full-text index for the keyword retrieval route), `re`, `json`, `dataclasses`. No external dependencies, no `requirements.txt` needed yet — deliberate, since it keeps setup to a single `python3` command and satisfies the "in-memory, no heavy vector DB" constraint for free. *Update this if a teammate adds something (e.g. an embeddings library) for a specific pillar.*

## Datasets and assets used

- Amazon Reviews 2023 (McAuley Lab, UCSD) — `Clothing_Shoes_and_Jewelry` category, frozen by the organizer to a 50,000-product catalog (SHA256-verified).
- 200 labeled public development sessions provided by the organizer (Buying, Browsing, Intent Override, and Boundary scenarios).
- 800 private evaluation sessions held by the organizer for final scoring (not accessible to us).
- No manually labeled data of our own.

---

*Reminder: also need — a short method/limitations report, and a latency/token-usage/cost disclosure, per `docs/submission_rules.md`. Those can reuse content from this draft.*
