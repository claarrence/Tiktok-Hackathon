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
| Hit Rate@10    | 0.125              | 0.940     | +0.815       |
| MRR            | 0.068              | 0.628     | +0.560       |
| MTTC           | 9.81               | 4.43      | −5.38 turns |
| TechnicalScore | 0.107              | 0.790     | ~7.4x        |

(These figures are now run-to-run stable: we removed a set-iteration-order dependency that was leaking `PYTHONHASHSEED` into score ties and swinging TechnicalScore by ~0.02 between runs of identical code.)

The single biggest lever turned out to be architectural rather than algorithmic: the baseline never asks a clarification question, so it can never unlock the additional product detail the (deterministic, rule-based) evaluator simulator only discloses in response to a targeted `ask_attribute`. Adding the dialog state machine's proactive clarification loop — on top of the same hybrid retrieval idea — was most of the initial lift.

A follow-up diagnosis of the remaining misses found ~86% were ranking failures, not recall failures — the correct product was almost always somewhere in the retrieval candidate pool, just not pushed into the top 10. Two ranking fixes closed a chunk of that gap: (1) disclosed budget amounts are now compared numerically against the catalog's actual `price` field instead of being (uselessly) token-matched against product text, and (2) disclosed constraint phrases — which are lifted near-verbatim from the target product's own listing text — now get an exact-substring match bonus in the ranker, which is a much stronger disambiguation signal than bag-of-words overlap for telling near-duplicate products apart.

The context-programming pillar targets the same ranking gap from a different angle. Rather than one static fusion of the retrieval routes, the distilled `ContextVector` drives a per-turn `precision_bias` that shifts weight toward exact slot/phrase matching in proportion to how much the shopper has actually disclosed — and detected intent-override turns give it an extra push, since the post-override constraint is the sharpest signal in the session. Isolated on the 200 dev sessions (against an otherwise identical agent with static weights), this lifts TechnicalScore 0.603 → 0.615, Hit Rate 0.715 → 0.730, and MRR 0.443 → 0.451, concentrated in the intent-override scenario (Hit Rate 0.867 → 0.900). A worked before/after walk-through is in `docs/context_distillation_example.md`.

The last big lever was replacing the vector-similarity route's scoring function. It originally used plain token-Jaccard overlap, which treats every shared word as equally meaningful — so a generic word like "comfortable" counted the same as a specific, rare detail that actually separates one product from ten near-duplicates in the same category. Swapping in TF-IDF cosine similarity (IDF computed once over the frozen catalog at index time, still pure Python stdlib — no embeddings dependency) let that route actually discriminate between similar products instead of just measuring rough topical overlap. Combined with the ranker's other precision signals (which were already tuned to reward exact matches), this was the single largest jump in the project: TechnicalScore 0.615 → 0.790, Hit Rate@10 0.730 → 0.940, MRR 0.451 → 0.628. *(Update these tables again if further tuning changes the numbers before submission.)*

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
