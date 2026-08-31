# Devpost Written Project Description

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
| MRR            | 0.068              | 0.641     | +0.573       |
| MTTC           | 9.81               | 4.18      | −5.63 turns |
| TechnicalScore | 0.107              | 0.804     | ~7.5x        |

(These figures are now run-to-run stable: we removed a set-iteration-order dependency that was leaking `PYTHONHASHSEED` into score ties and swinging TechnicalScore by ~0.02 between runs of identical code.)

The single biggest lever turned out to be architectural rather than algorithmic: the baseline never asks a clarification question, so it can never unlock the additional product detail the (deterministic, rule-based) evaluator simulator only discloses in response to a targeted `ask_attribute`. Adding the dialog state machine's proactive clarification loop — on top of the same hybrid retrieval idea — was most of the initial lift.

A follow-up diagnosis of the remaining misses found ~86% were ranking failures, not recall failures — the correct product was almost always somewhere in the retrieval candidate pool, just not pushed into the top 10. Two ranking fixes closed a chunk of that gap: (1) disclosed budget amounts are now compared numerically against the catalog's actual `price` field instead of being (uselessly) token-matched against product text, and (2) disclosed constraint phrases — which are lifted near-verbatim from the target product's own listing text — now get an exact-substring match bonus in the ranker, which is a much stronger disambiguation signal than bag-of-words overlap for telling near-duplicate products apart.

The context-programming pillar targets the same ranking gap from a different angle. Rather than one static fusion of the retrieval routes, the distilled `ContextVector` drives a per-turn `precision_bias` that shifts weight toward exact slot/phrase matching in proportion to how much the shopper has actually disclosed — and detected intent-override turns give it an extra push, since the post-override constraint is the sharpest signal in the session. Isolated on the 200 dev sessions (against an otherwise identical agent with static weights), this lifts TechnicalScore 0.603 → 0.615, Hit Rate 0.715 → 0.730, and MRR 0.443 → 0.451, concentrated in the intent-override scenario (Hit Rate 0.867 → 0.900). A worked before/after walk-through is in `docs/context_distillation_example.md`.

**Correction:** an earlier version of this draft attributed a "0.615 → 0.790, single largest jump in the project" swing to replacing the vector route's scoring function (Jaccard → TF-IDF cosine). That number was confounded — it compared against a stale 0.615 baseline that predated a separate, larger round of dialog-strategy fixes (shallow-disclosure follow-ups, question-budget exhaustion handling, slot override/broaden-reset behavior) landing in parallel on the same branch. Measured properly in isolation, holding the (already-improved) dialog-strategy code fixed and changing only the vector route: **TechnicalScore 0.7875 → 0.7899, a small, real gain (+0.003), not a dramatic one.** The dialog-strategy fixes are the actual largest single contributor to the project's score — roughly 0.615 → 0.7875 across those commits, though that comparison isn't as cleanly isolated as the vector-route A/B since other small changes landed in the same span.

TF-IDF is still worth keeping over Jaccard: it down-weights generic catalog-wide words ("clothing," "comfortable") relative to the specific, rare details that actually separate near-duplicate products, which is the theoretically correct behavior even where the 200-session dev set doesn't show a large effect from it. (Reciprocal Rank Fusion for the same route-combination step was also tried and measured — see `project-plan.md` — and reverted after tuning landed it at a tie with the simpler raw-score approach while costing MRR.)

A real, generalizable bug surfaced from tracing an actual failing session rather than guessing: `category_route`'s old top-200 cutoff could silently drop legitimately-matching products off the candidate list whenever more than 200 products tied for the best category-overlap score — common for broad categories (482 products fully matched "Card Cases & Money Organizers Wallets" in one session), since the tie-break fell out of arbitrary insertion order rather than any meaningful signal. The target in that session landed at alphabetical position 395 among the ties and was excluded entirely, scoring `category=0.0` despite a perfect category match, while an unrelated product at position 20 got full credit purely by luck. Fixed by guaranteeing every candidate tied for the best score is kept regardless of count, and only truncating the lower, partial-match tiers — this generalizes to any category width without a new heuristic or magic constant. This lifted TechnicalScore 0.7899 → 0.8020 (Hit Rate 0.940 → 0.950, MRR 0.628 → 0.635). Notably, it mostly helped hit rate and MRR broadly rather than fixing rank-1 precision on the specific scenario that motivated the investigation (Buying, still ~43% rank-1 among its hits).

That Buying gap turned out to have two distinct causes, found by tracing 44 actual near-miss transcripts rather than tuning weights blind. First: the fused ranking score was landing in genuine near-ties (median gap 0.038 out of ~1.0) between the target and a near-duplicate catalog item sharing the same disclosed constraint — two alloy necklaces, two "Imported, 100% Polyester" rain jackets — because catalog-wide IDF doesn't capture that a term is common *within that specific pair*, even when it's globally rare. We added a bounded second ranking pass: candidates within a small margin of the leading score get their disclosed phrases re-weighted by document frequency computed only within that tie cluster, nudged by no more than half that margin so it can settle a genuine tie but never overturn a lead earned elsewhere. That closed part of the gap safely (0 regressions across all 200 sessions, MRR 0.635 → 0.641) but left Buying's rank-1 rate itself unmoved (33/77 either way) — which pointed at the second, deeper cause: the challenge's own scoring rule locks in a session's rank at the *first* turn the target enters the top 10, and roughly a third of Buying sessions do that on turn 1, before any clarifying question is even asked, off a single generic disclosed word ("polyester," "alloy") that's shared by every near-duplicate in the catalog. No amount of ranking-side re-weighting can recover information that was never disclosed — that subset is a dialog-policy question (what to ask first), not a ranking one, and we're treating it as an open, explicitly-scoped-out item rather than a late risky change (see `project-plan.md`).

## Development tools used

- Python 3.10+ (no external dependencies — runs from a fresh clone with the standard `python3` interpreter, no notebook or IDE-specific tooling required)
- Git / GitHub for version control

## APIs used

- **None.** The "LLM Semantic Ranking" stage is implemented as a local, fully-offline scoring function (BM25-style keyword route + category-overlap + TF-IDF cosine similarity + IDF-weighted slot/phrase matching + profile-tag boost), which the challenge brief explicitly allows in place of a hosted LLM call. We deliberately did not swap this for a hosted LLM API: the local scorer already reaches TechnicalScore 0.804 on the dev set, adding an API call would introduce cost, latency, and a network dependency the organizer may disable for official scoring (per `docs/submission_rules.md`'s Model Policy), and there wasn't safe time this close to the deadline to A/B a swap without risking a regression we couldn't fully re-verify.

## Libraries and frameworks used

- **Python standard library only**: `sqlite3` (in-memory FTS5 full-text index for the keyword retrieval route), `re`, `json`, `dataclasses`, `math`, `statistics`. No external dependencies and no `requirements.txt` — deliberate, since it keeps setup to a single `python3` command and satisfies the "in-memory, no heavy vector DB" constraint for free.

## Datasets and assets used

- Amazon Reviews 2023 (McAuley Lab, UCSD) — `Clothing_Shoes_and_Jewelry` category, frozen by the organizer to a 50,000-product catalog (SHA256-verified).
- 200 labeled public development sessions provided by the organizer (Buying, Browsing, Intent Override, and Boundary scenarios).
- 800 private evaluation sessions held by the organizer for final scoring (not accessible to us).
- No manually labeled data of our own.

## Method, model choice & limitations

**Method:** four pipeline stages, wired together per turn in `starter/agent.py`. (1) Intent routing classifies each turn Buying vs. Browsing. (2) Multi-route retrieval (keyword FTS5, category inverted-index, TF-IDF cosine vector) unions candidates from all three routes into one pool. (3) A context-distillation stage folds the session so far (disclosed constraints, override events, buyer profile) into weights that re-orchestrate the next two stages. (4) A local ranking function fuses the three retrieval routes with slot-match, phrase-match, price, and rating signals into a final score, including a bounded second pass that re-resolves near-ties using rarity computed within just the leading cluster of candidates rather than the whole catalog. A dialog-state machine decides, each turn, whether to ask a clarifying question (and which one) or return recommendations, based on candidate-pool size, score confidence margin, and a per-question budget.

**Model choice:** no hosted LLM — see "APIs used" above. Ranking is a deterministic, fully-local scoring function rather than a learned or prompted model, chosen for zero cost, zero network dependency, and full reproducibility from a frozen catalog.

**Limitations / what we'd improve with more time:**
- **Buying scenario rank-1 precision (~43%)** is the clearest remaining gap — see the diagnosis above. About a third of Buying sessions resolve on turn 1 off a single generic disclosed word, before the dialog policy gets a chance to ask a more discriminating question; fixing this needs a deliberate dialog-policy change (which question to front-load, at what cost to MTTC), not another ranking tweak.
- **No real LLM reranking stage.** The brief allows a local scorer in place of one, and ours is competitive, but a prompted reranker over the top ~20–30 candidates was the original Pillar IV design and was deprioritized once the local scorer's marginal returns made the added latency/cost/network-dependency risk hard to justify this close to the deadline (see `project-plan.md`, "Remaining candidate, not yet implemented").
- **Boundary scenario has the smallest sample (10/200 sessions)** in the public dev set, so its MRR (0.53) is the noisiest of the four scenario breakdowns — a single session flips it several points. Worth more dev examples before trusting it as a tuning target.
- **Reciprocal Rank Fusion was tried and reverted** (see `project-plan.md`) after tuning landed it at a statistical tie with the simpler raw-score route fusion while costing Boundary-scenario MRR — noted here so it isn't re-attempted from scratch without that context.

## Latency, token usage & cost disclosure

- **Model cost: $0.** No hosted API is called; there is no per-token or per-request cost, and none of the code paths require network access at inference time (see "APIs used").
- **Token usage: 0 prompt / 0 completion tokens** per turn, reported honestly via the `usage` field in every `respond()` call (see `starter/agent.py`) — there's no LLM call to meter.
- **Latency**, measured locally on the 200-session public dev set (`evaluator/local_evaluator.py`, single-threaded, no GPU):
  - One-time catalog load + index build (SQLite FTS5 + TF-IDF over 50,000 products): **~3.6s**, paid once per process start, not per session or per turn.
  - Average `respond()` call: **~38ms** (826 total turns across 200 sessions, 31.1s wall time).
  - Average full session (up to 10 turns, ends early on a hit): **~155ms**.
- **Offline fallback:** the entire agent *is* the offline fallback — it has no online mode to fall back from. It runs unmodified under network restrictions, per the Model Policy in `docs/submission_rules.md`.
