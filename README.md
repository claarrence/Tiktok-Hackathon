# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Our Solution

Four pipeline stages, wired together per turn in `starter/agent.py`:

- **Intent routing** (`retrieval/intent_router.py`) — classifies each turn Buying (hard-constraint, high-precision) or Browsing (open-ended, diverse retrieval), so the retrieval strategy matches the shopper's actual mode.
- **Multi-route retrieval** (`retrieval/engine.py`) — keyword (in-memory SQLite FTS5), category (inverted index over the catalog's own category paths), and vector (TF-IDF cosine similarity) routes union their candidates into one pool.
- **Dialog state machine** (`dialog_state/state_machine.py`) — accumulates disclosed slots turn over turn, handles intent override (rewriting a slot in place rather than appending to it), and decides each turn whether the candidate pool is confident enough to answer or whether to ask one targeted clarifying question.
- **Context & personalization engine** (`context_engine/`) — distills the whole session so far (disclosed constraints, override events, buyer profile) into a `ContextVector` each turn, which re-weights the retrieval routes toward precision as more gets disclosed.
- **Ranking** (`ranking/ranker.py`) — fuses the three retrieval routes with slot-match, phrase-match, price-fit, and rating signals into one score, standing in for an LLM reranker without a model API call (see "Model Choice and Cost" below). A bounded second pass re-resolves near-ties among the leading candidates using rarity computed within just that cluster rather than the whole catalog.

**Current results on the 200 public dev sessions** (`python3 -m evaluator.local_evaluator`): Hit Rate@10 **0.950**, MRR **0.641**, MTTC **4.18**, TechnicalScore **0.804** — up from the weak BM25 baseline's 0.107 (`docs/baseline_results.json`). Full iteration history and the reasoning behind each change: see "Development History and Diagnosis" below.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is recommended. The starter uses only the Python standard library.

```bash
python3 -m evaluator.local_evaluator
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is an objective input to the `Technical Execution` assessment. It is not a separate judging criterion and does not represent the entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

**Our choice: no hosted LLM.** The ranking stage is a local, fully-offline scoring function (see "Our Solution" above), which the brief allows in place of a hosted call. This means: **$0 model cost**, **0 prompt/completion tokens** per turn (reported honestly via `usage` in every `respond()` call), and **no network dependency** at inference time — the agent runs unmodified under the organizer's official-scoring network restrictions, since it has no online mode to fall back from. Measured latency on the 200-session dev set: ~3.6s one-time catalog load/index build (paid once per process start), then ~38ms average per `respond()` call. Full reasoning for not swapping to a hosted reranker: `devpost-draft.md`, "APIs used".

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.

## Development History and Diagnosis

An earlier version of this agent existed as a reference implementation before tuning; treat the numbers below as the record of how it got from there to the current state, not just a final score.

**Local eval history on the 200 public dev sessions** (`python3 -m evaluator.local_evaluator`):

| Version | Hit Rate@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| Weak BM25 baseline | 0.125 | 0.068 | 9.81 | 0.107 |
| First working pipeline (all 4 pillars, no tuning) | 0.68 | 0.407 | 5.53 | 0.571 |
| + numeric budget matching + exact-phrase bonus | 0.725 | 0.438 | 5.14 | 0.611 |
| + IDF-weighted slot match, hard price filter, context-engine precision_bias (grid-searched, `context_engine/params.py`) | 0.730 | 0.451 | 5.29 | 0.615 |
| + dialog-strategy fixes (shallow-disclosure follow-ups, question-budget exhaustion, override/broaden-reset) — largest single contributor | 0.940 | 0.628 | 4.43 | 0.7875 |
| + TF-IDF cosine replacing Jaccard on the vector route (isolated effect, holding the row above constant) | 0.940 | 0.628 | 4.43 | 0.7899 |
| + category_route tie-preservation fix (below) | 0.950 | 0.635 | 4.18 | 0.8020 |
| + cluster-local phrase-rarity tie-break second pass (below) | **0.950** | **0.641** | **4.18** | **0.8037** |

**Correction:** an earlier internal draft of this table attributed a "0.615 → 0.790" jump to the TF-IDF change alone. That was confounded — it compared against a stale baseline that predated the dialog-strategy fixes landing in parallel on a different branch. Measured properly in isolation (holding the already-improved dialog-strategy code fixed, changing only the vector route): TechnicalScore 0.7875 → 0.7899, a small, real gain (+0.003), not the largest lever. The dialog-strategy fixes are the actual largest single contributor — roughly 0.615 → 0.7875 — though that comparison isn't as cleanly isolated as the vector-route A/B since other small changes landed in the same span.

**Diagnosis history:** the original finding was that most misses were ranking failures (target already in the candidate pool, just not surfaced in the top 10), not recall failures — this motivated most of the fixes below.

**`category_route` bugfix (`retrieval/engine.py`):** with a broad, widely-shared category path (e.g. "Wallets" — 1,636 products, "Card Cases & Money Organizers Wallets" — 482 full matches), the old `overlap.most_common(200)` hard-cut silently dropped some full-overlap ties in arbitrary (insertion/ASIN) order — a genuinely correct candidate could land at `category_score=0.0` purely by alphabetical bad luck while an unrelated tie-mate got full credit. Fixed by always keeping every candidate tied for the *best* overlap score, however many there are; the limit now only bounds the lower, partial-match tiers. Diagnosed from a real failing Buying session (`public_0017`). Lifted TechnicalScore 0.7899 → 0.8020, but barely moved Buying's rank-1 rate specifically, which pointed at a separate, still-open problem.

**Cluster-local phrase-rarity tie-break (`ranking/ranker.py`, `_disambiguate_ties`):** traced 44 Buying near-misses (target found somewhere in the top 10, but not rank 1) and found the fused score was landing in genuine near-ties — median gap 0.038 out of ~1.0, max 0.15 — between the target and a near-duplicate catalog item sharing the same disclosed constraint (two alloy necklaces, two "Imported, 100% Polyester" rain jackets). Catalog-wide IDF doesn't distinguish a phrase's rarity *within that specific tie cluster*, even when it's globally rare. Added a bounded second pass: candidates within a small margin (`TIE_BAND = 0.08`) of the leading score get their disclosed phrases re-weighted by document frequency computed only within that cluster, nudging the score by at most half that margin — enough to settle a genuine tie, never enough to overturn a lead earned outside the band. Phrase-substring only, not bag-of-words over single tokens: an earlier version scored individual tokens and let a template artifact (the literal word "color" from a disclosed "color: green") coincidentally match an unrelated product's "Color" spec key and flip a correct rank-1 — a cluster this small (single digits to ~20 candidates) can't survive that kind of noise. Verified against all 200 dev sessions: 0 regressions, MRR 0.635 → 0.641, TechnicalScore 0.8020 → 0.8037. Notably, **Buying's rank-1 rate itself didn't move** (still 33/77, 42.9%) — Browsing picked up the gain instead (55.7% → 58.2%).

**Current diagnosis:** recall is essentially solved — only 10/200 sessions miss entirely (Hit Rate@10 0.950). The remaining lever is **rank-1 precision** (drives MRR, 30% of TechnicalScore): only ~53% of hits land at rank 1 overall, and Buying specifically sits at ~43% despite disclosing a constraint from turn 1 — worse than Browsing (~58%) and much worse than Intent Override (~85%). Two ranking-fusion fixes have now landed and both barely moved Buying's rank-1 rate specifically, which narrows the diagnosis: about a third of Buying's 80 sessions hit the top-10 on turn 1, before any clarifying question is even asked, off a single disclosed constraint that's almost always a generic material word (alloy, polyester, rayon...) shared by every near-duplicate in the cluster. The challenge's own scoring rule locks in a session's rank at the *first* turn the target enters the top 10 (`docs/evaluation_config.json`), so there's no later turn for a smarter question to help — no amount of ranking-side re-weighting can recover information that was never disclosed. Reaching that subset needs a dialog-policy change (e.g. front-loading a more discriminating question), not another ranking weight tweak — a real precision/efficiency trade to make deliberately, not stumble into. Left open rather than risking a late, untested change this close to the deadline.

**Tried and deliberately not kept — Reciprocal Rank Fusion (RRF):** replacing the raw-score route combination with rank-position fusion. Implemented, then its weights were grid-searched (~63 trials, `context_engine/profile.py`'s `BASE_WEIGHTS` + a new `RRF_K` constant) since the existing hand/grid-tuned weights weren't calibrated for RRF's different score distribution. Best invariant-respecting result: TechnicalScore 0.789, a statistical tie with the raw-score baseline — but it traded MRR down for a slightly better hit rate, and Boundary-scenario MRR got meaningfully worse. Reverted: not worth the added `RRF_K` tuning surface for a tie.

**Remaining candidate, not implemented:** real LLM reranking on the top ~20-30 candidates — the originally-envisioned ranking-stage design. Only worth it once the cheaper fixes plateau, since it adds latency, cost, and a network dependency that needs an offline fallback documented per `docs/submission_rules.md`. With TechnicalScore already at 0.80, the marginal value here is shrinking — not worth the added complexity/risk this close to the deadline.

## Limitations and Future Work

- **Buying scenario rank-1 precision (~43% of hits)** is the clearest remaining gap. See "Current diagnosis" above — it's a dialog-policy problem (which question to front-load, traded against MTTC), not a ranking one, and is left open rather than risking a late untested change.
- **No hosted LLM reranking stage.** The local scorer is competitive (TechnicalScore 0.804) and keeps cost/latency/network-dependency at zero; a prompted reranker over the top ~20-30 candidates was the original design for this stage and remains a natural next step if time/budget allow an A/B against the local scorer.
- **Boundary scenario has the smallest sample** in the public dev set (10/200 sessions), so its metrics are the noisiest of the four scenario breakdowns and shouldn't be over-tuned against.
- **Reciprocal Rank Fusion was tried and reverted** for the route-combination step (see above) — tuned to a statistical tie with the simpler raw-score fusion while costing Boundary MRR. Worth knowing before re-attempting it from scratch.

## Team Contributions

Member A (Maegan) — Retrieval & Intent Routing (Pillar I)
  - Owned `retrieval/`: the dual-track Buying/Browsing intent classifier and the initial multi-route retrieval design (keyword, category, vector) that the other three pillars build candidate pools from.
  - *(Maegan/team: add specifics + before/after impact numbers here, mirroring Member B/C below — this entry is currently role-scope only.)*

Member B (Caro) — Dialog Strategy (Pillar II)
  - Built the `dialog_state/` module — the per-session memory that turns a run of
    separate messages into one evolving picture of what the shopper wants.
  - Slot accumulation: stated preferences pile up across turns (colour on turn 2,
    material on turn 4) instead of each message being read in isolation.
  - Intent override: when the shopper changes their mind ("actually, make it
    white"), the affected preference is rewritten in place, not left sitting next
    to the old one; the override turn is recorded for the ranker.
  - Clarifying questions: when the candidate pool is still too broad, picks the
    most useful attribute the shopper hasn't answered yet, and never re-asks
    something already covered or a dead-end field.
  - Fixed the phrase classifier so open-ended requirements ("buckle closure",
    "moisture-wicking") land in a real slot instead of being dropped — the ranker
    was being starved of that signal.
  - Confirmed "brand" is not an actual catalog field (checked all 50,000 rows)
    and kept it out of the question queue so no turn is wasted on it.
  - Made the "search wider" flag switch back off once results reconverge, rather
    than staying on for the rest of the session and dragging out convergence.
  - Stopped "I don't have a preference" replies from leaking filler words into the
    search query.
  - Added a unit-test suite covering the accumulation, override, question-timing,
    and broaden-flag paths.
  - Impact on the 200 local dev sessions: TechnicalScore 0.6146 -> 0.6583,
    Hit Rate@10 0.730 -> 0.795, MRR 0.451 -> 0.489.

Member C (Yi Ting) — Context & Personalization (Pillar III)
  - Built the `context_engine/` module.
  - Condenses the conversation so far into one summary each turn, and re-tunes how much the agent trusts each search method based on it.
  - The more the shopper commits to, the harder the agent leans on exact matches.
  - Feeds the shopper's profile (past-review style, stated preferences) into the ranking.
  - Wired it into the main agent; fixed a bug that made evaluator scores vary run to run.

Member D (Jessica) — Ranking & Evaluation (Pillar IV)
  - Owned `ranking/`: the local semantic-ranking function that fuses the retrieval routes into a final score, and ran the evaluator throughout to track Hit Rate@10/MRR/MTTC across iterations.
  - *(Jessica/team: add specifics + before/after impact numbers here, mirroring Member B/C above — this entry is currently role-scope only.)*

Member E (Clarence) — Integration, Docs & Submission
  - Owned the shared `Agent` data contract from day 1 and ran the Sunday integration merge across all four pillar branches.
  - Diagnosed and fixed a `category_route` bug found from tracing an actual failing Buying session: a top-200 cutoff was silently dropping legitimately-tied candidates in arbitrary order on broad categories, scoring a correct product `category=0.0` purely by alphabetical bad luck. Fix: keep every candidate tied for the best overlap score regardless of count (`retrieval/engine.py`). Lifted TechnicalScore 0.7899 -> 0.8020.
  - Replaced the vector route's Jaccard similarity with TF-IDF cosine (`retrieval/engine.py`), and consolidated slot-match scoring onto the same catalog-wide IDF the vector route uses, so both routes agree on what counts as a rare, discriminating term.
  - Diagnosed why Buying's rank-1 rate (~43% of hits) lagged every other scenario despite disclosing the most information per session, then added a bounded cluster-local phrase-rarity second pass to the ranker that resolves genuine near-ties among near-duplicate candidates without ever overturning a lead earned outside that tie band (`ranking/ranker.py`). Verified against all 200 dev sessions with zero regressions: MRR 0.635 -> 0.641, TechnicalScore 0.8020 -> 0.8037. Also identified that the remainder of the Buying gap is a dialog-policy problem, not a ranking one, and scoped it out of this submission rather than risking a late untested change (see "Development History and Diagnosis" above).
  - Tried and reverted Reciprocal Rank Fusion for route combination after grid-searching its weights to a statistical tie with the simpler raw-score approach, at a cost to Boundary-scenario MRR.
  - Repo hygiene, README, Devpost draft (including two documented corrections to earlier confounded benchmark claims — see `devpost-draft.md`), and final submission.