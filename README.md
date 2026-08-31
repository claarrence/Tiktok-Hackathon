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

**Current results on the 200 public dev sessions** (`python3 -m evaluator.local_evaluator`): Hit Rate@10 **0.950**, MRR **0.641**, MTTC **4.18**, TechnicalScore **0.804** — up from the weak BM25 baseline's 0.107 (`docs/baseline_results.json`). Full iteration history and the reasoning behind each change: `project-plan.md`.

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

## Limitations and Future Work

- **Buying scenario rank-1 precision (~43% of hits)** is the clearest remaining gap — well below Browsing (~58%) and Intent Override (~85%) despite Buying disclosing a constraint from turn 1. Tracing failing sessions found two causes: (1) near-duplicate catalog items sharing the same disclosed constraint (two alloy necklaces, two "Imported, 100% Polyester" rain jackets) landing in genuine near-ties, since global IDF doesn't capture rarity *within a specific pair* — addressed by a cluster-local phrase-rarity second pass in the ranker (see `project-plan.md`); and (2) about a third of Buying sessions hit the top-10 on turn 1, before any clarifying question is asked, off one generic disclosed word — the scoring rule locks in rank at the first turn a hit occurs, so there's no later turn for a smarter question to help. That second cause is a dialog-policy problem (which question to front-load, traded against MTTC), not a ranking one, and is left open rather than risking a late untested change.
- **No hosted LLM reranking stage.** The local scorer is competitive (TechnicalScore 0.804) and keeps cost/latency/network-dependency at zero; a prompted reranker over the top ~20-30 candidates was the original design for this stage and remains a natural next step if time/budget allow an A/B against the local scorer.
- **Boundary scenario has the smallest sample** in the public dev set (10/200 sessions), so its metrics are the noisiest of the four scenario breakdowns and shouldn't be over-tuned against.
- **Reciprocal Rank Fusion was tried and reverted** for the route-combination step (see `project-plan.md`) — tuned to a statistical tie with the simpler raw-score fusion while costing Boundary MRR. Worth knowing before re-attempting it from scratch.

## Team Contributions

Member A (Maegan) — Retrieval & Intent Routing (Pillar I)
  - Owned `retrieval/`: the dual-track Buying/Browsing intent classifier and the initial multi-route retrieval design (keyword, category, vector) that the other three pillars build candidate pools from.
  - *(Maegan/team: add specifics + before/after impact numbers here, mirroring Member B/C below — this entry is currently role-scope only, filled in from `project-plan.md`'s role plan rather than commit-level detail.)*

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
  - *(Jessica/team: add specifics + before/after impact numbers here, mirroring Member B/C above — this entry is currently role-scope only, filled in from `project-plan.md`'s role plan rather than commit-level detail.)*

Member E (Clarence) — Integration, Docs & Submission
  - Owned the shared `Agent` data contract from day 1 and ran the Sunday integration merge across all four pillar branches.
  - Diagnosed and fixed a `category_route` bug found from tracing an actual failing Buying session: a top-200 cutoff was silently dropping legitimately-tied candidates in arbitrary order on broad categories, scoring a correct product `category=0.0` purely by alphabetical bad luck. Fix: keep every candidate tied for the best overlap score regardless of count (`retrieval/engine.py`). Lifted TechnicalScore 0.7899 -> 0.8020.
  - Replaced the vector route's Jaccard similarity with TF-IDF cosine (`retrieval/engine.py`), and consolidated slot-match scoring onto the same catalog-wide IDF the vector route uses, so both routes agree on what counts as a rare, discriminating term.
  - Diagnosed why Buying's rank-1 rate (~43% of hits) lagged every other scenario despite disclosing the most information per session, then added a bounded cluster-local phrase-rarity second pass to the ranker that resolves genuine near-ties among near-duplicate candidates without ever overturning a lead earned outside that tie band (`ranking/ranker.py`). Verified against all 200 dev sessions with zero regressions: MRR 0.635 -> 0.641, TechnicalScore 0.8020 -> 0.8037. Also identified that the remainder of the Buying gap is a dialog-policy problem, not a ranking one, and scoped it out of this submission rather than risking a late untested change (see `project-plan.md`).
  - Tried and reverted Reciprocal Rank Fusion for route combination after grid-searching its weights to a statistical tie with the simpler raw-score approach, at a cost to Boundary-scenario MRR.
  - Repo hygiene, README, Devpost draft (including two documented corrections to earlier confounded benchmark claims — see `devpost-draft.md`), and final submission.