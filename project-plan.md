# Project Plan — Shopping Copilot (TechJam)

Deadline: **Tue 1 Sep, 12:00pm**. Starting point: Thu 27 Aug. Team of 5. Full brief in [challenge-brief.md](challenge-brief.md).

## Strategy

Judging is 35% Technical Execution — the pipeline actually working end-to-end and scoring well beats a fancier idea that's half-wired. Sequence:

1. Get the **weak BM25 baseline running through the official local evaluator on day 1** — this is your safety net score and confirms the API contract/harness before anyone writes custom logic.
2. Build the 4 pillars **in parallel on separate branches** against a shared Agent interface, so nobody blocks anyone.
3. Reserve **all of Sunday for integration** — parallel work always takes longer to merge than expected.
4. Freeze features **Monday evening** and spend the last ~18 hours purely on eval-driven tuning, docs, and the video — not new architecture.

## Timeline

### Thu 27 Aug — Kickoff & Baseline

- Whole team: read the brief together, clone `techjam-conversational-search`, pull the participant kit, verify the catalog SHA256.
- Run the starter BM25 Agent through the local evaluator on the 200 dev sessions — record baseline Hit Rate@10 / MRR / MTTC.
- Walk the Python Agent interface + API contract together so everyone codes to the same shape from hour one.
- Assign the 5 roles (below) and create one branch per pillar off `main`.
- Agree on the Agent's internal data contract (session state shape, slot schema, retrieval candidate object) — this is the one thing that *must* be decided as a group before splitting up, since all 4 pillars read/write it.

### Fri 28 Aug — Parallel Build, Day 1

- **4:00–4:45pm: attend the Technical Workshop webinar** — treat this as a hard checkpoint, bring questions about ranking/eval scoring specifics.
- Each pillar owner builds their module in isolation against mocked inputs/outputs matching the shared contract.
- End of day: each module should run standalone on a hand-written test case (not yet wired together).

### Sat 29 Aug — Parallel Build, Day 2

- Continue building out each pillar; start writing real unit tests per module.
- First rough end-to-end wiring attempt in the evening (even if broken) so integration issues surface with a full day of buffer left.

### Sun 30 Aug — Integration Day

- All hands: merge the 4 branches into one working Agent.
- Run the full pipeline against all 200 dev sessions through the local evaluator.
- Triage failures by pillar (retrieval miss vs. ranking miss vs. dialog-state bug vs. context bug) and assign fixes to the owning member.

### Mon 31 Aug — Tuning & Polish

- Morning/afternoon: error analysis on dev sessions with low Hit Rate/MRR or high MTTC; tune retrieval weights, clarification triggers, context distillation logic.
- **Evening: feature freeze.** No new logic after this point — only bug fixes.
- Start README, Devpost write-up draft, and demo video script/recording in parallel with tuning.

### Tue 1 Sep — Submission Day (due 12:00pm)

- **09:00 — hard code freeze.** Only submission logistics from here.
- 09:00–10:30: finish demo video edit, upload to YouTube as **public**, grab the link.
- 09:00–11:00: finalize README (reproduce steps, limitations/future work, contributions) and Devpost description.
- 11:00–11:45: final read-through of the public repo (no secrets/API keys committed, no mock ASINs, code runs clean from a fresh clone).
- 11:45: submit on Devpost. Do not wait until 12:00 — GitHub/YouTube propagation and Devpost form hiccups eat time.

## Role & Deliverable Plan (5 members)

Each role owns one pillar end-to-end (code + its own tests) so ownership is unambiguous during integration.

### Member A — Retrieval & Intent Routing Lead *(Pillar I)*

- Dual-track intent classifier: Buying vs. Browsing.
- Multi-route retrieval: keyword, category, vector similarity, with combination weights.
- The pipeline's ingestion/retrieval stage that feeds the ranking stage.
- **Deliverable:** `retrieval/` module + intent router, standalone tests, a short design note on how routing weights were chosen (feeds into Devpost write-up).

### Member B — Dialog Strategy Lead *(Pillar II)*

- Dynamic state tracker: incremental slot accumulation + intent-override (slot erasure/rewrite).
- Over-generality detection → retrieval cutoff → structured clarification prompt generation.
- **Deliverable:** `dialog_state/` module, a state-machine diagram (for the README/demo), test cases covering both accumulation and override paths.

### Member C — Context & Personalization Lead *(Pillar III)*

- Personalized Context Distillation: short-term session state + long-term user profile updates from dialog history.
- Adaptive orchestration: runtime re-orchestration/context programming that lets the agent adjust its own guidance logic mid-session.
- **Deliverable:** `context_engine/` module, before/after example showing context distillation changing a recommendation, notes for the "innovation" section of the Devpost write-up (this pillar is where most of the 20% Innovation score will come from).

### Member D — Ranking & Evaluation Lead *(Pillar IV + LLM ranking stage)*

- LLM semantic ranking stage (prompt design or local scoring logic to push the purchased item to #1).
- Owns running the official local evaluator, tracking Hit Rate@10 / MRR / MTTC over time, and error analysis during Sunday/Monday.
- **Deliverable:** `ranking/` module, an evaluation log/spreadsheet of scores per iteration (baseline → final), the "results" section content for the README and Devpost.

### Member E — Integration, Docs & Submission Lead *(clarence — unavailable Sat 29 Aug)*

- Owns the shared Agent interface/data contract from day 1 and does the Sunday integration merge.
- Repo hygiene: structure, comments, no committed secrets, clean install from scratch.
- README (overview, setup, reproduce steps, limitations/future work, contributions), Devpost description, demo video recording/editing/upload, final Devpost submission.
- **Deliverable:** working `main` branch, complete README, Devpost draft, published YouTube video, submitted entry.

## Current State & Ranking Precision Backlog

A working end-to-end agent already exists on `main` (`retrieval/`, `dialog_state/`, `context_engine/`, `ranking/`, wired together in `starter/agent.py`) as a reference implementation — treat it as a starting point to improve, not a finished system. Whoever picks up Pillar I (retrieval) or Pillar IV (ranking) should read this section before touching those files, so the context carries over even if you're starting a fresh Claude Code session.

**Local eval history on the 200 public dev sessions** (`python3 -m evaluator.local_evaluator`):

| Version | Hit Rate@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| Weak BM25 baseline | 0.125 | 0.068 | 9.81 | 0.107 |
| First working pipeline (all 4 pillars, no tuning) | 0.68 | 0.407 | 5.53 | 0.571 |
| + numeric budget matching + exact-phrase bonus | 0.725 | 0.438 | 5.14 | 0.611 |
| + IDF-weighted slot match, hard price filter, context-engine precision_bias (grid-searched, `context_engine/params.py`) | 0.730 | 0.451 | 5.29 | 0.615 |
| + dialog-strategy fixes (shallow-disclosure follow-ups, question-budget exhaustion, override/broaden-reset) — largest single contributor, measured on `main` before the two rows below | 0.940 | 0.628 | 4.43 | 0.7875 |
| + TF-IDF cosine replacing Jaccard on the vector route (isolated effect, holding the row above constant) | 0.940 | 0.628 | 4.43 | 0.7899 |
| + category_route tie-preservation fix (below) | 0.950 | 0.635 | 4.18 | 0.8020 |
| + cluster-local phrase-rarity tie-break second pass (below) | **0.950** | **0.641** | **4.18** | **0.8037** |

**Correction:** an earlier version of this table attributed a "0.615 → 0.790" jump to TF-IDF alone. That was confounded — it compared against a stale baseline that predated the dialog-strategy fixes landing in parallel on a different branch. See `devpost-draft.md` for the corrected, isolated numbers (the dialog-strategy row above).

**Diagnosis history:** the original finding — most misses were ranking failures (target already in the candidate pool, just not surfaced in top 10), not recall failures — held for a while and motivated most of the fixes below. Of the candidate fixes originally listed here, **1, 2, and 4 are now done**: TF-IDF cosine similarity replaced Jaccard on the vector route (`retrieval/engine.py`), slot-match scoring is IDF-weighted (`ranking/ranker.py`), and the context-engine's route weights were grid-searched against the dev set rather than hand-picked (`context_engine/params.py` — see its docstring for the search methodology). Also since done, not on the original list: a hard price filter for buying-intent sessions, numeric budget/exact-phrase matching, and a `category_route` bugfix (see below).

**`category_route` bugfix (retrieval/engine.py):** with a broad, widely-shared category path (e.g. "Wallets" — 1636 products, "Card Cases & Money Organizers Wallets" — 482 full matches), the old `overlap.most_common(200)` hard-cut silently dropped some full-overlap ties in arbitrary (insertion/ASIN) order — a genuinely correct candidate could land at `category_score=0.0` purely by alphabetical bad luck while an unrelated tie-mate got full credit. Fixed by always keeping every candidate tied for the *best* overlap score, however many there are; `limit` now only bounds the lower, partial-match tiers. Generalizes to any category width, no new heuristic needed. Diagnosed from a real failing Buying session (`public_0017`) — worth rereading if anyone hits a similar "this should obviously match but doesn't" case elsewhere in the pipeline; the pattern (a hard top-N cut silently dropping legitimate ties) could recur wherever candidates get truncated.

**cluster-local phrase-rarity tie-break (ranking/ranker.py, `_disambiguate_ties`):** traced 44 Buying near-misses (hit somewhere in top 10, but not rank 1) and found the fused score was landing in genuine near-ties — median gap 0.038, max 0.15 — between the target and a near-duplicate catalog item sharing the same disclosed constraint (two alloy necklaces, two "Imported, 100% Polyester" rain jackets). Global IDF doesn't distinguish a phrase's rarity *within that specific tie cluster*. Added a bounded second pass: candidates within `TIE_BAND` (0.08) of the leading score get their disclosed phrases re-weighted by local (cluster-only) document frequency, nudging the score by at most `TIE_BAND/2` so it can only settle a genuine tie, never overturn a lead earned outside the band. Phrase-substring only, not bag-of-words over single tokens — an earlier version scored individual tokens and let a template artifact (the literal word "color" from a disclosed "color: green") coincidentally match an unrelated product's "Color" spec key and flip a correct rank-1; a cluster this small (single digits to ~20 candidates) can't survive that kind of noise. Result: 0 regressions, MRR 0.635 → 0.641, TechnicalScore 0.8020 → 0.8037 — but **Buying's rank-1 rate itself didn't move (still 33/77, 42.9%)**. Browsing picked up the gain instead (55.7% → 58.2%). See below for why.

**Current diagnosis, as of this measurement:** recall is essentially solved — only 10/200 sessions miss entirely (Hit Rate@10 0.950), and nearly all of those still have the target somewhere in the candidate pool. The remaining lever is **rank-1 precision** (drives MRR, 30% of TechnicalScore): only ~53% of hits land at rank 1 overall, and Buying specifically sits at ~43% despite being the scenario with the most disclosed information per session — worse than Browsing (58%) and much worse than Intent Override (85%). Two ranking-fusion fixes (category-route, phrase-rarity tie-break) have now landed and both barely moved Buying's rank-1 rate specifically, which narrows the diagnosis: 30 of Buying's 80 sessions hit top-10 on turn 1, before any clarifying question is even asked, off a single disclosed constraint that's almost always a generic material word (alloy, polyester, rayon...) shared by every near-duplicate in the cluster — see `docs/evaluation_config.json`'s scoring formula, which locks in `target_rank` at the *first* turn the target enters top-10. There's no extra text signal in those transcripts for a ranking-level fix to find; the session ends before more could be disclosed. Reaching that subset likely needs a dialog-policy change (e.g. front-loading a more discriminating question) rather than another ranking weight tweak — a real precision/efficiency trade to make deliberately, not stumble into.

**Tried and deliberately not kept:**

- **Reciprocal Rank Fusion (RRF)**, replacing the raw-score route combination with rank-position fusion. Implemented, then its weights were grid-searched (~63 trials, `context_engine/profile.py`'s `BASE_WEIGHTS` + a new `RRF_K` constant) since the existing hand/grid-tuned weights weren't calibrated for RRF's different score distribution. Best invariant-respecting result: **TechnicalScore 0.789**, a statistical tie with the raw-score baseline — but it traded MRR down for a slightly better hit rate, and Boundary-scenario MRR got meaningfully worse. Reverted: not worth the added `RRF_K` tuning surface for a tie. (One real bug surfaced along the way and is now fixed regardless of the RRF decision: the tuned weights briefly floored the keyword-route weight to exactly 0 under buying intent before `precision_bias` even applied, which would have made `test_precision_bias_moves_weight_from_recall_to_precision` vacuous — worth remembering if anyone revisits `BASE_WEIGHTS["keyword"]`, it needs to stay above ~0.20 for that invariant to mean anything under buying intent given the current legacy delta.)

**Remaining candidate, not yet implemented:**

1. **Real LLM reranking on the top ~20–30 candidates** — the originally-envisioned Pillar IV design. Only worth it once the cheaper fixes plateau, since it adds latency, cost, and a network dependency that needs an offline fallback documented per `docs/submission_rules.md`. With TechnicalScore already at 0.80, the marginal value here is shrinking — probably not worth the added complexity/risk this close to the deadline unless there's clear time to spare.

After implementing a fix, re-run the recall-vs-ranking diagnostic (ask whoever owns integration if you need the script) to confirm misses are actually shrinking and not just moving around.

## Shared Checklist Before Submitting

- [ ] Agent runs end-to-end on a fresh clone with no manual setup steps missing from the README
- [ ] Local evaluator run recorded (Hit Rate@10, MRR, MTTC, TechnicalScore) and included in README/Devpost
- [ ] No session exceeds 10 turns
- [ ] No API keys/secrets committed
- [ ] No catalog mutation or mock ASIN injection
- [ ] GitHub repo is public
- [ ] YouTube video is public and linked in Devpost
- [ ] Team contributions listed in README
