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

**Diagnosis (still true as of the last measurement):** of the ~55 remaining misses, roughly 86% are ranking failures, not recall failures — the correct product is almost always already in the retrieval candidate pool, the ranker (`ranking/ranker.py`) just doesn't push it into the top 10. Most of those misses aren't close either (many rank 51–200th, not 11–20th), meaning this needs real scoring improvements, not just weight nudges. Already fixed: (1) disclosed budget amounts are now compared numerically against the catalog's `price` field instead of being token-matched against product text, and (2) disclosed constraint phrases (which are lifted near-verbatim from the target product's own listing text, by construction of the evaluator's session simulator) now get an exact-substring match bonus in the ranker.

**Candidate fixes not yet implemented, roughly in order of effort:**

1. **TF-IDF cosine similarity instead of Jaccard** for the vector-similarity route in `retrieval/engine.py`. Jaccard treats every token as equally important, so generic words ("clothing," "comfortable") count the same as the words that actually distinguish one product from ten similar ones. Still stdlib-only, no embeddings needed.
2. **IDF-weight the slot-match score** in `ranking/ranker.py` the same way — a matched "cotton" should count for less than a matched rare, specific detail, since the latter is far more decisive for telling near-duplicate products apart.
3. **Reciprocal Rank Fusion (RRF)** instead of averaging raw route scores when combining keyword/category/vector signals. BM25, Jaccard/TF-IDF, and category-overlap live on differently-shaped scales; averaging raw scores across them can distort the combination. RRF combines each route's *rank position* instead, which is simpler and more robust for hybrid retrieval.
4. **Search the ranking weights instead of hand-picking them** (`context_engine/profile.py`'s `BASE_WEIGHTS`). We have 200 labeled dev sessions — a small grid or random search optimizing directly for MRR would likely beat the current hand-tuned values. Costs eval runtime (~40s per full run), so worth trimming the search space.
5. **Real LLM reranking on the top ~20–30 candidates** — the originally-envisioned Pillar IV design. Only worth it once the cheaper fixes plateau, since it adds latency, cost, and a network dependency that needs an offline fallback documented per `docs/submission_rules.md`.

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
