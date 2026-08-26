# Shopping Copilot: AI Conversational Search and Recommendations

**Technical Workshop Webinar + Q&A:** Fri 28 Aug, 4:00–4:45pm — [join the webinar](#) *(paste the actual link here)*

## 4.1 Background

Traditional e-commerce search relies on static keyword matching and fails to capture genuine shifts in consumer psychology or distinguish open-ended browsing from high-intent buying. In conversational commerce, an intelligent agent that uses dynamic context programming is critical for bridging ambiguous user queries and complex product catalogs — and it directly moves core industry metrics.

## 4.2 Problem Statement

Architect a next-generation shopping agent that goes beyond rigid search filters, demonstrating deep cognitive understanding, runtime architectural agility, and commercial efficiency on the provided Amazon dataset. The system rests on four pillars:

### I. Core Architecture: Intent Routing & Hybrid Pipeline
- **Dual-Track Routing** — detect user intent instantly: a high-precision filter track for "Buying" (lock hard constraints), and a diverse dense retrieval track for "Browsing" (cross-category scenario matching).
- **Pipeline Base** — in-memory data stream: *Multi-Route Retrieval → LLM Semantic Ranking* (keyword + category + vector similarity combined).

### II. Dialog Strategy: Multi-Turn Scenario Evolution
- **Dynamic State Machine** — track Information Accumulation (incremental slots) and Intent Override (slot erasure/rewriting) gracefully.
- **Proactive Guidance** — on Over-Generality (candidate pool overload), cut off retrieval and generate structured clarification prompts to converge the user.

### III. Self-Evolution: Dynamic Context Programming
- **Runtime Adaptation** — use dialog history for Personalized Context Distillation: update short-term session state and long-term user profile continuously.
- **Adaptive Orchestration** — use dynamic Context Programming to re-orchestrate the workflow and align strategy at runtime, refining its own guidance logic.

### IV. Evaluation Matrix: Product & Efficiency Metrics
Anchored on the final purchased record:
- **Coverage** — Hit Rate@K (catalog recall/boundary capability at retrieval)
- **Precision** — MRR / Top-K Hit Rate (does the LLM push the purchased item to #1)
- **Efficiency** — MTTC, Mean Turns to Conversion (heavily rewards fewer turns to the right product)

## 4.3 Constraints & Scope

| Category | Details |
|---|---|
| **In scope** | Sensitive intent-detection (Buying vs Browsing); heterogeneous retrieval routing (weights, dynamic truncation, slot decay); runtime-adaptive memory for personalized context distillation; prompt/local-scoring tuning for LLM ranking |
| **Out of scope** | UI/UX (backend/headless only); training or full fine-tuning of base LLMs; heavy external vector DB clusters (must be in-memory); multi-modal processing (text only) |
| **Limits** | Hard cap of **10 turns/session** (zero score if exceeded); catalog is **strictly read-only**, no mock ASIN injection |
| **Assumptions** | Inputs are pre-cleaned text (no typo/ASR correction needed); catalog/pricing/categories are static; each session is single-user, isolated |

## 4.4 Available Resources & Data

**Competition data**
- Frozen catalog: 50,000 products, Amazon Reviews 2023 → `Clothing_Shoes_and_Jewelry`
- 200 labeled public dev sessions (local testing)
- 800 private sessions (final evaluation only) — separate users/targets from public set

**Participant resources**
- Weak BM25 starter Agent (Python)
- Deterministic local evaluator: Hit Rate@10, MRR, MTTC, Efficiency, combined TechnicalScore
- Python Agent interface + machine-readable API contract
- Eval config, reproducible baseline results, data docs, submission rules
- SHA256 checksum for the catalog

You may fully replace the starter Agent as long as you keep using the official evaluator. Keyword, rule-based, dense, hybrid, reranking, local models, or external APIs are all fair game. **No hosted model access or API keys are provided** — a paid LLM is not required, and no secrets should ever be committed to the repo.

**Links**
- Participant repo: https://github.com/TechJam2026/techjam-conversational-search
- Participant kit release: https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit
- Original data source: https://amazon-reviews-2023.github.io/

## 4.5 Deliverables

1. **Written Project Description** (Devpost) — problem-solution fit, dev tools used, APIs used, libraries/frameworks used, datasets/assets used.
2. **Public GitHub Repository** — well-structured/commented code covering every component, plus a README with: overview, setup/install, steps to reproduce results, limitations + what you'd improve with more time, team contributions.
3. **Demo Video** — end-to-end working demo, uploaded to YouTube (public), linked in Devpost, no unlicensed third-party trademarks/content. If there's no front end, a walkthrough of API usage / inference examples / result analysis is accepted.

## 4.6 Judging Criteria

| Criteria | Definition | Weight |
|---|---|---|
| Technical Execution | Engineering fundamentals, architecture, reliability of demo | 35% |
| Innovation & Problem Insight | Originality, sharpness of problem framing | 20% |
| Impact & Relevance | Real value to real users beyond the hackathon prompt | 20% |
| Feasibility & Practicality | Realistic, buildable, resource-proportionate | 15% |
| Presentation & Communication | Final-event pitch clarity and Q&A depth | 10% |

**Deadline: Tue 1 Sep, 12:00pm**
