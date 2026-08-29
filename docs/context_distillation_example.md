# Context Distillation — before / after

Pillar III (`context_engine/`) does not add new retrieval signals. It distills
the dialog history into a `ContextVector` each turn and lets that vector
**re-orchestrate the route weights** the ranker fuses with. This note walks one
public dev session where that re-orchestration alone turns a miss into a hit —
the disclosed constraints are byte-identical between the two runs.

Reproduce:

```bash
python3 -m evaluator.local_evaluator          # aggregate scores + results.json
python3 -m unittest tests.test_context_engine # pillar unit tests
```

---

## Session `public_0002` — intent override, "critical" buyer

Profile: `rating_style: critical`, `average_prior_rating: 1.0`,
`preference_tags: [fit, comfort, style]`. Hidden target: `B071X54486`
(a full-grain leather men's belt).

Simulated transcript (identical for both runs — the evaluator is deterministic):

| Turn | User discloses | Agent asks |
|---|---|---|
| 1 | "looking for Accessories Belts. Buckle closure" | material |
| 2 | "what matters is: leather; 100% Leather" | color |
| 3 | "Actually, ignore my earlier preference. What I need is: leather." | budget |
| 4–7 | "no additional preference for {budget,size,style,use_case}" | size … feature |
| 8 | "what matters is: Imported; Buckle closure" | — |

### Run A — pillar off (legacy intent-only weights)

| Turn | pool | precision_bias | w[slot/phrase/vector] | target rank |
|---|---|---|---|---|
| 1 | 250 | — | 0.14 / 0.16 / 0.24 | 119 |
| 2 | 246 | — | 0.14 / 0.16 / 0.24 | 32 |
| 3 | 248 | — | 0.33 / 0.16 / 0.05 | 30 |
| 4 | 255 | — | 0.14 / 0.15 / 0.32 | 60 |
| 5–7 | ~260 | — | 0.14 / 0.15 / 0.32 | 73 → 31 |
| 8 | 270 | — | 0.14 / 0.15 / 0.32 | 15 |
| 10 | 294 | — | 0.14 / 0.15 / 0.32 | **35 — MISS** |

The legacy table only reacts to the *current* turn's intent label. After the
override flips the turn to "buying" (turn 3) it briefly spikes `slot` to 0.33,
then the very next turn the intent classifier falls back to "browsing" (the reply
"I don't have a preference…" carries no buying cue) and the weights snap back to
vector-heavy. The target never gets a stable precision push and finishes 35th.

### Run B — context pillar on

| Turn | pool / trend | precision_bias | w[slot/phrase/vector] | target rank |
|---|---|---|---|---|
| 1 | 250 / unknown | 0.00 | 0.17 / 0.18 / 0.22 | 119 |
| 2 | 246 / expanding | 0.14 | 0.21 / 0.21 / 0.18 | 30 |
| 3 | 248 / stagnant | 0.26 | **0.41 / 0.23 / 0.00** | 16 |
| 4 | 255 / stagnant | 0.13 | 0.20 / 0.20 / 0.26 | 49 |
| 5–7 | ~260 / stagnant | 0.13 | 0.20 / 0.20 / 0.26 | 57 → 26 |
| 8 | 270 / stagnant | 0.16 | 0.21 / 0.20 / 0.25 | **10 — HIT (turn 8)** |

`precision_bias` is accumulated by the distiller from *disclosed information*, not
the turn counter: +per confirmed slot, +per disclosed phrase, **+0.12 for the
detected override event**, minus a penalty while the candidate pool refuses to
shrink. It peaks at turn 3 (right after the override) and again once turn 8 adds
two more constraint phrases — pulling `slot`/`phrase` up and `vector` to zero,
which is what finally lifts the leather belt into the top 10.

The turn 4–7 dip is real: with no new constraint disclosed and the pool
stagnating, the penalty pulls `precision_bias` back down and the target drifts to
~50th, exactly as intended — the agent stops over-committing to precision routes
that are not discriminating. When turn 8 discloses "Imported; Buckle closure" the
bias recovers and the ranker converges.

### Same story, buying scenario — `public_0179`

Legacy: target `B08JK818ZD` sits at ranks 12–24 for the whole session, ends
**35th, MISS**. Pillar: `precision_bias` from the turn-1 hard constraint plus the
turn-7 disclosures lifts it to **rank 7, HIT on turn 7**. One slot filled in both
runs — nothing else changed.

---

## Aggregate impact (200 public dev sessions, deterministic)

| Version | Hit Rate@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| Legacy intent-only weights | 0.715 | 0.4434 | 5.380 | 0.6029 |
| **+ context distillation & adaptive orchestration** | **0.730** | **0.4509** | **5.285** | **0.6146** |
| Δ | +0.015 | +0.0075 | −0.095 | **+0.0117** |

By scenario the lift concentrates where trajectory information is richest:
intent-override Hit Rate 0.867 → 0.900, MRR 0.697 → 0.735.

> Note: these numbers are only comparable because this change also removed a
> pre-existing non-determinism (set-iteration order leaking into score ties) that
> was swinging TechnicalScore by ~0.03 between `PYTHONHASHSEED` values. See the
> "Determinism note" in `project-plan.md`.
