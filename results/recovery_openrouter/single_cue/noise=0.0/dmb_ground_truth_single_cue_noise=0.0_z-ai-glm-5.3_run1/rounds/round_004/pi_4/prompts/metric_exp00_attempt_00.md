# metric_exp00_attempt_00

## System Prompt

You are a psychology researcher proposing a metric in the Decision Making (Binary Features) domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=10):** [0.99, 0.98, 0.95, 0.9, 0.85, 0.75, 0.65, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[0, 0, 1, 1, 0, 1, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  trial 2: A=[0, 1, 0, 1, 0, 1, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  trial 3: A=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
  trial 4: A=[0, 0, 1, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  trial 5: A=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  trial 6: A=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
  trial 7: A=[1, 0, 0, 1, 0, 1, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  trial 8: A=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 1, 0, 1, 0, 0, 1, 1]
  trial 9: A=[1, 0, 0, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  trial 10: A=[0, 0, 1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
  trial 11: A=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
  trial 12: A=[0, 0, 1, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
  trial 13: A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
  trial 14: A=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 1]
  trial 15: A=[0, 0, 1, 1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 1, 1]
  trial 16: A=[1, 1, 0, 1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
  trial 17: A=[1, 1, 0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
  trial 18: A=[0, 1, 0, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
  trial 19: A=[1, 0, 1, 1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 1]
  trial 20: A=[1, 0, 1, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 1, 0, 1, 0, 0, 1]
  trial 21: A=[0, 0, 1, 0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0, 1, 1, 1]
  trial 22: A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  trial 23: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 24: A=[1, 0, 1, 0, 1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

**Rationale:** CORE LOGIC. Both theories share the identical architecture (sign votes x validity weights x low-gain softmax x lapse) and differ ONLY in the weight function, and every validity in this design is >= 0.55, so BSST's low flank is inert and the entire dissociation is carried by two structural features never tested together: (a) BSST's FLAT PLATEAU over v in [0.55, 0.95] (all these weights equal w_plateau to within 5e-5) vs FDEI's compressed-but-strictly-MONOTONE gradient (w(0.55)=0.77 < w(0.95)=1.18 at center; the gradient is what delta > 0 exists to produce); (b) BSST's sharp 0.98/0.99 distrust cliff (w(0.98) in [0.47, 0.81], w(0.99) in [-1.35, -0.85]) vs FDEI's delta-flattened near-equality of 0.98 and 0.99 (w99 - w98 in [0.020, 0.044] across its ENTIRE box, so FDEI is locked within 0.8 probability points of invariance to a 0.98-vs-0.99 swap on any trial). Validities (10 experts): [0.99, 0.98, 0.95, 0.90, 0.85, 0.75, 0.65, 0.55, 0.55, 0.55]. 24 unique pairs x K=4 = 96 trials. Differentiation from prior experiments: no v=0.5 chance-flank logic ([4], [8]); no tally-margin decorrelation ([1]-[3]); no exchange-rate ladders against a trusted 0.90 reference ([7]); unlike [8], the cliff is tested MASKED inside balanced mid-band coalitions (where FDEI's near-invariance is parameter-free) rather than in isolated 0.98-vs-0.99 pairs, and the reciprocal PLATEAU-swap equality (invisible to BSST, visible to FDEI) is entirely new -- giving a double dissociation of SENSITIVITY ORDERING.

FAMILY V (plateau validity psychometric -- delta estimation). V1/V2/V3: A wins {0.95, v} with v = 0.65/0.75/0.85, B wins two 0.55s; V4: A wins {0.95, 0.55}, B wins two 0.55s. BSST must predict EXACTLY p(A)=0.50 on all four (every discriminator sits on the plateau, counts are 2-vs-2 tied, uniform softmax + uniform lapse) -- a parameter-free flatness knife. FDEI predicts a strictly monotone ladder in v (its w is strictly increasing for any delta > 0): centers .564 (V4), .588 (V1), .601 (V2), .612 (V3); box ranges e.g. V1 [.553,.626], V3 [.567,.662], every rung strictly above .50 across the whole box. The ladder slope is a direct delta estimate: log-odds = 2*beta*(llr(v)-tau)^delta regressed on log(llr-tau). Flat 0.50 across the ladder falsifies FDEI's delta in [0.1,0.2] outright; any single rung above ~.55 falsifies BSST with zero free parameters. Mirrors V1m/V3m control side bias.

FAMILY R (reference-validity crossover ladder -- sign dissociation and reference-ordering reversal). A wins one reference cue r in {0.95, 0.98, 0.99}, B wins k in {1,2} cues of 0.55. R1 (r=.95, k=1): BSST exactly 0.50 (plateau tie); FDEI .564 [.538,.594]. R2 (r=.98, k=1): FDEI .573 [.542,.607] vs BSST .457 [.420,.488] -- SIGN-DISJOINT around 0.50: FDEI says one 98% expert beats one 55% expert, BSST says it loses. R5 (r=.99, k=1): FDEI .577 [.545,.615] vs BSST .204 [.192,.265 with lapse] -- a 30+ point sign flip. R3/R4 (k=2): FDEI [.399,.472]/[.405,.482] vs BSST [.342,.359]/[.274,.348] -- disjoint. R6 (r=.99, k=2): FDEI [.408,.488] vs BSST [.096,.152] -- disjoint. Crossover structure: FDEI crosses 0.5 between k=1 and k=2 for EVERY reference; BSST never exceeds 0.50 for ANY reference at ANY k (its 0.98/0.99 weights are below the plateau). REFERENCE-ORDERING REVERSAL (quantitative knife both ways): R2 vs R1 and R4 vs R3 compare 0.98 vs 0.95 references at fixed k. FDEI predicts R2 > R1 and R4 > R3 STRICTLY for every parameter (llr(0.98) > llr(0.95) always), but by at most ~1 point (w98-w95 <= 0.079). BSST predicts R2 < R1 and R4 < R3 STRICTLY for every parameter (w98 <= 0.81 < plateau) -- by 2-8 points at k=1 and 4-9 points at k=2. The SIGN of the 0.95-vs-0.98 comparison separates the entire boxes of both theories. Mirrors R2m/R4m/R5m balance sides.

FAMILY M/P (masked cliff -- the sensitivity-ordering double dissociation, flagship). All cells share the balanced mid-band background A/B = {0.75, 0.55} vs {0.90, 0.85, 0.65} and embed one super cue in A's coalition. M98 (A adds 0.98): FDEI .482 (box [.475,.487] -- tightly pinned); BSST .457. M99 (A adds 0.99): FDEI .486, box [.482,.489]; BSST .204 [.192,.265]. CLIFF SWAP (M98 -> M99, a 1-point validity change): FDEI predicts |delta p| <= 0.8 points for EVERY (delta, tau, beta, epsilon) in its box (w99-w98 <= 0.044, 2*beta*that <= 0.033 logits) -- a parameter-free near-equality; BSST predicts a drop of 16-31 points. PLATEAU SWAPS: P95 (A adds 0.99+0.95+0.55... i.e., coalition {0.99, 0.95, 0.55}) and P55 (coalition {0.99, 0.55, 0.55}) differ by swapping a 0.95 for a 0.55 -- a 40-point validity change. BSST must predict P95 = P55 = M99 EXACTLY (all reduce to margin w99 - w_plateau; plateau swaps are invisible) -- parameter-free triple equality. FDEI predicts P95 (.512, box [.504,.524]) > M99 (.486) > P55 (.448, box [.429,.466]), a 4-9 point swing per swap, with P95 and P55 on OPPOSITE sides of 0.50... P55 box entirely below .466 vs BSST entirely below .265 (disjoint), P95 box entirely above .504 vs BSST entirely below .265 (disjoint, sign flip). The headline quantitative signature is the 2x2 of swap sensitivity: cliff swap (0.98->0.99): FDEI ~0 pts, BSST -16 to -31 pts; plateau swap (0.95->0.55): FDEI -4 to -9 pts, BSST exactly 0 pts. Each theory is parameter-free-insensitive to exactly the manipulation the other finds huge. Mirrors M99m/P55m balance sides.

HONEST KNIVES AGAINST MY OWN THEORY (FDEI). (i) R2: FDEI's box ceiling is .607 -- if subjects follow a lone 98% expert over a 55% expert at p >= 0.61, my flattening claim (delta <= 0.2 caps the 98:55 exchange rate at 1.81) is falsified outright. (ii) The V ladder: flat 0.50 kills delta > 0. (iii) M99 vs M98: any reliable cliff-sized drop inside balanced coalitions falsifies w99-w98 <= 0.044. (iv) G1 anchor (A wins 0.95/0.90/0.85/0.75/0.65 vs three 0.55s): FDEI box [.811,.918] core, BSST [.761,.786] -- p(G1) <= 0.77 strains FDEI, p(G1) >= 0.80 exceeds BSST's ceiling. CONTROLS: X1 (one 0.55 won by each side, everything else tied): BOTH theories must predict exactly 0.50 for every parameter -- a side/display-bias assay. R1 vs V4 is a surface-invariance check both theories demand (identical evidence w95-w55, different surface: 1-vs-1 vs 2-vs-2 discriminators) -- inequality would indicate count-based heuristics outside both models. Tie patterns mix 0/0 and 1/1 so 'count the 1s' is unviable (e.g., M98 shows A with five 1s vs B with three 1s while the theories nearly tie; G1m shows B with six 1s vs A with three). No all-tie trials; every trial has 2-7 discriminators among 10 experts requiring genuine scanning. Sides balanced (10 A-tilted / 9 B-tilted / 5 near-neutral under FDEI centers; every non-anchor family contains an A/B mirror). PREDICTED PATTERN IF FDEI IS RIGHT (centers): R1 .564, R2 .573, R3 .441, R4 .450, R5 .577, R6 .454, V4<V1<V2<V3 = .564/.588/.601/.612, M98 .482 = M99 .486, P95 .512 > P55 .448, G1 .872, X1 .500 -- a rising plateau psychometric, near-zero cliff sensitivity, several-points plateau sensitivity, crossovers between k=1 and k=2. IF BSST IS RIGHT: R1/V1-V4 exactly .500, R2 .457, R3 .351, R4 .313, R5 .204, R6 .122, M98 .457 >> M99 = P95 = P55 = .204 exactly, G1 .774, X1 .500 -- flat plateau psychometric, zero plateau sensitivity, massive cliff sensitivity, never following a 0.98/0.99 expert. The boxes are disjoint or parameter-free-separated on 16 of 24 cells, and the two sensitivity orderings are mutually exclusive.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Flattened-Diagnosticity Evidence Integration (the extreme-compression corner of the Bayesian cue-counting family): people integrate ALL binary expert ratings into one evidence score, where each discriminating cue casts a sign vote weighted by w_j = max(log(v_j/(1-v_j)) - tau, 0)^delta — chance cues (v=0.5) are annihilated exactly, but among genuinely diagnostic cues the subjective weight is a HEAVILY flattened (delta ~ 0.15) version of the normative log-odds, and choice is a low-gain softmax (2*beta ~ 0.6) plus a small uniform lapse. The new theoretical claim is that the compression exponent delta is itself the structural margin-compression device: shrinking delta equalizes weights, which GROWS count-dominated conflict margins (where cues split across options) while SHRINKING alignment-dominated cumulative margins (where several strong cues pile onto one side). This differential action — not any global gain knob, and not trial-level normalization — is what lets a single small beta simultaneously produce moderate TTB-leaning conflict-trial behavior (Exp 1), a slightly positive tally-vs-top-cue regression slope (Exp 2), graded ~0.62 tally-following (Exp 3), and near-chance |d|=1 agreement driven by the cancellation between super-expert matches and super-expert reversals (Exp 4).

**Parameters:**
- delta: [0.1, 0.2]
- tau: [0.0, 0.04]
- beta: [0.27, 0.37]
- epsilon: [0.0, 0.05]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Diagnosticity-Weighted Evidence Integration (Bayesian cue-counting).
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings (0/1).
    # Each feature j casts a sign vote sign(a_j - b_j); the vote is
    # weighted by the cue's log-likelihood-ratio diagnosticity
    # llr_j = log(v_j / (1 - v_j)), rectified by an attention floor
    # tau and compressed by an exponent delta:
    #   w_j = (max(llr_j - tau, 0))^delta
    # A v = 0.5 expert has llr = 0 -> exactly zero evidence.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    delta = float(parameters["delta"])
    tau = float(parameters["tau"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    a, b = stim[0], stim[1]

    # Feature-wise votes: +1 where A wins the feature, -1 where B
    # wins, 0 on ties (a tied cue is uninformative for the pair).
    votes = np.sign(a - b)

    # Normative diagnosticity: log-likelihood-ratio of each expert.
    # Clip away from v = 1.0 to keep the log finite; v = 0.5 maps to 0.
    v = np.clip(val, 0.5, 1.0 - 1e-9)
    llr = np.log(v / (1.0 - v))

    # Limited-attention floor: diagnosticity below tau is discarded;
    # the remainder is compressed by the exponent delta. delta = 1 is
    # the purely normative Bayesian weight; delta < 1 flattens the
    # diagnosticity gradient toward tallying over informative cues
    # (while v = 0.5 cues stay annihilated at any delta, since
    # llr(0.5) = 0 <= tau); delta > 1 sharpens it toward
    # lexicographic/TTB-like use of the best experts.
    w = np.power(np.maximum(llr - tau, 0.0), delta)

    # Weighted evidence score. B's score is the mirror of A's because
    # every vote is zero-sum across the two options.
    s_a = float(np.dot(w, votes))
    scores = np.array([s_a, -s_a])

    # Numerically stable low-temperature softmax. When all attended
    # cues tie (s_a == 0) the softmax is exactly uniform, which is the
    # correct behavior for undiscriminating evidence.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Independent lapse: with probability epsilon output a uniform
    # pick over the two options.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Broad-Spectrum Skepticism Tallying (BSST), low-gain/held-cliff calibration. People integrate ALL binary expert ratings as sign votes (+1/-1/0) into a zero-sum evidence score, where the subjective weight w(v) is a wide inverted-U over STATED validity with skepticism on both flanks: (i) chance experts (v = 0.5) carry a small negative weight (~0.23 trusted-cue units) — a coin-flipper's endorsement is mildly anti-diagnostic; (ii) the entire band v ~ [0.55, 0.95] is a flat trust plateau (~0.9), making mid-band experts nearly exchangeable and generating moderate unweighted-tally behavior; (iii) distrust of claimed certainty is CONCENTRATED in a very sharp cliff between 0.98 and 0.99: w(0.98) stays mildly positive while w(0.99) plunges to ~ -1.1 trusted-cue units. The calibration claim added this round is about the GAIN-FLANK DECOUPLING: the softmax gain beta can be lowered (0.34) while the chance-flank magnitude kappa_lo is raised (0.23) so that the product beta*kappa_lo (~0.077) — the quantity that pins the chance-cue signature — is held invariant, but the product beta*|w(0.99)| (~0.37) is allowed to FALL because kappa_hi is held at its iter-2 value rather than deepened. This is the one combination never tried in nine iterations: every prior beta reduction (iters 5, 9) was bundled with a kappa_hi increase that exactly cancelled the gain on the 0.99-driven cells.

**Parameters:**
- v_rise: [0.515, 0.54]
- w_plateau: [0.88, 0.93]
- v_c: [0.9855, 0.9875]
- s_hi: [0.0025, 0.0033]
- kappa_lo: [0.22, 0.235]
- kappa_hi: [1.68, 1.72]
- beta: [0.33, 0.35]
- epsilon: [0.05, 0.07]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Broad-Spectrum Skepticism Tallying (BSST), sharp-late-cliff edition.
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Each feature j casts a sign vote sign(a_j - b_j); the vote is
    # weighted by a wide inverted-U 'skepticism' function of the
    # expert's STATED validity v:
    #   v <= v_rise :  w = -kappa_lo * (v_rise - v)/(v_rise - 0.5)
    #                  (chance experts: linear lobe, w(0.5) = -kappa_lo;
    #                   a coin-flipper's endorsement is mildly
    #                   anti-diagnostic noise)
    #   v >  v_rise :  w = w_plateau - (w_plateau + kappa_hi)
    #                        * sigmoid((v - v_c)/s_hi)
    #                  (flat trust plateau over the mid-band -- the
    #                   logistic is ~0 for every v <= 0.95, so all
    #                   mid-band experts get the full plateau weight;
    #                   then an extremely SHARP distrust cliff packed
    #                   between 0.98 and 0.99: w(0.98) stays mildly
    #                   positive, w(0.99) plunges to ~ -1.1 trusted-cue
    #                   units, saturating at -kappa_hi as v -> 1)
    # Votes are summed into a zero-sum evidence score; choice is a
    # low-gain softmax with inverse temperature beta plus a uniform
    # lapse epsilon.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    v_rise = float(parameters["v_rise"])
    w_plateau = float(parameters["w_plateau"])
    v_c = float(parameters["v_c"])
    s_hi = float(parameters["s_hi"])
    kappa_lo = float(parameters["kappa_lo"])
    kappa_hi = float(parameters["kappa_hi"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    a, b = stim[0], stim[1]

    # Feature-wise votes: +1 where A wins the feature, -1 where B
    # wins, 0 on ties (a tied cue is uninformative for the pair).
    votes = np.sign(a - b)

    v = np.clip(val, 0.5, 1.0)
    w = np.empty(n_features, dtype=float)

    # (i) Low flank: chance experts carry a small negative weight,
    # linearly reaching -kappa_lo at v = 0.5 and 0 at v_rise.
    lo = v < v_rise
    denom_lo = max(v_rise - 0.5, 1e-6)
    w[lo] = -kappa_lo * np.clip(v_rise - v[lo], 0.0, None) / denom_lo

    # (ii)+(iii) Flat trust plateau followed by an extremely sharp
    # distrust cliff concentrated between v = 0.98 and v = 0.99.
    # In the mid-band the logistic term is ~0, so w ~ w_plateau
    # (nearly equal weights -> tally-like counting); at 0.98 the
    # weight is only mildly discounted, at 0.99 it is strongly
    # anti-diagnostic, and it saturates at -kappa_hi as v -> 1.
    hi = ~lo
    # Numerically stable logistic.
    x = (v[hi] - v_c) / s_hi
    sig = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                   np.exp(x) / (1.0 + np.exp(x)))
    w[hi] = w_plateau - (w_plateau + kappa_hi) * sig

    # Weighted evidence score. B's score is the mirror of A's
    # because every vote is zero-sum across the two options.
    s_a = float(np.dot(w, votes))
    scores = np.array([s_a, -s_a])

    # Numerically stable low-gain softmax. When all cues tie
    # (s_a == 0) the softmax is exactly uniform.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Independent lapse: with probability epsilon output a uniform
    # pick over the two options.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
- option_b_ratings: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
- response: 0 if subject chose A, 1 if subject chose B.

## IMPLEMENTATION GUARDRAILS
Any column in the schema above whose description names a list / tuple / np.ndarray (i.e. a per-trial sequence of values) holds non-scalar cells. Those cells are NOT hashable, so operations that hash row values fail with `TypeError: unhashable type: 'list'`. Treating `<seq_col>` as a placeholder for any such sequence-valued column:
- Avoid: `data.groupby('<seq_col>')`, `data['<seq_col>'].value_counts()`,     `data['<seq_col>'].nunique()`, `data['<seq_col>'].unique()` (returns     an object array but downstream `set()` / `in dict` will crash),     `set(data['<seq_col>'])`, `data['<seq_col>'].isin([...])` against list     values, or using a list cell as a dict key.
- If you need a hashable surrogate, project to one first, e.g.:
    - `data['<seq_col>_key'] = data['<seq_col>'].apply(tuple)` then group by `<seq_col>_key`
    - `data['<seq_col>_str'] = data['<seq_col>'].apply(lambda x: ''.join(map(str, x)))`
    Scalar columns (ints, floats, strings like `subject_id`, integer     responses, etc.) hash fine and can be used directly.
- Generator expressions inside function calls like `map()` or `join()` MUST be     parenthesized. For example:
    - WRONG: `map(str, int(v) for v in x)` → SyntaxError
    - RIGHT: `map(str, (int(v) for v in x))` or use a list comp: `[str(int(v)) for v in x]`
- Always verify your code is syntactically valid Python before returning it.

## METRICS YOU ALREADY TRIED AND FAILED ON
Each entry below is a metric you previously proposed in this round that did NOT discriminate the two theories at the human sample size — either it errored, its between-subject variance was unavailable, or Welch's t-test on `(self mean, self var, N)` vs. `(adv mean, adv var, N)` returned p ≥ alpha. The `outcome` line is the simulation result (means, between-subject variances, t-statistic and p-value at the human N) on the same `data_self` / `data_adv` your next metric will be evaluated on. Use the numbers to see where your hypothesised contrast collapsed — small mean gap, large per-subject variance, or both — and propose something qualitatively different. Don't repeat the same idea with cosmetic tweaks.
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # \"Follow-the-99%-expert rate\": across all trials where the single\n    # 0.99-validity expert (feature index 0 in the fixed validity vector\n    # [0.99, 0.98, 0.95, 0.90, 0.85, 0.75, 0.65, ., ., .]) discriminates\n    # between A and B, compute the proportion of choices that go to the\n    # option the 0.99 expert rated 1 (i.e., the option the top expert\n    # \"endorsed\").\n    #\n    # FDEI: w(0.99) is the LARGEST positive weight in the whole vector\n    # (its compression never inverts the validity ordering), so the\n    # 0.99-favored option is followed above chance on essentially every\n    # such trial -> metric ~ 0.60-0.65.\n    # BSST: w(0.99) is strongly NEGATIVE (~ -1.1 trusted-cue units, the\n    # distrust cliff), so the 0.99-favored option is chosen BELOW chance\n    # on every such trial (the background cues are balanced/weak in this\n    # design) -> metric ~ 0.15-0.35.\n    # The sign of (metric - 0.5) alone separates the two theories on\n    # every trial in the 0.99-discriminating set.\n\n    TOP = 0  # index of the 0.99-validity expert, fixed by the design\n\n    resp = data['response'].to_numpy()\n    follows = []\n\n    for i in range(len(data)):\n        a_raw = data['option_a_ratings'].iloc[i]\n        b_raw = data['option_b_ratings'].iloc[i]\n        try:\n            av = np.asarray(list(a_raw), dtype=int).ravel()\n            bv = np.asarray(list(b_raw), dtype=int).ravel()\n        except Exception:\n            continue\n        if av.size <= TOP or bv.size <= TOP:\n            continue\n\n        d = int(av[TOP]) - int(bv[TOP])\n        if d == 0:\n            continue  # 0.99 expert tied on this trial -> uninformative\n\n        favored_a = d > 0          # option the 0.99 expert endorsed\n        chose_a = (int(resp[i]) == 0)\n        follows.append(1.0 if (chose_a == favored_a) else 0.0)\n\n    if len(follows) == 0:\n        return 0.5  # neutral fallback (should not occur: 8 trials x 4 reps)\n\n    return float(np.mean(follows))",
  "rationale": "The single sharpest, most robust structural disagreement between FDEI and BSST in this design is the sign of the weight assigned to the 0.99-validity expert. FDEI's compressed-but-strictly-monotone weight function makes w(0.99) its LARGEST positive weight (about 1.16-1.36 cue units across its whole parameter box), so whenever the 0.99 expert discriminates, FDEI follows it above chance. BSST's distrust cliff makes w(0.99) strongly NEGATIVE (about -0.9 to -1.3 across its whole box), so BSST systematically chooses AGAINST the 0.99-endorsed option. In this 24-trial schedule, exactly 8 unique trials (trials 7, 8, 9, 17, 18, 19, 20, 21) have the feature-1 (0.99) expert discriminating; in every one of them the remaining discriminating cues form a balanced or weak background (mid-band cues split across sides, plus 0.5/0.55-flank cues that are near-zero under FDEI), so the 0.99 cue carries the sign of the evidence. With 4 repetitions this yields 32 informative trials per subject. Predicted values: FDEI ~0.62 (every parameter in its box gives follow-rates of roughly 0.56-0.73 on these trials, plus a small lapse pulling toward 0.5); BSST ~0.15-0.35 (anti-following on all 8 trial types under either the 0.5 or 0.55 flank-validity reading of the design, since the cliff weight dominates the balanced background). The gap (~0.3 in means) is large relative to per-subject sampling noise (binomial SD over 32 trials is about 0.08), so with N=25 subjects Welch's t-test separates the theories decisively. The metric is a plain behavioral proportion (no model fitting, no free parameters), works identically on the pooled data and on each single subject's slice, and is deliberately insensitive to the mid-band weight structure, the softmax gain, and the exact flank validities \u2014 it isolates exactly the one qualitative signature (follow vs. anti-follow a 99% expert) on which the two theories make opposite-signed, box-uniform predictions. It is not trivially true for FDEI: it is precisely the empirical claim at stake \u2014 that stated near-certainty is trusted (compressed, but not inverted) rather than distrusted \u2014 and a human follow-rate at or below 0.5 on these trials would falsify FDEI's entire parameter box."
}
```

## Usage

```json
{
  "input_tokens": 8673,
  "output_tokens": 10348,
  "reasoning_tokens": 9138,
  "provider": "Sail Research"
}
```
