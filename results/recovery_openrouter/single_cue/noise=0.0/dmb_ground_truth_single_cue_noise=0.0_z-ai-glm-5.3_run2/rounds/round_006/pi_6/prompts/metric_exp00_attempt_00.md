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
**Validities (n_features=6):** [0.86, 0.78, 0.95, 0.62, 0.7, 0.7]

**Trial pairs (n=16):**
  trial 1: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 2: A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 1, 1]
  trial 4: A=[1, 0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  trial 5: A=[0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  trial 6: A=[0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 8: A=[1, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 9: A=[0, 0, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 10: A=[0, 0, 1, 0, 1, 1]  B=[0, 0, 1, 1, 1, 0]
  trial 11: A=[1, 1, 0, 0, 0, 1]  B=[1, 1, 0, 0, 1, 0]
  trial 12: A=[1, 1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  trial 13: A=[0, 1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  trial 14: A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  trial 15: A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 1, 0]
  trial 16: A=[0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]

**Rationale:** GOAL: Quantitatively dissociate Attenuated Frugality with Position-Validity Mixture Heterogeneity (advocated) from Distrust-Weighted Frugality with Reversal-Coherent Mixture Heterogeneity (competing). Both are one-reason models sharing the anti-validity central component, so the dissociation must come from their DIFFERENT non-central heterogeneity channels: the advocated adds position-primacy (w = -position) and position-recency (w = +position) components whose masses grow LINEARLY in binding ambiguity with NO coherence dependence; the competing adds ONLY an anti-misbound component (w = -val[::-1]) whose adoption is gated by m_am = ambiguity x sigmoid((coherence-0.45)/0.08). Prior designs [10]-[12] all operated in the HIGH-coherence regime where the gate is OPEN ([12], the same theory pair with roles swapped, used coherence 0.727 and exact validity ties). This design occupies the opposite, unexplored corner: MAXIMAL ambiguity with NEAR-ZERO reversal coherence. This directly tests the gate-vs-linear structural difference: the competitor claims non-central heterogeneity requires reversal-coherent list structure; the advocated claims position heterogeneity tracks ambiguity ALONE.

SETUP: 6 features, ALL-DISTINCT validities [0.55, 0.86, 0.78, 0.95, 0.62, 0.70] (f0..f5). Arrangement statistics (both models' own formulas, distinct ranks): Spearman(position, validity) = 3/35 = 0.086, so binding ambiguity = 0.914; Spearman(validity, validity-reversed) = 1/35 = 0.029, so the competitor's gate = sigmoid((0.029-0.45)/0.08) = 0.005, essentially SHUT. A methodological novelty: because BOTH models scale their per-cue zeta distortion by ambiguity x (coherence or gate), both zeta scales are negligible here (advocated <= 0.037, competitor <= 0.008, vs. minimum validity/position weight gaps of 0.07/1.0) - predictions are SHARP without needing exact validity ties, and the free tie_break is inert in both models (no weight ties).

PARAMETER-FREE MIXTURE COMMITMENTS: Advocated: w_prim = 0.80x0.914, w_anti = 0.45, w_rec = 0.06x0.914, w_ttb = 0.04 -> normalized masses PRIMACY 0.573, ANTI-VALIDITY 0.353, RECENCY 0.043, TTB 0.031, MISBOUND 0. Competitor: m_am = 0.914 x 0.005 = 0.0047 -> ANTI-VALIDITY 0.995, MISBOUND 0.005, POSITION COMPONENTS 0 (structurally absent). The predicted type distributions have disjoint supports on the primacy mass (0.573 vs 0).

THE FIVE HIERARCHIES: ANTI (ascending validity): f0(.55) -> f4(.62) -> f5(.70) -> f2(.78) -> f1(.86) -> f3(.95). PRIMACY: f0 -> f1 -> f2 -> f3 -> f4 -> f5. RECENCY: f5 -> f4 -> f3 -> f2 -> f1 -> f0. TTB: f3 -> f1 -> f2 -> f5 -> f4 -> f0. MISBOUND: f2 -> f4 -> f3 -> f0 -> f1 -> f5. Crucially, ANTI and PRIMACY share f0 as top cue, so every dissociation trial keeps f0 TIED, forcing consultation to depth >= 2 where the two orders diverge: primacy's second cue is the 2nd-listed 86% expert f1, anti-validity's is the 5th-listed 62% expert f4.

SIGNATURE 1 - DIRECTION-FLIP DISSOCIATION GRID WITH DISJOINT INTERVALS (T1-T10): on T1/T3/T5 the earliest discriminator (f1/f2/f3, positions 1-3, validities 0.78-0.95) favors A while the weakest discriminator f4 (position 5, 62%) favors B; on T7/T9 the weakest discriminator is f5 (position 6, 70%). Deciders on every one of these cells: PRIMACY and TTB -> the early/strong cue; ANTI-VALIDITY, RECENCY, and MISBOUND -> the weak/late cue. Advocated: p(A) = 0.604*p_f + 0.396*(1-p_f) in [0.532, 0.567] (above 0.5 for ANY parameters). Competitor: p(A) = 1 - p_f (+0.005 misbound tilt on T3/T7/T9) in [0.208, 0.344] (below 0.5 for ANY parameters). THE INTERVALS ARE DISJOINT AND ON OPPOSITE SIDES OF 0.5 ON ALL TEN CELLS: the sign of the population preference on any single cell falsifies one theory at any parameter setting. T2/T4/T6/T8/T10 are exact A/B mirrors ruling out side bias. The grid also sweeps primacy-decider depth (2, 3, 4) and validity (0.78-0.95) and anti-decider identity (f4 vs f5): both theories predict a FLAT follow rate across the sweep - a shared one-reason flatness probe.

SIGNATURE 2 - THE PRIMACY-FLIP PAIRED CONTRAST, EXACTLY ZERO VS STRONGLY POSITIVE (T1/T5 vs T11, T2/T6 vs T12): T11 (f4->B, f5->A) has the IDENTICAL anti-validity and misbound deciders as T1/T5 (anti f4->B, mis f4->B), so the competitor predicts p(T11) = p(T1) = p(T5) EXACTLY (both = 1 - p_f) - it is structurally blind to the fact that the primacy, recency, and TTB deciders all flipped. The advocated predicts p(T1) - p(T11) = 0.530*(2*p_f - 1) in [0.165, 0.341]. The paired within-subject contrast is 0 (competitor) vs >= 0.165 (advocated) - disjoint. Mirrors give the symmetric negative contrast. This flat-vs-graded profile on a factor the competitor cannot represent is the central quantitative dissociation.

SIGNATURE 3 - THE ANTI-FLIP SWING MAGNITUDE (T13 vs T1): T13 (f1->A, f3->B) holds the primacy decider fixed (f1->A) while flipping the anti-validity decider (f1->A instead of f4->B). Competitor swing p(T13)-p(T1) = 1.98*p_f - 0.995 in [0.306, 0.577]; advocated swing = 0.644*p_f - 0.322 in [0.100, 0.208]. DISJOINT INTERVALS: the anti-flip swing alone separates the theories. Combined with Signature 2, the two swing magnitudes form a double dissociation: the competitor demands anti-flip swing (>= 0.31) >> primacy-flip swing (= 0); the advocated demands primacy-flip swing (>= 0.165) > anti-flip swing (<= 0.208), reflecting its 57/35 primacy-vs-anti mass split against the competitor's 99.5/0.5.

SIGNATURE 4 - PER-SUBJECT TYPE CLASSIFICATION (the sharpest test): 11 dissociation cells (T1-T10, T15) x 6 repetitions = 66 trials/subject. The competitor predicts ~99.5% of subjects are anti-validity-consistent (always following the weakest-advertised discriminator, f4 or f5); the advocated predicts only ~35% anti-consistent, ~57% PRIMACY-consistent (always following the earliest-listed discriminator), ~7% recency/TTB. A single primacy-consistent subject is strong evidence against the competitor (which assigns position rules zero mass); a near-100% anti-consistent sample falsifies the advocated's point-predicted 0.573 primacy mass - a genuinely two-sided test of each theory's heterogeneity structure at a parameter-free commitment.

SIGNATURE 5 - TTB CONFOUND GUARDS AND CONTROLS: T13/T14 are agreement cells (both theories ~0.69-0.72 toward the early cue f1) where standard validity-respecting TTB follows the 95% expert f3 the OTHER way - if subjects were TTB, p(T13) ~ 0.28, falsifying both target theories while validating the anti-validity claim shared by both. T15 pits PRIMACY ALONE (f1->A) against ANTI + RECENCY + TTB + MISBOUND (f3/f4->B): advocated p(A) = 0.573*p_f + 0.427*(1-p_f) in [0.523, 0.547] vs competitor 1 - p_f in [0.208, 0.344] - disjoint, and simultaneously separating primacy-following from strong-expert-following. T16 (full tie) pins both theories to exactly 0.50, anchoring lapse and side bias. The noise regimes also differ: per-subject follow-rate SD on the agreement cells (T13/T14) is ~0.02 under the competitor's tightened regime vs ~0.06 under the advocated's attenuated regime.

GLOBAL PREDICTED PROFILES (p_f ~ 0.72): Advocated: p(A) ~ 0.546 on T1,T3,T5,T7,T9; ~0.454 on T2,T4,T6,T8,T10; 0.312 on T11/T14; 0.688 on T12/T13; 0.532 on T15; 0.500 on T16. Competitor: ~0.28 on T1,T3,T5,T7,T9,T11,T15; ~0.72 on T2,T4,T6,T8,T10,T12,T13; ~0.28 on T14; 0.500 on T16. The competitor emits a two-level step function constant across ALL cells sharing its (anti, misbound) decider pair - in particular p(T1)=p(T11)=p(T15) exactly - while the advocated emits a graded profile on those same cells driven by its primacy mass. Model comparison on the 16-cell response profile separates the theories even after fitting beta, epsilon, and tie_break to each. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Attenuated Frugality with Position-Validity Mixture Heterogeneity. People are one-reason decision makers: cues are consulted one at a time and the first strictly discriminating cue decides, with no integration across cues (flat psychometric profile over tally margins, matching the margin-0 signatures of the knife-edge experiments). The subjective cue hierarchy is NOT universal: each subject draws a single consultation order from a graded mixture over a small candidate set: (1) anti-validity (distrust-the-weakest: consult the lowest advertised validity first) as the population-central tendency with a constant base weight; (2) position-primacy (consult experts in reading order, earliest-listed first); (3) position-recency (consult later-listed experts first, a minority variant); and (4) a residual standard-TTB component (descending validity). The position components' mixture weights grow smoothly and linearly with binding ambiguity A = 1 - |Spearman(display position, advertised validity)|, with no sigmoid gates of any kind: when the validity-to-position binding is unambiguous (monotone validity lists), the mixture collapses onto the anti-validity order (plus the small TTB residual), producing strong, consistent one-reason contrasts; when the binding is ambiguous (non-monotone lists), the position components enter, the pooled deciders disperse, and hierarchy-dissociation contrasts collapse toward chance. On top of the discrete mixture, a per-cue idiosyncratic weight distortion (zeta, fixed per subject, zero-centered) is applied, smoothly scaled by ambiguity times reversal-coherence and only when all advertised validities are distinct, so that all-distinct ambiguous designs degrade grain-by-grain while exact validity ties anchor the discrete rule mixture. Response discipline is attenuated and heterogeneous: the decider-follow probability p_f = (1-eps)*sigmoid(beta) + eps/2 spans roughly 0.55-0.85 across subjects with mean ~0.73 and SD ~0.06, so pooled contrasts are diluted both by mixture heterogeneity (disagreeing deciders) and by moderate, variable following. Validity ties are broken by a free per-subject tie-break; noise enters as softmax(beta) over the binary winner score plus an independent lapse epsilon; history is ignored (no feedback in this task).

**Parameters:**
- beta: [0.85, 1.8]
- epsilon: [0.10, 0.22]
- sigma_h: [0.4, 1.4]
- zeta: [(-1, 1)] * n_features
- tie_break: {0, 1}
- comp: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Attenuated Frugality with Position-Validity Mixture Heterogeneity.
    #
    # One-reason decision making: cues are consulted one at a time in
    # DESCENDING subjective weight and the first strictly discriminating
    # cue decides; no integration across cues (flat psychometric profile
    # over tally margins).
    #
    # Subjective hierarchy: each subject draws ONE consultation order from
    # a graded mixture over a small candidate set:
    #   - ANTI-VALIDITY (central):  w = -validity  (distrust the weakest
    #     advertised expert; consult ascending validity)
    #   - POSITION-PRIMACY:         w = -position  (reading order;
    #     earliest-listed expert first)
    #   - POSITION-RECENCY:         w = +position  (later-listed first;
    #     minority variant)
    #   - RESIDUAL TTB:             w = +validity  (descending validity)
    # The position components' weights grow SMOOTHLY and LINEARLY with
    # binding ambiguity
    #     A = 1 - |Spearman(display position, advertised validity)|
    # with no sigmoid gates: at A ~ 0 (monotone validity lists) the mixture
    # collapses onto anti-validity plus the small TTB residual; at high A
    # the position components enter and pooled deciders disperse, pulling
    # hierarchy-dissociation contrasts toward chance.
    #
    # Per-cue idiosyncratic distortion (zeta, zero-centered, fixed per
    # subject) is applied only when all advertised validities are DISTINCT,
    # smoothly scaled by ambiguity x reversal-coherence, so all-distinct
    # ambiguous designs degrade grain-by-grain while exact validity ties
    # anchor the discrete rule mixture.
    #
    # Noise: softmax(beta) over the binary winner score mixed with an
    # independent lapse epsilon. History is ignored (no feedback).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float).ravel()
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    sigma_h = float(parameters["sigma_h"])
    tie_break = float(parameters["tie_break"])
    comp = float(parameters["comp"])

    zeta = np.asarray(parameters["zeta"], dtype=float).ravel()
    if zeta.shape[0] != n_features:
        # Graceful fallback: no per-cue distortion.
        zeta = np.zeros(n_features)

    # ---- helper: average-tie ranks ----
    def avg_ranks(x):
        n = x.shape[0]
        srt = np.argsort(x, kind="stable")
        ranks = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[srt[j + 1]] == x[srt[i]]:
                j += 1
            ranks[srt[i:j + 1]] = 0.5 * (i + j)  # 0-indexed average rank
            i = j + 1
        return ranks

    # ---- binding ambiguity: A = 1 - |Spearman(position, validity)| ----
    # 0 when the validity list is monotone in display position (the
    # validity-to-position binding is unambiguous -> the mixture
    # concentrates on the anti-validity distrust order); growing with
    # position-validity conflict (ambiguous binding -> position-anchored
    # reading orders enter the mixture).
    if n_features > 1:
        ranks = avg_ranks(val)
        pos_r = np.arange(n_features, dtype=float)
        vp = pos_r - pos_r.mean()
        vr = ranks - ranks.mean()
        denom = np.sqrt(float((vp * vp).sum()) * float((vr * vr).sum()))
        if denom > 1e-12:
            rho = float((vp * vr).sum()) / denom
        else:
            # All validities tied: distrust order undefined -> maximal
            # idiosyncrasy.
            rho = 0.0
        ambiguity = 1.0 - abs(rho)
    else:
        ambiguity = 0.0

    # ---- reversal coherence: |Spearman(validity, validity[::-1])| ----
    # Used ONLY as a smooth (gate-free) multiplier on the per-cue zeta
    # distortion: idiosyncratic scrambling is strongest when the binding is
    # ambiguous AND the list structure is reversal-coherent enough that
    # small perturbations reshuffle the subjective order.
    if n_features > 1:
        ranks_v = avg_ranks(val)
        ranks_r = avg_ranks(val[::-1])
        dv = ranks_v - ranks_v.mean()
        dr = ranks_r - ranks_r.mean()
        denom2 = np.sqrt(float((dv * dv).sum()) * float((dr * dr).sum()))
        if denom2 > 1e-12:
            rho_rev = float((dv * dr).sum()) / denom2
        else:
            rho_rev = 0.0
        coherence = abs(rho_rev)
    else:
        coherence = 0.0

    # ---- graded mixture weights over the candidate subjective orders ----
    # Smooth in ambiguity; NO sigmoid gates. The per-subject draw `comp` ~
    # U(0,1) selects one component via the cumulative normalized weights.
    w_anti = 0.45                  # anti-validity: constant central mass
    w_rec = 0.06 * ambiguity       # position-recency: minority variant
    w_prim = 0.80 * ambiguity      # position-primacy: reading-order default
    w_ttb = 0.04                   # residual standard TTB
    z_tot = w_anti + w_rec + w_prim + w_ttb
    c_anti = w_anti / z_tot
    c_rec = c_anti + w_rec / z_tot
    c_prim = c_rec + w_prim / z_tot

    if comp < c_anti:
        # Distrust-the-weakest: anti-validity subjective weights.
        w_base = -val
    elif comp < c_rec:
        # Position-recency: later-listed experts consulted first.
        w_base = np.arange(n_features, dtype=float)
    elif comp < c_prim:
        # Position-primacy: earlier-listed experts consulted first.
        w_base = -np.arange(n_features, dtype=float)
    else:
        # Residual standard TTB: descending advertised validity.
        w_base = val

    # ---- per-cue idiosyncratic distortion (distinct lists only) ----
    # Exact validity ties anchor the discrete rule mixture (no per-cue
    # noise); all-distinct lists degrade grain-by-grain, smoothly scaled
    # by ambiguity x reversal-coherence.
    distinct = (np.unique(val).shape[0] == n_features)
    zeta_scale = sigma_h * ambiguity * coherence if distinct else 0.0

    w = w_base + zeta_scale * zeta

    # Consult cues in DESCENDING subjective weight. np.lexsort uses the
    # LAST key as primary. Exact ties in w are broken by the free
    # per-subject tie_break parameter: 0 -> earlier feature position
    # first, 1 -> later feature position first among equally weighted cues.
    pos = np.arange(n_features)
    secondary = pos if tie_break < 0.5 else (n_features - 1 - pos)
    cue_order = np.lexsort((secondary, -w))

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    n_opts = 2
    if winner is None:
        # No discriminating cue: pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta). Implied decider-follow
    # probability p = (1-eps)*sigmoid(beta) + eps/2 has mean ~0.73 with
    # per-subject SD ~0.06 over the sampled (beta, epsilon) ranges:
    # attenuated and variable, but high enough to reproduce the strong
    # margin-0 one-reason signatures of the monotone-validity designs.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / float(n_opts))
```

**`policy source code`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Distrust-Weighted Frugality with Reversal-Coherent Mixture Heterogeneity (softened-gate variant). People are one-reason decision makers: cues are consulted one at a time and the first strictly discriminating cue decides, with no integration across cues (flat psychometric slope over tally margins, consistent with the margin-0 signatures of Exps 3-8). The population-central subjective hierarchy is the anti-validity ('distrust') order: the weakest-advertised expert is consulted first (w = -validity). Heterogeneity is asymmetric and structured: each subject adopts one of a small family of re-encoding rules — distrust-the-weakest (w = -val) or distrust-the-misbinding (w = -val[::-1]) — with the anti-misbound rule's adoption probability gated by binding ambiguity (1 - |Spearman(position, validity)|) times a STEEP, SOFTENED reversal-coherence gate sigmoid((|Spearman(val, val[::-1])| - 0.45)/0.08): subjects adopt the anti-misbound re-encoding only when the binding is non-monotone AND the reversed list is structurally coherent enough to be learnable, with the gate threshold placed low enough (0.45 rather than 0.50) that moderately coherent designs (Exp 5, coherence 0.38) retain enough anti-misbound mass to reproduce their strong misbound-decider follow. Idiosyncratic per-cue distortion (zeta, zero-centered, fixed per subject) accumulates only when every advertised validity is distinct, scaled by the same ambiguity x coherence-gate product. Noise: softmax(beta) over the binary winner score plus lapse epsilon, in the tightened heterogeneous regime (implied follow p mean ~0.72, per-subject SD ~0.02). Validity ties are broken by a free per-subject tie-break. The theory reduces to Weakest-Expert-First Frugality (pi_4) in the ambiguity = 0 special case.

**Parameters:**
- beta: [0.9, 1.6]
- epsilon: [0.12, 0.26]
- sigma_h: [0.5, 1.75]
- zeta: [(-1, 1)] * n_features
- tie_break: {0, 1}
- comp: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Distrust-Weighted Frugality with Reversal-Coherent Mixture
    # Heterogeneity (softened coherence gate: center 0.45, width 0.08).
    #
    # One-reason decision making: cues are consulted one at a time in order
    # of DESCENDING subjective weight and the first strictly discriminating
    # cue decides; no integration across cues occurs (flat psychometric
    # profile over tally margins).
    #
    # Subjective hierarchy: each subject draws ONE component of a small
    # family of re-encoding rules:
    #   - distrust-the-weakest (central):  w_base = -validity
    #   - distrust-the-misbinding:         w_base = -validity[::-1]
    # The per-subject draw `comp` ~ U(0,1) selects the anti-misbound rule
    # when comp < m_am, where
    #     m_am = ambiguity * gate(coherence)
    #         ambiguity     = 1 - |Spearman(position, validity)|
    #         coherence     = |Spearman(validity list, its reversal)|
    #         gate(c)       = sigmoid((c - 0.45) / 0.08)
    # The gate is STEEP but its center is softened to 0.45 (from the
    # rejected 0.50): designs with coherence well above 0.45 (Exps 7, 8,
    # 10 at 0.62-0.71) admit the anti-misbound rule at high probability,
    # while moderately coherent designs (Exp 5 at 0.38) retain enough
    # anti-misbound mass (gate ~0.29) to reproduce their strong
    # misbound-decider follow, instead of being suppressed to near-zero
    # adoption as under the 0.50-centered gate.
    #
    # Per-cue idiosyncratic distortion (zeta, zero-centered, fixed per
    # subject) is applied ONLY when all advertised validities are DISTINCT,
    # and is scaled by the SAME ambiguity x gate(coherence) product so the
    # scrambling tracks the rule-mixture gating.
    #
    # Noise: softmax with inverse temperature beta over the binary winner
    # score, mixed with an independent lapse epsilon. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float).ravel()
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    sigma_h = float(parameters["sigma_h"])
    tie_break = float(parameters["tie_break"])
    comp = float(parameters["comp"])

    zeta = np.asarray(parameters["zeta"], dtype=float).ravel()
    if zeta.shape[0] != n_features:
        # Graceful fallback: no per-cue distortion.
        zeta = np.zeros(n_features)

    # ---- helper: average-tie ranks ----
    def avg_ranks(x):
        n = x.shape[0]
        srt = np.argsort(x, kind="stable")
        ranks = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[srt[j + 1]] == x[srt[i]]:
                j += 1
            ranks[srt[i:j + 1]] = 0.5 * (i + j)  # 0-indexed average rank
            i = j + 1
        return ranks

    # ---- binding ambiguity: 1 - |Spearman(position, validity)| ----
    # 0 when the validity list is monotone in display position (unambiguous
    # binding -> the population clusters on the pure anti-validity distrust
    # order), growing with position-validity conflict.
    if n_features > 1:
        ranks = avg_ranks(val)
        pos_r = np.arange(n_features, dtype=float)
        vp = pos_r - pos_r.mean()
        vr = ranks - ranks.mean()
        denom = np.sqrt(float((vp * vp).sum()) * float((vr * vr).sum()))
        if denom > 1e-12:
            rho = float((vp * vr).sum()) / denom
        else:
            # All validities tied: the distrust order is undefined ->
            # maximal idiosyncrasy.
            rho = 0.0
        ambiguity = 1.0 - abs(rho)
    else:
        ambiguity = 0.0

    # ---- reversal coherence + STEEP SOFTENED gate ----
    # coherence = |Spearman(validity, validity[::-1])|: how structurally
    # coherent the reversed validity list is. The adoption gate is
    #     gate(c) = sigmoid((c - 0.45) / 0.08)
    # computed via tanh for numerical stability. Center 0.45 (softened
    # from the rejected 0.50) so that moderately coherent designs keep
    # enough anti-misbound mass; width 0.08 keeps the discrimination
    # sharp between Exp 5 (0.38 -> gate ~0.29) and Exps 7/8/10
    # (0.62-0.71 -> gate ~0.90-0.98).
    if n_features > 1:
        ranks_v = avg_ranks(val)
        ranks_r = avg_ranks(val[::-1])
        dv = ranks_v - ranks_v.mean()
        dr = ranks_r - ranks_r.mean()
        denom2 = np.sqrt(float((dv * dv).sum()) * float((dr * dr).sum()))
        if denom2 > 1e-12:
            rho_rev = float((dv * dr).sum()) / denom2
        else:
            rho_rev = 0.0
        gate_x = (abs(rho_rev) - 0.45) / 0.08
        coherence_gate = 0.5 * (1.0 + float(np.tanh(0.5 * gate_x)))
        m_am = ambiguity * coherence_gate
    else:
        coherence_gate = 0.0
        m_am = 0.0

    # ---- tie anchoring: per-cue distortion only with all-distinct lists ----
    # Exact validity ties anchor the re-encoding (rule-like mixture adoption,
    # no per-cue noise); all-distinct lists degrade grain-by-grain (zeta on),
    # scaled by the same ambiguity x coherence-gate product.
    distinct = (np.unique(val).shape[0] == n_features)
    zeta_scale = sigma_h * ambiguity * coherence_gate if distinct else 0.0

    # ---- per-subject component draw over the hierarchy family ----
    if comp < m_am:
        # Distrust-the-misbinding: anti-misbound subjective weights.
        w_base = -val[::-1]
    else:
        # Central distrust-the-weakest: anti-validity subjective weights.
        w_base = -val

    w = w_base + zeta_scale * zeta

    # Consult cues in DESCENDING subjective weight. np.lexsort uses the
    # LAST key as primary. Exact ties in w are broken by the free
    # tie_break parameter: 0 -> earlier feature position first,
    # 1 -> later feature position first among equally weak cues.
    pos = np.arange(n_features)
    secondary = pos if tie_break < 0.5 else (n_features - 1 - pos)
    cue_order = np.lexsort((secondary, -w))

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    n_opts = 2
    if winner is None:
        # No discriminating cue: pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta). Implied decider-follow
    # probability p = (1-eps)*sigmoid(beta) + eps/2 has mean ~0.72 with
    # per-subject SD ~0.02 over the tightened (beta, epsilon) ranges.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / float(n_opts))
```

**`policy source code`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\n\ndef metric(data: pd.DataFrame) -> float:\n    \"\"\"\n    Primacy-type excess vs validity-respecting-type excess.\n\n    Two cell families are selected by their discriminating-cue sets\n    (feature positions where the two options' ratings differ):\n\n    Family 1 (\"early-cue cells\"; design trials T1-T4, T7-T8, T13, T15):\n        sets {1,4}, {2,4}, {1,5}, {1,3}, {1,3,4}. The earliest-listed\n        discriminating expert (f1 or f2) favors the opposite option from\n        every later discriminating expert, so a position-primacy reader\n        and an anti-validity (weakest-first) reader pick opposite options.\n\n    Family 2 (\"weak-cue cells\"; design trials T5-T6, T9-T10):\n        sets {3,4}, {3,5}. The earliest-listed discriminating expert is\n        the low-validity f3, opposing f4/f5, so primacy and anti-validity\n        readers disagree here as well, but a descending-validity (TTB-like\n        or anti-misbound) reader now sides with the anti-validity reader.\n\n    Per subject: X1 = rate of choosing the earliest-cue option on\n    Family 1, X2 = same on Family 2. Score = +1 if the subject follows\n    the earliest-listed expert on BOTH families (pure position-primacy\n    consultation order), -1 if it follows it on Family 1 but reverses on\n    Family 2 (validity-respecting order), 0 otherwise. The metric is the\n    mean score: the excess of primacy-type subjects over\n    validity-respecting-type subjects.\n    \"\"\"\n    fam1 = {frozenset((1, 4)), frozenset((2, 4)), frozenset((1, 5)),\n            frozenset((1, 3)), frozenset((1, 3, 4))}\n    fam2 = {frozenset((3, 4)), frozenset((3, 5))}\n\n    def _resp(v):\n        if isinstance(v, str):\n            return 0 if v.strip().upper() in (\"A\", \"0\") else 1\n        try:\n            return int(v)\n        except (TypeError, ValueError):\n            return -1\n\n    scores = []\n    for _, sdf in data.groupby(\"subject_id\"):\n        f1, f2 = [], []\n        for _, row in sdf.iterrows():\n            try:\n                a = np.asarray(row[\"option_a_ratings\"], dtype=int).ravel()\n                b = np.asarray(row[\"option_b_ratings\"], dtype=int).ravel()\n            except (TypeError, ValueError):\n                continue\n            if a.size == 0 or a.size != b.size:\n                continue\n            diff = np.nonzero(a != b)[0]\n            if diff.size == 0:\n                continue\n            dset = frozenset(int(j) for j in diff)\n            in1 = dset in fam1\n            in2 = dset in fam2\n            if not (in1 or in2):\n                continue\n            j0 = int(diff[0])  # earliest-listed discriminating expert\n            fav0 = 0 if a[j0] == 1 else 1  # option favored by that cue\n            # earliest cue must oppose every later discriminating cue\n            ok = True\n            for j in diff[1:]:\n                if (0 if a[int(j)] == 1 else 1) == fav0:\n                    ok = False\n                    break\n            if not ok:\n                continue\n            r = _resp(row[\"response\"])\n            if r not in (0, 1):\n                continue\n            follow = 1.0 if r == fav0 else 0.0\n            if in1:\n                f1.append(follow)\n            else:\n                f2.append(follow)\n        if not f1 or not f2:\n            continue\n        x1 = float(np.mean(f1))\n        x2 = float(np.mean(f2))\n        if x1 <= 0.5:\n            s = 0.0\n        elif x2 > 0.5:\n            s = 1.0\n        elif x2 < 0.5:\n            s = -1.0\n        else:\n            s = 0.0\n        scores.append(s)\n\n    if not scores:\n        return 0.0\n    return float(np.mean(scores))",
  "rationale": "Both theories are one-reason models sharing the anti-validity central order, so pooled single-cell contrasts are nearly collinear (their predicted p(A) profiles differ by at most ~0.13 on any cell). The real dispute is the STRUCTURE of between-subject heterogeneity: the advocated theory says a large position-primacy mass enters linearly with binding ambiguity (here ~36% of subjects consult experts in reading order), while the competing theory says heterogeneity is only anti-validity vs anti-misbound re-encoding, with position rules structurally absent. The metric therefore classifies each subject's consultation order from two cell families and returns the excess of primacy-type over validity-respecting-type subjects.\n\nFamily 1 (trials whose discriminating-cue sets are {f1,f4}, {f2,f4}, {f1,f5}, {f1,f3}, {f1,f3,f4}: T1-T4, T7-T8, T13, T15, 48 trials/subject): the earliest-listed discriminating expert opposes every later one, so primacy readers and anti-validity readers pick opposite options. Family 2 (sets {f3,f4}, {f3,f5}: T5-T6, T9-T10, 24 trials/subject): the earliest-listed discriminating expert is the weak 62% expert f3, so primacy readers now OPPOSE both the anti-validity readers and any descending-validity/anti-misbound reader. A subject following the earliest-listed expert on BOTH families is a pure position-primacy type (+1); one following it on Family 1 but reversing on Family 2 has a validity-respecting order (-1); anti-validity subjects score 0.\n\nPredicted values (following probabilities p_f in 0.69-0.73 give clean 0.27/0.73 per-cell rates, far from the 0.5 classification threshold over 48/24 trials):\n- Advocated: ~36% primacy-type subjects (+1), ~5% TTB-type (-1), ~59% anti/recency (0) -> metric ~ +0.31, per-subject SD ~ 0.56.\n- Competitor: 0% primacy-type (position rules have zero mass); its ~36% anti-misbound subjects follow the early cue on Family 1 but the weak cue on Family 2 (-1), its ~64% anti-validity subjects score 0 -> metric ~ -0.36, per-subject SD ~ 0.48.\n\nWelch t at N=25: t ~ 4.6 (p < 1e-4), comfortably below alpha=0.01, because the supports of the primacy-type mass are disjoint (0.36 vs 0) and the validity-type mass differs 7-fold in the opposite direction (0.05 vs 0.36). The contrast is also robust to the alternative all-distinct validity arrangement ([0.55,0.86,0.78,0.95,0.62,0.70]): there the advocated yields ~ +0.57 (57% primacy-type) vs ~ -0.01 for the competitor (gate shut, ~100% anti-validity, all scoring 0), t ~ 5.8. The metric is well-defined on a single subject's slice (score in {-1,0,+1}), so between-subject variance is always available, and it uses only stimulus-derived quantities (discriminating-cue positions), so it is invariant to any A/B label randomization."
}
```

## Usage

```json
{
  "input_tokens": 9940,
  "output_tokens": 73001,
  "reasoning_tokens": 71171,
  "provider": "Sail Research"
}
```
