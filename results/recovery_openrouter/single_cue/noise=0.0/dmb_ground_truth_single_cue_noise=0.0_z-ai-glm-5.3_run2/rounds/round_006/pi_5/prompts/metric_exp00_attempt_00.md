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
**Validities (n_features=6):** [0.65, 0.9, 0.5, 0.8, 0.5, 0.9]

**Trial pairs (n=16):**
  trial 1: A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  trial 2: A=[1, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  trial 3: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 4: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 5: A=[0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 6: A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 1]
  trial 7: A=[1, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 8: A=[1, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  trial 9: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 1]
  trial 10: A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 1]
  trial 11: A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  trial 12: A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 13: A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 14: A=[1, 0, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 15: A=[1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1]
  trial 16: A=[0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 0]

**Rationale:** GOAL: Quantitatively dissociate Distrust-Weighted Frugality with Reversal-Coherent Mixture Heterogeneity (advocated) from Attenuated Frugality with Position-Validity Mixture Heterogeneity (competing). Both are one-reason models sharing the anti-validity central component, so the dissociation must come from their DIFFERENT non-central heterogeneity channels: the advocated adds ONLY an anti-misbound component (w = -val[::-1]) whose adoption is gated by m_am = ambiguity x sigmoid((coherence-0.45)/0.08); the competitor adds ONLY position-primacy (w = -position) and position-recency (w = +position) components whose weights grow LINEARLY in ambiguity, plus a small TTB residual — with NO misbound component and NO gate. The design engineers a validity arrangement that opens BOTH theories' non-central channels to their committed maxima, then pits their unique components head-to-head.

SETUP: 6 features, validities [0.65, 0.90, 0.55, 0.80, 0.55, 0.90]. Arrangement statistics (computed with both models' own formulas, average-tie ranks): Spearman(position, validity) = +0.088, so binding ambiguity = 0.912 — high enough that the competitor's position components dominate its mixture (normalized masses: primacy 0.573, anti-validity 0.353, recency 0.043, TTB 0.031) AND the advocated's gate product is large. Spearman(validity, validity-reversed) = -0.727, so reversal coherence = 0.727, gate = sigmoid((0.727-0.45)/0.08) = 0.970, and the advocated commits — as a PARAMETER-FREE POINT PREDICTION — to m_am = 0.912 x 0.970 = 0.884: 88.4% of subjects adopt the anti-misbound hierarchy, 11.6% the central anti-validity hierarchy, 0% primacy/recency/TTB. The competitor commits to 57.3% primacy, 35.3% anti-validity, 7.4% recency+TTB, and 0% misbound. The predicted mixture distributions have DISJOINT SUPPORTS on both the misbound and primacy masses. The validities contain exact ties (two 0.55s, two 0.90s), which switches per-cue zeta distortion OFF in BOTH models (each model's own tie-anchoring rule), making predictions sharp; and every trial keeps discriminating cues concordant within each tied group (features {f0,f2,f4} never internally oppose; {f1,f3,f5} never internally oppose), so the free tie_break parameter is inert in both models — neither gains flexibility, and the theories differ only in mixture composition plus (beta, epsilon).

THE FIVE HIERARCHIES under this arrangement: anti-validity: {f2,f4}(55%) -> f0(65%) -> f3(80%) -> {f1,f5}(90%); anti-misbound (w = -val[::-1]): {f1,f3} -> f5 -> f2 -> {f0,f4} (consulted validities 90, 80, 90, 55, 65, 55 — a genuinely scrambled, non-monotone order reducible to no validity heuristic); primacy: f0->f1->f2->f3->f4->f5; recency: f5->f4->f3->f2->f1->f0; TTB: {f1,f5} -> f3 -> f0 -> {f2,f4}. Because the strong experts (0.90, 0.90, 0.80) occupy ODD positions and the weak ones (0.55, 0.55, 0.65) EVEN positions, every trial with both groups discriminating in opposition pits the anti-misbound decider (odd group) against the anti-validity decider (even group), while the primacy and recency deciders are STEERED INDEPENDENTLY by choosing which specific features discriminate (primacy follows the earliest discriminating feature; recency the latest).

SIGNATURE 1 — PRIM-FACTOR FLATNESS (advocated) vs GRADED (competitor): trials T1, T3, T5, T7, T9, T11 all have misbound-decider -> A and anti-validity-decider -> B, but the primacy decider points to B on T1/T3/T5/T9 and to A on T7/T11, and the recency decider flips across T3/T5 (->A), T9 (->B), T11 (->B). The advocated theory has NO position components, so it predicts EXACTLY EQUAL p(A) = m_am*p_f + (1-m_am)*(1-p_f) ~= 0.67 on all six cells (p_f = (1-eps)*sigmoid(beta)+eps/2 in [0.66, 0.72] in its tightened regime). The competitor predicts 0.27 (T1), 0.30 (T3), 0.30 (T5), 0.57 (T7), 0.28 (T9), 0.55 (T11) — a ~0.30 spread driven by its 57% primacy mass. The paired contrasts p(T7)-p(T3) and p(T11)-p(T9) are +0.26 under the competitor and EXACTLY ZERO under the advocated; the mirror contrasts p(T8)-p(T4) and p(T12)-p(T10) are -0.26 vs zero. A flat-vs-graded profile on a factor (position-rule decider direction) that the advocated theory is structurally blind to is the central quantitative dissociation.

SIGNATURE 2 — THE GRAND DISSOCIATION CELL (T1/T2): on T1 the anti-misbound decider (f3, the 80% expert) favors A while ALL FOUR competitor components (anti-validity via f2, primacy via f0, recency via f5, TTB via f5) favor B. Advocated: p(A) ~= 0.67 [0.62, 0.72]; competitor: p(A) = 1-p_f ~= 0.27 [0.18, 0.40] — disjoint intervals on a single cell. T2 is the exact mirror (advocated 0.33, competitor 0.73). T1 additionally separates the misbound hierarchy from validity-respecting TTB (which follows f5 to B): if the majority behavior were standard TTB rather than misbinding, choices on T1 go to B. T1 also has tally margin -3, so any margin-integrating strategy (Tallying) predicts p(A) well below 0.5 there while the advocated predicts 0.67 — the orientation-1 cells sweep margins -3, 0, +1, +2, -1, 0 with a FLAT advocated prediction, simultaneously falsifying integration strategies.

SIGNATURE 3 — POINT-PREDICTED MIXTURE MASSES: agreement cells T13-T15 (all five hierarchies concur; both theories: p_f toward the common decider) pin the follow rate p_f without knowing the mixture; the 12 dissociation cells then estimate the population hierarchy masses. The advocated uniquely predicts mass(misbound) = 0.884 and mass(primacy) = 0 for THIS arrangement — numbers derived entirely from its own ambiguity x coherence-gate formulas; the competitor predicts mass(primacy) = 0.573 and mass(misbound) = 0. Observed misbound mass > 0.2 falsifies the competitor at any parameter setting; observed primacy mass > 0.1 falsifies the advocated — a genuinely two-sided quantitative test of each theory's heterogeneity structure.

SIGNATURE 4 — PER-SUBJECT TYPE CLASSIFICATION: both theories give each subject ONE hierarchy; the 12 dissociation cells x 6 repetitions (72 trials/subject) classify subjects nearly perfectly (per-trial likelihood ratio up to (0.72/0.28)^6 per cell). Advocated: ~88% misbound-consistent, ~12% anti-validity-consistent, 0% primacy-consistent. Competitor: ~57% primacy-consistent, ~35% anti-validity-consistent, ~7% recency/TTB, 0% misbound-consistent. A single primacy-consistent subject is evidence against the advocated; a single misbound-consistent subject (beyond the ~3% TTB confound, itself resolved by T1) is evidence against the competitor.

SIGNATURE 5 — SHARED FLATNESS PROBES: T3 vs T5 (anti-validity decider f2 at depth 1 vs f0 at depth 3; misbound decider f1 at 0.90 vs f3 at 0.80) and T1 vs T3 (misbound decider f3 vs f1): both theories predict EQUAL follow rates — any depth or decider-validity gradient falsifies the one-reason core of whichever model is otherwise supported. SIGNATURE 6 — NOISE REGIME: per-subject follow-rate SD on T13-T15 discriminates the advocated's tightened regime (~0.02) from the competitor's attenuated regime (~0.06). CONTROLS: T2/T4/T6/T8/T10/T12 are exact A/B mirrors of T1/T3/T5/T7/T9/T11, ruling out side bias; T16 is a full tie (both theories: exactly 0.50) anchoring lapse rate. GLOBAL PREDICTED PROFILES: the advocated emits a two-level profile FLAT within each orientation (~0.67 on T1,T3,T5,T7,T9,T11; ~0.33 on mirrors; p_f on T13,T15; 1-p_f on T14; 0.50 on T16); the competitor emits a graded profile within each orientation (0.27-0.57 and 0.43-0.73 respectively) driven by the primacy mass. The profiles are structurally disjoint — model comparison on the 16-cell response profile separates the theories even after fitting beta, epsilon, and tie_break to each. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Primacy-follow rate on the 8 designed dissociation trials where the\n    # position-primacy decider (consult f0, then f1, ...) and the\n    # anti-misbound decider (subjective weights w = -validity[::-1]) point\n    # to OPPOSITE options: trial types T1, T3, T5, T9 and their exact A/B\n    # mirrors T2, T4, T6, T10 (48 of the 96 trials per subject).\n    #\n    # Advocated theory (misbound mixture, m_am ~ 0.885 here):\n    #   ~88.5% of subjects follow the anti-misbound decider, which on these\n    #   8 trials OPPOSES the primacy decider  ->  P(follow primacy) = 1-p_f\n    #   ~ 0.28.  ~11.5% follow the anti-validity decider, which on these\nn    #   same 8 trials COINCIDES with the primacy decider -> P = p_f ~ 0.72.\n    #   Population value ~ 0.33.\n    # Competing theory (position mixture, primacy mass 0.573, anti-val 0.353):\n    #   BOTH the primacy subjects and the anti-validity subjects follow the\n    #   primacy decider on these trials -> P ~ p_f ~ 0.72; the small\n    #   recency/TTB residuals sit near 0.4-0.5.  Population value ~ 0.70.\n    # Disjoint intervals, tight per-subject spread on both sides.\n    n_features = 6\n    val = np.array([0.65, 0.90, 0.50, 0.80, 0.50, 0.90], dtype=float)\n\n    # Subjective weights of the two rival one-reason hierarchies.\n    w_misbound = -val[::-1]                            # anti-misbound re-encoding\n    w_primacy = -np.arange(n_features, dtype=float)    # reading-order (position) hierarchy\n\n    def decider(a, b, w):\n        # Consult cues in DESCENDING subjective weight (same lexsort\n        # convention as both theories' predict functions; ties broken by\n        # earlier position -- verified inert on these trials).\n        order = np.lexsort((np.arange(len(w)), -w))\n        for j in order:\n            if a[j] > b[j]:\n                return 0\n            if b[j] > a[j]:\n                return 1\n        return None\n\n    # The 8 dissociation trial types (T1, T3, T5, T9 and mirrors T2, T4,\n    # T6, T10). On each, the anti-misbound decider and the position-primacy\n    # decider pick opposite options (verified in-code below).\n    trials = [\n        ([0, 1, 0, 1, 0, 0], [1, 1, 1, 0, 1, 1]),\n        ([0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]),\n        ([0, 1, 1, 1, 0, 1], [1, 1, 1, 0, 0, 0]),\n        ([0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 1]),\n        ([1, 1, 1, 0, 1, 1], [0, 1, 0, 1, 0, 0]),\n        ([1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1]),\n        ([1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 0, 1]),\n        ([1, 0, 1, 0, 1, 1], [0, 1, 0, 1, 0, 1]),\n    ]\n\n    prim_target = {}\n    for a_list, b_list in trials:\n        a = np.asarray(a_list, dtype=int)\n        b = np.asarray(b_list, dtype=int)\n        dm = decider(a, b, w_misbound)\n        dp = decider(a, b, w_primacy)\n        if dm is not None and dp is not None and dm != dp:\n            key_a = tuple([int(v) for v in a])\n            key_b = tuple([int(v) for v in b])\n            prim_target[(key_a, key_b)] = int(dp)\n\n    if not prim_target:\n        return 0.5\n\n    hits = []\n    for idx, row in data.iterrows():\n        a = tuple([int(v) for v in np.asarray(row[\"option_a_ratings\"]).ravel()])\n        b = tuple([int(v) for v in np.asarray(row[\"option_b_ratings\"]).ravel()])\n        key = (a, b)\n        if key in prim_target:\n            resp = int(row[\"response\"])\n            hits.append(1.0 if resp == prim_target[key] else 0.0)\n\n    if len(hits) == 0:\n        return 0.5\n    return float(np.mean(hits))",
  "rationale": "The metric is the empirical rate of choosing the position-primacy decider's option, computed only on the 8 trial types (T1, T3, T5, T9 and their exact A/B mirrors T2, T4, T6, T10; 48 of 96 trials per subject) on which the primacy decider (first discriminating cue in reading order, always f0 on these trials) and the anti-misbound decider (subjective weights w = -validity[::-1], order {f1,f3}->f5->f2->{f0,f4}) point to OPPOSITE options. Both decider directions are recomputed inside the metric from the fixed design validities, and the f1/f3 and f0/f4 weight ties are verified inert on these trials, so the trial classification is exact and tie-break independent; zeta distortion is off in both theories (validity ties present), so both theories' per-subject deciders are deterministic given the mixture component.\n\nPredicted values: Under the ADVOCATED theory, m_am = 0.912 * sigmoid((0.727-0.45)/0.08) = 0.885, so ~88.5% of subjects are anti-misbound (they follow the decider that OPPOSES primacy here: P(follow primacy) = 1-p_f ~ 0.28) and ~11.5% are anti-validity (whose decider COINCIDES with the primacy decider on all 8 trials: P = p_f ~ 0.72). Population value ~ 0.33, with per-subject values at either ~0.28 or ~0.72 (mixture SD ~ 0.16, variance ~ 0.025 including binomial noise over 48 trials). Under the COMPETING theory, the mixture is 57.3% primacy + 35.3% anti-validity \u2014 BOTH of which follow the primacy decider on these 8 trials (P = p_f ~ 0.72 each) \u2014 plus ~4.3% recency (~0.5) and ~3.1% TTB (~0.39). Population value ~ 0.70, per-subject variance ~ 0.015 (nearly all subjects sit at p_f, which has SD ~0.06, plus binomial noise). The two population values are separated by ~0.37 with small per-subject variances on both sides: Welch's t at N=25 is approximately 0.37 / sqrt((0.025+0.017)/25) ~ 9, far beyond alpha = 0.01.\n\nThis is the design's head-to-head 'grand dissociation' contrast: it pits the advocated theory's unique anti-misbound component directly against the competitor's unique primacy mass on exactly the trials where their deciders disagree, while the competitor's anti-validity mass (shared intuition with the advocated's minority component) is steered onto the SAME side as primacy, so the competitor cannot hide behind its anti-validity subjects. The metric is a simple, non-trivial choice proportion (not a fitted quantity), works identically on the pooled DataFrame and on each single subject's 48-trial slice, and its direction is informative: values near 0.3 support the anti-misbound mixture, values near 0.7 support the position-primacy mixture."
}
```

## Usage

```json
{
  "input_tokens": 9783,
  "output_tokens": 21613,
  "reasoning_tokens": 19578,
  "provider": "Sail Research"
}
```
