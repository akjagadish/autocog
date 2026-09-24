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
**Validities (n_features=6):** [0.8, 0.9, 0.9, 0.9, 0.8, 0.9]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  trial 2: A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  trial 3: A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  trial 4: A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  trial 5: A=[1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0, 0]
  trial 6: A=[1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0]
  trial 7: A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 8: A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 9: A=[0, 1, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1, 0]
  trial 10: A=[0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 11: A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  trial 12: A=[1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  trial 13: A=[1, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  trial 14: A=[1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 1, 1, 0]
  trial 15: A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  trial 16: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Rationale:** GOAL: Quantitatively dissociate Distrust-Weighted Frugality with Reversal-Coherent Mixture (advocated) from Weakest-Expert-First Frugality (competing). The two theories are behaviorally IDENTICAL whenever the mixture mass m_am = ambiguity x gate(coherence) is ~0, because the advocated model's central component (w = -val) IS the competitor. The entire dissociation therefore rests on engineering the validity arrangement so that the advocated theory's own gate formulas open the anti-misbound channel: the subject population should be ~91% positional-reversal misbinders and only ~9% value-based anti-validity deciders, while the competitor asserts 100% value-based anti-validity deciders. SETUP: 6 features with validities [0.80, 0.55, 0.90, 0.55, 0.80, 0.65] — an anti-symmetric ('alternating') arrangement chosen so that (i) Spearman(position, validity) = -0.088, giving binding ambiguity = 0.912 (the validity order is essentially uncorrelated with display position, so the binding of accuracy to expert is maximally ambiguous), and (ii) |Spearman(validity, validity-reversed)| = 0.909, so the reversal-coherence gate sigmoid((0.909-0.45)/0.08) ~= 0.997 is fully OPEN. The advocated theory therefore commits, as a POINT PREDICTION with no free parameter, to m_am = 0.912 x 0.997 = 0.909: 90.9% of subjects adopt the anti-misbound hierarchy w = -val[::-1] and 9.1% the central w = -val. Crucially, the validities contain exact TIES (two 0.55s, two 0.80s): per the advocated theory's own tie-anchoring rule, per-cue zeta distortion is switched OFF (zeta_scale = 0), so its predictions are sharp rather than diffuse; and the trial set contains NO oppositions within any tied group ({f1,f3}, {f0,f4} for the value sort; {f2,f4}, {f1,f5} for the reversal), so the free tie_break parameter is inert in BOTH models — neither gains flexibility, and the theories differ only in hierarchy direction plus (beta, epsilon). THE TWO HIERARCHIES: competitor (ascending validity): {f1,f3} (55%) -> f5 (65%) -> {f0,f4} (80%) -> f2 (90%). Anti-misbound (w = -val[::-1]): {f2,f4} -> f0 -> {f1,f5} -> f3 — i.e., the misbinding subject consults the 90% and 80% experts FIRST and the 55% experts LAST, the exact transpose of the competitor's claim. Note this alignment of the reversal-coherent misbound hierarchy with descending validity is not an accident but a consequence the advocated theory itself forces: with high ambiguity, the gate can only open on a NEGATIVELY coherent (anti-symmetric) arrangement, under which trusting by mirror-position weakness equals trusting by own strength. The design therefore pits the competitor's core anti-validity claim (follow the WEAKEST-advertised discriminator) directly against the advocated claim (follow the misbound decider, which here is the STRONGEST-advertised discriminator). SIGNATURE 1 — DISJOINT PREDICTION INTERVALS ON 13 OF 16 CELLS: on every dissociating cell (T1-T13) the competitor's first discriminating cue and the anti-misbound first discriminating cue point to OPPOSITE options. Population-level, with follow rate p_f = (1-eps)*sigmoid(beta)+eps/2 in [0.59, 0.80] for both models: competitor predicts p(anti-misbound decider's option) = 1 - p_f in [0.20, 0.40]; advocated predicts 0.909*p_f + 0.091*(1-p_f) in [0.57, 0.74]. The intervals are DISJOINT, so any single dissociating cell with p > 0.45 toward the anti-misbound decider falsifies the competitor at any parameter setting, and p < 0.55 falsifies the advocated. T1-T12 are six exact A/B mirror pairs, ruling out side bias. SIGNATURE 2 — THE MIXTURE PROPORTION IS A PARAMETER-FREE POINT PREDICTION: agreement cells T14 (only the 55% expert f1 discriminates; both hierarchies follow it, at competitor depth 1 vs anti-misbound depth 4) and T15 (only the 65% expert f5 discriminates; depths 3 vs 4) pin the follow rate p_f; the dissociating cells then estimate the anti-misbound mass m via p_dissoc = m*p_f + (1-m)*(1-p_f). The advocated theory uniquely predicts m_hat ~= 0.91 FOR THIS ARRANGEMENT (a number derived entirely from its ambiguity/coherence gate formulas); the competitor predicts m_hat = 0. Observed m_hat > 0.45 falsifies the competitor; m_hat far below ~0.85 falsifies the advocated theory's gate calibration — a genuinely two-sided quantitative test. SIGNATURE 3 — PER-SUBJECT CONSISTENCY AND CLASSIFICATION: each subject holds ONE hierarchy, so choices across the 13 dissociating cells x 6 repetitions (78 trials) should be near-perfectly consistent with one of the two orders; the advocated predicts ~91%/9% split of subject types, the competitor 0%/100%. Per-trial likelihood ratio for an anti-misbound subject is (0.72/0.28)^78 — overwhelming model comparison even after fitting beta, epsilon, and tie_break to each model. SIGNATURE 4 — DECIDER-VALIDITY AND CASCADE-DEPTH INVARIANCE: across T1, T3, T5, T7, T9, T11, T13 the anti-misbound decider sweeps f2 (90%), f4 (80%), f0 (80%), f5 (65%) at cascade depths 1-3, and the competitor's decider sweeps f1/f3 (55%), f5 (65%), f0 (80%) at depths 1-4. Both one-reason theories predict a FLAT follow rate toward their own decider across all of these cells — any validity gradient or depth effect falsifies the one-reason claim of whichever model is otherwise supported. T3/T4 (90% vs 80% expert in conflict) is the strong-strong cell that removes any weak-vs-strong confound in interpreting the tilt; T9/T10 (65% vs 55%) is the weak-weak counterpart. SIGNATURE 5 — COALITION CELL (T13): the 90% expert alone favors A while the 55%, 55%, and 65% experts favor B (tally margin -2). The competitor follows a lone 55% expert against the 90% expert AND a 3-vs-1 majority; the advocated follows the 90% expert; any margin-integrating strategy says B — a three-way separation on one cell, with the one-reason models additionally predicting p(T13) = p(T1) (same deciders, different context). CONTROLS: T16 is a full tie (both theories: exactly 0.50) anchoring lapse rate and side bias. GLOBAL PROFILES: the competitor emits p(A) ~= 0.72 on T2, T4, T6, T8, T10, T12, T14 and ~= 0.28 on T1, T3, T5, T7, T9, T11, T13, T15, with 0.50 on T16; the advocated emits the near-TRANSPOSE on the 13 dissociating cells (~0.68 toward the misbound decider, slightly diluted below p_f by the 9% central mass — itself a quantitative signature: p_dissoc = 0.818*p_f + 0.091) while agreeing on T14/T15/T16. Because the advocated model's central component is exactly the competitor, the advocated nests the competitor; the experiment is therefore a clean one-sided falsification of the competitor (if the anti-misbound mass is real) plus a point test of the advocated gate (if it is not). With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

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
**Description:** Weakest-Expert-First Frugality (value-based anti-validity TTB). People are one-reason decision makers: they consult cues one at a time and let the first strictly discriminating cue decide, ignoring all lower cues and all magnitude/tally information. However, the subjective cue hierarchy is anti-correlated in VALUE with the instructed validities: subjects systematically consult the cue with the LOWEST advertised validity first, then the next-lowest, and so on (equivalently, subjective weights w_j = -validity_j). Because no integration across cues occurs, the model predicts a flat choice profile across tally margins (zero psychometric slope). When no cue discriminates, the model guesses uniformly. Validity ties are broken by a free per-subject tie-breaking parameter (early-position vs late-position preference among equally weak cues). Response noise enters through a softmax with inverse temperature beta over the binary winner score, plus an independent lapse epsilon mixing in a uniform choice, giving an implied cascade-follow probability p = (1-eps)*sigmoid(beta) + eps/2 that spans roughly 0.54 to 0.86 across subjects with a mean near 0.70 — high enough to produce the steep contrasts in the non-monotone-validity experiments while moderate values cover the monotone ones.

**Parameters:**
- beta: [0.5, 1.5]
- epsilon: [0.06, 0.18]
- tie_break: {0, 1}
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Weakest-Expert-First Frugality (value-based anti-validity TTB).
    # One-reason decision making: cues are consulted one at a time and
    # the first strictly discriminating cue decides. The subjective
    # hierarchy is anti-correlated in VALUE with the instructed
    # validities: the cue with the LOWEST advertised validity is
    # consulted FIRST, then the next-lowest, etc. (subjective weights
    # w_j = -validity_j). This is a VALUE-based sort (ascending
    # validity), not a positional reversal of the validity list, so in
    # experiments where the validity list is not monotone in feature
    # position the two hierarchies genuinely differ. No integration
    # across cues occurs -> flat psychometric profile across tally
    # margins. Noise: softmax (inverse temperature beta) over the
    # binary winner score, plus an independent lapse epsilon mixing in
    # a uniform choice. History is ignored (no feedback in this task).
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

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    tie_break = float(parameters["tie_break"])

    # Weakest-expert-first hierarchy: sort cues by ASCENDING validity
    # value. Ties in validity are broken by the free tie_break
    # parameter: 0 -> earlier feature position first (stable),
    # 1 -> later feature position first among equally weak cues.
    # np.lexsort uses the LAST key as primary: primary = val ascending,
    # secondary = position (forward or reversed).
    pos = np.arange(n_features)
    secondary = pos if tie_break < 0.5 else (n_features - 1 - pos)
    cue_order = np.lexsort((secondary, val))

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
        # No discriminating cue — pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta). The implied cascade-follow
    # probability is p = (1-eps)*sigmoid(beta) + eps/2, which over the
    # sampled (beta, epsilon) ranges spans ~[0.54, 0.86] across
    # subjects with a mean near 0.70.
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"\n    Anti-misbound decider-follow rate on hierarchy-dissociating cells.\n\n    For each trial we compute the option picked by the competitor's\n    hierarchy (weakest-advertised-expert-first: cues sorted by ASCENDING\n    validity, ties by display position) and the option picked by the\n    advocated theory's anti-misbound re-encoding (w = -val[::-1],\n    consulted in descending subjective weight, which is equivalent to\n    sorting cues by ASCENDING val[::-1], ties by display position), using\n    the experiment's validities [0.80, 0.55, 0.90, 0.55, 0.80, 0.65].\n\n    On this design the two hierarchies' first discriminating cues point to\n    OPPOSITE options on 13 of the 16 unique pairs (T1-T13) and agree on\n    T14/T15 (T16 is a full tie). The deciders are invariant to either\n    theory's free tie_break parameter because no trial opposes two cues\n    tied within a hierarchy group. The metric is the proportion of\n    choices that follow the anti-misbound decider on those 13\n    dissociating cells only.\n\n    Predicted values:\n      - Weakest-Expert-First (competing): subjects follow their own\n        (central) decider with p_f ~ 0.70, hence follow the anti-misbound\n        decider with ~0.30.\n      - Advocated: ~95% of subjects are anti-misbinders (m_am =\n    ambiguity x coherence-gate ~ 0.95 for this arrangement), so the\n        follow rate toward the anti-misbound decider is ~0.70.\n    The two theories therefore sit on opposite sides of 0.50 with a\n    ~0.40 gap, while per-subject values are tight (each subject\n    contributes 13 cells x 6 reps = 78 dissociating trials).\n    \"\"\"\n    VAL = [0.80, 0.55, 0.90, 0.55, 0.80, 0.65]\n    n = len(VAL)\n\n    # Competitor hierarchy: ascending advertised validity, position ties.\n    comp_order = sorted(range(n), key=lambda j: (VAL[j], j))\n\n    # Anti-misbound hierarchy: w = -val[::-1] consulted in DESCENDING\n    # subjective weight == ascending val[::-1], position ties.\n    rev = VAL[::-1]\n    mis_order = sorted(range(n), key=lambda j: (rev[j], j))\n\n    def decider(a, b, order):\n        for j in order:\n            if a[j] > b[j]:\n                return 0\n            if b[j] > a[j]:\n                return 1\n        return None\n\n    a_col = data[\"option_a_ratings\"].tolist()\n    b_col = data[\"option_b_ratings\"].tolist()\n    r_col = data[\"response\"].tolist()\n\n    hits = []\n    for a_raw, b_raw, r in zip(a_col, b_col, r_col):\n        try:\n            a = [int(v) for v in a_raw]\n            b = [int(v) for v in b_raw]\n        except Exception:\n            continue\n        if len(a) != n or len(b) != n:\n            continue\n        cd = decider(a, b, comp_order)\n        md = decider(a, b, mis_order)\n        # Keep only cells where the two hierarchies genuinely dissociate.\n        if cd is None or md is None or cd == md:\n            continue\n        hits.append(1.0 if int(r) == int(md) else 0.0)\n\n    if not hits:\n        return 0.5\n    return float(np.mean(hits))",
  "rationale": "The design's validities [0.80, 0.55, 0.90, 0.55, 0.80, 0.65] make the advocated theory's ambiguity x reversal-coherence gate open almost fully (Spearman(position, validity) ~ -0.05 -> ambiguity ~ 0.95; |Spearman(val, val[::-1])| = 0.909 -> gate ~ 0.997; m_am ~ 0.95), so the advocated theory commits to ~95% of subjects adopting the anti-misbound hierarchy w = -val[::-1] (cue order f2,f4 -> f0 -> f1,f5 -> f3), while the competing theory asserts 100% weakest-expert-first (f1,f3 -> f5 -> f0,f4 -> f2). Working through the actual code of both predict functions, these two hierarchies pick OPPOSITE options on exactly 13 of the 16 unique pairs (T1-T13; they agree on T14/T15 and T16 is a full tie), and \u2014 crucially \u2014 the decider on every cell is invariant to each theory's free tie_break parameter, because no trial contains an opposition between two cues tied within either hierarchy's tied groups ({f1,f3}, {f0,f4}, {f2,f4}, {f1,f5} never conflict). So the metric 'proportion of choices following the anti-misbound decider on the dissociating cells' is a clean, parameter-free contrast: the competing theory (all-central subjects, implied follow rate p_f ~ 0.70 mean) predicts ~0.30, while the advocated theory predicts ~0.95*p_f + 0.05*(1-p_f) ~ 0.70. That is a ~0.40 gap in means, and per-subject values are tight because every subject supplies 13 cells x 6 repetitions = 78 dissociating trials (binomial SD ~ 0.05 on top of a small p_f heterogeneity SD of ~0.03-0.05), so Welch's t-test at N=25 separates the two simulated datasets by many sigma. The metric is well defined both on the pooled DataFrame (one grand proportion) and on a single subject's slice (that subject's proportion over their 78 dissociating trials), so the between-subject variance is available. It is not trivially true for the advocated theory: it directly measures the empirical follow rate toward a specific decider, and any value near 0.5 (or near the competitor's 0.30) would falsify the advocated gate calibration, consistent with the design's two-sided test of the mixture mass."
}
```

## Usage

```json
{
  "input_tokens": 7830,
  "output_tokens": 43178,
  "reasoning_tokens": 41717,
  "provider": "Sail Research"
}
```
