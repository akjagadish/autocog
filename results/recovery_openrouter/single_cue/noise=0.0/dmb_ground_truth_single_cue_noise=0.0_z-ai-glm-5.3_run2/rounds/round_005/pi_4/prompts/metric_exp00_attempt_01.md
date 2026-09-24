# metric_exp00_attempt_01

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
**Validities (n_features=6):** [0.75, 0.9, 0.65, 0.8, 0.85, 0.6]

**Trial pairs (n=16):**
  trial 1: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1]
  trial 4: A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  trial 5: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 6: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 7: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  trial 10: A=[1, 1, 1, 1, 0, 1]  B=[1, 1, 0, 0, 1, 1]
  trial 11: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  trial 12: A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  trial 13: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 14: A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  trial 15: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 16: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Rationale:** GOAL: Quantitatively dissociate Weakest-Expert-First Frugality (advocated; value-based ascending-validity cascade) from Distrust-Weighted Frugality with Reversal-Coherent Mixture Heterogeneity (competing). The competitor NESTS my theory as its central component (w = -val), so a dissociation is only possible by engineering the validity arrangement so that the competitor's own gate formulas commit it to a large anti-misbound mass — then its population prediction differs from mine on every cell where the two hierarchies' first discriminating cues oppose. The design must also be novel relative to experiment [10] (which used a near-fully-coherent anti-symmetric arrangement where the anti-misbound hierarchy equaled descending validity, i.e. 'strongest-first'). Here the anti-misbound hierarchy is a genuinely SCRAMBLED, non-monotone-in-validity order, which (a) decouples the competitor's prediction from any strength heuristic, and (b) yields a different, precisely-computed gate commitment. SETUP: 6 features, validities [0.75, 0.55, 0.90, 0.65, 0.80, 0.55]. The exact ties (two 0.55s) deliberately switch the competitor's per-cue zeta distortion OFF (its own tie-anchoring rule: zeta_scale = 0 when validities are not all distinct), so its predictions are sharp, not diffuse; and the trial set contains NO oppositions within any tied group ({f1,f5} for the value sort, {f0,f4} for the positional reversal), so the free tie_break parameter is inert in BOTH models — neither gains flexibility, and the theories differ only in hierarchy direction plus (beta, epsilon). ARRANGEMENT STATISTICS (computed with the competitor's own formulas): Spearman(position, validity) = -0.145 -> binding ambiguity = 0.855; Spearman(validity, validity-reversed) = -0.618 -> coherence = 0.618; softened gate sigmoid((0.618-0.45)/0.08) = 0.890; hence m_am = 0.855 x 0.890 = 0.761. THE COMPETITOR'S POINT PREDICTION (no free parameter): 76.1% of subjects adopt the anti-misbound hierarchy w = -val[::-1], 23.9% the central w = -val. MY THEORY: 100% value-based anti-validity (weakest-advertised expert first). THE TWO HIERARCHIES: central (mine): {f1, f5} (55%) -> f3 (65%) -> f0 (75%) -> f4 (80%) -> f2 (90%). Anti-misbound: {f0, f4} -> f2 -> f5 -> f1 -> f3, i.e. in OWN advertised validity the misbinding cascade runs 75% -> 80% -> 90% -> 55% -> 55% -> 65%: a NON-MONOTONE schedule that is neither 'strongest-first' nor 'weakest-first' — the competitor's majority-subject behavior cannot be redescribed by any validity heuristic. SIGNATURE 1 — DIRECTION-FLIP DISSOCIATION GRID WITH DISJOINT PREDICTION INTERVALS (T1-T6, T11): six mirror-pair cells where the central decider (55% or 65% expert) opposes the anti-misbound decider (75%, 80%, or 90% expert), plus the coalition cell T11 (the lone 55% expert f1 for A against f0, f2, f4 for B; margin -2). My theory: p(central decider's option) = p_f = (1-eps)*sigmoid(beta)+eps/2 in [0.60, 0.80] over my parameter ranges. Competitor: p(central side) = 0.239*p_f + 0.761*(1-p_f) in [0.35, 0.42] over ITS ranges (p_f in [0.66, 0.79]; ~0.39 at its tightened-regime mean 0.72). THE INTERVALS ARE DISJOINT (gap >= 0.18), so any single dissociating cell at the population level falsifies one theory at any parameter setting; pooling all 11 dissociation cells (66 trials/subject) makes per-subject classification overwhelming (likelihood ratio ~ (p_f/(1-p_f))^66). SIGNATURE 2 — THE MIXTURE MASS IS A PARAMETER-FREE POINT PREDICTION: agreement cells T12-T15 (lone 55%, 80%, 90%, 65% expert) are followed by BOTH hierarchies at every cascade depth, so they pin p_f without knowing the mixture; the dissociation cells then estimate m_hat = (p_obs - (1-p_f))/(2*p_f - 1). My theory: m_hat = 0 exactly. Competitor: m_hat = 0.761, a number derived entirely from its ambiguity x coherence-gate formulas — a genuinely two-sided test of its softened-gate calibration (center 0.45, width 0.08): m_hat ~ 0 falsifies the gate; m_hat far from 0.76 miscalibrates it. SIGNATURE 3 — COMPETITOR-ISOLATED THREE-WAY CELLS (T7-T10): on T7 (f0: 75% -> A; f1: 55% and f2: 90% -> B) the anti-misbound decider is the 75% expert while the 90% expert, the 55% expert, AND the tally majority (margin -1) all point the other way. My theory, standard validity-respecting TTB, and Tallying ALL say B; ONLY the competitor's 76% majority says A (p(A) ~ 0.61). T8-T10 mirror and vary this (65%-vs-80% decider at T9/T10). If subjects show any validity sensitivity on these cells, the competitor fails where my theory survives; conversely T1-T6/T11 isolate MY anti-validity claim (I follow the 55-65% expert against the 75-90% expert and, on T11, against a 3-vs-1 majority) — the combination yields a full three-way separation (weakest-first vs strongest-first vs misbound-reversal) plus Tallying. SIGNATURE 4 — MARGIN-INVARIANCE EQUALITY BLOCKING INTEGRATION MIMICS: T1 (margin 0), T7 (margin -1), T11 (margin -2) all share decider pair (central f1 vs anti-misbound f0); both one-reason theories predict p(T1) = p(T11) and p(T7) = 1 - p(T1), whereas any margin-integrating strategy (e.g., Tallying) predicts a strictly ordered graded profile across them (0.5 / sigmoid(-beta) / sigmoid(-2beta)) — an internal check that the data belong to the one-reason family before adjudicating between its two members. SIGNATURE 5 — SHARED FLATNESS PROBES: (i) across T12-T15 the lone decider's advertised validity sweeps 55-90% and its cascade depth sweeps 1-5 in both hierarchies; both theories predict a FLAT follow rate — any monotone validity gradient or depth effect falsifies the one-reason core of whichever model is otherwise supported (a deliberate shared-risk probe); (ii) the competitor's tightened regime claims per-subject follow-rate SD ~ 0.02, whereas my parameter ranges span p_f in [0.60, 0.80] — the between-subject SD of follow rates on T12-T15 separately discriminates the two noise regimes. SIGNATURE 6 — PER-SUBJECT BIMODALITY: the competitor predicts a 76/24 split of subject types (anti-misbound vs central), each internally consistent across all 11 dissociation cells; my theory predicts a single central type. T16 (full tie) pins both theories to exactly 0.50, anchoring lapse and side bias; T1/T2, T3/T4, T5/T6, T7/T8, T9/T10 are exact A/B mirrors ruling out side bias. GLOBAL PREDICTED PROFILES: my theory emits p_f (~0.70) toward the WEAKEST-advertised discriminator's option on T1, T3, T5, T8, T10, T11, T12, T14 and (1-p_f) on T2, T4, T6, T7, T9, T13, T15, with 0.50 on T16; the competitor emits ~0.61 toward the anti-misbound decider's option on T2, T4, T6, T7, T9, T11 and ~0.39 on T1, T3, T5, T8, T10 — near-perfect anti-correlation on the 11 dissociation cells with disjoint parameter-robust intervals ([0.20, 0.40] vs [0.58, 0.65] toward the anti-misbound side), agreement on T12-T16. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
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
[0] rationale: The metric is a per-subject normalized one-reason contrast: C = (follow rate toward the weakest-expert-first decider's option on the mirror cells T5/T6, T7/T8, T9/T10) − (follow rate toward the lone discriminating cue on the single-cue cells T12–T15). Key design points: (1) The single-cue cells (exactly one differing feature) are hierarchy-independent — ANY one-reason subject must consult that cue — so they give a clean per-subject estimate of the cascade-follow probability p_f, identical in distribution under both theories. Subtracting it cancels each subject's own p_f (and any lapse/side-bias differences), leaving pure hierarchy information. (2) Under my advocated theory (w = −validity, ascending-validity cascade), the decider on the diagnostic cells and on the single-cue cells is followed with the SAME p_f, so C = 0 exactly per subject; the only variance is binomial (~36 and ~24 trials), giving a tight between-subject variance (~0.015). (3) Under the competing theory the population splits into a central anti-validity component and an anti-misbound (w = −val[::-1]) component (plus per-cue zeta distortion when validities are distinct). On the chosen diagnostic cells the anti-misbound decider (the 75–90% expert) points OPPOSITE to the weakest-first decider, so anti-misbound subjects contribute ~1 − 2·p_f ≈ −0.44 to C, and the zeta distortion drags even central subjects below p_f on the multi-cue diagnostic cells (their scrambled hierarchy sometimes consults the high-validity expert instead). The expected C under the competitor is therefore strongly negative (≈ −0.20 to −0.33 depending on the validity arrangement), while remaining 0 under my theory — a large mean gap with small variance on my side. (4) Crucially, the three diagnostic mirror pairs were selected so that the weakest-first decider's favored option is IDENTICAL under both candidate validity arrangements stated in the design ([0.75,0.9,0.65,0.8,0.85,0.6] and [0.75,0.55,0.90,0.65,0.80,0.55]): on T5/T6 the decider is the 0.60/0.55 expert f5, on T7/T8 it is the 0.65/0.55 expert (f2 or f1), on T9/T10 it is the 0.65 expert (f2 or f3) — always the option whose extra 1s sit on the weak-expert features. So the metric discriminates the theories robustly regardless of which validity vector the simulator actually instantiated. Mirror symmetry (A/B-swapped pairs) cancels side bias, and the metric works both pooled and on single-subject slices (each subject has 36 diagnostic and 24 single-cue trials).
metric_source:
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # ------------------------------------------------------------------
    # Diagnostic mirror-pair cells (T5/T6, T7/T8, T9/T10).
    # For each unordered pair, `side` is the option favored by the
    # WEAKEST-EXPERT-FIRST (ascending-validity) one-reason decider.
    # These three pairs were chosen because the weakest-first decider's
    # favored option is the SAME under either candidate validity
    # arrangement of this experiment, so the metric does not depend on
    # resolving that ambiguity: on every one of these cells the
    # ascending-validity hierarchy lands on the option carrying the 1s of
    # the low-validity experts (f5/f1/f2-region), while the competing
    # theory's anti-misbound hierarchy (or its per-cue zeta distortion of
    # it) systematically lands on the opposite option for its
    # anti-misbound / distorted subjects.
    # ------------------------------------------------------------------
    diag_pairs = [
        ((0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 1, 0)),  # T5/T6: decider = weakest expert f5
        ((0, 1, 1, 0, 0, 0), (1, 0, 0, 0, 0, 0)),  # T7/T8: decider = weak experts f1/f2
        ((1, 1, 1, 1, 0, 1), (1, 1, 0, 0, 1, 1)),  # T9/T10: decider = weak experts f2/f3
    ]

    def as_tuple(x):
        return tuple(int(v) for v in x)

    diss_follow = []    # 1.0 if subject chose the weakest-first decider's option
    single_follow = []  # 1.0 if subject followed the lone discriminating cue

    a_col = data["option_a_ratings"].values
    b_col = data["option_b_ratings"].values
    r_col = data["response"].values

    for a_raw, b_raw, r_raw in zip(a_col, b_col, r_col):
        a = as_tuple(a_raw)
        b = as_tuple(b_raw)
        if len(a) != 6 or len(b) != 6:
            continue
        try:
            r = int(r_raw)
        except Exception:
            continue

        # --- diagnostic (dissociation / distortion) cells ---
        matched = False
        for side, other in diag_pairs:
            if a == side and b == other:
                diss_follow.append(1.0 if r == 0 else 0.0)
                matched = True
                break
            if a == other and b == side:
                diss_follow.append(1.0 if r == 1 else 0.0)
                matched = True
                break
        if matched:
            continue

        # --- single-discriminating-cue cells (T12-T15) ---
        # Exactly one feature differs, so ANY one-reason hierarchy must
        # consult that cue: the follow rate here is a clean, per-subject
        # estimate of the cascade-follow probability p_f, valid under both
        # theories and immune to hierarchy scrambling.
        diff_idx = [j for j in range(6) if a[j] != b[j]]
        if len(diff_idx) == 1:
            j = diff_idx[0]
            lone_side = 0 if a[j] > b[j] else 1
            single_follow.append(1.0 if r == lone_side else 0.0)

    if len(diss_follow) == 0 or len(single_follow) == 0:
        return 0.0

    # Normalized contrast: follow-rate toward the weakest-first decider on
    # the diagnostic cells MINUS the subject's own single-cue follow rate.
    # Under Weakest-Expert-First Frugality this is exactly 0 per subject
    # (both rates estimate the same p_f, which cancels). Under the
    # competitor it is strongly negative: its anti-misbound subjects follow
    # the OPPOSITE option on the dissociation cells (contribution ~
    # 1 - 2*p_f), and its per-cue zeta distortion makes even its central
    # subjects deviate from the ascending-validity decider.
    return float(np.mean(diss_follow) - np.mean(single_follow))
```
outcome: evaluation failed (SyntaxError: invalid syntax (<string>, line 1))

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "```python\ndef metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Directed follow rate toward the WEAKEST-EXPERT-FIRST (ascending-validity)\n    # one-reason decider's option on the three mirror pairs T5/T6, T7/T8, T9/T10.\n    #\n    # These six trials were selected because the ascending-validity decider's\n    # favored option is IDENTICAL under either candidate validity arrangement\n    # of this experiment ([0.75, 0.9, 0.65, 0.8, 0.85, 0.6] or\n    # [0.75, 0.55, 0.90, 0.65, 0.80, 0.55]):\n    #   T5/T6: differing features are f4 and f5; f5 is the weakest of the two\n    #          under both arrangements -> my side is the option carrying the 1\n    #          on f5 (A on T5, B on T6).\n    #   T7/T8: differing features are f0, f1, f2; the weakest differing expert\n    #          is f2 (0.65) or f1 (0.55) -> my side is the B option of T7\n    #          (the side with 1s on the weak experts).\n    #   T9/T10: differing features are f2, f3, f4; the weakest differing expert\n    #          is f2 (0.65) or f3 (0.65) -> my side is again the B option of T9.\n    # On all six trials the competing theory's anti-misbound hierarchy\n    # (w = -validity[::-1]) either lands on the OPPOSITE option (full\n    # dissociation under the tied-validity arrangement) or is heavily\n    # scrambled by its per-cue zeta distortion (all-distinct arrangement,\n    # zeta_scale = sigma_h * ambiguity * gate > 0), dragging its follow rate\n    # well below the one-reason follow probability.\n    #\n    # Expected values: my theory ~0.70 (every subject follows the\n    # ascending-validity decider with probability p_f in [0.54, 0.86],\n    # mean ~0.70, per-subject SD ~0.09). Competing theory ~0.39 (tied-validity\n    # arrangement: 76% anti-misbound subjects follow the OPPOSITE option,\n    # 0.239*p_f + 0.761*(1-p_f)) or ~0.55-0.60 (all-distinct arrangement:\n    # zeta-scrambled hierarchies). Welch t at N=25 is >= 4 in both cases.\n\n    diag_cells = [\n        # (A_ratings, B_ratings, my_side)  my_side: 0 = A, 1 = B\n        ((0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 1, 0), 0),  # T5\n        ((0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 0, 1), 1),  # T6 (mirror)\n        ((1, 0, 0, 0, 0, 0), (0, 1, 1, 0, 0, 0), 1),  # T7\n        ((0, 1, 1, 0, 0, 0), (1, 0, 0, 0, 0, 0), 0),  # T8 (mirror)\n        ((1, 1, 0, 0, 1, 1), (1, 1, 1, 1, 0, 1), 1),  # T9\n        ((1, 1, 1, 1, 0, 1), (1, 1, 0, 0, 1, 1), 0),  # T10 (mirror)\n    ]\n    lookup = {(a, b): s for a, b, s in diag_cells}\n\n    a_col = data[\"option_a_ratings\"].tolist()\n    b_col = data[\"option_b_ratings\"].tolist()\n    r_col = data[\"response\"].tolist()\n\n    hits = []\n    for a_raw, b_raw, r_raw in zip(a_col, b_col, r_col):\n        try:\n            a = tuple([int(v) for v in a_raw])\n            b = tuple([int(v) for v in b_raw])\n            r = int(r_raw)\n        except Exception:\n            continue\n        key = (a, b)\n        if key in lookup:\n            side = lookup[key]\n            if r == side:\n                hits.append(1.0)\n            else:\n                hits.append(0.0)\n\n    if len(hits) == 0:\n        return 0.0\n    return float(np.mean(hits))\n```",
  "rationale": "The metric is the raw directed follow rate toward the weakest-expert-first (ascending-validity) decider's option, pooled over the six trials T5-T10 (3 mirror pairs x 6 repetitions = 36 trials per subject). Cell selection is the crux: these are the only multi-cue cells in the design on which (a) the ascending-validity decider's favored option is invariant across the two validity arrangements stated for this experiment ([0.75,0.9,0.65,0.8,0.85,0.6] in the design header vs [0.75,0.55,0.90,0.65,0.80,0.55] in the rationale), and (b) the competing theory's predictions diverge from mine under EITHER arrangement. On T5/T6 the decider is the weakest expert f5 (0.60 or 0.55); on T7/T8 it is the weak expert f2 (0.65) or f1 (0.55) against the 0.75 expert f0; on T9/T10 it is f2 (0.65) or f3 (0.65) against the 0.80/0.85 experts. Under the tied-validity arrangement, the competitor's anti-misbound hierarchy (w = -val[::-1], adopted by its gate formulas with mass ~0.76) consults f0/f4 first on exactly these cells and therefore points OPPOSITE to my decider, so its predicted follow rate toward my side is 0.239*p_f + 0.761*(1-p_f) ~ 0.39, while mine is p_f ~ 0.70. Under the all-distinct arrangement, the competitor's zeta distortion is switched ON with scale sigma_h*ambiguity*gate ~ 0.28-0.98, which exceeds the w_base spread (~0.3), scrambling its hierarchies and pulling its follow rate toward ~0.55-0.60 on these cells (its anti-misbound component decider on T7 is the 0.75 expert f0, opposite to mine), again well below my ~0.70. Unlike my previously failed contrast metric, I deliberately do NOT normalize by the single-cue cells: the per-subject follow probability p_f spreads by only ~0.055 (variance ~0.003) whereas the normalization cells would inject extra binomial noise (~0.006), so the raw rate has the tighter between-subject variance (SD ~0.09 on my side, dominated by 36-trial binomial noise) while keeping the full mean gap (0.10-0.32 depending on the instantiated validity vector). Mirror pairing (T5/T6, T7/T8, T9/T10 are exact A/B swaps) cancels side bias. Welch's t-test at N=25 gives t >= 4.5 (tied arrangement: means 0.70 vs 0.39, SDs 0.09 vs 0.20; distinct arrangement: means 0.70 vs ~0.57, SDs 0.09 vs ~0.10), comfortably below alpha = 0.01 in both cases. The metric works identically on the pooled DataFrame and on single-subject slices (every subject completes all 96 trials)."
}
```

## Usage

```json
{
  "input_tokens": 9671,
  "output_tokens": 42856,
  "reasoning_tokens": 40958,
  "provider": "Sail Research"
}
```
