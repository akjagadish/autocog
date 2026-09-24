# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended decision-making theories make distinguishable predictions: avoid degenerate pairs where every theory agrees, and prefer pairs that maximally dissociate the candidate theories. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

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


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] The design exploits the core structural difference between Take-The-Best (TTB) and Tallying: TTB's choice is determined solely by the single highest-validity discriminating cue, whereas Tallying integrates the sign of every feature comparison. I use 5 features with a clear validity hierarchy [0.90, 0.80, 0.70, 0.60, 0.55] so the cue cascade is unambiguous. The trial set is built to produce a QUANTITATIVE dissociation, not just a qualitative one. (1) Conflict trials T1-T3: option A wins only on the top cue (validity 0.90) while B wins on progressively fewer lower cues (tally margins -3, -2, -1 for A). TTB predicts p(A) is IDENTICAL across T1, T2, T3 (a constant (1-eps)*sigmoid(beta)+eps/2, since the TTB score is always [1,0]); Tallying predicts a strictly graded p(A) that increases with the tally margin (sigmoid(-3*beta), sigmoid(-2*beta), sigmoid(-beta)). A flat vs. graded profile across these three trial types is the central quantitative signature. (2) Mirror conflict trials T4-T6 test the same dissociation in the opposite direction (TTB->B, Tallying->A), ruling out side-bias explanations. (3) Tally-tie trials T7-T8: the tally is exactly 1-1, so Tallying is forced to 50/50 regardless of beta, while TTB is near-deterministic (top cue decides). These trials alone can falsify Tallying at the aggregate level: any consistent preference is TTB-only. (4) Cascade-depth trials T11-T12: the top cue ties and the SECOND cue (0.80) discriminates against the tally majority — TTB follows cue 2, Tallying follows the 3-vs-2 count. This confirms subjects respect the validity ordering, not just 'any high cue wins'. (5) Agreement trials T9-T10 (both theories favor the same option, margin +1) and a full-tie trial T13 (both predict exactly 50/50) serve as controls: they anchor the lapse rate and verify that neither theory is needed where both agree. Crucially, TTB predicts a single constant choice probability across all five TTB->A trial types (T1, T2, T3, T7, T9), whereas Tallying predicts five distinct probabilities spanning from well below 0.5 to well above 0.5 — model comparison on the response profile across these cells cleanly separates the theories even after fitting beta and epsilon to each. With 13 unique pairs and K=7 repetitions (91 trials), each cell is sampled enough to estimate choice proportions per subject.

[1] This design dissociates Tallying from Take-The-Best (TTB) along two orthogonal quantitative dimensions simultaneously: (i) the tally MARGIN (a_wins - b_wins), which is the ONLY trial-dependent quantity in Tallying's softmax (p(A) = (1-eps)*sigmoid(beta*margin) + eps/2), and (ii) the identity/validity of the first discriminating cue, which is the ONLY trial-dependent quantity in TTB (p(A) = (1-eps)*sigmoid(beta) + eps/2 toward the top-cue winner, constant across ALL discriminating trials). I use 6 features with an unambiguous validity hierarchy [0.95, 0.85, 0.75, 0.65, 0.60, 0.55]. CENTRAL SIGNATURE 1 — the full crossing psychometric function (T1-T5 and mirror T6-T10): the top cue (0.95) always favors one option while the tally margin sweeps across its full range (-4, -2, 0, +2, +4). TTB predicts a FLAT p(A) = (1-eps)*sigmoid(beta)+eps/2 across all five trials in each direction; Tallying predicts a strictly monotone graded curve that passes through EXACTLY 0.5 at margin 0 (T3/T8) — a knife-edge prediction independent of beta that any consistent preference falsifies — and reverses sign at negative margins (T1/T2: TTB says A with high confidence, Tallying says B; T6/T7: the mirror sign reversal, ruling out side bias). T1 vs T10 is especially clean: identical tally margins (-4) but opposite TTB winners, so Tallying predicts identical p(A) on both while TTB predicts opposite extremes. CENTRAL SIGNATURE 2 — cascade-depth invariance (T11-T15): the top cue TIES and the TTB decision comes from cue 2 (0.85), cue 3 (0.75), or even cue 5 (0.60, T14). TTB predicts the SAME constant p(A) on T1, T2, T3, T11, T12, T13, T14 (all TTB->A regardless of which cue decided or how deep the cascade ran); Tallying predicts six distinct values spanning sigmoid(-2beta) to sigmoid(+1beta), including exact 0.5 on T3/T13/T14. T15 is a depth-2 sign reversal (cue 2 favors B, tally margin +2 favors A). T14 probes the bottom of the hierarchy: when cues 1-4 all tie, TTB commits to the single 0.60-validity cue while Tallying sits at exactly 50/50 — if subjects treat a lone low-validity cue as uninformative, TTB fails where Tallying succeeds. T16 (full tie) is the shared 50/50 control anchoring lapse rate and side bias. With 16 unique pairs and K=6 (96 trials exactly), each cell yields 6 repetitions per subject for stable choice-proportion estimates. Model comparison on the full 16-cell response profile — flat vs. graded across margins, constant vs. varied across cascade depths, and exact-0.5 vs. deterministic on margin-0 trials — cleanly separates the theories even after fitting beta and epsilon to each.

[2] This design dissociates Reversed-Hierarchy Frugality (misbinding TTB) from Tallying by exploiting their two orthogonal trial-dependent quantities: (i) Tallying's choice depends ONLY on the tally margin m = a_wins - b_wins, giving p(A) = (1-eps)*sigmoid(beta*m) + eps/2, a strictly graded psychometric curve; (ii) misbinding TTB's choice depends ONLY on which cue first discriminates when the cascade runs in ASCENDING true validity (kappa=1 binds the highest subjective validity to the LAST feature), giving a CONSTANT p(winner) = (1-eps)*sigmoid(beta) + eps/2 ~ 0.60 on every discriminating trial. I use 5 features with validities [0.90, 0.80, 0.70, 0.60, 0.55], so the reversed cascade consults the 0.55 expert first, then 0.60, 0.70, 0.80, 0.90. CENTRAL SIGNATURE 1 - margin sweep with the lowest-validity cue as fixed decider (trials 1-4 and 5-8): the 0.55 expert always favors one option while the tally margin sweeps across -3, -2, -1, 0 (and mirrored +3, +2, +1, 0). Misbinding TTB predicts a FLAT p ~ 0.60 toward the lowest-cue winner on all eight trials; Tallying predicts a strictly monotone graded curve that passes through EXACTLY 0.5 at margin 0 (trials 4, 8) - a knife-edge prediction independent of beta - and reverses sign at negative margins (trial 1: Tallying says B strongly, misbinding TTB says A). CENTRAL SIGNATURE 2 - matched-margin opposite-decider pairs, a parameter-free falsification for Tallying: pairs (2, 10), (4, 8), (6, 9), (7, 14) have IDENTICAL tally margins (-2, 0, +2, +1) but opposite lowest-cue winners. Tallying predicts IDENTICAL p(A) within each pair for ANY (beta, eps); misbinding TTB predicts a ~0.60 vs ~0.40 split within each pair. Trial 14 is especially diagnostic: it is the deepest cascade (only the 0.90 expert discriminates, all lower cues tie), and misbinding TTB predicts the SAME constant p there as on trial 1 (where the 0.55 expert decides) - cascade-depth invariance - while Tallying treats trials 7 and 14 identically (both margin +1) though misbinding TTB gives opposite choices. CENTRAL SIGNATURE 3 - reversed-hierarchy depth probes at margin 0 (trials 11, 12, 13): the decider is the 0.60, 0.70, or 0.80 expert while the tally is exactly 1-1, so Tallying is pinned to exactly 0.50 regardless of parameters while misbinding TTB stays at ~0.60/~0.40. Trial 15 is a depth-3 conflict (0.70 expert favors A, tally margin -1 favors B). Trial 5 additionally separates misbinding TTB from STANDARD TTB: the top-validity 0.90 expert and the tally both favor A, but the reversed cascade follows the 0.55 expert to B - if subjects choose A here, the misbinding claim fails. Trial 16 is a full-tie control anchoring lapse and side bias (both theories: exactly 0.5). The overall profile is non-monotone in margin (margin +3 -> p(A) ~ 0.40; margin -3 -> p(A) ~ 0.60), which no monotone psychometric function of margin - i.e., no parameterization of Tallying - can reproduce, whereas misbinding TTB predicts one constant across all 15 discriminating trials with variation only in direction. 16 unique pairs, K = 6 (96 trials exactly), give 6 repetitions per cell for stable per-subject choice proportions.

[3] This design dissociates Tallying from Reversed-Hierarchy Frugality (misbinding TTB, kappa=1) along three novel quantitative axes that the previous designs did not exploit. SETUP: 6 features with validities [0.90, 0.80, 0.70, 0.65, 0.60, 0.55]. Because validities descend across positions, the misbound hierarchy (positional reversal) equals an ascending-validity cascade: the reversed TTB model consults f5 (0.55) first, then f4, f3, f2, f1, f0 (0.90) last. AXIS 1 — THE CONFIDENCE CEILING (T3, T4, T12): the competing theory's beta is locked in [0.45, 0.55] with epsilon in [0.11, 0.15], so its predicted choice probability on ANY discriminating trial is capped at ~(0.87)*sigmoid(0.55)+0.075 ≈ 0.62. T3/T4 are unanimous trials (margin ±6) and T12 is a sparse trial where only the two highest-validity experts discriminate, both favoring A. Every sensible strategy — including Tallying with a moderate beta (p(A) = (1-eps)*sigmoid(2*beta)+eps/2, up to ~0.93) — predicts near-ceiling consistency here, but the misbinding model is mathematically FORBIDDEN from exceeding ~0.62. Human consistency above ~65% on these cells falsifies the competing theory at its parameter boundary while Tallying accommodates it. AXIS 2 — EXTREME-MARGIN CONFLICT WITH ORDINAL INVERSION (T1, T2, T14): on T1, A wins the four highest-validity experts (margin +4) but the lowest-validity expert (f5) favors B. Tallying predicts p(A) = (1-eps)*sigmoid(4*beta)+eps/2 ≥ 0.59 (well above 0.5 for any beta > 0.1); reversed TTB follows f5 to B, predicting p(A) ≈ 0.39. Crucially, compare T1 to T5 (margin 0, decider f5 favors A): Tallying predicts p(T1) >> p(T5) = exactly 0.50, while reversed TTB predicts p(T1) ≈ 0.39 < p(T5) ≈ 0.61 — a complete ORDINAL INVERSION of the two-cell response profile that no parameter setting of either model can reconcile with the other's prediction. T2 and T14 mirror this inversion, ruling out side bias. AXIS 3 — KNIFE-EDGE AND ISO-MARGIN FAMILIES (T5–T11, T13, T14, T16): five margin-0 trials (T5–T8, T16) pin Tallying to EXACTLY 0.50 regardless of (beta, epsilon) — a parameter-free prediction — while reversed TTB predicts 0.61/0.39 splits on every one, at shallow (f5), middle (f3), and deep (f1) cascade depths, i.e., depth-invariant. The iso-margin family {T9, T10, T11, T12} all have margin +2: Tallying predicts IDENTICAL p(A) on all four for any parameters; reversed TTB predicts 0.61, 0.39, 0.39, 0.61 (decider flips between f5, f3, f1). The mirror family {T13, T14} does the same at margin -2, where Tallying prefers B (~0.30 for A) but reversed TTB prefers A on T14 (0.61) — opposite sides of 0.5. T15 (full tie) is the shared 50/50 control anchoring lapse and side bias. GLOBALLY: reversed TTB's entire 16-cell profile is a two-level step function (≈0.61 toward the lowest-validity discriminator, ≈0.39 against, 0.50 on the full tie), whereas Tallying predicts a monotone psychometric function of margin with exact 0.5 at m=0 and identical values within margin levels — two cross-structured profiles that model comparison on the full response profile separates cleanly even after fitting beta and epsilon to each. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject.

[4] GOAL: Quantitatively dissociate Reversed-Hierarchy Frugality (misbinding TTB, kappa=1) from Tallying, while adding a novel axis absent from prior designs: a SCRAMBLED (non-monotone) validity arrangement that makes the misbound hierarchy a non-trivial permutation, enabling a 'subjective-hierarchy ladder' and a 'decider-validity invariance' test. SETUP: 6 features, validities [0.70, 0.90, 0.60, 0.95, 0.75, 0.55] (positions f0..f5). The misbound subjective weights are w = val reversed = [0.55, 0.75, 0.95, 0.60, 0.90, 0.70], so the misbinding cascade consults f2 -> f4 -> f1 -> f5 -> f3 -> f0. Crucially, the misbound model's FIRST cue is Expert 3 (advertised only 60%), while the truly best expert (95%, f3) is consulted 5th. Tallying ignores validities entirely, so its predictions depend only on the tally margin m = a_wins - b_wins: p(A) = (1-eps)*sigmoid(beta*m) + eps/2. The misbinding model's predictions depend only on the decider: constant p = (1-eps)*sigmoid(beta)+eps/2 in [0.63, 0.78] toward the first cue in the order f2,f4,f1,f5,f3,f0 that discriminates. SIGNATURE 1 - margin sweep with fixed decider (T1-T6): Expert 3 (f2) always favors A while the margin sweeps -4, -2, 0, +1, +3 (T1, T3, T4, T5, T6), plus the A/B mirror at margin +4 with decider->B (T2). Misbinding TTB predicts a FLAT p(A) in [0.63, 0.78] on all six; Tallying predicts a strictly graded psychometric curve that passes through EXACTLY 0.5 at T4 (margin 0, parameter-free knife-edge) and stays below 0.5 at negative margins (T1: Tallying <= 0.45 for ANY beta in [0.1, 20], eps in [0, 0.5], while misbinding TTB >= 0.63 - non-overlapping prediction intervals on this single cell). SIGNATURE 2 - extreme-margin matched pair (T2 vs T15): both have margin +4, but the decider is f2->B on T2 and f2->A on T15. Tallying predicts IDENTICAL p(A) on the two cells for ANY (beta, eps) - a parameter-free prediction - while misbinding TTB predicts ~0.22-0.37 on T2 vs ~0.63-0.78 on T15, opposite sides of 0.5. SIGNATURE 3 - the subjective-hierarchy ladder at margin 0 (T4, T7-T11): six knife-edge cells, each with exactly one cue favoring each option, so Tallying is pinned to EXACTLY 0.50 on every rung; any consistent preference on any rung falsifies Tallying regardless of parameters. The rungs pit adjacent cues of the MISBOUND order against each other (f2 vs f4; f2 vs f1; f4 vs f1; f5 vs f3 and its mirror; f3 vs f0), so subjects' choices trace out their subjective priority order. Because the arrangement is scrambled, five of six rungs also place standard validity-respecting TTB on the OPPOSITE side (e.g., T9/T10: the 55% expert f5 vs the 95% expert f3 - misbinding follows the 55% expert, standard TTB the 95% expert), making the ladder a simultaneous three-way separation: misbinding TTB ~0.7 toward its decider, Tallying exactly 0.5, standard TTB often the reverse. SIGNATURE 4 - decider-validity invariance at fixed margin (T5, T12, T13, T14): four margin-+1 cells whose misbound deciders are advertised at 60%, 75%, 90%, and 55% respectively. Misbinding TTB predicts the SAME p(A) on all four (the follow rate cannot depend on the decider's advertised accuracy); any monotone effect of decider validity falsifies the kappa=1 misbinding claim. These are agreement cells for Tallying (identical margin), but they pin the flat level as decider-independent, and the within-decider contrast T4 vs T5 (margin 0 vs +1, same decider) tests Tallying's strict margin effect where misbinding TTB predicts none. SIGNATURE 5 - consistency ceiling (T15): a margin-+4 agreement cell where only the 95% expert dissents. Misbinding TTB is capped at ~0.78 by its parameter ranges (beta <= 1.2, eps >= 0.08); human consistency above ~0.80 here falsifies the advocated theory at its boundary while Tallying accommodates it. CONTROLS: T16 is a full tie (both theories: exactly 0.5, anchoring lapse and side bias); T1/T2 are exact A/B mirrors; T10 and T8 provide decider->B cells at margin 0 to rule out side bias in the ladder. GLOBAL PREDICTED PROFILES: misbinding TTB produces a two-level step function over the 16 cells (~0.7 toward the misbound decider on all 15 discriminating trials, exactly 0.5 on T16), invariant to margin, cascade depth, and decider validity; Tallying produces a monotone psychometric function of margin with exact 0.5 on the six margin-0 cells and identical values within margin levels (T2 = T15; T5 = T12 = T13 = T14). The profiles are structurally disjoint (e.g., misbinding says p(T1) = p(T6); Tallying says p(T1) < 0.45 < p(T6)), so model comparison on the full 16-cell response profile separates the theories even after fitting beta and epsilon to each. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable choice proportions.

[5] NOVEL AXIS - THE CONCORDANCE LADDER: every prior design dissociated Tallying from TTB-family models using CONFLICT trials (some cues favor each side). This design's centerpiece is a family of zero-conflict trials where every discriminating cue favors the same option and only the NUMBER of supporting cues varies (ties fill the rest). Tallying's output depends only on the margin m = a_wins - b_wins, so it predicts a strictly GRADED psychometric curve p(A) = (1-eps)*sigmoid(beta*m) + eps/2 even when no cue disagrees — climbing from ~0.62 at m=+1 to ~0.97 at m=+6 for moderate beta. Reversed-Hierarchy Frugality is a one-reason model: on every concordance trial the first discriminating cue in its misbound cascade decides, so it predicts ONE CONSTANT p(A) = (1-eps)*sigmoid(beta)+eps/2 across the entire ladder — and, critically, its fixed noise regime (beta in [0.8,1.2], eps in [0.08,0.14]) mathematically CAPS its follow probability at ~0.75. Human accuracy above ~0.80 on high-rung concordance cells falsifies the competing theory at its parameter boundary while Tallying accommodates it. SETUP: 6 features with ASCENDING validities [0.55, 0.60, 0.65, 0.75, 0.85, 0.95] (f0..f5) — an arrangement no prior design used. The misbound hierarchy w = val reversed = [0.95, ..., 0.55] makes the reversed cascade consult f0 first: the FIRST-listed, LOWEST-validity (55%) expert is the misbinding model's most trusted cue, and the 95% expert is consulted last. SIGNATURE 1 - CONCORDANCE LADDER (T1-T5): A weakly dominates B, margins +1, +2, +4, +5, +6; the misbound decider sweeps across f5 (95%), f4 (85%), f2 (65%), f1 (60%), f0 (55%). Misbinding TTB: flat ~0.70 on all five rungs (jointly a cascade-depth- AND decider-validity-invariance claim); Tallying: strictly increasing, exceeding the misbinding cap on the top rungs. SIGNATURE 2 - SINGLE-CELL NON-OVERLAP CONFLICT SWEEP (T6-T8, T9-T11): the 55% expert f0 always decides for A (T6: m=-4, T7: m=-2, T8: m=0) or always for B (T9: m=+4, T10: m=+2, T11: m=0). On T6/T7 Tallying is mathematically BELOW 0.50 for ANY parameters (negative margin), while misbinding is at or above 0.663 — the prediction intervals are disjoint, so each cell alone falsifies one theory. On T9/T10 the inversion is complete (Tallying > 0.50, misbinding <= 0.337). T6 vs T9 are exact A/B mirrors ruling out side bias. SIGNATURE 3 - KNIFE-EDGE FAMILY (T8, T11, T13, T15): four margin-0 cells (1-1, 1-1, 2-2, 3-3) where Tallying is pinned to EXACTLY 0.50 independent of (beta, epsilon) — any consistent preference on any of these cells falsifies Tallying — while misbinding predicts 0.70/0.30 splits (decider f0) at every tally composition, i.e., invariance to how many cues sit on each side. SIGNATURE 4 - ISO-MARGIN OPPOSITE-DECIDER TRIPLETS (parameter-free for Tallying): {T2, T10, T12} all have m=+2 — Tallying predicts IDENTICAL p(A) on all three for any parameters; misbinding predicts 0.70 (T2, decider f4->A), 0.70 (T12, decider f0->A), 0.30 (T10, decider f0->B). {T3, T9} both m=+4: Tallying identical; misbinding 0.70 vs 0.30. {T1, T14} both m=+1 with deciders at opposite ends of the subjective hierarchy (f5, 95% vs f0, 55%): BOTH theories predict identical p(A) here — a shared control that pins misbinding's flat level and tests its claim that the decider's advertised accuracy is irrelevant. T16 is the full-tie control (both theories: exactly 0.50, anchoring lapse and side bias). GLOBAL PROFILES: misbinding TTB emits a two-level step function over the 16 cells (~0.70 on the eleven decider->A cells including ALL ladder rungs and ALL negative-margin conflicts; ~0.30 on the four decider->B cells; 0.50 on the tie), invariant to margin, tally composition, cascade depth, and decider validity. Tallying emits a monotone psychometric function of margin with exact 0.50 on the four margin-0 cells, exact equality within margin levels (T2=T10=T12; T3=T9; T1=T14), and values above the misbinding ceiling on T4/T5. The profiles are structurally disjoint — e.g., misbinding says p(T1)=p(T5)=p(T6); Tallying says p(T6)<0.50<p(T1)<p(T5) — so model comparison on the full 16-cell response profile separates the theories even after fitting beta and epsilon to each. With 16 unique pairs and K=6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

[6] GOAL: Quantitatively dissociate Reversed-Hierarchy Frugality (misbinding TTB, kappa=1) from Tallying via a validity arrangement whose MISBOUND hierarchy consults the three WEAKEST experts before any of the three strongest — a 'weak-expert dictatorship' schedule no prior design used. SETUP: 6 features, validities [0.60, 0.95, 0.55, 0.85, 0.65, 0.90] (f0..f5). The misbound weights w = val reversed = [0.90, 0.65, 0.85, 0.55, 0.95, 0.60], so the misbinding cascade consults f4 (advertised 65%) -> f0 (60%) -> f2 (55%) -> f1 (95%) -> f5 (90%) -> f3 (85%): the three lowest-validity experts occupy the top three rungs of the subjective hierarchy, and the 95% expert is not consulted until depth 4. Tallying ignores validities entirely: p(A) = (1-eps)*sigmoid(beta*m) + eps/2, depending only on the tally margin m. The misbinding model emits a constant p(decider winner) = (1-eps)*sigmoid(beta)+eps/2, which over its parameter ranges (beta in [0.6,1.2], eps in [0.08,0.18]) lies in [0.62, 0.80] — a hard ceiling. SIGNATURE 1 — WEAK-EXPERT DICTATORSHIP SWEEP (T1-T5): the misbound top cue f4 (a 65% expert) always decides for A while the margin sweeps -3, -1, 0, +1, +3. Misbinding TTB predicts a FLAT p(A) in [0.62, 0.80] on all five; Tallying predicts a strictly graded monotone curve passing through EXACTLY 0.5 at T3 (m=0, parameter-free knife-edge). Single-cell non-overlap at T1 (m=-3): Tallying is mathematically <= 0.463 for ANY (beta, eps) in its ranges (max at beta=0.1, eps=0.5), while misbinding is >= 0.62 — disjoint prediction intervals. T6-T8 are the A/B mirrors (decider f4 -> B at m=+3, 0, -3): at T6 the four STRONGEST experts (95%, 90%, 85%, 60%) all favor A with margin +3, yet misbinding follows the lone 65% expert to B (p(A) in [0.20, 0.38]) while Tallying is >= 0.537 — again disjoint. T5 provides the ceiling test: unanimous-ish margin +3 where Tallying can reach ~0.97 but misbinding is capped at 0.80; human consistency above ~0.82 falsifies the advocated theory at its parameter boundary. SIGNATURE 2 — ISO-MARGIN OPPOSITE-DECIDER PAIRS (parameter-free for Tallying): (T4, T12) at m=+1, (T3, T7) at m=0, (T5, T6) at m=+3 — identical margins, opposite f4 deciders. Tallying predicts IDENTICAL p(A) within each pair for ANY parameters; misbinding predicts a ~0.70 vs ~0.30 split within each pair, straddling 0.5. SIGNATURE 3 — KNIFE-EDGE LADDER AT m=0 WITH VARYING CASCADE DEPTH (T3, T7, T9, T10, T11): five margin-0 cells pinning Tallying to EXACTLY 0.50 regardless of (beta, epsilon) — any consistent preference on any rung falsifies Tallying — while the misbound decider sits at depth 1 (f4, T3/T7), depth 2 (f0, T9), depth 3 (f2, T10), and depth 4 (f1, T11), in both directions. Misbinding predicts the same ~0.70/0.30 split at every depth: depth invariance. SIGNATURE 4 — DECIDER-VALIDITY INVARIANCE AT m=+1 (T4, T13, T14, T15): four margin-+1 cells whose misbound deciders are advertised at 65% (f4), 60% (f0), 55% (f2), and 95% (f1). Misbinding predicts the SAME follow rate on all four — the follow probability cannot depend on the decider's advertised accuracy; any monotone effect of decider validity (e.g., subjects following the 95% expert in T15 more than the 55% expert in T14) falsifies the kappa=1 misbinding claim. These are agreement cells for Tallying (identical margin), so they isolate the misbinding model's internal structure. GLOBAL PROFILES: misbinding TTB emits a two-level step function over the 16 cells (~[0.62,0.80] toward the misbound decider on all 15 discriminating trials, exactly 0.5 on the full tie), invariant to margin, cascade depth, and decider validity; Tallying emits a monotone psychometric function of margin with exact 0.5 on all five margin-0 cells and exact equality within margin levels (T4=T12=T13=T14=T15; T5=T6; T1=T8). The profiles are structurally disjoint — e.g., misbinding says p(T1)=p(T5) (both decider f4->A) while Tallying says p(T1) <= 0.463 < 0.537 <= p(T5); misbinding says p(T3)=p(T4) while Tallying says p(T3)=0.5 < p(T4) — so model comparison on the full 16-cell response profile separates the theories even after fitting beta and epsilon to each. T16 (full tie) anchors lapse rate and side bias. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

[7] GOAL: Quantitatively dissociate Tallying (advocated) from Reversed-Hierarchy Frugality (misbinding TTB, kappa=1) via a design organized around a NOVEL structural centerpiece: a balanced, orthogonal 2x5 FACTORIAL GRID crossing tally margin with misbound-decider direction, plus a 'sandwich-scrambled' validity arrangement whose misbound hierarchy is non-monotone in advertised accuracy. SETUP: 7 features (no prior design used more than 6, giving a wider margin range up to +/-5) with validities [0.60, 0.90, 0.55, 0.80, 0.65, 0.75, 0.95] (f0..f6). The misbound weights w = val reversed = [0.95, 0.75, 0.65, 0.80, 0.55, 0.90, 0.60], so the misbinding cascade consults f0 -> f5 -> f3 -> f1 -> f2 -> f6 -> f4, i.e., advertised accuracies along the subjective hierarchy run 60% -> 75% -> 80% -> 90% -> 55% -> 95% -> 65%: a NON-MONOTONE sandwich in which the first-listed 60% expert is trusted most, the 95% expert is consulted SIXTH (next to last), and the 55% expert sits above it at depth 5. Tallying ignores validities entirely: p(A) = (1-eps)*sigmoid(beta*m) + eps/2, depending only on margin m. Misbinding TTB emits a constant follow probability p = (1-eps)*sigmoid(beta)+eps/2, which over its parameter ranges (beta in [0.6,1.2], eps in [0.08,0.18]) lies in [0.62, 0.75] toward the first discriminating cue in the order f0,f5,f3,f1,f2,f6,f4. SIGNATURE 1 - THE ORTHOGONAL FACTORIAL GRID (T1-T10): the misbound TOP cue f0 (advertised only 60%) always decides, toward A in T1-T5 and toward B in T6-T10 (exact A/B mirrors), while the margin sweeps -3, -1, 0, +1, +3. This is a fully balanced 2 (decider direction) x 5 (margin) design, and the two theories predict TRANSPOSED main-effect structures in the marginal means: Tallying predicts a NULL decider-row effect (both row means exactly 0.5, since margins are symmetric within each row) with a strictly graded margin-column effect passing through EXACTLY 0.5 at m=0; misbinding TTB predicts a strong decider-row effect (~0.70 vs ~0.30) with a NULL margin-column effect (every column mean exactly 0.5). A 2x5 ANOVA or model comparison on the grid cleanly awards the data to one theory: any significant margin effect with null decider effect falsifies misbinding; any significant decider effect with null margin effect falsifies Tallying. SIGNATURE 2 - SINGLE-CELL DISJOINT PREDICTION INTERVALS AT EXTREME MARGINS (T14, T15): on T14, ALL six other experts - including the 95% expert - favor A (margin +5) and only the first-listed 60% expert favors B. Misbinding follows the 60% expert to B, predicting p(A) <= 0.38; Tallying predicts p(A) = (1-eps)*sigmoid(5*beta)+eps/2 >= 0.56 even at its WORST parameters (beta=0.1, eps=0.5) - the intervals are disjoint, so this one cell alone falsifies one theory. T15 is the exact mirror (margin -5, lone 60% expert favors A): Tallying <= 0.44, misbinding >= 0.62. These cells also separate misbinding from STANDARD TTB (which follows the 95% expert to A on T14), making the design adversarial to the whole TTB family. SIGNATURE 3 - PARAMETER-FREE KNIFE EDGES WITH COMPOSITION CONTROL (T3, T8, T11): three margin-0 cells - a 2-2 tally with decider f0->A (T3), its mirror with decider f0->B (T8), and a sparse 1-1 tally with decider f0->A (T11). Tallying is pinned to EXACTLY 0.50 on all three regardless of (beta, eps) and regardless of tally composition; misbinding predicts ~0.70/~0.30 splits on all three, invariant to whether 2 or 6 cues sit on each side. SIGNATURE 4 - DECIDER-VALIDITY AND DEPTH INVARIANCE AT FIXED MARGIN (T2, T12, T13): three margin-+1 cells whose misbound deciders are the 60% first-listed expert at depth 1 (T2), the 90% expert at depth 4 (T12), and the 95% expert at depth 6 (T13, where it is the ONLY discriminating cue). Misbinding predicts the SAME follow rate on all three - the follow probability cannot depend on the decider's advertised accuracy or cascade depth; any monotone validity gradient (e.g., following the lone 95% expert more than the 60% expert) falsifies the kappa=1 misbinding claim. Tallying also predicts identical values here (same margin), so these cells isolate the competitor's internal structure. GLOBAL PROFILES: misbinding TTB emits a two-level step function over the 16 cells (~[0.62,0.75] toward the misbound decider on all 15 discriminating trials, exactly 0.5 on the full tie), invariant to margin, tally composition, cascade depth, and decider validity; Tallying emits a monotone psychometric function of margin with exact 0.5 on the three margin-0 cells, exact equality within margin levels (T1=T10-margin... specifically T1=T10? No: T1 (m=+3, dec->A) = T10 (m=+3, dec->B) in p(A); T2=T9=T12=T13; T5=T6; T14 highest, T15 lowest), and values far outside the misbinding cap on T14/T15. The profiles are structurally disjoint - e.g., misbinding says p(T15)=p(T1) (both decider->A) while Tallying says p(T15) <= 0.44 < 0.56 <= p(T1); misbinding says p(T3)=p(T2) while Tallying says p(T3)=0.5 < p(T2) - so model comparison on the full 16-cell response profile separates the theories even after fitting beta and epsilon to each. T16 (full tie) anchors lapse rate and side bias (both theories: exactly 0.5). With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

[8] GOAL: Quantitatively dissociate Reversed-Hierarchy Frugality (advocated; positional misbinding, w = val[::-1]) from Weakest-Expert-First Frugality (competing; value-based anti-validity sort, ascending validity). These two one-reason models are behaviorally IDENTICAL whenever the validity list is monotone in position (both then consult cues in ascending validity), so the design's foundation is a SCRAMBLED validity arrangement that makes the two hierarchies genuinely different permutations — the one structural axis that separates them. SETUP: 6 features with all-distinct validities [0.90, 0.55, 0.70, 0.60, 0.80, 0.95] (f0..f5). Distinctness is deliberate: it makes the competing model's free tie_break parameter irrelevant, stripping it of extra flexibility. The two hierarchies are: ADVOCATED (w = val reversed = [0.95, 0.80, 0.60, 0.70, 0.55, 0.90], descending): f0 -> f5 -> f1 -> f3 -> f2 -> f4, i.e., the misbinding model's most trusted cue is the FIRST-listed 90% expert and its second is the LAST-listed 95% expert. COMPETING (ascending validity): f1 (55%) -> f3 (60%) -> f2 (70%) -> f4 (80%) -> f0 (90%) -> f5 (95%), i.e., the weakest expert first and the two strongest experts consulted LAST. The hierarchies disagree on 12 of 15 cue pairs; crucially, whenever f0 or f5 discriminates against any of f1..f4, the advocated model follows the 90%/95% (STRONG) expert while the competing model follows the WEAKER discriminator — a head-to-head 'strong-expert vs weak-expert dictatorship' test of the two anti-validity claims. SIGNATURE 1 — STRONG-VS-WEAK DISSOCIATION LADDER (T1-T13): on T1-T8 the first-listed 90% expert (f0, the advocated model's top cue, consulted 5th by the competitor) favors one option while a single weaker expert (55%, 70%, 60%, or 80%) favors the other; on T9-T13 the last-listed 95% expert (f5, the advocated model's second cue, consulted LAST by the competitor) plays the same role. Advocated predicts p(A) = (1-eps)*sigmoid(beta)+eps/2 in [0.62, 0.75] toward the strong expert on every rung; competing predicts the same quantity toward the weak expert, i.e., p(A) in [0.20, 0.40]. THE PREDICTION INTERVALS ARE DISJOINT ON EVERY ONE OF THE 13 DISSOCIATING CELLS: any cell with p(A) > 0.55 falsifies the weak-expert-first theory; any cell with p(A) < 0.45 falsifies the misbinding theory. T1-T12 are exact A/B mirror pairs, ruling out side bias. SIGNATURE 2 — DECIDER-VALIDITY GRADIENT (parameter-free direction test): across T1, T3, T5, T7, T9, T11, T13 the advocated decider is always the highest-validity discriminator (90% or 95%) while the competing decider sweeps 55-80%. If subjects' choices track the STRONG expert's direction, the competing theory's core claim (subjective weights = -validity) is falsified regardless of (beta, eps, tie_break); if they track the WEAK expert's direction, the kappa=1 positional-misbinding claim is falsified. Both models additionally predict follow-rate INVARIANCE across which weak/strong expert decides (flat follow rate across rungs) — a shared internal check. SIGNATURE 3 — COALITION CONFLICT (T14): the 90% expert alone favors A while the 55%, 60%, and 70% experts favor B (tally margin -2). Advocated follows f0 -> A; competing follows f1 -> B; Tallying also says B. This cell simultaneously dissociates the advocated model from the competing theory AND from any margin-integrating strategy, and tests the shared one-reason claim p(T14) = p(T1) (same deciders, different margin and cascade context). SIGNATURE 4 — CONTROLS: T15 (only the 60% expert discriminates) is an agreement cell where both theories follow the same decider at different cascade depths (advocated depth 4, competing depth 2) — it anchors the follow rate and tests both models' depth-invariance claim; any difference between p(T15) and the dissociation-cell follow rates falsifies the flat-follow assumption of whichever model mispredicts it. T16 (full tie) pins both theories to exactly 0.50, anchoring lapse rate and side bias. GLOBAL PROFILES: the advocated model emits p(A) ≈ [0.62, 0.75] on T1,T3,T5,T7,T9,T11,T13,T14,T15 and ≈ [0.25, 0.38] on T2,T4,T6,T8,T10,T12, with 0.50 on T16; the competing model emits the EXACT TRANSPOSE (near-perfect anti-correlation on 14 of 16 cells, agreement only on T15 and T16). Because the intervals are disjoint cell-by-cell, model comparison on the full 16-cell response profile separates the theories even after fitting beta, epsilon, and tie_break to each — and even the AGGREGATE choice rate across the 12 mirror cells (advocated: strong-expert direction ~70%; competing: weak-expert direction ~70%) is diagnostic. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

[9] GOAL: Quantitatively dissociate Weakest-Expert-First Frugality (advocated; value-based ascending-validity cascade) from Reversed-Hierarchy Frugality (competing; positional misbinding, w = val[::-1]). The two theories are behaviorally IDENTICAL whenever the validity list is monotone in position, so the design rests on a scrambled arrangement that turns them into two genuinely different cue permutations — but with a novel twist relative to design [8]: here BOTH hierarchies are 'weak-expert dictatorships', so the design tests the MECHANISM of the anti-validity claim (value-sort vs positional misbinding) without the confound that subjects simply follow strong experts. SETUP: 6 features, all-distinct validities [0.55, 0.90, 0.65, 0.95, 0.60, 0.80] (f0..f5). Distinctness strips the advocated model's free tie_break parameter of any role. ADVOCATED hierarchy (ascending validity): f0 (55%) -> f4 (60%) -> f2 (65%) -> f5 (80%) -> f1 (90%) -> f3 (95%). COMPETING hierarchy (w = val[::-1] = [0.80, 0.60, 0.95, 0.65, 0.90, 0.55], descending): f2 (65%) -> f4 (60%) -> f0 (55%) -> f3 (95%) -> f1 (90%) -> f5 (80%). Note the competing model's TOP cue is the 65% expert f2 and the advocated model's top cue is the 55% expert f0 — both models trust weak experts first, but DIFFERENT weak experts, and both consult the 95% expert late. The two hierarchies disagree on exactly the cue pairs (f0,f2), (f0,f4), (f2,f4), (f1,f3), (f1,f5), (f3,f5); f4 and f1 are 'fixed points' sitting at the SAME cascade depth in both hierarchies (depth 2 and depth 5 respectively). SIGNATURE 1 — THE PURE PERMUTATION DISSOCIATION GRID AT THE TALLY KNIFE-EDGE (T1-T12): twelve cells, one per dissociating cue pair per direction, each with EXACTLY two discriminating cues in opposition and all others tied (tally margin exactly 0). On every one of these cells the advocated model follows its earlier cue and the competing model follows the OPPOSITE cue, so their predicted intervals are DISJOINT CELL-BY-CELL: advocated p(A) in [0.54, 0.80] toward its decider vs competing p(A) in [0.20, 0.43] — a direction flip on all twelve cells (T1-T12 come in six exact A/B mirror pairs, ruling out side bias). Because every cell sits at margin 0, ANY margin-integrating strategy (e.g. Tallying) is pinned to EXACTLY 0.50 on all twelve — a parameter-free knife-edge — so any consistent preference on any cell simultaneously falsifies the entire integration family while the two one-reason theories are the only candidates predicting systematic deviation: a three-way separation no prior design combined with the permutation axis. SIGNATURE 2 — WEAK-VS-WEAK DICTATORSHIP (novel axis vs design [8], which pitted strong against weak experts): the advocated decider sweeps 55% (f0: T1-T4), 60% (f4: T5-T6), 80% (f5: T9-T12), 90% (f1: T7-T8); the competing decider sweeps 65% (f2: T1,T2,T5,T6), 95% (f3: T7,T8,T11,T12), 90% (f1: T9,T10). If subjects follow the 95% expert on T7/T11 or the 90% expert on T9, BOTH anti-validity theories fail — a shared falsification probe that design [8] could not mount, since there the advocated model itself followed the strong experts. SIGNATURE 3 — CASCADE-DEPTH INVARIANCE WITHIN EACH MODEL: the deciding rung moves from depth 1 (T1-T4 for advocated; T1,T2,T5,T6 for competing) through depths 4-5 (T7-T12), and each model predicts the SAME follow probability p = (1-eps)*sigmoid(beta)+eps/2 regardless of whether its decider is consulted first or fifth — testable flatness across the grid, with the mirror structure balancing depth across options. SIGNATURE 4 — COALITION CELLS WITH ROLE REVERSAL (T13, T14): both at tally margin +1 for A. On T13 the advocated top cue f0 leads a 3-vs-2 majority for A while the competing top cue f2 dissents to B; on T14 the roles reverse (f0 dissents to B against a 3-2 majority, f2 leads the majority to A). Each model gets exactly one majority-siding and one minority-siding coalition cell, so a 'follow-the-majority' heuristic cannot mimic either theory's profile; both models also predict p(T13) = p(T1) = p(T3) (same decider f0) and p(T14) = p(T5) (same decider f2) — context invariance across margin and coalition size. SIGNATURE 5 — FIXED-POINT AND TIE CONTROLS: T15 has only f4 discriminating, a cue at depth 2 in BOTH hierarchies — an agreement cell where the two theories concur in choice AND in cascade depth, anchoring the follow rate and testing both models' depth-invariance claims where they cannot differ; any deviation of p(T15) from the dissociation-cell follow rates falsifies the flat-follow assumption of whichever model mispredicts it. T16 is a full tie (both theories: exactly 0.50), anchoring lapse rate and side bias. GLOBAL PROFILES: the advocated model emits p(A) ≈ [0.54, 0.80] on T1, T3, T6, T7, T10, T11, T13, T15 and ≈ [0.20, 0.46] on T2, T4, T5, T8, T9, T12, T14, with 0.50 on T16; the competing model emits the EXACT TRANSPOSE — a near-perfect sign-flipped anti-correlation on 14 of 16 cells (agreement only on T15 and T16). Because the intervals are disjoint cell-by-cell on all twelve permutation cells, model comparison on the full 16-cell response profile separates the theories even after fitting beta, epsilon, and tie_break to each — and even the aggregate direction rate across the twelve mirror cells (advocated: its decider's direction ~70%; competing: the opposite direction ~70%) is diagnostic. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

[10] GOAL: Quantitatively dissociate Distrust-Weighted Frugality with Reversal-Coherent Mixture (advocated) from Weakest-Expert-First Frugality (competing). The two theories are behaviorally IDENTICAL whenever the mixture mass m_am = ambiguity x gate(coherence) is ~0, because the advocated model's central component (w = -val) IS the competitor. The entire dissociation therefore rests on engineering the validity arrangement so that the advocated theory's own gate formulas open the anti-misbound channel: the subject population should be ~91% positional-reversal misbinders and only ~9% value-based anti-validity deciders, while the competitor asserts 100% value-based anti-validity deciders. SETUP: 6 features with validities [0.80, 0.55, 0.90, 0.55, 0.80, 0.65] — an anti-symmetric ('alternating') arrangement chosen so that (i) Spearman(position, validity) = -0.088, giving binding ambiguity = 0.912 (the validity order is essentially uncorrelated with display position, so the binding of accuracy to expert is maximally ambiguous), and (ii) |Spearman(validity, validity-reversed)| = 0.909, so the reversal-coherence gate sigmoid((0.909-0.45)/0.08) ~= 0.997 is fully OPEN. The advocated theory therefore commits, as a POINT PREDICTION with no free parameter, to m_am = 0.912 x 0.997 = 0.909: 90.9% of subjects adopt the anti-misbound hierarchy w = -val[::-1] and 9.1% the central w = -val. Crucially, the validities contain exact TIES (two 0.55s, two 0.80s): per the advocated theory's own tie-anchoring rule, per-cue zeta distortion is switched OFF (zeta_scale = 0), so its predictions are sharp rather than diffuse; and the trial set contains NO oppositions within any tied group ({f1,f3}, {f0,f4} for the value sort; {f2,f4}, {f1,f5} for the reversal), so the free tie_break parameter is inert in BOTH models — neither gains flexibility, and the theories differ only in hierarchy direction plus (beta, epsilon). THE TWO HIERARCHIES: competitor (ascending validity): {f1,f3} (55%) -> f5 (65%) -> {f0,f4} (80%) -> f2 (90%). Anti-misbound (w = -val[::-1]): {f2,f4} -> f0 -> {f1,f5} -> f3 — i.e., the misbinding subject consults the 90% and 80% experts FIRST and the 55% experts LAST, the exact transpose of the competitor's claim. Note this alignment of the reversal-coherent misbound hierarchy with descending validity is not an accident but a consequence the advocated theory itself forces: with high ambiguity, the gate can only open on a NEGATIVELY coherent (anti-symmetric) arrangement, under which trusting by mirror-position weakness equals trusting by own strength. The design therefore pits the competitor's core anti-validity claim (follow the WEAKEST-advertised discriminator) directly against the advocated claim (follow the misbound decider, which here is the STRONGEST-advertised discriminator). SIGNATURE 1 — DISJOINT PREDICTION INTERVALS ON 13 OF 16 CELLS: on every dissociating cell (T1-T13) the competitor's first discriminating cue and the anti-misbound first discriminating cue point to OPPOSITE options. Population-level, with follow rate p_f = (1-eps)*sigmoid(beta)+eps/2 in [0.59, 0.80] for both models: competitor predicts p(anti-misbound decider's option) = 1 - p_f in [0.20, 0.40]; advocated predicts 0.909*p_f + 0.091*(1-p_f) in [0.57, 0.74]. The intervals are DISJOINT, so any single dissociating cell with p > 0.45 toward the anti-misbound decider falsifies the competitor at any parameter setting, and p < 0.55 falsifies the advocated. T1-T12 are six exact A/B mirror pairs, ruling out side bias. SIGNATURE 2 — THE MIXTURE PROPORTION IS A PARAMETER-FREE POINT PREDICTION: agreement cells T14 (only the 55% expert f1 discriminates; both hierarchies follow it, at competitor depth 1 vs anti-misbound depth 4) and T15 (only the 65% expert f5 discriminates; depths 3 vs 4) pin the follow rate p_f; the dissociating cells then estimate the anti-misbound mass m via p_dissoc = m*p_f + (1-m)*(1-p_f). The advocated theory uniquely predicts m_hat ~= 0.91 FOR THIS ARRANGEMENT (a number derived entirely from its ambiguity/coherence gate formulas); the competitor predicts m_hat = 0. Observed m_hat > 0.45 falsifies the competitor; m_hat far below ~0.85 falsifies the advocated theory's gate calibration — a genuinely two-sided quantitative test. SIGNATURE 3 — PER-SUBJECT CONSISTENCY AND CLASSIFICATION: each subject holds ONE hierarchy, so choices across the 13 dissociating cells x 6 repetitions (78 trials) should be near-perfectly consistent with one of the two orders; the advocated predicts ~91%/9% split of subject types, the competitor 0%/100%. Per-trial likelihood ratio for an anti-misbound subject is (0.72/0.28)^78 — overwhelming model comparison even after fitting beta, epsilon, and tie_break to each model. SIGNATURE 4 — DECIDER-VALIDITY AND CASCADE-DEPTH INVARIANCE: across T1, T3, T5, T7, T9, T11, T13 the anti-misbound decider sweeps f2 (90%), f4 (80%), f0 (80%), f5 (65%) at cascade depths 1-3, and the competitor's decider sweeps f1/f3 (55%), f5 (65%), f0 (80%) at depths 1-4. Both one-reason theories predict a FLAT follow rate toward their own decider across all of these cells — any validity gradient or depth effect falsifies the one-reason claim of whichever model is otherwise supported. T3/T4 (90% vs 80% expert in conflict) is the strong-strong cell that removes any weak-vs-strong confound in interpreting the tilt; T9/T10 (65% vs 55%) is the weak-weak counterpart. SIGNATURE 5 — COALITION CELL (T13): the 90% expert alone favors A while the 55%, 55%, and 65% experts favor B (tally margin -2). The competitor follows a lone 55% expert against the 90% expert AND a 3-vs-1 majority; the advocated follows the 90% expert; any margin-integrating strategy says B — a three-way separation on one cell, with the one-reason models additionally predicting p(T13) = p(T1) (same deciders, different context). CONTROLS: T16 is a full tie (both theories: exactly 0.50) anchoring lapse rate and side bias. GLOBAL PROFILES: the competitor emits p(A) ~= 0.72 on T2, T4, T6, T8, T10, T12, T14 and ~= 0.28 on T1, T3, T5, T7, T9, T11, T13, T15, with 0.50 on T16; the advocated emits the near-TRANSPOSE on the 13 dissociating cells (~0.68 toward the misbound decider, slightly diluted below p_f by the 9% central mass — itself a quantitative signature: p_dissoc = 0.818*p_f + 0.091) while agreeing on T14/T15/T16. Because the advocated model's central component is exactly the competitor, the advocated nests the competitor; the experiment is therefore a clean one-sided falsification of the competitor (if the anti-misbound mass is real) plus a point test of the advocated gate (if it is not). With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

[11] GOAL: Quantitatively dissociate Weakest-Expert-First Frugality (advocated; value-based ascending-validity cascade) from Distrust-Weighted Frugality with Reversal-Coherent Mixture Heterogeneity (competing). The competitor NESTS my theory as its central component (w = -val), so a dissociation is only possible by engineering the validity arrangement so that the competitor's own gate formulas commit it to a large anti-misbound mass — then its population prediction differs from mine on every cell where the two hierarchies' first discriminating cues oppose. The design must also be novel relative to experiment [10] (which used a near-fully-coherent anti-symmetric arrangement where the anti-misbound hierarchy equaled descending validity, i.e. 'strongest-first'). Here the anti-misbound hierarchy is a genuinely SCRAMBLED, non-monotone-in-validity order, which (a) decouples the competitor's prediction from any strength heuristic, and (b) yields a different, precisely-computed gate commitment. SETUP: 6 features, validities [0.75, 0.55, 0.90, 0.65, 0.80, 0.55]. The exact ties (two 0.55s) deliberately switch the competitor's per-cue zeta distortion OFF (its own tie-anchoring rule: zeta_scale = 0 when validities are not all distinct), so its predictions are sharp, not diffuse; and the trial set contains NO oppositions within any tied group ({f1,f5} for the value sort, {f0,f4} for the positional reversal), so the free tie_break parameter is inert in BOTH models — neither gains flexibility, and the theories differ only in hierarchy direction plus (beta, epsilon). ARRANGEMENT STATISTICS (computed with the competitor's own formulas): Spearman(position, validity) = -0.145 -> binding ambiguity = 0.855; Spearman(validity, validity-reversed) = -0.618 -> coherence = 0.618; softened gate sigmoid((0.618-0.45)/0.08) = 0.890; hence m_am = 0.855 x 0.890 = 0.761. THE COMPETITOR'S POINT PREDICTION (no free parameter): 76.1% of subjects adopt the anti-misbound hierarchy w = -val[::-1], 23.9% the central w = -val. MY THEORY: 100% value-based anti-validity (weakest-advertised expert first). THE TWO HIERARCHIES: central (mine): {f1, f5} (55%) -> f3 (65%) -> f0 (75%) -> f4 (80%) -> f2 (90%). Anti-misbound: {f0, f4} -> f2 -> f5 -> f1 -> f3, i.e. in OWN advertised validity the misbinding cascade runs 75% -> 80% -> 90% -> 55% -> 55% -> 65%: a NON-MONOTONE schedule that is neither 'strongest-first' nor 'weakest-first' — the competitor's majority-subject behavior cannot be redescribed by any validity heuristic. SIGNATURE 1 — DIRECTION-FLIP DISSOCIATION GRID WITH DISJOINT PREDICTION INTERVALS (T1-T6, T11): six mirror-pair cells where the central decider (55% or 65% expert) opposes the anti-misbound decider (75%, 80%, or 90% expert), plus the coalition cell T11 (the lone 55% expert f1 for A against f0, f2, f4 for B; margin -2). My theory: p(central decider's option) = p_f = (1-eps)*sigmoid(beta)+eps/2 in [0.60, 0.80] over my parameter ranges. Competitor: p(central side) = 0.239*p_f + 0.761*(1-p_f) in [0.35, 0.42] over ITS ranges (p_f in [0.66, 0.79]; ~0.39 at its tightened-regime mean 0.72). THE INTERVALS ARE DISJOINT (gap >= 0.18), so any single dissociating cell at the population level falsifies one theory at any parameter setting; pooling all 11 dissociation cells (66 trials/subject) makes per-subject classification overwhelming (likelihood ratio ~ (p_f/(1-p_f))^66). SIGNATURE 2 — THE MIXTURE MASS IS A PARAMETER-FREE POINT PREDICTION: agreement cells T12-T15 (lone 55%, 80%, 90%, 65% expert) are followed by BOTH hierarchies at every cascade depth, so they pin p_f without knowing the mixture; the dissociation cells then estimate m_hat = (p_obs - (1-p_f))/(2*p_f - 1). My theory: m_hat = 0 exactly. Competitor: m_hat = 0.761, a number derived entirely from its ambiguity x coherence-gate formulas — a genuinely two-sided test of its softened-gate calibration (center 0.45, width 0.08): m_hat ~ 0 falsifies the gate; m_hat far from 0.76 miscalibrates it. SIGNATURE 3 — COMPETITOR-ISOLATED THREE-WAY CELLS (T7-T10): on T7 (f0: 75% -> A; f1: 55% and f2: 90% -> B) the anti-misbound decider is the 75% expert while the 90% expert, the 55% expert, AND the tally majority (margin -1) all point the other way. My theory, standard validity-respecting TTB, and Tallying ALL say B; ONLY the competitor's 76% majority says A (p(A) ~ 0.61). T8-T10 mirror and vary this (65%-vs-80% decider at T9/T10). If subjects show any validity sensitivity on these cells, the competitor fails where my theory survives; conversely T1-T6/T11 isolate MY anti-validity claim (I follow the 55-65% expert against the 75-90% expert and, on T11, against a 3-vs-1 majority) — the combination yields a full three-way separation (weakest-first vs strongest-first vs misbound-reversal) plus Tallying. SIGNATURE 4 — MARGIN-INVARIANCE EQUALITY BLOCKING INTEGRATION MIMICS: T1 (margin 0), T7 (margin -1), T11 (margin -2) all share decider pair (central f1 vs anti-misbound f0); both one-reason theories predict p(T1) = p(T11) and p(T7) = 1 - p(T1), whereas any margin-integrating strategy (e.g., Tallying) predicts a strictly ordered graded profile across them (0.5 / sigmoid(-beta) / sigmoid(-2beta)) — an internal check that the data belong to the one-reason family before adjudicating between its two members. SIGNATURE 5 — SHARED FLATNESS PROBES: (i) across T12-T15 the lone decider's advertised validity sweeps 55-90% and its cascade depth sweeps 1-5 in both hierarchies; both theories predict a FLAT follow rate — any monotone validity gradient or depth effect falsifies the one-reason core of whichever model is otherwise supported (a deliberate shared-risk probe); (ii) the competitor's tightened regime claims per-subject follow-rate SD ~ 0.02, whereas my parameter ranges span p_f in [0.60, 0.80] — the between-subject SD of follow rates on T12-T15 separately discriminates the two noise regimes. SIGNATURE 6 — PER-SUBJECT BIMODALITY: the competitor predicts a 76/24 split of subject types (anti-misbound vs central), each internally consistent across all 11 dissociation cells; my theory predicts a single central type. T16 (full tie) pins both theories to exactly 0.50, anchoring lapse and side bias; T1/T2, T3/T4, T5/T6, T7/T8, T9/T10 are exact A/B mirrors ruling out side bias. GLOBAL PREDICTED PROFILES: my theory emits p_f (~0.70) toward the WEAKEST-advertised discriminator's option on T1, T3, T5, T8, T10, T11, T12, T14 and (1-p_f) on T2, T4, T6, T7, T9, T13, T15, with 0.50 on T16; the competitor emits ~0.61 toward the anti-misbound decider's option on T2, T4, T6, T7, T9, T11 and ~0.39 on T1, T3, T5, T8, T10 — near-perfect anti-correlation on the 11 dissociation cells with disjoint parameter-robust intervals ([0.20, 0.40] vs [0.58, 0.65] toward the anti-misbound side), agreement on T12-T16. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

[12] GOAL: Quantitatively dissociate Distrust-Weighted Frugality with Reversal-Coherent Mixture Heterogeneity (advocated) from Attenuated Frugality with Position-Validity Mixture Heterogeneity (competing). Both are one-reason models sharing the anti-validity central component, so the dissociation must come from their DIFFERENT non-central heterogeneity channels: the advocated adds ONLY an anti-misbound component (w = -val[::-1]) whose adoption is gated by m_am = ambiguity x sigmoid((coherence-0.45)/0.08); the competitor adds ONLY position-primacy (w = -position) and position-recency (w = +position) components whose weights grow LINEARLY in ambiguity, plus a small TTB residual — with NO misbound component and NO gate. The design engineers a validity arrangement that opens BOTH theories' non-central channels to their committed maxima, then pits their unique components head-to-head.

SETUP: 6 features, validities [0.65, 0.90, 0.55, 0.80, 0.55, 0.90]. Arrangement statistics (computed with both models' own formulas, average-tie ranks): Spearman(position, validity) = +0.088, so binding ambiguity = 0.912 — high enough that the competitor's position components dominate its mixture (normalized masses: primacy 0.573, anti-validity 0.353, recency 0.043, TTB 0.031) AND the advocated's gate product is large. Spearman(validity, validity-reversed) = -0.727, so reversal coherence = 0.727, gate = sigmoid((0.727-0.45)/0.08) = 0.970, and the advocated commits — as a PARAMETER-FREE POINT PREDICTION — to m_am = 0.912 x 0.970 = 0.884: 88.4% of subjects adopt the anti-misbound hierarchy, 11.6% the central anti-validity hierarchy, 0% primacy/recency/TTB. The competitor commits to 57.3% primacy, 35.3% anti-validity, 7.4% recency+TTB, and 0% misbound. The predicted mixture distributions have DISJOINT SUPPORTS on both the misbound and primacy masses. The validities contain exact ties (two 0.55s, two 0.90s), which switches per-cue zeta distortion OFF in BOTH models (each model's own tie-anchoring rule), making predictions sharp; and every trial keeps discriminating cues concordant within each tied group (features {f0,f2,f4} never internally oppose; {f1,f3,f5} never internally oppose), so the free tie_break parameter is inert in both models — neither gains flexibility, and the theories differ only in mixture composition plus (beta, epsilon).

THE FIVE HIERARCHIES under this arrangement: anti-validity: {f2,f4}(55%) -> f0(65%) -> f3(80%) -> {f1,f5}(90%); anti-misbound (w = -val[::-1]): {f1,f3} -> f5 -> f2 -> {f0,f4} (consulted validities 90, 80, 90, 55, 65, 55 — a genuinely scrambled, non-monotone order reducible to no validity heuristic); primacy: f0->f1->f2->f3->f4->f5; recency: f5->f4->f3->f2->f1->f0; TTB: {f1,f5} -> f3 -> f0 -> {f2,f4}. Because the strong experts (0.90, 0.90, 0.80) occupy ODD positions and the weak ones (0.55, 0.55, 0.65) EVEN positions, every trial with both groups discriminating in opposition pits the anti-misbound decider (odd group) against the anti-validity decider (even group), while the primacy and recency deciders are STEERED INDEPENDENTLY by choosing which specific features discriminate (primacy follows the earliest discriminating feature; recency the latest).

SIGNATURE 1 — PRIM-FACTOR FLATNESS (advocated) vs GRADED (competitor): trials T1, T3, T5, T7, T9, T11 all have misbound-decider -> A and anti-validity-decider -> B, but the primacy decider points to B on T1/T3/T5/T9 and to A on T7/T11, and the recency decider flips across T3/T5 (->A), T9 (->B), T11 (->B). The advocated theory has NO position components, so it predicts EXACTLY EQUAL p(A) = m_am*p_f + (1-m_am)*(1-p_f) ~= 0.67 on all six cells (p_f = (1-eps)*sigmoid(beta)+eps/2 in [0.66, 0.72] in its tightened regime). The competitor predicts 0.27 (T1), 0.30 (T3), 0.30 (T5), 0.57 (T7), 0.28 (T9), 0.55 (T11) — a ~0.30 spread driven by its 57% primacy mass. The paired contrasts p(T7)-p(T3) and p(T11)-p(T9) are +0.26 under the competitor and EXACTLY ZERO under the advocated; the mirror contrasts p(T8)-p(T4) and p(T12)-p(T10) are -0.26 vs zero. A flat-vs-graded profile on a factor (position-rule decider direction) that the advocated theory is structurally blind to is the central quantitative dissociation.

SIGNATURE 2 — THE GRAND DISSOCIATION CELL (T1/T2): on T1 the anti-misbound decider (f3, the 80% expert) favors A while ALL FOUR competitor components (anti-validity via f2, primacy via f0, recency via f5, TTB via f5) favor B. Advocated: p(A) ~= 0.67 [0.62, 0.72]; competitor: p(A) = 1-p_f ~= 0.27 [0.18, 0.40] — disjoint intervals on a single cell. T2 is the exact mirror (advocated 0.33, competitor 0.73). T1 additionally separates the misbound hierarchy from validity-respecting TTB (which follows f5 to B): if the majority behavior were standard TTB rather than misbinding, choices on T1 go to B. T1 also has tally margin -3, so any margin-integrating strategy (Tallying) predicts p(A) well below 0.5 there while the advocated predicts 0.67 — the orientation-1 cells sweep margins -3, 0, +1, +2, -1, 0 with a FLAT advocated prediction, simultaneously falsifying integration strategies.

SIGNATURE 3 — POINT-PREDICTED MIXTURE MASSES: agreement cells T13-T15 (all five hierarchies concur; both theories: p_f toward the common decider) pin the follow rate p_f without knowing the mixture; the 12 dissociation cells then estimate the population hierarchy masses. The advocated uniquely predicts mass(misbound) = 0.884 and mass(primacy) = 0 for THIS arrangement — numbers derived entirely from its own ambiguity x coherence-gate formulas; the competitor predicts mass(primacy) = 0.573 and mass(misbound) = 0. Observed misbound mass > 0.2 falsifies the competitor at any parameter setting; observed primacy mass > 0.1 falsifies the advocated — a genuinely two-sided quantitative test of each theory's heterogeneity structure.

SIGNATURE 4 — PER-SUBJECT TYPE CLASSIFICATION: both theories give each subject ONE hierarchy; the 12 dissociation cells x 6 repetitions (72 trials/subject) classify subjects nearly perfectly (per-trial likelihood ratio up to (0.72/0.28)^6 per cell). Advocated: ~88% misbound-consistent, ~12% anti-validity-consistent, 0% primacy-consistent. Competitor: ~57% primacy-consistent, ~35% anti-validity-consistent, ~7% recency/TTB, 0% misbound-consistent. A single primacy-consistent subject is evidence against the advocated; a single misbound-consistent subject (beyond the ~3% TTB confound, itself resolved by T1) is evidence against the competitor.

SIGNATURE 5 — SHARED FLATNESS PROBES: T3 vs T5 (anti-validity decider f2 at depth 1 vs f0 at depth 3; misbound decider f1 at 0.90 vs f3 at 0.80) and T1 vs T3 (misbound decider f3 vs f1): both theories predict EQUAL follow rates — any depth or decider-validity gradient falsifies the one-reason core of whichever model is otherwise supported. SIGNATURE 6 — NOISE REGIME: per-subject follow-rate SD on T13-T15 discriminates the advocated's tightened regime (~0.02) from the competitor's attenuated regime (~0.06). CONTROLS: T2/T4/T6/T8/T10/T12 are exact A/B mirrors of T1/T3/T5/T7/T9/T11, ruling out side bias; T16 is a full tie (both theories: exactly 0.50) anchoring lapse rate. GLOBAL PREDICTED PROFILES: the advocated emits a two-level profile FLAT within each orientation (~0.67 on T1,T3,T5,T7,T9,T11; ~0.33 on mirrors; p_f on T13,T15; 1-p_f on T14; 0.50 on T16); the competitor emits a graded profile within each orientation (0.27-0.57 and 0.43-0.73 respectively) driven by the primacy mass. The profiles are structurally disjoint — model comparison on the 16-cell response profile separates the theories even after fitting beta, epsilon, and tie_break to each. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "GOAL: Quantitatively dissociate Attenuated Frugality with Position-Validity Mixture Heterogeneity (advocated) from Distrust-Weighted Frugality with Reversal-Coherent Mixture Heterogeneity (competing). Both are one-reason models sharing the anti-validity central component, so the dissociation must come from their DIFFERENT non-central heterogeneity channels: the advocated adds position-primacy (w = -position) and position-recency (w = +position) components whose masses grow LINEARLY in binding ambiguity with NO coherence dependence; the competing adds ONLY an anti-misbound component (w = -val[::-1]) whose adoption is gated by m_am = ambiguity x sigmoid((coherence-0.45)/0.08). Prior designs [10]-[12] all operated in the HIGH-coherence regime where the gate is OPEN ([12], the same theory pair with roles swapped, used coherence 0.727 and exact validity ties). This design occupies the opposite, unexplored corner: MAXIMAL ambiguity with NEAR-ZERO reversal coherence. This directly tests the gate-vs-linear structural difference: the competitor claims non-central heterogeneity requires reversal-coherent list structure; the advocated claims position heterogeneity tracks ambiguity ALONE.\n\nSETUP: 6 features, ALL-DISTINCT validities [0.55, 0.86, 0.78, 0.95, 0.62, 0.70] (f0..f5). Arrangement statistics (both models' own formulas, distinct ranks): Spearman(position, validity) = 3/35 = 0.086, so binding ambiguity = 0.914; Spearman(validity, validity-reversed) = 1/35 = 0.029, so the competitor's gate = sigmoid((0.029-0.45)/0.08) = 0.005, essentially SHUT. A methodological novelty: because BOTH models scale their per-cue zeta distortion by ambiguity x (coherence or gate), both zeta scales are negligible here (advocated <= 0.037, competitor <= 0.008, vs. minimum validity/position weight gaps of 0.07/1.0) - predictions are SHARP without needing exact validity ties, and the free tie_break is inert in both models (no weight ties).\n\nPARAMETER-FREE MIXTURE COMMITMENTS: Advocated: w_prim = 0.80x0.914, w_anti = 0.45, w_rec = 0.06x0.914, w_ttb = 0.04 -> normalized masses PRIMACY 0.573, ANTI-VALIDITY 0.353, RECENCY 0.043, TTB 0.031, MISBOUND 0. Competitor: m_am = 0.914 x 0.005 = 0.0047 -> ANTI-VALIDITY 0.995, MISBOUND 0.005, POSITION COMPONENTS 0 (structurally absent). The predicted type distributions have disjoint supports on the primacy mass (0.573 vs 0).\n\nTHE FIVE HIERARCHIES: ANTI (ascending validity): f0(.55) -> f4(.62) -> f5(.70) -> f2(.78) -> f1(.86) -> f3(.95). PRIMACY: f0 -> f1 -> f2 -> f3 -> f4 -> f5. RECENCY: f5 -> f4 -> f3 -> f2 -> f1 -> f0. TTB: f3 -> f1 -> f2 -> f5 -> f4 -> f0. MISBOUND: f2 -> f4 -> f3 -> f0 -> f1 -> f5. Crucially, ANTI and PRIMACY share f0 as top cue, so every dissociation trial keeps f0 TIED, forcing consultation to depth >= 2 where the two orders diverge: primacy's second cue is the 2nd-listed 86% expert f1, anti-validity's is the 5th-listed 62% expert f4.\n\nSIGNATURE 1 - DIRECTION-FLIP DISSOCIATION GRID WITH DISJOINT INTERVALS (T1-T10): on T1/T3/T5 the earliest discriminator (f1/f2/f3, positions 1-3, validities 0.78-0.95) favors A while the weakest discriminator f4 (position 5, 62%) favors B; on T7/T9 the weakest discriminator is f5 (position 6, 70%). Deciders on every one of these cells: PRIMACY and TTB -> the early/strong cue; ANTI-VALIDITY, RECENCY, and MISBOUND -> the weak/late cue. Advocated: p(A) = 0.604*p_f + 0.396*(1-p_f) in [0.532, 0.567] (above 0.5 for ANY parameters). Competitor: p(A) = 1 - p_f (+0.005 misbound tilt on T3/T7/T9) in [0.208, 0.344] (below 0.5 for ANY parameters). THE INTERVALS ARE DISJOINT AND ON OPPOSITE SIDES OF 0.5 ON ALL TEN CELLS: the sign of the population preference on any single cell falsifies one theory at any parameter setting. T2/T4/T6/T8/T10 are exact A/B mirrors ruling out side bias. The grid also sweeps primacy-decider depth (2, 3, 4) and validity (0.78-0.95) and anti-decider identity (f4 vs f5): both theories predict a FLAT follow rate across the sweep - a shared one-reason flatness probe.\n\nSIGNATURE 2 - THE PRIMACY-FLIP PAIRED CONTRAST, EXACTLY ZERO VS STRONGLY POSITIVE (T1/T5 vs T11, T2/T6 vs T12): T11 (f4->B, f5->A) has the IDENTICAL anti-validity and misbound deciders as T1/T5 (anti f4->B, mis f4->B), so the competitor predicts p(T11) = p(T1) = p(T5) EXACTLY (both = 1 - p_f) - it is structurally blind to the fact that the primacy, recency, and TTB deciders all flipped. The advocated predicts p(T1) - p(T11) = 0.530*(2*p_f - 1) in [0.165, 0.341]. The paired within-subject contrast is 0 (competitor) vs >= 0.165 (advocated) - disjoint. Mirrors give the symmetric negative contrast. This flat-vs-graded profile on a factor the competitor cannot represent is the central quantitative dissociation.\n\nSIGNATURE 3 - THE ANTI-FLIP SWING MAGNITUDE (T13 vs T1): T13 (f1->A, f3->B) holds the primacy decider fixed (f1->A) while flipping the anti-validity decider (f1->A instead of f4->B). Competitor swing p(T13)-p(T1) = 1.98*p_f - 0.995 in [0.306, 0.577]; advocated swing = 0.644*p_f - 0.322 in [0.100, 0.208]. DISJOINT INTERVALS: the anti-flip swing alone separates the theories. Combined with Signature 2, the two swing magnitudes form a double dissociation: the competitor demands anti-flip swing (>= 0.31) >> primacy-flip swing (= 0); the advocated demands primacy-flip swing (>= 0.165) > anti-flip swing (<= 0.208), reflecting its 57/35 primacy-vs-anti mass split against the competitor's 99.5/0.5.\n\nSIGNATURE 4 - PER-SUBJECT TYPE CLASSIFICATION (the sharpest test): 11 dissociation cells (T1-T10, T15) x 6 repetitions = 66 trials/subject. The competitor predicts ~99.5% of subjects are anti-validity-consistent (always following the weakest-advertised discriminator, f4 or f5); the advocated predicts only ~35% anti-consistent, ~57% PRIMACY-consistent (always following the earliest-listed discriminator), ~7% recency/TTB. A single primacy-consistent subject is strong evidence against the competitor (which assigns position rules zero mass); a near-100% anti-consistent sample falsifies the advocated's point-predicted 0.573 primacy mass - a genuinely two-sided test of each theory's heterogeneity structure at a parameter-free commitment.\n\nSIGNATURE 5 - TTB CONFOUND GUARDS AND CONTROLS: T13/T14 are agreement cells (both theories ~0.69-0.72 toward the early cue f1) where standard validity-respecting TTB follows the 95% expert f3 the OTHER way - if subjects were TTB, p(T13) ~ 0.28, falsifying both target theories while validating the anti-validity claim shared by both. T15 pits PRIMACY ALONE (f1->A) against ANTI + RECENCY + TTB + MISBOUND (f3/f4->B): advocated p(A) = 0.573*p_f + 0.427*(1-p_f) in [0.523, 0.547] vs competitor 1 - p_f in [0.208, 0.344] - disjoint, and simultaneously separating primacy-following from strong-expert-following. T16 (full tie) pins both theories to exactly 0.50, anchoring lapse and side bias. The noise regimes also differ: per-subject follow-rate SD on the agreement cells (T13/T14) is ~0.02 under the competitor's tightened regime vs ~0.06 under the advocated's attenuated regime.\n\nGLOBAL PREDICTED PROFILES (p_f ~ 0.72): Advocated: p(A) ~ 0.546 on T1,T3,T5,T7,T9; ~0.454 on T2,T4,T6,T8,T10; 0.312 on T11/T14; 0.688 on T12/T13; 0.532 on T15; 0.500 on T16. Competitor: ~0.28 on T1,T3,T5,T7,T9,T11,T15; ~0.72 on T2,T4,T6,T8,T10,T12,T13; ~0.28 on T14; 0.500 on T16. The competitor emits a two-level step function constant across ALL cells sharing its (anti, misbound) decider pair - in particular p(T1)=p(T11)=p(T15) exactly - while the advocated emits a graded profile on those same cells driven by its primacy mass. Model comparison on the 16-cell response profile separates the theories even after fitting beta, epsilon, and tie_break to each. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.",
  "validities": [
    0.86,
    0.78,
    0.95,
    0.62,
    0.7,
    0.7
  ],
  "trial_a_ratings": [
    [
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      1,
      0
    ],
    [
      1,
      0,
      1,
      0,
      0,
      1
    ],
    [
      1,
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      1,
      0,
      1,
      0,
      0
    ],
    [
      0,
      1,
      0,
      0,
      1,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1,
      0
    ],
    [
      0,
      0,
      1,
      0,
      1,
      1
    ],
    [
      1,
      1,
      0,
      0,
      0,
      1
    ],
    [
      1,
      1,
      0,
      0,
      1,
      0
    ],
    [
      0,
      1,
      1,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      0,
      1
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      0,
      1,
      0
    ],
    [
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      1,
      1
    ],
    [
      1,
      0,
      1,
      0,
      0,
      1
    ],
    [
      0,
      1,
      0,
      0,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      0
    ],
    [
      1,
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      1,
      0,
      1,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1,
      0
    ],
    [
      1,
      1,
      0,
      0,
      1,
      0
    ],
    [
      1,
      1,
      0,
      0,
      0,
      1
    ],
    [
      0,
      0,
      1,
      1,
      0,
      1
    ],
    [
      0,
      1,
      1,
      0,
      0,
      1
    ],
    [
      1,
      0,
      0,
      1,
      1,
      0
    ],
    [
      0,
      1,
      0,
      1,
      0,
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 24117,
  "output_tokens": 35796,
  "reasoning_tokens": 32660,
  "provider": "Sail Research"
}
```
