# Round 4 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3_2` — KILLED ✗

**Description:** Reversed-Hierarchy Frugality (misbinding TTB). People are one-reason decision makers: they consult cues one at a time and let the first strictly discriminating cue decide, ignoring all lower cues and all magnitude information. However, the subjective cue hierarchy is ANTI-correlated with the instructed validities: subjects misbind the communicated validity list to feature positions (binding the highest validity to the last-presented feature), so the cascade effectively runs in ascending true validity (kappa = 1, full misbinding). When no cue discriminates, the model guesses uniformly. Response noise enters through a softmax with inverse temperature beta over the binary winner score, plus an independent lapse epsilon mixing in a uniform choice. Because no integration across cues occurs, the model predicts a flat choice profile across tally margins (zero psychometric slope), in direct opposition to Tallying's graded curve. The noise regime is moderate: the cascade-follow probability p = (1-eps)*sigmoid(beta) + eps/2 sits near 0.60, matching the human follow rate implied by both experiments.

**Rationale:** The decision rule (kappa=1 reversed TTB cascade, uniform guess on ties, softmax+lapse output) is unchanged — the arbiter is right that the misfit is purely in the noise magnitudes. But I calibrated the new noise regime from the algebra of each metric rather than adopting the suggested ranges verbatim.

Deriving each experiment's implied follow probability p = (1-eps)*sigmoid(beta) + eps/2 from the observed values and the previous ACCEPTED candidate's (pi_3_1, p_bar≈0.703) predictions:

- Exp 1: on every margin<=0 trial the reversed cascade winner OPPOSES the TTB winner, so metric = 1 - p; real 0.3633 -> p ≈ 0.637 (pi_3_1 gave 0.2933, i.e. p too high).
- Exp 2: E[s*margin] = (2p-1)*C with C≈0.29 (recovered from pi_3: 0.0608 at p=0.606 and pi_3_1: 0.1208 at p=0.703); real 0.0358 -> p ≈ 0.56.
- Exp 3: metric = 2p-1 exactly; real 0.4533 -> p ≈ 0.727 (pi_3_1 already near-perfect at 0.4276).
- Exp 4: metric = 4p-2; real 0.7787 -> p ≈ 0.695 (pi_3_1 near-perfect at 0.8027).
- Exp 5: metric ≈ p + 0.015; real 0.6133 -> p ≈ 0.60 (pi_3_1 overshot at 0.7178).
- Exp 6: metric = 2p-1; real 0.34 -> p ≈ 0.67.

The per-experiment optima range from 0.56 to 0.73; minimizing the (per-experiment normalized) L2 distance over p gives an optimum at p_bar ≈ 0.68, versus the current 0.703. This lowers the predicted Exp 1 level (0.29 -> ~0.36), Exp 2 value (0.121 -> ~0.07), Exp 5 follow rate (0.718 -> ~0.69) and Exp 6 contrast (0.39 -> ~0.36) while only marginally degrading the excellent Exp 3/Exp 4 fits — a net aggregate-loss improvement.

Why I deviate from the arbiter's literal ranges (beta [0.1,0.9], eps [0.10,0.40]): those ranges imply p_bar ≈ 0.59 (e.g. beta=0.1, eps=0.25 gives p≈0.52; beta=0.9, eps=0.25 gives p≈0.66), which would drag Exp 3 down to ~0.18 (real 0.4533) and Exp 4 down to ~0.36 (real 0.7787) — exactly the two experiments where the arbiter itself says the reversed-hierarchy direction is strongly confirmed and should not be degraded. My ranges follow the arbiter's *direction* (lower, wider beta; higher epsilon; more between-subject heterogeneity, with per-subject p spanning ~[0.62, 0.75]) but center the follow rate at ~0.68, the value the cross-experiment algebra actually demands. kappa stays fixed at 1.0: for monotone validity lists any kappa in [0.85, 1] yields the identical fully-reversed cue order, so freeing it adds nothing, and values below 0.5 would flip the hierarchy and destroy the confirmed Exp 2/5/6 fits.

**Parameters:**
  - `beta`: `[0.6, 1.2]`
  - `epsilon`: `[0.08, 0.18]`
  - `kappa`: `{1.0}`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Reversed-Hierarchy Frugality (misbinding TTB). One-reason
    # decision making: cues are consulted one at a time and the first
    # strictly discriminating cue decides. The subjective hierarchy is
    # ANTI-correlated with the instructed validities (kappa = 1: the
    # communicated validity list is misbound to feature positions,
    # highest validity bound to the last-presented feature), so the
    # cascade runs in ascending true validity. No integration across
    # cues occurs -> flat psychometric profile across tally margins.
    # Noise: softmax (inverse temperature beta) over the binary winner
    # score, plus an independent lapse epsilon mixing in a uniform
    # choice. History is ignored (no feedback in this task).
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

    # Subjective validities: mixture of the instructed vector and its
    # end-to-end reversal. kappa = 1 -> fully reversed hierarchy
    # (full positional misbinding), per the theory's fixed claim.
    kappa = float(parameters["kappa"])
    w = (1.0 - kappa) * val + kappa * val[::-1]

    # Descending SUBJECTIVE validity; stable argsort so ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-w, kind="stable")

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

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta). The implied cascade-follow
    # probability is p = (1-eps)*sigmoid(beta) + eps/2, which over the
    # sampled (beta, epsilon) ranges averages ~0.68 with per-subject
    # values spanning ~[0.62, 0.75].
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / float(n_opts))
```

**`policy(probs)`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_4` — SURVIVED ✓

**Description:** Weakest-Expert-First Frugality (value-based anti-validity TTB). People are one-reason decision makers: they consult cues one at a time and let the first strictly discriminating cue decide, ignoring all lower cues and all magnitude/tally information. However, the subjective cue hierarchy is anti-correlated in VALUE with the instructed validities: subjects systematically consult the cue with the LOWEST advertised validity first, then the next-lowest, and so on (equivalently, subjective weights w_j = -validity_j). Because no integration across cues occurs, the model predicts a flat choice profile across tally margins (zero psychometric slope). When no cue discriminates, the model guesses uniformly. Validity ties are broken by a free per-subject tie-breaking parameter (early-position vs late-position preference among equally weak cues). Response noise enters through a softmax with inverse temperature beta over the binary winner score, plus an independent lapse epsilon mixing in a uniform choice, giving an implied cascade-follow probability p = (1-eps)*sigmoid(beta) + eps/2 that spans roughly 0.54 to 0.86 across subjects with a mean near 0.70 — high enough to produce the steep contrasts in the non-monotone-validity experiments while moderate values cover the monotone ones.

**Rationale:** The arbiter diagnosed the fatal flaw of the pi_3 family: it implemented the anti-validity hierarchy by POSITIONALLY reversing the communicated validity list. In the six monotone-validity experiments (where validity decreases with feature index) positional reversal coincides with ascending-validity sorting, so pi_3_2 succeeded there; but in the two non-monotone experiments (presented Experiments 7 and 8) positional reversal wrongly promotes mid- and high-validity cues to the top of the hierarchy (f4 at 0.65 in Experiment 7; f3 at 0.95 in Experiment 8), producing strongly POSITIVE metric values where the real data are strongly NEGATIVE (-0.31 and -0.29). The fix is to compute the anti-correlation on validity VALUES: sort cues by ascending true validity. I verified analytically that this flips both signatures: in Experiment 7 the metric becomes (7-14q)/12 (e.g., -0.245 at q=0.70, vs real -0.311, where pi_3_2 predicted +0.36); in Experiment 8 the single margin-0 cell T11 has the weakest cue f0 favoring A while the positional-reversal decider is f5 favoring B, so the metric becomes 0.5-q (e.g., -0.20 at q=0.70, vs real -0.289, where pi_3_2 predicted +0.19). Meanwhile, in every monotone-validity experiment the ascending-value order is IDENTICAL to the positional reversal, so the already-good predictions of pi_3_2 on Experiments 1-6 are preserved exactly (Experiment 1 metric = 1-q; Experiment 3 = 2q-1; Experiment 4 = 4q-2; Experiment 5 = q; Experiment 6 = 2q-1). I then solved for the implied follow probability q each experiment demands (0.64, 0.58, 0.73, 0.70, 0.61, 0.67, 0.76, 0.79 respectively) and set the noise ranges beta in [0.5, 1.5], epsilon in [0.06, 0.18] so that the mean implied q = E[(1-eps)*sigmoid(beta)+eps/2] lands at approximately 0.70 — the value that minimizes the aggregate squared error across all eight per-experiment constraints (q=0.70 yields RMS error ~0.06, better than both 0.68 and 0.72) — while the per-subject span [0.54, 0.86] covers the heterogeneity the arbiter requested. The tie_break parameter implements the arbiter's suggested free Bernoulli over early-vs-late position among equally weak cues; I verified it is inert for all eight metrics' diagnostic cells (tied cues never jointly determine a scored cell), so it adds subject heterogeneity without distorting predictions. The model retains the empirically validated one-reason, margin-blind structure (flat across tally margins, uniform guess on full ties) that Tallying (pi_2) lacked, while replacing the positional mechanism with the value-based one that produces the correct negative signs in the two non-monotone experiments.

**Parameters:**
  - `beta`: `[0.5, 1.5]`
  - `epsilon`: `[0.06, 0.18]`
  - `tie_break`: `{0, 1}`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_5` → slot 1 (via `new_theory`)

**Description:** Distrust-Weighted Frugality with Reversal-Coherent Mixture Heterogeneity (softened-gate variant). People are one-reason decision makers: cues are consulted one at a time and the first strictly discriminating cue decides, with no integration across cues (flat psychometric slope over tally margins, consistent with the margin-0 signatures of Exps 3-8). The population-central subjective hierarchy is the anti-validity ('distrust') order: the weakest-advertised expert is consulted first (w = -validity). Heterogeneity is asymmetric and structured: each subject adopts one of a small family of re-encoding rules — distrust-the-weakest (w = -val) or distrust-the-misbinding (w = -val[::-1]) — with the anti-misbound rule's adoption probability gated by binding ambiguity (1 - |Spearman(position, validity)|) times a STEEP, SOFTENED reversal-coherence gate sigmoid((|Spearman(val, val[::-1])| - 0.45)/0.08): subjects adopt the anti-misbound re-encoding only when the binding is non-monotone AND the reversed list is structurally coherent enough to be learnable, with the gate threshold placed low enough (0.45 rather than 0.50) that moderately coherent designs (Exp 5, coherence 0.38) retain enough anti-misbound mass to reproduce their strong misbound-decider follow. Idiosyncratic per-cue distortion (zeta, zero-centered, fixed per subject) accumulates only when every advertised validity is distinct, scaled by the same ambiguity x coherence-gate product. Noise: softmax(beta) over the binary winner score plus lapse epsilon, in the tightened heterogeneous regime (implied follow p mean ~0.72, per-subject SD ~0.02). Validity ties are broken by a free per-subject tie-break. The theory reduces to Weakest-Expert-First Frugality (pi_4) in the ambiguity = 0 special case.

**Rationale:** Minimal single-number edit of the running-best (iter-3) base, applying the iter-5 critic's pre-specified contingency exactly. The iter-5 candidate (this base + steep coherence gate centered at 0.50) was rejected at loss 0.0669 vs 0.0607, and the critic's diagnosis was concrete: the 0.50-centered gate over-suppressed Exp 5's anti-misbound adoption (m_am 0.315 -> 0.153), driving its point from a near-perfect 0.616 (real 0.613) to 0.681 — a +0.068 error, roughly 3 SE — while everything the gate was supposed to fix actually worked (Exp 10 index 187 -> 115.2 vs real 116.6, the best fit of any theory on that metric; Exp 7 -0.206 -> -0.226 vs real -0.311; Exp 2 point 0.097 -> 0.051 and variance 0.0435 -> 0.0330 vs real 0.036/0.018). The iter-3 critique had already pre-registered the remedy for exactly this failure mode ('if Exp 5 drifts beyond +0.04, soften the threshold to center 0.45, width 0.10'). THE EDIT: move the gate center from 0.50 to 0.45 (width 0.08), applied consistently to BOTH m_am and zeta_scale; nothing else changes — the one-reason cascade, softmax+lapse noise (beta [0.9,1.6], epsilon [0.12,0.26], mean p ~0.72, SD ~0.02 — the iter-4 interpolation to p~0.706 was rejected by the gate and is NOT reintroduced), free tie-break, tie-anchored zeta, and ambiguity scaffold are retained verbatim from the accepted base. Expected per-design consequences of the center shift (using the critic's own computed design statistics): Exp 5 (coherence 0.38, ambiguity ~0.84): gate rises 0.15 -> ~0.29, m_am ~0.25, restoring the point to ~0.62-0.64 (within the +0.03 target of real 0.613) while cutting the bimodality-driven variance excess relative to iter 3. Exp 10 (coherence 0.657): gate ~0.93, m_am ~0.80 (vs 0.75 at center 0.50, vs 0.58 linear at iter 3) — the index stays in the ~105-130 band around real 116.6. Exp 7 (coherence 0.618): gate ~0.90, m_am ~0.61, holding or slightly improving the -0.226 fit toward real -0.311. Exp 8 (coherence 0.71): gate ~0.98, essentially unchanged. Exp 9 (coherence 0.543, ambiguity 0.229): m_am rises only marginally (0.14 -> 0.17), keeping its good fit. Exp 2's zeta gating is slightly relaxed relative to center 0.50 but remains far below the ungated iter-3 scrambling, so its improved point/variance should largely persist. On the critic's multi-seed mandate: the accept margin at this stage (0.006) is comparable to Monte Carlo noise on individual experiments, and the expected movement here is dominated by a single large, mechanistically-guaranteed restoration (Exp 5, ~+0.06 error removed) plus retention of the already-demonstrated Exp 10/7/2 gains, so the edit should clear the gate on seed-mean rather than seed-luck; no stochastic-averaging device is available inside predict/policy, which are per-trial deterministic given parameters. Distinguishability from pi_4 is unchanged: pi_4 is the ambiguity = 0, comp >= m_am special case, and the theory predicts design-dependent amplification/attenuation of anti-hierarchy effects (via the coherence gate and tie anchoring) that pi_4 structurally cannot generate. Expected aggregate loss: ~0.045-0.055, below the 0.0607 floor.

**Parameters:**
  - `beta`: `[0.9, 1.6]`
  - `epsilon`: `[0.12, 0.26]`
  - `sigma_h`: `[0.5, 1.75]`
  - `zeta`: `[(-1, 1)] * n_features`
  - `tie_break`: `{0, 1}`
  - `comp`: `[0, 1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)
```
