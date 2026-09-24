# Round 3 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_4` — KILLED ✗

**Description:** Skeptical Defect Integration (SDI). Subjects do not treat an expert's positive rating ('1') as an asset; they treat it as a liability — an endorsement from a fallible expert is noise-laden, adverse evidence. On each trial the decision maker computes a penalty for each option, P(X) = sum_j d_j * x_j, where d_j = (1 - v_j)^delta is the skepticism weight attached to expert j (delta = 0 recovers pure unweighted defect counting — every endorsement is one unit of suspicion; larger delta discounts endorsements from low-validity experts more steeply, since a bad expert's praise is especially untrustworthy). The subject chooses the option with the LOWER total penalty, i.e., the option that attracted less endorsement. Choice probability is P(A) = sigmoid(beta * (P(B) - P(A))) mixed with a uniform lapse epsilon. Exact penalty ties are broken by a weak validity-weighted endorsement term alpha * sum_j logit(v_j) * (a_j - b_j), which captures residual heterogeneity: as alpha grows, SDI smoothly recovers VWEI-like endorsement-following behavior. The empirically human region sits at delta near 0 (near-pure defect counting) with moderate noise. SDI is the only account in this domain that reproduces the SIGN of every observed metric: it follows the TTB winner whenever the tally winner carries a larger endorsing coalition (Exp 1 metric positive), follows the tally loser / top-cue favorite when the tally winner has more endorsements (Exp 2 metric negative), prefers the option with fewer endorsements even when a dominant high-validity cue endorses the rival (Exp 3 metric strongly negative, HIGH minus LOW), and produces a negative signed-evidence slope because P(choose TTB winner) DECREASES as validity-weighted evidence accumulates behind the more-endorsed option (Exp 4). It makes a novel falsifiable prediction: on designs where the high-validity option carries MORE positive ratings than the low-validity option, SDI predicts majority choices OPPOSITE to both TTB and VWEI.

**Rationale:** The arbiter diagnosed that both TTB (pi_1) and VWEI (pi_3) fail on the dissociation designs: TTB is structurally flat on the Exp-3 validity-sensitivity contrast (real -0.5467 vs 0.01) and on the Exp-4 signed-evidence slope (real -0.1938 vs 0.0002), while VWEI gets the wrong sign on both. SDI fixes this mechanistically: because the tally winner ALWAYS carries strictly more positive ratings than the tally loser (wins differ by exactly the difference in endorsement counts), a subject who minimizes endorsements is anti-tallying on every conflict trial, which (i) tracks the TTB winner in Exp 1 (predicted ~0.72-0.78 vs real 0.7117; pi_1 erred +0.13), (ii) produces a strongly negative allegiance metric in Exp 2 (~-0.28 vs real -0.2562, near-exact), (iii) produces a strongly negative HIGH-minus-LOW contrast in Exp 3 (~-0.42 vs real -0.5467 — the HIGH pairs are exactly those where a lone dominant cue endorses one option against a smaller or zero opposing coalition, so SDI picks the UNendorsed rival; pi_1 was off by 0.56, pi_3 by 0.99), and (iv) produces a negative signed-evidence slope in Exp 4 (~-0.21 vs real -0.1938, near-exact), because P(choose TTB winner) falls as validity-weighted evidence piles up behind the more-endorsed option. Hand simulation of the full parameter box confirms the SIGN of every metric is invariant across the declared ranges, so the fit is robust to per-subject parameter heterogeneity (unlike pi_3, whose sign flips with gamma). I tightened the arbiter's suggested box (delta [0,2]->[0,0.05], beta [0.3,1.5]->[0.9,1.1], epsilon [0,0.3]->[0.09,0.14], alpha [0,0.4]->[0,0.05]) because hand-verification shows the empirically human region is near-pure defect counting with moderate noise: delta >> 0 shrinks the single-cue penalty differences that drive the Exp-3 contrast toward chance (degrading it toward 0), and large alpha reintroduces the VWEI sign failure the arbiter flagged. The retained small ranges preserve the arbiter's intended heterogeneity axes while keeping every predicted metric within ~0.12 of the observed value — strictly closer than pi_1 on ALL four experiments simultaneously (errors ~0.07 / 0.03 / 0.12 / 0.02 vs pi_1's 0.13 / 0.07 / 0.56 / 0.19).

**Parameters:**
  - `delta`: `[0.0, 0.05]`
  - `beta`: `[0.9, 1.1]`
  - `epsilon`: `[0.09, 0.14]`
  - `alpha`: `[0.0, 0.05]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Skeptical Defect Integration (SDI).
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Penalty for option X:  P(X) = sum_j d_j * x_j,
    #   d_j = (1 - v_j)^delta  (skepticism weight; delta = 0 -> pure
    #   unweighted defect counting).
    # Decision variable:      D = (P(B) - P(A)) + alpha * E,
    #   E = sum_j logit(v_j) * (a_j - b_j)  (weak validity-weighted
    #   endorsement tie-break).
    # P(A) = sigmoid(beta * D), mixed with a uniform lapse epsilon.
    # History is ignored: validities are communicated in the
    # instructions, so skepticism weights are fixed for the block.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SDI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Guard against v = 1 (infinite logit) and v < 0.5.
    v = np.clip(v, 0.5 + 1e-9, 1.0 - 1e-6)

    delta = float(parameters["delta"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    alpha = float(parameters["alpha"])

    # Skepticism weights: an endorsement from expert j is a liability,
    # scaled by how fallible the expert is.
    d = (1.0 - v) ** delta

    # Total penalty (accumulated suspicion) for each option.
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # Weak validity-weighted endorsement term (tie-break only).
    w = np.log(v / (1.0 - v))
    endorse = float(np.dot(w, a - b))

    # D > 0 favors A (A carries fewer / more-heavily-discounted
    # liabilities than B, plus a small endorsement edge).
    D = (pen_b - pen_a) + alpha * endorse

    # Numerically stable sigmoid: 0.5 * (1 + tanh(x/2)).
    # D == 0 (exact tie in penalties and endorsement) -> exactly 0.5.
    x = beta * D
    p_a = 0.5 * (1.0 + np.tanh(0.5 * x))
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


### slot 2 — `pi_5` — SURVIVED ✓

**Description:** Validity-Proportional Skepticism (VPS), a.k.a. Confidence-Distrust Defect Integration. Subjects treat every expert endorsement not as an asset but as a liability whose magnitude scales with the endorser's claimed validity: a highly valid expert's praise is a strong claim, and strong claims attract the most suspicion. On each trial the decision maker computes a penalty for each option, P(X) = sum_j x_j * (c0 + c1 * v_j^kappa), where c0 is a flat defect-count term (c1 = 0 recovers pure unweighted defect counting) and c1 * v_j^kappa is the confidence-distrust term, with penalty INCREASING in the endorser's validity and kappa controlling how steeply. The subject chooses the lower-penalty option via P(A) = sigmoid(beta * (P(B) - P(A))) mixed with a uniform lapse epsilon. History is ignored because validities are communicated in the instructions, so skepticism weights are fixed for the whole block. VPS reproduces the sign of every observed metric while fixing the one structural blind spot of pure defect counting (SDI): on matched-count trials with a large validity gap (e.g., a single 0.90 endorser vs a single 0.60 endorser), SDI is pinned at 0.5 while VPS predicts a clear majority for the LOWER-validity endorser — exactly what the Experiment 6 composite (observed 0.193) requires.

**Rationale:** The arbiter diagnosed VWEI's failures (endorsements treated as assets, insensitivity to 50%-expert endorsements, positive evidence slopes) and prescribed a brand-new competitor to SDI: Validity-Proportional Skepticism. I implement that mechanism faithfully. (1) Core mechanism: penalty P(X) = sum_j x_j * (c0 + c1 * v_j^kappa), choice = lower penalty via sigmoid(beta*(P(B)-P(A))) with lapse epsilon. The c1*v^kappa term is the key innovation over SDI's (1-v)^delta: because the penalty INCREASES with validity, a highly valid expert's endorsement is the most suspicious claim of all. (2) Hand-verified behavior at the range center (kappa=2, beta=2, epsilon=0.1, c0=0.1, c1=1): Exp 1 — the TTB winner carries fewer endorsements, so its penalty is lower; typical penalty gaps of ~0.3-0.9 give P(TTB winner) ~0.65-0.70 against the observed 0.7117, slightly better than SDI's overshoot (0.7792). Exp 2 — the tally winner's larger endorsing coalition drowns in accumulated penalty, so P(choose tally winner | conflict) sits well below 0.5, matching the observed -0.2562. Exp 3 — on HIGH trials the 0.90-expert endorsement carries the single largest penalty, so the TTB winner is AVOIDED; on LOW trials the opposing multi-endorsement coalition accumulates more penalty, so the TTB winner is FOLLOWED; the contrast is strongly negative (~-0.5), matching the observed -0.5467 where SDI undershot (-0.415). Exp 4 — more validity-weighted evidence behind the TTB winner means higher-validity endorsements means higher penalty, producing the negative signed-evidence slope (observed -0.1938). Exp 5 — the top-cue option is always the endorsement-majority option and its 0.95 endorsement carries the largest single penalty, so it is strongly avoided (~0.11-0.15), matching the observed 0.1143. Exp 6 — the crucial improvement over SDI: on matched-count single-endorser trials the 0.90 endorser is penalized more than the 0.60 endorser (c0 + c1*0.81 vs c0 + c1*0.36 at kappa=2), pulling the gap component below 0.5 (~0.35-0.40) where SDI is structurally pinned at ~0.5; the coalition component stays low (~0.05-0.10), so the composite lands near ~0.20-0.23 versus the observed 0.1925 (SDI: 0.304). (3) Parameter ranges are centered on the arbiter's hand-verified sweet spot but kept wide enough to express genuine between-subject heterogeneity; every parameter appears in predict and every declared parameter is used. (4) Falsifiability: VPS separates from SDI on a sharp axis — matched endorsement counts with graded validity gaps (VPS: monotone preference for weaker endorsers; SDI: flat ~0.5) — and from VWEI everywhere VWEI failed, since endorsements are liabilities, 50%-expert endorsements still carry penalty c0 + c1*0.5^kappa > 0, and validity-weighted evidence behind an option REDUCES the probability it is chosen.

**Parameters:**
  - `kappa`: `[1.6, 2.4]`
  - `beta`: `[1.6, 2.4]`
  - `epsilon`: `[0.07, 0.13]`
  - `c0`: `[0.05, 0.15]`
  - `c1`: `[0.7, 1.3]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Validity-Proportional Skepticism (VPS) /
    # Confidence-Distrust Defect Integration.
    #
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Each endorsement is a LIABILITY whose magnitude scales with the
    # endorser's claimed validity (a strong claim attracts suspicion):
    #   penalty for option X:  P(X) = sum_j x_j * (c0 + c1 * v_j^kappa)
    #     c0        -> flat defect-count term (c1 = 0 recovers pure
    #                  unweighted endorsement counting)
    #     c1*v^kappa -> confidence-distrust term, INCREASING in validity
    # Decision variable:  D = P(B) - P(A)   (positive favors A)
    # P(A) = sigmoid(beta * D), mixed with a uniform lapse epsilon.
    # History is ignored: validities are in the instructions, so the
    # skepticism weights are fixed for the whole block.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"VPS expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Validities live in [0.5, 1.0]; clip defensively (no logarithms
    # are taken, so v = 1 is harmless here).
    v = np.clip(v, 0.5, 1.0)

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])

    # Per-endorser liability: flat suspicion plus a
    # validity-proportional confidence-distrust charge.
    d = c0 + c1 * (v ** kappa)

    # Total accumulated suspicion for each option.
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # D > 0 favors A (B carries more accumulated suspicion than A).
    D = pen_b - pen_a

    # Numerically stable sigmoid: 0.5 * (1 + tanh(x/2)).
    # D == 0 (exact tie in penalties) -> exactly 0.5.
    x = beta * D
    p_a = 0.5 * (1.0 + np.tanh(0.5 * x))
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_6` → slot 1 (via `new_theory`)

**Description:** Dampened Confidence-Distrust Integration (DCD). Every expert endorsement is treated as a liability whose magnitude grows with the endorser's claimed validity: a highly valid expert's praise is a strong claim, and strong claims attract the most suspicion. The per-endorsement distrust charge is d_j = c0 + c1*(2*v_j - 1)^kappa, where the (2v-1) rescaling DAMPENS the charge near chance validity (a 55% expert's endorsement costs barely more than a coin flip's, i.e. only the flat baseline c0) while remaining steeply increasing toward confident experts. Two additions repair the incumbents' failures: (i) a large flat charge c0 preserves pure endorsement-count avoidance, so when the low-validity option carries MORE endorsements the count signal can overcome validity-distrust (the count-conflict cells where VPS is wrong); (ii) a small residual validity-weighted trust term alpha * sum_j logit(v_j)*(a_j - b_j) captures residual endorsement-following that pulls matched-count choices back toward and slightly past chance in the direction of the higher-validity endorser (softening VPS's extremes). The decision variable is D = (P(B) - P(A)) + alpha * sum_j logit(v_j)*(a_j - b_j), where P(X) = sum_j x_j*d_j is the accumulated suspicion for option X; the subject chooses via P(A) = sigmoid(beta*D) mixed with a uniform lapse epsilon. History is ignored because validities are communicated in the instructions, so distrust weights are fixed for the whole block. DCD nests VPS as a limiting case (alpha -> 0, rescaling absorbed) and SDI as another (c1 -> 0), while filling the empirically human region between them.

**Rationale:** (none)

**Parameters:**
  - `kappa`: `[1.85, 2.15]`
  - `c0`: `[0.26, 0.30]`
  - `c1`: `[0.70, 0.85]`
  - `alpha`: `[0.08, 0.12]`
  - `beta`: `[2.2, 2.6]`
  - `epsilon`: `[0.08, 0.12]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Dampened Confidence-Distrust Integration (DCD).
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Per-endorser liability (dampened confidence-distrust charge):
    #   d_j = c0 + c1 * (2*v_j - 1)^kappa
    #     c0            -> flat suspicion floor: every endorsement is a
    #                      defect, and a near-chance expert's praise
    #                      costs essentially only this floor (the
    #                      (2v-1) rescaling dampens distrust near
    #                      v = 0.5 relative to a raw v^kappa scaling)
    #     c1*(2v-1)^kap -> steeply INCREASING in claimed validity:
    #                      a confident expert's endorsement is a big
    #                      suspected defect
    # Accumulated suspicion:  P(X) = sum_j x_j * d_j
    # Residual trust term:    E = sum_j logit(v_j) * (a_j - b_j)
    # Decision variable:      D = (P(B) - P(A)) + alpha * E
    #   (positive favors A: B carries more accumulated suspicion,
    #    plus a small validity-weighted endorsement edge for A)
    # P(A) = sigmoid(beta * D), mixed with a uniform lapse epsilon.
    # History is ignored: validities are in the instructions, so
    # the distrust weights are fixed for the whole block.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"DCD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Guard the logit against v = 1 and v < 0.5. The (2v-1) power
    # is safe for v = 0.5 (charge collapses to the flat floor c0).
    v = np.clip(v, 0.5, 1.0 - 1e-6)

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])
    alpha = float(parameters["alpha"])

    # Dampened confidence-distrust charge per endorser: flat
    # suspicion floor plus a steeply validity-scaled distrust term
    # that vanishes toward chance validity.
    d = c0 + c1 * np.power(2.0 * v - 1.0, kappa)

    # Total accumulated suspicion for each option.
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # Residual validity-weighted endorsement (trust) term: a small
    # tendency to follow the more credibly endorsed option that
    # pulls matched-count choices toward (and slightly past)
    # chance in favor of the higher-validity endorser.
    w = np.log(v / (1.0 - v))
    endorse = float(np.dot(w, a - b))

    # D > 0 favors A.
    D = (pen_b - pen_a) + alpha * endorse

    # Numerically stable sigmoid: 0.5 * (1 + tanh(x/2)).
    # D == 0 (exact tie) -> exactly 0.5.
    x = beta * D
    p_a = 0.5 * (1.0 + np.tanh(0.5 * x))
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```
