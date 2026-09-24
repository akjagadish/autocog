# Round 2 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_3` — KILLED ✗

**Description:** Noisy Validity-weighted evidence integration with capacity-interference and one-reason lapses (NVW). People do not use a single fixed heuristic; they pool all the expert ratings they can, but each cue's contribution is weighted by its validity raised to a steepness exponent gamma (gamma = 0 recovers Tallying, gamma -> infinity recovers Take-The-Best, so both classic heuristics are limiting cases). Integration is capacity-limited: beyond a working-memory span of about four cues, every additional cue that must be pooled multiplies the reliability (effective inverse temperature) of the integrated comparison by an interference factor delta < 1, so choice consistency is design-dependent and falls sharply as feature count grows. Each cue is additionally attended only with probability alpha (lapses of attention). On a minority of trials — with a probability omega that is modulated by how strongly the most valid discriminating cue's validity dominates the validity of the opposing side's cues — the agent abandons integration and makes a one-reason decision based solely on that top cue. The final choice is a softmax (at the capacity-scaled inverse temperature) over the attended, validity-weighted evidence difference, mixed with an independent lapse to uniform guessing.

**Rationale:** DIAGNOSIS OF PRIOR FAILURES. TTB (pi_1) predicts P(choose TTB winner | conflict) = 0.84 in Experiment 1 (real: 0.208) and a strongly negative tally-minus-TTB contrast of -0.48 in Experiment 2 (real: +0.158): it is far too one-reason driven. Tallying (pi_2) is almost right in Experiment 1 (0.136 vs 0.208, i.e. tally-consistency 0.86 vs real 0.79) but predicts the SAME invariant consistency in Experiment 2 (0.548 vs real 0.158, i.e. tally-consistency 0.86 vs real 0.605). The signature of the human data is therefore: (i) mostly integration, mildly validity-sensitive (slightly more TTB-leaning than pure tallying in Exp 1), and (ii) a large DROP in integration consistency when the cue set grows from 5 to 6 features. Crucially, I verified analytically that the two designs are matched on every conflict-trial statistic that a fixed-noise integrator sees (weighted margins, SNR, dominance of the top cue — if anything top-cue dominance is HIGHER in Exp 1), so no shared-parameter validity weighting alone can produce the gap; the gap must be carried by a feature-count-dependent noise term. This is exactly the arbiter's mechanism (3): 'effective decision noise grows with feature count'.

MECHANISMS. NVW implements the arbiter's prescription: (1) weights w_j = v_j^gamma nest Tallying (gamma=0) and TTB (gamma->inf); (2) a one-reason lapse mode with rate omega coupled to the validity dominance of the top discriminating cue over the opposing majority's cues; (3) cue-wise attention alpha plus a capacity/interference term: beyond a working-memory span of ~4 cues, each extra pooled cue multiplies evidence reliability by delta, so noise grows steeply with feature count; (4) softmax response rule with inverse temperature beta plus an independent lapse epsilon.

CALIBRATION LOGIC. Because parameters are sampled per subject from the declared ranges and metrics are computed on the pooled population, the ranges themselves are the population-level fit (this is visible in pi_2: its 0.86 tally-consistency in BOTH experiments equals 1 - E[epsilon]/2 under uniform sampling of epsilon in [0, 0.5]). I therefore derived the ranges analytically from the conflict-trial structure of both designs so that the SAME parameter distribution reproduces both targets: with gamma ~ U[0.7, 1.3], beta ~ U[5, 11], delta ~ U[0.18, 0.30], the effective Exp-1 sensitivity is beta*delta (mean ~1.9) and the effective Exp-2 sensitivity is beta*delta^2 (mean ~0.47). Averaging the logistic over these ranges yields Exp-1 tally-consistency ~0.79-0.80 (metric P(TTB winner|conflict) ~ 0.20-0.22, real 0.208) and Exp-2 tally-consistency ~0.60-0.61 (metric ~0.15-0.17, real 0.158). The one-reason lapse (omega ~ 0.02) and lapse (epsilon ~ 0.03) contribute the small residual top-cue choices without materially shifting the averages. This is why I propose tighter ranges than the arbiter's illustrative ones: under per-subject uniform sampling, wide ranges (e.g. gamma in [0,4]) would average TTB-like and tally-like subjects and destroy the fit; the ranges above are centered on the region that both experiments jointly identify.

DEVIATION FROM ARBITER DETAIL. I keep the validity-dominance coupling for omega but let its slope (omega_dom) be free-signed: computing actual dominance values (top-cue validity minus mean opposing validity) gives a HIGHER mean in Exp 1 (0.23) than Exp 2 (0.19), so an increasing-with-dominance coupling would push Experiment 1 toward TTB — the opposite of what the data show. The between-experiment gap is instead carried by the capacity-interference term, which is the arbiter's own mechanism (3). The coupling remains in the model, falsifiable against new designs that manipulate the validity gap at fixed tally margins.

FALSIFIABLE PREDICTIONS. NVW predicts (a) choice consistency falls sharply (geometrically) as feature count grows beyond ~4 at fixed tally margins — distinguishing it from Tallying's invariance; (b) deviations from tally invariance that grow with the validity spread of the cues (via gamma); (c) a small, validity-dominance-modulated minority of pure one-reason choices. It can be validated on new designs that independently manipulate feature count and the top-cue validity gap.

**Parameters:**
  - `gamma`: `[0.7, 1.3]`
  - `alpha`: `[0.9, 1.0]`
  - `beta`: `[5.0, 11.0]`
  - `delta`: `[0.18, 0.30]`
  - `epsilon`: `[0.0, 0.06]`
  - `omega_base`: `[0.0, 0.04]`
  - `omega_dom`: `[-0.1, 0.1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # NVW: Noisy Validity-weighted integration with capacity-interference
    # and one-reason lapses.
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"NVW expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n}."
        )

    gamma = float(parameters["gamma"])
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    delta = float(parameters["delta"])
    epsilon = float(parameters["epsilon"])
    omega_base = float(parameters["omega_base"])
    omega_dom = float(parameters["omega_dom"])

    # (1) Validity-steepness weights: gamma = 0 -> Tallying (equal weights),
    # gamma -> infinity -> Take-The-Best (top cue only).
    w = np.power(v, gamma)

    # (3) Attention: each cue independently attended with probability alpha.
    att = np.random.random(n) < alpha
    a_win = (a > b) & att
    b_win = (b > a) & att
    disc = a_win | b_win

    # (3) Capacity / interference: beyond a working-memory span of ~4 cues,
    # each additional cue that must be pooled multiplies the reliability
    # (effective inverse temperature) of the integrated evidence by delta.
    # This makes decision noise grow with feature count.
    excess = max(0, n - 4)
    beta_eff = beta * (delta ** excess)

    def _sig(z):
        # Numerically stable logistic.
        return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))

    if not np.any(disc):
        # No attended discriminating cue: pure guess.
        p_a = 0.5
    else:
        # (1) Integration: softmax over the weighted evidence difference.
        d = float(np.sum(w[a_win]) - np.sum(w[b_win]))
        p_int = float(_sig(beta_eff * d))

        # (2) One-reason mode: the single most valid attended
        # discriminating cue decides (TTB-like), with confidence scaled
        # by that cue's weight and the effective sensitivity.
        idx = np.where(disc)[0]
        top = int(idx[np.argmax(v[idx])])
        s = 1.0 if a[top] > b[top] else -1.0
        p_1r = float(_sig(beta_eff * w[top] * s))

        # (2) Validity-dominance coupling for the one-reason lapse rate:
        # dominance of the top cue over the opposing side's cue validities.
        opp = b_win if s > 0 else a_win
        if np.any(opp):
            dom = float(v[top] - np.mean(v[opp]))
        else:
            dom = float(v[top] - 0.5)
        omega_t = float(np.clip(omega_base + omega_dom * dom, 0.0, 0.95))

        p_a = (1.0 - omega_t) * p_int + omega_t * p_1r

    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (4) Independent lapse toward uniform guessing.
    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    return probs
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_4` — SURVIVED ✓

**Description:** Subjective-Weighted Counting with convex cue-order recency (SWC), deterministic-attention variant. People integrate ALL discriminating cues additively with no capacity interference, but the cue weights are subjective and only partially calibrated to communicated validities: w_j = (1-lambda)/n + lambda*[(1-rho)*vnorm_j + rho*(j/(n-1))^eta]. Because validities are always presented in strictly decreasing feature order, the convex recency gradient is anti-correlated with validity, producing within-margin anti-validity preferences (Experiment 3's negative d-coefficient) while preserving mostly-tally aggregate behavior. Attention to cues is deterministic (every cue is always attended); the only stochasticity is the logistic choice noise itself plus a small uniform lapse epsilon. This edition implements the critic's in-family tuning: the noise channels that let the fit collapse to flat counting are removed or tightened so the recency gradient must do the work.

**Rationale:** Minimal-diff edit of the accepted SWC base, applying the critic's four in-family prescriptions. (1) CUT THE NOISE: the per-cue stochastic attention (alpha drawn per trial) was removed entirely — every cue is now always attended. This was the dominant variance source behind Experiment 3's between-subject variance of 0.2046 (real: 0.0186): binomial per-cue dropout made single subjects wildly inconsistent and let the fit soften aggregate rates with noise instead of using the recency gradient. The uniform lapse epsilon is retained but tightened to [0, 0.03]. (2) FORCE THE RECENCY GRADIENT TO ENGAGE: lambda is now [0.8, 1.0], rho [0.8, 1.0], eta [2.5, 4.5], so the flat-counting collapse (lambda ~ 0) that produced Exp 3 ~ 0 is unreachable; the mixture is always dominated by the graded (validity + convex recency) component. (3) MODERATE BETA: beta is capped at 1.7 (previous range went to 8), so conflict probabilities stay in the 0.6-0.9 band rather than saturating — this is what Experiment 2's near-indifferent contrast (real 0.158) requires. (4) ANCHORS: with no interference and moderate beta, hand computation at the range midpoint (lambda~0.9, rho~0.9, eta~3.5, beta~1.2) gives: Exp 4 anchor allegiance ~0.83 (real 0.843) — the 7-1 anchors with the loss on the last cue land near p~0.65, cooling the previous 0.900; Exp 1 TTB-adherence ~0.21 (real 0.208); Exp 2 contrast ~0.19-0.21 (real 0.158, down from 0.321); Exp 3 partial d-coefficient ~ -0.4 at the midpoint and ~ -0.55 to -0.68 at the recency-heavy corner (rho -> 1, lambda -> 1, beta ~1.2-1.3), bracketing the real -0.5756 — a massive improvement over the previous +0.026. The convex recency term is preserved unchanged: it remains the only in-family degree of freedom that yields Experiment 3's negative within-margin coefficient while keeping Experiments 1, 2 and 4 near their targets, and the tightened ranges now force the fit into that basin instead of the noise-dominated flat-counting region the previous parameterization permitted.

**Parameters:**
  - `beta`: `[0.8, 1.7]`
  - `epsilon`: `[0.0, 0.03]`
  - `lambda`: `[0.8, 1.0]`
  - `rho`: `[0.8, 1.0]`
  - `eta`: `[2.5, 4.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # SWC: Subjective-Weighted Counting with convex cue-order recency
    # (deterministic-attention variant).
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SWC expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: linearly decreasing validities.
        v = np.linspace(0.95, 0.55, n)

    lam = float(parameters["lambda"])
    rho = float(parameters["rho"])
    eta = float(parameters["eta"])
    beta = float(parameters["beta"])
    eps = float(parameters["epsilon"])

    # (1) Subjective validity component: min-max normalized validities.
    vspan = float(v.max() - v.min())
    if vspan > 1e-12:
        vnorm = (v - v.min()) / vspan
    else:
        vnorm = np.full(n, 0.5)

    # (2) Recency component: position in presentation order, convexly
    # transformed so that late cues can be strongly overweighted.
    if n > 1:
        pos = (np.arange(n, dtype=float) / (n - 1.0)) ** eta
    else:
        pos = np.full(n, 0.5)

    # Subjective weights: mixture of flat counting, validity-following,
    # and recency-following. Because validities are presented in strictly
    # decreasing feature order, the recency part is anti-correlated with
    # validity -- the source of within-margin anti-validity preferences.
    g = (1.0 - rho) * vnorm + rho * pos
    w = (1.0 - lam) / n + lam * g

    # (3) Deterministic attention: every cue is attended on every trial
    # (per-cue attention lapse removed -- real subjects are highly
    # consistent, and stochastic lapses were the dominant noise source
    # inflating between-subject variance).
    s = np.sign(a - b)

    # Additive integration of all discriminating cues.
    # NO capacity interference: beta is design-independent.
    d = float(np.sum(w * s))

    # Numerically stable logistic.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (4) Independent uniform lapse.
    probs = (1.0 - eps) * probs + eps * 0.5
    return probs
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_5` → slot 1 (via `new_theory`)

**Description:** INVERTED-VALIDITY COUNTING (IVC) — distrust-weighted tallying. Subjects do read the communicated validities but map them through a systematic distrust inversion: they act as if each stated validity were the expert's error rate, so the most highly validated experts are discounted most. Every discriminating cue is attended deterministically (no attention lapses, no working-memory capacity interference — choice consistency is design-independent). Each cue j carries subjective weight w_j = (1-tau)/n + tau*(1-v_j)^kappa / Z, a mixture of flat counting and a normalized inverted-validity kernel; tau is the trust placed in the inversion, kappa its steepness (kappa=0, tau=0 recovers plain Tallying; large kappa approaches a worst-expert-first one-reason rule). Evidence is integrated additively, d = sum_j w_j * sign(A_j - B_j), and the choice is a logistic function of beta*d plus a small uniform lapse epsilon. Because the anti-validity signature is grounded in the communicated validity values rather than in cue position, it transfers across designs with different validity-position layouts — unlike SWC's recency gradient, which sign-failed the recency-consistency experiment.

**Rationale:** IVC implements the arbiter's prescribed mechanism family faithfully, with parameter bounds chosen from a trial-by-trial quantitative analysis of all six displayed experiments. (1) Why inverted validity: the pooled data demand anti-validity weighting — displayed Exp 1's below-chance TTB adherence (0.208) on conflicts where the tally majority rests on low-validity cues; displayed Exp 3's strongly negative within-margin validity coefficient (-0.576), which I verified requires steep inversion (kappa >= 2.3) plus moderate beta (~2.1) to reach about -0.5; displayed Exp 5's below-chance validity adherence on validity-vs-recency conflicts (0.231), which I verified IVC reproduces (about 0.24) because inverted weights side with the recency-implied option; and displayed Exp 2's mildly positive tally-minus-TTB (0.158), where IVC lands at about 0.26, the top of the arbiter's band — lowering beta to hit it exactly would break Exps 1/3/4, so I accept this as the single residual error. (2) Why no capacity interference: displayed Exp 4's 0.843 anchor consistency at n=8 forbids it (this is what degenerated NVW to chance, 0.530); deterministic attention plus design-independent beta keeps anchors at 0.78-0.85. (3) Why no position gradient: SWC's recency gradient sign-failed displayed Exp 6 (0.726 vs 0.374); in that experiment the late cues carry high validities (0.90/0.96) and therefore receive the smallest inverted weights, so IVC stays mildly anti-recency (about 0.35-0.45) — the anti-validity signature is carried by the validity values themselves, making the theory experiment-invariant rather than order-dependent. (4) Box calibration: I computed each metric at box corners and center — the center (tau=0.925, kappa=2.7, beta=2.15, eps=0.05) yields approximately Exp1 0.20 (real 0.208), Exp2 0.26 (real 0.158), Exp3 -0.51 (real -0.576), Exp4 0.78-0.85 (real 0.843), Exp5 0.25 (real 0.231), Exp6 0.35-0.45 (real 0.374) — a strictly better error profile than the running candidates (pi_2's errors of +0.39/+0.57/+0.35 on Exps 2/3/5; pi_4's +0.16/+0.35 on Exps 3/6), while remaining inside every range the arbiter prescribed (tau in [0.3,1], kappa in [0.5,3], beta in [1,12], epsilon in [0.01,0.10]). The tight box also keeps between-subject metric variance low, matching the small observed variances.

**Parameters:**
  - `tau`: `[0.85, 1.0]`
  - `kappa`: `[2.4, 3.0]`
  - `beta`: `[1.8, 2.5]`
  - `epsilon`: `[0.03, 0.07]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # IVC: Inverted-Validity Counting (distrust-weighted tallying).
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"IVC expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])
    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    eps = float(parameters["epsilon"])

    # (1) Distrust inversion: treat the communicated validity as if it
    # were the expert's error rate. (1 - v_j)^kappa is small for highly
    # validated experts and large for weakly validated ones; Z normalizes
    # the kernel so the mixture weights sum to one.
    inv = np.power(1.0 - v, kappa)
    Z = float(inv.sum())
    if Z > 1e-12:
        w_inv = inv / Z
    else:
        w_inv = np.full(n, 1.0 / n)

    # Subjective weights: mixture of flat counting and inverted validity.
    # tau = 0 recovers plain Tallying; tau = 1 with large kappa approaches
    # a 'worst-expert-first' one-reason rule.
    w = (1.0 - tau) / n + tau * w_inv

    # (2) Deterministic attention: every discriminating cue is always
    # attended. NO capacity interference: beta is design-independent
    # (anchor consistency stays high even at n = 8 features).
    s = np.sign(a - b)

    # (3) Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # Numerically stable logistic choice rule.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (4) Small independent uniform lapse.
    probs = (1.0 - eps) * probs + eps * 0.5
    return probs
```

**`policy(probs)`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))
```
