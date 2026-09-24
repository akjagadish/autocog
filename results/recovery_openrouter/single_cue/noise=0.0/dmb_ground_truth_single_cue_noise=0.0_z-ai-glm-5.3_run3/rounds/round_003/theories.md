# Round 3 — Theories

**Verdict:** `new_model` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5` — KILLED ✗

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

### `pi_5_1` → slot 1 (via `new_model`)

**Description:** INVERTED-VALIDITY COUNTING (IVC) — distrust-weighted tallying. Subjects do read the communicated validities but map them through a systematic distrust inversion: they act as if each stated validity were the expert's error rate, so the most highly validated experts are discounted most. Every discriminating cue is attended deterministically (no attention lapses, no working-memory capacity interference — choice consistency is design-independent). Each cue j carries subjective weight w_j = (1-tau)/n + tau*(1-v_j)^kappa / Z, a mixture of flat counting and a normalized inverted-validity kernel; tau is the trust placed in the inversion, kappa its steepness (kappa=0, tau=0 recovers plain Tallying; large kappa approaches a worst-expert-first one-reason rule). Evidence is integrated additively, d = sum_j w_j * sign(A_j - B_j), and the choice is a logistic function of beta*d plus a small uniform lapse epsilon. Because the anti-validity signature is grounded in the communicated validity values rather than in cue position, it transfers across designs with different validity-position layouts — unlike SWC's recency gradient, which sign-failed the recency-consistency experiment.

**Rationale:** This is a minimal, purely parametric edit on the accepted iter-5 base, implementing the critic's iter-6 kernel-alignment diagnosis exactly. The critic's key insight is that the remaining residuals cannot be fixed by noise: Exp2 (+0.160, dominant) and Exp7 (-0.059) read out determinism in OPPOSITE directions through comparable |d| ranges, so the fix must come from the kernel shape. The diagnostic facts: (a) Exp7's metric scores adherence to a FIXED reference kernel (tau=0.925, kappa=2.7, uncentered), and the candidate's median kernel (gamma~3.2 with partial centering) is systematically steeper than that reference — on borderline Exp7 trials the candidate's d flips sign relative to the reference, leaking adherence at fixed noise; (b) gamma acts strongly on the spread-error designs (Exp1/2/3/5, error rates 0.05..0.5) where the 'too inversion-extreme' residuals live, and barely on Exp7's saturated bimodal error structure. EDIT 1: gamma lowered from [2.2, 4.2] to [2.3, 3.2] (median ~2.75, aligned with the kappa=2.7 reference). This raises Exp7 adherence via better sign agreement with the scored kernel WITHOUT another beta raise, while simultaneously softening the inversion on Exps 1/2/3/5 — all currently too extreme — and leaving Exp4 anchors untouched (d = ±1 when all cues agree, so only beta/eps touch anchors). EDIT 2: c trimmed from [0.0, 0.35] to [0.0, 0.2]: centering steepens the effective kernel, so trimming it reinforces the same alignment on Exp7 and further softens the inversion on Exp3/5/8. EDIT 3: log10_beta partially retreated from [0.35, 0.80] to [0.25, 0.70] (median ~3.0) — explicitly the midpoint between iter-3's [0.15, 0.70] and iter-5's [0.35, 0.80], not a revert: Exp2 and Exp3 both moved away from target after the iter-5 beta raise, so some retreat is required, but going below 0.25 would undo the real Exp4/Exp7 gains. EDIT 4: tau's lower bound split from 0.88 to 0.83 (midpoint of 0.78 and 0.88): the 0.88 raise amplified the inverted kernel's share and is a suspected co-contributor (with beta) to the Exp3/Exp5/Exp1 overshoots, but Exp7 did benefit — split the difference rather than full revert. HELD: epsilon at [0.08, 0.22] per the critic's explicit instruction to preserve the working beta/eps decoupling (Exp4 anchors landed at 0.8417 vs 0.8433 — essentially perfect); per-subject sampling scheme unchanged; per-cue bias term still omitted. Expectation management per the critic: Exp2 has never landed below 0.233 in any IVC configuration across five iterations, so it will likely land ~0.22-0.25 rather than 0.158 — that ~+0.07 is accepted as the in-family floor and NOT chased with noise increases that would re-break the Exp4/Exp7 pair. Projected landing: Exp1 ~0.19, Exp2 ~0.24, Exp3 ~-0.60, Exp4 ~0.84, Exp5 ~0.21, Exp6 ~0.40, Exp7 ~0.75-0.78, Exp8 ~0.35 — a net loss clearly below the 0.1047 floor, with no experiment materially worse than iter-5's residuals and five of eight improving.

**Parameters:**
  - `tau`: `[0.83, 1.0]`
  - `c`: `[0.0, 0.2]`
  - `gamma`: `[2.3, 3.2]`
  - `log10_beta`: `[0.25, 0.7]`
  - `epsilon`: `[0.08, 0.22]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # IVC: Inverted-Validity Counting (distrust-weighted tallying),
    # PARTIAL-CENTERING edition, iter-6 kernel-alignment retune.
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
    gamma = float(parameters["gamma"])
    c = float(parameters["c"])
    # log-uniform (heavy-tailed) subject-level inverse temperature.
    # Partially retreated to [0.25, 0.70] (beta in [1.8, 5.0], median ~3.0):
    # the iter-5 raise to [0.35, 0.80] fixed the Exp4/Exp7 pair but
    # overshot the moderate-|d| regime (Exp2 contrast and Exp3 coefficient
    # both moved AWAY from target). This is the midpoint between iter-3's
    # [0.15, 0.70] and iter-5's [0.35, 0.80] -- a partial retreat, not a
    # revert -- so the Exp4/Exp7 gains are largely preserved while the
    # spread-error designs (Exp1/2/3/5) recover.
    beta = 10.0 ** float(parameters["log10_beta"])
    eps = float(parameters["epsilon"])

    # (1) PARTIAL distrust inversion. The stated validity is mapped to an
    # error rate (1 - v_j); a FRACTION c of the smallest error rate is
    # subtracted before the gamma power:
    #     inv_j = ((1 - v_j) - c * min_k(1 - v_k))^gamma / Z
    # c = 0 recovers the uncentered kernel; c = 1 is hard centering.
    # c TRIMMED to [0.0, 0.2]: centering steepens the effective kernel;
    # trimming it moves the candidate toward the uncentered kappa=2.7
    # reference kernel that Exp7's metric scores against, and further
    # softens the inversion on the spread-error designs (Exp3/5/8).
    err = 1.0 - v
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Z = float(inv.sum())
    if Z > 1e-12:
        w_inv = inv / Z
    else:
        # Degenerate design (all validities equal): fall back to flat tally.
        w_inv = np.full(n, 1.0 / n)

    # Subjective weights: mixture of flat counting and the partially
    # centered inverted-validity kernel. tau's lower bound SPLIT to 0.83
    # (midpoint of iter-3's 0.78 and iter-5's 0.88): the 0.88 raise was a
    # suspected co-contributor (with beta) to the Exp3/Exp5/Exp1
    # overshoots, but Exp7 did benefit, so split the difference.
    w = (1.0 - tau) / n + tau * w_inv

    # (2) Deterministic attention: every discriminating cue is always
    # attended. NO capacity interference: beta is design-independent.
    s = np.sign(a - b)

    # (3) Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # Numerically stable logistic choice rule.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (4) Independent uniform lapse, HELD at [0.08, 0.22]: the beta/eps
    # decoupling from iter-5 is working (Exp4 anchors at 0.8417 vs 0.8433)
    # and must be preserved; if the beta trim drops Exp4 slightly, the
    # lapse continues to carry the anchor cap.
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
