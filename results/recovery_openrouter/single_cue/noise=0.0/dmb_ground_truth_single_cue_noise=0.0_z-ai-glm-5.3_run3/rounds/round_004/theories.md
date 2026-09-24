# Round 4 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_5_1` — SURVIVED ✓

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


### slot 2 — `pi_4` — KILLED ✗

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

### `pi_6` → slot 2 (via `new_theory`)

**Description:** PRIMACY-WEIGHTED INVERTED-VALIDITY COUNTING (PW-IVC). Subjects read the communicated validities but systematically distrust-invert them: they act as if each stated validity were the expert's error rate, so highly validated experts are discounted most (the anti-validity signature). Independently and additively, subjective attention is tilted toward EARLY-presented cues by a steep convex primacy gradient concentrated on the first one or two positions, at a small-to-moderate share of the subjective-weight mixture so the distrust kernel retains dominance wherever validity information varies. Each cue j carries weight w_j = (1-tau)/n + tau * [ (1-rho) * ((1-v_j) - c*min_k(1-v_k))^gamma / Z_v + rho * ((n-j)/(n-1))^eta / Z_p ]; evidence d = sum_j w_j * sign(A_j - B_j) is integrated additively over all discriminating cues; choice is logistic in beta*d with a small uniform lapse epsilon. On tied-validity designs the distrust kernel is exactly flat and the primacy kernel alone supplies the early-vs-late contrast, producing below-chance recency-following; on decreasing-validity layouts the distrust kernel dominates and reproduces the anti-validity regression signature.

**Rationale:** Minimal-diff edit exactly per the critic's iter-3 directive; the PW-IVC mechanism and all code are unchanged from the accepted iter-3 base, and only three parameter-box knobs move. (1) rho: [0.16, 0.34] -> [0.16, 0.30]. The iter-3 ceiling raise is what regressed Exp3 from -0.434 to -0.372 (squared error ~0.042, now the dominant loss term): in the strictly-decreasing-validity layout, primacy mass (pro-early = pro-high-validity) directly cancels the distrust kernel, and the fitter was buying Exp9/Exp10 gains with primacy mass at Exp3's expense. The modest ceiling trim — placed between iter-2's 0.28 and iter-3's 0.34, not back at either, to avoid the oscillation the critic warns about — restores distrust dominance in decreasing-validity layouts. (2) beta: [2.2, 6.5] -> [2.2, 8.0], and epsilon floor: 0.08 -> 0.06. These are the commitment knobs that COMPENSATE the tied-validity experiments for the rho trim: on Exp9's count-neutral trials the anti-recency deviation from 0.5 scales roughly with beta*rho at small evidence, so a ~12% rho cut is offset by the ~23% beta headroom, while on Exps 3/5/7/10 the distrust evidence already points the right way and sharper commitment pushes each under-signed metric toward its target (Exp3 more negative toward -0.576, Exp5 toward 0.231, Exp7 up toward 0.788, Exp10 toward -0.173). The lower epsilon floor additionally de-attenuates the Exp3 OLS coefficient, which the lapse shrinks toward zero by roughly (1-eps). (3) Everything else frozen: tau [0.88, 0.97], eta [3.5, 6.5], gamma [2.9, 3.8], c [0.0, 0.2] — all worked in both prior accepted iterations. Guardrails per the critic: Exp4's anchor is lapse-capped (adherence ~ 1 - eps/2 at saturation, so the lowered epsilon floor could push it up — the ceiling 0.22 keeps the mean near 0.84, and if Exp4 exceeds 0.88 the epsilon floor should be restored to 0.10 rather than capping beta); Exp8 is accepted as the family residual (all distrust-flavored theories undershoot the real 0.396) and is monitored but not purchased at the cost of the signature experiments; Exp2 (currently +0.055) is monitored — higher beta sharpens its tally-vs-TTB contrast in the wrong direction, but the signature experiments (3, 5, 7, 9, 10) take priority, exactly as directed. The iter-1 -> iter-3 trajectory (Exp3 error +0.51 -> +0.14 -> +0.20, aggregate 0.151 -> 0.101 -> 0.099) confirms the structure is right; this final coordinated nudge on commitment level and the rho ceiling targets a loss below the 0.0989 floor.

**Parameters:**
  - `tau`: `[0.88, 0.97]`
  - `rho`: `[0.16, 0.30]`
  - `eta`: `[3.5, 6.5]`
  - `gamma`: `[2.9, 3.8]`
  - `c`: `[0.0, 0.2]`
  - `beta`: `[2.2, 8.0]`
  - `epsilon`: `[0.06, 0.22]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # PW-IVC: Primacy-Weighted Inverted-Validity Counting.
    # iter-4 minimal-diff edit per critic's iter-3 directive:
    #   (1) rho ceiling trimmed 0.34 -> 0.30 (restores distrust dominance in
    #       strictly-decreasing-validity layouts; the iter-3 ceiling raise is
    #       what regressed Exp3 from -0.434 to -0.372);
    #   (2) beta ceiling raised 6.5 -> 8.0 and epsilon floor lowered
    #       0.08 -> 0.06 (the commitment knobs that compensate the
    #       tied-validity experiments for the rho trim and de-attenuate the
    #       Exp3 OLS coefficient);
    #   (3) everything else frozen (tau, eta, gamma, c).
    # Mechanism structure unchanged from the accepted iter-3 base.
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PW-IVC expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])      # trust in the non-flat kernel mixture
    rho = float(parameters["rho"])        # primacy share within the mixture
    eta = float(parameters["eta"])       # convexity of the primacy gradient
    gamma = float(parameters["gamma"])   # steepness of distrust inversion
    c = float(parameters["c"])           # partial centering of error rates
    beta = float(parameters["beta"])
    eps = float(parameters["epsilon"])

    # (1) DISTRUST INVERSION with partial centering: treat the stated
    # validity as an error rate; subtract a fraction c of the smallest
    # error rate before raising to the gamma power, then normalize.
    #     inv_j = ((1 - v_j) - c * min_k(1 - v_k))^gamma / Z_v
    # When all validities are tied this kernel is flat (or degenerate at
    # c -> 1, where we fall back to flat), so tied-validity designs are
    # carried entirely by the primacy kernel below.
    err = 1.0 - v
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Z_v = float(inv.sum())
    if Z_v > 1e-12:
        w_inv = inv / Z_v
    else:
        w_inv = np.full(n, 1.0 / n)

    # (2) STEEP CONVEX PRIMACY GRADIENT over presentation positions: cue j
    # gets raw primacy mass ((n - j)/(n - 1))^eta with eta >= 3.5, so the
    # gradient is concentrated on the FIRST one or two positions. This
    # maximizes the early-vs-late weight CONTRAST per unit of mixture
    # share rho (what the anti-recency metrics score), while the small-to-
    # moderate rho keeps the primacy kernel from fighting the distrust
    # kernel on decreasing-validity layouts (Exps 3, 5, 8).
    if n > 1:
        pos = np.arange(n, dtype=float)
        prim = np.power((n - pos) / (n - 1.0), eta)
    else:
        prim = np.ones(1)
    Z_p = float(prim.sum())
    w_prim = prim / Z_p

    # (3) Subjective weights: mixture of flat counting, the partially
    # centered inverted-validity kernel, and the primacy kernel.
    # tau ~ 0.9+ keeps the non-flat kernels dominant; rho <= 0.30 keeps
    # the distrust kernel the dominant carrier of validity variance.
    w = (1.0 - tau) / n + tau * ((1.0 - rho) * w_inv + rho * w_prim)

    # (4) Deterministic attention: every discriminating cue is always
    # attended; no capacity interference, beta is design-independent.
    s = np.sign(a - b)

    # (5) Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # Numerically stable logistic choice rule.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (6) Small independent uniform lapse, capping anchor adherence.
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
