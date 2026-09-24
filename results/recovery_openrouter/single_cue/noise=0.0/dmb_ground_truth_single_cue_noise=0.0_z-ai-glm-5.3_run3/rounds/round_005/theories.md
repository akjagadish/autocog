# Round 5 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_5_1` — KILLED ✗

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


### slot 2 — `pi_6` — SURVIVED ✓

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


## Replacement

### `pi_7` → slot 1 (via `new_theory`)

**Description:** PRIMACY-GATED DISTRUST TALLYING (PGDT), floored-gate edition with a rebalanced distrust share. Subjects scan the expert ratings in presentation order and attention is deployed through a steep convex primacy gate with a substantial nonzero floor: g_j = delta + (1-delta)*((n-j)/(n-1))^eta, so the first one or two experts carry the dominant share of attention but late experts retain a meaningful attention residue. Independently, the communicated validities are distrust-inverted exactly as in the shared kernel: subjects act as if each stated validity were the expert's error rate, inv_j = ((1-v_j) - c*min_k(1-v_k))^gamma / Z. The two kernels combine MULTIPLICATIVELY: w_j proportional to g_j * [(1-tau)/n + tau*inv_j], normalized. The refinement this iteration is a rebalanced distrust share: tau is lowered to [0.82, 0.92] so the flat component (1-tau)/n of the multiplicative bracket grows from ~1.5% to ~4-5% per cue. This directly feeds trusted high-validity LATE experts (whose bracket value is structurally pinned at the flat share regardless of gate shape), softens the distrust-driven allegiance contrasts that overshoot their targets, and pulls the effective weights off pure anti-validity where that overshoots. A modestly raised gate floor delta in [0.20, 0.45] compensates on decreasing-validity layouts by further concentrating the product weights onto the back-loaded low-validity experts, protecting the anti-validity regression. Evidence is integrated additively over all discriminating cues, d = sum_j w_j * sign(A_j - B_j), and choice is logistic in beta*d with a small uniform lapse epsilon.

**Rationale:** Minimal-diff edit per the critic's iter-3 directive: two parameter-range changes on the already-accepted floored-gate PGDT base, mechanism and code path identical. PRIMARY KNOB: tau lowered from [0.90, 0.97] to [0.82, 0.92] (inside the arbiter's prescribed [0.80, 0.97]). The iter-3 diagnosis identified a structural impasse the delta knob cannot break: for a high-validity late cue the multiplicative bracket is pinned at (1-tau)/n ~= 0.01-0.02 when tau >= 0.90, so no amount of gate flattening can rescue it -- this is exactly why Exp8 (0.278 vs 0.396), Exp6 (0.303 vs 0.374) and Exp1 (0.141 vs 0.208) barely moved across the iter-2 to iter-3 delta raise despite hand-checks predicting movement. Lowering tau grows the flat share to ~4-5%, directly feeding Exp8's final 0.90-validity expert, softening the over-committed distrust allegiances (Exp2 +0.162, Exp7 +0.083, Exp4 +0.056 -- the single largest remaining miss cluster), and pulling Exp1 back up off pure anti-validity. SECONDARY (compensating) KNOB: delta ceiling raised from 0.40 to 0.45 with floor 0.20, because the tau drop weakens the distrust kernel's dominance on decreasing-validity layouts and Exp3 (still +0.107 short of -0.576) needs the back-loading strengthened; the iter-2 to iter-3 trajectory verified delta's strong leverage there (-0.318 to -0.469). Cautions from the critique are honored: (i) Exp11 (0.234 vs 0.315) may be squeezed by the delta bump -- if it falls below ~0.20 the trade is accepted, since the combined Exp2+Exp8 misses (0.28) outweigh Exp11's (0.08) and every family theory except pi_1 undershoots Exp11 anyway; eta's ceiling stays at 6 so steep-gate subjects remain available to protect it. (ii) Exp12 is now matched (0.546 vs 0.540); if lowering tau pushes its slope above ~0.60, eta should be pulled toward 4 rather than reverting tau. (iii) Exp2 is treated as a stretch target: banking anything in the 0.20-0.28 range is a win given pi_3's 0.166 is the family best. gamma, c, beta, epsilon are frozen exactly as directed -- the distrust kernel's sharpness (gamma) is what the Exp3/Exp5/Exp7 regime depends on, and the accept gate protects against any regression. Both previous single-knob directives produced accepted candidates (loss 0.3072 -> 0.1413 -> 0.1175); this two-knob edit follows the same verified playbook, targeting the two coherent residual clusters (over-committed allegiances, starved trusted late cues) that the iter-3 diagnosis showed share a single structural cause in tau's dominance share.

**Parameters:**
  - `tau`: `[0.82, 0.92]`
  - `eta`: `[4.0, 6.0]`
  - `delta`: `[0.20, 0.45]`
  - `gamma`: `[2.8, 3.5]`
  - `c`: `[0.10, 0.25]`
  - `beta`: `[2.5, 6.0]`
  - `epsilon`: `[0.06, 0.15]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # PGDT: Primacy-Gated Distrust Tallying, FLOORED-GATE edition,
    # REBALANCED-DISTRUST-SHARE variant.
    # iter-4 minimal-diff edit per critic's iter-3 directive:
    #   (1) PRIMARY KNOB: LOWER TAU. Range moved from [0.90, 0.97] to
    #       [0.82, 0.92] (still inside the arbiter's prescribed
    #       [0.80, 0.97]). This grows the flat share (1-tau)/n of the
    #       multiplicative bracket from ~1.5% to ~4-5% per cue, which is
    #       the ONLY lever that can move Exp8's final 0.90-validity
    #       expert (its weight is structurally pinned at the flat share
    #       regardless of gate shape), and simultaneously softens the
    #       over-committed distrust allegiances on Exp2/Exp4/Exp7.
    #   (2) COMPENSATING KNOB: RAISE DELTA CEILING. Range moved from
    #       [0.15, 0.40] to [0.20, 0.45] so the iter-2->3 trajectory's
    #       verified leverage on Exp3's back-loading (stronger
    #       anti-validity as the early-gate flattens) compensates the
    #       Exp3 weakening that the tau drop would otherwise cause.
    #   (3) Everything else frozen: eta, gamma, c, beta, epsilon.
    # Mechanism structure unchanged from the accepted iter-3 base.
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PGDT expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])      # trust in the distrust kernel
    eta = float(parameters["eta"])       # convexity of the primacy gate
    delta = float(parameters["delta"])  # floor of the primacy gate
    gamma = float(parameters["gamma"])   # steepness of distrust inversion
    c = float(parameters["c"])           # partial centering of error rates
    beta = float(parameters["beta"])     # choice consistency
    eps = float(parameters["epsilon"])  # uniform lapse

    # (1) DISTRUST INVERSION with partial centering: treat the stated
    # validity as an error rate; subtract a fraction c of the smallest
    # error rate before raising to the gamma power, then normalize.
    err = 1.0 - v
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Z_v = float(inv.sum())
    if Z_v > 1e-12:
        w_inv = inv / Z_v
    else:
        # Degenerate design (all validities tied at the extreme): the
        # distrust kernel is flat and the gate alone carries position.
        w_inv = np.full(n, 1.0 / n)

    # (2) STEEP CONVEX PRIMACY GATE WITH RAISED FLOOR over presentation
    # positions. Cue j (0-indexed) gets gate mass
    #     g_j = delta + (1-delta) * ((n - j)/(n - 1))^eta
    # with delta now in [0.20, 0.45]: the FIRST position still carries
    # the dominant share, but every late position retains a substantial
    # attention residue, so the multiplicative product never approaches
    # a pure primacy gradient and late low-validity (maximum-distrust)
    # cues keep enough weight to carry the anti-validity signature.
    # The raised ceiling further back-loads the product on decreasing-
    # validity layouts, compensating the tau drop for Exp3/Exp5.
    if n > 1:
        pos = np.arange(n, dtype=float)
        g = delta + (1.0 - delta) * np.power((n - pos) / (n - 1.0), eta)
    else:
        g = np.ones(1)

    # (3) MULTIPLICATIVE combination: the gate scales the distrust
    # bracket cue-by-cue. Where the bracket is flat (tied validities)
    # this reduces to the (floored) primacy gradient; where the bracket
    # nearly vanishes (trusted high-validity early experts) the gate
    # multiplies almost nothing and the distrust kernel decides where
    # the remaining weight mass sits. With tau lowered, the flat share
    # (1-tau)/n is large enough that trusted high-validity LATE experts
    # retain meaningful weight through the gate's floor residue -- the
    # structural fix for the under-weighted late-cue cluster (Exp6/8)
    # that no amount of gate flattening could rescue at tau >= 0.90.
    bracket = (1.0 - tau) / n + tau * w_inv
    raw = g * bracket
    Z_t = float(raw.sum())
    if Z_t > 1e-12:
        w = raw / Z_t
    else:
        w = np.full(n, 1.0 / n)

    # (4) Deterministic attention: every discriminating cue is always
    # attended; no capacity interference, beta is design-independent.
    s = np.sign(a - b)

    # (5) Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # Numerically stable logistic choice rule.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (6) Small uniform lapse; kept below IVC's range because the gate
    # itself supplies response dilution on late-cue trials.
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
