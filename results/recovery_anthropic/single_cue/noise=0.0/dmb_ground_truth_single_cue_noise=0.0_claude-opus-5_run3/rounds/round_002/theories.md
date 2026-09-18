# Round 2 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_4` — KILLED ✗

**Description:** **Threshold-terminated compensatory integration with CONCLUSION-GRADED commitment.**

People treat expert ratings as evidence to be added up, not as a lexicographic hierarchy. Five claims:

1. **Cue weights depend ONLY on the communicated validity, never on where the rating sits on the screen.** The weight of expert j is a compressed, monotone function of its stated validity, w_j ∝ (v_j − 0.5)^rho, normalised to sum to 1. Compression (rho ≈ 1.25–1.45) lets a crowd of medium-validity experts out-vote or tie a single .95 expert.

2. **Integration is compensatory and additive, carried out incrementally in the order the ratings are displayed, and it can terminate early.** After each rating that actually discriminates, the decision maker performs a *best-case reversal check*: "could any single expert I have not yet read still overturn what I now believe?" The sufficiency bar is theta_k = phi · max_{j after the current screen position} w_j, and the probability of stopping is sigmoid(s·(|E_k|/theta_k − 1)). With phi ≈ 1 this is the rational rule "stop when no single remaining expert can reverse my lead". Nothing is discounted by position; position matters *only* through this truncation.

3. **NEW — confidence is conclusion-graded, not merely margin-graded.** Two psychologically distinct states can end a trial. (a) The reversal check *passes*: the person has reached a categorical conclusion ("nothing still unread can overturn this"), and choice is committed sharply — softmax on the raw margin with an amplified sharpness beta·kappa (kappa > 1). Crucially, *which* cue triggered the stop is then almost irrelevant: a mid-validity cue that has already cleared the sufficiency bar is followed about as decisively as the top cue is. (b) The list is *exhausted* without the check ever passing: the person never reached a conclusion and is left holding a raw, ungrounded residual margin; choice is then a shallow softmax on that margin with the ordinary beta, so unresolved trials sit close to chance and are only weakly tilted by the residual. Two trials with identical final margins therefore differ in decisiveness depending on whether the stopping criterion was ever met — the theory's sharpest new prediction.

4. **No renormalisation, no side bias, plus a lapse.** Evidence is never divided by the total discriminating weight (that would make one-cue and many-cue decisions equally confident); a lapse epsilon mixes in a uniform guess.

5. **Individual differences are unimodal jitter** in compression (rho), conservatism (phi), stopping sharpness (s), commitment amplification (kappa), residual sensitivity (beta) and lapse — no discrete strategy types, no primacy/recency "reader types".

Signature predictions: in a validity-sorted display the leading cue clears the bar immediately, search stops, and behaviour looks like Take-The-Best in BOTH conflict and agreement cells — and because stopping commitment is amplified, cells whose first cues tie and which stop on a *middle* cue are followed just as decisively, so the TTB-vs-tally contrast collapses to ≈0 instead of going negative. When the best expert is listed last, the bar stays high until a mid-validity crowd piles past it, search stops before the best expert is read, and behaviour looks like tallying. When a strong cue sits mid-list with another strong cue still unread, the check fails, integration runs to the end, and the shallow residual softmax leaves the choice near chance.

**Rationale:** **Minimal diff.** Everything is re-emitted verbatim from the accepted iter-3 base (compressed validity-only weights, incremental read in screen order, suffix-max reversal check, softmax on the RAW unnormalised margin, lapse, no side bias, no KAPPA floor, no sum-form bar, no position-dependent weights, phi=[1.00,1.12], s=[12,30], rho=[1.25,1.45] all RESTORED exactly as the critic demanded). Exactly ONE line of mechanism is added inside the loop: the read-out sharpness is `beta*kappa` when the trial ends because the *sufficiency test passed* (an early, conclusive stop) and stays `beta` when the trial ends because the *list was exhausted*. Parameter `kappa` is new; `beta` is re-ranged from [8,14] to [5,10] because kappa changes what beta means (it now governs ONLY unresolved, read-to-the-end trials).

**Why this implements the critic's diagnosis better than raising beta globally.** The critic located the whole Exp1 residual precisely: with w≈(0.352,0.281,0.186,0.134,0.046), the four single-top-cue conflict cells and all four agreement cells stop at position 0 and read out at sigmoid(beta·0.352)≈0.98, while the two tie-tie conflict cells (7 and 9) stop at position 2 and read out at only sigmoid(beta·0.186)≈0.89 — that 0.09 shortfall on 2 of 6 conflict cells IS the −0.03 metric. The critic's fix (beta → [16,22]) closes that gap but simultaneously sharpens every *exhausted-list* trial — and the counted cells of Exps 2, 3 and 4 are exactly the exhausted-list trials (I verified cell-by-cell: Exp2 cells 9/10 read all six cues, Exp3's double-dissociation cells read positions 0,3,5 with stop-prob <0.1 at position 3, Exp4 cells 1/2/7/8 have stop probs <0.005 at every step). Those three metrics all sit near chance and all have small residual margins pointing slightly the WRONG way, so a global beta increase drives Exp2 to ~0.76, Exp4 to ~0.487 and (per my reconstruction of Exp3's weights from the iter1↔iter3 outputs) Exp3 down to ~0.44. Splitting the temperature decouples the two populations of cells: kappa fixes Exp1 exactly where the critic said the damage is, and the *lower* beta pushes the three near-chance, unresolved metrics toward 0.5, which is where the humans are.

**Cell-by-cell simulation before shipping (rho=1.35, phi=1.05, s=20, beta=7.5, kappa=4 ⇒ beta_c=30, eps=0.03).**
• Exp1: agreement cells 5/6/11/12 stop at pos0 → 1.00; conflict cells 1/2/8/10 stop at pos0 with q=0.979 (leak 0.021 flips) → 0.979; conflict cells 7/9 stop at pos2 and now commit at sigmoid(30·0.186)=0.996 instead of 0.89. Metric = (4·0.979+2·0.996)/6 − 1.00 ≈ **−0.015** (was −0.0356; realistic in-family ceiling ≈ 0, since pure TTB itself scores only 0.004 here).
• Exp2: counted cells 1/2/3 — every early stop (pos0 .9-cue, pos1, pos2) lies on the TALLY side, so kappa pushes them to ≈1.00; cells 9/10 are exhausted-list, 1−sigmoid(7.5·0.0207)=0.461. Metric = (3·1.00+2·0.461)/5 → lapse-shrunk ≈ **0.776** (target 0.775, was 0.7695).
• Exp3: stop prob at the .95 cue is 0.06 (unchanged — phi/s untouched); that branch now commits at ≈1.0 instead of 0.93, and the dominant exhausted branch moves from sigmoid(−11·0.0205)=0.444 to sigmoid(−7.5·0.0205)=0.462 ⇒ **≈0.50** (target 0.5175, was 0.485).
• Exp4: both counted conflicts are exhausted-list; mean(sigmoid(7.5·0.0327), sigmoid(−7.5·0.0394)) = (0.561+0.427)/2 = **0.494** (target 0.5075, was 0.4913).
All four land inside the critic's acceptance bands except Exp1, which is at the structural ceiling of this family. Predicted aggregate loss ≈ 0.044 vs the 0.0573 floor — and, importantly, no metric is traded away: all four residuals shrink or hold.

**Experiment invariance.** The new ingredient is a single scale-free claim (conclusive stops are committed, unresolved read-outs are shallow) with no hand-set constants and no reference to feature counts, screen layout, or a particular validity vector; it transfers unchanged to any n_features/validity design.

**Parameters:**
  - `rho`: `[1.25, 1.45]`
  - `phi`: `[1.00, 1.12]`
  - `s`: `[12.0, 30.0]`
  - `beta`: `[5.0, 10.0]`
  - `kappa`: `[3.0, 5.0]`
  - `epsilon`: `[0.0, 0.06]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ------------------------------------------------------------------
    # 1. parse the trial stimulus into two rating vectors (screen order)
    # ------------------------------------------------------------------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(list(state['option_a_ratings']), dtype=float)
            b = np.asarray(list(state['option_b_ratings']), dtype=float)
    if a is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(2, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(2, -1)
        if arr.shape[0] != 2:
            arr = arr.reshape(2, -1)
        a = arr[0].astype(float)
        b = arr[1].astype(float)

    a = np.ravel(a).astype(float)
    b = np.ravel(b).astype(float)
    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.array([0.5, 0.5])
    a = a[:n]
    b = b[:n]
    d = a - b

    # ------------------------------------------------------------------
    # 2. validity-only cue weights, compressed toward equality
    # ------------------------------------------------------------------
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        # fall back: assume display is validity-sorted, descending
        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])
    val = np.clip(val, 0.5 + 1e-6, 1.0)

    rho = float(parameters.get('rho', 1.35))
    rho = float(np.clip(rho, 0.1, 5.0))
    raw = np.clip(val - 0.5, 1e-9, None) ** rho
    tot = float(np.sum(raw))
    if not np.isfinite(tot) or tot <= 0:
        w = np.ones(n, dtype=float) / n
    else:
        w = raw / tot                      # weights sum to 1, position-free

    phi = float(np.clip(parameters.get('phi', 1.05), 0.05, 3.0))
    s = float(np.clip(parameters.get('s', 20.0), 0.1, 200.0))
    beta = float(np.clip(parameters.get('beta', 7.5), 0.0, 200.0))
    kappa = float(np.clip(parameters.get('kappa', 4.0), 1.0, 20.0))
    eps = float(np.clip(parameters.get('epsilon', 0.0), 0.0, 1.0))

    # ------------------------------------------------------------------
    # 3. incremental integration in screen order with graded stopping
    #    sufficiency bar = phi * (largest weight among cues NOT YET READ)
    #    i.e. "can any single remaining expert still overturn my lead?"
    #    Commitment is CONCLUSION-GRADED: a stop that passes the check is
    #    read out with sharpness beta*kappa; an exhausted list (no
    #    conclusion ever reached) is read out with the shallow beta.
    # ------------------------------------------------------------------
    disc = np.nonzero(np.abs(d) > 1e-12)[0]     # discriminating cues, in order
    if disc.size == 0:
        return np.array([0.5, 0.5])

    steps = w[disc] * np.sign(d[disc])
    cum = np.cumsum(steps)                      # margin after each new reason
    m = cum.shape[0]

    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    p_a = 0.0
    remain = 1.0
    for k in range(m):
        E = float(cum[k])
        if k < m - 1:
            idx = int(disc[k])
            if idx + 1 < n:
                suffix_max = float(np.max(w[idx + 1:]))
            else:
                suffix_max = 0.0
            theta_k = max(phi * suffix_max, 1e-12)
            q = _sig(s * (abs(E) / theta_k - 1.0))   # graded termination
            p_choice = _sig(beta * kappa * E)        # committed conclusion
        else:
            q = 1.0                                  # nothing left to read
            p_choice = _sig(beta * E)                # unresolved residual
        p_a += remain * q * p_choice
        remain *= (1.0 - q)
        if remain <= 1e-12:
            break
    if remain > 1e-12:                           # numerical remainder
        p_a += remain * _sig(beta * float(cum[-1]))

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = (1.0 - eps) * p + eps * 0.5              # lapse, no side bias
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return p
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
```


### slot 2 — `pi_3` — SURVIVED ✓

**Description:** **Serial-position–driven, primacy/recency-graded evidence accumulation ("read-order scanning").**

People do not re-sort the expert ratings by the stated validities before deciding. They inspect the ratings in the order in which they appear on the screen (reading order) and their attention decays sharply across that inspection sequence, so the evidence carried by a cue is discounted geometrically by its *serial position*, not by its validity. Stated validities exert only a weak, partial pull on the inspection order (parameter gamma, small).

Two further claims:

1. **The gradient is steep but graded.** Weights fall off as w_j = exp(-lambda * rank_j) with |lambda| well above zero, so decisions are usually driven by the first cue(s) that discriminate — near-lexicographic but not strictly all-or-none: when the leading cues tie, the next-inspected cues take over, and when several later cues line up they can occasionally overturn a leading cue if the gradient is shallow.

2. **The gradient has a sign: most readers are primacy-oriented, a substantial minority are recency-oriented.** Reading a short list produces either an anchoring-on-the-first-item strategy (primacy, lambda > 0) or a last-impression-counts strategy (recency, lambda < 0). The population is bimodal: about one third of subjects anchor on the *last-read* cue. Near-zero gradients (i.e., genuine equal-weight tallying) are essentially absent — people do not integrate all six/five cues equally.

Evidence is evaluated *relatively*: the signed weighted difference is normalised by the total weight of the cues that actually discriminate, so a decision based on one early cue is as confident as a decision based on many, and only genuinely near-balanced weighted evidence produces hesitation. Choice is a softmax on this relative evidence with inverse temperature beta, plus a lapse epsilon.

Because validity order and screen order coincide in a validity-sorted display but dissociate in a scrambled display, this theory predicts TTB-looking behaviour in the former and tally-looking behaviour in the latter — without either heuristic being the actual mechanism. The causal driver is *where the cue sits on the screen*.

**Rationale:** **Why the incumbents fail.** pi_1 (validity-ordered TTB) reproduces Experiment 1 (metric ~0) but collapses in Experiment 2 (0.15 vs 0.775), because in Exp 2 the validities are scrambled relative to screen position and TTB's top cue (position 5, validity .95) systematically opposes the tally. pi_2 (Tallying) does the reverse (0.87 in Exp 2, -0.69 in Exp 1). Neither is experiment-invariant, and the key structural difference between the two experiments is precisely that Exp 1's display is validity-sorted while Exp 2's is not. That is the arbiter's diagnosis and it is exactly right: *screen position*, not validity, is the causal variable.

**What this model does mechanistically.** Weights decay geometrically along the inspection sequence, which is essentially the reading order (gamma <= 0.25 gives validities only a weak pull). In Exp 1 position order = validity order, so a primacy reader looks TTB-like (metric ~0). In Exp 2 a primacy reader always uses position 0 first; I verified by hand that on every one of the five qualifying conflict trials in Exp 2 (rows 1, 2, 3, 9, 10) the first *positionally* discriminating cue points at the **tally** winner, so primacy readers score ~1.0 there — i.e. they mimic Tallying without counting anything.

**Why the signed gradient (the novel ingredient).** A purely primacy model predicts exactly 0.0 in Exp 1, whereas humans are slightly *above* 0 (+0.093): they are marginally more TTB-consistent on conflict than on agree trials. No decay-only or one-reason model can produce a positive value; graded integration makes it negative. A recency-oriented reader (lambda < 0, anchoring on the last-read cue) yields exactly the required asymmetry: I computed that such a subject scores 1/3 on Exp-1 conflict trials but 0 on Exp-1 agree trials (metric = +1/3), and 0.4 on Exp-2 conflict trials. Mixing ~35% recency readers with ~65% primacy readers therefore predicts Exp 1 ~ +0.08 to +0.09 and Exp 2 ~ 0.76 to 0.78, hitting both observed values (0.093, 0.775) simultaneously — something no single-strategy model in the leaderboard can do. The bimodality claim (|lambda| bounded away from 0) is substantive: it asserts that true equal-weight integration is rare, which is what rules out the pi_2-style -0.69 in Exp 1.

**Relative-evidence normalisation.** Dividing the signed weighted sum by the total weight of discriminating cues means a decision resting on a single early cue is held with the same confidence as one resting on many, so trials whose leading cues tie (e.g. Exp 1 rows 7/9) do not degenerate into coin flips under a steep gradient. This keeps error rates roughly uniform across trial types and prevents the spurious negative bias that a raw-magnitude softmax would inject into the Exp-1 metric.

**Nesting and falsifiability.** lambda -> 0 recovers Tallying, lambda -> +inf with gamma -> 1 recovers TTB, gamma -> 0 gives pure reading-order scanning, and the sign of lambda captures primacy vs recency. The theory makes the sharp, testable prediction the arbiter asked for: scrambling the on-screen order of the same experts while holding the pair structure fixed should reverse choices on top-cue-vs-tally conflict pairs, whereas validity-based accounts predict invariance.

**Parameters:**
  - `lam`: `[1.2, 3.5]`
  - `gamma`: `[0.0, 0.25]`
  - `beta`: `[4.0, 12.0]`
  - `epsilon`: `[0.0, 0.10]`
  - `scan_orientation`: `[0, 1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---- unpack the trial stimulus -> (2, n_features) ------------------
    if isinstance(state, dict):
        a = np.asarray(list(state.get('option_a_ratings', [])), dtype=float)
        b = np.asarray(list(state.get('option_b_ratings', [])), dtype=float)
        stim = np.vstack([a, b])
    else:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            stim = stim.reshape(2, -1)
        elif stim.ndim > 2:
            stim = stim.reshape(2, -1)
    if stim.shape[0] != 2:
        stim = stim.reshape(2, -1)

    a = stim[0].astype(float)
    b = stim[1].astype(float)
    n = a.shape[0]
    d = a - b

    # ---- inspection ranks: screen position, weakly pulled by validity --
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        # fallback: assume the display is already validity-sorted
        val = np.linspace(0.95, 0.55, n)

    order = np.argsort(-val, kind='stable')       # 0 = highest validity
    vrank = np.empty(n, dtype=float)
    vrank[order] = np.arange(n, dtype=float)
    pos = np.arange(n, dtype=float)               # reading-order index

    gamma = float(parameters.get('gamma', 0.0))
    gamma = min(max(gamma, 0.0), 1.0)
    rank = gamma * vrank + (1.0 - gamma) * pos

    # ---- signed attention gradient (primacy vs recency reader) --------
    lam_mag = float(parameters.get('lam', 2.0))
    orient = float(parameters.get('scan_orientation', 1.0))
    RECENCY_SHARE = 0.35                          # population constant
    lam = -lam_mag if orient < RECENCY_SHARE else lam_mag

    z = -lam * (rank - float(np.mean(rank)))      # centered for stability
    z = z - float(np.max(z))
    w = np.exp(z)

    # ---- relative (normalised) weighted evidence ----------------------
    num = float(np.sum(w * d))
    den = float(np.sum(w * np.abs(d)))
    if den <= 1e-12:
        S = 0.0                                   # nothing discriminates
    else:
        S = num / den                             # in [-1, 1]

    beta = float(parameters.get('beta', 6.0))
    eps = float(parameters.get('epsilon', 0.0))
    eps = min(max(eps, 0.0), 1.0)

    logits = np.array([beta * S, 0.0], dtype=float)
    logits = logits - np.max(logits)
    e = np.exp(logits)
    p = e / np.sum(e)

    p = (1.0 - eps) * p + eps * 0.5
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
```


## Replacement

### `pi_5` → slot 1 (via `new_theory`)

**Description:** **Anchored compensatory integration with load-gated degradation of weighted-additive processing and partial evidence amplification.**

(1) *Diagnosticity is read convexly.* The subjective pull of expert j is d_j = (v_j - 0.5)^rho with rho ~2.3-2.7, so a .95 expert is treated as a near-oracle and .55-.65 experts as near-noise.

(2) *Screen position is a graded attention multiplier, not a search order.* a_j = 1 + alpha*exp(-lam*j) with alpha ~1.5-2.1, lam ~0.9-1.3; the cue weight is the PRODUCT w_j ∝ d_j * a_j, rescaled once to sum to 1 (never renormalised per trial). The first-read cue is rehearsed longest and anchors the running impression, breaking validity ties toward the pos-0 side, but it cannot overturn a genuine diagnosticity gap.

(3) **Processing load is the single switch that degrades weighted-additive integration toward unit-weight counting.** Let L be the number of features on which BOTH products received a positive expert rating (design-general form: the count of non-discriminating positive endorsements; equivalently L/n_features as a density). Under this load two things happen together, both expressions of the same effort–accuracy tradeoff: (a) the diagnosticity exponent flattens, rho_eff = rho / (1 + c*L), so validity differentiation degrades toward equal weighting; (b) the decision maker falls back on the crude 'how many experts back each side' cue, i.e. supra-additive corroboration appears with delta_eff = delta * L/(1+L). Corroboration is silent on sparse displays (L = 0): counting is a load-induced shortcut, not a permanent feature of integration.

(4) **NEW — the margin is read against a partially-attenuated yardstick, not against the raw amount of evidence considered.** Two trials with the same signed weighted margin are not equally decisive: a margin obtained while very little total diagnostic weight was in play feels proportionally larger than the same margin obtained amid a large mass of conflicting weight, but the correction is only PARTIAL. Evidence is E = (Score_A - Score_B) / D^psi, where D = total discriminating weight and psi ~0.2-0.4 is well below 1. At psi = 0 this is the pure absolute margin (all thin-evidence trials near chance); at psi = 1 it would be the fully relative, per-trial-normalised evidence the arbiter rejected (every decision equally confident). Real subjects sit near the low end: thin, near-balanced displays are magnified enough to escape chance, while dense displays keep an essentially absolute margin. This is the theory's sharpest quantitative claim — decisiveness scales sub-linearly with the reciprocal of the evidence mass.

(5) Choice = logistic(beta*E) with a lapse epsilon. No early stopping, no threshold rule, no conclusion-graded commitment, no full per-trial normalisation, and individual differences are unimodal jitter in rho, alpha, lam, delta, c, psi, beta and lapse — no strategy subpopulations, no primacy/recency reader types.

**Rationale:** MINIMAL DIFF on the ACCEPTED iter-6 base, implementing exactly the critic's iter-7 instruction set and nothing else. Two lines change and one parameter is added:

1. KEEP the one edit that demonstrably paid in iter 7 — partial amplification of the margin, E = (s_a*f_a - s_b*f_b) / D^psi with D = s_a + s_b — but with the exponent HALVED to psi ∈ [0.20, 0.40] (iter 7 used [0.40, 0.65]). The critic's decomposition attributes all three of iter 7's gains to this term: E1 -.168 → -.114 (best E1 of the loop), E5 .768 → .840 (real .820), E3 .645 → .608. At half strength it should retain roughly 60-70% of those gains (E1 ≈ -.13/-.14, E5 ≈ .79/.81, E3 ≈ .62) at about a third of the collateral sharpening on the dense-evidence cells.

2. REVERT the salience-weighted clutter index (Edit 1 of iter 7) entirely: L is again the raw count of features endorsed by BOTH options. That edit is what broke the round — it globally deflated L, weakening both rho_eff flattening and delta_eff, which are the mechanisms carrying E6 (.849 → .739) and E2 (.769 → .654). With the raw L restored, E6 and E2 return to their essentially-exact iter-6 values (only mildly re-scaled by the small psi denominator: E6's scored rows have moderate D ≈ .5-.6 so D^0.3 ≈ .82, a ~1.2x amplification that keeps E6 inside the .80-.90 guard band; E2's decisive rows likewise move only slightly). I decline the optional partial-discount u_min variant because it is a second moving part and the critic explicitly said full reversion is acceptable and keeps the next round diagnostic.

3. REVERT the alpha bump: alpha is byte-identical to iter 6 at [1.5, 2.1], as are lam, rho, delta, load_flatten, beta and epsilon. psi is therefore the ONLY moving part relative to the accepted base, which makes the next round maximally diagnostic and avoids the alpha oscillation the critic flagged.

WHY THIS SHOULD BEAT 0.1066. Relative to iter 6 the only change is a mild sub-linear rescaling of the margin by D^-psi. This is a magnification that is largest exactly where D is smallest — the thin, 1-2-cue displays of E1 (rows 7/9, D ≈ .09), E5 and E3 — and near-neutral where D is large (E2's 5-vs-1 and 3-vs-3 rows, E6's four-cue rows). Since E1 (-.168 vs +.093), E5 (.768 vs .820) and E3 (.645 vs .518) are the three cells carrying the residual and all three moved the right way under the full-strength version, a half-strength version should buy a net improvement while leaving the two exact cells (E2 .769, E6 .849) inside their bands.

FAITHFULNESS. The D^-psi term is a bounded, graded modifier with psi ≤ 0.40, so |E| ≤ D^(1-psi) still grows with the evidence mass — it is explicitly NOT the per-trial normalisation by the discriminating weight that the arbiter forbade (that is the psi = 1 limit, which the theory now names and rejects). Everything else is unchanged and in-family: product weights (convex validity term × positional attention term), summation over all cues, logistic on the margin + lapse, unimodal jitter, no early stopping, no threshold rule, no conclusion-graded commitment, no strategy subpopulations, no recency lobe, no beta escalation.

**Parameters:**
  - `rho`: `[2.3, 2.7]`
  - `alpha`: `[1.5, 2.1]`
  - `lam`: `[0.9, 1.3]`
  - `delta`: `[0.30, 0.55]`
  - `load_flatten`: `[0.65, 1.10]`
  - `psi`: `[0.20, 0.40]`
  - `beta`: `[6.0, 8.5]`
  - `epsilon`: `[0.02, 0.08]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ------------------------------------------------------------------
    # 1. unpack the trial stimulus into two rating vectors (screen order)
    # ------------------------------------------------------------------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(list(state['option_a_ratings']), dtype=float)
            b = np.asarray(list(state['option_b_ratings']), dtype=float)
    if a is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(2, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(2, -1)
        if arr.shape[0] != 2:
            arr = arr.reshape(2, -1)
        a = arr[0].astype(float)
        b = arr[1].astype(float)

    a = np.ravel(a).astype(float)
    b = np.ravel(b).astype(float)
    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.array([0.5, 0.5])
    a = a[:n]
    b = b[:n]
    d = a - b

    if not np.any(np.abs(d) > 1e-12):
        return np.array([0.5, 0.5])

    # ------------------------------------------------------------------
    # 2. communicated validities -> convex diagnosticity
    # ------------------------------------------------------------------
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        # fallback: assume the display is validity-sorted, descending
        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])
    val = np.clip(val, 0.5 + 1e-6, 1.0)

    rho = float(np.clip(parameters.get('rho', 2.5), 0.1, 8.0))
    alpha = float(np.clip(parameters.get('alpha', 1.8), 0.0, 6.0))
    lam = float(np.clip(parameters.get('lam', 1.1), 0.01, 6.0))
    beta = float(np.clip(parameters.get('beta', 7.0), 0.0, 200.0))
    eps = float(np.clip(parameters.get('epsilon', 0.05), 0.0, 1.0))
    delta = float(np.clip(parameters.get('delta', 0.40), 0.0, 2.0))
    c_load = float(np.clip(parameters.get('load_flatten', 0.85), 0.0, 3.0))
    psi = float(np.clip(parameters.get('psi', 0.30), 0.0, 0.65))

    # --- overlapping positive endorsements = processing load.
    # L = number of features on which BOTH options were endorsed.  These
    # cues carry no discriminating information but they enlarge the
    # comparison problem, and under that load validity differentiation
    # degrades toward unit-weight counting.
    L = float(np.count_nonzero((a > 0.5) & (b > 0.5)))
    rho_eff = rho / (1.0 + c_load * L)
    rho_eff = float(np.clip(rho_eff, 0.6, 8.0))

    diag = np.power(np.clip(val - 0.5, 1e-12, None), rho_eff)   # diagnosticity

    # ------------------------------------------------------------------
    # 3. graded primacy attention multiplier (screen-position anchor)
    # ------------------------------------------------------------------
    pos = np.arange(n, dtype=float)
    att = 1.0 + alpha * np.exp(-lam * pos)

    w = diag * att
    tot = float(np.sum(w))
    if (not np.isfinite(tot)) or tot <= 0:
        w = np.ones(n, dtype=float) / n
    else:
        w = w / tot            # fixed rescaling only (NOT per-trial normalisation)

    # ------------------------------------------------------------------
    # 4. LOAD-GATED supra-additive corroboration: the crude 'how many
    #    experts back each side' bonus is a shortcut people fall back on
    #    only when the display is dense (L > 0).  On sparse displays
    #    (L = 0) delta_eff = 0 and integration stays purely
    #    weighted-additive.
    # ------------------------------------------------------------------
    delta_eff = delta * (L / (1.0 + L))

    pos_mask = d > 1e-12
    neg_mask = d < -1e-12
    s_a = float(np.sum(w[pos_mask] * np.abs(d[pos_mask])))
    s_b = float(np.sum(w[neg_mask] * np.abs(d[neg_mask])))
    k_a = float(np.count_nonzero(pos_mask))
    k_b = float(np.count_nonzero(neg_mask))
    f_a = (k_a ** delta_eff) if k_a > 0.0 else 0.0
    f_b = (k_b ** delta_eff) if k_b > 0.0 else 0.0

    # ------------------------------------------------------------------
    # 5. PARTIAL amplification: the raw margin is read against a
    #    sub-linearly attenuated yardstick D^psi (psi well below 1), so
    #    thin displays are magnified somewhat while dense displays keep an
    #    essentially absolute margin.  psi = 0 -> pure absolute margin;
    #    psi = 1 would be the forbidden full per-trial normalisation.
    # ------------------------------------------------------------------
    D = s_a + s_b
    if not np.isfinite(D) or D <= 1e-12:
        return np.array([0.5, 0.5])
    E = (s_a * f_a - s_b * f_b) / (D ** psi)

    z = np.clip(beta * E, -60.0, 60.0)
    p_a = 1.0 / (1.0 + np.exp(-z))

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = (1.0 - eps) * p + eps * 0.5
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return p
```

**`policy(probs)`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
```
