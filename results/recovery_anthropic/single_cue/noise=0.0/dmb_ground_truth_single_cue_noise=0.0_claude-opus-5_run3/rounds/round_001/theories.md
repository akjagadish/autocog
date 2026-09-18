# Round 1 — Theories

**Verdict:** `new_theory` (slot 1 replaced)

## Starting theories

### slot 1 — `pi_1` — KILLED ✗

**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Rationale:** (none)

**Parameters:**
  - `beta`: `[0.1, 20.0]`
  - `epsilon`: `[0.0, 0.5]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy(probs)`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
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

### `pi_4` → slot 1 (via `new_theory`)

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
