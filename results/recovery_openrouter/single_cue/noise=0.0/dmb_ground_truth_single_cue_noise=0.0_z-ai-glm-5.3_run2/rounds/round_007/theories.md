# Round 7 — Theories

**Verdict:** `new_theory` (slot 2 replaced)

## Starting theories

### slot 1 — `pi_7` — SURVIVED ✓

**Description:** Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity (momentary-diffusion variant). People are one-reason decision makers: cues are consulted one at a time in descending subjective weight and the first strictly discriminating cue decides; no integration across cues (flat psychometric profile over tally margins). The population-central hierarchy is anti-validity (distrust-the-weakest, w = -validity). Each subject's weights are a continuous perturbation, w_j = -val_j + gamma*salience_j + A^3*(sigma_h*zeta_j + kappa*xi_j), where (i) salience_j is a positional gradient favoring early-listed experts with gamma drawn asymmetrically from [-0.05, +0.12] — a graded tilt, never a discrete primacy rule, bounded so it can flip nothing on the anchored designs that pin the anti-validity core; (ii) binding ambiguity A = 1 - |Spearman(position, validity)| enters CUBED, a steeply convex, gate-free gradient; and (iii) the idiosyncratic distortion is split into a small stable part (zeta_j, fixed per subject) and a large MOMENTARY part (xi_j, re-drawn every trial): under high binding ambiguity the consultation order is not merely idiosyncratic but diffuse — attention fluctuates from trial to trial, which keeps populations tight around chance-level conformity instead of spreading subjects into fixed opposing deciders. All distortion applies only when all advertised validities are distinct (exact ties anchor the distrust order). Noise is softmax(beta) over the binary winner score plus an independent lapse epsilon (implied follow p_f mean ~0.71, SD ~0.08); no discriminating cue yields exactly 0.5; validity ties are broken by a free per-subject tie-break (~80% early position); history is ignored.

**Rationale:** Minimal-diff edit of the ACCEPTED base (loss 0.0915), addressing the critic's (A)-(E) with one structural addition my own moment audit shows is necessary. (A) PATHWAY VERIFICATION — the critic suspects zeta is silently zeroed. It is not: a zeta-zeroed model (pure anti-validity core at this p_f regime, mean ~0.71) produces an Exp-10 index near the pi_4 level (~590), while the simulation returns 241; moreover iter-1 (weaker sigma_h, linear A) gave 255 and iter-2 (nearly doubled sigma_h, A-squared) gave 241 — a SATURATED lever, not an inert one. I did remove the silent zeros fallback (it now raises), so any future shape mismatch fails loudly. (B) THE REAL Exp-10 DIAGNOSIS — the residual +124 is carried by the dispersion term: our per-subject var is 0.0134 vs observed 0.0041. With FIXED per-subject consultation orders, scrambling MAXIMIZES between-subject follow-rate dispersion (each subject's order deterministically flips or matches each cue pair, giving between-subject SD ~0.13-0.17); the observed SD 0.064 sits barely above the binomial floor (0.053 at ~90 trials). No recalibration of a fixed zeta — including the critic's suggested sigma_h [0.6,1.2] with zeta +/-1, which by my pair-flip calculation still lands near ~200 — can produce a tight, chance-centered population. The fix: split the distortion into a small STABLE part (sigma_h*zeta, exactly the critic's suggested range) plus a large MOMENTARY per-trial diffusion (kappa*xi, fresh every trial). This keeps every pooled-mean metric identical to the fixed-zeta version (means are linear in the noise distribution) while collapsing the between-subject Gini to the binomial floor plus a small stable component. Moment math on Exp 10 (A=0.857, A^3=0.63, kappa~10 -> per-cue noise std ~3.6): mean follow ~0.505, per-subject SD ~0.06 -> index ~100-135 (observed 116.6, var ~0.004) instead of 241. On Exp 12 (A^3=0.32, noise ~1.9): analytic full-diffusion follow = 0.535 vs observed 0.532. I steepened the gradient from A^2 to A^3 so that Exp 9 (A=0.23, A^3~0.012 -> noise ~0.08, close to its current ~0.15) stays anchored at its excellent current fit (0.416 vs 0.428); a linear or quadratic scaling at Exp-10-appropriate magnitude would over-scramble Exp 9 toward 0.5. (C) Exp 4 — adopted the critic's asymmetric gamma [-0.05, +0.12] verbatim: the floor sits above the anchored-design flip threshold (-0.0625) and the cap below the Exp-14 primacy threshold (+0.144), so Exp 14 stays exactly 0.000 with var 0 and Exp 7 keeps its recovered opposition while a positive tail partially restores the early-position lean the gamma tightening removed. (D) GATE RELAXATION REJECTED, with a cell-level audit: on Exp 8 (tied, A=0.77) all four margin-0 deciders ALREADY oppose the reversed hierarchy — the model sits at the family's maximum opposition (1-p_f ~ 0.27; observed -0.289 exceeds even that), so dispersion can only wash it toward 0; on Exp 13 (tied, A=0.91) the observed 0.150 lies BELOW the pure-opposition floor 1-p_f ~ 0.29, and noise drifts the metric UP toward 0.5; on Exp 7 dispersion washes the signed contrast toward 0, the wrong direction from -0.196 vs observed -0.311. Only Exp 5 would benefit, and it cannot outweigh three regressions; I also verified the tie-break is inert on the metric cells of Exps 5, 6, 7, 11 and 13. (E) p_f untouched (mean ~0.71, SD ~0.08): Exp 3 pins it at 0.727 and the de-conformity on Exps 10/12 now comes from decider diffusion, not response weakness. Acknowledged family limits left in place: Exps 13 and 7 (observed opposition beyond 1-p_f), Exp 5 (tied-design over-conformity, unfixable without the rejected gate relaxation), Exp 1 (TTB-following level pinned near 1-p_f). Expected net movement: Exp 10 from +124 to roughly 0 +/- 20, Exp 12 from +0.037 to ~0, Exp 4 recovery toward +0.04 if the gamma mechanism is as the critic's audit indicates, with Exps 2, 3, 5, 6, 8, 9, 11, 13, 14 preserved — comfortably below the accepted loss floor.

**Parameters:**
  - `beta`: `[0.5, 1.9]`
  - `epsilon`: `[0.08, 0.24]`
  - `gamma`: `[-0.05, 0.12]`
  - `sigma_h`: `[0.6, 1.2]`
  - `kappa`: `[8, 12]`
  - `zeta`: `[(-1, 1)] * n_features`
  - `tie_break`: `[0, 1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity
    # (momentary-diffusion variant).
    #
    # One-reason decision making: cues are consulted one at a time in
    # DESCENDING subjective weight and the first strictly discriminating
    # cue decides; no integration across cues (flat psychometric profile
    # over tally margins).
    #
    # Subjective weights (continuous, per subject):
    #     w_j = -val_j + gamma * salience_j
    #            + A^3 * (sigma_h * zeta_j + kappa * xi_j)
    #
    # - ANTI-VALIDITY CORE: -val_j (distrust the weakest advertised
    #   expert; consult ascending validity).
    # - POSITIONAL SALIENCE: salience_j = (n-1-j)/(n-1) - 1/2, favoring
    #   early-listed experts; gamma is drawn ASYMMETRICALLY from
    #   [-0.05, +0.12]: the negative clamp sits above the anchored-design
    #   flip threshold (-0.0625) and the positive cap below the primacy
    #   type threshold (+0.144), so the tilt remains a graded perturbation
    #   that can never manufacture spurious pure types while retaining a
    #   partial early-position lean on anchored designs.
    # - BINDING AMBIGUITY: A = 1 - |Spearman(display position, validity)|,
    #   now entering CUBED. The steeply convex gradient keeps
    #   low/moderate-ambiguity designs anchored (A^3 ~ 0.01-0.02) while
    #   fully diffusing high-ambiguity all-distinct designs (A^3 ~ 0.3-0.6).
    # - DIFFUSE HETEROGENEITY: the idiosyncratic distortion is split into
    #   (a) a SMALL STABLE part, sigma_h * zeta_j (zeta fixed per subject,
    #   zero-centered), and (b) a LARGE MOMENTARY part, kappa * xi_j,
    #   where xi is RE-DRAWN EVERY TRIAL. Momentary consultation diffusion
    #   is what makes high-ambiguity populations tight around chance-level
    #   conformity: with fixed per-subject orders, scrambling MAXIMIZES
    #   between-subject follow-rate dispersion, whereas the observed
    #   population SD (~0.064 on the conformity-dispersion experiment)
    #   sits barely above the binomial floor. Both parts are applied ONLY
    #   when all advertised validities are distinct (exact ties anchor the
    #   distrust order).
    # NO reversal re-encoding, NO sigmoid gates, NO discrete component
    # draws.
    #
    # Noise: softmax(beta) over the binary winner score mixed with an
    # independent lapse epsilon (implied follow probability
    # p_f = (1-eps)*sigmoid(beta) + eps/2, mean ~0.71, SD ~0.08).
    # No discriminating cue -> exactly 0.5. History is ignored.
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
    gamma = float(parameters["gamma"])
    sigma_h = float(parameters["sigma_h"])
    kappa = float(parameters["kappa"])
    tie_break = float(parameters["tie_break"])

    zeta = np.asarray(parameters["zeta"], dtype=float).ravel()
    if zeta.shape[0] != n_features:
        # NO silent zeros fallback: a shape mismatch means the entire
        # heterogeneity mechanism would be silently disabled, so fail
        # loudly instead (per the arbiter's diagnosis request).
        raise ValueError(
            f"zeta length {zeta.shape[0]} != n_features {n_features}."
        )

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

    # ---- positional-salience gradient (early-listed experts favored) ----
    # Centered so the tilt is a pure graded perturbation: +0.5 at the
    # first listed expert, -0.5 at the last, linear in between.
    if n_features > 1:
        sal = ((n_features - 1 - np.arange(n_features, dtype=float))
               / (n_features - 1.0)) - 0.5
    else:
        sal = np.zeros(1, dtype=float)

    # ---- binding ambiguity: A = 1 - |Spearman(position, validity)| ----
    if n_features > 1:
        ranks = avg_ranks(val)
        pos_r = np.arange(n_features, dtype=float)
        vp = pos_r - pos_r.mean()
        vr = ranks - ranks.mean()
        denom = np.sqrt(float((vp * vp).sum()) * float((vr * vr).sum()))
        if denom > 1e-12:
            rho = float((vp * vr).sum()) / denom
        else:
            # All validities tied: distrust order undefined.
            rho = 0.0
        ambiguity = 1.0 - abs(rho)
    else:
        ambiguity = 0.0

    # ---- continuous subjective weights ----
    w = -val + gamma * sal

    # ---- diffuse idiosyncratic distortion (distinct lists only) ----
    # Exact validity ties anchor the distrust order (no distortion);
    # all-distinct lists degrade with the CUBE of binding ambiguity.
    # The distortion is split into a small STABLE per-subject part
    # (sigma_h * zeta) and a large MOMENTARY per-trial part (kappa * xi,
    # xi re-drawn fresh on every call), so that high-ambiguity designs
    # show tight, chance-centered conformity rather than fixed
    # idiosyncratic deciders that spread the population apart.
    distinct = (np.unique(val).shape[0] == n_features)
    if distinct and ambiguity > 0.0:
        amb3 = ambiguity * ambiguity * ambiguity
        xi = np.random.uniform(-1.0, 1.0, size=n_features)
        w = w + amb3 * (sigma_h * zeta + kappa * xi)

    # Consult cues in DESCENDING subjective weight. np.lexsort uses the
    # LAST key as primary. Exact ties in w are broken by the free
    # per-subject tie_break parameter (population-consistent: ~80% of
    # subjects break ties by early display position).
    pos = np.arange(n_features)
    secondary = pos if tie_break < 0.8 else (n_features - 1 - pos)
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
    # probability p_f = (1-eps)*sigmoid(beta) + eps/2 has mean ~0.71
    # with per-subject SD ~0.08 over the sampled (beta, epsilon) ranges.
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


### slot 2 — `pi_6` — KILLED ✗

**Description:** Attenuated Frugality with Position-Validity Mixture Heterogeneity. People are one-reason decision makers: cues are consulted one at a time and the first strictly discriminating cue decides, with no integration across cues (flat psychometric profile over tally margins, matching the margin-0 signatures of the knife-edge experiments). The subjective cue hierarchy is NOT universal: each subject draws a single consultation order from a graded mixture over a small candidate set: (1) anti-validity (distrust-the-weakest: consult the lowest advertised validity first) as the population-central tendency with a constant base weight; (2) position-primacy (consult experts in reading order, earliest-listed first); (3) position-recency (consult later-listed experts first, a minority variant); and (4) a residual standard-TTB component (descending validity). The position components' mixture weights grow smoothly and linearly with binding ambiguity A = 1 - |Spearman(display position, advertised validity)|, with no sigmoid gates of any kind: when the validity-to-position binding is unambiguous (monotone validity lists), the mixture collapses onto the anti-validity order (plus the small TTB residual), producing strong, consistent one-reason contrasts; when the binding is ambiguous (non-monotone lists), the position components enter, the pooled deciders disperse, and hierarchy-dissociation contrasts collapse toward chance. On top of the discrete mixture, a per-cue idiosyncratic weight distortion (zeta, fixed per subject, zero-centered) is applied, smoothly scaled by ambiguity times reversal-coherence and only when all advertised validities are distinct, so that all-distinct ambiguous designs degrade grain-by-grain while exact validity ties anchor the discrete rule mixture. Response discipline is attenuated and heterogeneous: the decider-follow probability p_f = (1-eps)*sigmoid(beta) + eps/2 spans roughly 0.55-0.85 across subjects with mean ~0.73 and SD ~0.06, so pooled contrasts are diluted both by mixture heterogeneity (disagreeing deciders) and by moderate, variable following. Validity ties are broken by a free per-subject tie-break; noise enters as softmax(beta) over the binary winner score plus an independent lapse epsilon; history is ignored (no feedback in this task).

**Rationale:** STRUCTURE (faithful to the arbiter's prescription). (1) The one-reason cascade backbone is kept: first strictly discriminating cue decides, flat over tally margins — the one structural claim both prior theories got right and the margin-0 signatures support. (2) The pure anti-validity hierarchy is replaced by a per-subject categorical draw from a GRADED mixture over {anti-validity, position-recency, position-primacy, residual TTB}, with weights that vary smoothly and linearly with binding ambiguity A = 1 - |Spearman(position, validity)| — no steep sigmoid gates anywhere (the only sigmoid in the model is the response softmax). In monotone designs (A ~ 0) the mixture collapses to anti-validity + a ~8% TTB residual; in ambiguous designs the position components enter, producing near-chance pooled behavior on hierarchy-dissociation cells (Exps 11, 12, 9) and attenuated tilts where orders align. (3) Response discipline is attenuated and heterogeneous: p_f spans ~0.55-0.85 with mean ~0.73 and per-subject SD ~0.06. (4) Free per-subject tie-break, softmax(beta) + lapse epsilon, history ignored, and a smooth ambiguity-times-coherence-scaled per-cue zeta distortion (distinct validities only) that replaces pi_5's steeply gated scrambling.

DELIBERATE DEVIATIONS FROM THE ARBITER'S NUMBERS, WITH EVIDENCE. (a) p_f mean ~0.62 was rejected: the margin-0 knife-edge signatures pin the effective decider-follow at ~0.70-0.73 (e.g., the Exp-4 contrast metric equals 4q-2 for any one-reason model, so observed 0.779 implies q ~ 0.695; the Exp-3 reversed-cascade follow implies q ~ 0.727). Setting p_f ~ 0.62 would underpredict those experiments by >0.2 each — worse than either current theory. The attenuation the arbiter wants is instead delivered where it is actually needed (pooled contrasts in ambiguous designs) by mixture dilution and the widened p_f spread, not by a uniformly low p_f. (b) Within the position family, primacy is weighted far more heavily than recency (0.80A vs 0.06A). Trial-by-trial analysis of the strongly anti-reversed experiments shows the arbiter's premise that 'recency jointly opposes the reversed cascade' there is factually inverted: in those designs the later-listed experts are the high-validity ones, so a recency-first order AGREES with the reversed/TTB-like cascade and pulls the pooled metric toward zero, away from the observed strongly negative values. The position component that genuinely opposes the reversed cascade in those experiments is primacy (reading order, which lands on the low-validity early features). Primacy-dominance reproduces the strong negative anti-reversed contrasts while recency is retained as a minority variant to serve the designs where it helps. (c) The per-cue zeta is retained from the running-best theory (pi_5) but with its steep reversal-coherence gate replaced by a smooth product moderator (ambiguity x coherence), honoring the 'no steep gates' instruction while preserving the grain-by-grain degradation that pi_5's zeta contributed in distinct-validity ambiguous designs.

EXPECTED IMPROVEMENTS over the running best (pi_5, score 0.849), which overpredicts contrast magnitude nearly everywhere: the ~8% TTB residual in monotone designs lowers the Exp-4 contrast from pi_5's 0.90 toward the observed 0.78 and the Exp-6 knife-edge contrast from ~0.50 toward the observed 0.34, while raising the Exp-1 level toward its observed 0.36; in ambiguous designs the position mixture plus smooth zeta pulls the Exp-12 follow rate down from 0.59 toward 0.53, the Exp-10 conformity index down from 184 toward ~117, and the Exp-11 rate toward chance, while the anti+primacy alignment preserves the strongly negative anti-reversed contrasts and the Exp-5 follow rate. The mixture is experiment-invariant by construction: n_features and validities are read symbolically, all weights are smooth functions of rank statistics of the validity list, and the theory reduces to Weakest-Expert-First Frugality in the unambiguous-binding limit.

**Parameters:**
  - `beta`: `[0.85, 1.8]`
  - `epsilon`: `[0.10, 0.22]`
  - `sigma_h`: `[0.4, 1.4]`
  - `zeta`: `[(-1, 1)] * n_features`
  - `tie_break`: `{0, 1}`
  - `comp`: `[0, 1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
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

**`policy(probs)`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)
```


## Replacement

### `pi_8` → slot 2 (via `new_theory`)

**Description:** Anchored Anti-Validity Frugality with Validity-Respecting Leak (frozen-discipline variant). People are one-reason decision makers: experts are consulted one at a time and the first strictly discriminating expert decides; no integration across cues. The population-central hierarchy is anti-validity (distrust-the-weakest) and the population is typeless — every subject is a continuous perturbation around that core. Three graded, design-scaled mechanisms modulate the core. (1) GAP-PROPORTIONAL ANCHORING: per-trial attention noise on expert j is scaled by the local validity gaps adjacent to j in the subjective order, gated by cubed binding ambiguity, so validity-isolated experts anchor the order while dense clusters diffuse; exact validity ties anchor the order entirely. (2) VALIDITY-RESPECTING LEAK, now TRIAL-CONDITIONAL and implemented as a probability mixture: the leak toward the standard TTB (descending-validity) order fires only when the anti-validity cascade's first discriminating expert on that trial is a strictly INTERIOR validity (min_val + 0.10 < val < max_val - 0.10), with design-level probability q = min(rho_leak * reversal_coherence * mean_gap, 0.18). When the anti-decider's carrier is a weakest or near-top expert, the leak is switched off — protecting the pure anti-validity-follow designs — while genuinely mid-validity carriers let a fraction of trials flip toward the strongest discriminator, generating below-chance reversed-cascade following where the data demand it. (3) DESIGN-SCALED RESPONSE DISCIPLINE with a DISTINCT-DESIGN BONUS: p_f = (1-eps)*sigmoid(beta_eff) + eps/2 with beta_eff = beta*(1 + boost*(1-A))*(1 + b_distinct) when all validities are distinct — distinct-validity designs (which the data show want tighter decider-following) get a bounded bonus, tie designs are untouched. Response-discipline and diffusion parameters are FROZEN at narrow windows reproducing the accepted base's behavior, so only the leak strength and the distinct bonus are free.

**Rationale:** This is a surgical edit of the accepted iter-1 base implementing the iter-3 critic's prescription, whose two prior rejection diagnoses I take as ground truth: (i) the leak-gate DIRECTION was empirically right both times (Exps 3/9/10 moved correctly in isolation), but (ii) range-narrowing failed twice as a drift-control strategy because the leak changed the joint likelihood landscape and dragged the shared response-discipline parameters (beta/boost) upward, inflating p_f and regressing every decider-follow metric (Exps 4, 5, 6, 10, 7, 13). I therefore change the drift-control STRATEGY, not just the ranges. (1) FREEZE: beta, boost, epsilon, gamma, kappa, sigma_h are confined to narrow windows centered on the base's mid-range values (I verified analytically that mid-range values reproduce the base's Exp-5 follow of ~0.68 and Exp-3 anti-follow of ~0.71, i.e., the base's operating point), so only rho_leak and the new b_distinct are effectively free. (2) SYMMETRIC INTERIOR GATE (delta = 0.10 hardcoded): the leak is eligible only when the anti-validity cascade's carrier j* on this trial satisfies min + 0.10 < val[j*] < max - 0.10. Hand-traced consequences: on Exp 3 the carrier is always the 0.5-validity minimum expert -> leak provably OFF -> Exp 3 must stay at (or slightly above, via the distinct bonus) the base's value, serving as the freeze regression test; on Exp 6 the carriers are the 0.6-validity minima -> leak OFF; on Exp 7 the design's validity set {0.6, 0.65, 0.85, 0.9, 0.95} contains NO strictly interior values, so the leak never fires there — this removes iter-2's wrong-direction T11 flip while restoring the base's pure anti-follow; on Exp 9 the dissociating carriers are weak experts (<= 0.7 = min + 0.10) -> leak OFF, protecting the near-exact 0.427 fit; on Exp 8 the interior carriers (0.75, 0.8) let the leak flip toward TTB exactly where TTB opposes the reversed cascade, pushing the metric more negative as the data demand. (3) PROBABILITY MIXTURE instead of a per-trial Bernoulli for the leak (numerically equivalent, lower Monte Carlo variance), with ceiling lowered to 0.18 and rho_leak widened to [0.5, 3.0] so the cap — not an attenuated coherence x gap product — does the bounding on high-product designs (Exp 13's product is ~0.097, so the leak can reach the cap there if the fit wants it). (4) DISTINCT-DESIGN BONUS b_distinct in [0, 0.25] multiplying beta_eff when all validities are distinct: this is the orthogonal p_f knob the critic prescribed, along the axis (distinct vs tied) the data actually support — it raises follow on the distinct designs that undershoot (Exps 1, 3, 10, 15) while leaving the tie designs (Exps 5, 6, 7, 13) exactly at base behavior, preventing the cross-contamination that killed iters 2 and 3. (5) Exp 10 is the top-priority target (largest base miss, -31): the interior gate switches the leak off on its weakest-carrier agreeing cells, and the bounded distinct bonus raises the mean follow toward the real level without the uncontrolled inflation that overshot iter-3 to 150.9. I accept Exp 13 (~0.30 vs 0.15) and the Exp 7 structural cap (~-0.20 vs -0.31) as residual misses shared by every one-reason competitor, per the critic's explicit instruction not to trade the winnable experiments for them. All changes stay strictly inside the prescribed Anchored Anti-Validity Frugality with Validity-Respecting Leak family.

**Parameters:**
  - `beta`: `[0.80, 1.00]`
  - `boost`: `[0.28, 0.42]`
  - `b_distinct`: `[0.0, 0.25]`
  - `epsilon`: `[0.12, 0.16]`
  - `gamma`: `[-0.01, 0.03]`
  - `kappa`: `[7.5, 9.5]`
  - `sigma_h`: `[0.18, 0.28]`
  - `rho_leak`: `[0.5, 3.0]`
  - `zeta`: `[(-1, 1)] * n_features`
  - `tie_break`: `[0, 1]`
  - `validities`: `validities`

**`predict(parameters, stimulus, history)`:**
```python
def predict(parameters, state, history):
    # Anchored Anti-Validity Frugality with Validity-Respecting Leak
    # (frozen-discipline variant).
    #
    # Minimal-diff edit of the accepted iter-1 base:
    #   (a) The validity-respecting leak is now TRIAL-CONDITIONAL: it is
    #       eligible only when the anti-validity cascade's first
    #       discriminating expert j* on this trial carries a strictly
    #       INTERIOR validity (min + 0.10 < val[j*] < max - 0.10), and
    #       it is implemented as a PROBABILITY MIXTURE (no Bernoulli
    #       draw) with ceiling 0.18.
    #   (b) A DISTINCT-DESIGN BONUS b_distinct multiplies beta_eff when
    #       all validities are distinct (tie designs untouched).
    #   (c) Response-discipline / diffusion parameters are FROZEN at
    #       narrow windows that reproduce the accepted base's behavior;
    #       only rho_leak and b_distinct are genuinely free.
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
    boost = float(parameters["boost"])
    b_distinct = float(parameters["b_distinct"])
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    kappa = float(parameters["kappa"])
    sigma_h = float(parameters["sigma_h"])
    rho_leak = float(parameters["rho_leak"])
    tie_break = float(parameters["tie_break"])

    zeta = np.asarray(parameters["zeta"], dtype=float).ravel()
    if zeta.shape[0] != n_features:
        raise ValueError(
            f"zeta length {zeta.shape[0]} != n_features {n_features}."
        )

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

    # ---- positional-salience gradient (early-listed experts favored) ----
    if n_features > 1:
        sal = ((n_features - 1 - np.arange(n_features, dtype=float))
               / (n_features - 1.0)) - 0.5
    else:
        sal = np.zeros(1, dtype=float)

    # ---- binding ambiguity: A = 1 - |Spearman(position, validity)| ----
    if n_features > 1:
        ranks = avg_ranks(val)
        pos_r = np.arange(n_features, dtype=float)
        vp = pos_r - pos_r.mean()
        vr = ranks - ranks.mean()
        denom = np.sqrt(float((vp * vp).sum()) * float((vr * vr).sum()))
        if denom > 1e-12:
            rho = float((vp * vr).sum()) / denom
        else:
            rho = 0.0
        ambiguity = 1.0 - abs(rho)
    else:
        ambiguity = 0.0

    # ---- reversal coherence: |Spearman(validity, validity[::-1])| ----
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

    # ---- mean adjacent validity gap over UNIQUE validities ----
    uvals = np.unique(val)
    if uvals.size > 1:
        mean_gap = float(np.mean(np.diff(np.sort(uvals))))
    else:
        mean_gap = 0.0

    distinct = (uvals.size == n_features)

    # ---- gap-proportional anchoring ratios (verbatim from base) ----
    ref_gap = 0.10
    gap_ratio = np.ones(n_features, dtype=float)
    if distinct and n_features > 1:
        srt = np.argsort(val, kind="stable")
        sv = val[srt]
        gs = np.ones(n_features, dtype=float)
        for r in range(n_features):
            g = 0.0
            if r > 0:
                g += sv[r] - sv[r - 1]          # gap to next weaker
            if r < n_features - 1:
                g += sv[r + 1] - sv[r]          # gap to next stronger
            gs[srt[r]] = max(g, 0.05)           # numerical floor
        gap_ratio = np.minimum(ref_gap / gs, 2.0)

    amb3 = ambiguity * ambiguity * ambiguity

    pos = np.arange(n_features)
    secondary = pos if tie_break < 0.8 else (n_features - 1 - pos)

    a, b = stim[0], stim[1]

    def cascade_winner(order):
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return None

    # ---- STABLE anti-validity weights (no momentary noise): used ONLY
    # to locate this trial's anti-validity carrier j* for the leak gate.
    w_stable = -val + gamma * sal
    if distinct and ambiguity > 0.0:
        w_stable = w_stable + amb3 * kappa * gap_ratio * sigma_h * zeta
    order_stable = np.lexsort((secondary, -w_stable))

    j_star = None
    for j in order_stable:
        if a[j] != b[j]:
            j_star = int(j)
            break

    # ---- TRIAL-CONDITIONAL leak eligibility: strictly interior carrier.
    # Weakest-expert carriers (val <= min + 0.10) and near-top carriers
    # (val >= max - 0.10) switch the leak OFF; only genuinely mid-
    # validity carriers let the TTB leak fire. Ceiling 0.18.
    leak_q = 0.0
    if j_star is not None and n_features > 1:
        vmin = float(val.min())
        vmax = float(val.max())
        if (vmin + 0.10) < val[j_star] < (vmax - 0.10):
            leak_q = float(min(rho_leak * coherence * mean_gap, 0.18))

    # ---- per-trial subjective hierarchy (momentary diffusion added) ----
    w = w_stable.copy()
    if distinct and ambiguity > 0.0:
        xi = np.random.normal(0.0, 1.0, size=n_features)
        w = w + amb3 * kappa * gap_ratio * xi
    order_anti = np.lexsort((secondary, -w))
    winner_anti = cascade_winner(order_anti)

    n_opts = 2
    if winner_anti is None:
        # No discriminating cue: pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    # ---- design-scaled response discipline with DISTINCT-DESIGN BONUS ----
    beta_eff = beta * (1.0 + boost * (1.0 - ambiguity))
    if distinct:
        beta_eff = beta_eff * (1.0 + b_distinct)

    def score_prob(winner):
        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        z = beta_eff * (scores - scores.max())
        e = np.exp(z)
        return e / e.sum()

    p_core = score_prob(winner_anti)

    # ---- VALIDITY-RESPECTING LEAK as a probability mixture (no
    # Bernoulli draw -> lower Monte Carlo variance in simulated metrics).
    if leak_q > 0.0:
        order_ttb = np.lexsort((secondary, -val))
        winner_ttb = cascade_winner(order_ttb)
        if winner_ttb is not None and winner_ttb != winner_anti:
            p_core = (1.0 - leak_q) * p_core + leak_q * score_prob(winner_ttb)

    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / float(n_opts))
```

**`policy(probs)`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)
```
