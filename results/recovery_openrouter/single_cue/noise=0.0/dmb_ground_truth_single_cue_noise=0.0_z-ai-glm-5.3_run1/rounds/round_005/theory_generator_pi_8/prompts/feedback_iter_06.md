# feedback_iter_06

## System Prompt

You are a renowned cognitive scientist critiquing a freshly proposed candidate theory and model in the Decision Making (Binary Features) domain.

The candidate has been simulated on every previously run experiment. For each experiment you are shown the design, the metric, the value the metric takes on real data, and the value it takes on the candidate's simulated data.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the feedback is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across multiple experiments. 
Your task is to determine whether the candidate captures the human/real behavior well enough across these experiments. Return a verdict:
  * "continue"   — the candidate is good enough; carry on.
  * "regenerate" — the candidate fails to capture the empirical pattern; the proposing agent must produce a new candidate, taking your rationale into account.

Justify the verdict with a concrete diagnosis (which experiments fail, in what direction, what mechanism is likely missing or miscalibrated).

## SCOPE OF YOUR CRITIQUE — STAY INSIDE THE ARBITER'S MECHANISM FAMILY
When an "## ARBITER RECOMMENDATION" block is present below, the proposer was explicitly instructed to implement the mechanism family the arbiter prescribed. Your job is to grade FIT QUALITY *within that prescribed family*, not to relitigate which family should be used — that is the arbiter's call, made one level above this loop.

Concretely:
  * If the candidate misses the data, you may push for MINOR ADJUSTMENTS that keep the prescribed mechanism intact: tightening / widening parameter ranges, adding a temperature, swapping a normalization scheme, fixing a softmax / distance metric, re-balancing attention weights, fixing a learning-rate sign, correcting a bug in the gating or recurrence, etc.
  * You MUST NOT recommend switching to a different mechanism family. Such a switch is the arbiter's prerogative; recommending it here will mislead the proposer into oscillating between families across iterations.
  * Also grade FAITHFULNESS to the recommendation explicitly: if the candidate has clearly drifted into a different family than the one prescribed, say so in the rationale and ask for a return to the prescribed family — again, with minor adjustments, not a re-design.

## ACCEPT GATE — HOW THE LOOP DECIDES WHAT TO BUILD ON NEXT
This propose-loop has a programmatic accept gate. After every iteration the candidate's `aggregate_loss` is compared against the running-best loss (`accepted_loss`):
  * `loss < accepted_loss` → ACCEPTED. The candidate becomes the new running-best base; the next iteration's proposer will build on THIS candidate.
  * `loss >= accepted_loss` → REJECTED. The base is unchanged; the next iteration's proposer will build on the SAME `accepted` candidate again, with your new feedback on top. Rejected candidates are discarded — the loop guarantees the base never regresses, so you do NOT need to ask the proposer to "revert" anything; that already happens for free.

Two consequences for your verdict:
  * If the candidate you are grading was REJECTED by the gate, returning `"continue"` is silently downgraded to `"regenerate"` (returning a worse candidate would defeat the gate). Spend your rationale on a NEW direction the proposer should try on top of the unchanged accepted base, not on defending the rejected attempt.
  * If the candidate was ACCEPTED, you can return `"continue"` to stop the loop and ship this candidate, or `"regenerate"` to keep tuning further.

## LEARN FROM YOUR OWN PAST ADVICE
When a "## YOUR PRIOR CRITIQUES" block is present below, each prior iteration ends with an "Outcome of your advice" line that says whether the next candidate the proposer produced was ACCEPTED (your advice helped — its loss strictly beat the running best) or REJECTED (your advice didn't help — the proposer discarded the result and reset to the previous accepted base). This is the loop's ground-truth signal on whether *your own previous critique was good*. Use it explicitly:
  * If a previous piece of advice was ACCEPTED, it is OK to repeat / extend it. Reinforce in the same direction.
  * If a previous piece of advice was REJECTED, do NOT repeat the same recommendation; in your new rationale, briefly acknowledge that the previous push in that direction was rejected by the gate and try a different in-family knob (or a smaller step in the same direction) instead.
  * If you find yourself oscillating (e.g. iter 1 said "increase α", iter 2 said "decrease α", iter 3 about to say "increase α" again), STOP and recommend a value between the two flanking iterations instead.
  * The "## LOSS TRAJECTORY" block at the top of the user prompt summarises the same information at the loop level — consult it before issuing a new regenerate-with-direction recommendation.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## ARBITER RECOMMENDATION (mechanism family the proposer was told to implement)
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_7`
- THEORY 2 = `pi_4`
- The recommendation below acts on THEORY 2 (= `pi_4`).

Replace FDEI (pi_4) with a new theory: Panel-Incoherence-Gated Distrust with Flattened Diagnosticity (PIGD). Core claims: (1) BASELINE — all experts at or below a skepticism ceiling v_c ≈ 0.98 are integrated as sign votes with flattened diagnosticity weights w_j = max(log(v_j/(1-v_j)) - tau, 0)^delta (delta ≈ 0.15–0.35), exactly FDEI's successful baseline, which reproduces its strong fits on Experiments 3–6 (graded tally-vs-top-cue slopes, ~0.62 tally-following, |d|=1 near-chance agreement). (2) ABOVE-CEILING DISTRUST IS COMPOSITION-DRIVEN, NOT GAP-DRIVEN — an expert claiming v > v_c receives w = w_base(v) - D * Phi(c_frac), where c_frac is the fraction of chance-level (v ≈ 0.5) experts in the panel and Phi is a sigmoid centered around c_frac ≈ 0.3–0.4; there is NO non-monotone rung-gap bump. This yields: strong distrust (w(0.99) < 0) on chance-laden panels, matching Experiments 8 and 9 near-exactly where pi_7 already succeeded; near-zero discount on coherent panels, matching Experiment 2's observed near-chance deference (0.4525) that falsified CREDT's spurious-precision bump; and full positive weight when the outlier sits atop a dense corroborating ladder (gap ≤ 0.02), matching Experiment 12's mild pro-99% following (0.599). (3) CHANCE CUES — v = 0.5 cues receive weight ≈ 0 (annihilated) on coherent panels, but a small negative weight on chance-laden panels with an above-ceiling outlier, capturing Experiment 7's observed negative chance-cue following (-0.168) that FDEI could not produce at any parameter. (4) ATTENUATED RESPONSE GAIN — a low-gain softmax (2*beta ≈ 0.4–0.6) plus a uniform lapse (epsilon ≈ 0.08–0.15), sized so that predicted effects are systematically smaller than CREDT's, matching the shrunken magnitudes in Experiments 1 and 2. This theory differs from FDEI by abandoning structural monotonicity (allowing context-dependent sign flips of the super-expert weight) and differs from CREDT by deleting the gap-based non-monotone bump and the consensus-gated positive trust of chance cues, replacing both with a single panel-incoherence drive — making it a genuinely stronger competitor to CREDT on the one regime (coherent moderate-gap panels) where CREDT is currently falsified, while inheriting CREDT's empirical wins everywhere else.


## CANDIDATE THEORY
Panel-Incoherence-Gated Distrust with Flattened Diagnosticity (PIGD), re-anchored-incoherence edition. People integrate all binary expert ratings as sign votes into a zero-sum evidence score with flattened diagnosticity weights, and the subjective weight of an above-ceiling claimant (v > ~0.985) is gated by panel INCOHERENCE. The single new cognitive claim this iteration: the incoherence statistic that drives distrust of a near-certain claim is measured against a BROAD skepticism band, not exact chance. When judging whether a 99% claim is credible, an expert at v ~ 0.55-0.60 is functionally noise — it cannot corroborate near-certainty — even though the same expert, on a claimant-free panel, is treated as a weakly trusted cue (its endorsement is never anti-diagnostic). Concretely: (1) baseline w_base(v) = max(log(v/(1-v)) - tau, 0)^delta; (2) distrust discount G = Phi(c_d) * mult, where c_d is the fraction of experts below a distrust-specific threshold v_cd ~ 0.56-0.61 (broader than the kappa branch's v_lo ~ 0.53), and mult = max(m_floor, 1 - rho_corr * corr_frac) is corroboration dilution in which a rung counts as corroborating down to v_corr ~ 0.76-0.80 (a 0.80 expert substantially corroborates a 0.99 claim; a 0.75 expert does not); the claimant loses d_top*G, the rest of the trusted scale d_panel*G; (3) chance-cue anti-weighting (-kappa * Phi_high(c)) fires only on claimant-free panels and keeps the narrow v_lo classification; (4) low-gain softmax plus uniform lapse.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Panel-Incoherence-Gated Distrust with Flattened Diagnosticity (PIGD),
    # re-anchored-incoherence edition.
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings (0/1).
    #  (i) BASELINE (flattened diagnosticity): w_base(v) =
    #      max(log(v/(1-v)) - tau, 0)^delta. Experts ~0.55-0.95 count
    #      nearly equally; exact-0.5 experts are annihilated.
    # (ii) DISTRUST BRANCH (fires only if some expert claims v > v_c,
    #      v_c strictly between 0.98 and 0.99): the incoherence input
    #      is now the fraction of experts below a DISTRUST-SPECIFIC
    #      threshold v_cd (~0.56-0.61), BROADER than the kappa branch's
    #      v_lo: when judging a near-certain claim, a barely-above-
    #      chance expert (v ~ 0.55-0.60) is functionally noise and
    #      counts toward panel incoherence, even though the same
    #      expert is weakly trusted on claimant-free panels. The
    #      discount strength is
    #         G = Phi(c_d) * mult,
    #      with mult = max(m_floor, 1 - rho_corr * corr_frac) a
    #      CORROBORATION DILUTION in which a sub-ceiling rung counts
    #      as corroborating down to v_corr ~ 0.76-0.80 (a 0.80 expert
    #      corroborates a 0.99 claim; a 0.75 expert does not). The
    #      claimant loses d_top * G; every other trusted expert loses
    #      d_panel * G. No rung-gap term anywhere.
    # (iii) CHANCE-CUE ANTI-WEIGHTING, gated on the ABSENCE of an
    #      above-ceiling claimant and keeping the NARROW v_lo chance
    #      classification (unchanged from the accepted base): only on
    #      claimant-free panels does a coin-flipper's endorsement
    #      become mildly anti-diagnostic (-kappa * Phi_high(c) * ramp).
    # (iv) Choice is a low-gain softmax (beta) plus a uniform lapse
    #      (epsilon).
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

    delta = float(parameters["delta"])
    tau = float(parameters["tau"])
    v_c = float(parameters["v_c"])
    v_cd = float(parameters["v_cd"])
    d_top = float(parameters["d_top"])
    d_panel = float(parameters["d_panel"])
    c_half = float(parameters["c_half"])
    s_c = float(parameters["s_c"])
    v_corr = float(parameters["v_corr"])
    rho_corr = float(parameters["rho_corr"])
    m_floor = float(parameters["m_floor"])
    kappa = float(parameters["kappa"])
    k_half = float(parameters["k_half"])
    s_k = float(parameters["s_k"])
    v_lo = float(parameters["v_lo"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    a, b = stim[0], stim[1]

    # Feature-wise votes: +1 where A wins the feature, -1 where B
    # wins, 0 on ties (a tied cue is uninformative for the pair).
    votes = np.sign(a - b)

    v = np.clip(val, 0.5, 1.0)

    # ---- (i) Baseline: flattened-diagnosticity weights. ----
    vv = np.clip(v, 0.5, 1.0 - 1e-9)
    llr = np.log(vv / (1.0 - vv))
    w = np.power(np.maximum(llr - tau, 0.0), delta)

    # ---- Panel composition. ----
    # Narrow classification (kappa branch, weight-side spread):
    chance = v < v_lo
    c_frac = float(np.mean(chance)) if n_features > 0 else 0.0
    # NEW: broad distrust-band classification. An expert below v_cd
    # (~0.56-0.61) is functionally noise WHEN JUDGING A NEAR-CERTAIN
    # CLAIM: it cannot corroborate the claimant, so it counts toward
    # the incoherence input of the distrust branch only. The kappa
    # branch and the d_panel spread keep the narrow v_lo mask, so
    # claimant-free panels are bit-identical to the accepted base.
    chance_d = v < v_cd
    c_frac_d = float(np.mean(chance_d)) if n_features > 0 else 0.0
    v_max = float(v.max()) if n_features > 0 else 0.0
    outlier = v_max > v_c

    if outlier:
        # ---- (ii) Distrust branch: broad-band incoherence gate,
        #      diluted by corroboration mass down to v_corr. ----
        phi_d = 0.5 * (1.0 + np.tanh(0.5 * (c_frac_d - c_half) / s_c))

        # Corroboration: fraction of trusted, SUB-ceiling experts
        # whose validity sits at or above v_corr. A 0.78-0.84 rung
        # now corroborates (it approaches the claim); a 0.75 rung
        # still does not.
        trusted_nc = (~chance) & (v <= v_c)
        n_tc = int(np.sum(trusted_nc))
        if n_tc > 0:
            corr_frac = float(np.sum(trusted_nc & (v >= v_corr))) / n_tc
        else:
            corr_frac = 0.0
        mult = max(m_floor, 1.0 - rho_corr * corr_frac)

        G = phi_d * mult

        # Distrust spreads over the whole trusted scale...
        trusted = ~chance
        w[trusted] = w[trusted] - d_panel * G
        # ...and hits the above-ceiling claimant(s) hardest, so the
        # claimant's total discount is d_top * G.
        above = v > v_c
        w[above] = w[above] - (d_top - d_panel) * G
        # Chance cues stay exactly annihilated on claimant panels.
    else:
        # ---- (iii) Chance-cue anti-weighting, claimant-free panels
        #      only: high-centered incoherence gate on the NARROW
        #      classification (unchanged from the accepted base). ----
        phi_k = 0.5 * (1.0 + np.tanh(0.5 * (c_frac - k_half) / s_k))
        if np.any(chance) and phi_k > 0.0:
            ramp = np.clip(
                (v_lo - v[chance]) / max(v_lo - 0.5, 1e-6), 0.0, 1.0
            )
            w[chance] = w[chance] - kappa * phi_k * ramp

    # Weighted evidence score. B's score is the mirror of A's
    # because every vote is zero-sum across the two options.
    s_a = float(np.dot(w, votes))
    scores = np.array([s_a, -s_a], dtype=float)

    # Numerically stable low-gain softmax. When all cues tie
    # (s_a == 0) the softmax is exactly uniform.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Independent lapse: with probability epsilon output a uniform
    # pick over the two options.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))

`parameters`:
- delta: [0.15, 0.19]
- tau: [0.01, 0.03]
- v_c: [0.982, 0.988]
- v_cd: [0.555, 0.615]
- d_top: [4.0, 4.6]
- d_panel: [2.4, 3.0]
- c_half: [0.25, 0.29]
- s_c: [0.06, 0.08]
- v_corr: [0.76, 0.80]
- rho_corr: [2.0, 2.4]
- m_floor: [0.21, 0.25]
- kappa: [0.34, 0.42]
- k_half: [0.38, 0.42]
- s_k: [0.06, 0.08]
- v_lo: [0.52, 0.54]
- beta: [0.28, 0.34]
- epsilon: [0.05, 0.09]
- validities: validities

`rationale`:
This is a three-knob minimal-diff edit of the accepted iter-5 base, implementing the iter-6 critic's primary directive: FIX THE INCOHERENCE INPUT, NOT THE SIGMOID CENTER. Six iterations proved E7's gate never opens because its weak block is 0.55-0.60-class experts that the v_lo ~ 0.53 classifier counts as trusted; lowering the center (iter 6) was gate-rejected because it opened E3's gate instead. The fix is a distrust-branch-specific threshold v_cd.

INSTRUMENTATION (best available without code execution — I cannot run code in this response, so instead of inferring composition from simulation swings, I use the validities STATED in each metric's docstring, which are fixed design constants): E5 = [0.95, 0.85, 0.75, 0.65, 0.50x4] (docstring); E6 = 0.99, 0.99, 0.75, 0.75 + four weak (metric classes); E8 = 0.98, 0.90, 0.75 + 6 unspecified (docstring, top three); E9 = 0.99, 0.99, 0.98, 0.90, 0.90, 0.55, 0.50x3+ (S2/V1 cell descriptions); E10 = [0.99, 0.98, 0.95, 0.90, 0.85, 0.75, 0.65, ., ., .] (docstring); E11 = [0.99, 0.90, 0.80, 0.70, 0.60, 0.50x5] (docstring); E12 = 0.99 atop a 0.94 rung, partially chance-laden (CREDT description); E2 top = 0.92 (docstring). E1/E3/E4 are not documented; E3 is INFERRED to have a claimant with graded rungs >= 0.60 and one 0.50 (the only hypothesis consistent with both iter-5 exactness at phi(0.167)~0.19 and iter-6's collapse at phi(0.167)~0.7 with weakened dilution); E1 is inferred claimant-free (its value was insensitive to iter-6's c_half swing).

EDIT 1 — v_cd (the only structural change): the distrust branch's c_frac now counts experts below v_cd ~ 0.555-0.615 instead of v_lo. Effects on stated compositions: E9's c rises 0.27 -> 0.36-0.55 (its 0.55 + 0.50s) -> phi -> ~1, G_9 0.166 -> ~0.23 (floored mult) -> anti-99 rises toward the real 0.56 (fixes half of the -0.048 error); E12's c rises ~0.25 -> ~0.375 -> phi 0.46 -> 0.82, G_12 ~0.10 -> ~0.19 -> deference falls from 0.547 toward ~0.50; E7's c rises from ~0 to 0.25-0.5 (its 0.55-0.60 block) -> G_7 0 -> 0.2-0.6 -> w(0.99) drops from +1.29 toward parity or below -> contrarianism moves from 0.31 toward 0.5-0.65. Panels with no 0.55-0.60 experts are UNCHANGED: E3 (rungs >= 0.60; the 0.60 is counted only by the ~25% of draws with v_cd > 0.60, giving at most a mild partial opening at a center of 0.27, far from iter-6's collapse regime), E4/E6 (c already saturated at 0.5), E10 (if its three hidden rungs are 0.50s; if any are 0.55s, c_10 -> 0.4-0.6 and following dips to ~0.57, a bounded -0.03 cost), E11 (0.60 counted only for v_cd > 0.60 draws, negligible). The kappa branch keeps v_lo, so E1, E2, E5, E8 are bit-identical to the accepted base by construction.

EDIT 2 — v_corr [0.80, 0.84] -> [0.76, 0.80]: the E7/E6 separator. Both panels have c ~ 0.5 and currently corr = 0, but E6's best sub-ceiling rung is 0.75 while a plausible E7 panel tops out at 0.78-0.90. Crediting rungs down to 0.78 gives E7's claimant mult ~ 0.3-0.63 (partial discount, the parity regime the critic demands) while E6 keeps mult = 1 (0.75 < 0.76) and its validated strong-side rejection. Cost: E3's 0.80 rung now corroborates, dropping its mult 0.45 -> 0.23 and G_3 0.086 -> 0.044 — E3 drifts from exactly 0.617 to ~0.62-0.63, a ~0.0002 squared cost against E7's potential ~0.05 gain. No other stated panel has rungs in [0.76, 0.80).

EDIT 3 — m_floor [0.18, 0.24] -> [0.21, 0.25]: I deliberately DEVIATE from the critic's suggested [0.12, 0.18]. Justification: iter-6's rejection showed that LOWERING m_floor cut E9's discount (its mult is floored at corr ~ 0.6) and moved E9 from 0.5125 to 0.4595, away from the real 0.56. Since v_cd already drives phi_9 to ~1, m_floor is now E9's only discount dial, and E9 needs MORE anti-99, not less. The raised floor also adds ~10% discount to the floored E12 panel (helps its +0.095 error) at a small E10 cost. rho_corr stays at the validated [2.0, 2.4].

RISK PROFILE (why this beats the gate's 0.0716 floor in expectation): E7's error is ~80% of the remaining loss. Under this edit, E7 either (a) lands in the parity/partial-discount regime (index ~0.5-0.65, error ~0.05), or (b) if its second rung is below 0.76 AND its weak block is exactly 0.60s, stays at 0.31 — the same squared error as now, i.e. bounded downside. E9 improves (toward 0.53-0.54), E12 improves (toward 0.49-0.51), E4/E6/E11 are unchanged, E1/E2/E5/E8 are bit-identical, E3/E10 carry bounded costs of ~0.02-0.03 each in adverse composition scenarios. The one path I do NOT take is re-lowering c_half: it is empirically falsified (it collapsed E3 at iter 6 without moving E7 at any center from 0.36 down to 0.09), and the c_half box [0.25, 0.29] already sits inside the critic's recommended re-derived band [0.22, 0.30] on the new v_cd scale.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.0999 -> ACCEPTED
- iter 2: loss=0.1393 -> REJECTED
- iter 3: loss=0.1117 -> REJECTED
- iter 4: loss=0.1215 -> REJECTED
- iter 5: loss=0.0716 -> ACCEPTED
- iter 6: loss=0.1026 -> REJECTED
- iter 7 (current candidate you are grading): loss=0.0885 -> REJECTED
Running-best (last accepted) base: iter 5 at loss=0.0716.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    rows_a = data["option_a_ratings"].tolist()
    rows_b = data["option_b_ratings"].tolist()
    resp = data["response"].tolist()
    n_rows = len(rows_a)
    contribs = []
    for i in range(n_rows):
        ca = rows_a[i]
        cb = rows_b[i]
        if isinstance(ca, str):
            ca = ca.replace("[", " ").replace("]", " ").replace(",", " ").split()
        if isinstance(cb, str):
            cb = cb.replace("[", " ").replace("]", " ").replace(",", " ").split()
        a = [int(float(v)) for v in ca]
        b = [int(float(v)) for v in cb]
        r = int(resp[i])
        n = len(a)
        if n == 0 or n != len(b):
            continue
        ttb = -1
        for j in range(n):
            if a[j] > b[j]:
                ttb = 0
                break
            if b[j] > a[j]:
                ttb = 1
                break
        if ttb < 0:
            continue
        aw = 0
        bw = 0
        for j in range(n):
            if a[j] > b[j]:
                aw = aw + 1
            elif b[j] > a[j]:
                bw = bw + 1
        if aw == bw:
            if r == ttb:
                contribs.append(0.5)
            else:
                contribs.append(-0.5)
        else:
            if bw > aw:
                tal = 1
            else:
                tal = 0
            if tal == ttb:
                continue
            if r == ttb:
                contribs.append(1.0)
            else:
                contribs.append(-1.0)
    if len(contribs) == 0:
        return 0.0
    total = 0.0
    for c in contribs:
        total = total + c
    return float(total / len(contribs))
```

**Observed (real) value:** -0.2180 (var=0.0140)
**Candidate trajectory (this loop):**
  - iter 1: -0.2050 (var=0.0113) (Δ vs real +0.0130)
  - iter 2: -0.2185 (var=0.0062) (Δ vs real -0.0005)
  - iter 3: -0.2042 (var=0.0084) (Δ vs real +0.0138)
  - iter 4: -0.2000 (var=0.0070) (Δ vs real +0.0180)
  - iter 5: -0.1817 (var=0.0050) (Δ vs real +0.0363)
  - iter 6: -0.1998 (var=0.0086) (Δ vs real +0.0182)
  - iter 7 (current): -0.2258 (var=0.0075) (Δ vs real -0.0078)
**Other theories' values on this metric (for reference):**
- pi_1: 0.6130 (var=0.0149)
- pi_2: -0.4425 (var=0.0124)
- pi_3: -0.2182 (var=0.0156)
- pi_4: -0.2188 (var=0.0064)
- pi_5: -0.4125 (var=0.0044)
- pi_6: -0.1973 (var=0.0076)
- pi_7: -0.2530 (var=0.0065)

### Experiment 2
**Design**
  A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Dual-dissociation regression metric.
    # Per trial, compute:
    #   d = tally margin  = (#features A wins) - (#features B wins)
    #   s = top-cue direction = sign(A[0] - B[0])   (feature 0 has the
    #       highest validity 0.92, and every designed pair discriminates on it)
    # Regress chose-A on [1, d, s] via OLS and return b_d - b_s.
    # Tallying: choice probability depends only on d  -> b_d > 0, b_s = 0 -> metric > 0.
    # TTB: choice probability depends only on s       -> b_d = 0, b_s > 0 -> metric < 0.
    # The two theories push this scalar in opposite directions, and the
    # partial regression cleanly removes the design's d-s correlation.
    df = data.reset_index(drop=True)

    A = np.stack([np.asarray(list(r), dtype=float) for r in df["option_a_ratings"].values])
    B = np.stack([np.asarray(list(r), dtype=float) for r in df["option_b_ratings"].values])

    # Tally margin: strict feature-wise wins; ties count for neither side.
    d = (A > B).sum(axis=1).astype(float) - (B > A).sum(axis=1).astype(float)

    # Direction of the highest-validity (top) cue, which always discriminates
    # in this design: +1 if it favors A, -1 if it favors B.
    s = np.sign(A[:, 0] - B[:, 0])

    # Chose-A indicator (response: 0 = A, 1 = B).
    y = 1.0 - df["response"].to_numpy(dtype=float)

    X = np.column_stack([np.ones(len(y)), d, s])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    b_d = float(coef[1])
    b_s = float(coef[2])

    # Tallying's signature (slope in d) minus TTB's signature (slope in s).
    return b_d - b_s
```

**Observed (real) value:** 0.0771 (var=0.0031)
**Candidate trajectory (this loop):**
  - iter 1: 0.0712 (var=0.0029) (Δ vs real -0.0059)
  - iter 2: 0.0804 (var=0.0029) (Δ vs real +0.0033)
  - iter 3: 0.0925 (var=0.0022) (Δ vs real +0.0155)
  - iter 4: 0.0762 (var=0.0032) (Δ vs real -0.0009)
  - iter 5: 0.0780 (var=0.0019) (Δ vs real +0.0009)
  - iter 6: 0.0832 (var=0.0026) (Δ vs real +0.0061)
  - iter 7 (current): 0.0817 (var=0.0023) (Δ vs real +0.0046)
**Other theories' values on this metric (for reference):**
- pi_2: 0.1196 (var=0.0023)
- pi_1: -0.3725 (var=0.0076)
- pi_3: 0.0861 (var=0.0026)
- pi_4: 0.0881 (var=0.0019)
- pi_5: 0.0789 (var=0.0021)
- pi_6: 0.0956 (var=0.0023)
- pi_7: 0.1225 (var=0.0025)

### Experiment 3
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Tally-following rate on decisive trials.

    For every trial, compute the raw (unweighted) tally margin
    d = (# cues where A wins) - (# cues where B wins).
    On trials with d != 0, check whether the subject's choice agrees
    with the option favored by the raw cue count (A if d > 0, B if
    d < 0). The metric is the fraction of such agreement over all
    decisive trials (d = 0 trials are excluded because the tallying
    heuristic itself is indifferent there).
    """
    df = data.reset_index(drop=True)
    n = len(df)
    if n == 0:
        return 0.5

    a_lists = df['option_a_ratings'].tolist()
    b_lists = df['option_b_ratings'].tolist()
    resp = df['response'].to_numpy(dtype=float)

    d = np.empty(n, dtype=float)
    for i in range(n):
        a = np.asarray(a_lists[i], dtype=float)
        b = np.asarray(b_lists[i], dtype=float)
        d[i] = float(np.sum(a > b) - np.sum(b > a))

    decisive = d != 0.0
    if not np.any(decisive):
        return 0.5

    dd = d[decisive]
    rr = resp[decisive]
    # Option favored by the raw tally: A (response 0) if d > 0,
    # otherwise B (response 1).
    agree = np.where(dd > 0.0, rr == 0.0, rr == 1.0)
    return float(np.mean(agree))
```

**Observed (real) value:** 0.6170 (var=0.0034)
**Candidate trajectory (this loop):**
  - iter 1: 0.6244 (var=0.0030) (Δ vs real +0.0074)
  - iter 2: 0.6252 (var=0.0026) (Δ vs real +0.0081)
  - iter 3: 0.6293 (var=0.0036) (Δ vs real +0.0122)
  - iter 4: 0.6207 (var=0.0028) (Δ vs real +0.0037)
  - iter 5: 0.6170 (var=0.0037) (Δ vs real +0.0000)
  - iter 6: 0.4456 (var=0.0026) (Δ vs real -0.1715)
  - iter 7 (current): 0.6063 (var=0.0042) (Δ vs real -0.0107)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6456 (var=0.0096)
- pi_2: 0.8389 (var=0.0095)
- pi_1: 0.6944 (var=0.0071)
- pi_4: 0.6485 (var=0.0027)
- pi_5: 0.4996 (var=0.0026)
- pi_6: 0.4811 (var=0.0035)
- pi_7: 0.6267 (var=0.0043)

### Experiment 4
**Design**
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0, 0, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Hard-ceiling test on the |d|=1 family (FACT 1 of the design).

    Metric = proportion of choices that agree with the UNWEIGHTED tally
    count-winner, restricted to trials whose raw tally margin is exactly
    +/-1 (the 8 trial types t1-t8; 48 of the 96 trials per subject).

    Tallying (any beta >= 1, any epsilon) predicts ~1 - epsilon/2 here,
    i.e. >= ~0.85 for any moderate lapse. The validity-weighted
    competitor is structurally capped: its choice probability on a |d|=1
    trial is (1-eps)*sigmoid(2*beta*m) with m = sum_j v_j^gamma*sign diff;
    the mean of |m| over the 8 |d|=1 trial types is <= 1 for every
    gamma in [0.05, 0.7] (equality only at gamma=0), and beta <= 0.55,
    so by concavity of tanh its mean agreement on this family is
    <= 0.5 + tanh(0.55)/2 = 0.750 over its ENTIRE parameter box.
    """
    def _mat(col):
        vals = []
        for v in col:
            if isinstance(v, np.ndarray):
                vals.append(v)
            else:
                vals.append(list(v))
        return np.asarray(vals, dtype=float)

    A = _mat(data["option_a_ratings"])
    B = _mat(data["option_b_ratings"])
    resp = np.asarray(data["response"], dtype=float)

    if A.ndim != 2 or A.shape[0] == 0 or A.shape != B.shape:
        return 0.5

    # Raw (unweighted) tally margin per trial.
    d = (A > B).sum(axis=1) - (B > A).sum(axis=1)

    # The |d| = 1 family: 8 unique pairs x 6 reps = 48 trials/subject.
    mask = np.abs(d) == 1
    if mask.sum() < 10:
        # Defensive fallback: use all non-tied trials.
        mask = d != 0
    if mask.sum() == 0:
        return 0.5

    dm = d[mask]
    rm = resp[mask]
    # Count-winner is A when d>0 (response 0), B when d<0 (response 1).
    consistent = np.where(dm > 0, rm == 0, rm == 1)
    return float(np.mean(consistent))
```

**Observed (real) value:** 0.5133 (var=0.0042)
**Candidate trajectory (this loop):**
  - iter 1: 0.4129 (var=0.0030) (Δ vs real -0.1004)
  - iter 2: 0.3829 (var=0.0018) (Δ vs real -0.1304)
  - iter 3: 0.4067 (var=0.0026) (Δ vs real -0.1067)
  - iter 4: 0.4592 (var=0.0033) (Δ vs real -0.0542)
  - iter 5: 0.4900 (var=0.0043) (Δ vs real -0.0233)
  - iter 6: 0.5233 (var=0.0037) (Δ vs real +0.0100)
  - iter 7 (current): 0.4871 (var=0.0041) (Δ vs real -0.0262)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8387 (var=0.0140)
- pi_3: 0.6062 (var=0.0085)
- pi_1: 0.6825 (var=0.0042)
- pi_4: 0.5192 (var=0.0023)
- pi_5: 0.3950 (var=0.0023)
- pi_6: 0.3850 (var=0.0041)
- pi_7: 0.5425 (var=0.0051)

### Experiment 5
**Design**
  A=[1, 0, 1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1, 0, 1]  B=[1, 1, 0, 1, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0, 1, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    # Chance-cue sensitivity on "chance-only" (Z-family) trials.
    # Design validities: [0.95, 0.85, 0.75, 0.65, 0.5, 0.5, 0.5, 0.5]
    # -> features 0-3 are diagnostic, features 4-7 are the four v=0.5 experts.
    # A Z-family trial is one where ALL diagnostic cues tie, so the pair is
    # discriminated ONLY by the 50%-accurate experts. On such a trial let
    # k = net number of chance cues won by A (signed). The competing theory
    # (llr-rectified weights) annihilates v=0.5 cues exactly, so p(A)=0.5
    # regardless of k -> subjects match the k-implied side at chance (metric 0).
    # My theory gives v=0.5 cues weight 0.5^gamma, so choices should follow
    # sign(k) above chance (metric > 0).
    DIAG = [0, 1, 2, 3]
    CHANCE = [4, 5, 6, 7]

    a_ratings = data["option_a_ratings"].apply(lambda x: np.asarray(list(x), dtype=int))
    b_ratings = data["option_b_ratings"].apply(lambda x: np.asarray(list(x), dtype=int))
    responses = data["response"].to_numpy()

    correct = []
    for av, bv, r in zip(a_ratings, b_ratings, responses):
        if av.shape[0] != 8 or bv.shape[0] != 8:
            continue
        # Keep only trials where every diagnostic cue ties (chance-only trials)
        if not np.array_equal(av[DIAG], bv[DIAG]):
            continue
        # Net chance-cue advantage for A
        k = int(np.sign(av[CHANCE] - bv[CHANCE]).sum())
        if k == 0:
            continue
        chose_a = (int(r) == 0)  # response: 0 = A, 1 = B
        followed = (k > 0 and chose_a) or (k < 0 and not chose_a)
        correct.append(1.0 if followed else 0.0)

    if len(correct) == 0:
        return 0.0
    # 2 * (accuracy - 0.5): 0 = chance-level following of chance cues,
    # positive = subjects systematically follow the 50%-accurate experts.
    return 2.0 * (float(np.mean(correct)) - 0.5)
```

**Observed (real) value:** -0.1683 (var=0.0124)
**Candidate trajectory (this loop):**
  - iter 1: -0.1733 (var=0.0181) (Δ vs real -0.0050)
  - iter 2: -0.1592 (var=0.0187) (Δ vs real +0.0092)
  - iter 3: -0.1633 (var=0.0188) (Δ vs real +0.0050)
  - iter 4: -0.2492 (var=0.0152) (Δ vs real -0.0808)
  - iter 5: -0.1575 (var=0.0172) (Δ vs real +0.0108)
  - iter 6: -0.1717 (var=0.0130) (Δ vs real -0.0033)
  - iter 7 (current): -0.1550 (var=0.0191) (Δ vs real +0.0133)
**Other theories' values on this metric (for reference):**
- pi_3: 0.2958 (var=0.0606)
- pi_4: 0.0308 (var=0.0233)
- pi_1: 0.1625 (var=0.0145)
- pi_2: 0.7617 (var=0.0356)
- pi_5: -0.1167 (var=0.0195)
- pi_6: -0.1350 (var=0.0257)
- pi_7: -0.1425 (var=0.0147)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Strong-Side Choice Rate on Multi-Weak Conflict trials (SSR).

    Selects the five conflict cells in which a high-validity side (99%
    experts, or 75% experts for R3) faces TWO OR MORE opposing weak
    experts: K2 (1x99 vs 2 weak), K3 (1x99 vs 3 weak), M3 (2x99 vs 3
    weak), M4 (2x99 vs 4 weak), R3 (2x75 vs 3 weak) -- 5 unique pairs
    x 2 mirrors x 4 reps = 40 trials per subject. Returns the
    proportion of trials on which the subject chose the strong side.
    """
    hits = 0
    n = 0

    a_col = data['option_a_ratings']
    b_col = data['option_b_ratings']
    r_col = data['response']

    for a, b, r in zip(a_col, b_col, r_col):
        a = [int(x) for x in a]
        b = [int(x) for x in b]
        if len(a) < 8 or len(b) < 8:
            continue

        # Validity-class vote balances (features 0-1 = 0.99 experts,
        # 2-3 = 0.75 experts, 4-7 = weak experts).
        d_hi = (a[0] + a[1]) - (b[0] + b[1])
        d_mid = (a[2] + a[3]) - (b[2] + b[3])
        d_lo = (a[4] + a[5] + a[6] + a[7]) - (b[4] + b[5] + b[6] + b[7])

        # K2/K3/M3/M4: super experts alone on one side (no 75s involved)
        # opposing >= 2 weak experts.
        if d_mid == 0 and d_hi != 0 and abs(d_lo) >= 2:
            target_is_a = d_hi > 0
        # R3: two 75% experts vs three weak experts.
        elif d_hi == 0 and abs(d_mid) == 2 and abs(d_lo) == 3:
            target_is_a = d_mid > 0
        else:
            # Excludes K1 (|d_lo|=1), R1, W2/W3 (no strong side),
            # W4a (|d_mid|=1), G1 (mixed 99+75 side), X1 (all tied).
            continue

        n += 1
        chose_a = (int(r) == 0)
        if chose_a == target_is_a:
            hits += 1

    if n == 0:
        return 0.5
    return hits / n

```

**Observed (real) value:** 0.1180 (var=0.0074)
**Candidate trajectory (this loop):**
  - iter 1: 0.2960 (var=0.0067) (Δ vs real +0.1780)
  - iter 2: 0.1435 (var=0.0030) (Δ vs real +0.0255)
  - iter 3: 0.1865 (var=0.0027) (Δ vs real +0.0685)
  - iter 4: 0.1255 (var=0.0028) (Δ vs real +0.0075)
  - iter 5: 0.1360 (var=0.0028) (Δ vs real +0.0180)
  - iter 6: 0.1065 (var=0.0027) (Δ vs real -0.0115)
  - iter 7 (current): 0.1270 (var=0.0028) (Δ vs real +0.0090)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7670 (var=0.0039)
- pi_3: 0.4190 (var=0.0107)
- pi_1: 0.8475 (var=0.0076)
- pi_2: 0.1180 (var=0.0091)
- pi_5: 0.2870 (var=0.0027)
- pi_6: 0.4360 (var=0.0065)
- pi_7: 0.1235 (var=0.0028)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Super-expert contrarianism index.

    Over all trials in which the feature-0 expert (the 99%-validity
    expert) discriminates between the two options (a[0] != b[0]),
    compute the proportion of choices that go AGAINST the option
    endorsed by that expert.

    Advocated theory (inverted trust): the 99% expert's endorsement is
    anti-diagnostic (w(0.99) ~ -1 trusted-cue unit), so subjects
    predominantly choose AGAINST it -> index ~0.8-0.9.

    Competing theory (flattened diagnosticity): the 99% expert gets the
    LARGEST positive weight, so subjects predominantly FOLLOW it ->
    index ~0.2-0.35.
    """
    n_anti = 0
    n_sel = 0
    for ra, rb, resp in zip(
        data["option_a_ratings"], data["option_b_ratings"], data["response"]
    ):
        a = np.asarray(ra, dtype=int).ravel()
        b = np.asarray(rb, dtype=int).ravel()
        if a.size == 0 or b.size == 0 or a[0] == b[0]:
            continue  # expert 0 does not discriminate on this trial
        # Option index endorsed by the 99%-validity (feature-0) expert.
        endorsed = 0 if a[0] == 1 else 1
        r = int(resp)
        n_sel += 1
        if r != endorsed:
            n_anti += 1
    if n_sel == 0:
        return 0.5
    return float(n_anti) / float(n_sel)

```

**Observed (real) value:** 0.5796 (var=0.0066)
**Candidate trajectory (this loop):**
  - iter 1: 0.3418 (var=0.0050) (Δ vs real -0.2378)
  - iter 2: 0.3191 (var=0.0056) (Δ vs real -0.2604)
  - iter 3: 0.3182 (var=0.0041) (Δ vs real -0.2613)
  - iter 4: 0.3462 (var=0.0048) (Δ vs real -0.2333)
  - iter 5: 0.3133 (var=0.0043) (Δ vs real -0.2662)
  - iter 6: 0.3302 (var=0.0051) (Δ vs real -0.2493)
  - iter 7 (current): 0.3253 (var=0.0054) (Δ vs real -0.2542)
**Other theories' values on this metric (for reference):**
- pi_5: 0.7902 (var=0.0033)
- pi_4: 0.3191 (var=0.0050)
- pi_1: 0.1662 (var=0.0114)
- pi_2: 0.2569 (var=0.0106)
- pi_3: 0.3831 (var=0.0085)
- pi_6: 0.6756 (var=0.0042)
- pi_7: 0.5787 (var=0.0053)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 1, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """P(follow the 98% expert | 98%-vs-90% conflict trials).

    A 'D-cell' trial is one where the top expert (v=0.98, feature 0) and the
    second expert (v=0.90, feature 1) endorse OPPOSITE options, while the
    third expert (v=0.75, feature 2) is tied. On these trials the metric is
    the fraction of choices that follow the option endorsed by the 98%
    expert. FDEI (compressed diagnosticity) treats 0.98 and 0.90 as nearly
    exchangeable, so this should be ~0.50 (coin flip). SITT (inverted trust)
    assigns the 98% expert a large NEGATIVE weight, so subjects should
    systematically choose AGAINST the 98% endorsement (~0.05-0.10).
    """
    if data is None or len(data) == 0:
        return 0.5

    follow_flags = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.ndim != 1 or a.shape[0] < 3 or a.shape != b.shape:
            continue

        d0 = a[0] - b[0]  # 98% expert's relative endorsement of A
        d1 = a[1] - b[1]  # 90% expert's relative endorsement of A
        d2 = a[2] - b[2]  # 75% expert must be tied (no interference)

        # Need a genuine 98-vs-90 conflict with the 75% expert tied.
        if d0 == 0 or d1 == 0 or d2 != 0:
            continue
        if np.sign(d0) == np.sign(d1):
            continue

        endorsed_a = d0 > 0          # 98% expert endorses option A
        chose_a = (int(row["response"]) == 0)
        follow_flags.append(1.0 if (endorsed_a == chose_a) else 0.0)

    if len(follow_flags) == 0:
        return 0.5
    return float(np.mean(follow_flags))
```

**Observed (real) value:** 0.4625 (var=0.0103)
**Candidate trajectory (this loop):**
  - iter 1: 0.4250 (var=0.0255) (Δ vs real -0.0375)
  - iter 2: 0.2225 (var=0.0119) (Δ vs real -0.2400)
  - iter 3: 0.4412 (var=0.0127) (Δ vs real -0.0213)
  - iter 4: 0.4313 (var=0.0200) (Δ vs real -0.0312)
  - iter 5: 0.4288 (var=0.0180) (Δ vs real -0.0338)
  - iter 6: 0.4625 (var=0.0098) (Δ vs real +0.0000)
  - iter 7 (current): 0.4813 (var=0.0200) (Δ vs real +0.0187)
**Other theories' values on this metric (for reference):**
- pi_4: 0.5400 (var=0.0192)
- pi_5: 0.0550 (var=0.0029)
- pi_1: 0.8688 (var=0.0163)
- pi_2: 0.6587 (var=0.0158)
- pi_3: 0.6062 (var=0.0141)
- pi_6: 0.4213 (var=0.0156)
- pi_7: 0.4813 (var=0.0186)

### Experiment 9
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1]  B=[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Anti-super-expert choice rate on the 10 pre-registered 'cliff conflict'
    trials (S1/S1m, S2/S2m, D1/D1m, D2/D2m, V1/V1m).

    On each of these trials exactly one option is endorsed by the
    99%-validity experts (features 0 and 1) while the rival option is
    endorsed only by experts of validity <= 0.98, and the 99%-endorsed
    side has no 0.90-class support (this excludes the G1/G1m agreement
    anchors).  The metric is the fraction of these trials on which the
    subject chooses the option NOT endorsed by the 99% experts.

    BSST (sharp distrust cliff between 0.98 and 0.99, w(0.99) ~ -1.1
    trusted-cue units, w(0.98) ~ +0.6, w(0.90) ~ +0.9, w(0.50) ~ -0.23):
    the 99%-endorsed side is disfavored on every one of these cells, with
    predicted anti-99% choice rates of roughly S1 .86-.90, S2 .90-.95,
    D1 .72-.79, D2 .85-.94, V1 .70-.76  ->  overall ~0.84.

    FDEI (delta ~ 0.15 compression makes 0.98 and 0.99 experts nearly
    exchangeable, w99 - w98 in [-0.21, +0.15]): it is locked near 0.50 on
    the D cells, mildly PRO-99% on the S cells (the 99-side also carries
    a tiny validity edge), and strongly pro-99% on V1 (three chance cues
    are annihilated exactly)  ->  overall anti-99% rate ~0.30-0.53,
    center ~0.44.  Sign-disjoint from BSST with a ~0.40 mean gap.
    """

    def as_tuple(cell):
        try:
            return tuple([int(x) for x in cell])
        except Exception:
            return None

    # key: (A ratings, B ratings) -> favored option (0 = A, 1 = B),
    # where 'favored' is the side NOT endorsed by the 99% experts.
    TRIALS = {
        # S1 / S1m  (A or B wins both 0.99; rival wins both 0.90)
        ((1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0), (0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0)): 1,
        ((0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0), (1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0)): 0,
        # S2 / S2m  (one side wins 0.99+0.99+0.98; rival 0.90+0.90+0.55)
        ((1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1), (0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1)): 1,
        ((0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1), (1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1)): 0,
        # D1 / D1m  (one side wins a 0.99; rival wins a 0.98)
        ((0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1), (1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1)): 0,
        ((1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1), (0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1)): 1,
        # D2 / D2m  (one side wins both 0.99; rival wins both 0.98)
        ((0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0), (1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0)): 0,
        ((1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0), (0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0)): 1,
        # V1 / V1m  (one side wins a 0.99; rival wins all three 0.50)
        ((1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0), (0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1)): 1,
        ((0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1), (1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0)): 0,
    }

    hits = []
    a_col = data['option_a_ratings']
    b_col = data['option_b_ratings']
    r_col = data['response']

    for a, b, r in zip(a_col, b_col, r_col):
        ka = as_tuple(a)
        kb = as_tuple(b)
        if ka is None or kb is None:
            continue
        fav = TRIALS.get((ka, kb))
        if fav is None:
            continue
        try:
            ri = int(r)
        except Exception:
            continue
        hits.append(1.0 if ri == fav else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.5600 (var=0.0052)
**Candidate trajectory (this loop):**
  - iter 1: 0.5890 (var=0.0101) (Δ vs real +0.0290)
  - iter 2: 0.7160 (var=0.0039) (Δ vs real +0.1560)
  - iter 3: 0.7680 (var=0.0064) (Δ vs real +0.2080)
  - iter 4: 0.4400 (var=0.0059) (Δ vs real -0.1200)
  - iter 5: 0.5125 (var=0.0067) (Δ vs real -0.0475)
  - iter 6: 0.4595 (var=0.0060) (Δ vs real -0.1005)
  - iter 7 (current): 0.5105 (var=0.0055) (Δ vs real -0.0495)
**Other theories' values on this metric (for reference):**
- pi_6: 0.8115 (var=0.0019)
- pi_4: 0.3985 (var=0.0055)
- pi_1: 0.1815 (var=0.0131)
- pi_2: 0.5685 (var=0.0047)
- pi_3: 0.5220 (var=0.0036)
- pi_5: 0.7830 (var=0.0038)
- pi_7: 0.5715 (var=0.0054)

### Experiment 10
**Design**
  A=[0, 0, 1, 1, 0, 1, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 1, 0, 1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 1, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # "Follow-the-99%-expert rate": across all trials where the single
    # 0.99-validity expert (feature index 0 in the fixed validity vector
    # [0.99, 0.98, 0.95, 0.90, 0.85, 0.75, 0.65, ., ., .]) discriminates
    # between A and B, compute the proportion of choices that go to the
    # option the 0.99 expert rated 1 (i.e., the option the top expert
    # "endorsed").
    #
    # FDEI: w(0.99) is the LARGEST positive weight in the whole vector
    # (its compression never inverts the validity ordering), so the
    # 0.99-favored option is followed above chance on essentially every
    # such trial -> metric ~ 0.60-0.65.
    # BSST: w(0.99) is strongly NEGATIVE (~ -1.1 trusted-cue units, the
    # distrust cliff), so the 0.99-favored option is chosen BELOW chance
    # on every such trial (the background cues are balanced/weak in this
    # design) -> metric ~ 0.15-0.35.
    # The sign of (metric - 0.5) alone separates the two theories on
    # every trial in the 0.99-discriminating set.

    TOP = 0  # index of the 0.99-validity expert, fixed by the design

    resp = data['response'].to_numpy()
    follows = []

    for i in range(len(data)):
        a_raw = data['option_a_ratings'].iloc[i]
        b_raw = data['option_b_ratings'].iloc[i]
        try:
            av = np.asarray(list(a_raw), dtype=int).ravel()
            bv = np.asarray(list(b_raw), dtype=int).ravel()
        except Exception:
            continue
        if av.size <= TOP or bv.size <= TOP:
            continue

        d = int(av[TOP]) - int(bv[TOP])
        if d == 0:
            continue  # 0.99 expert tied on this trial -> uninformative

        favored_a = d > 0          # option the 0.99 expert endorsed
        chose_a = (int(resp[i]) == 0)
        follows.append(1.0 if (chose_a == favored_a) else 0.0)

    if len(follows) == 0:
        return 0.5  # neutral fallback (should not occur: 8 trials x 4 reps)

    return float(np.mean(follows))
```

**Observed (real) value:** 0.5988 (var=0.0026)
**Candidate trajectory (this loop):**
  - iter 1: 0.5375 (var=0.0072) (Δ vs real -0.0613)
  - iter 2: 0.6162 (var=0.0109) (Δ vs real +0.0175)
  - iter 3: 0.5975 (var=0.0087) (Δ vs real -0.0012)
  - iter 4: 0.5913 (var=0.0073) (Δ vs real -0.0075)
  - iter 5: 0.5938 (var=0.0084) (Δ vs real -0.0050)
  - iter 6: 0.5869 (var=0.0062) (Δ vs real -0.0119)
  - iter 7 (current): 0.6081 (var=0.0052) (Δ vs real +0.0094)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6719 (var=0.0085)
- pi_6: 0.2988 (var=0.0058)
- pi_1: 0.8525 (var=0.0127)
- pi_2: 0.6687 (var=0.0078)
- pi_3: 0.6206 (var=0.0104)
- pi_5: 0.1444 (var=0.0034)
- pi_7: 0.6094 (var=0.0057)

### Experiment 11
**Design**
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 1, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 1, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 0, 1, 0, 0, 1, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0, 1, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    """
    Theory-discriminant index for the 10-expert panel
    validities = [0.99, 0.90, 0.80, 0.70, 0.60, 0.50, 0.50, 0.50, 0.50, 0.50].

    Both theories reduce to a FIXED weight vector over the 10 experts on
    this panel. At each theory's parameter-box midpoint we compute its
    predicted p(choose A) for the observed trial, take the difference
    d_t = p_CREDT(t) - p_FDEI(t), and average d_t * y_t over all trials,
    where y_t = +1 if the subject chose A and -1 if B.

    E[metric] = (1/T) * sum_t d_t * (2*p_true(t) - 1).
    If CREDT generated the data this is strongly POSITIVE (~ +0.045);
    if FDEI generated the data it is NEGATIVE (~ -0.022). The contrast
    is driven by the cells where the boxes are disjoint: the Z-family
    (chance-cue ladders: CREDT 0.59-0.74 vs FDEI exactly 0.50), the
    0.99-expert trials (CREDT ~0.28-0.36 vs FDEI ~0.55-0.60), the
    X-family sign flips, and K1 (four trusted experts vs four
    coin-flippers: CREDT ~0.50 vs FDEI ~0.78).
    """
    validities = np.array([0.99, 0.90, 0.80, 0.70, 0.60, 0.50, 0.50, 0.50, 0.50, 0.50])

    # ---------- CREDT weights at parameter-box midpoints ----------
    w_p, v_lo, v_hi, kappa_lo = 0.90, 0.53, 0.9815, 0.20
    g0, s_gap = 0.043, 0.013
    rho, d_w, lambda_ext = 1.36, 0.40, 0.40
    g_star, s_l, s_r = 0.048, 0.012, 0.033
    c_half, s_c, c_amp = 0.36, 0.06, 1.29
    beta_c, eps_c = 0.42, 0.05

    v = np.clip(validities, 0.5, 1.0)
    v_top = float(v.max())
    below = v[v < v_top - 1e-9]
    v_second = float(below.max()) if below.size > 0 else 0.5
    gap = v_top - v_second
    outlier = v_top > v_hi
    if outlier:
        sig_gap = 0.5 * (1.0 + np.tanh(0.5 * ((gap - g0) / s_gap)))
    else:
        sig_gap = 0.0
    gate = (lambda_ext + (1.0 - lambda_ext) * sig_gap) if outlier else 0.0

    w_c = np.full(10, w_p, dtype=float)
    flank = v < v_lo
    if np.any(flank):
        w_fl = -kappa_lo + (w_p + kappa_lo) * np.clip(v[flank] - 0.5, 0.0, None) / (v_lo - 0.5)
        w_c[flank] = (1.0 - gate) * w_fl + gate * w_p
    if outlier:
        c_frac = float(flank.mean())
        C = c_amp * 0.5 * (1.0 + np.tanh(0.5 * (c_frac - c_half) / s_c))
        width = s_l if gap < g_star else s_r
        P = float(np.exp(-0.5 * ((gap - g_star) / width) ** 2))
        w_top = w_p - d_w * (v_top - v_hi) / (1.0 - v_hi) - rho * (P + C * sig_gap)
        w_c[v >= v_top - 1e-9] = w_top

    # ---------- FDEI weights at parameter-box midpoints ----------
    delta, tau = 0.15, 0.02
    beta_f, eps_f = 0.32, 0.025
    vv = np.clip(validities, 0.5, 1.0 - 1e-9)
    llr = np.log(vv / (1.0 - vv))
    w_f = np.power(np.maximum(llr - tau, 0.0), delta)

    def _p_choose_a(s, beta, eps):
        # low-gain softmax over [s, -s] plus uniform lapse
        z = beta * s
        p_core = 1.0 / (1.0 + np.exp(-z))
        return (1.0 - eps) * p_core + eps * 0.5

    total = 0.0
    n = 0
    for a_cell, b_cell, resp in zip(data['option_a_ratings'],
                                    data['option_b_ratings'],
                                    data['response']):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        votes = np.sign(a - b)
        s_c = float(np.dot(w_c, votes))
        s_f = float(np.dot(w_f, votes))
        p_c = _p_choose_a(s_c, beta_c, eps_c)
        p_f = _p_choose_a(s_f, beta_f, eps_f)
        d_t = p_c - p_f
        y = 1.0 if int(resp) == 0 else -1.0
        total += d_t * y
        n += 1

    if n == 0:
        return 0.0
    return float(total / n)
```

**Observed (real) value:** 0.0206 (var=0.0003)
**Candidate trajectory (this loop):**
  - iter 1: 0.0324 (var=0.0004) (Δ vs real +0.0118)
  - iter 2: 0.0430 (var=0.0003) (Δ vs real +0.0224)
  - iter 3: 0.0339 (var=0.0002) (Δ vs real +0.0133)
  - iter 4: -0.0373 (var=0.0004) (Δ vs real -0.0579)
  - iter 5: 0.0141 (var=0.0003) (Δ vs real -0.0065)
  - iter 6: 0.0269 (var=0.0002) (Δ vs real +0.0064)
  - iter 7 (current): -0.0191 (var=0.0003) (Δ vs real -0.0397)
**Other theories' values on this metric (for reference):**
- pi_7: 0.0758 (var=0.0002)
- pi_4: -0.0398 (var=0.0003)
- pi_1: -0.0446 (var=0.0003)
- pi_2: 0.0431 (var=0.0003)
- pi_3: 0.0148 (var=0.0004)
- pi_5: -0.0288 (var=0.0001)
- pi_6: -0.0227 (var=0.0002)

### Experiment 12
**Design**
  A=[1, 1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 1, 0, 1]  B=[1, 1, 0, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Super-expert deference rate.

    Among all trials on which the highest-validity expert (feature 0, the
    99% expert) discriminates between option A and option B, return the
    fraction of choices that FOLLOW that expert's recommendation (i.e.
    chose the option the 99% expert rated 1).

    FDEI (flattened-diagnosticity integration) gives the 0.99 expert its
    largest positive weight, so it predicts deference well above chance
    (~0.57 overall, ~0.69 on isolated super-expert trials, but BELOW 0.5
    when a 0.94+0.65+0.50 coalition opposes it).

    CREDT, on this panel (0.99 sitting 0.05 above the 0.94 rung with a
    partially chance-laden scale), assigns the panel-max expert a large
    NEGATIVE weight, so it predicts systematic ANTI-deference (~0.21).
    """
    df = data.reset_index(drop=True)

    def _first_rating(cell):
        # Cells may be list / tuple / np.ndarray of 0/1 ints, or a string
        # serialization like "[1, 0, 1, ...]". Robustly extract element 0.
        if isinstance(cell, str):
            s = cell.strip().strip("[]")
            parts = s.replace(",", " ").split()
            return int(float(parts[0]))
        seq = list(cell)
        return int(seq[0])

    a_col = df["option_a_ratings"]
    b_col = df["option_b_ratings"]
    r_col = df["response"]

    n = 0
    follow = 0
    for i in range(len(df)):
        try:
            a0 = _first_rating(a_col.iloc[i])
            b0 = _first_rating(b_col.iloc[i])
        except Exception:
            continue
        # Skip trials where the 99% expert does not discriminate.
        if a0 not in (0, 1) or b0 not in (0, 1) or a0 == b0:
            continue
        # Option endorsed by the 99% expert: 0 = A, 1 = B.
        winner = 0 if a0 == 1 else 1
        resp = int(r_col.iloc[i])
        n += 1
        if resp == winner:
            follow += 1

    if n == 0:
        return 0.5
    return float(follow) / float(n)
```

**Observed (real) value:** 0.4525 (var=0.0028)
**Candidate trajectory (this loop):**
  - iter 1: 0.4713 (var=0.0051) (Δ vs real +0.0187)
  - iter 2: 0.3513 (var=0.0072) (Δ vs real -0.1013)
  - iter 3: 0.4229 (var=0.0077) (Δ vs real -0.0296)
  - iter 4: 0.5429 (var=0.0043) (Δ vs real +0.0904)
  - iter 5: 0.5471 (var=0.0045) (Δ vs real +0.0946)
  - iter 6: 0.5221 (var=0.0055) (Δ vs real +0.0696)
  - iter 7 (current): 0.5096 (var=0.0048) (Δ vs real +0.0571)
**Other theories' values on this metric (for reference):**
- pi_4: 0.6017 (var=0.0039)
- pi_7: 0.2367 (var=0.0031)
- pi_1: 0.8358 (var=0.0153)
- pi_2: 0.6479 (var=0.0042)
- pi_3: 0.5867 (var=0.0071)
- pi_5: 0.1121 (var=0.0017)
- pi_6: 0.2667 (var=0.0041)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: PIGD is a genuine step forward: it is faithful to the arbiter's prescribed family (flattened-diagnosticity baseline + composition-driven above-ceiling distrust, no gap bump), and it nails the experiments that falsified both parents — Exp 5 (-0.173 vs real -0.168, the negative chance-cue following FDEI could not produce) and Exp 12 (0.471 vs real 0.4525, the regime where CREDT's gap bump catastrophically failed). Exps 1, 2, 3, 8, 9, 11 are all essentially on top of the real values. However, the composition-driven distrust mechanism is systematically UNDER-calibrated, and the misses cluster exactly where the theory stakes its distinctive claim. (a) Experiment 7 (super-expert contrarianism): real 0.580, candidate 0.342 — a 0.24 miss in the PRO-following direction. The 0.99 expert still carries net positive weight on that ~3/8-chance panel; the sigmoid is centered too high (c_half ≈ 0.40) and/or d_top too small, so phi at c_frac ≈ 0.375 is too weak to flip the sign of w(0.99). (b) Experiment 6 (strong-side rejection): real 0.118, candidate 0.296 — an 0.18 miss. The panel-spread discount (d_panel) is not strong enough to make subjects abandon the 75%-expert side in the R3 cell; the proposer's own rationale predicted this cell would work, and it does not. (c) Experiment 4: real 0.513, candidate 0.413 — the chance-cue anti-weighting leaks too much negativity onto this panel, pushing |d|=1 tally agreement below chance where the real data sit at chance (FDEI's exact annihilation gives 0.519 there). (d) Experiment 10: real 0.599, candidate 0.538 — distrust fires too readily on this coherent dense-ladder panel (c_frac ≈ 0.3), dragging 99-following down. The common root cause is that the incoherence sigmoid is too shallow and too high-centered: it cannot simultaneously give phi ≈ 0 at c_frac ≈ 0.30 (Exp 10), moderate-to-strong phi at c_frac ≈ 0.375 (Exp 7), and strong phi at c_frac = 0.5 (Exps 5, 6). A sharper, lower-centered sigmoid separates these regimes. I also flag a likely bug: the v_c parameter range is listed as [0.971, 0.76] — min > max, an empty/inverted interval; this should be corrected (e.g., [0.965, 0.978]) so the fitter explores it properly.
Rationale: The candidate was accepted by the gate, but the fit is not yet good enough to ship: two large misses (Exp 7: 0.342 vs 0.580; Exp 6: 0.296 vs 0.118) sit precisely on the theory's signature mechanism, plus moderate misses on Exps 4 and 10. All fixes are in-family recalibrations of the existing PIGD machinery — do NOT change mechanism families. (1) Sharpen and lower-center the incoherence sigmoid: reduce s_c from 0.06–0.08 to roughly 0.04–0.05 and move c_half to roughly 0.35–0.37. This yields phi ≈ 0 at c_frac ≈ 0.30 (Exp 10 keeps full pro-99 following ~0.60), phi ≈ 0.5–0.6 at c_frac ≈ 0.375 (Exp 7's 0.99 weight flips mildly negative, index rises toward 0.58), and phi ≈ 1 at c_frac = 0.5 (Exps 5/6 unchanged). (2) Raise d_top (toward 5+) so w(0.99) actually goes negative at Exp 7's incoherence level; verify Exp 12 (partially chance-laden, phi ≈ 0.15–0.2) stays near chance ~0.45–0.50 and Exp 8 stays ~0.46. (3) Strengthen the panel spread for Exp 6: increase d_panel (or its ratio to d_top) so the 0.75 experts lose enough weight on a half-chance panel that R3's strong-side rate falls toward 0.12, while checking Exp 9's S-cells (currently good at 0.589 vs 0.56) do not degrade — the sharper sigmoid should keep phi low there. (4) For Exp 4, the same sharper sigmoid should reduce the kappa leakage at that panel's lower c_frac; if it still undershoots 0.51, trim kappa slightly (Exp 5 has c_frac = 0.5 where phi ≈ 1, so it tolerates a somewhat smaller kappa). (5) Fix the malformed v_c range [0.971, 0.76] to a valid interval around 0.97–0.98. Keep the two motivated deviations from the arbiter (composition-alone gating of the chance-cue anti-weight; the d_panel spread term) — both are within the prescribed family and both are validated by Exps 5 and 12.

**Outcome of your advice:** iter 1 candidate loss=0.0999 -> iter 2 candidate loss=0.1393 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The gate rejected this candidate (loss 0.1393 vs the accepted base's 0.0999), and the per-experiment pattern shows exactly why. The recalibration was a bundle of five changes (v_c bug fix, sharper/lower-centered sigmoid, higher d_top, higher d_panel, trimmed kappa), and its effects split cleanly into wins and losses. WINS: Exp 6 is essentially fixed (0.296 -> 0.1435 vs real 0.118; the d_panel raise did its job), Exp 10 is fixed (0.538 -> 0.616 vs real 0.599; the sharp sigmoid's low flank plus the v_c repair removed the leakage), and Exp 1 is now exact (-0.2185 vs -0.2180). LOSSES: the mid-incoherence range is now systematically over-distrusted — Exp 8 collapsed from 0.425 to 0.2225 (real 0.4625), Exp 9 overshot from 0.589 to 0.716 (real 0.560), Exp 12 dropped from 0.471 to 0.351 (real 0.4525), and Exp 4 slipped further from 0.413 to 0.383 (real 0.5133). All four moved in the anti-top-expert direction past the real values, which fingerprints a single cause: at c_frac in the 0.37-0.45 band, phi*d_top is now too large. My iteration-1 advice was therefore only half right — the sharpness (s_c) and d_panel recommendations paid off, but the lower center (c_half 0.36-0.39) combined with d_top 4.4-5.2 over-discounted exactly the panels (8, 9, 12) that iteration 1 had on target. The deepest problem, however, is untouched: Experiment 7. Its value did not move at all between iterations (0.342 -> 0.319) despite the sigmoid center shifting 0.025 and d_top rising ~20%, and 0.319 is numerically identical to pure FDEI (pi_4 = 0.3191). This is proof that phi ~ 0 on Exp 7's panel: the composition gate never opens there. The proposer's rationale assumed c_frac ~ 0.375 for Exp 7 ('Phi(0.375) ~ 0.5 (Exps 7/12)'), and the simulation falsifies that assumption — Exp 7's actual chance fraction (under the v_lo ~ 0.52-0.54 classification) must sit well below the sigmoid's active range. No parametric placement of a chance-count-only sigmoid can fix this: real Exp 7 (0.580) needs distrust at its c_frac while real Exp 10 (0.599) needs none at c_frac ~ 0.30, and if Exp 7's chance fraction is ~0.25 those two regimes are inseparable on that statistic. The incoherence INPUT, not the sigmoid's placement, is the miscalibrated part.
Rationale: Regenerate, with the following in-family corrections on top of the unchanged iter-1 base. (1) UNDO THE MID-RANGE OVERSHOOT, KEEP THE GOOD PARTS. Keep the sharp s_c (~0.035-0.045) and the v_c fix ([0.966, 0.976]) and the d_panel raise (2.5-3.0, which fixed Exp 6) and the trimmed kappa — these are validated. But move c_half back up to ~0.39-0.42 and pull d_top back to ~4.0-4.4 (i.e., between the two flanking iterations, not a full reversal). Target phi table: phi ~ 0.02-0.08 at c_frac ~ 0.30 (Exp 10 keeps ~0.60 following), phi ~ 0.3-0.4 at c_frac ~ 0.375-0.44 (Exps 8/12 land near their real near-chance values 0.46, not the current 0.22/0.35), phi >= 0.9 at c_frac = 0.5 (Exps 5/6 unchanged). (2) FIX EXP 7 BY FIXING THE INCOHERENCE INPUT, NOT THE SIGMOID. First, recompute the actual per-experiment c_frac using the true validities and the v_lo classification — the rationale's phi table is demonstrably wrong for Exp 7. If Exp 7's chance fraction is genuinely ~0.25, then counting only v~0.5 experts cannot separate 'distrust at 0.25' (Exp 7, real 0.580) from 'no distrust at 0.30' (Exp 10, real 0.599). In that case broaden the incoherence statistic WITHIN the prescribed family: count weak/unreliable experts, not just chance-level ones — e.g., fraction of experts below a threshold v_weak ~ 0.60-0.70, or a soft fractional count sum_j max(0, (v_weak - v_j)/(v_weak - 0.5)) — and re-place the sigmoid on that richer statistic so the phi table separates Exp 7 (distrust fires, w(0.99) goes mildly negative, index -> ~0.58) from Exp 10 (dense 0.98/0.95/0.90 ladder, phi stays ~0). This is a recalibration of the composition measure the arbiter's family already requires, not a new mechanism; do NOT reintroduce any rung-gap term. (3) EXP 4: both iterations sit ~0.10 below the real 0.513, which is FDEI-like (pi_4 = 0.519). Whatever branch activates on Exp 4's panel (chance anti-weight and/or the d_panel spread) over-fires there; after correcting the phi table, verify Exp 4's composition lands on the low flank so its weak cues are annihilated rather than inverted. (4) Keep the two validated deviations from the arbiter spec (composition-alone gating of the chance anti-weight — required by Exp 5's -0.168 without any above-ceiling outlier; and the d_panel spread — required by Exp 6). (5) Self-correction note: my iteration-1 push for a lower center and bigger d_top was rejected by the gate in this combined form; the recommendation above deliberately targets intermediate values (c_half ~0.39-0.42, d_top ~4.0-4.4) rather than oscillating back to the iter-1 settings wholesale, and redirects the remaining error budget to the one structural knob (the incoherence input) that two rounds of sigmoid tuning have proven cannot reach Experiment 7.

**Outcome of your advice:** iter 2 candidate loss=0.1393 -> iter 3 candidate loss=0.1117 -> the gate marked it REJECTED.

### Iteration 3
Verdict: regenerate
Interpretation: The gate rejected this candidate (loss 0.1117 vs the accepted base's 0.0999), so the verdict is regenerate regardless; the per-experiment pattern shows both real progress and a now-clearly-identified structural wall. WINS relative to iter 2: the v_c repair (ceiling strictly between 0.98 and 0.99) fixed Experiment 8 (0.441 vs real 0.4625) and largely restored Experiment 12 (0.423 vs 0.4525); Experiments 1, 2, 3, 5, 10, 11 remain essentially on target (Exp 10 is now exact: 0.5975 vs 0.5988). LOSSES: (a) Experiment 7 did not move AT ALL (0.342 -> 0.319 -> 0.318 vs real 0.580) despite the soft weak-mass statistic being introduced specifically to open its gate — it is still numerically identical to pure FDEI (pi_4 = 0.3191), so phi remains ~0 on that panel even under the soft statistic. (b) Experiment 9 got WORSE for the second straight iteration (0.589 -> 0.716 -> 0.768 vs real 0.560): the soft statistic added partial mass from Exp 9's 0.55-band experts, pushing its c past the sigmoid center and over-firing the claimant discount. (c) Experiment 4 is stuck ~0.10 low for the third iteration (0.413 / 0.383 / 0.407 vs real 0.513; FDEI reference 0.519). (d) Experiment 6 regressed (0.1435 -> 0.1865 vs real 0.118) because the recentered sigmoid (c_half ~0.415) dropped phi below saturation on its half-chance panel. The deep diagnosis: the soft weak-mass statistic orders the panels WRONG for the theory's own signature claims — Experiment 7 (needs STRONG distrust, w(0.99) < 0) now sits BELOW Experiment 9 (needs only MILD distrust) on the c statistic, so NO monotone sigmoid placement on weak-mass alone can fit both. Lowering c_half to open Exp 7's gate would further over-discount Exp 9; raising it does the reverse. Two rounds of sigmoid tuning (iter 2) and one round of input-statistic broadening (iter 3) have both failed on exactly this pair. The missing panel dimension is CORROBORATION: Exp 9's 0.99 sits atop 0.99/0.98/0.90/0.90 support and Exp 10's atop a dense 0.98/0.95/0.90 ladder, while Exp 7's 0.99 floats above much weaker experts with no near-rung support. The candidate remains faithful to the prescribed family (flattened-diagnosticity baseline, composition-driven distrust, no rung-gap term, low-gain softmax + lapse), and the two validated deviations (composition-alone kappa gating, d_panel spread) are correctly retained.
Rationale: Regenerate, on top of the unchanged iter-1 base, with the following in-family corrections. (1) ADD A SECOND COMPOSITION STATISTIC: CORROBORATION DILUTION. The distrust discount for an above-ceiling claimant should be multiplied by (1 - rho * corr_frac), where corr_frac is the soft mass of panel experts with validity in [v_corr, v_c] (v_corr ~ 0.85-0.93, to be placed from the actual validities). This is still panel-COMPOSITION driving distrust — it is the 'dense corroborating ladder' regime the arbiter's own spec names — and it is NOT a rung-gap bump: it is monotone in corroboration mass, never a non-monotone function of the gap. Implemented this way it stays inside the prescribed family. Predicted effects: Exp 9 (0.98 + two 0.90s + second 0.99 corroborating) gets strongly diluted -> anti-99 falls from 0.77 back toward 0.56; Exp 10 (dense ladder) stays at full following ~0.60; Exp 12 (0.94 + 0.90 partial corroboration) lands mildly distrustful ~0.45; Exp 7 (no near-top support) keeps the full Phi(c) discount. (2) WITH EXP 9/10 HANDLED BY DILUTION, THE SIGMOID IS FREED: lower c_half to ~0.32-0.36 on the soft weak-mass scale (sharp s_c ~ 0.04) so that Exp 7's c (~0.31) actually opens the gate, and verify d_top * phi > w_base(0.99) there so the 0.99 weight flips negative and contrarianism rises from 0.32 toward 0.58. This decoupling is what made the previous two rounds fail — the sigmoid was being asked to separate Exp 7 from Exp 9 on a statistic that orders them backwards; it cannot, so stop tuning it against that constraint. The lower center also restores phi ~ 1 at c = 0.5, which should return Exp 6 toward its iter-2 fit (0.14 vs real 0.118) and preserve Exp 5's -0.168. (3) EXPERIMENT 4: three iterations stuck at 0.38-0.41 vs real 0.513 while FDEI sits at 0.519 — some distrust branch over-fires on that panel. Before choosing the fix, INSTRUMENT THE MODEL: for each experiment, print c, phi, corr_frac, and the per-branch weight changes, and include that table in the next rationale — the phi tables in the last two rationales were both demonstrably wrong for at least one experiment, so no more assumed placements. If the culprit is the d_panel spread (plausible if Exp 4's panel has a 0.99 claimant over 0.85/0.75/0.65/0.5s), the corroboration term may fix it for free: place v_corr so Exp 4's second rung (0.85) counts as corroboration while Exp 6's (0.75) does not — that separates the FDEI-like panel (Exp 4, target 0.51) from the strong-distrust panel (Exp 6, target 0.12). If instead the kappa anti-weight is the culprit, compress it (e.g., cap the total chance-cue anti-contribution or make it sublinear in the number of opposing chance cues) so Exp 5's mild -0.168 survives while Exp 4's mixed diagnostic/chance |d|=1 trials return to FDEI-like indifference. (4) Keep v_c strictly in (0.98, 0.99) — that repair is validated by Exp 8 — and keep d_panel, kappa, beta, epsilon near current values. (5) Self-history note: my iter-2 push to 'fix the incoherence input' was directionally right but incomplete — the soft weak-mass alone cannot resolve the Exp 7 / Exp 9 inversion, as this rejection proves; the corroboration dimension is the extension of that same advice, not a reversal of it. Do not reintroduce any rung-gap term and do not switch mechanism families; all changes above are recalibrations of the composition statistics the prescribed PIGD family already requires.

**Outcome of your advice:** iter 3 candidate loss=0.1117 -> iter 4 candidate loss=0.1215 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The gate rejected this candidate (0.1215 vs the accepted iter-1 base's 0.0999), so the verdict is regenerate regardless; the per-experiment pattern shows the corroboration-dilution concept is right but its parameterization is a bludgeon, and the theory's signature experiment remains unreachable after four attempts. WINS vs iter 1: Exp 6 is now essentially exact (0.296 -> 0.1255 vs real 0.1180), Exp 4 improved substantially (0.413 -> 0.4592 vs real 0.5133), Exp 10 is exact (0.5913 vs 0.5988), and Exps 2, 3, 8 remain on target. LOSSES vs iter 1: (a) Exp 9 flipped from +0.029 error to -0.120 (0.4400 vs real 0.5600) — the dilution over-neutralized distrust on the 0.98/0.90/0.90-corroborated panel, turning mild anti-99 into mild pro-99; (b) Exp 12 overshot the same way (0.5429 vs real 0.4525, was 0.4713); (c) Exp 11 sign-flipped (-0.0373 vs real +0.0206, was +0.0324) — the model became more FDEI-like than the data on this panel; (d) Exp 5 regressed hard (-0.2492 vs real -0.1683, was -0.1733) because kappa was raised to 0.44-0.50. All four regressions point the same direction: wherever distrust or anti-weighting fires, it now fires too weakly or too strongly by a common factor — rho_corr (2.2-2.8 with m_floor 0.09-0.15) nearly annihilates the claimant discount on any panel with a 0.90-class rung, and kappa at 0.44-0.50 over-weights chance cues on saturated panels. (e) THE WALL: Experiment 7 did not move for the fourth consecutive iteration (0.342 / 0.319 / 0.318 / 0.346 vs real 0.5796), staying numerically identical to pure FDEI (pi_4 = 0.3191), which proves phi_7 ≈ 0 in the fitted model — directly contradicting the rationale's hand-computed claim of phi_7 ~ 0.5-0.6 and w(0.99) ~ -1.6. That is now the THIRD consecutive falsified hand-computed phi table (iter 2's c_frac ~ 0.375 claim, iter 3's soft-mass claim, iter 4's 'gate actually opens' claim). The proposer is optimizing against an imagined Experiment 7. Either the assumed validities for Exp 7 ('0.99 over 0.75/0.65/0.55-0.60 field') are wrong, or the fitter consistently lands on parameters (c_half at the top of the box, v_weak at the bottom) that keep the gate shut — and no amount of hand arithmetic will distinguish these. My iter-3 demand for machine-generated instrumentation was not met; the table in this rationale is again hand-computed and again falsified by the simulation.
Rationale: Regenerate, on top of the unchanged iter-1 base, with the following in-family corrections. (1) MANDATORY: MACHINE INSTRUMENTATION, NOT HAND TABLES. Before any tuning, run the actual predict() code on each experiment's true validities at the fitted parameters and print, per experiment: c, phi, corr_frac, mult, and the final weight vector. Include that machine-generated output in the next rationale. Three consecutive hand-computed phi tables have been falsified by the simulations (all predicted Exp 7's gate open; all four simulations show phi_7 ≈ 0). Also audit Experiment 7's actual validities against the assumption of a 0.55-0.60 weak field — if its panel is a graded ladder with no experts below ~0.65, the weak-mass statistic with v_weak in [0.63, 0.67] can never open its gate, and the incoherence input must be re-anchored (still composition-only: e.g., weak-mass measured relative to the claimant's validity rather than a fixed anchor — monotone in panel composition, no rung-gap term). (2) SOFTEN THE DILUTION — KEEP THE MECHANISM. The concept is validated (it fixed Exps 4, 6, 10 simultaneously), but rho_corr in [2.2, 2.8] with m_floor in [0.09, 0.15] crushes distrust to near-zero on every corroborated panel. Retune to rho_corr ~ [1.0, 1.6] and m_floor ~ [0.25, 0.40], targeting: Exp 9 anti-99 from 0.44 back to ~0.56, Exp 12 deference from 0.543 down to ~0.45, Exp 11 from -0.037 back to ~+0.02, while Exp 10 (low phi AND high corr) and Exp 6 (corr_frac = 0, mult = 1) stay put. Do NOT remove the dilution term — that would oscillate back toward iter 3's over-distrust (Exp 9 = 0.768). (3) REVERT KAPPA to iter-1-like values ([0.32, 0.40]) to fix Exp 5's regression (-0.249 -> ~-0.17). (4) EXP 4's residual -0.054 error is now most plausibly kappa-driven, not distrust-driven: Exp 4 and Exp 5 have IDENTICAL composition (c = 0.5) but need opposite chance-cue behavior (Exp 4: FDEI-like annihilation, real 0.513 ≈ pi_4's 0.519; Exp 5: anti-weight, real -0.168), and the ONLY panel feature separating them is the presence of the 0.99 claimant. If the instrumentation table confirms kappa drives Exp 4's residual, gate kappa's ramp on the ABSENCE of an above-ceiling claimant (a panel-structure gate, still composition-driven, no gap term) and verify Exp 6's strong-side rejection survives — it should, because the d_panel spread (w(0.75) ~ -2.5) carries that experiment, but confirm with the printed weights. (5) EXP 7, AFTER INSTRUMENTATION: if c_7 genuinely sits at 0.25-0.31, the single sigmoid provably cannot serve both the claimant-discount branch (needs phi high at c ≈ 0.28 for Exp 7) and the kappa branch (needs phi low at c ≈ 0.28 for Exps 2/3) — split them: a lower-centered sigmoid (c_half ~ 0.24-0.28) for the d_top/d_panel discount, a higher-centered one (c_half ~ 0.38-0.42) for kappa. Both remain composition gates; this is the decoupling iter 3 asked for, completed. If instead c_7 ≈ 0 under the true validities, report the validities and re-anchor the incoherence input per point (1) rather than tuning the sigmoid against a statistic that cannot reach the panel. (6) PROTECT THE WINS: keep v_c strictly in (0.98, 0.99) (validated by Exp 8), keep d_panel and d_top near current values, keep the soft weak-mass input, keep the trusted-scale-normalized corr_frac. Exps 1, 2, 3, 6, 8, 10 are currently within noise — check each retuning step against them (Exp 1 has drifted slightly, -0.200 vs -0.218; a small kappa reversion should restore it). Self-history note: my iter-3 push for corroboration dilution was directionally validated (Exps 4/6/10 fixed) but the parameter magnitudes I implicitly tolerated (rho up to 2.8, kappa up to 0.50) over-corrected; the recommendation above scales those back rather than reversing any mechanism.

**Outcome of your advice:** iter 4 candidate loss=0.1215 -> iter 5 candidate loss=0.0716 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: This is the first accepted improvement since iteration 1 (0.0999 -> 0.0716), and the per-experiment pattern confirms that the iter-5 edits — the split gates (low-centered distrust sigmoid, high-centered kappa sigmoid), the claimant-absence gating of kappa, and the softened corroboration dilution — all did what they were designed to do. Exp 4 jumped from a chronic ~0.10 deficit to -0.023 (the kappa gating resolved the Exp 4-vs-Exp 5 contradiction exactly as intended), Exp 6 is now essentially exact (0.136 vs 0.118), Exps 2, 3, 5, 10, 11 are on top of their real values, and Exps 8, 9 are within ~0.04. The claimant-free branch is bit-identical to the accepted base, which is why Exps 1-3/5 held. However, the loss is now dominated by two residuals, one of which is the theory's signature failure: (a) Experiment 7 sits at 0.3133 vs real 0.5796 — numerically indistinguishable from pure FDEI (pi_4 = 0.3191) for the FIFTH consecutive iteration. The rationale's central claim ('G_7 ~ 0.44, w(0.99) ~ -0.6, contrarianism moving toward ~0.58') is again falsified by the simulation: phi_7 ≈ 0 in the fitted model. Since the distrust center was lowered to 0.25-0.29 this iteration and Exp 7 still did not move an inch, the proposer's inferred c_7 ~ 0.25 is wrong — the simulation itself is the strongest evidence that Exp 7's chance fraction under the v_lo ~ 0.53 classification is well below 0.25, almost certainly 0.125 (one chance cue in eight). (b) Experiment 12 regressed relative to the iter-1 base (0.471 -> 0.547 vs real 0.4525, error +0.019 -> +0.095): with rho_corr 2.0-2.4, every panel with corr_frac >= ~0.35 is pinned at the m_floor ~0.21, so the dilution can no longer distinguish Exp 12's sparse 0.94/0.90 ladder from Exp 10's dense 0.98/0.95/0.90/0.85 ladder — Exp 12 lost the discount it needs. (c) Exp 1 has drifted to +0.036 (from +0.013) and Exp 9 sits at -0.048 — both small. The candidate is a genuine improvement and correctly becomes the new base, but roughly half the remaining squared loss is a single identifiable miss (Exp 7, 0.27) that the theory's core mechanism — sign-flipping distrust of an uncorroborated above-ceiling claimant — was built to explain and has never once fired.
Rationale: Regenerate on top of this accepted candidate (which is now the base — do not revert any of the iter-5 edits; the split gates, kappa-on-claimant-absence gating, v_c in (0.98, 0.99), and d_panel spread are all validated by Exps 4, 5, 6, 8 this iteration). Two targeted, in-family recalibrations remain: (1) OPEN EXPERIMENT 7'S GATE — THE CENTER IS STILL TOO HIGH. Five iterations of value ~0.32 (pure FDEI) prove phi_7 ≈ 0 at every center tried (0.36-0.415, then 0.25-0.29). The proposer's inference 'c_7 ~ 0.25' is contradicted by its own simulation; c_7 is almost certainly 0.125. Now that the distrust sigmoid is DECOUPLED from kappa (iter 5's validated edit), lowering c_half is finally SAFE: claimant-free panels (Exps 1, 2, 3, 5, 8) are untouched by construction. Lower c_half to roughly [0.10, 0.16] with sharp s_c ~0.04-0.06 so phi(0.125) ~ 0.4-0.6 and d_top * phi > w_base(0.99), flipping Exp 7's super-expert weight negative and moving contrarianism from 0.31 toward 0.58. The only other low-c claimant panel is Exp 10, which is protected by the dilution floor — which leads to (2). (2) RE-BALANCE THE DILUTION TO RESTORE EXP 12 WITHOUT BREAKING EXP 10. Currently rho_corr 2.0-2.4 with m_floor 0.18-0.24 floors Exps 9, 10, AND 12 identically, destroying the separation the dilution exists to provide (Exp 12 regressed +0.09 this iteration for exactly this reason). Lower rho_corr to ~1.4-1.8 and m_floor to ~0.10-0.16 so that: Exp 10 (corr ~0.57, dense ladder) stays floored at near-zero discount and keeps its exact 0.594 following; Exp 12 (corr ~0.33-0.40, sparse ladder) gets mult ~0.4-0.5, and — with phi_12 rising toward 1 under the lower center — enough total discount to bring deference from 0.547 back down toward 0.45; Exp 9 (high corr, c ~0.3-0.4) should land near its current mildly-anti 0.51-0.56. Note the needed ordering G_10 < G_9 ~ G_12 with corr_10 ~ corr_9 > corr_12 means corroboration alone cannot order Exp 9 vs Exp 12 — if their c values differ (c_9 > c_12), let phi carry that separation; if not, accept Exp 9's current -0.05 error and prioritize 7 and 12, which carry ~5x its squared loss. (3) EXP 1 (+0.036 drift): re-check after the retune; it was exact at iter 2 and should be insensitive to distrust-branch changes (claimant-free panel), so any residual drift is kappa-side — verify the kappa gate's high center (k_half ~0.40) is untouched. (4) NO MORE INFERRED COMPOSITION TABLES: the iter-5 rationale's 'instrumentation' (inferring c from simulation swings) produced a fourth consecutive falsified phi table for Exp 7. Before tuning, enumerate the ACTUAL validities of each experiment's panel (they are fixed and known — e.g., Exp 5's and Exp 10's are printed in their metric docstrings) and compute c and corr_frac exactly under the v_lo / v_corr boxes; if Exp 7's true validities show zero experts below v_lo, then the fix is re-anchoring the chance classification (e.g., v_lo up to ~0.56-0.60), not the sigmoid center — but the simulation's FDEI-identical value is most consistent with c_7 = 0.125, i.e., one 0.5 cue, which the lower center reaches. All changes above are recalibrations of the composition statistics and gate placements the prescribed PIGD family already contains — no rung-gap term, no mechanism switch. Self-history note: my iter-2 push to raise the (then single, shared) sigmoid center was made before the iter-5 split gates existed; recommending a lower center NOW for the distrust branch is not an oscillation — the decoupling removed the constraint (protecting claimant-free panels) that motivated the higher center, and the kappa branch keeps its high center unchanged.

**Outcome of your advice:** iter 5 candidate loss=0.0716 -> iter 6 candidate loss=0.1026 -> the gate marked it REJECTED.

### Iteration 6 (most recent)
Verdict: regenerate
Interpretation: The gate rejected this candidate (0.1026 vs the accepted iter-5 base's 0.0716), and the per-experiment decomposition shows exactly where the four moved knobs went wrong. WINS: Experiment 8 is now EXACT (0.4625 vs 0.4625), Experiment 4 had its best fit ever (0.5233 vs 0.5133), Experiment 12 improved slightly (+0.095 -> +0.070), and the claimant-free branch held (Exps 1, 2, 5, 11 all fine). LOSSES: (a) Experiment 3 collapsed from EXACT (0.6170) to 0.4456 vs real 0.6170 — a -0.17 catastrophic regression whose squared error (~0.029) is essentially the entire loss increase (0.0716 -> 0.1026). Since the kappa-branch parameters were untouched, this proves Experiment 3's panel HAS an above-ceiling claimant and that the lowered c_half / weakened dilution opened its distrust gate: the d_panel spread destroyed the mid-scale weights and tally-following fell to near chance. The proposer's claim that 'the claimant-free branch (Exps 1, 2, 3, 5, 8) is bit-identical by construction' is therefore false for E3 — the SEVENTH consecutive falsified hand-computed composition claim. (b) Experiment 9 regressed (-0.048 -> -0.100): the weakened dilution (rho_corr 1.4-1.8, m_floor 0.10-0.16) over-fired distrust on its corroborated panel. (c) THE WALL STANDS: Experiment 7 moved from 0.3133 to 0.3302 vs real 0.5796 — still numerically indistinguishable from pure FDEI (pi_4 = 0.3191) for the SIXTH consecutive iteration, despite the rationale's claim that 'w(0.99) is negative on E7 for EVERY parameter draw.' That claim is falsified by the simulation itself. The decisive inference: centers from 0.36 down to 0.09 have all failed to move E7, including a center (0.09-0.15) that would open the gate at any c_7 >= 0.125 under any draw in the box. The only consistent explanation is that c_7 ≈ 0 under the v_lo ~ 0.52-0.54 classification — Experiment 7's weak block is NOT 0.50-level cues but barely-above-chance experts (~0.55-0.60), which the current chance classifier does not count. No placement of the sigmoid center on this statistic can ever reach that panel. My own iter-5 endorsement of the proposer's 'c_7 = 0.125' hypothesis was wrong, and the iter-5 advice (lower c_half to 0.10-0.16 + weaken the dilution) was rejected by the gate with E3's collapse as the direct cost.
Rationale: REGENERATE, on top of the unchanged iter-5 base (do not re-lower c_half to 0.09-0.15 — that path is now empirically falsified: it wrecks E3 without moving E7). Direction for the next candidate, all inside the prescribed PIGD family: (1) FIX THE INCOHERENCE INPUT, NOT THE SIGMOID CENTER — ADD A DISTRUST-BRANCH-SPECIFIC CHANCE THRESHOLD. Six iterations prove E7 registers c ≈ 0 under v_lo ~ 0.53; its weak block is almost certainly 0.55-0.60-class experts that the current classifier treats as trusted (and the flattening then gives them w ≈ 0.8 each, which is also why the panel reads as coherent to the model). Introduce a separate threshold v_cd ∈ [0.56, 0.62] used ONLY in the distrust branch's chance-fraction computation (the kappa branch keeps v_lo unchanged, so claimant-free panels — Exps 1, 2, 3-if-claimant-free-branch, 5 — remain bit-identical). Re-derive c per panel on the new scale and re-place c_half accordingly (roughly [0.22, 0.30]). Predicted effects: E7's c rises to ~0.3-0.4, opening its gate at a moderate center; E9's 0.55s also raise its c but its high corroboration keeps mult floored, so its discount stays mild; E10 (dense 0.98/0.95/0.90 ladder, no weak experts) is untouched; E4/E6 (0.50 cues, corr = 0) keep full distrust. (2) TARGET ONLY A PARTIAL DISCOUNT ON E7 — PARITY, NOT A SIGN FLIP. Real E7 is 0.58, i.e., MILD contrarianism; with coalitions of flattened weak cues routinely opposing feature 0, reducing w(0.99) to rough parity with w(0.75) (d_top * G_7 ≈ 0.8-1.2) suffices. Do not demand w(0.99) < 0 — that overkill is what has driven the collateral damage on every previous attempt. (3) RESTORE THE DILUTION TO INTERMEDIATE STRENGTH: rho_corr ∈ [1.7, 2.1] and m_floor ∈ [0.12, 0.18] — between iter-5's (2.0-2.4 / 0.18-0.24, which protected E3/E9/E10 but starved E12) and iter-6's rejected weakening. E12's needed extra discount should come from phi under the re-derived center, not from weakening the corroboration shield that protects E3, E9, and E10. Target d_top * G_12 ≈ 0.9-1.3 to pull deference from 0.547 toward 0.45. (4) MANDATORY MACHINE INSTRUMENTATION — THIS IS NOW A HARD REQUIREMENT, NOT A SUGGESTION. Seven consecutive hand-computed composition tables have been falsified by the simulations (this iteration: E3 claimed bit-identical, collapsed; E7 claimed 'gate opens for every draw,' did not move). Before proposing any parameter box, run the actual predict() code on each experiment's true validities at 3-5 points in the candidate box and include the machine-generated per-experiment table of c, phi, corr_frac, mult, G, and the final weight vector. The iter-6 record itself is diagnostic data: E3's collapse proves E3 has a claimant; E4's improvement proves E4 has a rung at or above v_corr; use these as consistency checks on the table. (5) IF the machine table shows E3 and E7 are inseparable on (c, corr) under the new threshold — i.e., both low-corr and similar-c — then STOP chasing E7 with gate placement: accept it as the family's one known residual (a 0.27 error on a mild-contrarianism index) and spend the iteration protecting E3/E9/E12 instead; a candidate that fixes E3 and E12 while holding everything else at iter-5 quality will beat 0.0716 without ever touching E7. (6) KEEP VALIDATED MACHINERY UNTOUCHED: v_c strictly in (0.98, 0.99), the split gates (low-centered distrust sigmoid / high-centered kappa sigmoid), kappa-on-claimant-absence, the d_panel spread, the flattened baseline, low-gain softmax + lapse. Self-history note: my iter-5 push (c_half to 0.10-0.16 + weaker dilution) was rejected — the recommendation above deliberately reverts the dilution to intermediate values rather than oscillating back to iter-4's 2.2-2.8, and replaces the failed center-lowering path with a different in-family knob (the classification threshold) rather than another step in the same direction.

**Outcome of your advice:** iter 6 candidate loss=0.1026 -> CURRENT candidate loss=0.0885 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The gate rejected this candidate (0.0885 vs the accepted iter-5 base's 0.0716), so the verdict is regenerate regardless. The three-knob edit decomposes cleanly into one clear win, one clear loss, and one null result. WIN: m_floor raised to [0.21, 0.25] (EDIT 3) drove Experiment 12's improvement (+0.095 error -> +0.057) and, together with v_cd, Experiment 1's best-ever fit (-0.1817 -> -0.2258 vs real -0.2180) \u2014 note that E1's movement falsifies the rationale's claim that E1 is claimant-free and bit-identical (the EIGHTH consecutive falsified hand-computed composition claim; E1 evidently has an above-ceiling claimant, which is diagnostic gold). LOSS: EDIT 2 (v_corr lowered to [0.76, 0.80]) was designed to separate E7 from E6 but delivered nothing on E7 while breaking Experiment 11 (0.0141 -> -0.0191 vs real +0.0206): E11's 0.80 rung now counts as corroboration, corr_frac rises to ~0.5, mult floors, G halves, and the model drifts FDEI-ward on the one metric where the real data are mildly CREDT-like. It also accounts for E3's small drift (0.6170 -> 0.6063). NULL RESULT \u2014 THE WALL: Experiment 7 sits at 0.3253 vs real 0.5796 for the SEVENTH consecutive iteration, still numerically indistinguishable from pure FDEI (pi_4 = 0.3191). The iter-6 primary directive (broaden the incoherence input via v_cd up to 0.615) moved it by +0.012. Combined with six prior failed knob placements (centers 0.09-0.415, narrow and soft classifiers), the conclusion is now forced: under every composition statistic this family has tried, E7's panel either registers c ~ 0 (weak block at or above ~0.62) or its mid-rungs corroborate and floor the discount. Raising v_cd past 0.62 is empirically barred \u2014 it would count E3's 0.60+ rungs and repeat iter-6's E3 collapse; E3 (needs no distrust) and E7 (needs distrust) are graded-ladder claimant panels that are inseparable on (c, corr) under any broadened classifier. My iter-6 point (5) anticipated exactly this contingency and it has now been confirmed: E7 must be accepted as the family's residual. A secondary cost: between-subject variances rose on E1 (0.0050 -> 0.0075), E5 (0.0172 -> 0.0191), E7 (0.0043 -> 0.0054), and E8 (0.0180 -> 0.0200) \u2014 the widened boxes increased inconsistency, which plausibly explains why the point-estimate squared errors actually IMPROVED (sum ~0.086 -> ~0.073) while the aggregate loss worsened.",
  "verdict": "regenerate",
  "rationale": "REGENERATE, on top of the unchanged iter-5 base, with a strictly-positive-only edit set \u2014 stop spending error budget on Experiment 7. (1) REVERT EDIT 2 ENTIRELY: v_corr back to [0.80, 0.84]. It was justified purely by an E7 separation that did not materialize, and it is the identified cause of E11's regression (the 0.80 rung must NOT corroborate on E11's panel) and of E3's drift. This is not oscillation \u2014 v_corr [0.80, 0.84] is the iter-5 validated value that the current candidate moved away from without payoff. (2) KEEP EDIT 3 (m_floor [0.21, 0.25], optionally [0.22, 0.28]): it is the main driver of E12's improvement and is protected on E9/E10 by their high corroboration. Target d_top * G_12 ~ 0.9-1.3 to pull E12 deference from ~0.51 toward 0.45-0.48. (3) KEEP v_cd BUT CAP IT AT [0.555, 0.595]: the broadened distrust-band threshold demonstrably helped E1 (error 0.036 -> 0.008 \u2014 the simulation itself proves E1 has a claimant with a weak block in the 0.55-0.60 range) and contributed to E12, but draws with v_cd > 0.60 count E3's 0.60-class rungs (E3's small drift) and risk E10's hidden 0.65 rungs. Capping below 0.60 retains the E1/E12 gains while making E3/E10 bit-safe. (4) DO NOT TOUCH E7 AGAIN \u2014 no further v_cd raises, no center moves, no dilution weakening. Three independent mechanisms have now failed on it while each attempt cost collateral elsewhere (iter-6: E3 collapse; iter-7: E11 regression). E7's ~0.065 squared error is the family's accepted residual; the winnable margins are E12 (~0.003), E9 (~0.002), E11 (~0.0016), and E1 (~0.001), totaling enough to beat 0.0716 without E7. (5) TIGHTEN, DON'T WIDEN: the variance increases on E1/E5/E7/E8 suggest the aggregate loss penalizes inconsistency; keep every other box at iter-5 values (kappa [0.34, 0.42] with k_half [0.38, 0.42], v_c strictly in (0.98, 0.99), d_top/d_panel, beta, epsilon) so the fitter's search concentrates. Expected outcome: E12 ~0.47-0.49, E1 ~-0.22, E11 back to ~+0.015, E3 back to ~0.617, everything else held \u2014 an aggregate loss of roughly 0.065-0.068, beating the gate. All changes are in-family recalibrations of the prescribed PIGD machinery (flattened baseline, composition-gated distrust with corroboration dilution, split kappa gate, low-gain softmax + lapse); no rung-gap term, no mechanism switch."
}
```

## Usage

```json
{
  "input_tokens": 42671,
  "output_tokens": 12272,
  "reasoning_tokens": 10817,
  "provider": "Sail Research"
}
```
