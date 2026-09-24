# feedback_iter_08

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
- THEORY 1 = `pi_3_1`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 2 (= `pi_5`).

Replace HEAI (pi_5) with a context-referenced, trait-free anti-validity theory. Sketch — Context-Normalized Noisy Anti-Validity Integration (CNNAI): (1) Keep the empirically well-supported core shared by both theories: all cues integrated additively into a per-option value, a rating of 1 is a defect (sign inversion), softmax plus symmetric lapse, a modest guesser subpopulation. (2) Replace the stable per-subject weight map (whether SCWI's fixed tau or HEAI's lifetime gamma) with a CONTEXT-REFERENCED weight: w_j = -v_j * (v_j / v_ref)^eta, where v_ref is the mean/max validity of the cues actually displayed in the current trial set (or block) and eta is an elasticity that is RE-DRAWN per context/block from a broad unimodal distribution centered near 0.3–0.6 (roughly lognormal, wide). Because eta is redrawn when the validity context changes (e.g., between the 0.50-ladder and 0.60-ladder of Exp 1), the theory predicts cross-ladder consistency near r ≈ 0.15–0.25 — matching the observed 0.196 — while still producing strong within-context anti-validity behavior (negative Exp-3 slope, low Exp-6 tally-following). Range normalization also naturally produces intermediate psychometric rises (Exp 2's D ≈ 0.25–0.35) because the effective weight ratio compresses toward the context's validity spread rather than following a fixed power law. (3) Add per-trial multiplicative attention noise on cue weights (not fixed per-cue attention), which inflates between-subject variance on diagnostic composites toward the observed 0.06–0.09 without shifting means — fixing the systematic variance underprediction of both current theories. (4) Retain a heterogeneous beta/epsilon mixture but let it be uncorrelated with eta, so no latent trait couples behavior across contexts. This theory is a stronger competitor to SCWI than HEAI because it keeps SCWI's empirically successful anti-validity integration while explaining the two things SCWI gets wrong: near-zero cross-context consistency and elevated between-subject variance. It differs from both predecessors precisely where they were falsified: neither a fixed quadratic map nor a fixed power-law trait, but a context-normalized, redrawn-per-context gradient with per-trial attention diffusion.


## CANDIDATE THEORY
Context-Normalized Noisy Anti-Validity Integration (CNNAI), flat-weight-mixture variant. Subjects integrate all binary expert ratings additively into a per-option value, but a rating of 1 is comprehended as a DEFECT (sign inversion). The anti-weight of expert j is context-normalized: w_j = -v_j * (v_j / v_ref)^eta, where v_ref is the mean instructed validity of the cues endorsed on the current trial and eta is a per-context elasticity drawn at the level of coarse validity CONTEXT BUCKETS (mean endorsed validity rounded to 0.1). The elasticity decomposes into a small stable per-subject component (eta_base ~ N(0.02, 0.09)) and a larger per-bucket redraw (~N(0.10, 0.19)), total hard-clipped to [-0.25, 0.70], fixing latent cross-context consistency near r ~ 0.18. NEW in this iteration: a small MILD FLAT-WEIGHT subpopulation (~7% of engaged subjects, ~6% of the population) draws a STABLE, context-invariant eta uniform in [-0.45, -0.30] — SCWI-like integration with low curvature (w_h/w_l = (v_h/v_l)^(1+eta) ~ 1.1-1.2, validity ordering preserved), not tallying. Per-trial multiplicative attention diffusion (sd 0.06), softmax(beta * values) with nearly homogeneous engaged beta (lognormal median ~2.14, log-sd 0.20), narrow symmetric lapse (0.03-0.09), and a 13% true-guesser subpopulation uncorrelated with eta are retained.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Context-Normalized Noisy Anti-Validity Integration (CNNAI),
    # iter 9 -- disentangled flat-weight-subpopulation edit on the
    # accepted iter-6 base, implementing the iter-8 critic's
    # prescription after two confounded multi-knob rejections.
    # Mechanism family UNCHANGED: additive anti-validity integration,
    # context-referenced elasticity w_j = -v_j * (v_j/v_ref)^eta with
    # v_ref = mean validity of the endorsed cues, bucketed-mean context
    # key, per-trial attention diffusion, softmax + symmetric lapse,
    # guesser mixture. Exactly THREE coordinated changes vs the
    # accepted iter-6 base, each targeting a distinct diagnostic:
    #   (A) FLAT-WEIGHT SUBPOPULATION ADDED (~7% of engaged subjects,
    #       i.e. regime_seed u in [0.13, 0.191)): these subjects draw a
    #       STABLE, context-invariant eta uniform in [-0.45, -0.30]
    #       (no per-bucket redraw). At eta ~ -0.37 the effective weight
    #       exponent is ~0.63: validity ordering is preserved
    #       (w_h/w_l ~ 1.1-1.2), so these are low-curvature SCWI-like
    #       integrators, NOT tally-followers -- they keep conflict
    #       tally-following low (Exp 4) while lifting the slightly
    #       undershooting composites (Exps 4/6). The range is
    #       DELIBERATELY shallower than iter-8's rejected [-0.55,-0.40]
    #       to reduce the damage to Exp 2 and the Exp-7 overshoot.
    #       Pipeline fact (established by the critic's own control-
    #       screen analysis): every eta in [-0.45,-0.30] lies below the
    #       L-control knife at eta = -0.263, so flat subjects choose
    #       the top stack on the control cells, fail the cH <= 0.375
    #       keep filter, and are EXCLUDED from Exp 9's correlation --
    #       they are provably inert on the headline statistic.
    #   (B) DIFFUSION 0.05 -> 0.06 (NOT 0.08): diffusion 0.08 was the
    #       common factor of both broken iterations (7, 8) and the prime
    #       suspect for the Exp-9 collapse; 0.06 sits next to the
    #       healthy iter-6 value of 0.05 and buys only mild softening
    #       of the many-cue cells (Exps 8/10). A residual overshoot on
    #       Exp 8 is accepted rather than risking Exp 9.
    #   (C) eta_base sd STAYS AT 0.09 (the iter-8 trim to 0.07 is NOT
    #       carried over): the trim was justified by a subpopulation
    #       that is inert on Exp 9, so it only removed the stable
    #       coupling that produced iter-6's excellent r = 0.184.
    # Net effect on Exp 9: for every RETAINED subject the pipeline is
    # byte-identical to the accepted iter-6 base except diffusion
    # 0.05 -> 0.06 (flat subjects are screened out by the control
    # filter), so r should recover to ~0.17-0.19.
    import hashlib
    import math
    import numpy as np
    from scipy.special import ndtri

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"CNNAI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]
    ra, rb = stim[0], stim[1]

    def _to_vec(raw):
        if raw is None:
            return None
        if isinstance(raw, str):
            s = raw.strip().strip('[]()')
            toks = [t for t in s.replace(',', ' ').split() if t != '']
            if not toks:
                return None
            return np.array([float(t) for t in toks])
        arr = np.asarray(raw, dtype=float).ravel()
        return arr if arr.size > 0 else None

    val = _to_vec(parameters.get("validities"))
    if val is None or val.size != n_features:
        # Defensive fallback: uniform mid-range validities.
        val = np.full(n_features, 0.75)
    val = np.clip(val, 1e-3, 1.0)

    def _seed(name):
        try:
            x = float(parameters[name])
        except (KeyError, TypeError, ValueError):
            x = 0.5
        if not np.isfinite(x):
            x = 0.5
        return min(max(x, 0.0), 1.0)

    u = _seed("regime_seed")
    es = _seed("eta_base_seed")
    bs = min(max(_seed("beta_seed"), 1e-6), 1.0 - 1e-6)
    ps = _seed("epsilon_seed")
    ss = _seed("subject_seed")

    # --- Population mixture (uncorrelated with eta). ---
    is_flat = False
    if u < 0.13:
        # Guesser subpopulation (~13%): beta ~ 0 -> uniform core on
        # every trial; elevated lapse. Unchanged from the accepted base.
        t = u / 0.13
        beta = 0.02 + 0.08 * t            # 0.02 .. 0.10
        epsilon = 0.12 + 0.13 * ps       # 0.12 .. 0.25
        eta_base = 0.10                  # irrelevant at beta ~ 0
    else:
        # Engaged subpopulation (~87%).
        if u < 0.191:
            # (A) Mild flat-weight subpopulation: ~7% of engaged
            # subjects draw a STABLE low-curvature eta, uniform in
            # [-0.45, -0.30], held fixed across all contexts (no
            # per-bucket redraw). Validity ordering is preserved
            # (exponent 1+eta in [0.55, 0.70]), so these subjects
            # remain anti-validity integrators with flat weights --
            # they are screened out of Exp 9 by the control filter
            # (every eta here is below the L-control knife -0.263)
            # and therefore cannot touch the cross-ladder r.
            is_flat = True
            eta_base = -0.45 + 0.15 * es  # es ~ U[0,1] -> [-0.45, -0.30]
        else:
            # Regular engaged majority: small stable elasticity
            # component (sd 0.09 -- NOT the iter-8 trim to 0.07, which
            # removed real stable coupling for nothing) paired with a
            # larger per-context redraw (sd 0.19); the variance ratio
            # fixes latent cross-context consistency at r ~ 0.18.
            zs = min(max(es, 1e-6), 1.0 - 1e-6)
            eta_base = 0.02 + 0.09 * float(ndtri(zs))
            eta_base = min(max(eta_base, -0.15), 0.20)

        # Beta / epsilon unchanged from the accepted iter-6 base:
        # nearly homogeneous engaged beta (lognormal median ~2.14,
        # log-sd 0.20) and a narrow stable lapse (0.03-0.09) so no
        # latent trait couples behavior across contexts.
        beta = math.exp(0.76 + 0.20 * float(ndtri(bs)))  # median ~2.14
        beta = min(max(beta, 0.9), 4.5)
        epsilon = 0.03 + 0.06 * ps       # 0.03 .. 0.09

    # --- Context: bucketed mean endorsed validity (unchanged). ---
    active = np.flatnonzero((ra > 0.5) | (rb > 0.5))
    if active.size == 0:
        # Nothing endorsed by either option: no evidence either way.
        return np.full(2, 0.5)

    v_ref = float(np.mean(val[active]))
    bucket = float(np.round(v_ref, 1))
    key = f"{bucket:.1f}"
    h = int(hashlib.md5(f"ss:.9f|key".format(ss=ss, key=key).encode("utf-8")).hexdigest()[:14], 16)
    uc = ((h % 999983) + 0.5) / 999984.0
    uc = min(max(uc, 1e-6), 1.0 - 1e-6)

    if is_flat:
        # (A) Flat subjects keep their stable eta across contexts.
        eta_ctx = eta_base
    else:
        # Per-context redraw for the regular engaged majority,
        # centered low (engaged eta ~ N(0.12, 0.21), P(eta < 0.33)
        # ~ 0.83 -- the validated iter-6 knife-avoidance placement).
        eta_ctx = eta_base + 0.10 + 0.19 * float(ndtri(uc))
        eta_ctx = min(max(eta_ctx, -0.25), 0.70)

    # Context-normalized anti-validity weights.
    w = -val * np.power(val / v_ref, eta_ctx)

    # (B) Per-trial multiplicative attention diffusion, sd 0.05 ->
    # 0.06: a small step toward softening the many-cue cells (Exps
    # 8/10) while staying next to the iter-6 value that kept Exp 9
    # healthy (0.08 was present in both broken iterations and is the
    # prime suspect for the cross-ladder collapse).
    att = np.exp(0.06 * np.random.randn(n_features))
    w = w * att

    scores = np.array([float(np.dot(w, ra)), float(np.dot(w, rb))])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * 0.5 * np.ones(2)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- subject_seed: [0, 1]
- regime_seed: [0, 1]
- eta_base_seed: [0, 1]
- beta_seed: [0, 1]
- epsilon_seed: [0, 1]
- validities: validities

`rationale`:
This is a minimal-diff, deliberately DISENTANGLED edit of the accepted iter-6 base (loss 0.0645), implementing the iter-8 critic's prescription after two confounded multi-knob rejections. The mechanism family, context key, v_ref definition, softmax/lapse structure, and parameter set are unchanged; exactly three lines of behavior change. (A) FLAT-WEIGHT SUBPOPULATION, the one mechanism that has ever moved Exps 4/6 onto target (iter 8: 0.170/0.630 vs real 0.163/0.630), is retained but re-tuned per the prescription: ~7% of engaged subjects (regime_seed in [0.13, 0.191)) draw a STABLE eta uniform in [-0.45, -0.30] -- shallower than iter-8's rejected [-0.55, -0.40] to reduce the Exp-2 damage and the Exp-7 overshoot. Crucially, this subpopulation is provably inert on Exp 9: every eta in [-0.45, -0.30] lies below the L-control knife at eta = -0.263 (choose-top-stack condition 3 < 5*0.5^(1+eta) requires eta < -0.262), so flat subjects fail the cH <= 0.375 keep filter and are excluded from the cross-ladder correlation -- the pipeline fact the critic established when withdrawing the eta_base-trim justification. (B) DIFFUSION 0.05 -> 0.06, not 0.08: diffusion 0.08 was the common factor of both broken iterations (7, 8) and the prime remaining suspect for the Exp-9 collapse; 0.06 sits directly beside the healthy iter-6 value. (C) eta_base sd RESTORED to 0.09 (the base already has it; the iter-8 trim to 0.07 is simply not carried over). The resulting Exp-9 guarantee is the strongest available in-family argument: for every subject RETAINED by Exp 9's screens, the pipeline is identical to the accepted iter-6 base except diffusion 0.05 -> 0.06, so r should recover to ~0.17-0.19 (iter 6 delivered 0.184 vs real 0.196); the added per-trial noise is ~20% larger in sd on the attention component only, a small attenuation on top of binomial and lapse noise. Projected profile vs the iter-6 base, grounded in the loop's own empirical deltas rather than knife-count arithmetic (diffusion 0.05->0.08 alone moved Exp 2 by -0.022 and Exp 8 by ~-0.03 in iter 7; flat 8%-deep mass moved Exp 4 +0.032, Exp 6 +0.010, Exp 2 -0.028, Exp 7 +0.018 in iter 8): Exp 1 ~ -0.193 (from -0.201, real -0.175), Exp 2 ~ 0.17 (still under 0.274 -- the chronic residual, but recovering from iter-8's 0.139), Exp 3 ~ 0.134, Exp 4 ~ 0.160 (from 0.138, real 0.163), Exp 5 ~ 0.613 (residual accepted; no in-family lever identified), Exp 6 ~ 0.626, Exp 7 ~ 0.672 (slight guardrail tension accepted), Exp 8 ~ 0.68 (residual +0.07 accepted per the critic's explicit instruction to sacrifice Exp 8 rather than Exp 9), Exp 9 ~ 0.18, Exp 10 ~ 0.325 (from 0.366, real 0.296 -- flat mass pushes p_twin up and diffusion softens the many-cue cells). Every projected move is in the direction the corresponding iteration's data already demonstrated for that knob at comparable magnitude; the one genuinely uncertain cell is Exp 9 at diffusion 0.06, which is why the diffusion step was kept minimal and next to the proven value. Between-subject variance remains underpredicted (2-5x on the composites); per the iter-8 diagnosis this channel is exhausted in-family -- a discrete subpopulation is arithmetically incapable of sd ~0.26 composites and symmetric dispersion is capped by Exp 9 -- so no variance move is attempted this iteration, and the beta-heterogeneity increment is deferred until the mean profile first beats 0.0645, exactly as prescribed.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.1070 -> ACCEPTED
- iter 2: loss=0.1032 -> ACCEPTED
- iter 3: loss=0.0968 -> ACCEPTED
- iter 4: loss=0.1234 -> REJECTED
- iter 5: loss=0.0913 -> ACCEPTED
- iter 6: loss=0.0645 -> ACCEPTED
- iter 7: loss=0.0786 -> REJECTED
- iter 8: loss=0.0819 -> REJECTED
- iter 9 (current candidate you are grading): loss=0.1539 -> REJECTED
Running-best (last accepted) base: iter 6 at loss=0.0645.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Slope of P(choose the top-cue-favored option) regressed on the
    # tally margin of that favored option, across all trials where the
    # highest-validity discriminating cue exists (ties like trials 17/18
    # are skipped). Under TTB this probability is flat in the tally margin
    # (slope ~ 0); under Tallying it rises monotonically with the tally
    # margin (positive slope).
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.65])
    cue_order = np.argsort(-validities, kind="stable")

    xs = []
    ys = []
    ra_col = data["option_a_ratings"]
    rb_col = data["option_b_ratings"]
    resp_col = data["response"]

    for ra, rb, resp in zip(ra_col, rb_col, resp_col):
        a = np.asarray(ra, dtype=float)
        b = np.asarray(rb, dtype=float)

        # Find the first discriminating cue in descending-validity order.
        favored = None
        for j in cue_order:
            if a[j] > b[j]:
                favored = 0
                break
            if b[j] > a[j]:
                favored = 1
                break
        if favored is None:
            # Full tie: no top cue, no tally signal. Skip.
            continue

        # Tally margin from A's perspective, then re-sign so it is from
        # the top-cue-favored option's perspective.
        d = float(np.sum(a > b) - np.sum(b > a))
        if favored == 1:
            d = -d

        xs.append(d)
        ys.append(1.0 if int(resp) == favored else 0.0)

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 3:
        return 0.0

    xm = xs.mean()
    ym = ys.mean()
    sxx = float(np.sum((xs - xm) ** 2))
    if sxx <= 0.0:
        return 0.0

    slope = float(np.sum((xs - xm) * (ys - ym)) / sxx)
    return slope
```

**Observed (real) value:** -0.1747 (var=0.0108)
**Candidate trajectory (this loop):**
  - iter 1: -0.1790 (var=0.0088) (Δ vs real -0.0043)
  - iter 2: -0.1813 (var=0.0063) (Δ vs real -0.0067)
  - iter 3: -0.1850 (var=0.0087) (Δ vs real -0.0103)
  - iter 4: -0.2025 (var=0.0069) (Δ vs real -0.0278)
  - iter 5: -0.1825 (var=0.0081) (Δ vs real -0.0078)
  - iter 6: -0.2010 (var=0.0028) (Δ vs real -0.0263)
  - iter 7: -0.1973 (var=0.0040) (Δ vs real -0.0227)
  - iter 8: -0.1900 (var=0.0071) (Δ vs real -0.0153)
  - iter 9 (current): -0.1845 (var=0.0073) (Δ vs real -0.0098)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0023 (var=0.0008)
- pi_2: 0.1952 (var=0.0019)
- pi_3: -0.1877 (var=0.0014)
- pi_4: -0.1883 (var=0.0047)
- pi_3_1: -0.2228 (var=0.0055)
- pi_5: -0.1898 (var=0.0057)

### Experiment 2
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # Cue cascade order for validities [0.9, 0.8, 0.7, 0.6, 0.5]:
    # strictly descending, so TTB consults features 0,1,2,3,4 in order.
    cue_order = [0, 1, 2, 3, 4]

    def to_vec(x):
        if isinstance(x, str):
            s = x.strip().strip('[]()')
            return np.array([float(v) for v in s.replace(',', ' ').split()])
        return np.asarray(x, dtype=float).ravel()

    follow_disagree = 0
    n_disagree = 0
    follow_agree = 0
    n_agree = 0

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        a = to_vec(a_raw)
        b = to_vec(b_raw)
        if a.size != 5 or b.size != 5:
            continue

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))

        # Tallying must be decisive and the tally margin must be exactly 1:
        # on every such trial the softmax probability of following the tally
        # is sigmoid(beta * 1), identical across trials, so Tallying predicts
        # a perfectly FLAT follow-rate regardless of subset.
        if a_wins == b_wins:
            continue
        if abs(a_wins - b_wins) != 1:
            continue

        tally_picks_a = a_wins > b_wins

        ttb_picks_a = None
        for j in cue_order:
            if a[j] > b[j]:
                ttb_picks_a = True
                break
            if b[j] > a[j]:
                ttb_picks_a = False
                break
        if ttb_picks_a is None:
            continue

        chose_a = (float(resp) == 0.0)

        if ttb_picks_a == tally_picks_a:
            n_agree += 1
            if chose_a == tally_picks_a:
                follow_agree += 1
        else:
            n_disagree += 1
            if chose_a == tally_picks_a:
                follow_disagree += 1

    if n_disagree == 0 or n_agree == 0:
        return 0.0

    p_dis = follow_disagree / float(n_disagree)
    p_agr = follow_agree / float(n_agree)
    return p_dis - p_agr

```

**Observed (real) value:** 0.2737 (var=0.0930)
**Candidate trajectory (this loop):**
  - iter 1: 0.1862 (var=0.0254) (Δ vs real -0.0875)
  - iter 2: 0.1394 (var=0.0250) (Δ vs real -0.1344)
  - iter 3: 0.2075 (var=0.0172) (Δ vs real -0.0663)
  - iter 4: 0.2125 (var=0.0159) (Δ vs real -0.0612)
  - iter 5: 0.1900 (var=0.0086) (Δ vs real -0.0837)
  - iter 6: 0.1894 (var=0.0175) (Δ vs real -0.0844)
  - iter 7: 0.1669 (var=0.0141) (Δ vs real -0.1069)
  - iter 8: 0.1387 (var=0.0170) (Δ vs real -0.1350)
  - iter 9 (current): 0.1550 (var=0.0143) (Δ vs real -0.1188)
**Other theories' values on this metric (for reference):**
- pi_2: -0.0006 (var=0.0060)
- pi_1: -0.6981 (var=0.0507)
- pi_3: 0.1862 (var=0.0095)
- pi_4: 0.2669 (var=0.0115)
- pi_3_1: 0.1713 (var=0.0139)
- pi_5: 0.1550 (var=0.0256)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Slope of P(choose A) regressed on the anti-validity weighted cue
    margin D = sum_j u_j * (b_j - a_j), with u_j = v_j + tau * v_j^2
    (tau fixed at 0.35, the midpoint of SCWI's tau range).

    D > 0 means option A carries fewer/smaller anti-validity 'defects'
    and is therefore preferred by SCWI. SCWI predicts a POSITIVE slope
    (~0.2-0.3 with beta in [1.1,1.7], epsilon in [0.05,0.15]).
    Tallying cannot produce a positive slope: on count-tied trials
    (pairs 1-8, 24) it outputs exactly 0.5 for any beta/epsilon, and on
    all remaining trials its choice probability tracks the raw count
    margin, which is strongly ANTI-correlated with D across this design
    (the tally winner is almost always the weighted-margin loser). So
    Tallying's slope is <= 0 (approximately 0 for small beta, clearly
    negative for large beta). The sign separation is parameter-free.
    """
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.5])
    tau = 0.35
    u = validities + tau * np.square(validities)

    def _to_arr(cell):
        arr = np.asarray(cell, dtype=float).ravel()
        if arr.size != u.size:
            raise ValueError(
                f"Expected {u.size} ratings, got {arr.size}"
            )
        return arr

    a = np.vstack([_to_arr(c) for c in data["option_a_ratings"]])
    b = np.vstack([_to_arr(c) for c in data["option_b_ratings"]])

    # Signed SCWI evidence in favor of A (positive -> A preferred).
    D = (b - a) @ u

    # response: 0 = chose A, 1 = chose B  ->  indicator of choosing A.
    choice_a = 1.0 - np.asarray(data["response"], dtype=float).ravel()

    Dc = D - D.mean()
    Yc = choice_a - choice_a.mean()
    denom = float(np.sum(Dc * Dc))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(Dc * Yc) / denom)
```

**Observed (real) value:** 0.1227 (var=0.0008)
**Candidate trajectory (this loop):**
  - iter 1: 0.1391 (var=0.0016) (Δ vs real +0.0164)
  - iter 2: 0.1445 (var=0.0005) (Δ vs real +0.0218)
  - iter 3: 0.1375 (var=0.0022) (Δ vs real +0.0148)
  - iter 4: 0.1374 (var=0.0023) (Δ vs real +0.0147)
  - iter 5: 0.1255 (var=0.0037) (Δ vs real +0.0028)
  - iter 6: 0.1330 (var=0.0025) (Δ vs real +0.0103)
  - iter 7: 0.1379 (var=0.0022) (Δ vs real +0.0152)
  - iter 8: 0.1376 (var=0.0028) (Δ vs real +0.0149)
  - iter 9 (current): 0.1415 (var=0.0018) (Δ vs real +0.0188)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1378 (var=0.0002)
- pi_2: -0.1263 (var=0.0006)
- pi_1: -0.0888 (var=0.0010)
- pi_4: 0.1448 (var=0.0005)
- pi_3_1: 0.1444 (var=0.0038)
- pi_5: 0.1165 (var=0.0040)

### Experiment 4
**Design**
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Conflict-set tally-following rate.
    #
    # Selection rule: keep only trials where (i) the tally is non-zero (one
    # option strictly wins more feature-wise comparisons) and (ii) a
    # validity-weighted signed integrator with anti-validity weights
    # w_j = v_j + tau * v_j^2 (tau = 0.35, the midpoint of SCWI's range)
    # STRONGLY prefers the tally LOSER (|evidence| > 0.8).
    # In this design that selects exactly Probe B (pairs 7-12, |D| in
    # [1.0, 1.3]) plus the anchors (pairs 13-14, |D| ~ 5.2), i.e. 8 of the
    # 16 unique pairs = 48 of 96 trials per subject.
    #
    # Metric value = proportion of choices that follow the TALLY WINNER on
    # those conflict trials.
    #   Tallying: P >= sigmoid(beta*|m|) with |m| in {1,3,6}  -> > 0.5 for
    #             ANY beta > 0, epsilon < 1; ~0.95+ for typical beta.
    #   SCWI:     P = (1-eps)*(1-sigmoid(beta*|D|)) + eps/2 ~ 0.16-0.20 on
    #             Probe B and ~ eps/2 ~ 0.075 on anchors -> far below 0.5.
    # The two theories are separated by ~0.4-0.8 in mean with tight
    # per-subject binomial variance (48 trials/subject).

    import numpy as np

    VALIDITIES = np.array([1.0, 0.95, 0.65, 0.5, 0.5, 0.5])
    TAU = 0.35      # midpoint of SCWI's tau box [0.25, 0.45]
    THRESH = 0.8    # cleanly above Probe A's |D| <= 0.08, below Probe B's |D| >= 1.0

    w = VALIDITIES + TAU * np.square(VALIDITIES)  # positive anti-validity magnitudes

    a_col = data["option_a_ratings"].tolist()
    b_col = data["option_b_ratings"].tolist()
    r_col = data["response"].tolist()

    n_sel = 0
    n_match = 0
    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        if a.size == 0 or a.size != b.size or a.size != w.size:
            continue

        a_wins = int(np.sum(a > b))
        b_wins = int(np.sum(b > a))
        if a_wins == b_wins:
            # Full-tie trials (pairs 15/16): both theories say 0.5, uninformative.
            continue

        # Signed evidence for option A under a weighted anti-validity
        # integrator: value_A - value_B = sum_j w_j * (b_j - a_j).
        ev = float(np.dot(w, b - a))

        tally_a = a_wins > b_wins
        if tally_a:
            # A is the tally winner; keep only if the integrator strongly
            # prefers B (the tally loser).
            if ev > -THRESH:
                continue
        else:
            # B is the tally winner; keep only if the integrator strongly
            # prefers A (the tally loser).
            if ev < THRESH:
                continue

        r = int(round(float(resp)))  # 0 = chose A, 1 = chose B
        chose_tally_winner = (r == 0) if tally_a else (r == 1)
        n_sel += 1
        if chose_tally_winner:
            n_match += 1

    if n_sel == 0:
        return 0.5  # degenerate fallback (no conflict trials found)
    return float(n_match) / float(n_sel)
```

**Observed (real) value:** 0.1625 (var=0.0119)
**Candidate trajectory (this loop):**
  - iter 1: 0.1904 (var=0.0230) (Δ vs real +0.0279)
  - iter 2: 0.1725 (var=0.0185) (Δ vs real +0.0100)
  - iter 3: 0.1663 (var=0.0157) (Δ vs real +0.0038)
  - iter 4: 0.1583 (var=0.0173) (Δ vs real -0.0042)
  - iter 5: 0.1421 (var=0.0149) (Δ vs real -0.0204)
  - iter 6: 0.1375 (var=0.0132) (Δ vs real -0.0250)
  - iter 7: 0.1442 (var=0.0151) (Δ vs real -0.0183)
  - iter 8: 0.1696 (var=0.0236) (Δ vs real +0.0071)
  - iter 9 (current): 0.1375 (var=0.0132) (Δ vs real -0.0250)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8538 (var=0.0090)
- pi_3: 0.1608 (var=0.0029)
- pi_1: 0.6796 (var=0.0042)
- pi_4: 0.1696 (var=0.0044)
- pi_3_1: 0.1388 (var=0.0265)
- pi_5: 0.1979 (var=0.0276)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Instructed validities of this experiment (fixed at design time).
    VAL = np.array([1.0, 1.0, 0.85, 0.85, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    n_f = VAL.size

    # Extreme points of each theory's ADMISSIBLE parameter box.
    # SCWI: w(v) = -(v + tau*v^2), tau in [0.25, 0.45].
    W_SC = [-(VAL + t * VAL ** 2) for t in (0.25, 0.45)]
    # PAHI: w(v) = -v^gamma, gamma in [1.4, 2.4].
    W_PA = [-(VAL ** g) for g in (1.4, 2.4)]

    # Minimum |evidence margin| required to call a preference "guaranteed"
    # across the whole parameter box of a theory.
    M_SC = 0.08
    M_PA = 0.05

    def parse_vec(cell):
        if isinstance(cell, str):
            s = cell.strip().strip('[]()')
            toks = [t for t in s.replace(',', ' ').split() if t != '']
            return np.array([float(t) for t in toks], dtype=float)
        return np.asarray(cell, dtype=float).ravel()

    def parse_resp(r):
        if isinstance(r, str):
            rs = r.strip().upper()
            if rs == 'A':
                return 0
            if rs == 'B':
                return 1
            return int(float(rs))
        return int(round(float(r)))

    hits = []
    for a_cell, b_cell, r_cell in zip(data['option_a_ratings'],
                                       data['option_b_ratings'],
                                       data['response']):
        try:
            a = parse_vec(a_cell)
            b = parse_vec(b_cell)
            if a.size != n_f or b.size != n_f:
                continue
            d_sc = [float(np.dot(w, a) - np.dot(w, b)) for w in W_SC]
            d_pa = [float(np.dot(w, a) - np.dot(w, b)) for w in W_PA]
            resp = parse_resp(r_cell)
        except Exception:
            continue

        sc_lo, sc_hi = min(d_sc), max(d_sc)
        if sc_lo > M_SC:
            sc_pref = 1      # SCWI prefers A for EVERY admissible tau
        elif sc_hi < -M_SC:
            sc_pref = -1     # SCWI prefers B for EVERY admissible tau
        else:
            continue         # preference not guaranteed -> skip

        pa_lo, pa_hi = min(d_pa), max(d_pa)
        if pa_lo > M_PA:
            pa_pref = 1      # PAHI prefers A for EVERY admissible gamma
        elif pa_hi < -M_PA:
            pa_pref = -1     # PAHI prefers B for EVERY admissible gamma
        else:
            continue

        if sc_pref == pa_pref:
            continue         # theories agree -> no diagnostic content

        # Trial is a guaranteed sign-opposition trial: SCWI's preferred
        # option is exactly the one PAHI rejects (and vice versa).
        scwi_option_is_a = (sc_pref == 1)
        chose_scwi = (resp == 0) if scwi_option_is_a else (resp == 1)
        hits.append(1.0 if chose_scwi else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.6837 (var=0.0643)
**Candidate trajectory (this loop):**
  - iter 1: 0.6000 (var=0.0314) (Δ vs real -0.0837)
  - iter 2: 0.5525 (var=0.0185) (Δ vs real -0.1312)
  - iter 3: 0.5694 (var=0.0265) (Δ vs real -0.1144)
  - iter 4: 0.5006 (var=0.0185) (Δ vs real -0.1831)
  - iter 5: 0.5031 (var=0.0163) (Δ vs real -0.1806)
  - iter 6: 0.6194 (var=0.0196) (Δ vs real -0.0644)
  - iter 7: 0.5913 (var=0.0394) (Δ vs real -0.0925)
  - iter 8: 0.6094 (var=0.0299) (Δ vs real -0.0744)
  - iter 9 (current): 0.6625 (var=0.0150) (Δ vs real -0.0212)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5531 (var=0.0069)
- pi_4: 0.2281 (var=0.0125)
- pi_1: 0.8350 (var=0.0129)
- pi_2: 0.1300 (var=0.0103)
- pi_3_1: 0.6506 (var=0.0203)
- pi_5: 0.6019 (var=0.0710)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # "Top-cue adherence" on steepness-diagnostic ladder cells.
    # The design's validity vector: 4 cues at 1.00, 5 at 0.65, 5 at 0.60,
    # 14 at 0.50 (hardcoded from the experimental design).
    VALID = np.array([1.0] * 4 + [0.65] * 5 + [0.60] * 5 + [0.50] * 14)
    NF = VALID.size

    def to_vec(x):
        if isinstance(x, str):
            s = x.strip().strip('[]()')
            return np.array([float(t) for t in s.replace(',', ' ').split()])
        return np.asarray(x, dtype=float).ravel()

    # Excluded cells (signature = (top-cue count, sorted low-cue validities)):
    #  (2, 4x0.50): both theories prefer the low option -> no separation;
    #  (2, 5x0.60) and (2, 5x0.65): PAHI splits across its gamma range on
    #  these cells -> they add between-subject noise without adding gap.
    EXCLUDE = {
        (2, (0.5, 0.5, 0.5, 0.5)),
        (2, (0.6, 0.6, 0.6, 0.6, 0.6)),
        (2, (0.65, 0.65, 0.65, 0.65, 0.65)),
    }

    hits = []
    for a_raw, b_raw, resp in zip(
        data['option_a_ratings'], data['option_b_ratings'], data['response']
    ):
        a = to_vec(a_raw)
        b = to_vec(b_raw)
        if a.size != NF or b.size != NF:
            continue
        ta = int(np.sum(a[:4] > 0.5))
        tb = int(np.sum(b[:4] > 0.5))
        # Exactly one option must carry 1s on the perfect-validity cues.
        if ta > 0 and tb > 0:
            continue
        if ta == 0 and tb == 0:
            continue
        if ta > 0:
            top_vec, low_vec, top_idx, tc = a, b, 0, ta
        else:
            top_vec, low_vec, top_idx, tc = b, a, 1, tb
        if tc not in (2, 4):
            continue
        # Top option must have 1s ONLY on perfect cues; low option none.
        if np.sum(top_vec[4:] > 0.5) > 0:
            continue
        if np.sum(low_vec[:4] > 0.5) > 0:
            continue
        low_mask = low_vec > 0.5
        if int(np.sum(low_mask)) < 4:
            continue
        sig = (tc, tuple(sorted(float(x) for x in np.round(VALID[low_mask], 2))))
        if sig in EXCLUDE:
            continue
        hits.append(1.0 if int(resp) == top_idx else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.6300 (var=0.0703)
**Candidate trajectory (this loop):**
  - iter 1: 0.5796 (var=0.0327) (Δ vs real -0.0504)
  - iter 2: 0.5229 (var=0.0137) (Δ vs real -0.1071)
  - iter 3: 0.5383 (var=0.0281) (Δ vs real -0.0917)
  - iter 4: 0.5171 (var=0.0191) (Δ vs real -0.1129)
  - iter 5: 0.5575 (var=0.0193) (Δ vs real -0.0725)
  - iter 6: 0.6204 (var=0.0138) (Δ vs real -0.0096)
  - iter 7: 0.5975 (var=0.0244) (Δ vs real -0.0325)
  - iter 8: 0.6300 (var=0.0185) (Δ vs real +0.0000)
  - iter 9 (current): 0.6817 (var=0.0127) (Δ vs real +0.0517)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2267 (var=0.0083)
- pi_3: 0.5687 (var=0.0053)
- pi_1: 0.8783 (var=0.0092)
- pi_2: 0.1263 (var=0.0072)
- pi_3_1: 0.6646 (var=0.0194)
- pi_5: 0.7004 (var=0.0351)

### Experiment 7
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # Design-time constants (fixed by this experiment: 29 experts,
    # 6 @ .85, 5 @ .80, 6 @ .75, 12 @ .50).
    V = np.array([0.85] * 6 + [0.80] * 5 + [0.75] * 6 + [0.50] * 12, dtype=float)

    # Reference (midpoint) parameter values. The design guarantees that on
    # every critical cell the SIGN of each theory's evidence is constant
    # over its ENTIRE parameter box, so midpoint values classify trials
    # identically to any admissible parameter.
    TAU = 0.28      # SCWI box [0.15, 0.42]
    GAMMA = 1.9     # PAHI box [1.4, 2.4]
    w_scwi = -(V + TAU * np.square(V))   # SCWI anti-validity weights
    w_pahi = -(np.power(V, GAMMA))       # PAHI anti-validity weights

    def _to_arr(x):
        if isinstance(x, np.ndarray):
            return x.astype(float)
        if isinstance(x, (list, tuple)):
            return np.asarray(x, dtype=float)
        s = str(x).strip().strip('[]()')
        toks = [t for t in s.replace(',', ' ').split() if t != '']
        return np.array([float(t) for t in toks])

    hits = []
    for a_raw, b_raw, r_raw in zip(data['option_a_ratings'],
                                    data['option_b_ratings'],
                                    data['response']):
        try:
            a = _to_arr(a_raw)
            b = _to_arr(b_raw)
            r = int(r_raw)
        except Exception:
            continue
        if r not in (0, 1):
            continue
        n = int(min(a.size, b.size, V.size))
        if n == 0:
            continue
        diff = a[:n] - b[:n]
        d_scwi = float(np.dot(w_scwi[:n], diff))   # value_A - value_B under SCWI
        d_pahi = float(np.dot(w_pahi[:n], diff))   # value_A - value_B under PAHI

        # Keep only trials where the two theories make OPPOSITE predictions
        # with a non-trivial margin on both sides (this automatically selects
        # the knife-edge count-ratio cells and discards controls, ties,
        # anchors, and calibration trials, where the theories agree).
        if abs(d_scwi) <= 0.05 or abs(d_pahi) <= 0.05:
            continue
        if (d_scwi > 0) == (d_pahi > 0):
            continue

        scwi_option = 0 if d_scwi > 0 else 1   # 0 = A, 1 = B
        hits.append(1.0 if r == scwi_option else 0.0)

    if not hits:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.6462 (var=0.0395)
**Candidate trajectory (this loop):**
  - iter 1: 0.5569 (var=0.0327) (Δ vs real -0.0894)
  - iter 2: 0.5509 (var=0.0090) (Δ vs real -0.0953)
  - iter 3: 0.5663 (var=0.0426) (Δ vs real -0.0800)
  - iter 4: 0.5472 (var=0.0205) (Δ vs real -0.0991)
  - iter 5: 0.6069 (var=0.0145) (Δ vs real -0.0394)
  - iter 6: 0.6653 (var=0.0221) (Δ vs real +0.0191)
  - iter 7: 0.6703 (var=0.0324) (Δ vs real +0.0241)
  - iter 8: 0.6828 (var=0.0248) (Δ vs real +0.0366)
  - iter 9 (current): 0.7372 (var=0.0117) (Δ vs real +0.0909)
**Other theories' values on this metric (for reference):**
- pi_3_1: 0.6975 (var=0.0214)
- pi_4: 0.2169 (var=0.0147)
- pi_1: 0.8762 (var=0.0079)
- pi_2: 0.1344 (var=0.0070)
- pi_3: 0.6412 (var=0.0050)
- pi_5: 0.6056 (var=0.0640)

### Experiment 8
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Kill-composite: fraction of choices falling on the HIGH-validity stack
    # across the parameter-free sign-kill trials of this design.
    #
    # Kill trials pit a endorsements on higher-validity cues against b
    # endorsements on lower-validity cues (all other cues 0 for both
    # options, disjoint supports, each stack within one validity block)
    # at count ratios that lie strictly between the two theories'
    # weight-ratio windows:
    #   5 x 0.90 vs 7 x 0.70   (r = 1.40)
    #   9 x 0.70 vs 14 x 0.50  (r = 1.556)
    #   5 x 0.90 vs 11 x 0.50  (r = 2.20)
    # SCWI's ratio window w(v_h)/w(v_l) is BELOW each r for every admissible
    # tau, so SCWI always prefers the high-validity stack. PAHI's power-law
    # window is ABOVE each r for every admissible gamma, so PAHI always
    # prefers the low-validity stack. The metric is the fraction of kill
    # trials on which the subject chose the high-validity stack; mirrored
    # versions of each kill pair absorb side bias. It is well defined on
    # the pooled DataFrame and on a single subject slice (18 kill trials
    # per subject: 6 unique kill pairs x 3 repetitions).
    import numpy as np

    validities = np.array([0.9] * 5 + [0.7] * 9 + [0.5] * 14)
    n_feat = validities.size

    kill_keys = set()
    kill_keys.add(((5, 0.9), (7, 0.7)))
    kill_keys.add(((9, 0.7), (14, 0.5)))
    kill_keys.add(((5, 0.9), (11, 0.5)))

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    hits = 0
    total = 0

    for i in range(len(a_col)):
        a_cell = a_col[i]
        b_cell = b_col[i]
        resp = r_col[i]
        try:
            a = np.asarray(a_cell, dtype=float).ravel()
            b = np.asarray(b_cell, dtype=float).ravel()
        except Exception:
            continue
        if a.size != n_feat or b.size != n_feat:
            continue

        ones_a = np.flatnonzero(a == 1)
        ones_b = np.flatnonzero(b == 1)
        if ones_a.size == 0 or ones_b.size == 0:
            continue
        # Kill trials have disjoint supports (no tied cues).
        if np.intersect1d(ones_a, ones_b).size > 0:
            continue

        va = validities[ones_a]
        vb = validities[ones_b]
        # Each stack must sit entirely within one validity block.
        if np.unique(va).size != 1 or np.unique(vb).size != 1:
            continue

        sig_a = (int(ones_a.size), float(va[0]))
        sig_b = (int(ones_b.size), float(vb[0]))
        key = tuple(sorted([sig_a, sig_b]))
        if key not in kill_keys:
            continue
        if sig_a[1] == sig_b[1]:
            continue

        # High-validity stack is the one whose cues have larger validity.
        h_is_a = sig_a[1] > sig_b[1]

        try:
            r = int(round(float(resp)))
        except Exception:
            continue
        # response: 0 = chose A, 1 = chose B.
        chose_h = (r == 0) if h_is_a else (r == 1)
        if chose_h:
            hits += 1
        total += 1

    if total == 0:
        return 0.5
    return float(hits) / float(total)
```

**Observed (real) value:** 0.6044 (var=0.0653)
**Candidate trajectory (this loop):**
  - iter 1: 0.5900 (var=0.0332) (Δ vs real -0.0144)
  - iter 2: 0.5578 (var=0.0296) (Δ vs real -0.0467)
  - iter 3: 0.5544 (var=0.0482) (Δ vs real -0.0500)
  - iter 4: 0.4467 (var=0.0264) (Δ vs real -0.1578)
  - iter 5: 0.5622 (var=0.0329) (Δ vs real -0.0422)
  - iter 6: 0.6922 (var=0.0236) (Δ vs real +0.0878)
  - iter 7: 0.6656 (var=0.0299) (Δ vs real +0.0611)
  - iter 8: 0.6567 (var=0.0250) (Δ vs real +0.0522)
  - iter 9 (current): 0.7544 (var=0.0178) (Δ vs real +0.1500)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2078 (var=0.0291)
- pi_3_1: 0.7078 (var=0.0349)
- pi_1: 0.8611 (var=0.0123)
- pi_2: 0.1200 (var=0.0071)
- pi_3: 0.6056 (var=0.0099)
- pi_5: 0.6122 (var=0.0654)

### Experiment 9
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def _vec(x):
        if isinstance(x, np.ndarray):
            a = x.astype(float).ravel()
            if a.size > 0:
                return a
        try:
            a = np.asarray(x, dtype=float).ravel()
            if a.size > 0:
                return a
        except (TypeError, ValueError):
            pass
        s = str(x).strip().strip('[]()')
        toks = [t for t in s.replace(',', ' ').split() if t != '']
        return np.asarray([float(t) for t in toks], dtype=float)

    df = data.reset_index(drop=True)
    if df.shape[0] == 0 or 'response' not in df.columns:
        return 0.0

    A_list = [_vec(v) for v in df['option_a_ratings'].tolist()]
    B_list = [_vec(v) for v in df['option_b_ratings'].tolist()]
    R = np.asarray([float(r) for r in df['response'].tolist()], dtype=float)
    sids = df['subject_id'].tolist()

    rows = []
    for i in range(len(df)):
        a = A_list[i]
        b = B_list[i]
        r = R[i]
        n = a.size
        ta = float(a[:3].sum()); ma = float(a[3:12].sum()); la = float(a[12:].sum())
        tb = float(b[:3].sum()); mb = float(b[3:12].sum()); lb = float(b[12:].sum())
        row = {'anch0': np.nan, 'Lk': np.nan, 'Mk': np.nan, 'cH': np.nan}
        sa = ta + ma + la; sb = tb + mb + lb
        # ANCHOR: one option all-1s, the other all-0s. Record whether the
        # subject picked the all-0 option (engaged anti-validity subjects
        # do this with prob ~1 - eps/2; guessers 0.5).
        if n >= 12 and sa == float(n) and sb == 0.0:
            row['anch0'] = 1.0 if r == 1.0 else 0.0
        elif n >= 12 and sb == float(n) and sa == 0.0:
            row['anch0'] = 1.0 if r == 0.0 else 0.0
        # Top-stack (H) option: exactly the three v=1.00 cues endorsed.
        aH = (n >= 12) and (ta == 3.0) and (ma == 0.0) and (la == 0.0)
        bH = (n >= 12) and (tb == 3.0) and (mb == 0.0) and (lb == 0.0)
        if aH != bH:
            h = 0.0 if aH else 1.0
            opp = b if aH else a
            ot = float(opp[:3].sum()); om = float(opp[3:12].sum()); ol = float(opp[12:].sum())
            if ot == 0.0:
                if om == 0.0 and 8.0 <= ol <= 12.0:
                    # 0.50-ladder kill cell (3x1.00 vs 8-12x0.50)
                    row['Lk'] = 1.0 if r == h else 0.0
                elif ol == 0.0 and 6.0 <= om <= 9.0:
                    # 0.60-ladder kill cell (3x1.00 vs 6-9x0.60)
                    row['Mk'] = 1.0 if r == h else 0.0
                elif om == 0.0 and ol == 5.0:
                    # L-control (3x1.00 vs 5x0.50)
                    row['cH'] = 1.0 if r == h else 0.0
                elif ol == 0.0 and om == 4.0:
                    # M-control (3x1.00 vs 4x0.60)
                    row['cH'] = 1.0 if r == h else 0.0
        rows.append(row)

    tdf = pd.DataFrame(rows)
    tdf['sid'] = sids

    stats = []
    for sid, g in tdf.groupby('sid', sort=False):
        Lv = g['Lk'].dropna()
        Mv = g['Mk'].dropna()
        sL = float(Lv.mean()) if len(Lv) > 0 else 0.5
        sM = float(Mv.mean()) if len(Mv) > 0 else 0.5
        an = g['anch0'].dropna()
        ok_a = (float(an.sum()) == float(len(an))) if len(an) > 0 else True
        cv = g['cH'].dropna()
        ok_c = (float(cv.mean()) <= 0.375) if len(cv) > 0 else True
        stats.append({'sid': sid, 'sL': sL, 'sM': sM, 'keep': bool(ok_a and ok_c)})

    st = pd.DataFrame(stats)
    if st.shape[0] == 0:
        return 0.0
    scr = st[st['keep']]

    # Multi-subject (pooled) case: cross-ladder consistency = Pearson
    # correlation across engaged subjects between the 0.50-ladder and
    # 0.60-ladder top-stack choice rates.
    if scr.shape[0] >= 4:
        x = scr['sL'].to_numpy(dtype=float)
        y = scr['sM'].to_numpy(dtype=float)
        if x.std() > 1e-9 and y.std() > 1e-9:
            with np.errstate(all='ignore'):
                r = float(np.corrcoef(x, y)[0, 1])
            if np.isfinite(r):
                return r

    # Single-subject / degenerate case: within-subject cross-ladder
    # agreement, bounded in [-1, 1] and tight under both theories.
    d = np.abs(st['sL'].to_numpy(dtype=float) - st['sM'].to_numpy(dtype=float))
    return float(np.clip(1.0 - 2.0 * float(d.mean()), -1.0, 1.0))
```

**Observed (real) value:** 0.1963 (var=0.0601)
**Candidate trajectory (this loop):**
  - iter 1: 0.4412 (var=0.0444) (Δ vs real +0.2449)
  - iter 2: 0.1635 (var=0.0422) (Δ vs real -0.0328)
  - iter 3: 0.1342 (var=0.1626) (Δ vs real -0.0621)
  - iter 4: 0.0681 (var=0.1474) (Δ vs real -0.1282)
  - iter 5: 0.3517 (var=0.0488) (Δ vs real +0.1554)
  - iter 6: 0.1839 (var=0.0245) (Δ vs real -0.0124)
  - iter 7: -0.0215 (var=0.0402) (Δ vs real -0.2178)
  - iter 8: -0.1365 (var=0.0199) (Δ vs real -0.3328)
  - iter 9 (current): 0.7712 (var=0.0097) (Δ vs real +0.5749)
**Other theories' values on this metric (for reference):**
- pi_3_1: 0.3684 (var=0.0180)
- pi_5: 0.9529 (var=0.0191)
- pi_1: 0.9048 (var=0.0084)
- pi_2: 0.8735 (var=0.0093)
- pi_3: 0.2390 (var=0.0208)
- pi_4: 0.8613 (var=0.0179)

### Experiment 10
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Cell patterns are (top-cue count on H option, mid-cue count on L option,
    # low-cue count on L option), where H = the option whose 1s sit exclusively
    # on the v=1.00 cues and L = the option whose 1s sit exclusively on the
    # v<=0.60 cues.
    # TWIN-KILL cells (r = 1/v_l, the gamma=1 flip point for HEAI):
    #   4x1.00 vs 8x0.50  (trials 3, 9, 13, 19)   and
    #   3x1.00 vs 5x0.60  (trials 7, 10, 17, 20)  -> 32 rows/subject.
    # HIGH-RATIO cells (r = 3.0 on ladder 1, r = 2.0 on ladder 2):
    #   4x1.00 vs 12x0.50 (trials 5, 15)          and
    #   3x1.00 vs 6x0.60  (trials 8, 18)          -> 16 rows/subject.
    TWIN_PATTERNS = {(4.0, 0.0, 8.0), (3.0, 5.0, 0.0)}
    HIGH_PATTERNS = {(4.0, 0.0, 12.0), (3.0, 6.0, 0.0)}

    def _counts(ratings):
        try:
            arr = np.asarray(list(ratings), dtype=float).ravel()
        except Exception:
            return None
        if arr.size != 22:
            return None
        return (float(arr[0:4].sum()), float(arr[4:10].sum()), float(arr[10:22].sum()))

    def _is_H(c):
        # H option: all its 1s are on the four v=1.00 cues.
        return c[0] > 0 and c[1] == 0.0 and c[2] == 0.0

    def _is_L(c):
        # L option: no top-cue 1s, at least one 1 on the v<=0.60 cues.
        return c[0] == 0.0 and (c[1] > 0 or c[2] > 0)

    twin_hits = []
    high_hits = []

    responses = data["response"].values
    a_list = data["option_a_ratings"].values
    b_list = data["option_b_ratings"].values

    for resp, ra, rb in zip(responses, a_list, b_list):
        ca = _counts(ra)
        cb = _counts(rb)
        if ca is None or cb is None:
            continue
        try:
            r = int(float(resp))
        except Exception:
            continue
        if _is_H(ca) and _is_L(cb):
            h_cnt, l_cnt, h_is_a = ca, cb, True
        elif _is_H(cb) and _is_L(ca):
            h_cnt, l_cnt, h_is_a = cb, ca, False
        else:
            continue
        pattern = (float(h_cnt[0]), float(l_cnt[1]), float(l_cnt[2]))
        chose_h = (r == 0) if h_is_a else (r == 1)
        val = 1.0 if chose_h else 0.0
        if pattern in TWIN_PATTERNS:
            twin_hits.append(val)
        elif pattern in HIGH_PATTERNS:
            high_hits.append(val)

    if len(twin_hits) == 0 or len(high_hits) == 0:
        return 0.0

    p_twin = float(np.mean(twin_hits))
    p_high = float(np.mean(high_hits))
    # Signed psychometric rise between the twin-kill ratio region and the
    # high-ratio region. SCWI (step function pinned in [2.14, 2.35] ladder 1
    # and [1.76, 1.89] ladder 2) produces a LARGE rise (~0.55-0.60); HEAI
    # (bimodal gamma, shallow mode straddling the gamma=1 flip) produces a
    # SMALL rise (~0.20).
    return p_high - p_twin
```

**Observed (real) value:** 0.2963 (var=0.0806)
**Candidate trajectory (this loop):**
  - iter 1: 0.1994 (var=0.0635) (Δ vs real -0.0969)
  - iter 2: 0.1988 (var=0.0693) (Δ vs real -0.0975)
  - iter 3: 0.1731 (var=0.0258) (Δ vs real -0.1231)
  - iter 4: 0.3594 (var=0.0425) (Δ vs real +0.0631)
  - iter 5: 0.3406 (var=0.0368) (Δ vs real +0.0444)
  - iter 6: 0.3656 (var=0.0275) (Δ vs real +0.0694)
  - iter 7: 0.2844 (var=0.0495) (Δ vs real -0.0119)
  - iter 8: 0.3175 (var=0.0408) (Δ vs real +0.0213)
  - iter 9 (current): 0.3425 (var=0.0351) (Δ vs real +0.0463)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1713 (var=0.0509)
- pi_3_1: 0.4531 (var=0.0906)
- pi_1: 0.0181 (var=0.0129)
- pi_2: 0.0131 (var=0.0090)
- pi_3: 0.3713 (var=0.0215)
- pi_4: 0.1175 (var=0.0341)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate faithfully implements the prescribed CNNAI family (additive anti-validity integration, context-referenced elasticity, per-trial attention diffusion, guesser mixture) and its aggregate loss (0.1070) made it the accepted base. Per-experiment, however, there is a clear and systematic pattern of residual error rather than random noise. (1) The headline mechanism — near-zero cross-context consistency — FAILS: Exp 9 cross-ladder r = 0.441 vs real 0.196, worse than plain SCWI (pi_3: 0.239). Diagnosis: the subject-level ladder composites sL and sM average over many cells, so the per-cell context redraw (sd 0.36) averages out within each ladder, leaving the STABLE per-subject components (eta_base sd 0.25, and especially the heterogeneous beta with lognormal sd 0.45) to dominate between-subject differences. Stable beta is a hidden trait: high-beta subjects are more extreme on BOTH ladders, producing spurious cross-ladder correlation even with independent etas. (2) Between-subject variances are systematically underpredicted by roughly a factor of 2 on the diagnostic composites: Exp 2 var 0.025 vs 0.093, Exp 5 0.031 vs 0.064, Exp 6 0.033 vs 0.070, Exp 8 0.033 vs 0.065. The per-trial attention diffusion (sd 0.15) is too weak, and the averaging-out of per-cell eta draws suppresses exactly the variance the redraw was supposed to create. (3) The anti-validity steepness composites all undershoot in the same direction: Exp 2 (0.186 vs 0.274), Exp 5 (0.600 vs 0.684), Exp 6 (0.580 vs 0.630), Exp 7 (0.557 vs 0.646), Exp 10 rise (0.199 vs 0.296). The proposer deliberately centered eta at 0.19, below the kill-knife at ~0.31–0.34; the real data want more probability mass above the knife. Exps 1, 3, 4, 8 are essentially on target (slopes/composites within 0.01–0.03 of real), confirming the shared anti-validity core is sound.
Rationale: Regenerate, staying strictly inside the CNNAI family, with three targeted recalibrations. (A) Break the hidden cross-ladder coupling: shrink the stable elasticity component (eta_base sd 0.25 → ~0.10) and enlarge the per-context redraw (sd 0.36 → ~0.50), AND narrow the engaged beta distribution (lognormal sd 0.45 → ~0.25, or cap beta heterogeneity) so stable response extremity no longer correlates the two ladders. This should pull Exp 9 from 0.44 toward the observed 0.196 without touching the anti-validity core. (B) Restore between-subject variance: raise the per-trial attention diffusion (sd 0.15 → ~0.30) and/or let epsilon vary more across engaged subjects; the composites need var ≈ 0.06–0.09, not 0.03. Note the tension and its resolution: increasing per-context eta spread raises within-context variance while LOWERING cross-context correlation — both desired — whereas the current parameterization does neither because the context noise averages out over cells. (C) Shift the eta center modestly upward (0.19 → ~0.25–0.28, still below the 0.31–0.34 knife) so more contexts sit above the kill flip, lifting Exps 2/5/6/7 and the Exp 10 rise toward their real values while keeping enough mass below gamma = 1 for the twin cells. Watch Exp 3 (already slightly high at 0.139 vs 0.123) and Exp 1 slope as guardrails; if they overshoot, compensate by trimming the guesser fraction slightly (16% → ~13%) rather than reversing the eta shift. All changes are parameter recalibrations within the prescribed mechanism — no family switch needed.

**Outcome of your advice:** iter 1 candidate loss=0.1070 -> iter 2 candidate loss=0.1032 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The iter-2 recalibration did exactly what my iter-1 critique asked on its primary target, and the gate accepted it (loss 0.1070 -> 0.1032). Exp 9 cross-ladder r moved from 0.441 to 0.164 (real 0.196) — the headline CNNAI prediction of near-zero cross-context consistency is now essentially on target, and Exp 4 improved to within 0.01 of real. The narrowed beta, tiny stable eta component, and trimmed guesser fraction should be kept. However, the rest of the profile regressed or stagnated, and the pattern is diagnostic rather than noisy. (1) All anti-validity steepness composites UNDERSHOOT and got worse or flat vs iter 1: Exp 2 0.139 vs real 0.274 (error grew from -0.088 to -0.134), Exp 5 0.553 vs 0.684, Exp 6 0.523 vs 0.630, Exp 7 0.551 vs 0.646, Exp 8 0.558 vs 0.604, Exp 10 rise 0.199 vs 0.296 (unchanged despite the eta center moving 0.19 -> 0.26). (2) Between-subject variances got WORSE despite doubling the attention diffusion: Exp 5 var 0.031 -> 0.019 (real 0.064), Exp 6 0.033 -> 0.014 (real 0.070), Exp 2 stuck at 0.025 (real 0.093), Exp 8 0.030 (real 0.065). My iter-1 part-(B) prescription (inflate variance via per-trial diffusion) was wrong in mechanism, and I withdraw it: per-trial multiplicative noise is independent across trials, so it averages out within a subject's composite — it adds within-subject binomial-like noise, not between-subject dispersion. The same averaging-out logic dooms the steepness composites: the candidate redraws eta per endorsed-cue INDEX set, so each subject's composite averages over ~10-20 independent per-cell eta draws (many of which land near or below the kill knife, flipping or flattening individual cells), concentrating every subject near the population mean. This single design choice — per-cell context hashing — explains both failures simultaneously: diluted mean steepness AND suppressed between-subject variance, and it is also a drift from the arbiter sketch, which specified eta redrawn 'per context/block (e.g., between the 0.50-ladder and 0.60-ladder of Exp 1)', i.e., at the ladder/block level, not per trial cell.
Rationale: Regenerate, staying strictly inside the CNNAI family, with one structural recalibration plus parameter retuning. (A) COARSEN THE CONTEXT KEY: replace the hash over the endorsed-cue index set with a hash over the sorted set of distinct VALIDITY VALUES among the endorsed cues (equivalently, a bucketed v_ref / ladder identifier). All cells within one validity ladder then share a single per-subject eta draw: high-eta subjects are steep on every cell of that ladder, low-eta subjects shallow on every cell — restoring both the mean steepness composites (Exps 2/5/6/7/8/10) and the between-subject variance (target 0.06-0.09) in one move, because the composite is now driven by one shared draw instead of an average of many. Different ladders (Exp 9's {1.00,0.50} vs {1.00,0.60} cells; Exp 10's two ladders) remain distinct contexts, so cross-ladder decoupling and the hard-won Exp 9 r ~ 0.16-0.20 are preserved. (B) RETUNE: since pooling removes the dilution, keep the eta center at ~0.26 — do NOT raise it further; Exp 1 slope (-0.181 vs -0.175) and Exp 3 (0.145 vs 0.123) are already slightly steep and will steepen more under pooling, so be ready to trim the center toward ~0.22 if those guardrails overshoot. Reduce the per-trial attention diffusion back to sd ~0.15-0.20: at 0.30 it flattens psychometric composites (a contributor to the Exp 2 undershoot) and, as established, cannot create between-subject variance anyway. Keep the iter-2 constants that worked: narrow engaged beta (lognormal sd 0.25, clip [1,5]), tiny stable eta_base sd 0.10, 13% guessers, widened epsilon. (C) Verify against the variance targets explicitly: Exp 2 var 0.093, Exp 5 0.064, Exp 6 0.070, Exp 8 0.065 — if the shared-per-ladder eta still underdelivers, widen the per-context redraw sd (0.50 -> 0.60-0.70) rather than re-inflating beta or eta_base, since those were the coupling channels that broke Exp 9 in iter 1. All changes are within the prescribed CNNAI mechanism; no family switch.

**Outcome of your advice:** iter 2 candidate loss=0.1032 -> iter 3 candidate loss=0.0968 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The iter-3 candidate faithfully implements the prescribed CNNAI family, and the ladder-pooled context key I recommended in iter 2 did what it was supposed to do partially: loss improved again (0.1032 -> 0.0968), Exp 2's mean moved substantially toward target (0.139 -> 0.208 vs real 0.274), Exp 4 is essentially perfect (0.166 vs 0.163), and Exps 1/3 remain close. The ladder pooling stays. But the residual error pattern is still systematic, and this iteration reveals that the single remaining free knob — the eta dispersion — is now miscalibrated in BOTH directions at once, which is diagnostic of a deeper placement problem. (1) Steepness composites still undershoot across the board: Exp 5 0.569 vs 0.684, Exp 6 0.538 vs 0.630, Exp 7 0.566 vs 0.646, Exp 8 0.554 vs 0.604, Exp 10 rise 0.173 vs 0.296 (which actually got WORSE than iter 2's 0.199). The likely root cause is that the eta->behavior map is too FLAT: v_ref is the MEAN validity of endorsed cues, so the ratios v_j/v_ref sit close to 1 and even large eta draws produce only mildly discriminating weight profiles; meanwhile roughly a third of the engaged mass (center 0.23, sd 0.60) draws eta < 0, i.e. flat or validity-tolerant weights that actively dilute every steepness composite. (2) Between-subject variance is now wrong in opposite directions simultaneously: Exp 9 var exploded to 0.163 vs real 0.060 (the sd-0.60 per-ladder redraw makes subjects flip-flop steepness between the 0.50- and 0.60-ladders far more than real subjects do), while Exps 2/5/6/10 var remain 2-4x UNDER target (0.017-0.028 vs 0.064-0.093). A single sd knob cannot fix both — the variance is in the wrong place: too much cross-ladder inconsistency, too little within-ladder dispersion around a steep mean. (3) Exp 9's point estimate drifted low (0.134 vs 0.196) because the stable eta_base component (sd 0.10) is now dwarfed by the widened redraw. The mechanism family is sound and clearly the best-fitting theory so far (compare pi_3_1 and pi_5 rows); what fails is the placement and shape of the eta distribution and the strength of the normalization.
Rationale: Verdict: regenerate, staying strictly inside CNNAI, building on the accepted iter-3 base (keep the ladder-pooled context key, narrow beta sd 0.25, 13% guessers, attention diffusion 0.18 — all validated). Three coordinated recalibrations: (A) STEEPEN THE EFFECTIVE WEIGHT MAP. Switch v_ref from the mean to the MAX validity of the endorsed cues (the arbiter sketch explicitly allowed 'mean/max', so this is an in-family normalization swap), and/or raise the eta center from 0.23 to ~0.35-0.40. Mean-referencing compresses v_j/v_ref toward 1, which is why three iterations of eta tuning have barely moved Exps 5/6/7/10; max-referencing makes low-validity ratios substantially below 1, so moderate eta now produces genuinely SCWI-like discrimination. This is the primary lever for the undershooting means. (B) REBALANCE THE ETA DECOMPOSITION — this is not oscillation on the sd knob but a change of strategy: cut the per-context redraw sd from 0.60 to ~0.30-0.35 AND raise the stable eta_base sd from 0.10 to ~0.15. This simultaneously (i) shrinks the cross-ladder flip-flopping that blew up Exp 9's variance (0.163 -> toward 0.060), (ii) lifts Exp 9's point estimate toward 0.196 by strengthening the stable component's relative share, and (iii) — together with (A) — concentrates subjects across the per-cell kill knives so that composite between-subject variance on Exps 5/6 rises toward 0.064-0.070 (variance comes from subjects straddling the transition region, not from saturated extremes spread over two ladders). Aim for sigma_ctx roughly 2x sigma_base, which yields cross-ladder r near the observed 0.2 by construction. (C) RIGHT-SKEW THE ETA DISTRIBUTION: replace the symmetric normal redraw with a lognormal-like shape (as the arbiter sketch actually specified) so that little engaged mass falls below eta ~ 0.1 — the currently ~35% flat-weight subjects dilute the Exp 2 mean/variance and the Exp 5/6/7 composites. For Exp 2's large real variance (0.093), the right tail (strongly anti-validity subjects with composite differences near +0.7-0.9) is the missing mass, not the left tail. (D) GUARDRAILS: Exps 1 (-0.185 vs -0.175) and 3 (0.138 vs 0.123) are already slightly steep and will steepen under (A); compensate by trimming the engaged beta median (2.4 -> ~2.0) or widening epsilon slightly rather than reversing the eta shift, and verify Exp 9's r does not exceed ~0.25 and Exp 4 stays within ~0.02. Do NOT re-inflate beta heterogeneity beyond sd ~0.30 — that was the coupling channel that broke Exp 9 in iter 1.

**Outcome of your advice:** iter 3 candidate loss=0.0968 -> iter 4 candidate loss=0.1234 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The gate rejected iter 4 (loss 0.0968 -> 0.1234), and the per-experiment pattern shows my iter-3 prescriptions were implemented with two critical errors. Empirically, relative to the accepted iter-3 base: every kill/steepness composite moved FURTHER below target (Exp 5: 0.569->0.501 vs real 0.684; Exp 6: 0.538->0.517 vs 0.630; Exp 7: 0.566->0.547 vs 0.646; Exp 8: 0.554->0.447 vs 0.604), the steepness guardrails overshot (Exp 1 slope -0.203 vs real -0.175; Exp 10 rise 0.359 vs 0.296), and Exp 9 collapsed on both statistics (r 0.068 vs 0.196; var 0.147 vs 0.060). Only Exp 2 improved marginally (0.213 vs 0.274). Diagnosis 1 — arithmetic bug in the redraw dispersion: a lognormal with median 0.16 and log-sd 1.05 has sd = mean*sqrt(e^{sigma^2}-1) = 0.278*1.42 ~= 0.39, not the claimed ~0.21 (the proposer appears to have used the median instead of the mean in the sd formula). The per-ladder redraw is nearly twice as dispersed as designed, which is exactly why Exp 9's cross-ladder r fell to 0.068 and its variance stayed at 2.5x target. Diagnosis 2 — the heavy upper tail is actively harmful, not helpful: the kill composites are near-knife threshold statistics softened by attention noise, so below-knife subjects only reach p~0.6-0.7 on the tight cells (5x0.9 vs 7x0.7), while tail mass at eta 0.6-2 chooses the low-validity stack near-deterministically at beta 2.9. The composite is dragged down almost one-for-one by upper-tail mass, which is why F(eta<0.34)~0.68 still produced a composite of 0.45. The same tail overshot Exp 1's slope, and the strictly positive lognormal (essentially no mass at eta<=0) drove p_twin->0 on Exp 10's twin cells, inflating the rise to 0.359. Diagnosis 3 — the reference theories confirm what the real data want: pi_3 (SCWI, effective curvature entirely below the knife) scores 0.606 on Exp 8 (real 0.604) and a 0.371 rise on Exp 10 (real 0.296). The real engaged population is SCWI-like in effective curvature: eta bulk around 0.1-0.25 with a bounded upper tail, not a median-0.16 lognormal with mass out to eta 2. The proposer's own rationale flagged the risk ('I monitor the guardrails: Exps 1/3 may steepen slightly') but underestimated every guardrail breach and overestimated every gain — their analytic F(knife) threshold model of the composites ignores attention-noise softening and is systematically miscalibrated.
Rationale: Regenerate, building on the UNCHANGED iter-3 accepted base (the gate already reverted; do not re-derive from iter 4). My iter-3 advice needs explicit correction: the rebalance direction (cut redraw sd, raise stable sd) was right but was implemented with the wrong lognormal arithmetic, and the right-skew prescription backfired — I withdraw the push toward a heavy upper tail. The new recommendation keeps the iter-3 eta CENTER and fixes only the dispersion and shape. (1) REPLACE THE REDRAW DISTRIBUTION with a symmetric, tightly bounded one: center ~0.20-0.25, sd ~0.25-0.30, hard-clipped to roughly [-0.1, 0.8]. Targets: engaged F(eta < 0.34) ~ 0.65-0.75 and P(eta > 0.6) <= 0.05. If a lognormal is retained for the mild right-skew, use median ~0.20 with log-sd ~0.45-0.55 and VERIFY the actual sd numerically (sd = mean*sqrt(e^{sigma^2}-1)); do not trust the closed-form claim that produced the 0.21-vs-0.39 error. (2) Stable component: center ~0.05, sd 0.13-0.15, giving latent cross-ladder r = sigma_b^2/(sigma_b^2+sigma_c^2) ~ 0.2 with sigma_c ~ 0.27; check Exp 9 lands r in [0.15, 0.25] and var in [0.06, 0.10]. (3) REVERT beta to ~2.4-2.6 (the 2.9 raise overshot Exp 1 and Exp 10) and sharpen the near-knife mapping by CUTTING attention diffusion from 0.14 to ~0.08-0.10 instead — the kill cells have small score differences, so diffusion noise is what softens below-knife subjects to p~0.6; reducing it lifts the composites without steepening the global anti-validity slope. (4) GUARDRAILS: Exp 1 slope -0.175, Exp 3 ~0.123, Exp 4 ~0.163, Exp 10 rise 0.25-0.33. If Exp 10's rise falls back toward iter 3's 0.173, raise the eta center modestly (to 0.25-0.28) — do NOT re-inflate the upper tail, which is the channel that broke Exps 5/6/7/8 this iteration. (5) Exp 2 (mean 0.21 vs 0.274, var 0.016 vs 0.093) remains the stubborn residual: its trials fragment across many ladder keys, so per-subject composites average over multiple ctx draws. If variance is still far short after (1)-(3), COARSEN the context key (e.g., hash on the rounded max endorsed validity, or the top-two distinct validities) so more trials within an experiment share one draw — but never move to a per-experiment block key, which would re-couple Exp 9's two ladders and recreate the iter-1 r=0.44 failure. (6) METHODOLOGICAL: before committing parameters, numerically sweep the eta->composite map (including attention noise, beta, lapse, and the 13% guessers) on the actual Exp 5/6/8/10 cells instead of analytic F(knife) arguments — that argument predicted composites of 0.62-0.71 this iteration and delivered 0.45-0.52. All changes are parameter/shape recalibrations within the prescribed CNNAI mechanism (additive anti-validity integration, ladder-pooled context-referenced elasticity, softmax + lapse, guesser mixture); no family switch.

**Outcome of your advice:** iter 4 candidate loss=0.1234 -> iter 5 candidate loss=0.0913 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: The iter-5 candidate is the best-fitting theory this loop has produced (loss 0.0968 -> 0.0913, ACCEPTED), and my iter-4 prescriptions were implemented correctly: the symmetric bounded eta distribution fixed the iter-4 collapse (Exp 8 back to 0.562 from 0.447, Exp 7 up to 0.607 from 0.547, Exp 10 rise back to 0.341 from 0.359's overshoot toward the real 0.296, Exp 1 slope back to -0.183 from -0.203). Exps 1, 3, 4 are now essentially on target (|Δ| <= 0.02). However, the residual error is still systematic, not noise, and three failures have identifiable mechanisms. (1) EXP 5 IS THE WORST CELL AND GOT WORSE (0.503 vs real 0.684; iter 3 was 0.569). Root cause is structural: Exp 5's diagnostic cells discriminate the 1.00-vs-0.85 validity region, where mean-referenced normalization at eta~0.23 gives a weight ratio w(1.0)/w(0.85) = (1/0.85)^(1+eta_effective-ish) of only ~1.04 — essentially flat exactly where the metric needs SCWI-like curvature (~1.2+). Note even pi_3 (SCWI) only scores 0.553 and pi_3_1 scores 0.651, so the real population is MORE SCWI-following in this region than a lapse-laden SCWI; CNNAI at 0.503 is below all anti-validity references. (2) EXP 9's CROSS-LADDER r OVERSHOT (0.352 vs real 0.196) despite the by-construction latent r = 0.21. The gap between latent and empirical correlation means a residual stable coupling channel exists: the engaged epsilon range (0.02-0.17, stable across ladders) is now the dominant stable per-subject trait — high-epsilon subjects are less extreme on BOTH ladders, inflating r. Its variance is fine (0.049 vs 0.060), so the fix is correlation-specific. (3) BETWEEN-SUBJECT VARIANCE REMAINS 2-10x UNDER on Exps 2 (0.009 vs 0.093), 5 (0.016 vs 0.064), 6 (0.019 vs 0.070), 8 (0.033 vs 0.065), 10 (0.037 vs 0.081). Exp 2 is the extreme case: its trials fragment across many ladder keys ({0.9,0.8}, {0.9,0.7}, etc.), so each subject's composite averages over many independent eta draws and collapses toward the population mean — the same dilution mechanism I diagnosed in iter 2, still unaddressed for multi-ladder experiments.
Rationale: Regenerate, building on the accepted iter-5 base (keep: symmetric bounded eta, ladder-pooled context key, narrow beta sd 0.25, 13% guessers, attention diffusion 0.10). Four targeted moves, all in-family. (A) FIX EXP 5 VIA THE NORMALIZATION, NOT THE ETA CENTER: switch v_ref from the MEAN to the MAX endorsed validity. The arbiter sketch explicitly allowed 'mean/max', and this lever was never cleanly tested — iter-4's rejection is attributable to the lognormal arithmetic bug, not the reference swap. Max-referencing gives w(v_h)/w(v_l) = (v_h/v_l)^(1+eta), so at eta ~0.35-0.45 the 1.00-vs-0.85 ratio reaches ~1.22-1.28 (SCWI-like) and the 0.90-vs-0.70 kill ratio in Exp 8 crosses its knife, lifting Exps 5/6/7/8 means together. This is preferable to raising the eta center because it steepens the high-validity region specifically rather than globally. (B) FIX EXP 9's r WITH THE COUPLING CHANNEL, NOT THE REDRAW: shrink the engaged epsilon range from 0.02-0.17 to roughly 0.03-0.09 (it is the last stable trait correlating the two ladders), and cut the stable eta_base sd from 0.13 to ~0.10 so the latent target drops to ~0.14 — the empirical r should then land near 0.20 given the observed +0.14 latent-to-empirical gap. Verify numerically on the actual Exp 9 pipeline including the anchor/control screens. (C) COARSEN THE CONTEXT KEY for multi-ladder experiments: hash on the rounded MAX endorsed validity (or top-two distinct validities) rather than the full sorted distinct-validity set, so Exp 2's and Exp 5's within-experiment cells share draws and per-subject composites stop averaging over 5-10 independent etas. This is the primary lever for the chronic variance underprediction (Exp 2 var 0.009 vs 0.093) and it also helps the Exp 5/6 means. Do NOT go to a per-experiment key — that re-couples Exp 9's ladders and recreates the iter-1 r=0.44 failure. (D) GUARDRAILS: Exp 1 (-0.183 vs -0.175), Exp 3 (0.126 vs 0.123) and especially Exp 10 (0.341 vs 0.296, already +0.044 high) will steepen under (A); compensate by trimming the engaged beta median (2.34 -> ~2.0-2.2) rather than widening epsilon (which is now needed narrow for Exp 9) or reversing the normalization. Watch Exp 4 (0.142 vs 0.163, currently slightly low — should improve). As before, verify every composite claim by direct numerical simulation of the full pipeline (attention noise, lapse, guessers, screens) on the actual experimental cells — the analytic F(knife) arguments remain systematically optimistic.

**Outcome of your advice:** iter 5 candidate loss=0.0913 -> iter 6 candidate loss=0.0645 -> the gate marked it ACCEPTED.

### Iteration 6
Verdict: regenerate
Interpretation: The iter-6 candidate is a substantial, gate-confirmed improvement (0.0913 -> 0.0645) and is now the best-fitting theory in the entire comparison set. The proposer's two deliberate deviations from my iter-5 advice were both correct and I withdraw those prescriptions: (i) the algebraic invariance argument is right — w_h/w_l = (v_h/v_l)^(1+eta) is independent of v_ref, so max-referencing would only have deflated margins, not steepened ratios; (ii) the bucketed-mean context key (rather than my rounded-max suggestion, which would have re-coupled Exp 9's two ladders since both have max validity 1.0) was the smarter implementation of my coarsening advice. The results validate the knife-avoidance recalibration: Exp 9 is now essentially perfect on its headline statistic (r = 0.184 vs real 0.196), Exp 6 is on target (0.620 vs 0.630), Exp 7 slightly high (0.665 vs 0.646), Exps 3 close (0.133 vs 0.123). However, the residual error is still systematic, not noise, and now has a clean two-sided structure. (1) OVERSHOOTS from the eta-center/noise double move: Exp 8 jumped from -0.042 to +0.088 (0.692 vs 0.604) and Exp 10 from +0.044 to +0.069 (0.366 vs 0.296); Exp 1's slope is slightly too steep (-0.201 vs -0.175). (2) PERSISTENT UNDERSHOOTS: Exp 2 mean is stuck at 0.189 vs 0.274 across all six iterations, Exp 5 improved but remains low (0.619 vs 0.684, and note real subjects follow SCWI on Exp 5 MORE than a lapse-laden SCWI does: pi_3 scores only 0.553, pi_3_1 0.651), and Exp 4 is slightly low (0.138 vs 0.163). (3) THE MOST CHRONIC FAILURE IS NOW VARIANCE: between-subject variance is underpredicted on EVERY experiment — Exp 1 (0.003 vs 0.011), Exp 2 (0.018 vs 0.093), Exp 5 (0.020 vs 0.064), Exp 6 (0.014 vs 0.070), Exp 8 (0.024 vs 0.065), Exp 9 (0.025 vs 0.060), Exp 10 (0.028 vs 0.081) — a consistent 3-5x shortfall. The mechanism is identifiable: with eta centered at 0.12 and hard-clipped at [-0.25, 0.70], the engaged bulk sits deep in the saturated below-knife region where the eta->composite map is flat, so even the shared per-ladder draw cannot produce between-subject spread; real subjects (sd ~0.25-0.30 on these composites) must be spread across the transition region and into flatter-weight territory. The direction that fixes the most cells simultaneously is coherent: the undershooting experiments (2, 4, 5) all want FLATTER effective weights (lower eta), while the overshooting experiments (8, 10) have many-cue near-knife cells where restored attention diffusion (which scales with endorsement count) and a wider eta spread concentrated near the knives soften exactly them.
Rationale: Regenerate, building on the accepted iter-6 base (keep: bucketed-mean context key, mean-referenced v_ref, narrow beta sd 0.20, narrow epsilon 0.03-0.09, 13% guessers — all validated this iteration). The next move is a coordinated dispersion/flatness recalibration, NOT another center slide alone. (A) WIDEN THE ETA DISPERSION ~30-40% WHILE PRESERVING THE DECOMPOSITION RATIO: eta_base sd 0.09 -> ~0.12 and per-bucket redraw sd 0.19 -> ~0.26, keeping the center near 0.08-0.10 (a modest drop from 0.12). The ratio preservation keeps latent cross-ladder r at ~0.18-0.20, protecting the now-excellent Exp 9 point estimate, while the larger total spread (i) puts more engaged mass in the eta ~0.2-0.45 transition band where the eta->composite map has slope — the ONLY in-family source of the missing between-subject variance on Exps 5/6/8/9/10 — and (ii) adds near-knife mass that pulls Exp 8 (0.692 -> toward 0.60) and Exp 10 (0.366 -> toward 0.30) back down. (B) EXTEND THE LEFT CLIP from -0.25 to about -0.45 and let the widened distribution place modest mass at eta <= 0. Flatter weights (w ~ -v^0.6) are more tally-like on count-margin cells: this is the primary lever for the chronic Exp 2 undershoot (0.189 vs 0.274, unmoved in six iterations) and helps Exp 4 (0.138 vs 0.163) and the slightly-too-steep Exp 1 slope (-0.201 vs -0.175). It also raises variance everywhere (flat-weight subjects sit near composite ~0.5, far from the engaged cluster). CRITICAL: keep the RIGHT clip at 0.70 and the distribution symmetric-bounded — the iter-4 rejection was caused precisely by a heavy upper tail (mass at eta 0.6-2), which dragged every kill composite down one-for-one; do not recreate it. (C) RESTORE ATTENTION DIFFUSION from 0.05 to ~0.07-0.09. This is a DIFFERENTIATING lever, not a global one: per-trial noise accumulates with endorsement count, so it softens Exp 8's 9-14-cue kill cells and Exp 10's 8-12-cue cells far more than Exp 5's 2-7-cue cells or Exp 2's 5-cue cells — exactly the asymmetry needed to lower the overshooters while sparing the undershooters. It cannot create between-subject variance (my iter-2 lesson, still true), so treat it purely as a mean-softening channel for the many-cue experiments. (D) GUARDRAILS, checked by direct numerical simulation of the full pipeline (attention noise, lapse, guessers, anchor/control screens) on the actual cells — the proposer's iter-6 margin arithmetic was good; keep that discipline: Exp 6 is currently nearly perfect (0.620 vs 0.630) and its var (0.014 vs 0.070) should rise via (A) without moving the mean much; Exp 7 (0.665 vs 0.646) should drift down slightly under (B)/(C), which is fine; Exp 3 (0.133 vs 0.123) tolerates a small drop but verify it stays >= 0.10; Exp 9's r must stay in [0.15, 0.25] — if the widened redraw pushes it low, raise eta_base sd proportionally rather than re-narrowing the redraw. (E) If Exp 2's variance (0.018 vs 0.093) is still short after (A)+(B), the remaining gap likely requires a small explicitly flat-weight subpopulation (eta drawn near -0.8 to -1.0, where w_j -> -v_ref, i.e., pure tallying) of perhaps 8-12% of engaged subjects — test this numerically against Exp 4 (real 0.163; too many tally subjects overshoot it) and Exp 6 before committing. All changes are parameter/shape recalibrations strictly within the prescribed CNNAI mechanism; no family switch.

**Outcome of your advice:** iter 6 candidate loss=0.0645 -> iter 7 candidate loss=0.0786 -> the gate marked it REJECTED.

### Iteration 7
Verdict: regenerate
Interpretation: The gate rejected iter 7 (loss 0.0645 -> 0.0786), and the per-experiment decomposition shows the three-knob move split cleanly into one success and two failures. WHAT WORKED: restoring attention diffusion to 0.08 (knob C) did exactly what it was designed to do — Exp 8 fell from 0.692 to 0.666 (toward real 0.604) and Exp 10's rise landed essentially on target (0.284 vs 0.296, first time in the loop). The differentiating many-cue softening logic is validated; keep it. WHAT FAILED: the ~40% eta-dispersion widening (knob A) and the left-clip extension to -0.45 (knob B) broke the two things they were supposed to help. (1) EXP 9 COLLAPSED: cross-ladder r went from 0.184 (essentially perfect) to -0.021. The by-construction latent ratio (0.176) was preserved, but the empirical r is NOT the latent r when the added dispersion mass lands in the non-monotone region of the eta->composite map: the two ladders' kill cells flip at different etas (0.50-ladder knives span ~0.42-1.0, 0.60-ladder knives span ~0.36-1.05), so subjects drawn into the 0.3-0.6 band flip cells inconsistently ACROSS ladders, and the extra flat-region mass (eta <= 0, sitting near the composite midpoint) attenuates what correlation remains. Widening the redraw is therefore capped well below sd 0.26 by Exp 9 — iter-6's sd ~0.21 is close to the empirical ceiling. (2) THE UNDERSHOOTING MEANS GOT WORSE, IN THE DIRECTION OPPOSITE TO THE PROPOSER'S ARITHMETIC: Exp 2 fell 0.189 -> 0.167 (real 0.274), Exp 5 0.619 -> 0.591 (real 0.684), Exp 6 0.620 -> 0.598 (real 0.630). The proposer's cell arithmetic predicted 'roughly flat-to-slightly-up' for Exp 2 and material variance gains; both were wrong, continuing the pattern I flagged in iters 4 and 6 — analytic F(knife) arguments are systematically miscalibrated and must be replaced by full-pipeline numerical simulation. (3) VARIANCE IMPROVED ONLY MODESTLY (Exp 5 0.020->0.039, Exp 6 0.014->0.024, Exp 10 0.028->0.050) and at unacceptable cost. The chronic residuals on the iter-6 base remain: variance 3-5x under on every experiment, Exp 2/4/5 means under, Exp 8 over. My iter-6 knobs (A) and (B) were implemented at too-large magnitude and failed; (C) succeeded. The lesson is not to reverse direction but to change the variance CHANNEL: symmetric dispersion widening is exhausted (capped by Exp 9), so the missing between-subject variance and the Exp 2/5 undershoot must come from a DISCRETE mild flat-weight subpopulation instead.
Rationale: Verdict: regenerate, building on the UNCHANGED iter-6 accepted base (the gate already reverted; do not re-derive from iter 7). Keep from iter 7 only the attention diffusion at 0.08 — it is the one validated change (Exps 8/10 both improved). Then: (1) REVERT THE ETA DISPERSION TO ITER-6 VALUES: per-bucket redraw sd back to ~0.19-0.21, eta_base sd ~0.09-0.10, center ~0.11, left clip back to ~-0.30, right clip 0.70. The iter-7 widening is empirically capped by Exp 9: the empirical cross-ladder r decouples from the latent ratio once mass enters the inter-ladder knife band (~0.36-0.6) and the flat region, so total sd must stay near 0.21-0.22. Do not re-attempt the widened symmetric dispersion — that is the rejected iter-6 prescription (A)/(B), withdrawn. (2) ADD A MILD FLAT-WEIGHT SUBPOPULATION as the new variance channel: roughly 8-10% of ENGAGED subjects draw eta ~ -0.4 to -0.6 (stable across contexts). This revives my iter-6 conditional (E) in the milder form the proposer themselves suggested: the anti-tallying objection applies only at eta = -1 where weights become exactly constant; at eta ~ -0.5 the validity ordering is preserved (w_h/w_l = (v_h/v_l)^0.5 ~ 1.1-1.2), so these subjects are SCWI-like with low curvature, not tally-followers — they keep Exp 4's conflict tally-following low and do not prefer the tally loser. Their signature is exactly what the residuals want: on Exps 5/6 they follow SCWI strongly (lifting the undershooting means toward 0.684/0.630); on Exp 2 they prefer the fewer-endorsement option, which coincides with TTB's pick on a substantial share of disagreement trials (lifting the chronic 0.19-vs-0.274 undershoot and its 0.014-vs-0.093 variance, the most stubborn cell in the loop); on Exp 1/3 they flatten the slightly-too-steep slope/composite toward -0.175/0.123; and they sit at sL ~ sM ~ 1 on Exp 9's kill cells, adding between-subject variance on every composite. (3) COMPENSATIONS AND GUARDRAILS: flat subjects RAISE Exp 8's already-overshooting composite (they prefer the high-validity stack) and raise Exp 9's r (they are consistent across ladders). For Exp 8, raise diffusion from 0.08 to ~0.10-0.11 — it softens Exp 8's 9-14-cue kill cells hardest — but verify Exp 10's rise does not fall below ~0.25 (it is currently at target; the twin cells are 2:1 more numerous than the high-ratio cells, so flat mass plus diffusion both push the rise DOWN — if Exp 10 undershoots, cap diffusion at 0.08 and accept a residual +0.04 on Exp 8). For Exp 9, if r exceeds 0.25, trim eta_base sd (0.09 -> 0.07) rather than touching the flat subpopulation. Watch Exp 7 (already +0.019; flat mass pushes it up — verify it stays <= 0.68) and Exp 3 (<= 0.14). (4) METHODOLOGICAL, NOW MANDATORY: before committing, numerically simulate the FULL pipeline (attention noise, lapse, guessers, anchor/control screens, per-bucket hashing) on the actual cells of Exps 2, 5, 8, 9, 10 and report the predicted composites. The proposer's iter-7 margin arithmetic predicted Exp 2 'flat-to-slightly-up' (it fell 0.02), Exp 8 '~0.62' (delivered 0.666), and 'Exp 9 r stays in [0.15, 0.25]' (delivered -0.02) — three for three wrong, all in the optimistic direction. No analytic knife-count argument should appear in the next rationale. All changes are parameter/shape recalibrations strictly within the prescribed CNNAI mechanism (additive anti-validity integration, bucketed-mean context key, mean-referenced v_ref, softmax + lapse, guesser mixture); no family switch.

**Outcome of your advice:** iter 7 candidate loss=0.0786 -> iter 8 candidate loss=0.0819 -> the gate marked it REJECTED.

### Iteration 8 (most recent)
Verdict: regenerate
Interpretation: The gate rejected iter 8 (0.0645 -> 0.0819), and the per-experiment decomposition shows the flat-weight subpopulation move split into two real successes, two clear failures, and one diagnostic confound resolved. WHAT WORKED: Exps 4 (0.138 -> 0.170 vs real 0.163) and 6 (0.620 -> 0.630 vs 0.630) are now essentially on target — the flat subpopulation is what lifted them (iter 7, which had diffusion 0.08 but no flat mass, sat at 0.144/0.598). Exp 8 improved to 0.657 (still +0.05) and Exp 10 to 0.318 (+0.02). WHAT FAILED: (1) EXP 9 COLLAPSED (r = -0.137 vs real 0.196; iter 6 had 0.184). Crucially, iter 8 disentangles the confound I mis-attributed in iter 7: the dispersion was reverted to iter-6 values, yet Exp 9 broke again — so the iter-7 collapse was NOT primarily the dispersion widening. The common factors in iters 7/8 but not iter 6 are diffusion 0.08 and the eta_base trim. My armchair analysis of the control screen shows the flat subjects CANNOT be the cause: at eta in [-0.55, -0.40], the L-control knife sits at eta = -0.263 and the M-control knife at eta = -0.437, so flat subjects choose the top stack on the controls, fail the cH <= 0.375 keep filter, and are excluded from Exp 9's correlation entirely. The proposer's claim that flat subjects 'sit at sL ~ sM ~ 1 on the kill cells and add spread on every diagnostic composite' on Exp 9 is wrong for this reason — and the preemptive eta_base trim (0.09 -> 0.07) that was justified by that (inert) subpopulation removed real stable coupling for nothing. The remaining suspect for the sign flip is diffusion 0.08 softening the 11-15-endorsement kill cells toward 0.5 while the interleaved ladder knives (0.50-ladder at ~0.42-1.0, 0.60-ladder at ~0.36-0.71) generate a small anti-correlated band. (2) EXP 2 FELL TO ITS WORST VALUE OF THE LOOP (0.139 vs real 0.274; iter 6 was 0.189) — the flat subpopulation was predicted to lift it and did the opposite; fourth consecutive iteration in which the proposer's analytic cell arithmetic was wrong in the optimistic direction, despite my iter-7 mandate for full-pipeline numerical simulation, which was again replaced by 'back-of-envelope mixture arithmetic'. (3) THE VARIANCE CHANNEL FAILED: between-subject variance remains 2-5x under on every experiment (Exp 2: 0.017 vs 0.093; Exp 5: 0.030 vs 0.064; Exp 6: 0.019 vs 0.070; Exp 8: 0.025 vs 0.065; Exp 9: 0.020 vs 0.060; Exp 10: 0.041 vs 0.081), and Exp 4's variance now OVERSHOOTS (0.024 vs 0.012). An 8% discrete subpopulation is arithmetically incapable of delivering sd ~0.26 composites: a two-point mixture at 0.9/0.6 contributes ~0.006 of variance, an order of magnitude short. (4) Exp 7's overshoot grew (0.683 vs 0.646, guardrail was <= 0.68).
Rationale: Regenerate, building on the UNCHANGED iter-6 base (the gate already reverted). The loop has now lost two iterations to confounded multi-knob moves — mandate ONE disentangled change set this time, verified numerically. (1) RESTORE eta_base sd TO 0.09 (undo the trim). The trim was justified by a subpopulation that is provably inert on Exp 9 (screened out by the cH control), so it only removed the stable coupling that produced iter-6's excellent r = 0.184. Keep redraw sd 0.19, center 0.02, right clip 0.70; left clip can stay -0.30. (2) DROP DIFFUSION TO ~0.06 (from 0.08). Iter 8's disentanglement shows diffusion 0.08 is the prime remaining suspect for the Exp 9 collapse (present in both broken iters 7/8, absent in the healthy iter 6), while the Exp 8/10 softening it buys is roughly half-achieved by 0.06 given the flat mass also pushing those cells the right way. Accept a residual +0.05 on Exp 8 rather than sacrificing Exp 9's r — the loss is dominated by the headline correlation statistic. (3) KEEP THE FLAT SUBPOPULATION BUT RE-TUNE IT: it is the only mechanism in the loop that has ever moved Exps 4/6 onto target, so retain ~6-8% of engaged at stable eta, but shift the range shallower ([-0.45, -0.30]) to reduce the damage to Exp 2 (where it pushed the composite DOWN to a loop-worst 0.139) and the Exp 7 overshoot (0.683, guardrail breached). Verify numerically that Exps 4/6 stay within 0.02. (4) STOP TRYING TO FIX VARIANCE WITH THIS CHANNEL: a modest discrete subpopulation cannot produce sd ~0.26 composites, and the symmetric-dispersion route is capped by Exp 9 (iter-7 lesson). If any further variance is attempted, the only untested in-family knob is moderate per-subject beta heterogeneity (lognormal sd ~0.30 — NOT the 0.45 that broke iter 1), which spreads subjects along the softmax steepness axis on every composite simultaneously; test it as a SEPARATE small increment only if the mean profile first beats 0.0645. (5) METHODOLOGICAL, NOW A HARD REQUIREMENT: the iter-8 rationale once again substituted back-of-envelope mixture arithmetic for simulation and was wrong in sign on Exp 9 (predicted ~0.2, delivered -0.14) and wrong on Exp 2 (predicted lift, delivered a drop) — the control-screen exclusion of flat subjects from Exp 9 is exactly the kind of pipeline detail only a full simulation catches. Before submitting, simulate the complete pipeline (attention noise, lapse, guessers, per-bucket hashing, and Exp 9's anchor AND control keep-filters) on the actual cells of Exps 2, 5, 8, 9, 10 and report predicted values; a rationale containing analytic knife-count arguments should be treated as unverified. All changes are parameter/shape recalibrations strictly within the prescribed CNNAI mechanism (additive anti-validity integration, bucketed-mean context key, mean-referenced v_ref, softmax + lapse, guesser mixture); no family switch.

**Outcome of your advice:** iter 8 candidate loss=0.0819 -> CURRENT candidate loss=0.1539 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The gate rejected iter 9 (0.0645 -> 0.1539), and the failure is not a tuning failure \u2014 it is a silent implementation bug that disables the theory's defining mechanism. The proposer claimed the Exp-9 pipeline is 'byte-identical to the accepted iter-6 base except diffusion 0.05 -> 0.06' and predicted r ~ 0.17-0.19; the simulation delivered r = 0.771 (real 0.196), the largest single error of the entire loop and the signature of a stable-trait theory (compare pi_5: 0.953, pi_1: 0.905, pi_2: 0.874). The smoking gun is in the context-hash line: `hashlib.md5(f\"ss:.9f|key\".format(ss=ss, key=key)...)`. That string is an f-string with NO interpolation braces, so it evaluates to the literal 'ss:.9f|key'; the subsequent `.format(ss=ss, key=key)` has no replacement fields to fill and does nothing. The hash is therefore a CONSTANT \u2014 independent of both the subject seed and the bucket key. Consequently uc is one fixed number for every subject in every context, and eta_ctx = eta_base + 0.10 + 0.19*ndtri(uc_const) has NO per-context redraw and NO per-subject redraw: the 'context-normalized, redrawn-per-context elasticity' \u2014 the exact feature the arbiter prescribed to explain cross-ladder r ~ 0.196 \u2014 is functionally switched off, collapsing CNNAI back into the stable-trait HEAI-like model the arbiter explicitly rejected. The empirical fingerprint confirms this diagnosis on three independent axes: (1) Exp 9's r exploded to 0.771 (fully stable per-subject eta -> cross-ladder consistency), exactly the iter-1 failure mode I attributed to stable traits; (2) between-subject variance collapsed UNIFORMLY relative to iter 6 on every experiment (Exp 9 var 0.0245 -> 0.0097, Exp 7 0.0221 -> 0.0117, Exp 8 0.0236 -> 0.0178, Exp 6 0.0138 -> 0.0127) \u2014 impossible if the sd-0.19 redraw were still contributing variance, and exactly what a constant offset produces; (3) every composite shifted SCWI-ward (flatter effective eta): Exp 8 0.692 -> 0.754 (real 0.604, worst ever), Exp 7 0.665 -> 0.737 (real 0.646, worst ever, guardrail breached), Exp 6 0.620 -> 0.682, Exp 2 0.189 -> 0.155, Exp 1 slope -0.201 -> -0.185 \u2014 all consistent with the constant ndtri(uc) offset landing negative and pulling the whole engaged eta distribution down and tightening it. Because the bug confounds everything, the iter-8 prescription (flat subpopulation at [-0.45,-0.30], diffusion 0.06, eta_base sd 0.09) remains genuinely untested \u2014 the flat-subpopulation 'provable inertness' on Exp 9 cannot be evaluated while the redraw is dead, though I note the control-cell margins for flat subjects (~0.1-0.15 score units at beta ~2.1) are small enough that attention noise and lapse will let a nontrivial fraction pass the cH <= 0.375 filter, so the inertness claim still needs numerical verification once the bug is fixed.",
  "verdict": "regenerate",
  "rationale": "Regenerate, building on the UNCHANGED iter-6 accepted base (the gate already reverted). This is primarily a BUG FIX, not a re-design, and the mechanism family is untouched. (1) FIX THE CONTEXT HASH: the seed string must actually interpolate both the subject seed and the bucket key, e.g. `f\"{ss:.9f}|{key}\"` or `\"ss:{ss:.9f}|{key}\".format(ss=ss, key=key)`. Add a direct unit check that uc takes distinct values across distinct (subject_seed, bucket) pairs before submitting \u2014 this is the second silent pipeline failure in three iterations and the loop cannot afford a third. (2) RESUBMIT THE SAME DISENTANGLED EDIT on top of the fixed hash: flat subpopulation ~7% of engaged at stable eta in [-0.45, -0.30], diffusion 0.06, eta_base sd 0.09, all other iter-6 constants unchanged. Since the iter-8/9 prescriptions were never actually tested (every iter-9 result is confounded by the dead redraw), do NOT change direction \u2014 the disentangled knob set from my iter-8 advice is still the right experiment to run. (3) NUMERICALLY VERIFY THE EXP-9 INVARIANCE CLAIM instead of proving it analytically: simulate the full Exp-9 pipeline (anchor filter, cH control filter, per-bucket hashing, attention noise, lapse, guessers) and report (a) the fraction of flat subjects surviving the cH <= 0.375 screen \u2014 the L-control margin for eta in [-0.45,-0.30] is only ~0.1 score units, so with beta ~2.1, diffusion, and lapse, a meaningful share (roughly a quarter) will pass and then sit at (sL, sM) ~ (1, 1), inflating r; and (b) the resulting pooled r with target [0.15, 0.25]. If surviving flat subjects push r above 0.25, shrink the flat fraction (7% -> 4-5%) or shift the range just deep enough that the control pass-rate falls, rather than deepening it to the iter-8 [-0.55, -0.40] range that damaged Exp 2. (4) MANDATORY FULL-PREDSUBMISSION SIMULATION, now with teeth: the iter-9 rationale's central guarantee ('r should recover to ~0.17-0.19') was wrong by +0.60, and the loop's analytic knife-count arguments are now zero-for-five in the optimistic direction. The next rationale must report simulated values for Exps 2, 5, 8, 9, 10 from the actual pipeline, including the hash-variability check; a candidate whose rationale contains only arithmetic guarantees should be treated as unverified. All changes remain strictly within the prescribed CNNAI mechanism (additive anti-validity integration, bucketed-mean context key with a functioning per-context redraw, mean-referenced v_ref, softmax + lapse, guesser mixture); no family switch."
}
```

## Usage

```json
{
  "input_tokens": 56220,
  "output_tokens": 11381,
  "reasoning_tokens": 9901,
  "provider": "Sail Research"
}
```
