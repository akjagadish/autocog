# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended decision-making theories make distinguishable predictions: avoid degenerate pairs where every theory agrees, and prefer pairs that maximally dissociate the candidate theories. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
**Description:** Signed-Cue Weighted Integration (SCWI). When choosing between two options described by binary expert ratings, subjects do not use a lexicographic cascade (TTB) and do not merely count wins (Tallying). Instead they integrate ALL cues additively into a single subjective value per option — but the mapping from a rating to evidence carries a systematic SIGN INVERSION: a rating of 1 is treated as a defect/criticism rather than an endorsement (a comprehension inversion of the rating direction). Moreover, the inversion is not uniform across cues: the subjective weight of cue j is w_j = -(v_j + tau * v_j^2), i.e. an anti-validity weighting that grows SUPRALINEARLY with the instructed validity — the endorsements of the most trusted experts are avoided most strongly, while low-validity experts are nearly ignored (their anti-weight is small in magnitude). The option with the higher (less negative) signed value is preferred, with softmax noise (inverse temperature beta) and an independent lapse (epsilon) capturing response variability. This single task-invariant mechanism predicts both experimental signatures simultaneously: in Experiment 1, the top-cue-favored option carries more 1s and hence more 'defects', so P(choose the top-cue-favored option) DECREASES with its tally margin (negative slope); in Experiment 2, on margin-1 trials where TTB and the tally disagree the two options' signed values are close together (so noise frequently flips the subject onto the tally winner), whereas when they agree the value gap is large (so the subject reliably anti-follows) — yielding p_disagree > p_agree, a positive metric.

**Parameters:**
- beta_seed: [0, 1]
- tau: [0.15, 0.42]
- attention: [(0.90, 1.10)] * n_features
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Signed-Cue Weighted Integration (SCWI) -- corrected three-regime
    # population, iteration 7 (minimal-diff on the accepted iter-6 base).
    # Mechanism is UNCHANGED: subjective validity of cue j = a_j * v_j
    # (per-cue attention factor, fixed for the whole run), signed cue
    # weight w_j = -(v'_j + tau * v'_j^2) (a rating of 1 is a defect; the
    # anti-weight grows SUPRALINEARLY in the subjective instructed
    # validity), softmax over beta * values plus a symmetric lapse
    # epsilon. History is ignored (no trial-by-trial feedback).
    #
    # Population edits on top of the accepted iter-6 base (mechanism
    # untouched), per the iter-6 critic diagnosis that the remaining
    # gaps are pure mixture-calibration problems:
    #   * FRACTIONS 13/57/30 -> 16/50/34: the mid regime (57%) was the
    #     underperformer on Exp 5 and contributes almost nothing on the
    #     Exp-4 Probe-B conflict cells, so its mass is redistributed to
    #     BOTH tails -- the guesser tail lifts Exp 4 (guessers contribute
    #     0.5 on conflict cells vs mid's ~0.10) and the deterministic
    #     tail lifts Exp 5/6 (det ~0.9 vs mid's ~0.53).
    #   * MID BETA raised a small step, 1.6-2.2 -> 1.8-2.4 (NOT a swing
    #     back to iter-5's 1.8-2.8, which failed without real guessers;
    #     with 16% pinned at 0.5 the Exp-4 floor is higher).
    #   * WITHIN-REGIME SPREAD WIDENED for between-subject variance:
    #     deterministic beta 4.2-5.6 -> 3.8-6.0, mid epsilon 0.06-0.10
    #     -> 0.05-0.11, guesser epsilon 0.15-0.25 -> 0.12-0.28. These
    #     widen the per-subject hit-rate distribution without materially
    #     moving the population means.
    #   * Attention jitter stays exactly at the validated (0.90, 1.10);
    #     tau stays at [0.15, 0.42]; the beta~0 guesser implementation
    #     (the validated decoupling lever) is untouched.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SCWI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

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

    att = _to_vec(parameters.get("attention"))
    if att is None or att.size != n_features:
        # Defensive fallback: no attention distortion.
        att = np.ones(n_features)
    att = np.clip(att, 0.6, 1.4)

    # --- Population mixture over (beta, epsilon) from a single uniform
    # seed. Three regimes: TRUE guessers (beta ~ 0 -> uniform core on
    # every trial), a mid anti-validity regime, and a deterministic
    # anti-validity regime. Iter-7 fractions: 16% / 50% / 34%.
    try:
        u = float(parameters["beta_seed"])
    except (TypeError, ValueError):
        u = 0.5
    u = min(max(u, 0.0), 1.0)
    if u < 0.16:
        # True guessing subpopulation (~16%): beta ~ 0 makes the
        # softmax core essentially uniform on every trial, so these
        # subjects sit at ~0.5 on every diagnostic cell.
        t = u / 0.16
        beta = 0.02 + 0.08 * t         # 0.02 .. 0.10
        epsilon = 0.12 + 0.16 * t      # 0.12 .. 0.28
    elif u < 0.66:
        # Mid subpopulation (~50%): moderate beta (small step up from
        # the iter-6 1.6-2.2 to target the Exp-5 cells where mid
        # subjects sit too close to chance), low lapse.
        t = (u - 0.16) / 0.50
        beta = 1.8 + 0.6 * t           # 1.8 .. 2.4
        epsilon = 0.11 - 0.06 * t      # 0.11 .. 0.05
    else:
        # Deterministic subpopulation (~34%): high beta with widened
        # within-regime spread, minimal lapse. Fraction (not beta
        # ceiling) guards Exp 6.
        t = (u - 0.66) / 0.34
        beta = 3.8 + 2.2 * t           # 3.8 .. 6.0
        epsilon = 0.04 - 0.02 * t      # 0.04 .. 0.02

    tau = float(parameters["tau"])

    # Subjective validities (instructed validity x per-cue attention),
    # then anti-validity weights: the anti-weight of expert j is
    # v'_j + tau * v'_j^2, supralinear in the subjective validity, so
    # the endorsements of the most trusted experts are avoided most
    # strongly while low-validity experts are nearly ignored.
    v_eff = val * att
    w = -(v_eff + tau * np.square(v_eff))

    scores = np.array([float(np.dot(w, stim[0])), float(np.dot(w, stim[1]))])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Heterogeneous-Elasticity Anti-Validity Integration (HEAI), bimodal-population refinement. Subjects integrate ALL expert ratings additively into a per-option subjective value, but a rating of 1 is comprehended as a DEFECT (evidence against the option) — a sign inversion of the rating direction. The anti-weight of expert j is a per-subject power law of the instructed validity, w_j = -v_j^gamma, where gamma ('validity elasticity') is drawn from a BIMODAL continuous population: a shallow majority (75% of engaged subjects, gamma ~ N(1.04, 0.22) truncated [0.60, 1.42]) that sits cleanly below the kill-design count-ratio thresholds (~1.31-1.34) and therefore follows the high-validity stack, and a steep minority (25% of engaged subjects, gamma ~ N(2.10, 0.26) truncated [1.65, 2.75]) that sits cleanly above them, whose near-lexicographic top-cue dominance drives the Exp-2 dissociation and keeps the Exp-8/Exp-4 composites from overshooting. Inverse temperature is mode-linked (shallow: lognormal median ~2.6; steep: lognormal median ~3.3; both heavy-tailed), a 16% guesser subpopulation (beta ~ 0) and a symmetric lapse epsilon are retained unchanged. The population mean elasticity (~1.30) stays below the arbiter's 1.4 ceiling. The key change from the falsified wide-unimodal shape is that almost no mass is wasted in the ambiguous 1.2-1.6 band where subjects sit at chance on the diagnostic cells: the shallow mode's thin upper shoulder (1.32-1.42) is retained deliberately, because subjects just above the kill thresholds miss Exp 8 while still hitting the higher-threshold Exp 5/7 ladder cells — the lever that lets Exp 8's mean and variance stay calibrated without capping Exp 5/6/7.

**Parameters:**
- regime_seed: [0, 1]
- gamma_seed: [0, 1]
- beta_seed: [0, 1]
- epsilon_seed: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Heterogeneous-Elasticity Anti-Validity Integration (HEAI), iter 3.
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Mechanism (UNCHANGED from the accepted iter-2 base): every cue
    # contributes signed evidence w_j = -v_j**gamma per unit rating
    # (a rating of 1 is a defect; the anti-weight is a per-subject
    # power law of the instructed validity). Choice is a numerically
    # stable softmax over beta * option values, mixed with a symmetric
    # lapse epsilon. History is ignored (no trial-by-trial feedback).
    #
    # Population edits on top of the accepted iter-2 base (mechanism
    # untouched), implementing the iter-2 critic's diagnosis that a
    # wide UNIMODAL gamma dumps its spread into the ambiguous
    # threshold band (gamma ~ 1.2-1.6) where subjects sit at ~chance
    # on every diagnostic cell, capping the means on BOTH sides:
    #   * GAMMA MADE BIMODAL: the engaged population is now a mixture
    #     of a SHALLOW majority (75%: N(1.04, 0.22) trunc [0.60, 1.42])
    #     and a STEEP minority (25%: N(2.10, 0.26) trunc [1.65, 2.75]),
    #     with minimal mass in between. The shallow mode sits cleanly
    #     below the kill-design count-ratio thresholds (~1.31-1.34 on
    #     Exp 8; ~1.32+ on the Exp 5 ladder), so it drives high-stack
    #     adherence on Exp 5/6/7/8; its thin upper shoulder (1.32-1.42,
    #     ~10% of the mode) deliberately remains to keep Exp 8's mean
    #     and between-subject variance calibrated. The steep mode sits
    #     cleanly above ~1.8 (critic's prescription; extends to 2.75 so
    #     the top cue can dominate three lower cues), which is what
    #     lifts the Exp 2 dissociation mean/variance and holds Exp 4
    #     down. Population mean elasticity ~ 1.30 < 1.4 (arbiter box).
    #   * BETA MODE-LINKED AND HOTTER: shallow subjects draw lognormal
    #     beta, median exp(0.95) ~ 2.6 (up from ~2.3), sigma 0.70,
    #     clip [0.6, 7.0]; steep subjects draw a hotter lognormal,
    #     median exp(1.20) ~ 3.3, sigma 0.60, clip [1.2, 8.0] (the
    #     steep 'confident lexicographic' subjects are also the
    #     sharpest). Hotter shallow beta sharpens engaged adherence on
    #     Exp 5/6/7; hotter steep beta maximizes the Exp 2 contribution
    #     per steep subject while pushing steep Exp 8/Exp 4 hits down.
    #   * Guesser fraction (16%), epsilon ranges, seeds, and everything
    #     else are untouched.
    import math
    import numpy as np
    from scipy.special import ndtri

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"HEAI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

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
    # Keep CDF seeds strictly inside (0, 1) so ndtri stays finite.
    gs = min(max(_seed("gamma_seed"), 1e-6), 1.0 - 1e-6)
    bs = min(max(_seed("beta_seed"), 1e-6), 1.0 - 1e-6)
    es = _seed("epsilon_seed")

    if u < 0.16:
        # Guesser subpopulation (~16%): beta ~ 0 makes the softmax core
        # essentially uniform on every trial; lapse is elevated.
        t = u / 0.16
        beta = 0.02 + 0.08 * t          # 0.02 .. 0.10
        epsilon = 0.12 + 0.13 * es      # 0.12 .. 0.25
        gamma = 1.12                    # irrelevant at beta ~ 0
    else:
        if gs < 0.75:
            # SHALLOW majority (~75% of engaged): gamma ~ N(1.04, 0.22)
            # truncated to [0.60, 1.42]. Cleanly below the kill-design
            # thresholds (~1.31-1.34), with a thin upper shoulder that
            # keeps Exp 8's mean/variance calibrated.
            t = gs / 0.75
            tt = min(max(t, 1e-4), 1.0 - 1e-4)
            gamma = 1.04 + 0.22 * float(ndtri(tt))
            gamma = min(max(gamma, 0.60), 1.42)

            # Hot shallow beta: lognormal median ~2.6, heavy tail.
            z_b = float(ndtri(bs))
            beta = math.exp(0.95 + 0.70 * z_b)
            beta = min(max(beta, 0.6), 7.0)
        else:
            # STEEP minority (~25% of engaged): gamma ~ N(2.10, 0.26)
            # truncated to [1.65, 2.75]. Cleanly above ~1.8 so the top
            # cue can dominate three lower cues (the Exp 2 lever).
            t = (gs - 0.75) / 0.25
            tt = min(max(t, 1e-4), 1.0 - 1e-4)
            gamma = 2.10 + 0.26 * float(ndtri(tt))
            gamma = min(max(gamma, 1.65), 2.75)

            # Steep subjects are also sharper: lognormal median ~3.3.
            z_b = float(ndtri(bs))
            beta = math.exp(1.20 + 0.60 * z_b)
            beta = min(max(beta, 1.2), 8.0)

        # Symmetric lapse for engaged subjects (unchanged).
        epsilon = 0.03 + 0.14 * es      # 0.03 .. 0.17

    # Anti-validity power-law cue weights: a rating of 1 counts AGAINST
    # the option, with magnitude v_j**gamma.
    w = -np.power(val, gamma)

    scores = np.array([float(np.dot(w, stim[0])), float(np.dot(w, stim[1]))])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
    return np.random.choice(len(probabilities), p=probabilities)
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] This design exploits the core structural difference between Take The Best (TTB) and Tallying: TTB's choice depends ONLY on the single highest-validity discriminating cue, while Tallying depends on the total count of cue wins. I fix 5 experts with well-spread validities [0.95, 0.85, 0.75, 0.65, 0.55] and construct a family of critical trials where the top cue favors A but B wins on more (or all) of the remaining cues, with the tally margin parametrically varied. KEY QUANTITATIVE DISSOCIATION: across the four trial types with tally margins of B+4, B+3, B+2, and exact tie (pairs 1, 3, 5, 7), TTB predicts an IDENTICAL probability of choosing A on every one of them — P(A) = sigmoid(beta)*(1-epsilon) + epsilon/2, completely flat as a function of tally margin, because in all four cases the top cue (v=0.95) discriminates in favor of A. Tallying instead predicts a strictly graded, monotonically decreasing P(A) as the tally margin against A grows: 1/(1+e^{3*beta}), 1/(1+e^{2*beta}), 1/(1+e^{beta}), and exactly 0.5 respectively. So the signature is FLAT-vs-GRADED psychometric slope over tally margin at fixed TTB evidence. A second, orthogonal quantitative probe: pairs 11, 13, 15 hold the tally structure roughly constant but shift WHICH cue is the top discriminating one (cue 2 at v=0.85, cue 3 at v=0.75). TTB predicts the same confident A-choice regardless of which cue discriminates (its score is always 1 vs 0), whereas any validity-weighting or evidence-accumulation account would predict graded confidence. Pair 13 is especially diagnostic: the second cue favors A, all others tie, so TTB predicts a confident A choice while Tallying predicts exact 50/50 guessing. Pairs 9/10 are agreement controls (both theories predict the same confident choice) to calibrate shared noise parameters; pairs 17/18 are full-tie trials where both theories must guess, providing a pure lapse-rate baseline. All critical pairs are mirrored (A/B swapped) to absorb side/response biases. With 18 unique pairs, K = 96//18 = 5 repetitions yields 90 trials per subject. Because there is no feedback, subjective cue order stays anchored to the instructed validities, keeping TTB's cue cascade well-defined. If subjects' P(A) is flat across the margin series, TTB is supported; if it declines systematically with the tally margin (approaching 50% at tie), Tallying is supported; a mixed pattern (e.g., partial decline) would favor neither in its pure form.

[1] This design inverts the logic of the classic TTB-vs-Tallying conflict: instead of holding the top cue fixed and varying the tally margin (already explored), I hold the TALLY MARGIN fixed and parametrically vary WHICH cues the losing-side wins on. I use 5 experts with strictly ordered validities [0.90, 0.80, 0.70, 0.60, 0.55] so TTB's cascade order is unambiguous. CORE DISSOCIATION (pairs 1-7): every one of these pairs has the identical tally structure — A wins exactly 3 cues, B wins exactly 2 (margin +1 for A) — so Tallying predicts the exact same P(A) = sigmoid(beta) on all seven trials, a perfectly FLAT psychometric function over the series. TTB instead predicts a STEP FUNCTION: on pairs 1-3 the top cue (v=0.90) is one of B's wins, so TTB confidently picks B (P(A) = 1 - sigmoid(beta)); on pairs 4-7 the top cue favors A, so TTB confidently picks A (P(A) = sigmoid(beta)). The predicted discontinuity sits exactly between pair 3 (B wins cues {1,5}) and pair 4 (B wins cues {2,3}) — TTB specifies the step's location precisely, while Tallying specifies zero slope. This is a double dissociation in slope: the previously explored design had TTB flat / Tallying graded; this one has Tallying flat / TTB discontinuous, so the two designs jointly triangulate both theories. Pairs 8-14 are exact A/B mirrors of 1-7 to absorb side/response bias (Tallying: flat sigma(-beta); TTB: mirrored step). SECOND PROBE — cascade depth at graded tally (pairs 15-17): the top k cues are forced to tie so TTB's deciding cue moves down the validity ladder (cue 2, then cue 3, then cue 4) while it always picks B with the SAME confidence 1 - sigmoid(beta), because TTB's binary score is validity-invariant. Tallying, in contrast, predicts a strictly graded decline in P(A) — sigma(2*beta), sigma(beta), and exactly 0.5 — because the tally margin shrinks from +2 to +1 to 0 as ties remove A's winning cues. Pair 17 is the sharpest single trial: Tallying predicts exactly 50/50 guessing (tally 1-1), TTB predicts a confident B choice from cue 4 alone. Pairs 18-20 mirror these. CONTROLS: pair 21 (A wins all cues) is an agreement trial calibrating the shared sigmoid(beta) confidence asymptote; pair 22 (identical vectors) is a full-tie trial where both theories must guess, giving a pure lapse-rate baseline; pairs 23-24 are extreme single-cue-vs-all-others conflicts (top cue alone vs. four opposing cues) providing maximum-separation anchor points where the theories disagree most strongly in magnitude (Tallying: sigma(-3*beta) vs TTB: sigmoid(beta)). With 24 unique pairs, K = 96 // 24 = 4 repetitions yields exactly 96 trials. No feedback keeps subjective cue order anchored to instructed validities. DECISION RULE: if P(A) across pairs 1-7 is flat at the level fit on margin+1 trials, Tallying is supported; if it shows a sharp step exactly between pairs 3 and 4, TTB is supported; a graded monotone drift favoring neither step nor flatness would falsify both pure forms.

[2] ADVERSARIAL LOGIC. The advocated theory (SCWI) and the competitor (Tallying) have a deep structural relationship on binary cues: because SCWI's signed value difference between options is sum_j w_j*(a_j - b_j) with w_j = -(v_j + tau*v_j^2), tied cues cancel exactly, and SCWI effectively computes an ANTI-VALIDITY-WEIGHTED tally: it prefers the option that LOSES on more (and more heavily weighted) cues. Tallying prefers the option that WINS on more cues, ignoring validity entirely. Two things follow. (1) SIGN: on every trial with a nonzero tally margin the two theories prefer OPPOSITE options. (2) SLOPE: Tallying's evidence is the raw count margin; SCWI's evidence is the validity-weighted (inverted) margin, so the two theories order trials differently. The design exploits both, with a parameter-free kill shot: on any trial where the feature-wise counts are TIED (a_wins = b_wins), Tallying's softmax is exactly uniform for ANY beta and ANY epsilon (lapse is symmetric), pinning P(A) = 0.500 exactly. SCWI is not so pinned: a count tie with unequal validity composition yields a nonzero weighted margin. PROBE A (pairs 1-8, count ties): I construct 8 count-tie trials whose SCWI weighted margin M = sum|w| over B's winning cues minus sum|w| over A's winning cues sweeps from -0.92 to +0.92 (computed at tau=0.35; the sweep is robust across tau in [0.25,0.45]). Tallying predicts EXACTLY 0.500 on all eight, a horizontal line with zero free parameters. SCWI predicts a monotone psychometric sweep of P(A) from ~0.25 up to ~0.75, with the specific ordering implied by the supralinear anti-validity weights (e.g., pair 3, where A wins the two top experts and B wins the two worst, yields P(A)~0.25; pair 4, the cue-label swap, yields ~0.75). Any reliable deviation from 0.5 on these trials falsifies Tallying outright; the graded M-ordering then tests SCWI quantitatively. PROBE B (pairs 9-15, tally margin fixed at +1 for A: A wins exactly 3 cues, B wins 2): Tallying predicts a FLAT P(A) = sigmoid(beta)*(1-eps)+eps/2 on all seven, while SCWI predicts a strictly graded function of which cues A wins — from P(A)~0.11 when A wins the three most trusted experts (its 'defect load' is maximal) to P(A)~0.49 when A wins the three least trusted. This is a flat-vs-graded slope dissociation at fixed tally evidence, and every SCWI prediction sits BELOW 0.5 while Tallying's sits ABOVE 0.5: a double (sign + slope) dissociation. PROBE D (pairs 16-19) mirrors four Probe-B trials (A/B swapped) to absorb side and response bias; Tallying stays flat at sigmoid(-beta), SCWI stays graded above 0.5. ANCHORS (pairs 20-23): extreme trials where A wins all cues (pair 20), B wins all (21), or A wins four low-validity cues against B's single top cue (22/23). Tallying approaches deterministic choice of the tally winner (sigmoid(5*beta), sigmoid(3*beta)); SCWI approaches deterministic choice of the tally LOSER, with confidence tracking the weighted margin M rather than the count margin — plotting P(A) against count margin gives a rising function under Tallying and a falling function under SCWI. Pair 24 is a full tie (identical vectors) where both theories predict exactly 0.5, providing a pure lapse/side-bias baseline. With 24 unique pairs, K = 96//24 = 4 repetitions (96 trials). DECISION RULE: (i) if P(A) on pairs 1-8 is reliably different from 0.5 in either direction, Tallying is falsified (it cannot produce this for any parameter values); (ii) if P(A) on pairs 9-15 is flat, Tallying is supported and SCWI is falsified; (iii) if P(A) across pairs 1-15 is a monotone function of the anti-validity weighted margin M (increasing in M, crossing 0.5 near M=0, with the tally loser preferred), SCWI is supported; (iv) a preference for the tally WINNER that is nonetheless graded by validity weights would falsify both pure forms (it would indicate pro-validity weighted integration).

[3] ADVERSARIAL LOGIC (the 'transpose' of the count-tie trick). On binary cues, Tallying's evidence is the integer count margin m = a_wins - b_wins, while SCWI's evidence is the anti-validity weighted margin D = -sum_j (v_j + tau*v_j^2)*s_j (s_j = +1 if A wins cue j, -1 if B wins). The already-explored design pinned TALLYING at 0.5 (count ties) and let SCWI vary. I do the exact inverse: I engineer WEIGHTED TIES (D ~= 0, robustly across the full tau range [0.25, 0.45]) at NONZERO tally margins, so SCWI is pinned near 0.5 while Tallying is graded — and then hold D at a large fixed negative value while m varies, so SCWI is flat-low while Tallying rises. This yields a two-probe double slope dissociation in the opposite assignment from the prior design.

VALIDITY ENGINEERING. With validities [1.00, 0.95, 0.65, 0.54, 0.51, 0.50], a weighted tie between a cue set X and cue set Y requires sum_X v + tau*sum_X v^2 = sum_Y v + tau*sum_Y v^2. I choose sets where the constant and tau-coefficient nearly cancel simultaneously: e.g., the four low cues {3,4,5,6} (sum v = 2.20, sum v^2 = 1.224) vs the two top cues {1,2} (1.95, 1.9025): residual c(tau) = 0.25 - 0.678*tau, i.e., |c| <= 0.080 for ALL tau in [0.25, 0.45]. Similar cancellations: {4,5} vs {2} (|c| <= 0.058) and {3,6} vs {1} (|c| <= 0.068).

PROBE A (pairs 1-6, weighted ties at nonzero tally margin): pairs 1-2 have m = +1 and pair 3 has m = +2, all with |D| <= 0.08 across the entire SCWI parameter box. SCWI therefore predicts P(A) = 0.500 +/- 0.032 on ALL SIX (including mirrors) for ANY beta in [1.1,1.7], tau in [0.25,0.45], epsilon in [0.05,0.15] — a horizontal line with essentially no free parameters. Tallying predicts a graded ladder: P(A) = sigmoid(m*beta)*(1-eps)+eps/2, with lapse-corrected logits exactly proportional to m (the m=+1 and m=+2 trials must sit in 1:2 logit ratio, symmetric under mirroring). Any reliable, mirror-symmetric departure from 0.5 that grows with |m| falsifies SCWI outright; perfect flatness at 0.5 falsifies Tallying (which must exceed 0.5 at m>0 for any parameters).

PROBE B (pairs 7-12, fixed weighted evidence, varying tally margin): three trials with D pinned in a narrow confident band — pair 7: m=+1, D in [-1.30,-1.13]; pair 8: m=+2, D in [-1.18,-1.02]; pair 9: m=+3, D in [-1.30,-1.26]. SCWI predicts a FLAT-LOW P(A) ~ 0.15-0.26 on all three (mirrors ~0.74-0.85), because its evidence barely changes. Tallying predicts P(A) >= 0.5 always (m>0), strictly increasing in m: ~0.86 -> 0.96 -> 0.97 at beta=2. Huge vertical separation plus opposite slope sign. Note this is the inverse hold of the prior design's fixed-margin probe (there: m fixed, D varied; here: D fixed, m varies).

MINIMAL PAIR (pairs 3 vs 9): identical stimuli except cue 2 changes from a B-win to a tie. Tallying: m goes +2 -> +3, P(A) barely moves (0.96 -> 0.97). SCWI: D jumps from -0.01 to -1.28, P(A) collapses 0.50 -> 0.18. A single-cue manipulation that one theory treats as negligible and the other as decisive.

ANCHORS AND CONTROLS: pairs 13/14 (one option wins all six cues, m = +/-6, |D| ~ 5.0-5.6) are maximum-separation trials — Tallying ~0.97 vs SCWI ~epsilon/2 (~0.05). Pairs 15/16 are full ties (identical vectors, one all-0 and one mixed) where BOTH theories predict exactly 0.500, providing a pure lapse/side-bias baseline and verifying that tie values (0 vs 1) are inert. All critical pairs are mirrored to absorb side/response bias. 16 unique pairs, K = 96//16 = 6 repetitions = 96 trials.

DECISION RULES: (i) If P(A) on pairs 1-6 deviates from 0.5 symmetrically and grows with |m| (logit ratio ~1:2 for m=1 vs m=2), Tallying is supported and SCWI is falsified — SCWI cannot exceed 0.532 or dip below 0.468 there for any allowed parameters. (ii) If pairs 1-6 are flat at 0.5 while pairs 7-12 sit flat at ~0.2 (mirrors ~0.8) and pair 13 shows P(A) ~ 0.05 (anti-following the all-1s option), SCWI is supported and Tallying is falsified — Tallying cannot go below 0.5 - eps/2 at m>0. (iii) Flat 0.5 on Probe A combined with a Tallying-like increase on Probe B (or any preference for the tally winner graded by validity weights) falsifies both pure forms. Bonus: pair 9 also separates TTB (which picks B via cue 1) from Tallying, so a TTB-like subject is identifiable rather than confounded with SCWI.

[4] ADVERSARIAL LOGIC. SCWI and PAHI are structurally near-identical: both are anti-validity additive integrators (a rating of 1 counts AGAINST an option, tied cues cancel exactly, choice is a softmax plus symmetric lapse). The ONLY substantive difference is the validity-to-weight mapping: SCWI uses w(v) = v + tau*v^2 with tau in [0.25, 0.45]; PAHI uses w(v) = v^gamma with gamma in [1.4, 2.4]. Two mathematical facts drive the whole design. (1) UNIFORM STEEPNESS ORDERING: SCWI's log-elasticity is (1+2*tau*v)/(1+tau*v) <= 1.9/1.45 = 1.31 everywhere on v in [0.5, 1], while PAHI's elasticity is gamma >= 1.4. Therefore for ANY two validities, PAHI's relative weight w(v_hi)/w(v_lo) strictly exceeds SCWI's: PAHI is a uniformly steeper validity gradient. (2) LEVEL CROSSING: at v = 1.0 PAHI's weight is EXACTLY 1 (gamma-invariant) versus SCWI's 1.25-1.45, while at v = 0.5 SCWI's weight is 0.56-0.61 versus PAHI's 0.19-0.38. Consequence: on 'concentration' trials pitting a few high-validity endorsements against many low-validity endorsements, the preference flips at a knife-edge count ratio that differs between the theories, and integer ratios can be placed in the gap between the two knife-edges. THE KILL ZONE: with validities exactly 1.00 (two cues) and 0.50 (eight cues), let d = (number of 0.5-cue 1s on the low-cue option) - (number of 0.5-cue 1s on the top-cue option). SCWI prefers the top-cue option H iff d > 2(1+tau)/(0.5+0.25*tau), i.e., d in (4.44, 4.74) over the whole tau box -> flips strictly between d=4 and d=5. PAHI prefers H iff d > 2/0.5^gamma = 2^(gamma+1) in (5.28, 10.55) -> flips at d=6 or LATER, never before 5.28. So at d=5 the two theories make OPPOSITE predictions for every admissible parameter value: SCWI P(H) in [0.54, 0.62] (its evidence is 0.5-0.75*tau > 0 since tau < 2/3), PAHI P(H) in [0.04, 0.46] (its evidence is 5*0.5^gamma - 2 < 0 since gamma > log2(2.5) = 1.32). Note v = 1.0 exactly is essential: with v = 0.99, SCWI's flip condition tau < 0.389 fails inside its own box, destroying the parameter-free separation. PRIMARY INSTRUMENT (pairs 1-12): a d-sweep from 3 to 8, fully mirrored. SCWI predicts a psychometric curve crossing 0.5 strictly between d=4 (evidence = -tau < 0 always) and d=5, then saturating (P(H) ~ 0.67/0.77/0.84 at d=6/7/8; at d=8 its evidence is exactly +2.0 for ALL tau - a parameter-free anchor). PAHI predicts a crossing at d* = 2^(gamma+1) in (5, 10.6): at or beyond d=6 for gamma >= 1.59, and for gamma > 2 the curve never exceeds 0.5 within the observable range d <= 8. Three parameter-free statistics separate them: (i) the SIGN of P(H) at d=5 (16 trials per subject: pairs 3, 9, 13, 14 x 4 reps); (ii) the CROSSING LOCATION (SCWI: 4-5 for every subject, since the tau range is too narrow to move it; PAHI: >= 5.3, heterogeneous across subjects - SCWI predicts essentially zero between-subject variance in crossing point, PAHI predicts large variance); (iii) the saturation level at d=8 (SCWI >= 0.84; PAHI can sit below 0.5). SECONDARY KILLS at other validity mixtures, so the dissociation is not an artifact of the 1.0/0.5 pair alone: pair 15/16 (G): {1.0, 1.0, 0.85} vs seven 0.5-cues - SCWI margin 0.65-0.97*tau in [0.21, 0.41] favors the high option, PAHI margin 2+0.85^gamma-7*0.5^gamma in [0.14, 1.34] favors the low option; P(H) in [0.55, 0.66] vs [0.10, 0.46]. Pair 17/18 (S): two 0.85-cues vs four 0.5-cues - SCWI margin 0.3-0.445*tau > 0 always (tau < 0.674) favors the 0.85 option, PAHI margin 0.5^gamma*(1.7^gamma - 2) > 0 always (gamma > 1.305) favors the 0.5 option. Pair 19/20 (M2): two 1.0-cues vs {0.85, 0.85, 0.5} - SCWI margin 0.2-0.305*tau > 0 always (tau < 0.656), PAHI margin 2-2*0.85^gamma-0.5^gamma in [0.03, 0.48] > 0 always. GAMMA-LADDER (pairs 21/22 and d=6/7/8): pair W ({1.0, 0.85} vs five 0.5-cues) splits PAHI at gamma ~= 1.47; d=6 splits at 1.59; d=7 at 1.81; d=8 at 2.0. SCWI predicts uniformly high P(H) on all of W/d6/d7/d8 (>= 0.55/0.67/0.77/0.84). If PAHI is correct, the observed flips across this ladder must recover a single consistent gamma AND that gamma must place the d=5 choice below 0.5; if SCWI is correct, all ladder points sit high while d=5 sits above 0.5. ROBUSTNESS VARIANT (pairs 13/14): d=5 realized with an extra 1 on BOTH options (3 vs 6 total 1s instead of 2 vs 5) - identical difference, different totals; both theories predict behavior identical to pair 3, ruling out raw count-of-1s heuristics. CONTROLS: pair 23 (all-1 vs all-0) is a maximum-separation anti-validity anchor where both theories predict P(all-1 option) ~= epsilon/2, calibrating the lapse; pair 24 (identical mixed vectors) is a full tie where both predict exactly 0.500, calibrating side/response bias. All critical pairs are mirrored to absorb side bias. 24 unique pairs x K = 96//24 = 4 repetitions = 96 trials. DECISION RULES: (i) If the d=5 composite H-choice rate exceeds 0.5 (with d=4 below and d=6 above), SCWI is supported and PAHI is falsified - PAHI cannot exceed 0.463 there for any (beta, gamma, epsilon) in its box. (ii) If the d=5 rate is below 0.5, or the psychometric crossing sits at d >= 6 (or never), PAHI is supported and SCWI is falsified - SCWI cannot cross later than 4.74 low-cue units. (iii) If the crossing point varies widely across subjects (some never crossing by d=8), that specifically favors PAHI's heterogeneous gamma over SCWI's narrow tau. (iv) If the secondary kills G/S/M2 flip sign relative to the primary kill (e.g., high option preferred at 1.0/0.5 but not at 0.85/0.5), BOTH pure forms are falsified, since each theory's weight-shape prediction is sign-consistent across all these mixtures by construction.

[5] ADVERSARIAL LOGIC. PAHI (advocated) and SCWI (competitor) are structurally near-identical anti-validity additive integrators: a rating of 1 counts AGAINST an option, tied cues cancel, choice is a softmax over the anti-load difference plus a symmetric lapse. The ONLY substantive difference is the validity-to-weight map: PAHI uses w(v)=v^gamma with gamma in [1.4, 2.4]; SCWI uses w(v)=v+tau*v^2 with tau in [0.25, 0.45]. Three mathematical facts drive the design. (1) UNIFORM STEEPNESS ORDERING: SCWI's log-local elasticity is (1+2*tau*v)/(1+tau*v), which lies in [1.11, 1.31] for ALL v in [0.5, 1] and ALL tau in its box; PAHI's elasticity is exactly gamma >= 1.4. The elasticity ranges are DISJOINT, so for every v < 1, PAHI's top-to-low weight ratio w(1)/w(v) strictly exceeds SCWI's. (2) CROSSING-WINDOW DISJOINTNESS AT r=2: fix counts of 2 top cues (v=1.00, where PAHI's weight is exactly 1, gamma-invariant - a fixed 'currency') versus 4 low cues at validity v. The choice flips at v* solving 2*w(v) = w(1). SCWI: v* in [0.550, 0.576] across its ENTIRE parameter box - a 0.026-wide, effectively parameter-free window. PAHI: v* = 0.5^(1/gamma) in [0.610, 0.749]. The windows are disjoint, so each subject's measured flip validity classifies the theories with zero free parameters. (3) FLATNESS AT r=2.5: at counts 2:5, SCWI's flip validity v* in [0.449, 0.477] lies BELOW the floor 0.5, so SCWI must prefer the top-cue option for EVERY v in [0.5, 1] and every parameter - a parameter-free flat-H prediction. PAHI flips at 0.4^(1/gamma) in [0.520, 0.683] and must prefer the LOW option at v=0.50 for every gamma - parameter-free flat-L at the floor. Opposite pinned predictions. WHERE THIS SITS RELATIVE TO PRIOR DESIGNS: experiment [4] swept the COUNT RATIO at fixed validities (1.0 vs 0.5) and used a single knife-edge (d=5). This design is its TRANSPOSE: it fixes the count ratio and sweeps the LOW-CUE VALIDITY, making the statistic the CROSSING VALIDITY (its location, its between-subject spread, and its cross-ratio consistency), plus amplified stacked kills at validity mixtures (1.0/0.60/0.50 and 1.0/0.55) that [4] never touched. Only one cell (2x1.00 vs 5x0.50) coincides with [4]'s d=5 anchor, retained deliberately as a cross-experiment replication cell. INSTRUMENT: 28 experts - 4 at 1.00, 5 at 0.65, 5 at 0.60, 9 at 0.55, 5 at 0.50. Two ladders at r=2 (2:4) and r=2.5 (2:5) over v in {0.50, 0.55, 0.60, 0.65}, two stacked kills (4x1.00 vs 4x0.60+5x0.50; 4x1.00 vs 9x0.55, i.e. r=2.25 sitting inside the (2.11, 2.32) gap at v=0.55), all mirrored, plus tie / lapse / agreement controls. 24 unique pairs x K = 96//24 = 4 = 96 trials. KILL CELLS (parameter-free sign separation; SCWI evidence is linear in tau so endpoint checks suffice, PAHI evidence is monotone decreasing in gamma on every kill cell): R2-60 (2x1.00 vs 4x0.60): SCWI evidence +0.148..+0.26, P(top) in [0.535, 0.603]; PAHI evidence -0.044..-0.826, P(top) <= 0.487. This cell is the r=2 crossing detector: SCWI's crossing (<=0.576) is below 0.60, PAHI's (>=0.610) is above. R25-50: SCWI +0.163..+0.313, P >= 0.538; PAHI -0.106..-1.053, P <= 0.468. STK (4x1.00 vs 4x0.60+5x0.50): SCWI +0.311..+0.573, P in [0.572, 0.742]; PAHI -0.149..-1.879, P <= 0.456. STK2 (4x1.00 vs 9x0.55): SCWI +0.375..+0.630, P in [0.588, 0.733]; PAHI -0.103..-1.856, P <= 0.469. R2-55 is a REVERSE kill: SCWI sits exactly at its crossing (evidence -0.156..+0.002, P(top) in [0.464, 0.501], i.e. pinned at chance) while PAHI is confidently opposite (evidence -0.268..-1.047, P(top) <= 0.420). R25-55: SCWI P in [0.62, 0.73]; PAHI prefers the low option for 87% of its gamma range. GAMMA-MAPPING RUNGS: R2-65 (SCWI [0.61, 0.70]; PAHI splits at gamma=1.61, 79% prefer the low option), R25-60 (SCWI [0.70, 0.82]; PAHI splits at gamma=1.79, 61% low), R25-65 (SCWI [0.76, 0.88]; PAHI splits at gamma=2.13, 73% top). POPULATION SHAPE AND HETEROGENEITY SIGNATURES: plotting P(top-option) versus low validity at r=2, SCWI crosses 0.5 inside (0.55, 0.60) for EVERY subject with near-zero between-subject spread (the whole tau box moves v* by only 0.026), whereas PAHI crosses in (0.60, 0.75) with crossing = 0.5^(1/gamma) spread over 0.14 validity units - a fivefold variance dissociation in crossing location. At r=2.5, SCWI never crosses (flat above 0.5 down to v=0.50) while PAHI crosses inside (0.52, 0.68). Moreover PAHI's two crossings must be jointly consistent with a single gamma: v*_{2.5} = 0.8^(1/gamma) * v*_2, i.e. v*_{2.5} in [0.85, 0.91] x v*_2; SCWI has no observable r=2.5 crossing at all. CONTROLS: pair 21 (identical mixed vectors) pins side/response bias at exactly 0.500 for both theories; pair 22 (all-1 vs all-0) calibrates the lapse (both predict P(all-1 option) ~ epsilon/2, confirming anti-validity processing); pairs 23/24 are confident agreement cells (e.g. one 0.50-cue 1 vs two 1.00-cue 1s) calibrating the shared softmax/lapse on the anti-validity side. DECISION RULES: (i) SCWI is supported, and PAHI falsified, if P(top) > 0.5 on all four kill cells (especially STK/STK2 >= 0.57, which PAHI cannot exceed at 0.456/0.469 for any admissible parameters), the r=2 crossing lies in (0.55, 0.60) with tight between-subject clustering, r=2.5 is flat-H all the way down to v=0.50, and R2-55 sits at ~0.50. (ii) PAHI is supported, and SCWI falsified, if P(top) < 0.5 on the kill cells (STK/STK2 <= 0.47, below SCWI's 0.572/0.588 floor), the r=2 crossing lies at or above 0.61 (low-option choices persist through v=0.60 and top-option choices emerge only at 0.65+), an r=2.5 crossing is observed inside (0.52, 0.68) and is consistent with the r=2 crossing through a single gamma, R2-55 is clearly below 0.5, and crossing locations vary widely across subjects. (iii) If subjects prefer options with MORE high-validity 1s (pro-validity integration), or crossings appear below 0.55, or the profile fits neither map, both pure forms are falsified.

[6] ADVERSARIAL LOGIC. SCWI (advocated) and PAHI (competing) agree on everything except the validity-to-weight map: SCWI uses w(v) = v + tau*v^2 with tau in [0.15, 0.42]; PAHI uses w(v) = v^gamma with gamma in [1.4, 2.4]. Both are anti-validity additive integrators (a rating of 1 is a defect, tied cues cancel, softmax plus symmetric lapse), so the ONLY exploitable wedge is the SHAPE of w(v). The key mathematical fact: the local log-elasticity of SCWI's map, (1+2*tau*v)/(1+tau*v), lies in [1.13, 1.30] for ALL v in [0.5, 1] and ALL tau in its box, while PAHI's elasticity is exactly gamma >= 1.4. The ranges are DISJOINT, so for every validity pair (v_hi, v_lo) SCWI's weight ratio w(v_hi)/w(v_lo) is strictly SMALLER than PAHI's. Experiments [4] and [5] exploited this at the v = 1.00 anchor (where PAHI's weight is pinned at exactly 1). This design is the TRANSPOSE into the mid-validity region, using NO validity at 1.00: the knife-edge becomes a RATIONAL COUNT RATIO. For a trial pitting a endorsements on v_hi-cues against b endorsements on v_lo-cues, both theories prefer the high-validity-stack option iff their weight ratio is below b/a. SCWI ratio windows (over its full tau box): (0.75, 0.50): [1.552, 1.630]; (0.80, 0.50): [1.667, 1.766]; (0.85, 0.50): [1.783, 1.907]. PAHI windows (over gamma in [1.4, 2.4]): 1.5^gamma = [1.764, 2.646]; 1.6^gamma = [1.931, 3.090]; 1.7^gamma = [2.102, 3.573]. Each inter-window gap contains a simple rational: 5/3 = 1.667 and 7/4 = 1.75 in (1.630, 1.764); 9/5 = 1.80 in (1.766, 1.931); 2 in (1.907, 2.102). Placing the count ratio exactly at these rationals yields KILL CELLS with OPPOSITE sign predictions for EVERY admissible parameter value of BOTH theories.

KILL CELLS (evidence for the stated option, positive for all parameters): B3 = 6x0.75 vs 10x0.50 (ratio 5/3): SCWI prefers the 0.75-stack with evidence 0.5 - 0.875*tau in [0.13, 0.37]; PAHI prefers the 0.50-stack with evidence 0.5^gamma*(6*1.5^gamma - 10) in [0.22, 1.11]. B4 = 4x0.75 vs 7x0.50 (ratio 7/4): SCWI evidence 0.5 - 0.5*tau in [0.29, 0.43]; PAHI evidence in [0.02, 0.68]. K3 = 5x0.80 vs 9x0.50 (ratio 9/5): SCWI evidence 0.5 - 0.95*tau in [0.10, 0.36]; PAHI evidence in [0.25, 1.23]. A3 = 4x0.85 vs 8x0.50 (ratio 2): SCWI evidence 0.6 - 0.89*tau in [0.23, 0.47]; PAHI evidence in [0.16, 1.19]. Intuition: PAHI's steep power law makes mid-validity endorsements relatively heavy, so for PAHI a 1 on a 0.75-cue outweighs 5/3 endorsements on 0.50-cues; SCWI's near-linear-quadratic map makes them equivalent-or-lighter, so SCWI needs the count ratio to reach ~1.75-2.1 before flipping. POPULATION PREDICTIONS (SCWI mixture 16% guessers / 50% mid / 34% deterministic; PAHI with beta in [1.5,4], gamma in [1.4,2.4], epsilon in [0.06,0.2]): kill composite (48 trials/subject, pairs 1-12) SCWI = 0.62-0.72 (point ~0.66); PAHI = 0.15-0.45 (point ~0.25). Every non-guesser SCWI subject sits >= 0.55; every PAHI subject sits <= 0.50.

LADDER / FLIP-LOCATION DISSOCIATION. Bracketing rungs pin each theory's crossing point, per subject, at two independent validity pairs. On the 0.75/0.50 ladder: SCWI's flip ratio lies in (3/2, 5/3) for EVERY subject (low-stack at r = 3/2 [pair 13], high-stack at r = 5/3 [pairs 1-4]); PAHI's flip ratio is >= 1.764, hence at or above the 7/4 rung (low-stack at 5/3 AND at 7/4 [pairs 5-6], high-stack at r = 2 only if gamma < 1.71 [pairs 15-16]). On the 0.85/0.50 ladder: SCWI flips in (7/4, 2) (low-stack at 7/4 [pairs 17-18], high-stack at 2 [pairs 9-12]); PAHI flips above 2 (low-stack at 2, high-stack at 9/4 only if gamma < 1.53 [pairs 19-20]). A systematic ONE-RUNG SHIFT in the psychometric crossing over count ratio, replicated across two validity mixtures -- a quantitative signature SCWI's tau-box cannot cross into and PAHI's gamma-box cannot cross out of.

GAMMA-RUNGS (PAHI-specific consistency test). B5 (6x0.75 vs 12x0.50) splits PAHI exactly at gamma = 1.71; A4 (4x0.85 vs 9x0.50) splits PAHI at gamma = 1.53. SCWI predicts the HIGH-stack on both with huge margins (1.34-1.44 and 0.83-1.00): population P(high-stack) ~ 0.85 vs PAHI's ~0.30-0.40. A PAHI subject's rung and kill choices must be jointly consistent with a single gamma (e.g., high-stack on A4 implies gamma < 1.53, which forces high-stack on B5); SCWI predicts high-stack on all rungs uniformly, with essentially zero between-subject spread in crossing location (the whole tau box moves the 0.75-ladder crossing ratio by only 0.078, all inside the same rung gap), whereas PAHI's crossings spread over ~0.9 ratio units.

POPULATION-MIXTURE PROBE (novel; untouched by experiments [0]-[5]). SCWI's beta_seed mixture places 16% of subjects in a true-guesser regime (beta in [0.02, 0.10], epsilon in [0.12, 0.28]). On the all-1-vs-all-0 anchors (pairs 21-22) the total anti-load is ~23.5 (SCWI units) or ~17 (PAHI units), so these guessers sit at P(choose all-1) in [0.22, 0.35] -- and at ~0.5 on every kill and rung. PAHI has NO such regime: every subject has beta >= 1.5 and epsilon <= 0.20, so P(choose all-1) <= epsilon/2 <= 0.10 for EVERY subject. Statistic: fraction of subjects choosing the all-1 option on >= 3 of 8 anchor trials: SCWI ~ 0.08-0.13; PAHI ~ 0.01 (and no PAHI subject can exceed ~0.12 in true probability). SCWI additionally predicts a TRIMODAL per-subject anchor distribution (masses near 0.02, 0.05, and 0.3) with guesser-flagged subjects also at chance on kills/rungs; PAHI predicts a unimodal-low continuum.

CONTROLS. Pairs 13-14 (B2: 4x0.75 vs 6x0.50, r = 3/2) and 17-18 (A2: 4x0.85 vs 7x0.50, r = 7/4) are agreement cells where BOTH theories prefer the low-stack (SCWI evidence 0.75*tau in [0.11, 0.32] and 1.14*tau - 0.1 in [0.07, 0.38]; PAHI evidence 0.40-0.87 and 0.53-1.31): they calibrate the shared softmax/lapse on the anti-validity side and identify non-inverted TTB subjects (who pick the high-stack because the top discriminating cue favors it -- note TTB also picks the high-stack on the kills, so B2/A2 and the anchors are what exclude it) and pro-validity integrators (who pick all-1 on anchors). Pair 24 (CAL: one 0.85-endorsement vs one 0.50-endorsement) is a single-cue anti-validity calibration (both theories prefer the 0.50-1 option, P ~ 0.75-0.85). Pair 23 (TIE: identical mixed vectors) pins side/response bias at exactly 0.500 for both. All critical pairs are mirrored; B3 and A3 include cue-rotation variants (different low-cue subsets) to average within-subject per-cue attention (drawn from [0.90, 1.10]); kills that use every cue of a class (K3, B5) average attention by construction, and the population-level sign predictions are immune to attention because it is symmetric with mean 1. Worst-case attention analysis: flipping B3 requires ~8% adverse class-mean attention (~2 SD of a 10-cue class mean); B4/B5/A4 margins (0.29-1.44) require 15-30% and cannot flip.

DESIGN. 29 experts (6 at 0.85, 5 at 0.80, 6 at 0.75, 12 at 0.50); 24 unique pairs x K = 96//24 = 4 repetitions = 96 trials. Per subject: 48 kill trials, 32 rung trials, 8 anchor trials, 4 tie trials, 4 calibration trials.

DECISION RULES. (i) SCWI is supported, and PAHI falsified, if the kill composite P(high-stack) > 0.55 with per-subject high-stack majorities (PAHI cannot exceed 0.50 on any kill for any admissible parameters), the rung composite >= 0.75, the 0.75-ladder crossing lies in (1.5, 1.667) and the 0.85-ladder crossing in (1.75, 2) with tight between-subject clustering, and the anchor-tail fraction (subjects >= 3/8 all-1 choices, also near chance on kills) is ~0.10-0.15 where PAHI predicts ~0.01. (ii) PAHI is supported, and SCWI falsified, if all four kills show low-stack preference (composite < 0.45), the rung pattern is jointly consistent with a single gamma per subject (A4 high-stack => B5 high-stack; A4 low-stack with gamma > 1.53 => correspondingly graded B5), crossings sit at or above 5/3 (0.75-ladder) and at or above 2 (0.85-ladder) with wide between-subject spread, and the anchor distribution is unimodal-low with no subject above ~0.12. (iii) BOTH pure forms are falsified if subjects prefer the option with MORE high-validity endorsements (pro-validity: all-1 chosen on anchors, low-stack rejected on kills), or if a flat anti-tallying pattern appears (low-stack on EVERY cell including the B5/A4 rungs, which both theories reject), or if the profile fits neither map.

[7] ADVERSARIAL LOGIC. PAHI (advocated) and SCWI (competing) are structurally near-identical anti-validity additive integrators: a rating of 1 counts AGAINST an option, tied cues cancel exactly, and choice is a softmax over the anti-load difference plus a symmetric lapse. The ONLY exploitable wedge is the validity-to-weight map: PAHI uses w(v) = v^gamma with gamma in [1.4, 2.4]; SCWI uses w(v) = v + tau*v^2 with tau in [0.15, 0.42]. The exact weight-RATIO windows w(v_h)/w(v_l) over the FULL parameter boxes, for three validity pairs forming a triangle (none explored in [0]-[6], which used 1.00/0.50, 1.00/0.55-0.65, and 0.75-0.85/0.50): (0.90, 0.70): SCWI [1.321, 1.369] vs PAHI [1.422, 1.828]; (0.70, 0.50): SCWI [1.439, 1.497] vs PAHI [1.602, 2.243]; (0.90, 0.50): SCWI [1.900, 2.050] vs PAHI [2.277, 4.098]. In every pair the windows are DISJOINT and each gap contains a rational count ratio: 7/5 = 1.40, 14/9 = 1.556, 11/5 = 2.20. A trial pitting a endorsements on high-validity cues against b on low-validity cues (all other cues 0 for both options) is therefore a parameter-free SIGN KILL at those ratios: SCWI prefers the high-validity stack H (its ratio is below b/a for EVERY tau) while PAHI prefers the low stack L (its ratio is above b/a for EVERY gamma). KILL MARGINS: P1-K (5x0.90 vs 7x0.70): SCWI H by 0.14-0.31, PAHI L by 0.10-1.35; P2-K (9x0.70 vs 14x0.50): SCWI H by 0.32-0.56, PAHI L by 0.11-0.60; P3-K (5x0.90 vs 11x0.50): SCWI H by 0.45-0.81, PAHI L by 0.15-1.80. Population kill composite (18 trials/subject): SCWI ~0.55-0.79 (16% guessers at 0.50, non-guessers 0.55-0.88 with beta 1.8-6.0); PAHI ~0.15-0.45 (margins against H, beta 1.5-4.0, epsilon 0.06-0.20).

NOVELTY VS PRIOR DESIGNS. (1) The (0.90, 0.70) pair is the CLOSEST validity pair ever pitted: the two maps' absolute levels nearly coincide (SCWI 1.02-1.24 vs 0.77-0.91; PAHI 0.78-0.86 vs 0.36-0.60), so only relative steepness discriminates — a pure shape test with no level confound. (2) IMPLIED-EXPONENT TRIANGULATION (headline statistic): for a subject whose choices flip from L to H at count ratio r* on pair (v_h, v_l), define gamma_hat = ln(r*)/ln(v_h/v_l). PAHI: gamma_hat = gamma in [1.4, 2.4] at EVERY pair, jointly consistent with a single gamma, with large between-subject spread (SD ~0.29). SCWI: gamma_hat in [1.08, 1.25] at every pair (P1 [1.11, 1.25]; P2 [1.08, 1.20]; P3 [1.09, 1.22]) — strictly below PAHI's floor of 1.4 for ANY admissible parameters, with tight spread (SD ~0.05). The exponent LEVEL is thus a parameter-free classifier replicated three times, plus a fivefold between-subject variance dissociation. (3) A within-subject CONSISTENCY WEB: SCWI predicts H on every cell with r >= 1.5 (all six gamma-rungs and all three kills) for every subject regardless of tau — a step function crossing in (1.25, 1.40) at 0.9/0.7, (1.40, 1.556) at 0.7/0.5, (1.75, 2.20) at 0.9/0.5. PAHI predicts L on all kills and brackets, with rungs splitting at gamma = 1.52 (P2-C), 1.61 (P1-C), 1.66 (P2-D), 2.06 (P2-E), 2.13 (P3-E), 2.23 (P1-E); e.g., a PAHI subject choosing H on P1-E but L on P2-E has gamma in (2.06, 2.23), which FORCES L on P3-E, P2-D, P1-C and all kills; H on P1-C forces H on P2-C/D/E and P3-E. SCWI cannot generate any such covariance — it predicts H on all rungs unconditionally.

RUNGS AND BRACKETS. P1-B (r = 4/3) and P3-B (r = 2) are SCWI tau-rungs splitting at tau = 0.213 and tau = 0.323 (with the ordering constraint H on P1-B => H on P3-B); PAHI is confidently L on both (margins 0.18-0.88), so 'H on P1-B' alone is nearly a kill for low-tau SCWI. P1-A (r = 1.25), P2-A (r = 1.40), P3-A (r = 1.75) are agreement-L brackets (both theories prefer the low stack, margins 0.10-1.78); P1-F (r = 2) is the agreement-H bracket. TIE (identical vectors) pins side/response bias at exactly 0.500 for both theories; ANCHOR (all-1 vs all-0, total anti-load ~21 SCWI / ~12 PAHI units) calibrates the lapse and screens out pro-validity integrators and PAHI's rare s = +1 subjects, who choose the all-1 option. ATTENTION ROBUSTNESS: kill stacks use 5-14 cues per side; flipping a kill requires class-mean attention distortions of 2-8% (1-3 SD of the class-mean of SCWI's per-cue attention in [0.90, 1.10]); P2-K/P3-K need > 2 SD, and population-level sign predictions are immune because attention is symmetric with mean 1. No feedback keeps weights anchored to the instructed validities; every critical trial requires comparing many cues, satisfying the min-RT viewing requirement.

DESIGN. 28 experts (5 x 0.90, 9 x 0.70, 14 x 0.50); 32 unique pairs x K = 96//32 = 3 repetitions = 96 trials (18 kill, 12 tau-rung, 36 gamma-rung, 24 bracket, 3 tie, 3 anchor trials). All critical pairs are mirrored to absorb side bias.

DECISION RULES. (i) SCWI is supported, and PAHI falsified, if the kill composite P(H) >= 0.55 with per-subject H-majorities (PAHI cannot exceed 0.50 on any kill for any admissible parameters), H-majority on all six gamma-rungs, rung-implied exponents clustering in [1.05, 1.30] at all three pairs with between-subject SD < 0.08, and a coherent P1-B/P3-B tau pattern. (ii) PAHI is supported, and SCWI falsified, if the kill composite P(H) <= 0.45 (SCWI's population floor is ~0.55), rung choices are jointly consistent with a single gamma in [1.4, 2.4] per subject (any violation of the consistency web — e.g., H on P1-E with L on P2-C — falsifies PAHI), implied exponents >= 1.4 at all three pairs with wide between-subject spread, and ANCHOR choices at the lapse floor. (iii) BOTH are falsified if the all-1 option is chosen well above lapse on ANCHOR (pro-validity integration), if kills and rungs sit flat at 0.5 (pure guessing), or if implied exponents fall in the empty band (1.25, 1.40) or fit neither map.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "ADVERSARIAL LOGIC. SCWI (advocated) and HEAI (competing) are near-identical anti-validity additive integrators: a rating of 1 is a defect, tied cues cancel, softmax plus symmetric lapse, 16% guessers. Moreover, HEAI's SHALLOW mode (75% of engaged, gamma ~ N(1.04, 0.22)) is almost exactly degenerate with SCWI's map w(v) = v + tau*v^2 (tau in [0.15, 0.42], implied power-exponent 1.10-1.23 at every validity pair): their weight-ratio windows overlap at every mixture I could construct, so no single-cell sign kill between them exists. The honest adversarial move is therefore to target HEAI's DISTINCTIVE claim \u2014 the BIMODAL gamma, specifically the STEEP minority (25% of engaged = 21% of population, gamma in [1.65, 2.75], median 2.10) \u2014 whose validity gradient is far steeper than anything in SCWI's box, and to exploit the resulting POPULATION and DISTRIBUTIONAL dissociations, which SCWI's unimodal near-linear-quadratic map cannot reproduce for any parameters.\n\nMATHEMATICAL SETUP. With 3 cues at v=1.00 (where HEAI's weight is exactly 1, gamma-invariant) pitted against b cues at v=0.50, an anti-validity integrator prefers the TOP-STACK option H (the 3 high-validity 1s) iff 3*w(1.00) < b*w(0.50). SCWI's margin (defect_L - defect_H) = (0.5b - 3) + tau*(0.25b - 3): NEGATIVE (prefers the low stack) for b <= 6, and POSITIVE for every b >= 8 with margin 1 - tau in [0.58, 0.85] at b=8 up to EXACTLY 3.0 at b=12 (parameter-free anchor: the tau-term cancels). No SCWI parameter value, no attention draw (class-mean attention shifts the margin by SD ~0.12 at worst), and no beta/epsilon can make an engaged SCWI subject prefer the low stack at b >= 8. HEAI's flip occurs at gamma* = log2(b/3): b=8 -> 1.415, b=9 -> 1.585, b=10 -> 1.737, b=11 -> 1.874, b=12 -> 2.0. EVERY steep subject (gamma >= 1.65) prefers the LOW stack at b=8 and b=9 with large margins (e.g. at gamma=2.10, b=9: margin 0.90), and flips to H only as b/3 exceeds 2^gamma. A second, independent ladder at v=0.60 (3x1.00 vs b cues, gamma* = ln(b/3)/ln(5/3)) replicates the instrument at a different validity mixture: SCWI's flip ratio there is in [1.76, 1.89], so SCWI prefers H at b >= 6 (margin 0.25-0.47) for all tau, while steep subjects flip at gamma* = 1.357 (b=6), 1.658 (b=7), 1.921 (b=8), 2.151 (b=9).\n\nPREDICTED DISSOCIATIONS. (S1) KILL COMPOSITE (cells L2, L3, M2, M3 + mirrors, gamma* <= 1.66, 32 trials/subject): SCWI population P(H) ~ 0.80; HEAI ~ 0.62-0.66 (shallow confidently H, steep confidently L, guessers 0.5). (S2) SUBJECT-LEVEL KILL, parameter-free: on L3 (3x1.00 vs 9x0.50) SCWI's margin is >= 1.19 for EVERY admissible (tau, attention, regime) \u2014 no engaged SCWI subject can prefer the low stack. HEAI predicts ~21% of subjects prefer it with P ~ 0.8-0.9. The fraction of subjects with an L-majority on the kill composite is ~0-3% under SCWI vs ~22-25% under HEAI, and under HEAI those subjects form a distinct LOW MODE in the between-subject distribution (bimodal), whereas SCWI predicts only guesser mass near 0.5 and engaged mass >= 0.6. (S3) FLIP-LOCATION CEILING: SCWI's implied flip exponent is bounded in [1.10, 1.23] at (1.0, 0.5) \u2014 BELOW the lowest kill rung (gamma* = 1.357 at M2). So under SCWI, NO subject's psychometric flip (low-stack -> top-stack) lies inside the observable ladder; under HEAI, ~21% of subjects flip inside [1.4, 2.2], with the flip point estimating their personal gamma. (S4) CROSS-LADDER CONSISTENCY WEB: a HEAI steep subject's flips on the 0.50-ladder (gamma* = log2(b/3)) and the 0.60-ladder (gamma* = ln(b/3)/ln(5/3)) must recover the SAME gamma and be monotone in gamma*; e.g. L at L3 (gamma* 1.585) plus H at M5 (gamma* 2.151) implies gamma in (1.585, 2.151), which forces L at L2, M2, M3 and H at L6 only if gamma < 2.0. SCWI predicts no flips at all, hence no such covariance; L-choices across kill cells are uncorrelated noise under SCWI and strongly correlated (latent steep class) under HEAI \u2014 a testable correlation difference at matched marginal rates. (S5) PARAMETER-FREE ANCHOR L6 (3x1.00 vs 12x0.50): SCWI's margin is exactly 3.0 for every tau (the tau-coefficient 0.25*12 - 3 vanishes), so P(H) >= 1 - epsilon for every engaged subject; HEAI's steep subjects with gamma > 2.0 (65% of steep) still prefer the low stack there. POPULATION PSYCHOMETRICS over gamma*: SCWI is flat-high and saturated early (0.80 -> 0.92 from gamma* = 1.415 to 2.15, rising only through margin growth); HEAI is graded and rises late (0.64 -> 0.80), with the shallow mode's thin upper shoulder (gamma in [1.32, 1.42]) adding a small extra L-mass at L2 that SCWI cannot generate.\n\nCONTROLS AND EXCLUSION OF THIRD PARTIES. L-C (3x1.00 vs 5x0.50, r = 5/3) and M-C (3x1.00 vs 4x0.60, r = 4/3) sit BELOW every theory's flip ratio: both SCWI and HEAI (including all steep subjects) confidently prefer the LOW-validity stack there. These controls (i) anchor the bottom of the psychometric function, and (ii) exclude confounded strategies that mimic the kill predictions: a TTB subject follows the top cue and picks the top stack on EVERY cell (kills AND controls \u2014 the controls expose it); a 'fewer-1s' heuristic likewise picks the 3-1 option everywhere including controls; a tallying/more-1s subject picks the low stack on kills (mimicking steep-HEAI) but is exposed by the ANCHOR (all-1 vs all-0), where both target theories predict P(all-1) ~ epsilon/2 while tallying predicts ~1. TIE (identical mixed vectors) pins side/response bias at exactly 0.500 for both theories. ANCHOR also calibrates the shared lapse and screens pro-validity integrators. ATTENTION ROBUSTNESS: kill stacks use 3 vs 6-12 cues per side, so SCWI's per-cue attention jitter (class-mean SD 0.02-0.03) shifts margins by ~0.05-0.12 against margins >= 0.25 (M2, worst case) \u2014 no flip; the L3+ cells are z >= 10 safe. All critical pairs are mirrored to absorb side bias. Every trial requires comparing large multi-cue stacks (satisfying the min-RT viewing requirement), and with 24 unique pairs, K = 96 // 24 = 4 repetitions = 96 trials per subject (32 kill, 40 rung, 16 control, 4 tie, 4 anchor).\n\nDECISION RULES. (i) SCWI is supported, and HEAI falsified, if the kill composite sits at ~0.78-0.85 with NO subject (beyond guesser noise at ~0.5) showing an L-majority on L3/L4/L5/L6 or M3/M4/M5, no psychometric flips inside the ladder, L6 at ~0.92, and a unimodal-high plus guesser-tail subject distribution. HEAI's steep mode (21% of the population confidently anti-following at gamma* <= 1.6) cannot be produced by SCWI at any parameter values. (ii) HEAI is supported, and SCWI falsified, if ~20% of subjects confidently prefer the low stack on the low-gamma* kills, their flips lie inside the ladder and are jointly consistent with a single gamma across BOTH ladders (monotone in gamma*, gamma_hat agreeing between the 0.50- and 0.60-ladders), the kill composite sits at ~0.62-0.68, and the between-subject kill distribution is bimodal with an empty band between the shallow cluster (flips below 1.4) and the steep cluster (flips in [1.65, 2.4]). Any subject with an L-majority on L3 alone falsifies SCWI outright, since SCWI's margin there is >= 1.19 across its entire parameter and attention box. (iii) BOTH are falsified if the all-1 option is chosen well above lapse on ANCHOR (pro-validity integration or tallying), if the top stack is preferred on the L-C/M-C controls (TTB or fewer-1s heuristic), or if flips appear below gamma* = 1.3 or fit neither a power law nor the quadratic map.",
  "validities": [
    1.0,
    1.0,
    1.0,
    0.6,
    0.6,
    0.6,
    0.6,
    0.6,
    0.6,
    0.6,
    0.6,
    0.6,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    0.5
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      0
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 18185,
  "output_tokens": 49866,
  "reasoning_tokens": 44989,
  "provider": "Sail Research"
}
```
