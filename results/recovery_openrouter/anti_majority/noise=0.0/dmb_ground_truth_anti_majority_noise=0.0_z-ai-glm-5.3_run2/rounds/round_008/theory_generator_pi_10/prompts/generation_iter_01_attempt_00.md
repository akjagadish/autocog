# generation_iter_01_attempt_00

## System Prompt

You are a renowned cognitive scientist and an expert Python programmer.

Your job is to propose a new theory and its model instantiation in the Decision Making (Binary Features) domain based on the feedback provided by an arbiter. The feedback contains diagnoses of mechanistic failures of the previous theory along with suggestions for a new theory family that overcomes those failures. The newly proposed theory and model should display human-like behavior when simulated on experiment(s). 
The goal of the theory generation process is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across the majority of experiments. 
You will see a list of theories that have been proposed in the past but you should only use them as inspiration and not to choose from them. Propose a new theory that is different. 
If they fail to do so, you will receive feedback on their performance on the same experiment(s) and you will have to propose another new theory and model that meet the requirements, iterating until you succeed.

If you think the failure to capture human behavior is due to arbiter feedback that is inaccurate or unhelpful, you can propose a new theory and model that ignore the feedback, but you must provide rationale for why you are ignoring it and how your proposal overcomes the identified mechanistic failures.

## ACCEPT GATE & LOSS TRAJECTORY — HOW THE LOOP HANDLES YOUR EDITS
This propose-loop has a programmatic accept gate: after every iteration the candidate's `aggregate_loss` is compared against the running-best loss; strict improvement -> ACCEPTED (the candidate becomes the new running-best base); otherwise -> REJECTED (the candidate is discarded and the base is unchanged). You do NOT need to manually "revert" a regressed edit — the gate already does that for you.

The block rendered below as `## PREVIOUS CANDIDATE (this loop)` is ALWAYS the running-best (last ACCEPTED) candidate, NEVER your most recent attempt if it was rejected. So:
  * Treat `## PREVIOUS CANDIDATE` as a known-good base. Build on it.
  * The `## LOSS TRAJECTORY` block tags every iteration ACCEPTED or REJECTED. Use this as ground truth on which past critic advice actually moved the loop forward and which didn't.
  * The `## PRIOR FEEDBACK ITERATIONS` block annotates each prior critique with the same ACCEPTED/REJECTED tag of the candidate it elicited. Down-weight critic advice whose previous candidates were REJECTED, and reinforce / extend advice whose candidates were ACCEPTED.
  * Treat the best ACCEPTED iteration's loss as a soft floor — the next edit should plausibly land at-or-below it, otherwise the gate will reject your attempt and the base stays put.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

## PARAMETER NOTATION
`parameters` is a JSON object mapping each parameter name (snake_case string) to a *string* value that specifies its domain. Every value MUST be a string — never a bare list, number, tuple, or expression. Use exactly one of these notations per parameter:

1. Continuous interval — square brackets, two numeric bounds:
   "[min, max]"
   Examples: "[0, 1]", "[1.0, 10.0]", "[10, 1000]"

2. Discrete set — curly braces, comma-separated values:
   "{v1, v2, ...}"
   Example: "{1, 2}"

3. Vector of intervals whose length is set by the experiment — a bracketed tuple repeated by a symbolic length variable:
   "[(min, max)] * length_var"
   Example: "[(0, 1)] * n_features"

4. Symbolic reference — a bare variable name (no brackets, no angle brackets), used when the parameter takes its value from an experiment-defined constant rather than a range:
   "variable_name"
   Example: "n_features"

Rules:
- Do not use parentheses for intervals; square brackets only. Tuples `(a, b)` are reserved for the vector-of-intervals notation in (3).
- Do not mix notations within a single value (e.g., no "[0, 1] or {2, 3}").
- Do not quote numbers inside the notation (write "[0, 1]", not "['0', '1']").
- Every parameter referenced by `predict` or `policy` must appear as a key in `parameters`, and vice versa.
- Notations 3 and 4 may ONLY reference the experiment-defined symbolic identifiers listed under "ALLOWED SYMBOLIC IDENTIFIERS" below. Do not invent new identifier names. If a parameter's shape doesn't fit any of those variables, fall back to a literal interval (notation 1) or discrete set (notation 2). Use these names so the model adapts to any experiment in this domain instead of hardcoding shapes.

## ALLOWED SYMBOLIC IDENTIFIERS (for notations 3 and 4 above)
- n_features: Number of expert ratings per option (LLM-proposed via `validities` length).
- validities: Per-expert validities (LLM-proposed; each in [0.5, 1.0]); fixed across all trials.

## AVAILABLE IMPORTS inside `predict` and `policy`
- numpy as np
- pandas as pd
- scipy and its submodules
- torch and torch.nn.functional as F
- sklearn and its submodules
- math, random, and other standard Python libraries

## RUNTIME CONTRACT (function signatures and argument shapes)
`predict(parameters, state, history) -> np.ndarray`:
- `parameters`: dict[str, value]. One sample drawn from your declared `parameters` ranges, applied for the entire subject run.
- `state`: the per-trial input delivered by the experiment (shape is domain-specific — see the experiment description above and the `history` key list below, which mirrors the per-trial variables carried in `state`). Convert to an array with `np.asarray(state)` if you need array ops.
- `history`: dict-of-lists for past trials in this subject's run, NOT a list-of-dicts. The per-trial keys are:
  Each value below is a Python list in trial order; entry `i` is the value for trial `i`. On the first trial all lists are empty.
  - `"option_a_ratings"`: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
  - `"option_b_ratings"`: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
  - `"response"`: 0 if subject chose A, 1 if subject chose B.
Iterating `for x in history:` iterates the dict KEYS (strings); to walk trials index the lists in lock-step, e.g. `for i in range(len(next(iter(history.values())))): ...`.
- Returns: 1-D `np.ndarray` of choice probabilities over the experiment's discrete action set, summing to 1.

`policy(probs) -> int`:
- Receives the probability vector produced by `predict`.
- Returns: integer index in `[0, len(probs))` identifying the chosen action. If you sample with `np.random.choice(..., p=probs)`, normalise first (`probs = np.asarray(probs, dtype=np.float64); probs /= probs.sum()`) to avoid the "probabilities do not sum to 1" ValueError from float drift.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## ARBITER GUIDE
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_8`
- THEORY 2 = `pi_9`
- The recommendation below acts on THEORY 1 (= `pi_8`).

Propose a new theory to replace pi_8: 'TPQSAI — Trait-based Prestige-Quantity Salience Anti-validity Integration.' Core (retained from the validated CNNAI/PQSAI lineage): a rating of 1 is a defect (sign inversion); per-option value = sum_j w_j * rating_j with w_j = -v_j*(v_j/v_ref)^eta, v_ref = mean endorsed validity; eta drawn per bucketed-mean-endorsed-validity context with latent cross-context r ~ 0.18; softmax(beta*value) with symmetric lapse; ~13% guessers, ~12% hot tail. New/changed vs both current theories: (1) STABLE PRESTIGE TRAIT: sigma_i is drawn ONCE per subject from a quasi-bimodal distribution (~45-55% linear counters with sigma in [1.0, 1.2], remainder prestige-sensitive with sigma in [1.7, 2.2]), and is NOT redrawn across contexts or ladders — this within-subject stability is what produces the observed cross-ladder correlation (Exp 11 target ~0.20) that per-context redraws destroy. (2) RECALIBRATED LOAD DILUTION: sigma_eff = 1 + (sigma_i - 1)*exp(-gap/0.10)*D with D = 1/(1+(n_disc/n0)^1.1), n0 in [8, 10] — flatter and stronger than pi_9's current setting so that small-n_disc knife cells (Exp 13 AL, Exp 1 B1/B2) retain D ~ 0.65-0.75, landing the AL anti-validity excess near the observed 2.88 (between pi_8's undiluted 3.60 and pi_9's current 2.32), while large stacks (n_disc >= 20) still dilute toward the linear core (Exp 16, Exp 2 high rungs). (3) MARGIN-LEVEL NOISE: replace per-cue load-scaled attention diffusion with per-trial noise on the value difference, sd = s0 + s1*log(1+n_disc), with s1 small (~0.01-0.015) — this attenuates within-subject reliability on complex 24-31 feature trials (Exp 9 target ~0.20 cross-ladder correlation) without the over-attenuation that made pi_9 miss Exp 11. (4) BETA TRIM: engaged beta lognormal median ~1.85 (clip [0.8, 4.2]) so the winsorized single-validity ladder slope lands near the observed 1.41 rather than 1.58. This theory is a direct, stronger competitor to pi_9: it keeps every experiment pi_9 wins (Exps 1, 2, 15, 17, 18 via the dilution + bimodality architecture) while repairing pi_9's misses on Exps 11, 13, and 16 through the stable trait, the stronger small-n_disc amplification, and the beta trim.

## THEORY LEADERBOARD
A small set of prior picked theories shown for reference. Overall score is in `[0, 1]`, higher = better, computed as `1 - L2_norm(normalized_per_experiment_distances) / max_L2_norm`. 1.0 means closest to the real value on every experiment+metric pair; 0.0 means farthest. Each entry below carries the same depth of detail as the PREVIOUS MODEL INSTANCE above so you can borrow concrete mechanisms when useful.

### `pi_9` (overall score: 0.932)

**Description**
PQSAI (recalibrated) — Prestige-Quantity Salience Anti-validity Integration with Load Dilution and a Bimodal Salience Mixture. Subjects integrate binary expert ratings additively into a per-option value but comprehend a rating of 1 as a DEFECT (sign inversion), with context-normalized anti-weights w_j = -v_j*(v_j/v_ref)^eta (v_ref = mean endorsed validity; eta drawn per bucketed-mean-endorsed-validity context, latent cross-context r ~ 0.18). The top endorsed validity tier receives an additional multiplicative aversion sigma_eff = 1 + (sigma_i - 1)*exp(-gap/0.10)*D, where D = 1/(1 + (n_disc/n0)^1.1) is a FLAT load-dilution factor (exponent softened 1.5 -> 1.1, n0 raised 6 -> ~10): prestige aversion is strong on small knife-edge contests (n_disc = 6, D ~ 0.62-0.68) but retains SUBSTANTIAL amplification even on large stacks (n_disc = 13-14, D ~ 0.40-0.45), so the population sits between the pure linear core and full undiluted salience exactly where the real data lie. The engaged population is bimodal in salience: ~50% prestige-sensitive (sigma_i ~ U[1.6, 2.1]) and ~50% linear counters (sigma_i ~ U[1.0, 1.15]), generating large between-subject variance on near-knife composites. Per-trial attention diffusion scales with load, sd = s0 + s1*sqrt(n_active), attenuating within-subject reliability on complex multi-tier trials. Softmax(beta*value) with symmetric lapse; ~13% true guessers; ~12% hot tail; exactly linear within a validity family (tied-filler load ladders flat).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # PQSAI (recalibrated iter 2): Prestige-Quantity Salience
    # Anti-validity Integration with Load Dilution and a Bimodal
    # Salience Mixture.
    #
    # CHANGES VS THE ACCEPTED ITER-1 BASE (minimal diff, per the
    # critic's five-point recalibration — all in-family):
    #   (A) FLATTER DILUTION CURVE: exponent 1.5 -> 1.1 and
    #       dilution_n0 raised (default 6 -> 10, exposed range
    #       [7, 13]). New operating points: D(n_disc=6) ~ 0.62-0.68
    #       (was ~0.50) and D(n_disc=13) ~ 0.40-0.45 (was ~0.15-0.24).
    #       This simultaneously raises Exp 11 toward 2.88 (small-n_disc
    #       cells keep much more amplification) and pulls Exp 16 down
    #       from the pure-core 0.686 toward 0.495 (large-stack cells
    #       now RETAIN substantial amplification, landing the
    #       population between core and full SACNAI).
    #   (B) STRONGER SENSITIVE MODE: sigma_sensitive box widened to
    #       [1.6, 2.1] (default 1.95) and sensitive fraction toward
    #       0.5 (range [0.4, 0.6]). The 3v3 AL cells now receive
    #       sigma_eff ~ 1.35-1.40 on sensitive subjects — roughly a
    #       half-to-full recovery of SACNAI's undiluted amplification,
    #       which is what the real 2.88 (vs SACNAI's 3.60 and the
    #       core's 0.32) requires.
    #   (C) GAP SCALE KEPT AT 0.10: coarse-gap stacks (Exps 5/8/10)
    #       keep exp(-gap/0.10) <= 0.14-0.22, so the amplifier stays
    #       essentially off there and the validated core predictions
    #       are preserved.
    #   (D) EXP 2 VARIANCE: the widened sensitive box (up to 2.1) plus
    #       the flatter dilution enlarges the mode separation on
    #       gap-0.10, n_disc ~ 5-6 cells (sensitive sigma_eff ~ 1.25 vs
    #       linear ~ 1.02), injecting the missing between-subject
    #       spread without any new mechanism.
    #   (E) EXP 9 OVER-CONSISTENCY: noise_s1 raised (default 0.014 ->
    #       0.022, range [0.012, 0.03]) to attenuate within-subject
    #       reliability on the 24-feature complex trials and pull the
    #       cross-ladder correlation down toward 0.196; noise_s0 kept
    #       low (range [0.02, 0.04]) so Exp 13's flat ladder and
    #       Exp 14's slope are preserved.
    #
    # Everything else — the shared CNNAI/SACNAI core (defect sign,
    # context-normalized anti-validity weights, bucketed per-context
    # eta with latent r ~ 0.18, 13% guessers, 12% hot tail, exact
    # within-family linearity) — is retained verbatim from the
    # accepted base.
    import hashlib
    import math
    import numpy as np
    from scipy.special import ndtri

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"PQSAI expects a (2, n_features) stimulus; got shape {stim.shape}."
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

    def _pos(name, default, lo, hi):
        try:
            x = float(parameters[name])
        except (KeyError, TypeError, ValueError):
            x = default
        if not np.isfinite(x):
            x = default
        return min(max(x, lo), hi)

    u = _seed("regime_seed")
    es = _seed("eta_base_seed")
    bs = min(max(_seed("beta_seed"), 1e-6), 1.0 - 1e-6)
    ps = _seed("epsilon_seed")
    ss = _seed("subject_seed")
    ms = _seed("mode_seed")

    # --- PQSAI-specific knobs (recalibrated) ---
    n0 = _pos("dilution_n0", 10.0, 2.0, 20.0)          # dilution scale (raised)
    s0 = _pos("noise_s0", 0.03, 0.0, 0.2)              # noise intercept (kept low)
    s1 = _pos("noise_s1", 0.022, 0.0, 0.1)             # load-scaled slope (raised)
    sig_s = _pos("sigma_sensitive", 1.95, 1.0, 3.0)    # sensitive-mode sigma (raised)
    frac = _pos("sigma_sensitive_fraction", 0.52, 0.05, 0.95)  # toward 0.5

    # --- Bimodal salience mixture (applies to all non-guessers) ---
    if ms < frac:
        # Prestige-sensitive subject: sigma ~ U[1.6, 2.1] (drawn per
        # subject via the sigma_sensitive parameter).
        sigma = sig_s
    else:
        # Linear counter: sigma ~ U[1.0, 1.15].
        t = (ms - frac) / max(1e-9, 1.0 - frac)
        sigma = 1.0 + 0.15 * t

    # --- Population mixture (uncorrelated with eta). ---
    if u < 0.13:
        # Guesser subpopulation (~13%): beta ~ 0 -> uniform core on
        # every trial; elevated lapse. The amplifier is irrelevant
        # here, so pin sigma at 1.
        t = u / 0.13
        beta = 0.02 + 0.08 * t                 # 0.02 .. 0.10
        epsilon = 0.12 + 0.13 * ps             # 0.12 .. 0.25
        eta_base, redraw_mu, redraw_sd = 0.12, 0.0, 0.0
        sigma = 1.0
    elif u < 0.25:
        # Hot tail (~12%): sharp, low-lapse subjects whose elasticity
        # is pinned at the engaged median, so they amplify the CORE
        # sign pattern without shifting the effective curvature.
        beta = 3.4 + 1.8 * bs                  # 3.4 .. 5.2
        epsilon = 0.02 + 0.03 * ps             # 0.02 .. 0.05
        eta_base, redraw_mu, redraw_sd = 0.12, 0.0, 0.10
    else:
        # Engaged bulk (~75%). Small stable elasticity component
        # (sd 0.09) + larger per-context redraw (sd 0.19): the variance
        # ratio fixes latent cross-context consistency at r ~ 0.18.
        # Narrow stable lapse so it cannot act as a hidden
        # cross-context coupling channel.
        zs = min(max(es, 1e-6), 1.0 - 1e-6)
        eta_base = 0.02 + 0.09 * float(ndtri(zs))
        eta_base = min(max(eta_base, -0.15), 0.20)

        beta = math.exp(0.693 + 0.25 * float(ndtri(bs)))  # median ~2.0
        beta = min(max(beta, 0.8), 4.5)
        epsilon = 0.03 + 0.06 * ps             # 0.03 .. 0.09
        redraw_mu, redraw_sd = 0.10, 0.19

    # --- Context: bucketed mean endorsed validity. ---
    active = np.flatnonzero((ra > 0.5) | (rb > 0.5))
    if active.size == 0:
        # Nothing endorsed by either option: no evidence either way.
        return np.full(2, 0.5)

    v_ref = float(np.mean(val[active]))
    bucket = float(np.round(v_ref, 1))
    key = f"{bucket:.1f}"
    h = int(hashlib.md5(f"{ss:.9f}|{key}".encode("utf-8")).hexdigest()[:14], 16)
    uc = ((h % 999983) + 0.5) / 999984.0
    uc = min(max(uc, 1e-6), 1.0 - 1e-6)

    # Engaged eta ~ N(0.12, 0.21): centered well below the kill
    # knives (~0.31-0.34). Hot-tail eta is pinned at the engaged
    # median (0.12) with a tighter redraw.
    eta_ctx = eta_base + redraw_mu + redraw_sd * float(ndtri(uc))
    eta_ctx = min(max(eta_ctx, -0.25), 0.70)

    # Context-normalized anti-validity weights.
    w = -val * np.power(val / v_ref, eta_ctx)

    # --- Prestige aversion with QUANTITY DILUTION (flattened) ---
    # The top endorsed tier is amplified by
    #     sigma_eff = 1 + (sigma - 1) * exp(-gap/0.10) * D,
    #     D = 1 / (1 + (n_disc/n0)^1.1),
    # where n_disc counts the cues on which the two options differ.
    # FLATTENED curve (exponent 1.1, n0 ~ 10): small knife-edge
    # contests (3v3, n_disc = 6, D ~ 0.62-0.68) keep most of the
    # amplification -> strong prestige aversion (Exp 11 AL cells,
    # Exp 15 L-cells); large stacks (n_disc = 13-14, D ~ 0.40-0.45)
    # retain SUBSTANTIAL amplification, landing the population
    # between the linear core and full undiluted salience (Exp 16).
    # Single-family trials have no contrast tier: off, so
    # within-family load ladders stay exactly flat.
    tiers = np.unique(val[active])
    if tiers.size >= 2:
        gap = float(tiers[-1] - tiers[-2])
        n_disc = int(np.count_nonzero(ra != rb))
        D = 1.0 / (1.0 + (n_disc / n0) ** 1.1)
        sig_eff = 1.0 + (sigma - 1.0) * math.exp(-gap / 0.10) * D
        if sig_eff > 1.0 + 1e-9:
            top_mask = val >= (tiers[-1] - 1e-9)
            w = np.where(top_mask, w * sig_eff, w)

    # --- Load-scaled attention diffusion ---
    # sd = s0 + s1*sqrt(n_active): small-stack cells stay crisp while
    # complex multi-tier trials acquire within-subject stochasticity
    # (attenuating cross-ladder consistency on 24-feature designs).
    # Per-trial noise averages out within a subject, so it supplies
    # within-context stochasticity without damping population means.
    sd_att = s0 + s1 * math.sqrt(float(active.size))
    att = np.exp(sd_att * np.random.randn(n_features))
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
- mode_seed: [0, 1]
- sigma_sensitive: [1.6, 2.1]
- sigma_sensitive_fraction: [0.4, 0.6]
- dilution_n0: [7, 13]
- noise_s0: [0.02, 0.04]
- noise_s1: [0.012, 0.03]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.1747 (var=0.0108) vs this=-0.2100 (var=0.0048)
- Experiment 2: real=0.2737 (var=0.0930) vs this=0.2381 (var=0.0227)
- Experiment 3: real=0.1227 (var=0.0008) vs this=0.1167 (var=0.0051)
- Experiment 4: real=0.1625 (var=0.0119) vs this=0.2092 (var=0.0220)
- Experiment 5: real=0.6837 (var=0.0643) vs this=0.6162 (var=0.0241)
- Experiment 6: real=0.6300 (var=0.0703) vs this=0.6250 (var=0.0238)
- Experiment 7: real=0.6462 (var=0.0395) vs this=0.6347 (var=0.0219)
- Experiment 8: real=0.6044 (var=0.0653) vs this=0.6356 (var=0.0278)
- Experiment 9: real=0.1963 (var=0.0601) vs this=0.1063 (var=0.0178)
- Experiment 10: real=0.2963 (var=0.0806) vs this=0.3263 (var=0.0579)
- Experiment 11: real=2.8800 (var=0.0015) vs this=2.3200 (var=0.0009)
- Experiment 12: real=0.0913 (var=0.0729) vs this=0.0820 (var=0.0631)
- Experiment 13: real=0.0000 (var=0.4439) vs this=0.0000 (var=0.2898)
- Experiment 14: real=1.4067 (var=0.6946) vs this=1.5820 (var=0.5497)
- Experiment 15: real=0.5760 (var=0.0293) vs this=0.5880 (var=0.0275)
- Experiment 16: real=0.4950 (var=0.0950) vs this=0.5658 (var=0.0371)
- Experiment 17: real=0.2633 (var=0.1550) vs this=0.1842 (var=0.0918)
- Experiment 18: real=0.5539 (var=0.1118) vs this=0.5998 (var=0.0143)


---

### `pi_6` (overall score: 0.774)

**Description**
Context-Normalized Noisy Anti-Validity Integration (CNNAI), knife-avoidance recalibration. Subjects integrate all binary expert ratings additively into a per-option value, but a rating of 1 is comprehended as a DEFECT (sign inversion). The anti-weight of expert j is context-normalized: w_j = -v_j * (v_j / v_ref)^eta, where v_ref is the mean instructed validity of the cues endorsed on the current trial and eta is a per-context elasticity drawn at the level of coarse validity CONTEXT BUCKETS (the mean endorsed validity rounded to 0.1), so all trials sharing a bucket inherit one shared per-subject draw. The elasticity decomposes into a small stable per-subject component (eta_base ~ N(0.02, 0.09)) and a larger per-bucket redraw (~N(0.10, 0.19)), total hard-clipped to [-0.25, 0.70]; the variance ratio fixes latent cross-context consistency at r ~ 0.18 by construction. The engaged eta distribution (mean ~0.12, sd ~0.21) is deliberately centered well BELOW the kill-design knife region (eta ~ 0.31-0.34 where high-validity stacks flip to low-validity stacks), so the engaged bulk behaves SCWI-like in effective curvature while margins on the below-knife side stay large. A modest per-trial multiplicative attention diffusion (sd 0.05) supplies within-context stochasticity, choice is softmax(beta * values) with nearly homogeneous engaged beta (lognormal median ~2.14, log-sd 0.20) plus a narrow symmetric lapse (0.03-0.09), and a 13% true-guesser subpopulation is retained, uncorrelated with eta.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Context-Normalized Noisy Anti-Validity Integration (CNNAI),
    # iter 6 -- knife-avoidance recalibration of the accepted iter-5 base
    # per the iter-5 critic diagnosis. Mechanism family UNCHANGED:
    # additive anti-validity integration, context-referenced elasticity
    # w_j = -v_j * (v_j / v_ref)^eta with v_ref = mean validity of the
    # endorsed cues, per-trial attention diffusion, softmax + symmetric
    # lapse, guesser mixture. Changes vs the accepted iter-5 base:
    #   (A) ETA CENTER LOWERED 0.23 -> 0.12 (eta_base ~ N(0.02, 0.09) plus
    #       per-context redraw ~ N(0.10, 0.19), total clipped [-0.25, 0.70]).
    #       The kill composites (Exps 5/6/8) are near-knife statistics:
    #       their flip knives sit at eta ~ 0.30-0.34, and on the
    #       below-knife side the score margins GROW as eta decreases
    #       (e.g. Exp 5's 2x1.00-vs-5x0.50 cell: margin 0.14 at eta 0.23
    #       vs 0.33 at eta 0.10). Centering at 0.12 both moves engaged
    #       mass below the knives (P(eta < 0.33) ~ 0.83) and enlarges
    #       the margins, lifting the undershooting composites.
    #   (B) ATTENTION DIFFUSION CUT 0.10 -> 0.05: at 0.10 the per-trial
    #       margin noise (sd ~ 0.25 on a 12-cue cell) was comparable to
    #       the near-knife signals (~0.15-0.35), damping below-knife
    #       subjects to p ~ 0.55-0.6 -- the main softening channel on
    #       Exps 5/6/8. Per-trial noise averages out within a subject
    #       (iter-2 critic), so cutting it lifts means without touching
    #       between-subject variance.
    #   (C) CONTEXT KEY COARSENED to the mean endorsed validity rounded
    #       to 0.1 (a bucketed v_ref / ladder identifier). The critic's
    #       'rounded MAX endorsed validity' variant was NOT used: both
    #       Exp-9 ladders ({1.00,0.50} and {1.00,0.60} cells) have max
    #       validity 1.0, so a max-based key would map them to the SAME
    #       draw and re-couple them -- recreating the iter-1 r = 0.44
    #       failure the critic itself warns against. Bucketed mean keeps
    #       them distinct (0.6 vs 0.7) while merging within-experiment
    #       fragmentation (Exp 2: ~2-3 keys instead of ~6; Exp 5: 2 keys),
    #       so per-subject composites stop averaging over many
    #       independent eta draws.
    #   (D) EXP-9 COUPLING CHANNELS CLOSED: engaged epsilon narrowed
    #       0.02-0.17 -> 0.03-0.09 (stable lapse was the dominant
    #       residual cross-ladder trait), eta_base sd cut 0.13 -> 0.09
    #       (latent r = 0.0081/(0.0081+0.0361) ~ 0.18), beta median
    #       trimmed 2.34 -> 2.14 with log-sd 0.25 -> 0.20 (guards Exps
    #       1/3/10 against the noise cut, and removes the last sizeable
    #       stable-trait correlation channel).
    #   (E) v_ref STAYS MEAN-REFERENCED (not max): the weight ratio
    #       w(v_h)/w(v_l) = (v_h/v_l)^(1+eta) is invariant to v_ref, and
    #       max-referencing multiplies every margin by v_max^(-eta) <= 1,
    #       which would DEFLATE exactly the near-knife margins the
    #       composites need. The critic's goal (steeper high-validity
    #       region) is achieved via (A)+(B) instead.
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
    if u < 0.13:
        # Guesser subpopulation (~13%): beta ~ 0 -> uniform core on
        # every trial; elevated lapse. These subjects sit at ~0.5 on
        # every diagnostic cell and dilute population means/slopes.
        t = u / 0.13
        beta = 0.02 + 0.08 * t            # 0.02 .. 0.10
        epsilon = 0.12 + 0.13 * ps       # 0.12 .. 0.25
        eta_base = 0.10                  # irrelevant at beta ~ 0
    else:
        # Engaged subpopulation (~87%). Small stable elasticity
        # component (sd 0.09) paired with a larger per-context redraw
        # (sd 0.19): the variance ratio fixes latent cross-context
        # consistency at r ~ 0.18 by construction. The stable lapse
        # range is NARROW (0.03-0.09) so it can no longer act as a
        # hidden coupling channel across contexts, and beta is nearly
        # homogeneous (lognormal sd 0.20).
        zs = min(max(es, 1e-6), 1.0 - 1e-6)
        eta_base = 0.02 + 0.09 * float(ndtri(zs))
        eta_base = min(max(eta_base, -0.15), 0.20)

        beta = math.exp(0.76 + 0.20 * float(ndtri(bs)))  # median ~2.14
        beta = min(max(beta, 0.9), 4.5)
        epsilon = 0.03 + 0.06 * ps       # 0.03 .. 0.09

    # --- Context: bucketed mean endorsed validity. ---
    active = np.flatnonzero((ra > 0.5) | (rb > 0.5))
    if active.size == 0:
        # Nothing endorsed by either option: no evidence either way.
        return np.full(2, 0.5)

    # v_ref = mean validity of the cues actually displayed (endorsed)
    # this trial; the context key is v_ref rounded to one decimal, so
    # all cells of one validity ladder share a single per-subject eta
    # draw while different ladders (e.g. Exp 9's 0.50-ladder at bucket
    # 0.6 vs its 0.60-ladder at bucket 0.7) draw independently.
    v_ref = float(np.mean(val[active]))
    bucket = float(np.round(v_ref, 1))
    key = f"{bucket:.1f}"
    h = int(hashlib.md5(f"{ss:.9f}|{key}".encode("utf-8")).hexdigest()[:14], 16)
    uc = ((h % 999983) + 0.5) / 999984.0
    uc = min(max(uc, 1e-6), 1.0 - 1e-6)

    # Per-context redraw, centered low so the engaged bulk sits below
    # the kill knives (~0.31-0.34) with large below-knife margins:
    # engaged eta ~ N(0.12, 0.21), P(eta < 0.33) ~ 0.83.
    eta_ctx = eta_base + 0.10 + 0.19 * float(ndtri(uc))
    eta_ctx = min(max(eta_ctx, -0.25), 0.70)

    # Context-normalized anti-validity weights.
    w = -val * np.power(val / v_ref, eta_ctx)

    # Per-trial multiplicative attention diffusion on cue weights,
    # sd cut 0.10 -> 0.05: at 0.10 the margin noise on near-knife cells
    # was comparable to the signal and was the main softening channel
    # on the Exp 5/6/8 composites.
    att = np.exp(0.05 * np.random.randn(n_features))
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

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.1747 (var=0.0108) vs this=-0.1968 (var=0.0072)
- Experiment 2: real=0.2737 (var=0.0930) vs this=0.1650 (var=0.0201)
- Experiment 3: real=0.1227 (var=0.0008) vs this=0.1497 (var=0.0011)
- Experiment 4: real=0.1625 (var=0.0119) vs this=0.1342 (var=0.0135)
- Experiment 5: real=0.6837 (var=0.0643) vs this=0.6150 (var=0.0208)
- Experiment 6: real=0.6300 (var=0.0703) vs this=0.6208 (var=0.0179)
- Experiment 7: real=0.6462 (var=0.0395) vs this=0.6581 (var=0.0250)
- Experiment 8: real=0.6044 (var=0.0653) vs this=0.7011 (var=0.0213)
- Experiment 9: real=0.1963 (var=0.0601) vs this=0.1113 (var=0.0229)
- Experiment 10: real=0.2963 (var=0.0806) vs this=0.3438 (var=0.0509)
- Experiment 11: real=2.8800 (var=0.0015) vs this=0.3200 (var=0.0016)
- Experiment 12: real=0.0913 (var=0.0729) vs this=0.0000 (var=0.0000)
- Experiment 13: real=0.0000 (var=0.4439) vs this=0.0000 (var=0.2794)
- Experiment 14: real=1.4067 (var=0.6946) vs this=1.3546 (var=0.2588)
- Experiment 15: real=0.5760 (var=0.0293) vs this=0.4600 (var=0.0092)
- Experiment 16: real=0.4950 (var=0.0950) vs this=0.6958 (var=0.0184)
- Experiment 17: real=0.2633 (var=0.1550) vs this=0.3042 (var=0.0547)
- Experiment 18: real=0.5539 (var=0.1118) vs this=0.7213 (var=0.0157)


---

### `pi_3` (overall score: 0.767)

**Description**
Signed-Cue Weighted Integration (SCWI). When choosing between two options described by binary expert ratings, subjects do not use a lexicographic cascade (TTB) and do not merely count wins (Tallying). Instead they integrate ALL cues additively into a single subjective value per option — but the mapping from a rating to evidence carries a systematic SIGN INVERSION: a rating of 1 is treated as a defect/criticism rather than an endorsement (a comprehension inversion of the rating direction). Moreover, the inversion is not uniform across cues: the subjective weight of cue j is w_j = -(v_j + tau * v_j^2), i.e. an anti-validity weighting that grows SUPRALINEARLY with the instructed validity — the endorsements of the most trusted experts are avoided most strongly, while low-validity experts are nearly ignored (their anti-weight is small in magnitude). The option with the higher (less negative) signed value is preferred, with softmax noise (inverse temperature beta) and an independent lapse (epsilon) capturing response variability. This single task-invariant mechanism predicts both experimental signatures simultaneously: in Experiment 1, the top-cue-favored option carries more 1s and hence more 'defects', so P(choose the top-cue-favored option) DECREASES with its tally margin (negative slope); in Experiment 2, on margin-1 trials where TTB and the tally disagree the two options' signed values are close together (so noise frequently flips the subject onto the tally winner), whereas when they agree the value gap is large (so the subject reliably anti-follows) — yielding p_disagree > p_agree, a positive metric.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Signed-Cue Weighted Integration (SCWI).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Each cue contributes signed evidence w_j = -(v_j + tau * v_j^2)
    # per unit rating (anti-validity weighting, supralinear in the
    # instructed validity). Option values are the cue-weighted sums;
    # choice is a numerically stable softmax over beta * values, mixed
    # with a symmetric lapse. History is ignored (no feedback).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SCWI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    raw = parameters.get("validities")
    if isinstance(raw, str):
        s = raw.strip().strip('[]()')
        val = np.array([float(t) for t in s.replace(',', ' ').split()])
    else:
        val = np.asarray(raw, dtype=float).ravel()
    if val.size != n_features:
        # Defensive fallback: uniform mid-range validities.
        val = np.full(n_features, 0.75)

    beta = float(parameters["beta"])
    tau = float(parameters["tau"])
    epsilon = float(parameters["epsilon"])

    # Signed (anti-validity) cue weights: a rating of 1 is evidence
    # AGAINST an option, and the inversion strength grows supralinearly
    # with the instructed validity of the expert.
    w = -(val + tau * np.square(val))

    scores = np.array([float(np.dot(w, stim[0])), float(np.dot(w, stim[1]))])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
    return np.random.choice(len(probabilities), p=probabilities)

`parameters`:
- beta: [1.1, 1.7]
- tau: [0.25, 0.45]
- epsilon: [0.05, 0.15]
- validities: validities

**Per-experiment fit (real vs this theory's metric value):**
- Experiment 1: real=-0.1747 (var=0.0108) vs this=-0.1877 (var=0.0014)
- Experiment 2: real=0.2737 (var=0.0930) vs this=0.1862 (var=0.0095)
- Experiment 3: real=0.1227 (var=0.0008) vs this=0.1378 (var=0.0002)
- Experiment 4: real=0.1625 (var=0.0119) vs this=0.1608 (var=0.0029)
- Experiment 5: real=0.6837 (var=0.0643) vs this=0.5531 (var=0.0069)
- Experiment 6: real=0.6300 (var=0.0703) vs this=0.5687 (var=0.0053)
- Experiment 7: real=0.6462 (var=0.0395) vs this=0.6412 (var=0.0050)
- Experiment 8: real=0.6044 (var=0.0653) vs this=0.6056 (var=0.0099)
- Experiment 9: real=0.1963 (var=0.0601) vs this=0.2390 (var=0.0208)
- Experiment 10: real=0.2963 (var=0.0806) vs this=0.3713 (var=0.0215)
- Experiment 11: real=2.8800 (var=0.0015) vs this=1.0800 (var=0.0009)
- Experiment 12: real=0.0913 (var=0.0729) vs this=0.0000 (var=0.0000)
- Experiment 13: real=0.0000 (var=0.4439) vs this=-0.0563 (var=0.3210)
- Experiment 14: real=1.4067 (var=0.6946) vs this=1.1054 (var=0.0832)
- Experiment 15: real=0.5760 (var=0.0293) vs this=0.4387 (var=0.0088)
- Experiment 16: real=0.4950 (var=0.0950) vs this=0.6808 (var=0.0116)
- Experiment 17: real=0.2633 (var=0.1550) vs this=0.3925 (var=0.0292)
- Experiment 18: real=0.5539 (var=0.1118) vs this=0.7173 (var=0.0086)


## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision — only ACCEPTED candidates have ever been used as the base for a subsequent iteration. Use this together with PRIOR FEEDBACK ITERATIONS below to grade which past critic advice actually paid off.

- iter 1: loss=0.0805 -> ACCEPTED
Running-best (last ACCEPTED) base: iter 1 at loss=0.0805 — this is the source shown verbatim below under `## PREVIOUS CANDIDATE (this loop)`. Push the next edit's loss strictly below that floor or the gate will reject it.

## EXPERIMENTAL RESULTS
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
**Previous candidate values (this loop):**
  - iter 1 (most recent): -0.2055 (var=0.0036) (Δ vs real -0.0308)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0023 (var=0.0008)
- pi_2: 0.1952 (var=0.0019)
- pi_3: -0.1877 (var=0.0014)
- pi_4: -0.1883 (var=0.0047)
- pi_3_1: -0.2228 (var=0.0055)
- pi_5: -0.1898 (var=0.0057)
- pi_6: -0.1968 (var=0.0072)
- pi_7: -0.2270 (var=0.0080)
- pi_8: -0.1823 (var=0.0088)
- pi_9: -0.2100 (var=0.0048)

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
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.2194 (var=0.0123) (Δ vs real -0.0544)
**Other theories' values on this metric (for reference):**
- pi_2: -0.0006 (var=0.0060)
- pi_1: -0.6981 (var=0.0507)
- pi_3: 0.1862 (var=0.0095)
- pi_4: 0.2669 (var=0.0115)
- pi_3_1: 0.1713 (var=0.0139)
- pi_5: 0.1550 (var=0.0256)
- pi_6: 0.1650 (var=0.0201)
- pi_7: 0.2425 (var=0.0162)
- pi_8: 0.2213 (var=0.0129)
- pi_9: 0.2381 (var=0.0227)

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
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1279 (var=0.0037) (Δ vs real +0.0052)
**Other theories' values on this metric (for reference):**
- pi_3: 0.1378 (var=0.0002)
- pi_2: -0.1263 (var=0.0006)
- pi_1: -0.0888 (var=0.0010)
- pi_4: 0.1448 (var=0.0005)
- pi_3_1: 0.1444 (var=0.0038)
- pi_5: 0.1165 (var=0.0040)
- pi_6: 0.1497 (var=0.0011)
- pi_7: 0.1478 (var=0.0035)
- pi_8: 0.1421 (var=0.0027)
- pi_9: 0.1167 (var=0.0051)

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
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1713 (var=0.0136) (Δ vs real +0.0088)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8538 (var=0.0090)
- pi_3: 0.1608 (var=0.0029)
- pi_1: 0.6796 (var=0.0042)
- pi_4: 0.1696 (var=0.0044)
- pi_3_1: 0.1388 (var=0.0265)
- pi_5: 0.1979 (var=0.0276)
- pi_6: 0.1342 (var=0.0135)
- pi_7: 0.1325 (var=0.0322)
- pi_8: 0.1683 (var=0.0164)
- pi_9: 0.2092 (var=0.0220)

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
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.5794 (var=0.0268) (Δ vs real -0.1044)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5531 (var=0.0069)
- pi_4: 0.2281 (var=0.0125)
- pi_1: 0.8350 (var=0.0129)
- pi_2: 0.1300 (var=0.0103)
- pi_3_1: 0.6506 (var=0.0203)
- pi_5: 0.6019 (var=0.0710)
- pi_6: 0.6150 (var=0.0208)
- pi_7: 0.5531 (var=0.0405)
- pi_8: 0.5650 (var=0.0288)
- pi_9: 0.6162 (var=0.0241)

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
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.6179 (var=0.0220) (Δ vs real -0.0121)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2267 (var=0.0083)
- pi_3: 0.5687 (var=0.0053)
- pi_1: 0.8783 (var=0.0092)
- pi_2: 0.1263 (var=0.0072)
- pi_3_1: 0.6646 (var=0.0194)
- pi_5: 0.7004 (var=0.0351)
- pi_6: 0.6208 (var=0.0179)
- pi_7: 0.5408 (var=0.0201)
- pi_8: 0.6433 (var=0.0205)
- pi_9: 0.6250 (var=0.0238)

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
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.6641 (var=0.0244) (Δ vs real +0.0178)
**Other theories' values on this metric (for reference):**
- pi_3_1: 0.6975 (var=0.0214)
- pi_4: 0.2169 (var=0.0147)
- pi_1: 0.8762 (var=0.0079)
- pi_2: 0.1344 (var=0.0070)
- pi_3: 0.6412 (var=0.0050)
- pi_5: 0.6056 (var=0.0640)
- pi_6: 0.6581 (var=0.0250)
- pi_7: 0.5931 (var=0.0314)
- pi_8: 0.6562 (var=0.0253)
- pi_9: 0.6347 (var=0.0219)

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
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.6056 (var=0.0267) (Δ vs real +0.0011)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2078 (var=0.0291)
- pi_3_1: 0.7078 (var=0.0349)
- pi_1: 0.8611 (var=0.0123)
- pi_2: 0.1200 (var=0.0071)
- pi_3: 0.6056 (var=0.0099)
- pi_5: 0.6122 (var=0.0654)
- pi_6: 0.7011 (var=0.0213)
- pi_7: 0.5433 (var=0.0334)
- pi_8: 0.5767 (var=0.0220)
- pi_9: 0.6356 (var=0.0278)

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
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.4739 (var=0.0179) (Δ vs real +0.2776)
**Other theories' values on this metric (for reference):**
- pi_3_1: 0.3684 (var=0.0180)
- pi_5: 0.9529 (var=0.0191)
- pi_1: 0.9048 (var=0.0084)
- pi_2: 0.8735 (var=0.0093)
- pi_3: 0.2390 (var=0.0208)
- pi_4: 0.8613 (var=0.0179)
- pi_6: 0.1113 (var=0.0229)
- pi_7: 0.0213 (var=0.0715)
- pi_8: 0.1959 (var=0.0248)
- pi_9: 0.1063 (var=0.0178)

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
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.4125 (var=0.0346) (Δ vs real +0.1162)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1713 (var=0.0509)
- pi_3_1: 0.4531 (var=0.0906)
- pi_1: 0.0181 (var=0.0129)
- pi_2: 0.0131 (var=0.0090)
- pi_3: 0.3713 (var=0.0215)
- pi_4: 0.1175 (var=0.0341)
- pi_6: 0.3438 (var=0.0509)
- pi_7: 0.3656 (var=0.0729)
- pi_8: 0.3463 (var=0.0382)
- pi_9: 0.3263 (var=0.0579)

### Experiment 11
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def _to_vec(x):
        if isinstance(x, str):
            s = x.strip().strip('[]()')
            toks = [t for t in s.replace(',', ' ').split() if t != '']
            return np.array([float(t) for t in toks], dtype=float)
        return np.asarray(x, dtype=float).ravel()

    A = np.array([_to_vec(x) for x in data['option_a_ratings']], dtype=float)
    B = np.array([_to_vec(x) for x in data['option_b_ratings']], dtype=float)

    r = np.asarray(data['response'])
    try:
        resp = r.astype(float)
    except Exception:
        resp = np.array([0.0 if str(v).strip().upper() in ('0', 'A') else 1.0 for v in r],
                        dtype=float)

    subj = np.asarray(data['subject_id'])

    n_feat = A.shape[1]
    val = np.array([0.9] * 6 + [0.85] * 11 + [0.5] * 5 + [1.0] * 6, dtype=float)
    if val.size < n_feat:
        val = np.concatenate([val, np.full(n_feat - val.size, 0.5)])
    val = val[:n_feat]

    hi = np.isclose(val, 0.9)
    mid = np.isclose(val, 0.85)

    A_hi = A[:, hi].sum(axis=1)
    A_mid = A[:, mid].sum(axis=1)
    B_hi = B[:, hi].sum(axis=1)
    B_mid = B[:, mid].sum(axis=1)
    A_other = A.sum(axis=1) - A_hi - A_mid
    B_other = B.sum(axis=1) - B_hi - B_mid

    # AL (near-knife, equal-count) cells: one option carries exactly 3 endorsements
    # on the 0.90-validity experts, the other exactly 3 on the 0.85-validity
    # experts, and every other expert is silent for both options.
    al = ((A_other == 0) & (B_other == 0)
          & (((A_hi == 3) & (A_mid == 0) & (B_hi == 0) & (B_mid == 3))
             | ((A_hi == 0) & (A_mid == 3) & (B_hi == 3) & (B_mid == 0))))

    subjects = np.unique(subj)

    if subjects.size > 1:
        # Population statistic: mean signed excess of anti-validity choices on
        # the AL cells (per subject: 2*#anti-choices - #AL-trials, in [-6, +6]).
        excess = []
        for s in subjects:
            m = al & (subj == s)
            k = int(m.sum())
            if k == 0:
                continue
            # anti-validity-preferred option = the one whose 1s sit on the
            # LOWER-validity (0.85) experts (a rating of 1 is a defect).
            anti = np.where(A_mid[m] == 3, 1.0 - resp[m], resp[m])
            excess.append(2.0 * float(anti.sum()) - float(k))
        if len(excess) == 0:
            return 0.0
        return float(np.mean(excess))
    else:
        # Single-subject slice: the population excess has no single-subject
        # analogue, so return the subject's position-bias baseline (mean A
        # response). By the design's mirror-pair symmetry this is pinned near
        # 0.5 with only binomial noise under BOTH theories, i.e. it is the
        # stable per-subject quantity against which the population-level
        # excess is defined.
        return float(np.mean(resp))
```

**Observed (real) value:** 2.8800 (var=0.0015)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 2.8800 (var=0.0013) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_3_1: 1.4400 (var=0.0007)
- pi_6: 0.3200 (var=0.0016)
- pi_1: -4.3200 (var=0.0012)
- pi_2: 0.2000 (var=0.0010)
- pi_3: 1.0800 (var=0.0009)
- pi_4: 1.7600 (var=0.0014)
- pi_5: 0.7200 (var=0.0008)
- pi_7: 1.7600 (var=0.0012)
- pi_8: 3.6000 (var=0.0010)
- pi_9: 2.3200 (var=0.0009)

### Experiment 12
**Design**
  A=[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    df = data.reset_index(drop=True)

    # --- Design constants for this experiment --------------------------
    # 12 experts: validities [0.9 x 4, 0.6 x 8].
    VALID = np.array([0.9] * 4 + [0.6] * 8)
    N_F = VALID.size

    # Reference anti-validity defect magnitudes: a rating of 1 is treated
    # as a defect with weight -d_j. The reference ratio d(0.9)/d(0.6)
    # = 1.58 sits between SCWI's whole-box ratio range (1.562-1.651 over
    # tau in [0.15, 0.42]) and CNNAI's median ratio (1.5^(1+0.12) ~ 1.58),
    # so the per-trial signed margins fed to the fit are theory-neutral.
    D_HI, D_LO = 1.0, 0.633
    D_REF = np.where(VALID > 0.75, D_HI, D_LO)

    # Grid for the per-subject sensitivity (softmax inverse temperature)
    # maximum-likelihood fit, and the soft-threshold ramp applied to it.
    # The ramp [3.0, 4.2] targets exactly the region where the two
    # theories' beta distributions are disjoint: SCWI's deterministic
    # regime has beta ~ U[3.8, 6.0] (34% of subjects), while CNNAI's
    # engaged beta is lognormal(median 2.14, log-sd 0.20) clipped to
    # [0.9, 4.5] with 98% of engaged subjects below 3.2.
    GRID = np.linspace(0.0, 10.0, 501)
    RAMP_LO, RAMP_HI = 3.0, 4.2

    def _vec(raw):
        if raw is None:
            return None
        if isinstance(raw, str):
            s = raw.strip().strip('[]()')
            toks = [t for t in s.replace(',', ' ').split() if t != '']
            if not toks:
                return None
            return np.array([float(t) for t in toks])
        try:
            arr = np.asarray(raw, dtype=float).ravel()
        except Exception:
            return None
        return arr if arr.size > 0 else None

    def _subject_score(sub):
        ms = []
        ys = []
        for a_cell, b_cell, resp in zip(
            sub['option_a_ratings'].values,
            sub['option_b_ratings'].values,
            sub['response'].values,
        ):
            A = _vec(a_cell)
            B = _vec(b_cell)
            if A is None or B is None or A.size == 0 or A.size != B.size:
                continue
            if A.size == N_F:
                d = D_REF
            else:
                d = np.full(A.size, 0.8)
            w = -d
            va = float(np.dot(w, A))
            vb = float(np.dot(w, B))
            m = abs(vb - va)
            if m < 1e-9 or m > 2.0:
                # identical pairs (ties) and the all-ones-vs-all-zeros
                # anchor screens carry no usable sensitivity info
                # (both theories put them at ~0.5 / lapse rate), and the
                # ~9-unit anchor margin would corrupt the fit.
                continue
            b_pred = vb > va
            r = float(resp)
            y = 1.0 if ((r > 0.5) == b_pred) else 0.0
            ms.append(m)
            ys.append(y)
        if len(ms) < 10:
            return 0.0
        m = np.asarray(ms, dtype=float)            # (n_trials,)
        y = np.asarray(ys, dtype=float)[:, None]   # (n_trials, 1)  <-- broadcast fix
        z = np.outer(m, GRID)                      # (n_trials, n_grid)
        # Bernoulli log-likelihood of P(y=1) = sigmoid(beta * margin):
        # ll(beta) = -sum_i [ y_i * log(1+e^{-z}) + (1-y_i) * log(1+e^{z}) ]
        ll = -(y * np.logaddexp(0.0, -z)
               + (1.0 - y) * np.logaddexp(0.0, z)).sum(axis=0)
        bhat = float(GRID[int(np.argmax(ll))])
        g = (bhat - RAMP_LO) / (RAMP_HI - RAMP_LO)
        return float(min(max(g, 0.0), 1.0))

    subjects = pd.unique(df['subject_id'])
    if len(subjects) == 0:
        return 0.0
    vals = []
    for sid in subjects:
        sub = df[df['subject_id'] == sid]
        vals.append(_subject_score(sub))
    return float(np.mean(vals))
```

**Observed (real) value:** 0.0913 (var=0.0729)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0460 (var=0.0286) (Δ vs real -0.0453)
**Other theories' values on this metric (for reference):**
- pi_6: 0.0000 (var=0.0000)
- pi_3_1: 0.3587 (var=0.2287)
- pi_1: 0.0000 (var=0.0000)
- pi_2: 0.0000 (var=0.0000)
- pi_3: 0.0000 (var=0.0000)
- pi_4: 0.0000 (var=0.0000)
- pi_5: 0.0823 (var=0.0589)
- pi_7: 0.0777 (var=0.0589)
- pi_8: 0.0623 (var=0.0489)
- pi_9: 0.0820 (var=0.0631)

### Experiment 13
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Median across subjects of the within-subject LOG-ODDS CONFIDENCE DROP
    on the tied-filler load ladder.

    For every ladder trial (one validity family active, options differ by
    1-2 'defect' ratings, plus f tied ratings shared by both options) we
    record whether the subject chose the option with FEWER 1s (the
    'defect-free' option).  Trials are split by the defect option's total
    one-count on its active family:
      - LOW  (sub-knee):  family H (0.90 cues, first 9) with <=3 ones,
                          family L (0.60 cues) with <=5 ones.  For the whole
                          CAVI-K parameter box the weighted loads (count *
                          (v + tau*v^2), tau in [0.35,0.65]) stay below the
                          compression knee 4.45, so CAVI-K is exactly linear
                          here too.
      - HIGH (above-knee): family H with >=4 ones, family L with >=8 ones.
                          Loads are 4.7-13.9, strictly above 4.45 for every
                          admissible tau, so CAVI-K's power compression
                          (kappa in [0.55,0.85], always < 1) shrinks the
                          margin; CNNAI, being exactly linear in the
                          ratings, is unaffected.
    Ties (A==B) and anchor trials (both validity families active) are
    excluded.  Per subject: p_low and p_high are the smoothed proportions
    of defect-free choices in each pool; the statistic is
        logit(p_low) - logit(p_high).
    The metric returns the MEDIAN of this drop across subjects.

    CNNAI: within a family every rung has identical active-validity set
    (same bucket key, same eta draw), and tied cues cancel exactly in the
    score difference, so p_low and p_high are draws from the SAME
    distribution -> the drop is symmetric around exactly 0 for every
    subject (mirrored pairs also cancel side bias).
    CAVI-K: every non-guesser subject has kappa < 1 and top-rung loads far
    above the knee, so the drop is strictly positive (engaged ~1 logit,
    deterministic tail ~0.5, guessers ~0).
    """
    import numpy as np
    import pandas as pd

    df = data.reset_index(drop=True)

    # --- response as 0 (chose A) / 1 (chose B) ---
    try:
        resp = pd.to_numeric(df['response']).to_numpy(dtype=float)
    except Exception:
        resp = np.array(
            [0.0 if str(r).strip().upper().startswith('A') else 1.0
             for r in df['response']]
        )

    n_rows = len(df)
    grp = np.zeros(n_rows, dtype=int)      # 0 = ignore, 1 = LOW, 2 = HIGH
    clean = np.zeros(n_rows, dtype=float)  # 1 = chose the fewer-ones option

    for i in range(n_rows):
        a = np.asarray(df.at[i, 'option_a_ratings'], dtype=float).ravel()
        b = np.asarray(df.at[i, 'option_b_ratings'], dtype=float).ravel()
        if a.size == 0 or a.size != b.size:
            continue
        na = float(a.sum())
        nb = float(b.sum())
        if na == nb:
            continue  # exact-tie trials (21, 22)
        hs = min(9, a.size)  # first 9 experts at 0.90, remaining at 0.60
        aH = float(a[:hs].sum()); aL = float(a[hs:].sum())
        bH = float(b[:hs].sum()); bL = float(b[hs:].sum())
        totH = aH + bH
        totL = aL + bL
        if (totH > 0 and totL > 0) or (totH == 0 and totL == 0):
            continue  # anchor trials spanning both families (23, 24)
        if totH > 0:
            cnt = max(aH, bH)
            is_high = cnt >= 4.0  # >=4 ones on 0.90-cues: load > knee 4.45 for all tau
        else:
            cnt = max(aL, bL)
            is_high = cnt >= 8.0  # >=8 ones on 0.60-cues: load > knee; <=5 stays below
        r = resp[i]
        chose_clean = (r == 0.0) if (na < nb) else (r == 1.0)
        grp[i] = 2 if is_high else 1
        clean[i] = 1.0 if chose_clean else 0.0

    def _logit(p):
        p = min(max(p, 0.02), 0.98)
        return float(np.log(p / (1.0 - p)))

    subj = df['subject_id'].to_numpy()
    drops = []
    for s in pd.unique(subj):
        m = (subj == s)
        lo = m & (grp == 1)
        hi = m & (grp == 2)
        nl = int(lo.sum())
        nh = int(hi.sum())
        if nl < 5 or nh < 5:
            continue
        pl = (float(clean[lo].sum()) + 0.5) / (nl + 1.0)
        ph = (float(clean[hi].sum()) + 0.5) / (nh + 1.0)
        drops.append(_logit(pl) - _logit(ph))

    if len(drops) == 0:
        return 0.0
    return float(np.median(drops))
```

**Observed (real) value:** 0.0000 (var=0.4439)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.0000 (var=0.3747) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_7: 0.8108 (var=0.7400)
- pi_6: 0.0000 (var=0.2794)
- pi_1: 0.0000 (var=0.5140)
- pi_2: 0.0000 (var=0.5995)
- pi_3: -0.0563 (var=0.3210)
- pi_4: 0.0000 (var=0.3165)
- pi_3_1: -0.1555 (var=0.3226)
- pi_5: 0.0000 (var=0.3782)
- pi_8: 0.1640 (var=0.2822)
- pi_9: 0.0000 (var=0.2898)

### Experiment 14
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # Fixed design: 25 experts, 5 at each validity 1.00/0.90/0.75/0.60/0.50
    VALS = np.array([1.0] * 5 + [0.9] * 5 + [0.75] * 5 + [0.6] * 5 + [0.5] * 5)

    def _parse(cell):
        if isinstance(cell, str):
            toks = [t for t in cell.strip().strip('[]() ').replace(',', ' ').split() if t != '']
            return np.array([float(t) for t in toks])
        return np.asarray(cell, dtype=float).ravel()

    def _subject_slope(df):
        gs, ys = [], []
        for _, row in df.iterrows():
            try:
                a = _parse(row['option_a_ratings'])
                b = _parse(row['option_b_ratings'])
            except Exception:
                continue
            if a.size != VALS.size or b.size != VALS.size:
                continue
            na = int(np.sum(a > 0.5))
            nb = int(np.sum(b > 0.5))
            # pure count-discrimination ladder trial: a {2,1} split of ones
            if (na + nb) != 3 or na * nb != 2:
                continue
            active = np.flatnonzero((a > 0.5) | (b > 0.5))
            if active.size == 0:
                continue
            v = VALS[active]
            # all endorsed cues must come from one validity family
            if (v.max() - v.min()) > 1e-9:
                continue
            vv = float(v[0])
            # outcome: 1 = chose the option with FEWER endorsed cues (fewer defects)
            if na == 2:
                y = 1 if int(row['response']) == 1 else 0
            else:
                y = 1 if int(row['response']) == 0 else 0
            # theory-neutral severity index: midpoint of linear (v) and
            # fully quadratic (v + v^2) anti-validity maps
            gs.append(vv + 0.5 * vv * vv)
            ys.append(float(y))
        if len(gs) < 12:
            return 0.5
        g = np.asarray(gs, dtype=float)
        y = np.asarray(ys, dtype=float)
        # logistic regression THROUGH THE ORIGIN: P(fewer-defect) = sigmoid(b * g(v))
        b_hat = 1.0
        for _ in range(200):
            z = np.clip(b_hat * g, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-z))
            w = p * (1.0 - p)
            grad = float(np.sum(g * (y - p)))
            hess = float(np.sum(g * g * w))
            if hess < 1e-10:
                break
            step = grad / hess
            b_hat += step
            if b_hat > 10.0:
                b_hat = 10.0
            elif b_hat < -10.0:
                b_hat = -10.0
            if abs(step) < 1e-10:
                break
        # winsorize: floor bounds pure-noise slopes of chance-level subjects,
        # cap bounds ceiling-separated (near-deterministic) subjects
        return float(min(max(b_hat, 0.5), 3.5))

    vals = []
    for _, df in data.groupby('subject_id', sort=False):
        vals.append(_subject_slope(df))
    if not vals:
        return 0.5
    return float(np.mean(vals))
```

**Observed (real) value:** 1.4067 (var=0.6946)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 1.2619 (var=0.5805) (Δ vs real -0.1448)
**Other theories' values on this metric (for reference):**
- pi_6: 1.3546 (var=0.2588)
- pi_7: 2.1329 (var=0.9142)
- pi_1: 0.5000 (var=0.0000)
- pi_2: 0.5000 (var=0.0000)
- pi_3: 1.1054 (var=0.0832)
- pi_4: 1.2978 (var=0.1102)
- pi_3_1: 1.9258 (var=1.2285)
- pi_5: 1.3617 (var=0.5233)
- pi_8: 1.4953 (var=0.3606)
- pi_9: 1.5820 (var=0.5497)

### Experiment 15
**Design**
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Fixed design-time validities (n_features = 20).
    VALIDITIES = np.array([
        0.95, 0.9, 0.9, 0.9, 0.9, 0.9,
        0.85, 0.85, 0.85, 0.85, 0.85,
        0.8, 0.8, 0.8,
        0.75, 0.75, 0.75,
        0.7, 0.7, 0.7,
    ])

    def _parse(cell):
        if cell is None:
            return None
        if isinstance(cell, str):
            s = cell.strip().strip('[]()')
            toks = [t for t in s.replace(',', ' ').split() if t != '']
            if not toks:
                return None
            return np.array([float(t) for t in toks])
        try:
            arr = np.asarray(cell, dtype=float).ravel()
        except Exception:
            return None
        return arr if arr.size == VALIDITIES.size else None

    def _sig(cell):
        # Signature = sorted tuple of validities of endorsed (rating=1) experts.
        arr = _parse(cell)
        if arr is None:
            return None
        m = arr > 0.5
        if not np.any(m):
            return ()
        return tuple(sorted(round(float(v), 2) for v in VALIDITIES[m]))

    # Amplifier-critical cells: (sig_A, sig_B) -> signature of the side that
    # BOTH theories' anti-validity core prefers (the lower-validity stack).
    #   L05   : 3x0.90 vs 3x0.85            (near-knife gap-0.05 ladder)
    #   M05   : 3x0.85 vs 3x0.80            (near-knife gap-0.05 M-ladder)
    #   TLC05 : (0.7+3x0.90) vs (0.7+3x0.85) (tied low filler, gap still 0.05)
    #   X34   : 3x0.90 vs 4x0.85            (count-vs-prestige, knife at 3v4)
    #   X45   : 4x0.90 vs 5x0.85            (count-vs-prestige, knife at 4v5)
    s9x3 = (0.9, 0.9, 0.9)
    s9x4 = (0.9, 0.9, 0.9, 0.9)
    s85x3 = (0.85, 0.85, 0.85)
    s85x4 = (0.85, 0.85, 0.85, 0.85)
    s85x5 = (0.85, 0.85, 0.85, 0.85, 0.85)
    s8x3 = (0.8, 0.8, 0.8)
    tlc_hi = (0.7, 0.9, 0.9, 0.9)
    tlc_lo = (0.7, 0.85, 0.85, 0.85)

    TARGETS = {
        # L05 (both orientations / mirrors)
        (s9x3, s85x3): s85x3,
        (s85x3, s9x3): s85x3,
        # M05
        (s85x3, s8x3): s8x3,
        (s8x3, s85x3): s8x3,
        # TLC05 (tied 0.70 filler cancels; contest is 3x0.90 vs 3x0.85)
        (tlc_hi, tlc_lo): tlc_lo,
        (tlc_lo, tlc_hi): tlc_lo,
        # X34
        (s9x3, s85x4): s85x4,
        (s85x4, s9x3): s85x4,
        # X45
        (s9x4, s85x5): s85x5,
        (s85x5, s9x4): s85x5,
    }

    hits = []
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()
    for a_cell, b_cell, r in zip(a_col, b_col, r_col):
        sa = _sig(a_cell)
        sb = _sig(b_cell)
        if sa is None or sb is None:
            continue
        tgt = TARGETS.get((sa, sb))
        if tgt is None:
            continue
        try:
            ri = int(r)
        except Exception:
            continue
        chosen_sig = sb if ri == 1 else sa
        hits.append(1.0 if chosen_sig == tgt else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.5760 (var=0.0293)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.6293 (var=0.0322) (Δ vs real +0.0533)
**Other theories' values on this metric (for reference):**
- pi_8: 0.7653 (var=0.0162)
- pi_6: 0.4600 (var=0.0092)
- pi_1: 0.1520 (var=0.0102)
- pi_2: 0.6507 (var=0.0092)
- pi_3: 0.4387 (var=0.0088)
- pi_4: 0.4887 (var=0.0058)
- pi_3_1: 0.4313 (var=0.0107)
- pi_5: 0.4807 (var=0.0102)
- pi_7: 0.4840 (var=0.0074)
- pi_9: 0.5880 (var=0.0275)

### Experiment 16
**Design**
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import re
    import numpy as np

    # Instructed validities, index-aligned with the 31 rating slots.
    VAL = np.array([0.90] * 7 + [0.85] * 6 + [0.82, 0.80, 0.78, 0.76]
                   + [0.75] * 7 + [0.50] * 7, dtype=float)
    NF = int(VAL.size)
    ETA = 0.12      # engaged eta center of BOTH theories
    SIGMA = 1.45    # SACNAI amplifier center
    NUM = re.compile(r'-?\d+(?:\.\d+)?')

    def parse_vec(x):
        try:
            if x is None:
                return None
            if isinstance(x, str):
                toks = NUM.findall(x)
                if not toks:
                    return None
                return np.array([float(t) for t in toks], dtype=float)
            if isinstance(x, bytes):
                return parse_vec(x.decode('utf-8', 'ignore'))
            if isinstance(x, (list, tuple)):
                return np.array([float(v) for v in x], dtype=float)
            if isinstance(x, np.ndarray):
                return x.astype(float).ravel()
            if isinstance(x, (int, float, np.integer, np.floating)):
                return None
            arr = np.asarray(x, dtype=float).ravel()
            return arr if arr.size > 0 else None
        except Exception:
            return None

    def parse_resp(x):
        try:
            if x is None:
                return None
            if isinstance(x, str):
                t = x.strip().upper()
                if t == 'A' or t.startswith('A_') or t.startswith('OPTION_A'):
                    return 0
                if t == 'B' or t.startswith('B_') or t.startswith('OPTION_B'):
                    return 1
                m = NUM.findall(t)
                if not m:
                    return None
                x = float(m[0])
            f = float(x)
            if np.isnan(f):
                return None
            r = int(round(f))
            if r == 0 or r == -1:
                return 0
            if r == 1 or r == 2:
                return 1
            return None
        except Exception:
            return None

    def analyze(ra, rb):
        """Recompute both theories' deterministic cores from the actual
        rating vectors + instructed validities.
        M  = CNNAI margin (eta = 0.12):  A_anti - B_anti, >0 -> B better.
        Ms = SACNAI margin: same, but the top endorsed validity tier's
             anti-weights are amplified by
             sig_eff = 1 + (sigma-1)*exp(-gap/0.10), sigma = 1.45.
        Returns (cn_side, strength) only when the two signs DISAGREE;
        strength = min(|M|, |Ms|) is the weaker theory's margin."""
        try:
            active = np.flatnonzero((ra > 0.5) | (rb > 0.5))
            if active.size < 2:
                return None
            v_ref = float(np.mean(VAL[active]))
            anti = VAL * np.power(VAL / v_ref, ETA)
            ma = ra > 0.5
            mb = rb > 0.5
            a_anti = float(np.sum(anti[ma]))
            b_anti = float(np.sum(anti[mb]))
            m_c = a_anti - b_anti
            m_s = m_c
            tiers = np.unique(VAL[active])
            if tiers.size >= 2:
                gap = float(tiers[-1] - tiers[-2])
                sig_eff = 1.0 + (SIGMA - 1.0) * float(np.exp(-gap / 0.10))
                if sig_eff > 1.0 + 1e-9:
                    top = VAL >= (tiers[-1] - 1e-9)
                    anti_p = np.where(top, anti * sig_eff, anti)
                    m_s = float(np.sum(anti_p[ma])) - float(np.sum(anti_p[mb]))
            if m_c == 0.0 or m_s == 0.0:
                return None
            if (m_c > 0.0) == (m_s > 0.0):
                return None
            cn_side = 1 if m_c > 0.0 else 0
            strength = min(abs(m_c), abs(m_s))
            return (cn_side, strength)
        except Exception:
            return None

    try:
        col_a = list(data['option_a_ratings'])
        col_b = list(data['option_b_ratings'])
        col_r = list(data['response'])
    except Exception:
        return 0.5

    n = min(len(col_a), len(col_b), len(col_r))
    cache = {}
    recs = []
    for i in range(n):
        ra = parse_vec(col_a[i])
        rb = parse_vec(col_b[i])
        if ra is None or rb is None or ra.size != NF or rb.size != NF:
            continue
        resp = parse_resp(col_r[i])
        if resp is None:
            continue
        key = (ra.tobytes(), rb.tobytes())
        if key not in cache:
            cache[key] = analyze(ra, rb)
        res = cache[key]
        if res is None:
            continue
        recs.append((res[0], res[1], resp))

    # Strict sign-kill trials first; relax the margin floor only if the
    # strict pass finds almost nothing (defensive fallback).
    for thresh in (0.20, 0.05, 0.0):
        vals = [1.0 if resp == side else 0.0
                for (side, strength, resp) in recs if strength >= thresh]
        if len(vals) >= 6:
            return float(np.mean(vals))
    if recs:
        vals = [1.0 if resp == side else 0.0 for (side, _, resp) in recs]
        return float(np.mean(vals))
    return 0.5
```

**Observed (real) value:** 0.4950 (var=0.0950)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.5283 (var=0.0654) (Δ vs real +0.0333)
**Other theories' values on this metric (for reference):**
- pi_6: 0.6958 (var=0.0184)
- pi_8: 0.3392 (var=0.0277)
- pi_1: 0.8600 (var=0.0139)
- pi_2: 0.1425 (var=0.0115)
- pi_3: 0.6808 (var=0.0116)
- pi_4: 0.4250 (var=0.0242)
- pi_3_1: 0.7508 (var=0.0365)
- pi_5: 0.6600 (var=0.0614)
- pi_7: 0.6325 (var=0.0241)
- pi_9: 0.5658 (var=0.0371)

### Experiment 17
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data):
    import numpy as np

    def _parse(cell):
        if isinstance(cell, str):
            s = cell.strip().strip('[]()')
            toks = [t for t in s.replace(',', ' ').split() if t != '']
            return np.array([float(t) for t in toks])
        return np.asarray(list(cell), dtype=float)

    # Feature layout fixed by design: idx 0-11 -> 0.90, 12-26 -> 0.85, 27-34 -> 0.60
    LADDER = {(4, 5): 1, (8, 10): 2, (12, 15): 3}   # rung by (n x 0.90-stack, 1.25n x 0.85-stack)
    BCELL = set([(5, 6), (6, 7)])                   # r = 1.2 bimodality cells B1/B2

    rung_vals = []
    b_vals = []
    c_vals = []

    for _, row in data.iterrows():
        a = _parse(row['option_a_ratings'])
        b = _parse(row['option_b_ratings'])
        if a.size < 35 or b.size < 35:
            continue
        a90 = float(np.sum(a[0:12]))
        a85 = float(np.sum(a[12:27]))
        a60 = float(np.sum(a[27:35]))
        b90 = float(np.sum(b[0:12]))
        b85 = float(np.sum(b[12:27]))
        b60 = float(np.sum(b[27:35]))

        rung = 0
        is_b = False
        stack_a = None
        if a60 == 0.0 and b60 == 0.0:
            # A is the pure 0.90-stack, B the pure 0.85-stack
            if a85 == 0.0 and b90 == 0.0 and a90 > 0 and b85 > 0:
                pair = (int(a90), int(b85))
                if pair in LADDER:
                    rung = LADDER[pair]
                    stack_a = True
                elif pair in BCELL:
                    is_b = True
                    stack_a = True
            # Mirror: B is the 0.90-stack, A the 0.85-stack
            elif b85 == 0.0 and a90 == 0.0 and b90 > 0 and a85 > 0:
                pair = (int(b90), int(a85))
                if pair in LADDER:
                    rung = LADDER[pair]
                    stack_a = False
                elif pair in BCELL:
                    is_b = True
                    stack_a = False

        if rung > 0 or is_b:
            resp = int(row['response'])   # 0 = chose A, 1 = chose B
            chose_a = (resp == 0)
            # c90 = 1 iff the subject chose the 0.90-stack option
            if chose_a == stack_a:
                c_vals.append(1.0)
            else:
                c_vals.append(0.0)
            rung_vals.append(rung)
            b_vals.append(is_b)

    rung = np.asarray(rung_vals, dtype=int)
    bmask = np.asarray(b_vals, dtype=bool)
    c90 = np.asarray(c_vals, dtype=float)

    m1 = (rung == 1)
    m3 = (rung == 3)
    if m1.sum() == 0 or m3.sum() == 0 or bmask.sum() == 0:
        return 0.0

    # Ladder slope: change in P(choose 0.90-stack) from rung 1
    # (n_disc = 9) to rung 3 (n_disc = 27), same validity bucket,
    # same per-subject eta draw, same gap and count ratio.
    slope = float(np.mean(c90[m3]) - np.mean(c90[m1]))

    # Level term: deviation from 0.5 of P(choose 0.90-stack) on the
    # r = 1.2 bimodality cells B1/B2 (n_disc = 11/13, same bucket).
    level = float(np.mean(c90[bmask]) - 0.5)

    return slope + level

```

**Observed (real) value:** 0.2633 (var=0.1550)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.1133 (var=0.0852) (Δ vs real -0.1500)
**Other theories' values on this metric (for reference):**
- pi_8: -0.4533 (var=0.0814)
- pi_9: 0.1842 (var=0.0918)
- pi_1: 0.3950 (var=0.0211)
- pi_2: -0.3650 (var=0.0224)
- pi_3: 0.3925 (var=0.0292)
- pi_4: 0.3242 (var=0.0313)
- pi_3_1: 0.4317 (var=0.0391)
- pi_5: 0.3342 (var=0.0497)
- pi_6: 0.3042 (var=0.0547)
- pi_7: 0.3100 (var=0.0543)

### Experiment 18
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # Fixed design: 38 experts, validities [0.9]*11 + [0.85]*22 + [0.6]*5
    n_feat = 38
    idx90 = np.arange(0, 11)   # 0.90 tier
    idx60 = np.arange(33, 38)  # 0.60 tier

    def parse(x):
        if isinstance(x, str):
            s = x.strip().strip('[]()')
            toks = [t for t in s.replace(',', ' ').split() if t != '']
            return np.array([float(t) for t in toks])
        return np.asarray(x, dtype=float).ravel()

    ys = []     # 1 = subject chose the 0.90-stack option
    loads = []  # n_disc = number of cues on which A and B differ

    a_col = data['option_a_ratings']
    b_col = data['option_b_ratings']
    r_col = data['response']

    for i in range(len(data)):
        ra = parse(a_col.iloc[i])
        rb = parse(b_col.iloc[i])
        if ra.size != n_feat or rb.size != n_feat:
            continue
        a90 = float(ra[idx90].sum())
        b90 = float(rb[idx90].sum())
        a60 = float(ra[33:38].sum())
        b60 = float(rb[33:38].sum())

        # Exclude C1 (0.60-tier endorsements) and ANCHOR (all-1 side
        # touches the 0.60 tier).
        if a60 > 0 or b60 > 0:
            continue
        # Exclude TIE (both sides endorse 0.90 cues) and any trial
        # where neither side carries a 0.90 stack.
        if (a90 > 0) == (b90 > 0):
            continue
        # Exactly one side carries the 0.90 stack; require stack size
        # >= 3, which excludes the AG anchor (2x0.90 vs 6x0.85).
        stack_count = a90 if a90 > 0 else b90
        if stack_count < 3:
            continue

        r = float(r_col.iloc[i])
        if a90 > 0:
            y = 1.0 if r == 0.0 else 0.0   # A is the 0.90 stack
        else:
            y = 1.0 if r == 1.0 else 0.0   # B is the 0.90 stack
        ys.append(y)
        loads.append(int(np.count_nonzero(ra != rb)))

    if len(ys) < 10:
        return 0.5

    ys = np.asarray(ys, dtype=float)
    loads = np.asarray(loads, dtype=int)

    # Level: P(choose 0.90 stack) over all 72 near-knife ladder
    # trials (F1 rungs + F2 endpoints + tied-filler control).
    L = float(ys.mean())

    # Slope: high-load rungs (n_disc >= 21) minus low-load rungs
    # (n_disc <= 13). Middle rung (n_disc = 17) contributes to L only.
    hi = ys[loads >= 21]
    lo = ys[loads <= 13]
    S = 0.0
    if hi.size >= 5 and lo.size >= 5:
        S = float(hi.mean() - lo.mean())

    return L + 0.5 * S
```

**Observed (real) value:** 0.5539 (var=0.1118)
**Previous candidate values (this loop):**
  - iter 1 (most recent): 0.5670 (var=0.0368) (Δ vs real +0.0131)
**Other theories' values on this metric (for reference):**
- pi_9: 0.5998 (var=0.0143)
- pi_8: 0.3211 (var=0.0237)
- pi_1: 0.8501 (var=0.0097)
- pi_2: 0.1501 (var=0.0132)
- pi_3: 0.7173 (var=0.0086)
- pi_4: 0.6798 (var=0.0208)
- pi_3_1: 0.7758 (var=0.0339)
- pi_5: 0.7328 (var=0.0250)
- pi_6: 0.7213 (var=0.0157)
- pi_7: 0.6787 (var=0.0256)

## PREVIOUS CANDIDATE (this loop)
The RUNNING-BEST (last ACCEPTED) candidate in this critique loop — i.e. the source the loop's accept gate kept as the best base so far. If your most recent attempt was REJECTED by the gate, this is NOT that attempt; it is the previously-accepted base the gate rolled back to. Iterate on this source — the next critic feedback should be applied on top of it.

**Description**
TPQSAI — Trait-based Prestige-Quantity Salience Anti-validity Integration. Subjects integrate binary expert ratings additively into a per-option value but comprehend a rating of 1 as a DEFECT (sign inversion), with context-normalized anti-weights w_j = -v_j*(v_j/v_ref)^eta (v_ref = mean endorsed validity; eta drawn per bucketed-mean-endorsed-validity context from a small stable per-subject component plus a per-context redraw, latent cross-context r ~ 0.2). The single genuinely new claim is that prestige aversion is a STABLE SUBJECT TRAIT, not a per-context state: each subject carries one sigma_i drawn once from a quasi-bimodal distribution (~45-55% linear counters with sigma in [1.0, 1.2], the remainder prestige-sensitive with sigma in [1.7, 2.2]), never redrawn across contexts or ladders. On any trial pitting endorsements on the top endorsed validity tier against a lower tier, the top tier's anti-weights are amplified by sigma_eff = 1 + (sigma_i - 1)*exp(-gap/0.10)*D, where D = 1/(1+(n_disc/n0)^1.1) is a flat load-dilution factor: prestige aversion is strong on small knife-edge contests (n_disc ~ 6, D ~ 0.65-0.70) but dilutes toward the linear core on large stacks (n_disc >= 20). Stochasticity lives as PER-TRIAL NOISE ON THE VALUE DIFFERENCE, sd = s0 + s1*log(1+n_disc), rather than per-cue attention diffusion — this attenuates within-subject reliability on complex 24-31 feature trials without flattening small-contest amplification. Choice is softmax(beta*value) with symmetric lapse; ~13% true guessers, ~12% hot tail; engaged beta lognormal median ~1.85 (clip [0.8, 4.2]). Exactly linear within a validity family (tied-filler load ladders flat).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # TPQSAI -- Trait-based Prestige-Quantity Salience Anti-validity
    # Integration.
    #
    # Core (validated CNNAI/PQSAI lineage, retained): a rating of 1 is a
    # defect (sign inversion); w_j = -v_j*(v_j/v_ref)^eta with v_ref =
    # mean endorsed validity; eta drawn per bucketed-mean-endorsed-
    # validity context (stable component sd 0.10 + redraw sd 0.19 ->
    # latent cross-context r ~ 0.2); softmax(beta*value) with symmetric
    # lapse; ~13% guessers; ~12% hot tail; exact within-family
    # linearity.
    #
    # NEW vs the running-best (pi_9) and vs pi_8:
    #   (1) STABLE PRESTIGE TRAIT: sigma_i is drawn ONCE per subject
    #       from a quasi-bimodal distribution (~45-55% linear counters,
    #       sigma in [1.0, 1.2]; remainder prestige-sensitive, sigma in
    #       [1.7, 2.2]) and is NOT redrawn across contexts or ladders.
    #       This within-subject stability is what supports the observed
    #       cross-ladder consistency that per-context redraws destroy.
    #   (2) RECALIBRATED LOAD DILUTION: sigma_eff =
    #       1 + (sigma_i - 1)*exp(-gap/0.10)*D with
    #       D = 1/(1 + (n_disc/n0)^1.1), n0 ~ U[10, 14]. The exponent
    #       is kept at 1.1 but n0 is raised so that the ARBITER'S
    #       STATED OPERATING POINTS are actually achieved: D(n_disc=6)
    #       ~ 0.65-0.70 (small knife cells like the Exp 11 AL cells and
    #       the Exp 1 B-cells keep most of the amplification, landing
    #       the AL anti-validity excess between pi_8's undiluted 3.60
    #       and pi_9's over-diluted 2.32, near the observed 2.88),
    #       while large stacks (n_disc >= 20) still dilute strongly
    #       toward the linear core (Exp 16, Exp 2 high rungs).
    #   (3) MARGIN-LEVEL NOISE: per-trial Gaussian noise on the value
    #       DIFFERENCE, sd = s0 + s1*log(1+n_disc), replaces per-cue
    #       load-scaled attention diffusion. Because it is per-trial it
    #       supplies within-context stochasticity that averages out of
    #       population means (so Exp 11 / Exp 17 amplification is no
    #       longer over-attenuated, pi_9's failure mode), while still
    #       degrading within-subject reliability on complex 24-31
    #       feature trials (Exp 9 target ~0.20 cross-ladder
    #       correlation).
    #   (4) BETA TRIM: engaged beta lognormal median ~1.85 (log-sd
    #       0.28, clip [0.8, 4.2]) so the winsorized single-validity
    #       ladder slope (Exp 14) lands near the observed 1.41 rather
    #       than 1.58, and the Exp 1 slope flattens toward -0.175.
    import hashlib
    import math
    import numpy as np
    from scipy.special import ndtri

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TPQSAI expects a (2, n_features) stimulus; got shape {stim.shape}."
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

    def _pos(name, default, lo, hi):
        try:
            x = float(parameters[name])
        except (KeyError, TypeError, ValueError):
            x = default
        if not np.isfinite(x):
            x = default
        return min(max(x, lo), hi)

    u = _seed("regime_seed")
    es = _seed("eta_base_seed")
    bs = min(max(_seed("beta_seed"), 1e-6), 1.0 - 1e-6)
    ps = _seed("epsilon_seed")
    ss = _seed("subject_seed")
    ms = _seed("mode_seed")

    # --- TPQSAI-specific knobs ---
    n0 = _pos("dilution_n0", 12.0, 2.0, 30.0)       # dilution scale
    s0 = _pos("noise_s0", 0.03, 0.0, 0.2)           # margin-noise intercept
    s1 = _pos("noise_s1", 0.012, 0.0, 0.1)          # margin-noise load slope
    sig_s = _pos("sigma_sensitive", 1.95, 1.0, 3.0)  # sensitive-mode sigma
    frac = _pos("sigma_sensitive_fraction", 0.50, 0.05, 0.95)

    # --- (1) STABLE PRESTIGE TRAIT ---
    # Drawn ONCE per subject (mode_seed is a per-subject parameter),
    # never redrawn across contexts or ladders. Quasi-bimodal:
    # prestige-sensitive subjects sit at sigma ~ U[1.7, 2.2] (via the
    # sigma_sensitive parameter), linear counters at sigma ~ U[1.0, 1.2].
    if ms < frac:
        sigma = sig_s
    else:
        t = (ms - frac) / max(1e-9, 1.0 - frac)
        sigma = 1.0 + 0.20 * t

    # --- Population mixture (uncorrelated with the trait). ---
    if u < 0.13:
        # Guesser subpopulation (~13%): beta ~ 0 -> uniform core on
        # every trial; elevated lapse. Amplifier irrelevant; pin sigma.
        t = u / 0.13
        beta = 0.02 + 0.08 * t                 # 0.02 .. 0.10
        epsilon = 0.12 + 0.13 * ps             # 0.12 .. 0.25
        eta_base, redraw_mu, redraw_sd = 0.12, 0.0, 0.0
        sigma = 1.0
    elif u < 0.25:
        # Hot tail (~12%): sharp, low-lapse subjects whose elasticity
        # is pinned at the engaged median so they amplify the CORE sign
        # pattern without shifting effective curvature. The prestige
        # TRAIT is retained (stable within subject).
        beta = 3.4 + 1.8 * bs                  # 3.4 .. 5.2
        epsilon = 0.02 + 0.03 * ps             # 0.02 .. 0.05
        eta_base, redraw_mu, redraw_sd = 0.12, 0.0, 0.10
    else:
        # Engaged bulk (~75%). Stable elasticity component (sd 0.10)
        # plus per-context redraw (sd 0.19): variance ratio fixes
        # latent cross-context consistency at r ~ 0.2. Beta trimmed:
        # lognormal median 1.85, log-sd 0.28, clip [0.8, 4.2].
        zs = min(max(es, 1e-6), 1.0 - 1e-6)
        eta_base = 0.02 + 0.10 * float(ndtri(zs))
        eta_base = min(max(eta_base, -0.15), 0.20)

        beta = math.exp(0.6152 + 0.28 * float(ndtri(bs)))  # median ~1.85
        beta = min(max(beta, 0.8), 4.2)
        epsilon = 0.03 + 0.07 * ps             # 0.03 .. 0.10
        redraw_mu, redraw_sd = 0.10, 0.19

    # --- Context: bucketed mean endorsed validity. ---
    active = np.flatnonzero((ra > 0.5) | (rb > 0.5))
    if active.size == 0:
        # Nothing endorsed by either option: no evidence either way.
        return np.full(2, 0.5)

    # Load: number of cues on which the two options differ. Computed on
    # every trial (used by both the dilution factor and the noise).
    n_disc = int(np.count_nonzero(ra != rb))

    v_ref = float(np.mean(val[active]))
    bucket = float(np.round(v_ref, 1))
    key = f"{bucket:.1f}"
    h = int(hashlib.md5(f"{ss:.9f}|{key}".encode("utf-8")).hexdigest()[:14], 16)
    uc = ((h % 999983) + 0.5) / 999984.0
    uc = min(max(uc, 1e-6), 1.0 - 1e-6)

    # Engaged eta ~ N(0.12, 0.21): centered well below the kill
    # knives (~0.31-0.34). Hot-tail eta pinned at the engaged median.
    eta_ctx = eta_base + redraw_mu + redraw_sd * float(ndtri(uc))
    eta_ctx = min(max(eta_ctx, -0.25), 0.70)

    # Context-normalized anti-validity weights.
    w = -val * np.power(val / v_ref, eta_ctx)

    # --- (2) Prestige aversion with RECALIBRATED quantity dilution ---
    # sigma_eff = 1 + (sigma_i - 1) * exp(-gap/0.10) * D,
    # D = 1 / (1 + (n_disc/n0)^1.1), n0 ~ U[10, 14].
    # Operating points: D(n_disc=6) ~ 0.65-0.70 -> the 3v3 knife cells
    # (Exp 11 AL, Exp 15 L-cells, Exp 1 B-cells) retain most of the
    # amplification; D(n_disc=13-15) ~ 0.45-0.50; D(n_disc >= 20)
    # ~ 0.27-0.36 -> large stacks dilute toward the linear core
    # (Exp 16, Exp 2 high rungs). Coarse-gap stacks (gap >= 0.2) keep
    # exp(-gap/0.10) <= 0.14, so the amplifier stays essentially off
    # there and the validated core predictions are preserved
    # (Exps 5, 6, 8, 9, 10). Single-family trials have no contrast
    # tier: off, so within-family load ladders stay exactly flat.
    tiers = np.unique(val[active])
    if tiers.size >= 2:
        gap = float(tiers[-1] - tiers[-2])
        D = 1.0 / (1.0 + (n_disc / n0) ** 1.1)
        sig_eff = 1.0 + (sigma - 1.0) * math.exp(-gap / 0.10) * D
        if sig_eff > 1.0 + 1e-9:
            top_mask = val >= (tiers[-1] - 1e-9)
            w = np.where(top_mask, w * sig_eff, w)

    # --- (3) MARGIN-LEVEL NOISE ---
    # Per-trial Gaussian noise on the value difference,
    # sd = s0 + s1*log(1+n_disc). Per-trial noise averages out within
    # a subject, so it supplies within-context stochasticity (and
    # degrades cross-ladder reliability on complex designs, Exp 9)
    # WITHOUT damping population means or the small-contest
    # amplification (pi_9's per-cue diffusion over-attenuated exactly
    # the cells where amplification was needed).
    margin = float(np.dot(w, ra - rb))
    sd_m = s0 + s1 * math.log(1.0 + float(n_disc))
    margin += sd_m * float(np.random.randn())

    # --- Softmax over the two options with trimmed beta. ---
    z = beta * margin
    z = min(max(z, -60.0), 60.0)
    p_a = 1.0 / (1.0 + math.exp(-z))
    p = np.array([p_a, 1.0 - p_a])

    return (1.0 - epsilon) * p + epsilon * 0.5 * np.ones(2)

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
- mode_seed: [0, 1]
- sigma_sensitive: [1.7, 2.2]
- sigma_sensitive_fraction: [0.45, 0.55]
- dilution_n0: [10, 14]
- noise_s0: [0.02, 0.04]
- noise_s1: [0.008, 0.015]
- validities: validities

`rationale`: This proposal implements the arbiter's prescribed TPQSAI family as a direct, stronger competitor to the running-best pi_9 (0.932), keeping every mechanism that made pi_9 win while repairing its four clear misses (Exps 11, 14, 16, 17) and its Exp 9 under-shoot.

(1) STABLE PRESTIGE TRAIT vs pi_9's per-context-adjacent mixture. pi_9 already drew sigma per subject, but its operating points were tuned so the small-contest amplification fell short: Exp 11 landed at 2.32 vs the real 2.88 (pi_8's undiluted version overshot to 3.60). TPQSAI widens the sensitive mode to [1.7, 2.2], pins the linear mode to [1.0, 1.2], and — critically — keeps sigma_i fixed across all contexts and ladders, so the between-subject mode separation is expressed consistently wherever a top-tier contrast exists.

(2) DILUTION RECALIBRATED TO THE ARBITER'S STATED OPERATING POINTS. The arbiter asked for D ~ 0.65-0.75 at n_disc = 6 with large stacks still diluting. Under exponent 1.1 that requires n0 ~ 11-13, not [8, 10] (which yields D(6) ~ 0.58-0.64 and would leave Exp 11 stuck near pi_9's value); I therefore expose n0 ~ U[10, 14] to hit the intended operating points — this is a faithful implementation of the arbiter's INTENT (Exp 11 near 2.88, between pi_8's 3.60 and pi_9's 2.32) rather than its literal range. Working through the AL cells (3x0.90 vs 3x0.85, gap 0.05, n_disc 6): sigma_eff ~ 1.39 for the median sensitive subject, giving an engaged-mixture population excess ~ 2.7-2.9 vs pi_9's 2.32 — capturing most of the 0.56 gap. The same strengthening pulls Exp 16 down from 0.566 toward the real 0.495 (large-stack cells retain D ~ 0.33-0.45, landing the population between the linear core's 0.70 and pi_8's 0.34) and lifts Exp 17's dilution-induced ladder slope from 0.184 toward the real 0.263 (sensitive subjects flip on the low rung, n_disc = 9, D ~ 0.58, but not the high rung, n_disc = 27, D ~ 0.29).

(3) MARGIN-LEVEL NOISE fixes pi_9's over-attenuation. pi_9's per-cue load-scaled diffusion (sd up to ~0.11 on 13-20 active cues) injected margin noise comparable to the near-knife signals it was supposed to leave intact — the main channel by which its Exp 11 and Exp 17 amplification was washed out. Replacing it with sd = s0 + s1*log(1+n_disc) (s1 ~ 0.012) keeps complex-trial reliability attenuation for Exp 9 (target r ~ 0.20) while leaving small-contest margins essentially crisp. It also makes the Exp 13 tied-filler ladder exactly symmetric (n_disc is 1 on every rung), preserving the flat 0 drop.

(4) BETA TRIM (median 1.85, clip [0.8, 4.2]) moves the Exp 14 winsorized ladder slope from pi_9's 1.58 toward the observed 1.41, flattens the Exp 1 slope toward -0.175, and softens the Exp 10 twin-kill rise toward 0.296; the slightly wider log-sd (0.28) plus the stable trait and reduced measurement noise raise the Exp 9 cross-ladder correlation above pi_9's 0.106 toward the observed 0.196.

Everything else — the defect sign inversion, context-normalized anti-validity weights with bucketed per-context eta, 13% guessers, 12% hot tail, exact within-family linearity, coarse-gap experiments (5, 6, 8, 9, 10) with the amplifier essentially off — is retained from the accepted lineage, so pi_9's wins on Exps 1, 2, 3, 15, 18 are preserved while the recalibrated trait/dilution/noise triad repairs the misses. Expected net effect is a strict aggregate-loss improvement over the 0.932 floor.

## PRIOR FEEDBACK ITERATIONS
The critic's verdicts on each previous in-loop candidate, in order. Each block ends with an **Outcome of this advice** line saying whether the candidate the proposer produced AFTER this advice was ACCEPTED (loss strictly beat the running best) or REJECTED (the gate discarded it). Address the most recent iteration's feedback in your next edit, but down-weight past advice whose candidates were REJECTED.

### Iteration 1 (most recent — address this)
Verdict: regenerate
Interpretation: The candidate is a genuine improvement over pi_9 in aggregate (accepted at loss 0.0805) and it nails the arbiter's headline target: Exp 11 AL anti-validity excess lands at exactly 2.88 vs the real 2.88 (between pi_8's 3.60 and pi_9's 2.32), and Exps 3, 4, 6, 7, 8, 13, 16, 18 are all essentially on target. However, the simulation exposes one large mechanistic failure and several moderate misses that are all diagnosable within the prescribed TPQSAI family. (1) EXP 9 (biggest error, 0.474 vs real 0.196): the margin-level per-trial noise cannot do the job it was assigned. iid per-trial Gaussian noise on the value difference averages out of per-subject choice RATES, so it supplies no attenuation of the cross-ladder correlation between the 0.50-ladder and 0.60-ladder kill rates. What drives those rates is the stable sigma_i trait plus beta and eta — all fixed within subject — so the two ladders co-vary almost perfectly (r ~ 0.47 instead of ~0.20). The claim in the rationale that per-trial noise 'degrades within-subject reliability on complex trials' is simply wrong at the level of the metric, which is computed on within-subject rate correlations. pi_8 hit 0.196 and pi_9 hit 0.106 on this metric, so the family can reach the target; this parameterization cannot. (2) EXP 10 overshoots (0.413 vs 0.296; pi_9 was 0.326): the raised dilution scale n0 in [10,14] keeps too much amplification on the twin-kill cells (n_disc = 8-12) while the beta trim should have flattened the rise — the net effect went the wrong way relative to pi_9. (3) EXP 17 undershoots (0.113 vs 0.263; worse than pi_9's 0.184): the ladder slope (rung n_disc=9 vs 27) plus the B-cell level term are both too weak. (4) EXP 14 undershoots (1.262 vs 1.407): the beta trim over-corrected past the target (pi_9's 1.58 → candidate 1.26, target 1.41). (5) EXP 5 undershoots (0.579 vs 0.684, worse than pi_9's 0.616). The pattern is coherent: the trait/dilution/noise triad was calibrated to hit Exp 11's mean exactly, but the noise channel is inert where it was supposed to attenuate (Exp 9), and the dilution curve is too flat between n_disc = 6 and n_disc = 8-12 (Exp 10), while beta is now slightly too low for Exps 14/17.
Rationale: The candidate was ACCEPTED and is the new base — build the next iteration on top of it, keeping the TPQSAI family intact. Three targeted, in-family adjustments: (A) FIX EXP 9 (highest priority, error 0.28): replace the inert per-trial margin noise as the reliability-attenuation channel with a PER-CONTEXT multiplicative jitter on the prestige trait, e.g. sigma_eff,ctx = 1 + (sigma_i - 1) * kappa_ctx where kappa_ctx > 0 is drawn per (subject, validity-bucket) with mean ~1 and sd ~0.4-0.5 (lognormal). This preserves the arbiter's stable-trait architecture (sigma_i still drawn once and never redrawn) while adding exactly the kind of context-level variability that lowers the cross-ladder rate correlation toward ~0.20. Critically, symmetric kappa jitter leaves population MEANS untouched, so Exp 11's 2.88, Exp 15, and Exp 16 should not move; verify this. The per-trial margin noise (s0, s1) can stay but should be understood as pure trial stochasticity, not a reliability mechanism. (B) FIX EXP 10: steepen the dilution curve between n_disc = 6 and n_disc = 8-12 while holding D(6) at ~0.65-0.70 — e.g. lower n0 to ~[8,10] and raise the exponent to ~1.3-1.4 (or keep n0 ~ 12 with exponent ~1.5). This preserves the AL knife cells (n_disc = 6, Exp 11's perfect 2.88) while pulling the twin-kill cells (n_disc 8-12) back toward the linear core, moving the Exp 10 rise from 0.41 toward 0.30. (C) FIX EXPS 14/17: the beta trim over-shot downward. Raise the engaged beta lognormal median from 1.85 back toward ~2.0-2.1 (clip unchanged), which should lift the Exp 14 winsorized slope from 1.26 toward 1.41 and add level to Exp 17's B-cells; the dilution steepening in (B) counteracts the upward pressure this puts on Exp 10, and Exp 1 (currently -0.206 vs -0.175) should be re-checked but is a small residual error. Do not touch the guesser/hot-tail mixture, the sign inversion, or the eta architecture — those are carrying Exps 2-8, 12, 13, 15, 16, 18 well. Expected net effect: Exp 9 from 0.47 to ~0.2, Exp 10 from 0.41 to ~0.31, Exp 14 from 1.26 to ~1.40, Exp 17 from 0.11 toward ~0.2, with Exp 11 held at ~2.88 — a strict aggregate-loss improvement over the 0.0805 base.

**Outcome of this advice:** iter 1 candidate loss=0.0805 -> next candidate accept-status=PENDING (this is the advice you are addressing now).


## IMPLEMENTATION GUARDRAILS
- The parameters should be within the specified ranges.
- The model's predictions should be valid probability distributions (non-negative and sum to 1).
- When converting logits to probabilities via softmax, always use the numerically stable form: subtract the max before exponentiating (`x = x - np.max(x); p = np.exp(x); p /= p.sum()`). A naive `np.exp(x) / np.sum(np.exp(x))` overflows to Inf/NaN for large logits. Alternatively, use `scipy.special.softmax`.


## PROPOSAL — MINIMAL-DIFF EDIT (do NOT rewrite from scratch)
The RUNNING-BEST (last ACCEPTED) candidate is shown verbatim above under `## PREVIOUS CANDIDATE (this loop)`. This is the base the loop's accept gate is currently keeping; the most recent critic feedback (see `## PRIOR FEEDBACK ITERATIONS`) should be applied on top of it. Apply the SMALLEST edit that addresses the critic's diagnosis while staying inside the arbiter's prescribed mechanism family:
  - Re-emit the previous source verbatim, then change ONLY the lines needed to address the critic (a parameter range, a normalization, a softmax temperature, an attention scheme, a gating term, a buggy indexing line, etc.).
  - Keep all unaffected functions, equations, parameter names, and the overall mechanism intact.
  - Do NOT rewrite `predict` / `policy` end-to-end if a few lines would do, and do NOT switch mechanism families — that is the arbiter's decision, not yours in this loop.
  - Briefly explain the minimal edit in `rationale`.
If you genuinely believe a larger rewrite is required, you may do one, but justify in `rationale` why the minimal-diff path was insufficient.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- theory: The theoretical claim this theory makes about cognition.
- predict: Python source defining def predict(parameters, state, history) -> np.ndarray.
- policy: Python source defining def policy(probs) -> int.
- parameters: Parameter ranges (e.g. "[1.0, 10.0]") or choices (e.g. "{1, 2}"), sampled per subject.
- rationale: Reasoning behind this theory design; used to explain why this theory is better then the previous ones.


## Response

```json
```json
{
  "theory": "TPQSAI-2 — Trait-based Prestige-Quantity Salience Anti-validity Integration with Context-Jittered Trait Expression. Subjects integrate binary expert ratings additively into a per-option value but comprehend a rating of 1 as a DEFECT (sign inversion), with context-normalized anti-weights w_j = -v_j*(v_j/v_ref)^eta (v_ref = mean endorsed validity; eta drawn per bucketed-mean-endorsed-validity context from a small stable per-subject component plus a per-context redraw, latent cross-context r ~ 0.13). Prestige aversion is a STABLE SUBJECT TRAIT (sigma_i drawn once from a quasi-bimodal distribution: ~50% linear counters at sigma in [1.0, 1.2], remainder prestige-sensitive at sigma in [1.75, 2.25]) whose EXPRESSION is jitters per context by a multiplicative factor kappa_ctx ~ lognormal(median 1, sd ~ 0.45), drawn per (subject, validity-bucket): sigma_eff = 1 + (sigma_i - 1)*kappa_ctx*exp(-gap/0.10)*D with a STEEPENED load-dilution D = 1/(1+(n_disc/n0)^1.4), n0 ~ U[9,11] — D(6) ~ 0.67 is held fixed (preserving the Exp 11 AL operating point) while the curve drops faster through n_disc = 8-16 (D(8) ~ 0.58, D(12) ~ 0.44, D(16) ~ 0.35), pulling the twin-kill and large-stack cells back toward the linear core. Stochasticity is per-trial Gaussian noise on the value difference, sd = s0 + s1*log(1+n_disc) with a raised s1; together with the narrowed beta spread (lognormal median ~2.05, log-sd 0.20) and rebalanced eta variance this attenuates the cross-ladder rate correlation on complex designs toward the observed ~0.20. Choice is softmax(beta*value) with symmetric lapse; ~13% true guessers, ~12% hot tail; exactly linear within a validity family.",
  "predict": "def predict(parameters, state, history):\n    # TPQSAI iter 2 -- minimal-diff edit of the ACCEPTED iter-1 base,\n    # applying the critic's three in-family recalibrations:\n    #   (A) PER-CONTEXT TRAIT-EXPRESSION JITTER (kappa): the stable\n    #       prestige trait sigma_i is retained (drawn once per subject,\n    #       never redrawn), but its expression is modulated per\n    #       (subject, validity-bucket) by kappa_ctx ~ lognormal(median 1,\n    #       sd ~ 0.45). Symmetric jitter leaves population means\n    #       essentially untouched while decorrelating trait expression\n    #       across contexts/ladders.\n    #   (B) STEEPENED DILUTION: exponent 1.1 -> 1.4, n0 ~ U[9,11].\n    #       D(6) ~ 0.67 is preserved (Exp 11 AL cells keep the\n    #       amplification that landed 2.88), but the curve falls\n    #       faster through n_disc = 8-16, pulling the twin-kill\n    #       (Exp 10) and large-stack (Exp 16, Exp 17 high rungs) cells\n    #       back toward the linear core.\n    #   (C) BETA RESTORED toward ~2.05 (log-sd trimmed 0.28 -> 0.20):\n    #       lifts the Exp 14 winsorized ladder slope and Exp 17 level,\n    #       and (by shrinking the dominant shared-trait coupling\n    #       channel) helps decorrelate the Exp 9 ladders.\n    #   (D) EXP 9 (the critic's headline miss, 0.474 vs 0.196): NOTE --\n    #       the prestige amplifier is provably OFF on both Exp 9\n    #       ladders (top-two-tier gap 0.40-0.50 => exp(-gap/0.10) <=\n    #       0.02), so kappa alone cannot attenuate the cross-ladder\n    #       RATE correlation there. The actual coupling channels are\n    #       the stable beta, eta_base and epsilon traits. I therefore\n    #       additionally (i) trim the beta log-sd to 0.20, (ii)\n    #       rebalance the eta variance split (stable sd 0.08, redraw sd\n    #       0.21 -> latent r ~ 0.13), and (iii) raise the per-trial\n    #       margin-noise load slope s1 to ~0.035, which on the\n    #       near-knife 24-feature kill cells pushes choice\n    #       probabilities toward 0.5 and injects binomial sampling\n    #       variance into the per-subject rates -- the same channel by\n    #       which pi_6/pi_9 reached 0.11-0.19 on this metric.\n    import hashlib\n    import math\n    import numpy as np\n    from scipy.special import ndtri\n\n    stim = np.asarray(state, dtype=float)\n    if stim.ndim != 2 or stim.shape[0] != 2:\n        raise ValueError(\n            f\"TPQSAI expects a (2, n_features) stimulus; got shape {stim.shape}.\"\n        )\n    n_features = stim.shape[1]\n    ra, rb = stim[0], stim[1]\n\n    def _to_vec(raw):\n        if raw is None:\n            return None\n        if isinstance(raw, str):\n            s = raw.strip().strip('[]()')\n            toks = [t for t in s.replace(',', ' ').split() if t != '']\n            if not toks:\n                return None\n            return np.array([float(t) for t in toks])\n        arr = np.asarray(raw, dtype=float).ravel()\n        return arr if arr.size > 0 else None\n\n    val = _to_vec(parameters.get(\"validities\"))\n    if val is None or val.size != n_features:\n        # Defensive fallback: uniform mid-range validities.\n        val = np.full(n_features, 0.75)\n    val = np.clip(val, 1e-3, 1.0)\n\n    def _seed(name):\n        try:\n            x = float(parameters[name])\n        except (KeyError, TypeError, ValueError):\n            x = 0.5\n        if not np.isfinite(x):\n            x = 0.5\n        return min(max(x, 0.0), 1.0)\n\n    def _pos(name, default, lo, hi):\n        try:\n            x = float(parameters[name])\n        except (KeyError, TypeError, ValueError):\n            x = default\n        if not np.isfinite(x):\n            x = default\n        return min(max(x, lo), hi)\n\n    u = _seed(\"regime_seed\")\n    es = _seed(\"eta_base_seed\")\n    bs = min(max(_seed(\"beta_seed\"), 1e-6), 1.0 - 1e-6)\n    ps = _seed(\"epsilon_seed\")\n    ss = _seed(\"subject_seed\")\n    ms = _seed(\"mode_seed\")\n\n    # --- TPQSAI-specific knobs (recalibrated) ---\n    n0 = _pos(\"dilution_n0\", 10.0, 2.0, 30.0)        # dilution scale\n    s0 = _pos(\"noise_s0\", 0.03, 0.0, 0.2)           # margin-noise intercept\n    s1 = _pos(\"noise_s1\", 0.035, 0.0, 0.1)           # margin-noise load slope (raised)\n    sig_s = _pos(\"sigma_sensitive\", 2.0, 1.0, 3.0)   # sensitive-mode sigma\n    frac = _pos(\"sigma_sensitive_fraction\", 0.50, 0.05, 0.95)\n    kap_sd = _pos(\"kappa_sd\", 0.45, 0.1, 0.9)        # trait-expression jitter sd\n\n    # --- (1) STABLE PRESTIGE TRAIT (unchanged architecture) ---\n    if ms < frac:\n        sigma = sig_s\n    else:\n        t = (ms - frac) / max(1e-9, 1.0 - frac)\n        sigma = 1.0 + 0.20 * t\n\n    # --- Population mixture (uncorrelated with the trait). ---\n    if u < 0.13:\n        # Guesser subpopulation (~13%): beta ~ 0 -> uniform core.\n        t = u / 0.13\n        beta = 0.02 + 0.08 * t                 # 0.02 .. 0.10\n        epsilon = 0.12 + 0.13 * ps             # 0.12 .. 0.25\n        eta_base, redraw_mu, redraw_sd = 0.12, 0.0, 0.0\n        sigma = 1.0\n    elif u < 0.25:\n        # Hot tail (~12%): sharp, low-lapse; trait retained.\n        beta = 3.4 + 1.8 * bs                  # 3.4 .. 5.2\n        epsilon = 0.02 + 0.03 * ps             # 0.02 .. 0.05\n        eta_base, redraw_mu, redraw_sd = 0.12, 0.0, 0.10\n    else:\n        # Engaged bulk (~75%). Variance split rebalanced: stable sd 0.08\n        # + per-context redraw sd 0.21 -> latent cross-context r ~ 0.13\n        # (weaker shared-eta coupling across ladders, same total spread).\n        zs = min(max(es, 1e-6), 1.0 - 1e-6)\n        eta_base = 0.02 + 0.08 * float(ndtri(zs))\n        eta_base = min(max(eta_base, -0.15), 0.20)\n\n        # Beta restored toward the pi_6/pi_9 operating point but with a\n        # TRIMMED log-sd (0.28 -> 0.20): median ~2.05 lifts Exp 14/17,\n        # and the narrower spread removes the dominant stable-trait\n        # coupling channel behind the Exp 9 over-correlation.\n        beta = math.exp(0.718 + 0.20 * float(ndtri(bs)))  # median ~2.05\n        beta = min(max(beta, 0.8), 4.2)\n        epsilon = 0.03 + 0.07 * ps             # 0.03 .. 0.10\n        redraw_mu, redraw_sd = 0.10, 0.21\n\n    # --- Context: bucketed mean endorsed validity. ---\n    active = np.flatnonzero((ra > 0.5) | (rb > 0.5))\n    if active.size == 0:\n        # Nothing endorsed by either option: no evidence either way.\n        return np.full(2, 0.5)\n\n    n_disc = int(np.count_nonzero(ra != rb))\n\n    v_ref = float(np.mean(val[active]))\n    bucket = float(np.round(v_ref, 1))\n    key = f\"{bucket:.1f}\"\n\n    # Per-(subject, bucket) eta redraw.\n    h = int(hashlib.md5(f\"{ss:.9f}|{key}\".encode(\"utf-8\")).hexdigest()[:14], 16)\n    uc = ((h % 999983) + 0.5) / 999984.0\n    uc = min(max(uc, 1e-6), 1.0 - 1e-6)\n\n    # (A) Per-(subject, bucket) TRAIT-EXPRESSION JITTER kappa.\n    # Lognormal, median 1, sd ~ kap_sd: mean-preserving on the log scale,\n    # so population mean amplification is essentially unchanged while\n    # the effective sigma varies across contexts/ladders.\n    h2 = int(hashlib.md5(f\"{ss:.9f}|KAP|{key}\".encode(\"utf-8\")).hexdigest()[:14], 16)\n    uk = ((h2 % 999983) + 0.5) / 999984.0\n    uk = min(max(uk, 1e-6), 1.0 - 1e-6)\n    kappa = math.exp(kap_sd * float(ndtri(uk)))\n    kappa = min(max(kappa, 0.15), 4.0)\n\n    # Engaged eta ~ N(0.12, 0.22): centered well below the kill knives.\n    eta_ctx = eta_base + redraw_mu + redraw_sd * float(ndtri(uc))\n    eta_ctx = min(max(eta_ctx, -0.25), 0.70)\n\n    # Context-normalized anti-validity weights.\n    w = -val * np.power(val / v_ref, eta_ctx)\n\n    # --- (2) Prestige aversion with STEEPENED quantity dilution ---\n    # sigma_eff = 1 + (sigma_i - 1) * kappa * exp(-gap/0.10) * D,\n    # D = 1 / (1 + (n_disc/n0)^1.4), n0 ~ U[9, 11].\n    # Operating points: D(6) ~ 0.67 (Exp 11 AL cells keep the\n    # amplification that produced the on-target 2.88); D(8) ~ 0.58,\n    # D(12) ~ 0.44, D(16) ~ 0.35, D(27) ~ 0.20 -- the twin-kill cells\n    # (Exp 10) and large stacks (Exps 16/17 high rungs) are pulled back\n    # toward the linear core. Coarse-gap stacks (gap >= 0.2) keep the\n    # amplifier essentially off (Exps 5, 6, 8, 9, 10 cores preserved).\n    tiers = np.unique(val[active])\n    if tiers.size >= 2:\n        gap = float(tiers[-1] - tiers[-2])\n        D = 1.0 / (1.0 + (n_disc / n0) ** 1.4)\n        sig_eff = 1.0 + (sigma - 1.0) * kappa * math.exp(-gap / 0.10) * D\n        if sig_eff > 1.0 + 1e-9:\n            top_mask = val >= (tiers[-1] - 1e-9)\n            w = np.where(top_mask, w * sig_eff, w)\n\n    # --- (3) MARGIN-LEVEL NOISE (raised load slope) ---\n    # sd = s0 + s1*log(1+n_disc), s1 ~ 0.035. On the near-knife\n    # complex cells (Exp 9/10 kill cells, n_disc 9-16) this yields\n    # sd ~ 0.10-0.13, pushing choice probabilities toward 0.5 and\n    # injecting sampling variance into per-subject rates (the Exp 9\n    # attenuation channel), while small-contest (n_disc <= 5) and\n    # single-family ladder trials (n_disc 1-3) stay crisp (Exps 1-4,\n    # 13, 14 preserved).\n    margin = float(np.dot(w, ra - rb))\n    sd_m = s0 + s1 * math.log(1.0 + float(n_disc))\n    margin += sd_m * float(np.random.randn())\n\n    # --- Softmax over the two options. ---\n    z = beta * margin\n    z = min(max(z, -60.0), 60.0)\n    p_a = 1.0 / (1.0 + math.exp(-z))\n    p = np.array([p_a, 1.0 - p_a])\n\n    return (1.0 - epsilon) * p + epsilon * 0.5 * np.ones(2)",
  "policy": "def policy(probabilities):\n    probabilities = np.asarray(probabilities, dtype=np.float64)\n    probabilities = probabilities / probabilities.sum()  # Guard against float drift.\n    return np.random.choice(len(probabilities), p=probabilities)",
  "parameters": {
    "subject_seed": "[0, 1]",
    "regime_seed": "[0, 1]",
    "eta_base_seed": "[0, 1]",
    "beta_seed": "[0, 1]",
    "epsilon_seed": "[0, 1]",
    "mode_seed": "[0, 1]",
    "sigma_sensitive": "[1.75, 2.25]",
    "sigma_sensitive_fraction": "[0.45, 0.55]",
    "dilution_n0": "[9, 11]",
    "noise_s0": "[0.02, 0.04]",
    "noise_s1": "[0.025, 0.045]",
    "kappa_sd": "[0.35, 0.5]",
    "validities": "validities"
  },
  "rationale": "This is a minimal-diff edit of the ACCEPTED iter-1 TPQSAI base implementing the critic's three prescribed adjustments, plus one documented deviation where the critic's diagnosis is mechanically incomplete.\n\n(A) KAPPA TRAIT-EXPRESSION JITTER (critic's fix for Exp 9): implemented exactly as prescribed — sigma_i stays a stable per-subject trait, but sigma_eff is modulated per (subject, validity-bucket) by kappa ~ lognormal(median 1, sd ~ 0.45). Symmetric jitter leaves population means essentially intact, so Exp 11 (currently perfect at 2.88), Exp 15 and Exp 16 means should not move materially; the slight concavity-driven reduction in mean amplification is compensated by widening sigma_sensitive to [1.75, 2.25]. IMPORTANT CAVEAT, and the one place I extend beyond the critic's letter: I verified that the prestige amplifier is OFF on both Exp 9 ladders (top-two-endorsed-tier gaps of 0.40 and 0.50 give exp(-gap/0.10) <= 0.02), so kappa alone CANNOT attenuate the Exp 9 cross-ladder rate correlation — the critic's own rationale ('verify this') invites this check. The actual coupling channels are the stable beta, eta_base and epsilon traits. I therefore additionally (i) trimmed the engaged beta log-sd from 0.28 to 0.20 — the iter-1 widening was the main new coupling channel, and pi_6/pi_9 with log-sd 0.20/0.25 landed at 0.11/0.11 on this metric; (ii) rebalanced the eta variance split (stable sd 0.08, redraw sd 0.21, latent r ~ 0.13) so less eta variance is shared across buckets; and (iii) raised the per-trial margin-noise load slope s1 to ~0.035, which on the near-knife 24-feature kill cells (sd ~ 0.11-0.13 on margins of similar magnitude) pushes probabilities toward 0.5 and injects binomial sampling variance into the per-subject rates — precisely the channel by which pi_6 (per-cue diffusion ~0.17 effective margin noise) reached 0.11 while the iter-1 candidate (noise ~0.06) reached 0.47. The log(1+n_disc) scaling keeps low-load trials crisp, so Exps 1-4, 13, 14 are protected.\n\n(B) STEEPENED DILUTION (critic's fix for Exp 10): exponent 1.1 -> 1.4 with n0 ~ U[9,11]. D(6) = 0.67 matches the iter-1 value (0.68), so the Exp 11 AL cells and Exp 15 near-knife cells keep their calibration; but D falls faster through n_disc = 8-16 (0.58 / 0.44 / 0.35 vs iter-1's 0.61 / 0.50 / 0.42), pulling the Exp 10 twin-kill cells, Exp 16 large stacks, and Exp 17 high rungs toward the linear core. Note the amplifier is also off on Exp 10's own cells (gap 0.4-0.5), so the Exp 10 improvement is carried jointly by the reduced mean amplification elsewhere in the parameterization, the raised per-trial noise (which softens the decisive high-ratio cells that drive the p_high - p_twin rise), and the beta log-sd trim; the beta-median raise works slightly against Exp 10, an accepted residual risk given the larger Exp 14/17 gains.\n\n(C) BETA RESTORED (critic's fix for Exps 14/17): engaged lognormal median 1.85 -> 2.05 (log(2.05) = 0.718), log-sd 0.28 -> 0.20, clip unchanged. This should lift the Exp 14 winsorized ladder slope from 1.26 toward ~1.4, add level to Exp 17's B-cells, lift the undershooting Exps 2/5/12, and — via the narrowed spread — reinforce the Exp 9 decorrelation.\n\nEverything else — the defect sign inversion, context-normalized anti-validity weights with bucketed per-context eta, the stable quasi-bimodal trait, the guesser/hot-tail mixture, exact within-family linearity, and the coarse-gap experiments (5, 6, 8, 9, 10) with the amplifier essentially off — is retained verbatim from the accepted base. Expected net effect: Exp 9 from 0.47 toward ~0.2, Exp 10 from 0.41 toward ~0.33, Exp 14 from 1.26 toward ~1.40, Exp 17 from 0.11 toward ~0.2, with Exp 11 held near 2.88 and Exps 3/4/6/7/8/13/16/18 essentially unchanged — a strict aggregate-loss improvement over the 0.0805 floor."
}
```
```

## Usage

```json
{
  "input_tokens": 101169,
  "output_tokens": 22016,
  "reasoning_tokens": 16898,
  "provider": "Sail Research"
}
```
