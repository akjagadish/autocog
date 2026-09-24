# feedback_iter_07

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
- THEORY 1 = `pi_10`
- THEORY 2 = `pi_11`
- The recommendation below acts on THEORY 2 (= `pi_11`).

Replace pi_11 with a new theory that keeps the shared low-cue machinery (median anchoring of advertised validities, MAD-normalized deviance weight for below-median cues, lone low-cue amplification, and count/deviance-based attenuation of multiple low cues) but replaces the sign-flexible rank-ordered consensus term and the lapse floor with a low-bloc-size-gated high-cue term. Concretely: (i) when no below-median cues are diagnostic, fall back to equal weighting of all cues or median-cue majority so high cues do not dominate (captures Exp2); (ii) when a small number, roughly one or two, of below-median cues are diagnostic and their signed evidence is balanced or weak, give above-median cues a positive validity-gap weight so choices follow the high block (captures Exp1); (iii) when a large homogeneous bloc of below-median cues, roughly three or more per side, is diagnostic and balanced, make the high-cue gate negative or strongly downweighted so choices oppose the high block (captures Exp19); (iv) when low evidence is strongly one-sided, damp the high-cue term toward zero as conflict resolution; and (v) use a fixed-temperature softmax with no lapse floor (captures Exp15). The gate should be a continuous function of the number or mass of diagnostic below-median cues relative to the high bloc, not a strict three-cue license as in pi_10 and not an always-on rank bonus as in pi_11. This should preserve the good fits on Exp3-8, 14, 17, 18, and 20 while fixing the qualitative failures.


## CANDIDATE THEORY
People anchor advertised expert validities to the ensemble median and give below-median cues MAD-normalized deviance weight, with lone-cue amplification and count/deviance attenuation. Above-median cues enter only through a continuous validity-gap gate whose negative branch is licensed by a genuinely prominent, absolute high-gap cue rather than a mere normalized share or low-bloc count. When such a standout high cue exists, conflict damping is only partially suppressed, so the top cue can receive strong anti-following weight. When the high-gap profile is diffuse or modest, the gate returns control to the low-cue branch. Choice is a fixed-temperature softmax with no lapse floor, and heterogeneous multi-low-cue trials receive a lower effective temperature.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('GVC-prominence expects a (2, n_features) stimulus.')

    validities = np.asarray(parameters['validities'], dtype=float).reshape(-1)
    n_feat = validities.shape[0]
    if n_feat != stim.shape[1]:
        raise ValueError('validities length must match n_features.')

    c = float(parameters['c'])
    median = float(np.median(validities))
    abs_dev = np.abs(validities - median)
    mad = float(np.median(abs_dev))
    scale = mad + c
    if scale <= 1e-12:
        scale = 1e-12

    dev = np.maximum(0.0, median - validities) / scale
    gap = np.maximum(0.0, validities - median) / scale
    below = validities < median - 1e-12
    at_or_above = ~below

    diff = stim[0] - stim[1]
    active_below = below & (np.abs(diff) > 1e-12)
    n_active = int(np.count_nonzero(active_below))

    w0 = float(parameters['w0'])
    kappa = float(parameters['kappa'])
    gamma = float(parameters['gamma'])
    lam = float(parameters['lambda'])
    sigma = float(parameters['sigma'])
    phi = float(parameters['phi'])
    nu = float(parameters['nu'])
    opp_cap = float(parameters['opp_cap'])
    focal_ratio = float(parameters['focal_ratio'])

    raw = np.zeros(n_feat, dtype=float)
    dev_bounded = np.minimum(dev, 100.0)
    raw[below] = w0 + kappa * np.power(dev_bounded[below], gamma)

    def lone_amplification(idx):
        base_amp = 1.0 + lam * np.tanh(float(dev[idx]) / sigma)
        opp = at_or_above & (np.abs(diff) > 1e-12) & (diff * diff[idx] < 0.0)
        n_opp = int(np.count_nonzero(opp))
        return base_amp * (1.0 + nu * min(float(n_opp), opp_cap))

    if n_active == 1:
        focal = int(np.flatnonzero(active_below)[0])
        raw[focal] = raw[focal] * lone_amplification(focal)
    elif n_active >= 2:
        base_att = 1.0 / (1.0 + phi * (n_active - 1))
        active_devs = dev[active_below]
        i_max = int(np.argmax(active_devs))
        max_dev = float(active_devs[i_max])
        med_dev = float(np.median(active_devs))
        if med_dev > 1e-12 and max_dev > focal_ratio * med_dev:
            focal = int(np.flatnonzero(active_below)[i_max])
            others = below.copy()
            others[focal] = False
            raw[others] = raw[others] * base_att
            raw[focal] = raw[focal] * lone_amplification(focal)
        else:
            raw[below] = raw[below] * base_att

    beta = float(parameters['beta'])
    fallback_beta_scale = float(parameters['fallback_beta_scale'])

    if n_active == 0:
        evidence = float(np.mean(diff))
        beta_eff = beta * fallback_beta_scale
    else:
        beta_eff = beta
        mass_active = float(np.sum(raw[active_below]))
        E_low = float(np.dot(raw, diff))
        imbalance = 0.0
        if mass_active > 1e-12:
            imbalance = min(1.0, abs(E_low) / mass_active)

        K_bal = float(parameters['K_bal'])
        low_balance = 1.0 / (1.0 + (imbalance / K_bal) ** 2.0)

        H_spread = float(parameters['H_spread'])
        if n_active >= 2:
            spread_norm = float(np.std(validities[active_below])) / scale
            homog = 1.0 / (1.0 + (spread_norm / H_spread) ** 4.0)
        else:
            homog = 1.0

        beta_multi_scale = float(parameters['beta_multi_scale'])
        if n_active >= 2:
            beta_eff = beta * (beta_multi_scale + (1.0 - beta_multi_scale) * homog)

        D_half = float(parameters['D_half'])
        dmax = float(np.max(dev[active_below]))
        dev_damp = 1.0 / (1.0 + (dmax / D_half) ** 2.0)

        h_pos = float(parameters['h_pos'])
        h_neg = float(parameters['h_neg'])
        neg_bal_floor = float(parameters['neg_bal_floor'])
        neg_dev_floor = float(parameters['neg_dev_floor'])
        n_small = float(parameters['n_small'])
        s_small = float(parameters['s_small'])
        n_large = float(parameters['n_large'])
        s_large = float(parameters['s_large'])

        z_small = (float(n_active) - n_small) / s_small
        small_win = float(np.exp(-z_small * z_small))

        z_large = (float(n_active) - n_large) / s_large
        z_large = float(np.clip(z_large, -50.0, 50.0))
        large_win = 1.0 / (1.0 + float(np.exp(-z_large)))

        conc_boost_gain = float(parameters['conc_boost_gain'])
        conc_power = float(parameters['conc_power'])
        max_gap_share = 0.0
        max_gap_raw = 0.0
        if np.any(at_or_above):
            gap_above = gap[at_or_above]
            sum_gap = float(np.sum(gap_above))
            if sum_gap > 1e-12:
                max_gap_share = float(np.max(gap_above)) / sum_gap
            else:
                max_gap_share = 0.0
            raw_gap_above = np.maximum(0.0, validities[at_or_above] - median)
            max_gap_raw = float(np.max(raw_gap_above))
        else:
            max_gap_share = 0.0
            max_gap_raw = 0.0

        G_min = float(parameters['G_min'])
        G_s = float(parameters['G_s'])
        z_stand = float(np.clip((max_gap_raw - G_min) / G_s, -50.0, 50.0))
        standout = 1.0 / (1.0 + float(np.exp(-z_stand)))

        share_pow = max_gap_share ** conc_power
        conc_act = share_pow * standout
        conc_boost = 1.0 + conc_boost_gain * conc_act
        conc_mult = 1.0 + conc_boost_gain * conc_act

        neg_count_floor_low = float(parameters['neg_count_floor_low'])
        neg_count_floor_high = float(parameters['neg_count_floor_high'])
        neg_count_floor = neg_count_floor_low + (neg_count_floor_high - neg_count_floor_low) * standout
        neg_count_factor = neg_count_floor + (1.0 - neg_count_floor) * large_win

        strong_bal_floor = float(parameters['strong_bal_floor'])
        strong_dev_floor = float(parameters['strong_dev_floor'])
        eff_bal_floor = neg_bal_floor + (strong_bal_floor - neg_bal_floor) * standout
        eff_dev_floor = neg_dev_floor + (strong_dev_floor - neg_dev_floor) * standout
        neg_damp_bal = eff_bal_floor + (1.0 - eff_bal_floor) * low_balance
        neg_damp_dev = eff_dev_floor + (1.0 - eff_dev_floor) * dev_damp

        gate = (h_pos * small_win * homog * low_balance * dev_damp
                - h_neg * homog * conc_act * conc_mult * neg_count_factor
                * neg_damp_bal * neg_damp_dev)
        if n_active == 1:
            gate = gate * float(parameters['one_cue_gate_scale'])
        lower_bound = -h_neg * conc_boost
        gate = float(np.clip(gate, lower_bound, h_pos))

        rho = float(parameters['rho'])
        alpha = float(parameters['alpha'])
        branch_boost = float(parameters['branch_boost'])
        hi_boost = 1.0
        if n_active >= 2:
            hi_boost = 1.0 + branch_boost * small_win * homog * low_balance

        hi_pre = np.zeros(n_feat, dtype=float)
        gap_bounded = np.minimum(gap, 100.0)
        hi_pre[at_or_above] = rho * hi_boost * np.power(gap_bounded[at_or_above], alpha)

        raw_high = gate * hi_pre
        raw_all = raw + raw_high
        if n_active == 1:
            norm = float(np.sum(raw[below]))
        else:
            norm = float(np.sum(raw[below])) + float(np.sum(np.abs(raw_high[at_or_above])))
        if norm <= 1e-12:
            evidence = float(np.mean(diff))
        else:
            evidence = float(np.dot(raw_all, diff) / norm)

    logits = np.array([evidence, 0.0], dtype=float)
    z = beta_eff * (logits - float(np.max(logits)))
    z = np.clip(z, -60.0, 60.0)
    e = np.exp(z)
    probs = e / np.sum(e)
    return probs


`policy(probs) -> int`:
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))


`parameters`:
- c: [0.008, 0.012]
- w0: [0.007, 0.012]
- kappa: [2.30, 2.80]
- gamma: [1.25, 1.45]
- lambda: [8.0, 11.0]
- sigma: [0.14, 0.22]
- phi: [0.38, 0.55]
- nu: [0.15, 0.25]
- opp_cap: {5}
- focal_ratio: [3.0, 4.0]
- rho: [0.08, 0.14]
- alpha: [1.1, 1.8]
- h_pos: [0.005, 0.02]
- h_neg: [0.12, 0.20]
- neg_bal_floor: [0.25, 0.45]
- neg_dev_floor: [0.25, 0.45]
- n_small: [0.8, 1.2]
- s_small: [0.10, 0.25]
- n_large: [7.5, 9.2]
- s_large: [1.5, 2.5]
- K_bal: [0.40, 0.65]
- H_spread: [0.003, 0.012]
- D_half: [0.40, 0.65]
- beta: [1.8, 2.3]
- fallback_beta_scale: [0.05, 0.12]
- one_cue_gate_scale: [0.15, 0.35]
- conc_boost_gain: [2.5, 4.5]
- conc_power: {2}
- branch_boost: [0.0, 0.08]
- G_min: [0.08, 0.12]
- G_s: [0.01, 0.025]
- neg_count_floor_low: [0.03, 0.10]
- neg_count_floor_high: [0.30, 0.45]
- strong_bal_floor: [0.55, 0.80]
- strong_dev_floor: [0.55, 0.80]
- beta_multi_scale: [0.60, 0.90]
- validities: validities

`rationale`:
This is a minimal in-family edit of the accepted iter-7 base addressing the iter-7 diagnosis without repeating rejected levers. (1) The negative gate is now gated by absolute high-gap prominence: a logistic on the raw, unnormalized maximum above-median validity gap multiplies the normalized max_gap_share concentration term. Exp19's diffuse/modest high-gap profile therefore no longer triggers the anti-high branch, returning control to the low-cue path, while Exp20's genuinely large top gap still sets standout near 1. (2) The old fixed 0.35 negative count floor is replaced by a concentration-gated floor: low (0.03-0.10) in non-standout regimes and higher only when a true standout high cue exists, so low-bloc count alone cannot hold the negative channel open. (3) The low-balance and deviance damping floors are themselves raised toward 0.55-0.80 only under high absolute prominence, allowing Exp20's one-sided low evidence to coexist with a strong anti-top contribution instead of suppressing it. (4) A branch-specific inverse-temperature multiplier is applied when n_active >= 2 using the existing homogeneity score, lowering decision strength for heterogeneous multi-low-cue blocs and targeting Exp6 without touching h_pos, n_small, s_small, or branch_boost. The shared low-cue machinery, lone-cue normalization, no-lapse fixed-temperature softmax, and fallback branch are unchanged.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.1685 -> ACCEPTED
- iter 2: loss=0.1311 -> ACCEPTED
- iter 3: loss=0.1031 -> ACCEPTED
- iter 4: loss=0.1360 -> REJECTED
- iter 5: loss=0.1009 -> ACCEPTED
- iter 6: loss=0.1443 -> REJECTED
- iter 7: loss=0.0999 -> ACCEPTED
- iter 8 (current candidate you are grading): loss=0.0984 -> ACCEPTED
Running-best (last accepted) base: iter 8 at loss=0.0984.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1, 1, 0, 1]  B=[0, 0, 1, 0, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 1, 0, 1]
  A=[0, 1, 0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0

    def subject_slope(sdf):
        a = np.array(sdf['option_a_ratings'].tolist(), dtype=float)
        b = np.array(sdf['option_b_ratings'].tolist(), dtype=float)
        resp = sdf['response'].to_numpy(dtype=float)
        n = a.shape[0]

        top = np.full(n, -1, dtype=int)
        top[a[:, 0] > b[:, 0]] = 0
        top[b[:, 0] > a[:, 0]] = 1
        valid = top >= 0

        opp_wins = np.zeros(n, dtype=int)
        for j in range(1, a.shape[1]):
            opp_wins[(top == 0) & (b[:, j] > a[:, j])] += 1
            opp_wins[(top == 1) & (a[:, j] > b[:, j])] += 1

        choose_top = np.where(resp == top, 1.0, 0.0)

        x = opp_wins[valid].astype(float)
        y = choose_top[valid]
        if x.size < 2 or np.ptp(x) == 0:
            return 0.0
        return float(np.polyfit(x, y, 1)[0])

    slopes = data.groupby('subject_id', sort=False).apply(subject_slope)
    vals = np.atleast_1d(np.asarray(slopes, dtype=float))
    if vals.size == 0:
        return 0.0
    return float(np.mean(vals))
```

**Observed (real) value:** -0.0959 (var=0.0010)
**Candidate trajectory (this loop):**
  - iter 1: -0.1049 (var=0.0002) (Δ vs real -0.0090)
  - iter 2: -0.1046 (var=0.0003) (Δ vs real -0.0087)
  - iter 3: -0.1046 (var=0.0003) (Δ vs real -0.0087)
  - iter 4: -0.1049 (var=0.0003) (Δ vs real -0.0090)
  - iter 5: -0.1129 (var=0.0003) (Δ vs real -0.0170)
  - iter 6: -0.1207 (var=0.0003) (Δ vs real -0.0247)
  - iter 7: -0.1194 (var=0.0002) (Δ vs real -0.0235)
  - iter 8 (current): -0.1010 (var=0.0004) (Δ vs real -0.0051)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0006 (var=0.0002)
- pi_2: -0.1360 (var=0.0009)
- pi_3: -0.1493 (var=0.0006)
- pi_4: -0.1505 (var=0.0003)
- pi_5: -0.1025 (var=0.0003)
- pi_6: -0.1359 (var=0.0002)
- pi_7: -0.1246 (var=0.0002)
- pi_8: -0.1140 (var=0.0002)
- pi_9: -0.1230 (var=0.0002)
- pi_10: -0.1024 (var=0.0003)
- pi_11: -0.0895 (var=0.0003)

### Experiment 2
**Design**
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.6, 0.6], dtype=float)
    cue_order = np.argsort(-validities, kind='stable')

    y = data['response'].astype(float).to_numpy()
    dirs = np.zeros(len(data), dtype=float)
    margins = np.zeros(len(data), dtype=float)

    for i, (_, row) in enumerate(data.iterrows()):
        a = np.asarray(row['option_a_ratings'], dtype=int)
        b = np.asarray(row['option_b_ratings'], dtype=int)

        margins[i] = float(np.sum(a > b) - np.sum(b > a))

        direction = 0.0
        for j in cue_order:
            if a[j] > b[j]:
                direction = 1.0
                break
            if b[j] > a[j]:
                direction = -1.0
                break
        dirs[i] = direction

    d_pos = dirs == 1.0
    d_neg = dirs == -1.0
    m_pos = margins > 0.0
    m_neg = margins < 0.0

    directional_contrast = y[d_neg].mean() - y[d_pos].mean()
    margin_contrast = y[m_neg].mean() - y[m_pos].mean()

    return float(directional_contrast - margin_contrast)

```

**Observed (real) value:** -0.8592 (var=0.0589)
**Candidate trajectory (this loop):**
  - iter 1: -0.6574 (var=0.0128) (Δ vs real +0.2018)
  - iter 2: -0.6581 (var=0.0211) (Δ vs real +0.2011)
  - iter 3: -0.7003 (var=0.0147) (Δ vs real +0.1589)
  - iter 4: -0.6828 (var=0.0106) (Δ vs real +0.1764)
  - iter 5: -0.7574 (var=0.0100) (Δ vs real +0.1018)
  - iter 6: -0.7589 (var=0.0107) (Δ vs real +0.1003)
  - iter 7: -0.7208 (var=0.0164) (Δ vs real +0.1383)
  - iter 8 (current): -0.7239 (var=0.0106) (Δ vs real +0.1353)
**Other theories' values on this metric (for reference):**
- pi_2: -0.7597 (var=0.0423)
- pi_1: 0.7279 (var=0.0420)
- pi_3: -0.6946 (var=0.0665)
- pi_4: -0.8321 (var=0.0089)
- pi_5: -0.7328 (var=0.0080)
- pi_6: -0.8021 (var=0.0109)
- pi_7: -0.7272 (var=0.0075)
- pi_8: -0.7515 (var=0.0116)
- pi_9: -0.7776 (var=0.0100)
- pi_10: -0.6397 (var=0.0123)
- pi_11: -0.7337 (var=0.0110)

### Experiment 3
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data):
    pairs_ordered = [
        (0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        (1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
        (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
        (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0),
        (0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0),
        (1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0),
        (1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0),
        (1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0),
        (1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1),
        (1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1),
        (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1)
    ]
    order = {p: i for i, p in enumerate(pairs_ordered)}

    def pair_key(row):
        return tuple(list(row['option_a_ratings']) + list(row['option_b_ratings']))

    keys = data.apply(pair_key, axis=1)
    pos = keys.map(order)
    if pos.isna().any():
        return 0.0

    rank = (pos % 8).to_numpy(dtype=float) + 1.0
    chose_a = (data['response'].to_numpy() == 0).astype(float)

    subj_corrs = []
    for sid in data['subject_id'].unique():
        mask = data['subject_id'].to_numpy() == sid
        if np.std(rank[mask]) == 0.0 or np.std(chose_a[mask]) == 0.0:
            subj_corrs.append(0.0)
            continue
        r = float(np.corrcoef(rank[mask], chose_a[mask])[0, 1])
        subj_corrs.append(r)

    return float(np.mean(subj_corrs))
```

**Observed (real) value:** -0.3485 (var=0.0198)
**Candidate trajectory (this loop):**
  - iter 1: -0.4841 (var=0.0037) (Δ vs real -0.1356)
  - iter 2: -0.4676 (var=0.0070) (Δ vs real -0.1191)
  - iter 3: -0.4926 (var=0.0061) (Δ vs real -0.1441)
  - iter 4: -0.4857 (var=0.0063) (Δ vs real -0.1372)
  - iter 5: -0.5387 (var=0.0069) (Δ vs real -0.1902)
  - iter 6: -0.5329 (var=0.0070) (Δ vs real -0.1844)
  - iter 7: -0.5378 (var=0.0070) (Δ vs real -0.1893)
  - iter 8 (current): -0.4960 (var=0.0103) (Δ vs real -0.1475)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5096 (var=0.0191)
- pi_2: 0.0215 (var=0.0080)
- pi_1: 0.5469 (var=0.0262)
- pi_4: -0.2535 (var=0.0057)
- pi_5: -0.3414 (var=0.0125)
- pi_6: -0.3640 (var=0.0059)
- pi_7: -0.2726 (var=0.0068)
- pi_8: -0.4045 (var=0.0105)
- pi_9: -0.4851 (var=0.0096)
- pi_10: -0.4246 (var=0.0071)
- pi_11: -0.5035 (var=0.0056)

### Experiment 4
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    high_a = (1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
    low_a = (0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)

    a_key = data['option_a_ratings'].apply(tuple)
    mask_high_a = a_key == high_a
    mask_low_a = a_key == low_a

    if mask_high_a.sum() == 0 or mask_low_a.sum() == 0:
        return 0.0

    pA_high = 1.0 - float(data.loc[mask_high_a, 'response'].mean())
    pA_low = 1.0 - float(data.loc[mask_low_a, 'response'].mean())

    return float(pA_high - pA_low)

```

**Observed (real) value:** -0.6533 (var=0.0604)
**Candidate trajectory (this loop):**
  - iter 1: -0.7092 (var=0.0130) (Δ vs real -0.0558)
  - iter 2: -0.6817 (var=0.0200) (Δ vs real -0.0283)
  - iter 3: -0.7008 (var=0.0140) (Δ vs real -0.0475)
  - iter 4: -0.6942 (var=0.0210) (Δ vs real -0.0408)
  - iter 5: -0.7633 (var=0.0119) (Δ vs real -0.1100)
  - iter 6: -0.7875 (var=0.0091) (Δ vs real -0.1342)
  - iter 7: -0.7542 (var=0.0113) (Δ vs real -0.1008)
  - iter 8 (current): -0.6633 (var=0.0196) (Δ vs real -0.0100)
**Other theories' values on this metric (for reference):**
- pi_2: -0.1783 (var=0.0317)
- pi_3: 0.8683 (var=0.0156)
- pi_1: 0.6692 (var=0.0641)
- pi_4: -0.7908 (var=0.0142)
- pi_5: -0.6958 (var=0.0145)
- pi_6: -0.7150 (var=0.0209)
- pi_7: -0.6675 (var=0.0201)
- pi_8: -0.7300 (var=0.0175)
- pi_9: -0.7383 (var=0.0103)
- pi_10: -0.6833 (var=0.0225)
- pi_11: -0.5908 (var=0.0222)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    total = 0.0
    n = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        a_low = int(a[0]) + int(a[1])
        b_low = int(b[0]) + int(b[1])
        if a_low == b_low:
            continue
        low_side = 'A' if a_low > b_low else 'B'
        opposing = b if low_side == 'A' else a
        k = sum(1 for j in range(2, 9) if int(opposing[j]) == 1)
        if 3 <= k <= 6:
            chose_low = ((low_side == 'A' and int(row['response']) == 0) or
                         (low_side == 'B' and int(row['response']) == 1))
            total += float(chose_low)
            n += 1
    if n == 0:
        return 0.5
    return total / float(n)
```

**Observed (real) value:** 0.8275 (var=0.0166)
**Candidate trajectory (this loop):**
  - iter 1: 0.8500 (var=0.0044) (Δ vs real +0.0225)
  - iter 2: 0.8517 (var=0.0022) (Δ vs real +0.0242)
  - iter 3: 0.8387 (var=0.0034) (Δ vs real +0.0112)
  - iter 4: 0.8588 (var=0.0031) (Δ vs real +0.0312)
  - iter 5: 0.8688 (var=0.0035) (Δ vs real +0.0413)
  - iter 6: 0.8892 (var=0.0021) (Δ vs real +0.0617)
  - iter 7: 0.8817 (var=0.0028) (Δ vs real +0.0542)
  - iter 8 (current): 0.8208 (var=0.0055) (Δ vs real -0.0067)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8642 (var=0.0036)
- pi_2: 0.1492 (var=0.0089)
- pi_1: 0.1346 (var=0.0111)
- pi_3: 0.0708 (var=0.0019)
- pi_5: 0.8250 (var=0.0030)
- pi_6: 0.7621 (var=0.0044)
- pi_7: 0.8321 (var=0.0036)
- pi_8: 0.8754 (var=0.0037)
- pi_9: 0.8846 (var=0.0017)
- pi_10: 0.8383 (var=0.0040)
- pi_11: 0.8846 (var=0.0020)

### Experiment 6
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 1, 1]  B=[1, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    pos1 = {
        ((1, 1, 1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1, 1, 1)),
        ((0, 1, 1, 0, 0, 1, 0, 0), (1, 0, 0, 0, 0, 0, 1, 1)),
        ((0, 1, 1, 1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1, 1, 1)),
        ((0, 0, 1, 1, 0, 1, 0, 0), (0, 1, 0, 0, 0, 0, 1, 1)),
    }
    neg1 = {
        ((0, 0, 1, 0, 0, 1, 1, 0), (1, 1, 0, 0, 0, 0, 0, 1)),
        ((0, 0, 0, 0, 0, 1, 1, 1), (1, 1, 1, 0, 0, 0, 0, 0)),
        ((0, 0, 0, 1, 0, 1, 1, 0), (0, 1, 1, 0, 0, 0, 0, 1)),
        ((0, 0, 0, 0, 0, 1, 1, 1), (0, 1, 1, 1, 0, 0, 0, 0)),
    }
    early2 = {
        ((1, 1, 1, 1, 1, 0, 0, 0), (0, 0, 0, 0, 0, 1, 1, 1)),
        ((1, 1, 1, 1, 0, 1, 0, 0), (0, 0, 0, 0, 1, 0, 1, 1)),
        ((0, 1, 1, 1, 1, 1, 0, 0), (1, 0, 0, 0, 0, 0, 1, 1)),
        ((0, 1, 1, 1, 0, 0, 1, 1), (1, 0, 0, 0, 1, 1, 0, 0)),
    }
    late2 = {
        ((0, 0, 1, 1, 0, 1, 1, 1), (1, 1, 0, 0, 1, 0, 0, 0)),
        ((0, 0, 0, 1, 1, 1, 1, 1), (1, 1, 1, 0, 0, 0, 0, 0)),
        ((0, 0, 1, 0, 1, 1, 1, 1), (1, 1, 0, 1, 0, 0, 0, 0)),
        ((1, 0, 0, 0, 1, 1, 1, 1), (0, 1, 1, 1, 0, 0, 0, 0)),
    }
    akey = data['option_a_ratings'].apply(lambda x: tuple([int(v) for v in x]))
    bkey = data['option_b_ratings'].apply(lambda x: tuple([int(v) for v in x]))
    pairs = list(zip(akey, bkey))
    p_a = 1.0 - data['response'].to_numpy(dtype=float)
    p_pos1 = p_a[[k in pos1 for k in pairs]].mean()
    p_neg1 = p_a[[k in neg1 for k in pairs]].mean()
    p_early2 = p_a[[k in early2 for k in pairs]].mean()
    p_late2 = p_a[[k in late2 for k in pairs]].mean()
    block1_contrast = p_pos1 - p_neg1
    block2_contrast = p_early2 - p_late2
    return float(0.5 * (block1_contrast + block2_contrast))
```

**Observed (real) value:** 0.1525 (var=0.0118)
**Candidate trajectory (this loop):**
  - iter 1: 0.4188 (var=0.0074) (Δ vs real +0.2662)
  - iter 2: 0.4108 (var=0.0068) (Δ vs real +0.2583)
  - iter 3: 0.4096 (var=0.0079) (Δ vs real +0.2571)
  - iter 4: 0.4217 (var=0.0067) (Δ vs real +0.2692)
  - iter 5: 0.4733 (var=0.0067) (Δ vs real +0.3208)
  - iter 6: 0.4750 (var=0.0098) (Δ vs real +0.3225)
  - iter 7: 0.4750 (var=0.0092) (Δ vs real +0.3225)
  - iter 8 (current): 0.3687 (var=0.0114) (Δ vs real +0.2162)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0092 (var=0.0068)
- pi_4: 0.6663 (var=0.0054)
- pi_1: -0.4425 (var=0.0114)
- pi_3: -0.6067 (var=0.0096)
- pi_5: 0.3704 (var=0.0099)
- pi_6: 0.3529 (var=0.0067)
- pi_7: 0.4271 (var=0.0084)
- pi_8: 0.4717 (var=0.0121)
- pi_9: 0.4829 (var=0.0072)
- pi_10: 0.3608 (var=0.0101)
- pi_11: 0.3121 (var=0.0077)

### Experiment 7
**Design**
  A=[1, 1, 1, 1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    # High-validity experts are the first six rating positions.
    # Low-validity experts are positions 6,7,8; their split identifies the block.
    blocks = {}
    for row in data.itertuples(index=False):
        a = np.asarray(row.option_a_ratings, dtype=float)
        b = np.asarray(row.option_b_ratings, dtype=float)
        high_diff = a[:6] - b[:6]
        h = float(np.sum(high_diff))
        key = tuple(int(x) for x in high_diff)
        low_sign = int((a[6] - b[6]) + (a[7] - b[7]) + (a[8] - b[8]))
        yA = 1.0 if int(row.response) == 0 else 0.0
        d = blocks.setdefault(low_sign, {})
        entry = d.setdefault(key, [0.0, 0, h])
        entry[0] += yA
        entry[1] += 1
    vals = []
    for d in blocks.values():
        if len(d) < 2:
            continue
        entries = list(d.values())
        hs = np.array([e[2] for e in entries], dtype=float)
        ps = np.array([e[0] / e[1] if e[1] > 0 else 0.5 for e in entries], dtype=float)
        z = hs - hs.mean()
        denom = float(np.sqrt(np.mean(z * z)))
        if denom <= 0.0:
            continue
        vals.append(float(np.mean((z / denom) * ps)))
    return float(np.mean(vals)) if vals else 0.0
```

**Observed (real) value:** -0.0103 (var=0.0009)
**Candidate trajectory (this loop):**
  - iter 1: 0.0270 (var=0.0018) (Δ vs real +0.0372)
  - iter 2: 0.0050 (var=0.0023) (Δ vs real +0.0152)
  - iter 3: 0.0095 (var=0.0026) (Δ vs real +0.0197)
  - iter 4: 0.0196 (var=0.0022) (Δ vs real +0.0298)
  - iter 5: 0.0106 (var=0.0029) (Δ vs real +0.0208)
  - iter 6: 0.0089 (var=0.0023) (Δ vs real +0.0192)
  - iter 7: 0.0069 (var=0.0021) (Δ vs real +0.0172)
  - iter 8 (current): 0.0097 (var=0.0019) (Δ vs real +0.0200)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2803 (var=0.0012)
- pi_5: 0.0068 (var=0.0020)
- pi_1: 0.1888 (var=0.0043)
- pi_2: 0.2909 (var=0.0059)
- pi_3: 0.3502 (var=0.0022)
- pi_6: 0.0282 (var=0.0018)
- pi_7: 0.0224 (var=0.0016)
- pi_8: 0.0207 (var=0.0025)
- pi_9: 0.0194 (var=0.0024)
- pi_10: 0.0036 (var=0.0016)
- pi_11: 0.0097 (var=0.0020)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np

def metric(data):
    chose_a = (data['response'].to_numpy(dtype=float) == 0).astype(float)
    a_high = np.array([sum(r[2:]) for r in data['option_a_ratings']], dtype=float)
    b_high = np.array([sum(r[2:]) for r in data['option_b_ratings']], dtype=float)
    high_evidence = a_high - b_high
    if float(high_evidence.std()) < 1e-12 or float(chose_a.std()) < 1e-12:
        return 0.0
    return float(np.corrcoef(high_evidence, chose_a)[0, 1])
```

**Observed (real) value:** 0.0472 (var=0.0067)
**Candidate trajectory (this loop):**
  - iter 1: 0.0082 (var=0.0071) (Δ vs real -0.0390)
  - iter 2: 0.0027 (var=0.0091) (Δ vs real -0.0444)
  - iter 3: 0.0077 (var=0.0084) (Δ vs real -0.0394)
  - iter 4: 0.0125 (var=0.0067) (Δ vs real -0.0347)
  - iter 5: 0.0094 (var=0.0089) (Δ vs real -0.0378)
  - iter 6: -0.0101 (var=0.0090) (Δ vs real -0.0573)
  - iter 7: 0.0085 (var=0.0060) (Δ vs real -0.0387)
  - iter 8 (current): -0.0031 (var=0.0066) (Δ vs real -0.0502)
**Other theories' values on this metric (for reference):**
- pi_5: -0.0010 (var=0.0097)
- pi_4: 0.6671 (var=0.0037)
- pi_1: 0.3105 (var=0.0223)
- pi_2: 0.6061 (var=0.0242)
- pi_3: 0.7504 (var=0.0068)
- pi_6: 0.0794 (var=0.0104)
- pi_7: 0.0680 (var=0.0084)
- pi_8: -0.0014 (var=0.0073)
- pi_9: 0.0032 (var=0.0066)
- pi_10: -0.0054 (var=0.0055)
- pi_11: -0.0115 (var=0.0070)

### Experiment 9
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    df = data.copy()
    df['y'] = (df['response'] == 0).astype(float)
    df['low_a'] = df['option_a_ratings'].apply(lambda r: int(np.sum(np.asarray(r)[:4])))
    df['high_a'] = df['option_a_ratings'].apply(lambda r: int(np.sum(np.asarray(r)[4:9])))
    df['low_tie'] = (df['low_a'] == 2).astype(float)
    df['a_maj'] = (df['low_a'] == 3).astype(float)

    slopes = []
    for subj_id, sub in df.groupby('subject_id', sort=False):
        X = np.column_stack((
            np.ones(len(sub)),
            sub['low_tie'].to_numpy(dtype=float),
            sub['a_maj'].to_numpy(dtype=float),
            sub['high_a'].to_numpy(dtype=float)
        ))
        y = sub['y'].to_numpy(dtype=float)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        slopes.append(float(beta[3]))

    return float(np.mean(slopes))
```

**Observed (real) value:** 0.0032 (var=0.0007)
**Candidate trajectory (this loop):**
  - iter 1: -0.0003 (var=0.0004) (Δ vs real -0.0035)
  - iter 2: 0.0028 (var=0.0008) (Δ vs real -0.0004)
  - iter 3: -0.0011 (var=0.0008) (Δ vs real -0.0043)
  - iter 4: -0.0049 (var=0.0008) (Δ vs real -0.0080)
  - iter 5: -0.0001 (var=0.0007) (Δ vs real -0.0033)
  - iter 6: 0.0025 (var=0.0006) (Δ vs real -0.0007)
  - iter 7: 0.0016 (var=0.0007) (Δ vs real -0.0016)
  - iter 8 (current): -0.0035 (var=0.0010) (Δ vs real -0.0067)
**Other theories' values on this metric (for reference):**
- pi_6: 0.0500 (var=0.0010)
- pi_5: 0.0003 (var=0.0008)
- pi_1: 0.1056 (var=0.0010)
- pi_2: 0.1807 (var=0.0017)
- pi_3: 0.2205 (var=0.0005)
- pi_4: 0.1640 (var=0.0005)
- pi_7: 0.0023 (var=0.0005)
- pi_8: -0.0043 (var=0.0007)
- pi_9: 0.0048 (var=0.0012)
- pi_10: -0.0062 (var=0.0005)
- pi_11: 0.0018 (var=0.0009)

### Experiment 10
**Design**
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np

def metric(data):
    validities = np.array([0.5, 0.84, 0.843, 0.846, 0.849, 0.852, 0.855, 0.858, 0.861, 0.864, 0.867])
    order = np.argsort(validities)
    ranks = np.empty(len(validities), dtype=float)
    ranks[order] = np.arange(1, len(validities) + 1, dtype=float)
    u = (ranks - 1.0) / (len(validities) - 1.0)
    alpha = 4.75
    raw = np.power(1.0 - u, alpha)
    weights = raw / raw.sum()

    A = np.stack([np.asarray(v, dtype=float) for v in data['option_a_ratings']])
    B = np.stack([np.asarray(v, dtype=float) for v in data['option_b_ratings']])
    signed_evidence = (A - B) @ weights
    chose_a = 1.0 - np.asarray(data['response'], dtype=float)
    return float(np.mean(signed_evidence * chose_a))
```

**Observed (real) value:** 0.2001 (var=0.0006)
**Candidate trajectory (this loop):**
  - iter 1: 0.1912 (var=0.0004) (Δ vs real -0.0089)
  - iter 2: 0.1859 (var=0.0003) (Δ vs real -0.0142)
  - iter 3: 0.1892 (var=0.0003) (Δ vs real -0.0110)
  - iter 4: 0.1893 (var=0.0004) (Δ vs real -0.0108)
  - iter 5: 0.1930 (var=0.0002) (Δ vs real -0.0071)
  - iter 6: 0.1987 (var=0.0002) (Δ vs real -0.0014)
  - iter 7: 0.1971 (var=0.0003) (Δ vs real -0.0030)
  - iter 8 (current): 0.1787 (var=0.0003) (Δ vs real -0.0215)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1290 (var=0.0004)
- pi_6: 0.1905 (var=0.0004)
- pi_1: 0.1377 (var=0.0003)
- pi_2: 0.2228 (var=0.0005)
- pi_3: 0.2121 (var=0.0003)
- pi_4: 0.2385 (var=0.0002)
- pi_7: 0.2038 (var=0.0002)
- pi_8: 0.2034 (var=0.0002)
- pi_9: 0.1985 (var=0.0003)
- pi_10: 0.1841 (var=0.0004)
- pi_11: 0.1913 (var=0.0003)

### Experiment 11
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    response = np.asarray(data['response'], dtype=float)
    choice_a = (response == 0).astype(float)

    a = np.asarray(data['option_a_ratings'].tolist(), dtype=float)
    b = np.asarray(data['option_b_ratings'].tolist(), dtype=float)
    diff = a - b

    low_favors_a = diff[:, 0] > 0.5
    high_a_count = np.sum(diff[:, 1:], axis=1)

    la_k0 = low_favors_a & (high_a_count == 0)
    la_k7 = low_favors_a & (high_a_count == 7)
    lb_k0 = (~low_favors_a) & (high_a_count == 0)
    lb_k7 = (~low_favors_a) & (high_a_count == 7)

    def prop_a(mask):
        n = int(np.sum(mask))
        if n == 0:
            return 0.0
        return float(np.sum(choice_a[mask]) / n)

    la_contrast = prop_a(la_k7) - prop_a(la_k0)
    lb_contrast = prop_a(lb_k7) - prop_a(lb_k0)
    return 0.5 * (la_contrast + lb_contrast)

```

**Observed (real) value:** 0.4833 (var=0.0144)
**Candidate trajectory (this loop):**
  - iter 1: 0.4917 (var=0.0128) (Δ vs real +0.0083)
  - iter 2: 0.4867 (var=0.0104) (Δ vs real +0.0033)
  - iter 3: 0.5017 (var=0.0085) (Δ vs real +0.0183)
  - iter 4: 0.4917 (var=0.0098) (Δ vs real +0.0083)
  - iter 5: 0.4933 (var=0.0038) (Δ vs real +0.0100)
  - iter 6: 0.5167 (var=0.0100) (Δ vs real +0.0333)
  - iter 7: 0.5083 (var=0.0070) (Δ vs real +0.0250)
  - iter 8 (current): 0.5033 (var=0.0086) (Δ vs real +0.0200)
**Other theories' values on this metric (for reference):**
- pi_7: 0.7900 (var=0.0109)
- pi_5: 0.4950 (var=0.0193)
- pi_1: 0.8517 (var=0.0151)
- pi_2: 0.8650 (var=0.0147)
- pi_3: 0.9367 (var=0.0079)
- pi_4: 0.9033 (var=0.0065)
- pi_6: 0.6117 (var=0.0132)
- pi_8: 0.5033 (var=0.0080)
- pi_9: 0.4800 (var=0.0077)
- pi_10: 0.5083 (var=0.0098)
- pi_11: 0.5017 (var=0.0068)

### Experiment 12
**Design**
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    a = data['option_a_ratings']
    b = data['option_b_ratings']

    # Feature index 0 is the 0.50-validity cue.
    # Feature indices 1..5 are the five 0.80-validity cues.
    f1 = a.apply(lambda x: float(x[0])) - b.apply(lambda x: float(x[0]))
    grp = a.apply(lambda x: float(sum(x[1:6]))) - b.apply(lambda x: float(sum(x[1:6])))

    # High-conflict trials: the lone 0.50 cue points opposite the 0.80 bloc,
    # and the 0.80 bloc has a net majority of at least 3 (i.e. +/-3 or +/-5).
    conflict = ((f1 * grp) < 0) & (grp.abs() >= 3)

    if int(conflict.sum()) == 0:
        return 0.5

    # Follow the lone 0.50 cue: choose A when it favors A, choose B when it favors B.
    follow_lone = ((f1 > 0) & (data['response'] == 0)) | ((f1 < 0) & (data['response'] == 1))

    return float(follow_lone[conflict].mean())

```

**Observed (real) value:** 0.8375 (var=0.0133)
**Candidate trajectory (this loop):**
  - iter 1: 0.8381 (var=0.0048) (Δ vs real +0.0006)
  - iter 2: 0.8375 (var=0.0037) (Δ vs real +0.0000)
  - iter 3: 0.8337 (var=0.0052) (Δ vs real -0.0038)
  - iter 4: 0.8400 (var=0.0046) (Δ vs real +0.0025)
  - iter 5: 0.8744 (var=0.0044) (Δ vs real +0.0369)
  - iter 6: 0.8819 (var=0.0033) (Δ vs real +0.0444)
  - iter 7: 0.8738 (var=0.0031) (Δ vs real +0.0363)
  - iter 8 (current): 0.8237 (var=0.0057) (Δ vs real -0.0138)
**Other theories' values on this metric (for reference):**
- pi_5: 0.6706 (var=0.0116)
- pi_7: 0.3594 (var=0.0070)
- pi_1: 0.3181 (var=0.0072)
- pi_2: 0.1288 (var=0.0070)
- pi_3: 0.0569 (var=0.0024)
- pi_4: 0.1156 (var=0.0048)
- pi_6: 0.5475 (var=0.0085)
- pi_8: 0.8650 (var=0.0050)
- pi_9: 0.7462 (var=0.0067)
- pi_10: 0.8344 (var=0.0038)
- pi_11: 0.8413 (var=0.0040)

### Experiment 13
**Design**
  A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    a_low = data['option_a_ratings'].apply(lambda v: int(v[0]) + int(v[1]) + int(v[2]))
    b_low = data['option_b_ratings'].apply(lambda v: int(v[0]) + int(v[1]) + int(v[2]))
    d = (a_low - b_low).to_numpy(dtype=float)
    resp = data['response'].to_numpy(dtype=float)

    follow = np.where(d > 0, 1.0 - resp, resp)
    mask = np.abs(d) >= 1.0
    d = d[mask]
    resp = resp[mask]
    follow = follow[mask]

    unan = np.abs(d) == 3.0
    split = np.abs(d) == 1.0
    n_u = int(unan.sum())
    n_s = int(split.sum())
    if n_u < 5 or n_s < 5:
        return 0.0

    p_u = float(np.clip(follow[unan].mean(), 0.001, 0.999))
    p_s = float(np.clip(follow[split].mean(), 0.001, 0.999))

    def logit(p):
        p = float(p)
        return float(np.log(p / (1.0 - p)))

    emax = min(0.45, 2.0 * (1.0 - p_u) - 1e-3, 2.0 * (1.0 - p_s) - 1e-3)
    if emax <= 0.0:
        return 0.0

    best_e = 0.0
    best_loss = 1e12
    for j in range(401):
        e = emax * float(j) / 400.0
        denom = 1.0 - e
        if denom <= 0.0:
            continue
        q_u = (p_u - 0.5 * e) / denom
        q_s = (p_s - 0.5 * e) / denom
        if q_u <= 0.0 or q_u >= 1.0 or q_s <= 0.0 or q_s >= 1.0:
            continue
        loss = abs(logit(q_u) - 3.0 * logit(q_s))
        if loss < best_loss:
            best_loss = loss
            best_e = e
    return float(best_e)
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Candidate trajectory (this loop):**
  - iter 1: 0.0000 (var=0.0156) (Δ vs real +0.0000)
  - iter 2: 0.0000 (var=0.0173) (Δ vs real +0.0000)
  - iter 3: 0.0385 (var=0.0208) (Δ vs real +0.0385)
  - iter 4: 0.0860 (var=0.0194) (Δ vs real +0.0860)
  - iter 5: 0.0000 (var=0.0136) (Δ vs real +0.0000)
  - iter 6: 0.0532 (var=0.0187) (Δ vs real +0.0532)
  - iter 7: 0.0000 (var=0.0089) (Δ vs real +0.0000)
  - iter 8 (current): 0.0392 (var=0.0129) (Δ vs real +0.0392)
**Other theories' values on this metric (for reference):**
- pi_8: 0.0342 (var=0.0105)
- pi_5: 0.1512 (var=0.0183)
- pi_1: 0.0000 (var=0.0000)
- pi_2: 0.0000 (var=0.0103)
- pi_3: 0.0000 (var=0.0039)
- pi_4: 0.0080 (var=0.0088)
- pi_6: 0.0000 (var=0.0233)
- pi_7: 0.0585 (var=0.0121)
- pi_9: 0.0530 (var=0.0135)
- pi_10: 0.0000 (var=0.0173)
- pi_11: 0.1185 (var=0.0145)

### Experiment 14
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data):
    import numpy as np

    ratings = np.array(data['option_a_ratings'].tolist(), dtype=float)
    responses = np.asarray(data['response'].tolist(), dtype=int)
    subjects = np.asarray(data['subject_id'].tolist())
    choose_a = (responses == 0).astype(float)

    rel_idx = np.arange(1, 16)
    tallies = ratings[:, rel_idx].sum(axis=1)

    low = tallies <= 4
    high = tallies >= 8

    vals = []
    for sid in np.unique(subjects):
        idx = subjects == sid
        low_idx = idx & low
        high_idx = idx & high
        if np.count_nonzero(low_idx) == 0 or np.count_nonzero(high_idx) == 0:
            continue
        vals.append(float(np.mean(choose_a[high_idx]) - np.mean(choose_a[low_idx])))

    if not vals:
        return 0.0
    return float(np.mean(vals))
```

**Observed (real) value:** 0.0278 (var=0.0063)
**Candidate trajectory (this loop):**
  - iter 1: 0.3233 (var=0.0087) (Δ vs real +0.2956)
  - iter 2: 0.3406 (var=0.0149) (Δ vs real +0.3128)
  - iter 3: 0.0500 (var=0.0148) (Δ vs real +0.0222)
  - iter 4: 0.0633 (var=0.0143) (Δ vs real +0.0356)
  - iter 5: 0.0672 (var=0.0087) (Δ vs real +0.0394)
  - iter 6: 0.0250 (var=0.0128) (Δ vs real -0.0028)
  - iter 7: 0.0189 (var=0.0139) (Δ vs real -0.0089)
  - iter 8 (current): 0.0317 (var=0.0123) (Δ vs real +0.0039)
**Other theories' values on this metric (for reference):**
- pi_5: 0.2411 (var=0.0134)
- pi_8: 0.0667 (var=0.0215)
- pi_1: 0.3689 (var=0.0171)
- pi_2: 0.6983 (var=0.0351)
- pi_3: 0.8700 (var=0.0083)
- pi_4: 0.7717 (var=0.0078)
- pi_6: 0.3933 (var=0.0108)
- pi_7: 0.3833 (var=0.0125)
- pi_9: 0.0706 (var=0.0146)
- pi_10: -0.0217 (var=0.0149)
- pi_11: 0.0383 (var=0.0140)

### Experiment 15
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np


def metric(data):
    a0 = data['option_a_ratings'].apply(lambda x: int(x[0]))
    b0 = data['option_b_ratings'].apply(lambda x: int(x[0]))
    low_a = (a0 == 1) & (b0 == 0)
    low_b = (a0 == 0) & (b0 == 1)
    resp = data['response'].astype(float)
    ra = resp[low_a].mean()
    rb = resp[low_b].mean()
    if np.isnan(ra) or np.isnan(rb):
        return 0.0
    return float(rb - ra)

```

**Observed (real) value:** 0.7742 (var=0.0352)
**Candidate trajectory (this loop):**
  - iter 1: 0.5729 (var=0.0122) (Δ vs real -0.2013)
  - iter 2: 0.6982 (var=0.0098) (Δ vs real -0.0760)
  - iter 3: 0.6920 (var=0.0072) (Δ vs real -0.0822)
  - iter 4: 0.6902 (var=0.0057) (Δ vs real -0.0840)
  - iter 5: 0.7658 (var=0.0060) (Δ vs real -0.0084)
  - iter 6: 0.7676 (var=0.0045) (Δ vs real -0.0067)
  - iter 7: 0.7707 (var=0.0058) (Δ vs real -0.0036)
  - iter 8 (current): 0.7560 (var=0.0049) (Δ vs real -0.0182)
**Other theories' values on this metric (for reference):**
- pi_8: 0.1440 (var=0.0168)
- pi_9: 0.7493 (var=0.0087)
- pi_1: -0.0071 (var=0.0075)
- pi_2: 0.0831 (var=0.0046)
- pi_3: 0.0964 (var=0.0021)
- pi_4: 0.0831 (var=0.0052)
- pi_5: 0.3538 (var=0.0183)
- pi_6: 0.4036 (var=0.0073)
- pi_7: 0.2671 (var=0.0058)
- pi_10: 0.6627 (var=0.0044)
- pi_11: 0.7404 (var=0.0043)

### Experiment 16
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np


def metric(data: pd.DataFrame) -> float:
    low_a = np.asarray([x[0] for x in data['option_a_ratings']], dtype=float)
    resp = np.asarray(data['response'], dtype=float)
    follow_low = ((low_a == 1.0) & (resp == 0.0)) | ((low_a == 0.0) & (resp == 1.0))
    return float(np.mean(follow_low))
```

**Observed (real) value:** 0.8392 (var=0.0101)
**Candidate trajectory (this loop):**
  - iter 1: 0.6196 (var=0.0029) (Δ vs real -0.2196)
  - iter 2: 0.7010 (var=0.0027) (Δ vs real -0.1381)
  - iter 3: 0.7917 (var=0.0036) (Δ vs real -0.0475)
  - iter 4: 0.8198 (var=0.0026) (Δ vs real -0.0194)
  - iter 5: 0.8735 (var=0.0012) (Δ vs real +0.0344)
  - iter 6: 0.7900 (var=0.0028) (Δ vs real -0.0492)
  - iter 7: 0.8760 (var=0.0013) (Δ vs real +0.0369)
  - iter 8 (current): 0.8490 (var=0.0018) (Δ vs real +0.0098)
**Other theories' values on this metric (for reference):**
- pi_9: 0.8350 (var=0.0017)
- pi_8: 0.5506 (var=0.0025)
- pi_1: 0.5008 (var=0.0009)
- pi_2: 0.5677 (var=0.0017)
- pi_3: 0.5329 (var=0.0011)
- pi_4: 0.6112 (var=0.0009)
- pi_5: 0.6748 (var=0.0043)
- pi_6: 0.7431 (var=0.0023)
- pi_7: 0.6675 (var=0.0024)
- pi_10: 0.8198 (var=0.0021)
- pi_11: 0.8092 (var=0.0019)

### Experiment 17
**Design**
  A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np

VALIDITIES = np.array([0.7985, 0.7985, 0.7985, 0.7985, 0.7985, 0.7985, 0.8, 0.8, 0.8, 0.8, 0.86, 0.88, 0.9, 0.95, 0.99])
MEDIAN = float(np.median(VALIDITIES))
HIGH_MASK = VALIDITIES > MEDIAN + 1e-12
HIGH_WEIGHTS = VALIDITIES - MEDIAN

def metric(data):
    a = data['option_a_ratings'].tolist()
    b = data['option_b_ratings'].tolist()
    r = data['response'].to_numpy()
    match = []
    for ai, bi, ri in zip(a, b, r):
        ai_arr = np.asarray(ai, dtype=float)
        bi_arr = np.asarray(bi, dtype=float)
        ev = float(np.dot(HIGH_WEIGHTS[HIGH_MASK], ai_arr[HIGH_MASK] - bi_arr[HIGH_MASK]))
        if ev > 1e-12:
            rec = 0
        elif ev < -1e-12:
            rec = 1
        else:
            continue
        match.append(1.0 if ri == rec else 0.0)
    if not match:
        return float('nan')
    return float(np.mean(match))
```

**Observed (real) value:** 0.3783 (var=0.0022)
**Candidate trajectory (this loop):**
  - iter 1: 0.2606 (var=0.0030) (Δ vs real -0.1177)
  - iter 2: 0.2754 (var=0.0025) (Δ vs real -0.1029)
  - iter 3: 0.3008 (var=0.0021) (Δ vs real -0.0775)
  - iter 4: 0.2698 (var=0.0023) (Δ vs real -0.1085)
  - iter 5: 0.2971 (var=0.0032) (Δ vs real -0.0813)
  - iter 6: 0.2338 (var=0.0020) (Δ vs real -0.1446)
  - iter 7: 0.3648 (var=0.0037) (Δ vs real -0.0135)
  - iter 8 (current): 0.3848 (var=0.0060) (Δ vs real +0.0065)
**Other theories' values on this metric (for reference):**
- pi_10: 0.6129 (var=0.0043)
- pi_9: 0.5133 (var=0.0025)
- pi_1: 0.7302 (var=0.0056)
- pi_2: 0.7373 (var=0.0043)
- pi_3: 0.7998 (var=0.0012)
- pi_4: 0.7013 (var=0.0013)
- pi_5: 0.5038 (var=0.0035)
- pi_6: 0.5517 (var=0.0019)
- pi_7: 0.5256 (var=0.0024)
- pi_8: 0.4973 (var=0.0024)
- pi_11: 0.3569 (var=0.0017)

### Experiment 18
**Design**
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    a = np.array(data['option_a_ratings'].tolist(), dtype=float)
    b = np.array(data['option_b_ratings'].tolist(), dtype=float)

    low_side = (b[:, 0] > a[:, 0]).astype(float)
    resp = data['response'].to_numpy(dtype=float)
    agree_low = (resp == low_side).astype(float)

    k = (a[:, 2:] != b[:, 2:]).sum(axis=1).astype(float)

    frame = data[['subject_id']].copy()
    frame['agree_low'] = agree_low
    frame['k'] = k

    slopes = []
    for sid, sub in frame.groupby('subject_id'):
        x = sub['k'].to_numpy(dtype=float)
        y = sub['agree_low'].to_numpy(dtype=float)
        xm = x.mean()
        ym = y.mean()
        denom = np.sum((x - xm) ** 2)
        if denom <= 1e-12:
            slopes.append(0.0)
        else:
            slopes.append(float(np.sum((x - xm) * (y - ym)) / denom))

    return float(np.mean(slopes)) if slopes else float('nan')
```

**Observed (real) value:** 0.0004 (var=0.0000)
**Candidate trajectory (this loop):**
  - iter 1: -0.0118 (var=0.0000) (Δ vs real -0.0122)
  - iter 2: -0.0074 (var=0.0001) (Δ vs real -0.0078)
  - iter 3: -0.0014 (var=0.0000) (Δ vs real -0.0018)
  - iter 4: 0.0020 (var=0.0000) (Δ vs real +0.0016)
  - iter 5: -0.0034 (var=0.0001) (Δ vs real -0.0038)
  - iter 6: -0.0039 (var=0.0001) (Δ vs real -0.0043)
  - iter 7: -0.0014 (var=0.0000) (Δ vs real -0.0018)
  - iter 8 (current): -0.0007 (var=0.0000) (Δ vs real -0.0011)
**Other theories' values on this metric (for reference):**
- pi_9: -0.0098 (var=0.0000)
- pi_10: 0.0003 (var=0.0000)
- pi_1: -0.0154 (var=0.0000)
- pi_2: -0.0150 (var=0.0000)
- pi_3: -0.0178 (var=0.0000)
- pi_4: -0.0182 (var=0.0000)
- pi_5: -0.0135 (var=0.0000)
- pi_6: -0.0182 (var=0.0000)
- pi_7: -0.0180 (var=0.0000)
- pi_8: -0.0037 (var=0.0000)
- pi_11: 0.0024 (var=0.0000)

### Experiment 19
**Design**
  A=[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    kA = data['option_a_ratings'].apply(lambda x: sum(x[7:13]))
    low = data.loc[kA <= 2, 'response'].mean()
    high = data.loc[kA >= 3, 'response'].mean()
    contrast = low - high
    n_subj = data['subject_id'].nunique()
    se = np.sqrt(0.25 * (1.0 / 48.0 + 1.0 / 48.0))
    return float(contrast / (se / np.sqrt(n_subj)))
```

**Observed (real) value:** 7.8384 (var=0.7057)
**Candidate trajectory (this loop):**
  - iter 1: 33.3420 (var=0.5688) (Δ vs real +25.5036)
  - iter 2: 27.1643 (var=2.3845) (Δ vs real +19.3260)
  - iter 3: -0.3464 (var=7.7943) (Δ vs real -8.1848)
  - iter 4: -27.6839 (var=1.1911) (Δ vs real -35.5223)
  - iter 5: -1.1547 (var=12.0267) (Δ vs real -8.9931)
  - iter 6: -24.1332 (var=8.8101) (Δ vs real -31.9716)
  - iter 7: -16.3390 (var=1.7424) (Δ vs real -24.1774)
  - iter 8 (current): -15.2998 (var=1.4900) (Δ vs real -23.1381)
**Other theories' values on this metric (for reference):**
- pi_10: -1.2702 (var=0.7861)
- pi_11: -23.9023 (var=1.1453)
- pi_1: 24.9704 (var=1.1071)
- pi_2: 45.0911 (var=2.0926)
- pi_3: 59.8135 (var=0.5103)
- pi_4: 41.7713 (var=0.5290)
- pi_5: 11.9800 (var=1.1921)
- pi_6: 10.3923 (var=0.9683)
- pi_7: 16.4545 (var=0.9383)
- pi_8: 3.8105 (var=1.0363)
- pi_9: 2.1073 (var=0.8937)

### Experiment 20
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0
    top_a = data['option_a_ratings'].apply(lambda x: int(x[-1]))
    top_b = data['option_b_ratings'].apply(lambda x: int(x[-1]))
    top_favors_a = (top_a > top_b).astype(int)
    chose_a = (data['response'] == 0).astype(int)
    congruent = (top_favors_a == chose_a).astype(int)
    return float(2.0 * congruent.mean() - 1.0)

```

**Observed (real) value:** -0.7183 (var=0.0285)
**Candidate trajectory (this loop):**
  - iter 1: -0.4767 (var=0.0103) (Δ vs real +0.2417)
  - iter 2: -0.4571 (var=0.0063) (Δ vs real +0.2612)
  - iter 3: -0.1329 (var=0.0086) (Δ vs real +0.5854)
  - iter 4: -0.1192 (var=0.0120) (Δ vs real +0.5992)
  - iter 5: -0.0529 (var=0.0105) (Δ vs real +0.6654)
  - iter 6: -0.0383 (var=0.0148) (Δ vs real +0.6800)
  - iter 7: -0.0404 (var=0.0064) (Δ vs real +0.6779)
  - iter 8 (current): -0.0304 (var=0.0095) (Δ vs real +0.6879)
**Other theories' values on this metric (for reference):**
- pi_11: 0.1733 (var=0.0156)
- pi_10: -0.4392 (var=0.0094)
- pi_1: 0.6879 (var=0.0443)
- pi_2: -0.7512 (var=0.0213)
- pi_3: -0.5600 (var=0.1248)
- pi_4: -0.8175 (var=0.0028)
- pi_5: -0.4387 (var=0.0077)
- pi_6: -0.8054 (var=0.0024)
- pi_7: -0.3554 (var=0.0109)
- pi_8: -0.0658 (var=0.0075)
- pi_9: -0.5467 (var=0.0130)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is a faithful implementation of the arbiter's continuous low-bloc-gated high-cue mechanism and fixes the sign of Exp19, which pi_11 got backwards, while preserving several strong fits: Exp1 (-0.1049 vs -0.0959), Exp5 (0.850 vs 0.828), Exp9-13, and Exp18 are all near observed values, and Exp20 has the correct anti-top-cue sign even if too shallow (-0.477 vs -0.718). The main problem is magnitude calibration of the high-cue gate and the relative potency of the low-cue branch. Exp19 saturates: 33.34 vs observed 7.84, meaning the large homogeneous low-bloc anti-high gate is too strong. Exp14 (0.323 vs 0.028) and Exp6 (0.419 vs 0.153) overshoot in the opposite direction, indicating the small-bloc positive high-cue window is too strong or too wide. Exp15 (0.573 vs 0.774) and Exp16 (0.620 vs 0.839) under-follow the lone low cue, so the preserved low-cue machinery is being outcompeted by the above-median contribution. Exp2 (-0.657 vs -0.859) and Exp17 (0.261 vs 0.378) are also shallow. These are consistent with a gate whose positive and negative gains are too coarsely scaled, not with the wrong architecture.
Rationale: Keep the same low-cue machinery and continuous gate family. Rebalance parameters rather than redesign: (1) reduce h_neg from 0.70-1.10 to roughly 0.30-0.50, and/or shift n_large up toward 5.5-6.5 with a wider s_large, so the Exp19 contrast falls from 33 toward the observed 7.8 instead of hitting a near-deterministic anti-high choice. (2) tighten the positive small-bloc window by lowering s_small to about 0.5-0.8 and reducing h_pos to about 0.05-0.12; the current window is broad enough to produce too much high-block following in Exp14 and Exp6. (3) increase low-cue potency, e.g. raise kappa to 2.3-2.8 and lambda to 5.5-7.0, or re-normalize so the lone low cue is not diluted by the positive high-cue term, which should lift Exp15 and Exp16 toward their observed values. (4) Re-check Exp2 and Exp17 after those changes; if Exp2 remains too shallow, increase rho or alpha selectively inside the balanced small-bloc branch instead of raising h_pos globally. Do not reintroduce a lapse floor or the sign-flexible rank consensus term; the architecture is in-family and only the gate strength and low/high weight balance need correction.

**Outcome of your advice:** iter 1 candidate loss=0.1685 -> iter 2 candidate loss=0.1311 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: This candidate stays inside the prescribed continuous low-bloc-gated high-cue family and the parameter-level recalibration clearly helped overall: aggregate loss dropped from 0.1685 to 0.1311 and the gate accepted it. The wins are visible per experiment: Exp15 moved from 0.573 to 0.698 (real 0.774), Exp16 from 0.620 to 0.701 (real 0.839), Exp19 from 33.3 to 27.2 (real 7.8), Exp4 from 0.709 to 0.682 (real 0.653), Exp3 from 0.485 to 0.468 (real 0.349), and Exp7, Exp9, Exp11 tightened to near zero deltas. Several experiments are essentially perfect or excellent: Exp1 (-0.105 vs -0.096), Exp5 (0.852 vs 0.828), Exp9 (0.0028 vs 0.0032), Exp10 (0.186 vs 0.200), Exp11 (0.487 vs 0.483), Exp12 (0.8375 vs 0.8375), Exp13 (0.0 vs 0.0), Exp18 (-0.007 vs 0.0004). So the low-cue potency boost (kappa, lambda) worked as intended. However the candidate still misses the arbiter's target experiments by large margins. Exp19 remains about 3.5 times too strong at 27.2 vs 7.8 observed. Exp14 got WORSE after the h_pos/s_small reduction (0.323 to 0.341 vs 0.028 observed), which means that lever does not drive the Exp14 overshoot. Exp6 is still high (0.411 vs 0.153). Exp20 is still too shallow (-0.457 vs -0.718). Exp2 is unchanged and shallow (-0.658 vs -0.859). Exp16 (-0.138) and Exp15 (-0.076) still under-follow the lone low cue. This is a calibration and branching problem, not a wrong architecture, but it is not yet experiment-invariant enough to ship.
Rationale: Keep the same machine and gate family, but do not repeat the h_pos/s_small lever for Exp14 since lowering it failed there. Concrete next moves, in priority order: (1) Exp19 is still about 3.5x too large, so cut the negative gate much harder: h_neg to about 0.08-0.20, n_large up to 6.5-8.0, s_large wider (1.2-2.5), and relax the multi-cue attenuation phi to about 0.35-0.55 or add a floor to base_att so the low bloc keeps enough weight to partially counter the negative gate; the observed raw contrast is a partial reversal (about 0.15), not a full flip, so the low branch and the negative high term should be near-equal rather than the high term dominating. (2) Exp14 worsened despite weaker h_pos, so the +0.34 overshoot probably comes from the n_active == 0 equal-weight fallback integrating the 15 aligned cues with the full beta; soften that branch specifically, e.g., an effective temperature multiplier on beta of about 0.3-0.7 used only in the n_active == 0 fallback, or a count/redundancy cap on the fallback evidence so a large homogeneous bloc of tied-or-high cues is not integrated linearly. Re-check Exp2 after this, since the fallback also drives that metric and it currently sits 0.20 too high. (3) Exp20's negative gate is too weak while Exp19's is too strong, so gate the negative term on the concentration of the high bloc: multiply the negative gate by a factor that grows when the highest-validity cue stands alone or where the high-bloc gap is large, e.g., a term based on the maximum gap of the at-or-above cues; this raises the anti-top magnitude in Exp20 without touching Exp19's homogeneous bloc. (4) Exp15/16 still under-follow the lone low cue: raise lambda to 6.5-9.0 and sigma to 0.12-0.20, and in the n_active == 1 branch reduce the dilution of the lone cue by making the positive high-cue contribution smaller there (e.g., scale the one-cue branch gate by about 0.5), rather than boosting beta globally, which would re-inflate Exp14. (5) Exp2 and Exp17 remain shallow: raise rho or alpha selectively inside the balanced small-bloc branch (as previously suggested and not yet tried) instead of raising h_pos globally. Do not reintroduce the lapse floor or the sign-flexible rank consensus term, and do not switch mechanism families; these are all in-family calibrations of the gate shape, the fallback normalization, and the low/high weight balance.

**Outcome of your advice:** iter 2 candidate loss=0.1311 -> iter 3 candidate loss=0.1031 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: This candidate is faithful to the arbiter's prescribed continuous low-bloc-gated high-cue family and the aggregate loss improved from 0.1311 to 0.1031. The fallback-temperature fix worked as intended: Exp14 moved from 0.341 to 0.050 against observed 0.028. The lone-cue changes lifted Exp16 from 0.701 to 0.792 and Exp15 remains reasonably close at 0.692 versus 0.774. Exp1, Exp5, Exp9 through Exp12, and Exp18 are all near the observed values. However the current parameter set overshot the Exp19 correction: it now gives -0.346 whereas the real value is +7.838, a full sign reversal. The same weakened negative gate left Exp20 too shallow at -0.133 versus -0.718. Exp6 remains too high at 0.410 versus 0.153, Exp2 remains too shallow at -0.700 versus -0.859, Exp17 is still low at 0.301 versus 0.378, Exp3 is too negative at -0.493 versus -0.349, and Exp13 regressed from 0.000 to 0.039. These are calibration failures of the gate strength and low/high weight balance rather than evidence against the prescribed architecture.
Rationale: Regenerate with in-family calibration, not redesign. First, restore h_neg to an intermediate band around 0.30 to 0.50, and move n_large to about 5.8 to 6.8 with s_large around 1.0 to 1.8. This is the midpoint between the earlier saturating 0.70 to 1.10 regime and the current 0.12 to 0.20 regime; it should recover a positive but partial Exp19 contrast instead of the current negative one. Second, strengthen the Exp20 anti-top pathway through the concentration factor: raise conc_gain to about 1.5 to 2.5 or use max_gap_share raised to a power greater than 1, so a standout top cue receives a much stronger negative multiplier without re-amplifying Exp19's less concentrated high bloc. Third, trim the positive small-bloc window that is driving Exp6 over-following: lower h_pos to about 0.01 to 0.04 and/or set n_small to 0.8 to 1.5 with s_small to 0.20 to 0.40. If Exp2 and Exp17 become too shallow as a result, raise rho or alpha inside the balanced small-bloc branch rather than restoring h_pos globally. Fourth, to close the remaining Exp15 and Exp16 gaps, lower one_cue_gate_scale to about 0.15 to 0.35 and increase lambda toward 8 to 11 with sigma around 0.12 to 0.22. Keep fallback_beta_scale near its current range because it fixed Exp14. Recheck Exp3, Exp8, and Exp13 after these balance changes. Do not add a lapse floor or the sign-flexible rank-consensus term, and stay inside the continuous gated family.

**Outcome of your advice:** iter 3 candidate loss=0.1031 -> iter 4 candidate loss=0.1360 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The current candidate stays inside the prescribed continuous low-bloc-gated high-cue family, but its recalibration overshot: aggregate loss rose from 0.1031 (accepted iter-3 base) to 0.1360 and the gate marked it REJECTED. The single biggest failure is Exp19: the proposed restoration of h_neg to 0.30-0.50 with n_large 5.8-6.8 turned a small negative contrast (-0.35) into a strongly negative one (-27.68), the opposite of the intended 'partial positive' recovery (real +7.84). Tellingly, the Exp20 anti-top fix (conc_gain 1.5-2.5) did essentially nothing: Exp20 moved only from -0.133 to -0.119 vs real -0.718. This strongly suggests the negative gate is already hitting its hard clip at [-h_neg, h_pos], so multiplying by conc_factor cannot amplify it — the parameter was inert at its upper range. Secondary regressions vs the accepted base: Exp17 (0.301 -> 0.270, real 0.378), Exp13 (0.039 -> 0.086, real 0.000), Exp14 (0.050 -> 0.063, real 0.028), Exp6 (0.410 -> 0.422, real 0.153), Exp2 (-0.700 -> -0.683, real -0.859), Exp7 (0.0095 -> 0.0196, real -0.010), Exp5 (0.839 -> 0.859, real 0.828). Gains were small: Exp16 (0.792 -> 0.820, real 0.839) and Exp11/Exp12 marginal. The architecture looks right and most experiments (Exp1, 4, 9, 10, 18, 8) are stable; the problem is the shape/bounds of the gate as a function of low-bloc size/homogeneity, low-evidence balance, and high-gap concentration, plus the normalization that dilutes the lone low cue.
Rationale: The iter-4 direction was rejected, so do NOT repeat 'restore h_neg to 0.30-0.50 with n_large 5.8-6.8' — that lever crashed Exp19. Build on the accepted iter-3 base with different, smaller in-family moves: (1) keep iter-3 h_neg and instead make the MAGNITUDE of the negative gate concentration-driven: remove or scale the gate's lower clamp by the concentration factor (e.g., effective lower bound -h_neg * (1 + conc_gain_p * max_gap_share^p)) so Exp20's stand-out top cue can push the anti-top weight well below -h_neg, while Exp19's diffuse high-gap profile stays near iter-3 strength. The inertness of conc_gain in iter 4 implicates the clamp or the norm; fix that rather than raising h_neg. (2) Pull Exp19 out of the negative window structurally: raise n_large to roughly 7.0-8.5 and/or require BOTH homogeneity and a n_active range for the negative term, so Exp19's configuration gets a near-zero or slightly positive gate while Exp20 still qualifies through its distinct high-gap concentration. If Exp19's lows are balanced, a small fixed positive weight in the imbalance~0 / diffuse-high case (e.g., gate ~ +0.01-0.03) plus the low branch would push the contrast toward the observed +7.84 without re-saturating to iter-2's +27. (3) Exp6 over-following: do not touch h_pos again (already tiny and the lever was rejected); instead attenuate the positive gate for heterogeneous multi-cue low blocs (stronger H_spread or phi dependence), then compensate Exp2/Exp17 with modest rho/alpha increases restricted to the balanced, homogeneous small-bloc branch — these two experiments remain the shallowest after Exp20. (4) Lone low cue (Exp15): keep lambda 8-11 but reduce dilution in the n_active==1 branch by excluding the near-zero high-cue terms from the normalization (use sum(raw[below]) as norm there), which should lift Exp15 without touching Exp16. (5) After rebalancing, re-check Exp13 and Exp14 and raise beta slightly (2.0-2.5) or tighten fallback_beta_scale (0.08-0.20) if noise-based metrics regress. Keep the no-lapse fixed-temperature softmax, the normalization scheme, and the continuous gate — all changes are within the prescribed gate family.

**Outcome of your advice:** iter 4 candidate loss=0.1360 -> iter 5 candidate loss=0.1009 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate was accepted by the gate (aggregate loss 0.1009, best so far) and it is a faithful in-family implementation of the prescribed continuous low-bloc-gated high-cue architecture with the shared low-cue machinery, no-lapse fixed-temperature softmax, and continuous gate. It genuinely fixed several previous weak points: Exp13 is back to exactly 0.000 (real 0.000), Exp15 is now 0.766 against 0.774 (previously 0.69), Exp16 is 0.874 against 0.839, and Exp1, Exp5, Exp9, Exp10, Exp11, Exp18 all remain close. However, the two signature targets the arbiter assigned this family remain qualitatively missed, and one experiment regressed badly. Exp19 is still the wrong sign: candidate -1.15 against real +7.84, with huge subject variance (12.03), indicating the negative anti-high gate never opens in the intended regime and choices sit near a flip boundary. Exp20 remains ~10-14 times too shallow: -0.053 against -0.718, so the concentration-driven anti-top amplification is essentially inert. The reason is visible in the code: conc_boost only rescales the lower clamp bound, not the gate magnitude, and the gate magnitude itself is dominated by large_win, which is nearly zero because n_large sits at 7.5-9.2 while the diagnostic low-bloc counts in these experiments are much smaller, plus low_balance and dev_damp further shrink it. So the negative term never reaches -h_neg, let alone beyond it. Secondary problems: Exp6 worsened to 0.473 against 0.153 (the positive small-bloc pathway still over-follows high cues), Exp2 remains shallow (-0.757 vs -0.859), Exp17 is low (0.297 vs 0.378), Exp14 slightly overshoots (0.067 vs 0.028), and Exp3/Exp4 drifted a bit. These are calibration and gate-shape problems, not evidence against the architecture.
Rationale: Regenerate with in-family calibrations, keeping h_neg fixed since raising it to 0.30-0.50 was the rejected lever that crashed Exp19. Priority 1: make the negative gate actually fire. (a) Apply the concentration factor to the gate magnitude, not only the clip bound: compute gate = (h_pos * small_win - h_neg * large_win * (1 + conc_boost_gain * max_gap_share ^ conc_power)) * homog * low_balance * dev_damp and keep the lower clip at -h_neg * (1 + conc_boost_gain * max_gap_share ^ conc_power); consider conc_power = 1 and conc_boost_gain in 2.0-3.5 so an Exp20 standout top cue drives the gate well below its diffuse value. (b) The large-window trigger is centered wrong for Exp19: n_large 7.5-9.2 with s_large 1.5-2.5 means large_win is near zero at the low-bloc counts that actually occur. Move n_large to roughly 3.0-5.0 with s_large 0.8-1.5, or better, replace absolute n_active with a continuous function of the diagnostic below-median cue mass relative to the high bloc as the arbiter specified, so the negative gate opens for Exp19 without re-saturating to iter-2's +27. (c) Check that Exp19's low cues are actually classified below-median and that dev_damp/low_balance are not suppressing them; consider raising D_half or relaxing dev_damp for large homogeneous balanced blocs. Priority 2: fix Exp6 without touching h_pos again. Cap branch_boost to 0.05-0.25 and raise H_spread to 0.02-0.06 so heterogeneous multi-cue low blocs lose the positive high-cue boost while homogeneous small blocs (Exp2/Exp17) keep it; if Exp2 and Exp17 dip, compensate only inside the balanced homogeneous branch with modest rho/alpha increases rather than h_pos. Priority 3: tighten fallback_beta_scale to 0.05-0.12 to bring Exp14 from 0.067 toward 0.028. Keep the n_active==1 normalization that uses sum(raw[below]) (it fixed Exp15 and Exp16), keep the no-lapse fixed-temperature softmax, and do not reintroduce the sign-flexible rank consensus or a lapse floor. After these changes, re-check Exp3, Exp4, Exp12, and Exp16; Exp16 is now slightly over (0.874 vs 0.839), so a small lambda or beta reduction may be needed once the gate is recalibrated.

**Outcome of your advice:** iter 5 candidate loss=0.1009 -> iter 6 candidate loss=0.1443 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The candidate is a faithful in-family implementation of the prescribed continuous low-bloc-gated high-cue architecture, but its recalibration is badly miscalibrated and the gate rejected it: aggregate loss rose from the accepted 0.1009 (iter 5) to 0.1443. The dominant failure is Exp19: the candidate now produces -24.13 against the observed +7.84, essentially reproducing the rejected pi_11 value of -23.90. This happened because moving n_large from 7.5-9.2 down to 3.5-5.0 opened the negative large_win branch on the low-bloc counts actually present in Exp19, but with the wrong overall sign/strength. The same negative gate over-fires in Exp17, which fell from 0.297 (iter 5) to 0.234 against the real 0.378. Exp20 remains almost flat at -0.038 against -0.718: the concentration factor is present in the gate arithmetic now, but it is multiplied afterwards by low_balance and dev_damp, which are near their minimum when low evidence is strongly one-sided, so the anti-top contribution never reaches usable magnitude. Exp6 is still a large miss at 0.475 versus 0.153, showing the H_spread and branch_boost changes did not address the over-following problem. Secondary regressions from the current parameter set include Exp16 (0.790 vs 0.839 real, caused by the lambda reduction), Exp13 (0.053 vs 0.000), and mild drift in Exp4, Exp5, Exp8, Exp11 and Exp12. The one clear success is Exp14: the tightened fallback_beta_scale moved it to 0.025 against 0.028. Exp15 and Exp10 also remain good, but the losses are far larger than the gains.
Rationale: Build on the accepted iter-5 base, not on this rejected candidate. Keep the fallback_beta_scale tightening that fixed Exp14, but discard the n_large=3.5-5.0, the broad negative large_win activation, the H_spread widening, the lambda reduction and the phi reduction; those changes caused the Exp19/Exp17/Exp16 regressions. Concretely: (1) restore n_large to roughly the iter-5 range and do not let low-bloc count alone activate the negative gate. Instead gate the negative branch on high-bloc concentration--for example, a separate concentration-activation term based on max_gap_share--so Exp19's diffuse high-gap profile gets a near-zero or small positive high-cue gate and the low-cue branch can drive the observed positive contrast, while Exp20's standout top cue still receives a strong negative contribution. (2) For Exp20, stop multiplying the concentration-negative term by the full low_balance and dev_damp suppression. Those factors are near zero exactly when low evidence is one-sided and the top cue stands out; use a floor or an interaction that lets high concentration override the conflict damping, e.g. multiply the negative concentration term by something like (0.3+0.7*low_balance) and (0.3+0.7*dev_damp), or split it into an only partially damped negative channel. This is the in-family fix for why conc_boost remains inert. (3) For Exp6, do not reuse H_spread/branch_boost as the main lever since it did not move the metric. Shrink the positive small-bloc window more directly: n_small toward 0.8-1.2, s_small toward 0.10-0.25, lower h_pos toward 0.005-0.02, and cap branch_boost near zero for n_active>=2. If Exp2 and Exp17 then become too shallow, compensate only inside the balanced homogeneous small-bloc branch via a modest rho/alpha increase, not via h_pos or branch_boost. (4) Restore lambda to the higher accepted range (about 8-11) to recover Exp16 and keep Exp15, while leaving one_cue_gate_scale low so lone-cue choices do not get diluted. (5) After these changes, explicitly re-check Exp13 should return to 0.000, and Exp17 should return toward the observed 0.378 rather than being pulled negative by the large-bloc gate. Do not change mechanism family and do not reintroduce a lapse floor or sign-flexible rank consensus.

**Outcome of your advice:** iter 6 candidate loss=0.1443 -> iter 7 candidate loss=0.0999 -> the gate marked it ACCEPTED.

### Iteration 7 (most recent)
Verdict: regenerate
Interpretation: The candidate is a faithful in-family implementation of the prescribed continuous low-bloc-gated high-cue architecture, and it is the accepted best so far (aggregate loss 0.0999). Many fits are now excellent or very close: Exp9 (0.0016 vs 0.0032), Exp10 (0.1971 vs 0.2001), Exp11 (0.5083 vs 0.4833), Exp13 (0.0000 vs 0.0000), Exp14 (0.0189 vs 0.0278), Exp15 (0.7707 vs 0.7742), Exp16 (0.8760 vs 0.8392), Exp17 (0.3648 vs 0.3783), and Exp18 (-0.0014 vs 0.0004). However, the two signature experiments this mechanism family was explicitly introduced to fix remain qualitatively missed. Exp19 is still wrong-sign and enormous: candidate -16.34 versus real +7.84 (Δ -24.18), with candidate variance 1.74, so the negative gate still opens in the diffuse/modest high-gap regime where it should not. Exp20 remains essentially at chance: -0.040 versus real -0.718 (Δ +0.678), so the anti-top concentration pathway is still far too weak when it should dominate. A third large miss is Exp6: 0.475 versus 0.153, nearly three times the observed block contrast. The likely core defect is that the concentration trigger `max_gap_share = max(gap_above)/sum(gap_above)` is purely normalized within the above-median bloc: a single modest above-median cue automatically receives share 1.0, making it indistinguishable from a genuinely standout top cue. That is the most plausible reason Exp19 gets a large negative high-cue gate while Exp20's true standout top cue remains under-weighted. In addition, the fixed 0.35 floor in `neg_count_factor` keeps the negative channel partially open even when the large-bloc count window is off, and the balance/deviance damping floors leave Exp20's negative contribution too small. The architecture is therefore directionally correct but the gate's prominence metric, count floor, and conflict-damping interaction are miscalibrated.
Rationale: Regenerate within the same continuous low-bloc-gated high-cue family, building on the accepted iter-7 base. Do not raise h_neg globally and do not move n_large back down; those were the previously rejected levers that crashed Exp19. Concrete in-family changes: (1) Make the negative gate depend on absolute high-gap prominence, not only normalized share, e.g. multiply or gate `max_gap_share^conc_power` by a factor such as `tanh(max_gap / G_half)` or a logistic in `max_gap - G_min`, so a modest lone above-median cue in Exp19 no longer triggers the anti-high branch while Exp20's genuinely large top gap still fires strongly. (2) Reduce the `neg_count_factor` floor toward about 0.05-0.12 or make it concentration-gated, so the negative branch is near zero in non-standout regimes; after this, verify Exp19's low-cue branch recovers a positive contrast near the observed +7.84 rather than re-saturating. (3) For Exp20, let high absolute concentration override the low-balance/deviance damping more strongly: make `neg_bal_floor` and `neg_dev_floor` concentration-dependent so they rise toward roughly 0.5-0.8 only when a true standout top cue is present, instead of uniformly suppressing the anti-top contribution. (4) For Exp6, do not re-tighten n_small/s_small or touch h_pos/branch_boost again—those levers already failed; instead soften decision strength specifically for heterogeneous multi-cue low blocs via a branch-specific inverse-temperature (for example a beta_multi around 0.6-0.9 applied when n_active>=2 and the below-median validity spread is large), which should shrink the 0.475 contrast toward 0.153 without weakening lone-cue effects in Exp15/16. (5) Keep the no-lapse fixed-temperature softmax and all shared low-cue machinery intact, and re-check Exp2, Exp3, Exp4, Exp5, and Exp17 after rebalancing to avoid regressions in the currently good fits.

**Outcome of your advice:** iter 7 candidate loss=0.0999 -> CURRENT candidate loss=0.0984 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "Iter 8 is a real improvement and was correctly accepted: aggregate loss fell to 0.0984, and many fits are now excellent or close \u2014 Exp1 (-0.101 vs -0.096), Exp4 (-0.663 vs -0.653), Exp5 (0.821 vs 0.828), Exp7, Exp9, Exp11, Exp12, Exp14-18. Exp16 and Exp17 also tightened. The beta_multi_scale lever made Exp6 less extreme (0.475 -> 0.369, real 0.153). However the two signature experiments this family was introduced to fix remain qualitatively wrong, so the theory is not yet experiment-invariant enough to ship. Exp19 is still wrong-signed and large: -15.30 vs +7.84, with low variance, meaning the candidate deterministically opposes the block humans follow. The absolute-gap prominence gate (G_min 0.08-0.12) is evidently still firing in Exp19's high-gap regime, and max_gap_share^2 can mark a cue as concentrated whenever the above-median set is small, so the anti-high branch stays open. Exp20 is still effectively at chance: -0.030 vs -0.718. This is consistent with, and the code makes likely, a structural bypass: when n_active == 0 the current implementation returns a weak mean(diff) fallback and never evaluates the prominence-gated high-cue term at all, so the strong anti-top contribution promised by the theory cannot manifest. Exp6 remains a large miss (0.369 vs 0.153), showing the multi-cue temperature correction is insufficient but directionally correct. There are also smaller residual drifts: Exp2 is shallow (-0.724 vs -0.859), Exp3 too negative (-0.496 vs -0.349), Exp8 slightly negative (-0.003 vs 0.047), Exp10 a bit low (0.179 vs 0.200), and Exp13 noisy (0.039 vs 0.000). The architecture is still in-family; the remaining problems are branch bypass, prominence metric calibration, and decision-strength scaling.",
  "verdict": "regenerate",
  "rationale": "Keep the prescribed continuous low-bloc-gated high-cue family, the shared low-cue machinery, fixed-temperature softmax, and no-lapse floor. Do not repeat the rejected h_neg restoration or n_large-down moves. Three in-family corrections should be tried on top of the accepted iter-8 base. (1) Fix the n_active == 0 bypass for Exp20: the fallback branch must still evaluate the high-cue gate rather than immediately returning mean(diff) with the small fallback temperature. For a standout top cue with no active below-median cues, add the gated (negative) high-gap evidence to the fallback evidence, with enough effective temperature or weight that Exp20's anti-top congruence moves from near chance toward the observed -0.718. Re-check Exp2 and Exp14 afterward to ensure the no-high-dominance fallback still works. (2) Make the negative-gate prominence truly selective for Exp19 vs Exp20: use a relative/sharper metric, e.g. require the largest above-median raw gap to exceed the second-largest gap by a margin, and/or raise G_min to roughly 0.15-0.25 with narrower G_s. Also ensure large_win cannot act as a substitute for standout prominence in the negative channel; the negative branch should be near zero for the diffuse/modest high-gap profile in Exp19 so the low-cue path can produce the observed positive contrast. (3) Push the accepted beta_multi_scale direction further for Exp6: lower it to about 0.35-0.65 and/or strengthen the heterogeneity dependence so heterogeneous multi-low-cue trials are softer, without touching h_pos, n_small, or branch_boost. After these changes explicitly re-check Exp2, Exp3, Exp8, Exp13, Exp14, Exp17 and Exp20, since they are the main residual or collateral targets."
}
```

## Usage

```json
{
  "input_tokens": 49313,
  "output_tokens": 14702,
  "reasoning_tokens": 13826,
  "provider": "NextBit"
}
```
