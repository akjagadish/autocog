# feedback_iter_01

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
- THEORY 2 = `pi_9`
- The recommendation below acts on THEORY 2 (= `pi_9`).

Propose a new theory in which subjects combine a damped validity-weighted inverse-valence tally with a continuously active, smoothly attenuated top-coalition bonus. Unlike pi_9, the top-coalition effect must not be gated by |D|<=window or abs(E)<=tie_tol; it should survive nonzero D at modest strength, so E1 top-following is partial (~0.30-0.40, not 0.145) and E20's h-trend is large (~3.4). Unlike pi_10, the tied-top component should be much weaker and saturating: no large m=2 multiplier of 2.7, no extreme m-based amplifications, and no sharp suppression for three-cue top groups facing five lower cues; it should increase gradually with top-group size and validity gap, decay smoothly with opposition and net tally, and use tanh-type saturation so E2 stays near 0.09 and E16 near 0.40. Unique-top precedence should be probabilistic and still available in large multi-cue conflicts but with conflict-dependent damping that keeps it partial rather than ceiling-level. Large-conflict validity weighting should follow pi_9's damped-lambda style rather than pi_10's boost, letting raw majority dominate when |D| is large while the small top-coalition bonus partially counters it. Finally, subject-level parameters should remain heavy-tailed but with enough variance in the top-coalition and precedence components to reproduce the observed between-subject variances (e.g., E1 ~0.05, E8 ~0.42, E16 ~0.33), avoiding pi_9's near-zero population variance.


## CANDIDATE THEORY
Subjects code a binary expert rating of 0 as a clean/favorable cue and 1 as a defect, so every discriminating expert contributes inverse-valence evidence. The default evidence is a damped validity-weighted tally, with lambda modest but only mildly reduced in large conflicts so high-validity cues still move moderately against raw majorities. A continuously active, smoothly attenuated top-coalition bonus follows tied highest-validity expert groups: it grows approximately linearly with top-group size for small coalitions (m=2-3) but is sharply and smoothly penalized for larger coalitions (m=4+), while opposition from lower-validity cues is damped in a way that scales with top-group size. Unique highest-validity expert precedence is probabilistic and survives large conflicts, but is attenuated when the unique top cue merely agrees with the raw majority. Choice is a softmax over the resulting evidence with heavy-tailed subject-level gain and lapse, preserving between-subject heterogeneity.

`predict(parameters, state, history) -> np.ndarray`:
import math
import numpy as np
from scipy.stats import t as student_t


def _tqt(u, mu, sigma, lo, hi, df):
    u = float(np.clip(u, 1e-8, 1.0 - 1e-8))
    q_lo = student_t.cdf((lo - mu) / sigma, df)
    q_hi = student_t.cdf((hi - mu) / sigma, df)
    q = q_lo + u * (q_hi - q_lo)
    return mu + sigma * student_t.ppf(q, df)


def _logit(p):
    p = float(np.clip(p, 1e-8, 1.0 - 1e-8))
    return math.log(p / (1.0 - p))


def _ilogit(z):
    return 1.0 / (1.0 + math.exp(-z))


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')

    a = stim[0]
    b = stim[1]
    n_features = int(stim.shape[1])

    val = np.asarray(parameters['validities'], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    # ---------- Heavy-tailed subject-level parameters -------------------
    def u_raw(x):
        return float(np.clip((float(x) - 0.02) / 0.96, 0.0, 1.0))

    df_ht = 1.8

    gain = float(np.exp(_tqt(
        u_raw(parameters['gain_raw']),
        math.log(1.35), 0.55, math.log(0.45), math.log(3.60), df_ht)))

    epsilon = float(_ilogit(_tqt(
        u_raw(parameters['epsilon_raw']),
        _logit(0.20), 0.65, _logit(0.03), _logit(0.55), df_ht)))

    lam = float(np.exp(_tqt(
        u_raw(parameters['lambda_raw']),
        math.log(0.24), 0.75, math.log(0.005), math.log(0.50), df_ht)))

    lam_large_scale = float(_ilogit(_tqt(
        u_raw(parameters['lam_large_scale_raw']),
        _logit(0.85), 0.45, _logit(0.10), _logit(0.98), df_ht)))

    p_prec = float(_ilogit(_tqt(
        u_raw(parameters['prec_p_raw']),
        _logit(0.75), 0.90, _logit(0.06), _logit(0.98), df_ht)))

    tie_p = float(_ilogit(_tqt(
        u_raw(parameters['tie_p_raw']),
        _logit(0.60), 0.75, _logit(0.15), _logit(0.99), df_ht)))

    prec_str = float(np.exp(_tqt(
        u_raw(parameters['prec_strength_raw']),
        math.log(0.95), 0.90, math.log(0.15), math.log(2.80), df_ht)))

    prec_gap_amp = float(np.exp(_tqt(
        u_raw(parameters['prec_gap_amp_raw']),
        math.log(0.90), 0.50, math.log(0.10), math.log(2.20), df_ht)))

    prec_opp_decay = float(np.exp(_tqt(
        u_raw(parameters['prec_opp_decay_raw']),
        math.log(0.50), 0.50, math.log(0.03), math.log(1.20), df_ht)))

    prec_d_decay = float(np.exp(_tqt(
        u_raw(parameters['prec_d_decay_raw']),
        math.log(0.10), 0.60, math.log(0.02), math.log(0.80), df_ht)))

    prec_conf_att = float(np.exp(_tqt(
        u_raw(parameters['prec_conf_att_raw']),
        math.log(0.06), 0.55, math.log(0.01), math.log(0.60), df_ht)))

    align_scale = float(_ilogit(_tqt(
        u_raw(parameters['align_scale_raw']),
        _logit(0.40), 0.60, _logit(0.05), _logit(0.90), df_ht)))

    align_d_att = float(np.exp(_tqt(
        u_raw(parameters['align_d_att_raw']),
        math.log(0.60), 0.60, math.log(0.03), math.log(1.60), df_ht)))

    tie_str = float(np.exp(_tqt(
        u_raw(parameters['tie_strength_raw']),
        math.log(1.10), 1.00, math.log(0.10), math.log(7.00), df_ht)))

    tie_coal_scale = float(np.exp(_tqt(
        u_raw(parameters['tie_coal_scale_raw']),
        math.log(2.80), 0.30, math.log(1.20), math.log(5.00), df_ht)))

    tie_size_amp = float(np.exp(_tqt(
        u_raw(parameters['tie_size_amp_raw']),
        math.log(0.50), 0.45, math.log(0.05), math.log(1.50), df_ht)))

    tie_large_decay = float(np.exp(_tqt(
        u_raw(parameters['tie_large_decay_raw']),
        math.log(3.00), 0.45, math.log(0.80), math.log(9.00), df_ht)))

    tie_gap_amp = float(np.exp(_tqt(
        u_raw(parameters['tie_gap_amp_raw']),
        math.log(0.70), 0.55, math.log(0.05), math.log(2.00), df_ht)))

    tie_opp_decay = float(np.exp(_tqt(
        u_raw(parameters['tie_opp_decay_raw']),
        math.log(0.60), 0.60, math.log(0.03), math.log(1.50), df_ht)))

    tie_opp_size = float(np.exp(_tqt(
        u_raw(parameters['tie_opp_size_raw']),
        math.log(0.80), 0.50, math.log(0.10), math.log(1.80), df_ht)))

    tie_d_decay = float(np.exp(_tqt(
        u_raw(parameters['tie_d_decay_raw']),
        math.log(0.12), 0.60, math.log(0.02), math.log(0.80), df_ht)))

    # ---------- Inverse-valence + damped validity weighting ------------
    changed = a != b
    disc_idx = np.flatnonzero(changed)
    if disc_idx.size == 0:
        return np.array([0.5, 0.5])

    diff = (b > a).astype(float) - (a > b).astype(float)
    D = float(np.sum(diff))
    disc_size = int(disc_idx.size)

    is_large_weight = disc_size >= 6 and n_features >= 7
    lam_eff = lam * (lam_large_scale if is_large_weight else 1.0)

    mean_v = float(np.mean(val))
    vals_disc = val[disc_idx]
    w_disc = 1.0 + lam_eff * (vals_disc - mean_v)
    E = float(np.sum(w_disc * diff[disc_idx]))

    # ---------- Top-group / unique-top structure -----------------------
    top_val = float(np.max(vals_disc))
    top_mask = np.isclose(vals_disc, top_val, rtol=0.0, atol=1e-9)
    top_idx = disc_idx[top_mask]

    unique_top = False
    top_tied = False
    sign_top = 0.0
    top_clean = 0.0
    gap = 0.04

    if top_idx.size == 1:
        unique_top = True
        top_pos = int(np.flatnonzero(top_mask)[0])
        sign_top = 1.0 if diff[disc_idx[top_pos]] > 0.0 else -1.0
        others = vals_disc[~top_mask]
        if others.size > 0:
            gap = float(top_val - float(np.max(others)))
            if gap < 1e-9:
                gap = 0.04
        else:
            gap = 0.25
    else:
        top_tied = True
        top_clean = float(np.sum(diff[top_idx]))
        others = vals_disc[~top_mask]
        if others.size > 0:
            gap = float(top_val - float(np.max(others)))
            if gap < 1e-9:
                gap = 0.04

    lower_positions = disc_idx[~top_mask]

    # ---------- Unique-top precedence ----------------------------------
    if unique_top:
        if abs(D) <= 1e-12:
            if np.random.rand() < tie_p:
                gap_factor = 1.0 + prec_gap_amp * math.tanh(gap / 0.08)
                E = sign_top * prec_str * gap_factor
        else:
            p_apply = p_prec / (1.0 + prec_conf_att * max(0.0, float(disc_size - 2)))
            if np.random.rand() < p_apply:
                gap_factor = 1.0 + prec_gap_amp * math.tanh(gap / 0.08)
                opp = 0.0
                if lower_positions.size > 0:
                    opp = float(np.sum(diff[lower_positions] * sign_top < 0.0))
                opp_factor = 1.0 / (1.0 + prec_opp_decay * (opp ** 0.7))
                d_factor = math.exp(-prec_d_decay * min(abs(D), 8.0))

                bonus = prec_str * gap_factor * opp_factor * d_factor

                # Smooth attenuation when the unique top cue merely repeats
                # the direction of the raw majority.
                if sign_top * D > 0.0 and disc_size >= 3:
                    align_factor = align_scale / (1.0 + align_d_att * abs(D))
                    bonus *= align_factor

                E = E + sign_top * bonus

    # ---------- Smooth continuous tied-top coalition bonus -------------
    elif top_tied and abs(top_clean) > 1e-9:
        sign_top = 1.0 if top_clean > 0.0 else -1.0
        m = int(top_idx.size)
        c = abs(top_clean)

        opp = 0.0
        if lower_positions.size > 0:
            opp = float(np.sum(diff[lower_positions] * sign_top < 0.0))

        # Linear in m for small coalitions, with a smooth penalty for large
        # coalitions that would otherwise overwhelm lower-cue majorities.
        coalition = float(m) * math.tanh(c / tie_coal_scale)
        size_factor = 1.0 + tie_size_amp * math.tanh((float(m) - 2.0) / 2.0)
        large_att = 1.0 / (1.0 + tie_large_decay * max(0.0, float(m - 3.0)))

        gap_factor = 1.0 + tie_gap_amp * math.tanh(gap / 0.08)
        opp_mult = 1.0 + tie_opp_size * max(0.0, float(m - 3.0))
        opp_factor = 1.0 / (1.0 + tie_opp_decay * opp_mult * (opp ** 0.7))
        d_factor = math.exp(-tie_d_decay * min(abs(D), 8.0))

        bonus = tie_str * coalition * size_factor * large_att * gap_factor * opp_factor * d_factor
        bonus = min(bonus, 8.0)
        E = E + sign_top * bonus

    # ---------- Softmax choice -----------------------------------------
    logits = gain * np.array([E, 0.0], dtype=float)
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core = p_core / p_core.sum()

    out = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    out = np.asarray(out, dtype=float)
    if not np.all(np.isfinite(out)) or out.sum() <= 0.0:
        return np.array([0.5, 0.5])
    out = out / out.sum()
    return out


`policy(probs) -> int`:
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))


`parameters`:
- gain_raw: [0.02, 0.98]
- epsilon_raw: [0.02, 0.98]
- lambda_raw: [0.02, 0.98]
- lam_large_scale_raw: [0.02, 0.98]
- prec_p_raw: [0.02, 0.98]
- tie_p_raw: [0.02, 0.98]
- prec_strength_raw: [0.02, 0.98]
- prec_gap_amp_raw: [0.02, 0.98]
- prec_opp_decay_raw: [0.02, 0.98]
- prec_d_decay_raw: [0.02, 0.98]
- prec_conf_att_raw: [0.02, 0.98]
- align_scale_raw: [0.02, 0.98]
- align_d_att_raw: [0.02, 0.98]
- tie_strength_raw: [0.02, 0.98]
- tie_coal_scale_raw: [0.02, 0.98]
- tie_size_amp_raw: [0.02, 0.98]
- tie_large_decay_raw: [0.02, 0.98]
- tie_gap_amp_raw: [0.02, 0.98]
- tie_opp_decay_raw: [0.02, 0.98]
- tie_opp_size_raw: [0.02, 0.98]
- tie_d_decay_raw: [0.02, 0.98]
- validities: validities

`rationale`:
This edit keeps the accepted inverse-valence plus damped validity-weighting frame and applies the critic's in-family fixes directly. The tied-top coalition is re-weighted from sqrt(m) to m*tanh(c/scale), which strongly raises m=2-3 coalition following (E5, E8, E18, E20) while a new smooth large-coalition penalty and top-size-scaled opposition damping stop m=4-5 groups from overwhelming lower-cue majorities (E13, E14, E16, E12). Large-conflict validity damping is relaxed by moving lam_large_scale's median from 0.45 to 0.85, directly targeting the E6 sign error. Unique-top precedence is made more available in large conflicts by lowering conflict and net-tally damping, but a new aligned-with-majority attenuation term lowers overuse when the top cue merely agrees with the raw tally (E12). Finally, subject-level transforms use heavier-tailed Student-t/logit-Student-t draws with wider sigma and lower df for the top-coalition, precedence, and lambda parameters, restoring the larger between-subject variance observed in E5, E6, E8, E19, and E20.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.1332 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.0699 -> ACCEPTED
Running-best (last accepted) base: iter 2 at loss=0.0699.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return 0.0

    A = np.array(data['option_a_ratings'].tolist(), dtype=int)
    B = np.array(data['option_b_ratings'].tolist(), dtype=int)
    y = 1.0 - data['response'].to_numpy(dtype=float)

    top_a = A[:, 0] > B[:, 0]
    top_b = A[:, 0] < B[:, 0]

    lower_a_wins = (A[:, 1:] > B[:, 1:]).sum(axis=1)
    lower_b_wins = (B[:, 1:] > A[:, 1:]).sum(axis=1)
    lower_net = lower_a_wins - lower_b_wins

    high = lower_net >= 1
    low = lower_net <= -1

    def diff(top_mask):
        hi = y[top_mask & high]
        lo = y[top_mask & low]
        if len(hi) == 0 or len(lo) == 0:
            return 0.0
        return float(hi.mean() - lo.mean())

    return (diff(top_a) + diff(top_b)) / 2.0
```

**Observed (real) value:** -0.5117 (var=0.0161)
**Candidate trajectory (this loop):**
  - iter 1: -0.5367 (var=0.0178) (Δ vs real -0.0250)
  - iter 2 (current): -0.5429 (var=0.0111) (Δ vs real -0.0313)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0092 (var=0.0043)
- pi_2: 0.6258 (var=0.0234)
- pi_3: -0.5379 (var=0.0084)
- pi_4: -0.5413 (var=0.0057)
- pi_5: -0.5308 (var=0.0142)
- pi_4_1: -0.5442 (var=0.0102)
- pi_6: -0.3700 (var=0.2536)
- pi_7: -0.4754 (var=0.0242)
- pi_8: -0.4688 (var=0.0076)
- pi_9: -0.4625 (var=0.0083)
- pi_10: -0.5287 (var=0.0134)

### Experiment 2
**Design**
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_rows = data['option_a_ratings'].tolist()
    b_rows = data['option_b_ratings'].tolist()
    tally = []
    for ra, rb in zip(a_rows, b_rows):
        a_wins = sum(1 for x, y in zip(ra, rb) if x > y)
        b_wins = sum(1 for x, y in zip(ra, rb) if y > x)
        tally.append(a_wins - b_wins)
    tally = np.asarray(tally, dtype=float)
    choose_a = (data['response'].to_numpy(dtype=float) == 0).astype(float)
    pos_rate = choose_a[tally > 0].mean()
    neg_rate = choose_a[tally < 0].mean()
    return float(pos_rate - neg_rate)
```

**Observed (real) value:** -0.6438 (var=0.0430)
**Candidate trajectory (this loop):**
  - iter 1: -0.6731 (var=0.0315) (Δ vs real -0.0294)
  - iter 2 (current): -0.5919 (var=0.0291) (Δ vs real +0.0519)
**Other theories' values on this metric (for reference):**
- pi_2: 0.7112 (var=0.0379)
- pi_1: -0.0013 (var=0.0059)
- pi_3: -0.6069 (var=0.0076)
- pi_4: -0.5913 (var=0.0097)
- pi_5: -0.5938 (var=0.0426)
- pi_4_1: -0.6038 (var=0.0238)
- pi_6: -0.6238 (var=0.2467)
- pi_7: -0.6138 (var=0.0287)
- pi_8: -0.5431 (var=0.0192)
- pi_9: -0.5531 (var=0.0300)
- pi_10: -0.5869 (var=0.0358)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.9, 0.84, 0.78, 0.72, 0.66, 0.6, 0.55])
    cue_order = np.argsort(-validities, kind='stable')

    a = np.array([list(r) for r in data['option_a_ratings']], dtype=float)
    b = np.array([list(r) for r in data['option_b_ratings']], dtype=float)
    resp = np.asarray(data['response'], dtype=int)

    n = a.shape[0]
    if n == 0:
        return 0.0

    predicted = np.full(n, -1, dtype=int)
    for cue in cue_order:
        undecided = predicted == -1
        disc = np.logical_and(undecided, a[:, cue] != b[:, cue])
        if not np.any(disc):
            continue
        predicted[disc] = np.where(a[disc, cue] > b[disc, cue], 0, 1)

    decided = predicted != -1
    if not np.any(decided):
        return 0.0

    match = (resp[decided] == predicted[decided]).astype(float)
    return float(np.mean(2.0 * match - 1.0))
```

**Observed (real) value:** -0.4300 (var=0.0221)
**Candidate trajectory (this loop):**
  - iter 1: -0.3133 (var=0.0360) (Δ vs real +0.1167)
  - iter 2 (current): -0.3254 (var=0.0497) (Δ vs real +0.1046)
**Other theories' values on this metric (for reference):**
- pi_1: 0.7113 (var=0.0398)
- pi_3: 0.0175 (var=0.0062)
- pi_2: 0.0117 (var=0.0115)
- pi_4: -0.4575 (var=0.0063)
- pi_5: -0.4596 (var=0.0245)
- pi_4_1: -0.4029 (var=0.0226)
- pi_6: -0.3550 (var=0.1762)
- pi_7: -0.6425 (var=0.0137)
- pi_8: -0.3579 (var=0.0112)
- pi_9: -0.4450 (var=0.0283)
- pi_10: -0.4000 (var=0.0432)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a = np.array(data['option_a_ratings'].tolist(), dtype=int)
    b = np.array(data['option_b_ratings'].tolist(), dtype=int)
    # inverse-tally advantage for A: #(A=0,B=1) - #(A=1,B=0)
    s = (b > a).sum(axis=1) - (a > b).sum(axis=1)
    mask_pos = s == 2
    mask_neg = s == -2
    pos = data.loc[mask_pos, 'response']
    neg = data.loc[mask_neg, 'response']
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    p_pos = (pos == 0).mean()
    p_neg = (neg == 0).mean()
    return float(p_pos - p_neg)
```

**Observed (real) value:** 0.4925 (var=0.0658)
**Candidate trajectory (this loop):**
  - iter 1: 0.6588 (var=0.0260) (Δ vs real +0.1663)
  - iter 2 (current): 0.6269 (var=0.0349) (Δ vs real +0.1344)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6381 (var=0.0147)
- pi_1: -0.0244 (var=0.0050)
- pi_2: -0.7525 (var=0.0447)
- pi_4: 0.4313 (var=0.0096)
- pi_5: 0.4237 (var=0.0670)
- pi_4_1: 0.5069 (var=0.0237)
- pi_6: 0.4681 (var=0.1627)
- pi_7: 0.5363 (var=0.0483)
- pi_8: 0.5619 (var=0.0213)
- pi_9: 0.5325 (var=0.0266)
- pi_10: 0.6331 (var=0.0232)

### Experiment 5
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

PAIR_SIGNS = {
    ((0, 0, 0, 0, 0, 1), (1, 0, 0, 0, 0, 0)): 1.0,
    ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)): -1.0,
    ((0, 0, 0, 0, 0, 1), (0, 1, 0, 0, 0, 0)): 1.0,
    ((0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)): -1.0,
    ((0, 0, 0, 0, 0, 1), (0, 0, 1, 0, 0, 0)): 1.0,
    ((0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 1)): -1.0,
    ((1, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 1)): -1.0,
    ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 1)): -1.0,
    ((1, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1)): -1.0,
    ((0, 0, 0, 1, 1, 1), (1, 1, 0, 0, 0, 0)): 1.0,
    ((0, 0, 0, 0, 1, 1), (1, 0, 0, 0, 0, 0)): 1.0,
    ((0, 0, 0, 1, 1, 1), (1, 0, 1, 0, 0, 0)): 1.0,
}

def metric(data: pd.DataFrame) -> float:
    contributions = []
    for _, row in data.iterrows():
        key = (tuple(row['option_a_ratings']), tuple(row['option_b_ratings']))
        sign = PAIR_SIGNS.get(key)
        if sign is None:
            continue
        y = 1.0 if row['response'] == 0 else -1.0
        contributions.append(y * sign)
    return float(np.mean(contributions)) if contributions else 0.0
```

**Observed (real) value:** 0.2658 (var=0.1584)
**Candidate trajectory (this loop):**
  - iter 1: 0.0379 (var=0.0219) (Δ vs real -0.2279)
  - iter 2 (current): 0.1321 (var=0.0256) (Δ vs real -0.1337)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2971 (var=0.0096)
- pi_3: -0.2708 (var=0.0095)
- pi_1: -0.6967 (var=0.0260)
- pi_2: 0.3704 (var=0.0144)
- pi_5: 0.3858 (var=0.0225)
- pi_4_1: 0.2446 (var=0.0156)
- pi_6: 0.3300 (var=0.1102)
- pi_7: 0.2608 (var=0.0101)
- pi_8: 0.2696 (var=0.0180)
- pi_9: 0.3075 (var=0.0162)
- pi_10: 0.0975 (var=0.0284)

### Experiment 6
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return 0.0

    validities = np.array([0.95, 0.9, 0.85, 0.8, 0.7, 0.65, 0.6, 0.55], dtype=float)
    weights = 1.0 + 2.0 * (validities - validities.mean())

    pos = []
    neg = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        clean_a = (b > a).astype(float)
        clean_b = (a > b).astype(float)
        diff = clean_a - clean_b
        d = float(np.sum(diff))
        s = float(np.dot(weights, diff))
        y = 1.0 - float(row['response'])
        if d < 0.0 and s > 0.0:
            pos.append(y)
        elif d > 0.0 and s < 0.0:
            neg.append(y)

    if not pos or not neg:
        return 0.0

    return float(np.mean(pos) - np.mean(neg))
```

**Observed (real) value:** 0.0867 (var=0.4244)
**Candidate trajectory (this loop):**
  - iter 1: -0.3650 (var=0.0351) (Δ vs real -0.4517)
  - iter 2 (current): -0.2483 (var=0.0584) (Δ vs real -0.3350)
**Other theories' values on this metric (for reference):**
- pi_3: -0.5333 (var=0.0223)
- pi_4: 0.2158 (var=0.0210)
- pi_1: -0.7375 (var=0.0404)
- pi_2: 0.6717 (var=0.0518)
- pi_5: 0.1383 (var=0.0736)
- pi_4_1: 0.1075 (var=0.0264)
- pi_6: 0.0358 (var=0.0678)
- pi_7: -0.2058 (var=0.0270)
- pi_8: 0.0692 (var=0.0205)
- pi_9: 0.0483 (var=0.0368)
- pi_10: 0.1267 (var=0.1923)

### Experiment 7
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data):
    import numpy as np
    a0 = data['option_a_ratings'].apply(lambda x: x[0]).to_numpy(dtype=float)
    b0 = data['option_b_ratings'].apply(lambda x: x[0]).to_numpy(dtype=float)
    ld = data.apply(lambda row: sum([1 for x, y in zip(row['option_a_ratings'][1:], row['option_b_ratings'][1:]) if x != y]), axis=1).to_numpy(dtype=float)
    choose_a = 1.0 - data['response'].to_numpy(dtype=float)
    level = np.where(ld == 1, 1, np.where(ld == 3, 2, np.where(ld == 5, 3, 0)))
    topA = ((a0 == 0) & (b0 == 1))
    topB = ((a0 == 1) & (b0 == 0))
    m1 = choose_a[topA & (level == 1)].mean() if (topA & (level == 1)).any() else 0.0
    m2 = choose_a[topB & (level == 3)].mean() if (topB & (level == 3)).any() else 0.0
    m3 = choose_a[topA & (level == 3)].mean() if (topA & (level == 3)).any() else 0.0
    m4 = choose_a[topB & (level == 1)].mean() if (topB & (level == 1)).any() else 0.0
    return float(m1 + m2 - m3 - m4)
```

**Observed (real) value:** 0.0725 (var=0.0580)
**Candidate trajectory (this loop):**
  - iter 1: 0.0400 (var=0.0562) (Δ vs real -0.0325)
  - iter 2 (current): 0.0863 (var=0.0284) (Δ vs real +0.0138)
**Other theories' values on this metric (for reference):**
- pi_4: -0.0025 (var=0.0525)
- pi_5: 0.3325 (var=0.0612)
- pi_1: 0.0288 (var=0.0221)
- pi_2: 0.0137 (var=0.0602)
- pi_3: -0.0025 (var=0.0792)
- pi_4_1: -0.0400 (var=0.0672)
- pi_6: 0.1800 (var=0.1187)
- pi_7: 0.0812 (var=0.0474)
- pi_8: -0.0275 (var=0.0571)
- pi_9: -0.0062 (var=0.0541)
- pi_10: 0.0213 (var=0.0926)

### Experiment 8
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data):
    counts = {}
    totals = {}

    for row in data.itertuples(index=False):
        a = [int(x) for x in row.option_a_ratings]
        b = [int(x) for x in row.option_b_ratings]

        # h = 0 means the three high-validity cues favour B under inverse valence;
        # h = 1 is the mirror where they favour A.
        h = 0 if sum(a[:3]) > sum(b[:3]) else 1

        # Number of opposing low-validity cues.
        n = sum(1 for i in range(3, len(a)) if a[i] != b[i])

        key = (h, n)
        totals[key] = totals.get(key, 0) + 1
        if int(row.response) == 0:
            counts[key] = counts.get(key, 0) + 1

    props = {}
    for h in (0, 1):
        for n in (3, 4, 5):
            tot = totals.get((h, n), 0)
            props[(h, n)] = (counts.get((h, n), 0) / tot) if tot > 0 else 0.5

    d3 = props[(0, 3)] - props[(1, 3)]
    d4 = props[(0, 4)] - props[(1, 4)]
    d5 = props[(0, 5)] - props[(1, 5)]

    return float(-d3 + d4 + 2.0 * d5)
```

**Observed (real) value:** 1.0100 (var=2.8518)
**Candidate trajectory (this loop):**
  - iter 1: 1.6362 (var=1.0526) (Δ vs real +0.6263)
  - iter 2 (current): 0.5838 (var=1.6699) (Δ vs real -0.4262)
**Other theories' values on this metric (for reference):**
- pi_5: 1.3512 (var=0.8533)
- pi_4: -0.1263 (var=0.1938)
- pi_1: 1.4112 (var=0.1813)
- pi_2: -2.0162 (var=0.4200)
- pi_3: 1.7625 (var=0.1534)
- pi_4_1: 0.5012 (var=0.5143)
- pi_6: 1.1075 (var=2.7433)
- pi_7: 1.6462 (var=0.3404)
- pi_8: 0.8500 (var=0.3186)
- pi_9: 0.9838 (var=0.4229)
- pi_10: 1.3738 (var=0.2713)

### Experiment 9
**Design**
  A=[1, 0, 0, 1, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0, 0, 1]

**Metric**
```python
import numpy as np
import pandas as pd

_VALIDITIES = np.array([0.95, 0.90, 0.65, 0.64, 0.63, 0.62, 0.60, 0.60])
_FIRST_CUE_ORDER = np.argsort(-_VALIDITIES, kind='stable')


def metric(data: pd.DataFrame) -> float:
    if len(data) == 0:
        return 0.0

    a = np.asarray(data['option_a_ratings'].tolist(), dtype=np.int64)
    b = np.asarray(data['option_b_ratings'].tolist(), dtype=np.int64)
    r = np.asarray(data['response'].tolist(), dtype=np.int64)

    n = len(data)
    first_col = np.full(n, -1, dtype=np.int64)
    assigned = np.zeros(n, dtype=bool)

    for j in _FIRST_CUE_ORDER:
        differs = a[:, j] != b[:, j]
        newly = differs & ~assigned
        first_col[newly] = j
        assigned |= differs

    valid = first_col >= 0
    if not valid.any():
        return 0.0

    idx_valid = np.nonzero(valid)[0]
    cols_valid = first_col[valid]
    a_first = a[idx_valid, cols_valid]
    b_first = b[idx_valid, cols_valid]
    favors_a = (a_first == 0) & (b_first == 1)

    high_mask = np.zeros(n, dtype=bool)
    low_mask = np.zeros(n, dtype=bool)
    high_mask[idx_valid[favors_a]] = True
    low_mask[idx_valid[~favors_a]] = True

    if high_mask.sum() == 0 or low_mask.sum() == 0:
        return 0.0

    chose_a = r == 0
    return float(chose_a[high_mask].mean() - chose_a[low_mask].mean())

```

**Observed (real) value:** 0.3050 (var=0.0605)
**Candidate trajectory (this loop):**
  - iter 1: 0.2812 (var=0.0423) (Δ vs real -0.0238)
  - iter 2 (current): 0.3133 (var=0.0408) (Δ vs real +0.0083)
**Other theories' values on this metric (for reference):**
- pi_4_1: 0.4188 (var=0.0158)
- pi_5: 0.2196 (var=0.0342)
- pi_1: -0.7067 (var=0.0351)
- pi_2: 0.0204 (var=0.0104)
- pi_3: -0.0142 (var=0.0125)
- pi_4: 0.4296 (var=0.0093)
- pi_6: 0.1742 (var=0.1020)
- pi_7: 0.4579 (var=0.0150)
- pi_8: 0.2754 (var=0.0135)
- pi_9: 0.3433 (var=0.0229)
- pi_10: 0.2175 (var=0.0401)

### Experiment 10
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    scores = []
    for sid, sub in data.groupby('subject_id'):
        low = []
        for a, b, resp in zip(sub['option_a_ratings'], sub['option_b_ratings'], sub['response']):
            s = 0
            for x, y in zip(a, b):
                if x == 0 and y == 1:
                    s += 1
                elif x == 1 and y == 0:
                    s -= 1
            if s > 0:
                low.append(1.0 if resp == 0 else 0.0)
            elif s < 0:
                low.append(1.0 if resp == 1 else 0.0)
        if not low:
            scores.append(0.0)
        else:
            q = float(np.mean(low))
            scores.append(q * q - q)
    return float(np.mean(scores))
```

**Observed (real) value:** -0.1667 (var=0.0053)
**Candidate trajectory (this loop):**
  - iter 1: -0.1423 (var=0.0027) (Δ vs real +0.0244)
  - iter 2 (current): -0.1657 (var=0.0026) (Δ vs real +0.0009)
**Other theories' values on this metric (for reference):**
- pi_5: -0.2057 (var=0.0022)
- pi_4_1: -0.2374 (var=0.0002)
- pi_1: -0.1112 (var=0.0033)
- pi_2: -0.1264 (var=0.0029)
- pi_3: -0.1529 (var=0.0007)
- pi_4: -0.2445 (var=0.0000)
- pi_6: -0.1839 (var=0.0030)
- pi_7: -0.1824 (var=0.0024)
- pi_8: -0.1707 (var=0.0019)
- pi_9: -0.1827 (var=0.0021)
- pi_10: -0.1847 (var=0.0032)

### Experiment 11
**Design**
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a_list = data['option_a_ratings'].tolist()
    b_list = data['option_b_ratings'].tolist()
    k = np.array([sum(1 for x, y in zip(a, b) if x == 0 and y == 1) for a, b in zip(a_list, b_list)], dtype=float)
    resp = np.asarray(data['response'], dtype=float)
    subject_ids = data['subject_id'].to_numpy()
    informative = k != 4
    inv_pred = np.where(k < 4, 1.0, 0.0)
    inv_match = resp == inv_pred

    def subject_consistency(mask):
        valid = informative & mask
        if not valid.any():
            return 0.5
        return float(np.mean(inv_match[valid]))

    unique_ids = np.unique(subject_ids)
    if len(unique_ids) > 1:
        positive_count = []
        for sid in unique_ids:
            s = subject_consistency(subject_ids == sid)
            positive_count.append(1.0 if s < 0.5 else 0.0)
        return float(np.mean(positive_count))
    s = subject_consistency(np.ones(len(data), dtype=bool))
    return float(max(s, 1.0 - s))

```

**Observed (real) value:** 0.0000 (var=0.0045)
**Candidate trajectory (this loop):**
  - iter 1: 0.0000 (var=0.0047) (Δ vs real +0.0000)
  - iter 2 (current): 0.0000 (var=0.0045) (Δ vs real +0.0000)
**Other theories' values on this metric (for reference):**
- pi_6: 0.0400 (var=0.0007)
- pi_5: 0.0000 (var=0.0037)
- pi_1: 0.9200 (var=0.0019)
- pi_2: 1.0000 (var=0.0083)
- pi_3: 0.0000 (var=0.0019)
- pi_4: 0.0000 (var=0.0011)
- pi_4_1: 0.0000 (var=0.0015)
- pi_7: 0.0000 (var=0.0038)
- pi_8: 0.0000 (var=0.0019)
- pi_9: 0.0000 (var=0.0028)
- pi_10: 0.0000 (var=0.0065)

### Experiment 12
**Design**
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def score_one(g):
        a = np.array([list(x) for x in g['option_a_ratings']], dtype=int)
        b = np.array([list(x) for x in g['option_b_ratings']], dtype=int)
        resp = g['response'].to_numpy().astype(int)

        d = ((a == 0) & (b == 1)).sum(axis=1) - ((a == 1) & (b == 0)).sum(axis=1)
        choice_a = (resp == 0).astype(float)

        far = np.abs(d) == 3
        far_pos = far & (d == 3)
        far_neg = far & (d == -3)
        if np.any(far_pos) and np.any(far_neg):
            orient_diff = float(np.mean(choice_a[far_pos]) - np.mean(choice_a[far_neg]))
            orient = 1.0 if orient_diff >= 0.0 else -1.0
        else:
            corr = float(np.corrcoef(d.astype(float), choice_a)[0, 1])
            orient = 1.0 if np.isfinite(corr) and corr >= 0.0 else -1.0

        aligned_d = orient * d
        majority_opt = np.where(aligned_d > 0, 0, 1)

        raw_top_dir = np.where((a[:, 0] == 0) & (b[:, 0] == 1), 1.0, -1.0)
        top_opt = np.where(orient * raw_top_dir > 0.0, 0, 1)
        top_align = (top_opt == majority_opt)
        followed_top = (resp == top_opt).astype(float)

        near = np.abs(d) == 1
        far2 = np.abs(d) == 3

        def arcs(mask):
            vals = followed_top[mask]
            n = vals.shape[0]
            if n == 0:
                return 0.0
            p = (float(np.sum(vals)) + 0.5) / (float(n) + 1.0)
            p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
            return float(np.arcsin(np.sqrt(p)))

        near_aligned = arcs(near & top_align)
        near_opposed = arcs(near & ~top_align)
        far_aligned = arcs(far2 & top_align)
        far_opposed = arcs(far2 & ~top_align)

        return (near_aligned - near_opposed) + (far_aligned - far_opposed)

    if data.empty:
        return 0.0

    scores = []
    for sid in data['subject_id'].unique():
        sub = data[data['subject_id'] == sid]
        scores.append(score_one(sub))
    return float(np.mean(scores))
```

**Observed (real) value:** 1.2973 (var=0.2536)
**Candidate trajectory (this loop):**
  - iter 1: 1.5635 (var=0.1851) (Δ vs real +0.2662)
  - iter 2 (current): 1.4297 (var=0.1343) (Δ vs real +0.1324)
**Other theories' values on this metric (for reference):**
- pi_5: 1.5845 (var=0.1245)
- pi_6: 1.9342 (var=0.0366)
- pi_1: 0.1091 (var=0.0296)
- pi_2: 1.6796 (var=0.3099)
- pi_3: 1.2888 (var=0.0583)
- pi_4: 1.3818 (var=0.0296)
- pi_4_1: 1.3853 (var=0.0963)
- pi_7: 1.1792 (var=0.1659)
- pi_8: 1.2583 (var=0.0676)
- pi_9: 1.3103 (var=0.0593)
- pi_10: 1.4052 (var=0.1406)

### Experiment 13
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    a = np.array([list(row) for row in data['option_a_ratings']], dtype=float)
    b = np.array([list(row) for row in data['option_b_ratings']], dtype=float)

    clean_a = (a == 0) & (b == 1)
    clean_b = (a == 1) & (b == 0)

    D = clean_a.sum(axis=1) - clean_b.sum(axis=1)
    x = clean_a[:, :5].sum(axis=1) - clean_b[:, :5].sum(axis=1)

    chose_a = (data['response'].to_numpy(dtype=float) == 0)

    mask_pos = (D == 2) & (x == -3)
    mask_mirror = (D == -2) & (x == 3)

    if mask_pos.sum() == 0 or mask_mirror.sum() == 0:
        return 0.0

    p_pos = float(chose_a[mask_pos].mean())
    p_mirror = float(chose_a[mask_mirror].mean())
    return p_pos - p_mirror

```

**Observed (real) value:** 0.7300 (var=0.0558)
**Candidate trajectory (this loop):**
  - iter 1: 0.5175 (var=0.1063) (Δ vs real -0.2125)
  - iter 2 (current): 0.5200 (var=0.0752) (Δ vs real -0.2100)
**Other theories' values on this metric (for reference):**
- pi_7: 0.4625 (var=0.0677)
- pi_5: 0.1475 (var=0.1242)
- pi_1: -0.7650 (var=0.0460)
- pi_2: -0.7350 (var=0.0554)
- pi_3: 0.6325 (var=0.0265)
- pi_4: -0.5125 (var=0.0583)
- pi_4_1: -0.2175 (var=0.0968)
- pi_6: 0.1750 (var=0.1163)
- pi_8: 0.6225 (var=0.0341)
- pi_9: 0.5800 (var=0.0580)
- pi_10: 0.6250 (var=0.1075)

### Experiment 14
**Design**
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    crit_a1 = (1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
    crit_b1 = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)
    crit_a2 = (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1)
    crit_b2 = (1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)

    def pair_key(row):
        return (tuple(row['option_a_ratings']), tuple(row['option_b_ratings']))

    keys = data.apply(pair_key, axis=1)
    mask1 = (keys == (crit_a1, crit_b1)).to_numpy()
    mask2 = (keys == (crit_a2, crit_b2)).to_numpy()
    resp = data['response'].to_numpy()

    if mask1.sum() == 0 or mask2.sum() == 0:
        return 0.0

    p_a1 = float((resp[mask1] == 0).mean())
    p_a2 = float((resp[mask2] == 0).mean())
    return p_a1 - p_a2

```

**Observed (real) value:** 0.3950 (var=0.3296)
**Candidate trajectory (this loop):**
  - iter 1: 0.0787 (var=0.2142) (Δ vs real -0.3163)
  - iter 2 (current): 0.2800 (var=0.0913) (Δ vs real -0.1150)
**Other theories' values on this metric (for reference):**
- pi_5: 0.2938 (var=0.0899)
- pi_7: -0.2175 (var=0.1258)
- pi_1: 0.7537 (var=0.0387)
- pi_2: -0.7450 (var=0.0311)
- pi_3: 0.5400 (var=0.0262)
- pi_4: -0.0587 (var=0.0308)
- pi_4_1: 0.0538 (var=0.0355)
- pi_6: 0.1925 (var=0.1178)
- pi_8: 0.4813 (var=0.0208)
- pi_9: 0.3987 (var=0.0481)
- pi_10: 0.1500 (var=0.1973)

### Experiment 15
**Design**
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 1, 1, 1]  B=[1, 0, 0, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    a = np.array([list(v) for v in data['option_a_ratings']], dtype=float)
    b = np.array([list(v) for v in data['option_b_ratings']], dtype=float)
    y = 1.0 - data['response'].to_numpy(dtype=float)
    sid = data['subject_id'].to_numpy()

    clean_a_high = ((a[:, :3] == 0) & (b[:, :3] == 1)).sum(axis=1)
    clean_b_high = ((a[:, :3] == 1) & (b[:, :3] == 0)).sum(axis=1)
    h = clean_a_high - clean_b_high

    clean_a_all = ((a == 0) & (b == 1)).sum(axis=1)
    clean_b_all = ((a == 1) & (b == 0)).sum(axis=1)
    D = clean_a_all - clean_b_all

    slopes = []
    for subj in np.unique(sid):
        m = sid == subj
        subj_slopes = []
        for D_val in (2.0, -2.0):
            dm = m & (D == D_val)
            if np.sum(dm) < 2:
                continue
            hh = h[dm].astype(float)
            yy = y[dm]
            hh_c = hh - hh.mean()
            yy_c = yy - yy.mean()
            denom = float(np.sum(hh_c * hh_c))
            if denom > 1e-12:
                subj_slopes.append(float(np.sum(hh_c * yy_c) / denom))
        slopes.append(float(np.mean(subj_slopes)) if subj_slopes else 0.0)

    if not slopes:
        return 0.0
    return float(np.mean(slopes))
```

**Observed (real) value:** 0.0608 (var=0.0035)
**Candidate trajectory (this loop):**
  - iter 1: 0.0203 (var=0.0010) (Δ vs real -0.0405)
  - iter 2 (current): 0.0601 (var=0.0029) (Δ vs real -0.0007)
**Other theories' values on this metric (for reference):**
- pi_8: 0.0028 (var=0.0004)
- pi_5: 0.0792 (var=0.0012)
- pi_1: -0.1013 (var=0.0012)
- pi_2: 0.0023 (var=0.0002)
- pi_3: 0.0020 (var=0.0003)
- pi_4: 0.1189 (var=0.0003)
- pi_4_1: 0.0888 (var=0.0011)
- pi_6: 0.0698 (var=0.0043)
- pi_7: 0.0219 (var=0.0003)
- pi_9: 0.0088 (var=0.0006)
- pi_10: 0.0120 (var=0.0004)

### Experiment 16
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    def top_group_sign(row):
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        diff = (b[:4] > a[:4]).astype(float) - (a[:4] > b[:4]).astype(float)
        h = float(np.sum(diff))
        if h > 0:
            return 1.0
        if h < 0:
            return -1.0
        return 0.0

    df = data.copy()
    df['sign_h'] = df.apply(top_group_sign, axis=1)
    df['chose_a'] = 1.0 - df['response'].astype(float)

    pos = df.loc[df['sign_h'] > 0, 'chose_a']
    neg = df.loc[df['sign_h'] < 0, 'chose_a']
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    return float(pos.mean() - neg.mean())
```

**Observed (real) value:** 0.2492 (var=0.0989)
**Candidate trajectory (this loop):**
  - iter 1: 0.3133 (var=0.0627) (Δ vs real +0.0642)
  - iter 2 (current): 0.2463 (var=0.0596) (Δ vs real -0.0029)
**Other theories' values on this metric (for reference):**
- pi_5: 0.3954 (var=0.0294)
- pi_8: 0.0162 (var=0.0093)
- pi_1: -0.7196 (var=0.0295)
- pi_2: 0.0000 (var=0.0081)
- pi_3: -0.0071 (var=0.0111)
- pi_4: 0.4225 (var=0.0057)
- pi_4_1: 0.4133 (var=0.0171)
- pi_6: 0.4467 (var=0.1357)
- pi_7: 0.3862 (var=0.0094)
- pi_9: 0.2429 (var=0.0280)
- pi_10: 0.3746 (var=0.0795)

### Experiment 17
**Design**
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd


def metric(data: pd.DataFrame) -> float:
    a = np.array(data['option_a_ratings'].tolist(), dtype=int)
    b = np.array(data['option_b_ratings'].tolist(), dtype=int)
    if a.size == 0:
        return 0.0

    diff = (b > a).astype(int) - (a > b).astype(int)
    h = diff[:, :4].sum(axis=1).astype(float)
    d = diff.sum(axis=1).astype(int)

    resp = np.asarray(data['response'], dtype=float)
    y = 1.0 - resp

    h_c = h.copy()
    y_c = y.copy()
    for dval in np.unique(d):
        mask = d == dval
        h_c[mask] -= h_c[mask].mean()
        y_c[mask] -= y_c[mask].mean()

    denom = float(np.sum(h_c * h_c))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(h_c * y_c) / denom)

```

**Observed (real) value:** 0.0608 (var=0.0012)
**Candidate trajectory (this loop):**
  - iter 1: 0.0410 (var=0.0011) (Δ vs real -0.0198)
  - iter 2 (current): 0.0542 (var=0.0013) (Δ vs real -0.0066)
**Other theories' values on this metric (for reference):**
- pi_8: 0.0036 (var=0.0002)
- pi_9: 0.0373 (var=0.0004)
- pi_1: -0.0778 (var=0.0005)
- pi_2: 0.0014 (var=0.0002)
- pi_3: 0.0011 (var=0.0002)
- pi_4: 0.0906 (var=0.0001)
- pi_5: 0.0870 (var=0.0005)
- pi_4_1: 0.0777 (var=0.0005)
- pi_6: 0.0665 (var=0.0059)
- pi_7: 0.0530 (var=0.0004)
- pi_10: 0.0545 (var=0.0013)

### Experiment 18
**Design**
  A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def _trend_z_for_subject(sub):
    h_order = [-3.0, -1.0, 1.0, 3.0]
    score_vals = np.tanh(np.array(h_order) / 2.8)
    zs = []

    for d_target in (1.0, -1.0):
        h_list = []
        y_list = []
        for _, row in sub.iterrows():
            a = np.asarray(row['option_a_ratings'], dtype=float)
            b = np.asarray(row['option_b_ratings'], dtype=float)
            diff = (b > a).astype(float) - (a > b).astype(float)
            D = float(diff.sum())
            if abs(D - d_target) > 1e-9:
                continue
            h = float(diff[0] + diff[1] + diff[2])
            y = 1.0 if float(row['response']) == 0.0 else 0.0
            h_list.append(h)
            y_list.append(y)

        if not h_list:
            zs.append(0.0)
            continue

        h_arr = np.array(h_list, dtype=float)
        y_arr = np.array(y_list, dtype=float)

        r = []
        n = []
        for hv in h_order:
            mask = np.isclose(h_arr, hv)
            r.append(float(y_arr[mask].sum()))
            n.append(float(mask.sum()))

        r = np.array(r, dtype=float)
        n = np.array(n, dtype=float)
        if n.sum() == 0.0:
            zs.append(0.0)
            continue

        pbar = r.sum() / n.sum()
        if pbar <= 0.0 or pbar >= 1.0:
            zs.append(0.0)
            continue

        numerator = np.sum(score_vals * (r - n * pbar))
        denominator = np.sqrt(pbar * (1.0 - pbar) * np.sum(n * score_vals * score_vals))
        zs.append(float(numerator / denominator) if denominator > 1e-12 else 0.0)

    return float(np.mean(zs))

def metric(data: pd.DataFrame) -> float:
    if data is None or data.empty:
        return 0.0
    subject_vals = [_trend_z_for_subject(sub) for _, sub in data.groupby('subject_id')]
    return float(np.mean(subject_vals)) if subject_vals else 0.0
```

**Observed (real) value:** 3.4448 (var=3.3537)
**Candidate trajectory (this loop):**
  - iter 1: 1.5524 (var=1.9218) (Δ vs real -1.8925)
  - iter 2 (current): 3.5746 (var=2.4559) (Δ vs real +0.1298)
**Other theories' values on this metric (for reference):**
- pi_9: 1.0841 (var=0.9280)
- pi_8: -0.0410 (var=0.4858)
- pi_1: -4.5331 (var=1.5322)
- pi_2: -0.0042 (var=0.5248)
- pi_3: 0.1400 (var=0.4901)
- pi_4: 3.4585 (var=0.3754)
- pi_5: 3.3993 (var=1.0210)
- pi_4_1: 2.6923 (var=1.0039)
- pi_6: 2.1446 (var=13.2383)
- pi_7: 1.1573 (var=1.6319)
- pi_10: 3.9261 (var=1.6381)

### Experiment 19
**Design**
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1]

**Metric**
```python
import numpy as np
def metric(data):
    a = [row for row in data['option_a_ratings']]
    b = [row for row in data['option_b_ratings']]
    resp = data['response'].to_numpy()
    d_vals = []
    top_choices = []
    for ar, br, r in zip(a, b, resp):
        d = sum([1 if (y > x) else -1 for x, y in zip(ar, br) if x != y])
        d_vals.append(d)
        top_choices.append(((ar[0] == 0) and (br[0] == 1) and (r == 0)) or
                           ((ar[0] == 1) and (br[0] == 0) and (r == 1)))
    d_vals = np.array(d_vals)
    top_choices = np.array(top_choices, dtype=float)
    mask = np.abs(d_vals) == 3
    if mask.sum() == 0:
        return 0.5
    return float(np.mean(top_choices[mask]))
```

**Observed (real) value:** 0.3422 (var=0.0518)
**Candidate trajectory (this loop):**
  - iter 1: 0.1236 (var=0.0033) (Δ vs real -0.2186)
  - iter 2 (current): 0.1711 (var=0.0064) (Δ vs real -0.1711)
**Other theories' values on this metric (for reference):**
- pi_10: 0.3828 (var=0.0886)
- pi_9: 0.1447 (var=0.0036)
- pi_1: 0.1692 (var=0.0121)
- pi_2: 0.8639 (var=0.0094)
- pi_3: 0.1728 (var=0.0021)
- pi_4: 0.3950 (var=0.0032)
- pi_5: 0.3572 (var=0.0283)
- pi_4_1: 0.2986 (var=0.0101)
- pi_6: 0.3608 (var=0.0509)
- pi_7: 0.1628 (var=0.0101)
- pi_8: 0.1489 (var=0.0022)

### Experiment 20
**Design**
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data):
    import numpy as np
    import pandas as pd

    A = np.asarray(data['option_a_ratings'].tolist(), dtype=float)
    B = np.asarray(data['option_b_ratings'].tolist(), dtype=float)

    diff = (B > A).astype(float) - (A > B).astype(float)
    D = diff.sum(axis=1)
    top = diff[:, 0] + diff[:, 1]
    chose_a = (data['response'].to_numpy() == 0).astype(float)

    aux = pd.DataFrame({
        'sid': data['subject_id'].to_numpy(),
        'D': D,
        'top': top,
        'chose_a': chose_a,
    })

    def subj_metric(g):
        pp = g.loc[(g['D'] > 0) & (g['top'] > 0), 'chose_a'].mean()
        pn = g.loc[(g['D'] > 0) & (g['top'] < 0), 'chose_a'].mean()
        np_ = g.loc[(g['D'] < 0) & (g['top'] > 0), 'chose_a'].mean()
        nn = g.loc[(g['D'] < 0) & (g['top'] < 0), 'chose_a'].mean()
        return 0.5 * ((pp - pn) + (np_ - nn))

    scores = aux.groupby('sid', sort=False).apply(subj_metric)
    return float(scores.mean())

```

**Observed (real) value:** 0.0900 (var=0.0603)
**Candidate trajectory (this loop):**
  - iter 1: 0.0325 (var=0.0121) (Δ vs real -0.0575)
  - iter 2 (current): 0.1375 (var=0.0615) (Δ vs real +0.0475)
**Other theories' values on this metric (for reference):**
- pi_9: 0.0181 (var=0.0090)
- pi_10: 0.3738 (var=0.1102)
- pi_1: -0.6606 (var=0.0427)
- pi_2: 0.0037 (var=0.0075)
- pi_3: -0.0150 (var=0.0113)
- pi_4: 0.1319 (var=0.0140)
- pi_5: 0.0819 (var=0.0153)
- pi_4_1: 0.0825 (var=0.0070)
- pi_6: 0.0394 (var=0.0047)
- pi_7: 0.0269 (var=0.0088)
- pi_8: 0.0094 (var=0.0065)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The candidate preserves the inverse-valence baseline well (E1, E2, E7, E9, E10, E11 are close), but the top-coalition and precedence components are not experiment-invariant. The clearest diagnostic pattern is a size-dependent miscalibration of the tied-top coalition bonus: it is too weak for small top groups and too strong for large top groups. Small-top-group cases are underpowered: E5 is 0.038 vs 0.266, E18 is 1.55 vs 3.44, E20 is 0.033 vs 0.090, and E8 shows too much lower-majority dominance (1.64 vs 1.01), implying the m=2-3 coalition bonus is too small. Large-top-group cases are overpowered: E13 is 0.518 vs 0.730, E14 is 0.079 vs 0.395, E16 is 0.313 vs 0.249, and E12 is 1.56 vs 1.30. This points to the sqrt(m)*size_factor term over-amplifying m=4-5 while underpowering m=2-3. The large-conflict validity-weighting damping is also too aggressive: E6 flips sign (-0.365 vs 0.087), indicating lam_large_scale is too low and raw majority dominates too much. Unique-top precedence is miscalibrated across conflict sizes: too weak in E19 (0.124 vs 0.342) but too strong in E12 (1.56 vs 1.30), suggesting the conflict damping and the treatment of aligned-vs-opposed top cues need adjustment. Finally, the candidate is underdispersed relative to real between-subject variance in E5, E6, E8, E19, and E20, so the heavy-tailed subject-level draws are still not wide enough.
Rationale: Stay within the prescribed damped inverse-valence plus continuous top-coalition family, but recalibrate the following in-family knobs. (1) Fix the tied-top size scaling: the current sqrt(m)*size_factor makes large top groups too dominant and small top groups too weak. Flatten/saturate the size term earlier, reduce size_amp for m=4-5, and raise the base tie_str so m=2-3 coalitions are stronger; also consider making opposition damping scale with top-group size so large top groups facing many lower cues are more attenuated. This should move E13/E14/E16/E12 down and E5/E8/E18/E20 up. (2) Raise lam_large_scale toward roughly 0.8-1.0, or otherwise reduce the large-conflict lambda damping, to fix the E6 sign error; compensate with top-coalition damping rather than validity-weight damping. (3) Recalibrate unique-top precedence: soften the conflict-size and net-tally damping in large conflicts to raise E19, while adding smooth attenuation when the unique top cue agrees with D to lower E12. (4) Widen the subject-level t/logit-t sigmas for tie/precedence/lambda parameters to restore the observed between-subject variances, especially E5, E6, E8, E19, and E20.

**Outcome of your advice:** iter 1 candidate loss=0.1332 -> CURRENT candidate loss=0.0699 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The iter-2 candidate is a large improvement and is close on many experiments (E1, E2, E7, E9, E10, E11, E15, E16, E17, E18, E20 are essentially within noise or only modestly off), but it still misses several central, theory-relevant signatures. The most damaging is E6: the real effect is slightly positive (0.087) but the candidate remains sign-reversed (-0.248). This shows that in large conflicts the validity-weighted inverse-valence tally is still too weak relative to the raw majority, despite the relaxed lam_large_scale. The tied-top coalition is now better for small coalitions (E18 is close) but still too weak in E5 (0.132 vs 0.266) and E8 (0.584 vs 1.010), while the smooth large-coalition penalty is too aggressive, depressing E13 (0.520 vs 0.730) and E14 (0.280 vs 0.395). Unique-top precedence is miscalibrated in opposite directions: too weak in E19 (0.171 vs 0.342) but too strong in E12 (1.430 vs 1.297). Finally, the promised between-subject heterogeneity is still underdispersed\u2014especially E6 (candidate var 0.058 vs real 0.424), E5 (0.026 vs 0.158), E19 (0.006 vs 0.052), and E14 (0.091 vs 0.330). These are not just sampling noise; they reflect systematic mis-settings of the same few in-family knobs.",
  "verdict": "regenerate",
  "rationale": "Keep the accepted damped inverse-valence plus continuous top-coalition architecture, but make the following in-family recalibrations. (1) Fix E6's sign error by strengthening large-conflict validity weighting more aggressively: raise the lambda median well above 0.24 (toward roughly 0.4-0.6) and/or push lam_large_scale toward or above 1.0, and widen the lambda subject-level distribution so some subjects strongly follow the validity-weighted tally while others remain raw-majority driven. Without this, E6 will stay negative. (2) Rebalance tied-top coalitions without the current harsh large-group penalty: increase base tie_str (e.g., median 1.5-2.0) and reduce tie_opp_decay/tie_d_decay for m=2-3 to lift E5 and E8; replace or sharply soften the inverse-linear tie_large_decay term\u2014currently 1/(1+3*(m-3)) suppresses m=5 by about 7x\u2014with a saturating tanh penalty that only bites for m>=5, so E13/E14 rise toward their real values. (3) Recalibrate unique-top precedence in a conflict-dependent way: raise p_prec and reduce prec_conf_att, prec_d_decay, and prec_opp_decay to lift E19's moderate-conflict top-following, but simultaneously strengthen the aligned-with-majority attenuation (lower align_scale, increase align_d_att, and extend it to smaller discriminating sets) to pull E12 back down. This split should fix the E19-too-low/E12-too-high asymmetry without changing the mechanism family. (4) Widen the heavy-tailed subject-level sigmas for lambda, tie_strength/tie_opp terms, and precedence terms to restore the observed between-subject variances, especially E5/E6/E8/E14/E19. After these changes, re-check E2/E4/E3/E10 to ensure the strengthened validity weighting does not over-amplify raw-majority sensitivity in those designs; if E4 rises too far, a small gain reduction or slight lapse increase is the appropriate in-family compensation."
}
```

## Usage

```json
{
  "input_tokens": 29615,
  "output_tokens": 11609,
  "reasoning_tokens": 10775,
  "provider": "AtlasCloud"
}
```
