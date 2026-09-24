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
- THEORY 1 = `pi_6`
- THEORY 2 = `pi_5`
- The recommendation below acts on THEORY 1 (= `pi_6`).

Replace pi_6 with a 'bounded contextual salience' theory. The new theory should compute median-normalized deviance as in pi_5, but add three safeguards. First, include a rank-based high-validity baseline so high-validity cues are not completely ignored when only one low-validity outlier is present. Second, cap the normalized weight share of any single cue, preventing the lone-outlier monopoly that caused pi_5's Exp 2 failure. Third, replace pi_6's hard bottom-share threshold and pi_5's binary high/low beta gates with smooth functions of low-bloc size, validity gap, and effective number of cues. This should preserve pi_5's successes in Exps 1, 3, 5, 6, 7, and 9, fix the Exp 2 and Exp 10 failures by retaining high-cue weight, and reduce the overpredicted contrast in Exp 8. It differs from pi_6 by eliminating the hard threshold/floor discontinuity, and differs from pi_5 by bounding outlier dominance and adding a high-rank baseline.


## CANDIDATE THEORY
People evaluate each advertised expert validity relative to the ensemble median, so that cues below the median acquire context-sensitive salience through a median-normalized deviance score. A rank-based baseline keeps higher-validity cues from being completely ignored, but only when the genuine low-validity outlier bloc is small; as the number of below-median outlier cues grows, this high-validity baseline is smoothly suppressed. Cue weights are capped so that no single cue can monopolize the decision, and choice gain is set by smooth functions of low-outlier bloc size, validity gap, and effective number of competing cues.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np
from scipy.stats import rankdata


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Bounded contextual salience expects a (2, n_features) stimulus.')

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

    deviance = np.maximum(0.0, median - validities) / scale

    low_dev_threshold = float(parameters['low_dev_threshold'])
    low_bloc_frac = float(np.mean(deviance > low_dev_threshold))

    w0 = float(parameters['w0'])
    kappa = float(parameters['kappa'])
    gamma = float(parameters['gamma'])
    dev_weight = w0 + kappa * np.power(deviance, gamma)

    ranks = rankdata(validities, method='average')
    if n_feat > 1:
        u = (ranks - 1.0) / (n_feat - 1.0)
    else:
        u = np.array([0.5])

    b_scale = float(parameters['b_scale'])
    b_base = float(parameters['b_base'])
    b_alpha = float(parameters['b_alpha'])
    gap_center = float(parameters['gap_center'])
    gap_width = float(parameters['gap_width'])

    gap_norm = (float(np.max(validities)) - float(np.min(validities))) / (float(np.median(validities)) + c)
    gap_activ = 1.0 / (1.0 + np.exp(-float(np.clip((gap_norm - gap_center) / gap_width, -500.0, 500.0))))

    rank_gate_center = float(parameters['rank_gate_center'])
    rank_gate_width = float(parameters['rank_gate_width'])
    rank_gate = 1.0 / (1.0 + np.exp(float(np.clip((rank_gate_center - low_bloc_frac) / rank_gate_width, -500.0, 500.0))))
    rank_base = b_scale * (b_base + gap_activ) * np.power(u, b_alpha) * rank_gate

    raw = dev_weight + rank_base
    total = float(np.sum(raw))
    if total <= 0.0:
        weights = np.ones(n_feat, dtype=float) / float(max(n_feat, 1))
    else:
        weights = raw / total

    cap_share = float(parameters['cap_share'])
    for _ in range(100):
        max_w = float(np.max(weights)) if weights.size else 0.0
        if max_w <= cap_share + 1e-12:
            break
        weights = np.minimum(weights, cap_share)
        total_w = float(np.sum(weights))
        if total_w <= 1e-12:
            weights = np.ones(n_feat, dtype=float) / float(max(n_feat, 1))
            break
        weights = weights / total_w

    diff = stim[0] - stim[1]
    evidence_A = float(np.dot(weights, diff))

    sorted_w = np.sort(weights)[::-1]
    top1 = float(sorted_w[0]) if sorted_w.size > 0 else 1.0
    top2 = float(sorted_w[1]) if sorted_w.size > 1 else top1
    top3 = float(sorted_w[2]) if sorted_w.size > 2 else (float(sorted_w[-1]) if sorted_w.size > 0 else 0.0)
    top3_sim = 1.0 - float(top1 - top3)
    effective_n = 1.0 / max(float(np.sum(weights * weights)), 1e-12)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-float(np.clip(x, -500.0, 500.0))))

    share_hi_center = float(parameters['share_hi_center'])
    share_hi_width = float(parameters['share_hi_width'])
    gap_hi_center = float(parameters['gap_hi_center'])
    gap_hi_width = float(parameters['gap_hi_width'])
    share_gate = sigmoid((top1 - share_hi_center) / share_hi_width)
    weight_gap_gate = sigmoid(((top1 - top2) - gap_hi_center) / gap_hi_width)
    high_gate = share_gate * weight_gap_gate

    eff_n_center = float(parameters['eff_n_center'])
    eff_n_width = float(parameters['eff_n_width'])
    top3_sim_center = float(parameters['top3_sim_center'])
    top3_sim_width = float(parameters['top3_sim_width'])
    low_bloc_center = float(parameters['low_bloc_center'])
    low_bloc_width = float(parameters['low_bloc_width'])

    eff_gate = np.exp(-0.5 * ((effective_n - eff_n_center) / eff_n_width) ** 2)
    balance_gate = np.exp(-0.5 * ((top3_sim - top3_sim_center) / top3_sim_width) ** 2)
    low_size_gate = np.exp(-0.5 * ((low_bloc_frac - low_bloc_center) / low_bloc_width) ** 2)
    low_gate = eff_gate * balance_gate * low_size_gate

    beta_base = float(parameters['beta_base'])
    beta_floor_hi = float(parameters['beta_floor_hi'])
    beta_boost_lo = float(parameters['beta_boost_lo'])
    beta_eff = beta_base + beta_boost_lo * low_gate
    beta_eff = beta_eff * (1.0 - high_gate) + beta_floor_hi * high_gate
    beta_eff = float(np.clip(beta_eff, 0.10, 12.0))

    epsilon = float(parameters['epsilon'])
    logits = np.array([evidence_A, 0.0], dtype=float)
    z = beta_eff * (logits - float(np.max(logits)))
    e = np.exp(z)
    core = e / e.sum()

    n_opts = core.shape[0]
    return (1.0 - epsilon) * core + epsilon * np.ones(n_opts, dtype=float) / float(n_opts)


`policy(probs) -> int`:
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))


`parameters`:
- c: [0.008, 0.014]
- low_dev_threshold: {0.70}
- rank_gate_center: {0.19}
- rank_gate_width: {0.07}
- w0: [0.005, 0.015]
- kappa: [1.8, 2.2]
- gamma: [1.3, 1.5]
- b_scale: [0.30, 0.50]
- b_base: [0.15, 0.25]
- b_alpha: [1.3, 1.7]
- gap_center: {0.38}
- gap_width: {0.10}
- cap_share: {0.34}
- share_hi_center: {0.85}
- share_hi_width: {0.02}
- gap_hi_center: {0.40}
- gap_hi_width: {0.06}
- eff_n_center: {3.0}
- eff_n_width: {0.14}
- top3_sim_center: {0.95}
- top3_sim_width: {0.07}
- low_bloc_center: {0.48}
- low_bloc_width: {0.10}
- beta_base: [2.4, 2.8]
- beta_floor_hi: [0.35, 0.50]
- beta_boost_lo: [0.35, 0.60]
- epsilon: [0.04, 0.06]
- validities: validities

`rationale`:
This is a localized correction of the previous bounded-contextual-salience candidate. The previous model gated the rank-based high-validity baseline only through a raw validity gap and used `validities <= median` as the low-validity bloc measure, which lumped near-median cues into the low bloc and left the rank baseline too active in multi-low-cue experiments. The edit computes a genuine low-validity bloc fraction from median-normalized deviance outliers, then multiplies the rank baseline by a smooth decreasing sigmoid gate on that fraction. The gate is near 0.8 for a single glaring low-validity outlier such as Exp 10, but falls quickly as the outlier bloc grows to a third or half of the ensemble, suppressing the high-validity baseline in Exp 2, Exp 3, Exp 6, and Exp 8. The same corrected low-bloc fraction feeds the existing smooth beta/choice-strength path, preserving the capped cue weights and softmax machinery while removing the unconditional high-rank influence identified by the critic.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.1041 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.0931 -> ACCEPTED
Running-best (last accepted) base: iter 2 at loss=0.0931.

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
  - iter 1: -0.1245 (var=0.0002) (Δ vs real -0.0286)
  - iter 2 (current): -0.1251 (var=0.0002) (Δ vs real -0.0292)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0006 (var=0.0002)
- pi_2: -0.1360 (var=0.0009)
- pi_3: -0.1493 (var=0.0006)
- pi_4: -0.1505 (var=0.0003)
- pi_5: -0.1025 (var=0.0003)
- pi_6: -0.1359 (var=0.0002)

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
  - iter 1: -0.7567 (var=0.0164) (Δ vs real +0.1025)
  - iter 2 (current): -0.7432 (var=0.0095) (Δ vs real +0.1160)
**Other theories' values on this metric (for reference):**
- pi_2: -0.7597 (var=0.0423)
- pi_1: 0.7279 (var=0.0420)
- pi_3: -0.6946 (var=0.0665)
- pi_4: -0.8321 (var=0.0089)
- pi_5: -0.7328 (var=0.0080)
- pi_6: -0.8021 (var=0.0109)

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
  - iter 1: -0.2586 (var=0.0065) (Δ vs real +0.0899)
  - iter 2 (current): -0.2514 (var=0.0104) (Δ vs real +0.0971)
**Other theories' values on this metric (for reference):**
- pi_3: 0.5096 (var=0.0191)
- pi_2: 0.0215 (var=0.0080)
- pi_1: 0.5469 (var=0.0262)
- pi_4: -0.2535 (var=0.0057)
- pi_5: -0.3414 (var=0.0125)
- pi_6: -0.3640 (var=0.0059)

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
  - iter 1: -0.7175 (var=0.0145) (Δ vs real -0.0642)
  - iter 2 (current): -0.6742 (var=0.0130) (Δ vs real -0.0208)
**Other theories' values on this metric (for reference):**
- pi_2: -0.1783 (var=0.0317)
- pi_3: 0.8683 (var=0.0156)
- pi_1: 0.6692 (var=0.0641)
- pi_4: -0.7908 (var=0.0142)
- pi_5: -0.6958 (var=0.0145)
- pi_6: -0.7150 (var=0.0209)

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
  - iter 1: 0.7908 (var=0.0036) (Δ vs real -0.0367)
  - iter 2 (current): 0.8054 (var=0.0034) (Δ vs real -0.0221)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8642 (var=0.0036)
- pi_2: 0.1492 (var=0.0089)
- pi_1: 0.1346 (var=0.0111)
- pi_3: 0.0708 (var=0.0019)
- pi_5: 0.8250 (var=0.0030)
- pi_6: 0.7621 (var=0.0044)

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
  - iter 1: 0.4008 (var=0.0095) (Δ vs real +0.2483)
  - iter 2 (current): 0.3746 (var=0.0060) (Δ vs real +0.2221)
**Other theories' values on this metric (for reference):**
- pi_2: 0.0092 (var=0.0068)
- pi_4: 0.6663 (var=0.0054)
- pi_1: -0.4425 (var=0.0114)
- pi_3: -0.6067 (var=0.0096)
- pi_5: 0.3704 (var=0.0099)
- pi_6: 0.3529 (var=0.0067)

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
  - iter 1: 0.0242 (var=0.0022) (Δ vs real +0.0345)
  - iter 2 (current): 0.0296 (var=0.0024) (Δ vs real +0.0399)
**Other theories' values on this metric (for reference):**
- pi_4: 0.2803 (var=0.0012)
- pi_5: 0.0068 (var=0.0020)
- pi_1: 0.1888 (var=0.0043)
- pi_2: 0.2909 (var=0.0059)
- pi_3: 0.3502 (var=0.0022)
- pi_6: 0.0282 (var=0.0018)

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
  - iter 1: 0.1869 (var=0.0096) (Δ vs real +0.1397)
  - iter 2 (current): 0.1938 (var=0.0070) (Δ vs real +0.1466)
**Other theories' values on this metric (for reference):**
- pi_5: -0.0010 (var=0.0097)
- pi_4: 0.6671 (var=0.0037)
- pi_1: 0.3105 (var=0.0223)
- pi_2: 0.6061 (var=0.0242)
- pi_3: 0.7504 (var=0.0068)
- pi_6: 0.0794 (var=0.0104)

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
  - iter 1: -0.0025 (var=0.0007) (Δ vs real -0.0057)
  - iter 2 (current): 0.0059 (var=0.0007) (Δ vs real +0.0027)
**Other theories' values on this metric (for reference):**
- pi_6: 0.0500 (var=0.0010)
- pi_5: 0.0003 (var=0.0008)
- pi_1: 0.1056 (var=0.0010)
- pi_2: 0.1807 (var=0.0017)
- pi_3: 0.2205 (var=0.0005)
- pi_4: 0.1640 (var=0.0005)

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
  - iter 1: 0.1890 (var=0.0003) (Δ vs real -0.0111)
  - iter 2 (current): 0.1973 (var=0.0002) (Δ vs real -0.0028)
**Other theories' values on this metric (for reference):**
- pi_5: 0.1290 (var=0.0004)
- pi_6: 0.1905 (var=0.0004)
- pi_1: 0.1377 (var=0.0003)
- pi_2: 0.2228 (var=0.0005)
- pi_3: 0.2121 (var=0.0003)
- pi_4: 0.2385 (var=0.0002)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The candidate faithfully implements the prescribed bounded contextual salience family: median-normalized deviance, a capped weight vector, a rank-based high-validity baseline, and smooth beta/choice-strength gates. It lands close on Exp 10 (candidate 0.189 vs real 0.200), Exp 9 (-0.003 vs 0.003), Exp 7 (0.024 vs -0.010), and is broadly reasonable on Exp 1, Exp 4, and Exp 5. However, a coherent failure pattern remains: the high-validity rank baseline is applied almost unconditionally, producing too much choice influence for high-validity cues whenever the stimulus set is not a lone-outlier context. This shows up as overpredicted positive/high-cue effects in Exp 8 (0.187 vs real 0.047, and worse than pi_6 at 0.079 despite the arbiter explicitly asking to reduce that contrast), Exp 6 (0.401 vs real 0.153), and insufficiently negative directional effects in Exp 2 (-0.757 vs real -0.859) and Exp 3 (-0.259 vs real -0.349). The model's low_bloc_frac is computed with validities <= median, which lumps near-median high-validity cues into the 'low bloc' and therefore cannot appropriately gate the rank baseline as a function of genuine low-validity outlier size.
Rationale: Keep the architecture of bounded contextual salience but fix the rank-based baseline gating. Compute a genuine low-validity bloc fraction from normalized deviance outliers (for example, the fraction of cues with deviance above a modest threshold such as 1.0-1.5), and multiply rank_base by a smooth decreasing gate on that fraction so the high-validity baseline is mainly active for a small genuine low-validity bloc (as in Exp 10) and is suppressed when several below-median cues are present (as in Exp 2, Exp 3, Exp 6, Exp 8). Use the same corrected low-bloc measure inside the existing low_size_gate/beta path rather than the current 'validities <= median' definition. Optionally reduce b_base/b_scale modestly and stiffen the gap gate, but only after verifying Exp 10 remains near 0.19-0.20. This is a localized parameter and gating correction within the prescribed mechanism family, not a change of theory.

**Outcome of your advice:** iter 1 candidate loss=0.1041 -> CURRENT candidate loss=0.0931 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate is the best aggregate fit so far (loss 0.0931) and is close on Exps 4, 5, 9 and 10, so the previous correction removed the worst Exp 10 failure. However, it still shows a coherent, systematic over-weighting of high-validity cues whenever the low-validity bloc is small or sits near the median. Exp 8 is the clearest miss: candidate 0.1938 vs real 0.0472, which is worse than both pi_5 (-0.0010) and pi_6 (0.0794) and directly violates the arbiter request to reduce that overpredicted contrast. Exp 6 remains too positive (0.3746 vs real 0.1525), Exp 2 is not fixed enough (-0.7432 vs real -0.8592, with pi_6 at -0.8021), and Exp 3 loses the pi_5-like behavior the recommendation wanted preserved (-0.2514 vs real -0.3485 and pi_5 -0.3414). The rank-based high-validity baseline is therefore still too broad. The main culprit is the gate on low_bloc_frac: it is built from deviance greater than 0.70 and centered at rank_gate_center 0.19 with width 0.07, which is calibrated for Exp 10's single 1/11 outlier but still leaves the high-validity baseline active for Exp 8's 2/9 low-cue fraction. In addition, a deviance-outlier count can fail to recognize a low-validity majority that sits exactly at or near the median, as in Exp 3, leaving the rank baseline nearly fully on when it should be suppressed.",
  "verdict": "regenerate",
  "rationale": "Regenerate within the bounded contextual salience family. First, recalibrate the rank gate so the high-validity baseline dies off quickly once the genuine low-validity bloc grows beyond a single outlier: move rank_gate_center from 0.19 down to roughly 0.10-0.12 and narrow rank_gate_width to about 0.04-0.06. That should turn the baseline off for Exp 8's 2/9 fraction while keeping Exp 10's 1/11 fraction substantially active. Second, augment low_bloc_frac with a gap-aware low-cluster term, for example a smooth blend of the current deviance-outlier fraction and the fraction of cues at or below the median, so near-median low-validity majorities such as Exp 3 are counted. Avoid reverting to a raw below-or-equal-median switch; instead keep it smooth and multiply by the existing validity-gap and effective-number gates. Third, modestly reduce high-cue gain by lowering b_scale to about 0.20-0.35 and b_base to about 0.10-0.18, and verify that Exp 10 does not fall much below 0.19 and that Exp 2 moves toward -0.86. Keep the cap and softmax machinery intact; the problem is the pre-cap rank baseline, not the cap itself."
}
```

## Usage

```json
{
  "input_tokens": 20334,
  "output_tokens": 10768,
  "reasoning_tokens": 10081,
  "provider": "Baidu"
}
```
