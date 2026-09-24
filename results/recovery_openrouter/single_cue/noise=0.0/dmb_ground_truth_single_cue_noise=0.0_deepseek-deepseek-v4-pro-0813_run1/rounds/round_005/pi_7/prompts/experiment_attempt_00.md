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
**Description:** People evaluate each advertised expert validity relative to the ensemble median, so that cues below the median acquire context-sensitive salience through a median-normalized deviance score. A rank-based baseline keeps higher-validity cues from being completely ignored, but only when the genuine low-validity outlier bloc is small; as the number of below-median outlier cues grows, this high-validity baseline is smoothly suppressed. Cue weights are capped by a smooth, low-bloc-size-dependent maximal share, so that no single cue can monopolize the decision while still allowing redistribution toward high-validity cues when several low-validity cues compete. Choice gain is set by smooth functions of low-outlier bloc size, validity gap, and effective number of competing cues.

**Parameters:**
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
- cap_base: {0.26}
- cap_peak: {0.44}
- cap_center: {0.22}
- cap_width: {0.10}
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

**`predict source code`:**
```python
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

    cap_base = float(parameters['cap_base'])
    cap_peak = float(parameters['cap_peak'])
    cap_center = float(parameters['cap_center'])
    cap_width = float(parameters['cap_width'])
    cap_share = cap_base + (cap_peak - cap_base) * np.exp(-0.5 * ((low_bloc_frac - cap_center) / cap_width) ** 2)
    cap_share = float(np.clip(cap_share, 0.20, 0.50))

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
```

**`policy source code`:**
```python
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))
```


## COMPETING THEORY
**Description:** People normalize each advertised cue validity relative to the current task ensemble, giving salience only to cues whose validity falls below the ensemble median. Low-validity outliers dominate the signed feature tally, with ties contributing zero. The decisiveness of the softmax choice rule is itself context-gated: when a single cue becomes a near-monopoly in normalized decision weight, the response temperature is quenched toward chance; when a small set of low-validity cues is nearly co-equal, the response is sharpened. This keeps equal-weight tallying as a limiting case while allowing reversed-validity-like and regime-sensitive choice behavior.

**Parameters:**
- w0: [0.005, 0.04]
- kappa: [1.5, 4.0]
- gamma: [1.0, 1.5]
- c: [0.005, 0.02]
- beta: [1.7, 2.3]
- epsilon: [0.12, 0.18]
- share_hi_center: [0.84, 0.92]
- share_hi_width: [0.012, 0.035]
- gap_hi_center: [0.42, 0.58]
- gap_hi_width: [0.03, 0.08]
- beta_floor_hi: [0.25, 0.35]
- eff_n_center: [2.95, 3.10]
- eff_n_width: [0.10, 0.18]
- top3_sim_center: [0.92, 0.98]
- top3_sim_width: [0.04, 0.10]
- beta_boost_lo: [0.8, 1.4]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus array.')

    validities = np.asarray(parameters['validities'], dtype=float).reshape(-1)
    if validities.shape[0] != stim.shape[1]:
        raise ValueError('validities length must match n_features.')

    median = np.median(validities)
    mad = np.median(np.abs(validities - median))
    c = float(parameters['c'])

    denom = mad + c
    if denom <= 0.0:
        denom = 1e-12
    deviance = np.maximum(0.0, median - validities) / denom

    w0 = float(parameters['w0'])
    kappa = float(parameters['kappa'])
    gamma = float(parameters['gamma'])
    weights = w0 + kappa * np.power(deviance, gamma)

    total = weights.sum()
    if total <= 0.0:
        weights = np.ones_like(weights) / max(weights.size, 1)
    else:
        weights = weights / total

    diff = stim[0] - stim[1]
    evidence_A = float(np.dot(weights, diff))

    sorted_w = np.sort(weights)[::-1]
    nw = sorted_w.shape[0]
    top1 = sorted_w[0]
    top2 = sorted_w[1] if nw > 1 else top1
    top3 = sorted_w[2] if nw > 2 else (sorted_w[-1] if nw > 0 else 0.0)
    top3_similarity = 1.0 - float(top1 - top3)
    effective_n = 1.0 / max(float(np.sum(weights ** 2)), 1e-12)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-float(np.clip(x, -500.0, 500.0))))

    share_hi_center = float(parameters['share_hi_center'])
    share_hi_width = float(parameters['share_hi_width'])
    gap_hi_center = float(parameters['gap_hi_center'])
    gap_hi_width = float(parameters['gap_hi_width'])
    beta_floor_hi = float(parameters['beta_floor_hi'])

    share_gate = sigmoid((top1 - share_hi_center) / share_hi_width)
    gap_gate = sigmoid(((top1 - top2) - gap_hi_center) / gap_hi_width)
    high_gate = share_gate * gap_gate

    eff_n_center = float(parameters['eff_n_center'])
    eff_n_width = float(parameters['eff_n_width'])
    top3_sim_center = float(parameters['top3_sim_center'])
    top3_sim_width = float(parameters['top3_sim_width'])
    beta_boost_lo = float(parameters['beta_boost_lo'])

    eff_gate = np.exp(-0.5 * ((effective_n - eff_n_center) / eff_n_width) ** 2)
    balance_gate = np.exp(-0.5 * ((top3_similarity - top3_sim_center) / top3_sim_width) ** 2)
    low_gate = eff_gate * balance_gate

    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    beta_eff = beta + beta_boost_lo * low_gate
    beta_eff = beta_eff * (1.0 - high_gate) + beta_floor_hi * high_gate
    beta_eff = float(np.clip(beta_eff, 0.05, 15.0))

    logits = np.array([evidence_A, 0.0])
    z = beta_eff * (logits - np.max(logits))
    e = np.exp(z)
    core = e / e.sum()

    n_opts = core.shape[0]
    return (1.0 - epsilon) * core + epsilon * np.ones(n_opts) / n_opts
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Eight experts are used. Expert 0 is advertised as 90% accurate; the remaining seven are clearly lower (75% down to 50%). Every critical trial makes Expert 0 discriminate: it favors one option. The seven lower experts are arranged so that k = 0,1,...,7 of them favor the opposite option. Take The Best stops at Expert 0 and therefore predicts a constant probability of choosing the top-cued option across all k. Tallying counts all eight experts; the tally margin for the top-cued option is 8 - 2k, so its predicted choice probability is a decreasing sigmoid in k, crossing chance at k=4 and reversing for k>4. Two mirrored blocks are included (Expert 0 favors A in one block, B in the other) to double observations and control for response bias. This yields a strong quantitative dissociation: a flat psychometric function over k under Take The Best versus a negatively sloped sigmoid under Tallying. The 16 unique pairs are repeated 6 times for roughly 96 trials.

[1] Six experts with validities 0.90, 0.80, 0.70, 0.60, 0.55, 0.50 define a fixed TTB cue order. The 16 unique pairs form a cue-locus × tally-margin design. In A-first trials (A0-A7) the first discriminating cue in validity order favors A; in B-first trials (B0-B7) it favors B. Higher-validity cues are tied on r=1 and r=2 trials so the first discriminator is not always the top expert, extending the already-explored design. Lower cues are arranged to produce tally margins (#A wins - #B wins) of -4, -3, -2, 0, +2, +3, +4. TTB stops at the first discriminator, so it predicts a single p(A) for all A-first trials and a single p(A) for all B-first trials, independent of the lower-cue margin. Tallying counts all six cues, so it predicts p(A) = sigmoid(beta * margin), increasing monotonically with the margin and equal to chance at margin 0. Thus TTB predicts a flat step function in margin within each first-discriminator direction, whereas Tallying predicts a graded psychometric function; zero-margin trials additionally oppose TTB's directional prediction to Tallying's chance prediction. The 16 unique pairs are repeated K=6 times for 96 trials.

[2] Six experts with validities 0.92, 0.84, 0.76, 0.68, 0.58, 0.50. The design holds the Tallying score constant within two blocks while varying validity-weighted evidence. The first eight trials are zero tally-margin pairs: A wins exactly one feature and B wins exactly one feature, so Tallying gives a 1-1 tie and predicts p(A)=0.5 for every pair. The discriminating feature pair is varied from (A wins the 50% expert, B wins the 92% expert) to the reverse, so the advocated validity-weighted evidence increases monotonically from w5-w0 to w0-w5, predicting a psychometric curve from below to above chance. The last eight trials are constant +2 tally-margin pairs: A wins 4 features, B wins 2, so Tallying predicts the same p(A)=sigmoid(2*beta) for all eight. The four A-win features are chosen along a dominance chain (each adjacent pair swaps one A-win from a lower-validity to a higher-validity feature), so the validity-weighted evidence strictly increases for any monotone validity-weighting function. With offset=0.5, gamma=1 the +2 block runs from -0.24 (weighted favors B despite A winning 4 of 6 features) to +1.12. Thus Tallying predicts two flat lines (0.5 and a single >0.5 value), while the advocated theory predicts two increasing psychometric functions of weighted evidence. The 16 unique pairs are repeated K=6 for about 96 trials.

[3] Twelve experts: six high-validity (0.94-0.99) and six low-validity (0.50-0.55). The eight unique pairs form four groups. Group A (A1-A3): A wins 3-5 high-validity features while B wins 4-6 low-validity features, so Tallying margin is -1 (predicts p(A)<0.5), but validity-weighted evidence is positive (predicts p(A)>0.5). Group B (B1-B3): A wins 4-6 low-validity features while B wins 3-5 high-validity features, so Tallying margin is +1 (predicts p(A)>0.5), but validity-weighted evidence is negative (predicts p(A)<0.5). Groups C and D have zero Tallying margin (predicts p(A)=0.5) but all-high vs all-low wins give positive and negative validity-weighted evidence respectively. Thus Tallying and validity-weighted integration make opposite directional predictions for every pair, and the within-group graded evidence allows estimating beta for both accounts.

[4] This design tests whether low-validity experts are overweighted (reversed-validity) against the equal-count Tallying benchmark. Two experts are advertised near chance (50%, 51%) and seven are near-perfect (98%–99.5%). On every trial the two low-validity experts favor one option while k = 0…7 high-validity experts favor the other option; all remaining experts are tied. In A-low trials Tallying gives A exactly 2 wins and B k wins, so its tally margin is 2 − k: it crosses chance at k = 2 and predicts B for k ≥ 3. Reversed-validity instead gives the 50%/51% experts weights of about 0.75–0.81 each (the largest weights) and the 98%–99.5% experts weights of about 0.12–0.31 each. The two low weights sum to about 1.5–1.6, which outweighs six of the seven high weights (and all seven for gamma ≈ 0.4). Thus reversed-validity predicts A for k = 0…6 in A-low trials and B for k = 0…6 in B-low trials. The critical k = 2…6 trials therefore produce opposite directional predictions: Tallying chooses the high-validity side (or guesses at k = 2), while reversed-validity chooses the low-validity side. The psychometric crossover is shifted from k = 2 under Tallying to k = 7 or beyond under reversed-validity, a large quantitative dissociation. The two mirrored blocks double observations and control for response bias.

[5] Eight experts with four near-chance validities (0.50-0.53) and four near-perfect validities (0.97-0.995). Reversed-validity weights near-chance experts heavily (weight roughly 0.74-0.81) and discounts near-perfect experts (weight roughly 0.12-0.29), while Tallying gives every expert equal unit weight. The 16 unique pairs form two blocks. Block 1 (pairs 1-8) holds the Tallying margin at 0 (A wins exactly 3 features, B wins exactly 3, 2 ties), so Tallying predicts p(A)=0.5 for every pair. The pairs swap A's wins from low-validity to high-validity features, so reversed-weighted evidence decreases from about +1.7 to -1.7, predicting a graded psychometric curve crossing chance. Block 2 (pairs 9-16) holds the Tallying margin at +2 (A wins 5, B wins 3), so Tallying predicts the same >0.5 p(A) for all eight pairs. The same low-to-high validity swap makes reversed-weighted evidence decrease from about +2.8 to -0.7, so reversed-validity predicts a decreasing curve that crosses below chance on the last three pairs, opposite in direction to Tallying. Thus Tallying predicts two flat lines (0.5 and a single >0.5 value), whereas reversed-validity predicts two graded psychometric functions of weighted evidence, with several pairs yielding opposite directional predictions.

[6] Nine experts are used: three near-chance experts (0.53, 0.52, 0.51) and six higher-validity experts (0.99 down to 0.90). The three near-chance experts are deliberately close together. In the competing theory they become the top three normalized weights, each about 0.32 to 0.33, so the top3-similarity gate is near 1 and the effective-n gate is near 3; this holds the context-gated beta roughly constant. All six higher validities together contribute less than about 0.02 of normalized weight, so the competing theory's evidence is set almost entirely by which option wins more of the three low-validity experts. Block A has A win two low experts and B win one, giving competing evidence about +0.33 for A on every trial, so p(A) stays near 0.65 to 0.70 regardless of the high-feature pattern. Block B mirrors this, giving competing evidence about -0.33 and p(A) near 0.30 to 0.35. The advocated theory does not normalize and weights each expert by max(1 - v, 0)^gamma. Its low-expert weights are near 0.80, but its high-expert weights remain substantial, from about 0.50 for the 0.90 expert down to about 0.25 for the 0.99 expert. Eight high-feature patterns are chosen so that the B-winning high-feature weight total S equals 0, 0.349, 0.752, 0.951, 1.202, 1.511, 1.707, or 2.267. In the two-low A block, advocated evidence for A therefore runs +3.08, +2.38, +1.58, +1.18, +0.68, +0.06, -0.33, -1.45, giving p(A) from near ceiling to near floor and crossing chance between the last two intermediate patterns. In the mirrored block it runs +1.45, +0.76, -0.05, -0.45, -0.95, -1.57, -1.96, -3.08. Thus the competitor predicts a flat line within each low-split block, while the advocated theory predicts a steep psychometric function of the high-validity pattern; four pairs are directionally opposite. The 16 unique pairs repeated 6 times give roughly 96 trials.

[7] This nine-expert design isolates ensemble-normalized cue weighting from fixed reversed-validity weighting. The validities are two 0.50 experts and seven 0.80-0.84 experts, so the median is 0.80. Under the advocated normalization, only the two 0.50 experts lie below the median and receive large deviance weights; the seven 0.80-0.84 experts have zero or negative deviance and receive only the tiny baseline w0. After normalization the two low-validity experts carry nearly all decision weight, giving effective_n near 2 and a softmax beta near 2.0. The competing reversed-diagnosticity model instead assigns weights near 0.78 to each 0.50 expert and weights near 0.53-0.57 to each 0.80-0.84 expert, so the seven high experts have about 3.9 total weight versus about 1.6 for the two low experts. The first eight trials hold the two low experts in a 1-1 split and sweep the high-validity block from all favoring B to all favoring A. The advocated theory predicts a flat p(A) near 0.50 across all eight patterns because the high experts are normalized into near-zero weights. The competing theory predicts a sharp psychometric step from near floor to near ceiling as the high-expert evidence crosses zero. The second eight trials give both low experts to A and repeat the same high-block sweep. The advocated theory again predicts a flat line near 0.82 because evidence is approximately +1.0 and beta is about 2.0, while the competing theory predicts a steep psychometric function shifted by the low-expert advantage. This yields a strong quantitative dissociation: two flat lines under the advocated theory versus two steep high-block psychometric functions under the competing theory, with several pairs producing opposite directional predictions. The 16 unique pairs are repeated K = 6 times for about 96 trials.

[8] Four tied low-validity experts at 0.50 and five tied high-validity experts at 0.85 give bottom_share = 4/9 = 0.444, which crosses the advocated theory's 0.42 threshold and engages its delta_high floor. The tied high ranks then receive a non-negligible normalized weight floor, about 0.20 to 0.25 of total decision weight. By contrast, the competing theory computes the median as 0.85 because the five high experts are the majority, so the high experts sit exactly at the ensemble median, receive only the tiny baseline w0, and are essentially ignored. The four low experts lie below the median and carry nearly all competing decision weight. In the low-tie block the four low experts are split 2 for A and 2 for B, so competing evidence is zero and the competing model predicts p(A) near 0.50 for every high-block split. The advocated model's high-validity floor makes high-block evidence vary from about -0.23 to +0.23 as the number of high experts favoring A goes from 0 to 5, so its predicted p(A) moves from about 0.33 to 0.67. In the A-majority low block (3 low experts favor A) the competing model predicts a flat p(A) near 0.70, while the advocated model predicts an increasing curve from about 0.61 to 0.86. In the B-majority low block the competing model predicts a flat p(A) near 0.30, while the advocated model predicts an increasing curve from about 0.14 to 0.39. This yields three flat lines under the competing theory and three monotonic high-block psychometric curves under the advocated theory, providing strong quantitative dissociations. Sixteen unique pairs are repeated six times for about 96 trials.

[9] Eleven experts: one lone 50% cue and ten tightly clustered 84.0%-86.7% cues. The median validity is 85.2% and MAD is about 0.9 percentage points, so under the advocated normalized-deviance model the 50% cue is a huge deviance outlier. It receives roughly 96% of normalized weight, while the three just-below-median experts swept in the design (84.0%, 84.3%, 84.6%) together carry only about 3% of weight and the higher experts carry almost none. This near-monopoly activates the single-cue high-share/gap gate and quenches the softmax beta to its low floor, so the advocated model predicts essentially flat choice functions across the three-expert sign sweep: p(A) near 0.50 in the low-tie block and a nearly constant value above 0.50 in the low-favors-A block. In the competing rank model bottom_share is only 1/11, so delta=0 and beta_low near 1.7 applies. Rank weights give the lone 50% cue about 0.44 of total weight, while the top three swept high cues together retain about 0.50 of weight, so the same 2x2x2 sign sweep moves competing signed evidence from about -0.50 to +0.50 in the tie block (p(A) roughly 0.30 to 0.70) and from about -0.06 to +0.94 in the low-favors-A block (p(A) rising from below 0.50 to above 0.80). The first low-favors-A pair is a directional reversal: the lone low cue favors A while all three swept high cues favor B, giving advocated p(A) above 0.50 but competing p(A) around 0.47. Features 4-10 are tied at 1 in every trial so they do not affect either model. This one-low plus tight-high-validity ensemble is distinct from the already-explored two-, three-, and four-low designs.

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "Validities are one 50% expert, four 80% experts (the ensemble median), and 85/90/95% experts. The 50% cue has huge normalized deviance (median=0.80, MAD\u22480.025), so in the competing model it receives roughly 99.6% of normalized weight; its near-monopoly then quenches beta to its low floor, giving nearly flat p(A)\u22480.57 when the low cue favors A and \u22480.43 when it favors B, regardless of the high-cue pattern. In the advocated model the same lone low cue is capped at cap_share\u22480.375 (low_bloc_frac=1/8), and the cap redistributes the remaining \u22480.625 weight to the seven high-validity cues. The high-cue sweep therefore moves advocated signed evidence from about -0.25 to +1.00 in the low-A block and from -1.00 to +0.25 in the low-B block, producing a steep psychometric function that crosses chance. Key directional reversals include low-A k=0 (advocated p(A)\u22480.35 vs competing\u22480.57) and low-B k=7 (advocated\u22480.65 vs competing\u22480.43). The 16 unique pairs are two blocks (low cue favors A vs B) \u00d7 k=0..7 high cues favoring A in ascending validity order; K=6 gives 96 trials.",
  "validities": [
    0.5,
    0.8,
    0.8,
    0.8,
    0.8,
    0.85,
    0.9,
    0.95
  ],
  "trial_a_ratings": [
    [
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
      0
    ],
    [
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
      1,
      1,
      1,
      1,
      1
    ],
    [
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
      1,
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      0,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      0,
      0
    ],
    [
      0,
      1,
      1,
      1,
      1,
      1,
      1,
      0
    ],
    [
      0,
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
      1,
      1,
      1,
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      1,
      1,
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      1,
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      1,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      1,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
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
      1
    ],
    [
      1,
      0,
      1,
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
      0,
      0,
      1,
      1,
      1,
      1,
      1
    ],
    [
      1,
      0,
      0,
      0,
      1,
      1,
      1,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0,
      1,
      1,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      1,
      1
    ],
    [
      1,
      0,
      0,
      0,
      0,
      0,
      0,
      1
    ],
    [
      1,
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
  "input_tokens": 8967,
  "output_tokens": 40039,
  "reasoning_tokens": 26218,
  "provider": "CoreWeave"
}
```
