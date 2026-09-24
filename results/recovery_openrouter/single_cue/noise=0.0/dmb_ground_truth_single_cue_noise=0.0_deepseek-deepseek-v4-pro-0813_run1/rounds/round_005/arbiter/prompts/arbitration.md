# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_7" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_7" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_7
People evaluate each advertised expert validity relative to the ensemble median, so that cues below the median acquire context-sensitive salience through a median-normalized deviance score. A rank-based baseline keeps higher-validity cues from being completely ignored, but only when the genuine low-validity outlier bloc is small; as the number of below-median outlier cues grows, this high-validity baseline is smoothly suppressed. Cue weights are capped by a smooth, low-bloc-size-dependent maximal share, so that no single cue can monopolize the decision while still allowing redistribution toward high-validity cues when several low-validity cues compete. Choice gain is set by smooth functions of low-outlier bloc size, validity gap, and effective number of competing cues.

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


`policy(probs) -> int`:
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))


## THEORY 2 — pi_5
People normalize each advertised cue validity relative to the current task ensemble, giving salience only to cues whose validity falls below the ensemble median. Low-validity outliers dominate the signed feature tally, with ties contributing zero. The decisiveness of the softmax choice rule is itself context-gated: when a single cue becomes a near-monopoly in normalized decision weight, the response temperature is quenched toward chance; when a small set of low-validity cues is nearly co-equal, the response is sharpened. This keeps equal-weight tallying as a limiting case while allowing reversed-validity-like and regime-sensitive choice behavior.

`predict(parameters, state, history) -> np.ndarray`:
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


`policy(probs) -> int`:
def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))


## EXPERIMENT 1 (proposed by pi_7)

### DESIGN
**Validities (n_features=8):** [0.5, 0.8, 0.8, 0.8, 0.8, 0.85, 0.9, 0.95]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  trial 4: A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  trial 5: A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  trial 6: A=[1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]
  trial 7: A=[1, 1, 1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 8: A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  trial 10: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1]
  trial 11: A=[0, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1, 1, 1]
  trial 12: A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 1]
  trial 13: A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1, 1]
  trial 14: A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1]
  trial 15: A=[0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 1]
  trial 16: A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** Validities are one 50% expert, four 80% experts (the ensemble median), and 85/90/95% experts. The 50% cue has huge normalized deviance (median=0.80, MAD≈0.025), so in the competing model it receives roughly 99.6% of normalized weight; its near-monopoly then quenches beta to its low floor, giving nearly flat p(A)≈0.57 when the low cue favors A and ≈0.43 when it favors B, regardless of the high-cue pattern. In the advocated model the same lone low cue is capped at cap_share≈0.375 (low_bloc_frac=1/8), and the cap redistributes the remaining ≈0.625 weight to the seven high-validity cues. The high-cue sweep therefore moves advocated signed evidence from about -0.25 to +1.00 in the low-A block and from -1.00 to +0.25 in the low-B block, producing a steep psychometric function that crosses chance. Key directional reversals include low-A k=0 (advocated p(A)≈0.35 vs competing≈0.57) and low-B k=7 (advocated≈0.65 vs competing≈0.43). The 16 unique pairs are two blocks (low cue favors A vs B) × k=0..7 high cues favoring A in ascending validity order; K=6 gives 96 trials.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This is a within-block psychometric endpoint contrast. Separately for the low-cue-favors-A and low-cue-favors-B blocks, it takes P(choose A) when all seven high-validity cues favor A minus P(choose A) when all seven high-validity cues favor B. The competing theory gives the single below-median cue nearly all normalized weight, quenches beta, and therefore leaves P(A) almost flat across the high-cue sweep, so both within-block differences are near zero. The advocated theory caps the low cue near a 0.375 share and redistributes the remaining weight to high-validity cues, so the same sweep moves signed evidence from about -0.25 to 1.00 in the low-A block and from -1.00 to 0.25 in the low-B block, producing an aggregate metric near 0.55 instead of near 0. The endpoint trials are repeated six times per subject per block, keeping the per-subject estimate stable.

Source:
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


### RESULTS
- Predicted under pi_7 (simulated): 0.7900 (var=0.0109)
- Predicted under pi_5 (simulated): 0.4950 (var=0.0193)
- Observed on real data: 0.4833 (var=0.0144)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=12):** [0.5, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 4: A=[1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 5: A=[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  trial 6: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
  trial 7: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 8: A=[0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 9: A=[0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 10: A=[0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 11: A=[0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  trial 12: A=[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

**Rationale:** This 12-expert design sets the ensemble median to 0.85 and MAD to 0.05. Six cues are below the median: one 0.50 cue and five 0.80 cues; six 0.90 cues are above the median and are tied at 1 in every trial so they add no signed evidence. Under the advocated model the deviance of the 0.50 cue is 7 times the deviance of each 0.80 cue, so after normalization the lone 0.50 cue receives about 0.56-0.79 of decision weight while all five 0.80 cues together receive only about 0.21-0.44. The 0.90 cues receive only the tiny baseline weight. The top-share never reaches the single-cue floor gate, so beta stays in the normal range. Therefore the advocated model follows the lone 0.50 cue: in Block A, where the 0.50 cue favors A, it predicts p(A) above 0.5 even when all five 0.80 cues oppose A (k=0 and k=1); in Block B, where the 0.50 cue favors B, it predicts p(A) below 0.5 even when all five 0.80 cues favor A (k=4 and k=5). Under the competing model, low_bloc_frac is 6/12 = 0.5, which suppresses the rank-based baseline and sets cap_share near 0.263. The lone 0.50 cue is capped at about 0.263 and the remaining weight is redistributed to the five 0.80 cues, which together carry about 0.737 of decision weight. The competing model therefore follows the majority of the 0.80 cues, producing opposite directional predictions in those same critical pairs: Block A k=0 and k=1 predict p(A) below 0.5, while Block B k=4 and k=5 predict p(A) above 0.5. The k=0..5 sweep in each block also yields different psychometric slopes for the two theories. The 12 unique pairs repeated 8 times give roughly 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the four high-conflict trial types (A k=0, A k=1, B k=4, B k=5) where the lone 0.50-validity cue and the five 0.80-validity cues point in opposite directions, with the 0.80 cue net majority being +/-5 or +/-3. The advocated theory gives normalized decision weight dominantly to the 0.50 cue, so choices should follow it on these trials. The competing theory caps the 0.50 cue and redistributes roughly 0.74 of the weight to the 0.80 cues, so choices should follow the 0.80 bloc instead. Averaging the follow-the-0.50-cue proportion over these 32 trials per subject gives a large mean gap while keeping per-subject variance driven mainly by binomial sampling noise, and it discards the tied 0.90 cues and low-conflict/agreeing trials that dilute the contrast.

Source:
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


### RESULTS
- Predicted under pi_7 (simulated): 0.3594 (var=0.0070)
- Predicted under pi_5 (simulated): 0.6706 (var=0.0116)
- Observed on real data: 0.8375 (var=0.0133)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
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
**Predicted under pi_7:** -0.1246 (var=0.0002)
**Predicted under pi_5:** -0.1025 (var=0.0003)

### Experiment 4
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
**Predicted under pi_7:** -0.7272 (var=0.0075)
**Predicted under pi_5:** -0.7328 (var=0.0080)

### Experiment 5
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
**Predicted under pi_7:** -0.2726 (var=0.0068)
**Predicted under pi_5:** -0.3414 (var=0.0125)

### Experiment 6
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
**Predicted under pi_7:** -0.6675 (var=0.0201)
**Predicted under pi_5:** -0.6958 (var=0.0145)

### Experiment 7
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
**Predicted under pi_7:** 0.8321 (var=0.0036)
**Predicted under pi_5:** 0.8250 (var=0.0030)

### Experiment 8
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
**Predicted under pi_7:** 0.4271 (var=0.0084)
**Predicted under pi_5:** 0.3704 (var=0.0099)

### Experiment 9
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
**Predicted under pi_7:** 0.0224 (var=0.0016)
**Predicted under pi_5:** 0.0068 (var=0.0020)

### Experiment 10
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
**Predicted under pi_7:** 0.0680 (var=0.0084)
**Predicted under pi_5:** -0.0010 (var=0.0097)

### Experiment 11
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
**Predicted under pi_7:** 0.0023 (var=0.0005)
**Predicted under pi_5:** 0.0003 (var=0.0008)

### Experiment 12
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
**Predicted under pi_7:** 0.2038 (var=0.0002)
**Predicted under pi_5:** 0.1290 (var=0.0004)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all 12 metrics, pi_5 is clearly the better theory. It is closer to the observed value in 8 of 12 experiments, and its total absolute error across all experiments is about 0.72, versus about 1.37 for pi_7. The decisive failures belong to pi_7's signature cap-and-redistribution mechanism. In Experiment 1, pi_7 predicts 0.790 against an observed 0.483, overshooting the high-cue sweep by a large margin. In Experiment 2, pi_7 predicts 0.359 while the observed follow-the-lone-0.50-cue rate is 0.838: pi_7 follows the moderate 0.80-validity bloc, but humans strongly follow the extreme low-validity cue. That directly contradicts pi_7's core claim that the low cue should be capped and weight redistributed to higher-validity cues. Pi_5's median-normalized low-deviance dominance assumption fits Experiments 1, 3, 4, 5, 7, and 9 well, and it has the correct direction in Experiment 2, but it is not fully task-invariant: it underpredicts Experiment 2 by about 0.17 and overpredicts Experiment 8 by about 0.22. Pi_7 edges out pi_5 only in Experiments 6, 10, 11, and 12, usually by small margins, probably because of its high-validity rank baseline. Together, this suggests pi_5 captures the main regularity but pi_7 is degenerate and neither theory is good enough across all experiments to justify only a model patch.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_7 with a distinctiveness-gated low-cue dominance theory. Keep the useful median/MAD-normalized deviance idea from pi_5, but do not use pi_7's hard share cap or its size-suppressed rank baseline. In the new theory, a below-median cue's weight should increase with both its deviance and its distinctiveness, defined as the normalized validity gap to the next-lowest cue. A single extreme low cue should therefore receive very large weight, producing the strong follow-the-0.50-cue behavior observed in Experiment 2, while a homogeneous below-median bloc should share weight more evenly. Add a small and strictly bounded high-validity rank baseline whose strength depends on high-bloc size and low-cue distinctiveness, allowing the high-cue sweep in Experiment 1 to produce a contrast near 0.48 and preserving the positive high-validity correlation in Experiment 12 without overriding an extreme outlier. Replace the current product-of-gates beta machinery with a simpler entropy/conflict gate: choice temperature should be low when weight entropy is low and one distinct low cue dominates the signed evidence, moderate when a high-cue sweep opposes a solitary outlier, and near-chance when evidence is low-spread, balanced, or ambiguous due to many similar cues, as in Experiments 8 and 9. This theory differs from pi_7 by removing the cap-and-redistribution core and by making the high-rank baseline conditional rather than size-suppressed; it differs from pi_5 by adding explicit cue distinctiveness and a simpler entropy-based stochasticity term, so it can sharpen Experiment 2 and damp the overpredicted Experiment 8 while retaining pi_5's good fits elsewhere."
}
```

## Usage

```json
{
  "input_tokens": 22097,
  "output_tokens": 10340,
  "reasoning_tokens": 9605,
  "provider": "Baidu"
}
```
