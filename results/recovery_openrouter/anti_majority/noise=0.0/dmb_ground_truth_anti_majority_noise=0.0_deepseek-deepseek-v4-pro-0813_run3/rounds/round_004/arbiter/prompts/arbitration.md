# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4_1" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4_1
People invert the valence of binary expert ratings: a 1 is treated as a costly defect and a 0 as a clean, favorable signal. Across all features they compute an inverse-tally advantage for A, weakly weighting the inverse cues by the experts' stated validities. When the raw inverse tally is nonzero, choice is a noisy softmax over the two weakly validity-weighted inverse tallies. When the raw inverse tally is exactly tied, people do not guess uniformly; instead they break the tie lexicographically by consulting features in descending validity order and choosing the option that has a clean 0 on the first discriminating feature. Response noise is captured by a softmax plus an independent uniform lapse. This differs from Take The Best because it aggregates all cues rather than stopping at the first discriminator, and from plain inverse tallying because ties are resolved by validity-ordered clean-cue use instead of guessing.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np
from scipy.stats import norm


def _trunc_qnorm(u, mu, sigma, lo, hi):
    """Map a (0,1) draw to a normal quantile truncated to [lo, hi]."""
    u = float(np.clip(u, 0.0, 1.0))
    q_lo = norm.cdf((lo - mu) / sigma)
    q_hi = norm.cdf((hi - mu) / sigma)
    q = q_lo + u * (q_hi - q_lo)
    return mu + sigma * norm.ppf(q)


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    val = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    # --- Subject-level parameter transforms ---------------------------------
    # Raw declared ranges are mapped through truncated normal/lognormal
    # distributions so the intended medians and dispersions are realized
    # per subject rather than sampled uniformly.
    beta_tally_raw = float(parameters['beta_tally'])
    beta_tiebreak_raw = float(parameters['beta_tiebreak'])
    epsilon_raw = float(parameters['epsilon'])
    validity_weight_raw = float(parameters['validity_weight'])

    # Lower median inverse temperature to soften the inverse-tally gradient,
    # especially the s = +/-2 separation in Experiment 4, while retaining a
    # moderately wide support for between-subject variance.
    u_beta = (beta_tally_raw - 0.22) / (0.85 - 0.22)
    beta_tally = np.exp(_trunc_qnorm(
        u_beta, np.log(0.45), 0.40, np.log(0.22), np.log(0.85)))
    beta_tally = float(np.clip(beta_tally, 0.22, 0.85))

    # Keep tie-break temperature near 1.0 but widen the subject-level range.
    # The median is nudged to 1.02 to hold Experiment 3 near its observed
    # anti-TTB tie-break rate once the lapse distribution is widened.
    u_tie = (beta_tiebreak_raw - 0.50) / (1.70 - 0.50)
    beta_tiebreak = np.exp(_trunc_qnorm(
        u_tie, np.log(1.02), 0.40, np.log(0.50), np.log(1.70)))
    beta_tiebreak = float(np.clip(beta_tiebreak, 0.50, 1.70))

    # Shift the single skewed validity-weight distribution upward as advised:
    # median near 1.60 and a small upper tail toward 2.40. This preserves a
    # single unimodal distribution with no rejected mixture component.
    u_w = (validity_weight_raw - 0.60) / (2.40 - 0.60)
    z_w = _trunc_qnorm(
        u_w, np.log(1.10), 0.40, np.log(0.10), np.log(1.90))
    validity_weight = 0.50 + np.exp(z_w)
    validity_weight = float(np.clip(validity_weight, 0.60, 2.40))

    # Widen the lapse distribution only modestly, centered near 0.09, so
    # central tendencies in Experiments 3, 5, and 7 are not flattened while
    # enough heterogeneity is added.
    logit_lo = np.log(0.04 / 0.96)
    logit_hi = np.log(0.18 / 0.82)
    logit_mu = np.log(0.09 / 0.91)
    u_eps = (epsilon_raw - 0.04) / (0.18 - 0.04)
    z_eps = _trunc_qnorm(u_eps, logit_mu, 0.40, logit_lo, logit_hi)
    epsilon = 1.0 / (1.0 + np.exp(-z_eps))
    epsilon = float(np.clip(epsilon, 0.04, 0.18))

    # --- Inverse clean-cue comparison --------------------------------------
    clean_a = (b > a)
    clean_b = (a > b)
    raw_s = float(np.sum(clean_a) - np.sum(clean_b))

    if raw_s != 0.0:
        # Weak-to-moderate, all-cue validity weighting. Mean-centering keeps
        # the weights near one so low-validity majorities can still move the
        # tally when present.
        if validity_weight > 0.0 and n_features > 1:
            weights = 1.0 + validity_weight * (val - np.mean(val))
        else:
            weights = np.ones(n_features)

        s = float(np.sum(weights * clean_a) - np.sum(weights * clean_b))
        scores = np.array([s, -s])
        beta = beta_tally
    else:
        # Exact raw tally tie: consult features in descending validity order
        # and choose the alternative with a clean 0 on the first discriminator.
        cue_order = np.argsort(-val, kind='stable')
        winner = None
        for j in cue_order:
            if a[j] != b[j]:
                winner = 0 if (a[j] == 0 and b[j] == 1) else 1
                break

        if winner is None:
            return np.ones(2) / 2.0

        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        beta = beta_tiebreak

    logits = beta * scores
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core = p_core / p_core.sum()

    return (1.0 - epsilon) * p_core + epsilon * np.ones(2) / 2.0

`policy(probs) -> int`:
import numpy as np


def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_5
People invert the valence of binary expert ratings: a 1 is treated as a defect and a 0 as a clean, favorable signal. Each feature j probabilistically enters an inverse clean-cue tally with inclusion probability p_include_j = logistic(alpha * validity_j + gamma); excluded cues receive exactly zero weight. The included clean-cue tally drives a noisy softmax choice, and exact zero tallies are resolved by consulting the highest-validity included discriminating cue and choosing the option with the clean 0 there. If no included cue discriminates, people guess. Subject-level alpha, gamma, beta_tally, beta_tiebreak, and epsilon create individual differences, with heavy-tailed/logit-transformed subject-level distributions so that some subjects are nearly deterministic while others remain noisy, preserving the pooled inverse-valence pattern and realistically large between-subject variance.

`predict(parameters, state, history) -> np.ndarray`:
import numpy as np
from scipy.special import expit

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    val = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    # Heavy-tailed subject-level transforms of the sampled raw parameters.
    beta_tally = float(np.exp(float(parameters['beta_tally_log'])))
    beta_tiebreak = float(np.exp(float(parameters['beta_tiebreak_log'])))
    epsilon = float(expit(float(parameters['epsilon_logit'])))
    alpha = float(parameters['alpha'])
    gamma = -4.5 + 2.0 * float(expit(float(parameters['gamma_logit'])))

    p_include = expit(alpha * val + gamma)
    include = np.random.random(n_features) < p_include

    clean_a = ((b > a) & include).astype(float)
    clean_b = ((a > b) & include).astype(float)
    s = float(np.sum(clean_a) - np.sum(clean_b))

    if s != 0.0:
        p_a = expit(2.0 * beta_tally * s)
        p_core = np.array([p_a, 1.0 - p_a])
    else:
        winner = None
        cue_order = np.argsort(-val, kind='stable')
        for j in cue_order:
            if include[j] and a[j] != b[j]:
                winner = 0 if (a[j] == 0 and b[j] == 1) else 1
                break
        if winner is None:
            return np.ones(2) / 2.0
        p_a = expit(beta_tiebreak) if winner == 0 else expit(-beta_tiebreak)
        p_core = np.array([p_a, 1.0 - p_a])

    return (1.0 - epsilon) * p_core + epsilon * np.ones(2) / 2.0


`policy(probs) -> int`:
import numpy as np

def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_4_1)

### DESIGN
**Validities (n_features=8):** [0.95, 0.9, 0.65, 0.64, 0.63, 0.62, 0.6, 0.6]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 1, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0, 1, 1]
  trial 4: A=[1, 0, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0, 0, 1]
  trial 5: A=[1, 0, 0, 0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0, 0, 0, 1]
  trial 6: A=[1, 0, 0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0, 0, 1]

**Rationale:** This design holds the raw inverse tally at exactly 0 on every unique pair, so the advocated theory is always in its validity-ordered tie-break mode and chooses using only the first discriminating feature. The top two experts (validities 0.95 and 0.90) are tied in all pairs, so the first discriminator is placed at validity 0.65 or 0.64 after those ties. The six pairs form a first-direction by lower-cue-mass design. Advocated theory predicts a flat step of approximately p(A)=0.71 when the first discriminating cue is A=0/B=1 and approximately p(A)=0.29 when it is A=1/B=0, with no effect of later cues. The competing stochastic-inclusion theory does not have this deterministic raw-tie mode: it includes later cues probabilistically, and in the k=3 pairs the three mid-validity opposite cues (0.64, 0.63, 0.62) plus two 0.50 same-side cues give an expected included tally of about -0.40. Under typical parameters alpha=7, gamma=-3.5, beta_tally around 1.1, beta_tiebreak around 2, p(A) is approximately 0.47 for the first-cue-A k=3 pair and approximately 0.53 for its mirror, opposite in direction to the advocated tie-break. The k=2 pairs provide a weaker opposite lower-cue mass and therefore an intermediate quantitative gradient. This separates a deterministic validity-ordered raw-tie-break from stochastic all-cue inclusion while avoiding the previously explored top-cue tie-break designs.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
All six unique pairs in this design have equal numbers of raw inverse clean-cue advantages for A and B, so the advocated theory enters its validity-ordered tie-break mode. The metric therefore isolates the first discriminating feature for every trial, ordering features by the fixed validities 0.95, 0.90, 0.65, 0.64, 0.63, 0.62, 0.60, 0.60. A trial is labelled A-first when the first discriminating feature is A=0/B=1, and B-first when it is A=1/B=0. The returned value is mean P(A) on A-first trials minus mean P(A) on B-first trials. The advocated theory predicts a large positive per-subject contrast, because choices follow the first clean 0 despite the raw tally tie. The competing stochastic-inclusion theory mixes later cues into the tally and predicts a much smaller or reversed first-cue contrast. Each subject has 48 A-first and 48 B-first trials, so the contrast is tightly estimated within subject.

Source:
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


### RESULTS
- Predicted under pi_4_1 (simulated): 0.4188 (var=0.0158)
- Predicted under pi_5 (simulated): 0.2196 (var=0.0342)
- Observed on real data: 0.3050 (var=0.0605)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=11):** [1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.62, 0.6, 0.7, 0.61, 0.61]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 4: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** All six pairs have a raw inverse tally of +2 or -2, so the competing theory is forced to use its mean-centered validity-weighted tally rather than a raw tie-break. The validities are deliberately skewed: features 0-5 are high (1.00 to 0.95), but features 3-5 are tied on every trial and act as high-validity anchors that raise the validity mean to about 0.795. The five low-validity features 6-10 (0.62, 0.60, 0.58, 0.56, 0.54) actually do the discriminating work. At median advocated parameters alpha=7 and gamma=-3.5, inclusion probabilities are about 0.97 for the high cues and 0.70, 0.67, 0.64, 0.60, 0.57 for the low cues. In the competing theory the tied high anchors lower the mean-centered weights of the low cues to about 0.72 to 0.59, while high-cue weights are about 1.33. For the raw +2 pairs, a low-validity majority of 3, 4, or 5 A-clean cues is opposed by 1, 2, or 3 high-validity B-clean cues. The competing weighted score goes from +0.73 through -0.08 to -0.66, so p(A) drops from about 0.64 to 0.49 to 0.37. The advocated expected included tally stays positive at about 1.03, 0.54, and 0.27, so advised p(A) stays at about 0.84, 0.73, and 0.62. The raw -2 mirrors reverse the signs and create a crossover: at the 4- and 5-low-cue levels the two theories make opposite directional predictions. The key manipulation is the tied high anchors, which change the competing mean-centered weights without changing the advocated inclusion tally. Six unique pairs repeated K=16 gives 96 trials.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
For each subject compute q, the proportion of trials on which the option with the larger number of inverse clean-cue features is chosen, pooling all six displayed stimulus types. The returned value is the average across subjects of q squared minus q. A simple mean q separates the two theories only weakly and has large between-subject variance. The quadratic form q squared minus q is nearly constant across subjects because it is a negative consistency term, q minus q squared, so it is much tighter per subject. It still detects the difference in low-validity-majority choice propensity because the advocated inclusion-tally mechanism produces both a higher mean low-cue choice rate and greater subject heterogeneity than the validity-weighted competitor, shifting the average of this nonlinear consistency score in opposite directions for the two theories.

Source:
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

### RESULTS
- Predicted under pi_4_1 (simulated): -0.2374 (var=0.0002)
- Predicted under pi_5 (simulated): -0.2057 (var=0.0022)
- Observed on real data: -0.1667 (var=0.0053)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
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
**Predicted under pi_4_1:** -0.5442 (var=0.0102)
**Predicted under pi_5:** -0.5308 (var=0.0142)

### Experiment 4
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
**Predicted under pi_4_1:** -0.6038 (var=0.0238)
**Predicted under pi_5:** -0.5938 (var=0.0426)

### Experiment 5
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
**Predicted under pi_4_1:** -0.4029 (var=0.0226)
**Predicted under pi_5:** -0.4596 (var=0.0245)

### Experiment 6
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
**Predicted under pi_4_1:** 0.5069 (var=0.0237)
**Predicted under pi_5:** 0.4237 (var=0.0670)

### Experiment 7
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
**Predicted under pi_4_1:** 0.2446 (var=0.0156)
**Predicted under pi_5:** 0.3858 (var=0.0225)

### Experiment 8
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
**Predicted under pi_4_1:** 0.1075 (var=0.0264)
**Predicted under pi_5:** 0.1383 (var=0.0736)

### Experiment 9
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
**Predicted under pi_4_1:** -0.0400 (var=0.0672)
**Predicted under pi_5:** 0.3325 (var=0.0612)

### Experiment 10
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
**Predicted under pi_4_1:** 0.5012 (var=0.5143)
**Predicted under pi_5:** 1.3512 (var=0.8533)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all ten experiments, pi_4_1 achieves slightly better point estimates on several older experiments (E4-E9), but this is offset by a structural failure: it systematically underestimates between-subject variance (e.g., E7 predicted var 0.0156 vs observed 0.1584; E8 0.0264 vs 0.4244; E10 0.5143 vs 2.8518). Its deterministic validity-ordered tie-break is too strong in the diagnostic E1 (0.4188 vs observed 0.3050), and its weak validity-weighted tally is too weak in E2 (-0.2374 vs -0.1667). pi_5 captures variance much better and wins the two newest experiments (E1, E2) as well as E3 and E10, but it badly overpredicts the lower-cue override effect in E9 (0.3325 vs 0.0725) and also overshoots E7 and E8. The observed pattern across experiments is best described as strong pooled inverse valence, moderate first-cue tie-breaking, near-zero top/lower-cue interaction effects, and very large between-subject heterogeneity. Neither theory contains a mixture of valence orientations or near-deterministic inclusion of high-validity cues, both of which seem necessary to explain this pattern. pi_4_1 is the more degenerate theory because its deterministic all-cue tie-break cannot generate the observed heterogeneity and fails the newer diagnostic experiments, so it should be replaced by a new theory.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a valence-mixture validity-gated tally theory. Each subject has a latent valence orientation: with probability about 0.8 they treat 1 as a defect and 0 as clean (inverse valence), otherwise they use the opposite positive valence. Within a trial, each feature j enters the tally with probability logistic(alpha * validity_j + gamma); calibrate alpha and gamma so the highest-validity cues are included with probability >0.95, which should produce the near-zero top/lower-cue interaction in E9 and moderate tie-break contrast in E1, while lower-validity cues are included with moderate probability so their majorities can still move the tally in E2 and E10. The included tally drives a noisy softmax with subject-specific beta_tally. Exact zero tallies are broken by scanning included cues in descending validity order and choosing the valence-consistent cue (0 for inverse subjects, 1 for positive subjects) on the first discriminator; if none discriminates, guess. Subject-level orientation, alpha, gamma, beta_tally, beta_tiebreak, and epsilon should be drawn from heavy-tailed or logit-normal distributions to reproduce the large observed between-subject variances. This differs from pi_4_1 by replacing deterministic all-cue tie-breaking with probabilistic inclusion and adding valence mixture; it differs from pi_5 by adding valence mixture and making high-validity cue inclusion nearly deterministic rather than merely high probability."
}
```

## Usage

```json
{
  "input_tokens": 14356,
  "output_tokens": 18920,
  "reasoning_tokens": 16462,
  "provider": "CoreWeave"
}
```
