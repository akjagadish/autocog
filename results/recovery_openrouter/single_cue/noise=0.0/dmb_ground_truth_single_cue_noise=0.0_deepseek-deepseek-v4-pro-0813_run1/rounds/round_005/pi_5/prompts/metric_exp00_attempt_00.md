# metric_exp00_attempt_00

## System Prompt

You are a psychology researcher proposing a metric in the Decision Making (Binary Features) domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
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



## ADVOCATED THEORY
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


## COMPETING THEORY
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


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
- option_b_ratings: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
- response: 0 if subject chose A, 1 if subject chose B.

## IMPLEMENTATION GUARDRAILS
Any column in the schema above whose description names a list / tuple / np.ndarray (i.e. a per-trial sequence of values) holds non-scalar cells. Those cells are NOT hashable, so operations that hash row values fail with `TypeError: unhashable type: 'list'`. Treating `<seq_col>` as a placeholder for any such sequence-valued column:
- Avoid: `data.groupby('<seq_col>')`, `data['<seq_col>'].value_counts()`,     `data['<seq_col>'].nunique()`, `data['<seq_col>'].unique()` (returns     an object array but downstream `set()` / `in dict` will crash),     `set(data['<seq_col>'])`, `data['<seq_col>'].isin([...])` against list     values, or using a list cell as a dict key.
- If you need a hashable surrogate, project to one first, e.g.:
    - `data['<seq_col>_key'] = data['<seq_col>'].apply(tuple)` then group by `<seq_col>_key`
    - `data['<seq_col>_str'] = data['<seq_col>'].apply(lambda x: ''.join(map(str, x)))`
    Scalar columns (ints, floats, strings like `subject_id`, integer     responses, etc.) hash fine and can be used directly.
- Generator expressions inside function calls like `map()` or `join()` MUST be     parenthesized. For example:
    - WRONG: `map(str, int(v) for v in x)` → SyntaxError
    - RIGHT: `map(str, (int(v) for v in x))` or use a list comp: `[str(int(v)) for v in x]`
- Always verify your code is syntactically valid Python before returning it.

## METRICS YOU ALREADY TRIED AND FAILED ON
Each entry below is a metric you previously proposed in this round that did NOT discriminate the two theories at the human sample size — either it errored, its between-subject variance was unavailable, or Welch's t-test on `(self mean, self var, N)` vs. `(adv mean, adv var, N)` returned p ≥ alpha. The `outcome` line is the simulation result (means, between-subject variances, t-statistic and p-value at the human N) on the same `data_self` / `data_adv` your next metric will be evaluated on. Use the numbers to see where your hypothesised contrast collapsed — small mean gap, large per-subject variance, or both — and propose something qualitatively different. Don't repeat the same idea with cosmetic tweaks.
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "import numpy as np\nimport pandas as pd\n\n\ndef metric(data: pd.DataFrame) -> float:\n    a = data['option_a_ratings']\n    b = data['option_b_ratings']\n\n    # Feature index 0 is the 0.50-validity cue.\n    # Feature indices 1..5 are the five 0.80-validity cues.\n    f1 = a.apply(lambda x: float(x[0])) - b.apply(lambda x: float(x[0]))\n    grp = a.apply(lambda x: float(sum(x[1:6]))) - b.apply(lambda x: float(sum(x[1:6])))\n\n    # High-conflict trials: the lone 0.50 cue points opposite the 0.80 bloc,\n    # and the 0.80 bloc has a net majority of at least 3 (i.e. +/-3 or +/-5).\n    conflict = ((f1 * grp) < 0) & (grp.abs() >= 3)\n\n    if int(conflict.sum()) == 0:\n        return 0.5\n\n    # Follow the lone 0.50 cue: choose A when it favors A, choose B when it favors B.\n    follow_lone = ((f1 > 0) & (data['response'] == 0)) | ((f1 < 0) & (data['response'] == 1))\n\n    return float(follow_lone[conflict].mean())\n",
  "rationale": "This metric isolates the four high-conflict trial types (A k=0, A k=1, B k=4, B k=5) where the lone 0.50-validity cue and the five 0.80-validity cues point in opposite directions, with the 0.80 cue net majority being +/-5 or +/-3. The advocated theory gives normalized decision weight dominantly to the 0.50 cue, so choices should follow it on these trials. The competing theory caps the 0.50 cue and redistributes roughly 0.74 of the weight to the 0.80 cues, so choices should follow the 0.80 bloc instead. Averaging the follow-the-0.50-cue proportion over these 32 trials per subject gives a large mean gap while keeping per-subject variance driven mainly by binomial sampling noise, and it discards the tied 0.90 cues and low-conflict/agreeing trials that dilute the contrast."
}
```

## Usage

```json
{
  "input_tokens": 7115,
  "output_tokens": 17812,
  "reasoning_tokens": 17265,
  "provider": "Baidu"
}
```
