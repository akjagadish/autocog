# metric_exp01_attempt_00

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
**Validities (n_features=4):** [0.9, 0.8, 0.6, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  trial 4: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 5: A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  trial 6: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]

**Rationale:** To quantitatively dissociate Probabilistic Cue Sampling (PCS) from the Weighted Additive rule (WADD), this design exploits the core difference in their integration mechanisms: PCS uses a softmax over validities to sample a single feature, amplifying the influence of the highest-validity cue, whereas WADD linearly integrates all features. We use four cues with linearly spaced validities. We include 'compensatory' trials where one option possesses the single highest-validity cue but the other possesses all remaining cues (PCS prefers the former, WADD strongly prefers the latter). We also include 'tie-breaker' trials where the sum of validities is equal across both options (yielding a tie in WADD), but the convexity of the softmax in PCS leads to a strong preference for the option with the most valid cue.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Cue Sampling with Softmax Validities: Decision-makers evaluate options by sampling a single feature on each trial. The probability of sampling a feature is determined by a softmax function over the objective validities scaled by a sensitivity parameter gamma. The option that is superior on the sampled feature is chosen; ties result in guessing. This predicts intermediate choice proportions when validities are mixed, avoiding deterministic winner-takes-all behavior.

**Parameters:**
- gamma: [0.0, 2.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])

    # Compute subjective validities using softmax for sampling probabilities
    z = gamma * validities
    z = z - np.max(z)  # numerical stability
    e = np.exp(z)
    p_feat = e / np.sum(e)

    a, b = stim[0], stim[1]

    # Identify wins and ties for each feature
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    ties = (a == b).astype(float)

    # Analytically marginalize over the sampled feature:
    # P(Choose A) = sum_i P(sample i) * P(Choose A | sample i)
    # P(Choose A | sample i) = 1 if a_i > b_i, 0 if b_i > a_i, 0.5 if a_i == b_i
    p_A = np.sum(p_feat * (a_wins + 0.5 * ties))
    p_B = np.sum(p_feat * (b_wins + 0.5 * ties))

    p_core = np.array([p_A, p_B])

    # Apply uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson
    # 1993). Stimulus is the pair of option feature vectors for the
    # current trial: array-like of shape (2, n_features), row 0 =
    # option A, row 1 = option B. Each option's score is the dot
    # product of its feature vector with the (subjective) validity
    # weights. History is ignored: the choice depends only on the
    # current stimulus.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"weights length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ (validities * w)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax (standard choice-with-lapse
    # formulation; epsilon=0.5 yields equiprobable options, matching
    # the paper's maximum-noise condition).
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
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
[0] rationale: This metric calculates the proportion of times the subject chooses Option A in trials where Option A is favored by the most valid cue (Feature 1), but Option B is favored by a greater number of less valid cues (meaning the sum of Option B's features is greater than Option A's). WADD is compensatory, so it will consistently favor Option B in these trials because the combined weight of the numerous lower-validity cues outweighs the single best cue. PCS, however, samples cues probabilistically based on a softmax over validities, meaning it will frequently sample the most valid cue and subsequently choose Option A. Thus, PCS will predict a significantly higher choice rate for Option A on these target trials compared to WADD.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_first = data['option_a_ratings'].apply(lambda x: x[0] == 1)
    b_first = data['option_b_ratings'].apply(lambda x: x[0] == 0)
    a_sum = data['option_a_ratings'].apply(lambda x: sum(x))
    b_sum = data['option_b_ratings'].apply(lambda x: sum(x))
    
    is_target = a_first & b_first & (b_sum > a_sum)
    
    if is_target.sum() == 0:
        return 0.5
        
    return float((data.loc[is_target, 'response'] == 0).mean())
outcome: self_sim=0.4354 (var=0.0055) adversary_sim=0.3533 (var=0.0315) welch_t=+2.135 p=0.04048 (N=25, alpha=0.01) -> reject

[1] rationale: Probabilistic Cue Sampling (PCS) is inherently stochastic because it samples a single cue on each trial. Given the validities and the softmax function, the maximum probability of sampling any single cue is relatively low (~0.28 even at max gamma). Consequently, the predicted choice probabilities for PCS are always tightly bounded around 0.5 (e.g., between 0.3 and 0.7). In contrast, the Weighted Additive (WADD) model integrates all cues into a single scalar score for each option. Since WADD can have high inverse temperature (beta up to 20.0), it frequently makes near-deterministic predictions (choice proportions near 0 or 1) for trials where one option has a higher weighted sum. Therefore, the mean squared deviation of trial-level choice proportions from 0.5 will be significantly larger for WADD than for PCS.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Create a hashable trial type identifier from the option ratings
    data['trial_type'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, int(v) for v in x))) + '_' + data['option_b_ratings'].apply(lambda x: ''.join(map(str, int(v) for v in x)))
    
    # Calculate the proportion of times Option A was chosen for each unique trial type
    p_A = data.groupby('trial_type')['response'].apply(lambda x: (x == 0).mean())
    
    # Calculate the mean squared deviation of choice proportions from 0.5
    return float(((p_A - 0.5) ** 2).mean())
outcome: evaluation failed (SyntaxError: Generator expression must be parenthesized (<string>, line 3))

[2] rationale: Probabilistic Cue Sampling (PCS) is inherently stochastic because it samples a single cue on each trial. Given the validities and the softmax function, the maximum probability of sampling any single cue is relatively low (~0.28 even at max gamma). Consequently, the predicted choice probabilities for PCS are always tightly bounded around 0.5 (e.g., between 0.3 and 0.7). In contrast, the Weighted Additive (WADD) model integrates all cues into a single scalar score for each option. Since WADD can have high inverse temperature (beta up to 20.0), it frequently makes near-deterministic predictions (choice proportions near 0 or 1) for trials where one option has a higher weighted sum. Therefore, the mean squared deviation of trial-level choice proportions from 0.5 will be significantly larger for WADD than for PCS.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Create a hashable trial type identifier from the option ratings
    data['trial_type'] = data['option_a_ratings'].apply(tuple)
    
    # Calculate the proportion of times Option A was chosen for each unique trial type
    p_A = data.groupby('trial_type')['response'].apply(lambda x: (x == 0).mean())
    
    # Calculate the mean squared deviation of choice proportions from 0.5
    return float(((p_A - 0.5) ** 2).mean())
outcome: self_sim=0.0054 (var=0.0001) adversary_sim=0.0155 (var=0.0018) welch_t=-1.144 p=0.2622 (N=25, alpha=0.01) -> reject

[3] rationale: This metric isolates Trials 1 and 7, where the option containing the single best cue (cue 1) is heavily outweighed by the sheer number of lower-validity cues in the alternative option. Because WADD integrates all cues using subjective weights, the combined weight of the opposing cues consistently dominates, driving WADD's choice rate for the cue-1 option very low (close to the lapse rate). PCS, however, evaluates options by sampling a single cue. Because cue 1 has the highest validity, it is sampled most frequently (e.g., ~24-28% of the time). Whenever cue 1 is sampled, the cue-1 option is chosen. Thus, PCS guarantees a baseline choice rate for the cue-1 option that is significantly and consistently higher than WADD's prediction. By pooling these two trials (24 trials per subject), the metric achieves a large mean gap while maintaining low between-subject variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Convert feature lists to strings for safe and fast comparison
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Identify Trial 1 and Trial 7 patterns where one option has the highest validity cue
    # but the other option has a strictly greater number of lower-validity cues.
    t1_a, t1_b = '10000', '01111'
    t7_a, t7_b = '10001', '01110'
    
    mask_1_straight = (a_str == t1_a) & (b_str == t1_b)
    mask_1_swapped  = (a_str == t1_b) & (b_str == t1_a)
    mask_7_straight = (a_str == t7_a) & (b_str == t7_b)
    mask_7_swapped  = (a_str == t7_b) & (b_str == t7_a)
    
    valid_trials = mask_1_straight | mask_1_swapped | mask_7_straight | mask_7_swapped
    
    if not valid_trials.any():
        return 0.5
        
    # Check if the subject chose the option containing cue 1
    chose_cue1 = pd.Series(False, index=data.index)
    chose_cue1[mask_1_straight | mask_7_straight] = (data['response'] == 0)
    chose_cue1[mask_1_swapped | mask_7_swapped] = (data['response'] == 1)
    
    return float(chose_cue1[valid_trials].mean())
outcome: self_sim=0.3750 (var=0.0093) adversary_sim=0.2792 (var=0.0411) welch_t=+2.134 p=0.0401 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    \n    t6_a, t6_b = '1100', '0011'\n    \n    mask_straight = (a_str == t6_a) & (b_str == t6_b)\n    mask_swapped = (a_str == t6_b) & (b_str == t6_a)\n    \n    valid_trials = mask_straight | mask_swapped\n    \n    if not valid_trials.any():\n        return 0.5\n        \n    chose_1100 = pd.Series(False, index=data.index)\n    chose_1100[mask_straight] = (data['response'] == 0)\n    chose_1100[mask_swapped] = (data['response'] == 1)\n    \n    return float(chose_1100[valid_trials].mean())",
  "rationale": "This metric measures the choice proportion for the option [1, 1, 0, 0] when paired against [0, 0, 1, 1] (Trial 6). The Weighted Additive (WADD) model strongly prefers [1, 1, 0, 0] because the sum of the top two validities (0.9 + 0.8) heavily outweighs the bottom two (0.6 + 0.5), leading to a near-deterministic choice (close to 1.0, minus lapses). In contrast, Probabilistic Cue Sampling (PCS) is bounded by the softmax probabilities of sampling the individual features. Even at the maximum gamma parameter (2.0), the combined probability of sampling either of the top two cues is only around 64%. Therefore, PCS predicts a much lower, more equivocal choice rate for [1, 1, 0, 0] (~0.5 to 0.64). This trial provides a massive, consistent gap in predicted means while keeping between-subject variance low by focusing on a single, highly discriminatory trial type."
}
```

## Usage

```json
{
  "prompt_token_count": 5428,
  "candidates_token_count": 551,
  "total_token_count": 8177
}
```
