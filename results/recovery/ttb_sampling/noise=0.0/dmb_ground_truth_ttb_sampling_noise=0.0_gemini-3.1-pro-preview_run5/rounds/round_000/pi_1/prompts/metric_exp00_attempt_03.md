# metric_exp00_attempt_03

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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 8: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Rationale:** To dissociate Tallying from WADD, we use a 5-feature design with a steep drop-off in validities. Tallying makes decisions by counting the number of features favoring each option, treating all features equally. WADD, conversely, weights each feature by its validity. By creating trials where one option is favored by a few high-validity features while the other is favored by many low-validity features, we create a sharp dissociation. For example, an option winning on the 3 lowest-validity features but losing on the 2 highest will be strongly preferred by Tallying but strongly rejected by WADD. We also include agreement trials to ensure baseline performance checks.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
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
[0] rationale: Tallying decides based purely on the count of features favoring each option. In both Set 1 (Trials 1 & 2) and Set 2 (Trials 3, 5, 7, 8), the absolute difference in feature counts between options is exactly 1. Thus, Tallying predicts the same accuracy (consistency with the count-favored option) across both sets, yielding a metric difference of 0 in expectation. WADD, however, uses feature validities. In Set 1, the option with fewer features has its 1s on the two most highly valid features, making the weighted sums almost identical (a near-tie for WADD). In Set 2, the weighted sum difference is much larger. Therefore, WADD predicts significantly lower accuracy on Set 1 compared to Set 2, yielding a positive metric value.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    set1_correct = []
    set2_correct = []

    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        sum_a = np.sum(a)
        sum_b = np.sum(b)

        if sum_a == sum_b:
            continue

        tally_favors_a = sum_a > sum_b
        chose_a = (row['response'] == 0)
        is_correct = (tally_favors_a == chose_a)

        abs_diff = abs(sum_a - sum_b)

        if abs_diff == 1:
            # Identify the option with fewer positive features
            sum_min_option = a if sum_a < sum_b else b
            
            # Set 1: The 'weak' option has its 1s exactly on the two most valid features.
            # This creates a near-tie for WADD but a clear win for Tallying.
            if sum_min_option[0] == 1 and sum_min_option[1] == 1 and sum_min_option[2] == 0:
                set1_correct.append(is_correct)
            else:
                # Set 2: The 'weak' option does not perfectly offset the count deficit with validities.
                set2_correct.append(is_correct)

    if not set1_correct or not set2_correct:
        return 0.0

    return float(np.mean(set2_correct) - np.mean(set1_correct))
outcome: self_sim=-0.0017 (var=0.0080) adversary_sim=0.0196 (var=0.1223) welch_t=-0.294 p=0.7707 (N=25, alpha=0.01) -> reject

[1] rationale: Tallying predicts that choice probabilities depend entirely on the difference in feature counts. In this design, there are 6 distinct trial types where the absolute difference in feature counts is exactly 1. For a subject using Tallying, the true probability of choosing the tally-favored option is identical across all 6 of these trial types; any variance in the observed proportions is purely due to binomial sampling noise. In contrast, WADD computes weighted sums using subjective validities. Across these 6 trial types, the difference in WADD scores varies significantly because the specific features contributing to the counts differ. Therefore, a subject using WADD will have widely varying true choice probabilities across these 6 trial types. By calculating the variance of the choice proportions across these specific trial types, we expect a low, stable value for Tallying (reflecting only binomial noise) and a significantly higher value for WADD.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Calculate the sum of features for each option
    a_sum = data['option_a_ratings'].apply(np.sum)
    b_sum = data['option_b_ratings'].apply(np.sum)
    
    # Filter for trials where the absolute difference in tally scores is exactly 1
    mask = abs(a_sum - b_sum) == 1
    subset = data[mask].copy()
    
    if len(subset) == 0:
        return 0.0
        
    # Determine if the subject chose the option favored by Tallying
    subset['tally_favored_chosen'] = ((a_sum > b_sum) & (subset['response'] == 0)) | \
                                     ((b_sum > a_sum) & (subset['response'] == 1))
                                     
    # Create a unique string identifier for each trial type based on the option features
    subset['trial_type'] = subset.apply(
        lambda row: ''.join(map(str, row['option_a_ratings'])) + '_' + ''.join(map(str, row['option_b_ratings'])), 
        axis=1
    )
    
    # Calculate the proportion of times the tally-favored option was chosen for each trial type
    proportions = subset.groupby('trial_type')['tally_favored_chosen'].mean()
    
    if len(proportions) < 2:
        return 0.0
        
    # Return the variance of these proportions across the different trial types
    return float(np.var(proportions, ddof=1))
outcome: self_sim=0.0002 (var=0.0001) adversary_sim=0.0018 (var=0.0041) welch_t=-0.125 p=0.9018 (N=25, alpha=0.01) -> reject

[2] rationale: To cleanly dissociate Tallying from WADD, we leverage the fact that Tallying ignores feature validities and bases its confidence solely on the absolute difference in feature wins. We identify 6 trials where the absolute tally difference is exactly 1 (meaning Tallying predicts identical choice probabilities across all of them). We partition these into trials where Tallying favors Option A (diff = +1) and Option B (diff = -1). Within each partition, the objective WADD score difference varies dramatically. For example, when tally_diff = +1, Trial 1 has a tiny WADD difference (+0.15) while Trials 3 and 8 have large WADD differences (+0.75, +0.55). We compute the difference in the proportion of choosing Option A between the 'High WADD confidence' and 'Low WADD confidence' trials within both the +1 and -1 tally partitions, and subtract them. For Tallying, because the count difference is identical within each partition, this metric perfectly cancels out to 0 (with variance driven only by binomial noise). For WADD, choice probabilities scale with the WADD difference, yielding a strongly positive value.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    tally1_low = []
    tally1_high = []
    tallyM1_low = []
    tallyM1_high = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        tally_diff = a_wins - b_wins
        
        wadd_diff = np.dot(a, validities) - np.dot(b, validities)
        chose_a = 1.0 if row['response'] == 0 else 0.0
        
        if tally_diff == 1:
            if wadd_diff < 0.3:
                tally1_low.append(chose_a)
            elif wadd_diff > 0.4:
                tally1_high.append(chose_a)
        elif tally_diff == -1:
            if wadd_diff > -0.3:
                tallyM1_low.append(chose_a)
            elif wadd_diff < -0.4:
                tallyM1_high.append(chose_a)
                
    if not all([tally1_low, tally1_high, tallyM1_low, tallyM1_high]):
        return 0.0
        
    term1 = np.mean(tally1_high) - np.mean(tally1_low)
    term2 = np.mean(tallyM1_high) - np.mean(tallyM1_low)
    
    return float(term1 - term2)
outcome: self_sim=-0.0033 (var=0.0319) adversary_sim=0.0392 (var=0.4893) welch_t=-0.294 p=0.7707 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Extract features for options A and B\n    # We convert lists to 2D numpy arrays\n    A = np.vstack(data['option_a_ratings'].values)\n    B = np.vstack(data['option_b_ratings'].values)\n    \n    # Design matrix: difference in features between A and B\n    X = A - B\n    \n    # Response vector: 1 if A was chosen, -1 if B was chosen\n    y = np.where(data['response'] == 0, 1.0, -1.0)\n    \n    # Fit an Ordinary Least Squares (OLS) model to estimate the subjective weight of each feature\n    # beta_hat = (X^T X)^-1 X^T y\n    # We use lstsq for numerical stability\n    beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)\n    \n    # For Tallying, all features are treated exactly equally, so the true data-generating weights are identical.\n    # For WADD, features are weighted by subjective validities: w_i * objective_validity_i.\n    # We apply a fixed contrast vector 'k' that sums to 0. \n    # This guarantees that if all estimated weights are equal (Tallying), the dot product is exactly 0.\n    # The vector is constructed to positively weight the high-validity features and negatively weight the low ones,\n    # maximizing the expected value for WADD while keeping variance low.\n    k = np.array([0.3, 0.2, 0.1, -0.1, -0.5])\n    \n    return float(np.dot(k, beta_hat))",
  "rationale": "To cleanly discriminate Tallying from WADD despite the high inter-subject variance caused by WADD's random subjective weights, we estimate the implied linear weights of the 5 features from the subject's choices using OLS. Tallying treats all features interchangeably, meaning the expected estimated weights are identical across all 5 features. By applying a zero-sum contrast vector (0.3, 0.2, 0.1, -0.1, -0.5) to the estimated weights, Tallying mathematically evaluates to 0 in expectation, with minimal variance driven only by response noise. Conversely, WADD scales features by their objective validities (which monotonically decrease from 0.95 to 0.55). The contrast vector is aligned with this decrease, ensuring that WADD produces a robustly positive score across subjects, effectively bypassing the noise introduced by the random subjective weight parameters."
}
```

## Usage

```json
{
  "prompt_token_count": 5774,
  "candidates_token_count": 635,
  "total_token_count": 37863
}
```
