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
**Validities (n_features=8):** [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  trial 3: A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0, 1]
  trial 4: A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0]
  trial 5: A=[1, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 6: A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1, 1]

**Rationale:** This design quantitatively dissociates the pure Weighted Additive (WADD) model (Advocated) from the Mixture model (Competing) by exploiting a structural difference in how they scale evidence. The Competing model normalizes the WADD score difference by the sum of all subjective weights before applying the inverse temperature 'beta'. By using 8 features with high validities, the sum of weights becomes very large (e.g., > 6.5). The pure WADD model does not normalize by the sum of weights. Consequently, for trials where the unnormalized score difference is moderate but the total sum of weights is large, the pure WADD model can easily predict highly deterministic choices (using its beta parameter up to 5.0). In contrast, the Competing model divides this score difference by ~6.5, shrinking it so much that even with its maximum beta of 10.0, it is mathematically incapable of predicting the same level of determinism. By including trials where Tallying predicts a tie (further drawing the Mixture model toward 0.5) alongside trials with varying WADD score differences, we create a regime where the Advocated model can capture high choice probabilities that the Competing model strictly underpredicts.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People make choices by computing a weighted sum of all available features for each option, but the weights they use are a non-linear transformation of the objective cue validities. By exponentiating the validities with a free parameter gamma, the strategy can smoothly interpolate between equal weighting (Tallying, gamma=0) and a strong reliance on the most valid cues (approximating Take The Best, gamma > 1). The choice is then made probabilistically based on the difference between the options' weighted sums, with a bounded inverse temperature beta to allow for more stochastic choices.

**Parameters:**
- beta: [0.01, 5.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 5.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) state; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Non-linear scaling of validities
    subjective_weights = val ** gamma
    
    # Calculate weighted sums for both options
    a, b = stim[0], stim[1]
    score_a = np.sum(a * subjective_weights)
    score_b = np.sum(b * subjective_weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate response noise (lapse rate)
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Decision makers employ a dual-process or strategy mixture approach when evaluating multi-attribute options. Rather than relying entirely on a single strategy, choices are generated by a probabilistic mixture of a simple, unweighted Tallying heuristic (which counts the number of strictly winning features) and a compensatory Weighted Additive (WADD) strategy (which integrates all features weighted by their subjective validities). To ensure equitable application of choice determinism, the evidence scores for both strategies are normalized to a common [0, 1] scale before applying a shared inverse temperature parameter. The mixture parameter 'alpha' dictates the reliance on Tallying versus WADD, allowing the model to capture exact chance-level responding in scenarios where features tie while maintaining sensitivity to cue validities in general.

**Parameters:**
- beta: [0.01, 10.0]
- gamma: [0.0, 5.0]
- alpha: [0.0, 1.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    alpha = float(parameters["alpha"])
    epsilon = float(parameters["epsilon"])
    
    # WADD Component: Weighted sum using non-linearly scaled validities, normalized to [0, 1]
    subjective_weights = val ** gamma
    sum_weights = np.sum(subjective_weights)
    score_a_wadd = np.sum(a * subjective_weights) / sum_weights
    score_b_wadd = np.sum(b * subjective_weights) / sum_weights
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    z_wadd = beta * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # Tallying Component: Count of strict feature-wise wins, normalized to [0, 1]
    a_wins = float(np.sum(a > b)) / n_features
    b_wins = float(np.sum(b > a)) / n_features
    scores_tally = np.array([a_wins, b_wins])
    
    z_tally = beta * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # Mixture of the two strategies
    p_mixed = alpha * p_tally + (1.0 - alpha) * p_wadd
    
    # Incorporate response noise (lapse rate)
    return (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
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
[0] rationale: Measures the proportion of times subjects choose Option A on trials where Tallying predicts a strict tie (Trials 1 and 7). The pure WADD model strongly favors Option A on these trials due to cue validity differences. The Mixture model, however, is forced to pull its prediction toward 0.5 on these trials because the Tallying component predicts a tie, leading to a lower overall probability of choosing A.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def is_tie_trial(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == (1, 0, 0, 0, 0) and b == (0, 1, 0, 0, 0):
            return True
        if a == (1, 0, 1, 0, 0) and b == (0, 1, 0, 1, 0):
            return True
        return False
        
    mask = data.apply(is_tie_trial, axis=1)
    tie_data = data[mask]
    
    if len(tie_data) == 0:
        return 0.5
        
    return float((tie_data['response'] == 0).mean())
outcome: self_sim=0.5950 (var=0.0129) adversary_sim=0.5650 (var=0.0148) welch_t=+0.902 p=0.3716 (N=25, alpha=0.01) -> reject

[1] rationale: This metric calculates the proportion of choices that align with the Tallying heuristic (choosing the option with more winning features) on trials where Tallying does not predict a tie. The Competing Theory (Mixture model) explicitly incorporates Tallying into its decision process, so it should produce a higher proportion of Tallying-consistent choices. The Advocated Theory (pure WADD) scales validities non-linearly, which often leads it to strongly favor the option with the highest validity cue even when it loses on a sheer count of features, resulting in a lower score on this metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    def tally_match(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        
        if a_wins > b_wins:
            return 1.0 if row['response'] == 0 else 0.0
        elif b_wins > a_wins:
            return 1.0 if row['response'] == 1 else 0.0
        else:
            return np.nan
            
    matches = data.apply(tally_match, axis=1)
    return float(matches.dropna().mean())
outcome: self_sim=0.5614 (var=0.0195) adversary_sim=0.6125 (var=0.0165) welch_t=-1.347 p=0.1844 (N=25, alpha=0.01) -> reject

[2] rationale: This metric calculates the difference in choice extremeness (absolute deviation from 0.5 probability) between trials where Tallying predicts a tie and trials where Tallying strongly favors one option. Under the Mixture model, Tallying ties pull the predicted choice probability strongly toward 0.5, resulting in lower extremeness on tie trials compared to non-tie trials (yielding a negative metric value). Under the Pure WADD model, tie trials actually feature strong, unambiguous cue validity differences (e.g., cue 1 vs cue 2), leading to highly extreme choices, while non-tie trials often pit multiple weaker cues against a stronger one, leading to less extreme choices (yielding a positive metric value).
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Create a hashable trial signature
    data['trial_sig'] = data.apply(lambda row: (tuple(row['option_a_ratings']), tuple(row['option_b_ratings'])), axis=1)
    
    # Trials where Tallying predicts a strict tie
    tie_sigs = set([
        ((1, 0, 0, 0, 0), (0, 1, 0, 0, 0)),
        ((1, 0, 1, 0, 0), (0, 1, 0, 1, 0))
    ])
    
    def subject_score(sub_df):
        # P(Choose A) for each trial type (response == 0 means A was chosen)
        p_a_by_trial = sub_df.groupby('trial_sig')['response'].apply(lambda x: 1.0 - x.mean())
        
        ties_dev = []
        nonties_dev = []
        for sig, p_a in p_a_by_trial.items():
            dev = abs(p_a - 0.5)
            if sig in tie_sigs:
                ties_dev.append(dev)
            else:
                nonties_dev.append(dev)
                
        if not ties_dev or not nonties_dev:
            return np.nan
            
        return np.mean(ties_dev) - np.mean(nonties_dev)
        
    scores = data.groupby('subject_id').apply(subject_score).dropna()
    if len(scores) == 0:
        return 0.0
    return float(scores.mean())
outcome: self_sim=-0.0569 (var=0.0129) adversary_sim=-0.0453 (var=0.0101) welch_t=-0.385 p=0.7021 (N=25, alpha=0.01) -> reject

[3] rationale: This metric calculates the across-trial variance in a subject's choice probabilities (i.e., how extreme their average preferences are across the 8 distinct trial types). The Advocated Theory (pure WADD) allows for highly deterministic choices because the non-linear scaling of validities can create massive score disparities that push choice probabilities toward 0 or 1. The Competing Theory (Mixture model) structurally compresses choice probabilities toward 0.5 on most trials in this design because the Tallying component and the WADD component are pitted against each other, forcing a compromise that dilutes extremeness. Thus, pure WADD will exhibit a significantly higher variance in choice probabilities across trials than the Mixture model.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Project ratings to a hashable string representation to group by trial type
    data['trial_sig'] = data.apply(lambda row: ''.join(map(str, row['option_a_ratings'])) + '_' + ''.join(map(str, row['option_b_ratings'])), axis=1)
    
    def subject_extremity(sub_df):
        # Calculate the choice probability for Option A for each unique trial type
        p_a = sub_df.groupby('trial_sig')['response'].apply(lambda x: (x == 0).mean())
        # Measure how extreme the probabilities are by taking their variance
        return np.var(p_a)
        
    # Compute the variance of choice probabilities for each subject
    subj_vars = data.groupby('subject_id').apply(subject_extremity).dropna()
    
    if len(subj_vars) == 0:
        return 0.0
        
    return float(subj_vars.mean())
outcome: self_sim=0.0355 (var=0.0006) adversary_sim=0.0381 (var=0.0006) welch_t=-0.372 p=0.7119 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data['is_t1'] = data.apply(\n        lambda row: tuple(row['option_a_ratings']) == (1, 1, 1, 1, 0, 0, 0, 0) and \n                    tuple(row['option_b_ratings']) == (0, 0, 0, 0, 1, 1, 1, 1),\n        axis=1\n    )\n    t1_data = data[data['is_t1']]\n    if len(t1_data) == 0:\n        return 0.5\n    return float((t1_data['response'] == 0).mean())",
  "rationale": "Measures the choice probability of Option A on Trial 1. On this trial, the pure WADD model (Advocated) will predict a highly deterministic choice for Option A because the unnormalized score difference is easily scaled by its beta parameter to produce a probability close to 1.0. In contrast, the Mixture model (Competing) strictly normalizes the score difference by the sum of all weights (which is large for 8 high-validity features), shrinking the effective difference so much that even its maximum beta cannot push the choice probability as high. Additionally, Tallying predicts a tie on this trial, further anchoring the Mixture model's prediction toward 0.5."
}
```

## Usage

```json
{
  "prompt_token_count": 5657,
  "candidates_token_count": 327,
  "total_token_count": 8948
}
```
