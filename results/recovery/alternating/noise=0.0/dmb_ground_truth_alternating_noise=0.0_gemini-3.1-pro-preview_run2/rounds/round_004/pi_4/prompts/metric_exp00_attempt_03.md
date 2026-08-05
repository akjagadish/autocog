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
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 7: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]

**Rationale:** To uniquely validate TTB with High Noise over Contrarian WADD, we exploit the fact that TTB is strictly lexicographic, whereas Contrarian WADD computes a weighted sum (with a potentially negative alpha). By keeping the highest-validity cue fixed in favor of one option while drastically varying the remaining cues, we create trials where the WADD score difference radically changes sign. If subjects consistently show a weak preference for the option favored by the highest-validity cue (e.g., Option A in Trial 1 and Trial 2), Contrarian WADD cannot fit this behavior: a positive alpha would fail on Trial 1 (where B has a much higher weighted sum), and a negative alpha would fail on Trial 2 (where A has a much higher weighted sum). Thus, constant weak preference for the top cue strictly falsifies Contrarian WADD while perfectly matching TTB with High Noise.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Take-The-Best (TTB) with High Noise: Decision makers employ a lexicographic heuristic, searching through cues in order of descending validity. They stop at the first cue that discriminates between the two options and choose the option with the higher value on that cue. However, to accommodate the empirical observation that agreement with any deterministic strategy hovers around 50%, the model incorporates a very high lapse rate (epsilon) and a low softmax inverse temperature (beta). This restricts the model to primarily exhibit random guessing, with only a weak TTB signal, matching the high degree of noise in the observed data.

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.8, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Sort cues by descending validity; stable sort handles ties
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    winner = None
    
    # Lexicographic search
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        # No cue discriminates, guess uniformly
        return np.array([0.5, 0.5])
        
    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over binary TTB scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Contrarian WADD: Decision makers evaluate options by computing a weighted sum of their features, but they may distrust the provided expert ratings or view them as added complexity. Thus, they apply a scaling factor to the validities that can be negative, leading to an 'Anti-Tallying' or contrarian preference for options with lower scores. This weak contrarian signal is obscured by a very high rate of random guessing (lapse rate).

**Parameters:**
- alpha: [-2.0, 1.0]
- beta: [0.0, 0.5]
- epsilon: [0.8, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Contrarian WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
        
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Compute scores with the alpha scaling factor (which can be negative)
    score_a = np.dot(stim[0], val) * alpha
    score_b = np.dot(stim[1], val) * alpha
    scores = np.array([score_a, score_b])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
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
[0] rationale: Under Take-The-Best (TTB), the decision is entirely determined by the first discriminating cue. Since cue 0 discriminates in every single trial, TTB has a constant expected probability of choosing the option favored by cue 0 across all 8 trial types. Consequently, the variance of this choice rate across trial types will be near zero (reflecting only binomial sampling noise). In contrast, Contrarian WADD computes a weighted sum across all features. Because the feature values of cues 1-4 vary wildly across the 8 trial types, the WADD difference fluctuates significantly (sometimes strongly favoring the cue 0 option, sometimes strongly opposing it). Thus, for Contrarian WADD, the probability of choosing the cue-0-favored option will vary widely across the 8 trial types, resulting in a substantially higher variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # The highest validity cue is cue 0.
    # In all trial pairs, cue 0 is exactly 1 for one option and 0 for the other.
    # We determine if the subject chose the option where cue 0 == 1.
    cue0_A = data['option_a_ratings'].apply(lambda x: x[0])
    chose_cue0 = ((cue0_A == 1) & (data['response'] == 0)) | ((cue0_A == 0) & (data['response'] == 1))
    
    # Create a string representation of the trial to use as a hashable group key
    trial_id = data['option_a_ratings'].apply(lambda x: "".join(map(str, x))) + "_" + \
               data['option_b_ratings'].apply(lambda x: "".join(map(str, x)))
               
    # Create a temporary dataframe
    df = pd.DataFrame({'trial_id': trial_id, 'chose_cue0': chose_cue0})
    
    # Calculate the mean rate of choosing the cue-0-favored option for each trial type
    trial_means = df.groupby('trial_id')['chose_cue0'].mean()
    
    # Return the variance of these choice rates across the 8 unique trial types
    return float(trial_means.var())
outcome: self_sim=0.0002 (var=0.0001) adversary_sim=0.0005 (var=0.0001) welch_t=-0.117 p=0.9074 (N=25, alpha=0.01) -> reject

[1] rationale: Under Take-The-Best (TTB), the decision is driven by the most valid cue (Cue 0), so the probability of choosing the option favored by Cue 0 is always > 0.5, regardless of the other cues. Thus, the deviation from 0.5 is positive for both Set 1 (opposing cues) and Set 2 (aligned cues), yielding a positive product. Under Contrarian WADD, the model evaluates the weighted sum. Because alpha can be positive or negative, the model will strongly favor the Cue 0 option in one set and strongly oppose it in the other (e.g., if alpha > 0, it prefers the Cue 0 option in Set 2 but opposes it in Set 1; if alpha < 0, it prefers the Cue 0 option in Set 1 but opposes it in Set 2). Therefore, for WADD, the deviations from 0.5 will have opposite signs, consistently yielding a negative product.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Extract cue 0 for A and B
    cue0_a = data['option_a_ratings'].apply(lambda x: x[0])
    
    # Identify if subject chose the option with cue 0 = 1
    chose_cue0 = ((cue0_a == 1) & (data['response'] == 0)) | ((cue0_a == 0) & (data['response'] == 1))
    
    # Sum of features to identify the trial types
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Set 1 (Opposing): The option with Cue 0 has no other cues (sum=1), and the other option has all other cues (sum=4).
    # This corresponds to Trial 1 and Trial 3.
    is_set1 = ((sum_a == 1) & (cue0_a == 1) & (sum_b == 4)) | ((sum_b == 1) & (cue0_a == 0) & (sum_a == 4))
    
    # Set 2 (Aligned): The option with Cue 0 has all cues (sum=5), and the other option has no cues (sum=0).
    # This corresponds to Trial 2 and Trial 4.
    is_set2 = ((sum_a == 5) & (cue0_a == 1) & (sum_b == 0)) | ((sum_b == 5) & (cue0_a == 0) & (sum_a == 0))
    
    if is_set1.sum() == 0 or is_set2.sum() == 0:
        return 0.0
        
    p_set1 = chose_cue0[is_set1].mean()
    p_set2 = chose_cue0[is_set2].mean()
    
    # The metric is the product of the deviations from 0.5
    return float((p_set1 - 0.5) * (p_set2 - 0.5))
outcome: self_sim=0.0000 (var=0.0001) adversary_sim=-0.0006 (var=0.0001) welch_t=+0.211 p=0.8336 (N=25, alpha=0.01) -> reject

[2] rationale: Under Take-The-Best (TTB), the decision is solely determined by the first discriminating cue. Since Cue 0 discriminates in every single trial, the true probability of choosing the Cue-0-favored option is constant across all 8 trial types. Under Contrarian WADD, the choice depends on the weighted sum of all cues, which fluctuates wildly across the 8 trial types, meaning the true probability varies. A naive variance of choice rates across trial types is overwhelmed by binomial sampling noise (due to the high lapse rate). However, by splitting the trials into two independent halves and computing the covariance of the choice rates across the 8 trial types, we obtain an unbiased estimator of the true variance. For TTB, this split-half covariance is expected to be exactly 0. For Contrarian WADD, it will be strictly positive. This cleanly separates the models without being drowned out by the binomial noise.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Identify if the subject chose the option favored by the most valid cue (Cue 0)
    cue0_a = data['option_a_ratings'].apply(lambda x: x[0])
    chose_cue0 = ((cue0_a == 1) & (data['response'] == 0)) | ((cue0_a == 0) & (data['response'] == 1))
    
    # Create a unique string identifier for each trial type
    trial_id = data['option_a_ratings'].apply(lambda x: "".join(map(str, x))) + "_" + \
               data['option_b_ratings'].apply(lambda x: "".join(map(str, x)))
               
    df = pd.DataFrame({
        'subject_id': data['subject_id'],
        'trial_id': trial_id,
        'chose_cue0': chose_cue0
    })
    
    # To remove binomial noise from the variance estimate, we use a split-half covariance approach.
    # We split the 12 repetitions of each trial type into two halves (even and odd reps).
    df['rep'] = df.groupby(['subject_id', 'trial_id']).cumcount()
    df['half'] = df['rep'] % 2
    
    # Calculate the choice rate for the Cue 0 option in each half
    means = df.groupby(['subject_id', 'trial_id', 'half'])['chose_cue0'].mean().unstack('half')
    
    def compute_cov(sub_df):
        # We need at least 2 trial types and no missing data to compute covariance
        if sub_df.shape[0] < 2 or sub_df[0].isnull().any() or sub_df[1].isnull().any():
            return 0.0
        x = sub_df[0].values
        y = sub_df[1].values
        # Return the sample covariance between the two halves across the 8 trial types
        return float(np.cov(x, y)[0, 1])
        
    subject_covs = means.groupby('subject_id').apply(compute_cov)
    return float(subject_covs.mean())
outcome: self_sim=-0.0001 (var=0.0002) adversary_sim=-0.0011 (var=0.0002) welch_t=+0.273 p=0.7859 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n    \n    # Identify if the subject chose the option favored by the most valid cue (Cue 0)\n    cue0_a = data['option_a_ratings'].apply(lambda x: x[0])\n    chose_cue0 = ((cue0_a == 1) & (data['response'] == 0)) | ((cue0_a == 0) & (data['response'] == 1))\n    \n    # Calculate the sum of the remaining cues (Cues 1-4) for both options\n    sum14_a = data['option_a_ratings'].apply(lambda x: sum(x[1:]))\n    sum14_b = data['option_b_ratings'].apply(lambda x: sum(x[1:]))\n    \n    # Group trials into two sets based on whether Cues 1-4 ALIGN with or OPPOSE Cue 0\n    # Aligned: The option favored by Cue 0 also has a higher sum of Cues 1-4\n    align_a = (cue0_a == 1) & (sum14_a > sum14_b)\n    align_b = (cue0_a == 0) & (sum14_b > sum14_a)\n    is_aligned = align_a | align_b\n    \n    # Opposed: The option favored by Cue 0 has a lower sum of Cues 1-4\n    oppose_a = (cue0_a == 1) & (sum14_a < sum14_b)\n    oppose_b = (cue0_a == 0) & (sum14_b < sum14_a)\n    is_opposed = oppose_a | oppose_b\n    \n    # Compute the choice rate for the Cue-0-favored option in both sets\n    p_align = chose_cue0[is_aligned].mean() if is_aligned.sum() > 0 else 0.5\n    p_oppose = chose_cue0[is_opposed].mean() if is_opposed.sum() > 0 else 0.5\n    \n    # The metric is the difference in choice rates between Aligned and Opposed trials\n    return float(p_align - p_oppose)\n",
  "rationale": "Under Take-The-Best (TTB), the decision is strictly lexicographic and driven entirely by the first discriminating cue (Cue 0). Because Cue 0 discriminates in every trial, TTB ignores Cues 1-4 entirely. Thus, the probability of choosing the option favored by Cue 0 is identical whether Cues 1-4 align with it or oppose it, yielding an expected difference of exactly 0. Under Contrarian WADD, the decision is based on a weighted sum of all cues, scaled by an alpha parameter that is predominantly negative (mean = -0.5). Because of this contrarian tendency, WADD will generally avoid the option with the higher overall sum. In 'Aligned' trials, the Cue 0 option has a much higher sum, so WADD avoids it. In 'Opposed' trials, the Cue 0 option has a lower sum, so WADD prefers it. This creates a systematically negative difference (p_align - p_oppose) for WADD, cleanly separating the two theories with a simple linear contrast that avoids variance-inflation issues."
}
```

## Usage

```json
{
  "prompt_token_count": 5527,
  "candidates_token_count": 806,
  "total_token_count": 33131
}
```
