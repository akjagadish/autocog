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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Rationale:** To quantitatively dissociate the advocated Decisiveness-Dependent Strategy Selection model from the competing Tally-plus-Configural WADD Hybrid, we exploit how each model resolves tied tallies. The advocated model resolves ties by strictly falling back to a non-compensatory Take-The-Best (TTB) mechanism, relying solely on the highest-validity discriminating cue and completely ignoring feature configurality. In contrast, the competing model resolves ties using a configural WADD process that applies a symmetrical penalty to clustered (adjacent) features, actively suppressing options with adjacent cues. We designed tied-tally trials where Option A always wins the single highest-validity cue. In Trial 1, Option A's cues are clustered (adjacent) while Option B's are spaced out. In Trial 2, Option A's cues are spaced out while Option B's are clustered. The advocated model predicts identical choice probabilities for Option A across both trials, as it ignores clustering. The competing model is structurally forced to predict that Option A is chosen significantly more often in Trial 2 than in Trial 1, potentially even reversing preference to Option B in Trial 1 due to the clustering penalty. Unequal-tally trials are included to constrain baseline tallying determinism.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Decisiveness-Dependent Strategy Selection with Sharp Transition: Decision-makers probabilistically select between a compensatory Tallying strategy and a non-compensatory Take-The-Best (TTB) strategy on a trial-by-trial basis. The probability of using Tallying is a logistic function of the absolute difference in tally scores between the two options. By strictly constraining the sensitivity (theta) to be positive and the threshold to [0.1, 0.9], the model naturally transitions to a sharp step function where Tallying heavily dominates for decisive tally differences (delta >= 1), while TTB is strictly reserved as a tie-breaker for complex/tied stimuli (delta == 0).

**Parameters:**
- theta: [1.0, 20.0]
- threshold: [0.1, 0.9]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    threshold = float(parameters["threshold"])
    epsilon = float(parameters["epsilon"])
    
    # Tallying prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    delta_tally = abs(a_wins - b_wins)
    
    if a_wins > b_wins:
        p_a_tally = 1.0
    elif b_wins > a_wins:
        p_a_tally = 0.0
    else:
        p_a_tally = 0.5
        
    # Take-The-Best (TTB) prediction
    order = np.argsort(val)[::-1]
    p_a_ttb = 0.5
    for idx in order:
        if a[idx] > b[idx]:
            p_a_ttb = 1.0
            break
        elif b[idx] > a[idx]:
            p_a_ttb = 0.0
            break
            
    # Strategy selection probability
    # Probability of using Tallying depends on the decisiveness of the tally
    exponent = -theta * (delta_tally - threshold)
    exponent = np.clip(exponent, -500.0, 500.0) # Prevent overflow
    p_use_tally = 1.0 / (1.0 + np.exp(exponent))
    
    p_a_core = p_use_tally * p_a_tally + (1.0 - p_use_tally) * p_a_ttb
    p_b_core = 1.0 - p_a_core
    
    p_core = np.array([p_a_core, p_b_core])
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Tally-plus-Configural WADD Hybrid with Symmetrical Clustering Penalty (Active Suppression): Decision-makers primarily rely on a simple tally of positive features. When tallies are tied or close, they break the tie using a weighted additive process that features configural cue processing. Instead of just discounting clustered features, the model symmetrically penalizes ALL features in a cluster, allowing the penalty factor to be negative. This means clustered cues can be actively detrimental, heavily suppressing options with adjacent top cues (e.g., 11000) and strongly favoring options with spaced-out cues (e.g., 10001), perfectly capturing the extreme preference reversals observed in human data.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- theta: [0.1, 3.0]
- lambda_adj: [-1.0, 1.0]
- w_tally: [0.0, 10.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    theta = float(parameters["theta"])
    lambda_adj = float(parameters["lambda_adj"])
    w_tally = float(parameters["w_tally"])
    
    # Non-linear weighting of validities
    w = val ** theta
    w = w / np.sum(w)
    
    def calc_value(x):
        val_x = 0.0
        n = len(x)
        for i in range(n):
            if x[i] == 1:
                # Symmetrical adjacency penalty: penalize if adjacent to ANY other 1
                is_clustered = False
                if i > 0 and x[i-1] == 1:
                    is_clustered = True
                if i < n - 1 and x[i+1] == 1:
                    is_clustered = True
                    
                if is_clustered:
                    val_x += w[i] * lambda_adj
                else:
                    val_x += w[i]
        return val_x
        
    # Score is a hybrid of Tallying and Configural WADD
    score_a = w_tally * np.sum(a) + calc_value(a)
    score_b = w_tally * np.sum(b) + calc_value(b)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data = data.copy()\n    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))\n    \n    # Trial 1: Option A has clustered cues (11000)\n    t1 = data[data['A_str'] == '11000'].groupby('subject_id')['response'].mean()\n    \n    # Trial 2: Option A has spaced cues (10100)\n    t2 = data[data['A_str'] == '10100'].groupby('subject_id')['response'].mean()\n    \n    # Calculate the absolute difference in choice rates between Trial 1 and Trial 2 per subject\n    diff = (t1 - t2).abs()\n    return float(diff.mean())",
  "rationale": "The advocated theory (Decisiveness-Dependent Strategy Selection) predicts that for tied tallies (Trial 1 and Trial 2), the decision strictly falls back to Take-The-Best. Since Option A wins on the highest validity cue in both trials, the choice probability for A should be identical, leaving only binomial noise as the source of any difference. The competing theory (Tally-plus-Configural WADD) applies a symmetrical penalty (or bonus) to clustered features. Because Option A is clustered in Trial 1 and spaced in Trial 2, the competing theory predicts a systematic, large difference in the choice probabilities across the two trials. By computing the absolute difference in the proportion of B choices between Trial 1 and Trial 2 per subject and averaging, we obtain a metric that is near zero for the advocated theory and substantially higher for the competing theory."
}
```

## Usage

```json
{
  "prompt_token_count": 3777,
  "candidates_token_count": 405,
  "total_token_count": 6533
}
```
