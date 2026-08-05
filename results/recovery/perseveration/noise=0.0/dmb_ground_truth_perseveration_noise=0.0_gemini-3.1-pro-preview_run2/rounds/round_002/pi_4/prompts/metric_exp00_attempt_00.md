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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 6: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 7: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Rationale:** This design aims to completely dissociate the Random Guessing / High-Noise Tallying theory from the Weighted Additive (WADD) theory by including trials that vary from extreme evidence disparities to complex trade-offs. WADD predicts that choice probabilities will systematically track the weighted sum of validities (transformed by gamma), leading to highly deterministic choices (probabilities near 1.0 or 0.0) when one option dominates (e.g., all 1s vs all 0s) or when the weighted evidence strongly favors one side. In stark contrast, the advocated Random Guessing theory predicts that subjects largely ignore the validities and feature values due to high noise and lapse rates, resulting in choice proportions that remain universally close to 50/50 across all trials, with only a marginal pull toward the option with more unweighted feature wins.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Random Guessing / High-Noise Tallying: Due to the lack of trial-by-trial feedback and potentially low engagement, subjects largely ignore cue validities and feature values. Instead, they make choices that are nearly indistinguishable from uniform random guessing, which can be modeled as a simple unweighted tally of features heavily diluted by extreme decision noise and high lapse rates. This explains the ~0.5 agreement with Take The Best, ~0.44 agreement with Tallying, and the near-zero variance in choice proportions across different trial types.

**Parameters:**
- beta: [0.0, 0.05]
- epsilon: [0.8, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Unweighted tally of winning features
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    # Blend with uniform guessing based on epsilon
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
**Description:** People use a Weighted Additive (WADD) strategy when comparing two options, but they subjectively scale the stated cue validities. By applying a non-linear transformation to the validities (exponentiating them by a parameter gamma), decision-makers can amplify the differences between cues (mimicking Take The Best) or compress them (mimicking Tallying). Expanding the range of gamma allows for extreme weight disparities, accommodating individuals who rely heavily on the most valid cue.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 20.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    gamma = float(parameters["gamma"])
    val = np.asarray(parameters["validities"], dtype=float) ** gamma
    
    # Compute the weighted sum of features for each option
    ev_a = np.sum(val * a)
    ev_b = np.sum(val * b)
    scores = np.array([ev_a, ev_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n    \n    # Create a hashable string key for each unique trial type\n    a_str = data['option_a_ratings'].apply(lambda x: \"\".join(map(str, x)))\n    b_str = data['option_b_ratings'].apply(lambda x: \"\".join(map(str, x)))\n    \n    df = data.copy()\n    df['trial_key'] = a_str + \"_\" + b_str\n    \n    # Calculate the proportion of choosing option B (response = 1) for each subject and trial type\n    subject_trial_means = df.groupby(['subject_id', 'trial_key'])['response'].mean()\n    \n    # Calculate the absolute difference from 0.5 (extremeness of choice probability)\n    abs_devs = (subject_trial_means - 0.5).abs()\n    \n    # Return the average extremeness across all subjects and trial types\n    return float(abs_devs.mean())",
  "rationale": "The Weighted Additive (WADD) theory predicts highly deterministic choices (probabilities near 0 or 1) because subjects consistently apply subjective weights to the cues, leading to strong preferences on most trials. The Random Guessing theory, conversely, predicts that choice proportions will hover closely around 0.5 due to high noise and lapse rates. By calculating the absolute deviation of each subject's choice proportion from 0.5 for each trial type and averaging them, we directly quantify the 'extremeness' of their choices. WADD will yield high values (close to 0.5), whereas Random Guessing will yield low values (driven mostly by finite-sample binomial variance, around 0.22 for 12 repetitions)."
}
```

## Usage

```json
{
  "prompt_token_count": 3150,
  "candidates_token_count": 427,
  "total_token_count": 5164
}
```
