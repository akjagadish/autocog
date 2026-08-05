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

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 5: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 7: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 8: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 9: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 10: A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  trial 11: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  trial 12: A=[1, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]

**Rationale:** By holding both the Take-The-Best (TTB) prediction and the Tallying score difference strictly constant across a sequence of trials, we fix the predicted choice probability of Theory 1 (Constant-Mixture TTB + Tallying) to be perfectly flat. However, by systematically swapping which specific lower-validity cues support each option, we smoothly vary the continuous weighted sum difference. Theory 2 (WADD) is sensitive to the exact validities of the supporting cues and thus predicts a monotonically changing choice probability across the sequence. This provides a stark quantitative dissociation: constant vs. graded choice probabilities.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Constant-Mixture Strategy Selection with dampened Tallying: Decision-makers evaluate options using a probabilistic mixture of a non-compensatory strategy (Take-The-Best) and a compensatory strategy (Tallying). Individuals have a fixed, trait-like probability of using TTB versus Tallying on any given trial, independent of cue depth. To account for zero covariance between tally score differences and choices in certain environments, the temperature parameter for the Tallying component can reach zero, dampening its linear tracking of score differences.

**Parameters:**
- alpha: [0.0, 1.0]
- tau: [0.0, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Determine Take-The-Best (TTB) prediction
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = -1
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        p_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        p_ttb = np.array([0.0, 1.0])
    else:
        p_ttb = np.array([0.5, 0.5])
        
    # Determine Tallying prediction using softmax
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])
    tau = float(parameters["tau"])
    
    z = tau * (scores - np.max(scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)
        
    # Probabilistic strategy switch (constant mixture, independent of depth)
    alpha = float(parameters["alpha"])
    p_core = alpha * p_ttb + (1.0 - alpha) * p_tally
    
    # Independent lapse rate
    epsilon = float(parameters["epsilon"])
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
**Description:** People evaluate options using a non-linearly Weighted Additive (WADD) strategy. Each option's value is the sum of its features weighted by their perceived importance, which is a non-linear power function of the objective cue validities. This allows a single high-validity cue to balance out multiple lower-validity cues, resulting in compensatory trade-offs and choice probabilities near 0.5 on conflict trials.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [1.0, 10.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Compute weighted sum of features for each option
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
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
    return int(np.random.choice(len(probabilities), p=probabilities))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n    \n    df = data.copy()\n    df['A_str'] = df['option_a_ratings'].apply(lambda x: ''.join(map(str, int(v) for v in x)) if isinstance(x[0], (int, float, str)) else ''.join(map(str, x)))\n    df['A_str'] = df['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    df['B_str'] = df['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    \n    # chose_A is 1 if response is 0, else 0\n    df['chose_A'] = 1 - df['response']\n    \n    agg = df.groupby(['A_str', 'B_str'])['chose_A'].mean().to_dict()\n    \n    def get_p(a_str, b_str):\n        return agg.get((a_str, b_str), 0.5)\n        \n    # Group 1: TTB predicts A, Tally predicts B\n    p1 = get_p('11000', '00111')\n    p4 = get_p('10001', '01110')\n    \n    # Group 2: TTB predicts B, Tally predicts A\n    p5 = get_p('00111', '11000')\n    p8 = get_p('01110', '10001')\n    \n    # Group 3: TTB predicts A, Tally predicts A\n    p9 = get_p('11100', '00010')\n    p12 = get_p('10011', '01000')\n    \n    # Contrast: Expected to be 0 for TTB+Tallying, strictly positive for WADD\n    contrast = (p1 - p4) + (p8 - p5) + (p9 - p12)\n    return float(contrast)",
  "rationale": "By taking advantage of the experimental design, we can compute the difference in the probability of choosing Option A between trials where the TTB and Tallying scores are identical but the specific lower-validity cues change. Under the Constant-Mixture (TTB + Tallying) theory, the choice probability depends only on the TTB direction and the Tallying score difference, both of which are constant across these trial pairs; hence, the expected difference is 0. Under the WADD theory, choice probability is sensitive to the exact validities of the cues, so shifting the support to lower-validity cues strictly decreases the option's overall value. Summing the expected positive contrasts across three different trial groupings yields a metric that is near 0 for TTB+Tallying and robustly positive for WADD."
}
```

## Usage

```json
{
  "prompt_token_count": 3458,
  "candidates_token_count": 729,
  "total_token_count": 8567
}
```
