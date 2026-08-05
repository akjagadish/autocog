# metric_exp00_attempt_01

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

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 2: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** Both Theory 1 and Theory 2 implement exactly the same decision logic: Tallying with a right-to-left tie-breaking mechanism. The only difference between the two implementations is syntactic (using `reversed(range(len(a)))` versus `range(len(a) - 1, -1, -1)`). Because their mathematical predictions are identical for all possible inputs, no experimental design can quantitatively dissociate them. This design provides a diverse set of trials, including ties where the right-most differentiating feature favors either A or B, to verify the shared right-to-left tie-breaking behavior.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



## ADVOCATED THEORY
**Description:** Tallying with Recency-Driven Tie-Breaking: Decision makers primarily evaluate options using a Tallying heuristic, counting the number of features where one option strictly dominates the other. When the tallies are tied, they do not guess randomly. Instead, they rely on the cues most recently encoded into working memory to break the tie. Assuming a standard left-to-right reading order, the right-most features are the most recently processed. The model resolves ties by scanning features from right to left, favoring the option that wins on the first differentiating feature encountered.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    # Primary strategy: Tallying
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    
    score_a = float(a_wins)
    score_b = float(b_wins)
    
    # Recency-driven tie-breaking (right-to-left)
    if a_wins == b_wins:
        for i in reversed(range(len(a))):
            if a[i] > b[i]:
                score_a += 1.0
                break
            elif b[i] > a[i]:
                score_b += 1.0
                break
                
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Softmax choice with epsilon-greedy lapse
    scores = np.array([score_a, score_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(scores)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))
```


## COMPETING THEORY
**Description:** Tallying with Right-to-Left Position-Based Tie-Breaking

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    # Tallying: count strict feature-wise wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])
    
    # Position-based lexicographic tie-breaking (right-to-left)
    if a_wins == b_wins:
        for i in range(len(a) - 1, -1, -1):
            if a[i] > b[i]:
                scores[0] += 1.0
                break
            elif b[i] > a[i]:
                scores[1] += 1.0
                break
                
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64)
    p /= p.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(p), p=p))
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
[0] rationale: Since the two theories are mathematically identical and only differ in syntactic implementation, they will produce identical predictions. This metric measures the overall proportion of choosing Option B. As expected, it will not be able to discriminate between the two identical theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
outcome: self_sim=0.4320 (var=0.0013) adversary_sim=0.4240 (var=0.0017) welch_t=+0.732 p=0.4675 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Identify tied trials based on the experimental design\n    # Trial 3: A=[1, 1, 0, 0, 0], B=[0, 0, 1, 1, 0] -> Tie (2 vs 2)\n    # Trial 5: A=[1, 0, 0, 0, 1], B=[0, 1, 1, 0, 0] -> Tie (2 vs 2)\n    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    tied_trials = data[data['a_str'].isin(['11000', '10001'])]\n    if len(tied_trials) == 0:\n        return 0.0\n    \n    # On Trial 3 ('11000'), B wins the right-to-left tie-breaker at index 3.\n    # On Trial 5 ('10001'), A wins the right-to-left tie-breaker at index 4.\n    # We measure the proportion of choices that align with this right-to-left rule.\n    trial_3_correct = (tied_trials['a_str'] == '11000') & (tied_trials['response'] == 1)\n    trial_5_correct = (tied_trials['a_str'] == '10001') & (tied_trials['response'] == 0)\n    \n    correct_tie_breaks = trial_3_correct | trial_5_correct\n    return float(correct_tie_breaks.mean())",
  "rationale": "Since the two models are mathematically identical in their predictions, they cannot be dissociated on any experimental design. This metric isolates the specific trials where tallying results in a tie (Trials 3 and 5) and evaluates how consistently the choices follow the right-to-left tie-breaking rule implemented by both models."
}
```

## Usage

```json
{
  "prompt_token_count": 3174,
  "candidates_token_count": 474,
  "total_token_count": 8561
}
```
