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
  trial 1: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 2: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  trial 4: A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 1]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 8: A=[1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0]

**Rationale:** Theory 1 (Lexicographic Tallying) and Theory 2 (Equal-Weight Additive) make identical predictions when one option has strictly more positive features than the other, because any tally difference >= 1 triggers the Tallying rule in Theory 1, which perfectly mimics Equal-Weight. However, when the two options have an equal number of positive features (a tally tie), Theory 2 predicts a 50/50 guess. In contrast, Theory 1 falls back to a Take-The-Best strategy, predicting a strong preference for the option that possesses the highest-validity discriminating cue. This design consists primarily of trials where the tallies are tied but the options differ on the highest-validity cues, to maximally dissociate the two theories. We also include trials with tally differences to constrain the softmax beta parameter.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Lexicographic Tallying: Subjects evaluate options primarily by tallying the total number of positive features. If the difference in tallies between the options exceeds a certain threshold, they choose the option with the higher tally, ignoring specific cue validities. However, if the tally difference is small or zero (indicating a tie or weak evidence), they fall back to a lexicographic 'Take-The-Best' strategy, relying on the single most valid cue that discriminates between the options.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- delta: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Lexicographic Tallying expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    delta = float(parameters["delta"])
    
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    
    # Primary strategy: Tallying if difference is large enough
    if abs(tally_a - tally_b) >= delta:
        score_a = tally_a
        score_b = tally_b
    else:
        # Fallback strategy: Take-The-Best (Lexicographic)
        idx = np.argsort(val)[::-1]
        score_a, score_b = 0.0, 0.0
        for i in idx:
            if a[i] > b[i]:
                score_a = 1.0
                score_b = 0.0
                break
            elif b[i] > a[i]:
                score_a = 0.0
                score_b = 1.0
                break
        # If completely tied on all cues
        if score_a == 0.0 and score_b == 0.0:
            score_a = 0.5
            score_b = 0.5
            
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for response noise
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
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Equal-Weight Additive Rule: People integrate all available cues but ignore their differential validities, assigning equal weight to each cue. They compute an overall score for each option by summing its positive features. This is a fully compensatory process that discards validity magnitudes but retains the quantity of evidence. The option with the higher total score is favored, with response noise introduced via a softmax function and a uniform lapse rate.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Equal-Weight expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    # Sum all features (equal weighting)
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n\n    # Find trials where the overall tally is tied\n    tally_a = data['option_a_ratings'].apply(sum)\n    tally_b = data['option_b_ratings'].apply(sum)\n    tied_mask = tally_a == tally_b\n\n    # Determine the Take-The-Best prediction (features are sorted by validity)\n    def ttb_predict(row):\n        a = row['option_a_ratings']\n        b = row['option_b_ratings']\n        for i in range(len(a)):\n            if a[i] > b[i]:\n                return 0\n            elif b[i] > a[i]:\n                return 1\n        return -1\n\n    ttb_preds = data.apply(ttb_predict, axis=1)\n    valid_mask = tied_mask & (ttb_preds != -1)\n\n    if valid_mask.sum() == 0:\n        return 0.5\n\n    # Calculate proportion of choices consistent with Take-The-Best on tied trials\n    consistent = (data.loc[valid_mask, 'response'] == ttb_preds.loc[valid_mask]).mean()\n    return float(consistent)",
  "rationale": "The two theories make identical predictions when one option has a strictly higher tally than the other (since delta <= 1.0, Lexicographic Tallying will always use the tally rule for non-ties). They only diverge on trials where the tallies are tied. Equal-Weight Additive predicts a 50/50 guess on tied trials, regardless of the features. Lexicographic Tallying, however, falls back to a Take-The-Best strategy, relying on the highest-validity cue that discriminates the options. This metric isolates the tied tally trials and measures the proportion of choices that align with the Take-The-Best prediction. Equal-Weight Additive will yield a value around 0.5, whereas Lexicographic Tallying will yield a value significantly higher than 0.5."
}
```

## Usage

```json
{
  "prompt_token_count": 3318,
  "candidates_token_count": 511,
  "total_token_count": 6248
}
```
