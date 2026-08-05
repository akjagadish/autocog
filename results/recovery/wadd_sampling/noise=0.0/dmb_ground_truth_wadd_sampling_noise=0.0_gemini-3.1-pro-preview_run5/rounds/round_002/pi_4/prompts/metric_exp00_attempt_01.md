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

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 5: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 6: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Rationale:** This design aims to dissociate Validity-Thresholded Tallying from Weighted Additive (WADD) models by presenting choice sets that induce preference reversals under a thresholding mechanism but not under smooth weighting. The validities are distributed linearly from 0.9 to 0.5. Under Thresholded Tallying, cues below a subjective threshold are completely ignored, and the remaining cues are tallied with equal weight. This leads to distinct non-monotonic choice patterns as the threshold varies. For example, in a trial where A is supported by cues 1, 4, 5 and B by cues 2, 3, WADD consistently prefers A across most gamma values, whereas Thresholded Tallying will prefer A (if threshold is high, using only cue 1), switch to a Tie (using cues 1, 2), switch to B (using cues 1, 2, 3), and then back to A (using all cues). Such discrete switches cannot be captured by WADD's continuous exponentiation of validities.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Validity-Thresholded Tallying: Decision-makers simplify choices by ignoring cues with low validities and applying a simple tallying heuristic to the remaining high-validity cues. A subjective threshold determines which cues are considered reliable enough to use. For the included cues, the decision-maker counts how many times each option has a higher feature value than the other. The option with more wins among the thresholded cues is chosen. This boundedly rational strategy combines the frugality and robustness of tallying with the validity-sensitivity of weighted additive models, effectively breaking ties in favor of options that excel on more valid cues without requiring complex mental arithmetic.

**Parameters:**
- tau: [0.0, 1.0]
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Thresholded Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(f"validities length {val.shape[0]} != n_features {stim.shape[1]}.")
        
    tau = float(parameters["tau"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Identify which cues meet the subjective validity threshold
    valid_mask = val >= tau
    
    if not np.any(valid_mask):
        # If no cues are reliable enough, the decision-maker has no preference
        scores = np.zeros(2)
    else:
        # Tally wins only on the included cues
        a_filtered = stim[0, valid_mask]
        b_filtered = stim[1, valid_mask]
        
        a_wins = float(np.sum(a_filtered > b_filtered))
        b_wins = float(np.sum(b_filtered > a_filtered))
        
        scores = np.array([a_wins, b_wins])
        
    # Softmax over the tally scores
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


## COMPETING THEORY
**Description:** Decision-makers integrate all available information by taking a weighted sum of each option's features, where the weights are subjective transformations of the cue validities. By exponentiating the raw validities by a free parameter gamma, the weighting scheme can smoothly interpolate between equal weighting (Tallying), proportional weighting (raw Weighted Additive), and lexicographic-like steep weighting (Take The Best). Choice probabilities are generated via a softmax over these subjectively weighted sums, combined with a lapse rate. Human behavior is best described by relatively flat (Tally-like) weights combined with substantial choice noise (lower beta).

**Parameters:**
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 2.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(f"validities length {val.shape[0]} != n_features {stim.shape[1]}.")
        
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Calculate the weighted sum of features for each option
    scores = np.dot(stim, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores with max-subtraction for numerical stability
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
[0] rationale: This metric calculates the proportion of times the subject chooses the option with cues {2, 3} over the option with cues {1, 4, 5}. Under the Weighted Additive (WADD) model, the option with cues {1, 4, 5} is always strictly preferred regardless of the gamma parameter (since it dominates both in sum of raw validities and in maximum validity). Thus, WADD will only choose {2, 3} due to random choice noise. However, under Validity-Thresholded Tallying, if the subjective threshold falls between 0.6 and 0.7, cues 4 and 5 are ignored, and the option with cues {2, 3} wins the tally against cue {1}. Furthermore, thresholds causing ties will also elevate the choice probability of {2, 3}. Therefore, Thresholded Tallying predicts a significantly higher rate of choosing {2, 3} on this specific trial compared to WADD.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def is_target_trial(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == (1, 0, 0, 1, 1) and b == (0, 1, 1, 0, 0):
            return True
        if a == (0, 1, 1, 0, 0) and b == (1, 0, 0, 1, 1):
            return True
        return False
        
    mask = data.apply(is_target_trial, axis=1)
    target_data = data[mask]
    
    if len(target_data) == 0:
        return 0.0
        
    def chose_23(row):
        a = tuple(row['option_a_ratings'])
        if a == (0, 1, 1, 0, 0):
            return row['response'] == 0
        else:
            return row['response'] == 1
            
    choices = target_data.apply(chose_23, axis=1)
    return float(choices.mean())
outcome: self_sim=0.3237 (var=0.0506) adversary_sim=0.3150 (var=0.0283) welch_t=+0.156 p=0.8769 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    def get_choice(row):\n        a = tuple(row['option_a_ratings'])\n        b = tuple(row['option_b_ratings'])\n        \n        # Trial 2 & 6: {1, 5} is (1, 0, 0, 0, 1), {2, 3, 4} is (0, 1, 1, 1, 0)\n        if a == (1, 0, 0, 0, 1) and b == (0, 1, 1, 1, 0):\n            return ('t2', row['response'] == 0)\n        elif a == (0, 1, 1, 1, 0) and b == (1, 0, 0, 0, 1):\n            return ('t2', row['response'] == 1)\n            \n        # Trial 4: {2, 4} is (0, 1, 0, 1, 0), {1, 3, 5} is (1, 0, 1, 0, 1)\n        elif a == (0, 1, 0, 1, 0) and b == (1, 0, 1, 0, 1):\n            return ('t4', row['response'] == 0)\n        elif a == (1, 0, 1, 0, 1) and b == (0, 1, 0, 1, 0):\n            return ('t4', row['response'] == 1)\n            \n        return (None, False)\n\n    choices = data.apply(get_choice, axis=1)\n    \n    t2_choices = [c[1] for c in choices if c[0] == 't2']\n    t4_choices = [c[1] for c in choices if c[0] == 't4']\n    \n    p_t2 = np.mean(t2_choices) if t2_choices else 0.0\n    p_t4 = np.mean(t4_choices) if t4_choices else 0.0\n    \n    return float(p_t2 - p_t4)",
  "rationale": "This metric isolates a specific difference in choice probabilities between two trial types. On Trial 2, the option with cues {1, 5} competes against {2, 3, 4}. On Trial 4, the option with cues {2, 4} competes against {1, 3, 5}. Under the Weighted Additive (WADD) model, the options {1, 5} and {2, 4} are both strongly disfavored across all valid gamma parameters, and their choice probabilities are driven almost entirely by symmetrical choice noise, leading to a difference near 0. However, under Validity-Thresholded Tallying, the option {1, 5} will actually win outright when the threshold eliminates cues 3, 4, and 5 (leaving cue 1 to dominate), whereas {2, 4} can never win outright against {1, 3, 5}. Thus, Tallying predicts a significantly positive difference, while WADD predicts zero."
}
```

## Usage

```json
{
  "prompt_token_count": 3938,
  "candidates_token_count": 775,
  "total_token_count": 9295
}
```
