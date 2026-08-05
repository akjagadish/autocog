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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.55, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** This design quantitatively dissociates the 'Evidence Averaging Heuristic Theory' (Advocated) from the 'Reference-Dependent Quality Inference' (Competing) by exploiting their diametrically opposed treatments of multiple strong cues. The Competing theory evaluates cues additively relative to a reference point; if a cue is above the reference point, it is a strength and strictly adds to the option's value. The Advocated theory averages the active cues with a base prior. Trial 2 establishes whether the second-best cue (0.85) is perceived as a strength (better than zero features). If it is, the Competing theory mathematically guarantees that an option with the top two cues (Trial 1, Option B) will strictly dominate an option with only the top cue (Trial 1, Option A) because the second cue adds positive value. In contrast, the Advocated theory averages the cues: the average of 0.95 and 0.85 is lower than 0.95 alone. Thus, the Advocated theory can paradoxically predict that adding a strong, highly valid cue to an even stronger cue reduces the option's choice probability (a 'less-is-more' effect for strong cues), a reversal that the Competing theory cannot produce.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Decision-makers evaluate options based on the relative average validity of their active cues, rather than the simple sum. A base prior is included to prevent zero-division and establish a baseline expectation. This average-based integration naturally predicts the 'less-is-more' effect, as adding weak cues dilutes the overall average evidence of an option, while shared cues have a context-dependent impact by simultaneously altering the numerator and denominator.

**Parameters:**
- lambda_val: [0.1, 10.0]
- prior_count: [0.01, 10.0]
- prior_sum: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    lambda_val = float(parameters["lambda_val"])
    prior_count = float(parameters["prior_count"])
    prior_sum = float(parameters["prior_sum"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting of validities
    w = val ** lambda_val
    
    n_a = np.sum(a)
    n_b = np.sum(b)
    
    # Calculate average validity with a base prior to prevent zero-division
    v_a = (np.sum(w * a) + prior_sum) / (n_a + prior_count)
    v_b = (np.sum(w * b) + prior_sum) / (n_b + prior_count)
    
    # Convert scores to choice probabilities
    scores = np.array([v_a, v_b])
    z = beta * scores
    z = z - np.max(z)
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
**Description:** Reference-Dependent Quality Inference: Decision-makers maintain a default reference point (aspiration level) for evaluating product cues. Cues with validities above this reference point are perceived as strengths and add to the option's overall value, while cues below the reference point are perceived as defects and subtract from it. An option with zero features remains at the neutral baseline. This naturally explains why an option with zero features can be strictly preferred over an option with multiple weak cues, as the latter accumulates negative value. A loss aversion mechanism further allows negative signals to be weighted more heavily than positive ones.

**Parameters:**
- lambda_val: [0.1, 3.0]
- reference_point: [0.0, 1.0]
- loss_aversion: [1.0, 5.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    lambda_val = float(parameters["lambda_val"])
    reference_point = float(parameters["reference_point"])
    loss_aversion = float(parameters["loss_aversion"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting of validities
    w = val ** lambda_val
    
    def get_score(opt):
        # Evaluate each cue against the reference point
        diff = w - reference_point
        # Apply loss aversion to cues that fall below the reference point (defects)
        weighted_diff = np.where(diff > 0, diff, loss_aversion * diff)
        # Sum the evidence only for active features
        return np.sum(opt * weighted_diff)
        
    v_a = get_score(a)
    v_b = get_score(b)
    
    # Convert scores to choice probabilities using softmax
    scores = np.array([v_a, v_b])
    z = beta * scores
    z = z - np.max(z)  # numerical stability
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Identify Trial 1: A=[1, 0, 0, 0, 0], B=[1, 1, 0, 0, 0]\n    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))\n    t1_data = data[t1_mask]\n    if len(t1_data) == 0:\n        return 0.0\n    # The Advocated theory predicts a 'less-is-more' effect where A is chosen over B\n    # because the average validity of A's single cue is higher than the average of B's two cues.\n    # The Competing theory predicts B is chosen over A because the second cue strictly adds value.\n    return float((t1_data['response'] == 0).mean())",
  "rationale": "This metric calculates the proportion of times subjects choose Option A (the single best cue) over Option B (the top two cues) in Trial 1. The Advocated theory (Evidence Averaging) predicts a high rate of choosing A due to the less-is-more effect (averaging dilutes the strong first cue). The Competing theory predicts subjects will overwhelmingly choose B because the second cue adds positive value. Thus, the choice probability of A on Trial 1 strongly discriminates the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 3322,
  "candidates_token_count": 369,
  "total_token_count": 4745
}
```
