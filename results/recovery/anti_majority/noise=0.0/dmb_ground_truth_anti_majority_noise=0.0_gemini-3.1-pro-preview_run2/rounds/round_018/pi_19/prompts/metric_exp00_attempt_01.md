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
**Validities (n_features=5):** [0.95, 0.85, 0.8, 0.55, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0]
  trial 3: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Rationale:** This design quantitatively dissociates the 'Evidence Averaging Heuristic Theory' (Advocated) from 'Reference-Dependent Quality Inference' (Competing) by exploiting the mathematical invariance of shared cues in the Competing theory. The Competing theory evaluates cues independently and additively relative to a reference point; thus, adding a shared cue to both options adds the exact same value to both, perfectly canceling out in the score difference and predicting identical choice probabilities across trials. In contrast, the Advocated theory averages the active cues. Adding a shared weak cue dilutes the average of an option with fewer cues (Option A) more heavily than an option with more cues (Option B), predicting a massive shift in preference—potentially a full reversal—that the Competing theory mathematically cannot produce.

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
[0] rationale: This metric calculates the difference in the probability of choosing Option A between Trial 1 and Trial 3. In Trial 1, Option A has 1 strong cue and Option B has 2 moderately strong cues. In Trial 3, both options have 2 additional shared weak cues. Under the Competing theory (Reference-Dependent Quality Inference), shared cues add identical value to both options, so the choice probability for A should remain exactly the same across these two trials (difference ~ 0). Under the Advocated theory (Evidence Averaging), the shared weak cues dilute Option A's average more heavily than Option B's, causing a significant drop in preference for Option A. Thus, the difference will be strongly positive for the Advocated theory and near zero for the Competing theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    
    t1_mask = data['A_tuple'] == (1, 0, 0, 0, 0)
    t3_mask = data['A_tuple'] == (1, 0, 0, 1, 1)
    
    if t1_mask.sum() == 0 or t3_mask.sum() == 0:
        return 0.0
        
    p_A_t1 = (data[t1_mask]['response'] == 0).mean()
    p_A_t3 = (data[t3_mask]['response'] == 0).mean()
    
    return float(p_A_t1 - p_A_t3)
outcome: self_sim=0.0475 (var=0.0170) adversary_sim=-0.0008 (var=0.0114) welch_t=+1.435 p=0.1581 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data['A_tuple'] = data['option_a_ratings'].apply(tuple)\n    \n    t1_mask = data['A_tuple'] == (1, 0, 0, 0, 0)\n    t2_mask = data['A_tuple'] == (1, 0, 0, 1, 0)\n    t3_mask = data['A_tuple'] == (1, 0, 0, 1, 1)\n    \n    if t1_mask.sum() == 0 or t2_mask.sum() == 0 or t3_mask.sum() == 0:\n        return 0.0\n        \n    p1 = (data[t1_mask]['response'] == 0).mean()\n    p2 = (data[t2_mask]['response'] == 0).mean()\n    p3 = (data[t3_mask]['response'] == 0).mean()\n    \n    return float(max(p1, p2, p3) - min(p1, p2, p3))",
  "rationale": "Under the Competing theory (Reference-Dependent Quality Inference), adding shared cues to both options adds the exact same objective value to both, leaving the score difference mathematically identical. Thus, the choice probability for Option A should remain constant across Trials 1, 2, and 3. The range of these three probabilities should theoretically be zero (and practically just reflect binomial noise for individual subjects). Under the Advocated theory (Evidence Averaging), adding shared weak cues dilutes the average of Option A (which has fewer cues) more heavily than Option B, causing systematic shifts in choice probabilities across these three trials. By computing the maximum shift (range) across the three trials, we capture the magnitude of the dilution effect regardless of its direction or parameter-dependent scale, ensuring a robust discrimination from the Competing theory's prediction of zero variance."
}
```

## Usage

```json
{
  "prompt_token_count": 3659,
  "candidates_token_count": 453,
  "total_token_count": 8298
}
```
