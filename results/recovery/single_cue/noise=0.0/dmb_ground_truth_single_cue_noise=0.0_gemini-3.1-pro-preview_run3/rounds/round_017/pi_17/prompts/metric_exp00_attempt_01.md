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
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 3: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]

**Rationale:** To quantitatively dissociate the advocated Decisiveness-Dependent Strategy Selection model from the competing Tallying with Graded Validity Tie-Breaker model, we exploit how each model resolves tied tallies. The advocated model resolves ties by probabilistically falling back to a strictly non-compensatory Take-The-Best (TTB) rule, meaning its choice on a tied-tally trial depends solely on the single highest-validity discriminating cue. Consequently, for any two tied-tally trials where Option A wins the top cue, the advocated model predicts exactly identical choice probabilities for A. In contrast, the competing model resolves ties using a compensatory weighted sum of exponentiated validities (WADD). We designed two tied-tally trials (Trials 1 and 2). In both, Option A wins the highest-validity cue. However, in Trial 1, Option B wins the 2nd and 3rd best cues (providing strong compensatory counter-weight), whereas in Trial 2, Option B wins the 2nd and 4th best cues. The competing model is structurally forced to predict that Option A is chosen with significantly higher probability in Trial 2 than in Trial 1, whereas the advocated model strictly predicts equal probabilities. Unequal-tally trials are included to constrain baseline tallying determinism.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



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
**Description:** Tallying with Graded Validity Tie-Breaker: Decision-makers primarily use a simple Tallying heuristic (comparing the total number of positive features). When tallies are unequal, choices are highly deterministic. However, when tallies are tied, instead of rigidly applying a non-compensatory Take-The-Best rule (which overpredicts determinism), they fall back to a compensatory Weighted Additive (WADD) evaluation of the features. This tie-breaking process is governed by its own sensitivity parameter, allowing tied decisions to exhibit the softer, empirically observed determinism (~0.60) while maintaining the structural dominance of Tallying.

**Parameters:**
- beta_tally: [0.1, 20.0]
- beta_wadd: [0.1, 20.0]
- theta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    a = np.asarray(stimulus[0], dtype=float)
    b = np.asarray(stimulus[1], dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta_tally = float(parameters["beta_tally"])
    beta_wadd = float(parameters["beta_wadd"])
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    
    if tally_a != tally_b:
        # Primary decision based on tally difference
        z = beta_tally * (tally_a - tally_b)
    else:
        # Fallback to WADD-based tie-breaker with exponentiated validities
        val_w = val ** theta
        wadd_a = np.sum(val_w * a)
        wadd_b = np.sum(val_w * b)
        max_wadd = np.sum(val_w)
        if max_wadd == 0:
            max_wadd = 1.0
        # Normalize by max possible WADD for scale invariance
        z = beta_wadd * (wadd_a - wadd_b) / max_wadd
        
    # Numerically stable logistic function
    p_a = 1.0 / (1.0 + np.exp(-z))
    
    # Apply trembling hand epsilon
    p_a = (1.0 - epsilon) * p_a + epsilon * 0.5
    
    return np.array([p_a, 1.0 - p_a])
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
[0] rationale: Calculates the difference in the probability of choosing Option A between Trial 2 and Trial 1. Both are tied-tally trials where Option A wins the highest validity cue. The advocated model predicts an identical choice probability for A across both trials (difference ~ 0), as it strictly resolves ties via the single best cue. The competing model resolves ties via a compensatory WADD rule, and because Option B's counter-weight is much stronger in Trial 1 than Trial 2, it is structurally forced to predict a higher probability of choosing A in Trial 2 (difference > 0).
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    t1_mask = a_str == '10010'
    t2_mask = a_str == '10100'
    
    p_a_t1 = (1 - data.loc[t1_mask, 'response']).mean()
    p_a_t2 = (1 - data.loc[t2_mask, 'response']).mean()
    
    if pd.isna(p_a_t1) or pd.isna(p_a_t2):
        return 0.0
        
    return float(p_a_t2 - p_a_t1)
outcome: self_sim=-0.0274 (var=0.0145) adversary_sim=0.0442 (var=0.0178) welch_t=-1.991 p=0.05224 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    t1_mask = a_str == '10010'\n    if t1_mask.sum() == 0:\n        return 0.5\n    p_a_t1 = (1 - data.loc[t1_mask, 'response']).mean()\n    return float(p_a_t1)",
  "rationale": "Focuses purely on the probability of choosing Option A in Trial 1. In Trial 1, the validities of A's features (0.9, 0.6) and B's features (0.8, 0.7) perfectly balance out under a linear weighted additive model (1.5 vs 1.5). For the competing theory's WADD tie-breaker, even with exponentiated validities, the difference between A and B remains extremely close to zero across its entire parameter range, forcing its prediction near 0.5. In contrast, the advocated theory's TTB tie-breaker strictly looks at the single best cue (0.9 vs 0.8), predicting Option A with probability 1.0 (minus epsilon). This creates a massive, stable gap in the absolute probability of choosing A on this specific trial."
}
```

## Usage

```json
{
  "prompt_token_count": 4074,
  "candidates_token_count": 327,
  "total_token_count": 6626
}
```
