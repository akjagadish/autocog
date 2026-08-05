# metric_exp00_attempt_02

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
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 7: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 9: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 10: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  trial 11: A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 12: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate the pure Take The Best (TTB) model from the Strategy Selection (TTB + Tallying) model, we exploit the compensatory nature of Tallying. Pure TTB relies exclusively on the highest-validity discriminating feature and predicts a constant choice probability for the favored option regardless of how many lower-validity features favor the alternative. In contrast, the Strategy Selection model assumes occasional lapses into Tallying. By keeping the TTB prediction constant but varying the Tallying prediction (Agreement, Conflict, or Tie), we can test whether the choice probability for the TTB-favored option drops systematically when Tallying favors the alternative. The trials vary the highest discriminating cue to ensure the effect is robust across different cue validities.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Take The Best (TTB) Heuristic: Decision makers use a non-compensatory, lexicographic strategy to choose between options. They search through cues in descending order of subjective validity (or informativeness). The first cue that discriminates between the two options strictly determines the choice, and all remaining lower-validity cues are ignored. If no cues discriminate, the decision maker guesses. Response noise is modeled as a uniform lapse.

**Parameters:**
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues in descending order of validity. 
    # We use a stable sort so that if validities are tied, left-to-right order is preserved.
    order = np.argsort(-validities, kind='stable')
    
    # Default to guessing if no cues discriminate
    p_core = np.array([0.5, 0.5])
    
    # Lexicographic search
    for i in order:
        if a[i] > b[i]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[i] > a[i]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Strategy Selection: Decision-makers predominantly rely on a non-compensatory Take The Best (TTB) heuristic but probabilistically mix in a simple compensatory Tallying strategy. Unlike validity-dependent lapse models, this mixture maintains flat adherence across different validities of discriminating cues, while allowing small but consistent deviations in choice probabilities when TTB and Tallying conflict. The probability of tallying is kept very small to align with the near-zero deviations observed in experiments.

**Parameters:**
- p_tally: [0.0, 0.05]
- epsilon: [0.0, 0.4]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Take The Best (TTB) prediction
    order = np.argsort(-validities, kind='stable')
    p_ttb_choice = np.array([0.5, 0.5])
    for i in order:
        if a[i] > b[i]:
            p_ttb_choice = np.array([1.0, 0.0])
            break
        elif b[i] > a[i]:
            p_ttb_choice = np.array([0.0, 1.0])
            break
            
    # Tallying prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    if a_wins > b_wins:
        p_tally_choice = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        p_tally_choice = np.array([0.0, 1.0])
    else:
        p_tally_choice = np.array([0.5, 0.5])
        
    p_tally = float(parameters["p_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Mixture of strategies
    p_core = (1.0 - p_tally) * p_ttb_choice + p_tally * p_tally_choice
    
    # Apply uniform response noise (lapse)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
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
[0] rationale: To distinguish pure Take The Best (TTB) from the Strategy Selection model (which mixes TTB with Tallying), we compare the subject's adherence to the TTB prediction on trials where TTB and Tallying agree versus trials where they conflict. For a pure TTB agent, the probability of choosing the TTB-favored option depends only on the uniform lapse rate, so the difference in TTB compliance between Agreement and Conflict trials should be zero. For the Strategy Selection model, occasional lapses into Tallying will boost TTB compliance on Agreement trials and diminish it on Conflict trials, producing a strictly positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    # TTB choice: lexicographic search (weights are powers of 2 descending)
    diff = a_ratings - b_ratings
    weights = 2 ** np.arange(diff.shape[1])[::-1]
    ttb_vals = diff.dot(weights)
    ttb_choice = np.where(ttb_vals > 0, 0, np.where(ttb_vals < 0, 1, -1))
    
    # Tallying choice: simple count of winning cues
    tally_vals = np.sum(a_ratings > b_ratings, axis=1) - np.sum(b_ratings > a_ratings, axis=1)
    tally_choice = np.where(tally_vals > 0, 0, np.where(tally_vals < 0, 1, -1))
    
    responses = data['response'].values
    ttb_match = (responses == ttb_choice)
    
    # Masks for Agreement and Conflict trials
    agree_mask = (ttb_choice == tally_choice) & (ttb_choice != -1)
    conflict_mask = (ttb_choice != tally_choice) & (tally_choice != -1) & (ttb_choice != -1)
    
    if np.sum(agree_mask) == 0 or np.sum(conflict_mask) == 0:
        return 0.0
        
    return float(np.mean(ttb_match[agree_mask]) - np.mean(ttb_match[conflict_mask]))
outcome: self_sim=-0.0009 (var=0.0088) adversary_sim=0.0081 (var=0.0075) welch_t=-0.355 p=0.7243 (N=25, alpha=0.01) -> reject

[1] rationale: To reliably detect the subtle Strategy Selection mixture component amidst high response noise, we compute the two-sample t-statistic of the difference in TTB non-compliance between Conflict and Agree trials. By scaling the mean difference by its standard error, the metric acts as a signal-to-noise ratio. In single subjects, this SNR is small due to limited trials. However, when applied to the pooled data, the standard error shrinks, allowing the metric to heavily amplify the consistent but small tallying effect in the Strategy Selection model, while remaining tightly centered around zero for pure TTB.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    diff = a_ratings - b_ratings
    weights = 2 ** np.arange(diff.shape[1])[::-1]
    ttb_vals = diff.dot(weights)
    ttb_choice = np.where(ttb_vals > 0, 0, np.where(ttb_vals < 0, 1, -1))
    
    tally_vals = np.sum(a_ratings > b_ratings, axis=1) - np.sum(b_ratings > a_ratings, axis=1)
    tally_choice = np.where(tally_vals > 0, 0, np.where(tally_vals < 0, 1, -1))
    
    responses = data['response'].values
    non_ttb = (responses != ttb_choice).astype(float)
    
    agree_mask = (ttb_choice == tally_choice) & (ttb_choice != -1)
    conflict_mask = (ttb_choice != tally_choice) & (tally_choice != -1) & (ttb_choice != -1)
    
    if np.sum(agree_mask) == 0 or np.sum(conflict_mask) == 0:
        return 0.0
        
    agree_rates = non_ttb[agree_mask]
    conflict_rates = non_ttb[conflict_mask]
    
    mean_diff = np.mean(conflict_rates) - np.mean(agree_rates)
    
    var_agree = np.var(agree_rates, ddof=1) if len(agree_rates) > 1 else 0
    var_conflict = np.var(conflict_rates, ddof=1) if len(conflict_rates) > 1 else 0
    
    se = np.sqrt(var_agree / len(agree_rates) + var_conflict / len(conflict_rates))
    if se < 1e-8:
        return 0.0
        
    return float(mean_diff / se)
outcome: self_sim=-0.0697 (var=1.8833) adversary_sim=0.6727 (var=1.7704) welch_t=-1.942 p=0.05804 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_ratings = np.vstack(data['option_a_ratings'].values)\n    b_ratings = np.vstack(data['option_b_ratings'].values)\n    \n    diff = a_ratings - b_ratings\n    weights = 2 ** np.arange(diff.shape[1])[::-1]\n    ttb_vals = diff.dot(weights)\n    ttb_choice = np.where(ttb_vals > 0, 0, 1)\n    \n    a_wins = np.sum(a_ratings > b_ratings, axis=1)\n    b_wins = np.sum(b_ratings > a_ratings, axis=1)\n    \n    tally_tied = (a_wins == b_wins)\n    tally_favors_a = (a_wins > b_wins)\n    tally_favors_b = (b_wins > a_wins)\n    \n    tally_support = np.zeros(len(data))\n    for i in range(len(data)):\n        if tally_tied[i]:\n            tally_support[i] = 0.5\n        elif (ttb_choice[i] == 0 and tally_favors_a[i]) or (ttb_choice[i] == 1 and tally_favors_b[i]):\n            tally_support[i] = 1.0\n        else:\n            tally_support[i] = 0.0\n            \n    responses = data['response'].values\n    y = (responses == ttb_choice).astype(float)\n    \n    if np.var(tally_support) == 0:\n        return 0.0\n        \n    slope = np.cov(tally_support, y)[0, 1] / np.var(tally_support, ddof=1)\n    return float(slope)",
  "rationale": "To reliably measure the subtle mixture of Tallying in the Strategy Selection model, we define 'Tallying Support' for the TTB choice: 1.0 on Agree trials, 0.5 on Tie trials, and 0.0 on Conflict trials. We then compute the OLS slope of the subject's binary TTB-compliance against this Tallying Support variable. For a pure TTB agent, TTB-compliance depends only on the uniform lapse rate and is independent of Tallying Support, yielding a slope of 0. For the Strategy Selection model, occasional Tallying systematically modulates TTB-compliance, yielding a strictly positive slope. By regressing over all 96 trials simultaneously, this metric maximizes statistical power and minimizes within-subject variance compared to computing noisy pairwise differences."
}
```

## Usage

```json
{
  "prompt_token_count": 4847,
  "candidates_token_count": 658,
  "total_token_count": 9895
}
```
