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

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  trial 8: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 9: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  trial 10: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** This design quantitatively dissociates the Weighted Additive (WADD) theory from the Strategy Mixture (TTB + Tallying) model by exploiting the Mixture model's blindness to the validities of lower-ranked cues. In the Strategy Mixture model, the Take The Best (TTB) component only cares about the highest-validity discriminating cue, while the Tallying component only counts the sheer number of cues favoring each option, ignoring their validities entirely. Therefore, if we hold the highest discriminating cue constant (so TTB's preference is unchanged) and hold the number of opposing cues constant (so the Tally difference is unchanged), the Strategy Mixture model predicts identical choice probabilities, regardless of which specific lower-validity cues oppose the choice. WADD, by contrast, integrates the scaled validities of all cues. It predicts that choice probability will systematically vary depending on the exact validities of the opposing cues (e.g., an opposing cue with 0.8 validity will reduce choice probability much more than an opposing cue with 0.5 validity). By systematically shifting the opposing cues across different validity ranks while keeping the Tally difference constant, we can clearly separate the two models.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Weighted Additive (WADD) theory with non-linear scaling posits that decision-makers integrate all available information by weighting each feature by its validity, but subjectively scale these validities to adjust the degree of compensation. By exponentiating validities by a parameter gamma and then normalizing them, the model flexibly interpolates between equal weighting (Tallying), proportional weighting, and highly non-compensatory weighting (Take The Best) while keeping the integrated scores on a stable scale. Choices are made probabilistically using a softmax over the options' scores.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 30.0]
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
    
    # Non-linear scaling of validities with normalization to prevent score shrinkage
    weights = val ** gamma
    weights = weights / np.sum(weights)
    
    # Calculate overall scores by weighting features by their scaled validity
    a_score = np.dot(a, weights)
    b_score = np.dot(b, weights)
    scores = np.array([a_score, b_score])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
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
**Description:** Decision-makers do not exclusively rely on a single decision rule; rather, choices reflect a probabilistic mixture of strategies on a trial-by-trial basis (or heterogeneity across individuals). The Strategy Mixture theory posits that a decision is made by selecting either a non-compensatory strategy (Take The Best) with probability alpha, or a compensatory strategy (Tallying) with probability 1 - alpha. Take The Best relies solely on the most valid discriminating cue, making it highly frugal and non-compensatory, whereas Tallying integrates information by counting the total number of cues favoring each option. By averaging across these two distinct decision rules, the model seamlessly bridges the non-compensatory invariance observed in Experiment 2 and the compensatory sensitivity observed in Experiment 1.

**Parameters:**
- alpha: [0.0, 1.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Mixture model expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Take The Best (TTB) component
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        scores_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        scores_ttb = np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.0, 0.0])
        
    # Tallying component
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    alpha = float(parameters["alpha"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for TTB
    z_ttb = beta * (scores_ttb - scores_ttb.max())
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / e_ttb.sum()
    
    # Softmax for Tallying
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # Mixture of strategies
    p_mix = alpha * p_ttb + (1.0 - alpha) * p_tally
    
    # Lapse rate
    n_opts = p_mix.shape[0]
    return (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Convert option_b_ratings to tuples to make them hashable and comparable\n    b_tuples = data['option_b_ratings'].apply(tuple)\n    \n    # Weakest opponent B options: cues 4 and 5\n    weak_b = [(0, 0, 0, 0, 1), (0, 0, 0, 1, 1)]\n    # Strongest opponent B options: cues 2 and 3\n    strong_b = [(0, 1, 0, 0, 0), (0, 1, 1, 0, 0)]\n    \n    is_weak = b_tuples.isin(weak_b)\n    is_strong = b_tuples.isin(strong_b)\n    \n    # Calculate probability of choosing Option A (response == 0)\n    p_a_weak = (data.loc[is_weak, 'response'] == 0).mean()\n    p_a_strong = (data.loc[is_strong, 'response'] == 0).mean()\n    \n    if pd.isna(p_a_weak) or pd.isna(p_a_strong):\n        return 0.0\n        \n    return float(p_a_weak - p_a_strong)",
  "rationale": "This metric calculates the difference in the probability of choosing Option A when facing the 'weakest' B options (those relying on the lowest validity cues) versus the 'strongest' B options (those relying on the highest validity cues available to B), holding the sheer number of cues favoring B constant. Under the Strategy Mixture theory, the Tallying component only counts cues and the Take The Best component only looks at the highest validity cue (which A always wins). Thus, the Mixture model predicts no difference in choice probability (metric \u2248 0). In contrast, the Weighted Additive (WADD) theory integrates the exact validities, predicting a higher probability of choosing A when B's cues are weaker (metric > 0)."
}
```

## Usage

```json
{
  "prompt_token_count": 3725,
  "candidates_token_count": 482,
  "total_token_count": 6436
}
```
