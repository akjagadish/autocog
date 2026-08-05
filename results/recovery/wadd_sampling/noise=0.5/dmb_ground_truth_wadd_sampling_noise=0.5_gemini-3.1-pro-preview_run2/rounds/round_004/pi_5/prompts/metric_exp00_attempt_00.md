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

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Rationale:** This design sharply dissociates the Advocated Probabilistic Strategy Selection Theory from the Competing Softened Integrated Strategy Value Theory by exploiting their fundamentally different treatments of the Take-The-Best (TTB) heuristic. In the Advocated theory, TTB is an all-or-nothing categorical rule: if TTB selects an option, it contributes its full base-rate probability to that option, regardless of which cue in the hierarchy broke the tie. In contrast, the Competing theory scales the TTB score by the normalized validity of the specific discriminating cue. We design a sequence of 'TTB-dilution' trials where the cue that breaks the TTB tie shifts progressively from the most valid feature down to the third most valid feature. In all these trials, both Tallying and WADD strongly oppose the TTB choice. The Competing theory predicts a progressively weakening preference for the TTB-favored option as the discriminating cue's validity drops, whereas the Advocated theory predicts a constant, undiluted TTB contribution across the sequence.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Strategy Selection Theory: Decision-makers maintain a repertoire of distinct decision heuristics (Take-The-Best, Tallying, and Weighted Additive/WADD). On each trial, they probabilistically sample one strategy from this repertoire according to base-rate strategy weights, and execute it purely. This discrete mixture of strategies accounts for intermediate aggregate choice proportions while naturally predicting exact ties when the sampled strategy itself encounters a tie.

**Parameters:**
- w_ttb: [0.0, 1.0]
- w_tal: [0.0, 1.0]
- w_wadd: [0.0, 1.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB)
    diff = a - b
    order = np.argsort(val)[::-1]
    ttb_pred = np.array([0.5, 0.5])
    for idx in order:
        if diff[idx] > 0:
            ttb_pred = np.array([1.0, 0.0])
            break
        elif diff[idx] < 0:
            ttb_pred = np.array([0.0, 1.0])
            break
            
    # Tallying
    tal_diff = np.sum(a) - np.sum(b)
    if tal_diff > 0:
        tal_pred = np.array([1.0, 0.0])
    elif tal_diff < 0:
        tal_pred = np.array([0.0, 1.0])
    else:
        tal_pred = np.array([0.5, 0.5])
        
    # Weighted Additive (WADD)
    wadd_diff = np.sum(val * a) - np.sum(val * b)
    if wadd_diff > 0:
        wadd_pred = np.array([1.0, 0.0])
    elif wadd_diff < 0:
        wadd_pred = np.array([0.0, 1.0])
    else:
        wadd_pred = np.array([0.5, 0.5])
        
    w_ttb = float(parameters["w_ttb"])
    w_tal = float(parameters["w_tal"])
    w_wadd = float(parameters["w_wadd"])
    epsilon = float(parameters["epsilon"])
    
    w_arr = np.array([w_ttb, w_tal, w_wadd])
    sum_w = np.sum(w_arr)
    if sum_w == 0:
        p_strat = np.array([1/3, 1/3, 1/3])
    else:
        p_strat = w_arr / sum_w
        
    p_core = p_strat[0] * ttb_pred + p_strat[1] * tal_pred + p_strat[2] * wadd_pred
    
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
**Description:** Softened Integrated Strategy Value Theory: Decision-makers evaluate options by integrating evidence from multiple heuristics (Take-The-Best, Tallying, and Weighted Additive) into a single continuous subjective value for each option. To prevent non-compensatory heuristics from disproportionately dominating the integrated value, the TTB signal is softened by scaling it according to the normalized validity of the discriminating cue. These integrated values are then compared via a stochastic decision process (softmax) to produce a choice. This allows the model to gracefully capture indifference in delicately balanced trials while still reflecting heuristic-aligned preferences.

**Parameters:**
- w_ttb: [0.0, 1.0]
- w_tal: [0.0, 1.0]
- w_wadd: [0.0, 1.0]
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    n_features = len(val)
    sum_val = np.sum(val)
    
    # Take-The-Best (TTB) Score (softened)
    diff = a - b
    order = np.argsort(val)[::-1]
    score_ttb_a, score_ttb_b = 0.0, 0.0
    for idx in order:
        if diff[idx] > 0:
            score_ttb_a, score_ttb_b = val[idx] / sum_val, 0.0
            break
        elif diff[idx] < 0:
            score_ttb_a, score_ttb_b = 0.0, val[idx] / sum_val
            break
            
    # Tallying Score
    score_tal_a = np.sum(a) / n_features
    score_tal_b = np.sum(b) / n_features
    
    # Weighted Additive (WADD) Score
    score_wadd_a = np.sum(val * a) / sum_val
    score_wadd_b = np.sum(val * b) / sum_val
    
    w_ttb = float(parameters["w_ttb"])
    w_tal = float(parameters["w_tal"])
    w_wadd = float(parameters["w_wadd"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Normalize weights
    w_sum = w_ttb + w_tal + w_wadd
    if w_sum > 0:
        w_ttb /= w_sum
        w_tal /= w_sum
        w_wadd /= w_sum
    else:
        w_ttb = w_tal = w_wadd = 1.0 / 3.0
        
    # Integrated Option Values
    score_a = w_ttb * score_ttb_a + w_tal * score_tal_a + w_wadd * score_wadd_a
    score_b = w_ttb * score_ttb_b + w_tal * score_tal_b + w_wadd * score_wadd_b
    
    scores = np.array([score_a, score_b])
    
    # Single Softmax Decision Process
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    # Lapse rate
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_keys = data['option_a_ratings'].apply(tuple)\n    b_keys = data['option_b_ratings'].apply(tuple)\n    \n    # Trial 1: TTB tiebreaker is cue 0. TTB favors (1,0,0,0,0)\n    t1_mask_dir = (a_keys == (1,0,0,0,0)) & (b_keys == (0,1,1,1,0))\n    t1_mask_rev = (a_keys == (0,1,1,1,0)) & (b_keys == (1,0,0,0,0))\n    \n    # Trial 3: TTB tiebreaker is cue 2. TTB favors (1,1,1,0,0)\n    t3_mask_dir = (a_keys == (1,1,1,0,0)) & (b_keys == (1,1,0,1,1))\n    t3_mask_rev = (a_keys == (1,1,0,1,1)) & (b_keys == (1,1,1,0,0))\n    \n    t1_ttb_choices = 0\n    t1_total = 0\n    if t1_mask_dir.sum() > 0:\n        t1_ttb_choices += (data.loc[t1_mask_dir, 'response'] == 0).sum()\n        t1_total += t1_mask_dir.sum()\n    if t1_mask_rev.sum() > 0:\n        t1_ttb_choices += (data.loc[t1_mask_rev, 'response'] == 1).sum()\n        t1_total += t1_mask_rev.sum()\n        \n    t3_ttb_choices = 0\n    t3_total = 0\n    if t3_mask_dir.sum() > 0:\n        t3_ttb_choices += (data.loc[t3_mask_dir, 'response'] == 0).sum()\n        t3_total += t3_mask_dir.sum()\n    if t3_mask_rev.sum() > 0:\n        t3_ttb_choices += (data.loc[t3_mask_rev, 'response'] == 1).sum()\n        t3_total += t3_mask_rev.sum()\n        \n    p_t1 = t1_ttb_choices / t1_total if t1_total > 0 else 0.5\n    p_t3 = t3_ttb_choices / t3_total if t3_total > 0 else 0.5\n    \n    return float(p_t1 - p_t3)\n",
  "rationale": "This metric calculates the difference in the probability of choosing the TTB-favored option between Trial 1 (where the most valid cue breaks the tie) and Trial 3 (where the third most valid cue breaks the tie). In the Advocated theory, TTB is an all-or-nothing categorical rule, so its contribution to the choice probability is unaffected by which cue breaks the tie, leading to a difference near 0. In the Competing theory, the TTB signal is softened by the validity of the discriminating cue, so the preference for the TTB-favored option will systematically drop from Trial 1 to Trial 3, resulting in a noticeably positive difference."
}
```

## Usage

```json
{
  "prompt_token_count": 3947,
  "candidates_token_count": 840,
  "total_token_count": 8341
}
```
