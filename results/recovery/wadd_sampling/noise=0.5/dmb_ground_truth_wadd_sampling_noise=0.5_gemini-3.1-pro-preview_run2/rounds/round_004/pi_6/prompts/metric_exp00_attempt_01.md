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
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  trial 2: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  trial 5: A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  trial 6: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 8: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]

**Rationale:** This design sharply dissociates the Advocated Softened Integrated Strategy Value Theory from the Competing Probabilistic Strategy Selection Theory by exploiting their fundamentally different treatments of the Weighted Additive (WADD) score magnitude. The Competing theory assumes decision-makers sample pure, discrete strategies; thus, if WADD favors Option A, the WADD heuristic outputs a categorical preference for A regardless of whether the WADD score difference is 0.05 or 0.85. In contrast, the Advocated theory integrates the continuous WADD score into a softmax decision rule, meaning the probability of choosing A scales directly with the magnitude of the WADD difference. We employ 'Tallying-Tied' trials where Take-The-Best (TTB) consistently points to Option A and Tallying is perfectly tied, but the magnitude of the WADD difference favoring Option A varies from marginal to massive. The Competing theory is mathematically constrained to predict the exact same choice probabilities across these trials, while the Advocated theory predicts a smoothly increasing preference for Option A.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
[0] rationale: This metric calculates the difference in the probability of choosing the option favored by both TTB and WADD between trials with a massive WADD difference (>0.8) and trials with a marginal WADD difference (<0.1), restricting to trials where Tallying is perfectly tied. The Competing theory (discrete strategy selection) predicts this difference will be exactly zero because the categorical outputs of TTB, Tallying, and WADD are identical across both trial types. The Advocated theory (continuous integration) predicts a positive difference because the larger WADD magnitude directly increases the softmax probability of choosing the favored option.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55, 0.5])
    
    wadd_diff = data.apply(lambda row: np.sum(val * np.array(row['option_a_ratings'])) - np.sum(val * np.array(row['option_b_ratings'])), axis=1)
    tally_diff = data.apply(lambda row: np.sum(row['option_a_ratings']) - np.sum(row['option_b_ratings']), axis=1)
    
    tally_tied = tally_diff == 0
    favored_option = (wadd_diff < 0).astype(int)
    chose_favored = (data['response'] == favored_option).astype(int)
    
    high_mag = tally_tied & (np.abs(wadd_diff) > 0.8)
    low_mag = tally_tied & (np.abs(wadd_diff) < 0.1)
    
    if high_mag.sum() == 0 or low_mag.sum() == 0:
        return 0.0
        
    p_high = chose_favored[high_mag].mean()
    p_low = chose_favored[low_mag].mean()
    
    return float(p_high - p_low)
outcome: self_sim=0.0742 (var=0.0176) adversary_sim=-0.0008 (var=0.0140) welch_t=+2.110 p=0.04015 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55, 0.5])\n    \n    # Convert lists to numpy arrays for vectorized operations\n    a_ratings = np.array(data['option_a_ratings'].tolist())\n    b_ratings = np.array(data['option_b_ratings'].tolist())\n    \n    # The first cue (index 0) has the highest validity (0.95) and is always the\n    # discriminating cue for Take-The-Best (TTB) in this specific experimental design.\n    # Favored option is 0 (A) if A has the cue (a_ratings[:, 0] == 1), else 1 (B).\n    favored_option = 1 - a_ratings[:, 0]\n    chose_favored = (data['response'] == favored_option).astype(float)\n    \n    # Identify Tally-tied vs Tally-opposes trials\n    sum_a = a_ratings.sum(axis=1)\n    sum_b = b_ratings.sum(axis=1)\n    tally_tied = (sum_a == sum_b)\n    \n    # Calculate WADD difference magnitude\n    wadd_a = (a_ratings * val).sum(axis=1)\n    wadd_b = (b_ratings * val).sum(axis=1)\n    wadd_diff_abs = np.abs(wadd_a - wadd_b)\n    \n    # Isolate Trial types 1 & 4 (Tally tied, marginal WADD diff ~ 0.05)\n    is_t1_t4 = tally_tied & (wadd_diff_abs < 0.2)\n    \n    # Isolate Trial types 7 & 8 (Tally strictly opposes TTB)\n    is_t7_t8 = ~tally_tied\n    \n    if is_t1_t4.sum() == 0 or is_t7_t8.sum() == 0:\n        return 0.0\n        \n    p_tied_low = chose_favored[is_t1_t4].mean()\n    p_opposes = chose_favored[is_t7_t8].mean()\n    \n    # Theoretical contrast metric\n    return float(1.0 - 2.0 * p_tied_low + p_opposes)\n",
  "rationale": "This metric exploits a rigid algebraic constraint in the Competing (Probabilistic Strategy Selection) theory. In the Competing theory, the probability of choosing the TTB-favored option when Tally is tied is P_tied = (1-eps)*(w_ttb + w_wadd + 0.5*w_tal) + 0.5*eps. When Tally strictly opposes TTB, the probability is P_opposes = (1-eps)*(w_ttb + w_wadd) + 0.5*eps. Substituting these into the contrast `1 - 2*P_tied + P_opposes` algebraically collapses to exactly `0.5 * eps`, which is entirely invariant to the subject's strategy weights (w_ttb, w_tal, w_wadd). Because it cancels out the strategy weights, the between-subject variance for the Competing theory is strictly bounded to the variance of epsilon plus binomial noise.\n\nIn contrast, the Advocated (Softened Integrated Strategy Value) theory integrates the Tallying penalty (-1/6) directly into the continuous value score before applying a non-linear softmax. This causes P_opposes to drop significantly lower than what a linear mixture would predict, making the value of `1 - 2*P_tied + P_opposes` substantially larger than 0.5 * eps. By isolating this exact linear combination, we drastically reduce the Competing theory's variance while maintaining a large mean gap, yielding a highly significant Welch's t-statistic."
}
```

## Usage

```json
{
  "prompt_token_count": 4642,
  "candidates_token_count": 957,
  "total_token_count": 33050
}
```
