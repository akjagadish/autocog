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

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 5: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 7: A=[0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 9: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 10: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 11: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 13: A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 14: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 16: A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Rationale:** To quantitatively dissociate pure Take-The-Best (TTB) from the Dual-Process Hybrid Model (TTB + WADD), we systematically vary the compensatory support (WADD) while holding the TTB prediction constant. Pure TTB is strictly non-compensatory and predicts that the choice probability depends entirely on the single most valid discriminating feature. Therefore, it predicts perfectly flat choice probabilities across trials that share the same best discriminating feature, regardless of how many lower-validity features support or oppose the choice. In contrast, the Hybrid model incorporates a WADD component. As the lower-validity features shift from supporting to strongly opposing the TTB winner, the WADD component changes, leading the Hybrid model to predict a graded modulation of choice probabilities. By comparing trials with identical TTB scores but vastly different WADD scores, we can definitively detect the presence of the secondary compensatory process.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Dual-Process Hybrid Model (Take-The-Best + Weighted Additive) with soft compensatory noise: Human decision-making is predominantly non-compensatory (Take-The-Best), but subjects sometimes fall back on a compensatory strategy (Weighted Additive). Because WADD operates on sums of validities, its response noise is decoupled and constrained to extremely low values to reflect the highly soft, almost guessing-like nature of human compensatory choice. A high potential lapse rate captures general response noise and further scales down overconfident predictions.

**Parameters:**
- beta_ttb: [0.0, 10.0]
- beta_wadd: [0.0, 0.15]
- p_ttb: [0.0, 1.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Hybrid expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB) evaluation
    ranked_features = np.argsort(validities, kind='stable')[::-1]
    ttb_scores = np.array([0.0, 0.0])
    for idx in ranked_features:
        if a[idx] > b[idx]:
            ttb_scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_scores[1] = 1.0
            break
            
    # Weighted Additive (WADD) evaluation
    wadd_a = np.sum(a * validities)
    wadd_b = np.sum(b * validities)
    wadd_scores = np.array([wadd_a, wadd_b])
    
    beta_ttb = float(parameters["beta_ttb"])
    beta_wadd = float(parameters["beta_wadd"])
    
    # Softmax probabilities for TTB
    if ttb_scores[0] == 0.0 and ttb_scores[1] == 0.0:
        p_ttb_vec = np.array([0.5, 0.5])
    else:
        z_ttb = beta_ttb * ttb_scores
        z_ttb = z_ttb - np.max(z_ttb)
        e_ttb = np.exp(z_ttb)
        p_ttb_vec = e_ttb / np.sum(e_ttb)
        
    # Softmax probabilities for WADD
    if wadd_a == wadd_b:
        p_wadd_vec = np.array([0.5, 0.5])
    else:
        z_wadd = beta_wadd * wadd_scores
        z_wadd = z_wadd - np.max(z_wadd)
        e_wadd = np.exp(z_wadd)
        p_wadd_vec = e_wadd / np.sum(e_wadd)
        
    # Mix the two processes
    p_ttb_weight = float(parameters["p_ttb"])
    p_core = p_ttb_weight * p_ttb_vec + (1.0 - p_ttb_weight) * p_wadd_vec
    
    # Apply uniform lapse
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Take-The-Best (TTB) heuristic: People make decisions by ranking features according to their validities and choosing the option that is favored by the single most valid discriminating feature. If no feature discriminates, they guess. This is a lexicographic, non-compensatory strategy. However, human execution of this strategy is highly noisy, so choice probabilities are heavily tempered by response noise (low beta) and random guessing lapses (high epsilon).

**Parameters:**
- beta: [0.0, 2.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    # We use a stable sort to preserve the original feature order in case of ties.
    ranked_features = np.argsort(validities, kind='stable')[::-1]
    
    scores = np.array([0.0, 0.0])
    # Find the first feature that discriminates between the two options
    for idx in ranked_features:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    # If no feature discriminates, the core preference is uniform
    if scores[0] == 0.0 and scores[1] == 0.0:
        p_core = np.array([0.5, 0.5])
    else:
        beta = float(parameters["beta"])
        # Softmax over the scores to introduce response noise
        z = beta * scores
        z = z - np.max(z)
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities
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
[0] rationale: This metric isolates the compensatory effect of the lower-validity features by comparing trials where the Weighted Additive (WADD) strategy agrees with the Take-The-Best (TTB) strategy against trials where WADD disagrees with TTB. The pure TTB model predicts that choice probabilities depend only on the highest discriminating feature, so the difference in the proportion of TTB-consistent choices between these two sets of trials should be zero (subject to sampling noise). The Dual-Process Hybrid model predicts a positive difference, as the WADD component systematically increases the probability of choosing the TTB winner when they agree and decreases it when they disagree.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    agree_chose_ttb = []
    disagree_chose_ttb = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        ttb_winner = -1
        for i in range(5):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        wadd_a = np.sum(a * validities)
        wadd_b = np.sum(b * validities)
        wadd_winner = 0 if wadd_a > wadd_b else 1 if wadd_b > wadd_a else -1
        
        if ttb_winner != -1 and wadd_winner != -1:
            chose_ttb = 1 if row['response'] == ttb_winner else 0
            if ttb_winner == wadd_winner:
                agree_chose_ttb.append(chose_ttb)
            else:
                disagree_chose_ttb.append(chose_ttb)
                
    if not agree_chose_ttb or not disagree_chose_ttb:
        return 0.0
        
    return float(np.mean(agree_chose_ttb) - np.mean(disagree_chose_ttb))
outcome: self_sim=-0.0209 (var=0.0117) adversary_sim=0.0071 (var=0.0101) welch_t=-0.949 p=0.3474 (N=25, alpha=0.01) -> reject

[1] rationale: The critical difference between the two models lies in their maximum possible determinism. In the Pure TTB model, the softmax inverse-temperature `beta` is capped at 2.0, meaning the core probability of choosing the TTB winner never exceeds ~0.88. In the Hybrid model, `beta_ttb` can reach 10.0, allowing for near-perfect determinism (p=1.0) before the lapse rate is applied. Consequently, the Hybrid model's distribution of subject consistencies has a much heavier right tail. By computing the 5th power of each subject's overall TTB consistency, we disproportionately reward highly consistent subjects, yielding a significantly higher expected value for the Hybrid model while maintaining a stable per-subject metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    subj_scores = []
    for subj, subj_df in data.groupby('subject_id'):
        ttb_consistent = 0
        for _, row in subj_df.iterrows():
            a = np.array(row['option_a_ratings'])
            b = np.array(row['option_b_ratings'])
            
            winner = -1
            for i in range(5):
                if a[i] > b[i]:
                    winner = 0
                    break
                elif b[i] > a[i]:
                    winner = 1
                    break
            
            if row['response'] == winner:
                ttb_consistent += 1
                
        p = ttb_consistent / len(subj_df)
        # Using the 5th power to strongly amplify the right tail of the consistency distribution
        subj_scores.append(p ** 5)
        
    return float(np.mean(subj_scores))
outcome: self_sim=0.1002 (var=0.0125) adversary_sim=0.1121 (var=0.0131) welch_t=-0.369 p=0.7134 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])\n    \n    subject_covariances = []\n    for subj, subj_df in data.groupby('subject_id'):\n        wadd_diffs = []\n        chose_a = []\n        \n        for _, row in subj_df.iterrows():\n            a = np.array(row['option_a_ratings'])\n            b = np.array(row['option_b_ratings'])\n            \n            # Determine TTB winner\n            ttb_winner = -1\n            for i in range(5):\n                if a[i] > b[i]:\n                    ttb_winner = 0\n                    break\n                elif b[i] > a[i]:\n                    ttb_winner = 1\n                    break\n            \n            # Only look at trials where TTB predicts A to hold TTB constant\n            if ttb_winner == 0:\n                wadd_a = np.sum(a * validities)\n                wadd_b = np.sum(b * validities)\n                wadd_diffs.append(wadd_a - wadd_b)\n                chose_a.append(1 if row['response'] == 0 else 0)\n                \n        if len(wadd_diffs) > 1:\n            cov = np.cov(wadd_diffs, chose_a)[0, 1]\n            subject_covariances.append(cov)\n            \n    if not subject_covariances:\n        return 0.0\n        \n    return float(np.mean(subject_covariances))",
  "rationale": "To rigorously dissociate the Dual-Process Hybrid model from pure TTB, we must isolate the compensatory WADD component while holding the non-compensatory TTB prediction strictly constant. In the pure TTB model, choice probabilities depend exclusively on the single best discriminating feature; therefore, within the subset of trials where TTB predicts Option A, the true probability of choosing A is constant, and its covariance with the continuous WADD score difference is exactly zero in expectation. In contrast, the Hybrid model's secondary WADD process continuously modulates the choice probability based on the total compensatory support. By computing the intra-subject covariance between the WADD score difference (wadd_a - wadd_b) and the binary choice (1 if A, 0 if B) strictly within TTB-A trials, we extract a pure measure of compensatory modulation. The Hybrid model predicts a strictly positive covariance, whereas the pure TTB model predicts zero."
}
```

## Usage

```json
{
  "prompt_token_count": 5220,
  "candidates_token_count": 649,
  "total_token_count": 37323
}
```
