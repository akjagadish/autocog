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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** This design quantitatively dissociates Strategy Mixture Theory from Sequential Evidence Accumulation (SEA) Theory by exploiting their different mechanisms for handling conflict between the most valid cue and the sheer number of cues. Strategy Mixture predicts a trial-by-trial coin flip (choice probability ~0.5) whenever Take-The-Best (TTB) and Tallying disagree, regardless of the size of the tallying majority. SEA, however, deterministically follows a threshold: a low threshold mimics TTB (always choosing the option favored by the best cue), while a higher threshold allows evidence to accumulate, mimicking Tallying. By varying the point at which the tallying majority overcomes the best cue across trials, SEA is forced to strongly favor one option or the other depending on its fitted threshold, whereas Strategy Mixture consistently predicts ~0.5. Control trials where both heuristics agree ensure that baseline accuracy is maintained, preventing SEA from fitting a near-zero beta to mimic the 0.5 predictions.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Strategy Mixture Theory: Instead of relying on a single heuristic, decision-makers draw from a repertoire of strategies on a trial-by-trial basis. Specifically, individuals mix between a non-compensatory lexicographic rule (Take-The-Best) and a compensatory rule (Tallying). On any given trial, a subject employs TTB with probability `p_ttb` and Tallying with probability `1 - p_ttb`. This intra-individual strategy variation naturally accounts for the aggregate ~0.50 choice proportions observed in conflict trials where the two heuristics prescribe different options, while a relatively stable mixture proportion across the population explains the low between-subject variance.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- p_ttb: [0.4, 0.6]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Strategy Mixture expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    # Strategy 1: Take-The-Best (TTB)
    order = np.argsort(validities)[::-1]
    score_ttb = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            score_ttb[0] = 1.0
            break
        elif b[idx] > a[idx]:
            score_ttb[1] = 1.0
            break
            
    # Strategy 2: Tallying (Compensatory)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    score_tally = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    p_ttb = float(parameters["p_ttb"])
    
    # Softmax for TTB
    z_ttb = beta * score_ttb
    z_ttb = z_ttb - np.max(z_ttb)
    e_ttb = np.exp(z_ttb)
    prob_ttb = e_ttb / np.sum(e_ttb)
    
    # Softmax for Tallying
    z_tally = beta * score_tally
    z_tally = z_tally - np.max(z_tally)
    e_tally = np.exp(z_tally)
    prob_tally = e_tally / np.sum(e_tally)
    
    # Mix the two strategies
    p_core = p_ttb * prob_ttb + (1.0 - p_ttb) * prob_tally
    
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
**Description:** Sequential Evidence Accumulation Theory: Decision-makers sample cues sequentially in order of validity and accumulate evidence (counts of features favoring each option). If the difference in accumulated evidence reaches an internal threshold, the search is terminated and a choice is made based on the evidence collected up to that point. If all cues are exhausted without reaching the threshold, a choice is made based on the final accumulated tallies. This single mechanistic process naturally interpolates between Take-The-Best behavior (low threshold) and Tallying behavior (high threshold), while a softmax over the naturally accumulated evidence accounts for the observed stochasticity.

**Parameters:**
- beta: [0.0, 5.0]
- epsilon: [0.0, 0.5]
- threshold: [1.0, 5.0]
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
    # Sample cues in order of validity
    order = np.argsort(validities)[::-1]
    
    threshold = float(parameters["threshold"])
    
    ev_A = 0.0
    ev_B = 0.0
    
    for idx in order:
        if a[idx] > b[idx]:
            ev_A += 1.0
        elif b[idx] > a[idx]:
            ev_B += 1.0
            
        # Check if the evidence difference reached the threshold
        if ev_A - ev_B >= threshold:
            break
        elif ev_B - ev_A >= threshold:
            break
            
    scores = np.array([ev_A, ev_B])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over accumulated evidence
    z = beta * scores
    z = z - np.max(z)
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
[0] rationale: This metric isolates the effect of the second most valid cue on choices when it aligns with the most valid cue, while Tallying still favors the opposite option. Under Strategy Mixture Theory, because TTB only looks at the first discriminating cue, both Trial 1 and Trial 3 trigger exactly the same conflict (TTB vs Tallying), yielding identical choice probabilities (~p_ttb). The difference is zero. Under Sequential Evidence Accumulation, a threshold between 1.0 and 2.0 will cause the process to stop and choose the TTB-favored option in Trial 3 (where evidence reaches +2) but not in Trial 1 (where evidence only reaches +1 before swinging back). This creates a strictly positive difference under SEA (expected ~0.50), cleanly discriminating the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Create a string representation of option A to identify specific trials
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 1: A=[1, 0, 0, 0, 0] vs B=[0, 1, 1, 1, 1]
    t1 = data[data['a_str'] == '10000']
    # Trial 3: A=[1, 1, 0, 0, 0] vs B=[0, 0, 1, 1, 1]
    t3 = data[data['a_str'] == '11000']
    
    # Trial 5: A=[0, 1, 0, 1, 1] vs B=[1, 0, 1, 0, 0]
    t5 = data[data['a_str'] == '01011']
    # Trial 6: A=[0, 0, 1, 1, 1] vs B=[1, 1, 0, 0, 0]
    t6 = data[data['a_str'] == '00111']
    
    # Calculate probability of choosing the option favored by the best cue (TTB prediction)
    # For T1 and T3, TTB favors A (response 0)
    p_A_t1 = 1.0 - t1['response'].mean() if len(t1) > 0 else 0.0
    p_A_t3 = 1.0 - t3['response'].mean() if len(t3) > 0 else 0.0
    
    # For T5 and T6, TTB favors B (response 1)
    p_B_t5 = t5['response'].mean() if len(t5) > 0 else 0.0
    p_B_t6 = t6['response'].mean() if len(t6) > 0 else 0.0
    
    # Metric is the sum of differences in choice probabilities between trials where 
    # the TTB option has 2 supporting cues vs 1 supporting cue.
    diff1 = p_A_t3 - p_A_t1
    diff2 = p_B_t6 - p_B_t5
    
    return float(diff1 + diff2)
outcome: self_sim=0.0513 (var=0.0779) adversary_sim=0.3138 (var=0.2685) welch_t=-2.230 p=0.03193 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Create a string representation of option A to identify specific trials\n    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    \n    # Trial 2: A=[0, 1, 1, 1, 0] vs B=[1, 0, 0, 0, 1] -> TTB option is B (response 1)\n    t2 = data[data['a_str'] == '01110']\n    \n    # Trial 3: A=[1, 1, 0, 0, 0] vs B=[0, 0, 1, 1, 1] -> TTB option is A (response 0)\n    t3 = data[data['a_str'] == '11000']\n    \n    # Trial 5: A=[0, 1, 0, 1, 1] vs B=[1, 0, 1, 0, 0] -> TTB option is B (response 1)\n    t5 = data[data['a_str'] == '01011']\n    \n    # Trial 6: A=[0, 0, 1, 1, 1] vs B=[1, 1, 0, 0, 0] -> TTB option is B (response 1)\n    t6 = data[data['a_str'] == '00111']\n    \n    # Calculate probability of choosing the TTB option for each trial\n    p_ttb_t2 = t2['response'].mean() if len(t2) > 0 else 0.5\n    p_ttb_t3 = 1.0 - t3['response'].mean() if len(t3) > 0 else 0.5\n    p_ttb_t5 = t5['response'].mean() if len(t5) > 0 else 0.5\n    p_ttb_t6 = t6['response'].mean() if len(t6) > 0 else 0.5\n    \n    # Group 1: Trials where TTB option reaches +2 evidence early (T3, T6)\n    # Group 2: Trials where TTB option only reaches +1 evidence early (T2, T5)\n    group1_ttb = (p_ttb_t3 + p_ttb_t6) / 2.0\n    group2_ttb = (p_ttb_t2 + p_ttb_t5) / 2.0\n    \n    return float(group1_ttb - group2_ttb)",
  "rationale": "Under Strategy Mixture Theory, Trials 2, 3, 5, and 6 are perfectly symmetric: the TTB option is favored by a TTB score difference of 1, and the Tallying option is favored by a Tallying score difference of 1 (3 cues vs 2 cues). Thus, Strategy Mixture predicts identical choice probabilities across all four trials, and the difference between Group 1 and Group 2 should be exactly 0. Under Sequential Evidence Accumulation, however, the TTB option reaches a sequential evidence difference of +2 in Trials 3 and 6, triggering early stopping for moderate thresholds, whereas in Trials 2 and 5 the evidence only reaches +1 before swinging toward the Tallying option. This structural difference causes SEA to predict a much higher probability of choosing the TTB option in Group 1 than in Group 2, creating a strong, positive contrast."
}
```

## Usage

```json
{
  "prompt_token_count": 4408,
  "candidates_token_count": 856,
  "total_token_count": 8822
}
```
