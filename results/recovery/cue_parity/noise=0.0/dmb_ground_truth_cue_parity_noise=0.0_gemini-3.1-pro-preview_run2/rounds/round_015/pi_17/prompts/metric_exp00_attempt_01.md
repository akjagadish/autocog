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

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 6: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 9: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 10: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 11: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 12: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Rationale:** This design quantitatively dissociates the advocated 'Probabilistic Heuristic Integration' theory from the competing 'Sequential Evidence Accumulation' (SCA) theory by exploiting the advocated theory's dynamic conflict-based strategy mixing. The advocated theory explicitly calculates a 'conflict' metric based on the raw tally difference between the two options; as the tally difference shrinks (approaching a tie), conflict increases, which dynamically shifts the mixing weight between the Weighted Additive (WADD) and Tallying strategies. The competing SCA theory, however, evaluates cues sequentially by validity and stops as soon as a threshold of accumulated evidence is reached, completely ignoring the raw tally of lower-validity cues if the top cues are decisive. By presenting a spectrum of trials where the top cue strongly favors one option (potentially triggering early stopping in SCA) while the opposing tally difference systematically varies from 0 to 4, we can cleanly separate the theories. SCA predicts flat, early-stopping behavior across these variations, whereas the advocated theory predicts highly non-linear shifts in choice probabilities driven by the changing conflict-modulated mixture of WADD and Tallying.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Heuristic Integration with Independent Sensitivities

**Parameters:**
- wadd_gamma: [0.0, 10.0]
- beta_tally: [0.1, 50.0]
- beta_wadd: [0.1, 50.0]
- conflict_weight: [-10.0, 10.0]
- dispersion_weight: [-10.0, 10.0]
- base_mix: [-10.0, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    wadd_gamma = float(parameters["wadd_gamma"])
    beta_tally = float(parameters["beta_tally"])
    beta_wadd = float(parameters["beta_wadd"])
    conflict_weight = float(parameters["conflict_weight"])
    dispersion_weight = float(parameters["dispersion_weight"])
    base_mix = float(parameters["base_mix"])
    epsilon = float(parameters["epsilon"])
    
    # Tallying Heuristic
    tally_a = float(np.sum(a > b))
    tally_b = float(np.sum(b > a))
    scores_tally = np.array([tally_a, tally_b])
        
    # Weighted Additive Strategy (WADD)
    weights = np.maximum(val - 0.5, 0.001) ** wadd_gamma
    wadd_a = float(np.sum((a > b) * weights))
    wadd_b = float(np.sum((b > a) * weights))
    scores_wadd = np.array([wadd_a, wadd_b])
    
    # Softmax conversion to probabilities
    z_tally = beta_tally * scores_tally
    z_tally = z_tally - np.max(z_tally)
    p_tally = np.exp(z_tally) / np.sum(np.exp(z_tally))
    
    z_wadd = beta_wadd * scores_wadd
    z_wadd = z_wadd - np.max(z_wadd)
    p_wadd = np.exp(z_wadd) / np.sum(np.exp(z_wadd))
    
    # Dynamic strategy mixing
    max_diff = len(val)
    conflict = 1.0 - (abs(tally_a - tally_b) / max_diff) if max_diff > 0 else 0.0
    dispersion = float(np.std(val))
    
    logit_wadd = base_mix + conflict_weight * conflict + dispersion_weight * dispersion
    logit_wadd = np.clip(logit_wadd, -20.0, 20.0)
    prob_wadd = 1.0 / (1.0 + np.exp(-logit_wadd))
    
    p_mix = prob_wadd * p_wadd + (1.0 - prob_wadd) * p_tally
    
    # Lapse rate
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Sequential Evidence Accumulation: Decision-makers evaluate cues sequentially in descending order of validity. Each cue provides evidence proportional to a non-linear transformation of its validity above chance. Evidence is accumulated as a running difference between the two options. If the absolute accumulated evidence exceeds a threshold, search is terminated and a choice is made based on the current evidence. If all cues are evaluated without crossing the threshold, a decision is made based on the final accumulated evidence. This allows for fast, non-compensatory decisions when top cues are highly valid, while gracefully falling back to compensatory integration when early cues are less decisive.

**Parameters:**
- theta: [0.0, 10.0]
- gamma: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale weights by transforming validity above chance, allowing better separation
    weights = np.maximum(val - 0.5, 0.001) ** gamma
    
    # Search in order of descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    E = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        if diff != 0:
            E += diff * weights[j]
            # Stop if absolute accumulated evidence reaches the threshold
            if abs(E) >= theta:
                break
            
    scores = np.array([E, -E])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
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
[0] rationale: This metric measures the difference in the probability of choosing the option supported by the single most valid cue (cue 1) when it is opposed by 0 cues versus when it is opposed by 4 lower-validity cues. In the Sequential Evidence Accumulation (SCA) model, evaluating the most valid cue first can trigger early stopping, leading to a high probability of choosing the cue-1 option regardless of the opposing cues (making the difference small). In the Probabilistic Heuristic Integration (PHI) model, the 4 opposing cues heavily skew the tally difference, dynamically increasing conflict and pulling the choice probabilities significantly away from the cue-1 option, resulting in a large difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract cue arrays
    a_cues = np.array([list(x) for x in data['option_a_ratings']])
    b_cues = np.array([list(x) for x in data['option_b_ratings']])
    
    # Identify trials where exactly one option has ONLY cue 1 (and no other cues)
    # and the other option does not have cue 1.
    a_is_cue1_only = (a_cues[:, 0] == 1) & (np.sum(a_cues, axis=1) == 1)
    b_is_cue1_only = (b_cues[:, 0] == 1) & (np.sum(b_cues, axis=1) == 1)
    
    a_is_other = (a_cues[:, 0] == 0)
    b_is_other = (b_cues[:, 0] == 0)
    
    mask_A = a_is_cue1_only & b_is_other
    mask_B = b_is_cue1_only & a_is_other
    
    valid_trials = mask_A | mask_B
    if not np.any(valid_trials):
        return 0.0
    
    df_valid = data[valid_trials].copy()
    a_cues_v = a_cues[valid_trials]
    b_cues_v = b_cues[valid_trials]
    mask_A_v = mask_A[valid_trials]
    
    # Count the number of cues in the opposing option
    k = np.where(mask_A_v, np.sum(b_cues_v, axis=1), np.sum(a_cues_v, axis=1))
    
    # Determine if the subject chose the option with cue 1
    chose_cue1 = np.where(mask_A_v, df_valid['response'] == 0, df_valid['response'] == 1)
    
    # Calculate choice probabilities at the extremes of the tally difference
    k0_mask = (k == 0)
    k4_mask = (k == 4)
    
    p0 = float(np.mean(chose_cue1[k0_mask])) if np.any(k0_mask) else 0.0
    p4 = float(np.mean(chose_cue1[k4_mask])) if np.any(k4_mask) else 0.0
    
    return p0 - p4

outcome: self_sim=0.4437 (var=0.1452) adversary_sim=0.2087 (var=0.1438) welch_t=+2.186 p=0.03376 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_cues = np.array([list(x) for x in data['option_a_ratings']])\n    b_cues = np.array([list(x) for x in data['option_b_ratings']])\n    \n    a_has_cue1 = a_cues[:, 0] == 1\n    b_has_cue1 = b_cues[:, 0] == 1\n    \n    # Filter to trials where exactly one option has cue 1\n    valid = a_has_cue1 != b_has_cue1\n    if not np.any(valid):\n        return 0.0\n        \n    df_v = data[valid]\n    a_v = a_cues[valid]\n    b_v = b_cues[valid]\n    a_h1 = a_has_cue1[valid]\n    \n    # k is the sum of cues of the option that DOES NOT have cue 1\n    k = np.where(a_h1, np.sum(b_v, axis=1), np.sum(a_v, axis=1))\n    \n    # 1 if they chose the option with cue 1, 0 otherwise\n    chose_cue1 = np.where(a_h1, df_v['response'] == 0, df_v['response'] == 1)\n    \n    ps = []\n    for val in [1, 2, 3, 4]:\n        mask = (k == val)\n        if np.sum(mask) > 0:\n            ps.append(np.mean(chose_cue1[mask]))\n            \n    if len(ps) < 2:\n        return 0.0\n        \n    # Return the standard deviation of the choice probabilities across k=1,2,3,4\n    return float(np.std(ps))",
  "rationale": "By calculating the standard deviation of the probability of choosing the cue-1 option across different numbers of opposing cues (k=1 to 4), we can cleanly distinguish the theories. SCA predicts a flat response profile (low standard deviation) because early stopping on cue 1 makes the remaining cues irrelevant. PHI predicts a highly variable response profile (high standard deviation) because the shifting tally dynamically alters the conflict and mixing weights, pulling the choice probability away from the cue-1 option as opposing cues accumulate."
}
```

## Usage

```json
{
  "prompt_token_count": 4708,
  "candidates_token_count": 593,
  "total_token_count": 8631
}
```
