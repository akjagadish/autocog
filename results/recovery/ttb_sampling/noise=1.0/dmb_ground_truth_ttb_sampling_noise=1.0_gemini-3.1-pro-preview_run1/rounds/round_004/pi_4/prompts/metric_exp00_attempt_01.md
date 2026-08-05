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

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** This design quantitatively dissociates Strategy Mixture Theory from the Non-linear Weighted Additive (WADD) Theory by varying the compensatory majority while keeping the TTB/Tallying conflict constant. Strategy Mixture assumes individuals mix Take-The-Best (TTB) and Tallying trial-by-trial with a stable probability. Thus, whenever TTB and Tallying conflict, Strategy Mixture predicts a choice probability of ~0.50, regardless of the size of the Tallying majority (e.g., whether Tallying wins by 1 cue or 3 cues). In contrast, WADD integrates all cues. By presenting a series of conflict trials where the highest validity cue is pitted against an increasing number of lower validity cues, WADD is forced to predict varying choice probabilities across these trials (as the sum of weights for the compensatory option grows). Strategy Mixture uniquely predicts a flat ~0.50 choice proportion across all these conflict trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



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
**Description:** Non-linear Weighted Additive (WADD) Theory with Power Weights and Max-Normalized Validities: Decision-makers integrate all available cues simultaneously in a compensatory manner. The subjective weight assigned to each cue is determined by a power transformation of its objective validity, scaled by a parameter gamma. To ensure gamma operates consistently across experiments with varying validity scales, the objective validities are first normalized by their maximum value. This stable, bounded non-linear mapping ensures that cues with zero validity receive zero subjective weight, while allowing the model to naturally balance the top cue against the remaining cues on conflict trials. A temperature-parameterized softmax choice rule is used to smoothly control stochasticity.

**Parameters:**
- tau: [0.01, 5.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 20.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    tau = float(parameters["tau"])
    epsilon = float(parameters["epsilon"])
    
    # Max-normalize validities to make gamma scale-invariant across experiments
    v_norm = validities / np.max(validities)
    
    # Power transformation of normalized validities to subjective weights
    weights = v_norm ** gamma
    weights = weights / np.sum(weights)
    
    # Compute weighted sum of features for each option
    scores = stim @ weights
    
    # Standard softmax choice rule with temperature
    z = scores / tau
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
[0] rationale: This metric calculates the difference in the probability of choosing Option B between Trial 3 (where Option B has 4 compensatory cues) and Trial 1 (where Option B has only 2 compensatory cues). According to Strategy Mixture Theory, the probability of choosing the Tallying-favored option (Option B) is largely determined by the stable mixing probability `1 - p_ttb`, because Tallying simply counts wins and strongly favors Option B in both trials, leading to a small difference (near 0). In contrast, the Non-linear Weighted Additive (WADD) Theory integrates all cues, so adding two more valid cues to Option B in Trial 3 significantly increases its subjective value and thus the probability of it being chosen, yielding a much larger positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    t1 = data[(data['A_tuple'] == (1, 0, 0, 0, 0)) & (data['B_tuple'] == (0, 1, 1, 0, 0))]
    t3 = data[(data['A_tuple'] == (1, 0, 0, 0, 0)) & (data['B_tuple'] == (0, 1, 1, 1, 1))]
    
    if len(t1) == 0 or len(t3) == 0:
        return 0.0
        
    p_b_t1 = t1['response'].mean()
    p_b_t3 = t3['response'].mean()
    
    return float(p_b_t3 - p_b_t1)
outcome: self_sim=-0.0017 (var=0.0465) adversary_sim=0.0833 (var=0.0331) welch_t=-1.507 p=0.1387 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    def is_conflict(row):\n        a = row['option_a_ratings']\n        b = row['option_b_ratings']\n        ttb_pref = None\n        for i in range(len(a)):\n            if a[i] > b[i]:\n                ttb_pref = 0\n                break\n            elif b[i] > a[i]:\n                ttb_pref = 1\n                break\n        a_wins = sum(1 for i in range(len(a)) if a[i] > b[i])\n        b_wins = sum(1 for i in range(len(a)) if b[i] > a[i])\n        if a_wins > b_wins:\n            tally_pref = 0\n        elif b_wins > a_wins:\n            tally_pref = 1\n        else:\n            tally_pref = None\n        return ttb_pref is not None and tally_pref is not None and ttb_pref != tally_pref\n\n    conflict_mask = data.apply(is_conflict, axis=1)\n    conflict_data = data[conflict_mask].copy()\n    \n    if len(conflict_data) == 0:\n        return 0.0\n        \n    def get_ttb_pref(row):\n        a = row['option_a_ratings']\n        b = row['option_b_ratings']\n        for i in range(len(a)):\n            if a[i] > b[i]:\n                return 0\n            elif b[i] > a[i]:\n                return 1\n        return 0\n\n    conflict_data['ttb_pref'] = conflict_data.apply(get_ttb_pref, axis=1)\n    conflict_data['ttb_chosen'] = (conflict_data['response'] == conflict_data['ttb_pref']).astype(int)\n    \n    subj_means = conflict_data.groupby('subject_id')['ttb_chosen'].mean()\n    return float((subj_means - 0.5).abs().mean())",
  "rationale": "This metric isolates all 'conflict trials' where the Take-The-Best (TTB) strategy and the Tallying strategy prefer different options. For Strategy Mixture Theory, subjects mix between TTB and Tallying trial-by-trial with a stable probability p_ttb in [0.4, 0.6]. Therefore, the proportion of times they choose the TTB-favored option will tightly cluster around 0.5, resulting in an absolute deviation from 0.5 near zero. In contrast, the Non-linear WADD Theory integrates all cues via a power function on normalized validities. Depending on their gamma parameter, WADD subjects will either heavily favor the top cue (behaving like deterministic TTB, proportion near 1.0) or heavily favor the compensatory majority (proportion near 0.0). Consequently, WADD predicts extreme choice proportions on these conflict trials, yielding an absolute deviation from 0.5 that is much closer to 0.4 or 0.5. This creates a massive, stable gap in the expected metric value between the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 4073,
  "candidates_token_count": 768,
  "total_token_count": 9632
}
```
