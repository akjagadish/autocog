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

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 5: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Rationale:** This design quantitatively dissociates the Strategy Mixture (TTB + Tallying) model from the Weighted Additive with Diminishing Returns (WADD-DR) model. WADD-DR applies a concave transformation (alpha < 1) to the total accumulated evidence for each option, meaning that additional winning cues provide diminishing marginal returns. In contrast, the Strategy Mixture model combines a strictly non-compensatory component (TTB, which only considers the single best discriminating cue) and a strictly linear component (Tallying, which counts net feature wins without diminishing returns). By systematically pitting the highest-validity cue against coalitions of lower-validity cues of varying sizes, WADD-DR predicts compressed probability margins due to its sub-additive evidence transformation. The Mixture model, constrained by linear tallying and strict TTB, cannot mimic these specific non-linear shifts in choice probabilities.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



## ADVOCATED THEORY
**Description:** Strategy Mixture (TTB + Tallying): Decision makers do not universally adhere to a single strategy. Instead, they use a probabilistic mixture of a non-compensatory strategy (Take The Best) and a compensatory strategy (Tallying). A parameter P_TTB dictates the probability of using TTB on any given trial, while 1 - P_TTB is the probability of using Tallying. This accounts for intermediate levels of TTB-consistency and Tallying-consistency observed in empirical data across subjects and trials. The mixture captures a balance between TTB and Tallying, avoiding over-reliance on uniform guessing.

**Parameters:**
- beta: [0.1, 20.0]
- p_ttb: [0.35, 0.95]
- epsilon: [0.0, 0.2]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np

    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take The Best (TTB) Strategy
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
        
    # Tallying Strategy
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    # Normalize by n_features to keep scale comparable to TTB for the shared beta
    scores_tally = np.array([a_wins, b_wins]) / max(1.0, float(n_features))
    
    beta = float(parameters["beta"])
    
    # TTB Probabilities
    z_ttb = beta * (scores_ttb - scores_ttb.max())
    e_ttb = np.exp(z_ttb)
    p_ttb_dist = e_ttb / e_ttb.sum()
    
    # Tallying Probabilities
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally_dist = e_tally / e_tally.sum()
    
    # Mixture
    p_ttb_weight = float(parameters["p_ttb"])
    epsilon = float(parameters["epsilon"])
    
    p_core = p_ttb_weight * p_ttb_dist + (1.0 - p_ttb_weight) * p_tally_dist
    
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
**Description:** Weighted Additive with Diminishing Returns (WADD-DR): Decision-makers integrate all available discriminating cues but apply a sub-additive (concave) transformation to the accumulated evidence. Cue validities are first scaled non-linearly to reflect subjective weighting. Then, the total accumulated evidence for each option undergoes a concave transformation before being converted to choice probabilities. This naturally accounts for the dilution effect when multiple weaker cues are added, pulling choice probabilities toward 0.5, while maintaining sensitivity to the overall balance of evidence.

**Parameters:**
- gamma: [0.1, 30.0]
- alpha: [0.01, 1.0]
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale validities: subtract 0.5 so a random cue provides 0 evidence, then apply non-linear scaling
    w = np.maximum(0.0, val - 0.5) ** gamma
    
    # Accumulate evidence for each option based on discriminating cues
    diff = a - b
    ev_a = np.sum(w[diff > 0])
    ev_b = np.sum(w[diff < 0])
    
    # Apply sub-additive (concave) transformation to accumulated evidence
    # alpha < 1 yields diminishing returns for additional evidence
    ev_a_trans = (ev_a + 1e-9) ** alpha
    ev_b_trans = (ev_b + 1e-9) ** alpha
    
    # Convert transformed evidence to choice probabilities via softmax
    scores = np.array([ev_a_trans, ev_b_trans])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
[0] rationale: This metric calculates the difference in the probability of choosing Option A between Trial 3 and Trial 1. In Trial 1, Option A has only the most valid cue, while Option B has all other four cues. In Trial 3, Option A has the top two cues, and Option B has the remaining three. The Strategy Mixture model uses a linear Tallying component which heavily penalizes Option A in Trial 1 (1 vs 4 wins) compared to Trial 3 (2 vs 3 wins), leading to a large expected difference in choice probabilities. In contrast, the WADD-DR model applies a sub-additive transformation to accumulated evidence, which compresses the impact of Option B's numerous weaker cues in Trial 1, pulling the choice probabilities closer together and resulting in a different margin.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_ratings = data['option_a_ratings'].apply(tuple)
    
    # Trial 1: A=[1, 0, 0, 0, 0], B=[0, 1, 1, 1, 1]
    t1_mask = a_ratings == (1, 0, 0, 0, 0)
    # Trial 3: A=[1, 1, 0, 0, 0], B=[0, 0, 1, 1, 1]
    t3_mask = a_ratings == (1, 1, 0, 0, 0)
    
    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean()
    p_a_t3 = (data.loc[t3_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_t1) or pd.isna(p_a_t3):
        return 0.0
        
    return float(p_a_t3 - p_a_t1)
outcome: self_sim=0.0179 (var=0.0181) adversary_sim=0.0621 (var=0.0200) welch_t=-1.132 p=0.2634 (N=25, alpha=0.01) -> reject

[1] rationale: The Strategy Mixture model predicts identical choice probabilities for Trial 3 and Trial 5. In both trials, Option A wins on the most valid cue (winning the TTB component), and Option A has exactly 2 positive cues versus Option B's 3 positive cues (yielding identical scores for the Tallying component). Thus, for the Mixture model, P(A|T3) - P(A|T5) is expected to be exactly 0. In contrast, WADD-DR weights cues by their validities. In Trial 3, Option A possesses the 1st and 2nd best cues, whereas in Trial 5, Option A possesses the 1st and 4th best cues. Consequently, WADD-DR accumulates substantially higher evidence for Option A in Trial 3 than in Trial 5, predicting a strictly positive difference. This metric directly isolates this zero-versus-positive contrast.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_ratings = data['option_a_ratings'].apply(tuple)
    
    # Trial 3: A=[1, 1, 0, 0, 0], B=[0, 0, 1, 1, 1]
    t3_mask = a_ratings == (1, 1, 0, 0, 0)
    # Trial 5: A=[1, 0, 0, 1, 0], B=[0, 1, 1, 0, 1]
    t5_mask = a_ratings == (1, 0, 0, 1, 0)
    
    p_a_t3 = (data.loc[t3_mask, 'response'] == 0).mean()
    p_a_t5 = (data.loc[t5_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_t3) or pd.isna(p_a_t5):
        return 0.0
        
    return float(p_a_t3 - p_a_t5)
outcome: self_sim=-0.0263 (var=0.0195) adversary_sim=0.0453 (var=0.0267) welch_t=-1.664 p=0.1027 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_ratings = data['option_a_ratings'].apply(tuple)\n    \n    # Trial 3: A=[1, 1, 0, 0, 0], B=[0, 0, 1, 1, 1]\n    t3_mask = a_ratings == (1, 1, 0, 0, 0)\n    # Trial 4: A=[0, 1, 1, 0, 0], B=[1, 0, 0, 1, 1]\n    t4_mask = a_ratings == (0, 1, 1, 0, 0)\n    \n    p_a_t3 = (data.loc[t3_mask, 'response'] == 0).mean()\n    p_a_t4 = (data.loc[t4_mask, 'response'] == 0).mean()\n    \n    if pd.isna(p_a_t3) or pd.isna(p_a_t4):\n        return 0.0\n        \n    return float(p_a_t3 + p_a_t4)",
  "rationale": "The Strategy Mixture model computes choices as a probabilistic mixture of TTB and Tallying. For Trial 3 and Trial 4, the Tallying component yields identical scores (Option A has 2 positive cues vs Option B's 3), so P(A|Tally) is identical and strictly < 0.5. However, TTB favors A in Trial 3 and B in Trial 4. Because the TTB probabilities sum exactly to 1 across these two symmetric conditions, the sum of choice probabilities P(A|T3) + P(A|T4) analytically simplifies to `p_ttb + 2*(1-p_ttb)*P(A|Tally)`. Since Tallying favors B, this sum is strictly bounded below 1.0 (averaging ~0.65). In contrast, WADD-DR uses a sub-additive evidence transformation (alpha < 1). In Trial 3, Option A has the top two cues, strongly defeating B's weaker cues and pushing P(A|T3) near 1.0. In Trial 4, A loses the top cue but retains the 2nd and 3rd, making it a narrow loss. The sub-additivity prevents P(A|T4) from dropping to 0, keeping it near 0.4. Consequently, WADD-DR systematically predicts the sum P(A|T3) + P(A|T4) will be \u2265 1.0. This metric directly exploits this structural divergence to produce a large, reliable gap."
}
```

## Usage

```json
{
  "prompt_token_count": 4692,
  "candidates_token_count": 634,
  "total_token_count": 29658
}
```
