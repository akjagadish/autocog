# metric_exp00_attempt_03

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

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 7: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 8: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 9: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 10: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]

**Rationale:** To maximally distinguish pure WADD (THEORY 1) from the adaptive TTB/WADD mixture (THEORY 2), this design targets the explicit conflict mechanism in THEORY 2. THEORY 2 uses an independent linear combination of the top discriminating cue's validity (v_top) and the sum of opposing cues' validities (v_opp_sum) to determine the probability of switching from TTB to WADD. By systematically varying v_top and the number/validity of opposing cues, we create pairs where v_opp_sum changes substantially (driving strategy shifts in THEORY 2) while the non-linear WADD score differences (THEORY 1) follow a different, smoother gradient. This breaks the collinearity between pure compensatory evaluation and the explicit conflict-triggered strategy switch.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** People use a Weighted Additive (WADD) strategy to make decisions, where each feature is weighted by its subjective importance (a non-linear transformation of its validity). The total score for an option is the sum of the weighted features it possesses. This compensatory mechanism allows multiple lower-validity cues to sometimes outweigh a single high-validity cue, naturally interpolating between Take The Best (when validity differences are heavily magnified) and Tallying (when validities are ignored). Response noise and lapses account for stochasticity in choice.

**Parameters:**
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 30.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Transform validities into subjective weights using a power function.
    # gamma = 0 yields equal weights (Tallying); gamma -> inf yields lexicographic weights (Take The Best).
    weights = val ** gamma
    
    # Compute weighted additive scores for each option
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## COMPETING THEORY
**Description:** Decision-makers use an adaptive toolbox of strategies, dynamically selecting between Take The Best (TTB) and Weighted Additive (WADD) evaluation on a trial-by-trial basis. The probability of using TTB over WADD depends on the conflict between the cues: when the top discriminating cue is much stronger than the opposing cues, TTB dominates; when multiple opposing cues rival the top cue's validity, the decision-maker is more likely to switch to the compensatory WADD strategy. By applying independent scaling weights to the top cue's validity and the sum of opposing validities, the model can penalize a large number of weak opposing cues, keeping TTB dominant in those cases while still allowing WADD to differentiate between strong compensatory profiles.

**Parameters:**
- theta_top: [0.0, 20.0]
- theta_opp: [0.0, 20.0]
- bias_ttb: [-10.0, 10.0]
- gamma: [0.0, 10.0]
- beta_ttb: [0.1, 20.0]
- beta_wadd: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # --- Take The Best (TTB) ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    v_top = 0.0
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            v_top = val[j]
            break
        if b[j] > a[j]:
            winner_ttb = 1
            v_top = val[j]
            break
            
    if winner_ttb is None:
        p_ttb_core = np.array([0.5, 0.5])
        p_ttb_weight = 0.5
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb_core = e_ttb / np.sum(e_ttb)
        
        # Dynamic mixture weight based on conflict
        v_opp_sum = 0.0
        for j in range(len(val)):
            if winner_ttb == 0 and b[j] > a[j]:
                v_opp_sum += val[j]
            elif winner_ttb == 1 and a[j] > b[j]:
                v_opp_sum += val[j]
                
        theta_top = float(parameters["theta_top"])
        theta_opp = float(parameters["theta_opp"])
        bias_ttb = float(parameters["bias_ttb"])
        z_mix = theta_top * v_top - theta_opp * v_opp_sum + bias_ttb
        p_ttb_weight = 1.0 / (1.0 + np.exp(-z_mix))
        
    # --- Weighted Additive (WADD) ---
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores_wadd = np.array([score_a, score_b])
    
    beta_wadd = float(parameters["beta_wadd"])
    z_wadd = beta_wadd * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd_core = e_wadd / np.sum(e_wadd)
    
    # --- Strategy Mixture ---
    p_mixed = p_ttb_weight * p_ttb_core + (1.0 - p_ttb_weight) * p_wadd_core
    
    # --- Lapse Noise ---
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
    
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
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
[0] rationale: This metric contrasts the probability of choosing Option B in Trial 1 versus Trial 4. In both trials, Option A has the strongest cue (validity 0.95). In Trial 1, Option B has two strong opposing cues (0.85, 0.75), yielding a non-linear WADD score that is typically higher than Option B's score in Trial 4, which has three weaker opposing cues (0.65, 0.55, 0.50). Therefore, pure WADD strongly predicts P(B|Trial 1) > P(B|Trial 4). However, the Competing Theory's strategy-switch mechanism relies on the linear sum of opposing validities. Trial 4 has a higher linear sum of opposing validities (1.70) than Trial 1 (1.60), causing the Competing Theory to switch to the compensatory WADD strategy more often in Trial 4. This dampens or reverses the difference, making P(B|Trial 1) - P(B|Trial 4) a strong discriminator between the non-linear compensatory evaluation of WADD and the linear sum-based conflict resolution of the mixture model.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t1_mask = (data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0, 0))) & (data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0, 0)))
    t4_mask = (data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0, 0))) & (data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1, 1)))
    
    p_b_t1 = data.loc[t1_mask, 'response'].mean()
    p_b_t4 = data.loc[t4_mask, 'response'].mean()
    
    if pd.isna(p_b_t1) or pd.isna(p_b_t4):
        return 0.0
    
    return float(p_b_t1 - p_b_t4)
outcome: self_sim=0.0889 (var=0.0533) adversary_sim=0.1222 (var=0.0821) welch_t=-0.453 p=0.6528 (N=25, alpha=0.01) -> reject

[1] rationale: This metric contrasts the probability of choosing Option B in Trial 10 versus Trial 2. In Trial 2, A has cue 1 (0.95) and B has cues 2 and 4 (0.85, 0.65). In Trial 10, A additionally gains cue 3 (0.75) while B additionally gains cue 5 (0.55). For pure WADD, A gains a much stronger cue than B does, widening the utility gap in favor of A. Thus, WADD strongly predicts P(B|Trial 10) < P(B|Trial 2). However, under the Competing Theory's mixture model, the sum of opposing validities increases significantly from 1.50 in Trial 2 to 2.05 in Trial 10, which triggers a substantial increase in the probability of switching away from pure Take-The-Best (which strictly chooses A) to a compensatory strategy. This conflict-induced strategy shift causes the mixture model to predict a higher relative rate of choosing B in Trial 10 than WADD does, yielding a starkly different value for the difference P(B|Trial 10) - P(B|Trial 2).
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Trial 2: A=[1, 0, 0, 0, 0, 0], B=[0, 1, 0, 1, 0, 0]
    t2_mask = (data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0, 0))) & (data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 0, 1, 0, 0)))
    # Trial 10: A=[1, 0, 1, 0, 0, 0], B=[0, 1, 0, 1, 1, 0]
    t10_mask = (data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 1, 0, 0, 0))) & (data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 0, 1, 1, 0)))
    
    p_b_t2 = data.loc[t2_mask, 'response'].mean()
    p_b_t10 = data.loc[t10_mask, 'response'].mean()
    
    if pd.isna(p_b_t2) or pd.isna(p_b_t10):
        return 0.0
    
    return float(p_b_t10 - p_b_t2)
outcome: self_sim=-0.0267 (var=0.0526) adversary_sim=-0.1667 (var=0.0772) welch_t=+1.943 p=0.0581 (N=25, alpha=0.01) -> reject

[2] rationale: By aggregating over multiple trials, this metric significantly reduces between-subject variance. It contrasts the probability of choosing Option B when Option A relies on the strongest possible cue (Trials 1-4) versus when Option A relies on a weaker top cue (Trials 5-9). Pure WADD predicts a smooth, compensatory increase in P(B) as A's validity drops. However, the Competing Theory's mixture model predicts a sharp, compounded increase: dropping A's validity not only reduces A's compensatory score but also drastically decreases 'v_top', triggering a massive structural shift away from Take-The-Best (which strictly favors A) toward the compensatory WADD strategy. This non-linear strategy switch in the mixture model creates a much larger aggregate difference than pure WADD.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # A has a single cue with validity 0.95 in Trials 1-4
    a_high = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0, 0))
    # A has a single cue with lower validity (0.85 or 0.75) in Trials 5-9
    a_low = data['option_a_ratings'].apply(lambda x: tuple(x) in [(0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0)])
    
    p_b_high = data.loc[a_high, 'response'].mean()
    p_b_low = data.loc[a_low, 'response'].mean()
    
    if pd.isna(p_b_high) or pd.isna(p_b_low):
        return 0.0
        
    return float(p_b_low - p_b_high)
outcome: self_sim=0.1497 (var=0.0223) adversary_sim=0.0500 (var=0.0123) welch_t=+2.682 p=0.01025 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Count the number of positive cues for Option B\n    b_cues = data['option_b_ratings'].apply(sum)\n    \n    # Group into trials where Option B has 3 cues vs 2 cues\n    p_b_3cues = data.loc[b_cues == 3, 'response'].mean()\n    p_b_2cues = data.loc[b_cues == 2, 'response'].mean()\n    \n    if pd.isna(p_b_3cues) or pd.isna(p_b_2cues):\n        return 0.0\n        \n    return float(p_b_3cues - p_b_2cues)",
  "rationale": "This metric contrasts the probability of choosing Option B when it is supported by 3 cues versus when it is supported by 2 cues. Under the pure Weighted Additive (WADD) theory, subjective weights are a non-linear power function of validities, meaning that adding a third weak cue often contributes very little marginal value compared to having two strong cues. Conversely, the Competing Theory's strategy-selection mechanism relies on the linear sum of opposing validities (v_opp_sum). Three cues, even if individually weaker, produce a substantially larger linear sum than two cues, reliably triggering a shift away from Take-The-Best (which strictly favors Option A) toward the compensatory strategy. Thus, the Competing Theory predicts a much sharper increase in P(B) when moving from 2 to 3 opposing cues than pure WADD does. By aggregating over all trials, this metric ensures very low between-subject variance, boosting statistical power."
}
```

## Usage

```json
{
  "prompt_token_count": 5910,
  "candidates_token_count": 391,
  "total_token_count": 8373
}
```
