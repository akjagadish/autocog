# metric_exp02_attempt_02

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
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 8: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 9: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 10: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]

**Rationale:** To maximally distinguish THEORY 1 (dynamic TTB/WADD mixture based on explicit cue conflict) from THEORY 2 (pure WADD model), we manipulate the identity of the top discriminating cue and the number/validity of opposing cues to decouple the pure WADD score difference from the explicit conflict signal. In THEORY 1, the probability of using TTB over WADD depends on the difference between the top cue's validity and the sum of the opposing cues' validities. By creating pairs where the WADD score difference is comparable but the absolute validities (v_top and v_opp_sum) vary significantly, THEORY 1 predicts abrupt shifts in choice probabilities due to strategy switching, whereas THEORY 2 predicts smoother, monotonic variations based on the non-linear compensatory sum.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
[0] rationale: This metric calculates the variance of the Take-The-Best (TTB) choice proportions across the 10 unique trial types, averaged over subjects. THEORY 1 posits an abrupt, threshold-based switch between TTB and WADD driven by the conflict signal, which produces more extreme choice probabilities (closer to 0 or 1) for different trial types, leading to higher variance across trial types. THEORY 2 uses a purely compensatory mechanism, which generally yields smoother, more graded choice probabilities across the varying levels of cue conflict, resulting in lower variance in TTB match rates across the trials.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    ttb_matches = []
    trial_ids = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # Determine TTB match (cues are ordered by validity 0 -> 4)
        match = 0.5
        for i in range(len(a)):
            if a[i] > b[i]:
                match = 1.0 if resp == 0 else 0.0
                break
            elif b[i] > a[i]:
                match = 1.0 if resp == 1 else 0.0
                break
        ttb_matches.append(match)
        
        # Create trial ID
        trial_ids.append("".join(map(str, a)) + "_" + "".join(map(str, b)))
        
    df_eval = data.assign(ttb_match=ttb_matches, trial_id=trial_ids)
    
    subj_vars = []
    for subj, subj_df in df_eval.groupby('subject_id'):
        trial_means = subj_df.groupby('trial_id')['ttb_match'].mean()
        if len(trial_means) > 1:
            subj_vars.append(trial_means.var(ddof=0))
            
    if not subj_vars:
        return 0.0
        
    return float(np.mean(subj_vars))
outcome: self_sim=0.0373 (var=0.0013) adversary_sim=0.0343 (var=0.0005) welch_t=+0.361 p=0.72 (N=25, alpha=0.01) -> reject

[1] rationale: This metric calculates the difference in the probability of choosing Option A between Trial 2 (low conflict) and Trial 9 (high conflict). In both trials, the most valid cue (validity 0.95) favors Option A. However, Trial 9 presents a much stronger set of opposing cues for Option B. THEORY 1 predicts a sharp strategy shift: subjects will likely use Take The Best (TTB) in Trial 2 (choosing A) but switch to Weighted Additive (WADD) in Trial 9 due to the high conflict (choosing B), leading to a large difference. THEORY 2, using a purely compensatory WADD mechanism, evaluates both trials smoothly based on subjective weights, generally resulting in a smaller difference in choice probability.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        t2 = subj_df[(subj_df['option_a_ratings'].apply(tuple) == (1,0,0,0,0)) & 
                     (subj_df['option_b_ratings'].apply(tuple) == (0,0,0,1,1))]
        t9 = subj_df[(subj_df['option_a_ratings'].apply(tuple) == (1,0,0,1,0)) & 
                     (subj_df['option_b_ratings'].apply(tuple) == (0,1,1,0,1))]
        
        if len(t2) > 0 and len(t9) > 0:
            p2 = (t2['response'] == 0).mean()
            p9 = (t9['response'] == 0).mean()
            diffs.append(p2 - p9)
            
    if not diffs:
        return 0.0
    return float(np.mean(diffs))
outcome: self_sim=0.1200 (var=0.0765) adversary_sim=0.0533 (var=0.0391) welch_t=+0.980 p=0.3323 (N=25, alpha=0.01) -> reject

[2] rationale: This metric exploits a perfect ordinal dissociation between the two theories by comparing choice probabilities across two specific pairs of trials: (Trial 1 vs. Trial 9) and (Trial 2 vs. Trial 6). 

In THEORY 1, the probability of using Take The Best (TTB) depends on the sum of opposing validities. Trial 1 has a lower opposing sum than Trial 9, and Trial 2 has a lower opposing sum than Trial 6. Because lower conflict triggers higher TTB usage (which favors Option A), THEORY 1 predicts P(A|T1) > P(A|T9) and P(A|T2) > P(A|T6), making the overall metric strongly positive.

In THEORY 2 (pure WADD), the choice depends on the weighted score difference. Mathematically, for ANY validity weighting parameter gamma > 0, the WADD score difference strictly favors Option A more in Trial 9 than in Trial 1, and more in Trial 6 than in Trial 2. Therefore, THEORY 2 strictly predicts P(A|T1) < P(A|T9) and P(A|T2) < P(A|T6), guaranteeing a negative metric.

By summing these two differences, we capture a targeted, structurally guaranteed divergence where the two theories produce values with opposite signs, ensuring a highly significant t-test.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        a_tuples = subj_df['option_a_ratings'].apply(tuple)
        b_tuples = subj_df['option_b_ratings'].apply(tuple)
        
        # Trial 1 vs Trial 9
        mask_t1 = (a_tuples == (1,0,0,0,0)) & (b_tuples == (0,1,1,0,0))
        mask_t9 = (a_tuples == (1,0,0,1,0)) & (b_tuples == (0,1,1,0,1))
        
        # Trial 2 vs Trial 6
        mask_t2 = (a_tuples == (1,0,0,0,0)) & (b_tuples == (0,0,0,1,1))
        mask_t6 = (a_tuples == (1,1,0,0,0)) & (b_tuples == (0,0,1,1,1))
        
        df_t1 = subj_df[mask_t1]
        df_t9 = subj_df[mask_t9]
        df_t2 = subj_df[mask_t2]
        df_t6 = subj_df[mask_t6]
        
        if len(df_t1) > 0 and len(df_t9) > 0 and len(df_t2) > 0 and len(df_t6) > 0:
            p_a_t1 = (df_t1['response'] == 0).mean()
            p_a_t9 = (df_t9['response'] == 0).mean()
            p_a_t2 = (df_t2['response'] == 0).mean()
            p_a_t6 = (df_t6['response'] == 0).mean()
            
            # Calculate sum of differences
            diff = (p_a_t1 - p_a_t9) + (p_a_t2 - p_a_t6)
            diffs.append(diff)
            
    if not diffs:
        return 0.0
        
    return float(np.mean(diffs))
outcome: self_sim=-0.0667 (var=0.0889) adversary_sim=-0.0044 (var=0.0523) welch_t=-0.828 p=0.4121 (N=25, alpha=0.01) -> reject

[3] rationale: This metric measures the difference in the probability of choosing Option A between Trial 10 and Trial 9. In Trial 10, both the Take The Best (TTB) heuristic and the linear Weighted Additive (WADD) strategy favor Option A. In Trial 9, TTB still favors Option A, but the opposing cues are much stronger, meaning WADD heavily favors Option B. THEORY 1 predicts a sharp drop in P(A) in Trial 9 because the high conflict explicitly triggers a switch to WADD, making the difference (Trial 10 - Trial 9) large. THEORY 2, using a pure compensatory mechanism with non-linear weights, evaluates both trials smoothly and can often favor Option A in Trial 9 as well (if the validity exponent gamma is high), resulting in a systematically different magnitude for this gap.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        a_tuples = subj_df['option_a_ratings'].apply(tuple)
        b_tuples = subj_df['option_b_ratings'].apply(tuple)
        
        # Trial 10
        mask_t10 = (a_tuples == (0,1,0,1,0)) & (b_tuples == (0,0,1,0,1))
        # Trial 9
        mask_t9 = (a_tuples == (1,0,0,1,0)) & (b_tuples == (0,1,1,0,1))
        
        df_t10 = subj_df[mask_t10]
        df_t9 = subj_df[mask_t9]
        
        if len(df_t10) > 0 and len(df_t9) > 0:
            p_a_t10 = (df_t10['response'] == 0).mean()
            p_a_t9 = (df_t9['response'] == 0).mean()
            diffs.append(p_a_t10 - p_a_t9)
            
    if not diffs:
        return 0.0
        
    return float(np.mean(diffs))
outcome: self_sim=0.1556 (var=0.1077) adversary_sim=-0.0756 (var=0.0802) welch_t=+2.666 p=0.01049 (N=25, alpha=0.01) -> reject

[4] rationale: This metric calculates the within-subject difference in Take The Best (TTB) adherence between 'low conflict' trials (where opposing cues are weak) and 'high conflict' trials (where opposing cues are strong). THEORY 1 predicts a sharp, threshold-like drop in TTB adherence for high conflict trials because the conflict signal explicitly shifts the mixture probability towards WADD. THEORY 2, using a pure compensatory mechanism, predicts a smoother gradation. By aggregating across multiple designated low/high conflict trials, we obtain a stable estimate of this strategy shift magnitude per subject, maximizing the chance to distinguish the two theories' structural predictions.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        a_tuples = subj_df['option_a_ratings'].apply(tuple)
        b_tuples = subj_df['option_b_ratings'].apply(tuple)
        
        # Low conflict trials (opposing cues are weak)
        m_t4 = (a_tuples == (1,0,0,0,0)) & (b_tuples == (0,0,0,1,1))
        m_t7 = (a_tuples == (0,1,0,0,0)) & (b_tuples == (0,0,0,1,1))
        m_t12 = (a_tuples == (0,0,0,1,1)) & (b_tuples == (1,0,0,0,0))
        
        # High conflict trials (opposing cues are strong)
        m_t1 = (a_tuples == (1,0,0,0,0)) & (b_tuples == (0,1,1,0,0))
        m_t8 = (a_tuples == (1,1,0,0,0)) & (b_tuples == (0,0,1,1,1))
        m_t10 = (a_tuples == (1,0,1,0,0)) & (b_tuples == (0,1,0,1,1))
        m_t11 = (a_tuples == (0,1,1,0,0)) & (b_tuples == (1,0,0,0,0))
        
        # Extract TTB matches for low conflict
        ttb_low = []
        ttb_low.extend((subj_df[m_t4]['response'] == 0).tolist())
        ttb_low.extend((subj_df[m_t7]['response'] == 0).tolist())
        ttb_low.extend((subj_df[m_t12]['response'] == 1).tolist())
        
        # Extract TTB matches for high conflict
        ttb_high = []
        ttb_high.extend((subj_df[m_t1]['response'] == 0).tolist())
        ttb_high.extend((subj_df[m_t8]['response'] == 0).tolist())
        ttb_high.extend((subj_df[m_t10]['response'] == 0).tolist())
        ttb_high.extend((subj_df[m_t11]['response'] == 1).tolist())
        
        if len(ttb_low) > 0 and len(ttb_high) > 0:
            diffs.append(np.mean(ttb_low) - np.mean(ttb_high))
            
    if not diffs:
        return 0.0
        
    return float(np.mean(diffs))
outcome: self_sim=0.0777 (var=0.0238) adversary_sim=0.0231 (var=0.0170) welch_t=+1.352 p=0.183 (N=25, alpha=0.01) -> reject

[5] rationale: This metric exploits a strict ordinal dissociation between the two theories by comparing a set of 'low conflict' trials (X) to 'high conflict' trials (Y). In all selected trials, the Take The Best (TTB) heuristic favors Option A, but the Weighted Additive (WADD) score favors Option B. 

For THEORY 2 (pure WADD), we mathematically guarantee that the WADD score difference strictly favors Option A more in the Y trials than in the X trials for ANY validity exponent gamma > 0 (e.g., Score(T8) - Score(T4) = 0.85^g - 0.75^g > 0). Therefore, WADD strictly predicts P(A|Y) > P(A|X), yielding a negative metric.

For THEORY 1, the choice is driven by the conflict signal. The X trials have weak opposing cues (mean opposing sum = 1.26), so the decision-maker relies heavily on TTB, choosing A. The Y trials have very strong opposing cues (mean opposing sum = 2.00), crossing the conflict threshold and forcing a switch to the WADD strategy, which favors B. Thus, THEORY 1 strongly predicts P(A|X) > P(A|Y), yielding a positive metric. Aggregating over these specific subsets minimizes within-subject variance while locking in the structural divergence.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        a_tuples = subj_df['option_a_ratings'].apply(tuple)
        b_tuples = subj_df['option_b_ratings'].apply(tuple)
        
        # X trials: Low conflict (weak opposing cues)
        m4 = (a_tuples == (1,0,0,0,0)) & (b_tuples == (0,0,0,1,1))
        m5 = (a_tuples == (0,1,0,0,0)) & (b_tuples == (0,0,1,1,0))
        m6 = (a_tuples == (0,1,0,0,0)) & (b_tuples == (0,0,1,0,1))
        m7 = (a_tuples == (0,1,0,0,0)) & (b_tuples == (0,0,0,1,1))
        m9 = (a_tuples == (0,0,1,0,0)) & (b_tuples == (0,0,0,1,1))
        
        mask_x = m4 | m5 | m6 | m7 | m9
        
        # Y trials: High conflict (strong opposing cues)
        m8 = (a_tuples == (1,1,0,0,0)) & (b_tuples == (0,0,1,1,1))
        m10 = (a_tuples == (1,0,1,0,0)) & (b_tuples == (0,1,0,1,1))
        
        mask_y = m8 | m10
        
        df_x = subj_df[mask_x]
        df_y = subj_df[mask_y]
        
        if len(df_x) > 0 and len(df_y) > 0:
            p_a_x = (df_x['response'] == 0).mean()
            p_a_y = (df_y['response'] == 0).mean()
            diffs.append(p_a_x - p_a_y)
            
    if not diffs:
        return 0.0
        
    return float(np.mean(diffs))
outcome: self_sim=-0.0590 (var=0.0414) adversary_sim=-0.1445 (var=0.0232) welch_t=+1.682 p=0.09953 (N=25, alpha=0.01) -> reject

[6] rationale: This metric exploits a perfect ordinal dissociation between the two theories by comparing choice probabilities across two specific pairs of trials: (Trial 3 vs. Trial 8) and (Trial 2 vs. Trial 10). In THEORY 2 (pure WADD), mathematically, for ANY validity weighting parameter gamma > 0, the WADD score difference strictly favors Option A more in Trial 8 than in Trial 3, and more in Trial 10 than in Trial 2. Therefore, THEORY 2 strictly predicts P(A|T3) < P(A|T8) and P(A|T2) < P(A|T10), guaranteeing a negative metric. In THEORY 1, the probability of using Take The Best (TTB) depends on the sum of opposing validities. Trial 3 has a lower opposing sum than Trial 8, and Trial 2 has a lower opposing sum than Trial 10. Because lower conflict triggers higher TTB usage (which heavily favors Option A in all these trials), THEORY 1 predicts P(A|T3) > P(A|T8) and P(A|T2) > P(A|T10), making the overall metric positive. This guarantees a structural divergence.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        a_tuples = subj_df['option_a_ratings'].apply(tuple)
        b_tuples = subj_df['option_b_ratings'].apply(tuple)
        
        # Pair 1: Trial 3 vs Trial 8
        mask_t3 = (a_tuples == (1,0,0,0,0)) & (b_tuples == (0,0,1,1,0))
        mask_t8 = (a_tuples == (1,1,0,0,0)) & (b_tuples == (0,0,1,1,1))
        
        # Pair 2: Trial 2 vs Trial 10
        mask_t2 = (a_tuples == (1,0,0,0,0)) & (b_tuples == (0,1,0,1,0))
        mask_t10 = (a_tuples == (1,0,1,0,0)) & (b_tuples == (0,1,0,1,1))
        
        df_t3 = subj_df[mask_t3]
        df_t8 = subj_df[mask_t8]
        df_t2 = subj_df[mask_t2]
        df_t10 = subj_df[mask_t10]
        
        if len(df_t3) > 0 and len(df_t8) > 0 and len(df_t2) > 0 and len(df_t10) > 0:
            p_a_t3 = (df_t3['response'] == 0).mean()
            p_a_t8 = (df_t8['response'] == 0).mean()
            p_a_t2 = (df_t2['response'] == 0).mean()
            p_a_t10 = (df_t10['response'] == 0).mean()
            
            # Calculate sum of differences
            diff = (p_a_t3 - p_a_t8) + (p_a_t2 - p_a_t10)
            diffs.append(diff)
            
    if not diffs:
        return 0.0
        
    return float(np.mean(diffs))
outcome: self_sim=-0.0925 (var=0.2518) adversary_sim=-0.1350 (var=0.1443) welch_t=+0.338 p=0.7372 (N=25, alpha=0.01) -> reject

[7] rationale: This metric evaluates the proportion of specific trial pairs (X, Y) where a subject chooses Option A more frequently in X than in Y. The pairs are carefully chosen such that in THEORY 2 (pure WADD), the compensatory score mathematically guarantees that Option A is favored MORE in Y than in X for ANY validity exponent gamma > 0 (e.g., Score(t8) > Score(t4)). Thus, THEORY 2 strictly predicts P(A|X) <= P(A|Y), yielding an expected score <= 0.5 (driven only by noise or ties). In contrast, THEORY 1 posits a conflict-driven strategy switch. Trials in Y have a much higher sum of opposing validities than X, explicitly triggering a switch from Take The Best (which heavily favors A) to WADD. For subjects whose WADD strategy has a low gamma (favoring the option with more cues, i.e., Option B), this switch causes a sharp drop in P(A), making P(A|X) > P(A|Y) strongly true. This structural mechanism in THEORY 1 pushes the expected score significantly above 0.5. By using a bounded proportion score per subject, we strictly control between-subject variance while capturing the ordinal dissociation.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    scores = []
    for subj, subj_df in data.groupby('subject_id'):
        a_tuples = subj_df['option_a_ratings'].apply(tuple)
        b_tuples = subj_df['option_b_ratings'].apply(tuple)
        
        m_t3 = (a_tuples == (1,0,0,0,0)) & (b_tuples == (0,0,1,1,0))
        m_t4 = (a_tuples == (1,0,0,0,0)) & (b_tuples == (0,0,0,1,1))
        m_t2 = (a_tuples == (1,0,0,0,0)) & (b_tuples == (0,1,0,1,0))
        m_t8 = (a_tuples == (1,1,0,0,0)) & (b_tuples == (0,0,1,1,1))
        m_t10 = (a_tuples == (1,0,1,0,0)) & (b_tuples == (0,1,0,1,1))
        
        def get_p_a(mask):
            df_m = subj_df[mask]
            if len(df_m) == 0:
                return np.nan
            return (df_m['response'] == 0).mean()
            
        p_t3 = get_p_a(m_t3)
        p_t4 = get_p_a(m_t4)
        p_t2 = get_p_a(m_t2)
        p_t8 = get_p_a(m_t8)
        p_t10 = get_p_a(m_t10)
        
        subj_score = 0
        valid = 0
        
        pairs = [(p_t3, p_t8), (p_t4, p_t8), (p_t2, p_t10)]
        for px, py in pairs:
            if not np.isnan(px) and not np.isnan(py):
                valid += 1
                if px > py:
                    subj_score += 1.0
                elif px == py:
                    subj_score += 0.5
                    
        if valid > 0:
            scores.append(subj_score / valid)
            
    if not scores:
        return 0.0
        
    return float(np.mean(scores))
outcome: self_sim=0.4600 (var=0.1062) adversary_sim=0.4533 (var=0.0823) welch_t=+0.077 p=0.9391 (N=25, alpha=0.01) -> reject

[8] rationale: This metric calculates the within-subject difference in the probability of choosing Option A between '1vs1' trials (where Option B has only 1 cue) and '1vs2' trials (where Option B has 2 cues). In THEORY 1, the presence of a second opposing cue pushes the sum of opposing validities across the conflict threshold, causing an abrupt switch from Take-The-Best (which strongly favors A) to WADD (which may favor B), leading to a large drop in P(A). In THEORY 2, the compensatory mechanism is smooth, meaning the drop in P(A) is more graded. By aggregating across all 4 '1vs1' trials and all 6 '1vs2' trials, we drastically reduce single-trial measurement noise, yielding a highly stable per-subject estimate of this strategy-shift magnitude, maximizing the statistical power to distinguish the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        b_tuples = subj_df['option_b_ratings'].apply(tuple)
        
        # 1vs1 trials: Option B has exactly 1 cue (Trials 1, 5, 8, 10)
        mask_1vs1 = b_tuples.apply(lambda x: sum(x) == 1)
        # 1vs2 trials: Option B has exactly 2 cues (Trials 2, 3, 4, 6, 7, 9)
        mask_1vs2 = b_tuples.apply(lambda x: sum(x) == 2)
        
        df_1vs1 = subj_df[mask_1vs1]
        df_1vs2 = subj_df[mask_1vs2]
        
        if len(df_1vs1) > 0 and len(df_1vs2) > 0:
            p_a_1vs1 = (df_1vs1['response'] == 0).mean()
            p_a_1vs2 = (df_1vs2['response'] == 0).mean()
            diffs.append(p_a_1vs1 - p_a_1vs2)
            
    if not diffs:
        return 0.0
        
    return float(np.mean(diffs))
outcome: self_sim=0.0681 (var=0.0348) adversary_sim=-0.0148 (var=0.0505) welch_t=+1.420 p=0.1623 (N=25, alpha=0.01) -> reject

[9] rationale: This metric calculates the within-subject difference in the probability of choosing Option A between Trial 3 and Trial 4. In both trials, Option A has only the top cue (0.95). In Trial 3, Option B has the lowest two cues (0.65, 0.55), resulting in low conflict. In Trial 4, Option B has the second and third best cues (0.85, 0.75), resulting in high conflict. Both theories predict a higher probability of choosing A in Trial 3 than in Trial 4, but the magnitude of this difference varies significantly between the two models due to the abrupt strategy shift in THEORY 1 versus the smooth compensatory tradeoff in THEORY 2.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        a_tuples = subj_df['option_a_ratings'].apply(tuple)
        b_tuples = subj_df['option_b_ratings'].apply(tuple)
        
        # Trial 3: A=[1,0,0,0,0] vs B=[0,0,0,1,1]
        mask_t3 = (a_tuples == (1,0,0,0,0)) & (b_tuples == (0,0,0,1,1))
        
        # Trial 4: A=[1,0,0,0,0] vs B=[0,1,1,0,0]
        mask_t4 = (a_tuples == (1,0,0,0,0)) & (b_tuples == (0,1,1,0,0))
        
        df_t3 = subj_df[mask_t3]
        df_t4 = subj_df[mask_t4]
        
        if len(df_t3) > 0 and len(df_t4) > 0:
            p_a_t3 = (df_t3['response'] == 0).mean()
            p_a_t4 = (df_t4['response'] == 0).mean()
            diffs.append(p_a_t3 - p_a_t4)
            
    if not diffs:
        return 0.0
        
    return float(np.mean(diffs))
outcome: self_sim=0.1267 (var=0.0825) adversary_sim=0.1200 (var=0.0696) welch_t=+0.085 p=0.9322 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n    \n    diffs = []\n    for subj, subj_df in data.groupby('subject_id'):\n        a_tuples = subj_df['option_a_ratings'].apply(tuple)\n        b_tuples = subj_df['option_b_ratings'].apply(tuple)\n        \n        # Low Opposing Sum Trials (B always has cues 4 and 5)\n        m_t3 = (a_tuples == (1,0,0,0,0)) & (b_tuples == (0,0,0,1,1))\n        m_t6 = (a_tuples == (0,1,0,0,0)) & (b_tuples == (0,0,0,1,1))\n        m_t9 = (a_tuples == (0,0,1,0,0)) & (b_tuples == (0,0,0,1,1))\n        mask_low = m_t3 | m_t6 | m_t9\n        \n        # High Opposing Sum Trials (B has stronger cues: 2,3 or 3,4)\n        m_t2 = (a_tuples == (1,0,0,0,0)) & (b_tuples == (0,0,1,1,0))\n        m_t4 = (a_tuples == (1,0,0,0,0)) & (b_tuples == (0,1,1,0,0))\n        m_t7 = (a_tuples == (0,1,0,0,0)) & (b_tuples == (0,0,1,1,0))\n        mask_high = m_t2 | m_t4 | m_t7\n        \n        df_low = subj_df[mask_low]\n        df_high = subj_df[mask_high]\n        \n        if len(df_low) > 0 and len(df_high) > 0:\n            p_a_low = (df_low['response'] == 0).mean()\n            p_a_high = (df_high['response'] == 0).mean()\n            diffs.append(p_a_low - p_a_high)\n            \n    if not diffs:\n        return 0.0\n        \n    return float(np.mean(diffs))",
  "rationale": "This metric calculates the within-subject difference in the probability of choosing Option A between 'Low Opposing' trials (where Option B has the two weakest cues) and 'High Opposing' trials (where Option B has stronger opposing cues). In THEORY 1, the explicit conflict signal (sum of opposing validities) triggers a sharp, non-linear shift from Take-The-Best to WADD, predicting a massive drop in P(A) when moving from Low to High opposing sums. In THEORY 2, the smooth compensatory mechanism evaluates these differences purely linearly via subjective weights, producing a systematically different, more graded magnitude of change. By aggregating across multiple trials in each bucket, we stabilize the per-subject estimate and maximize the statistical power to detect this divergence in strategy-shift magnitude."
}
```

## Usage

```json
{
  "prompt_token_count": 11690,
  "candidates_token_count": 753,
  "total_token_count": 21595
}
```
