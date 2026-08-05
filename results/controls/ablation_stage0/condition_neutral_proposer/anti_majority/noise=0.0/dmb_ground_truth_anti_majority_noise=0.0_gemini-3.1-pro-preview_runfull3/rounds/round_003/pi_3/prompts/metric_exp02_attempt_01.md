# metric_exp02_attempt_01

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
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.6, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 7: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 8: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 9: A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 10: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 11: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 12: A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]

**Rationale:** To maximally distinguish pure WADD (THEORY 1) from the adaptive TTB/WADD mixture (THEORY 2), we exploit the specific functional form of the strategy switch in THEORY 2. THEORY 2 transitions from TTB to WADD based on a linear combination of the top discriminating cue's validity (v_top) and the sum of the opposing cues' validities (v_opp_sum). By contrast, THEORY 1 uses a non-linear compensatory sum (via the gamma parameter) to evaluate all cues continuously. We construct a 6-feature environment and design trial pairs that decouple the non-linear WADD score differences from the linear v_opp_sum. For instance, we compare trials where a high-validity top cue is opposed by a few strong cues versus many weak cues. These configurations can yield similar WADD score differences (depending on gamma) but radically different v_opp_sum values, triggering a sharp strategy switch in THEORY 2 while producing only smooth probability changes in THEORY 1.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



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

[3] rationale: This metric contrasts the probability of choosing Option B when it is supported by 3 cues versus when it is supported by 2 cues. Under the pure Weighted Additive (WADD) theory, subjective weights are a non-linear power function of validities, meaning that adding a third weak cue often contributes very little marginal value compared to having two strong cues. Conversely, the Competing Theory's strategy-selection mechanism relies on the linear sum of opposing validities (v_opp_sum). Three cues, even if individually weaker, produce a substantially larger linear sum than two cues, reliably triggering a shift away from Take-The-Best (which strictly favors Option A) toward the compensatory strategy. Thus, the Competing Theory predicts a much sharper increase in P(B) when moving from 2 to 3 opposing cues than pure WADD does. By aggregating over all trials, this metric ensures very low between-subject variance, boosting statistical power.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Count the number of positive cues for Option B
    b_cues = data['option_b_ratings'].apply(sum)
    
    # Group into trials where Option B has 3 cues vs 2 cues
    p_b_3cues = data.loc[b_cues == 3, 'response'].mean()
    p_b_2cues = data.loc[b_cues == 2, 'response'].mean()
    
    if pd.isna(p_b_3cues) or pd.isna(p_b_2cues):
        return 0.0
        
    return float(p_b_3cues - p_b_2cues)
outcome: self_sim=-0.0500 (var=0.0112) adversary_sim=-0.0624 (var=0.0105) welch_t=+0.421 p=0.6755 (N=25, alpha=0.01) -> reject

[4] rationale: This metric contrasts the rate at which subjects choose the option with the strongest single cue (validity 0.95) when the opposing cues have a low linear sum (1.40, Set 1) versus a high linear sum (1.95, Set 2). Under the purely compensatory WADD theory, subjective weights are non-linearly transformed, so adding a single weak cue to the opposing option causes only a moderate, smooth decrease in the probability of choosing the top cue. In contrast, the Competing Theory's strategy-selection mechanism relies on the raw linear sum of opposing validities. Moving from an opposing sum of 1.40 to 1.95 drastically increases the probability of switching away from Take-The-Best (which strictly favors the top cue) to a compensatory strategy. This structural shift causes the mixture model to predict a much sharper drop in top-cue choices than pure WADD. Averaging across symmetric and structurally similar trials reduces between-subject variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Set 1: Low opposing sum (1.40)
    t3_mask = (data['option_a_ratings'].apply(tuple) == (1, 0, 0, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 1, 0, 0, 1))
    t4_mask = (data['option_a_ratings'].apply(tuple) == (1, 0, 0, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 0, 1, 1, 0))
    
    # Set 2: High opposing sum (1.95)
    t5_mask = (data['option_a_ratings'].apply(tuple) == (1, 0, 0, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 0, 1, 1, 1))
    t12_mask = (data['option_a_ratings'].apply(tuple) == (0, 0, 1, 1, 1)) & (data['option_b_ratings'].apply(tuple) == (1, 0, 0, 0, 0))
    
    # Calculate probability of choosing the option with the top cue (0.95)
    # For trials 3, 4, 5, top cue is Option A (response = 0)
    # For trial 12, top cue is Option B (response = 1)
    c_top_3 = 1.0 - data.loc[t3_mask, 'response'].mean()
    c_top_4 = 1.0 - data.loc[t4_mask, 'response'].mean()
    c_top_5 = 1.0 - data.loc[t5_mask, 'response'].mean()
    c_top_12 = data.loc[t12_mask, 'response'].mean()
    
    if pd.isna(c_top_3) or pd.isna(c_top_4) or pd.isna(c_top_5) or pd.isna(c_top_12):
        return 0.0
        
    set1_mean = (c_top_3 + c_top_4) / 2.0
    set2_mean = (c_top_5 + c_top_12) / 2.0
    
    return float(set1_mean - set2_mean)
outcome: self_sim=-0.0033 (var=0.0333) adversary_sim=-0.0017 (var=0.0243) welch_t=-0.035 p=0.9725 (N=25, alpha=0.01) -> reject

[5] rationale: This metric calculates the difference in the rate at which subjects choose the option favored by the most valid discriminating cue (the 'top cue') when the top cue has a validity of 0.95 versus when it has a validity of 0.85. Under the pure Weighted Additive (WADD) theory, a drop in the top cue's validity from 0.95 to 0.85 in this design is accompanied by a corresponding drop in the strength of the opposing cues (e.g., Trials 6-8 have weaker opposing cues than Trials 1-5). Consequently, WADD predicts a relatively stable or even higher rate of top-cue choices in the 0.85 trials. In contrast, the Competing Theory's strategy-selection mechanism depends directly on the unweighted value of the top cue (`v_top`). Dropping `v_top` from 0.95 to 0.85 significantly reduces the probability of using Take-The-Best, triggering a structural shift toward the compensatory strategy. This causes the mixture model to predict a much sharper decline in top-cue choices compared to WADD. By aggregating across multiple trials based on top-cue validity, this metric minimizes between-subject variance while capturing a core structural divergence.
metric_source:
def metric(data: pd.DataFrame) -> float:
    def get_top_cue_info(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(5):
            if a[i] > b[i]:
                return 0, i
            elif b[i] > a[i]:
                return 1, i
        return 0, -1

    top_cue_info = data.apply(get_top_cue_info, axis=1)
    top_cue_option = top_cue_info.apply(lambda x: x[0])
    top_cue_idx = top_cue_info.apply(lambda x: x[1])
    chose_top_cue = (data['response'] == top_cue_option).astype(float)
    
    p_top_95 = chose_top_cue[top_cue_idx == 0].mean()
    p_top_85 = chose_top_cue[top_cue_idx == 1].mean()
    
    if pd.isna(p_top_95) or pd.isna(p_top_85):
        return 0.0
        
    return float(p_top_95 - p_top_85)
outcome: self_sim=0.1447 (var=0.0207) adversary_sim=0.0931 (var=0.0149) welch_t=+1.368 p=0.1778 (N=25, alpha=0.01) -> reject

[6] rationale: This metric contrasts the probability of choosing Option B in Trial 1 versus Trial 5. In both trials, Option A has the top cue (validity 0.95). In Trial 1, Option B has two strong opposing cues (0.85, 0.75). In Trial 5, Option B has three weaker opposing cues (0.75, 0.65, 0.55). Under the Advocated Theory (pure WADD), validities are subjectively weighted via a non-linear power function, meaning two strong cues often outweigh three weaker ones; thus, WADD predicts a higher probability of choosing B in Trial 1 than in Trial 5. Conversely, the Competing Theory relies on the raw linear sum of opposing validities to trigger a strategy switch from Take-The-Best (which strictly favors A) to WADD. Since the linear sum in Trial 5 (1.95) is much larger than in Trial 1 (1.60), the Competing Theory predicts a massive structural shift away from TTB in Trial 5, leading to a higher probability of choosing B in Trial 5 than in Trial 1. This creates a sharp divergence in the sign of the difference P(B|T1) - P(B|T5).
metric_source:
def metric(data: pd.DataFrame) -> float:
    t1_mask = (data['option_a_ratings'].apply(tuple) == (1, 0, 0, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 1, 1, 0, 0))
    t5_mask = (data['option_a_ratings'].apply(tuple) == (1, 0, 0, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 0, 1, 1, 1))
    
    p_b_t1 = data.loc[t1_mask, 'response'].mean()
    p_b_t5 = data.loc[t5_mask, 'response'].mean()
    
    if pd.isna(p_b_t1) or pd.isna(p_b_t5):
        return 0.0
        
    return float(p_b_t1 - p_b_t5)
outcome: self_sim=0.0633 (var=0.0743) adversary_sim=0.0867 (var=0.0814) welch_t=-0.296 p=0.7688 (N=25, alpha=0.01) -> reject

[7] rationale: This metric measures the marginal effect of adding supplementary cues to the option that already possesses the strongest valid cue (the 'Top' option), while holding the opponent's profile strictly constant. Under the pure WADD theory, adding cues to an option strictly increases its compensatory score, leading to a substantial increase in the probability of choosing it. However, under the Competing Theory's mixture model, the probability of selecting the Take-The-Best (TTB) strategy depends solely on the validity of the top cue (`v_top`) and the sum of the opponent's cues (`v_opp_sum`). Because both `v_top` and `v_opp_sum` are identical between the Base and Enhanced trials, the mixture model's probability of using TTB remains completely unchanged. Since TTB strictly chooses the Top option regardless of supplementary cues, the TTB component contributes exactly zero to the difference, heavily diluting the effect in the mixture model. This creates a robust divergence where WADD predicts a large positive difference, while the mixture model predicts a significantly dampened one.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Identify Base Trials where the Top option has ONLY the 0.95 cue.
    # T1: Top A, Opponent B has [0,1,1,0,0]
    t1 = (data['option_a_ratings'].apply(tuple) == (1,0,0,0,0)) & (data['option_b_ratings'].apply(tuple) == (0,1,1,0,0))
    # T4: Top A, Opponent B has [0,0,1,1,0]
    t4 = (data['option_a_ratings'].apply(tuple) == (1,0,0,0,0)) & (data['option_b_ratings'].apply(tuple) == (0,0,1,1,0))
    
    # Identify Enhanced Trials where the Top option has the 0.95 cue PLUS additional cues, 
    # but the Opponent is EXACTLY the same as in the Base Trials.
    # T14: Top A gets +[0,0,0,1,1], Opponent B is [0,1,1,0,0] (matches T1)
    t14 = (data['option_a_ratings'].apply(tuple) == (1,0,0,1,1)) & (data['option_b_ratings'].apply(tuple) == (0,1,1,0,0))
    # T9: Symmetric to T14. Top B gets +[0,0,0,1,1], Opponent A is [0,1,1,0,0] (matches T1)
    t9 = (data['option_a_ratings'].apply(tuple) == (0,1,1,0,0)) & (data['option_b_ratings'].apply(tuple) == (1,0,0,1,1))
    # T15: Top B gets +[0,1,0,0,0], Opponent A is [0,0,1,1,0] (matches T4)
    t15 = (data['option_a_ratings'].apply(tuple) == (0,0,1,1,0)) & (data['option_b_ratings'].apply(tuple) == (1,1,0,0,0))
    
    # Probability of choosing the Top option
    p_top_t1 = 1.0 - data.loc[t1, 'response'].mean()
    p_top_t4 = 1.0 - data.loc[t4, 'response'].mean()
    
    p_top_t14 = 1.0 - data.loc[t14, 'response'].mean()
    p_top_t9 = data.loc[t9, 'response'].mean()
    p_top_t15 = data.loc[t15, 'response'].mean()
    
    base_mean = np.nanmean([p_top_t1, p_top_t4])
    enhanced_mean = np.nanmean([p_top_t14, p_top_t9, p_top_t15])
    
    if pd.isna(base_mean) or pd.isna(enhanced_mean):
        return 0.0
        
    return float(enhanced_mean - base_mean)
outcome: self_sim=0.0444 (var=0.0532) adversary_sim=0.1817 (var=0.0729) welch_t=-1.932 p=0.05939 (N=25, alpha=0.01) -> reject

[8] rationale: This metric contrasts the probability of choosing Option A in Trial 10 versus Trial 1. In Trial 1, Option A has the strongest cue (0.95) and Option B has two opposing cues (0.85, 0.75). In Trial 10, Option A gains cue 4 (0.65) and Option B gains cue 5 (0.60). Under the purely compensatory WADD theory, Option A gains a stronger cue than Option B, so the probability of choosing A should strictly increase from Trial 1 to Trial 10. However, under the Competing Theory's mixture model, adding cue 5 to Option B increases the linear sum of opposing validities from 1.60 to 2.20. This massive increase triggers a sharp structural shift away from the Take-The-Best strategy (which strictly favors A) toward the compensatory WADD strategy. Consequently, the mixture model predicts a substantial drop in the probability of choosing A. This qualitative divergence (WADD predicts an increase, Mixture predicts a decrease) makes `P(A|Trial 10) - P(A|Trial 1)` a highly discriminative metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
    t1_mask = (data['option_a_ratings'].apply(tuple) == (1, 0, 0, 0, 0, 0)) & \
              (data['option_b_ratings'].apply(tuple) == (0, 1, 1, 0, 0, 0))
              
    # Trial 10: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0]
    t10_mask = (data['option_a_ratings'].apply(tuple) == (1, 0, 0, 1, 0, 0)) & \
               (data['option_b_ratings'].apply(tuple) == (0, 1, 1, 0, 1, 0))
               
    # Probability of choosing Option A (response == 0)
    p_a_t1 = 1.0 - data.loc[t1_mask, 'response'].mean()
    p_a_t10 = 1.0 - data.loc[t10_mask, 'response'].mean()
    
    if pd.isna(p_a_t1) or pd.isna(p_a_t10):
        return 0.0
        
    return float(p_a_t10 - p_a_t1)
outcome: self_sim=0.0075 (var=0.0296) adversary_sim=-0.0150 (var=0.0485) welch_t=+0.402 p=0.6893 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Group 2: Trials where the top cue is opposed by exactly 2 cues.\n    # T1: A=[1,0,0,0,0,0], B=[0,1,1,0,0,0] (Top A)\n    t1 = (data['option_a_ratings'].apply(tuple) == (1, 0, 0, 0, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 1, 1, 0, 0, 0))\n    # T3: A=[0,1,0,0,0,0], B=[0,0,1,1,0,0] (Top A)\n    t3 = (data['option_a_ratings'].apply(tuple) == (0, 1, 0, 0, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 0, 1, 1, 0, 0))\n    # T8: A=[0,1,1,0,0,0], B=[1,0,0,1,0,0] (Top B)\n    t8 = (data['option_a_ratings'].apply(tuple) == (0, 1, 1, 0, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (1, 0, 0, 1, 0, 0))\n    \n    p_top_t1 = 1.0 - data.loc[t1, 'response'].mean()\n    p_top_t3 = 1.0 - data.loc[t3, 'response'].mean()\n    p_top_t8 = data.loc[t8, 'response'].mean()\n    \n    # Group 3: Trials where the top cue is opposed by exactly 3 cues.\n    # T2: A=[1,0,0,0,0,0], B=[0,0,0,1,1,1] (Top A)\n    t2 = (data['option_a_ratings'].apply(tuple) == (1, 0, 0, 0, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 0, 0, 1, 1, 1))\n    # T4: A=[0,1,0,0,0,0], B=[0,0,0,1,1,1] (Top A)\n    t4 = (data['option_a_ratings'].apply(tuple) == (0, 1, 0, 0, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 0, 0, 1, 1, 1))\n    # T6: A=[1,0,0,0,0,0], B=[0,0,1,1,1,0] (Top A)\n    t6 = (data['option_a_ratings'].apply(tuple) == (1, 0, 0, 0, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 0, 1, 1, 1, 0))\n    # T9: A=[0,0,1,1,1,0], B=[0,1,0,0,0,1] (Top B)\n    t9 = (data['option_a_ratings'].apply(tuple) == (0, 0, 1, 1, 1, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 1, 0, 0, 0, 1))\n    # T10: A=[1,0,0,1,0,0], B=[0,1,1,0,1,0] (Top A)\n    t10 = (data['option_a_ratings'].apply(tuple) == (1, 0, 0, 1, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 1, 1, 0, 1, 0))\n    \n    p_top_t2 = 1.0 - data.loc[t2, 'response'].mean()\n    p_top_t4 = 1.0 - data.loc[t4, 'response'].mean()\n    p_top_t6 = 1.0 - data.loc[t6, 'response'].mean()\n    p_top_t9 = data.loc[t9, 'response'].mean()\n    p_top_t10 = 1.0 - data.loc[t10, 'response'].mean()\n    \n    import numpy as np\n    group2_mean = np.nanmean([p_top_t1, p_top_t3, p_top_t8])\n    group3_mean = np.nanmean([p_top_t2, p_top_t4, p_top_t6, p_top_t9, p_top_t10])\n    \n    if pd.isna(group2_mean) or pd.isna(group3_mean):\n        return 0.0\n        \n    return float(group3_mean - group2_mean)",
  "rationale": "This metric contrasts the average probability of choosing the option supported by the strongest single cue (the 'top cue') when it is opposed by exactly 3 cues versus when it is opposed by exactly 2 cues. Under the Advocated Theory (pure WADD), cues are weighted non-linearly. Two strong opposing cues (e.g., 0.85 and 0.75) are generally much more competitive than three weaker opposing cues (e.g., 0.65, 0.60, 0.55). Thus, WADD predicts the top cue will win MORE often when facing 3 weak cues than 2 strong ones (a positive difference). In stark contrast, the Competing Theory's strategy-selection mechanism relies strictly on the linear sum of opposing validities. Three cues produce a substantially higher linear sum than two cues, triggering a much higher rate of switching away from Take-The-Best to the noisier compensatory strategy. Consequently, the mixture model predicts the top cue will win LESS often when facing 3 opposing cues compared to 2 (a negative difference). This qualitative reversal provides a highly discriminating contrast."
}
```

## Usage

```json
{
  "prompt_token_count": 10509,
  "candidates_token_count": 1524,
  "total_token_count": 18088
}
```
