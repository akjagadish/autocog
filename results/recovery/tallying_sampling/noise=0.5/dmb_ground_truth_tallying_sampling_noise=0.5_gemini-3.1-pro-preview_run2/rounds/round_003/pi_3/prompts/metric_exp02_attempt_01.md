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
**Validities (n_features=5):** [1.0, 0.9, 0.6, 0.5, 0.5]

**Trial pairs (n=9):**
  trial 1: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 2: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 6: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 9: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Rationale:** To cleanly dissociate Pure Tallying from Tally-Gated Validity Bias, we focus on trials where the tally scores are strictly unequal (e.g., Option A has 3 features, Option B has 2 features). By utilizing a highly skewed validity distribution, we can create trial pairs where the +1 tally advantage for Option A is pitted against varying validity differences. Pure Tallying predicts a constant, positive choice probability for Option A across all these trials because the tally difference remains fixed at +1. In contrast, the Tally-Gated Validity Bias model predicts a graded response: as Option B's features become more valid compared to Option A's, the preference for A should diminish, and can even reverse if the validity difference opposes the tally difference strongly enough. Including tally-tie trials serves as a baseline check, as both models predict exactly 50/50 guessing when the number of positive features is equal.

**Computed schedule:** 9 unique pairs × 10 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** People make decisions between options using a Tallying (Equal Weight) heuristic. Instead of weighting features by their validities or relying on a single discriminating cue, decision-makers simply count the number of positive features for each option. They choose the option with the higher total count, treating all cues as equally important. When counts are tied, they guess. Response noise is modeled via a softmax over the tally scores and a uniform lapse rate. The choice is relatively noisy, preventing the strategy from becoming perfectly deterministic even when one option has a clear tally advantage.

**Parameters:**
- beta: [0.1, 1.5]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tallying: compute the sum of features for each option (equal weighting)
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tally scores with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Tally-Gated Validity Bias: Decision-makers primarily rely on a Tallying heuristic, simply counting the number of positive features for each option. If the tally results in a tie, the decision process abruptly concludes and they guess randomly, without falling back on cue validities. However, if there is a difference in tally scores, the strength of their preference is modulated by the explicit cue validities. This means validities act as a secondary confidence-adjuster rather than a tie-breaker, explaining why validity bias appears in overall choices but is absent when options have an equal number of positive features.

**Parameters:**
- beta: [0.1, 2.0]
- epsilon: [0.0, 0.5]
- w_val: [0.0, 0.6]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
    
    # Primary strategy: Tallying
    tally_scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_val = float(parameters["w_val"])
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # If tallying results in a tie, guess randomly (no validity tie-breaking)
    if tally_scores[0] == tally_scores[1]:
        p_core = np.ones(2) / 2.0
    else:
        # If there is a tally difference, validities modulate the response strength
        val_scores = stim @ validities
        scores = (1.0 - w_val) * tally_scores + w_val * val_scores
        z = beta * (scores - scores.max())
        e = np.exp(z)
        p_core = e / e.sum()
        
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
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
[0] rationale: This metric calculates the difference in the probability of choosing the option with the higher tally score between 'congruent' trials (where the tally winner also has a higher validity sum) and 'incongruent' trials (where the tally winner has a lower validity sum). Pure Tallying ignores validities, so it predicts this difference to be near zero. Tally-Gated Validity Bias predicts that validities modulate choice when tallies differ, so it predicts a significantly positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    validities = np.array([0.95, 0.9, 0.6, 0.55, 0.5])
    
    tally_diff = []
    val_diff = []
    tally_winner_chosen = []
    
    for i, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        ta = np.sum(a)
        tb = np.sum(b)
        
        va = np.sum(a * validities)
        vb = np.sum(b * validities)
        
        td = ta - tb
        vd = va - vb
        
        tally_diff.append(td)
        val_diff.append(vd)
        
        if td > 0:
            tally_winner_chosen.append(1 if row['response'] == 0 else 0)
        elif td < 0:
            tally_winner_chosen.append(1 if row['response'] == 1 else 0)
        else:
            tally_winner_chosen.append(np.nan)
            
    data_eval = pd.DataFrame({
        'td': tally_diff,
        'vd': val_diff,
        'tally_winner_chosen': tally_winner_chosen
    })
    
    congruent_mask = (data_eval['td'] * data_eval['vd']) > 0
    incongruent_mask = (data_eval['td'] * data_eval['vd']) < 0
    
    p_cong = data_eval.loc[congruent_mask, 'tally_winner_chosen'].mean()
    p_incong = data_eval.loc[incongruent_mask, 'tally_winner_chosen'].mean()
    
    if pd.isna(p_cong): p_cong = 0.0
    if pd.isna(p_incong): p_incong = 0.0
    
    return float(p_cong - p_incong)
outcome: self_sim=-0.0079 (var=0.0107) adversary_sim=0.0271 (var=0.0172) welch_t=-1.048 p=0.3002 (N=25, alpha=0.01) -> reject

[1] rationale: By isolating the most 'extreme congruent' trials (Trials 1 and 4, where the tally winner has a massive +1.4 validity advantage) and the 'incongruent' trials (Trials 3 and 6, where the tally winner has a -0.2 validity disadvantage), we can cleanly separate the two theories. Pure Tallying predicts exactly equal choice rates for the tally winner across these two trial types, resulting in a difference of 0. Tally-Gated Validity Bias predicts a significantly higher choice rate for the tally winner in the extreme congruent trials, yielding a large positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    t14_chosen_tally_winner = []
    t36_chosen_tally_winner = []
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        opt_chosen = a if resp == 0 else b
        
        # Extreme Congruent: Trials 1 and 4
        if set([a, b]) == {(1, 1, 1, 0, 0), (0, 0, 0, 1, 1)}:
            if sum(opt_chosen) == 3:
                t14_chosen_tally_winner.append(1)
            else:
                t14_chosen_tally_winner.append(0)
                
        # Incongruent: Trials 3 and 6
        elif set([a, b]) == {(0, 0, 1, 1, 1), (1, 1, 0, 0, 0)}:
            if sum(opt_chosen) == 3:
                t36_chosen_tally_winner.append(1)
            else:
                t36_chosen_tally_winner.append(0)
                
    m1 = np.mean(t14_chosen_tally_winner) if len(t14_chosen_tally_winner) > 0 else 0.5
    m2 = np.mean(t36_chosen_tally_winner) if len(t36_chosen_tally_winner) > 0 else 0.5
    
    return float(m1 - m2)
outcome: self_sim=-0.0108 (var=0.0173) adversary_sim=0.0333 (var=0.0222) welch_t=-1.111 p=0.2721 (N=25, alpha=0.01) -> reject

[2] rationale: By fitting a linear slope of the tally winner choice against the validity difference on trials where the tally scores are not tied, we efficiently aggregate the choice modulation across all relevant trials. Pure Tallying ignores validities entirely, so the expected slope is exactly zero. Tally-Gated Validity Bias posits that validities act as a secondary modulator when tally scores differ, predicting a positive slope. Using all non-tied trials to compute a single slope per subject significantly reduces binomial variance compared to a simple point-to-point contrast, enhancing the discriminatory power of the metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.9, 0.6, 0.55, 0.5])
    
    x = []
    y = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        ta = np.sum(a)
        tb = np.sum(b)
        
        # Only evaluate trials where tally scores differ (trials 1-6)
        if ta != tb:
            va = np.sum(a * validities)
            vb = np.sum(b * validities)
            
            if ta > tb:
                val_diff = va - vb
                chosen = 1 if row['response'] == 0 else 0
            else:
                val_diff = vb - va
                chosen = 1 if row['response'] == 1 else 0
                
            x.append(val_diff)
            y.append(chosen)
            
    if len(x) < 2:
        return 0.0
        
    x = np.array(x)
    y = np.array(y)
    
    # Calculate the slope of the linear regression of choice on validity difference
    cov = np.cov(x, y)[0, 1]
    var_x = np.var(x, ddof=1)
    
    if var_x == 0:
        return 0.0
        
    return float(cov / var_x)
outcome: self_sim=-0.0067 (var=0.0066) adversary_sim=0.0209 (var=0.0087) welch_t=-1.121 p=0.268 (N=25, alpha=0.01) -> reject

[3] rationale: To maximize the discriminative power between Pure Tallying and Tally-Gated Validity Bias, we group trials with tally differences by their validity difference. We compute the empirical probability of choosing the tally winner for each group and apply the empirical logit transformation. Since the logit function stretches differences near the extremes, and we compute the linear slope of these log-odds against the validity difference, we amplify the signal of validity modulation (predicted by Tally-Gated Validity Bias) while maintaining a theoretical slope of zero for Pure Tallying. This approach reduces binomial variance effects that washed out simple linear probability differences in previous attempts.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    k1 = 0; n1 = 0
    k2 = 0; n2 = 0
    k3 = 0; n3 = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        ta = sum(a)
        tb = sum(b)
        if ta == tb:
            continue
            
        opt_chosen = a if resp == 0 else b
        is_tally_winner = 1 if sum(opt_chosen) > min(ta, tb) else 0
        
        pair = set([a, b])
        if pair == {(1, 1, 1, 0, 0), (0, 0, 0, 1, 1)}:
            k1 += is_tally_winner
            n1 += 1
        elif pair == {(1, 0, 1, 1, 0), (0, 1, 0, 0, 1)}:
            k2 += is_tally_winner
            n2 += 1
        elif pair == {(0, 0, 1, 1, 1), (1, 1, 0, 0, 0)}:
            k3 += is_tally_winner
            n3 += 1
            
    if n1 == 0 or n2 == 0 or n3 == 0:
        return 0.0
        
    y1 = np.log((k1 + 0.5) / (n1 - k1 + 0.5))
    y2 = np.log((k2 + 0.5) / (n2 - k2 + 0.5))
    y3 = np.log((k3 + 0.5) / (n3 - k3 + 0.5))
    
    slope = (0.7667 * y1 + 0.0667 * y2 - 0.8333 * y3) / 1.28667
    return float(slope)
outcome: self_sim=-0.0282 (var=0.1161) adversary_sim=0.0898 (var=0.1908) welch_t=-1.065 p=0.2926 (N=25, alpha=0.01) -> reject

[4] rationale: To maximize the signal-to-noise ratio, we focus exclusively on trials where the tally difference is exactly 1 (Trials 1 through 6). By splitting these trials into a 'High Validity Advantage' group (Trials 1, 2, 3 where the tally winner has a validity advantage > 0.5) and a 'Low/Negative Validity Advantage' group (Trials 4, 5, 6 where the tally winner has a validity advantage < 0.5), we aggregate over 36 trials per group per subject. Pure Tallying predicts a difference of exactly zero because it ignores validities completely. Tally-Gated Validity Bias predicts a positive difference. Grouping 36 trials per bucket drastically reduces the binomial variance that previously washed out the effect, allowing the true difference to easily reach statistical significance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.9, 0.6, 0.55, 0.5])
    
    high_chosen = []
    low_chosen = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        ta = np.sum(a)
        tb = np.sum(b)
        
        # Restrict to trials where tally difference is exactly 1
        if abs(ta - tb) == 1:
            va = np.sum(a * validities)
            vb = np.sum(b * validities)
            
            if ta > tb:
                val_diff = va - vb
                chosen_tally_winner = 1 if row['response'] == 0 else 0
            else:
                val_diff = vb - va
                chosen_tally_winner = 1 if row['response'] == 1 else 0
                
            # Group trials by the validity advantage of the tally winner
            if val_diff > 0.5:
                high_chosen.append(chosen_tally_winner)
            else:
                low_chosen.append(chosen_tally_winner)
                
    if len(high_chosen) == 0 or len(low_chosen) == 0:
        return 0.0
        
    return float(np.mean(high_chosen) - np.mean(low_chosen))
outcome: self_sim=0.0244 (var=0.0126) adversary_sim=0.0467 (var=0.0117) welch_t=-0.712 p=0.4797 (N=25, alpha=0.01) -> reject

[5] rationale: Previous attempts using simple differences or linear slopes failed to reach significance because the true effect size is heavily compressed by the softmax ceiling effect and lapse rates, causing the signal to be overwhelmed by binomial noise. To overcome this fundamental limit, this metric computes the mathematically optimal linear combination of tally-winner choice probabilities across all 8 trial types. The weights are strictly proportional to `(Expected_Prob_TallyGated - Expected_Prob_Tallying) / Variance`, amplifying trials where the models diverge most (e.g., Trial 1 and Trial 7) and suppressing noise from trials where they perform similarly. This maximizes the expected theoretical difference relative to the standard error, yielding the highest possible t-statistic.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Optimal weights derived from the expected probability gaps between 
    # Tally-Gated Validity Bias and Pure Tallying, normalized by binomial variance.
    weights = {
        frozenset([(1, 1, 1, 0, 0), (0, 0, 0, 1, 1)]): 3.06,  # Trial 1
        frozenset([(1, 0, 1, 1, 0), (0, 1, 0, 0, 1)]): 1.31,  # Trial 2
        frozenset([(0, 1, 1, 1, 0), (1, 0, 0, 0, 1)]): 1.04,  # Trial 3
        frozenset([(0, 0, 1, 1, 1), (1, 0, 0, 0, 1)]): -0.05, # Trial 4
        frozenset([(0, 0, 1, 1, 1), (1, 1, 0, 0, 0)]): -1.20, # Trial 5
        frozenset([(0, 0, 0, 1, 1), (1, 0, 0, 0, 0)]): -0.33, # Trial 6
        frozenset([(1, 1, 1, 1, 0), (0, 0, 0, 0, 1)]): 2.21,  # Trial 7
        frozenset([(0, 1, 1, 1, 1), (1, 0, 0, 0, 0)]): 1.23   # Trial 8
    }
    
    score = 0.0
    
    # Create hashable representations of the ratings
    data_copy = data.copy()
    data_copy['a_tuple'] = data_copy['option_a_ratings'].apply(tuple)
    data_copy['b_tuple'] = data_copy['option_b_ratings'].apply(tuple)
    
    groups = data_copy.groupby(['a_tuple', 'b_tuple'])
    
    for (a, b), group in groups:
        pair = frozenset([a, b])
        if pair in weights:
            ta = sum(a)
            tb = sum(b)
            if ta == tb:
                continue
                
            # Calculate empirical probability of choosing the tally winner
            if ta > tb:
                p_winner = (group['response'] == 0).mean()
            else:
                p_winner = (group['response'] == 1).mean()
                
            score += weights[pair] * p_winner
            
    return float(score)
outcome: self_sim=5.3225 (var=0.4978) adversary_sim=5.4751 (var=0.7683) welch_t=-0.678 p=0.501 (N=25, alpha=0.01) -> reject

[6] rationale: To achieve maximum discriminative power, this metric isolates the two trials with the largest divergence in validity advantage while holding the tally difference constant at exactly 1. In Trial 1 (Extreme Congruent), the tally winner has a massive +1.4 validity advantage. In Trial 5 (Incongruent), the tally winner actually has a -0.2 validity disadvantage. Pure Tallying strictly ignores validities, so it predicts the exact same choice probability for the tally winner in both trials, yielding an expected difference of 0. Tally-Gated Validity Bias predicts that the validities modulate the choice strength, leading to a much higher choice probability for the tally winner in Trial 1 compared to Trial 5. By directly contrasting these two extreme endpoints, we maximize the expected gap between the theories while avoiding the dilution of effect size that occurs when grouping moderate trials.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    data_copy = data.copy()
    data_copy['a_str'] = data_copy['option_a_ratings'].apply(lambda x: "".join(map(str, x)))
    data_copy['b_str'] = data_copy['option_b_ratings'].apply(lambda x: "".join(map(str, x)))
    
    # Trial 1: Extreme Congruent (Tally winner has +1.4 validity advantage)
    t1_mask = ((data_copy['a_str'] == "11100") & (data_copy['b_str'] == "00011")) | \
              ((data_copy['a_str'] == "00011") & (data_copy['b_str'] == "11100"))
              
    # Trial 5: Incongruent (Tally winner has -0.2 validity disadvantage)
    t5_mask = ((data_copy['a_str'] == "00111") & (data_copy['b_str'] == "11000")) | \
              ((data_copy['a_str'] == "11000") & (data_copy['b_str'] == "00111"))
              
    t1_data = data_copy[t1_mask]
    t5_data = data_copy[t5_mask]
    
    def get_tally_winner_chosen(row):
        a_tally = sum(row['option_a_ratings'])
        b_tally = sum(row['option_b_ratings'])
        if a_tally > b_tally:
            return 1 if row['response'] == 0 else 0
        elif b_tally > a_tally:
            return 1 if row['response'] == 1 else 0
        return 0.5

    if len(t1_data) == 0 or len(t5_data) == 0:
        return 0.0

    t1_chosen = t1_data.apply(get_tally_winner_chosen, axis=1).mean()
    t5_chosen = t5_data.apply(get_tally_winner_chosen, axis=1).mean()
    
    return float(t1_chosen - t5_chosen)

outcome: self_sim=0.0600 (var=0.0306) adversary_sim=0.0983 (var=0.0355) welch_t=-0.746 p=0.4594 (N=25, alpha=0.01) -> reject

[7] rationale: Previous metrics failed because the true effect size is compressed by softmax ceiling effects and lapse rates, which allows binomial noise to wash out the differences. To overcome this, this metric computes the Log-Likelihood Ratio (LLR) test statistic between the two theories via grid-search Maximum Likelihood Estimation. Because Pure Tallying is strictly nested within Tally-Gated Validity Bias (it is exactly the case where w_val = 0), we can directly compare their max log-likelihoods on each subject's data. Under Pure Tallying, the LLR will be tightly distributed near 0. Under Tally-Gated Validity Bias, the MLE will identify the optimal w_val > 0, yielding a significantly larger LLR. This approach mathematically optimally combines all trials while perfectly accounting for parameter variations, non-linearities, and lapse rates, maximizing statistical power.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    validities = np.array([0.95, 0.9, 0.6, 0.55, 0.5])
    subjects = data['subject_id'].unique()
    
    # Define parameter grids for Maximum Likelihood Estimation
    betas = np.linspace(0.1, 2.0, 20)
    epsilons = np.linspace(0.0, 0.5, 11)
    w_vals = np.linspace(0.0, 0.6, 13)
    
    total_llr = 0.0
    
    for subj in subjects:
        subj_data = data[data['subject_id'] == subj]
        
        a_ratings = np.stack(subj_data['option_a_ratings'].values)
        b_ratings = np.stack(subj_data['option_b_ratings'].values)
        choices = subj_data['response'].values
        
        tally_a = a_ratings.sum(axis=1)
        tally_b = b_ratings.sum(axis=1)
        
        val_a = a_ratings @ validities
        val_b = b_ratings @ validities
        
        is_tied = (tally_a == tally_b)
        
        # Compute scores for all w_vals. Shape: (W, N)
        scores_a = (1 - w_vals[:, None]) * tally_a[None, :] + w_vals[:, None] * val_a[None, :]
        scores_b = (1 - w_vals[:, None]) * tally_b[None, :] + w_vals[:, None] * val_b[None, :]
        
        max_scores = np.maximum(scores_a, scores_b)
        
        # Apply softmax for all betas. Shape: (B, W, N)
        z_a = betas[:, None, None] * (scores_a - max_scores)[None, :, :]
        z_b = betas[:, None, None] * (scores_b - max_scores)[None, :, :]
        
        exp_a = np.exp(z_a)
        exp_b = np.exp(z_b)
        sum_exp = exp_a + exp_b
        
        p_core_a = exp_a / sum_exp
        p_core_b = exp_b / sum_exp
        
        # Override tied trials (both theories guess 50/50 on ties)
        p_core_a[:, :, is_tied] = 0.5
        p_core_b[:, :, is_tied] = 0.5
        
        # Blend with uniform lapse rate for all epsilons. Shape: (E, B, W, N)
        p_final_a = (1 - epsilons[:, None, None, None]) * p_core_a[None, :, :, :] + epsilons[:, None, None, None] * 0.5
        p_final_b = (1 - epsilons[:, None, None, None]) * p_core_b[None, :, :, :] + epsilons[:, None, None, None] * 0.5
        
        # Select the probability of the chosen option
        condition = (choices == 0)[None, None, None, :]
        p_chosen = np.where(condition, p_final_a, p_final_b)
        p_chosen = np.clip(p_chosen, 1e-6, 1.0)
        
        # Compute Log-Likelihood. Shape: (E, B, W)
        LL = np.sum(np.log(p_chosen), axis=3)
        
        # Maximum Log-Likelihood for Pure Tallying (w_val = 0 is index 0)
        LL_pure = np.max(LL[:, :, 0])
        
        # Maximum Log-Likelihood for Tally-Gated Validity Bias (any w_val)
        LL_gated = np.max(LL)
        
        # Log-Likelihood Ratio Test Statistic (Deviance)
        llr = 2 * (LL_gated - LL_pure)
        total_llr += llr
        
    return float(total_llr / len(subjects))
outcome: self_sim=0.5301 (var=0.8508) adversary_sim=1.0317 (var=2.5663) welch_t=-1.357 p=0.1828 (N=25, alpha=0.01) -> reject

[8] rationale: To maximize discriminative power, we isolate the two trials that produce the most extreme divergence in validity advantage while maintaining an identical tally difference of 1. In Trial 1, the tally winner enjoys a massive +1.5 validity advantage. In Trial 4, the tally winner actually suffers a -0.3 validity disadvantage. Pure Tallying completely ignores validities, predicting the exact same choice probability for the tally winner in both trials (expected difference = 0). Tally-Gated Validity Bias predicts that validities heavily modulate the choice strength, resulting in a significantly higher choice rate for the tally winner in Trial 1 compared to Trial 4. Contrasting these specific endpoints yields the largest possible effect size.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    t1_chosen = []
    t4_chosen = []
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        pair = frozenset([a, b])
        
        # Trial 1: Extreme Congruent. Tally difference is 1, Validity advantage is +1.5
        t1 = frozenset([(1, 1, 1, 0, 0), (0, 0, 0, 1, 1)])
        # Trial 4: Incongruent. Tally difference is 1, Validity advantage is -0.3
        t4 = frozenset([(0, 0, 1, 1, 1), (1, 1, 0, 0, 0)])
        
        if pair not in [t1, t4]:
            continue
            
        ta = sum(a)
        tb = sum(b)
        resp = row['response']
        
        if ta > tb:
            tally_winner_chosen = 1 if resp == 0 else 0
        elif tb > ta:
            tally_winner_chosen = 1 if resp == 1 else 0
        else:
            continue
            
        if pair == t1:
            t1_chosen.append(tally_winner_chosen)
        elif pair == t4:
            t4_chosen.append(tally_winner_chosen)
            
    if not t1_chosen or not t4_chosen:
        return 0.0
        
    return float(np.mean(t1_chosen) - np.mean(t4_chosen))
outcome: self_sim=-0.0040 (var=0.0372) adversary_sim=0.0780 (var=0.0473) welch_t=-1.410 p=0.1649 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    high_chosen = []\n    low_chosen = []\n    \n    # High validity advantage for the tally winner:\n    # Trial 1: val diff 1.5\n    # Trial 5: val diff 1.3\n    t1 = {(1, 1, 1, 0, 0), (0, 0, 0, 1, 1)}\n    t5 = {(1, 1, 0, 0, 0), (0, 0, 1, 0, 0)}\n    \n    # Low/negative validity advantage for the tally winner:\n    # Trial 4: val diff -0.3\n    # Trial 6: val diff 0.1\n    # Trial 7: val diff 0.0\n    t4 = {(0, 0, 1, 1, 1), (1, 1, 0, 0, 0)}\n    t6 = {(0, 0, 1, 1, 0), (1, 0, 0, 0, 0)}\n    t7 = {(0, 0, 0, 1, 1), (1, 0, 0, 0, 0)}\n    \n    for _, row in data.iterrows():\n        a = tuple(row['option_a_ratings'])\n        b = tuple(row['option_b_ratings'])\n        \n        ta = sum(a)\n        tb = sum(b)\n        \n        if ta == tb:\n            continue\n            \n        if ta > tb:\n            chosen_winner = 1 if row['response'] == 0 else 0\n        else:\n            chosen_winner = 1 if row['response'] == 1 else 0\n            \n        pair = set([a, b])\n        \n        if pair in [t1, t5]:\n            high_chosen.append(chosen_winner)\n        elif pair in [t4, t6, t7]:\n            low_chosen.append(chosen_winner)\n            \n    if not high_chosen or not low_chosen:\n        return 0.0\n        \n    return float(np.mean(high_chosen) - np.mean(low_chosen))",
  "rationale": "Previous attempts grouped trials incorrectly (e.g., classifying Trial 5 as 'low validity advantage' when it actually has a massive +1.3 validity advantage for the tally winner). This metric correctly pools the two trials with the highest validity advantage for the tally winner (Trials 1 and 5) and contrasts them against the three trials with the lowest/negative validity advantage (Trials 4, 6, and 7). By strictly contrasting these two correctly specified pools, we aggregate over 50 trials per subject, drastically reducing binomial variance while maximizing the expected divergence between Pure Tallying (which predicts a difference of 0) and Tally-Gated Validity Bias (which predicts a strongly positive difference)."
}
```

## Usage

```json
{
  "prompt_token_count": 10269,
  "candidates_token_count": 725,
  "total_token_count": 13983
}
```
