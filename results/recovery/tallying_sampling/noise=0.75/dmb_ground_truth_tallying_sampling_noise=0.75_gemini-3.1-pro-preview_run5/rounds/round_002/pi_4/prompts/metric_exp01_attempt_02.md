# metric_exp01_attempt_02

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
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 8: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Rationale:** To quantitatively dissociate pure Tallying from a Mixture of Tallying and Take-The-Best (TTB), we use a 5-feature design that systematically pits the raw tally of positive features against the highest-validity cue. Pure Tallying predicts choice probabilities entirely based on the difference in tallies, completely ignoring cue validities; notably, it predicts an exact 50/50 split when tallies are tied. The Mixture model incorporates a proportion of choices driven by TTB, which strictly follows the highest-validity discriminating cue. By including 'tally-tied' trials where the top discriminating cue favors one option, we can isolate the independent pull of TTB. If the pure Tallying model is correct, choice probabilities will map perfectly onto tally differences and show no preference in tied-tally trials regardless of which specific cue is positive. Conflicting trials where the tally favors one option but the highest validity cue favors the other further help to estimate the mixture weight.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Mixture of Tallying and Take-The-Best (TTB): Decision makers are heterogeneous in their strategy use. While the majority of choices are made using a compensatory equal-weight heuristic (Tallying), a smaller proportion of decisions rely on a non-compensatory, one-reason heuristic (Take-The-Best), which evaluates cues sequentially by validity and stops at the first discriminating cue. This mixture model captures both the dominant compensatory behavior and the minority non-compensatory behavior, providing a better fit to aggregate human data than either heuristic alone.

**Parameters:**
- w_ttb: [0.0, 0.25]
- beta_tally: [0.0, 2.0]
- beta_ttb: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    # Strategy 1: Tallying (Equal-Weight)
    scores_tally = np.sum(stim, axis=1)
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # Strategy 2: Take-The-Best (TTB)
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
    # Mixture
    w_ttb = float(parameters["w_ttb"])
    epsilon = float(parameters["epsilon"])
    
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # Uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## COMPETING THEORY
**Description:** Tallying (Equal-Weight) Heuristic: People evaluate options by simply counting the number of positive features (or cues favoring each option) and choosing the option with the higher total count. This compensatory strategy ignores the differential validities or subjective importance of different cues, treating all pieces of evidence equally. The choice probability is determined by a softmax over the total feature tallies for each option, combined with a uniform lapse rate. Crucially, the softmax temperature is constrained to produce softer choice probabilities, reflecting that humans do not apply the tallying rule completely deterministically.

**Parameters:**
- beta: [0.0, 1.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tally the number of positive cues for each option
    score_a = np.sum(stim[0])
    score_b = np.sum(stim[1])
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tallies
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
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
[0] rationale: By isolating trials where the total number of positive features is equal (tied tallies), the Pure Tallying model is forced to predict a 50/50 choice probability regardless of cue validity. The Mixture model, however, incorporates a Take-The-Best component that breaks the tie using the highest validity cue. Calculating the proportion of choices that align with the TTB winner on these tied-tally trials will yield approximately 0.5 for Pure Tallying, but significantly higher than 0.5 for the Mixture theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    responses = data['response'].values
    
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    
    # Focus only on trials where the simple tally of positive features is tied
    tied_mask = (sum_a == sum_b)
    
    if not np.any(tied_mask):
        return 0.5
        
    a_tied = a_ratings[tied_mask]
    b_tied = b_ratings[tied_mask]
    resp_tied = responses[tied_mask]
    
    matches = 0
    valid_trials = 0
    
    for i in range(len(a_tied)):
        a = a_tied[i]
        b = b_tied[i]
        ttb_winner = None
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is not None:
            if resp_tied[i] == ttb_winner:
                matches += 1
            valid_trials += 1
            
    if valid_trials == 0:
        return 0.5
    return float(matches / valid_trials)
outcome: self_sim=0.5242 (var=0.0053) adversary_sim=0.5008 (var=0.0053) welch_t=+1.132 p=0.2631 (N=25, alpha=0.01) -> reject

[1] rationale: This metric leverages a double-dissociation between the two models by computing the difference in the probability of choosing the TTB (Take-The-Best) winner between 'tied' trials and 'opposed' trials. In this experimental design, all trials with unequal tallies are 'opposed'—the TTB winner is strictly the tally loser. 

For pure Tallying, choices in tied trials are exactly 50/50, and choices in opposed trials penalize the TTB winner based purely on the tally difference (governed by beta in [0, 1]). 

For the Mixture model, two effects stretch this difference: 
1) The TTB component pulls choices toward the TTB winner in tied trials, raising P(TTB | tied) above 0.5.
2) The Mixture model allows a much higher sensitivity to tally differences (beta_tally in [0, 2] vs Tallying's [0, 1]). This stronger compensatory behavior heavily penalizes the tally loser in opposed trials, driving P(TTB | opposed) even lower than pure Tallying predicts.

By subtracting P(TTB | opposed) from P(TTB | tied), the metric aggregates both the TTB-driven boost in tied trials and the amplified tally-driven penalty in opposed trials, creating a robust, low-variance contrast across all 96 trials.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    responses = data['response'].values
    
    ttb_winners = []
    tally_diffs = []
    for i in range(len(a_ratings)):
        a = a_ratings[i]
        b = b_ratings[i]
        winner = -1
        for j in range(len(a)):
            if a[j] > b[j]:
                winner = 0
                break
            elif b[j] > a[j]:
                winner = 1
                break
        ttb_winners.append(winner)
        tally_diffs.append(np.sum(a) - np.sum(b))
        
    ttb_winners = np.array(ttb_winners)
    tally_diffs = np.array(tally_diffs)
    
    chose_ttb = (responses == ttb_winners)
    is_tied = (tally_diffs == 0)
    is_opposed = (tally_diffs != 0)
    
    p_tied = np.mean(chose_ttb[is_tied]) if np.sum(is_tied) > 0 else 0.5
    p_opposed = np.mean(chose_ttb[is_opposed]) if np.sum(is_opposed) > 0 else 0.5
    
    return float(p_tied - p_opposed)
outcome: self_sim=0.1475 (var=0.0136) adversary_sim=0.1283 (var=0.0141) welch_t=+0.575 p=0.568 (N=25, alpha=0.01) -> reject

[2] rationale: This metric leverages the adversarial nature of the experimental design. In the provided design, the highest-validity cue (Feature 0) consistently opposes the simple tally of positive features in all unequal-tally trials. 

For the pure Tallying model, choices are driven entirely by the tally difference. Because Feature 0 is negatively correlated with the tally difference, the marginal correlation of choices with Feature 0 will be strongly negative, while correlations with other features will be positive. This creates a high variance among the marginal feature correlations (r_j).

For the Mixture model, the Take-The-Best (TTB) component relies heavily on Feature 0 (the highest-validity cue) to break ties and drive choices. This creates a strong positive correlation between TTB's choices and Feature 0, which perfectly opposes and cancels out the negative correlation produced by the tallying component. As a result, the marginal feature correlations for the Mixture model are severely compressed towards zero, yielding a significantly lower variance across the 5 features compared to the pure Tallying model. This provides a robust, low-noise contrast between the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Convert ratings to numpy arrays
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    # choice_a is 1 if Option A was chosen, -1 if Option B was chosen
    choice_a = np.where(data['response'].values == 0, 1.0, -1.0)
    
    # Feature differences (A - B)
    delta = a_ratings - b_ratings
    
    # Calculate the marginal correlation (covariance) of the choice with each feature difference
    # r_j = mean(choice_a * delta_j) over all trials
    r = np.mean(choice_a[:, None] * delta, axis=0)
    
    # Return the variance of these feature correlations
    return float(np.var(r))
outcome: self_sim=0.0067 (var=0.0003) adversary_sim=0.0090 (var=0.0003) welch_t=-0.457 p=0.6499 (N=25, alpha=0.01) -> reject

[3] rationale: By isolating trials with the maximum tally difference (difference of 2), we probe the upper limits of determinism permitted by each model. In the pure Tallying model, the sensitivity parameter 'beta' is strictly bounded between [0.0, 1.0], meaning the model can never produce highly deterministic choice probabilities even when the tally difference is at its maximum. In contrast, the Mixture model allows 'beta_tally' to range up to 2.0. This allows the compensatory component of the Mixture model to be far more deterministic. Although the Take-The-Best (TTB) component opposes the tally winner on these trials, the ability of the Mixture model to reach much higher compensatory log-odds (due to the doubled beta upper bound) systematically stretches the distribution of Tally-winner choice proportions, yielding a reliably higher mean proportion of Tally-consistent choices for the Mixture model on these extreme trials.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    responses = data['response'].values
    
    # Calculate the tally of positive features for each option
    tally_a = a_ratings.sum(axis=1)
    tally_b = b_ratings.sum(axis=1)
    tally_diff = np.abs(tally_a - tally_b)
    
    # Identify which option is the Tally winner
    tally_winner = np.where(tally_a > tally_b, 0, np.where(tally_b > tally_a, 1, -1))
    
    # Filter for trials where the tally difference is exactly 2 (the most extreme opposed trials)
    diff2_mask = (tally_diff == 2)
    if not np.any(diff2_mask):
        return 0.5
        
    tally_winner_diff2 = tally_winner[diff2_mask]
    responses_diff2 = responses[diff2_mask]
    
    # Calculate the proportion of choices that align with the Tally winner
    matches = (responses_diff2 == tally_winner_diff2)
    
    return float(np.mean(matches))
outcome: self_sim=0.6542 (var=0.0161) adversary_sim=0.6650 (var=0.0190) welch_t=-0.289 p=0.7737 (N=25, alpha=0.01) -> reject

[4] rationale: This metric contrasts the probability of choosing the Tally Winner in opposed trials against the probability of choosing the TTB Winner in tied trials. For pure Tallying, choices in tied trials are essentially random noise around 0.5, so P(TTB|Tied) is 0.5, leaving the metric largely driven by P(Tally|Opposed). For the Mixture model, TTB actively pulls choices toward the TTB winner in tied trials (raising P(TTB|Tied) > 0.5) and simultaneously acts against the Tally Winner in opposed trials (lowering P(Tally|Opposed)). This dual-action of the TTB component in the Mixture model shrinks the difference between these two probabilities significantly more than in the pure Tallying model, providing a robust contrast that leverages all informative trials.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    responses = data['response'].values
    
    tally_a = a_ratings.sum(axis=1)
    tally_b = b_ratings.sum(axis=1)
    tally_diff = np.abs(tally_a - tally_b)
    tally_winner = np.where(tally_a > tally_b, 0, np.where(tally_b > tally_a, 1, -1))
    
    ttb_winner = np.zeros(len(a_ratings), dtype=int)
    for i in range(len(a_ratings)):
        a, b = a_ratings[i], b_ratings[i]
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_winner[i] = 0
                break
            elif b[j] > a[j]:
                ttb_winner[i] = 1
                break
                
    tied_mask = (tally_diff == 0)
    opposed_mask = (tally_diff > 0) & (tally_winner != ttb_winner)
    
    if not np.any(tied_mask) or not np.any(opposed_mask):
        return 0.0
        
    p_ttb_tied = np.mean(responses[tied_mask] == ttb_winner[tied_mask])
    p_tally_opposed = np.mean(responses[opposed_mask] == tally_winner[opposed_mask])
    
    return float(p_tally_opposed - p_ttb_tied)
outcome: self_sim=0.0818 (var=0.0234) adversary_sim=0.1012 (var=0.0117) welch_t=-0.519 p=0.6062 (N=25, alpha=0.01) -> reject

[5] rationale: This metric exploits the difference in the maximum allowed determinism between the two models. In the pure Tallying model, the sensitivity parameter 'beta' is strictly bounded between [0.0, 1.0], which mathematically limits the maximum probability of choosing the Tally winner even in the most extreme trials (tally difference of 2). In the Mixture model, 'beta_tally' can range up to 2.0. Although the Take-The-Best component consistently penalizes the Tally winner in opposed trials, subjects in the Mixture model with a low 'w_ttb' and a high 'beta_tally' will exhibit choice probabilities for the Tally winner that far exceed the theoretical maximum permitted by the pure Tallying model. By calculating the maximum proportion of Tally-consistent choices across the unique opposed trials for each subject, we capture this heavy right tail of determinism. The Mixture model will produce significantly more subjects hitting ceiling (e.g., 12/12 consistent choices on a trial) than the strictly bounded Tallying model.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    responses = data['response'].values
    
    # Calculate tallies for each option
    tally_a = a_ratings.sum(axis=1)
    tally_b = b_ratings.sum(axis=1)
    tally_diff = np.abs(tally_a - tally_b)
    
    # Identify the tally winner (0 for A, 1 for B, -1 for tie)
    tally_winner = np.where(tally_a > tally_b, 0, np.where(tally_b > tally_a, 1, -1))
    
    # We want to isolate the opposed trials (where tally difference > 0)
    opposed_mask = (tally_diff > 0)
    if not np.any(opposed_mask):
        return 0.5
        
    # Get the responses and winners for opposed trials
    opposed_responses = responses[opposed_mask]
    opposed_winners = tally_winner[opposed_mask]
    
    # Group by the specific trial types to find the maximum determinism
    # We compute the proportion of tally-consistent choices for each unique opposed trial
    # To do this, we can just group by the tally difference and the specific feature patterns
    # An easy surrogate is the string representation of the ratings
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    trial_ids = a_str + "_" + b_str
    
    opposed_trial_ids = trial_ids.values[opposed_mask]
    unique_trials = np.unique(opposed_trial_ids)
    
    max_prop = 0.0
    for t_id in unique_trials:
        idx = (opposed_trial_ids == t_id)
        if np.sum(idx) > 0:
            prop = np.mean(opposed_responses[idx] == opposed_winners[idx])
            if prop > max_prop:
                max_prop = prop
                
    return float(max_prop)
outcome: self_sim=0.6600 (var=0.0187) adversary_sim=0.6383 (var=0.0143) welch_t=+0.596 p=0.5542 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    # Convert ratings to numpy arrays\n    a_ratings = np.array(data['option_a_ratings'].tolist())\n    b_ratings = np.array(data['option_b_ratings'].tolist())\n    \n    # choice is 1.0 if Option A was chosen, -1.0 if Option B was chosen\n    choice = np.where(data['response'].values == 0, 1.0, -1.0)\n    \n    # Feature differences (A - B) for the highest-validity cue and two lower-validity cues\n    f0_diff = a_ratings[:, 0] - b_ratings[:, 0]\n    f2_diff = a_ratings[:, 2] - b_ratings[:, 2]\n    f3_diff = a_ratings[:, 3] - b_ratings[:, 3]\n    \n    # Custom contrast V that perfectly opposes TTB and Tallying\n    # Tallying predicts a strongly negative correlation with V\n    # TTB predicts a strongly positive correlation with V\n    V = f0_diff - f2_diff - f3_diff\n    \n    # Return the mean correlation of choice with the custom contrast V\n    return float(np.mean(choice * V))\n",
  "rationale": "By analyzing the specific trial structure, we can construct a custom feature contrast `V = F0_diff - F2_diff - F3_diff`. For the pure Tallying model, choice probability is driven entirely by the simple tally difference. Across the 8 unique trials, this contrast `V` is strongly negatively correlated with the tally difference, meaning pure Tallying predicts a robustly negative correlation between choices and `V`. In stark contrast, the Take-The-Best (TTB) heuristic relies primarily on the highest-validity cue (F0), and its decisions perfectly align with the sign of `V` on every single trial where `V` is non-zero. Consequently, pure TTB predicts a strongly positive correlation with `V`. The Mixture model, which incorporates a TTB component, will therefore exhibit a significantly higher (less negative) mean correlation with `V` than the pure Tallying model. This custom contrast effectively aggregates the opposing predictions of the two theories across all informative trials, yielding a large effect size with low per-subject variance."
}
```

## Usage

```json
{
  "prompt_token_count": 7557,
  "candidates_token_count": 556,
  "total_token_count": 26042
}
```
