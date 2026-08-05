# metric_exp01_attempt_03

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
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  trial 2: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** To dissociate pure Tallying from the Mixture of Tallying and TTB, we exploit trials where the highest-validity cue conflicts with the overall count of positive features. Pure Tallying predicts choices strictly based on the difference in tallies, yielding 50/50 probabilities when counts are tied, and extreme probabilities when one option has a large count advantage, entirely regardless of which specific cues are active. The Mixture model predicts that the highest-validity cue will exert an independent pull. Thus, on tally-tied trials, the Mixture model expects a systematic deviation towards the option with the top cue, and on tally-advantaged trials where the top cue opposes the tally, it expects a dampened choice probability compared to pure Tallying.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
[0] rationale: This metric isolates trials where the Tallying (Equal-Weight) heuristic predicts a perfect tie (i.e., both options have an equal number of positive cues). Under the pure Tallying theory, choice probabilities on these trials are exactly 50/50, leading to a theoretical mean of 0.5 across subjects. Under the Mixture theory, the addition of the Take-The-Best (TTB) component breaks the tie in favor of the option with the highest-validity discriminating cue. By calculating the proportion of choices that align with the TTB winner on these specific 'Tally-tie' trials, we expect the Mixture model to produce a mean significantly greater than 0.5, while the Tallying model remains anchored at 0.5.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        
        # Only consider trials where Tallying predicts a tie
        if sum_a == sum_b:
            diff = a - b
            non_zero = np.where(diff != 0)[0]
            if len(non_zero) > 0:
                # Since validities are strictly decreasing from index 0 to 4,
                # the first non-zero difference indicates the TTB winner.
                first_diff_idx = non_zero[0]
                ttb_winner = 0 if diff[first_diff_idx] > 0 else 1
                
                if row['response'] == ttb_winner:
                    ttb_match.append(1.0)
                else:
                    ttb_match.append(0.0)
                    
    if not ttb_match:
        return 0.5
    return float(np.mean(ttb_match))
outcome: self_sim=0.4917 (var=0.0063) adversary_sim=0.5394 (var=0.0101) welch_t=-1.865 p=0.06858 (N=25, alpha=0.01) -> reject

[1] rationale: This metric isolates trials where the Tallying heuristic strongly favors one option (a difference of 3 positive cues). We split these trials into two sets: 'Aligned' trials where Take-The-Best (TTB) also favors the Tallying winner, and 'Opposed' trials where TTB favors the Tallying loser. Under the pure Tallying theory, the choice probability depends entirely on the cue sum difference, meaning the rate of choosing the Tallying winner will be identical across both sets (yielding a metric of 0). Under the Mixture theory, the TTB component boosts the probability of choosing the Tallying winner on Aligned trials and depresses it on Opposed trials, yielding a strictly positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    aligned_choices = []
    opposed_choices = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        
        # Only look at trials with a cue sum difference of exactly 3
        if abs(sum_a - sum_b) == 3:
            tally_winner = 0 if sum_a > sum_b else 1
            
            # Determine TTB winner (first cue that differs, since validities are strictly decreasing)
            diff = a - b
            non_zero = np.where(diff != 0)[0]
            if len(non_zero) > 0:
                first_diff_idx = non_zero[0]
                ttb_winner = 0 if diff[first_diff_idx] > 0 else 1
            else:
                continue
                
            is_aligned = (tally_winner == ttb_winner)
            chose_tally_winner = (row['response'] == tally_winner)
            
            if is_aligned:
                aligned_choices.append(1.0 if chose_tally_winner else 0.0)
            else:
                opposed_choices.append(1.0 if chose_tally_winner else 0.0)
                
    if not aligned_choices or not opposed_choices:
        return 0.0
        
    return float(np.mean(aligned_choices) - np.mean(opposed_choices))
outcome: self_sim=0.0292 (var=0.0255) adversary_sim=0.0883 (var=0.0220) welch_t=-1.357 p=0.181 (N=25, alpha=0.01) -> reject

[2] rationale: By calculating the overall proportion of choices that align with the Take-The-Best (TTB) winner across all trials, we obtain a highly stable metric (averaging over all ~96 trials per subject, minimizing within-subject variance). In this specific experimental design, the TTB winner is heavily opposed to the Tallying winner on most non-tie trials. Pure Tallying will actively choose against the TTB winner on these opposed trials, leading to a systematically low overall TTB match rate. The Mixture model, however, includes a persistent TTB component that uniformly boosts the probability of choosing the TTB winner across all trial types. Evaluating this across the entire trial set maximizes the signal-to-noise ratio compared to subsetting.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_matches = 0
    total_trials = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # TTB winner is determined by the first cue that differs
        # (since validities are strictly decreasing from left to right)
        diff = a - b
        non_zero = np.where(diff != 0)[0]
        if len(non_zero) > 0:
            first_diff_idx = non_zero[0]
            ttb_winner = 0 if diff[first_diff_idx] > 0 else 1
            
            if row['response'] == ttb_winner:
                ttb_matches += 1
            total_trials += 1
            
    if total_trials == 0:
        return 0.5
    return float(ttb_matches / total_trials)
outcome: self_sim=0.4596 (var=0.0039) adversary_sim=0.4719 (var=0.0071) welch_t=-0.587 p=0.5605 (N=25, alpha=0.01) -> reject

[3] rationale: To perfectly discriminate the pure Tallying model from the Mixture model, we can construct a linear combination of choice probabilities across specific trial types such that the expected value is exactly 0 under pure Tallying, but strictly positive under the Mixture model. Pure Tallying depends only on the difference in cue tallies. Therefore, trials with identical tally differences must yield identical choice probabilities. For a tally difference of +3, Tallying predicts equal probability of choosing A for Trial 1 and Trial 2. However, Take-The-Best (TTB) favors B on Trial 1 and A on Trial 2. For a tally difference of -3 (Trial 5), Tallying predicts the exact complement probability, whereas TTB strongly favors A. For a tally difference of 0 (Trials 3, 4, and 8), Tallying predicts exactly 50% for all. TTB favors A on Trials 3 and 8, but favors B on Trial 4. By computing `Mean(A|T2) - 0.5*Mean(A|T1) + 0.5*Mean(A|T5) + 0.5*Mean(A|T3) + 0.5*Mean(A|T8) - Mean(A|T4) - 0.5`, all base choice probabilities under pure Tallying perfectly cancel out to 0 for every subject, regardless of their individual parameters. This reduces the between-subject variance under Tallying to pure sampling noise. Under the Mixture model, the TTB component breaks these symmetries systematically, yielding an expected value proportional to `w_ttb`. This combination maximizes the signal-to-noise ratio and provides a highly significant contrast.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    def get_trial_type(a, b):
        a_t = tuple(a)
        b_t = tuple(b)
        if a_t == (0, 1, 1, 1, 1) and b_t == (1, 0, 0, 0, 0): return 'T1'
        if a_t == (1, 1, 1, 1, 0) and b_t == (0, 0, 0, 0, 1): return 'T2'
        if a_t == (1, 0, 1, 0, 0) and b_t == (0, 1, 0, 1, 0): return 'T3'
        if a_t == (0, 1, 0, 1, 0) and b_t == (1, 0, 1, 0, 0): return 'T4'
        if a_t == (1, 0, 0, 0, 0) and b_t == (0, 1, 1, 1, 1): return 'T5'
        if a_t == (1, 1, 0, 0, 0) and b_t == (1, 0, 1, 0, 0): return 'T8'
        return None

    scores = []
    
    for subject, subj_df in data.groupby('subject_id'):
        counts = {'T1': 0, 'T2': 0, 'T3': 0, 'T4': 0, 'T5': 0, 'T8': 0}
        sums = {'T1': 0, 'T2': 0, 'T3': 0, 'T4': 0, 'T5': 0, 'T8': 0}
        
        for _, row in subj_df.iterrows():
            tt = get_trial_type(row['option_a_ratings'], row['option_b_ratings'])
            if tt is not None:
                counts[tt] += 1
                # response == 0 means subject chose option A
                if row['response'] == 0:
                    sums[tt] += 1
                    
        means = {}
        for tt in counts:
            if counts[tt] > 0:
                means[tt] = sums[tt] / counts[tt]
            else:
                means[tt] = 0.5
                
        # Linear combination of choice probabilities designed to exactly cancel out under pure Tallying
        m_val = (means['T2'] - 0.5 * means['T1'] + 0.5 * means['T5'] + 
                 0.5 * means['T3'] + 0.5 * means['T8'] - means['T4'] - 0.5)
        
        scores.append(m_val)
        
    return float(np.mean(scores))
outcome: self_sim=0.0158 (var=0.0665) adversary_sim=0.1658 (var=0.0879) welch_t=-1.909 p=0.06241 (N=25, alpha=0.01) -> reject

[4] rationale: This metric isolates trials where the Tallying heuristic predicts a moderate preference (a difference of exactly 1 positive cue). In this design, trials with a tally difference of 1 are perfectly split into two types: those where Take-The-Best (TTB) aligns with the Tallying winner (Trials 7 and 8), and those where TTB opposes the Tallying winner (Trials 5 and 6). Under the pure Tallying theory, the probability of choosing the Tallying winner depends strictly on the tally difference, meaning the choice rate will be identical on average across both sets (yielding an expected difference of 0). Under the Mixture theory, the TTB component boosts the probability of choosing the Tallying winner on Aligned trials and depresses it on Opposed trials. The difference between these two conditional probabilities is exactly proportional to the mixture weight of the TTB strategy, providing a strong, highly stable signal with near-zero theoretical mean for pure Tallying.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    scores = []
    for subject, subj_df in data.groupby('subject_id'):
        aligned_choices = []
        opposed_choices = []
        for _, row in subj_df.iterrows():
            a = np.array(row['option_a_ratings'])
            b = np.array(row['option_b_ratings'])
            sum_a = np.sum(a)
            sum_b = np.sum(b)
            
            if abs(sum_a - sum_b) == 1:
                tally_winner = 0 if sum_a > sum_b else 1
                
                diff = a - b
                non_zero = np.where(diff != 0)[0]
                if len(non_zero) > 0:
                    first_diff_idx = non_zero[0]
                    ttb_winner = 0 if diff[first_diff_idx] > 0 else 1
                else:
                    continue
                    
                chose_tally_winner = 1.0 if row['response'] == tally_winner else 0.0
                
                if tally_winner == ttb_winner:
                    aligned_choices.append(chose_tally_winner)
                else:
                    opposed_choices.append(chose_tally_winner)
                    
        if len(aligned_choices) > 0 and len(opposed_choices) > 0:
            scores.append(float(np.mean(aligned_choices) - np.mean(opposed_choices)))
        else:
            scores.append(0.0)
            
    return float(np.mean(scores))
outcome: self_sim=0.0058 (var=0.0226) adversary_sim=0.0658 (var=0.0197) welch_t=-1.459 p=0.1512 (N=25, alpha=0.01) -> reject

[5] rationale: This metric constructs a targeted linear combination of choice probabilities that perfectly cancels to 0 in expectation under the pure Tallying theory. Under Tallying, the choice probability depends only on the difference in tallies. Trials 7 and 5 have exactly the same tally difference (+1 for A), so P(A|T7) = P(A|T5). Similarly, Trials 6 and 8 have a tally difference of -1, so P(A|T6) = P(A|T8). Trials 1 and 2 are exact ties, so P(A|T1) = P(A|T2) = 0.5. Thus, the contrast (A7 - A5) + (A6 - A8) + (A1 - A2) equals 0 for every subject under pure Tallying. However, under the Mixture model, Take-The-Best strongly breaks these symmetries: it favors A on T7, T6, and T1, and favors B on T5, T8, and T2. This drives the contrast strictly positive, proportional to the mixture weight of the TTB strategy, generating a highly robust discriminator with near-zero theoretical variance for pure Tallying.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    def get_trial_type(a, b):
        a_t = tuple(a)
        b_t = tuple(b)
        if a_t == (1, 0, 0, 1, 1) and b_t == (0, 1, 1, 1, 0): return 'T1'
        if a_t == (0, 1, 1, 0, 1) and b_t == (1, 0, 0, 1, 1): return 'T2'
        if a_t == (0, 1, 1, 0, 0) and b_t == (1, 0, 0, 0, 0): return 'T5'
        if a_t == (1, 0, 0, 0, 0) and b_t == (0, 1, 1, 0, 0): return 'T6'
        if a_t == (1, 1, 1, 0, 0) and b_t == (0, 0, 0, 1, 1): return 'T7'
        if a_t == (0, 0, 0, 1, 1) and b_t == (1, 1, 1, 0, 0): return 'T8'
        return None

    scores = []
    for subject, subj_df in data.groupby('subject_id'):
        counts = {k: 0 for k in ['T1', 'T2', 'T5', 'T6', 'T7', 'T8']}
        sums = {k: 0 for k in ['T1', 'T2', 'T5', 'T6', 'T7', 'T8']}
        
        for _, row in subj_df.iterrows():
            tt = get_trial_type(row['option_a_ratings'], row['option_b_ratings'])
            if tt is not None:
                counts[tt] += 1
                # response == 0 means option A was chosen
                if row['response'] == 0:
                    sums[tt] += 1
                    
        means = {}
        for tt in counts:
            if counts[tt] > 0:
                means[tt] = sums[tt] / counts[tt]
            else:
                means[tt] = 0.5
                
        # Contrast designed to perfectly cancel out under pure Tallying
        c = (means['T7'] - means['T5']) + (means['T6'] - means['T8']) + (means['T1'] - means['T2'])
        scores.append(c)
        
    return float(np.mean(scores))
outcome: self_sim=0.0100 (var=0.1243) adversary_sim=0.2100 (var=0.1153) welch_t=-2.043 p=0.04662 (N=25, alpha=0.01) -> reject

[6] rationale: By focusing on trials where the Tallying heuristic predicts either a tie (tally difference 0) or a weak preference (tally difference 1), we perfectly isolate the effect of the Take-The-Best (TTB) strategy. Under pure Tallying, the probability of choosing the TTB-favored option on these trials perfectly balances out: the TTB winner aligns with the Tallying winner on half of the tally-difference-1 trials, and opposes it on the other half, while tie trials are split 50/50. Consequently, the expected proportion of TTB-aligned choices across this specific subset of trials is exactly 0.5 for every subject under pure Tallying, regardless of their individual noise or temperature parameters. This reduces the between-subject variance under Tallying to its theoretical minimum (pure binomial sampling noise over 72 trials). Under the Mixture model, however, the TTB component consistently biases choices toward the TTB winner across all these trials, pushing the expected proportion strictly above 0.5. By pooling 72 trials per subject, we maximize the signal-to-noise ratio, providing a highly significant contrast.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    scores = []
    for subject, subj_df in data.groupby('subject_id'):
        matches = 0
        count = 0
        
        for _, row in subj_df.iterrows():
            a = row['option_a_ratings']
            b = row['option_b_ratings']
            
            sum_a = sum(a)
            sum_b = sum(b)
            
            # Select trials where the tally difference is 0 or 1
            if abs(sum_a - sum_b) <= 1:
                # In this experimental design, the most valid cue (index 0)
                # ALWAYS differs between A and B. Thus, TTB always chooses
                # the option where cue 0 is 1.
                ttb_winner = 0 if a[0] > b[0] else 1
                
                if row['response'] == ttb_winner:
                    matches += 1
                count += 1
                
        if count > 0:
            scores.append(matches / count)
        else:
            scores.append(0.5)
            
    return float(np.mean(scores))
outcome: self_sim=0.5017 (var=0.0035) adversary_sim=0.5350 (var=0.0032) welch_t=-2.043 p=0.04662 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    scores = []\n    for subject, subj_df in data.groupby('subject_id'):\n        tally_1_choices = []\n        tally_2_choices = []\n        \n        for _, row in subj_df.iterrows():\n            a = np.array(row['option_a_ratings'])\n            b = np.array(row['option_b_ratings'])\n            sum_a = np.sum(a)\n            sum_b = np.sum(b)\n            \n            diff = abs(sum_a - sum_b)\n            if diff == 0:\n                continue\n                \n            tally_winner = 0 if sum_a > sum_b else 1\n            chose_tally_winner = 1.0 if row['response'] == tally_winner else 0.0\n            \n            if diff == 1:\n                tally_1_choices.append(chose_tally_winner)\n            elif diff == 2:\n                tally_2_choices.append(chose_tally_winner)\n                \n        if len(tally_1_choices) > 0 and len(tally_2_choices) > 0:\n            p1 = float(np.mean(tally_1_choices))\n            p2 = float(np.mean(tally_2_choices))\n            \n            # Under pure Tallying (softmax over tally differences), P(+2) can be \n            # perfectly predicted from P(+1) using the logistic function's properties.\n            denom = p1**2 + (1.0 - p1)**2\n            pred_p2 = (p1**2) / denom if denom > 0 else 0.5\n            \n            scores.append(p2 - pred_p2)\n        else:\n            scores.append(0.0)\n            \n    return float(np.mean(scores))",
  "rationale": "This metric exploits a mathematical invariant of the pure Tallying model to control for individual differences in temperature/noise, while perfectly isolating the structural asymmetry of the Take-The-Best (TTB) strategy in this specific experimental design.\n\nUnder pure Tallying, choices are made via a softmax over tally differences. This dictates a deterministic relationship between the probability of choosing the tally winner on diff=1 trials (p1) and diff=2 trials (p2): specifically, p2 = p1^2 / (p1^2 + (1-p1)^2). Thus, the residual `p2 - pred_p2` is expected to be exactly 0 for every subject under pure Tallying, regardless of their individual beta parameter.\n\nUnder the Mixture model, the TTB strategy is invoked on a proportion of trials. In this experimental design, TTB aligns with the tally winner on exactly half of the diff=1 trials, but it OPPOSES the tally winner on ALL diff=2 trials (Trials 3 and 4). Consequently, the TTB component heavily depresses the empirical `p2`, while leaving `p1` relatively unaffected (as its effects cancel out on average). This drives the metric strongly negative under the Mixture model, providing a highly significant discriminator with near-zero theoretical variance under pure Tallying."
}
```

## Usage

```json
{
  "prompt_token_count": 8599,
  "candidates_token_count": 779,
  "total_token_count": 40672
}
```
