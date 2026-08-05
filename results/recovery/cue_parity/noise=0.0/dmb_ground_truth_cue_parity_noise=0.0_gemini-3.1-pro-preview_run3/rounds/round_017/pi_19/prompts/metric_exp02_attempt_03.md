# metric_exp02_attempt_03

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

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 3: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 5: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 6: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Validity-Threshold with WADD Fallback model (Advocated Theory) from the pure Take The Best model (Competing Theory), we systematically shift the top discriminating cue down the validity hierarchy while ensuring all remaining lower-validity cues favor the opposing option. Pure TTB strictly stops at the first discriminating cue, completely ignoring its validity magnitude and all subsequent cues; thus, it predicts an identical, perfectly flat probability of choosing Option A across all trials. In contrast, the Advocated Theory posits a validity threshold. When the top discriminating cue's validity is above the threshold, the model uses TTB and chooses Option A. However, when the top cue's validity falls below the threshold, the model falls back to WADD. Because all lower-validity cues oppose the top cue, this fallback causes a sudden reversal in preference toward Option B. As the top cue shifts even lower, the number of opposing cues decreases, and the WADD score shifts back toward Option A. This highly non-monotonic pattern—high probability for A, sudden drop to B, and gradual return to A—uniquely identifies the threshold-fallback mechanism and directly contradicts the flat line predicted by pure TTB.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Validity-Threshold Model with WADD Fallback: Decision-makers process cues sequentially in order of validity, but their choice of strategy depends on the absolute strength of the best available evidence. If the first discriminating cue has a high validity (exceeding an internal threshold), the decision-maker relies purely on Take The Best (TTB), ignoring all other cues. If the top discriminating cue falls below this threshold, the decision-maker falls back to a compensatory Weighted Additive (WADD) strategy, integrating all available cues weighted by their validities.

**Parameters:**
- validity_threshold: [0.0, 1.0]
- beta_ttb: [0.1, 20.0]
- beta_comp: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    threshold = float(parameters["validity_threshold"])
    beta_ttb = float(parameters["beta_ttb"])
    beta_comp = float(parameters["beta_comp"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    disc_cue_idx = -1
    for j in cue_order:
        if a[j] != b[j]:
            disc_cue_idx = j
            break
            
    if disc_cue_idx == -1:
        return np.array([0.5, 0.5])
        
    v_disc = val[disc_cue_idx]
    
    if v_disc >= threshold:
        # Take The Best: rely solely on the first discriminating cue
        scores = np.array([1.0, 0.0]) if a[disc_cue_idx] > b[disc_cue_idx] else np.array([0.0, 1.0])
        z = beta_ttb * scores
    else:
        # WADD: weighted sum of cues as a compensatory fallback
        wadd_a = float(np.sum(val * a))
        wadd_b = float(np.sum(val * b))
        scores = np.array([wadd_a, wadd_b])
        z = beta_comp * scores
        
    z -= np.max(z)  # Numerical stability
    p_core = np.exp(z) / np.sum(np.exp(z))
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
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
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
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
[0] rationale: In all 6 trials of the experimental design, the top discriminating cue favors Option A. Thus, pure Take The Best (TTB) predicts a constant probability of choosing A across all trials (ignoring all lower-validity cues). However, the Advocated Theory posits a fallback to the Weighted Additive (WADD) strategy when the top cue's validity falls below a threshold. In trials where WADD favors Option B (which corresponds to trials where the sum of features for B is greater than A), the probability of choosing A will drop. This metric computes the difference in the rate of choosing A between trials where WADD favors A and trials where WADD favors B. TTB predicts this difference will be near 0, whereas the Advocated Theory predicts a positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Calculate sum of ratings for A and B
    sum_a = data['option_a_ratings'].apply(np.sum)
    sum_b = data['option_b_ratings'].apply(np.sum)
    
    # Trials where WADD favors A vs B
    wadd_favors_a = sum_a >= sum_b
    
    # Proportion of choosing A (response == 0)
    p_a_given_wadd_a = (data.loc[wadd_favors_a, 'response'] == 0).mean()
    p_a_given_wadd_b = (data.loc[~wadd_favors_a, 'response'] == 0).mean()
    
    return float(p_a_given_wadd_a - p_a_given_wadd_b)
outcome: self_sim=0.0616 (var=0.0254) adversary_sim=0.0125 (var=0.0055) welch_t=+1.397 p=0.1714 (N=25, alpha=0.01) -> reject

[1] rationale: In Trial 1, the first discriminating cue has a very high validity (0.95), making it extremely likely to exceed the threshold in the Advocated Theory, leading to a TTB strategy (which favors A). In Trial 6, the first discriminating cue has a low validity (0.65), making it much more likely to fall below the threshold and trigger the WADD fallback strategy (which favors B). Thus, the Advocated Theory predicts a large drop in the probability of choosing A from Trial 1 to Trial 6. In contrast, pure TTB ignores both the validity magnitude and the lower-validity cues, predicting a perfectly flat probability of choosing A across both trials.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Identify trials by the sum of features to distinguish them
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Trial 1: Top cue is Feature 1 (validity 0.95). TTB favors A, WADD favors B.
    trial_1_mask = (sum_a == 1) & (sum_b == 5)
    
    # Trial 6: Top cue is Feature 4 (validity 0.65). TTB favors A, WADD favors B.
    trial_6_mask = (sum_a == 4) & (sum_b == 5)
    
    # Probability of choosing Option A (response == 0)
    p_a_t1 = (data.loc[trial_1_mask, 'response'] == 0).mean()
    p_a_t6 = (data.loc[trial_6_mask, 'response'] == 0).mean()
    
    # Pure TTB predicts this difference will be 0, because the top cue favors A in both.
    # The Advocated Theory predicts a positive difference because Feature 1's high validity 
    # triggers TTB (choosing A), while Feature 4's lower validity triggers WADD fallback (choosing B).
    return float(p_a_t1 - p_a_t6)
outcome: self_sim=0.0837 (var=0.0763) adversary_sim=0.0250 (var=0.0217) welch_t=+0.938 p=0.3544 (N=25, alpha=0.01) -> reject

[2] rationale: To reduce per-subject variance and maximize the contrast between the theories, this metric compares the probability of choosing Option A across two pooled sets of trials. In Trials 4 and 5, both the Take The Best (TTB) and Weighted Additive (WADD) strategies favor Option A, leading to a high probability of choosing A under both models. In Trials 3 and 6, TTB still favors A, but WADD strongly favors B. Because the top discriminating cues in Trials 3 and 6 have relatively low validities (0.75 and 0.65), the Advocated Theory predicts these cues will often fall below the subject's validity threshold, triggering the WADD fallback and dropping the probability of choosing A. Pure TTB, ignoring validity magnitudes and lower-validity cues, predicts a flat, high probability of choosing A across all these trials. By pooling two trials per condition, we reduce response noise and stabilize the per-subject metric, yielding a large positive difference for the Advocated Theory and a near-zero difference for TTB.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate the sum of ratings for options A and B to uniquely identify trials
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Trials 4 and 5: Both TTB and WADD favor Option A.
    # T4: sum_a=4, sum_b=4. T5: sum_a=6, sum_b=2.
    mask_wadd_a = ((sum_a == 4) & (sum_b == 4)) | ((sum_a == 6) & (sum_b == 2))
    
    # Trials 3 and 6: TTB favors Option A, but WADD strongly favors Option B.
    # Moreover, the top discriminating cues in these trials have lower validities (0.75 and 0.65),
    # making them highly likely to fall below the Validity-Threshold, triggering WADD fallback.
    # T3: sum_a=3, sum_b=5. T6: sum_a=4, sum_b=5.
    mask_wadd_b = ((sum_a == 3) & (sum_b == 5)) | ((sum_a == 4) & (sum_b == 5))
    
    # Proportion of choosing A (response == 0)
    p_a_wadd_a = (data.loc[mask_wadd_a, 'response'] == 0).mean()
    p_a_wadd_b = (data.loc[mask_wadd_b, 'response'] == 0).mean()
    
    return float(p_a_wadd_a - p_a_wadd_b)
outcome: self_sim=0.0988 (var=0.0507) adversary_sim=0.0169 (var=0.0088) welch_t=+1.678 p=0.1031 (N=25, alpha=0.01) -> reject

[3] rationale: To maximize the contrast between pure Take The Best (TTB) and the Validity-Threshold with WADD Fallback model, we isolate Trials 3 and 5. In both trials, the top discriminating cue is the exact same (Feature 3, validity 0.75) and favors Option A. Pure TTB strictly follows this top cue, predicting identical probabilities of choosing A across both trials. However, if the subject's validity threshold exceeds 0.75, the Advocated Theory predicts a fallback to the compensatory WADD strategy. In Trial 5, WADD strongly favors Option A, but in Trial 3, WADD strongly favors Option B. By comparing the choice rates specifically on these two extreme trials, we avoid diluting the effect with weaker trials (like Trials 4 and 6), thereby maximizing the mean difference for the Advocated Theory while keeping TTB's prediction exactly at zero.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Calculate the sum of ratings to uniquely identify trials
    sum_a = data['option_a_ratings'].apply(np.sum)
    sum_b = data['option_b_ratings'].apply(np.sum)
    
    # Trial 5: Top discriminating cue is Feature 3 (validity 0.75).
    # WADD strongly favors Option A (WADD_A - WADD_B = +2.5).
    mask_t5 = (sum_a == 6) & (sum_b == 2)
    
    # Trial 3: Top discriminating cue is also Feature 3 (validity 0.75).
    # WADD strongly favors Option B (WADD_A - WADD_B = -1.05).
    mask_t3 = (sum_a == 3) & (sum_b == 5)
    
    # Proportion of choosing Option A (response == 0) in these trials
    p_a_t5 = (data.loc[mask_t5, 'response'] == 0).mean()
    p_a_t3 = (data.loc[mask_t3, 'response'] == 0).mean()
    
    return float(p_a_t5 - p_a_t3)
outcome: self_sim=0.1012 (var=0.0728) adversary_sim=0.0100 (var=0.0172) welch_t=+1.521 p=0.1374 (N=25, alpha=0.01) -> reject

[4] rationale: In all trials of the experiment, the top discriminating cue favors Option A. Thus, pure Take The Best (TTB) predicts a constant probability of choosing A across all trials. However, the Advocated Theory posits a fallback to the Weighted Additive (WADD) strategy when the top cue's validity falls below a threshold. In Trials 3 and 5, WADD strongly favors Option A, whereas in Trials 1 and 4, WADD favors Option B. By comparing the rate of choosing A between these two pooled sets of trials (sum_b == 0 vs sum_b >= 2), we compute a metric that pure TTB expects to be zero, while the Advocated Theory expects to be significantly positive. Pooling multiple trials per condition reduces within-subject response noise, leading to tighter per-subject estimates and higher discriminability.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Calculate the sum of ratings for option B to distinguish trials
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Trials 3 and 5: sum_b == 0. 
    # In these trials, WADD strongly favors Option A.
    mask_wadd_a = (sum_b == 0)
    
    # Trials 1 and 4: sum_b >= 2.
    # In these trials, WADD favors Option B.
    mask_wadd_b = (sum_b >= 2)
    
    # Proportion of choosing Option A (response == 0)
    p_a_wadd_a = (data.loc[mask_wadd_a, 'response'] == 0).mean()
    p_a_wadd_b = (data.loc[mask_wadd_b, 'response'] == 0).mean()
    
    return float(p_a_wadd_a - p_a_wadd_b)
outcome: self_sim=0.1521 (var=0.0815) adversary_sim=0.0058 (var=0.0039) welch_t=+2.503 p=0.01884 (N=25, alpha=0.01) -> reject

[5] rationale: To maximize the discriminability between the theories and reduce per-subject variance, we must carefully select which trials to pool. Pure TTB predicts an identical, flat probability of choosing Option A across all 5 trials because Option A always wins on the top discriminating cue. The Advocated Theory expects a drop in the probability of choosing A only when the top cue's validity is low enough to trigger the WADD fallback AND WADD favors Option B. Trial 1 perfectly fits this: the top cue is Feature 3 (validity 0.65), and WADD strongly favors B. In contrast, while WADD also favors B in Trial 4, its top cue is Feature 1 (validity 0.90), meaning the WADD fallback is rarely triggered. Previous attempts diluted the effect by pooling Trial 4 with Trial 1. By strictly comparing Trial 1 against a pooled baseline of Trials 2, 3, and 5 (where both TTB and WADD favor A), we maximize the mean difference while significantly reducing the variance of the baseline estimate.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Calculate the sum of ratings for option B to distinguish trials
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Trials 2, 3, and 5: sum_b <= 1.
    # In these trials, both Take The Best (TTB) and the Weighted Additive (WADD) fallback favor Option A.
    mask_high = sum_b <= 1
    
    # Trial 1: sum_b == 2.
    # The top discriminating cue is Feature 3 (validity 0.65), which will frequently fall below the validity threshold.
    # When this happens, the WADD fallback is triggered, which favors Option B, lowering the probability of choosing A.
    mask_low = sum_b == 2
    
    # Proportion of choosing Option A (response == 0)
    p_a_high = (data.loc[mask_high, 'response'] == 0).mean()
    p_a_low = (data.loc[mask_low, 'response'] == 0).mean()
    
    return float(p_a_high - p_a_low)
outcome: self_sim=0.1537 (var=0.0744) adversary_sim=0.0112 (var=0.0089) welch_t=+2.468 p=0.01958 (N=25, alpha=0.01) -> reject

[6] rationale: This metric isolates the two trials (Trial 1 and Trial 3) that share the exact same top discriminating cue (Feature 3, validity 0.65). Because the top cue is identical and favors Option A in both trials, pure Take The Best (TTB) predicts the exact same probability of choosing Option A for both, yielding a difference of zero. However, the Advocated Theory posits that a low-validity top cue (like 0.65) is likely to fall below the subject's validity threshold, triggering a fallback to the compensatory Weighted Additive (WADD) strategy. When WADD is triggered, it strongly favors Option A in Trial 3 but strongly favors Option B in Trial 1. Thus, the Advocated Theory predicts a large positive difference between the choice rates of A in these two specific trials. By strictly comparing these two perfectly matched trials, we isolate the WADD fallback effect without diluting the mean difference with trials where the top cue is strong (e.g., Trial 4 or 5).
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Calculate the sum of ratings to uniquely identify trials
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Trial 3: Top cue is Feature 3 (validity 0.65). WADD strongly favors Option A.
    mask_t3 = (sum_a == 3) & (sum_b == 0)
    
    # Trial 1: Top cue is also Feature 3 (validity 0.65). WADD strongly favors Option B.
    mask_t1 = (sum_a == 1) & (sum_b == 2)
    
    # Proportion of choosing Option A (response == 0)
    p_a_t3 = (data.loc[mask_t3, 'response'] == 0).mean()
    p_a_t1 = (data.loc[mask_t1, 'response'] == 0).mean()
    
    return float(p_a_t3 - p_a_t1)
outcome: self_sim=0.1884 (var=0.1094) adversary_sim=0.0084 (var=0.0098) welch_t=+2.607 p=0.01442 (N=25, alpha=0.01) -> reject

[7] rationale: By strictly isolating Trials 1, 2, and 3, we control for the top discriminating cue (Feature 3, validity 0.65). Because this cue always favors Option A, pure Take The Best (TTB) predicts identical choice rates across all three trials, yielding a difference of zero. However, the Advocated Theory posits that Feature 3's lower validity will frequently trigger a fallback to the compensatory Weighted Additive (WADD) strategy. In Trials 2 and 3, WADD favors Option A, but in Trial 1, WADD favors Option B. By comparing the pooled choice rate of A in Trials 2 and 3 against Trial 1, we capture the large drop predicted by the Advocated Theory while reducing per-subject variance by pooling multiple trials for the baseline.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Calculate the sum of ratings to identify specific trials
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Trials 1, 2, and 3 uniquely have Feature 3 as the top discriminating cue.
    # They can be identified by the total number of features across A and B being <= 3.
    is_f3_top = (sum_a + sum_b) <= 3
    
    # In Trials 2 and 3, the WADD fallback strongly favors Option A.
    mask_t23 = is_f3_top & (sum_b <= 1)
    
    # In Trial 1, the WADD fallback strongly favors Option B.
    mask_t1 = is_f3_top & (sum_b == 2)
    
    # Proportion of choosing Option A (response == 0)
    p_a_t23 = (data.loc[mask_t23, 'response'] == 0).mean()
    p_a_t1 = (data.loc[mask_t1, 'response'] == 0).mean()
    
    return float(p_a_t23 - p_a_t1)
outcome: self_sim=0.1347 (var=0.0631) adversary_sim=0.0068 (var=0.0093) welch_t=+2.377 p=0.02382 (N=25, alpha=0.01) -> reject

[8] rationale: In this specific experimental design, Option A is always favored by the top discriminating cue. As a result, pure Take The Best (TTB) predicts a constant, high probability of choosing Option A across all 6 trials. However, the Advocated Theory (Validity-Threshold with WADD Fallback) predicts a non-monotonic, U-shaped pattern in the probability of choosing A. In Trials 1 and 2, the top cue validity is high enough to trigger TTB (choosing A). In Trials 5 and 6, the top cue validity is low, triggering WADD, but WADD also favors A. In contrast, for Trials 3 and 4, the top cue validity is low enough to frequently trigger WADD, but WADD favors Option B, causing a sudden drop in the rate of choosing A. By pooling the high-probability trials (1, 2, 5, 6) and comparing them against the drop-off trials (3, 4), this metric captures the unique non-monotonic signature of the threshold fallback mechanism while significantly reducing per-subject variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Option B's sum of ratings uniquely identifies the trial in this design
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Trials 1, 2, 5, and 6 correspond to sum_b in [0, 1, 4, 5].
    # In Trials 1 & 2, the top cue validity is very high (0.95, 0.85), so the Advocated 
    # Theory almost always uses TTB (which favors Option A).
    # In Trials 5 & 6, the top cue validity is low, triggering WADD fallback, but 
    # WADD also strongly favors Option A.
    mask_high = sum_b.isin([0, 1, 4, 5])
    
    # Trials 3 & 4 correspond to sum_b in [2, 3].
    # Here, the top cue validity is moderate (0.75, 0.65), frequently falling below 
    # the subject's threshold and triggering WADD fallback. Crucially, in these trials, 
    # WADD favors Option B, which lowers the probability of choosing Option A.
    mask_low = sum_b.isin([2, 3])
    
    # Pure TTB predicts a flat probability of choosing A across all trials because 
    # the top discriminating cue always favors A. Thus, TTB expects a difference of 0.
    # The Advocated Theory expects a significant positive difference due to the WADD fallback.
    p_a_high = (data.loc[mask_high, 'response'] == 0).mean()
    p_a_low = (data.loc[mask_low, 'response'] == 0).mean()
    
    return float(p_a_high - p_a_low)

outcome: self_sim=0.1197 (var=0.0670) adversary_sim=0.0138 (var=0.0059) welch_t=+1.963 p=0.05963 (N=25, alpha=0.01) -> reject

[9] rationale: To maximize discriminability and minimize per-subject variance, this metric compares two carefully chosen sets of trials. In Trials 5 and 6, both the Take The Best (TTB) and Weighted Additive (WADD) strategies favor Option A, ensuring that the Advocated Theory predicts a consistently high probability of choosing A across all subjects, regardless of their individual validity thresholds. This provides a very stable, low-variance baseline. In Trials 3 and 4, TTB still favors Option A, but WADD strongly favors Option B. Because the top discriminating cues in these trials have moderate validities (0.75 and 0.65), they are highly likely to fall below the subject's validity threshold, triggering the WADD fallback and causing a drop in the rate of choosing A. Pure TTB ignores lower-validity cues and predicts an identical, flat probability of choosing A across all these trials, yielding a difference of zero.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Option B's sum of ratings uniquely identifies the trial in this design
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Trials 5 and 6 correspond to sum_b in [0, 1].
    # In these trials, both Take The Best (TTB) and the Weighted Additive (WADD) 
    # fallback strategy favor Option A. Thus, the Advocated Theory predicts a 
    # consistently high probability of choosing A for all subjects, regardless of 
    # their specific validity threshold, minimizing between-subject variance.
    mask_high = sum_b <= 1
    
    # Trials 3 and 4 correspond to sum_b in [2, 3].
    # Here, TTB favors Option A, but WADD strongly favors Option B. Because the top 
    # cue validities in these trials are moderate (0.75, 0.65), they frequently fall 
    # below the subject's threshold, triggering the WADD fallback and significantly 
    # lowering the probability of choosing Option A.
    mask_low = sum_b.isin([2, 3])
    
    # Pure TTB predicts a flat probability of choosing A across all trials because 
    # the top discriminating cue always favors A. Thus, TTB expects a difference of 0.
    # The Advocated Theory expects a significant positive difference.
    p_a_high = (data.loc[mask_high, 'response'] == 0).mean()
    p_a_low = (data.loc[mask_low, 'response'] == 0).mean()
    
    return float(p_a_high - p_a_low)
outcome: self_sim=0.1056 (var=0.0696) adversary_sim=0.0150 (var=0.0067) welch_t=+1.640 p=0.1119 (N=25, alpha=0.01) -> reject

[10] rationale: To maximize discriminability and minimize variance, we compute a per-subject metric that isolates the WADD fallback effect and applies a non-linear clipping transformation. We compare the choice rate of Option A in Trials 5 and 6 (where both TTB and WADD favor A, ensuring high P(A) for all subjects) against Trial 4 (where TTB favors A but WADD favors B). Because Trial 4's top cue has a validity of 0.65, a large proportion of subjects in the Advocated Theory will have an internal validity threshold above this, triggering the WADD fallback and dropping their rate of choosing A. Pure TTB predicts a flat P(A) across all these trials. By computing the difference P(A|T5,T6) - P(A|T4) *per subject* and clipping it at 0 (max(0, diff)), we zero out the negative binomial noise from pure TTB subjects. This dramatically reduces the between-subject variance for the Competing Theory while preserving the large positive signal from the Advocated Theory's WADD-fallback subjects, leading to a highly significant t-statistic.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # We compute the metric per subject and return the mean.
    # This ensures the pooled metric perfectly matches the mean of per-subject metrics,
    # allowing us to apply a non-linear transformation (clipping) per subject to reduce variance.
    if 'subject_id' in data.columns and data['subject_id'].nunique() > 1:
        subject_vals = []
        for _, subj_df in data.groupby('subject_id'):
            sum_b = subj_df['option_b_ratings'].apply(sum)
            
            # Trials 5 and 6: Both TTB and WADD favor Option A.
            mask_high = sum_b.isin([0, 1])
            # Trial 4: TTB favors A, but WADD strongly favors B. 
            # Validity of top cue is 0.65, triggering WADD for ~35% of subjects.
            mask_low = sum_b == 2
            
            p_high = (subj_df.loc[mask_high, 'response'] == 0).mean() if mask_high.sum() > 0 else 0.5
            p_low = (subj_df.loc[mask_low, 'response'] == 0).mean() if mask_low.sum() > 0 else 0.5
            
            # Clip at 0 to eliminate negative noise from pure TTB subjects
            subject_vals.append(max(0.0, float(p_high - p_low)))
            
        return float(np.mean(subject_vals))
    else:
        sum_b = data['option_b_ratings'].apply(sum)
        mask_high = sum_b.isin([0, 1])
        mask_low = sum_b == 2
        
        p_high = (data.loc[mask_high, 'response'] == 0).mean() if mask_high.sum() > 0 else 0.5
        p_low = (data.loc[mask_low, 'response'] == 0).mean() if mask_low.sum() > 0 else 0.5
        
        return max(0.0, float(p_high - p_low))
outcome: self_sim=0.1625 (var=0.0631) adversary_sim=0.0475 (var=0.0050) welch_t=+2.204 p=0.03598 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Option B's sum of ratings uniquely identifies the trial in this design\n    sum_b = data['option_b_ratings'].apply(sum)\n    \n    # Trials 5 and 6 correspond to sum_b <= 1.\n    # In these trials, both Take The Best (TTB) and the Weighted Additive (WADD) \n    # fallback strategy favor Option A. Thus, the Advocated Theory predicts a \n    # consistently high probability of choosing A for all subjects.\n    mask_high = sum_b <= 1\n    \n    # Trials 1, 2, 3, and 4 correspond to sum_b >= 2.\n    # In all these trials, TTB favors Option A, but WADD favors Option B.\n    # Depending on the subject's validity threshold, the WADD fallback will be \n    # triggered in at least some of these trials (especially Trials 3 and 4 \n    # where the top cue validity is lower), lowering the overall rate of choosing A.\n    mask_low = sum_b >= 2\n    \n    # Pure TTB predicts a flat probability of choosing A across all trials because \n    # the top discriminating cue always favors A. Thus, TTB expects a difference of 0.\n    # By pooling 32 trials for mask_high and 64 trials for mask_low, we drastically \n    # minimize the binomial response noise, reducing the per-subject variance for TTB \n    # and maximizing the t-statistic for the contrast.\n    p_a_high = (data.loc[mask_high, 'response'] == 0).mean()\n    p_a_low = (data.loc[mask_low, 'response'] == 0).mean()\n    \n    return float(p_a_high - p_a_low)",
  "rationale": "By aggressively pooling the trials into two maximum-size bins (32 trials where WADD favors A vs 64 trials where WADD favors B), we drastically reduce the binomial noise for the Take The Best model, shrinking its per-subject variance. The Advocated Theory still predicts a robust positive difference because the WADD fallback, when triggered, only pulls preference toward B in the second bin. This combination of a large mean gap and minimized baseline variance maximizes the discriminability between the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 10768,
  "candidates_token_count": 547,
  "total_token_count": 15354
}
```
