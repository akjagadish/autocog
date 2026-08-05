# metric_exp01_attempt_00

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

**Trial pairs (n=5):**
  trial 1: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 3: A=[0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 4: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 5: A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate the pure Take The Best (TTB) model (Advocated Theory) from the Validity-Threshold with WADD Fallback model (Competing Theory), we systematically vary the support from lower-validity cues while holding the highest-validity discriminating cue constant at a relatively low validity. Pure TTB strictly stops at the first discriminating cue, completely ignoring all subsequent cues, and thus predicts an identical, flat probability of choosing Option A across all trials where the top cue favors A. In contrast, the Competing Theory posits a validity threshold: if the top discriminating cue's validity is below the threshold, it falls back to a compensatory WADD strategy. By setting the top discriminating cue to lower validities (e.g., Feature 3 or 4) and parametrically shifting the remaining cues from strongly favoring Option B to strongly favoring Option A, the Competing Theory predicts a monotonic shift in choice probability (due to WADD fallback), directly contradicting the perfectly flat line predicted by pure TTB.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
[0] rationale: This metric calculates the difference in the probability of choosing Option A between Trial 1 (where the discriminating cue has a very high validity of 0.95) and Trial 3 (where the discriminating cue has a lower validity of 0.70). Pure Take The Best (TTB) uses only the first discriminating cue regardless of its absolute validity, thus it predicts identical, high rates of choosing Option A for both trials (difference ~ 0). The Validity-Threshold with WADD Fallback model, however, is much more likely to abandon TTB and fall back to WADD on Trial 3 because the discriminating cue's validity is lower and more likely to fall below the internal threshold. Since WADD strongly favors Option B on both trials, this fallback causes a sharp drop in the probability of choosing Option A on Trial 3 compared to Trial 1, yielding a positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 and Trial 3 by Option A's features
    is_t1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    is_t3 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 1, 0, 0))
    
    # Calculate the proportion of times Option A was chosen (response == 0)
    t1_prob = (data.loc[is_t1, 'response'] == 0).mean()
    t3_prob = (data.loc[is_t3, 'response'] == 0).mean()
    
    # For pure TTB, the probability of choosing A is flat across these trials.
    # For the Competing Theory, Trial 3 triggers the WADD fallback more often, 
    # drastically reducing the probability of choosing A.
    return float(t1_prob - t3_prob)
outcome: self_sim=0.0000 (var=0.0114) adversary_sim=0.0989 (var=0.0784) welch_t=-1.651 p=0.1089 (N=25, alpha=0.01) -> reject

[1] rationale: Trial 3 is specifically designed so that Option A wins according to pure Take The Best (TTB), but Option B wins according to the compensatory WADD strategy. Furthermore, the discriminating cue for this trial has a validity of 0.70, which is the lowest among all trials where TTB and WADD disagree. Pure TTB strictly ignores the validity magnitude and predicts a near 100% choice rate for Option A. The Validity-Threshold with WADD Fallback model, however, has a high probability (~30%) of abandoning TTB on this trial due to the 0.70 validity falling below the threshold, leading to a fallback to WADD and a choice of Option B. By measuring the raw choice probability of Option A on Trial 3 alone, we obtain a metric that is extremely stable (near 1.0 with ~0 variance) for TTB, but significantly lower (~0.7) for the Competing Theory. This direct approach maximizes the t-statistic by avoiding the compound variance of differencing multiple trials.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Isolate Trial 3 where the discriminating cue has the lowest validity (0.70)
    # and the compensatory WADD score strongly favors Option B.
    is_t3 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 1, 0, 0))
    
    # Return the proportion of times Option A was chosen on Trial 3.
    return float((data.loc[is_t3, 'response'] == 0).mean())
outcome: self_sim=0.8568 (var=0.0110) adversary_sim=0.7147 (var=0.0868) welch_t=+2.273 p=0.03038 (N=25, alpha=0.01) -> reject

[2] rationale: By taking the difference in the probability of choosing Option B between Trial 3 and Trial 4, we perfectly control for the pure Take The Best (TTB) prediction. For pure TTB, both trials share the exact same discriminating cue (validity 0.70), and TTB favors Option A in both cases. Consequently, the probability of choosing Option B is purely driven by response noise and is expected to be identical, yielding a difference of 0. In contrast, the Competing Theory will fall back to WADD if the 0.70 validity is below its threshold. In Trial 3, WADD favors Option B, increasing P(B). In Trial 4, WADD favors Option A, keeping P(B) low. This results in a strong positive difference that isolates the structural divergence between the models while minimizing baseline variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Trial 3: TTB chooses A (cue 3), WADD chooses B
    is_t3 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 1, 0, 0))
    # Trial 4: TTB chooses A (cue 3), WADD chooses A
    is_t4 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 1, 1, 1))
    
    # Calculate the probability of choosing Option B for both trials
    p_b_t3 = (data.loc[is_t3, 'response'] == 1).mean()
    p_b_t4 = (data.loc[is_t4, 'response'] == 1).mean()
    
    # For pure TTB, both trials have the exact same top discriminating cue (index 2)
    # and TTB strictly chooses A on both. P(B) is purely noise and identical across both.
    # For the Competing Theory, if the validity (0.70) falls below threshold, 
    # Trial 3 falls back to WADD and chooses B, whereas Trial 4 falls back to WADD 
    # and still chooses A. Thus, this difference perfectly isolates the WADD fallback.
    return float(p_b_t3 - p_b_t4)
outcome: self_sim=0.0032 (var=0.0105) adversary_sim=0.0958 (var=0.1080) welch_t=-1.346 p=0.189 (N=25, alpha=0.01) -> reject

[3] rationale: Previous metrics failed because the Competing Theory's threshold parameter creates a high between-subject variance: ~70% of its simulated subjects never trigger the WADD fallback on Trial 3 and behave exactly like pure TTB. A simple mean difference is heavily diluted by this 70% overlap. By computing the probability of choosing Option B on Trial 3 per subject and applying a cubic transformation (P(B)^3), we heavily penalize low-probability noise (driving the ~0.15 error rate of TTB subjects down to ~0.003) while preserving the high-probability signal of the 30% of subjects who successfully fall back to WADD. This drastically reduces the baseline variance of the metric, allowing the subset of WADD-fallback subjects to drive a highly significant divergence between the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 3, where the discriminating cue has the lowest validity (0.70)
    # and the compensatory WADD score strongly favors Option B.
    is_t3 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 1, 0, 0))
    t3_data = data[is_t3]
    
    if len(t3_data) == 0:
        return 0.0
        
    # Calculate the proportion of times Option B was chosen per subject on this trial.
    p_b_per_subj = t3_data.groupby('subject_id')['response'].mean()
    
    # Apply a cubic non-linear transformation to the probability.
    # Pure TTB predicts a low P(B) (driven only by noise), so P(B)^3 will be extremely close to 0.
    # The Competing Theory has a ~30% chance of falling back to WADD (which strongly favors B).
    # For these WADD-fallback subjects, P(B) is high, and P(B)^3 remains high.
    # The cubic transformation suppresses the noise variance from the TTB-like subjects
    # while strongly amplifying the signal from the WADD-fallback subjects, maximizing the t-statistic.
    return float((p_b_per_subj ** 3).mean())
outcome: self_sim=0.0082 (var=0.0002) adversary_sim=0.1272 (var=0.0634) welch_t=-2.358 p=0.0268 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data_copy = data.copy()\n    data_copy['a_str'] = data_copy['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))\n    \n    # Trial 4: A=[0, 0, 0, 1, 0, 0], B=[0, 0, 0, 0, 1, 1]\n    t4_mask = data_copy['a_str'] == '000100'\n    # Trial 5: A=[0, 0, 0, 1, 1, 0], B=[0, 0, 0, 0, 0, 1]\n    t5_mask = data_copy['a_str'] == '000110'\n    \n    if t4_mask.sum() == 0 or t5_mask.sum() == 0:\n        return 0.0\n        \n    p_b_t4 = (data_copy.loc[t4_mask, 'response'] == 1).mean()\n    p_b_t5 = (data_copy.loc[t5_mask, 'response'] == 1).mean()\n    \n    # Pure TTB predicts the exact same probability of choosing B (driven purely by noise) \n    # for both trials, as the top discriminating cue is identical (validity 0.65 favoring A).\n    # The Competing Theory will fall back to WADD if the threshold is > 0.65.\n    # In Trial 4, WADD strongly favors B, while in Trial 5, WADD favors A.\n    # This difference isolates the WADD fallback effect on the lower-validity trials.\n    return float(p_b_t4 - p_b_t5)",
  "rationale": "By contrasting Trial 4 and Trial 5, we look at trials where the top discriminating cue has a lower validity (0.65) compared to the 0.75 cues used in previous failed metrics. Because 0.65 is lower, it has a substantially higher probability (35% vs 25%) of falling below the Competing Theory's uniform(0,1) threshold, triggering more WADD fallback across subjects. For pure Take The Best (TTB), both Trial 4 and Trial 5 have the exact same top discriminating cue favoring Option A, meaning any choice of Option B is due to noise, and the expected difference is 0. For the Competing Theory, WADD favors B on Trial 4 but favors A on Trial 5. The difference P(B|T4) - P(B|T5) captures this exact divergence while inherently canceling out subject-specific baseline noise."
}
```

## Usage

```json
{
  "prompt_token_count": 5912,
  "candidates_token_count": 651,
  "total_token_count": 8997
}
```
