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
**Validities (n_features=6):** [0.99, 0.95, 0.9, 0.65, 0.6, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 2: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 3: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 5: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 6: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 8: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]

**Rationale:** This design quantitatively dissociates Strategy Mixture Theory (SMT) from Probabilistic Cue Search (PCS) by exploiting how each model scales confidence in non-compensatory decisions. In SMT, the Take-The-Best (TTB) component assigns a discrete score of 1.0 to the option favored by the first discriminating cue, regardless of that cue's actual validity. In contrast, PCS evaluates evidence sequentially, and the evidence at the first stopping point is proportional to the normalized validity of the first discriminating cue. By introducing trials where the first discriminating cue is of very low validity (because all high-validity cues are tied), SMT's TTB component remains highly confident, while PCS predicts a very weak initial preference. Pitting this low-validity first discriminating cue against multiple opposing lower-validity cues creates a stark divergence in predicted choice probabilities.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Strategy Mixture Theory: Decision-makers do not uniformly apply a single choice rule. Instead, they possess a repertoire of strategies and flexibly draw from them. On any given trial, a subject acts as a mixture model, choosing to apply a non-compensatory heuristic (Take-The-Best) with probability alpha, and a compensatory rule (Weighted Additive / Tallying) with probability 1 - alpha. The compensatory rule weights features by its subjective validities, naturally subsuming Tallying and WADD. Crucially, the compensatory scores are normalized to the [0, 1] scale to perfectly match the scale of the heuristic's discrete scores, allowing a single temperature parameter to symmetrically control the determinism of both strategies without numerical compromise.

**Parameters:**
- alpha: [0.0, 1.0]
- beta: [0.1, 20.0]
- gamma: [0.0, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Mixture model expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # --- Take-The-Best (TTB) Prediction ---
    order = np.argsort(validities)[::-1]
    a, b = stim[0], stim[1]
    ttb_scores = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_scores[1] = 1.0
            break
            
    z_ttb = beta * (ttb_scores - ttb_scores.max())
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / e_ttb.sum()
    
    # --- Compensatory (WADD/Tallying) Prediction ---
    # Subjective validities: gamma=0 yields Tallying, gamma=1 yields strict WADD
    subjective_weights = validities ** gamma
    wadd_scores = stim @ subjective_weights
    
    # Normalize WADD scores to [0, 1] scale to match TTB scores
    wadd_scores = wadd_scores / np.sum(subjective_weights)
    
    z_wadd = beta * (wadd_scores - wadd_scores.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # --- Strategy Mixture ---
    p_core = alpha * p_ttb + (1.0 - alpha) * p_wadd
    
    # --- Uniform Lapse ---
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Probabilistic Cue Search (Sequential Evidence Accumulation): Decision-makers evaluate features sequentially in descending order of validity. However, instead of strictly stopping at the first discriminating cue (as in pure Take-The-Best) or exhaustively accumulating all cues (as in Weighted Additive), they exhibit a probabilistic stopping rule. After evaluating each discriminating cue and updating their internal evidence, they stop searching and make a choice with probability `theta`. If they do not stop, they continue to the next discriminating cue, accumulating its evidence. This naturally produces a graded interpolation between non-compensatory and compensatory decision-making without relying on a discrete mixture of distinct strategies.

**Parameters:**
- theta: [0.0, 1.0]
- beta: [0.1, 20.0]
- gamma: [0.0, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    theta = float(parameters["theta"])
    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    a, b = stim[0], stim[1]
    
    # Subjective weights normalized so the sum of weights is 1.0
    w = validities ** gamma
    if np.sum(w) > 0:
        w = w / np.sum(w)
    
    # Sort all features by validity descending
    order = np.argsort(validities)[::-1]
    
    # Find discriminating cues in order of validity
    discrim_indices = [i for i in order if a[i] != b[i]]
    
    if len(discrim_indices) == 0:
        p_core = np.array([0.5, 0.5])
    else:
        p_A_total = 0.0
        p_B_total = 0.0
        p_continue = 1.0
        
        score_A = 0.0
        score_B = 0.0
        
        for j, i in enumerate(discrim_indices):
            # Accumulate evidence from the current discriminating cue
            score_A += w[i] * a[i]
            score_B += w[i] * b[i]
            
            # Determine stopping probability
            if j < len(discrim_indices) - 1:
                p_stop = theta
            else:
                p_stop = 1.0  # Must stop at the last discriminating cue
                
            # Softmax over accumulated scores so far
            z_A = beta * score_A
            z_B = beta * score_B
            max_z = max(z_A, z_B)
            e_A = np.exp(z_A - max_z)
            e_B = np.exp(z_B - max_z)
            p_A_given_stop = e_A / (e_A + e_B)
            p_B_given_stop = e_B / (e_A + e_B)
            
            # Marginalize over the stopping probability
            p_A_total += p_continue * p_stop * p_A_given_stop
            p_B_total += p_continue * p_stop * p_B_given_stop
            
            # Update the probability of continuing to the next cue
            p_continue *= (1.0 - p_stop)
            
        p_core = np.array([p_A_total, p_B_total])
        
    # Apply uniform lapse
    n_opts = 2
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
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
[0] rationale: This metric calculates the proportion of times the subject chooses Option A in trials where Option A is favored by the first discriminating cue (which has low validity), while Option B is favored by multiple subsequent cues. Strategy Mixture Theory's TTB component assigns a full score of 1.0 to Option A regardless of the cue's validity, leading to a higher rate of choosing A. Probabilistic Cue Search, however, evaluates the evidence proportionally to the validity, resulting in very weak initial evidence for A and ultimately favoring B due to the accumulation of multiple lower-validity cues.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    mask = (
        ((data['A_str'] == '000100') & (data['B_str'] == '000011')) |
        ((data['A_str'] == '001000') & (data['B_str'] == '000111'))
    )
    
    if mask.sum() == 0:
        return 0.0
        
    return float((data.loc[mask, 'response'] == 0).mean())
outcome: self_sim=0.6433 (var=0.0308) adversary_sim=0.5942 (var=0.0270) welch_t=+1.023 p=0.3116 (N=25, alpha=0.01) -> reject

[1] rationale: This metric calculates the difference in the probability of choosing Option A between Trial 6 and Trial 7. In Trial 6, the first discriminating cue has a very high validity (0.99), while in Trial 7, the first discriminating cue has a much lower validity (0.65). Under Strategy Mixture Theory, the Take-The-Best component assigns a discrete score of 1.0 to Option A in both trials regardless of the absolute validity of the cue, leading to a relatively small difference in choice probabilities between the two trials. Under Probabilistic Cue Search, the accumulated evidence is proportional to the cue's validity, resulting in a much stronger preference for Option A in Trial 6 compared to Trial 7. Thus, the difference P(A | Trial 6) - P(A | Trial 7) should be significantly larger for Probabilistic Cue Search than for Strategy Mixture Theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, int(v) for v in x)) if isinstance(x[0], (int, float, str)) else ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, int(v) for v in x)) if isinstance(x[0], (int, float, str)) else ''.join(map(str, x)))
    
    # Trial 6: A=[1,0,0,0,0,0] vs B=[0,1,0,0,0,0]
    mask6 = (data['A_str'] == '100000') & (data['B_str'] == '010000')
    # Trial 7: A=[0,0,0,1,0,0] vs B=[0,0,0,0,1,0]
    mask7 = (data['A_str'] == '000100') & (data['B_str'] == '000010')
    
    p_A_6 = (data.loc[mask6, 'response'] == 0).mean() if mask6.sum() > 0 else 0.5
    p_A_7 = (data.loc[mask7, 'response'] == 0).mean() if mask7.sum() > 0 else 0.5
    
    return float(p_A_6 - p_A_7)
outcome: evaluation failed (SyntaxError: Generator expression must be parenthesized (<string>, line 2))

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    \n    mask6 = (data['A_str'] == '100000') & (data['B_str'] == '010000')\n    mask7 = (data['A_str'] == '000100') & (data['B_str'] == '000010')\n    \n    p_A_6 = (data.loc[mask6, 'response'] == 0).mean()\n    p_A_7 = (data.loc[mask7, 'response'] == 0).mean()\n    \n    if pd.isna(p_A_6): p_A_6 = 0.5\n    if pd.isna(p_A_7): p_A_7 = 0.5\n    \n    return float(p_A_6 - p_A_7)",
  "rationale": "This metric calculates the difference in the probability of choosing Option A between Trial 6 and Trial 7. In Trial 6, the first discriminating cue has a very high validity (0.99), while in Trial 7, the first discriminating cue has a much lower validity (0.65). Under Strategy Mixture Theory, the Take-The-Best component assigns a discrete score of 1.0 to Option A in both trials regardless of the absolute validity of the cue, leading to a relatively small difference in choice probabilities between the two trials. Under Probabilistic Cue Search, the accumulated evidence is proportional to the cue's validity, resulting in a much stronger preference for Option A in Trial 6 compared to Trial 7. Thus, the difference P(A | Trial 6) - P(A | Trial 7) should be significantly larger for Probabilistic Cue Search than for Strategy Mixture Theory."
}
```

## Usage

```json
{
  "prompt_token_count": 5123,
  "candidates_token_count": 492,
  "total_token_count": 8441
}
```
