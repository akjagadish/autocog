# metric_exp00_attempt_01

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

**Trial pairs (n=7):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  trial 4: A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 6: A=[1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0]
  trial 7: A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]

**Rationale:** This design quantitatively dissociates the Lateral Inhibition Evidence Accumulator (Advocated) from the Strategy Mixture (Competing) model by exploiting two core structural differences: the 'perverse' self-dilution effect and invariance to symmetric evidence addition. In the Competing model, choice probability is a mixture of TTB (based on the highest validity discriminating cue) and Mean-Normalized Tallying. Adding unique lower-validity cues to Option A strictly increases its tallying score without changing the TTB winner, forcing the Competing model to predict a monotonic increase in preference for A. In contrast, the Advocated model employs divisive normalization; adding lower-validity cues to A dilutes its primary high-validity cue, which can paradoxically decrease A's choice probability. Furthermore, adding shared cues symmetrically to both options leaves the TTB winner and the tallying difference strictly unchanged in the Competing model, mandating identical choice probabilities. The Advocated model, however, processes these shared cues holistically, increasing both self and conflict suppression, thus predicting systematic shifts in preference.

**Computed schedule:** 7 unique pairs × 13 reps = 91 trials per subject.



## ADVOCATED THEORY
**Description:** Lateral Inhibition Evidence Accumulator (Extreme Dilution): Subjects evaluate options by comparing their feature vectors holistically, where the evidence provided by each cue is dynamically suppressed by other active cues via divisive normalization (lateral inhibition). Supporting secondary cues dilute the perceived value of the primary cue (self-inhibition), while conflicting cues from the alternative option also suppress evidence. This non-linear squashing before integration naturally produces strong non-monotonic and 'perverse' effects, where having too many secondary cues can paradoxically weaken an option's overall appeal compared to an option with a single strong primary cue.

**Parameters:**
- alpha: [0.0, 5000.0]
- beta: [0.0, 100.0]
- gamma: [0.1, 5.0]
- theta: [0.1, 500.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    # Transform validities into base weights
    w = np.power(np.maximum(val - 0.5, 0.0), gamma)
    
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    
    e_a = 0.0
    e_b = 0.0
    
    sum_w_a = np.sum(w * a)
    sum_w_b = np.sum(w * b)
    
    # Calculate laterally inhibited evidence for each option
    for i in range(len(a)):
        if a[i] > 0:
            other_a = sum_w_a - w[i] * a[i]
            conflicting_b = sum_w_b
            # Divisive normalization: self-dilution (alpha) + conflict suppression (beta)
            denom = 1.0 + alpha * other_a + beta * conflicting_b
            e_a += (w[i] * a[i]) / denom
            
        if b[i] > 0:
            other_b = sum_w_b - w[i] * b[i]
            conflicting_a = sum_w_a
            denom = 1.0 + alpha * other_b + beta * conflicting_a
            e_b += (w[i] * b[i]) / denom
            
    theta = float(parameters["theta"])
    z = theta * np.array([e_a, e_b])
    # Numerically stable softmax
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.sum(p)
    
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
**Description:** Strategy Mixture with Validity-Scaled TTB, Mean-Normalized Tallying, and Logistic Dispersion Modulation

**Parameters:**
- w_base: [-20.0, 20.0]
- gamma: [-50.0, 50.0]
- beta_ttb: [0.1, 50.0]
- beta_tally: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take-The-Best (TTB) Strategy - scaled by the validity of the discriminating cue
    ev_ttb = np.array([0.0, 0.0])
    for i in cue_order:
        if a[i] > b[i]:
            ev_ttb = np.array([val[i], 0.0])
            break
        elif b[i] > a[i]:
            ev_ttb = np.array([0.0, val[i]])
            break
            
    # Tallying Strategy (Unit-Weight Additive) - normalized by total cues to match scale
    ev_tally = np.array([np.mean(a), np.mean(b)])
    
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    
    # TTB probabilities
    z_ttb = beta_ttb * ev_ttb
    z_ttb = z_ttb - np.max(z_ttb)
    p_ttb = np.exp(z_ttb)
    p_ttb = p_ttb / np.sum(p_ttb)
    
    # Tallying probabilities
    z_tally = beta_tally * ev_tally
    z_tally = z_tally - np.max(z_tally)
    p_tally = np.exp(z_tally)
    p_tally = p_tally / np.sum(p_tally)
    
    # Strategy Mixture Weight (Logistic modulation based on dispersion)
    w_base = float(parameters["w_base"])
    gamma = float(parameters["gamma"])
    dispersion = np.std(val) if len(val) > 1 else 0.0
    
    logit_w = w_base + gamma * dispersion
    w = 1.0 / (1.0 + np.exp(-logit_w))
    
    p_mix = w * p_ttb + (1.0 - w) * p_tally
    
    # Lapse rate
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final
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
[0] rationale: This metric exploits a strict invariance in the Competing theory. In trials 5 and 7, Option A and Option B have the same number of active cues, so the Tallying strategy predicts no difference between them. The Take-The-Best (TTB) strategy favors Option A in both trials based on the exact same primary cue (cue 1 for A, cue 2 for B). Therefore, the Competing theory predicts the exact same choice probability for Option A across both trials, leading to an expected difference of 0 in the mean response. In contrast, the Advocated theory's divisive normalization means that adding shared cues (from trial 5 to trial 7) increases lateral inhibition, diluting the primary cue's impact and driving the choice probability closer to 0.5. This predicts a systematic shift in preference, yielding a non-zero difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t5_mask = (data['option_a_ratings'].apply(tuple) == (1, 0, 1, 0, 0)) & (data['option_b_ratings'].apply(tuple) == (0, 1, 1, 0, 0))
    t7_mask = (data['option_a_ratings'].apply(tuple) == (1, 0, 1, 1, 1)) & (data['option_b_ratings'].apply(tuple) == (0, 1, 1, 1, 1))
    
    t5_mean = data.loc[t5_mask, 'response'].mean()
    t7_mean = data.loc[t7_mask, 'response'].mean()
    
    if pd.isna(t5_mean) or pd.isna(t7_mean):
        return 0.0
        
    return float(t7_mean - t5_mean)
outcome: self_sim=0.0215 (var=0.0320) adversary_sim=0.0231 (var=0.0268) welch_t=-0.032 p=0.9748 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_tuples = data['option_a_ratings'].apply(tuple)\n    b_tuples = data['option_b_ratings'].apply(tuple)\n    \n    t1_mask = (a_tuples == (1, 0, 0, 0, 0)) & (b_tuples == (0, 1, 0, 0, 0))\n    t4_mask = (a_tuples == (1, 0, 1, 1, 1)) & (b_tuples == (0, 1, 0, 0, 0))\n    \n    t1_mean = data.loc[t1_mask, 'response'].mean()\n    t4_mean = data.loc[t4_mask, 'response'].mean()\n    \n    if pd.isna(t1_mean) or pd.isna(t4_mean):\n        return 0.0\n        \n    return float(t4_mean - t1_mean)",
  "rationale": "This metric directly tests the 'perverse self-dilution' effect versus monotonic evidence accumulation. Between trial 1 and trial 4, Option A gains additional unique lower-validity cues while Option B remains unchanged. The Competing Theory (Strategy Mixture) predicts a strict increase in the probability of choosing Option A, because the TTB winner is unchanged and the Tallying score for Option A strictly increases. Therefore, the mean response (where 1 = choosing B) should decrease from trial 1 to trial 4. In contrast, the Advocated Theory predicts that adding lower-validity cues to Option A dilutes the impact of its primary high-validity cue via divisive normalization, potentially decreasing the probability of choosing A and thus increasing the mean response. The difference in mean response between trial 4 and trial 1 directly captures this opposing directional prediction."
}
```

## Usage

```json
{
  "prompt_token_count": 4248,
  "candidates_token_count": 434,
  "total_token_count": 6177
}
```
