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
**Validities (n_features=5):** [0.85, 0.8, 0.7, 0.65, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 6: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 8: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** This design quantitatively dissociates the Sequential Evidence Accumulation model from the Dual-Process Strategy Mixture (TTB + WADD). In the Mixture model, the non-compensatory component (TTB) is exclusively controlled by the single highest-validity discriminating cue. Consequently, lower-validity cues only influence the WADD component, which processes them in a linear, compensatory manner. In contrast, the Sequential Accumulation model adds evidence cue-by-cue. If the threshold is intermediate, the first cue alone might not trigger a decision, but the first AND second cues together might hit the threshold, abruptly terminating search and ignoring all subsequent cues. By presenting trials where Cue 1 favors Option A and Cues 3-5 favor Option B, we manipulate Cue 2 to favor B, be Tied, or favor A. The Mixture model predicts a linear, symmetric shift in choice probability across these three states. The Sequential model predicts a massive non-linear jump: when Cue 2 favors A, the accumulator hits the threshold early, forcing a hard choice for A and blinding the model to the overwhelming evidence for B in Cues 3-5.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Dual-Process Strategy Mixture: Decision-making is driven by a probabilistic mixture of two distinct strategies. With a certain probability (mixture_p), subjects employ a non-compensatory Take-The-Best (TTB) heuristic, making a choice based solely on the most valid discriminating cue. Otherwise, they use a compensatory Weighted Additive (WADD) strategy, integrating all available features weighted by their validities into a comprehensive utility score. This blend captures both the strict, flat sensitivity of heuristic processing and the graded, trade-off sensitivity of compensatory processing.

**Parameters:**
- mixture_p: [0.0, 1.0]
- beta: [0.1, 3.5]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters['validities'], dtype=float)
    
    mixture_p = float(parameters['mixture_p'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # --- TTB Component ---
    cue_order = np.argsort(-val, kind='stable')
    a, b = stim[0], stim[1]
    
    p_ttb = np.array([0.5, 0.5])
    for j in cue_order:
        if a[j] > b[j]:
            p_ttb = np.array([1.0, 0.0])
            break
        elif b[j] > a[j]:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # --- WADD Component ---
    # WADD uses validities as weights
    scores = stim @ val
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_wadd = e / e.sum()
    
    # --- Mixture ---
    p_core = mixture_p * p_ttb + (1.0 - mixture_p) * p_wadd
    
    # --- Lapse ---
    p_final = (1.0 - epsilon) * p_core + epsilon * 0.5
    
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
**Description:** Sequential Evidence Accumulation: Decision-making is driven by a sequential sampling process where features are evaluated in order of their subjective validity. As each feature is processed, the validity-weighted difference between the options is added to a running accumulator. If this accumulated evidence reaches a predefined threshold at any point, search is immediately terminated and a choice is made (mimicking non-compensatory heuristics like Take-The-Best). If all features are exhausted without the evidence hitting the boundary, the subject makes a probabilistic choice based on the final accumulated tally (mimicking compensatory strategies like WADD). This single-process model naturally unifies fast-and-frugal heuristics and exhaustive compensatory integration depending on the height of the evidence threshold.

**Parameters:**
- threshold: [0.0, 5.0]
- beta: [0.1, 1.5]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    val = np.asarray(parameters['validities'], dtype=float)
    
    threshold = float(parameters['threshold'])
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Sort cues by descending validity; stable sort to preserve original order on ties
    cue_order = np.argsort(-val, kind='stable')
    
    a, b = stim[0], stim[1]
    
    A_val = 0.0
    decided = False
    p_a = 0.5
    
    # Sequentially accumulate evidence
    for j in cue_order:
        A_val += val[j] * (a[j] - b[j])
        if A_val >= threshold:
            p_a = 1.0
            decided = True
            break
        elif A_val <= -threshold:
            p_a = 0.0
            decided = True
            break
            
    # If boundary not reached, make a soft choice based on the final tally
    if not decided:
        z = beta * A_val
        # Numerically stable logistic function for P(A)
        if z > 100:
            p_a = 1.0
        elif z < -100:
            p_a = 0.0
        else:
            p_a = 1.0 / (1.0 + np.exp(-z))
            
    p_b = 1.0 - p_a
    p_core = np.array([p_a, p_b])
    
    # Blend with standard uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * 0.5
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
[0] rationale: This metric calculates the non-linearity (second derivative) of the choice probability curve as a function of the second cue's evidence. By isolating trials where Cue 1 strongly favors one option and Cues 3-5 strongly favor the other, we observe how manipulating Cue 2 shifts the choice. The Dual-Process Strategy Mixture model predicts a relatively linear shift because the non-compensatory component is fixed by Cue 1, leaving Cue 2 to only influence the linear compensatory WADD component. Conversely, the Sequential Evidence Accumulation model predicts a massive non-linear jump: when Cue 2 aligns with Cue 1, the accumulated evidence abruptly hits the decision threshold, truncating search and causing a sudden surge in choice probability. The metric quantifies this predicted convexity.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a1 = data['option_a_ratings'].apply(lambda x: x[0])
    b1 = data['option_b_ratings'].apply(lambda x: x[0])
    
    # Filter to only trials where exactly one option has cue1=1 (Trials 1-3, 6-8)
    mask = a1 != b1
    if not mask.any():
        return 0.0
        
    df = data[mask].copy()
    a1 = a1[mask]
    b1 = b1[mask]
    
    a2 = df['option_a_ratings'].apply(lambda x: x[1])
    b2 = df['option_b_ratings'].apply(lambda x: x[1])
    
    # The target option is the one where cue 1 = 1
    target_is_a = (a1 == 1)
    
    # Did the subject choose the target option?
    chose_target = np.where(target_is_a, df['response'] == 0, df['response'] == 1)
    
    # Cue 2 difference: target option's cue 2 minus the other option's cue 2
    target_cue2 = np.where(target_is_a, a2, b2)
    other_cue2 = np.where(target_is_a, b2, a2)
    cue2_diff = target_cue2 - other_cue2
    
    df_calc = pd.DataFrame({'chose_target': chose_target, 'cue2_diff': cue2_diff})
    
    # Calculate choice probability for the target option at each level of cue 2 support
    p1 = df_calc[df_calc['cue2_diff'] == -1]['chose_target'].mean()
    p2 = df_calc[df_calc['cue2_diff'] == 0]['chose_target'].mean()
    p3 = df_calc[df_calc['cue2_diff'] == 1]['chose_target'].mean()
    
    if pd.isna(p1) or pd.isna(p2) or pd.isna(p3):
        return 0.0
        
    # Return the second derivative (non-linearity) of the choice curve
    return float(p3 - 2 * p2 + p1)
outcome: self_sim=0.0375 (var=0.0719) adversary_sim=0.0558 (var=0.0853) welch_t=-0.231 p=0.8182 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))\n    \n    # Trials where Cue 1 and Cue 2 favor the TTB option, Cues 3-5 favor the compensatory option\n    t3 = (a_str == \"11000\") & (b_str == \"00111\")\n    t8 = (a_str == \"00111\") & (b_str == \"11000\")\n    \n    # Trials where Cue 1 favors the TTB option, Cue 2 is tied, Cues 3-5 favor the compensatory option\n    t2 = (a_str == \"11000\") & (b_str == \"01111\")\n    t7 = (a_str == \"01111\") & (b_str == \"11000\")\n    \n    data_t38 = data[t3 | t8]\n    if len(data_t38) == 0:\n        return 0.0\n    ttb_chosen_t38 = np.where(t3[t3 | t8], data_t38['response'] == 0, data_t38['response'] == 1).mean()\n    \n    data_t27 = data[t2 | t7]\n    if len(data_t27) == 0:\n        return 0.0\n    ttb_chosen_t27 = np.where(t2[t2 | t7], data_t27['response'] == 0, data_t27['response'] == 1).mean()\n    \n    return float(ttb_chosen_t38 - ttb_chosen_t27)",
  "rationale": "This metric calculates the difference in the probability of choosing the option favored by the strongest cue (the TTB option) between two specific conditions. In the first condition (Trials 3 & 8), Cue 1 and Cue 2 both favor the TTB option, while Cues 3-5 favor the other. In the second condition (Trials 2 & 7), Cue 1 favors the TTB option, Cue 2 is tied, and Cues 3-5 favor the other. The Dual-Process Mixture model predicts a relatively small difference here: its TTB component ignores Cue 2 entirely because Cue 1 already discriminates, leaving only the linear WADD component to respond to the Cue 2 difference. The Sequential Evidence Accumulation model, however, predicts a massive jump: the combined evidence of Cues 1 and 2 in the first condition easily crosses the decision threshold, truncating search and guaranteeing a TTB-option choice, whereas in the second condition, the evidence never exceeds the single-cue value and often fails to hit the threshold, causing the model to integrate the subsequent cues and choose the compensatory option."
}
```

## Usage

```json
{
  "prompt_token_count": 4323,
  "candidates_token_count": 736,
  "total_token_count": 9593
}
```
