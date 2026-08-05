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
**Validities (n_features=5):** [0.95, 0.8, 0.75, 0.7, 0.65]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 6: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 8: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Rationale:** To quantitatively dissociate the Strategy Mixture (TTB + WADD) from the Threshold-based Sequential Search (TSS) model, we exploit TSS's early termination mechanism which is dependent on the sequence of cue validities. We use a 5-feature design with strictly descending validities. In the critical 'compensatory' trials, the most valid cue always favors Option A (so TTB chooses A), while the sum of the remaining cues favors Option B (so WADD chooses B). For the Strategy Mixture model, these trials all represent a conflict between TTB and WADD, leading to a mixed choice probability that shifts only slightly depending on the exact WADD difference. However, for TSS, the order of the supporting cues matters immensely: if the first two cues both favor A, TSS accumulates enough evidence to cross its threshold and terminates early (choosing A). If the first cue favors A but the second favors B, the accumulated evidence drops, forcing TSS to evaluate all cues and ultimately choose B. This creates a stark qualitative reversal in choice predictions for TSS that the Strategy Mixture model cannot reproduce.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Strategy Mixture (Take-The-Best and Weighted Additive)

**Parameters:**
- beta_ttb: [0.1, 5.0]
- beta_wadd: [0.1, 10.0]
- mixture_ttb: [0.0, 1.0]
- epsilon: [0.0, 0.5]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Mixture expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    a, b = stim[0], stim[1]
    
    # --- TTB (Take-The-Best) Process ---
    order = np.argsort(validities)[::-1]
    scores_ttb = np.zeros(2)
    for idx in order:
        if a[idx] > b[idx]:
            scores_ttb[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores_ttb[1] = 1.0
            break
            
    beta_ttb = float(parameters["beta_ttb"])
    z_ttb = beta_ttb * (scores_ttb - scores_ttb.max())
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / e_ttb.sum()
    
    # --- WADD (Weighted Additive) Process ---
    w = np.asarray(parameters["weights"], dtype=float)
    scores_wadd = stim @ (validities * w)
    
    beta_wadd = float(parameters["beta_wadd"])
    z_wadd = beta_wadd * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # --- Mixture ---
    mix = float(parameters["mixture_ttb"])
    p_mix = mix * p_ttb + (1.0 - mix) * p_wadd
    
    # --- Lapse ---
    epsilon = float(parameters["epsilon"])
    n_opts = p_mix.shape[0]
    return (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
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
**Description:** Threshold-based Sequential Search integrates the fast-and-frugal nature of Take-The-Best with the compensatory evidence accumulation of Weighted Additive (WADD) models. Decision-makers search through cues in descending order of their validity, maintaining a running tally of the evidence (weighted by each cue's validity). Instead of stopping at the very first discriminating cue, search terminates only when the absolute accumulated evidence exceeds an internal confidence threshold. If all cues are exhausted without reaching this threshold, the option with the higher accumulated evidence is chosen. This allows for fast, one-reason decisions when a highly valid cue strongly favors one option, while enabling compensatory behavior when early cues provide weak or conflicting evidence.

**Parameters:**
- threshold: [0.0, 1.25]
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.25]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues in descending order of validity
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    evidence = 0.0
    threshold = float(parameters["threshold"])
    
    # Sequential search with evidence accumulation
    for idx in order:
        diff = a[idx] - b[idx]
        evidence += diff * validities[idx]
        
        # Stop search if the confidence threshold is met or exceeded
        if abs(evidence) >= threshold:
            break
            
    # Convert accumulated evidence into discrete choice scores
    scores = np.zeros(2)
    if evidence > 0:
        scores[0] = 1.0
    elif evidence < 0:
        scores[1] = 1.0
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend in uniform lapse
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
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
[0] rationale: This metric calculates the difference in the proportion of choices consistent with the Take-The-Best (TTB) heuristic between two sets of conflict trials. In Trials 1 & 4, the first two features point in the same direction, which will cause the Threshold-based Sequential Search (TSS) model to cross its threshold early and choose the TTB option. In Trials 2, 3 & 5, the first two features conflict, preventing early threshold crossing; TSS will then evaluate all features and select the WADD-favored option (which opposes TTB). Thus, TSS predicts a large positive difference in TTB-adherence between these conditions. The Strategy Mixture model, however, evaluates TTB and WADD independently of cue order and predicts similar TTB-adherence across both conditions since the WADD evidence differences are relatively stable. This leads to a stark quantitative separation between the models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    # Identify option strings to exactly match the trials of interest
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))

    # TTB choice is determined by the first feature (validity 0.95)
    a0 = data['option_a_ratings'].apply(lambda x: x[0])
    b0 = data['option_b_ratings'].apply(lambda x: x[0])
    ttb_is_A = a0 > b0
    chose_ttb = ((ttb_is_A) & (data['response'] == 0)) | ((~ttb_is_A) & (data['response'] == 1))

    # Condition 1: Early threshold crossing for TSS (Trials 1 and 4)
    # The first two features agree and strongly favor the TTB option, causing TSS to terminate early.
    t1_t4 = ((a_str == '11000') & (b_str == '00111')) | ((a_str == '00111') & (b_str == '11000'))

    # Condition 2: No early threshold crossing, fallback to WADD for TSS (Trials 2, 3, and 5)
    # The first two features conflict, preventing early termination. TSS evaluates all features and chooses the WADD option.
    t2_t3_t5 = ((a_str == '10100') & (b_str == '01011')) | \
               ((a_str == '10010') & (b_str == '01101')) | \
               ((a_str == '01011') & (b_str == '10100'))
               
    rate_cond1 = chose_ttb[t1_t4].mean()
    rate_cond2 = chose_ttb[t2_t3_t5].mean()
    
    if pd.isna(rate_cond1) or pd.isna(rate_cond2):
        return 0.0
        
    return float(rate_cond1 - rate_cond2)

outcome: self_sim=0.0336 (var=0.0324) adversary_sim=0.1864 (var=0.1011) welch_t=-2.091 p=0.04332 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n\n    # Identify the option that the Take-The-Best (TTB) heuristic would favor\n    # (TTB relies entirely on the first cue since it has the highest validity)\n    a0 = data['option_a_ratings'].apply(lambda x: x[0])\n    b0 = data['option_b_ratings'].apply(lambda x: x[0])\n    ttb_is_A = a0 > b0\n    chose_ttb = ((ttb_is_A) & (data['response'] == 0)) | ((~ttb_is_A) & (data['response'] == 1))\n\n    # Convert ratings to strings for exact trial matching\n    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))\n\n    # Condition 1: TTB and WADD agree (Trial 6)\n    # Both heuristics strongly favor the same option.\n    t6 = (a_str == '11100') & (b_str == '00011')\n\n    # Condition 2: TTB and WADD conflict, but TSS stops early (Trials 1 and 4)\n    # The first two cues agree, so TSS accumulates 0.95 + 0.80 = 1.75 evidence.\n    # Since the max threshold is 1.25, TSS *always* stops early and chooses the TTB option.\n    # Strategy Mixture, however, integrates all cues for WADD, so WADD prefers the opposite option.\n    t1_t4 = ((a_str == '11000') & (b_str == '00111')) | ((a_str == '00111') & (b_str == '11000'))\n\n    rate_agree = chose_ttb[t6].mean()\n    rate_conflict_early = chose_ttb[t1_t4].mean()\n\n    if pd.isna(rate_agree) or pd.isna(rate_conflict_early):\n        return 0.0\n\n    # Return the difference in TTB adherence\n    return float(rate_agree - rate_conflict_early)",
  "rationale": "This metric compares the rate at which subjects choose the Take-The-Best (TTB) favored option when TTB and WADD agree (Trial 6) versus when they conflict but the first two cues point in the same direction (Trials 1 and 4). For the Threshold-based Sequential Search (TSS) model, the accumulated evidence after the first two cues in Trials 1 and 4 is 1.75, which exceeds the maximum possible threshold (1.25). Thus, TSS terminates early and chooses the TTB option on Trials 1/4 just as reliably as it does on Trial 6, yielding a difference near 0 (with very low between-subject variance). In contrast, the Strategy Mixture model evaluates TTB and WADD independently. On Trial 6, both models agree, so the TTB option is chosen ~100% of the time. On Trials 1 and 4, TTB and WADD conflict, so the TTB option is only chosen according to the mixture probability. This results in a large, positive difference (~0.5 on average) for the Strategy Mixture model, robustly discriminating the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 4400,
  "candidates_token_count": 836,
  "total_token_count": 10483
}
```
