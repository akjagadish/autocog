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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** This design quantitatively dissociates the advocated Sequential Evidence Accumulation theory from the competing Probabilistic Heuristic Integration theory by probing for early stopping versus continuous mixture integration. The advocated theory evaluates cues sequentially and stops if the accumulated evidence exceeds a threshold. Thus, if a highly valid cue strongly favors Option A, the decision-maker may stop search immediately, completely ignoring the number and validities of any remaining cues favoring Option B. This predicts identical choice probabilities across trials where the top cue is constant but the opposing cues vary. The competing theory, however, dynamically mixes a Weighted Additive (WADD) strategy with Tallying. It continuously integrates all cues and counts the sheer number of opposing cues, meaning choice probabilities will systematically shift as the number of lower-validity cues favoring Option B increases. By holding the top validity cue constant for Option A and systematically increasing the number of lower-validity cues favoring Option B, we can cleanly separate the flat, early-stopping predictions of the advocated theory from the graded, tally-sensitive predictions of the competing theory.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Sequential Evidence Accumulation: Decision-makers evaluate cues sequentially in descending order of validity. Each cue provides evidence proportional to a non-linear transformation of its validity above chance. Evidence is accumulated as a running difference between the two options. If the absolute accumulated evidence exceeds a threshold, search is terminated and a choice is made based on the current evidence. If all cues are evaluated without crossing the threshold, a decision is made based on the final accumulated evidence. This allows for fast, non-compensatory decisions when top cues are highly valid, while gracefully falling back to compensatory integration when early cues are less decisive.

**Parameters:**
- theta: [0.0, 10.0]
- gamma: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale weights by transforming validity above chance, allowing better separation
    weights = np.maximum(val - 0.5, 0.001) ** gamma
    
    # Search in order of descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    E = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        if diff != 0:
            E += diff * weights[j]
            # Stop if absolute accumulated evidence reaches the threshold
            if abs(E) >= theta:
                break
            
    scores = np.array([E, -E])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Probabilistic Heuristic Integration with Independent Sensitivities

**Parameters:**
- wadd_gamma: [0.0, 10.0]
- beta_tally: [0.1, 50.0]
- beta_wadd: [0.1, 50.0]
- conflict_weight: [-10.0, 10.0]
- dispersion_weight: [-10.0, 10.0]
- base_mix: [-10.0, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    wadd_gamma = float(parameters["wadd_gamma"])
    beta_tally = float(parameters["beta_tally"])
    beta_wadd = float(parameters["beta_wadd"])
    conflict_weight = float(parameters["conflict_weight"])
    dispersion_weight = float(parameters["dispersion_weight"])
    base_mix = float(parameters["base_mix"])
    epsilon = float(parameters["epsilon"])
    
    # Tallying Heuristic
    tally_a = float(np.sum(a > b))
    tally_b = float(np.sum(b > a))
    scores_tally = np.array([tally_a, tally_b])
        
    # Weighted Additive Strategy (WADD)
    weights = np.maximum(val - 0.5, 0.001) ** wadd_gamma
    wadd_a = float(np.sum((a > b) * weights))
    wadd_b = float(np.sum((b > a) * weights))
    scores_wadd = np.array([wadd_a, wadd_b])
    
    # Softmax conversion to probabilities
    z_tally = beta_tally * scores_tally
    z_tally = z_tally - np.max(z_tally)
    p_tally = np.exp(z_tally) / np.sum(np.exp(z_tally))
    
    z_wadd = beta_wadd * scores_wadd
    z_wadd = z_wadd - np.max(z_wadd)
    p_wadd = np.exp(z_wadd) / np.sum(np.exp(z_wadd))
    
    # Dynamic strategy mixing
    max_diff = len(val)
    conflict = 1.0 - (abs(tally_a - tally_b) / max_diff) if max_diff > 0 else 0.0
    dispersion = float(np.std(val))
    
    logit_wadd = base_mix + conflict_weight * conflict + dispersion_weight * dispersion
    logit_wadd = np.clip(logit_wadd, -20.0, 20.0)
    prob_wadd = 1.0 / (1.0 + np.exp(-logit_wadd))
    
    p_mix = prob_wadd * p_wadd + (1.0 - prob_wadd) * p_tally
    
    # Lapse rate
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
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
[0] rationale: This metric calculates the difference in the probability of choosing Option A between Trial 4 (where B has only 1 lower-validity cue) and Trial 1 (where B has 4 lower-validity cues), while Option A consistently holds only the single highest-validity cue. The advocated Sequential Evidence Accumulation theory predicts that decision-makers will often stop searching early upon encountering the highly valid cue for A, largely ignoring the number of subsequent opposing cues for B. This leads to a relatively flat probability of choosing A across these trials, resulting in a difference near zero. Conversely, the competing Probabilistic Heuristic Integration theory continuously mixes a Weighted Additive approach with Tallying, meaning it counts the sheer number of opposing cues. As the number of cues favoring B increases from 1 to 4, the Tallying component strongly shifts the preference toward B, producing a significantly larger positive difference in the metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Identify trials where option A has only the most valid cue: A = [1, 0, 0, 0, 0]
    is_A_10000 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    
    # Sum the cues for option B to distinguish between Trial 1 (sum=4) and Trial 4 (sum=1)
    b_sums = data['option_b_ratings'].apply(sum)
    
    mask_t4 = is_A_10000 & (b_sums == 1)
    mask_t1 = is_A_10000 & (b_sums == 4)
    
    # Calculate the probability of choosing A for both trial types
    p_A_t4 = (data[mask_t4]['response'] == 0).mean()
    p_A_t1 = (data[mask_t1]['response'] == 0).mean()
    
    if pd.isna(p_A_t4):
        p_A_t4 = 0.5
    if pd.isna(p_A_t1):
        p_A_t1 = 0.5
        
    return float(p_A_t4 - p_A_t1)

outcome: self_sim=0.1250 (var=0.1077) adversary_sim=0.2675 (var=0.0672) welch_t=-1.704 p=0.09521 (N=25, alpha=0.01) -> reject

[1] rationale: By pooling Trials 1 & 2 (where Option B has 3-4 lower-validity cues) and Trials 3 & 4 (where Option B has 1-2 cues), we effectively halve the within-subject estimation variance compared to single-trial comparisons. The Sequential Evidence Accumulation theory predicts early stopping upon encountering Option A's highly valid top cue, making the choice probability relatively insensitive to the number of opposing cues for B, yielding a difference near zero. The competing Probabilistic Heuristic Integration theory counts the sheer number of opposing cues via its Tallying component, leading to a systematically higher probability of choosing B in the high-cue group, generating a significantly larger positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Identify trials where option A has only the most valid cue: A = [1, 0, 0, 0, 0]
    is_A_10000 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    
    # Sum the cues for option B to distinguish between trials
    b_sums = data['option_b_ratings'].apply(sum)
    
    # Group into high opposing cues (Trials 1 & 2: 3-4 cues) and low opposing cues (Trials 3 & 4: 1-2 cues)
    mask_high = is_A_10000 & (b_sums >= 3)
    mask_low = is_A_10000 & (b_sums <= 2)
    
    # Calculate the probability of choosing B for both groups
    p_B_high = data[mask_high]['response'].mean()
    p_B_low = data[mask_low]['response'].mean()
    
    if pd.isna(p_B_high):
        p_B_high = 0.5
    if pd.isna(p_B_low):
        p_B_low = 0.5
        
    return float(p_B_high - p_B_low)
outcome: self_sim=0.0825 (var=0.0340) adversary_sim=0.1206 (var=0.0273) welch_t=-0.770 p=0.4452 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    # Project sequences to strings for safe hashing and comparison\n    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))\n    \n    # Trial 1: A has the best cue (0), B has all four remaining cues\n    mask_t1 = (data['A_str'] == '10000') & (data['B_str'] == '01111')\n    # Trial 5: A has the second-best cue (1), B has the three remaining cues\n    mask_t5 = (data['A_str'] == '01000') & (data['B_str'] == '00111')\n    \n    # Calculate probability of choosing Option A\n    p_A_t1 = (data[mask_t1]['response'] == 0).mean()\n    p_A_t5 = (data[mask_t5]['response'] == 0).mean()\n    \n    if pd.isna(p_A_t1):\n        p_A_t1 = 0.5\n    if pd.isna(p_A_t5):\n        p_A_t5 = 0.5\n        \n    return float(p_A_t1 - p_A_t5)",
  "rationale": "This metric contrasts the probability of choosing Option A in Trial 1 versus Trial 5. In Trial 1, Option A holds the highest-validity cue (0.95), while B holds four opposing cues. In Trial 5, A holds the second highest-validity cue (0.85), while B holds three opposing cues. The advocated Sequential Evidence Accumulation theory evaluates cues in order of validity; because cue 0 is significantly more valid than cue 1, it predicts a substantially higher probability of choosing A in Trial 1 than in Trial 5. In contrast, the competing Probabilistic Heuristic Integration theory relies heavily on Tallying, which penalizes Option A more severely in Trial 1 (a 1-to-4 tally disadvantage) than in Trial 5 (a 1-to-3 tally disadvantage). This opposing sensitivity means the competing theory will produce a much smaller, or even negative, difference compared to the strongly positive difference predicted by the advocated theory."
}
```

## Usage

```json
{
  "prompt_token_count": 4800,
  "candidates_token_count": 590,
  "total_token_count": 9937
}
```
