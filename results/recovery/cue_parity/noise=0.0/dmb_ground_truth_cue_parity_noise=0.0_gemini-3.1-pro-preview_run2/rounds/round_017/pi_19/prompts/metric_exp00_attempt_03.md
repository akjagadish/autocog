# metric_exp00_attempt_03

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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=10):**
  trial 1: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 5: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 6: A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 7: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 9: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 1]
  trial 10: A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1]

**Rationale:** This design quantitatively dissociates the advocated Rank-Weighted Sequential Sampling theory from the competing LCA with Configural Weighting theory by exploiting their divergent handling of tied cues at the top of the validity hierarchy. The advocated theory evaluates cues sequentially. If the highest-validity cue is tied, the accumulated evidence remains zero, the stopping probability is zero, and the evaluation seamlessly proceeds to the next cues exactly as if the top cue were absent. Thus, it strictly predicts identical choice probabilities whether a tied top cue is present or absent. The competing theory, however, applies a configural weight to all cues based on the total number of cues favoring each option. Adding a tied top cue increases the total cue sum for both options, non-linearly shifting the effective weights of all subsequent discriminating cues and leading to systematically different choice probabilities.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Rank-Weighted Sequential Sampling with Forward-Looking Stopping: Decision-makers evaluate cues sequentially in descending order of validity, accumulating evidence at each step. To decide whether to stop or continue, they compare the currently accumulated evidence against the maximum possible evidence that could be provided by the remaining cues. The probability of halting increases as the current evidence outweighs the potential remaining evidence. This forward-looking dynamic prevents premature stopping when remaining cues could overturn the decision, seamlessly bridging Take-The-Best and Tallying-like compensatory behavior.

**Parameters:**
- lambda_stop: [0.0, 3.0]
- beta: [0.1, 25.0]
- gamma: [0.0, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    lambda_stop = float(parameters["lambda_stop"])
    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by validity descending (stable sort to preserve original order on ties)
    order = np.argsort(-val, kind='stable')
    
    E = 0.0
    p_continue = 1.0
    p_choose_A = 0.0
    
    for i, idx in enumerate(order):
        # Accumulate evidence weighted by validity
        w = val[idx] ** gamma
        E += w * (a[idx] - b[idx])
        
        # Stopping probability depends on absolute accumulated evidence relative to remaining possible evidence
        if i < len(order) - 1:
            max_remaining = sum((val[j] ** gamma) for j in order[i+1:])
            S = 1.0 - np.exp(-lambda_stop * (abs(E) / (max_remaining + 1e-3)))
        else:
            S = 1.0  # Must stop at the last cue
            
        p_stop_here = p_continue * S
        p_continue *= (1.0 - S)
        
        # Choice probability if stopped at this step
        z = beta * E
        if z > 20.0:
            p_A_given_stop = 1.0
        elif z < -20.0:
            p_A_given_stop = 0.0
        else:
            p_A_given_stop = 1.0 / (1.0 + np.exp(-z))
            
        p_choose_A += p_stop_here * p_A_given_stop
        
    # Apply lapse rate
    p_A_final = (1.0 - epsilon) * p_choose_A + epsilon * 0.5
    
    return np.array([p_A_final, 1.0 - p_A_final])
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
**Description:** Leaky Competing Accumulator with Non-linear Configural Weighting and Evidence Transduction: Decision-makers evaluate cues simultaneously, with evidence for each option dynamically inhibiting the other in a leaky competing accumulator (LCA). Cues have a configural impact, scaled non-linearly by the total number of supporting cues. Additionally, the integrated evidence for each option is passed through a non-linear transducer (alpha) before entering the accumulation process, allowing the model to amplify the differences driven by high-validity cues and capture strong non-compensatory reversals.

**Parameters:**
- gamma: [-10.0, 10.0]
- leak: [0.1, 2.0]
- inhibition: [0.0, 5.0]
- theta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- alpha: [0.1, 10.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    leak = float(parameters["leak"])
    inhibition = float(parameters["inhibition"])
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    alpha = float(parameters["alpha"])
    
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    
    # Configural weighting: effective validity of a cue is non-linearly modulated by the total number of cues
    # Using max(1e-6, sum) to avoid 0^negative_gamma undefined errors
    sum_a_safe = max(1e-6, sum_a)
    sum_b_safe = max(1e-6, sum_b)
    
    w_a = val * (sum_a_safe ** gamma)
    w_b = val * (sum_b_safe ** gamma)
    
    # Make sure inputs are non-negative and apply non-linear transducer alpha
    I_A = max(0.0, np.sum(w_a * a)) ** alpha
    I_B = max(0.0, np.sum(w_b * b)) ** alpha
    
    # Leaky Competing Accumulator (LCA) simulation
    x_a, x_b = 0.0, 0.0
    dt = 0.1
    steps = 100
    
    for _ in range(steps):
        dx_a = (I_A - leak * x_a - inhibition * x_b) * dt
        dx_b = (I_B - leak * x_b - inhibition * x_a) * dt
        
        x_a = max(0.0, x_a + dx_a)
        x_b = max(0.0, x_b + dx_b)
        
    # Softmax choice based on final activations
    z = theta * np.array([x_a, x_b])
    z = z - np.max(z)
    e = np.exp(z)
    p = e / np.sum(e)
    
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
[0] rationale: This metric calculates the mean absolute difference in choice probabilities between trial pairs that differ only in the presence or absence of a tied top-validity cue. The advocated Rank-Weighted Sequential Sampling theory strictly predicts identical choice probabilities across these pairs (difference near 0), as tied top cues contribute no accumulated evidence and do not alter the stopping probability. The competing LCA with Configural Weighting theory predicts a systematic difference (difference > 0), because adding a tied top cue increases the total cue sum for both options, non-linearly shifting the effective weights of all subsequent discriminating cues.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_rest = data['option_a_ratings'].apply(lambda x: tuple(x[1:]))
    b_rest = data['option_b_ratings'].apply(lambda x: tuple(x[1:]))
    top_cue = data['option_a_ratings'].apply(lambda x: x[0])
    
    df = pd.DataFrame({
        'a_rest': a_rest,
        'b_rest': b_rest,
        'top_cue': top_cue,
        'response': data['response']
    })
    
    diffs = []
    for (ar, br), group in df.groupby(['a_rest', 'b_rest']):
        p0 = group[group['top_cue'] == 0]['response'].mean()
        p1 = group[group['top_cue'] == 1]['response'].mean()
        if not pd.isna(p0) and not pd.isna(p1):
            diffs.append(abs(p0 - p1))
            
    if len(diffs) == 0:
        return 0.0
        
    return float(np.mean(diffs))
outcome: self_sim=0.0231 (var=0.0044) adversary_sim=0.0516 (var=0.0093) welch_t=-1.213 p=0.2318 (N=25, alpha=0.01) -> reject

[1] rationale: Instead of computing the mean absolute difference across all pairs (which suffers from positive bias and high variance), this metric computes the signed difference in the probability of choosing Option A when a tied top cue is present versus absent, specifically for trials where Option A has fewer total cues than Option B. Under Rank-Weighted Sequential Sampling, the tied top cue is ignored, predicting a difference of exactly zero. Under LCA with Configural Weighting, adding a tied top cue to both options changes the ratio of total cues (e.g., from 1:2 to 2:3), systematically shifting the relative configural weights and thereby increasing the probability of choosing Option A. This directional contrast produces a robust gap with tight variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Identify trials where sum of cues 2-5 for A is less than for B
    a_rest_sum = data['option_a_ratings'].apply(lambda x: sum(x[1:]))
    b_rest_sum = data['option_b_ratings'].apply(lambda x: sum(x[1:]))
    
    # We focus on trials where option A is numerically at a disadvantage in the remaining cues
    mask = a_rest_sum < b_rest_sum
    df_subset = data[mask].copy()
    
    if len(df_subset) == 0:
        return 0.0
        
    df_subset['top_cue'] = df_subset['option_a_ratings'].apply(lambda x: x[0])
    df_subset['chose_A'] = (df_subset['response'] == 0).astype(float)
    
    p_A_1 = df_subset[df_subset['top_cue'] == 1]['chose_A'].mean()
    p_A_0 = df_subset[df_subset['top_cue'] == 0]['chose_A'].mean()
    
    if pd.isna(p_A_1) or pd.isna(p_A_0):
        return 0.0
        
    return float(p_A_1 - p_A_0)
outcome: self_sim=0.0044 (var=0.0181) adversary_sim=-0.0822 (var=0.0201) welch_t=+2.216 p=0.03148 (N=25, alpha=0.01) -> reject

[2] rationale: To robustly discriminate the theories while aggressively suppressing binomial noise from finite trials, this metric isolates three strictly independent pairs of trials where a tied top-validity cue is added to options with an unequal number of cues (A < B). 

Under the advocated Rank-Weighted Sequential Sampling model, tied top cues are entirely ignored, meaning the true choice probability shift for all three contrasts (C1, C2, C3) is exactly zero. Because the contrasts are calculated on mutually exclusive trials, their estimation errors are perfectly independent. Consequently, the expected value of their pairwise cross-products (C1*C2 + C2*C3 + C1*C3) is mathematically guaranteed to be exactly zero, leaving only a minuscule variance floor.

Under the competing LCA with Configural Weighting theory, adding a tied cue changes the ratio of total cues (e.g., from 1/2 to 2/3), systematically shifting the relative configural weights. Critically, because the parameter 'gamma' is fixed per subject, the direction of this shift will be identical across all three contrasts for a given subject. Thus, C1, C2, and C3 will consistently share the same sign. The pairwise cross-products will therefore be strictly positive, cleanly extracting the variance of the true effect while discarding mean-zero binomial noise.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def get_pA(df, a_tuple, b_tuple):
        mask = (df['option_a_ratings'].apply(tuple) == a_tuple) & \
               (df['option_b_ratings'].apply(tuple) == b_tuple)
        return (df[mask]['response'] == 0).mean()
        
    scores = []
    for subj, subj_df in data.groupby('subject_id'):
        # Contrast 1: Pair 2 vs Pair 1 (Tied cue 1 added, A has 1 vs B has 2 cues)
        pA_2 = get_pA(subj_df, (1, 1, 0, 0, 0), (1, 0, 1, 1, 0))
        pA_1 = get_pA(subj_df, (0, 1, 0, 0, 0), (0, 0, 1, 1, 0))
        
        # Contrast 2: Pair 10 vs Pair 9 (Tied cue 1 added, A has 2 vs B has 3 cues)
        pA_10 = get_pA(subj_df, (1, 1, 0, 0, 1), (1, 0, 1, 1, 1))
        pA_9  = get_pA(subj_df, (0, 1, 0, 0, 1), (0, 0, 1, 1, 1))
        
        # Contrast 3: Pair 8 vs Pair 7 (Tied cue 1 added, A has 1 vs B has 2 cues)
        pA_8 = get_pA(subj_df, (1, 0, 1, 0, 0), (1, 0, 0, 1, 1))
        pA_7 = get_pA(subj_df, (0, 0, 1, 0, 0), (0, 0, 0, 1, 1))
        
        if pd.isna(pA_1) or pd.isna(pA_2) or pd.isna(pA_9) or pd.isna(pA_10) or pd.isna(pA_7) or pd.isna(pA_8):
            continue
            
        C1 = pA_2 - pA_1
        C2 = pA_10 - pA_9
        C3 = pA_8 - pA_7
        
        # Sum of pairwise cross-products of the three independent contrasts
        scores.append(C1 * C2 + C2 * C3 + C1 * C3)
        
    if not scores:
        return 0.0
        
    return float(sum(scores) / len(scores))
outcome: self_sim=0.0165 (var=0.0094) adversary_sim=0.0405 (var=0.0250) welch_t=-0.646 p=0.522 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    target_rests = [\n        ((1, 0, 0, 0), (0, 1, 1, 0)),\n        ((0, 1, 0, 0), (0, 0, 1, 1)),\n        ((1, 0, 0, 1), (0, 1, 1, 1))\n    ]\n    \n    data = data.copy()\n    data['a_rest'] = data['option_a_ratings'].apply(lambda x: tuple(x[1:]))\n    data['b_rest'] = data['option_b_ratings'].apply(lambda x: tuple(x[1:]))\n    data['top_cue'] = data['option_a_ratings'].apply(lambda x: x[0])\n    data['chose_A'] = (data['response'] == 0).astype(int)\n    \n    scores = []\n    for subj, subj_df in data.groupby('subject_id'):\n        subj_score = 0.0\n        valid_pairs = 0\n        \n        for a_rest, b_rest in target_rests:\n            mask = (subj_df['a_rest'] == a_rest) & (subj_df['b_rest'] == b_rest)\n            pair_df = subj_df[mask]\n            \n            df0 = pair_df[pair_df['top_cue'] == 0]\n            df1 = pair_df[pair_df['top_cue'] == 1]\n            \n            n0 = len(df0)\n            n1 = len(df1)\n            \n            if n0 == 0 or n1 == 0:\n                continue\n                \n            X = df0['chose_A'].sum()\n            Y = df1['chose_A'].sum()\n            \n            L0 = np.log((X + 0.5) / (n0 - X + 0.5))\n            L1 = np.log((Y + 0.5) / (n1 - Y + 0.5))\n            \n            subj_score += (L1 - L0)\n            valid_pairs += 1\n            \n        if valid_pairs > 0:\n            scores.append(subj_score / valid_pairs)\n            \n    if not scores:\n        return 0.0\n        \n    return float(np.mean(scores))",
  "rationale": "Previous attempts using raw probability differences suffered from signal compression because choice probabilities are often pushed near 0 or 1 by the competing theory's non-linear accumulator, making the absolute shift appear small relative to binomial noise. This metric overcomes the compression by calculating the Haldane-Anscombe empirical log odds ratio of choosing Option A when the tied top cue is present versus absent. We focus strictly on the three trial pairs where Option A has fewer total cues than Option B. Under the advocated Rank-Weighted Sequential Sampling theory, the tied top cue is ignored, meaning the true probabilities are identical and the expected log odds difference is exactly zero. Under the competing LCA with Configural Weighting theory, adding the tied cue systematically changes the ratio of total cues, shifting the underlying evidence. The log odds transformation massively amplifies this underlying evidence shift at the extremes, yielding a highly discriminative signal with stable variance."
}
```

## Usage

```json
{
  "prompt_token_count": 5993,
  "candidates_token_count": 808,
  "total_token_count": 32514
}
```
