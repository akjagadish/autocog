# metric_exp01_attempt_01

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
**Validities (n_features=5):** [0.95, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate Noisy Tallying from Noisy WADD, we use an environment with two highly valid features and three weakly valid features. The critical trial pits an option supported by the two highly valid features against an option supported by the three weakly valid ones. Noisy Tallying ignores validities and strictly prefers the option with three features (3 vs 2 wins). In contrast, Noisy WADD integrates the validities and strictly prefers the option with the two highly valid features, because their combined sum exceeds the sum of the three weakly valid features. Additional trials include cases where Tallying ties but WADD has a strict preference, providing further constraints to definitively separate the models.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Noisy Tallying (Equal Weights): Decision-makers ignore feature validities and instead rely on a simple tallying heuristic, counting the total number of positive features (1s) for each option. They tend to choose the option with the higher count. To account for the highly consistent ~0.5 choice probabilities observed across experiments, this process is heavily smoothed by a prominent noise mechanism, modeled via a high uniform lapse rate and a low inverse temperature in the softmax choice rule.

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.5, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Tallying: Sum of active features for each option (ignoring validities)
    scores = np.sum(stim, axis=1)
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
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


## COMPETING THEORY
**Description:** Noisy Weighted Additive (WADD) Model: Decision-makers compute a global value for each option by summing the validities of its active features. Rather than relying on a single discriminating feature (like Take-The-Best) or ignoring validities (like Tallying), WADD integrates all available validities compensatorily. To account for the highly consistent ~0.5 choice probabilities and near-zero differences observed across the experiments, the decision process incorporates a highly prominent noise mechanism. This is modeled via a very low inverse temperature in the softmax choice rule and a potentially high uniform lapse rate, smoothing out predictions and avoiding deterministic swings.

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.5, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # WADD: Sum of validities for active features for each option
    scores = np.sum(stim * validities, axis=1)
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
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
[0] rationale: This metric isolates the critical trial pair where Option 1 has 3 active features of low validity (0.6, 0.55, 0.5) and Option 2 has 2 active features of high validity (0.95, 0.85). The Noisy Tallying model ignores validities and strictly prefers Option 1 (more active features), leading to a choice probability < 0.5 for Option 2. Conversely, the Noisy WADD model integrates validities and strictly prefers Option 2 (higher weighted sum: 1.8 vs 1.65), leading to a choice probability > 0.5 for Option 2. Computing the empirical probability of choosing Option 2 directly distinguishes the two models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    target_opt1 = (0, 0, 1, 1, 1)
    target_opt2 = (1, 1, 0, 0, 0)
    
    count = 0
    total = 0
    
    for _, row in data.iterrows():
        a_val = tuple(row['option_a_ratings'])
        b_val = tuple(row['option_b_ratings'])
        
        if a_val == target_opt1 and b_val == target_opt2:
            total += 1
            if row['response'] == 1:
                count += 1
        elif a_val == target_opt2 and b_val == target_opt1:
            total += 1
            if row['response'] == 0:
                count += 1
                
    if total == 0:
        return 0.5
    return count / total
outcome: self_sim=0.4825 (var=0.0170) adversary_sim=0.5125 (var=0.0211) welch_t=-0.768 p=0.4462 (N=25, alpha=0.01) -> reject

[1] rationale: By taking a within-subject contrast across three specific trial types, we subtract out baseline choice biases and cleanly separate the models. In Trials 1, 2, and 4, the Tallying model predicts an identical difference in active features (+1 in favor of option A), meaning it predicts the exact same choice probability for A across all three trials. The WADD model, however, predicts varying differences in weighted sums (+0.55, +0.20, and -0.15 respectively). By computing the contrast (P(A|Trial 4) + P(A|Trial 2) - 2*P(A|Trial 1)), Tallying expects a value of roughly 0, while WADD expects a robustly positive value.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t1_A = (0, 0, 1, 1, 1)
    t1_B = (1, 1, 0, 0, 0)
    
    t2_A = (0, 0, 1, 1, 0)
    t2_B = (1, 0, 0, 0, 0)
    
    t4_A = (0, 1, 1, 1, 0)
    t4_B = (1, 0, 0, 0, 1)
    
    counts_A = {1: 0, 2: 0, 4: 0}
    totals = {1: 0, 2: 0, 4: 0}
    
    for _, row in data.iterrows():
        a_val = tuple(row['option_a_ratings'])
        b_val = tuple(row['option_b_ratings'])
        
        if (a_val == t1_A and b_val == t1_B):
            totals[1] += 1
            if row['response'] == 0: counts_A[1] += 1
        elif (a_val == t1_B and b_val == t1_A):
            totals[1] += 1
            if row['response'] == 1: counts_A[1] += 1
            
        elif (a_val == t2_A and b_val == t2_B):
            totals[2] += 1
            if row['response'] == 0: counts_A[2] += 1
        elif (a_val == t2_B and b_val == t2_A):
            totals[2] += 1
            if row['response'] == 1: counts_A[2] += 1
            
        elif (a_val == t4_A and b_val == t4_B):
            totals[4] += 1
            if row['response'] == 0: counts_A[4] += 1
        elif (a_val == t4_B and b_val == t4_A):
            totals[4] += 1
            if row['response'] == 1: counts_A[4] += 1

    p1 = counts_A[1] / totals[1] if totals[1] > 0 else 0.5
    p2 = counts_A[2] / totals[2] if totals[2] > 0 else 0.5
    p4 = counts_A[4] / totals[4] if totals[4] > 0 else 0.5
    
    return (p4 + p2) - 2 * p1
outcome: self_sim=-0.0162 (var=0.1148) adversary_sim=0.0650 (var=0.1009) welch_t=-0.875 p=0.3862 (N=25, alpha=0.01) -> reject

[2] rationale: This metric exploits a fundamental structural difference in the internal confidence of the two models across the experimental design. Because Tallying computes raw feature sums (which are integers) and WADD computes weighted sums (which are fractional and bounded by the maximum validities), the inputs to the softmax function (z = beta * score_diff) are systematically much larger for Tallying than for WADD across almost all trials. For example, on Trial 6, Tallying sees a score difference of 2, whereas WADD sees a score difference of only 0.70. Even with identical noise parameters (beta and epsilon), Tallying will produce choice probabilities that are pushed much further away from 0.5. By summing the probability of the Tallying-favored option across the 5 trials where Tallying has a strict preference, we aggregate this confidence gap. Tallying expects a large positive sum (driven by z-values of beta, beta, beta, beta, and 2*beta), whereas WADD expects a much smaller sum (driven by z-values of -0.15*beta, 0.20*beta, 0.55*beta, 0.30*beta, and 0.70*beta). This aggregation over 80 trials per subject produces a highly stable metric with a massive mean difference between the theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Trial types defined by their option A ratings
    t1_A = (0, 0, 1, 1, 1)
    t2_A = (0, 0, 1, 1, 0)
    t4_A = (0, 1, 1, 1, 0)
    t5_A = (0, 1, 0, 0, 0)
    t6_A = (1, 0, 0, 0, 0)
    
    counts = {1: 0, 2: 0, 4: 0, 5: 0, 6: 0}
    totals = {1: 0, 2: 0, 4: 0, 5: 0, 6: 0}
    
    for _, row in data.iterrows():
        a_val = tuple(row['option_a_ratings'])
        b_val = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a_val == t1_A:
            totals[1] += 1
            if resp == 0: counts[1] += 1
        elif b_val == t1_A:
            totals[1] += 1
            if resp == 1: counts[1] += 1
            
        elif a_val == t2_A:
            totals[2] += 1
            if resp == 0: counts[2] += 1
        elif b_val == t2_A:
            totals[2] += 1
            if resp == 1: counts[2] += 1
            
        elif a_val == t4_A:
            totals[4] += 1
            if resp == 0: counts[4] += 1
        elif b_val == t4_A:
            totals[4] += 1
            if resp == 1: counts[4] += 1
            
        elif a_val == t5_A:
            totals[5] += 1
            if resp == 1: counts[5] += 1 # Note: counting B choices
        elif b_val == t5_A:
            totals[5] += 1
            if resp == 0: counts[5] += 1
            
        elif a_val == t6_A:
            totals[6] += 1
            if resp == 1: counts[6] += 1 # Note: counting B choices
        elif b_val == t6_A:
            totals[6] += 1
            if resp == 0: counts[6] += 1

    p1 = counts[1] / totals[1] if totals[1] > 0 else 0.5
    p2 = counts[2] / totals[2] if totals[2] > 0 else 0.5
    p4 = counts[4] / totals[4] if totals[4] > 0 else 0.5
    p5 = counts[5] / totals[5] if totals[5] > 0 else 0.5
    p6 = counts[6] / totals[6] if totals[6] > 0 else 0.5
    
    return p1 + p2 + p4 + p5 + p6
outcome: self_sim=2.5094 (var=0.0432) adversary_sim=2.5269 (var=0.0525) welch_t=-0.283 p=0.7785 (N=25, alpha=0.01) -> reject

[3] rationale: This metric utilizes a strict within-subject contrast between Trial 4 and Trial 1 to cancel out baseline noise and strictly separate the models. For Tallying, both Trial 1 and Trial 4 present an identical difference in the number of active features (+1 in favor of option A). Thus, Tallying predicts exactly the same choice probability for Option A on both trials, meaning the expected value of P(A|Trial 4) - P(A|Trial 1) is exactly 0. For WADD, however, Trial 4 strongly favors Option A (weighted difference of +0.55), while Trial 1 actually favors Option B (weighted difference of -0.15). Therefore, WADD predicts a strictly positive value for P(A|Trial 4) - P(A|Trial 1). By limiting the contrast to only these two trials, we minimize the binomial variance from trial counts, producing a tight distribution around 0 for Tallying and a robustly positive mean for WADD.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t1_a = (0, 0, 1, 1, 1)
    t1_b = (1, 1, 0, 0, 0)
    t4_a = (0, 1, 1, 1, 0)
    t4_b = (1, 0, 0, 0, 1)
    
    p1_a = 0
    n1 = 0
    p4_a = 0
    n4 = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == t1_a and b == t1_b:
            n1 += 1
            if resp == 0: p1_a += 1
        elif a == t1_b and b == t1_a:
            n1 += 1
            if resp == 1: p1_a += 1
            
        elif a == t4_a and b == t4_b:
            n4 += 1
            if resp == 0: p4_a += 1
        elif a == t4_b and b == t4_a:
            n4 += 1
            if resp == 1: p4_a += 1
            
    p1 = p1_a / n1 if n1 > 0 else 0.5
    p4 = p4_a / n4 if n4 > 0 else 0.5
    
    return p4 - p1
outcome: self_sim=-0.0200 (var=0.0269) adversary_sim=0.0237 (var=0.0440) welch_t=-0.821 p=0.4159 (N=25, alpha=0.01) -> reject

[4] rationale: Because the noise parameters push choice probabilities very close to 0.5, raw single-trial differences are heavily masked by binomial variance. To maximize the statistical separation between Tallying and WADD, we must project the four choice probabilities onto the optimal linear contrast vector. The z-scores for Tallying across the four trials are proportional to V_T = [-1, 0, -1, 2], while for WADD they are proportional to V_W = [0.15, 0.10, -0.35, 1.05]. The difference vector D = V_T - V_W = [-1.15, -0.10, -0.65, 0.95]. Weighting the empirical probabilities by these exact coefficients maximally separates the expected values of the two models while strictly controlling the variance. Previous attempts failed due to typos in the hardcoded trial arrays that caused them to match zero trials; this metric correctly matches the prompt's arrays.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t1_a = (1, 1, 0, 0, 0)
    t1_b = (0, 0, 1, 1, 1)
    
    t2_a = (1, 0, 0, 0, 0)
    t2_b = (0, 1, 0, 0, 0)
    
    t3_a = (1, 0, 1, 0, 0)
    t3_b = (0, 1, 0, 1, 1)
    
    t4_a = (0, 1, 1, 1, 0)
    t4_b = (1, 0, 0, 0, 0)
    
    counts_A = {1: 0, 2: 0, 3: 0, 4: 0}
    totals = {1: 0, 2: 0, 3: 0, 4: 0}
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == t1_a and b == t1_b:
            totals[1] += 1
            if resp == 0: counts_A[1] += 1
        elif a == t1_b and b == t1_a:
            totals[1] += 1
            if resp == 1: counts_A[1] += 1
            
        elif a == t2_a and b == t2_b:
            totals[2] += 1
            if resp == 0: counts_A[2] += 1
        elif a == t2_b and b == t2_a:
            totals[2] += 1
            if resp == 1: counts_A[2] += 1
            
        elif a == t3_a and b == t3_b:
            totals[3] += 1
            if resp == 0: counts_A[3] += 1
        elif a == t3_b and b == t3_a:
            totals[3] += 1
            if resp == 1: counts_A[3] += 1
            
        elif a == t4_a and b == t4_b:
            totals[4] += 1
            if resp == 0: counts_A[4] += 1
        elif a == t4_b and b == t4_a:
            totals[4] += 1
            if resp == 1: counts_A[4] += 1
            
    p1 = counts_A[1] / totals[1] if totals[1] > 0 else 0.5
    p2 = counts_A[2] / totals[2] if totals[2] > 0 else 0.5
    p3 = counts_A[3] / totals[3] if totals[3] > 0 else 0.5
    p4 = counts_A[4] / totals[4] if totals[4] > 0 else 0.5
    
    return -1.15 * p1 - 0.10 * p2 - 0.65 * p3 + 0.95 * p4
outcome: self_sim=-0.4495 (var=0.0261) adversary_sim=-0.4200 (var=0.0253) welch_t=-0.651 p=0.5182 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    t1_a = (1, 1, 0, 0, 0)\n    t1_b = (0, 0, 1, 1, 1)\n    \n    t4_a = (0, 1, 1, 1, 0)\n    t4_b = (1, 0, 0, 0, 0)\n    \n    p1_a = 0\n    n1 = 0\n    p4_a = 0\n    n4 = 0\n    \n    for _, row in data.iterrows():\n        a = tuple(row['option_a_ratings'])\n        b = tuple(row['option_b_ratings'])\n        resp = row['response']\n        \n        if a == t1_a and b == t1_b:\n            n1 += 1\n            if resp == 0: p1_a += 1\n        elif a == t1_b and b == t1_a:\n            n1 += 1\n            if resp == 1: p1_a += 1\n            \n        elif a == t4_a and b == t4_b:\n            n4 += 1\n            if resp == 0: p4_a += 1\n        elif a == t4_b and b == t4_a:\n            n4 += 1\n            if resp == 1: p4_a += 1\n            \n    p1 = p1_a / n1 if n1 > 0 else 0.5\n    p4 = p4_a / n4 if n4 > 0 else 0.5\n    \n    return p4 - p1",
  "rationale": "This metric fixes a critical typo in a previous attempt that caused Trial 4 to be completely ignored. By computing the strictly within-subject contrast P(A|Trial 4) - P(A|Trial 1), we isolate the largest expected divergence between the models. Tallying predicts a massive swing in preference between these two trials (z-score difference of 3.0), expecting a strongly positive value. WADD, however, predicts a much smaller swing (z-score difference of 0.90), expecting a value closer to 0. Focusing on just these two trials minimizes the compounding binomial variance of combining all four trials, maximizing the signal-to-noise ratio."
}
```

## Usage

```json
{
  "prompt_token_count": 7224,
  "candidates_token_count": 585,
  "total_token_count": 37431
}
```
