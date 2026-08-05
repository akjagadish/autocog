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
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 3: A=[0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Non-linear Feature Difference Model with Flexible Validity Centering) from the Competing Theory (Leaky Competing Accumulator), we exploit the Advocated Theory's 'theta' parameter, which allows for flexible centering of validities. In the Competing Theory, validities are strictly centered at 0.5, meaning any feature with a validity > 0.5 always provides positive evidence for the option that wins it. Consequently, if Option A dominates Option B on low-validity features, the Competing Theory structurally must accumulate positive evidence for A. In contrast, the Advocated Theory subtracts 'theta' from the validities. If 'theta' is higher than the validity of a feature, winning that feature yields a NEGATIVE weight, actively penalizing the option. By constructing trials where Option A wins on the highest validity feature AND multiple low-validity features, while Option B wins only on the second-highest validity feature, the Competing Theory predicts a robust preference for A (due to early accumulation from F1 and support from F3-F5). However, the Advocated Theory can predict a strong preference for B, as the 'wins' on the low-validity features heavily penalize Option A.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Non-linear Feature Difference Model with Flexible Validity Centering

**Parameters:**
- theta: [0.0, 1.0]
- gamma: [0.1, 5.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Center validities with a flexible threshold rather than hardcoded 0.5
    theta = float(parameters["theta"])
    w = val - theta
    
    diff = a - b
    # Compute net evidence for option A over B
    net_ev = np.sum(w * diff)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Apply non-linear value function to the net evidence (Prospect Theory style)
    score = np.sign(net_ev) * (np.abs(net_ev) ** gamma)
    
    # Create scores for A and B
    scores = np.array([score, -score])
    
    # Softmax choice rule
    z = beta * scores
    z -= np.max(z) # For numerical stability
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
**Description:** Decision-makers evaluate options using a Leaky Competing Accumulator (LCA) process, where attention shifts sequentially across features in order of their validity. Evidence for each option accumulates continuously over time, subject to information decay (leakage) and lateral inhibition. When lateral inhibition is strong, early evidence from highly valid features quickly suppresses the competing option, locking in a choice and producing non-compensatory 'Take-The-Best' behavior. Conversely, when leakage and inhibition are low, evidence from all features integrates more evenly without suppression, resulting in compensatory 'Tallying' or WADD-like behavior. This provides a unified mechanistic account for the spectrum of decision strategies without requiring explicit rule-switching.

**Parameters:**
- leak: [0.0, 2.0]
- inhibition: [0.0, 5.0]
- steps_per_feature: [1, 50]
- gamma: [0.1, 5.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity (descending) to simulate sequential attention
    order = np.argsort(val)[::-1]
    
    leak = float(parameters["leak"])
    inhibition = float(parameters["inhibition"])
    steps_per_feature = int(float(parameters["steps_per_feature"]))
    gamma = float(parameters["gamma"])
    
    x_A, x_B = 0.0, 0.0
    dt = 0.1
    
    for idx in order:
        # Non-linear scaling of validities centered at chance
        v = max(0.0, val[idx] - 0.5) ** gamma
        I_A = a[idx] * v
        I_B = b[idx] * v
        
        # Accumulate evidence over time for the current feature
        for _ in range(steps_per_feature):
            dx_A = (I_A - leak * x_A - inhibition * x_B) * dt
            dx_B = (I_B - leak * x_B - inhibition * x_A) * dt
            
            # Rectified linear units (activations cannot be negative)
            x_A = max(0.0, x_A + dx_A)
            x_B = max(0.0, x_B + dx_B)
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    scores = np.array([x_A, x_B])
    
    # Softmax choice rule with numerical stability
    z = beta * scores
    z -= np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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
[0] rationale: Focuses on Trial 1, where Option A dominates Option B on low-validity features while Option B has no positive features. Under the Competing Theory, validities are centered at 0.5, so Option A always receives positive evidence while B receives zero, leading to a strong preference for A. Under the Advocated Theory, flexible validity centering (theta parameter) means that if theta is greater than the validities of the features A possesses, A actually receives negative evidence, resulting in a preference for B. Thus, the proportion of times Option B is chosen on this trial provides a clear discriminant between the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 0, 0))
    if t1_mask.sum() == 0:
        return 0.0
    return float(data.loc[t1_mask, 'response'].mean())
outcome: self_sim=0.3850 (var=0.0866) adversary_sim=0.4850 (var=0.0352) welch_t=-1.433 p=0.1596 (N=25, alpha=0.01) -> reject

[1] rationale: Contrasts Trial 2 (A=[1,0,0,0,0]) and Trial 3 (A=[1,0,0,1,1]), which have the same Option B. Under the Competing Theory, validities are centered at 0.5, so features 4 and 5 (validities 0.65 and 0.55) provide strictly positive evidence. Thus, adding them to A in Trial 3 must increase the probability of choosing A compared to Trial 2. Under the Advocated Theory, the flexible centering parameter 'theta' can exceed 0.65, meaning features 4 and 5 can act as negative evidence. Consequently, the Advocated Theory predicts that adding these features can paradoxically decrease the probability of choosing A. The metric computes the difference in the probability of choosing A between Trial 2 and Trial 3.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t2_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0))
    t3_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 1)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0))
    
    if t2_mask.sum() == 0 or t3_mask.sum() == 0:
        return 0.0
        
    p_a_t2 = 1.0 - data.loc[t2_mask, 'response'].mean()
    p_a_t3 = 1.0 - data.loc[t3_mask, 'response'].mean()
    
    return float(p_a_t2 - p_a_t3)
outcome: self_sim=-0.1317 (var=0.2378) adversary_sim=-0.0467 (var=0.0431) welch_t=-0.802 p=0.4285 (N=25, alpha=0.01) -> reject

[2] rationale: Focuses on Trial 6 (A=[1,0,0,0,1], B=[0,1,0,1,0]). In the Advocated Theory, the net evidence is the sum of validity differences: (0.95 - theta) - (0.85 - theta) - (0.65 - theta) + (0.55 - theta). The thetas perfectly cancel out, and the validity values evaluate to exactly 0.95 - 0.85 - 0.65 + 0.55 = 0.0. Thus, the Advocated Theory mathematically guarantees a choice probability of exactly 0.5 for Option A, regardless of any parameter values, producing extremely tight binomial variance across subjects. In contrast, the Competing Theory (LCA) processes features sequentially by validity. Option A receives the strongest feature (0.95) first, gaining an early lead that laterally inhibits Option B before B's features (0.85, 0.65) are fully processed. Thus, LCA predicts a robust preference for Option A (>0.5).
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 6 where A=[1, 0, 0, 0, 1] and B=[0, 1, 0, 1, 0]
    t6_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 1)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 0, 1, 0))
    if t6_mask.sum() == 0:
        return 0.5
    # Return the proportion of times Option A was chosen
    return float(1.0 - data.loc[t6_mask, 'response'].mean())
outcome: self_sim=0.5233 (var=0.0161) adversary_sim=0.5100 (var=0.0280) welch_t=+0.318 p=0.7523 (N=25, alpha=0.01) -> reject

[3] rationale: This metric calculates the overall proportion of times Option B is chosen across Trials 2, 4, and 5. In all three of these trials, Option A possesses the single most valid feature available, while Option B possesses multiple features of lower validity. The Competing Theory (LCA) processes features sequentially by validity; thus, Option A's best feature is evaluated first, granting it an early lead that laterally inhibits Option B, leading to a strong, consistent preference for Option A (a lower metric value). Conversely, the Advocated Theory integrates all features simultaneously. Because Option B has a greater number of features, their combined weight typically outweighs Option A's single feature (unless the flexible centering parameter 'theta' is unusually high), leading to a much higher proportion of Option B choices (a higher metric value). Pooling across these three structurally similar trials reduces within-subject variance while maintaining a sharp contrast between the compensatory and non-compensatory mechanisms of the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    mask = a_tuples.isin([
        (1, 0, 0, 0, 0),  # Trial 2
        (0, 0, 1, 0, 0),  # Trial 4
        (0, 1, 0, 0, 0)   # Trial 5
    ])
    if mask.sum() == 0:
        return 0.0
    return float(data.loc[mask, 'response'].mean())
outcome: self_sim=0.5422 (var=0.0587) adversary_sim=0.5067 (var=0.0103) welch_t=+0.677 p=0.5035 (N=25, alpha=0.01) -> reject

[4] rationale: Advocated theory predicts that Trial 1 (A=[1,0,1,1,1], B=[0,1,0,0,0]) and Trial 3 (A=[0,0,1,1,1], B=[0,0,0,0,0]) have almost identical net evidence. Specifically, net_ev1 - net_ev3 = (1.7 - 3*theta) - (1.65 - 3*theta) = 0.05. Because this difference is tiny and mathematically independent of the flexible centering parameter theta, the Advocated theory predicts P(A|T1) ≈ P(A|T3) for any subject, yielding a difference near 0 with low variance. Conversely, the Competing theory (LCA) evaluates options sequentially by validity. In Trial 3, B has no positive features, so A receives all evidence and wins almost deterministically. In Trial 1, B possesses the second most valid feature (F2), which provides strong competing evidence immediately after F1 is processed. This competition significantly reduces P(A|T1) compared to Trial 3. Thus, LCA predicts a robustly positive difference P(A|T3) - P(A|T1).
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t1_mask = (a_str == '10111') & (b_str == '01000')
    t3_mask = (a_str == '00111') & (b_str == '00000')
    
    if t1_mask.sum() == 0 or t3_mask.sum() == 0:
        return 0.0
        
    p_a_t1 = 1.0 - data.loc[t1_mask, 'response'].mean()
    p_a_t3 = 1.0 - data.loc[t3_mask, 'response'].mean()
    
    return float(p_a_t3 - p_a_t1)
outcome: self_sim=0.0025 (var=0.0175) adversary_sim=-0.0038 (var=0.0296) welch_t=+0.144 p=0.8861 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    # Safely create string representations of the ratings to avoid unhashable list errors\n    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    \n    # Combine to form a unique trial identifier\n    trial_id = a_str + '_' + b_str\n    \n    # Create a temporary dataframe to avoid modifying the original\n    temp_df = data[['subject_id', 'response']].copy()\n    temp_df['trial_id'] = trial_id\n    \n    # Group by subject and trial, compute mean response\n    subj_trial_means = temp_df.groupby(['subject_id', 'trial_id'])['response'].mean()\n    \n    # Compute absolute deviation from random guessing (0.5)\n    abs_dev = np.abs(subj_trial_means - 0.5)\n    \n    # Return the average absolute deviation\n    return float(abs_dev.mean())",
  "rationale": "The Competing Theory (LCA) scales validities by `(val - 0.5)^gamma`. Because all validities in this design are relatively close to 0.5, when the parameter `gamma` is greater than 1 (which occurs 80% of the time given its uniform distribution over [0.1, 5.0]), the evidence values are squashed to near zero. This leads to a lack of evidence accumulation, resulting in choices driven primarily by noise, yielding choice probabilities close to 0.5 for almost all trials. Thus, the mean absolute deviation of choice proportions from 0.5 will be low (approaching the expectation for random guessing). In contrast, the Advocated Theory computes net evidence using `val - theta`. For most values of `theta`, the net evidence for any given trial is substantially non-zero. Combined with the non-linear value function and softmax, this produces highly deterministic choice probabilities (close to 0 or 1). Therefore, the mean absolute deviation of choice proportions from 0.5 will be significantly higher in the Advocated Theory. By averaging this absolute deviation across all trials, we obtain a robust measure of 'choice determinism' that cleanly discriminates the theories with minimal between-subject variance."
}
```

## Usage

```json
{
  "prompt_token_count": 5944,
  "candidates_token_count": 570,
  "total_token_count": 33729
}
```
