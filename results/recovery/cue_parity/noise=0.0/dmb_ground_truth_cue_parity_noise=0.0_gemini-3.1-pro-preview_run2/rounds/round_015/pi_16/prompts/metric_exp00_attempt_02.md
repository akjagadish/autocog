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

**Trial pairs (n=14):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0]
  trial 4: A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 6: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0]
  trial 7: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 1]
  trial 8: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 9: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 10: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  trial 11: A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 1, 0]
  trial 12: A=[1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1]
  trial 13: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 14: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1]

**Rationale:** This design quantitatively dissociates the advocated LCA model with Configural Weighting from the competing Sequential Evidence Accumulation (SPRT) theory by exploiting their divergent treatment of tied cues. The competing theory processes cues sequentially based on validity and calculates log-odds evidence differences; crucially, it completely ignores any cue that is tied (where both options have a 1), meaning its choice predictions remain exactly identical regardless of how many tied cues are added. The advocated theory, however, utilizes a configural weighting mechanism where the effective weight of every cue is non-linearly scaled by the total number of cues favoring that option. By presenting base trials with a fixed set of discriminating cues and systematically adding tied cues (1s for both options) to the remaining features, the competing theory predicts perfectly flat, invariant choice probabilities across each set. The advocated theory predicts systematic, graded shifts in choice probabilities as the increasing total cue counts modulate the configural weights of the discriminating cues.

**Computed schedule:** 14 unique pairs × 6 reps = 84 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
**Description:** Sequential Evidence Accumulation with Dynamic Stopping Rule (SPRT Log-Odds Update)

**Parameters:**
- threshold: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Process cues sequentially, ordered by validity (descending)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    diff_evidence = 0.0
    for j in cue_order:
        # Evidence weight for the current cue using SPRT log-odds
        v = np.clip(val[j], 1e-5, 1.0 - 1e-5)
        weight = np.log(v / (1.0 - v))
        
        if a[j] > b[j]:
            diff_evidence += weight
        elif b[j] > a[j]:
            diff_evidence -= weight
            
        # Dynamic stopping rule: stop search if evidence difference exceeds threshold
        if abs(diff_evidence) >= threshold:
            break
            
    # Map accumulated evidence to choice probabilities using softmax
    # diff_evidence represents the relative evidence of A over B
    z = np.array([beta * diff_evidence, 0.0])
    z = z - np.max(z)
    e = np.exp(z)
    p = e / np.sum(e)
    
    # Apply lapse rate
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
[0] rationale: This metric isolates the core divergence between the two models: their treatment of tied cues. The competing SPRT model evaluates evidence sequentially and completely ignores tied cues (where both options have a 1), meaning it predicts perfectly identical choice probabilities for any trials that share the same discriminating cues. The advocated LCA model with Configural Weighting predicts that adding tied cues alters the total cue count, which non-linearly scales the effective weights of the discriminating cues, causing systematic shifts in choice probabilities. By grouping trials by their 'base' discriminating cues and calculating the variance of the choice rates within these groups, we directly measure this shift. The SPRT model will produce a low variance (reflecting only binomial sampling noise), while the LCA model will produce a significantly higher variance due to the systematic configural shifts.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    def get_base_pair(a, b):
        a_disc = tuple(i for i, (x, y) in enumerate(zip(a, b)) if x == 1 and y == 0)
        b_disc = tuple(i for i, (x, y) in enumerate(zip(a, b)) if x == 0 and y == 1)
        return (a_disc, b_disc)
        
    base_pairs = []
    trial_sigs = []
    for a, b in zip(data['option_a_ratings'], data['option_b_ratings']):
        base_pairs.append(get_base_pair(a, b))
        trial_sigs.append((tuple(a), tuple(b)))
        
    df = pd.DataFrame({
        'base_pair': base_pairs,
        'trial_sig': trial_sigs,
        'chose_a': (data['response'] == 0).astype(float)
    })
    
    # Calculate the mean choice rate for Option A for each unique trial
    trial_means = df.groupby(['base_pair', 'trial_sig'])['chose_a'].mean().reset_index()
    
    # Calculate the variance of these choice rates within each base pair group
    # (i.e., across trials that only differ by tied cues)
    variances = trial_means.groupby('base_pair')['chose_a'].var(ddof=0)
    
    # Return the average variance across all base pair groups
    return float(variances.mean())
outcome: self_sim=0.0028 (var=0.0003) adversary_sim=0.0004 (var=0.0001) welch_t=+0.657 p=0.5151 (N=25, alpha=0.01) -> reject

[1] rationale: To overcome the high binomial noise that masked the configural shift in previous attempts, this metric uses an unbiased estimator of the squared OLS slope between the total cue count and choice probability. According to the competing SPRT model, tied cues are completely ignored, meaning the true choice probability is perfectly invariant (slope = 0) across trials sharing the same discriminating cues. Any observed slope is purely due to binomial sampling noise. By calculating the squared OLS slope and exactly subtracting its expected variance under the null hypothesis, the metric produces an expected value of exactly 0 for the SPRT model, with extremely tight variance across subjects. Conversely, the advocated LCA model with Configural Weighting predicts that total cue counts non-linearly scale the effective evidence, creating systematic, monotonic shifts in choice probabilities (non-zero slope). Because the metric squares the slope, it accumulates a strong positive signal for LCA regardless of whether the shift is positive or negative (which depends on the subject's configural gamma parameter). By focusing on 'balanced' trial groups (Trials 1-4 and 9-12) where the shift is most pronounced, this metric perfectly isolates the theoretical divergence while analytically eliminating the noise.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    def get_base_pair(a, b):
        a_disc = tuple(i for i, (x, y) in enumerate(zip(a, b)) if x == 1 and y == 0)
        b_disc = tuple(i for i, (x, y) in enumerate(zip(a, b)) if x == 0 and y == 1)
        return (a_disc, b_disc)
        
    data = data.copy()
    data['base_pair'] = [get_base_pair(a, b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    
    # Filter for balanced base pairs where the number of discriminating cues is equal.
    # For these, the SPRT log-odds difference is identical across tied cue additions.
    data['is_balanced'] = data['base_pair'].apply(lambda bp: len(bp[0]) == len(bp[1]))
    bal_data = data[data['is_balanced']].copy()
    
    bal_data['sum_a'] = bal_data['option_a_ratings'].apply(lambda arr: sum(arr))
    bal_data['chose_a'] = (bal_data['response'] == 0).astype(float)
    
    scores = []
    
    for subj, subj_df in bal_data.groupby('subject_id'):
        subj_scores = []
        for bp, bp_df in subj_df.groupby('base_pair'):
            unique_sums = bp_df['sum_a'].nunique()
            # Only use trial groups with a wide range of tied cue additions (Trials 1-4 and 9-12)
            if unique_sums > 2:
                x = bp_df['sum_a'].values
                y = bp_df['chose_a'].values
                
                n = len(x)
                if n < 2:
                    continue
                    
                x_mean = np.mean(x)
                ss_x = np.sum((x - x_mean)**2)
                
                if ss_x > 0:
                    # OLS slope of choice probability vs total cue count
                    beta = np.sum((x - x_mean) * y) / ss_x
                    
                    # Calculate expected variance of the slope under the null hypothesis (SPRT)
                    # where choices are i.i.d. Bernoulli with constant probability
                    p_hat = np.mean(y)
                    p_var_unbiased = p_hat * (1.0 - p_hat) * (n / (n - 1))
                    expected_var_beta = p_var_unbiased / ss_x
                    
                    # Subtract the expected binomial noise variance to get an unbiased estimator 
                    # of the true squared slope
                    excess_beta2 = beta**2 - expected_var_beta
                    subj_scores.append(excess_beta2)
                    
        if subj_scores:
            scores.append(np.mean(subj_scores))
            
    if not scores:
        return 0.0
        
    # Scale by 1000 for numerical readability
    return float(np.mean(scores)) * 1000.0
outcome: self_sim=3.1700 (var=79.2082) adversary_sim=0.2652 (var=21.5372) welch_t=+1.447 p=0.1565 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])\n    scores = []\n    \n    for subj, subj_df in data.groupby('subject_id'):\n        high_choices = []\n        mid_choices = []\n        \n        for _, row in subj_df.iterrows():\n            a = np.array(row['option_a_ratings'])\n            b = np.array(row['option_b_ratings'])\n            \n            sum_a = np.sum(a)\n            sum_b = np.sum(b)\n            \n            # Only look at trials where both options have the same total number of cues.\n            # This perfectly neutralizes the variance caused by LCA's configural gamma parameter.\n            if sum_a == sum_b:\n                val_a = np.sum(a * val)\n                val_b = np.sum(b * val)\n                \n                chose_a = 1.0 if row['response'] == 0 else 0.0\n                \n                # Trials 1-4 and 9-12: The linear sum of validities strongly favors A.\n                if val_a > val_b + 1e-4:\n                    high_choices.append(chose_a)\n                # Trials 13-14: The linear sum of validities is EXACTLY equal for A and B.\n                elif abs(val_a - val_b) <= 1e-4:\n                    mid_choices.append(chose_a)\n                    \n        if len(high_choices) > 0 and len(mid_choices) > 0:\n            scores.append(np.mean(high_choices) - np.mean(mid_choices))\n            \n    if not scores:\n        return 0.0\n        \n    return float(np.mean(scores))",
  "rationale": "This metric exploits a structural divergence between LCA's linear summation of validities and SPRT's sequential log-odds updating. In Trials 13 and 14, the linear sum of validities for Option A and Option B are exactly equal, and their total cue counts are also equal. Because LCA evaluates evidence by linearly summing validities (modulated by total cue count) before its non-linear transducer, it predicts a choice probability of EXACTLY 0.5 for these trials, regardless of its parameter values. SPRT, however, processes cues sequentially using log-odds. Because the most valid cue favors Option A, and the sum of the non-linear log-odds yields a net positive evidence for A (+0.731), SPRT consistently predicts a high choice probability for A (>0.5). By subtracting the choice rate of Trials 13-14 from the choice rate of Trials 1-12 (where both models strongly favor A), we get a metric that is near 0 for SPRT (since it strongly favors A in both sets) but strongly positive (~0.35) for LCA (since it drops to exactly 0.5 in the latter set). Filtering for equal total cue counts across these sets completely neutralizes the between-subject variance from LCA's configural gamma parameter, ensuring extremely tight within-theory predictions."
}
```

## Usage

```json
{
  "prompt_token_count": 5554,
  "candidates_token_count": 790,
  "total_token_count": 28179
}
```
