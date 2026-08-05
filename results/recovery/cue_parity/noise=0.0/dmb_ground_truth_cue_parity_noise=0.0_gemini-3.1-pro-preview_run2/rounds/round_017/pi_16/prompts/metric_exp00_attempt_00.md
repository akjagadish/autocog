# metric_exp00_attempt_00

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
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.55, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 3: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  trial 4: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 6: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  trial 7: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 9: A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 0, 0, 1, 1]
  trial 10: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 11: A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0]
  trial 12: A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1]

**Rationale:** This design quantitatively dissociates the advocated Leaky Competing Accumulator (LCA) with Configural Weighting from the competing Rank-Weighted Sequential Sampling theory by exploiting their fundamentally divergent treatment of tied cues. The competing theory evaluates cues sequentially and accumulates evidence based on the difference between options (a[idx] - b[idx]). For any cue where both options have a '1' (a tied cue), the evidence difference is exactly zero, and the stopping probability (which depends only on accumulated evidence and maximum remaining possible evidence) is unchanged. Thus, the competing theory predicts absolutely identical choice probabilities across sets of trials where tied cues are added. The advocated LCA theory, however, applies a configural weight to all validities based on the total number of cues favoring each option (sum ** gamma). Adding tied cues increases the total sum for both options, which non-linearly scales the effective weights of the discriminating cues. Because this scaling is non-linear, it disproportionately boosts the option with fewer initial cues, predicting a systematic, graded shift in choice probabilities as tied cues are added. By presenting base trials with fixed discriminating cues and systematically adding 1 or 2 tied low-validity cues, we can cleanly test for configural tied-cue integration (LCA) versus tied-cue invariance (Sequential Sampling).

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n    \n    # The first 4 cues define the 'base' discrimination, while the last 2 cues are the manipulated tied cues.\n    # Create a key for the base trial by taking the first 4 cues of option A.\n    data['base_key'] = data['option_a_ratings'].apply(lambda x: tuple(x[:4]))\n    \n    # Count how many tied cues are present (sum of the last 2 cues)\n    data['tied_cues'] = data['option_a_ratings'].apply(lambda x: sum(x[4:]))\n    \n    # Calculate whether option A was chosen\n    data['chose_A'] = (data['response'] == 0).astype(float)\n    \n    # Average choice proportion for Option A grouped by base and number of tied cues\n    agg = data.groupby(['base_key', 'tied_cues'])['chose_A'].mean().unstack('tied_cues')\n    \n    if 0 in agg.columns and 2 in agg.columns:\n        # Calculate the absolute shift in choice probability between 2 tied cues and 0 tied cues\n        diffs = (agg[2] - agg[0]).dropna()\n        if len(diffs) > 0:\n            return float(np.mean(np.abs(diffs)))\n            \n    return 0.0",
  "rationale": "The advocated LCA theory with Configural Weighting predicts that adding tied cues non-linearly scales the effective weights of the discriminating cues, causing a systematic shift in choice probabilities (especially when the initial number of supporting cues is asymmetric). The competing Rank-Weighted Sequential Sampling theory, however, evaluates differences sequentially; tied cues provide zero evidence difference, leading to little to no change in the choice probabilities. This metric calculates the mean absolute shift in the probability of choosing Option A between trials with 0 tied cues and 2 tied cues across the different base trial types."
}
```

## Usage

```json
{
  "prompt_token_count": 4130,
  "candidates_token_count": 477,
  "total_token_count": 7710
}
```
