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
**Validities (n_features=6):** [0.95, 0.85, 0.8, 0.7, 0.6, 0.55]

**Trial pairs (n=11):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1, 0]
  trial 4: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  trial 5: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 7: A=[0, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1]
  trial 8: A=[1, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  trial 9: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 10: A=[1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 11: A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]

**Rationale:** This design quantitatively dissociates the advocated Sequential Cue Sampling theory from the competing Leaky Competing Accumulator (LCA) theory by exploiting their fundamentally divergent treatment of tied cues. The advocated theory calculates the conflict-modulated threshold based strictly on the tally difference of *discriminating* cues, and evaluates evidence sequentially as the difference between options. Consequently, adding tied cues (where both options have a 1) neither changes the threshold nor the accumulated evidence, leading the advocated theory to predict absolutely identical choice probabilities across sets of trials where tied cues are systematically added. The competing LCA theory, however, applies a configural weight to all validities based on the *total* number of cues favoring each option (sum ** gamma). Adding tied cues increases this total sum for both options, which non-linearly scales the effective weights of the discriminating cues. Because this scaling is non-linear, it disproportionately affects the option with fewer initial cues, predicting a systematic, graded shift in choice probabilities as tied cues are added. By presenting base trials with fixed discriminating cues and systematically adding tied cues, we can cleanly test for configural tied-cue integration (LCA) versus tied-cue invariance (Sequential Sampling).

**Computed schedule:** 11 unique pairs × 8 reps = 88 trials per subject.



## ADVOCATED THEORY
**Description:** Sequential Cue Sampling with Conflict-Modulated Threshold: Decision-makers sample cues sequentially in order of their validities. Instead of integrating all cues simultaneously, they accumulate evidence until it reaches a decision threshold. Crucially, the level of conflict between the options (inversely related to the absolute tally difference) dynamically modulates this threshold. Under low conflict, the threshold is low, leading to fast, non-compensatory decisions (like Take-The-Best). Under high conflict, the threshold increases, enforcing extensive sampling and yielding compensatory behavior. This provides a natural speed-accuracy tradeoff without mathematical artifacts.

**Parameters:**
- theta_0: [0.0, 3.0]
- theta_1: [0.0, 5.0]
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
    
    theta_0 = float(parameters["theta_0"])
    theta_1 = float(parameters["theta_1"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Calculate conflict based on the absolute tally difference of discriminating cues
    N_a = np.sum((a > b).astype(int))
    N_b = np.sum((b > a).astype(int))
    tally_diff = abs(N_a - N_b)
    
    # High tally difference means low conflict; zero tally difference means maximal conflict
    conflict = 1.0 / (1.0 + tally_diff)
    
    # Decision threshold is modulated by the level of conflict
    theta = theta_0 + theta_1 * conflict
    
    # Sequential sampling of cues prioritized by validity
    cue_order = np.argsort(-val, kind="stable")
    
    E = 0.0
    for j in cue_order:
        E += (a[j] - b[j]) * val[j]
        if abs(E) >= theta:
            break
            
    # Softmax choice probability based on accumulated evidence
    z = beta * np.array([E, -E])
    z = z - np.max(z)
    p = np.exp(z) / np.sum(np.exp(z))
    
    # Incorporate lapse rate
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    def get_base_pair(row):\n        a = row['option_a_ratings']\n        b = row['option_b_ratings']\n        disc_a = tuple(1 if a[i]==1 and b[i]==0 else 0 for i in range(len(a)))\n        disc_b = tuple(1 if b[i]==1 and a[i]==0 else 0 for i in range(len(b)))\n        return str((disc_a, disc_b))\n        \n    data['base_pair'] = data.apply(get_base_pair, axis=1)\n    data['tied_count'] = data.apply(lambda row: sum(1 for i in range(len(row['option_a_ratings'])) if row['option_a_ratings'][i]==1 and row['option_b_ratings'][i]==1), axis=1)\n    \n    data['chose_A'] = (data['response'] == 0).astype(float)\n    \n    grouped = data.groupby(['subject_id', 'base_pair', 'tied_count'])['chose_A'].mean().reset_index()\n    \n    diffs = []\n    for (subj, bp), group in grouped.groupby(['subject_id', 'base_pair']):\n        if len(group) > 1:\n            min_tied = group['tied_count'].min()\n            max_tied = group['tied_count'].max()\n            p_min = group.loc[group['tied_count'] == min_tied, 'chose_A'].values[0]\n            p_max = group.loc[group['tied_count'] == max_tied, 'chose_A'].values[0]\n            diffs.append(abs(p_max - p_min))\n            \n    return float(np.mean(diffs)) if diffs else 0.0",
  "rationale": "This metric isolates the core divergence between the advocated Sequential Cue Sampling (SCS) theory and the competing Leaky Competing Accumulator (LCA) theory: the effect of tied cues. SCS evaluates evidence based on the tally difference of *discriminating* cues, ignoring tied cues entirely. Thus, the choice probabilities for a given set of discriminating cues should remain identical regardless of how many tied cues are added. LCA, on the other hand, configuraly scales cue validities based on the total number of cues supporting each option, meaning that adding tied cues systematically alters the choice probabilities. By grouping trials into 'base pairs' (identical discriminating cues) and calculating the absolute difference in the probability of choosing Option A between the trial with the maximum tied cues and the trial with the minimum tied cues, we directly quantify this shift. SCS predicts a difference of 0 (subject only to binomial sampling noise), whereas LCA predicts a large, systematic shift."
}
```

## Usage

```json
{
  "prompt_token_count": 3954,
  "candidates_token_count": 674,
  "total_token_count": 8385
}
```
