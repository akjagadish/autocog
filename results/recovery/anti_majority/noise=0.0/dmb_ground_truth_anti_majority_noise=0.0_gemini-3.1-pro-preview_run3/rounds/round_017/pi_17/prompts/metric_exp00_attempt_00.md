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
**Validities (n_features=5):** [0.95, 0.85, 0.7, 0.6, 0.55]

**Trial pairs (n=9):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  trial 4: A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 7: A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 8: A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 9: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** This design quantitatively dissociates the Lateral Inhibition Evidence Accumulator (Advocated) from the Salience-Weighted Evidence Accumulator (Competing) by exploiting the Advocated model's 'perverse' self-dilution effect. In the Competing model, adding unique positive cues to an option strictly increases its accumulated evidence, thereby monotonically increasing its predicted choice probability. In stark contrast, the Advocated model employs divisive normalization before evidence integration. Adding lower-validity secondary cues to an option dilutes the perceived value of its primary high-validity cue (self-inhibition). If this self-dilution parameter is sufficiently strong, adding secondary cues can paradoxically decrease the option's overall accumulated evidence, leading to a non-monotonic drop in choice probability. By starting with a baseline trial that pits a single strong cue against a single medium cue, and systematically adding weak secondary cues to the strong option across subsequent trials, we force the Competing model to predict a strictly increasing preference for the strong option, while the Advocated model can predict a perverse reversal.

**Computed schedule:** 9 unique pairs × 10 reps = 90 trials per subject.



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
**Description:** Salience-Weighted Evidence Accumulator: Decision-makers process options by accumulating evidence from their features, but features shared between options receive reduced attentional weight (salience down-weighting) rather than being perfectly cancelled. This partially discounted shared evidence is added to the unique evidence, and the total accumulated evidence for each option undergoes a mild non-linear squashing (e.g., a parameterized power function). This compressive non-linearity ensures that adding shared features subtly reduces the discriminability between options, naturally predicting the small but non-zero choice shifts observed when common cues are introduced.

**Parameters:**
- theta: [0.1, 5.0]
- phi: [0.0, 1.0]
- gamma: [0.1, 2.0]
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
    phi = float(parameters["phi"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale validities to subjective weights
    w = np.maximum(0.0, val - 0.5) ** theta
    
    # Identify unique and shared positive features
    shared = (a == 1) & (b == 1)
    unique_a = (a == 1) & (b == 0)
    unique_b = (a == 0) & (b == 1)
    
    # Accumulate evidence with salience down-weighting for shared features
    ev_a = np.sum(w[unique_a]) + phi * np.sum(w[shared])
    ev_b = np.sum(w[unique_b]) + phi * np.sum(w[shared])
    
    # Mild non-linear squashing (power function)
    sq_a = ev_a ** gamma if ev_a > 0 else 0.0
    sq_b = ev_b ** gamma if ev_b > 0 else 0.0
    
    # Convert to probabilities via softmax
    z = beta * np.array([sq_a, sq_b])
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.sum(p)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(i)) for i in x]))\n    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(i)) for i in x]))\n    \n    t1 = (a_str == '10000') & (b_str == '01000')\n    t4 = (a_str == '10111') & (b_str == '01000')\n    \n    if t1.sum() == 0 or t4.sum() == 0:\n        return 0.0\n        \n    p_a_t1 = 1.0 - data.loc[t1, 'response'].mean()\n    p_a_t4 = 1.0 - data.loc[t4, 'response'].mean()\n    \n    return float(p_a_t4 - p_a_t1)",
  "rationale": "This metric calculates the difference in the probability of choosing Option A between Trial 4 (where A has one strong cue and three weak cues) and Trial 1 (where A has only the single strong cue), against a constant Option B (one medium cue). The Competing model, which accumulates evidence additively, predicts a positive difference because Option A has more positive features in Trial 4. The Advocated model predicts a negative difference because the weak secondary features in Option A will dilute the perceived value of its primary strong feature via lateral inhibition, reducing its overall choice probability."
}
```

## Usage

```json
{
  "prompt_token_count": 3708,
  "candidates_token_count": 395,
  "total_token_count": 5944
}
```
