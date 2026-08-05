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
**Validities (n_features=6):** [0.95, 0.75, 0.75, 0.75, 0.75, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1]
  trial 4: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 5: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 6: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 7: A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 8: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]

**Rationale:** This design quantitatively dissociates the advocated Sequential Evidence Accumulation theory from the competing Parallel Subjective Evidence Integration theory by exploiting their divergent evidence weighting and stopping rules. The competing theory uses a logarithmic transformation of validities (LLR) and integrates all cues in parallel, falling back to unweighted tallying when evidence is weak or conflicting. Because LLR grows relatively slowly for very high validities compared to the sum of several moderate validities, the competing theory predicts that an option supported by multiple moderate cues will reliably defeat an option supported solely by a single highly valid cue (both in the LLR calculation and the Tallying fallback). The advocated theory, however, evaluates cues sequentially and applies a zero-centered, potentially expansive non-linear transformation (max(val - 0.5)^gamma). This allows a single extremely valid top cue to either trigger early stopping or dominate the accumulated evidence, leading to a strict preference for the high-validity option. By pitting a single ~0.95 validity cue against combinations of ~0.75 validity cues, we cleanly separate the sequential, extreme-weighting predictions of the advocated theory from the parallel, tally-prone predictions of the competing theory.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



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
**Description:** Parallel Subjective Evidence Integration with Tallying Fallback: Decision-makers evaluate all available cues in parallel, accumulating evidence based on subjective log-likelihood ratios. Objective validities are scaled into subjective log-odds. Cues with 0.5 validity provide exactly zero evidence. If the total accumulated subjective evidence exceeds a confidence bound, the decision is made based on this evidence. If the evidence is weak or highly conflicting (falling below the bound), subjects fall back to a simple, unweighted tallying heuristic.

**Parameters:**
- gamma: [0.0, 5.0]
- theta: [0.0, 20.0]
- beta_E: [0.1, 20.0]
- beta_T: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    theta = float(parameters["theta"])
    beta_E = float(parameters["beta_E"])
    beta_T = float(parameters["beta_T"])
    epsilon = float(parameters["epsilon"])
    
    # Clip validities to avoid division by zero or log(0)
    val = np.clip(val, 0.5, 0.999)
    
    # Subjective LLR using scaled log-odds form:
    # LLR = gamma * ln(v / (1 - v))
    llr = gamma * np.log(val / (1.0 - val))
    
    # Accumulate evidence across all cues in parallel
    E = np.sum(llr * (a - b))
    
    # Check if accumulated evidence exceeds the confidence bound
    if abs(E) > theta:
        score_a = E
        score_b = -E
        beta = beta_E
    else:
        # Fallback to simple tallying
        T = np.sum(a - b)
        score_a = T
        score_b = -T
        beta = beta_T
        
    scores = np.array([score_a, score_b])
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_cues = np.array(data['option_a_ratings'].tolist())\n    b_cues = np.array(data['option_b_ratings'].tolist())\n    resp = data['response'].values\n    \n    # The first cue (index 0) is the highest validity cue (0.95).\n    # We calculate how often the subject chose the option that had a '1' for this cue.\n    chose_high_val_cue = ((a_cues[:, 0] == 1) & (resp == 0)) | ((b_cues[:, 0] == 1) & (resp == 1))\n    \n    return float(np.mean(chose_high_val_cue))",
  "rationale": "This metric measures the proportion of trials where the decision-maker chooses the option endorsed by the single most valid cue (validity 0.95). The Sequential Evidence Accumulation theory predicts a high proportion because it evaluates cues in descending order of validity, often resulting in early stopping or heavy weighting on the first cue. Conversely, the Parallel Subjective Evidence Integration theory (with tallying fallback) predicts a low proportion, because it integrates all cues simultaneously; the combined subjective evidence (or simple tally) of several moderately valid cues (validity 0.75) reliably outweighs the single highly valid cue."
}
```

## Usage

```json
{
  "prompt_token_count": 3532,
  "candidates_token_count": 335,
  "total_token_count": 6520
}
```
