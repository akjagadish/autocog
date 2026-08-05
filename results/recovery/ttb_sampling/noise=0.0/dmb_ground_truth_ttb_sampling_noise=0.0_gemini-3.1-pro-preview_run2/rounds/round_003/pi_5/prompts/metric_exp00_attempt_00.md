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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  trial 5: A=[0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** To quantitatively dissociate pure Take The Best (TTB) from TTB with Validity-Dependent Confidence (TTB-VDC), we must manipulate the validity of the highest discriminating cue across trials. Pure TTB assumes a constant lapse rate regardless of which cue breaks the tie, predicting a flat probability of choosing the TTB-favored option across these trials. In contrast, TTB-VDC posits that confidence scales with the validity of the discriminating cue, predicting that the lapse rate will parametrically increase (and accuracy decrease) as the decision is forced onto lower-validity cues. By systematically tying the upper cues and allowing progressively lower-validity cues to discriminate, we can observe whether choice consistency degrades in a validity-dependent manner.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Take The Best with Validity-Dependent Confidence: Decision makers use a non-compensatory, lexicographic strategy (Take The Best) to choose between options. However, their execution of this strategy is noisy, and the degree of noise (lapse rate) depends on the validity of the discriminating cue. When options are discriminated by a highly valid cue, confidence is high and the lapse rate is low. When the discriminating cue has low validity, confidence is lower, leading to a higher probability of guessing or lapsing. This maintains the non-compensatory nature of TTB while naturally explaining variations in performance across different trial structures.

**Parameters:**
- epsilon: [0.0, 0.5]
- gamma: [0.0, 2.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues in descending order of validity
    order = np.argsort(-validities, kind='stable')
    
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    
    p_core = np.array([0.5, 0.5])
    v_discriminating = None
    
    # Lexicographic search
    for i in order:
        if a[i] > b[i]:
            p_core = np.array([1.0, 0.0])
            v_discriminating = validities[i]
            break
        elif b[i] > a[i]:
            p_core = np.array([0.0, 1.0])
            v_discriminating = validities[i]
            break
            
    if v_discriminating is not None:
        # Lapse rate increases as validity of the discriminating cue decreases
        lapse = epsilon + gamma * (1.0 - v_discriminating)
        lapse = max(0.0, min(1.0, lapse))
        return (1.0 - lapse) * p_core + lapse * np.array([0.5, 0.5])
    else:
        # Guess if no cues discriminate
        return np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Take The Best (TTB) Heuristic: Decision makers use a non-compensatory, lexicographic strategy to choose between options. They search through cues in descending order of subjective validity (or informativeness). The first cue that discriminates between the two options strictly determines the choice, and all remaining lower-validity cues are ignored. If no cues discriminate, the decision maker guesses. Response noise is modeled as a uniform lapse.

**Parameters:**
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues in descending order of validity. 
    # We use a stable sort so that if validities are tied, left-to-right order is preserved.
    order = np.argsort(-validities, kind='stable')
    
    # Default to guessing if no cues discriminate
    p_core = np.array([0.5, 0.5])
    
    # Lexicographic search
    for i in order:
        if a[i] > b[i]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[i] > a[i]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n    \n    def get_disc_cue(row):\n        a = row['option_a_ratings']\n        b = row['option_b_ratings']\n        for i in range(len(a)):\n            if a[i] != b[i]:\n                return i\n        return -1\n\n    cues = data.apply(get_disc_cue, axis=1)\n    # In all designed trials, option A is favored by the highest discriminating cue.\n    # Thus, '1 - response' computes the TTB accuracy (1 if chose A, 0 if chose B).\n    acc = 1.0 - data['response']\n    \n    high_val = acc[cues.isin([0, 1])]\n    low_val = acc[cues.isin([3, 4])]\n    \n    if len(high_val) == 0 or len(low_val) == 0:\n        return 0.0\n        \n    return float(high_val.mean() - low_val.mean())",
  "rationale": "The experimental design systematically varies the index of the highest discriminating cue across trials, while keeping option A as the TTB-favored choice. The advocated theory (TTB-VDC) posits that as the decision relies on lower-validity cues, confidence decreases and the lapse rate increases. Therefore, accuracy relative to the TTB prediction should drop for trials where the discriminating cue is low-validity (cues 3 and 4) compared to high-validity (cues 0 and 1). The competing theory (pure TTB) assumes a constant lapse rate across all trials, predicting a difference of approximately zero. This metric directly calculates the difference in choice consistency between high-validity and low-validity discriminating cues."
}
```

## Usage

```json
{
  "prompt_token_count": 3174,
  "candidates_token_count": 446,
  "total_token_count": 6191
}
```
