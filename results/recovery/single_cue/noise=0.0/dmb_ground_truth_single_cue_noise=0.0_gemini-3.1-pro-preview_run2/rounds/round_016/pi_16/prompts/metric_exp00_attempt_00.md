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
**Validities (n_features=5):** [0.55, 0.9, 0.85, 0.8, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1]

**Rationale:** To quantitatively dissociate Recency-Biased Cue Overweighting (Advocated Theory) from Serial Position Dual-Overweighting (Competing Theory), we exploit the critical difference in their treatment of the first cue. The Advocated Theory treats the first cue exactly like any middle cue, weighting it strictly according to its stated validity (scaled by gamma). The Competing Theory, however, posits an independent, unnormalized 'primacy_weight' for the first cue, allowing it to dominate choices regardless of its stated validity. We use a 5-feature design where the first and last cues have low validities, while the middle cues have high validities. The trial set includes 'primacy conflicts' where the first cue favors one option while the high-validity middle cues favor the alternative (Advocated predicts middle cues win; Competing allows the first cue to win). We also include direct conflicts between the first and last cues to contrast the independent primacy and recency weights in the Competing Theory against the isolated recency weight in the Advocated Theory.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Recency-Biased Cue Overweighting: Decision-makers evaluate options by attempting to integrate all available features, but due to visual recency and short-term memory effects, the final feature in the sequence is disproportionately salient. While the first N-1 features are weighted according to their stated validities (subject to non-linear scaling), the final feature is assigned an independent, often much larger weight. This mechanism explains boundary cases where subjects' choices are driven by the nominally least valid cue, effectively overriding both compensatory tallying and the expected Take-The-Best hierarchy.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- recency_weight: [0.0, 10.0]
- gamma: [0.0, 5.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    recency_weight = float(parameters["recency_weight"])
    gamma = float(parameters["gamma"])
    
    # Scale validities for integration
    w = val ** gamma
    # Overweight the final feature due to recency
    w[-1] = recency_weight
    
    # Compute evidence for each option
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    score_a = np.sum(w * a_wins)
    score_b = np.sum(w * b_wins)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
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


## COMPETING THEORY
**Description:** Serial Position Dual-Overweighting: Decision-makers evaluate options by integrating features, but due to memory and attention constraints at the sequence boundaries, both the first (primacy) and the last (recency) cues are assigned independent, disproportionately large weights. Unlike models that normalize attention or weights, these boundary weights are unnormalized, allowing them to independently dominate choice when necessary. Middle cues are weighted by their stated validities, scaled non-linearly. This mechanism captures both extreme primacy and extreme recency effects without the dampening effect of weight normalization.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 5.0]
- primacy_weight: [0.0, 10.0]
- recency_weight: [0.0, 10.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")

    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    primacy_weight = float(parameters["primacy_weight"])
    recency_weight = float(parameters["recency_weight"])
    gamma = float(parameters["gamma"])
    
    # Scale validities for integration
    w = val ** gamma
    
    # Overweight the first and final features due to primacy and recency
    if len(w) > 1:
        w[0] = primacy_weight
        w[-1] = recency_weight
    elif len(w) == 1:
        w[0] = primacy_weight + recency_weight
        
    # Compute evidence for each option
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    score_a = np.sum(w * a_wins)
    score_b = np.sum(w * b_wins)
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Identify trials where Option A is supported ONLY by the first cue\n    # and Option B is supported either by strong middle cues (Trial 1) \n    # or the final cue (Trial 3).\n    is_A_10000 = data['option_a_ratings'].apply(lambda x: list(x) == [1, 0, 0, 0, 0])\n    is_B_01100 = data['option_b_ratings'].apply(lambda x: list(x) == [0, 1, 1, 0, 0])\n    is_B_00001 = data['option_b_ratings'].apply(lambda x: list(x) == [0, 0, 0, 0, 1])\n    \n    mask = is_A_10000 & (is_B_01100 | is_B_00001)\n    \n    if mask.sum() == 0:\n        return 0.0\n        \n    # Return the proportion of times Option A was chosen (response == 0)\n    return float((data.loc[mask, 'response'] == 0).mean())\n",
  "rationale": "This metric calculates the choice proportion of an option supported solely by the first cue (Option A in Trials 1 and 3) when pitted against options supported by high-validity middle cues or the final cue. Under the Advocated Theory (Recency-Biased Cue Overweighting), the first cue is evaluated using its low stated validity, meaning Option A will almost never be chosen. Under the Competing Theory (Serial Position Dual-Overweighting), the first cue is assigned an independent, unnormalized 'primacy_weight' which allows Option A to be chosen at a significantly higher rate."
}
```

## Usage

```json
{
  "prompt_token_count": 3444,
  "candidates_token_count": 439,
  "total_token_count": 6648
}
```
