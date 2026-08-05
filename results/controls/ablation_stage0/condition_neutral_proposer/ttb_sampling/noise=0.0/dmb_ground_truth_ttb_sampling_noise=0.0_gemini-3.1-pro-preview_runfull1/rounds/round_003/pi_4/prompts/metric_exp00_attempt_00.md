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
  trial 3: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 4: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** To distinguish deterministic Take The Best (TTB) from Probabilistic Take The Best (PTTB), we exploit the stochastic stopping rule of PTTB. TTB stops strictly at the first discriminating cue, whereas PTTB has a probability (1 - gamma) of skipping the first discriminating cue and evaluating subsequent ones. We design trials where the first discriminating cue strongly favors one option, but all subsequent discriminating cues favor the other option. TTB will almost exclusively choose the option favored by the first cue, whereas PTTB will show a systematic shift toward the opposite option due to the probability of skipping the first cue and stopping on later cues. We also include control trials where multiple top cues favor the same option, where both models should make identical predictions.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Take The Best (PTTB) posits that decision-makers search through features sequentially in order of their subjective validities. However, unlike deterministic Take The Best, the stopping rule is stochastic: upon finding a feature that discriminates between the two options, the decision-maker stops and bases their choice on that feature with probability gamma. If they do not stop, they continue to the next feature. If all features are evaluated without a decision, they guess randomly. This model captures the primarily non-compensatory nature of human decision-making while allowing for occasional deviations where lower-validity cues might influence the choice.

**Parameters:**
- gamma: [0.8, 1.0]
- epsilon: [0.0, 0.2]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"PTTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort indices by validity in descending order
    order = np.argsort(validities)[::-1]
    
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    p_choice = np.array([0.0, 0.0])
    p_reach = 1.0
    
    # Search for discriminating cues with stochastic stopping
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            p_stop = gamma
            p_choice += p_reach * p_stop * np.array([1.0, 0.0])
            p_reach *= (1.0 - p_stop)
        elif stim[1, idx] > stim[0, idx]:
            p_stop = gamma
            p_choice += p_reach * p_stop * np.array([0.0, 1.0])
            p_reach *= (1.0 - p_stop)
            
    # If no decision is made after evaluating all cues, guess
    p_choice += p_reach * np.array([0.5, 0.5])
        
    # Apply uniform lapse rate
    return (1.0 - epsilon) * p_choice + epsilon * np.array([0.5, 0.5])
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
**Description:** Take The Best (TTB) posits that decision-makers consider features sequentially in order of their subjective validities and stop searching as soon as they find a feature that discriminates between the two options. The choice is then based entirely on this single cue, providing a non-compensatory heuristic alternative to compensatory models like WADD.

**Parameters:**
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort indices by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a_wins = False
    b_wins = False
    
    # Search for the first discriminating cue
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            a_wins = True
            break
        elif stim[1, idx] > stim[0, idx]:
            b_wins = True
            break
            
    if a_wins:
        p_core = np.array([1.0, 0.0])
    elif b_wins:
        p_core = np.array([0.0, 1.0])
    else:
        p_core = np.array([0.5, 0.5])
        
    epsilon = float(parameters["epsilon"])
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    agree_match = []\n    disagree_match = []\n    \n    for _, row in data.iterrows():\n        a = row['option_a_ratings']\n        b = row['option_b_ratings']\n        resp = row['response']\n        \n        disc_cues = []\n        for i in range(5):\n            if a[i] > b[i]:\n                disc_cues.append(0)\n            elif b[i] > a[i]:\n                disc_cues.append(1)\n                \n        if len(disc_cues) >= 2:\n            first = disc_cues[0]\n            second = disc_cues[1]\n            \n            chose_first = (resp == first)\n            \n            if first == second:\n                agree_match.append(chose_first)\n            else:\n                disagree_match.append(chose_first)\n                \n    if len(agree_match) == 0 or len(disagree_match) == 0:\n        return 0.0\n        \n    return float(np.mean(agree_match) - np.mean(disagree_match))\n",
  "rationale": "This metric calculates the difference in the rate at which subjects choose the option favored by the *first* discriminating cue, conditional on whether the *second* discriminating cue agrees or disagrees with the first. For the deterministic Take The Best (TTB) model, the decision is strictly determined by the first discriminating cue, so the agreement of the second cue has no effect (the difference should be zero in expectation). For Probabilistic Take The Best (PTTB), subjects sometimes skip the first cue and rely on the second; thus, when the second cue agrees, they are more likely to make the same choice as the first cue, and when it disagrees, they are more likely to make the opposite choice (yielding a strongly positive difference)."
}
```

## Usage

```json
{
  "prompt_token_count": 3170,
  "candidates_token_count": 478,
  "total_token_count": 7403
}
```
