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
**Validities (n_features=4):** [0.95, 0.68, 0.88, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 1, 0]  B=[0, 1, 1, 0]
  trial 2: A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0]  B=[0, 1, 0, 1]
  trial 4: A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  trial 5: A=[1, 0, 1, 1]  B=[1, 1, 1, 1]
  trial 6: A=[0, 0, 1, 0]  B=[1, 1, 0, 1]
  trial 7: A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 8: A=[1, 0, 0, 1]  B=[1, 0, 1, 1]
  trial 9: A=[1, 0, 0, 1]  B=[0, 0, 1, 1]
  trial 10: A=[0, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 11: A=[0, 0, 1, 0]  B=[0, 0, 1, 1]
  trial 12: A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  trial 13: A=[1, 0, 0, 0]  B=[0, 0, 1, 0]
  trial 14: A=[1, 1, 0, 0]  B=[0, 0, 0, 1]
  trial 15: A=[1, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 16: A=[1, 0, 1, 0]  B=[1, 1, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Take-The-Best (PTTB): People evaluate cues sequentially to make decisions, but the sequence is stochastic rather than strictly deterministic. The probability of sampling a cue is proportional to an exponential function of its subjective validity. When sampling without replacement, the probability that the first discriminating cue encountered is cue $i$ is exactly proportional to its weight among all discriminating cues. This model bridges Tallying (when temperature is high/gamma is 0, all cues are equally likely to be the deciding one) and deterministic Take-The-Best (when gamma is large, the highest validity cue almost always determines the choice), providing a nuanced non-compensatory heuristic.

**Parameters:**
- gamma: [0.0, 35.0]
- epsilon: [0.0, 0.3]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("PTTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Identify cues that discriminate between the two options
    discrim = (a != b)
    
    if not np.any(discrim):
        # No discriminating cues, guess uniformly
        p_core = np.array([0.5, 0.5])
    else:
        # The probability of a discriminating cue being the first one found
        # is proportional to exp(gamma * validity).
        val_discrim = validities[discrim]
        # Max-subtraction for numerical stability
        z = gamma * (val_discrim - np.max(val_discrim))
        w = np.exp(z)
        w = w / np.sum(w)
        
        # The probability of choosing an option is the sum of the probabilities
        # of the discriminating cues that favor it.
        p_a = np.sum(w[a[discrim] > b[discrim]])
        p_b = np.sum(w[b[discrim] > a[discrim]])
        
        p_core = np.array([p_a, p_b])
        
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Take-The-Best (TTB) heuristic: People make decisions between multi-attribute options by ranking features according to their validities and comparing the options lexicographically. The decision-maker examines the feature with the highest validity first; if one option is strictly better on that feature, it is chosen immediately and no further features are considered. If there is a tie, they move to the second most valid feature, and so on. This non-compensatory, one-reason decision making process implies that a single highly valid cue can completely determine the choice, overriding any number of smaller advantages on less valid cues.

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
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    choice = None
    for idx in order:
        if a[idx] > b[idx]:
            choice = 0
            break
        elif b[idx] > a[idx]:
            choice = 1
            break
            
    epsilon = float(parameters["epsilon"])
    n_opts = 2
    
    if choice is None:
        p_core = np.ones(n_opts) / n_opts
    else:
        p_core = np.zeros(n_opts)
        p_core[choice] = 1.0
        
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_ratings = np.vstack(data['option_a_ratings'].values)\n    b_ratings = np.vstack(data['option_b_ratings'].values)\n    \n    # Cue 0 has the highest validity (0.95), Cue 2 has the second highest (0.88)\n    cue0_diff = a_ratings[:, 0] - b_ratings[:, 0]\n    cue2_diff = a_ratings[:, 2] - b_ratings[:, 2]\n    \n    # Find trials where Cue 0 and Cue 2 disagree\n    disagree = (cue0_diff != 0) & (cue2_diff != 0) & (cue0_diff != cue2_diff)\n    \n    if np.sum(disagree) == 0:\n        return 0.5\n        \n    # TTB predicts the option favored by Cue 0 (the highest validity cue)\n    ttb_pred = (cue0_diff[disagree] < 0).astype(int)\n    resp = data['response'].values[disagree]\n    \n    return float(np.mean(ttb_pred == resp))",
  "rationale": "This metric calculates the proportion of times a subject's choice aligns with the highest validity cue (Cue 0, validity 0.95) specifically on trials where it is opposed by the second highest validity cue (Cue 2, validity 0.88). TTB strictly follows the highest validity cue, so it will almost always choose the option favored by Cue 0 on these trials. PTTB, however, samples cues probabilistically based on their validities. Because 0.95 and 0.88 are very close, PTTB has a substantial probability of sampling Cue 2 first and thus choosing the option favored by Cue 2, leading to a visibly lower alignment with Cue 0."
}
```

## Usage

```json
{
  "prompt_token_count": 3363,
  "candidates_token_count": 460,
  "total_token_count": 7329
}
```
