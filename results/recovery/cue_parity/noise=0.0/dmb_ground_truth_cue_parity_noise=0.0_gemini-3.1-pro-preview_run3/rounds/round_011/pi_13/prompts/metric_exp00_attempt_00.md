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
**Validities (n_features=10):** [0.95, 0.91, 0.87, 0.83, 0.79, 0.75, 0.71, 0.67, 0.63, 0.59]

**Trial pairs (n=7):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate the pure Take The Best (TTB) model (Competing Theory) from the TTB with Confirmatory Search model (Advocated Theory), we systematically vary the *net* contradiction from lower-validity cues. The Advocated Theory posits that decision-makers compute a net contradiction by subtracting the number of supporting lower-validity cues from the number of opposing lower-validity cues. Across our trials, the highest-validity cue always favors Option A, and we maintain a high number of opposing cues (5 or 6) while parametrically varying the number of supporting cues to cancel them out. Pure TTB strictly ignores all lower-validity cues (both opposing and supporting) and predicts an identical, flat probability of choosing Option A across all trials. In contrast, the Advocated Theory predicts that as supporting cues are removed, the net contradiction increases, eventually crossing a threshold that triggers a sharp, step-function drop in choice confidence.

**Computed schedule:** 7 unique pairs × 13 reps = 91 trials per subject.



## ADVOCATED THEORY
**Description:** Take The Best with Confirmatory Search: Decision-makers initially anchor their choice on the first discriminating cue (Take The Best). However, rather than stopping search entirely, they perform a confirmatory check of the remaining lower-validity cues. If the subsequent cues present strong contradictory evidence (the net number of opposing cues exceeds a specific threshold), their confidence is slightly undermined, leading to a small reduction in the probability of choosing the TTB winner rather than a complete strategy shift. Otherwise, they stick with the initial TTB choice with high confidence.

**Parameters:**
- epsilon: [0.0, 0.2]
- threshold: {2, 3, 4, 5, 6}
- contradiction_confidence: [0.5, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    epsilon = float(parameters["epsilon"])
    threshold = int(float(parameters["threshold"]))
    contradiction_confidence = float(parameters["contradiction_confidence"])
    
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    ttb_winner = -1
    ttb_idx = -1
    for i, j in enumerate(cue_order):
        if a[j] > b[j]:
            ttb_winner = 0
            ttb_idx = i
            break
        elif b[j] > a[j]:
            ttb_winner = 1
            ttb_idx = i
            break
            
    if ttb_winner == -1:
        return np.array([0.5, 0.5])
        
    # Confirmatory search in the remaining cues
    net_contradiction = 0
    for j in cue_order[ttb_idx+1:]:
        if ttb_winner == 0:
            if b[j] > a[j]:
                net_contradiction += 1
            elif a[j] > b[j]:
                net_contradiction -= 1
        else:
            if a[j] > b[j]:
                net_contradiction += 1
            elif b[j] > a[j]:
                net_contradiction -= 1
                
    if net_contradiction >= threshold:
        p_winner = contradiction_confidence
    else:
        p_winner = 1.0 - epsilon
        
    p = np.zeros(2)
    p[ttb_winner] = p_winner
    p[1 - ttb_winner] = 1.0 - p_winner
    
    return p
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n    \n    def get_net(row):\n        a = row['option_a_ratings']\n        b = row['option_b_ratings']\n        net = 0\n        for j in range(1, len(a)):\n            if b[j] > a[j]:\n                net += 1\n            elif a[j] > b[j]:\n                net -= 1\n        return net\n        \n    net_c = data.apply(get_net, axis=1)\n    chose_a = 1.0 - data['response']\n    \n    low_mask = net_c <= 2\n    high_mask = net_c >= 5\n    \n    if low_mask.sum() == 0 or high_mask.sum() == 0:\n        return 0.0\n        \n    return float(chose_a[low_mask].mean() - chose_a[high_mask].mean())",
  "rationale": "The metric computes the difference in the probability of choosing Option A between trials with a low net contradiction (<= 2) and trials with a high net contradiction (>= 5). According to the Competing Theory (pure Take The Best), this difference should be approximately zero, as Option A wins on the highest validity cue in all trials, and lower validity cues are ignored. According to the Advocated Theory (TTB with Confirmatory Search), choice probability for Option A will drop significantly in high contradiction trials because the net contradiction crosses the threshold, leading to a positive difference."
}
```

## Usage

```json
{
  "prompt_token_count": 4037,
  "candidates_token_count": 394,
  "total_token_count": 6782
}
```
