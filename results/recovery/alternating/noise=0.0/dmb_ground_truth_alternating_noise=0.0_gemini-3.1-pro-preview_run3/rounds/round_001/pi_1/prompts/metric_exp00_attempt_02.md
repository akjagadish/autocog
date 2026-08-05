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

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 7: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 8: A=[1, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Rationale:** This design pits Take The Best (TTB) against the Weighted Additive (WADD) model by systematically varying the compensatory evidence from lower-validity cues while keeping the highest-discriminating cue constant. TTB is a non-compensatory, 'one-reason' heuristic: it decides solely based on the highest-validity cue that discriminates between the options, completely ignoring all other cues. Consequently, TTB predicts identical choice probabilities across all trials where the same cue determines the choice. In contrast, WADD computes a weighted sum of all features. By presenting trials where the highest-validity cue favors one option, but an increasing number of lower-validity cues oppose it, WADD predicts a graded change in choice probabilities (even flipping the choice if the gamma parameter allows for compensatory weighting). TTB predicts a completely flat probability profile across these same trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
**Description:** Weighted Additive (WADD) model with non-linear validity scaling and extended lapse rate: Decision makers evaluate options by computing a weighted sum of their feature values. The weights are formed by centering the cue validities (val - 0.5) and raising their absolute values to a power gamma (preserving sign), which allows the model to interpolate between equal-weighting, proportional weighting, and non-compensatory behavior without extreme log-odds scaling. To account for the high degree of noise or chance-level behavior observed in specific paradigms, the lapse rate (epsilon) can range up to 1.0, and beta down to 0.0, allowing the model to capture completely random guessing natively.

**Parameters:**
- beta: [0.0, 20.0]
- epsilon: [0.0, 1.0]
- gamma: [0.0, 10.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Center validities so 0.5 gives 0 weight, avoiding discrete parameter
    base_w = val - 0.5
        
    # Signed exponentiation to handle negative base_w with fractional gamma safely
    w = np.sign(base_w) * (np.abs(base_w) ** gamma)
        
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    return (1.0 - epsilon) * p_core + epsilon * 0.5
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
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
[0] rationale: This metric calculates the difference in the probability of choosing Option A between Trial 3 and Trial 1. In both trials, the highest-validity cue (feature 0) favors Option A. According to Take The Best (TTB), this single cue strictly determines the decision, so the probability of choosing A should be identical across both trials (metric ≈ 0). In contrast, the Weighted Additive (WADD) model integrates all cues. In Trial 1, the remaining four cues strongly oppose Option A, whereas in Trial 3, they strongly support Option A. Therefore, WADD predicts a significantly higher probability of choosing A in Trial 3 compared to Trial 1, resulting in a positive metric value.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t1_mask = data.apply(lambda row: row['option_a_ratings'][0] == 1 and row['option_b_ratings'][0] == 0 and sum(row['option_a_ratings']) == 1 and sum(row['option_b_ratings']) == 4, axis=1)
    t3_mask = data.apply(lambda row: row['option_a_ratings'][0] == 1 and row['option_b_ratings'][0] == 0 and sum(row['option_a_ratings']) == 4 and sum(row['option_b_ratings']) == 1, axis=1)
    
    p_a_t1 = 1.0 - data[t1_mask]['response'].mean()
    p_a_t3 = 1.0 - data[t3_mask]['response'].mean()
    
    if pd.isna(p_a_t1) or pd.isna(p_a_t3):
        return 0.0
        
    return float(p_a_t3 - p_a_t1)
outcome: self_sim=0.0133 (var=0.0162) adversary_sim=0.0950 (var=0.0700) welch_t=-1.391 p=0.1732 (N=25, alpha=0.01) -> reject

[1] rationale: This metric isolates Trials 7 and 8, where the highest validity cue (Cue 0) is tied, making Cue 1 the deciding factor for Take The Best (TTB). TTB predicts identical choice probabilities for the option favored by Cue 1 across both trials, yielding a metric of 0. However, in Trial 7, the lower validity cues strongly oppose the Cue 1 favored option, whereas in Trial 8, they strongly support it. The Weighted Additive (WADD) model integrates these lower validity cues and therefore predicts a significantly higher probability of choosing the Cue 1 favored option in Trial 8 compared to Trial 7, resulting in a large positive metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t7_favored_chosen = []
    t8_favored_chosen = []
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 7: Cue 0 tied, Cue 1 favors the (1,1,0,0,0) option. Cues 2,3,4 oppose.
        if a == (1, 1, 0, 0, 0) and b == (1, 0, 1, 1, 1):
            t7_favored_chosen.append(1 if resp == 0 else 0)
        elif b == (1, 1, 0, 0, 0) and a == (1, 0, 1, 1, 1):
            t7_favored_chosen.append(1 if resp == 1 else 0)
            
        # Trial 8: Cue 0 tied, Cue 1 favors the (1,1,0,1,1) option. Cues 3,4 support, Cue 2 opposes.
        elif a == (1, 1, 0, 1, 1) and b == (1, 0, 1, 0, 0):
            t8_favored_chosen.append(1 if resp == 0 else 0)
        elif b == (1, 1, 0, 1, 1) and a == (1, 0, 1, 0, 0):
            t8_favored_chosen.append(1 if resp == 1 else 0)
            
    if not t7_favored_chosen or not t8_favored_chosen:
        return 0.0
        
    p7 = sum(t7_favored_chosen) / len(t7_favored_chosen)
    p8 = sum(t8_favored_chosen) / len(t8_favored_chosen)
    
    return float(p8 - p7)
outcome: self_sim=0.0100 (var=0.0255) adversary_sim=0.0100 (var=0.0710) welch_t=+0.000 p=1 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    supported_adherence = []\n    opposed_adherence = []\n    \n    for _, row in data.iterrows():\n        a = row['option_a_ratings']\n        b = row['option_b_ratings']\n        \n        # Find the first discriminating cue to determine the TTB prediction\n        ttb_favors_A = None\n        for i in range(len(a)):\n            if a[i] > b[i]:\n                ttb_favors_A = True\n                break\n            elif b[i] > a[i]:\n                ttb_favors_A = False\n                break\n                \n        if ttb_favors_A is None:\n            continue\n            \n        sum_a = sum(a)\n        sum_b = sum(b)\n        chose_A = (row['response'] == 0)\n        \n        if ttb_favors_A:\n            adherence = 1 if chose_A else 0\n            if sum_a > sum_b:\n                supported_adherence.append(adherence)\n            elif sum_a < sum_b:\n                opposed_adherence.append(adherence)\n        else:\n            adherence = 1 if not chose_A else 0\n            if sum_b > sum_a:\n                supported_adherence.append(adherence)\n            elif sum_b < sum_a:\n                opposed_adherence.append(adherence)\n                \n    if not supported_adherence or not opposed_adherence:\n        return 0.0\n        \n    return float(np.mean(supported_adherence) - np.mean(opposed_adherence))",
  "rationale": "This metric calculates the difference in the rate at which subjects choose the option favored by Take The Best (TTB) when the unweighted sum of features SUPPORTS the TTB choice versus when the unweighted sum OPPOSES it. For TTB, the decision is strictly based on the first discriminating cue, so the adherence rate should be identical in both conditions (metric \u2248 0). For the Weighted Additive (WADD) model, the compensatory influence of the lower-validity cues will pull choices away from the TTB prediction when the sum opposes it, and push choices toward the TTB prediction when the sum supports it, leading to a consistently positive metric. By aggregating over 48 trials per condition, this metric minimizes per-subject variance."
}
```

## Usage

```json
{
  "prompt_token_count": 4910,
  "candidates_token_count": 607,
  "total_token_count": 9009
}
```
