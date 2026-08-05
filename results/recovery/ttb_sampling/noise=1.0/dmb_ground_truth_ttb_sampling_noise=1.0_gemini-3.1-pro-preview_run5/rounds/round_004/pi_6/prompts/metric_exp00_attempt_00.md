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
**Validities (n_features=5):** [0.98, 0.92, 0.86, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  trial 6: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1]
  trial 7: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 8: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** This design contrasts a Mixture of Take-The-Best and Tallying against Probabilistic Cue Sampling (PCS) by exploiting trials where Tallying is indifferent but TTB and PCS strongly disagree. We use a 5-cue design where the sum of the sampling probabilities for the second and third cues exceeds the sum for the first and fourth (or fifth) cues under PCS's softmax, even at maximum gamma. In these critical trials, Option A possesses the highest-validity cue and one low-validity cue, while Option B possesses the second and third highest-validity cues. Because the number of winning cues is tied, Tallying is perfectly indifferent. TTB, however, deterministically favors Option A due to the top cue. Consequently, any mixture of TTB and Tallying must predict a preference for Option A (P(A) >= 0.5). In contrast, PCS samples cues proportionally to a softmax distribution; because the combined probability of the second and third cues is greater, PCS systematically predicts a preference for Option B (P(A) < 0.5). This provides a strict, parameter-free qualitative dissociation.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Mixture of Simple Heuristics (Take-The-Best and Tallying) with Softened Determinism

**Parameters:**
- beta: [0.0, 2.0]
- epsilon: [0.0, 1.0]
- w_ttb: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    w_ttb = float(parameters["w_ttb"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # TTB Heuristic: find the first discriminating cue in descending order of validity
    order = np.argsort(validities)[::-1]
    ttb_diff = 0.0
    for idx in order:
        if stim[0, idx] != stim[1, idx]:
            ttb_diff = stim[0, idx] - stim[1, idx]
            break

    # Tallying Heuristic: unweighted count of winning features
    a_wins = np.sum(stim[0] > stim[1])
    b_wins = np.sum(stim[1] > stim[0])
    tally_diff = a_wins - b_wins

    # Translate differences into probabilities using softmax (decision noise)
    # Bound the differences to prevent overflow
    z_ttb = np.clip(beta * ttb_diff, -100, 100)
    p_ttb_A = 1.0 / (1.0 + np.exp(-z_ttb))
    
    z_tally = np.clip(beta * tally_diff, -100, 100)
    p_tally_A = 1.0 / (1.0 + np.exp(-z_tally))

    # Mixture of the two heuristics
    p_core_A = w_ttb * p_ttb_A + (1.0 - w_ttb) * p_tally_A
    p_core_B = 1.0 - p_core_A

    # Apply uniform lapse rate
    p_final = (1.0 - epsilon) * np.array([p_core_A, p_core_B]) + epsilon * np.array([0.5, 0.5])

    return p_final
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Probabilistic Cue Sampling with Softmax Validities: Decision-makers evaluate options by sampling a single feature on each trial. The probability of sampling a feature is determined by a softmax function over the objective validities scaled by a sensitivity parameter gamma. The option that is superior on the sampled feature is chosen; ties result in guessing. This predicts intermediate choice proportions when validities are mixed, avoiding deterministic winner-takes-all behavior.

**Parameters:**
- gamma: [0.0, 2.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])

    # Compute subjective validities using softmax for sampling probabilities
    z = gamma * validities
    z = z - np.max(z)  # numerical stability
    e = np.exp(z)
    p_feat = e / np.sum(e)

    a, b = stim[0], stim[1]

    # Identify wins and ties for each feature
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    ties = (a == b).astype(float)

    # Analytically marginalize over the sampled feature:
    # P(Choose A) = sum_i P(sample i) * P(Choose A | sample i)
    # P(Choose A | sample i) = 1 if a_i > b_i, 0 if b_i > a_i, 0.5 if a_i == b_i
    p_A = np.sum(p_feat * (a_wins + 0.5 * ties))
    p_B = np.sum(p_feat * (b_wins + 0.5 * ties))

    p_core = np.array([p_A, p_B])

    # Apply uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_has_cue0 = data['option_a_ratings'].apply(lambda x: x[0] == 1)\n    b_has_cue0 = data['option_b_ratings'].apply(lambda x: x[0] == 1)\n    \n    chose_a = data['response'] == 0\n    chose_b = data['response'] == 1\n    \n    chose_cue0 = (chose_a & a_has_cue0) | (chose_b & b_has_cue0)\n    return float(chose_cue0.mean())",
  "rationale": "The experimental design pits the highest validity cue (Cue 0) against the combination of the second and third highest validity cues (Cues 1 and 2), while Tallying remains indifferent. The Mixture of TTB and Tallying (Advocated Theory) will systematically favor the option with Cue 0 because TTB deterministically chooses it, yielding a choice proportion > 0.5. Conversely, Probabilistic Cue Sampling (Competing Theory) samples cues proportionally to their softmax validities; since the sum of probabilities for Cues 1 and 2 exceeds that of Cue 0 and any remaining low-validity cue, PCS systematically favors the option without Cue 0, yielding a choice proportion < 0.5. Measuring the overall proportion of choices for the option with Cue 0 perfectly discriminates the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 3521,
  "candidates_token_count": 342,
  "total_token_count": 5914
}
```
