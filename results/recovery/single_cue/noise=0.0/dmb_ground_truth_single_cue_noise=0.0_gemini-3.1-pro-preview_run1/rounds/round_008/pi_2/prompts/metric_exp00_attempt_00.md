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

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 5: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 7: A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  trial 8: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  trial 9: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 10: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Pure Tallying) from the Competing Theory (Left-to-Right TTB with Thresholded Compensatory Tallying and Recency Tie-Breaker), we focus on two critical types of trials. First, we use 'Tally Tie' trials where both options have an equal number of feature wins. Pure Tallying predicts an exact 50/50 guess on these trials. The Competing Theory, however, will either make a deterministic choice based on the first discriminating feature (TTB) if the opposing wins are below threshold, or use a recency-weighted tie-breaker if above threshold. By carefully arranging the winning features, we pit the TTB prediction against the recency prediction, ensuring the Competing Theory always diverges from 50/50. Second, we include 'Compensatory' trials where the option with fewer wins is favored by the first discriminating cue. Pure Tallying strictly favors the option with more wins, while the Competing Theory may favor the TTB winner depending on the subject's threshold parameter.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
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
**Description:** Decision-makers use a 'Take-The-Best (TTB) with Thresholded Compensatory Tallying' strategy. They initially attempt to use a lexicographic TTB approach, favoring the option that wins on the first discriminating feature. However, they evaluate the remaining features: if the number of opposing wins on subsequent features reaches or exceeds a subject-specific threshold, they abandon TTB and switch to a compensatory Tallying strategy. When Tallying results in a tie, a secondary recency (anti-primacy) bias acts as a tie-breaker, which is explicitly constrained to only apply when the primary tally counts are exactly equal, preventing it from overriding strict tallying wins.

**Parameters:**
- theta: [0.0, 3.0]
- w_recency: [-5.0, 5.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    theta = float(parameters["theta"])
    w_recency = float(parameters["w_recency"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # 1. Identify TTB winner and the first discriminating cue
    ttb_winner = None
    k = -1
    for i in range(n_features):
        if a[i] > b[i]:
            ttb_winner = 0
            k = i
            break
        elif b[i] > a[i]:
            ttb_winner = 1
            k = i
            break
            
    if ttb_winner is None:
        return np.array([0.5, 0.5])
        
    # 2. Evaluate remaining features
    if ttb_winner == 0:
        opposing_wins = np.sum(b[k+1:] > a[k+1:])
    else:
        opposing_wins = np.sum(a[k+1:] > b[k+1:])
        
    diff_val = float(opposing_wins)
    
    # 3. Calculate Tallying scores with Recency tie-breaker
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    weights = np.arange(1, n_features + 1, dtype=float)
    weights /= np.sum(weights)
    recency_a = np.sum(a * weights)
    recency_b = np.sum(b * weights)
    
    # Explicitly restrict tie-breaker to only apply when tally counts are equal
    if a_wins == b_wins:
        tally_scores = np.array([
            a_wins + w_recency * recency_a,
            b_wins + w_recency * recency_b
        ])
    else:
        tally_scores = np.array([a_wins, b_wins])
    
    # 4. Apply Threshold Logic
    if diff_val >= theta:
        scores = tally_scores
    else:
        scores = np.array([1.0, 0.0]) if ttb_winner == 0 else np.array([0.0, 1.0])
        
    # Softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    probs = e / np.sum(e)
    
    return (1.0 - epsilon) * probs + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    def check_tie(row):\n        a = np.array(row['option_a_ratings'])\n        b = np.array(row['option_b_ratings'])\n        return np.sum(a > b) == np.sum(b > a)\n        \n    is_tie = data.apply(check_tie, axis=1)\n    tie_data = data[is_tie].copy()\n    \n    if len(tie_data) == 0:\n        return 0.0\n        \n    tie_data['trial_id'] = tie_data.apply(\n        lambda x: tuple(x['option_a_ratings']) + tuple(x['option_b_ratings']), axis=1\n    )\n    \n    tie_data['chose_a'] = (tie_data['response'] == 0).astype(float)\n    \n    means = tie_data.groupby(['subject_id', 'trial_id'])['chose_a'].mean()\n    \n    return float(np.abs(means - 0.5).mean())",
  "rationale": "This metric calculates the mean absolute deviation of the choice probability from 0.5 on 'Tally Tie' trials (where both options have an equal number of feature wins). The Advocated Theory (Pure Tallying) predicts exactly 50/50 guessing on these trials, so the sample proportion will be close to 0.5 (yielding a small absolute deviation governed only by binomial variance). The Competing Theory, however, will resolve these ties deterministically either via Take-The-Best (if the opposing wins are below threshold) or via the recency tie-breaker. This results in choice probabilities near 0 or 1, yielding a mean absolute deviation close to 0.5."
}
```

## Usage

```json
{
  "prompt_token_count": 3903,
  "candidates_token_count": 445,
  "total_token_count": 7261
}
```
