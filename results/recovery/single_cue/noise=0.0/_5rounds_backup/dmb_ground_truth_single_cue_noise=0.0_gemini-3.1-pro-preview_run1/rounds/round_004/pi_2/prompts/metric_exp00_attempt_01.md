# metric_exp00_attempt_01

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
**Validities (n_features=6):** [0.9, 0.8, 0.7, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  trial 2: A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 0]
  trial 4: A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 5: A=[1, 1, 1, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1]
  trial 6: A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 7: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 8: A=[1, 1, 1, 0, 0, 1]  B=[0, 0, 0, 1, 0, 1]

**Rationale:** This design fully crosses the tally difference (0, 1, and 2) with the ownership of the final feature (won by Option A, won by Option B, or tied) to quantitatively dissociate standard Tallying from Recency-Biased Tallying. Standard Tallying predicts that choice probabilities are determined strictly by the tally difference, meaning all trials with a tally difference of 1 will yield identical choice distributions, as will all trials with a difference of 2. In contrast, Recency-Biased Tallying predicts that confidence will systematically diverge based on who wins the final feature, creating a main effect of last-feature ownership that modulates the primary tally difference. The inclusion of trials where the last feature is tied provides a critical baseline where both models generate identical predictions, ensuring the divergence is isolated to the recency mechanism.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



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
**Description:** Recency-Biased Tallying: People primarily evaluate options by counting the number of features where one option strictly dominates the other (Tallying). However, rather than giving a bonus or penalty based on cue validity, subjects exhibit a structural recency effect. They give a secondary tie-breaking bonus to the option that is superior on the most recently evaluated feature (the last cue). This preserves the dominance of Tallying for clear differences while elegantly explaining why tie-breaking appears at chance-level relative to the most valid (first) cue in most experiments, yet systematically favors the option winning the final cue when specifically tested.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- w: [0.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Recency-Biased Tallying expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    # Primary mechanism: Tallying strict wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    # Secondary mechanism: Recency bonus (winning the last feature)
    w = float(parameters["w"])
    a_last_win = float(a[-1] > b[-1])
    b_last_win = float(b[-1] > a[-1])
    
    score_a = a_wins + w * a_last_win
    score_b = b_wins + w * b_last_win
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
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
[0] rationale: This metric calculates the difference in the probability of choosing Option A between trials where Option A wins the last feature and trials where Option B wins the last feature, restricted to trials where Option A has a positive tally difference (1 or 2). Standard Tallying ignores feature order, meaning the last feature has no special status; thus, the choice probabilities should be identical across these two sets of trials, yielding a metric near 0. Recency-Biased Tallying, however, assigns a tie-breaking bonus to the winner of the final feature, so Option A will be chosen significantly more often when it wins the last feature compared to when Option B wins it, yielding a positive metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Convert ratings to arrays to compute tally differences
    a_arr = data['option_a_ratings'].apply(np.array)
    b_arr = data['option_b_ratings'].apply(np.array)
    
    a_wins = (a_arr > b_arr).apply(np.sum)
    b_wins = (b_arr > a_arr).apply(np.sum)
    tally_diff = a_wins - b_wins
    
    # Identify who won the last feature
    last_a = data['option_a_ratings'].apply(lambda x: x[-1])
    last_b = data['option_b_ratings'].apply(lambda x: x[-1])
    
    mask_a_wins_last = last_a > last_b
    mask_b_wins_last = last_b > last_a
    
    # Focus on trials where A wins the tally by 1 or 2
    mask_valid = (tally_diff == 1) | (tally_diff == 2)
    
    # Compute probability of choosing A (response == 0)
    p_a_a_last = (1 - data.loc[mask_valid & mask_a_wins_last, 'response']).mean()
    p_a_b_last = (1 - data.loc[mask_valid & mask_b_wins_last, 'response']).mean()
    
    if pd.isna(p_a_a_last) or pd.isna(p_a_b_last):
        return 0.0
        
    return float(p_a_a_last - p_a_b_last)
outcome: evaluation failed (ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all())

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_wins_last = []\n    b_wins_last = []\n    tally_diffs = []\n    \n    # Iterate directly to avoid pandas apply issues with array comparisons\n    for a, b in zip(data['option_a_ratings'], data['option_b_ratings']):\n        a_arr = np.array(a)\n        b_arr = np.array(b)\n        \n        a_wins = np.sum(a_arr > b_arr)\n        b_wins = np.sum(b_arr > a_arr)\n        tally_diffs.append(a_wins - b_wins)\n        \n        a_wins_last.append(a_arr[-1] > b_arr[-1])\n        b_wins_last.append(b_arr[-1] > a_arr[-1])\n        \n    # Use a local copy to avoid SettingWithCopyWarning\n    df = data.copy()\n    df['tally_diff'] = tally_diffs\n    df['a_wins_last'] = a_wins_last\n    df['b_wins_last'] = b_wins_last\n    \n    # Restrict to trials where A is strictly favored by Tallying\n    mask_diff = df['tally_diff'] > 0\n    \n    # Probability of choosing A (response == 0)\n    p_a_when_a_wins_last = 1.0 - df.loc[mask_diff & df['a_wins_last'], 'response'].mean()\n    p_a_when_b_wins_last = 1.0 - df.loc[mask_diff & df['b_wins_last'], 'response'].mean()\n    \n    if pd.isna(p_a_when_a_wins_last) or pd.isna(p_a_when_b_wins_last):\n        return 0.0\n        \n    return float(p_a_when_a_wins_last - p_a_when_b_wins_last)",
  "rationale": "This metric isolates the effect of the last feature by calculating the difference in the probability of choosing Option A when Option A wins the last feature versus when Option B wins it, specifically on trials where Option A already holds a positive tally difference. Standard Tallying ignores feature order, so the identity of the last feature's winner is irrelevant, and the metric will be near 0. Recency-Biased Tallying assigns a structural tie-breaking bonus to the winner of the final feature, so Option A will be chosen at a significantly higher rate when it wins the last feature compared to when Option B wins it, yielding a robust positive value."
}
```

## Usage

```json
{
  "prompt_token_count": 4074,
  "candidates_token_count": 660,
  "total_token_count": 7710
}
```
