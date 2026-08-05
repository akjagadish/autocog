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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 3: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 8: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Rationale:** To quantitatively dissociate Pure Tallying (Advocated Theory) from Strict Tallying with Probabilistic TTB Tie-Breaker (Competing Theory), we exploit the specific condition where their predictions diverge: tally ties. When the tally difference is non-zero, both theories predict identical choices based strictly on the tally difference with softmax noise. However, when the tally is tied, Pure Tallying predicts a strict 50/50 guess, while the Competing Theory applies a Take-The-Best (TTB) tie-breaker in probability space. By including 'tally tie' trials where the highest-validity discriminating cue strongly favors one option, we can observe whether choices remain strictly at chance or shift systematically toward the TTB prediction. We also include 'tally win' trials to anchor the shared beta and epsilon parameters.

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
**Description:** Strict Tallying with Probabilistic Take-The-Best (TTB) Tie-Breaker: Decision-makers primarily rely on a simple tally of winning features to evaluate options, choosing the option with the most winning features via a softmax choice rule. When the tallies are strictly tied, they bypass the tallying confidence mechanism entirely and fall back on a lexicographic Take-The-Best strategy. To capture the empirical finding that subjects are much less deterministic on tie trials, the tie-breaker is applied directly in probability space, allowing for a high rate of random guessing independent of the determinism on clear tally wins.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- p_ttb: [0.0, 1.0]
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
    order = np.argsort(-val, kind="stable")
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    p_ttb = float(parameters["p_ttb"])
    
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    diff = a_wins - b_wins
    
    # If the tally difference is non-zero, strictly use Tallying with Softmax
    if diff != 0:
        score_a = float(np.sign(diff))
        score_b = float(-np.sign(diff))
        scores = np.array([score_a, score_b])
        
        # Softmax choice rule
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        p_core = e / np.sum(e)
    else:
        # Otherwise, fall back to Take-The-Best tie-breaker directly in probability space
        ttb_winner = 0
        for idx in order:
            if a[idx] > b[idx]:
                ttb_winner = 1
                break
            elif b[idx] > a[idx]:
                ttb_winner = -1
                break
                
        if ttb_winner == 1:
            p_core = np.array([0.5 + p_ttb / 2.0, 0.5 - p_ttb / 2.0])
        elif ttb_winner == -1:
            p_core = np.array([0.5 - p_ttb / 2.0, 0.5 + p_ttb / 2.0])
        else:
            p_core = np.array([0.5, 0.5])
    
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_ratings = np.stack(data['option_a_ratings'].values)\n    b_ratings = np.stack(data['option_b_ratings'].values)\n    responses = data['response'].values\n    \n    a_wins = np.sum(a_ratings > b_ratings, axis=1)\n    b_wins = np.sum(b_ratings > a_ratings, axis=1)\n    \n    tie_mask = (a_wins == b_wins)\n    \n    if not np.any(tie_mask):\n        return 0.5\n        \n    a_tie = a_ratings[tie_mask]\n    b_tie = b_ratings[tie_mask]\n    resp_tie = responses[tie_mask]\n    \n    ttb_winners = []\n    for i in range(len(a_tie)):\n        winner = -1\n        for j in range(5):\n            if a_tie[i, j] > b_tie[i, j]:\n                winner = 0\n                break\n            elif b_tie[i, j] > a_tie[i, j]:\n                winner = 1\n                break\n        ttb_winners.append(winner)\n        \n    ttb_winners = np.array(ttb_winners)\n    valid_mask = (ttb_winners != -1)\n    \n    if not np.any(valid_mask):\n        return 0.5\n        \n    match = (resp_tie[valid_mask] == ttb_winners[valid_mask])\n    return float(np.mean(match))",
  "rationale": "This metric isolates trials where the tally of winning features is exactly tied between the two options. Pure Tallying predicts that decision-makers will guess uniformly at random (50%) on these trials. The Competing Theory, however, posits that decision-makers fall back on a probabilistic Take-The-Best (TTB) tie-breaker, systematically favoring the option that wins on the single most valid discriminating cue. By calculating the proportion of choices that align with the TTB-favored option exclusively on tally-tie trials, we obtain a value near 0.5 for Pure Tallying and systematically higher than 0.5 for the Competing Theory."
}
```

## Usage

```json
{
  "prompt_token_count": 3636,
  "candidates_token_count": 568,
  "total_token_count": 6405
}
```
