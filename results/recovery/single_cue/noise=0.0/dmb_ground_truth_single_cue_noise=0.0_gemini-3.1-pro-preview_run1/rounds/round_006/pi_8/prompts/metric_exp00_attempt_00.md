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
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 6: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 7: A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (a mixture of Left-to-Right Take-The-Best, Tallying, and a 'Top-Heavy Penalty') from the Competing Theory (Pure Tallying), we exploit the unique structural sensitivities of the advocated model. Pure Tallying strictly counts feature-wise wins and guesses uniformly on ties, completely ignoring the spatial/index position of the features. The Advocated Theory, however, uses the Left-to-Right (LTR) primacy rule and a Top-Heavy Penalty (THP) that explicitly favors features further to the right (higher indices). Crucially, when Tallying results in a tie, the Advocated Theory boosts the weight of the THP. We include 'Tally Tie' trials where Tallying predicts exactly 50/50, but LTR and THP pull strongly in specific directions (e.g., LTR prefers A, but THP strongly prefers B). We also include 'Compensatory' trials where Tallying prefers one option (e.g., 3 wins vs 2 wins), but LTR and/or THP favor the other option, allowing us to tease apart the specific mixture components and the conditional tie-breaker boost.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Decision-makers utilize a conditional mixture of three distinct heuristic strategies: Primacy-based Take-The-Best, Tallying, and a 'Top-Heavy Penalty' (Anti-Primacy) strategy. The Top-Heavy Penalty heavily penalizes options that rely on early, high-validity cues by using a linear index-based weight (starting from 1 instead of 0), preferring options whose positive features are shifted further to the right without entirely discounting the primary cue. When Tallying results in a tie, the reliance on this Top-Heavy Penalty is significantly boosted to act as a decisive tie-breaker, whereas in non-tie scenarios it provides a baseline balancing effect alongside Take-The-Best and Tallying.

**Parameters:**
- w_ltr: [0.0, 10.0]
- w_thp: [0.0, 10.0]
- w_thp_tie: [0.0, 50.0]
- w_tally: [0.0, 10.0]
- beta_ttb: [0.1, 20.0]
- beta_tally: [0.1, 20.0]
- beta_thp: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be of shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    w_ltr = float(parameters["w_ltr"])
    w_thp = float(parameters["w_thp"])
    w_thp_tie = float(parameters["w_thp_tie"])
    w_tally = float(parameters["w_tally"])
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    beta_thp = float(parameters["beta_thp"])
    epsilon = float(parameters["epsilon"])
    
    # 3. Tallying
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins])
    
    # Normalize mixture weights conditionally
    if a_wins == b_wins:
        current_w_thp = w_thp_tie
    else:
        current_w_thp = w_thp
        
    total_w = w_ltr + current_w_thp + w_tally
    if total_w == 0:
        p_ltr, p_thp, p_tally = 1/3, 1/3, 1/3
    else:
        p_ltr = w_ltr / total_w
        p_thp = current_w_thp / total_w
        p_tally = w_tally / total_w
        
    # 1. Left-to-Right Take-The-Best (Primacy)
    ltr_scores = np.array([0.0, 0.0])
    for i in range(n_features):
        if a[i] > b[i]:
            ltr_scores[0] = 1.0
            break
        elif b[i] > a[i]:
            ltr_scores[1] = 1.0
            break
            
    # 2. Top-Heavy Penalty (Anti-Primacy)
    # Penalize options that rely on early cues using a linear index-based penalty.
    thp_weights = np.arange(1, n_features + 1, dtype=float)
    thp_a = np.sum(a * thp_weights)
    thp_b = np.sum(b * thp_weights)
    thp_scores = np.array([thp_a, thp_b])
            
    # Helper to compute softmax probabilities safely
    def get_probs(scores, beta):
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        return e / np.sum(e)
        
    probs_ltr = get_probs(ltr_scores, beta_ttb)
    probs_thp = get_probs(thp_scores, beta_thp)
    probs_tally = get_probs(tally_scores, beta_tally)
    
    # Mix strategies
    mixed_probs = p_ltr * probs_ltr + p_thp * probs_thp + p_tally * probs_tally
    
    # Apply random lapse rate
    final_probs = (1.0 - epsilon) * mixed_probs + epsilon * np.array([0.5, 0.5])
    
    return final_probs
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_tuples = data['option_a_ratings'].apply(tuple)\n    \n    # Trial 2: A=[1, 0, 0, 0, 1] vs B=[0, 1, 1, 0, 0]\n    t2_mask = (a_tuples == (1, 0, 0, 0, 1))\n    # Trial 8: A=[0, 1, 1, 0, 0] vs B=[1, 0, 0, 0, 1]\n    t8_mask = (a_tuples == (0, 1, 1, 0, 0))\n    \n    val = 0.0\n    if t2_mask.any():\n        val += (data.loc[t2_mask, 'response'] == 0).mean()\n    if t8_mask.any():\n        val += (data.loc[t8_mask, 'response'] == 1).mean()\n        \n    return float(val)",
  "rationale": "Under the Competing Theory (Tallying), trials 2 and 8 result in exact ties (2 feature wins for A, 2 for B), so the probability of choosing either option is 0.5. The metric sums the proportion of choosing A in trial 2 and B in trial 8, which yields an expected value of 1.0 under Tallying. Under the Advocated Theory, both Left-to-Right Take-The-Best and the Top-Heavy Penalty strongly agree on A in trial 2 and on B in trial 8. Because these trials are 'Tally Ties', the Advocated Theory boosts the weight of the Top-Heavy Penalty, leading to a much higher expected value (> 1.0) for this metric."
}
```

## Usage

```json
{
  "prompt_token_count": 4148,
  "candidates_token_count": 441,
  "total_token_count": 8002
}
```
