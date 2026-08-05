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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 7: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 8: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 9: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 10: A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate a pure Weighted Additive (WADD) model with subjective weights from a Mixture of Heuristics (Tallying + objective WADD), this design uses 5 features with a linear spread of validities. The trials comprehensively pit individual higher-validity features against combinations of lower-validity features. The Mixture model is restricted to a convex combination of equal weights (Tallying) and strictly objective validities (WADD). In contrast, the advocated pure WADD model allows for subjective weighting, meaning it can capture non-linear transformations of the validities (e.g., extreme overweighting of the most valid cue, resembling Take-The-Best, or idiosyncratic trade-offs between middle-tier cues) that cannot be expressed as any mixture of objective WADD and Tallying. By spanning diverse trade-offs across the validity hierarchy, we can identify choice probability profiles that fall outside the rigid manifold of the Mixture model but are perfectly captured by WADD with free subjective weights.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson
    # 1993). Stimulus is the pair of option feature vectors for the
    # current trial: array-like of shape (2, n_features), row 0 =
    # option A, row 1 = option B. Each option's score is the dot
    # product of its feature vector with the (subjective) validity
    # weights. History is ignored: the choice depends only on the
    # current stimulus.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"weights length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ (validities * w)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax (standard choice-with-lapse
    # formulation; epsilon=0.5 yields equiprobable options, matching
    # the paper's maximum-noise condition).
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
**Description:** Mixture of Heuristics (Tallying and WADD)

**Parameters:**
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.1]
- w_tally: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    w_tally = float(parameters["w_tally"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # --- WADD ---
    scores_wadd = stim @ val
    z_wadd = beta * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
        
    # --- Tallying (Equal Weight) ---
    scores_tally = stim.sum(axis=1)
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # --- Mixture ---
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_wadd
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
[0] rationale: In the experimental design, Trials 6 and 7 are perfectly balanced under both objective WADD and Tallying. For Trial 6, objective WADD gives 1.6 to both options, and Tallying gives 2 to both. For Trial 7, objective WADD gives 1.4 to both, and Tallying gives 2 to both. Thus, the Competing Theory predicts exactly a 50% probability of choosing A or B on these trials, regardless of the mixture weight. Subjects simulated under the Competing Theory will exhibit choice proportions close to 0.5 (only deviating due to finite trial binomial noise). In contrast, the Advocated Theory uses subjective, idiosyncratic weights. It is highly unlikely that a subject's random subjective weights will perfectly balance these options, meaning subjects will typically have a strong preference for either A or B. By calculating the absolute deviation of a subject's choice proportion from 0.5 on these trials, we obtain a metric that will be near 0 for the Competing Theory and substantially greater than 0 (approaching 0.5) for the Advocated Theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Identify trials 6 and 7 where the Competing Theory (Mixture of objective WADD and Tallying)
    # predicts an exact tie (50/50 choice probability).
    # Trial 6: A=[0, 1, 1, 0, 0], B=[1, 0, 0, 1, 0]
    # Trial 7: A=[0, 0, 1, 1, 0], B=[0, 1, 0, 0, 1]
    
    is_t6_a = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0))
    is_t6_b = data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 0))
    
    is_t7_a = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 0))
    is_t7_b = data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 0, 0, 1))
    
    mask = (is_t6_a & is_t6_b) | (is_t7_a & is_t7_b)
    subset = data[mask]
    
    if len(subset) == 0:
        return 0.0
        
    # Calculate each subject's mean response on these specific tie trials
    subj_means = subset.groupby('subject_id')['response'].mean()
    
    # Return the average absolute deviation from 0.5 across subjects
    return float(subj_means.apply(lambda x: abs(x - 0.5)).mean())
outcome: self_sim=0.1178 (var=0.0091) adversary_sim=0.0900 (var=0.0052) welch_t=+1.163 p=0.251 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Identify trials 1 to 5 where option B has more positive features than option A.\n    # In the design, these are exactly the trials where sum(A) < sum(B).\n    # For these trials, both Tallying and Objective WADD strongly favor option B.\n    a_sum = data['option_a_ratings'].apply(sum)\n    b_sum = data['option_b_ratings'].apply(sum)\n    mask = a_sum < b_sum\n    \n    subset = data[mask]\n    if len(subset) == 0:\n        return 0.0\n        \n    # Calculate the proportion of times Option A was chosen (response == 0)\n    # We compute it per subject first to ensure stable between-subject variance.\n    subj_prop_a = subset.groupby('subject_id')['response'].apply(lambda x: (x == 0).mean())\n    \n    return float(subj_prop_a.mean())",
  "rationale": "In the experimental design, Trials 1 through 5 are characterized by Option B having more positive features than Option A. Consequently, the Competing Theory (a mixture of Tallying and Objective WADD) will heavily favor Option B on all these trials, as both the equal-weight heuristic and the objective validities point in the same direction. The probability of choosing Option A under the Competing Theory will be near zero, bounded only by the lapse rate (epsilon). In contrast, the Advocated Theory uses subjective, unconstrained weights. A participant might idiosyncratically assign a very high weight to the single feature present in Option A, overcoming the multiple features in Option B. Thus, the Advocated Theory predicts a significantly higher proportion of Option A choices on these trials. Measuring the mean rate of choosing Option A on this specific subset of trials will cleanly discriminate the two models."
}
```

## Usage

```json
{
  "prompt_token_count": 4332,
  "candidates_token_count": 433,
  "total_token_count": 7071
}
```
