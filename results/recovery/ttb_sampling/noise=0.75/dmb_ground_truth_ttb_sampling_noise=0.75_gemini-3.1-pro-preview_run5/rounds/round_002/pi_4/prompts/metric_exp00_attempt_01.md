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

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 8: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]

**Rationale:** To dissociate the Probabilistic Strategy Selection (Mixture of TTB and Tallying) model from the Weighted Additive (WADD) model, we exploit the fact that the mixture model relies on two specific, distinct sub-strategies: a purely lexicographic one (TTB) and a purely unweighted compensatory one (Tallying). WADD, on the other hand, evaluates options using a single set of continuous, validity-scaled weights, which must behave as a linear, compensatory model. We construct a 5-feature design with descending validities. The trials are designed to create specific conflicts between TTB and Tallying, and to probe choice patterns that a linear model (WADD) struggles to fit simultaneously. For example, some trials pit the single highest-validity cue against multiple lower-validity cues (where TTB and Tallying disagree, causing the mixture model to predict an intermediate choice probability depending on p_lex). By varying the number of opposing low-validity cues across trials, we can observe whether the choice probabilities shift in a step-function manner (as predicted by the mixture model, since Tallying only counts wins) or in a graded manner (as predicted by WADD, which sums the continuous weights).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Strategy Selection (Mixture of TTB and Tallying)

**Parameters:**
- p_lex: [0.0, 1.0]
- beta: [0.1, 20.0]
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
    p_lex = float(parameters["p_lex"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    a, b = stim[0], stim[1]
    
    # Strategy 1: Take-The-Best (Lexicographic)
    order = np.argsort(-validities, kind='stable')
    scores_ttb = np.zeros(2)
    for idx in order:
        if a[idx] > b[idx]:
            scores_ttb[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores_ttb[1] = 1.0
            break
            
    # Strategy 2: Tallying (Compensatory)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tal = np.array([a_wins, b_wins])
    
    # Softmax for TTB
    z_ttb = beta * (scores_ttb - scores_ttb.max())
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / e_ttb.sum()
    
    # Softmax for Tallying
    z_tal = beta * (scores_tal - scores_tal.max())
    e_tal = np.exp(z_tal)
    p_tal = e_tal / e_tal.sum()
    
    # Mixture of strategies
    p_core = p_lex * p_ttb + (1.0 - p_lex) * p_tal
    
    # Uniform lapse blended into the mixture
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


## COMPETING THEORY
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
[0] rationale: This metric calculates the difference in the probability of choosing Option 1 between Trial 7 and Trial 1. In both trials, Option 1 wins on the highest validity cue (TTB differential = 1) and Option 2 has exactly one more positive cue than Option 1 (Tallying differential = -1). Consequently, the Probabilistic Strategy Selection (Mixture) model predicts identical choice probabilities for both trials, yielding a metric near 0. In contrast, the Weighted Additive (WADD) model evaluates the specific continuous weights of the cues involved, which generally leads to a non-zero (and typically positive) difference in choice probability.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    data = data.copy()
    data['a_tup'] = data['option_a_ratings'].apply(tuple)
    data['b_tup'] = data['option_b_ratings'].apply(tuple)
    
    # Trial 1: Option 1 = [1, 0, 0, 0, 0], Option 2 = [0, 1, 1, 0, 0]
    m1_norm = (data['a_tup'] == (1, 0, 0, 0, 0)) & (data['b_tup'] == (0, 1, 1, 0, 0))
    m1_swap = (data['a_tup'] == (0, 1, 1, 0, 0)) & (data['b_tup'] == (1, 0, 0, 0, 0))
    
    p_a_t1_norm = (data.loc[m1_norm, 'response'] == 0).mean() if m1_norm.sum() > 0 else np.nan
    p_a_t1_swap = (data.loc[m1_swap, 'response'] == 1).mean() if m1_swap.sum() > 0 else np.nan
    
    t1_vals = [x for x in [p_a_t1_norm, p_a_t1_swap] if not np.isnan(x)]
    p_a_t1 = np.mean(t1_vals) if t1_vals else 0.5
    
    # Trial 7: Option 1 = [1, 0, 1, 0, 0], Option 2 = [0, 1, 0, 1, 1]
    m7_norm = (data['a_tup'] == (1, 0, 1, 0, 0)) & (data['b_tup'] == (0, 1, 0, 1, 1))
    m7_swap = (data['a_tup'] == (0, 1, 0, 1, 1)) & (data['b_tup'] == (1, 0, 1, 0, 0))
    
    p_a_t7_norm = (data.loc[m7_norm, 'response'] == 0).mean() if m7_norm.sum() > 0 else np.nan
    p_a_t7_swap = (data.loc[m7_swap, 'response'] == 1).mean() if m7_swap.sum() > 0 else np.nan
    
    t7_vals = [x for x in [p_a_t7_norm, p_a_t7_swap] if not np.isnan(x)]
    p_a_t7 = np.mean(t7_vals) if t7_vals else 0.5
    
    return float(p_a_t7 - p_a_t1)

outcome: self_sim=-0.0167 (var=0.0406) adversary_sim=0.1250 (var=0.1128) welch_t=-1.809 p=0.07819 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    data = data.copy()\n    data['a_tup'] = data['option_a_ratings'].apply(tuple)\n    data['b_tup'] = data['option_b_ratings'].apply(tuple)\n    \n    # Trial 1: A=[1, 0, 0, 0, 0], B=[0, 1, 1, 0, 0]\n    m1_norm = (data['a_tup'] == (1, 0, 0, 0, 0)) & (data['b_tup'] == (0, 1, 1, 0, 0))\n    m1_swap = (data['a_tup'] == (0, 1, 1, 0, 0)) & (data['b_tup'] == (1, 0, 0, 0, 0))\n    p_a_t1_norm = (data.loc[m1_norm, 'response'] == 0).mean() if m1_norm.sum() > 0 else np.nan\n    p_a_t1_swap = (data.loc[m1_swap, 'response'] == 1).mean() if m1_swap.sum() > 0 else np.nan\n    t1_vals = [x for x in [p_a_t1_norm, p_a_t1_swap] if not np.isnan(x)]\n    p_a_t1 = np.mean(t1_vals) if t1_vals else 0.5\n    \n    # Trial 3: A=[1, 0, 0, 0, 0], B=[0, 1, 1, 1, 1]\n    m3_norm = (data['a_tup'] == (1, 0, 0, 0, 0)) & (data['b_tup'] == (0, 1, 1, 1, 1))\n    m3_swap = (data['a_tup'] == (0, 1, 1, 1, 1)) & (data['b_tup'] == (1, 0, 0, 0, 0))\n    p_a_t3_norm = (data.loc[m3_norm, 'response'] == 0).mean() if m3_norm.sum() > 0 else np.nan\n    p_a_t3_swap = (data.loc[m3_swap, 'response'] == 1).mean() if m3_swap.sum() > 0 else np.nan\n    t3_vals = [x for x in [p_a_t3_norm, p_a_t3_swap] if not np.isnan(x)]\n    p_a_t3 = np.mean(t3_vals) if t3_vals else 0.5\n    \n    return float(p_a_t1 - p_a_t3)",
  "rationale": "This metric evaluates the difference in the probability of choosing Option A between Trial 1 and Trial 3. In both trials, Option A has only the highest-validity cue, while Option B has multiple lower-validity cues. From Trial 1 to Trial 3, Option B gains two additional low-validity cues. The Probabilistic Strategy Selection (Mixture) model predicts a very small difference because Tallying already strongly favors Option B in Trial 1, so adding more cues to B in Trial 3 doesn't change the Tallying choice probability much (it stays near 1). The WADD model, however, integrates all cues linearly, so the two additional cues for Option B in Trial 3 significantly increase its continuous weighted sum, leading to a much larger drop in the probability of choosing Option A."
}
```

## Usage

```json
{
  "prompt_token_count": 4687,
  "candidates_token_count": 908,
  "total_token_count": 9268
}
```
