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
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 5: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 6: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Rationale:** This design quantitatively dissociates the Strategy Mixture Theory (which posits intra-individual trial-by-trial mixing of Take-The-Best and Tallying) from the Weighted Additive (WADD) rule. The validities are structured with two highly valid cues and three less valid cues. We construct critical trials where TTB and Tallying point in opposite directions (e.g., Option A wins on the highest validity cue, but Option B wins on a greater number of less valid cues). Because Strategy Mixture Theory assumes individuals mix TTB and Tallying (with p_ttb ~ 0.5), it predicts a choice proportion near 0.5 for these conflict trials. In contrast, WADD integrates all cardinal validities and will strongly favor one option depending on the exact sum of weights, driving choice probabilities toward 1 or 0. By varying which option WADD favors while keeping the TTB/Tallying conflict constant, we can perfectly dissociate the two models.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Strategy Mixture Theory: Instead of relying on a single heuristic, decision-makers draw from a repertoire of strategies on a trial-by-trial basis. Specifically, individuals mix between a non-compensatory lexicographic rule (Take-The-Best) and a compensatory rule (Tallying). On any given trial, a subject employs TTB with probability `p_ttb` and Tallying with probability `1 - p_ttb`. This intra-individual strategy variation naturally accounts for the aggregate ~0.50 choice proportions observed in conflict trials where the two heuristics prescribe different options, while a relatively stable mixture proportion across the population explains the low between-subject variance.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- p_ttb: [0.4, 0.6]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Strategy Mixture expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    # Strategy 1: Take-The-Best (TTB)
    order = np.argsort(validities)[::-1]
    score_ttb = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            score_ttb[0] = 1.0
            break
        elif b[idx] > a[idx]:
            score_ttb[1] = 1.0
            break
            
    # Strategy 2: Tallying (Compensatory)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    score_tally = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    p_ttb = float(parameters["p_ttb"])
    
    # Softmax for TTB
    z_ttb = beta * score_ttb
    z_ttb = z_ttb - np.max(z_ttb)
    e_ttb = np.exp(z_ttb)
    prob_ttb = e_ttb / np.sum(e_ttb)
    
    # Softmax for Tallying
    z_tally = beta * score_tally
    z_tally = z_tally - np.max(z_tally)
    e_tally = np.exp(z_tally)
    prob_tally = e_tally / np.sum(e_tally)
    
    # Mix the two strategies
    p_core = p_ttb * prob_ttb + (1.0 - p_ttb) * prob_tally
    
    n_opts = p_core.shape[0]
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n\n    # Create a hashable trial identifier\n    data['trial_id'] = data.apply(\n        lambda row: (tuple(row['option_a_ratings']), tuple(row['option_b_ratings'])), \n        axis=1\n    )\n    \n    # Identify conflict trials where TTB and Tallying prescribe different options.\n    # TTB relies on the first cue (index 0). Tallying relies on the sum of cues.\n    def is_conflict(row):\n        a = row['option_a_ratings']\n        b = row['option_b_ratings']\n        ttb_a = a[0] > b[0]\n        ttb_b = b[0] > a[0]\n        tally_a = sum(a) > sum(b)\n        tally_b = sum(b) > sum(a)\n        \n        if ttb_a and tally_b:\n            return True\n        if ttb_b and tally_a:\n            return True\n        return False\n\n    data['is_conflict'] = data.apply(is_conflict, axis=1)\n    conflict_data = data[data['is_conflict']]\n    \n    if len(conflict_data) == 0:\n        return 0.0\n        \n    # For each subject and each unique conflict trial, compute the choice proportion.\n    # response == 0 means A, response == 1 means B. \n    # The mean of response is the proportion of B choices (p_B).\n    # The absolute difference from 0.5 measures how deterministic the subject's choice is.\n    grouped = conflict_data.groupby(['subject_id', 'trial_id'])['response'].mean().reset_index()\n    grouped['extremity'] = (grouped['response'] - 0.5).abs()\n    \n    # Average the extremity of preferences per subject across all conflict trials,\n    # then return the grand mean.\n    subj_extremity = grouped.groupby('subject_id')['extremity'].mean()\n    return float(subj_extremity.mean())\n",
  "rationale": "Strategy Mixture Theory posits that individuals mix between TTB and Tallying on a trial-by-trial basis. On conflict trials where the two heuristics disagree, an SMT agent will stochasticly alternate between the two options across repetitions of the exact same trial, yielding a choice proportion near 0.5 (low extremity). In contrast, WADD computes a deterministic weighted sum based on subjective validities. For any specific trial, a WADD agent will consistently favor one option over the other across all repetitions (modulo some response noise), yielding choice proportions closer to 0 or 1 (high extremity). By measuring the absolute deviation of the intra-individual choice proportion from 0.5 on identical conflict trials, we can cleanly separate the stochastic strategy mixing of SMT from the deterministic weighting of WADD."
}
```

## Usage

```json
{
  "prompt_token_count": 3760,
  "candidates_token_count": 702,
  "total_token_count": 9797
}
```
