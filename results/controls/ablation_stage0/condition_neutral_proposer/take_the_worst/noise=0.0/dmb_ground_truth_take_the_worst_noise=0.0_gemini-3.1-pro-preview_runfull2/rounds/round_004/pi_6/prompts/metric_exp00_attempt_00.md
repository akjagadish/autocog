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
**Validities (n_features=5):** [0.95, 0.88, 0.65, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 6: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 7: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 8: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]

**Rationale:** To maximally distinguish Theory 1 (WADD with Subjective Weighting) from Theory 2 (Tallying with Take-The-Best Tiebreaker), we use a 5-feature design with specific validities that create opposing predictions in two critical scenarios. First, in 'Tally Tie' trials, both options have an equal number of winning features. Theory 2 uses the single highest-validity discriminating cue (TTB) to break the tie, whereas Theory 1 evaluates the subjective weighted sum. By pitting the 1st and 5th features against the 2nd and 3rd features, TTB strongly favors the option with the 1st feature, while WADD favors the option with the 2nd and 3rd features combined. Second, in 'Unequal Tally' trials, we pit the top two highest-validity features against the bottom three. Theory 2's Tallying favors the option with three features, but if Theory 1's gamma parameter is high enough, the subjective weights of the top two features will overpower the three lower-validity features, leading to opposing predictions.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Weighted Additive (WADD) with Subjective Weighting: Decision-makers evaluate options using a fully compensatory strategy where each option's value is the weighted sum of its features. Instead of using the raw validities as weights, they apply a subjective transformation modeled as a power law (validity^gamma). This parameterization allows the model to capture a spectrum of strategies: when gamma is 0, it reduces to Equal-Weight (Tallying); when gamma is 1, it is standard WADD. By constraining gamma to be very small, the model predominantly relies on tallying-like behavior while allowing slight compensatory deviations to capture nuances in specific experiments.

**Parameters:**
- beta: [0.1, 20.0]
- gamma: [0.0, 0.8]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Transform validities into subjective weights via a power law
    weights = val ** gamma
    
    # Calculate the weighted sum for each option
    scores = np.sum(stim * weights, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
    p = np.asarray(probabilities, dtype=np.float64)
    p /= p.sum()  # Ensure valid probabilities
    return np.random.choice(len(p), p=p)
```


## COMPETING THEORY
**Description:** Tallying with Take-The-Best Tiebreaker: Decision makers primarily use a simple Tallying heuristic, counting the number of features that favor each option. If one option has a higher tally, it is chosen. However, if the tallies are tied, they do not simply guess; instead, they fall back to the Take-The-Best (TTB) heuristic, breaking the tie by choosing the option favored by the single most valid differentiating feature. This tie-breaking influence can be parameterized to allow for both positive reinforcement or penalty depending on the specific cue structures.

**Parameters:**
- beta: [0.1, 15.0]
- epsilon: [0.0, 0.5]
- tau: [-2.0, 2.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    # Primary heuristic: Tallying (counting strict wins)
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    
    score_a = float(a_wins)
    score_b = float(b_wins)
    
    # Tie-breaker: Take-The-Best
    if score_a == score_b:
        tau = float(parameters["tau"])
        val = np.asarray(parameters["validities"], dtype=float)
        # Sort indices by descending validity
        order = np.argsort(-val, kind="stable")
        for idx in order:
            if a[idx] > b[idx]:
                score_a += tau
                break
            elif b[idx] > a[idx]:
                score_b += tau
                break
                
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
    p = np.asarray(probabilities, dtype=np.float64)
    p /= p.sum()  # Ensure valid probabilities
    return np.random.choice(len(p), p=p)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Create hashable representations of the options to identify specific trials\n    a_tuples = data['option_a_ratings'].apply(tuple)\n    \n    # Trial 1: A=[1, 1, 0, 0, 0], B=[0, 0, 1, 1, 1]\n    # Theory 2 (Tallying) prefers B because 3 features > 2 features.\n    # Theory 1 (WADD) prefers A because the top 2 features outweigh the bottom 3.\n    t1_mask = a_tuples == (1, 1, 0, 0, 0)\n    \n    # Trials 2 & 3: A=[1, 0, 0, 0, 1] or [1, 0, 0, 1, 0], B=[0, 1, 1, 0, 0]\n    # Theory 2 (TTB tiebreaker) prefers A because tally is tied (2 vs 2) and A has the best feature.\n    # Theory 1 (WADD) prefers B because features 2+3 combined outweigh feature 1 + bottom feature.\n    t23_mask = (a_tuples == (1, 0, 0, 0, 1)) | (a_tuples == (1, 0, 0, 1, 0))\n    \n    p_b_t1 = data.loc[t1_mask, 'response'].mean()\n    p_b_t23 = data.loc[t23_mask, 'response'].mean()\n    \n    # Metric: P(Choose B in Trials 2 & 3) - P(Choose B in Trial 1)\n    # Theory 1 (WADD) predicts high P(B) in T2/3 and low P(B) in T1 -> Positive metric ~ 1.0\n    # Theory 2 (Tally+TTB) predicts low P(B) in T2/3 and high P(B) in T1 -> Negative metric ~ -1.0\n    return float(p_b_t23 - p_b_t1)\n",
  "rationale": "This metric isolates the specific trials where the two theories make diametrically opposed predictions. In Trial 1 ('Unequal Tally'), Tallying favors the option with more features (B), while WADD favors the option with fewer, higher-validity features (A). In Trials 2 and 3 ('Tally Tie'), Tallying falls back to Take-The-Best, which favors the option with the highest validity feature (A), while WADD favors the combined weight of the second and third highest validity features (B). By subtracting the proportion of B choices in Trial 1 from the proportion of B choices in Trials 2 and 3, we get a metric that is strongly positive for Theory 1 and strongly negative for Theory 2."
}
```

## Usage

```json
{
  "prompt_token_count": 3412,
  "candidates_token_count": 681,
  "total_token_count": 7103
}
```
