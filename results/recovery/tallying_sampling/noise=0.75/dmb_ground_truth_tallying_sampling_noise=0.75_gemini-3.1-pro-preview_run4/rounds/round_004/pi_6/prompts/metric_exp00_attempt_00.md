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
**Validities (n_features=9):** [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0]
  trial 6: A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Minimalist Heuristic from Tallying, we exploit the inverse of a previously explored relationship. While both models ignore cue validities, Tallying relies on the absolute difference in positive features (tally_A - tally_B), whereas the Minimalist Heuristic relies on the proportion of discriminating cues favoring the option (d_A / (d_A + d_B)). We construct sets of trials where the proportion of discriminating cues is strictly held constant (e.g., 2/3 or 3/4 or 1) while the absolute difference in tallies varies (e.g., d_A=2 vs d_B=1 gives a difference of 1; d_A=4 vs d_B=2 gives a difference of 2; d_A=6 vs d_B=3 gives a difference of 3). The Advocated Minimalist model predicts identical choice probabilities across trials within these sets because the relative share of favorable cues is unchanged. In contrast, the Competing Tallying model predicts that preference strength will increase monotonically as the absolute tally difference grows, despite the constant proportion.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Minimalist Heuristic (Trial-Specific Search): Decision-makers evaluate options sequentially using a one-reason lexicographic process, completely ignoring objective cue validities. Instead of maintaining a stable subjective cue hierarchy, individuals search through the available cues in a random order that varies from trial to trial. They stop at the first cue that discriminates between the two options and choose the favored option. Analytically, the probability of choosing an option is proportional to its share of the discriminating cues. This trial-by-trial stochasticity produces low per-subject determinism on symmetrically tied trials, matching human data.

**Parameters:**
- beta: [0.0, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    # Validities are ignored in the Minimalist heuristic
    _ = parameters["validities"]
    
    # Trial-specific random cue order is analytically equivalent to choosing 
    # proportional to the number of discriminating cues for each option.
    diff = stim[0] - stim[1]
    d_a = np.sum(diff > 0)
    d_b = np.sum(diff < 0)
    
    if d_a + d_b > 0:
        p_a = d_a / (d_a + d_b)
        p_b = d_b / (d_a + d_b)
    else:
        p_a = 0.5
        p_b = 0.5
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the binary scores [1, 0] or [0, 1] marginalized over the random cue order
    S = np.exp(beta) / (np.exp(beta) + 1.0)
    
    prob_a = p_a * S + p_b * (1.0 - S)
    prob_b = p_b * S + p_a * (1.0 - S)
    
    p_core = np.array([prob_a, prob_b])
    
    # Uniform lapse
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
**Description:** Tallying (Equal Weighting): Decision-makers completely ignore the provided cue validities and simply count the number of positive features for each option. The option with the highest tally is chosen. If the tallies are equal, the decision-maker guesses randomly. This heuristic provides an extremely fast and frugal way to compare options, perfectly explaining chance-level performance in 1-on-1 single-cue comparisons (where tallies tie) and highly consistent choices when one option has strictly more positive features. The decision process is subject to significant response noise, reflecting the inherent stochasticity in human choice behavior.

**Parameters:**
- beta: [0.0, 1.5]
- epsilon: [0.2, 0.8]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    # Validities are ignored in Tallying, but we read them to satisfy the parameter reference rule.
    _ = parameters["validities"]
    
    # Tallying: sum the features for each option (ignoring validities)
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over tallies
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Uniform lapse
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    # Calculate tallies for each option\n    tA = data['option_a_ratings'].apply(sum)\n    tB = data['option_b_ratings'].apply(sum)\n    \n    # 1 if choice was A, 0 if B\n    chose_A = (data['response'] == 0).astype(float)\n    \n    df = pd.DataFrame({'tA': tA, 'tB': tB, 'chose_A': chose_A})\n    \n    # Average probability of choosing A for each (tally_A, tally_B) pair\n    means = df.groupby(['tA', 'tB'])['chose_A'].mean().to_dict()\n    \n    def get_p(a, b):\n        return means.get((a, b), 0.5)\n        \n    # Set 1: Proportion of discriminating cues = 2/3\n    # Compare tally diff of 3 vs tally diff of 1\n    diff1 = get_p(6, 3) - get_p(2, 1)\n    \n    # Set 2: Proportion of discriminating cues = 3/4\n    # Compare tally diff of 4 vs tally diff of 2\n    diff2 = get_p(6, 2) - get_p(3, 1)\n    \n    # Set 3: Proportion of discriminating cues = 1.0\n    # Compare tally diff of 5 vs tally diff of 1\n    diff3 = get_p(5, 0) - get_p(1, 0)\n    \n    # The metric is the sum of these probability differences\n    return float(diff1 + diff2 + diff3)",
  "rationale": "The experimental design groups trials into sets where the proportion of discriminating cues favoring option A is perfectly constant, but the absolute difference in tallies varies. According to the Minimalist heuristic, choice probabilities are determined by the proportion of favorable discriminating cues, so the probability of choosing A should be identical within each set (metric \u2248 0). According to the Tallying model, choice probabilities are driven by the absolute difference in tallies, so the probability of choosing A should monotonically increase as the tally difference grows (metric > 0). By summing the within-set differences in choice rates, we strongly capture this divergence."
}
```

## Usage

```json
{
  "prompt_token_count": 3573,
  "candidates_token_count": 569,
  "total_token_count": 7749
}
```
