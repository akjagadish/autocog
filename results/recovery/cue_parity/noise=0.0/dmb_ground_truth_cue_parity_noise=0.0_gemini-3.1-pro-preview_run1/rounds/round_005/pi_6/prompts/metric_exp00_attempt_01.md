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

**Trial pairs (n=6):**
  trial 1: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 6: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Additive Utility with a power-law transformation of chance-centered validities) from the Competing Theory (a mixture of Tallying and WADD with Bayesian log-odds weights), we exploit the divergent scaling properties of their weighting functions. The log-odds transformation in the Competing Theory's WADD component grows extremely fast as validity approaches 1.0, while the Advocated Theory's power-law function allows for a broader, smoother interpolation between equal weighting (Tallying) and Take-The-Best (TTB) via the gamma parameter. Furthermore, the Competing Theory explicitly mixes in a pure Tallying component. By pitting options with single high-validity features against options with multiple lower-validity features, we can create trials where the log-odds WADD and pure Tallying strongly conflict, forcing the Competing Theory to predict a diluted, mixture-driven choice probability. In contrast, the Advocated Theory can predict stark, deterministic choices depending on the gamma parameter, without the structural constraint of mixing two distinct choice distributions.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People use a single-stage Additive Utility evaluation where features are weighted by a power-law transformation of their chance-centered objective validities. By avoiding artificial weight normalization and allowing for sufficiently high softmax temperatures and power parameters, the decision-making process can smoothly and stably interpolate between Tallying (equal weights), proportional weighting, and highly deterministic Take-The-Best behavior (where the most valid cues dominate completely).

**Parameters:**
- beta: [0.1, 50.0]
- gamma: [0.0, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Transform validities using a simple power law on chance-centered values
    # gamma=0 -> Tallying, gamma=1 -> WADD, gamma>1 -> TTB
    centered_val = val - 0.5
    weights = (centered_val + 1e-6) ** gamma
    
    scores = np.array([np.sum(a * weights), np.sum(b * weights)])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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
**Description:** Mixture of Strategies with Bayesian Weights: The decision-making population consists of individuals who employ a mixture of two distinct strategies. A proportion of choices (governed by parameter 'alpha') is made using a simple non-compensatory Tallying heuristic, which completely ignores feature validities and simply counts the number of features where one option strictly dominates the other. The remaining proportion of choices ('1 - alpha') is made using a fully compensatory Weighted Additive (WADD) strategy, where features are weighted by their Bayesian log-odds (log(v / (1-v))). This non-linear transformation naturally stretches the weights of highly valid features, allowing the model to capture strong compensatory preferences while interpolating with pure heuristic behavior.

**Parameters:**
- alpha: [0.0, 1.0]
- beta_tally: [0.1, 10.0]
- beta_wadd: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    alpha = float(parameters["alpha"])
    epsilon = float(parameters["epsilon"])
    
    # Strategy 1: Tallying (ignores validities, counts strict wins)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # Strategy 2: Weighted Additive (WADD) with Bayesian log-odds weights
    val = np.asarray(parameters["validities"], dtype=float)
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    w = np.log(val_clipped / (1.0 - val_clipped))
    scores_wadd = np.array([np.sum(a * w), np.sum(b * w)])
    
    beta_wadd = float(parameters["beta_wadd"])
    z_wadd = beta_wadd * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # Mixture of the two strategies
    p_core = alpha * p_tally + (1.0 - alpha) * p_wadd
    
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
[0] rationale: This metric calculates the probability of choosing the option with fewer, but more valid cues ([1, 1, 0, 0, 0]) over the option with more, but less valid cues ([0, 0, 1, 1, 1]) in Trial 6. Under the Competing Theory, the Tallying component strongly favors the [0, 0, 1, 1, 1] option while the WADD component strongly favors the [1, 1, 0, 0, 0] option. Because the Competing Theory mixes these two strategies, it predicts a diluted choice probability (averaging around 0.5 depending on the mixture parameter alpha). In contrast, the Advocated Theory consistently favors the [1, 1, 0, 0, 0] option across its parameter space, predicting a choice proportion very close to 1.0. This structural difference yields a large, stable gap between the theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    is_A_11000 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    is_B_00111 = data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    is_A_00111 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    is_B_11000 = data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    
    mask1 = is_A_11000 & is_B_00111
    mask2 = is_A_00111 & is_B_11000
    
    target_trials = data[mask1 | mask2]
    if len(target_trials) == 0:
        return 0.0
        
    chose_11000 = ((target_trials['response'] == 0) & mask1) | ((target_trials['response'] == 1) & mask2)
    return float(chose_11000.mean())
outcome: self_sim=0.0998 (var=0.0012) adversary_sim=0.0877 (var=0.0014) welch_t=+1.181 p=0.2435 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    is_A_01111 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 1, 1))\n    is_B_10000 = data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))\n    is_A_10000 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))\n    is_B_01111 = data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 1, 1))\n    \n    mask1 = is_A_01111 & is_B_10000\n    mask2 = is_A_10000 & is_B_01111\n    \n    target_trials = data[mask1 | mask2]\n    if len(target_trials) == 0:\n        return 0.0\n        \n    chose_10000 = ((target_trials['response'] == 1) & mask1) | ((target_trials['response'] == 0) & mask2)\n    return float(chose_10000.mean())",
  "rationale": "In Trial 1, the options are [0, 1, 1, 1, 1] and [1, 0, 0, 0, 0]. The Competing Theory strongly prefers [0, 1, 1, 1, 1] because both Tallying (4 vs 1) and WADD with Bayesian log-odds weights (sum of weights 3.65 vs 2.94) favor it. Thus, it predicts choosing [1, 0, 0, 0, 0] only due to random noise. In contrast, the Advocated Theory uses a power-law transformation of chance-centered validities. For gamma > 2.5, the single most valid cue outweighs the sum of the other four, leading to a strong preference for [1, 0, 0, 0, 0]. Since gamma ranges up to 10, the Advocated Theory predicts a much higher choice proportion for [1, 0, 0, 0, 0] than the Competing Theory."
}
```

## Usage

```json
{
  "prompt_token_count": 4126,
  "candidates_token_count": 589,
  "total_token_count": 8504
}
```
