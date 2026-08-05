# metric_exp00_attempt_03

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

**Trial pairs (n=4):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]

**Rationale:** To distinguish the pure Take The Best (TTB) model from the TTB-Tallying mixture model, this design includes trials where the single highest-validity cue strongly conflicts with the simple sum of features (Tallying). In the pure TTB model, the number of lower-validity cues opposing the choice has zero effect on the choice probabilities; only the top discriminating cue and the global noise parameters matter. In the mixture model, a strong opposition from Tallying (e.g., 1 feature vs. 4 features) will systematically drag the choice probability toward the Tallying favorite or toward 50/50, depending on the mixture weight. We also include a trial where Tallying predicts a tie while TTB makes a deterministic choice, isolating the mixture model's tendency to dilute TTB's confidence with Tallying's uncertainty.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
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
**Description:** Decision makers maintain a repertoire of strategies, specifically Take The Best (TTB) and Tallying. On any given decision, they select between these strategies probabilistically, relying on TTB with probability w_ttb and Tallying with probability 1 - w_ttb. This strategy selection mixture allows for both strong non-compensatory choices and occasional compensatory behavior depending on individual tendencies. To capture the empirical dominance of TTB in certain setups, the probability of selecting TTB is constrained to be at least 0.5.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.2]
- w_ttb: [0.5, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Identify the TTB winner
    cue_order = np.argsort(-val, kind="stable").tolist()
    ttb_winner = None
    for j in cue_order:
        if a[j] > b[j]:
            ttb_winner = 0
            break
        if b[j] > a[j]:
            ttb_winner = 1
            break
            
    ttb_scores = np.array([0.0, 0.0])
    if ttb_winner == 0:
        ttb_scores = np.array([1.0, 0.0])
    elif ttb_winner == 1:
        ttb_scores = np.array([0.0, 1.0])
        
    # Count total wins for each option (Tallying)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    tally_scores = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_ttb = float(parameters["w_ttb"])
    
    # TTB probabilities
    z_ttb = beta * (ttb_scores - np.max(ttb_scores))
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / np.sum(e_ttb)
    
    # Tallying probabilities
    z_tally = beta * (tally_scores - np.max(tally_scores))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # Mixture of strategies
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
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
[0] rationale: In the pure Take The Best (TTB) model, the number of contradicting features (Tallying score) has no effect on the choice probability; the model relies exclusively on the top discriminating cue. Therefore, the rate of TTB-consistent choices should be identical across all trials, regardless of whether Tallying strongly opposes TTB (Trials 1, 2, 4) or is perfectly neutral (Trial 3). In contrast, the competing mixture model explicitly blends TTB with Tallying. When Tallying opposes TTB, it drags the choice probability toward the TTB loser; when Tallying is neutral, it dilutes the choice toward 50/50. This metric calculates the difference in TTB-consistency between neutral-Tallying trials and opposing-Tallying trials. The pure TTB model predicts this difference is zero, while the mixture model predicts a positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    ttb_winner = [0 if a[0] > b[0] else 1 for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    ttb_consistent = (data['response'] == ttb_winner)
    
    sum_a = [sum(a) for a in data['option_a_ratings']]
    sum_b = [sum(b) for b in data['option_b_ratings']]
    
    is_tie = [a == b for a, b in zip(sum_a, sum_b)]
    
    df = pd.DataFrame({'ttb_consistent': ttb_consistent, 'is_tie': is_tie})
    
    rate_tie = df[df['is_tie']]['ttb_consistent'].mean()
    rate_oppose = df[~df['is_tie']]['ttb_consistent'].mean()
    
    # If for some reason a subject is missing one of the conditions, fallback to 0
    if pd.isna(rate_tie) or pd.isna(rate_oppose):
        return 0.0
        
    return float(rate_tie - rate_oppose)

outcome: self_sim=0.0247 (var=0.0078) adversary_sim=0.0889 (var=0.0091) welch_t=-2.468 p=0.01723 (N=25, alpha=0.01) -> reject

[1] rationale: By isolating the trials where Tallying most strongly opposes TTB (Trials 1 and 2, where the difference in tally scores is 3) versus where Tallying is perfectly neutral (Trial 3, where the difference is 0), we maximize the predicted contrast between the two models. Pure TTB predicts a difference of exactly 0, whereas the mixture model predicts a robust positive difference because the strong Tallying opposition in Trials 1 and 2 pulls the choice probabilities away from the TTB winner much more than in neutral trials.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # TTB always relies on the first feature (validity 0.95)
    ttb_winner = [0 if a[0] > b[0] else 1 for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    ttb_consistent = (data['response'] == ttb_winner)
    
    sum_a = [sum(a) for a in data['option_a_ratings']]
    sum_b = [sum(b) for b in data['option_b_ratings']]
    
    abs_diff = [abs(a - b) for a, b in zip(sum_a, sum_b)]
    
    df = pd.DataFrame({'ttb_consistent': ttb_consistent, 'abs_diff': abs_diff})
    
    # Trial 3 has a difference in sums of 0 (Tallying is neutral)
    rate_tie = df[df['abs_diff'] == 0]['ttb_consistent'].mean()
    
    # Trials 1 and 2 have a difference in sums of 3 (Tallying strongly opposes TTB)
    rate_extreme = df[df['abs_diff'] == 3]['ttb_consistent'].mean()
    
    if pd.isna(rate_tie) or pd.isna(rate_extreme):
        return 0.0
        
    return float(rate_tie - rate_extreme)
outcome: self_sim=0.0271 (var=0.0079) adversary_sim=0.0938 (var=0.0103) welch_t=-2.467 p=0.01731 (N=25, alpha=0.01) -> reject

[2] rationale: In the pure Take The Best (TTB) model, the presence of opposing features (Tallying) has no effect on choice probabilities, so the difference in TTB-consistency between neutral trials and extreme-opposition trials is zero. The mixture model predicts a positive difference because Tallying pulls choices away from the TTB winner when they conflict. Previous absolute difference metrics failed due to high between-subject variance driven by individual differences in noise (beta and epsilon). By normalizing the difference by the sum of the rates, we scale out the subject's baseline noise level, tightening the within-theory distribution and increasing the statistical power to discriminate the two models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Identify the Take The Best (TTB) winner for each trial (based on the first cue)
    ttb_winner = [0 if a[0] > b[0] else 1 for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    ttb_consistent = (data['response'] == ttb_winner)
    
    # Calculate the Tallying score difference
    sum_a = [sum(a) for a in data['option_a_ratings']]
    sum_b = [sum(b) for b in data['option_b_ratings']]
    abs_diff = [abs(a - b) for a, b in zip(sum_a, sum_b)]
    
    df = pd.DataFrame({'ttb_consistent': ttb_consistent, 'abs_diff': abs_diff})
    
    # Trial 3: Tallying is perfectly neutral (difference of 0)
    rate_tie = df[df['abs_diff'] == 0]['ttb_consistent'].mean()
    
    # Trials 1 and 2: Tallying strongly opposes TTB (difference of 3)
    rate_extreme = df[df['abs_diff'] == 3]['ttb_consistent'].mean()
    
    if pd.isna(rate_tie) or pd.isna(rate_extreme) or (rate_tie + rate_extreme) == 0:
        return 0.0
        
    # Use a normalized difference to reduce between-subject variance caused by global noise parameters
    return float((rate_tie - rate_extreme) / (rate_tie + rate_extreme))
outcome: self_sim=0.0161 (var=0.0033) adversary_sim=0.0612 (var=0.0052) welch_t=-2.441 p=0.01859 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n    \n    # Identify the Take The Best (TTB) winner for each trial based on the highest validity cue (index 0)\n    ttb_winner = [0 if a[0] > b[0] else 1 for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]\n    ttb_consistent = (data['response'] == ttb_winner)\n    \n    # Calculate the Tallying score difference to identify trial types\n    sum_a = [sum(a) for a in data['option_a_ratings']]\n    sum_b = [sum(b) for b in data['option_b_ratings']]\n    abs_diff = [abs(a - b) for a, b in zip(sum_a, sum_b)]\n    \n    df = pd.DataFrame({'ttb_consistent': ttb_consistent, 'abs_diff': abs_diff})\n    \n    # Trial 3: Tallying is perfectly neutral (difference of 0)\n    df_tie = df[df['abs_diff'] == 0]\n    k1 = df_tie['ttb_consistent'].sum()\n    n1 = len(df_tie)\n    \n    # Trials 1 and 2: Tallying strongly opposes TTB (difference of 3)\n    df_extreme = df[df['abs_diff'] == 3]\n    k2 = df_extreme['ttb_consistent'].sum()\n    n2 = len(df_extreme)\n    \n    if n1 == 0 or n2 == 0:\n        return 0.0\n        \n    # Apply Laplace smoothing (Beta(0.5, 0.5) prior) to stabilize variance for deterministic subjects\n    k1_prime = k1 + 0.5\n    n1_prime = n1 + 1.0\n    p1 = k1_prime / n1_prime\n    \n    k2_prime = k2 + 0.5\n    n2_prime = n2 + 1.0\n    p2 = k2_prime / n2_prime\n    \n    # Pooled proportion for the standard error\n    p_pool = (k1_prime + k2_prime) / (n1_prime + n2_prime)\n    variance = p_pool * (1.0 - p_pool) * (1.0 / n1_prime + 1.0 / n2_prime)\n    \n    if variance <= 0:\n        return 0.0\n        \n    # Z-score for the difference in proportions\n    z = (p1 - p2) / np.sqrt(variance)\n    \n    return float(z)",
  "rationale": "Previous metrics failed because the raw difference in TTB consistency (P_tie - P_extreme) has a variance that heavily depends on the subject's baseline noise level. Noisy subjects produce large variance but zero signal, diluting the metric's power. By computing the z-score (standardized difference in proportions) with Laplace smoothing, we divide the difference by the subject's standard error. For the pure TTB model, this z-score follows a standard normal distribution N(0, 1) regardless of the subject's noise level, tightly bounding the between-subject variance to ~1.0. For the mixture model, deterministic subjects (who have a large true difference and small standard error) yield massive z-scores, drastically amplifying the signal. This statistically optimal contrast maximizes the t-statistic between the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 5489,
  "candidates_token_count": 860,
  "total_token_count": 29962
}
```
