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
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.55, 0.5, 0.5]

**Trial pairs (n=5):**
  trial 1: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 2: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  trial 3: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 5: A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]

**Rationale:** This design quantitatively dissociates the Sequential Evidence Accumulation (SEA) model from the Strategy Mixture (Take-The-Best + Tallying) model by exploiting their different tie-breaking and stopping rules. We use 6 features with a specific validity profile: one very high validity (0.95), two high validities (0.90, 0.85), and three low validities (0.55, 0.50, 0.50). In Trials 1 and 2, Tallying sees a perfect tie (3 vs 3), forcing the Mixture model to default entirely to TTB, which strictly follows the 0.95 cue. However, SEA (with a high threshold and gamma near 1) integrates all evidence and correctly calculates that the sum of the two high validities plus one low validity (0.90+0.85+0.55=2.30) outweighs the highest validity plus two low validities (0.95+0.50+0.50=1.95), leading to the opposite prediction. In Trials 3 and 4, one option has a massive Tallying advantage (5 vs 1) but loses on the most valid cue. The Mixture model overwhelmingly predicts the Tallying winner. In contrast, SEA with a low threshold (e.g., theta = 0.8) will terminate search immediately after the first cue, predicting the TTB winner and ignoring the remaining cues entirely.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



## ADVOCATED THEORY
**Description:** Sequential Evidence Accumulation: Decision-makers inspect cues sequentially in order of validity, accumulating evidence for the favored option. The evidence contributed by each cue is its validity scaled by a non-linear parameter gamma. Search terminates when the absolute evidence difference reaches a threshold theta, or when all cues are exhausted. A choice is then made based on the accumulated evidence with softmax noise. This unified mechanism smoothly interpolates between Take-The-Best (low threshold), Tallying (high threshold, gamma=0), and Weighted Additive (high threshold, gamma>0).

**Parameters:**
- theta: [0.0, 3.0]
- gamma: [0.0, 1.0]
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    theta = float(parameters["theta"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale validities
    v = np.power(val, gamma)
    
    # Order cues by descending validity
    cue_order = np.argsort(-val, kind="stable")
    
    E = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        if diff != 0:
            E += v[j] * diff
            if abs(E) >= theta:
                break
                
    # E > 0 means A is favored; E < 0 means B is favored
    scores = np.array([E, 0.0])
    
    # Softmax choice
    z = beta * (scores - np.max(scores))
    e_vals = np.exp(z)
    p = e_vals / np.sum(e_vals)
    
    # Add lapse rate
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
**Description:** Strategy Mixture (Take-The-Best and Tallying with Tallying Bias): Decision-makers are heterogeneous in their strategy use, probabilistically switching between strategies. On any given choice, a subject has a probability 'p_ttb' of applying a one-reason lexicographic heuristic (Take-The-Best) and a probability '1 - p_ttb' of applying an equal-weight compensatory heuristic (Tallying). Empirical data suggests that people generally favor Tallying over Take-The-Best in these environments, so the probability of using Take-The-Best is bounded between 10% and 50%, ensuring the mixture slightly favors Tallying to better match observed aggregate choice probabilities.

**Parameters:**
- p_ttb: [0.1, 0.5]
- beta_ttb: [0.1, 20.0]
- beta_tally: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Take-The-Best (TTB) Prediction
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb_core = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb_core = e_ttb / np.sum(e_ttb)
        
    # Tallying Prediction
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally_core = e_tally / np.sum(e_tally)
    
    # Mixture
    p_ttb_weight = float(parameters["p_ttb"])
    p_mixed = p_ttb_weight * p_ttb_core + (1.0 - p_ttb_weight) * p_tally_core
    
    # Lapse rate
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
    
    return p_final
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
[0] rationale: This metric calculates the difference in the probability of choosing the Take-The-Best (TTB) winner between Trials 1/2 and Trials 3/4. In Trials 1/2, the Mixture model defaults to TTB because Tallying is tied, leading to a high rate of choosing the TTB winner. In Trials 3/4, the Mixture model overwhelmingly follows Tallying (5 vs 1), leading to a low rate of choosing the TTB winner. Thus, the Mixture model predicts a large positive difference. In contrast, the Sequential Evidence Accumulation (SEA) model's choice depends on its threshold parameter theta, which is fixed per subject: if theta is low, it chooses the TTB winner in BOTH trial types; if theta is high, it integrates all evidence and chooses the non-TTB winner in BOTH trial types. Therefore, SEA predicts this difference to be near zero regardless of its parameters.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    a_v0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_v0 = data['option_b_ratings'].apply(lambda x: x[0])
    
    ttb_winner = (b_v0 > a_v0).astype(int)
    chose_ttb = (data['response'] == ttb_winner).astype(float)
    
    is_t12 = (a_sums == 3) & (b_sums == 3)
    is_t34 = ((a_sums == 5) & (b_sums == 1)) | ((a_sums == 1) & (b_sums == 5))
    
    if is_t12.sum() == 0 or is_t34.sum() == 0:
        return 0.0
        
    return float(chose_ttb[is_t12].mean() - chose_ttb[is_t34].mean())
outcome: self_sim=0.1542 (var=0.0355) adversary_sim=0.2537 (var=0.0123) welch_t=-2.275 p=0.02853 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    # The most valid cue (cue 0) always discriminates in this design.\n    # Determine the Take-The-Best (TTB) winner for each trial (0 for A, 1 for B).\n    a_v0 = data['option_a_ratings'].apply(lambda x: x[0])\n    b_v0 = data['option_b_ratings'].apply(lambda x: x[0])\n    ttb_winner = (b_v0 > a_v0).astype(int)\n    \n    # Record whether the subject chose the TTB winner\n    chose_ttb = (data['response'] == ttb_winner).astype(float)\n    \n    # Create a safe, hashable string identifier for each unique trial type\n    trial_id = data['option_a_ratings'].apply(lambda x: \"\".join([str(v) for v in x])) + \"_\" + \\\n               data['option_b_ratings'].apply(lambda x: \"\".join([str(v) for v in x]))\n               \n    df_temp = pd.DataFrame({\n        'subject_id': data['subject_id'],\n        'trial_id': trial_id,\n        'chose_ttb': chose_ttb\n    })\n    \n    # Calculate the proportion of times the TTB winner was chosen per subject, per trial type\n    means = df_temp.groupby(['subject_id', 'trial_id'])['chose_ttb'].mean()\n    \n    # Metric: Mean absolute deviation of these choice probabilities from 0.5\n    return float((means - 0.5).abs().mean())\n",
  "rationale": "The Sequential Evidence Accumulation (SEA) model uses a unified deterministic mechanism (with softmax noise). Whether it stops early (acting like TTB) or integrates all cues (acting like Tallying), it strongly commits to one option per trial type, producing extreme choice probabilities (near 0 or 1). In contrast, the Strategy Mixture model posits that subjects stochastically switch between TTB and Tallying from trial to trial. Because the probability of using TTB is constrained between 10% and 50%, the Mixture model's predicted choice probabilities are an intermediate blend of the two heuristics. By calculating the empirical probability of choosing the TTB winner for each trial type and measuring its absolute deviation from 0.5, we can robustly separate the models: SEA will show a high deviation (~0.4) due to its extreme probabilities, while the Mixture model will show a significantly lower deviation (~0.15) due to its intermediate, blended probabilities."
}
```

## Usage

```json
{
  "prompt_token_count": 4248,
  "candidates_token_count": 629,
  "total_token_count": 24514
}
```
