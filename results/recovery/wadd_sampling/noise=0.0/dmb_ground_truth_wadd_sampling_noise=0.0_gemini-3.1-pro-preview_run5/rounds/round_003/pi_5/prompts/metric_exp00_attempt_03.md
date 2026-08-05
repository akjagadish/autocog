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

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 6: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 7: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 8: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 9: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 10: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate the pure WADD model from the Strategy Selection (WADD + Probabilistic Tallying) mixture model, this design completely orthogonalizes the WADD evidence difference and the Tallying evidence difference. We include trials where Tallying predicts a strict tie (equal number of winning features) but WADD strongly prefers one option. In a pure WADD model, choice probabilities on these trials are determined by the same gamma and beta parameters as all other trials. However, the mixture model predicts that the Tallying component will drag the choice probability toward 50/50 on these specific trials, creating a systematic 'flattening' of the psychometric function that pure WADD cannot accommodate without misfit on other trials. Additionally, we include trials where WADD evidence is weak or balanced but Tallying evidence is very strong (e.g., 3 wins vs 1 win), forcing pure WADD to stretch its gamma parameter to extremes to mimic the mixture model's sensitivity to simple feature counts.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Strategy Selection (WADD and Probabilistic Tallying): Decision-makers probabilistically alternate between a purely compensatory strategy (Weighted Additive) and a simpler Tallying heuristic on a trial-by-trial basis. The Tallying heuristic is probabilistic, using a softmax over win counts to generate choice probabilities rather than deterministic choices. This mixture allows individuals to exhibit graded sensitivity to cue evidence on some trials while defaulting to unweighted, softer cue-counting on others, effectively explaining both the high tallying agreement in certain environments and the near-zero extremeness differences in others.

**Parameters:**
- w_wadd: [0.0, 1.0]
- gamma: [0.1, 5.0]
- beta: [0.1, 10.0]
- beta_tally: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Stimulus expects shape (2, n_features); got {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # WADD Strategy
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    wadd_weights = val ** gamma
    wadd_scores = np.dot(stim, wadd_weights)
    
    z = beta * (wadd_scores - np.max(wadd_scores))
    e = np.exp(z)
    p_wadd = e / np.sum(e)
    
    # Tallying Strategy (Probabilistic)
    a_wins = float(np.sum(stim[0] > stim[1]))
    b_wins = float(np.sum(stim[1] > stim[0]))
    tally_scores = np.array([a_wins, b_wins])
    
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * (tally_scores - np.max(tally_scores))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
        
    # Mixture
    w_wadd = float(parameters["w_wadd"])
    epsilon = float(parameters["epsilon"])
    
    p_core = w_wadd * p_wadd + (1.0 - w_wadd) * p_tally
    
    n_opts = len(p_core)
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
**Description:** Decision-makers integrate all available information by taking a weighted sum of each option's features, where the weights are subjective transformations of the cue validities. By exponentiating the raw validities by a free parameter gamma, the weighting scheme can smoothly interpolate between equal weighting (Tallying), proportional weighting (raw Weighted Additive), and lexicographic-like steep weighting (Take The Best). Choice probabilities are generated via a softmax over these subjectively weighted sums, combined with a lapse rate. Human behavior is best described by relatively flat (Tally-like) weights combined with substantial choice noise (lower beta).

**Parameters:**
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 2.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(f"validities length {val.shape[0]} != n_features {stim.shape[1]}.")
        
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Calculate the weighted sum of features for each option
    scores = np.dot(stim, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores with max-subtraction for numerical stability
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
[0] rationale: Measures the overall extremeness of choice probabilities across the 10 unique trial types. The Mixture model (Strategy Selection) predicts that choices will be pulled strongly toward 0.5 on trials where Tallying predicts a tie, but pushed to extremes when Tallying and WADD agree. Pure WADD, constrained by a single weighting scheme and noise parameter, cannot simultaneously flatten some trials and extremize others as effectively, leading to a different overall mean absolute deviation from 0.5.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['trial_str'] = data.apply(lambda row: ''.join(map(str, row['option_a_ratings'])) + '_' + ''.join(map(str, row['option_b_ratings'])), axis=1)
    means = data.groupby('trial_str')['response'].mean()
    return float((means - 0.5).abs().mean())
outcome: self_sim=0.1584 (var=0.0034) adversary_sim=0.1624 (var=0.0054) welch_t=-0.214 p=0.8318 (N=25, alpha=0.01) -> reject

[1] rationale: Contrasts Trial 3 and Trial 5 to cleanly separate pure WADD from the Mixture model. In Trial 3, the WADD evidence difference is very small (0.15 favoring B), but Tallying strongly favors B (3 vs 2). In Trial 5, the WADD evidence difference is slightly larger (0.20 favoring A), but Tallying predicts a strict tie (2 vs 2). Pure WADD predicts that the choice probability of the winner in Trial 5 will be slightly higher than in Trial 3. The Mixture model predicts the opposite: Trial 3's choice probability will be boosted by Tallying, while Trial 5's will be dragged toward 0.5. Thus, P(B|Trial 3) - P(A|Trial 5) will be negative under pure WADD but strongly positive under the Mixture model.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t3_mask = data['option_a_ratings'].apply(lambda x: list(x) == [1, 1, 0, 0, 0]) & data['option_b_ratings'].apply(lambda x: list(x) == [0, 0, 1, 1, 1])
    t5_mask = data['option_a_ratings'].apply(lambda x: list(x) == [1, 0, 1, 0, 0]) & data['option_b_ratings'].apply(lambda x: list(x) == [0, 1, 0, 1, 0])
    
    p_b_t3 = data.loc[t3_mask, 'response'].mean() if t3_mask.sum() > 0 else 0.5
    p_a_t5 = 1.0 - data.loc[t5_mask, 'response'].mean() if t5_mask.sum() > 0 else 0.5
    
    return float(p_b_t3 - p_a_t5)
outcome: self_sim=-0.0667 (var=0.1477) adversary_sim=-0.0733 (var=0.0857) welch_t=+0.069 p=0.9453 (N=25, alpha=0.01) -> reject

[2] rationale: Contrasts Trial 1 and Trial 3 to exploit a divergence between WADD and Tallying evidence. In Trial 1, WADD evidence strongly favors B (difference of 0.65) and Tallying favors B by 1 win (2 vs 1). In Trial 3, WADD evidence only weakly favors B (difference of 0.15) but Tallying again favors B by 1 win (3 vs 2). Pure WADD predicts a substantially higher choice probability for B in Trial 1 compared to Trial 3 due to the large gap in weighted evidence. The Strategy Selection (Mixture) model, however, will heavily regress both probabilities toward each other because its Tallying component sees both trials as identical 1-win advantages for B. Thus, P(B|Trial 1) - P(B|Trial 3) will be significantly larger under pure WADD than under the Mixture model.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t1_m = data['option_a_ratings'].apply(lambda x: list(x) == [1,0,0,0,0]) & data['option_b_ratings'].apply(lambda x: list(x) == [0,1,1,0,0])
    t3_m = data['option_a_ratings'].apply(lambda x: list(x) == [1,1,0,0,0]) & data['option_b_ratings'].apply(lambda x: list(x) == [0,0,1,1,1])
    
    p_b_t1 = data.loc[t1_m, 'response'].mean() if t1_m.sum() > 0 else 0.5
    p_b_t3 = data.loc[t3_m, 'response'].mean() if t3_m.sum() > 0 else 0.5
    
    return float(p_b_t1 - p_b_t3)
outcome: self_sim=0.1778 (var=0.0528) adversary_sim=0.1933 (var=0.0834) welch_t=-0.211 p=0.834 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    def get_trial_id(a, b):\n        a_l = list(a)\n        b_l = list(b)\n        if a_l == [1,0,0,0,0] and b_l == [0,1,1,0,0]: return 1\n        if a_l == [1,1,0,0,0] and b_l == [0,0,1,1,1]: return 3\n        if a_l == [0,1,1,0,0] and b_l == [1,0,0,0,0]: return 4\n        if a_l == [0,1,1,1,0] and b_l == [1,0,0,0,1]: return 8\n        if a_l == [0,1,1,0,1] and b_l == [1,0,0,1,0]: return 9\n        return 0\n\n    data['trial_id'] = data.apply(lambda row: get_trial_id(row['option_a_ratings'], row['option_b_ratings']), axis=1)\n    \n    sub_data = data[data['trial_id'] > 0]\n    if len(sub_data) == 0:\n        return 0.0\n        \n    means = sub_data.groupby('trial_id')['response'].mean()\n    extremeness = (means - 0.5).abs()\n    \n    return float(np.var(extremeness))",
  "rationale": "This metric calculates the variance in choice extremeness across a specific subset of trials (Trials 1, 3, 4, 8, and 9). In all of these trials, the Tallying heuristic predicts exactly a 1-win advantage for one of the options (e.g., 2 wins vs 1 win, or 3 wins vs 2 wins). Because Tallying sees these trials as identically strong evidence, the Strategy Selection (Mixture) model will pull the choice probabilities toward a constant extremeness level, reducing the variance across them. In contrast, the pure WADD model evaluates these trials based on weighted evidence differences, which vary wildly across this subset (from a minimal 0.15 difference in Trial 3 to a large 0.75 difference in Trial 8). Consequently, pure WADD predicts a much higher variance in extremeness across these specific trials than the Mixture model."
}
```

## Usage

```json
{
  "prompt_token_count": 4901,
  "candidates_token_count": 601,
  "total_token_count": 10782
}
```
