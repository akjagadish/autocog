# metric_exp01_attempt_02

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
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 3: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 7: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 8: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 9: A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 10: A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate pure WADD from a mixture of WADD and Probabilistic Tallying, we must target the core difference: the mixture model assumes that on a subset of trials, subjects ignore validities entirely and rely on a soft count of winning features. Because pure WADD is formally nested within the mixture model (when w_wadd = 1), the optimal adversarial design for the WADD advocate is to generate a stimulus set where the WADD evidence difference and the Tallying evidence difference are completely orthogonal. By including trials where Tallying predicts a strong advantage for one option but WADD predicts a slight advantage for the other, alongside trials where Tallying predicts a dead tie but WADD predicts a strong advantage, we force the mixture model to attempt to explain the data. If human subjects truly use a unified WADD process with a single non-linear validity transformation (gamma) and noise parameter (beta), the mixture model will be forced to estimate w_wadd near 1, rendering the Tallying component superfluous and confirming the pure WADD theory.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
[0] rationale: Trial 4 and Trial 5 have similar and small WADD score differences under linear weighting (Trial 4 favors B slightly, Trial 5 favors A slightly). However, Tallying strongly favors B in Trial 4 (3 to 2) while predicting a perfect tie in Trial 5 (2 to 2). The mixture model, which includes a Probabilistic Tallying component, will pull Trial 5's choice probability strongly toward 50% while pushing Trial 4's choice probability heavily toward B, creating a large difference in P(A). Pure WADD scales both trials' differences similarly, predicting a smaller gap between the choice probabilities.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 4 and Trial 5 based on their A and B ratings
    # Trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
    # Trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
    
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    trial_4_mask = (data['A_str'] == '11000') & (data['B_str'] == '00111')
    trial_5_mask = (data['A_str'] == '10100') & (data['B_str'] == '01010')
    
    # Calculate probability of choosing A (response == 0)
    p_A_trial_4 = (data[trial_4_mask]['response'] == 0).mean()
    p_A_trial_5 = (data[trial_5_mask]['response'] == 0).mean()
    
    if pd.isna(p_A_trial_4) or pd.isna(p_A_trial_5):
        return 0.0
        
    return float(p_A_trial_4 - p_A_trial_5)
outcome: self_sim=-0.2100 (var=0.0667) adversary_sim=-0.1517 (var=0.0596) welch_t=-0.820 p=0.416 (N=25, alpha=0.01) -> reject

[1] rationale: Trial 1 and Trial 4 provide a strong dissociation between Pure WADD and the Mixture model. Under pure WADD, the utility differences are very similar and close to zero: Trial 1 has a slight advantage for A (+0.10) and Trial 4 has a slight advantage for B (-0.15). Thus, pure WADD predicts P(A) to be near 50% for both, making the difference P(A|T1) - P(A|T4) relatively small. In contrast, Tallying sees Trial 1 as a perfect tie (1 vs 1, predicting exactly 50%) but sees Trial 4 as a clear win for B (2 vs 3). The Mixture model, heavily influenced by Tallying, will therefore show a much larger gap between the choice probabilities of these two trials.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    mask_1 = (data['A_str'] == '10000') & (data['B_str'] == '01000')
    mask_4 = (data['A_str'] == '11000') & (data['B_str'] == '00111')
    
    p_a_1 = (data[mask_1]['response'] == 0).mean()
    p_a_4 = (data[mask_4]['response'] == 0).mean()
    
    if pd.isna(p_a_1) or pd.isna(p_a_4):
        return 0.0
        
    return float(p_a_1 - p_a_4)
outcome: self_sim=0.1250 (var=0.0481) adversary_sim=0.1833 (var=0.0594) welch_t=-0.889 p=0.3783 (N=25, alpha=0.01) -> reject

[2] rationale: Trials 1, 2, and 5 are perfect ties under the Tallying heuristic. The Mixture theory, which probabilistically relies on Tallying, will heavily pull the choice probabilities on these specific trials toward exactly 0.5. In contrast, pure WADD processes these trials using weighted sums that consistently favor Option A (due to the higher validity of the cues A possesses compared to B), leading to predicted probabilities that systematically deviate from 0.5. Measuring the cumulative absolute deviation from 0.5 on these Tally-tie trials robustly discriminates the theories, as the Mixture model exerts a strong regularizing force toward indifference here that Pure WADD lacks.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Tally tie trials
    mask_1 = (data['A_str'] == '10000') & (data['B_str'] == '01000')
    mask_2 = (data['A_str'] == '01000') & (data['B_str'] == '00100')
    mask_5 = (data['A_str'] == '10100') & (data['B_str'] == '01010')
    
    p_1 = (data[mask_1]['response'] == 0).mean()
    p_2 = (data[mask_2]['response'] == 0).mean()
    p_5 = (data[mask_5]['response'] == 0).mean()
    
    if pd.isna(p_1) or pd.isna(p_2) or pd.isna(p_5):
        return 0.0
        
    # Measure the total deviation from 0.5 (indifference) on these trials
    return float(abs(p_1 - 0.5) + abs(p_2 - 0.5) + abs(p_5 - 0.5))
outcome: self_sim=0.1867 (var=0.0314) adversary_sim=0.1233 (var=0.0311) welch_t=+1.267 p=0.2114 (N=25, alpha=0.01) -> reject

[3] rationale: Trial 6 (4 cues vs 1 cue) and Trial 3 (1 cue vs 3 cues) provide the most extreme Tallying differences in the design (+3 for A in T6, and +2 for B in T3). The Mixture model, heavily incorporating the Tallying heuristic, will produce extreme choice probabilities on these trials, resulting in a large difference between P(A|T6) and P(A|T3). The pure WADD model, operating with relatively flat weights and high noise (low beta), will heavily dampen these probabilities toward 0.5, predicting a much smaller gap.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    mask_6 = (data['A_str'] == '01111') & (data['B_str'] == '10000')
    mask_3 = (data['A_str'] == '10000') & (data['B_str'] == '00111')
    
    p_a_6 = (data[mask_6]['response'] == 0).mean()
    p_a_3 = (data[mask_3]['response'] == 0).mean()
    
    if pd.isna(p_a_6) or pd.isna(p_a_3):
        return 0.0
        
    return float(p_a_6 - p_a_3)
outcome: self_sim=0.5933 (var=0.0863) adversary_sim=0.5017 (var=0.0976) welch_t=+1.069 p=0.2906 (N=25, alpha=0.01) -> reject

[4] rationale: Trial 8 pits three weak cues against the single best cue (Tallying strongly favors A by +2, WADD linearly favors A by +1.0). Trial 1 pits the two best cues against three weak cues (Tallying favors B by -1, WADD linearly favors B by -0.15). The Mixture model, driven heavily by Tallying, will produce a very large difference in P(A) between these trials. Pure WADD, operating with high noise and relatively flat weights, will compress these probabilities, predicting a smaller difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    mask_8 = (data['A_str'] == '00111') & (data['B_str'] == '10000')
    mask_1 = (data['A_str'] == '11000') & (data['B_str'] == '00111')
    
    p_a_8 = (data[mask_8]['response'] == 0).mean()
    p_a_1 = (data[mask_1]['response'] == 0).mean()
    
    if pd.isna(p_a_8) or pd.isna(p_a_1):
        return 0.0
        
    return float(p_a_8 - p_a_1)
outcome: self_sim=0.4111 (var=0.0752) adversary_sim=0.3400 (var=0.1325) welch_t=+0.780 p=0.4394 (N=25, alpha=0.01) -> reject

[5] rationale: Trial 8 has a Tallying difference of +2 for Option A (3 cues vs 1 cue), while Trials 7 and 9 have a Tallying difference of +1 for Option A (3 cues vs 2 cues). However, under linear WADD, all three trials have nearly identical utility differences favoring Option A (Trial 8: +1.00; Trials 7 & 9: +0.95). Pure WADD therefore predicts that the choice probability for Option A should be virtually identical across these trials. In contrast, the Mixture model, which relies on Probabilistic Tallying on a subset of trials, will predict a substantially higher probability of choosing Option A in Trial 8 due to the larger Tallying gap. The difference P(A|Trial 8) - P(A|Trials 7 & 9) isolates this divergence, expected to be near zero for Pure WADD and significantly positive for the Mixture model.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Trial 8: Tally diff = +2, WADD diff = +1.00
    m8 = (data['A_str'] == '00111') & (data['B_str'] == '10000')
    
    # Trials 7 and 9: Tally diff = +1, WADD diff = +0.95
    m79 = ((data['A_str'] == '10110') & (data['B_str'] == '01001')) | \
          ((data['A_str'] == '11001') & (data['B_str'] == '00110'))
          
    p8 = (data[m8]['response'] == 0).mean()
    p79 = (data[m79]['response'] == 0).mean()
    
    if pd.isna(p8) or pd.isna(p79):
        return 0.0
        
    return float(p8 - p79)
outcome: self_sim=0.0211 (var=0.0246) adversary_sim=-0.0556 (var=0.0717) welch_t=+1.235 p=0.2241 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    \n    # Trials 7 and 9: Tally diff = +1, WADD diff = +0.95\n    m79 = ((data['A_str'] == '10110') & (data['B_str'] == '01001')) | \\\n          ((data['A_str'] == '11001') & (data['B_str'] == '00110'))\n          \n    # Trial 3: Tally diff = +1, WADD diff = +0.55\n    m3 = (data['A_str'] == '10011') & (data['B_str'] == '01100')\n    \n    p79 = (data[m79]['response'] == 0).mean()\n    p3 = (data[m3]['response'] == 0).mean()\n    \n    if pd.isna(p79) or pd.isna(p3):\n        return 0.0\n        \n    return float(p79 - p3)",
  "rationale": "Trials 3, 7, and 9 all have the exact same Tallying difference (+1 for Option A). However, under linear WADD, Trials 7 and 9 have a much stronger evidence difference favoring A (+0.95) compared to Trial 3 (+0.55). The Pure WADD model predicts a substantially higher probability of choosing A in Trials 7/9 than in Trial 3 because it is fully sensitive to this +0.40 WADD difference. The Mixture model, however, probabilistically defaults to Tallying on a subset of trials. Since Tallying predicts the exact same choice probabilities for all three of these trials, it acts as an anchor, significantly diluting the difference between them. Therefore, the gap P(A|Trials 7 & 9) - P(A|Trial 3) will be large for Pure WADD and noticeably compressed for the Mixture model."
}
```

## Usage

```json
{
  "prompt_token_count": 6805,
  "candidates_token_count": 549,
  "total_token_count": 13515
}
```
