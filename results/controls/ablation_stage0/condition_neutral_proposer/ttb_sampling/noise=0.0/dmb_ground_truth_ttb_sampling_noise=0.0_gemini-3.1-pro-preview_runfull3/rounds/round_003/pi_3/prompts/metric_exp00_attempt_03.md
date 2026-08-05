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
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 6: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 9: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 10: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Rationale:** To maximally distinguish Take-The-Best (TTB) from Probabilistic Cue Search (PCS), we manipulate the number and validities of lower-ranking discriminating cues. TTB bases its decision solely on the single highest-validity discriminating cue, ignoring all others. In contrast, PCS calculates choice probabilities proportionally based on the validities of all discriminating cues (a Plackett-Luce choice rule). We use a 5-feature design with linearly spaced validities. The critical trials hold the top discriminating cue constant while varying the number and validities of the remaining discriminating cues that favor the alternative option. TTB predicts the exact same choice probabilities across these variations, whereas PCS predicts shifts in probability as the competing cues accumulate or change.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** People compare two options using the Take-The-Best (TTB) heuristic. TTB is a fast-and-frugal lexicographic strategy that searches through cues in descending order of their validity. The first cue that discriminates between the two options completely determines the choice, and all remaining lower-validity cues are ignored. If no cue discriminates, the decision maker guesses randomly. Response noise enters through a softmax over the binary TTB outcome with inverse temperature beta, plus an independent lapse rate epsilon.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.zeros(2)
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            scores[0] = 1.0
            break
        elif stim[1, idx] > stim[0, idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
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
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Probabilistic Cue Search (PCS): Decision makers evaluate options by searching through cues sequentially. However, unlike the strict deterministic order of Take-The-Best, the search order is probabilistic. The probability of retrieving and evaluating a cue is proportional to an exponential function of its validity (a Plackett-Luce choice process). Because non-discriminating cues simply prompt the decision maker to continue searching, the probability that the decision is ultimately determined by a specific discriminating cue is exactly proportional to its weight relative to all other discriminating cues. This model elegantly bridges random search (Tallying) and deterministic search (TTB) through a single search-temperature parameter.

**Parameters:**
- gamma: [0.0, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Find discriminating cues
    diff = stim[0] - stim[1]
    discrim_mask = diff != 0
    
    if not np.any(discrim_mask):
        p_core = np.array([0.5, 0.5])
    else:
        # Due to the Plackett-Luce property of sampling without replacement,
        # the probability that a particular discriminating cue is encountered 
        # before any other discriminating cue is simply its proportional weight
        # among the set of discriminating cues. Non-discriminating cues drop out.
        discrim_validities = validities[discrim_mask]
        max_v = np.max(discrim_validities)
        
        # Calculate weights for discriminating cues with numerical stability
        weights = np.zeros_like(validities)
        weights[discrim_mask] = np.exp(gamma * (validities[discrim_mask] - max_v))
        
        w_A = np.sum(weights[diff > 0])
        w_B = np.sum(weights[diff < 0])
        
        total_w = w_A + w_B
        if total_w == 0:
            p_core = np.array([0.5, 0.5])
        else:
            p_core = np.array([w_A / total_w, w_B / total_w])
            
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
[0] rationale: Measures the difference in the probability of choosing Option A between Trial 5 (where B only has one weak cue) and Trial 1 (where B has four cues, including the second best). TTB predicts this difference is 0, because the top cue exclusively dictates the choice in both cases. PCS predicts a large positive difference, as the accumulation of competing cues in Trial 1 pulls probability away from Option A.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    is_trial_1 = (a_str == '10000') & (b_str == '01111')
    is_trial_5 = (a_str == '10000') & (b_str == '00001')
    
    p_a_trial_1 = (data[is_trial_1]['response'] == 0).mean()
    p_a_trial_5 = (data[is_trial_5]['response'] == 0).mean()
    
    if pd.isna(p_a_trial_1) or pd.isna(p_a_trial_5):
        return 0.0
        
    return float(p_a_trial_5 - p_a_trial_1)
outcome: self_sim=0.0133 (var=0.0309) adversary_sim=0.0867 (var=0.0545) welch_t=-1.255 p=0.2161 (N=25, alpha=0.01) -> reject

[1] rationale: This metric calculates the difference in compliance with the Take-The-Best (TTB) choice between two sets of trials: 'aligned/neutral' trials (where lower-validity cues either support the TTB choice or are balanced) and 'conflict' trials (where an accumulation of lower-validity cues strongly opposes the TTB choice). For TTB, choice probability depends strictly on the top discriminating cue, which is structurally identical across both trial sets, leading to an expected difference of exactly 0. For Probabilistic Cue Search (PCS), the opposing lower-validity cues in the conflict trials dynamically pull the choice probability away from the TTB option, resulting in a strictly positive difference. By aggregating across multiple trials, this metric minimizes per-subject variance and robustly cancels out base rates of noise (epsilon) and temperature (beta/gamma).
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    pair_str = a_str + "_" + b_str
    
    ttb_choices = {
        '10000_01111': 0,
        '10111_01111': 0,
        '11000_10111': 0,
        '10001_01000': 0,
        '10000_00001': 0,
        '00100_00011': 0,
        '01100_10000': 1,
        '11000_00111': 0,
        '01000_00111': 0,
        '10100_01011': 0
    }
    
    conflict_pairs = {
        '10000_01111',
        '11000_10111',
        '00100_00011',
        '01100_10000',
        '11000_00111',
        '01000_00111',
        '10100_01011'
    }
    
    aligned_pairs = {
        '10111_01111',
        '10001_01000',
        '10000_00001'
    }
    
    data['ttb_choice'] = pair_str.map(ttb_choices)
    data['is_ttb_match'] = (data['response'] == data['ttb_choice']).astype(float)
    
    is_conflict = pair_str.isin(conflict_pairs)
    is_aligned = pair_str.isin(aligned_pairs)
    
    conflict_match = data.loc[is_conflict, 'is_ttb_match'].mean()
    aligned_match = data.loc[is_aligned, 'is_ttb_match'].mean()
    
    if pd.isna(conflict_match) or pd.isna(aligned_match):
        return 0.0
        
    return float(aligned_match - conflict_match)
outcome: self_sim=0.0146 (var=0.0055) adversary_sim=0.0334 (var=0.0111) welch_t=-0.730 p=0.4693 (N=25, alpha=0.01) -> reject

[2] rationale: This metric calculates the covariance between the subject's compliance with the Take-The-Best (TTB) choice and the number of lower-validity cues opposing that choice across all 90 trials. For TTB, choice probability is determined strictly by the single best discriminating cue, which is structurally identical across all 10 trial types. Therefore, TTB predicts a constant probability of choosing the TTB option (driven only by constant beta and epsilon noise), yielding an expected covariance of exactly 0. For Probabilistic Cue Search (PCS), the probability of choosing the TTB option dynamically decreases as the number and weight of opposing cues increase. By computing the covariance across all 90 trials, we optimally pool the data to minimize binomial noise while cleanly isolating the signature of PCS (a strictly negative covariance) from TTB (zero covariance), perfectly robust to the high base rates of noise (epsilon).
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))
    pair_str = a_str + "_" + b_str
    
    # The choice Take-The-Best (TTB) would make for each unique trial pair
    ttb_choices = {
        '10000_01111': 0,
        '10111_01111': 0,
        '11000_10111': 0,
        '10001_01000': 0,
        '10000_00001': 0,
        '00100_00011': 0,
        '01100_10000': 1,
        '11000_00111': 0,
        '01000_00111': 0,
        '10100_01011': 0
    }
    
    # The number of discriminating cues that oppose the TTB choice
    num_opposing = {
        '10000_01111': 4,
        '10111_01111': 1,
        '11000_10111': 3,
        '10001_01000': 1,
        '10000_00001': 1,
        '00100_00011': 2,
        '01100_10000': 2,
        '11000_00111': 3,
        '01000_00111': 3,
        '10100_01011': 3
    }
    
    data['ttb_choice'] = pair_str.map(ttb_choices)
    data['is_ttb_match'] = (data['response'] == data['ttb_choice']).astype(float)
    data['opposing'] = pair_str.map(num_opposing)
    
    # Compute the covariance between TTB compliance and the number of opposing cues
    cov = data['is_ttb_match'].cov(data['opposing'])
    
    if pd.isna(cov):
        return 0.0
        
    return float(cov)
outcome: self_sim=-0.0049 (var=0.0016) adversary_sim=-0.0068 (var=0.0023) welch_t=+0.147 p=0.8837 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))\n    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, (int(v) for v in x))))\n    data['pair_str'] = a_str + \"_\" + b_str\n    \n    ttb_choices = {\n        '10000_01111': 0,\n        '10111_01111': 0,\n        '11000_10111': 0,\n        '10001_01000': 0,\n        '10000_00001': 0,\n        '00100_00011': 0,\n        '01100_10000': 1,\n        '11000_00111': 0,\n        '01000_00111': 0,\n        '10100_01011': 0\n    }\n    \n    data['ttb_choice'] = data['pair_str'].map(ttb_choices)\n    data['is_ttb_match'] = (data['response'] == data['ttb_choice']).astype(float)\n    \n    subject_scores = []\n    for subj, subj_df in data.groupby('subject_id'):\n        trial_means = subj_df.groupby('pair_str')['is_ttb_match'].mean()\n        p_overall = subj_df['is_ttb_match'].mean()\n        \n        v_obs = trial_means.var(ddof=1)\n        v_exp = p_overall * (1.0 - p_overall) / 9.0\n        \n        if pd.notna(v_obs) and pd.notna(v_exp):\n            subject_scores.append(v_obs - v_exp)\n            \n    if not subject_scores:\n        return 0.0\n        \n    return float(np.mean(subject_scores))",
  "rationale": "For Take-The-Best (TTB), the probability of choosing the TTB option depends strictly on the top discriminating cue, which is structurally identical across all 10 trial types. Thus, TTB predicts exactly zero variance in the true choice probabilities across trial types; the observed variance in match rates is entirely due to binomial sampling noise. For Probabilistic Cue Search (PCS), the true choice probabilities vary widely depending on the number and weights of opposing cues. This metric calculates the excess variance in TTB match rates across trial types beyond what is expected from binomial noise. For TTB, this value tightly clusters around 0. For PCS, it is strictly positive, robustly distinguishing the theories while controlling for each subject's base level of noise."
}
```

## Usage

```json
{
  "prompt_token_count": 5706,
  "candidates_token_count": 736,
  "total_token_count": 13188
}
```
