# metric_exp01_attempt_00

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
**Validities (n_features=7):** [0.96, 0.91, 0.82, 0.73, 0.64, 0.58, 0.53]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0]
  trial 2: A=[0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  trial 6: A=[0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]
  trial 8: A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1]

**Rationale:** This design utilizes 7 features with a wide and non-linear spread of validities to definitively separate the unweighted counting mechanism of Tallying from the log-odds validity weighting of WADD. Tallying ignores validity entirely, counting only the number of features favoring each option. WADD evaluates the weighted sum of feature differences. By including trials where one option wins on a small number of highly valid features while the other wins on a larger number of low-validity features, the models predict choices in opposite directions. Furthermore, 'tie' trials for Tallying (where both options win on an equal number of features) will yield 50/50 guessing under Tallying but deterministic, validity-driven preferences under WADD. This 7-feature structure avoids the already-explored 5- and 6-feature paradigms while providing maximum quantitative divergence.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People make decisions by computing a weighted sum of feature differences, where the weights are proportional to the log-odds of the cue validities. A parameter gamma interpolates between equal weighting (Tallying, gamma=0) and full log-odds weighting (gamma=1). This Compensatory Weighted Additive (WADD) strategy allows multiple lower-validity cues to outweigh a single high-validity cue, while capturing slight validity-driven effects that pure Tallying misses. Response noise is modeled via a softmax function and an independent lapse rate.

**Parameters:**
- beta: [0.0, 1.0]
- epsilon: [0.4, 0.9]
- gamma: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Calculate log-odds of validities
    clipped_val = np.clip(val, 0.5001, 0.9999)
    log_odds = np.log(clipped_val / (1.0 - clipped_val))
    
    # Apply gamma to scale the log-odds (gamma=0 -> equal weights/Tallying)
    weights = np.maximum(log_odds, 1e-9) ** gamma
    
    # Compute weighted sum of feature differences (strict wins)
    a_wins = np.sum((a > b) * weights)
    b_wins = np.sum((b > a) * weights)
    scores = np.array([a_wins, b_wins])
    
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
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.0, 1.0]
- epsilon: [0.4, 0.9]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
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
[0] rationale: This metric calculates the proportion of trials where the subject's choice aligns with the choice favored by the Compensatory Weighted Additive (WADD) strategy. Given the experimental design, WADD and Tallying make conflicting predictions on almost all trials (either Tallying predicts a 50/50 guess while WADD has a strong preference, or Tallying predicts the exact opposite of WADD). Consequently, data generated by WADD will yield a high proportion of matches (>0.5), whereas data generated by Tallying will yield a low proportion of matches (~0.33).
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    log_odds = np.log(val / (1.0 - val))
    
    wadd_matches = 0
    total = len(data)
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_score = np.sum((a > b) * log_odds)
        b_score = np.sum((b > a) * log_odds)
        
        if a_score > b_score:
            wadd_choice = 0
        elif b_score > a_score:
            wadd_choice = 1
        else:
            wadd_choice = -1
            
        if row['response'] == wadd_choice:
            wadd_matches += 1
            
    return float(wadd_matches / total)
outcome: self_sim=0.5189 (var=0.0031) adversary_sim=0.4978 (var=0.0028) welch_t=+1.372 p=0.1764 (N=25, alpha=0.01) -> reject

[1] rationale: By strictly isolating 'Reversal' trials—where the option with more winning features is different from the option with a higher log-odds weighted sum—we can cleanly separate the two theories. On these trials, Tallying will consistently prefer the option with more features (scoring low on this metric, bounded only by the lapse rate), whereas WADD will prefer the option with fewer but higher-validity features (scoring significantly higher, depending on gamma). Filtering out agreement and tie-breaker trials removes the noise that diluted the previous metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Approximate log-odds for validities [0.95, 0.85, 0.75, 0.65, 0.55]
    log_odds = np.array([2.9444, 1.7346, 1.0986, 0.6190, 0.2007])
    
    wadd_choices = 0
    reversal_trials = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        a_wadd = np.sum((a > b) * log_odds)
        b_wadd = np.sum((b > a) * log_odds)
        
        # Identify Reversal trials: where the simple feature count (Tallying)
        # points in the opposite direction of the log-odds weighted sum (WADD).
        if (a_wins - b_wins) * (a_wadd - b_wadd) < 0:
            reversal_trials += 1
            wadd_favors_A = a_wadd > b_wadd
            
            if wadd_favors_A and row['response'] == 0:
                wadd_choices += 1
            elif not wadd_favors_A and row['response'] == 1:
                wadd_choices += 1
                
    if reversal_trials == 0:
        return 0.5
        
    return float(wadd_choices / reversal_trials)
outcome: self_sim=0.5065 (var=0.0077) adversary_sim=0.4500 (var=0.0095) welch_t=+2.151 p=0.03658 (N=25, alpha=0.01) -> reject

[2] rationale: This metric isolates 'Tie-Breaker' trials where both options win on an equal number of features. On these trials, Tallying predicts a perfect 50/50 guess regardless of its noise parameters, yielding a mean of exactly 0.5. In contrast, WADD resolves the tie using the log-odds weights of the validities, consistently preferring the option with fewer but higher-validity cues. By measuring the proportion of choices that align with WADD's preference strictly on these tie-breaker trials, we create a metric where Tallying is anchored at 0.5 while WADD will score significantly higher.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    log_odds = np.array([2.9444, 1.7346, 1.0986, 0.6190, 0.2007])
    
    wadd_choices = 0
    tie_trials = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Isolate 'Tie-Breaker' trials where Tallying sees a perfect tie
        if a_wins == b_wins:
            tie_trials += 1
            a_wadd = np.sum((a > b) * log_odds)
            b_wadd = np.sum((b > a) * log_odds)
            
            wadd_favors_A = a_wadd > b_wadd
            if wadd_favors_A and row['response'] == 0:
                wadd_choices += 1
            elif not wadd_favors_A and row['response'] == 1:
                wadd_choices += 1
                
    if tie_trials == 0:
        return 0.5
        
    return float(wadd_choices / tie_trials)
outcome: self_sim=0.5220 (var=0.0084) adversary_sim=0.5160 (var=0.0074) welch_t=+0.238 p=0.8126 (N=25, alpha=0.01) -> reject

[3] rationale: To maximize the discrimination between Tallying and WADD while minimizing per-subject variance, we utilize all 90 trials per subject by computing the Pearson correlation between the subject's binary choices and the theoretical score differences of each model. The experimental design ensures that the Tallying score difference (simple count) and the WADD score difference (log-odds weighted sum) are essentially orthogonal or slightly negatively correlated. For a subject using Tallying, their choices will positively correlate with the Tallying score difference and zero-to-negatively correlate with the WADD score difference, yielding a reliably negative metric (r_wadd - r_tally). For a subject using WADD, their choices will align significantly better with the WADD score difference, making the metric positive. By aggregating over all trials via correlation, binomial noise is heavily suppressed compared to subsetting specific trial types.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    log_odds = np.log(val / (1.0 - val))
    
    tally_diffs = np.zeros(len(data))
    wadd_diffs = np.zeros(len(data))
    choices = np.zeros(len(data))
    
    for i, (_, row) in enumerate(data.iterrows()):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        tally_diffs[i] = np.sum(a > b) - np.sum(b > a)
        wadd_diffs[i] = np.sum((a > b) * log_odds) - np.sum((b > a) * log_odds)
        choices[i] = 1.0 if row['response'] == 0 else 0.0
        
    if np.std(choices) == 0:
        return 0.0
        
    r_tally = np.corrcoef(tally_diffs, choices)[0, 1]
    r_wadd = np.corrcoef(wadd_diffs, choices)[0, 1]
    
    if np.isnan(r_tally): r_tally = 0.0
    if np.isnan(r_wadd): r_wadd = 0.0
    
    return float(r_wadd - r_tally)
outcome: self_sim=0.0203 (var=0.0351) adversary_sim=-0.1135 (var=0.0395) welch_t=+2.449 p=0.01802 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    val = np.array([0.96, 0.91, 0.82, 0.73, 0.64, 0.58, 0.53])\n    log_odds = np.log(val / (1.0 - val))\n    \n    subject_scores = []\n    for subj, subj_df in data.groupby('subject_id'):\n        t_diffs = []\n        w_diffs = []\n        choices = []\n        \n        for _, row in subj_df.iterrows():\n            a = np.array(row['option_a_ratings'])\n            b = np.array(row['option_b_ratings'])\n            \n            t_diff = int(np.sum(a > b) - np.sum(b > a))\n            w_diff = float(np.sum((a > b) * log_odds) - np.sum((b > a) * log_odds))\n            \n            t_diffs.append(t_diff)\n            w_diffs.append(round(w_diff, 4))\n            choices.append(1.0 if row['response'] == 0 else 0.0)\n            \n        df_subj = pd.DataFrame({'t_diff': t_diffs, 'w_diff': w_diffs, 'choice': choices})\n        \n        score = 0.0\n        for t_d in [-2, 2, -1]:\n            group = df_subj[df_subj['t_diff'] == t_d]\n            if len(group) > 0:\n                unique_w = group['w_diff'].unique()\n                if len(unique_w) > 1:\n                    max_w = np.max(unique_w)\n                    min_w = np.min(unique_w)\n                    \n                    mean_max = group[group['w_diff'] == max_w]['choice'].mean()\n                    mean_min = group[group['w_diff'] == min_w]['choice'].mean()\n                    \n                    score += (mean_max - mean_min)\n                    \n        subject_scores.append(score)\n        \n    return float(np.mean(subject_scores))",
  "rationale": "By grouping trials that have the exact same Tallying score difference (e.g., A wins on 2 features, B wins on 4, yielding a difference of -2), we control for Tallying's primary mechanism. For any subject using Tallying, their probability of choosing Option A is identical across all trials within the same Tallying-difference group, regardless of their noise parameters (beta and epsilon). However, within these groups, the WADD theory evaluates the options differently depending on the log-odds validities of the specific features. By computing the difference in the choice rate of Option A between the trial with the highest WADD score difference and the trial with the lowest WADD score difference within each group, we isolate the validity-weighting effect. Tallying will yield an expected score of exactly 0.0 (with low variance driven purely by binomial noise), whereas WADD will yield a strictly positive score."
}
```

## Usage

```json
{
  "prompt_token_count": 5956,
  "candidates_token_count": 778,
  "total_token_count": 14944
}
```
