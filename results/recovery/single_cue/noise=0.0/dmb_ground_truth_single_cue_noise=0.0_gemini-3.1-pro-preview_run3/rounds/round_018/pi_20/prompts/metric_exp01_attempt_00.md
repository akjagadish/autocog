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
**Validities (n_features=5):** [0.92, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=4):**
  trial 1: A=[1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 2: A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 3: A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 4: A=[1, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0]

**Rationale:** To quantitatively dissociate the advocated Probabilistic Cue Sampling Tie-Breaker model from the competing Decisiveness-Dependent Strategy Selection model, we exploit how each theory scales determinism when different cues break a tied tally. The competing model resolves tied tallies using a strictly non-compensatory Take-The-Best (TTB) rule. Because TTB simply outputs a deterministic 1.0 preference for the option winning the highest available discriminating cue, the competing model is structurally forced to predict identical choice probabilities across any tied-tally trials where TTB favors an option, regardless of whether the tie is broken by the 1st-best or 2nd-best cue. In contrast, the advocated model resolves ties by sampling cues proportionally to their exponentiated log-odds validities. Because the log-odds evidence for the 1st cue is mathematically larger than for the 2nd cue, the advocated model is structurally forced to predict higher determinism when the 1st cue breaks the tie compared to when the 2nd cue breaks the tie. We designed tied-tally trials where either the 1st cue or the 2nd cue is the primary discriminator, alongside unequal-tally trials to anchor baseline determinism.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Cue Sampling Tie-Breaker with Log-Odds Evidence Scale: Decision-makers primarily rely on a Tallying heuristic to choose between options. However, when tallies are tied, they probabilistically sample a single discriminating feature to break the tie. The probability of sampling a specific feature is proportional to the exponentiated log-odds of its validity. This maps validities onto an evidence scale, providing a theoretically grounded non-linear stretching that captures the softer determinism observed in tied trials without allowing lower-validity cues to deterministically override higher-validity ones.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- theta: [0.1, 20.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    theta = float(parameters["theta"])
    
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    tally_a = np.sum(a_wins)
    tally_b = np.sum(b_wins)
    
    if tally_a != tally_b:
        # Primary heuristic: Tallying with softmax
        scores = np.array([tally_a, tally_b])
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        p_core = e / np.sum(e)
    else:
        # Tie-breaker: Probabilistic cue sampling using log-odds evidence scale
        mask = a != b
        if np.sum(mask) == 0:
            p_core = np.array([0.5, 0.5])
        else:
            val_clipped = np.clip(val, 0.501, 0.999)
            lo = np.log(val_clipped / (1.0 - val_clipped))
            surviving_lo = (lo ** theta) * mask
            w = surviving_lo / np.sum(surviving_lo)
            
            prob_a = np.sum(w * a_wins)
            prob_b = np.sum(w * b_wins)
            p_core = np.array([prob_a, prob_b])
            
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
**Description:** Decisiveness-Dependent Strategy Selection with Sharp Transition: Decision-makers probabilistically select between a compensatory Tallying strategy and a non-compensatory Take-The-Best (TTB) strategy on a trial-by-trial basis. The probability of using Tallying is a logistic function of the absolute difference in tally scores between the two options. By strictly constraining the sensitivity (theta) to be positive and the threshold to [0.1, 0.9], the model naturally transitions to a sharp step function where Tallying heavily dominates for decisive tally differences (delta >= 1), while TTB is strictly reserved as a tie-breaker for complex/tied stimuli (delta == 0).

**Parameters:**
- theta: [1.0, 20.0]
- threshold: [0.1, 0.9]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    threshold = float(parameters["threshold"])
    epsilon = float(parameters["epsilon"])
    
    # Tallying prediction
    a_wins = np.sum(a > b)
    b_wins = np.sum(b > a)
    delta_tally = abs(a_wins - b_wins)
    
    if a_wins > b_wins:
        p_a_tally = 1.0
    elif b_wins > a_wins:
        p_a_tally = 0.0
    else:
        p_a_tally = 0.5
        
    # Take-The-Best (TTB) prediction
    order = np.argsort(val)[::-1]
    p_a_ttb = 0.5
    for idx in order:
        if a[idx] > b[idx]:
            p_a_ttb = 1.0
            break
        elif b[idx] > a[idx]:
            p_a_ttb = 0.0
            break
            
    # Strategy selection probability
    # Probability of using Tallying depends on the decisiveness of the tally
    exponent = -theta * (delta_tally - threshold)
    exponent = np.clip(exponent, -500.0, 500.0) # Prevent overflow
    p_use_tally = 1.0 / (1.0 + np.exp(exponent))
    
    p_a_core = p_use_tally * p_a_tally + (1.0 - p_use_tally) * p_a_ttb
    p_b_core = 1.0 - p_a_core
    
    p_core = np.array([p_a_core, p_b_core])
    
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
[0] rationale: The metric computes the difference in the probability of choosing Option A between Trial 2 and Trial 1. In both trials, the tallies are tied (2 vs 2), and Option A wins the most valid cue (index 0). The competing theory strictly uses Take-The-Best for tied tallies, meaning it will always choose Option A in both trials, resulting in a difference of 0. The advocated theory probabilistically samples cues based on their exponentiated log-odds validity. Because Option A has stronger cues overall in Trial 2 compared to Trial 1, the advocated theory predicts a higher probability of choosing A in Trial 2, yielding a positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_tuple'] = data['option_a_ratings'].apply(tuple)
    
    t1_mask = data['a_tuple'] == (1, 0, 0, 1, 0)
    t2_mask = data['a_tuple'] == (1, 1, 0, 0, 0)
    
    # response == 0 means option A was chosen.
    # 1.0 - mean(response) gives the proportion of times A was chosen.
    p_A_t1 = 1.0 - data.loc[t1_mask, 'response'].mean()
    p_A_t2 = 1.0 - data.loc[t2_mask, 'response'].mean()
    
    return float(p_A_t2 - p_A_t1)
outcome: self_sim=0.0250 (var=0.0099) adversary_sim=0.0000 (var=0.0097) welch_t=+0.892 p=0.377 (N=25, alpha=0.01) -> reject

[1] rationale: In Trial 1, the tallies are tied (2 vs 2), but Option A wins the highest-validity cue. The Competing theory strictly falls back to Take-The-Best, yielding a very high probability of choosing A (only reduced by epsilon noise). The Advocated theory uses a probabilistic cue sampling tie-breaker. Because Option B wins the 2nd and 3rd best cues in Trial 1, it exerts a strong compensatory pull in the Advocated model, systematically lowering the probability of choosing A compared to the Competing model. We simply measure the proportion of times Option B is chosen in Trial 1 (i.e., response == 1).
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_tuple'] = data['option_a_ratings'].apply(tuple)
    t1_mask = data['a_tuple'] == (1, 0, 0, 1, 0)
    return float(data.loc[t1_mask, 'response'].mean())
outcome: self_sim=0.1808 (var=0.0198) adversary_sim=0.1308 (var=0.0110) welch_t=+1.426 p=0.1609 (N=25, alpha=0.01) -> reject

[2] rationale: This metric exploits a perfect structural cancellation in the Competing model. In the Competing model, Trial 1 triggers strict Take-The-Best (because tallies are tied, delta=0) which chooses A. Trial 4 triggers strict Tallying (because delta=3, heavily passing the threshold) which chooses B. In both cases, the dominant choice is selected with probability exactly `1 - epsilon/2`. Thus, for the Competing model, `P(B|T4) - P(A|T1)` is mathematically constrained to have an expectation of exactly 0.0, completely independent of the subject's epsilon noise parameter.

In contrast, the Advocated model relies on two independent parameters for these trials. Trial 4 is driven by the primary Tallying heuristic with softmax parameter `beta` (where delta=3 creates a massive logit, pushing `P(B|T4)` extremely close to 1.0). Trial 1 is driven by the Probabilistic Cue Sampling tie-breaker with parameter `theta`. Because A's cues in Trial 1 are relatively weak compared to B's compensatory cues, `P(A|T1)` is significantly lower than `P(B|T4)`. Therefore, the Advocated model produces a strictly positive difference, reliably separating it from the Competing model's zero-centered baseline.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 1: Tally tied, A wins highest validity cue (index 0)
    t1_mask = data['a_str'] == '10010'
    # Trial 4: Tally heavily favors B (1 vs 4), B wins highest validity cue
    t4_mask = data['a_str'] == '00010'
    
    # p_A_T1 is the probability of choosing A in Trial 1
    p_A_T1 = 1.0 - data.loc[t1_mask, 'response'].mean()
    # p_B_T4 is the probability of choosing B in Trial 4
    p_B_T4 = data.loc[t4_mask, 'response'].mean()
    
    # Return the difference between the dominant choices in T4 and T1
    return float(p_B_T4 - p_A_T1)
outcome: self_sim=0.0433 (var=0.0242) adversary_sim=0.0217 (var=0.0068) welch_t=+0.615 p=0.5421 (N=25, alpha=0.01) -> reject

[3] rationale: This metric contrasts the probability of choosing Option A in Trial 3 (decisive tally favoring A) versus Trial 1 (tied tally where A wins the best cue but B has strong compensatory cues). In the Competing model, both trials overwhelmingly favor A: Trial 3 triggers Tallying (which chooses A) and TTB (which chooses A), while Trial 1 predominantly triggers TTB (which also chooses A). Thus, the Competing model predicts P(A|T3) and P(A|T1) to be nearly identical, yielding a difference close to 0. In the Advocated model, Trial 3 strongly favors A via the primary Tallying heuristic, but Trial 1 relies on Probabilistic Cue Sampling where B's compensatory cues significantly drag down the probability of choosing A. This results in a substantial positive difference for the Advocated model, clearly discriminating the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 1: Tally tied (2 vs 2), A wins highest validity cue, B wins 2nd and 3rd
    t1_mask = data['a_str'] == '10010'
    # Trial 3: Tally favors A (3 vs 2), A wins highest validity cue
    t3_mask = data['a_str'] == '11100'
    
    p_A_T1 = 1.0 - data.loc[t1_mask, 'response'].mean()
    p_A_T3 = 1.0 - data.loc[t3_mask, 'response'].mean()
    
    return float(p_A_T3 - p_A_T1)
outcome: self_sim=0.0200 (var=0.0238) adversary_sim=0.0100 (var=0.0120) welch_t=+0.264 p=0.7929 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data = data.copy()\n    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    \n    t1_mean = data.loc[data['a_str'] == '10100', 'response'].mean()\n    t2_mean = data.loc[data['a_str'] == '11100', 'response'].mean()\n    \n    if pd.isna(t1_mean) or pd.isna(t2_mean):\n        return 0.0\n        \n    # response == 0 means Option A was chosen. \n    # Mean response is P(B).\n    # P(A|T1) - P(A|T2) = (1 - t1_mean) - (1 - t2_mean) = t2_mean - t1_mean\n    return float(t2_mean - t1_mean)",
  "rationale": "The metric calculates P(A|Trial 1) - P(A|Trial 2). In both Trial 1 and Trial 2, the tally is tied (2 vs 2 and 3 vs 3 respectively). The Competing theory resolves tied tallies by strictly falling back to Take-The-Best. In Trial 1, cue 0 discriminates and favors A; in Trial 2, cue 0 is tied, but cue 1 discriminates and favors A. Because TTB is a non-compensatory step-function, it assigns exactly the same deterministic preference to A in both trials, meaning the Competing model mathematically constrains this difference to exactly 0 (regardless of its epsilon noise or threshold parameters). In contrast, the Advocated theory resolves ties via probabilistic cue sampling proportional to exponentiated log-odds validity. Because cue 0 (which breaks the tie in T1) has a higher validity than cue 1 (which breaks the tie in T2), the Advocated model predicts a stronger preference for A in Trial 1 than in Trial 2, yielding a strictly positive difference."
}
```

## Usage

```json
{
  "prompt_token_count": 5580,
  "candidates_token_count": 488,
  "total_token_count": 9354
}
```
