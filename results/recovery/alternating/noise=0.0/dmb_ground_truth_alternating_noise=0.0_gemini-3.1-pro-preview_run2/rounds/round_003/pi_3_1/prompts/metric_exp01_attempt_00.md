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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 6: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 7: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 8: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** TTB with High Noise is highly constrained: because it applies a large lapse rate to a lexicographic rule, it must predict a weak, near-50/50 preference for the option possessing the highest-validity discriminating cue on every single trial. It cannot predict strong preferences, nor can it predict reversals when multiple lower-validity cues oppose the highest-validity cue. Non-linear WADD, on the other hand, computes a weighted sum and can span a wide range of choice probabilities. By mixing 'dominance' trials (where both models agree on the winner, but WADD predicts a strong preference while TTB predicts a weak one) with 'compensatory' trials (where a coalition of lower-validity cues outweighs the top cue, causing WADD to predict a strong reversal while TTB continues to predict a weak preference for the top cue), this design strictly dissociates the two models both in direction and magnitude of the predicted choice probabilities.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Decision makers evaluate options by computing a weighted sum of their feature values, where the weights correspond to a non-linear scaling of the subjective or instructed validities of the cues. This non-linear Weighted Additive (WADD) strategy includes an exponent parameter (gamma) on the validities, allowing the decision process to smoothly interpolate between equal weighting (Tallying, gamma=0), proportional weighting (standard WADD, gamma=1), and lexicographic one-reason decision making (Take The Best, gamma > 1). The option with the higher total weighted score is favored, with response noise modeled via a softmax function over the scores and an independent lapse rate. To capture the fact that human behavior can sometimes be entirely random, the lapse rate and inverse temperature parameters are allowed to span their full theoretical ranges (up to pure guessing).

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.7, 1.0]
- gamma: [0.0, 5.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
        
    # Apply non-linear scaling to validities
    gamma = float(parameters["gamma"])
    val = val ** gamma
    
    a, b = stim[0], stim[1]
    
    # Compute the weighted sum (WADD score) for each option.
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
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
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Take-The-Best (TTB) with High Noise: Decision makers employ a lexicographic heuristic, searching through cues in order of descending validity. They stop at the first cue that discriminates between the two options and choose the option with the higher value on that cue. However, to accommodate the empirical observation that agreement with any deterministic strategy hovers around 50%, the model incorporates a very high lapse rate (epsilon) and a low softmax inverse temperature (beta). This restricts the model to primarily exhibit random guessing, with only a weak TTB signal, matching the high degree of noise in the observed data.

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.8, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Sort cues by descending validity; stable sort handles ties
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    winner = None
    
    # Lexicographic search
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        # No cue discriminates, guess uniformly
        return np.array([0.5, 0.5])
        
    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over binary TTB scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
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
[0] rationale: Calculates the difference in the probability of choosing Option A between a strong dominance trial (Trial 6, where A has all positive cues) and a weak compensatory trial (Trial 1, where A only has the single best cue while B has all others). Take-The-Best (TTB) evaluates both trials identically because Cue 1 discriminates in favor of A in both cases, leading to an expected difference of exactly 0. In contrast, the Non-linear WADD model integrates all cues, predicting a much stronger preference for A in Trial 6 than in Trial 1, yielding a strictly positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 6: A dominates B completely
    idx6 = (a_str == '11111') & (b_str == '00000')
    # Trial 1: A has highest validity cue, B has all others
    idx1 = (a_str == '10000') & (b_str == '01111')
    
    p_a_6 = (data.loc[idx6, 'response'] == 0).mean()
    p_a_1 = (data.loc[idx1, 'response'] == 0).mean()
    
    if pd.isna(p_a_6) or pd.isna(p_a_1):
        return 0.0
        
    return float(p_a_6 - p_a_1)
outcome: self_sim=0.0567 (var=0.0404) adversary_sim=0.0433 (var=0.0415) welch_t=+0.233 p=0.8167 (N=25, alpha=0.01) -> reject

[1] rationale: Because epsilon is extremely high in both models (0.7 to 1.0), raw choice proportions are compressed tightly around 0.5. To extract a robust signal, we must aggregate across all trials using a contrast vector that aligns with WADD's evidence integration but is strictly orthogonal to TTB's lexicographic predictions. The weights used here correspond to the Equal Weights (EW) score differences (A - B) for each trial. TTB predicts the exact same weak preference for the winner in almost all trials, causing its expected value on this specific contrast to perfectly cancel out to 0. In contrast, WADD scales its choice probabilities dynamically with the integrated evidence, yielding a strictly positive expected score that maximizes the signal-to-noise ratio.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, int(v) for v in x)) if isinstance(x, (list, tuple)) else ''.join(map(str, x)))
    
    w_map = {
        '10000': -3,
        '11000': -1,
        '10101': 1,
        '11100': -1,
        '01100': 0,
        '11111': 5,
        '00111': 2,
        '10011': 1
    }
    
    score = 0.0
    for trial_str, w in w_map.items():
        trial_data = data[a_str == trial_str]
        if len(trial_data) > 0:
            score += w * (trial_data['response'] == 0).mean()
            
    return float(score)
outcome: evaluation failed (SyntaxError: Generator expression must be parenthesized (<string>, line 3))

[2] rationale: To robustly differentiate the models despite high lapse rates, we use a zero-sum linear contrast over the 7 trials where TTB predicts Option A. Because TTB's lexicographic rule leads it to predict the exact same weak probability for A across all 7 trials, its expected value on this contrast perfectly cancels out to 0. In contrast, WADD integrates all cues, meaning its preference for A scales with the number of cues favoring A. By assigning negative weights to trials where A has few cues and positive weights where A has many cues, WADD will yield a strictly positive expected score, maximizing the signal-to-noise ratio.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def to_str(x):
        if isinstance(x, (list, tuple)):
            return ''.join([str(int(v)) for v in x])
        return str(x)
        
    a_str = data['option_a_ratings'].apply(to_str)
    
    # These 7 trials are all ones where TTB predicts Option A.
    # We assign weights that sum to exactly 0, meaning TTB's expected score is 0.
    # The weights correlate with the number of cues favoring A, so WADD will score > 0.
    w_map = {
        '10000': -4,
        '11000': -2,
        '01100': -1,
        '10101': 1,
        '11100': 1,
        '10011': 1,
        '11111': 4
    }
    
    score = 0.0
    for trial_str, w in w_map.items():
        trial_data = data[a_str == trial_str]
        if len(trial_data) > 0:
            score += w * (trial_data['response'] == 0).mean()
            
    return float(score)
outcome: self_sim=0.1400 (var=0.7996) adversary_sim=0.2067 (var=0.6817) welch_t=-0.274 p=0.7854 (N=25, alpha=0.01) -> reject

[3] rationale: To robustly discriminate Non-linear WADD from TTB with High Noise, we exploit a structural feature of TTB: because it is lexicographic and maps its decision to a binary score before applying softmax, it predicts the exact same weak choice probability for the winning option across all trials. By constructing a zero-sum linear contrast over the 7 trials where TTB predicts Option A, we mathematically guarantee that TTB's expected score on this metric is exactly 0, regardless of the subject's specific noise parameters. In contrast, WADD integrates all cues, producing a dynamic range of choice probabilities. By assigning strong negative weights to trials where WADD predicts B (e.g., Trial 1) and strong positive weights where WADD predicts A (e.g., Trial 6), WADD yields a strictly positive expected score. This isolates the evidence-integration signal while perfectly canceling out TTB's constant baseline, maximizing the signal-to-noise ratio.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def to_str(x):
        return ''.join([str(int(v)) for v in x])
        
    a_str = data['option_a_ratings'].apply(to_str)
    
    # These 7 trials are exactly the ones where TTB predicts Option A.
    # Because TTB's lexicographic rule maps its decision to a binary score before softmax,
    # it predicts the exact same weak probability for A across all 7 of these trials.
    # By making the weights sum to 0, TTB's expected score perfectly cancels out to 0,
    # regardless of the subject's specific noise parameters (beta, epsilon).
    # In contrast, WADD integrates all cues, meaning its preference for A scales dynamically.
    # The weights correlate with WADD's expected score differences (e.g., strongly negative 
    # for Trial 1 where WADD prefers B, strongly positive for Trial 6 where WADD strongly prefers A),
    # yielding a strictly positive expected score for WADD and maximizing the signal-to-noise ratio.
    w_map = {
        '10000': -16,
        '11000': -3,
        '10101': 2,
        '11100': -3,
        '01100': -1,
        '11111': 22,
        '10011': -1
    }
    
    score = 0.0
    for trial_str, w in w_map.items():
        trial_data = data[a_str == trial_str]
        if len(trial_data) > 0:
            score += w * (trial_data['response'] == 0).mean()
            
    return float(score)
outcome: self_sim=1.2150 (var=15.6678) adversary_sim=0.7533 (var=16.3055) welch_t=+0.408 p=0.6849 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    \n    # Trial 1: A dominates (TTB chooses A, WADD strongly chooses A)\n    idx1 = (a_str == '11111') & (b_str == '00000')\n    # Trial 3: A has top cue, B has many lower cues (TTB chooses A, WADD chooses B)\n    idx3 = (a_str == '10000') & (b_str == '00111')\n    # Trial 4: B has top cue, A has many lower cues (TTB chooses B, WADD chooses A)\n    idx4 = (a_str == '01111') & (b_str == '10000')\n    # Trial 7: B has top two cues, A has bottom three (TTB chooses B, WADD chooses B)\n    idx7 = (a_str == '00111') & (b_str == '11000')\n    \n    p_a_1 = (data.loc[idx1, 'response'] == 0).mean()\n    p_a_3 = (data.loc[idx3, 'response'] == 0).mean()\n    p_a_4 = (data.loc[idx4, 'response'] == 0).mean()\n    p_a_7 = (data.loc[idx7, 'response'] == 0).mean()\n    \n    p_a_1 = p_a_1 if pd.notna(p_a_1) else 0.5\n    p_a_3 = p_a_3 if pd.notna(p_a_3) else 0.5\n    p_a_4 = p_a_4 if pd.notna(p_a_4) else 0.5\n    p_a_7 = p_a_7 if pd.notna(p_a_7) else 0.5\n    \n    return float((p_a_1 - p_a_3) + (p_a_4 - p_a_7))",
  "rationale": "This metric leverages two sets of paired trials to strictly isolate compensatory weighting (WADD) from lexicographic choice (TTB). In Trial 1 and Trial 3, TTB predicts Option A because Cue 0 favors A in both cases, leading to an identical weak preference for A. Thus, expected p(A|T1) - p(A|T3) = 0 for TTB. However, WADD strongly prefers A in Trial 1 (dominance) but prefers B in Trial 3 (compensatory), making this difference strictly positive. Similarly, for Trial 4 and Trial 7, TTB predicts Option B in both cases based on Cue 0, so p(A|T4) - p(A|T7) = 0. WADD prefers A in Trial 4 but B in Trial 7, again yielding a positive difference. By summing these two differences, we mathematically guarantee an expected score of exactly 0 for TTB regardless of its specific noise parameters, while WADD produces a robust, strictly positive score, heavily maximizing the signal-to-noise ratio."
}
```

## Usage

```json
{
  "prompt_token_count": 5623,
  "candidates_token_count": 832,
  "total_token_count": 11142
}
```
