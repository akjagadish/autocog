# metric_exp00_attempt_02

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
**Validities (n_features=5):** [0.93, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 2: A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 8: A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** To maximally distinguish Theory 1 (Non-linear log-odds WADD) from Theory 2 (Mixture of TTB and Linear WADD), we exploit the fundamental difference in how they weight cues. Theory 2's WADD component uses raw validities as linear weights, which sums up quickly and strongly favors options with more positive cues (similar to Tallying). Theory 1 transforms validities into log-odds before applying a non-linear scaling parameter (gamma). Because log-odds grow asymptotically as validity approaches 1.0, Theory 1 can strongly favor a single highly valid cue over multiple moderately valid cues, even without a strict TTB mechanism. By creating trials where the linear sum of raw validities strongly opposes the highest cue, but the log-odds sum supports it (or vice-versa depending on the exact combination), we can cleanly separate the probabilistic mixture of step-function TTB and linear WADD from the deterministic, non-linearly scaled log-odds WADD.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** The Weighted Additive (WADD) model with non-linear weight scaling posits that decision makers integrate all available information by computing a weighted sum of features. However, the weighting of cues is not strictly proportional to their log-odds validity. Instead, decision makers apply a non-linear transformation (parameterized by gamma) to the log-odds, allowing them to stretch the weight differential. This permits WADD to approximate lexicographic (TTB-like) choice when gamma > 1, or more uniform (Tallying-like) weighting when gamma < 1, while remaining a fully compensatory integration process.

**Parameters:**
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- gamma: [0.5, 2.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Transform validities to log-odds weights, clipping to avoid infinity
    val_clipped = np.clip(val, 0.5001, 0.9999)
    log_odds = np.log(val_clipped / (1.0 - val_clipped))
    
    gamma = float(parameters["gamma"])
    w = np.sign(log_odds) * (np.abs(log_odds) ** gamma)
    
    # Compute weighted sum for each option
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the weighted scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
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
**Description:** Strategy Mixture Theory (TTB + WADD): Decision makers do not universally adopt a single monolithic strategy. Instead, choices are generated from a probabilistic mixture of decision rules. On any given trial, an individual uses a non-compensatory heuristic (Take The Best) with probability 'alpha', and a compensatory strategy (Weighted Additive - WADD) with probability '1 - alpha'. Mixing these strategies captures intermediate rates of compensatory and non-compensatory choices, while WADD leverages cue validities for a more nuanced compensatory evaluation.

**Parameters:**
- alpha: [0.5, 1.0]
- beta: [1.0, 20.0]
- epsilon: [0.0, 0.2]
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
    alpha = float(parameters["alpha"])
    epsilon = float(parameters["epsilon"])
    
    # Strategy 1: Take The Best (TTB)
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
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta * (scores_ttb - scores_ttb.max())
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / e_ttb.sum()
        
    # Strategy 2: WADD (Weighted Additive)
    score_a_wadd = np.sum(a * val)
    score_b_wadd = np.sum(b * val)
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    z_wadd = beta * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Mixture of the two strategies
    p_mix = alpha * p_ttb + (1.0 - alpha) * p_wadd
    
    # Apply lapse rate
    n_opts = p_mix.shape[0]
    p_final = (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
    
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
[0] rationale: This metric calculates the difference in the rate of choosing the TTB-favored option between Trial 1 (where TTB is opposed by 2 cues) and Trials 3/6 (where TTB is opposed by 4 cues). Under the Competing theory (Strategy Mixture), the agent mixes between TTB and Linear WADD. Because Linear WADD deterministically chooses the non-TTB option in both cases, the probability of choosing the TTB option is simply the mixture parameter `alpha`, resulting in a difference near 0. Under the Advocated theory (Non-linear log-odds WADD), the non-linear scaling of weights means that 4 opposing cues will much more reliably overcome the most valid cue than 2 opposing cues will, leading to a significantly higher TTB-choice rate in Trial 1 compared to Trials 3/6 (a positive difference).
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t1 = (a_str == '01100') & (b_str == '10000')
    t3 = (a_str == '10000') & (b_str == '01111')
    t6 = (a_str == '01111') & (b_str == '10000')
    
    t1_ttb = (data.loc[t1, 'response'] == 1)
    t3_ttb = (data.loc[t3, 'response'] == 0)
    t6_ttb = (data.loc[t6, 'response'] == 1)
    
    t1_rate = t1_ttb.mean() if len(t1_ttb) > 0 else 0.5
    
    t36_ttb = pd.concat([t3_ttb, t6_ttb])
    t36_rate = t36_ttb.mean() if len(t36_ttb) > 0 else 0.5
    
    return float(t1_rate - t36_rate)
outcome: self_sim=0.0708 (var=0.0411) adversary_sim=-0.0067 (var=0.0250) welch_t=+1.507 p=0.1389 (N=25, alpha=0.01) -> reject

[1] rationale: This metric calculates the difference in the rate of choosing the TTB-favored option between Trial 7 and Trial 1. Under the Competing theory (TTB + Linear WADD mixture), Linear WADD opposes TTB in BOTH of these trials (in Trial 7, Linear WADD favors the option with 3 cues; in Trial 1, Linear WADD favors the option with 2 cues). Thus, the Competing theory chooses the TTB option with probability 'alpha' in both trials, yielding a difference near 0. Under the Advocated theory (Non-linear Log-odds WADD), the log-odds validities heavily favor the TTB option in Trial 7 (because cue 1 + cue 3 > cue 2 + cue 4 + cue 5 in log-odds), but oppose the TTB option in Trial 1 (because cue 2 + cue 3 > cue 1 in log-odds). Therefore, the Advocated theory will show a very high TTB choice rate in Trial 7 and a very low TTB choice rate in Trial 1, resulting in a large positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 7: TTB option has cues [1, 0, 1, 0, 0], non-TTB has [0, 1, 0, 1, 1]
    t7_1 = (a_str == '10100') & (b_str == '01011')
    t7_2 = (a_str == '01011') & (b_str == '10100')
    
    # Trial 1: TTB option has cues [1, 0, 0, 0, 0], non-TTB has [0, 1, 1, 0, 0]
    t1_1 = (a_str == '01100') & (b_str == '10000')
    t1_2 = (a_str == '10000') & (b_str == '01100')
    
    t7_ttb = pd.concat([
        data.loc[t7_1, 'response'] == 0,
        data.loc[t7_2, 'response'] == 1
    ])
    
    t1_ttb = pd.concat([
        data.loc[t1_1, 'response'] == 1,
        data.loc[t1_2, 'response'] == 0
    ])
    
    t7_rate = t7_ttb.mean() if len(t7_ttb) > 0 else 0.5
    t1_rate = t1_ttb.mean() if len(t1_ttb) > 0 else 0.5
    
    return float(t7_rate - t1_rate)
outcome: self_sim=0.1533 (var=0.1034) adversary_sim=0.0067 (var=0.0208) welch_t=+2.081 p=0.04524 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))\n    \n    t7_a = (a_str == '10100') & (b_str == '01011')\n    t7_b = (a_str == '01011') & (b_str == '10100')\n    \n    t36_a = (a_str == '10000') & (b_str == '01111')\n    t36_b = (a_str == '01111') & (b_str == '10000')\n    \n    t7_ttb = pd.concat([\n        data.loc[t7_a, 'response'] == 0,\n        data.loc[t7_b, 'response'] == 1\n    ])\n    \n    t36_ttb = pd.concat([\n        data.loc[t36_a, 'response'] == 0,\n        data.loc[t36_b, 'response'] == 1\n    ])\n    \n    t7_rate = t7_ttb.mean() if len(t7_ttb) > 0 else 0.5\n    t36_rate = t36_ttb.mean() if len(t36_ttb) > 0 else 0.5\n    \n    return float(t7_rate - t36_rate)",
  "rationale": "This metric calculates the difference in the rate of choosing the TTB-favored option between Trial 7 and Trials 3/6. Under the Competing theory (Strategy Mixture), the agent mixes between TTB and Linear WADD. In both Trial 7 and Trials 3/6, Linear WADD strongly favors the non-TTB option. Thus, the agent only chooses the TTB option when using the TTB heuristic, which happens with probability 'alpha'. Because 'alpha' is consistent, the difference in TTB choice rate between these trials is near 0. Under the Advocated theory (Non-linear log-odds WADD), the decision depends on the non-linear scaling of log-odds validities. In Trial 7, the log-odds of the TTB option's cues are strong enough to outweigh the opposing cues for ANY value of the scaling parameter 'gamma', leading to a consistently high TTB choice rate (~1.0). In Trials 3/6, the TTB option has only the top cue against all four other cues; for smaller values of 'gamma', the sum of the four non-TTB cues outweighs the top cue, pulling the average TTB choice rate down significantly. Therefore, the Advocated theory predicts a large positive difference."
}
```

## Usage

```json
{
  "prompt_token_count": 5082,
  "candidates_token_count": 694,
  "total_token_count": 21217
}
```
