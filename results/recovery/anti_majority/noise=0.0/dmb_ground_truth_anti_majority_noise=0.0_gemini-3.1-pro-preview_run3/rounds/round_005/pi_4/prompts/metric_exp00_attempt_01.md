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

**Trial pairs (n=7):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 2: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 5: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 6: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]

**Rationale:** This design leverages an invariance property of the Strategy Mixture (TTB + Tallying) model to quantitatively dissociate it from the Sequential Evidence Accumulation model. In the Strategy Mixture model, choice probabilities depend solely on the identity of the single best discriminating cue (TTB) and the total count of winning cues (Tallying). By holding the highest-validity cue and the total cue-win counts constant across a subset of trials (e.g., Trials 1, 2, and 3), the Strategy Mixture model is mathematically forced to predict the exact same choice probabilities. However, the Sequential Evidence Accumulation model integrates the actual validities of the specific lower-tier cues until a threshold is reached. By shifting which specific lower-validity cues Option A vs. B wins across these trials, the accumulated evidence varies. Consequently, if the threshold is large enough to evaluate these cues, the Sequential model predicts systematically shifting choice probabilities across trials where the Strategy Mixture model predicts strict equivalence.

**Computed schedule:** 7 unique pairs × 13 reps = 91 trials per subject.



## ADVOCATED THEORY
**Description:** Strategy Mixture (TTB + Tallying): Decision makers do not universally adhere to a single strategy. Instead, they use a probabilistic mixture of a non-compensatory strategy (Take The Best) and a compensatory strategy (Tallying). A parameter P_TTB dictates the probability of using TTB on any given trial, while 1 - P_TTB is the probability of using Tallying. This accounts for intermediate levels of TTB-consistency and Tallying-consistency observed in empirical data across subjects and trials. The mixture captures a balance between TTB and Tallying, avoiding over-reliance on uniform guessing.

**Parameters:**
- beta: [0.1, 20.0]
- p_ttb: [0.35, 0.95]
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
    n_features = len(a)
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take The Best (TTB) Strategy
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        scores_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        scores_ttb = np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.0, 0.0])
        
    # Tallying Strategy
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    # Normalize by n_features to keep scale comparable to TTB for the shared beta
    scores_tally = np.array([a_wins, b_wins]) / max(1.0, float(n_features))
    
    beta = float(parameters["beta"])
    
    # TTB Probabilities
    z_ttb = beta * (scores_ttb - scores_ttb.max())
    e_ttb = np.exp(z_ttb)
    p_ttb_dist = e_ttb / e_ttb.sum()
    
    # Tallying Probabilities
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally_dist = e_tally / e_tally.sum()
    
    # Mixture
    p_ttb_weight = float(parameters["p_ttb"])
    epsilon = float(parameters["epsilon"])
    
    p_core = p_ttb_weight * p_ttb_dist + (1.0 - p_ttb_weight) * p_tally_dist
    
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
**Description:** Sequential Evidence Accumulation with Flexible Threshold: Decision-makers search through cues sequentially in order of their validity. As they evaluate each cue, they accumulate evidence for the favored option. The search stops as soon as the absolute difference in accumulated evidence between the two options reaches a subjective threshold. If the threshold is low (or zero), this mechanism perfectly mimics Take-The-Best by stopping at the first discriminating cue. If the threshold is high, it evaluates all available cues, naturally transitioning into compensatory strategies like Weighted Additive (WADD) or Tallying.

**Parameters:**
- gamma: [0.0, 3.0]
- theta: [0.0, 1.0]
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    theta = float(parameters["theta"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Search through cues in order of validity (highest first)
    cue_order = np.argsort(-val, kind="stable")
    
    # Scale validities non-linearly to represent subjective evidence weights
    w = val ** gamma
    
    ev_a = 0.0
    ev_b = 0.0
    
    # Sequential evidence accumulation
    for i in cue_order:
        if a[i] > b[i]:
            ev_a += w[i]
        elif b[i] > a[i]:
            ev_b += w[i]
            
        # Stop search if the evidence difference reaches the threshold
        # (and ensure we don't stop prematurely if no discriminating evidence has been found yet)
        if abs(ev_a - ev_b) >= theta and abs(ev_a - ev_b) > 0:
            break
            
    # Convert accumulated evidence into choice probabilities via softmax
    scores = np.array([ev_a, ev_b])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate (guessing)
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
[0] rationale: In Trials 1, 2, and 3, Option A wins the highest validity cue (cue 1) but loses the tallying count (2 wins vs 3 wins). Because both the highest validity cue and the total number of winning cues are identical across these trials, the Strategy Mixture model (TTB + Tallying) predicts the exact same choice probability for Option A across all three trials. Therefore, the difference in the proportion of A choices between Trial 3 and Trial 1 should be zero (subject to sampling noise). However, the Sequential Evidence Accumulation model integrates the specific validities of the lower-tier cues. In Trial 3, Option A wins a much stronger secondary cue (cue 3) compared to Trial 1 (where it wins cue 5). If the threshold is high enough to evaluate these cues, the Sequential model predicts systematically higher choice probabilities for Option A in Trial 3 compared to Trial 1. This metric calculates the mean intra-subject difference in P(A) between Trial 3 and Trial 1.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    
    # Trial 1: A wins cue 1 and 5
    t1 = (1, 0, 0, 0, 1)
    # Trial 3: A wins cue 1 and 3
    t3 = (1, 0, 1, 0, 0)
    
    subj_diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        df_t1 = subj_df[subj_df['A_tuple'] == t1]
        df_t3 = subj_df[subj_df['A_tuple'] == t3]
        
        if len(df_t1) == 0 or len(df_t3) == 0:
            continue
            
        # response = 0 means option A was chosen
        p_a_t1 = 1.0 - df_t1['response'].mean()
        p_a_t3 = 1.0 - df_t3['response'].mean()
        
        subj_diffs.append(p_a_t3 - p_a_t1)
        
    if not subj_diffs:
        return 0.0
        
    return float(np.mean(subj_diffs))

outcome: self_sim=0.0485 (var=0.0285) adversary_sim=0.0354 (var=0.0277) welch_t=+0.276 p=0.7838 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    data = data.copy()\n    data['A_tuple'] = data['option_a_ratings'].apply(tuple)\n    \n    # Conflict trials: TTB prefers A (cue 1), but Tally prefers B (2 vs 3)\n    t1 = (1, 0, 0, 0, 1)\n    t2 = (1, 0, 0, 1, 0)\n    t3 = (1, 0, 1, 0, 0)\n    conflict_trials = {t1, t2, t3}\n    \n    # Agreement trial: TTB prefers A (cue 1), and Tally prefers A (2 vs 1)\n    t6 = (1, 1, 0, 0, 0)\n    \n    subj_diffs = []\n    for subj, subj_df in data.groupby('subject_id'):\n        df_conflict = subj_df[subj_df['A_tuple'].isin(conflict_trials)]\n        df_agree = subj_df[subj_df['A_tuple'] == t6]\n        \n        if len(df_conflict) == 0 or len(df_agree) == 0:\n            continue\n            \n        # response = 0 means option A was chosen\n        p_a_conflict = 1.0 - df_conflict['response'].mean()\n        p_a_agree = 1.0 - df_agree['response'].mean()\n        \n        subj_diffs.append(p_a_agree - p_a_conflict)\n        \n    if not subj_diffs:\n        return 0.0\n        \n    return float(np.mean(subj_diffs))",
  "rationale": "In Trials 1, 2, and 3, Option A wins the highest validity cue (cue 1), but Option B wins more total cues (3 vs 2). The Strategy Mixture model uses a probabilistic combination of Take-The-Best (which strongly prefers A) and Tallying (which prefers B), resulting in a moderated probability of choosing A. In Trial 6, Option A wins both the highest validity cue and the total cue count (2 vs 1), aligning both TTB and Tallying to strongly prefer A. Thus, the Strategy Mixture model predicts a substantial increase in the probability of choosing A in Trial 6 compared to Trials 1-3. Conversely, the Sequential Evidence Accumulation model searches cues in validity order and stops when the evidence difference exceeds a threshold. Because cue 1 has a very high validity (0.95), the threshold is almost always met at the first cue. Consequently, the Sequential model evaluates only cue 1 in both sets of trials and predicts virtually no difference in the choice probabilities between Trial 6 and Trials 1-3. This metric isolates this divergence by calculating the intra-subject difference in P(A) between the agreement trial (T6) and the conflict trials (T1, T2, T3)."
}
```

## Usage

```json
{
  "prompt_token_count": 4367,
  "candidates_token_count": 711,
  "total_token_count": 8862
}
```
