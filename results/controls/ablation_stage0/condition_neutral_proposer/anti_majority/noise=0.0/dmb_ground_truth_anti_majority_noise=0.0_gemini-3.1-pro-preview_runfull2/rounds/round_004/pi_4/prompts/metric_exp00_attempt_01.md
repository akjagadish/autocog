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
**Validities (n_features=5):** [0.95, 0.85, 0.8, 0.75, 0.7]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 6: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 8: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 9: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 10: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** Theory 1 predicts choices as a probabilistic mixture of Take-The-Best (TTB) and a Linear Weighted Additive rule (WADD). Because Linear WADD uses raw validities as weights, it is strongly compensatory: the sum of several moderate validities easily exceeds a single high validity. Theory 2, on the other hand, uses rank-dependent tallying where the weight of a feature decays as an inverse power of its rank (gamma >= 2.0). Mathematically, for any gamma >= 2.0, the weight of the highest-ranked feature (1.0) is strictly greater than the sum of all subsequent feature weights combined (which maxes out at ~0.645). Thus, Theory 2 acts as a deterministic, strictly non-compensatory lexicographic model (essentially a pure TTB rule). By pitting the highest-validity discriminating cue against multiple lower-validity cues, we can cleanly separate the two theories: Theory 2 will uniformly choose the option favored by the single best cue, while Theory 1 will exhibit a mixed response profile reflecting its compensatory WADD component.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
**Description:** Rank-Dependent Tallying: Decision-makers primarily evaluate options based on the sheer number of positive features (a tallying-like process), but the weight of each feature is subject to rank-based discounting. Rather than scaling exponentially with log-odds, a feature's weight decays as a power law of its validity rank. This ensures a strong compensatory mechanism where multiple moderate cues can easily overpower a single high-validity cue, and tallying differences dominate choice probabilities unless the validity rank differences are extreme.

**Parameters:**
- gamma: [2.0, 8.0]
- beta: [0.1, 20.0]
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
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Assign ranks to features based on validity (1 = highest validity)
    n_features = len(val)
    order = np.argsort(-val, kind="stable")
    ranks = np.zeros(n_features)
    ranks[order] = np.arange(1, n_features + 1)
    
    # Rank-based discounting: weight decays as an inverse power of rank
    w = 1.0 / (ranks ** gamma)
    
    # Compute weighted tally for each option
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
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
[0] rationale: Theory 2 (Rank-Dependent Tallying with gamma >= 2.0) acts as a strict, deterministic lexicographic model (Take-The-Best), meaning it will almost always choose the option favored by the highest-ranked discriminating cue. Theory 1 (Strategy Mixture) probabilistically mixes TTB with a compensatory Weighted Additive (WADD) rule. By calculating the proportion of choices that align with TTB on trials where TTB and WADD strictly disagree, we can clearly separate the two theories. Theory 2 will show a very high TTB match rate (bounded only by its lapse rate), while Theory 1 will show a significantly lower TTB match rate due to the compensatory WADD component pulling choices toward the opposing option.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    
    validities = np.array([0.95, 0.85, 0.8, 0.75, 0.7])
    
    # WADD predictions
    wadd_diff = (A - B).dot(validities)
    wadd_winner = np.where(wadd_diff > 0, 0, 1)
    
    # TTB predictions
    diff = A - B
    diff_bool = diff != 0
    first_diff_idx = diff_bool.argmax(axis=1)
    first_diff_val = diff[np.arange(len(diff)), first_diff_idx]
    ttb_winner = np.where(first_diff_val > 0, 0, 1)
    
    # Find trials where TTB and WADD make opposite predictions
    disagree = (ttb_winner != wadd_winner) & diff_bool.any(axis=1)
    
    if not np.any(disagree):
        return 0.5
        
    resp = data['response'].values
    match = (resp[disagree] == ttb_winner[disagree])
    
    return float(np.mean(match))
outcome: self_sim=0.7164 (var=0.0226) adversary_sim=0.7022 (var=0.0047) welch_t=+0.429 p=0.6707 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    A = np.stack(data['option_a_ratings'].values)\n    B = np.stack(data['option_b_ratings'].values)\n    resp = data['response'].values\n    \n    # Identify TTB predictions\n    diff = A - B\n    diff_bool = diff != 0\n    first_diff_idx = diff_bool.argmax(axis=1)\n    first_diff_val = diff[np.arange(len(diff)), first_diff_idx]\n    ttb_winner = np.where(first_diff_val > 0, 0, 1)\n    \n    # Identify WADD predictions\n    validities = np.array([0.95, 0.85, 0.8, 0.75, 0.7])\n    wadd_diff = (A - B).dot(validities)\n    wadd_winner = np.where(wadd_diff > 0, 0, 1)\n    \n    # Restrict to trials where TTB and WADD make opposite predictions\n    disagree = (ttb_winner != wadd_winner) & diff_bool.any(axis=1)\n    \n    # Group 1: Disagree trials where the highest valid cue (Cue 0) discriminates\n    group1 = disagree & (first_diff_idx == 0)\n    \n    # Group 2: Disagree trials where Cue 0 is tied, so Cue 1 discriminates\n    group2 = disagree & (first_diff_idx == 1)\n    \n    if np.sum(group1) == 0 or np.sum(group2) == 0:\n        return 0.0\n        \n    # Calculate TTB match rate in both groups\n    match1 = np.mean(resp[group1] == ttb_winner[group1])\n    match2 = np.mean(resp[group2] == ttb_winner[group2])\n    \n    # The contrast is the difference in TTB match rate\n    return float(match1 - match2)",
  "rationale": "In Theory 1 (Strategy Mixture), choices on disagree trials are a probabilistic mixture of TTB and WADD. Because both strategies are locally highly deterministic on these trials, the probability of choosing the TTB-favored option is simply the mixture parameter 'alpha', regardless of whether the first or second cue discriminates. Thus, Theory 1 predicts the TTB match rate will be identical across Group 1 and Group 2, making the difference ~0. \n\nIn Theory 2 (Rank-Dependent Tallying), the decision relies on the difference in weighted tallies. When Cue 0 discriminates (Group 1), the score difference is massive (e.g. ~0.65), leading to a high TTB match rate. However, when Cue 0 is tied and Cue 1 discriminates (Group 2), the weight of Cue 1 (0.25) barely exceeds the sum of the remaining cues (0.2135), resulting in a tiny score difference (0.0365). Because Theory 2 uses a softmax rule, this tiny difference pushes the choice probability very close to 50%. Thus, Theory 2 predicts a drastically lower TTB match rate in Group 2 compared to Group 1, yielding a strongly positive difference."
}
```

## Usage

```json
{
  "prompt_token_count": 4332,
  "candidates_token_count": 802,
  "total_token_count": 12529
}
```
