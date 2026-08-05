# metric_exp00_attempt_00

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
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 6: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]

**Rationale:** To cleanly dissociate Theory 1 (Cue Difference Threshold with log-odds WADD fallback) from Theory 2 (Mixture of TTB and linear WADD), we exploit the difference between log-odds weighting and linear weighting, as well as the threshold-triggered fallback mechanism. We use five features with a wide spread of validities. Because log-odds grow non-linearly, the highest validity cue (0.95) has a massive log-odds weight compared to moderate cues, but a relatively small linear weight advantage. In trials where the 0.95 cue is pitted against several moderate cues (e.g., 0.75, 0.65, 0.55), the linear sum strongly favors the moderate cues (Linear WADD), while the log-odds sum still favors the 0.95 cue. Theory 1 will either stick to TTB or fall back to log-odds WADD—both of which favor the single highest cue, resulting in a deterministic choice. Theory 2, however, mixes TTB (favoring the top cue) and Linear WADD (favoring the multiple moderate cues), predicting a mixed response. By varying the difference between the top cues and the number of opposing cues, we can also map out the threshold and deficit limit parameters of Theory 1.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Cue Difference Threshold Theory: Decision-makers evaluate options lexicographically but demand that the best discriminating cue provides a decisive advantage. A cue is deemed decisive if its validity exceeds the best opposing cue by a sufficient threshold, or if the sheer number of opposing cues is small enough (below a tallying deficit limit). If the top cue's advantage is challenged by a concentrated block of moderately high opposing cues (failing both conditions), the decision-maker abandons the non-compensatory heuristic and falls back to a compensatory Weighted Additive (WADD) process to resolve the choice.

**Parameters:**
- threshold: [0.0, 1.0]
- deficit_limit: {0, 1, 2, 3, 4, 5}
- beta: [0.1, 25.0]
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
    
    threshold = float(parameters["threshold"])
    deficit_limit = int(parameters["deficit_limit"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Identify which cues favor which option
    favor_a = (a > b)
    favor_b = (b > a)
    
    max_val_a = np.max(val[favor_a]) if np.any(favor_a) else 0.0
    max_val_b = np.max(val[favor_b]) if np.any(favor_b) else 0.0
    
    winner_ttb = 0 if max_val_a > max_val_b else (1 if max_val_b > max_val_a else None)
    
    if winner_ttb is not None:
        # Number of cues favoring the TTB loser
        num_opposing = np.sum(favor_b) if winner_ttb == 0 else np.sum(favor_a)
        
        # Two conditions for TTB to be considered decisive:
        # 1. The validity difference between the best cues of each option is large enough.
        # 2. The number of opposing cues is within the acceptable deficit limit.
        cue_diff_decisive = abs(max_val_a - max_val_b) >= threshold
        deficit_ok = num_opposing <= deficit_limit
        
        if cue_diff_decisive or deficit_ok:
            # Decisive advantage: stick to TTB
            scores = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        else:
            # Not decisive: fall back to compensatory WADD
            val_clipped = np.clip(val, 0.5001, 0.9999)
            log_odds = np.log(val_clipped / (1.0 - val_clipped))
            score_a = np.sum(log_odds * a)
            score_b = np.sum(log_odds * b)
            
            # Normalize compensatory scores by total weight to keep the scale 
            # comparable to the [0, 1] TTB scores for the softmax temperature.
            total_w = np.sum(log_odds)
            scores = np.array([score_a, score_b]) / total_w if total_w > 0 else np.array([0.5, 0.5])
    else:
        scores = np.array([0.5, 0.5])
        
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Identify target trials where the single highest cue is favored by both TTB and Log-odds WADD,\n    # but opposed by Linear WADD (which favors the sum of multiple moderate cues).\n    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))\n    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))\n    \n    target_pairs = {\n        frozenset([\"10000\", \"00111\"]),\n        frozenset([\"10000\", \"00011\"]),\n        frozenset([\"01000\", \"00110\"]),\n        frozenset([\"00100\", \"00011\"])\n    }\n    \n    is_target = data.apply(lambda row: frozenset([row['A_str'], row['B_str']]) in target_pairs, axis=1)\n    subset = data[is_target]\n    \n    if len(subset) == 0:\n        return 0.5\n        \n    def chose_single_cue(row):\n        # The option with exactly one cue is the TTB winner in these specific trials.\n        if sum(row['option_a_ratings']) == 1:\n            return 1.0 if row['response'] == 0 else 0.0\n        else:\n            return 1.0 if row['response'] == 1 else 0.0\n            \n    return float(subset.apply(chose_single_cue, axis=1).mean())",
  "rationale": "This metric isolates trials where the non-linear scaling of log-odds WADD fundamentally diverges from linear WADD. In the targeted trials, the single highest validity cue has a larger log-odds weight than the sum of the opposing moderate cues, meaning Theory 1 (which falls back to log-odds WADD) and TTB both agree on the winner. Consequently, Theory 1 predicts a deterministically high rate of choosing the single cue. Conversely, linear WADD favors the multiple moderate cues, so Theory 2 (which mixes TTB and linear WADD) predicts a significantly lower choice rate for the single cue, reflecting its compensatory mixture."
}
```

## Usage

```json
{
  "prompt_token_count": 4168,
  "candidates_token_count": 570,
  "total_token_count": 9831
}
```
