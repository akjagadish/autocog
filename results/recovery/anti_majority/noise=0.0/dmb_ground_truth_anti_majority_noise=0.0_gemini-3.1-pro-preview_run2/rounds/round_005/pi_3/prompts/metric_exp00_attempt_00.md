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

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** This design aims to dissociate the Weighted Additive (WADD) theory from Sequential Evidence Accumulation (SEA) by systematically varying the distribution of evidence across the cue hierarchy, pitting single highly-valid cues against coalitions of lower-validity cues. By presenting both extreme conflicts (e.g., Cue 1 vs Cues 2-5) and shifted conflicts (e.g., Cue 2 vs Cues 3-5), we can observe how each model scales evidence. WADD's exponentiation of validities naturally makes it more non-compensatory at lower levels of the hierarchy, whereas SEA's fixed absolute threshold and sequential sampling make it more compensatory at lower levels because individual lower-validity cues struggle to cross the threshold alone.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People make decisions by integrating all available information rather than stopping at the first discriminating cue or simply counting features. According to the Weighted Additive (WADD) theory, decision-makers evaluate each option by computing a sum of its features, weighted by the subjective validity of each feature. However, people may non-linearly amplify the importance of highly valid cues. To capture this, validities are exponentiated by a scaling parameter and then normalized, allowing the model to smoothly interpolate between highly compensatory (Tallying-like) and non-compensatory (TTB-like) decision strategies without shrinking the overall scale of the evidence. Choice probabilities are generated via a softmax function over the weighted sums, with an independent lapse rate for random guessing.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [1.0, 30.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    val = val ** gamma
    val = val / np.sum(val)
    
    a, b = stim[0], stim[1]
    
    # Compute weighted sum of features for each option
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the WADD scores
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
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Sequential Evidence Accumulation with Lateral Inhibition, Leak, and a Collapsing Decision Threshold: Decision-makers sample features sequentially with probability proportional to their validities. Options accumulate evidence subject to leak and lateral inhibition. To prevent indecision when evidence is mixed or tied, the decision threshold collapses over time (urgency signal). A highly valid cue can quickly cross the initially high threshold (mimicking Take-The-Best), but if early cues conflict, the threshold gradually lowers, forcing a decision based on the integrated evidence of later sampled cues (mimicking Tallying or WADD).

**Parameters:**
- gamma: [0.1, 20.0]
- theta: [1.0, 15.0]
- inhibition: [0.0, 1.0]
- leak: [0.0, 1.0]
- collapse_rate: [0.0, 0.5]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    a, b = stim[0], stim[1]
    
    gamma = float(parameters["gamma"])
    theta = float(parameters["theta"])
    inhibition = float(parameters["inhibition"])
    leak = float(parameters["leak"])
    collapse_rate = float(parameters["collapse_rate"])
    epsilon = float(parameters["epsilon"])
    
    # Sampling probability based on validity
    w = val ** gamma
    if np.sum(w) == 0:
        p_sample = np.ones_like(w) / len(w)
    else:
        p_sample = w / np.sum(w)
        
    n_sims = 2000
    max_steps = 100
    
    # Pre-sample all cues for all sims and steps for speed
    samples = np.random.choice(len(val), size=(n_sims, max_steps), p=p_sample)
    
    inc_a = a[samples]
    inc_b = b[samples]
    
    ea = np.zeros(n_sims)
    eb = np.zeros(n_sims)
    
    decided = np.zeros(n_sims, dtype=bool)
    winner = np.full(n_sims, -1)
    
    for step in range(max_steps):
        mask = ~decided
        if not np.any(mask):
            break
            
        curr_inc_a = inc_a[mask, step]
        curr_inc_b = inc_b[mask, step]
        
        # LCA update rule with lateral inhibition and leak
        new_ea = np.maximum(0.0, ea[mask] * (1.0 - leak) + curr_inc_a - inhibition * eb[mask])
        new_eb = np.maximum(0.0, eb[mask] * (1.0 - leak) + curr_inc_b - inhibition * ea[mask])
        
        ea[mask] = new_ea
        eb[mask] = new_eb
        
        # Collapsing threshold
        curr_theta = max(0.01, theta - step * collapse_rate)
        cross_a = new_ea >= curr_theta
        cross_b = new_eb >= curr_theta
        
        just_decided = cross_a | cross_b
        
        if np.any(just_decided):
            jd_indices = np.where(just_decided)[0]
            
            for idx in jd_indices:
                if cross_a[idx] and not cross_b[idx]:
                    winner_val = 0
                elif cross_b[idx] and not cross_a[idx]:
                    winner_val = 1
                else:
                    if new_ea[idx] > new_eb[idx]:
                        winner_val = 0
                    elif new_eb[idx] > new_ea[idx]:
                        winner_val = 1
                    else:
                        winner_val = np.random.choice([0, 1])
                
                orig_idx = np.where(mask)[0][idx]
                winner[orig_idx] = winner_val
                decided[orig_idx] = True

    undecided = ~decided
    if np.any(undecided):
        ea_un = ea[undecided]
        eb_un = eb[undecided]
        
        # For those that haven't crossed threshold, highest evidence wins
        ties = ea_un == eb_un
        win_un = np.where(ea_un > eb_un, 0, 1)
        if np.any(ties):
            win_un[ties] = np.random.choice([0, 1], size=ties.sum())
            
        winner[undecided] = win_un
        
    p_A = np.sum(winner == 0) / n_sims
    p_B = np.sum(winner == 1) / n_sims
    
    p = np.array([p_A, p_B])
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_sum = data['option_a_ratings'].apply(sum)\n    b_sum = data['option_b_ratings'].apply(sum)\n    mask = (a_sum == 1) & (b_sum > 1)\n    if mask.sum() == 0:\n        return 0.0\n    return float((data[mask]['response'] == 0).mean())",
  "rationale": "Measures the probability of choosing Option A when it is supported by exactly one cue but Option B is supported by multiple lower-validity cues (Trials 1, 2, and 6). WADD's exponentiation can strongly favor the single highly valid cue, whereas SEA's random sequential sampling and collapsing threshold often lead to selecting the option with more numerous positive features (Option B). This structural difference in how the models handle single vs. multiple cues should yield a significant contrast."
}
```

## Usage

```json
{
  "prompt_token_count": 3959,
  "candidates_token_count": 221,
  "total_token_count": 6355
}
```
