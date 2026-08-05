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
**Validities (n_features=6):** [0.95, 0.85, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  trial 2: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 3: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 4: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]

**Rationale:** To dissociate Rank-Based Threshold Tallying (advocated) from Tally-then-TTB (competing), we use a 6-feature design that pits global tallying against subset tallying. Tally-then-TTB integrates all cues for its primary decision, only falling back to the single highest-validity cue if the global tally is perfectly tied. Rank-Based Threshold Tallying restricts its primary tally to the top-K valid cues. In key dissociation trials, the global tally is either tied or favors one option, but the tally restricted to the top 2 or 3 cues favors the opposite option. For example, when the global tally is tied (3 vs 3) and Option A wins the single best cue but Option B wins the 2nd and 3rd best cues, Tally-then-TTB deterministically chooses A (via TTB tie-breaker), whereas the Rank-Based model chooses B (via top-K tally). Furthermore, we include unequal global tally trials where the minority of cues are high-validity, forcing Tally-then-TTB to follow the global majority while the Rank-Based model follows the top-K majority.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Rank-Based Threshold Tallying: Decision-makers do not integrate all available cues indiscriminately, nor do they rely on a single cue. Instead, they form a consideration set of the top-K most valid cues and apply a compensatory tallying heuristic strictly within this subset. If this primary tally yields a clear preference, they choose that option. If the high-validity cues result in a tie, decision-makers fall back to a secondary tally of the remaining low-validity cues to break the tie probabilistically. This captures the robust tallying behavior in most tasks while restricting integration to 'important' cues, explaining deviations in environments where unweighted tallying over all cues breaks down.

**Parameters:**
- top_k_prop: [0.1, 1.0]
- fallback_weight: [0.0, 0.1]
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
    n_features = len(val)
    
    top_k_prop = float(parameters["top_k_prop"])
    k = max(1, int(np.round(top_k_prop * n_features)))
    
    # Rank validities: lowest = 0, highest = n_features - 1
    ranks = np.argsort(np.argsort(val))
    primary_mask = ranks >= (n_features - k)
    secondary_mask = ~primary_mask
    
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    a_wins_primary = np.sum(a_wins[primary_mask])
    b_wins_primary = np.sum(b_wins[primary_mask])
    
    a_wins_secondary = np.sum(a_wins[secondary_mask])
    b_wins_secondary = np.sum(b_wins[secondary_mask])
    
    fallback = float(parameters["fallback_weight"])
    
    # Normalize secondary wins by number of secondary features 
    # so its maximum contribution is strictly < 1.0, ensuring it acts only as a tie-breaker.
    num_secondary = np.sum(secondary_mask)
    if num_secondary > 0:
        sec_a = a_wins_secondary / num_secondary
        sec_b = b_wins_secondary / num_secondary
    else:
        sec_a = 0.0
        sec_b = 0.0
        
    score_a = a_wins_primary + fallback * sec_a
    score_b = b_wins_primary + fallback * sec_b
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
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
**Description:** Tally-then-TTB (Tally with Validity Tie-Breaker): Decision-makers first compare options by tallying the total number of winning features for each option, ignoring cue validities. If one option has strictly more wins, it is chosen. This captures the compensatory nature of decision-making when there is a clear majority of supporting cues. However, if the tally results in a tie, the decision-maker falls back to a non-compensatory 'Take-The-Best' strategy to break the tie, choosing the option that wins on the single most valid cue. This hybrid model preserves robust Tallying performance on unequal-tally trials while capturing deterministic validity-based tie-breaking on ambiguous trials.

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
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Tallying phase: count strict feature-wise wins
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    score_a = a_wins
    score_b = b_wins
    
    # Tie-breaking phase: if tallies are equal, use TTB
    if a_wins == b_wins:
        cue_order = np.argsort(-val, kind="stable")
        for idx in cue_order:
            if a[idx] > b[idx]:
                score_a += 1.0
                break
            elif b[idx] > a[idx]:
                score_b += 1.0
                break
                
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    # The Tally-then-TTB model's predicted choice exactly matches the value of Option A's 2nd feature (index 1)\n    # across all 4 trial types, whereas the Rank-Based model always predicts the opposite.\n    a_feat1 = data['option_a_ratings'].apply(lambda x: x[1])\n    return float(np.mean(data['response'] == a_feat1))",
  "rationale": "By analyzing the trial designs, we can see that Tally-then-TTB always predicts a choice that corresponds exactly to the value of Option A's second feature (index 1), whereas the Rank-Based Threshold Tallying model predicts the exact opposite choice on all four trial types. Therefore, the proportion of times the subject's response matches Option A's second feature perfectly discriminates the two models."
}
```

## Usage

```json
{
  "prompt_token_count": 3703,
  "candidates_token_count": 220,
  "total_token_count": 6581
}
```
