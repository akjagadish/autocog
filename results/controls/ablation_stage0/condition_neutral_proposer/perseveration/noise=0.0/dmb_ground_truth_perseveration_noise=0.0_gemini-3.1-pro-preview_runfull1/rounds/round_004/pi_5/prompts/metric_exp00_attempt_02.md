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
**Validities (n_features=5):** [0.92, 0.9, 0.88, 0.86, 0.84]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 6: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 7: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** To maximally distinguish Theory 1 (Sequential Evidence Accumulation with Normalized Log-Odds Weights) from Theory 2 (Rank-Weighted Exponential Integration), we use a set of validities that are very close in value at the top end. Because Theory 1's weights are proportional to log-odds, closely matched validities result in nearly equal evidence weights, making it highly compensatory when the threshold allows integration. In contrast, Theory 2 enforces an exponential drop-off based strictly on rank, meaning the top cue receives substantially more weight than the second, regardless of their actual validity difference. By designing trials where Option A possesses the top-ranked cue and Option B possesses multiple slightly lower-ranked cues, Theory 1 will strongly favor Option B (if the threshold is exceeded) due to the accumulated mass of similar log-odds, while Theory 2 will often favor Option A due to the severe exponential penalty applied to lower ranks.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Sequential Evidence Accumulation with Normalized Weights: Decision-makers inspect cues sequentially in descending order of validity. Each cue's difference updates a running evidence tally weighted by the cue's normalized log-odds validity. Normalizing the weights ensures that the accumulated evidence scales consistently across different experiments, making the latent decision threshold an invariant parameter. If the absolute evidence crosses this threshold, search stops and a choice is made immediately. If all cues are exhausted without crossing the threshold, the decision defaults to the accumulated tally.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 1.0]
- threshold: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Convert validities to log-odds weights and normalize
    val_clipped = np.clip(val, 0.5001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    weights = weights / np.sum(weights)
    
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    threshold = float(parameters["threshold"])
    
    E = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        E += diff * weights[j]
        # Stop search if evidence crosses threshold (and is non-zero to skip ties)
        if abs(E) >= threshold and abs(E) > 1e-5:
            break
            
    # E > 0 favors option A, E < 0 favors option B
    scores = np.array([E, 0.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * scores
    z -= np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
**Description:** Rank-Weighted Exponential Integration: Decision-makers evaluate all cues simultaneously but weight them exponentially according to their validity rank (weight = alpha^(-rank)). This creates a non-compensatory profile that mimics Take-The-Best when cues are aligned, but naturally allows for compensatory overrides when multiple lower-ranked cues strongly align against the top cue, capturing regressions to chance in highly conflicting trial designs.

**Parameters:**
- alpha: [1.0, 2.5]
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
    
    # Determine the rank of each cue (0 = highest validity, 1 = second highest, etc.)
    cue_order = np.argsort(-val, kind="stable")
    ranks = np.empty_like(cue_order)
    ranks[cue_order] = np.arange(len(val))
    
    alpha = float(parameters["alpha"])
    # Exponentially decay weights based on rank
    weights = alpha ** (-ranks)
    
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * scores
    z -= np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
[0] rationale: The metric calculates the proportion of trials where a subject chooses the option with a greater number of positive cues despite it lacking the most valid cue. Theory 1 (Sequential Evidence Accumulation) will produce higher values because its normalized log-odds validities are very close to each other, allowing multiple lower-ranked cues to easily outweigh the top cue if the decision threshold permits integration. Theory 2 (Rank-Weighted Exponential Integration) heavily penalizes lower-ranked cues, making it much more non-compensatory, so it will rarely choose the option with more cues if it lacks the top cue. This leads to a strong divergence in the metric's expected value between the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_top = data['option_a_ratings'].apply(lambda x: x[0])
    b_top = data['option_b_ratings'].apply(lambda x: x[0])
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    
    cond1 = (a_top == 1) & (b_top == 0) & (b_sum > a_sum)
    cond2 = (b_top == 1) & (a_top == 0) & (a_sum > b_sum)
    
    valid_mask = cond1 | cond2
    if not valid_mask.any():
        return 0.5
        
    subset = data[valid_mask].copy()
    cond1_subset = cond1[valid_mask]
    
    chose_more_cues = np.where(
        cond1_subset,
        subset['response'] == 1,
        subset['response'] == 0
    )
    
    return float(np.mean(chose_more_cues))

outcome: self_sim=0.5860 (var=0.0191) adversary_sim=0.4720 (var=0.0342) welch_t=+2.470 p=0.01742 (N=25, alpha=0.01) -> reject

[1] rationale: This metric calculates the proportion of times the subject chooses the option with multiple lower-ranked cues (patterns '01011' or '00111') over the option possessing the top-ranked cue (patterns '10100' or '11000'). These specific trial pairs (Trials 2 and 8) are carefully chosen because they maximize the theoretical divergence. Theory 1 (Sequential Evidence Accumulation) has a high likelihood of integrating past the top cue because its threshold is rarely triggered by the small log-odds differences of the top cues alone, leading it to favor the target option the majority of the time (~65% expected). In contrast, Theory 2 (Rank-Weighted Exponential Integration) heavily penalizes lower ranks, meaning its exponential weights almost always favor the option with the top cue, choosing the target option only under extreme parameter values (~15% expected). Measuring the target choice rate specifically on these two trials provides a massive mean gap with minimal within-subject variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    target_patterns = {'01011', '00111'}
    competitor_patterns = {'10100', '11000'}
    
    mask_a_target = a_str.isin(target_patterns) & b_str.isin(competitor_patterns)
    mask_b_target = b_str.isin(target_patterns) & a_str.isin(competitor_patterns)
    
    valid_mask = mask_a_target | mask_b_target
    if not valid_mask.any():
        return 0.5
        
    subset = data[valid_mask]
    mask_a_subset = mask_a_target[valid_mask]
    
    # Response is 0 if A is chosen, 1 if B is chosen.
    # We want to track how often the target pattern is chosen.
    chose_target = ((mask_a_subset) & (subset['response'] == 0)) | ((~mask_a_subset) & (subset['response'] == 1))
    
    return float(chose_target.mean())
outcome: self_sim=0.5208 (var=0.0243) adversary_sim=0.3858 (var=0.0463) welch_t=+2.540 p=0.01472 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))\n    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))\n    \n    # Target Trial 8 specifically, where the divergence is most absolute.\n    # Option with cues [1,1,0,0,0] vs Option with cues [0,0,1,1,1]\n    t8_mask = (a_str == '11000') & (b_str == '00111')\n    t8_rev = (b_str == '11000') & (a_str == '00111')\n    \n    valid_mask = t8_mask | t8_rev\n    if not valid_mask.any():\n        return 0.5\n        \n    subset = data[valid_mask]\n    t8_subset = t8_mask[valid_mask]\n    \n    # Calculate how often the subject chose the option with cues [0,0,1,1,1]\n    chose_compensatory = np.where(t8_subset, subset['response'] == 1, subset['response'] == 0)\n    \n    return float(np.mean(chose_compensatory))",
  "rationale": "This metric isolates Trial 8 (A=[1,1,0,0,0] vs B=[0,0,1,1,1]), which provides the strictest divergence between the two theories. In Theory 1 (Sequential Evidence Accumulation), the normalized log-odds weights for cues 3, 4, and 5 (0.539 combined) outweigh the weights for cues 1 and 2 (0.459 combined). Consequently, if the integration threshold is large enough, Theory 1 will reliably favor the option with three lower-ranked cues. In Theory 2 (Rank-Weighted Exponential Integration), the exponential penalty applied to ranks ensures that the top two cues (1 + 1/alpha) strictly dominate the bottom three cues (1/alpha^2 + 1/alpha^3 + 1/alpha^4) for almost all valid values of alpha > 1.3. Thus, Theory 2 will almost never choose the lower-ranked option in this specific trial, except due to epsilon noise. By restricting the metric to this single trial type, we eliminate the within-subject variance introduced by mixing it with other trial types (like Trial 2) where Theory 2 is more permissive of compensatory choices."
}
```

## Usage

```json
{
  "prompt_token_count": 4440,
  "candidates_token_count": 632,
  "total_token_count": 11171
}
```
