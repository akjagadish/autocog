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

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 4: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 6: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 7: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 8: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 9: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 10: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** This design quantitatively dissociates the Advocated theory (TTB + Tallying) from the Competing theory (Subjective WADD + Tallying) by exploiting the step-function nature of Take-The-Best versus the continuous linear nature of WADD. In pairs 1-4, Tallying is tied (1 positive feature each), and TTB favors Option A based on adjacent cues (Cue 1 vs 2, Cue 2 vs 3, etc.). Because TTB makes a discrete 100% decision as long as it finds a discriminating cue, the Advocated theory predicts the exact same choice probability for Option A across all these pairs. For the Competing theory to mimic this constant probability, its subjective weights must form a strict arithmetic progression (w1 - w2 = w2 - w3 = ...). However, if the Competing theory's weights are linear, it must predict exactly 50% (complete guessing) on symmetrically balanced pairs (like Pair 5 and 6, where the sum of indices is equal). The Advocated theory, meanwhile, continues to predict the same strong preference for Option A on these balanced pairs because TTB strictly breaks the tie using the highest validity cue. This creates a geometric impossibility for the Competing theory to simultaneously fit the adjacent-cue trials and the balanced-cue trials.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Mixture of Take-The-Best (TTB) and Tallying: Decision-makers probabilistically mix between two cognitively simple heuristics. Take-The-Best (TTB) is a strict lexicographic rule that inspects cues in order of their validity and chooses based entirely on the first discriminating cue. Tallying is an equal-weight rule that simply counts the number of positive features for each option. The mixture is biased towards Tallying to account for the strong equal-weighting pull observed in human data, while still allowing TTB to explain some non-compensatory choices. A small uniform lapse rate accounts for general response errors.

**Parameters:**
- beta: [0.01, 10.0]
- epsilon: [0.0, 0.3]
- w_tally: [0.5, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    w_tally = float(parameters["w_tally"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # --- Take-The-Best (TTB) ---
    # Order cues by validity (descending)
    cue_order = np.argsort(val)[::-1]
    diff = stim[0] - stim[1]
    
    p_ttb = np.array([0.5, 0.5])
    for cue in cue_order:
        if diff[cue] > 0:
            p_ttb = np.array([1.0, 0.0])
            break
        elif diff[cue] < 0:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # --- Tallying (Equal Weight) ---
    scores_tally = stim.sum(axis=1)
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # --- Mixture ---
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_ttb
    
    n_opts = p_core.shape[0]
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
**Description:** Mixture of Subjective WADD and Tallying: Decision-makers probabilistically mix between a simple equal-weight heuristic (Tallying) and a weighted additive rule that uses subjective, free-varying feature weights rather than objective validities. The subjective weights allow the WADD component to capture non-compensatory, Take-The-Best-like behavior (by assigning heavily skewed weights to features), while the Tallying component accounts for the strong equal-weighting pull observed when individuals fall back on simply counting positive features. Choice probabilities are a mixture of the softmax probabilities derived from each strategy, further blended with a uniform lapse rate to account for response errors. Increased choice noise bounds allow the model to better match human sub-optimal choice frequencies.

**Parameters:**
- beta: [0.01, 5.0]
- epsilon: [0.0, 1.0]
- w_tally: [0.0, 1.0]
- weights: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    w_tally = float(parameters["w_tally"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # --- Subjective WADD ---
    # Uses free subjective weights instead of objective validities
    scores_wadd = stim @ w
    z_wadd = beta * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
        
    # --- Tallying (Equal Weight) ---
    scores_tally = stim.sum(axis=1)
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # --- Mixture ---
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_wadd
    
    n_opts = p_core.shape[0]
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
[0] rationale: In the Advocated theory (TTB + Tallying), for pairs 1 through 7, Tallying always results in a tie (the sum of features is equal for A and B), and TTB always favors Option A because the highest validity discriminating cue points to A. Therefore, the Advocated theory predicts the exact same underlying probability of choosing Option A across all these 7 pairs, meaning the variance of choice proportions across these pairs will be near zero (limited only by binomial noise). In contrast, the Competing theory uses Subjective WADD, which assigns unconstrained, random weights to each feature. Since the feature combinations differ significantly across these 7 pairs, the subjective value difference (and thus the probability of choosing A) will vary widely from pair to pair for any given subject. This metric explicitly captures this structural difference by measuring the within-subject variance of choice proportions across these specific pairs.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    valid_a = {
        (1, 0, 0, 0, 0): 1,
        (0, 1, 0, 0, 0): 2,
        (0, 0, 1, 0, 0): 3,
        (0, 0, 0, 1, 0): 4,
        (1, 0, 0, 0, 1): 5,
        (1, 0, 0, 1, 0): 6,
        (0, 1, 0, 0, 1): 7
    }
    
    # Map each trial to its pair ID (1-7) or 0 if it's not one of the target pairs
    pair_ids = data['option_a_ratings'].apply(lambda x: valid_a.get(tuple(x), 0))
    subset = data[pair_ids > 0].copy()
    subset['pair_id'] = pair_ids[pair_ids > 0]
    
    if len(subset) == 0:
        return 0.0
        
    # Calculate the proportion of times Option A was chosen (response == 0) for each pair
    props = subset.groupby('pair_id')['response'].apply(lambda x: 1.0 - x.mean())
    
    # Return the variance of these choice probabilities
    return float(np.var(props))
outcome: self_sim=0.0005 (var=0.0001) adversary_sim=0.0003 (var=0.0003) welch_t=+0.026 p=0.9792 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    ttb_match = []\n    \n    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):\n        # Only consider trials where Tallying is completely tied\n        if sum(a) == sum(b):\n            ttb_choice = None\n            # Find the choice favored by Take-The-Best (first discriminating cue)\n            # Since validities are strictly decreasing by index, the first difference determines TTB\n            for i in range(len(a)):\n                if a[i] > b[i]:\n                    ttb_choice = 0\n                    break\n                elif b[i] > a[i]:\n                    ttb_choice = 1\n                    break\n            \n            if ttb_choice is not None:\n                ttb_match.append(1 if resp == ttb_choice else 0)\n                \n    if not ttb_match:\n        return 0.5\n        \n    return float(np.mean(ttb_match))",
  "rationale": "This metric calculates the proportion of times a subject's choice matches the Take-The-Best (TTB) heuristic specifically on trials where the Tallying heuristic is perfectly tied (sum of features for A equals sum of features for B). In the Advocated theory (TTB + Tallying), Tallying provides no directional pull on these trials, so the choice is driven by TTB (pulling towards the TTB choice) and random noise, leading to an expected match rate strictly greater than 0.5. In the Competing theory (Subjective WADD + Tallying), the subjective weights are assigned independently of objective validities. Because the feature differences on these tied trials are perfectly symmetric, a random subjective weight vector is equally likely to favor the TTB option as it is to oppose it. Therefore, the Competing theory predicts an expected TTB match rate of exactly 0.5 on these trials. This creates a large mean difference between the theories while pooling over many trials per subject to keep within-subject binomial noise low."
}
```

## Usage

```json
{
  "prompt_token_count": 4262,
  "candidates_token_count": 508,
  "total_token_count": 13841
}
```
