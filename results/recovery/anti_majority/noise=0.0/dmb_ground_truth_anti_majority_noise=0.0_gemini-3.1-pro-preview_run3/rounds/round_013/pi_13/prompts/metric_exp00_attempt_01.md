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

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 5: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 6: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 9: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 10: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 11: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 12: A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Rationale:** This design quantitatively dissociates the Weighted Additive with Power-Function Validity Scaling (WADD-PWR) model (Advocated) from the Sequential Cue Evaluation with Probabilistic Stopping model (Competing) by exploiting a strict mathematical invariance in WADD-PWR. Because WADD-PWR integrates all cues simultaneously and maps the accumulated evidence to choice probabilities via a single softmax function, its predictions are strictly linear in log-odds space. That is, if the cue difference between options in a composite trial (e.g., Trial 3) is the exact sum of the cue differences in two component trials (e.g., Trial 1 and Trial 2), WADD-PWR mathematically guarantees that the log-odds of choosing Option A in the composite trial will exactly equal the sum of the log-odds in the component trials. In contrast, the Competing model evaluates cues sequentially and stops probabilistically, creating a mixture of choices made at different stopping points. This sequential mixture mechanism fundamentally violates log-odds additivity. By presenting multiple triplets of trials with additive cue differences, we can clearly isolate the linear integration of the Advocated model from the probabilistic stopping mechanism of the Competing model.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Weighted Additive with Power-Function Validity Scaling (WADD-PWR): Decision-makers integrate all available cues but weight them non-linearly using a power function of their stated validities. A single parameter, gamma, controls the weighting scheme: positive values approximate Take-The-Best by heavily favoring the most valid cues, zero yields Tallying by weighting all cues equally, and negative values approximate Reverse TTB by favoring less valid cues. The power function naturally anchors a validity of 1.0 to a weight of 1.0, providing built-in numerical stability across the full spectrum of strategies without requiring artificial normalization.

**Parameters:**
- gamma: [-20.0, 20.0]
- beta: [0.1, 20.0]
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
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Apply power-function weighting scheme to capture TTB, Tallying, and Reverse TTB
    # Validities are in [0.5, 1.0], so val ** gamma is numerically stable.
    w = val ** gamma
    
    # Accumulate evidence based on discriminating cues
    diff = a - b
    ev_a = np.sum(w[diff > 0])
    ev_b = np.sum(w[diff < 0])
    
    # Convert to choice probabilities via softmax
    scores = np.array([ev_a, ev_b])
    z = beta * scores
    e = np.exp(z - np.max(z))
    p_core = e / np.sum(e)
    
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


## COMPETING THEORY
**Description:** Sequential Cue Evaluation with Probabilistic Stopping: Decision-makers evaluate cues sequentially in descending order of validity. Upon finding a discriminating cue, they stop with a certain probability and choose based on accumulated evidence. If they continue, they integrate further cues, naturally blending non-compensatory (TTB) and compensatory (Tallying/WADD) behaviors.

**Parameters:**
- p_stop: [0.0, 1.0]
- beta: [0.1, 20.0]
- kappa: [0.0, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    p_stop = float(parameters["p_stop"])
    beta = float(parameters["beta"])
    kappa = float(parameters["kappa"])
    epsilon = float(parameters["epsilon"])
    
    ev_A = 0.0
    ev_B = 0.0
    
    w_reach = 1.0
    p_A_total = 0.0
    p_B_total = 0.0
    
    for i, j in enumerate(cue_order):
        weight = val[j] ** kappa
        ev_A += a[j] * weight
        ev_B += b[j] * weight
        
        is_last = (i == len(cue_order) - 1)
        
        # Stop probabilistically only if the cue discriminates
        if a[j] != b[j]:
            p_s = p_stop
        else:
            p_s = 0.0
            
        # Must stop at the last cue
        if is_last:
            p_s = 1.0
            
        w_stop = w_reach * p_s
        
        # Choice probabilities if stopping at this step
        z = beta * np.array([ev_A, ev_B])
        e = np.exp(z - np.max(z))
        p_choice = e / np.sum(e)
        
        p_A_total += w_stop * p_choice[0]
        p_B_total += w_stop * p_choice[1]
        
        # Update probability of reaching the next step
        w_reach *= (1.0 - p_s)
        
    p_final = np.array([p_A_total, p_B_total])
    return (1.0 - epsilon) * p_final + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
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
[0] rationale: This metric directly tests the core additivity property of the WADD-PWR model. It computes the empirical log-odds of choosing Option A for each trial using Laplace smoothing. It then sums the absolute deviation from log-odds additivity across the three composite triplets. The Advocated WADD-PWR model evaluates cues simultaneously and maps to choice probabilities via softmax, making its log-odds approximately additive (perfectly additive if epsilon=0). The Competing model evaluates cues sequentially with probabilistic stopping, fundamentally violating log-odds additivity, which creates a large and measurable gap between the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    diffs = []
    for a, b in zip(data['option_a_ratings'], data['option_b_ratings']):
        diffs.append(tuple(x - y for x, y in zip(a, b)))
        
    df = pd.DataFrame({
        'diff': diffs,
        'chose_A': 1 - data['response']
    })
    
    grouped = df.groupby('diff')['chose_A'].agg(['sum', 'count'])
    
    def get_log_odds(d):
        if d not in grouped.index:
            return 0.0
        row = grouped.loc[d]
        p = (row['sum'] + 1.0) / (row['count'] + 2.0)
        return np.log(p / (1.0 - p))
        
    triplets = [
        ((1, -1, 0, 0, 0), (0, 0, 1, -1, 0), (1, -1, 1, -1, 0)),
        ((0, 1, -1, 0, 0), (0, 0, 0, 1, -1), (0, 1, -1, 1, -1)),
        ((1, 0, -1, 0, 0), (0, 1, 0, -1, 0), (1, 1, -1, -1, 0))
    ]
    
    score = 0.0
    for c1, c2, comp in triplets:
        lo1 = get_log_odds(c1)
        lo2 = get_log_odds(c2)
        locomp = get_log_odds(comp)
        score += abs(locomp - (lo1 + lo2))
        
    return float(score)
outcome: evaluation failed (IndexingError: Too many indexers)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    def get_trial_type(a, b):\n        a = tuple(a)\n        b = tuple(b)\n        if a == (1, 0, 0, 0, 0) and b == (0, 1, 0, 0, 0): return 1\n        if a == (0, 0, 1, 0, 0) and b == (0, 0, 0, 1, 0): return 2\n        if a == (1, 0, 1, 0, 0) and b == (0, 1, 0, 1, 0): return 3\n        \n        if a == (0, 1, 0, 0, 0) and b == (0, 0, 1, 0, 0): return 4\n        if a == (0, 0, 0, 1, 0) and b == (0, 0, 0, 0, 1): return 5\n        if a == (0, 1, 0, 1, 0) and b == (0, 0, 1, 0, 1): return 6\n        \n        if a == (1, 0, 0, 0, 0) and b == (0, 0, 1, 0, 0): return 7\n        if a == (0, 1, 0, 0, 0) and b == (0, 0, 0, 1, 0): return 8\n        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 0): return 9\n        \n        return 0\n        \n    trial_types = []\n    for a, b in zip(data['option_a_ratings'], data['option_b_ratings']):\n        trial_types.append(get_trial_type(a, b))\n        \n    df = pd.DataFrame({\n        'tt': trial_types,\n        'chose_A': 1 - data['response']\n    })\n    \n    grouped = df.groupby('tt')['chose_A'].agg(['sum', 'count'])\n    \n    def get_log_odds(tt):\n        if tt not in grouped.index:\n            return 0.0\n        row = grouped.loc[tt]\n        p = (row['sum'] + 1.0) / (row['count'] + 2.0)\n        return np.log(p / (1.0 - p))\n        \n    score = 0.0\n    \n    # triplet 1: trial 3 is the composite of 1 and 2\n    lo1 = get_log_odds(1)\n    lo2 = get_log_odds(2)\n    lo3 = get_log_odds(3)\n    score += abs(lo3 - (lo1 + lo2))\n    \n    # triplet 2: trial 6 is the composite of 4 and 5\n    lo4 = get_log_odds(4)\n    lo5 = get_log_odds(5)\n    lo6 = get_log_odds(6)\n    score += abs(lo6 - (lo4 + lo5))\n    \n    # triplet 3: trial 9 is the composite of 7 and 8\n    lo7 = get_log_odds(7)\n    lo8 = get_log_odds(8)\n    lo9 = get_log_odds(9)\n    score += abs(lo9 - (lo7 + lo8))\n    \n    return float(score)",
  "rationale": "This metric evaluates the strictly linear log-odds additivity property of the WADD-PWR model. By computing empirical log-odds (with Laplace smoothing) for choices where the cue differences in a composite trial exactly equal the sum of differences in two component trials, it measures the deviation from additivity. The WADD-PWR model evaluates cues simultaneously and maps to probability via softmax, making its log-odds additive. The Competing model stops probabilistically, violating this additivity. This version replaces tuple indexing with integer trial IDs to avoid pandas IndexingErrors."
}
```

## Usage

```json
{
  "prompt_token_count": 4263,
  "candidates_token_count": 1040,
  "total_token_count": 7593
}
```
