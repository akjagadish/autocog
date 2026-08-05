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
**Validities (n_features=5):** [0.92, 0.9, 0.88, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  trial 4: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 8: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Rationale:** To quantitatively and qualitatively dissociate Probabilistic Search Take-The-Best (PS-TTB) from the Environment-Adaptive Strategy Selection Model (EASSM), we exploit how each theory handles ties in a compensatory (Tallying) context. EASSM is a mixture of Tallying and strict TTB. In trials where both options have the same number of positive features (Tallying is indifferent), EASSM will break the tie using its strict TTB component, which deterministically favors the option with the single highest-validity discriminating cue. In contrast, PS-TTB samples cue search orders probabilistically based on validities. By creating a cluster of three high-validity cues (e.g., 0.92, 0.90, 0.88), we can design 'conflict' trials where Option A is favored by the single highest-validity cue (Cue 1), but Option B is favored by the next two high-validity cues (Cues 2 and 3). Because the validities are very close, PS-TTB is highly likely to sample Cue 2 or Cue 3 before Cue 1, leading it to favor Option B. Meanwhile, EASSM will strictly favor Option A (since Tallying is tied and strict TTB chooses A). This yields a powerful qualitative dissociation (EASSM predicts P(A) > 0.5, while PS-TTB predicts P(A) < 0.5).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Search Take-The-Best (PS-TTB)

**Parameters:**
- tau: [0.01, 100.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    diff = stim[0] - stim[1]
    
    validities = np.asarray(parameters['validities'], dtype=float)
    tau = float(parameters['tau'])
    epsilon = float(parameters['epsilon'])
    
    n_features = len(validities)
    n_samples = 1000
    
    # Gumbel-max trick to sample permutations without replacement
    # probabilities proportional to softmax(validities / tau)
    logits = validities / (tau + 1e-6)
    gumbels = np.random.gumbel(size=(n_samples, n_features))
    orders = np.argsort(-(logits + gumbels), axis=1)
    
    diff_sign = np.sign(diff)
    ordered_diffs = diff_sign[orders]
    
    # Find the first discriminating cue in each sampled search order
    abs_diffs = np.abs(ordered_diffs)
    first_non_zero_idx = np.argmax(abs_diffs, axis=1)
    has_non_zero = np.any(abs_diffs > 0, axis=1)
    
    first_non_zero_vals = ordered_diffs[np.arange(n_samples), first_non_zero_idx]
    
    wins_a = np.sum((first_non_zero_vals == 1) & has_non_zero)
    wins_b = np.sum((first_non_zero_vals == -1) & has_non_zero)
    
    total = wins_a + wins_b
    if total > 0:
        p = np.array([wins_a / total, wins_b / total])
    else:
        p = np.array([0.5, 0.5])
        
    return (1.0 - epsilon) * p + epsilon * (np.ones(2) / 2.0)
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
**Description:** Environment-Adaptive Strategy Selection Model (TTB vs. Tallying). Decision-makers probabilistically choose between a non-compensatory strategy (Take-The-Best) and a compensatory strategy (Tallying) on a trial-by-trial basis. Crucially, the probability of deploying TTB is not fixed but adapts to the structure of the environment: it is a logistic function of the dispersion (standard deviation) of the cue validities. In environments with steep validity gradients (high dispersion), TTB is strongly preferred; in environments with flat gradients (low dispersion), decision-makers naturally fall back to Tallying.

**Parameters:**
- disp_slope: [0.0, 100.0]
- disp_threshold: [0.0, 0.5]
- beta_tally: [0.0, 5.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    dispersion = np.std(validities)
    slope = float(parameters["disp_slope"])
    threshold = float(parameters["disp_threshold"])
    
    # Calculate w_ttb dynamically based on the dispersion of cue validities
    w_ttb = 1.0 / (1.0 + np.exp(-slope * (dispersion - threshold)))
    
    beta_tally = float(parameters["beta_tally"])
    epsilon = float(parameters["epsilon"])
    
    # --- TTB Strategy ---
    # Sort cues by validity (descending)
    # We add a tiny amount of noise to validities to break ties consistently if they exist
    order = np.argsort(-(validities + np.random.uniform(0, 1e-6, size=len(validities))))
    
    p_ttb = np.array([0.5, 0.5])
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            p_ttb = np.array([1.0, 0.0])
            break
        elif stim[1, idx] > stim[0, idx]:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # --- Tallying Strategy ---
    scores = np.sum(stim, axis=1)
    z = beta_tally * scores
    z = z - np.max(z)
    e = np.exp(z)
    p_tally = e / np.sum(e)
    
    # --- Mixture ---
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # --- Lapse ---
    n_opts = len(p_core)
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    cue1_chosen = []\n    for _, row in data.iterrows():\n        A = row['option_a_ratings']\n        B = row['option_b_ratings']\n        resp = row['response']\n        \n        # Check if it's a conflict trial where Cue 1 favors one option,\n        # and Cues 2 and 3 favor the other option.\n        if A[0] != B[0] and A[1] != B[1] and A[2] != B[2]:\n            if A[0] != A[1] and A[0] != A[2]:\n                # Conflict trial identified\n                if A[0] > B[0]:\n                    # Cue 1 favors Option A\n                    cue1_chosen.append(1 if resp == 0 else 0)\n                else:\n                    # Cue 1 favors Option B\n                    cue1_chosen.append(1 if resp == 1 else 0)\n                    \n    if len(cue1_chosen) == 0:\n        return 0.5\n    return float(np.mean(cue1_chosen))",
  "rationale": "This metric calculates the proportion of times the subject chooses the option favored by the single highest-validity cue (Cue 1) on 'conflict' trials. Conflict trials are specifically those where Tallying is tied, Cue 1 favors one option, and the next two high-validity cues (Cues 2 and 3) favor the alternative option. Under EASSM, the tie in Tallying causes the model to rely on its strict TTB component, which deterministically chooses the option favored by Cue 1 (yielding a metric close to 1.0). In contrast, PS-TTB searches cues probabilistically; because Cues 2 and 3 also have very high validities, PS-TTB is highly likely to encounter one of them before Cue 1, leading it to favor the alternative option (yielding a metric well below 0.5). This creates a massive, robust qualitative dissociation between the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 3676,
  "candidates_token_count": 503,
  "total_token_count": 7042
}
```
