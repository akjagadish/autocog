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
**Validities (n_features=7):** [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  trial 3: A=[0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 1]
  trial 4: A=[1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 0]
  trial 5: A=[0, 1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 1, 1]
  trial 6: A=[1, 0, 1, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1, 0, 0]
  trial 7: A=[0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]
  trial 8: A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Sequential Evidence Accumulation) from the Competing Theory (a Strategy Selection mixture of TTB, Tallying, and WADD), we exploit the Advocated Theory's unweighted accumulation and early stopping mechanism. By setting a specific sequence of validities, we can construct trials where the accumulated difference temporarily reaches a threshold (e.g., +2) favoring Option A, causing the Advocated Theory to stop early and choose A. However, by ensuring that Option B wins on the single most valid feature (so TTB chooses B), wins on a greater total number of features (so Tallying chooses B), and wins on the log-odds weighted sum of features (so WADD chooses B), the Competing Theory is structurally forced to predict a strict preference for B across all of its sub-models. This creates a stark qualitative and quantitative dissociation where the Advocated Theory can predict a choice that is literally impossible under the Competing Theory's framework.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Decision-makers use a Sequential Evidence Accumulation strategy with a stopping rule. They inspect features one by one in descending order of their validity, maintaining a running sum of the differences between the options. If the absolute accumulated evidence reaches or exceeds a specific threshold, they stop and make a choice based on that evidence. If the threshold is not reached, they evaluate all features and decide based on the final tally. This allows the model to smoothly transition between Take-The-Best (low threshold) and Tallying (high threshold) behaviors.

**Parameters:**
- threshold: [0.0, 3.0]
- beta: [0.1, 15.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity descending
    order = np.argsort(val)[::-1]
    
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    diff = a - b
    accumulated_evidence = 0.0
    
    for idx in order:
        accumulated_evidence += diff[idx]
        if abs(accumulated_evidence) >= threshold and abs(accumulated_evidence) > 0:
            break
            
    scores = np.array([accumulated_evidence, -accumulated_evidence])
    
    z = beta * scores
    z -= np.max(z)
    p_core = np.exp(z) / np.sum(np.exp(z))
    
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
**Description:** Heuristic Toolbox / Strategy Selection Theory with Graded Sub-models: Decision-makers possess a repertoire of distinct decision heuristics (Take-The-Best, Tallying, and Weighted Additive). Instead of assuming pure determinism within each heuristic, each strategy produces graded choice probabilities via a softmax rule applied to its internal evidence (e.g., validity of the discriminating cue for TTB, cue sum for Tallying, log-odds sum for WADD) using a shared choice temperature. Individuals probabilistically select one of these softened heuristics on any given trial.

**Parameters:**
- w_ttb: [0.0, 50.0]
- w_tally: [0.0, 50.0]
- w_wadd: [0.0, 50.0]
- beta_choice: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    w_ttb = float(parameters["w_ttb"])
    w_tally = float(parameters["w_tally"])
    w_wadd = float(parameters["w_wadd"])
    beta_choice = float(parameters["beta_choice"])
    epsilon = float(parameters["epsilon"])
    
    # Normalize weights to get mixture probabilities for the strategies
    total_w = w_ttb + w_tally + w_wadd + 1e-9
    p_ttb = w_ttb / total_w
    p_tally = w_tally / total_w
    p_wadd = w_wadd / total_w
    
    # Strategy 1: Take-The-Best (TTB) Evidence
    order = np.argsort(val)[::-1]
    ttb_score_a, ttb_score_b = 0.0, 0.0
    for idx in order:
        if a[idx] > b[idx]:
            ttb_score_a, ttb_score_b = val[idx], 0.0
            break
        elif b[idx] > a[idx]:
            ttb_score_a, ttb_score_b = 0.0, val[idx]
            break
            
    z_ttb = beta_choice * np.array([ttb_score_a, ttb_score_b])
    z_ttb -= np.max(z_ttb)
    e_ttb = np.exp(z_ttb)
    ttb_probs = e_ttb / np.sum(e_ttb)
    ttb_a, ttb_b = ttb_probs[0], ttb_probs[1]
            
    # Strategy 2: Tallying (Equal Weights) Evidence
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    
    z_tally = beta_choice * np.array([sum_a, sum_b])
    z_tally -= np.max(z_tally)
    e_tally = np.exp(z_tally)
    tally_probs = e_tally / np.sum(e_tally)
    tally_a, tally_b = tally_probs[0], tally_probs[1]
        
    # Strategy 3: Weighted Additive (WADD) Evidence
    val_clipped = np.clip(val, 0.501, 0.999)
    w = np.log(val_clipped / (1.0 - val_clipped))
    score_a = np.sum(a * w)
    score_b = np.sum(b * w)
    
    z_wadd = beta_choice * np.array([score_a, score_b])
    z_wadd -= np.max(z_wadd)
    e_wadd = np.exp(z_wadd)
    wadd_probs = e_wadd / np.sum(e_wadd)
    wadd_a, wadd_b = wadd_probs[0], wadd_probs[1]
        
    # Probabilistic mixture of discrete strategy predictions
    p_A_core = p_ttb * ttb_a + p_tally * tally_a + p_wadd * wadd_a
    p_B_core = p_ttb * ttb_b + p_tally * tally_b + p_wadd * wadd_b
    p_core = np.array([p_A_core, p_B_core])
    
    # Apply lapse rate
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
[0] rationale: This metric isolates the specific trials (Trial 1 and Trial 2) engineered to dissociate the theories. On these trials, the Competing Theory's sub-models (TTB, Tallying, WADD) all unanimously favor one option (e.g., Option B in Trial 1). However, the Advocated Theory's sequential evidence accumulation can temporarily reach a threshold favoring the opposite option (e.g., Option A in Trial 1) before all features are inspected. By computing the proportion of choices for this 'early stopping' option, we obtain a value that will be near zero for the Competing Theory (only driven by lapses) but significantly higher for the Advocated Theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    def is_t1(x):
        return tuple(x) == (0, 1, 1, 1, 0, 0, 0)
        
    def is_t2(x):
        return tuple(x) == (1, 0, 0, 0, 1, 1, 1)
        
    t1_mask = data['option_a_ratings'].apply(is_t1)
    t2_mask = data['option_a_ratings'].apply(is_t2)
    
    t1_data = data[t1_mask]
    t2_data = data[t2_mask]
    
    t1_rate = (t1_data['response'] == 0).mean() if len(t1_data) > 0 else np.nan
    t2_rate = (t2_data['response'] == 1).mean() if len(t2_data) > 0 else np.nan
    
    rates = [r for r in [t1_rate, t2_rate] if not np.isnan(r)]
    return float(np.mean(rates)) if rates else 0.0
outcome: self_sim=0.3867 (var=0.1358) adversary_sim=0.2458 (var=0.0193) welch_t=+1.788 p=0.08371 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n    \n    # Project list columns to tuples to make them hashable/comparable\n    data['opt_a_tup'] = data['option_a_ratings'].apply(tuple)\n    \n    # Define the Option A and Option B rating patterns for the target trials\n    t1_a = (0, 1, 1, 1, 0, 0, 0)\n    t1_b = (1, 0, 0, 0, 1, 1, 1)\n    \n    t3_a = (0, 0, 1, 1, 1, 1, 0)\n    t3_b = (1, 1, 0, 0, 0, 0, 1)\n    \n    t7_a = (0, 0, 0, 1, 1, 1, 1)\n    t7_b = (1, 1, 1, 0, 0, 0, 0)\n    \n    def get_rate(t_a, t_b):\n        mask_f = data['opt_a_tup'] == t_a\n        mask_b = data['opt_a_tup'] == t_b\n        \n        rates = []\n        if mask_f.sum() > 0:\n            rates.append((data.loc[mask_f, 'response'] == 0).mean())\n        if mask_b.sum() > 0:\n            rates.append((data.loc[mask_b, 'response'] == 1).mean())\n            \n        return float(np.mean(rates)) if rates else 0.0\n\n    r1 = get_rate(t1_a, t1_b)\n    r3 = get_rate(t3_a, t3_b)\n    r7 = get_rate(t7_a, t7_b)\n    \n    return float(r1 + r3 - r7)",
  "rationale": "By design, Option B wins the most valid cue in all trials, meaning Take-The-Best (and thus the Advocated Theory with a low threshold <= 1) always chooses B. However, for higher thresholds, the Advocated Theory's sequential accumulation can favor Option A. Specifically, a medium threshold (1 < thresh <= 2) triggers early stopping in Trial 1, leading to an Option A choice. A high threshold (thresh > 2) prevents early stopping in Trial 3, allowing the final tally to favor Option A, while still triggering early stopping for Option B in Trial 7. Thus, the Advocated Theory predicts that subjects with threshold > 1 will choose Option A in EITHER Trial 1 OR Trial 3, but not Trial 7. In contrast, the Competing Theory's sub-models either uniformly favor B (Trial 1) or have identical strategy mixtures for Trial 3 and Trial 7 (Tallying favors A, TTB/WADD favor B). Therefore, the metric `P(A|Trial 1) + P(A|Trial 3) - P(A|Trial 7)` will be near 0 for the Competing Theory (since P(A|T3) \u2248 P(A|T7) and P(A|T1) \u2248 0), but will be strongly positive (~0.67) for the Advocated Theory, reliably discriminating the two models despite individual parameter differences."
}
```

## Usage

```json
{
  "prompt_token_count": 4647,
  "candidates_token_count": 825,
  "total_token_count": 19980
}
```
