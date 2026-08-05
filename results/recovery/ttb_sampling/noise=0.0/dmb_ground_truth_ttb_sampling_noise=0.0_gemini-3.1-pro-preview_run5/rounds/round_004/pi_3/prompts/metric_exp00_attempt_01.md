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
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 5: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  trial 7: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 8: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 9: A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 10: A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]

**Rationale:** To dissociate pure Take The Best (TTB) from Probabilistic Cue Retrieval TTB, we vary the absolute validity of the first discriminating cue while keeping its relative rank highest among discriminating cues. Pure TTB strictly stops at the first discriminating cue and predicts that the choice probability for the favored option will be constant (1 - epsilon) across all trials, regardless of whether the first discriminating cue has a validity of 0.95 or 0.65. In contrast, Probabilistic Cue Retrieval TTB posits that the probability of skipping a cue increases as its validity decreases (skip rate is proportional to 1 - validity). Therefore, it predicts a systematic drop in the choice probability for the TTB-favored option on trials where the highest valid discriminating cue has lower absolute validity. By tying the higher-validity cues and forcing the models to rely on cues with progressively lower validities (0.95 down to 0.65), we create a quantitative dissociation: pure TTB predicts a flat accuracy function across these trial types, while Probabilistic TTB predicts a declining accuracy function.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Take The Best (TTB) posits that decision-makers do not integrate all available information. Instead, they rank features by their validity (informativeness) and search through them sequentially. The first feature that discriminates between the two options completely determines the choice, and all subsequent features are ignored. If no features discriminate, the decision-maker guesses randomly. This non-compensatory heuristic allows a single highly valid cue to decisively outweigh any number of less valid cues.

**Parameters:**
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    # We use a stable sort to maintain original order in case of ties.
    order = np.argsort(-validities, kind='stable')
    
    score_a = 0.0
    score_b = 0.0
    
    # Sequential search: find the first discriminating feature
    for idx in order:
        if a[idx] > b[idx]:
            score_a = 1.0
            break
        elif b[idx] > a[idx]:
            score_b = 1.0
            break
            
    if score_a == score_b:
        p_core = np.array([0.5, 0.5])
    else:
        p_core = np.array([score_a, score_b])
        
    epsilon = float(parameters["epsilon"])
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
**Description:** Probabilistic Cue Retrieval TTB: Decision-makers follow the non-compensatory, sequential search architecture of Take The Best (TTB), but execution is noisy at the cue level. When a discriminating cue is evaluated, there is a probability of skipping it or failing to retrieve its value. This skip probability is a function of a base skip rate and is inversely proportional to the cue's validity (operationalized as proportional to 1 - validity). If a cue is skipped, the decision-maker continues to the next most valid cue. This produces slightly higher deviation rates on trials where the best cue is noisy, without defaulting to a fully compensatory linear integration.

**Parameters:**
- p_skip: [0.0, 0.4]
- epsilon: [0.0, 0.4]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    order = np.argsort(-validities, kind='stable')
    
    p_skip_base = float(parameters["p_skip"])
    
    p_A = 0.0
    p_B = 0.0
    p_continue = 1.0
    
    # Sequential search with probabilistic cue retrieval
    for idx in order:
        if a[idx] != b[idx]:
            # Probability of skipping this cue is proportional to its invalidity
            p_skip_cue = p_skip_base * (1.0 - validities[idx])
            p_eval = 1.0 - p_skip_cue
            
            if a[idx] > b[idx]:
                p_A += p_continue * p_eval
            else:
                p_B += p_continue * p_eval
                
            p_continue *= p_skip_cue
            
    # If all discriminating cues are skipped, or no cues discriminate, guess randomly
    p_A += p_continue * 0.5
    p_B += p_continue * 0.5
    
    p_core = np.array([p_A, p_B])
    
    # Global response noise
    epsilon = float(parameters["epsilon"])
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
[0] rationale: Pure Take The Best (TTB) predicts a constant choice probability for the TTB-favored option across all trials, regardless of whether the first discriminating cue has a high validity (e.g., 0.95) or a lower validity (e.g., 0.65). Probabilistic TTB predicts that the probability of skipping a cue increases as its validity decreases. Therefore, it predicts lower adherence to the TTB-favored option on trials where the first discriminating cue appears later in the search order (lower validity). This metric computes the difference in TTB adherence between trials discriminated by early cues (0 and 1) and trials discriminated by later cues (2, 3, and 4).
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    adherence_early = []
    adherence_late = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        disc_cue = -1
        ttb_choice = -1
        for i in range(len(a)):
            if a[i] != b[i]:
                disc_cue = i
                ttb_choice = 0 if a[i] > b[i] else 1
                break
                
        if disc_cue in [0, 1]:
            adherence_early.append(1 if resp == ttb_choice else 0)
        elif disc_cue in [2, 3, 4]:
            adherence_late.append(1 if resp == ttb_choice else 0)
            
    mean_early = np.mean(adherence_early) if adherence_early else 0.0
    mean_late = np.mean(adherence_late) if adherence_late else 0.0
    
    return float(mean_early - mean_late)
outcome: self_sim=0.0139 (var=0.0041) adversary_sim=0.0322 (var=0.0040) welch_t=-1.024 p=0.311 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    cue_indices = []\n    ttb_adherences = []\n    \n    for _, row in data.iterrows():\n        a = row['option_a_ratings']\n        b = row['option_b_ratings']\n        resp = row['response']\n        \n        disc_cue = -1\n        ttb_choice = -1\n        for i in range(len(a)):\n            if a[i] != b[i]:\n                disc_cue = i\n                ttb_choice = 0 if a[i] > b[i] else 1\n                break\n                \n        if disc_cue != -1:\n            cue_indices.append(disc_cue)\n            ttb_adherences.append(1 if resp == ttb_choice else 0)\n            \n    if len(cue_indices) == 0:\n        return 0.0\n        \n    x = np.array(cue_indices, dtype=float)\n    y = np.array(ttb_adherences, dtype=float)\n    \n    if np.std(x) == 0 or np.std(y) == 0:\n        return 0.0\n        \n    r = np.corrcoef(x, y)[0, 1]\n    if np.isnan(r):\n        return 0.0\n    return float(r)",
  "rationale": "Instead of computing the raw difference in TTB adherence between early and late cues, this metric computes the point-biserial correlation between the index of the first discriminating cue and whether the choice adhered to TTB. For pure TTB, adherence is constant across all trials (plus binomial noise), yielding an expected correlation of 0. For Probabilistic Cue Retrieval TTB, adherence systematically decreases as the first discriminating cue index increases (due to higher skip rates for lower validity cues), yielding a negative correlation. Using a correlation coefficient normalizes the slope against trial-level variance, magnifying the effect size of the systematic trend."
}
```

## Usage

```json
{
  "prompt_token_count": 3996,
  "candidates_token_count": 502,
  "total_token_count": 8264
}
```
