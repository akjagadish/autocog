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

**Trial pairs (n=11):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 3: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 7: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 9: A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 10: A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 11: A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]

**Rationale:** This design leverages an invariance property of the Strategy Mixture (Advocated) model to quantitatively dissociate it from the Sequential Cue Evaluation with Probabilistic Stopping (Competing) model. In the Strategy Mixture model, the predicted choice probability is a fixed combination of the strict Take-The-Best (TTB) prediction (based solely on the first discriminating cue) and the unweighted Tallying prediction (based solely on the total cue count). By holding the highest-validity discriminating cue and the total cue-win counts constant across specific subsets of trials (e.g., Trials 4-6), the Strategy Mixture model is mathematically forced to predict identical choice probabilities. However, the Sequential model evaluates cues sequentially, accumulating validity-weighted evidence and stopping probabilistically. By shifting which specific lower-validity cues are won by each option across these invariant subsets, the accumulated evidence at each stopping point varies. Consequently, the Sequential model predicts systematically shifting choice probabilities across trials where the Strategy Mixture model predicts strict equivalence.

**Computed schedule:** 11 unique pairs × 8 reps = 88 trials per subject.



## ADVOCATED THEORY
**Description:** Strategy Mixture: Decision-makers possess a repertoire of distinct, pure heuristic strategies—such as strict Take-The-Best (TTB) and strict unweighted Tallying. On any given trial, an individual probabilistically selects and executes one of these strategies in its entirety. This approach naturally generates the bimodal and contradictory choice patterns observed in human data, especially in conflict trials, by mixing discrete deterministic predictions rather than softening or blending a single sequential process.

**Parameters:**
- w_ttb: [0.0, 1.0]
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
    
    # Strategy 1: Strict Take-The-Best (TTB)
    p_ttb = 0.5
    for j in cue_order:
        if a[j] > b[j]:
            p_ttb = 1.0
            break
        elif b[j] > a[j]:
            p_ttb = 0.0
            break
            
    # Strategy 2: Strict Unweighted Tallying
    a_count = np.sum(a)
    b_count = np.sum(b)
    if a_count > b_count:
        p_tally = 1.0
    elif b_count > a_count:
        p_tally = 0.0
    else:
        p_tally = 0.5
        
    w_ttb = float(parameters["w_ttb"])
    
    # Probabilistic mixture of discrete deterministic predictions
    p_a = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # Motor/execution noise
    epsilon = float(parameters["epsilon"])
    p_a = (1.0 - epsilon) * p_a + epsilon * 0.5
    
    return np.array([p_a, 1.0 - p_a])
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
[0] rationale: This metric calculates the difference in choice probability for the option with the highest-validity cue (Option A: [1,0,0,0,0]) between Trial 5 and Trial 4. In both trials, Option A has 1 cue and Option B has 3 cues, so the Strategy Mixture model (mixing Take-The-Best and Tallying) predicts identical choice probabilities, yielding a difference of 0. However, the Sequential model accumulates evidence and stops probabilistically; since Option B's cues are weaker in Trial 5 ([0,0,1,1,1]) than in Trial 4 ([0,1,1,1,0]), the Sequential model predicts Option A will be chosen more frequently in Trial 5, yielding a positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    data = data.copy()
    data['A'] = data['option_a_ratings'].apply(tuple)
    data['B'] = data['option_b_ratings'].apply(tuple)
    
    t4_A = (1, 0, 0, 0, 0)
    t4_B = (0, 1, 1, 1, 0)
    
    t5_A = (1, 0, 0, 0, 0)
    t5_B = (0, 0, 1, 1, 1)
    
    mask4 = (data['A'] == t4_A) & (data['B'] == t4_B)
    mask4_rev = (data['A'] == t4_B) & (data['B'] == t4_A)
    
    mask5 = (data['A'] == t5_A) & (data['B'] == t5_B)
    mask5_rev = (data['A'] == t5_B) & (data['B'] == t5_A)
    
    data['chose_T1'] = np.nan
    data.loc[mask4 | mask5, 'chose_T1'] = (data['response'] == 0).astype(float)
    data.loc[mask4_rev | mask5_rev, 'chose_T1'] = (data['response'] == 1).astype(float)
    
    is_t4 = mask4 | mask4_rev
    is_t5 = mask5 | mask5_rev
    
    p_t1_t4 = data.loc[is_t4, 'chose_T1'].mean()
    p_t1_t5 = data.loc[is_t5, 'chose_T1'].mean()
    
    if pd.isna(p_t1_t4) or pd.isna(p_t1_t5):
        return 0.0
        
    return float(p_t1_t5 - p_t1_t4)
outcome: self_sim=-0.0063 (var=0.0324) adversary_sim=0.0088 (var=0.0291) welch_t=-0.303 p=0.7635 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    data = data.copy()\n    data['A'] = data['option_a_ratings'].apply(tuple)\n    data['B'] = data['option_b_ratings'].apply(tuple)\n    \n    t1_A = (1, 0, 0, 1, 1)\n    t1_B = (0, 1, 1, 0, 0)\n    \n    t2_A = (1, 1, 1, 0, 0)\n    t2_B = (0, 0, 0, 1, 1)\n    \n    mask1 = (data['A'] == t1_A) & (data['B'] == t1_B)\n    mask1_rev = (data['A'] == t1_B) & (data['B'] == t1_A)\n    \n    p_a_t1_list = []\n    if mask1.any():\n        p_a_t1_list.extend((data.loc[mask1, 'response'] == 0).astype(float).tolist())\n    if mask1_rev.any():\n        p_a_t1_list.extend((data.loc[mask1_rev, 'response'] == 1).astype(float).tolist())\n        \n    p_a_t1 = np.mean(p_a_t1_list) if p_a_t1_list else 0.5\n    \n    mask2 = (data['A'] == t2_A) & (data['B'] == t2_B)\n    mask2_rev = (data['A'] == t2_B) & (data['B'] == t2_A)\n    \n    p_a_t2_list = []\n    if mask2.any():\n        p_a_t2_list.extend((data.loc[mask2, 'response'] == 0).astype(float).tolist())\n    if mask2_rev.any():\n        p_a_t2_list.extend((data.loc[mask2_rev, 'response'] == 1).astype(float).tolist())\n        \n    p_a_t2 = np.mean(p_a_t2_list) if p_a_t2_list else 0.5\n    \n    return float(p_a_t2 - p_a_t1)",
  "rationale": "This metric calculates the difference in the choice probability of Option A between Trial 2 and Trial 1. In both trials, Option A wins the highest-validity cue (Cue 1) and has 3 total cue wins against Option B's 2. Therefore, the Strategy Mixture model (which mixes Take-The-Best and Tallying) predicts identical, extremely high choice probabilities for Option A in both trials, yielding an expected difference of 0. However, the Sequential model evaluates cues sequentially and stops probabilistically. If it continues past Cue 1, Option A wins the strong subsequent cues (Cues 2 and 3) in Trial 2, but Option B wins them in Trial 1. Thus, the accumulated evidence strongly favors A in Trial 2 but shifts toward B in Trial 1, causing the Sequential model to predict a significantly higher choice probability for A in Trial 2 than in Trial 1."
}
```

## Usage

```json
{
  "prompt_token_count": 4391,
  "candidates_token_count": 812,
  "total_token_count": 13366
}
```
