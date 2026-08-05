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

**Trial pairs (n=13):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 9: A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 10: A=[1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 11: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 12: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 13: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]

**Rationale:** This design isolates the effect of the specific ranks of secondary cues while holding constant the identity of the best discriminating cue and the difference in total cue counts between options. The Dynamic Dual Process Strategy Mixture theory computes its predictions based solely on the validity of the top discriminating cue (which determines the mixture weight) and the unweighted sum of all cues (Tallying). Consequently, it predicts identical choice probabilities across trials within each matched set. In contrast, the Rank-Weighted Additive theory integrates all cues using a rank-based exponential decay, making it sensitive to exactly which secondary cues are present. It therefore predicts robust, systematic differences in choice probabilities within the matched sets as the secondary cues shift from higher to lower ranks.

**Computed schedule:** 13 unique pairs × 7 reps = 91 trials per subject.



## ADVOCATED THEORY
**Description:** Rank-Weighted Additive Theory: Individuals integrate all cues but weight them according to an exponential decay based solely on their rank-order of validity. This creates a 'soft' lexicographic rule that acts primarily like Take-The-Best, but allows multiple secondary cues to exert a small, non-zero compensatory pull on the decision. Response variability is captured via a softmax choice rule and a lapse rate.

**Parameters:**
- decay: [0.01, 1.0]
- beta: [0.01, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Ranks: 0 is highest validity
    order = np.argsort(-val, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(val))
    
    # Exponential decay based on rank
    decay = float(parameters["decay"])
    weights = decay ** ranks
    
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    
    return p_final
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
**Description:** Dynamic Dual Process Strategy Mixture: Individuals probabilistically switch between a non-compensatory 'Take-The-Best' (TTB) heuristic and a compensatory 'Tallying' strategy. Crucially, the probability of deploying TTB is not fixed but depends on the structural clarity of the choice—specifically, the validity of the best discriminating cue. When the best discriminating cue is highly valid, individuals are more likely to rely on TTB; when it is weaker, they shift towards Tallying (which integrates all positive cues with equal weight). To account for trials where choice behavior strongly diverges from both heuristics, the model allows for a wide range of decision noise (lapse rate) and potentially inverted or very soft Tallying temperatures.

**Parameters:**
- w_base: [0.0, 1.0]
- alpha: [-2.0, 2.0]
- beta_tally: [-1.0, 10.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Identify the best discriminating cue for Take-The-Best (TTB)
    cue_order = np.argsort(-val, kind="stable")
    winner_ttb = None
    v_disc = 0.5  # default if no cues discriminate
    
    for j in cue_order:
        if a[j] != b[j]:
            winner_ttb = 0 if a[j] > b[j] else 1
            v_disc = val[j]
            break
            
    if winner_ttb == 0:
        p_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        p_ttb = np.array([0.0, 1.0])
    else:
        p_ttb = np.array([0.5, 0.5])
        
    # Strategy 2: Tallying (unit-weight additive)
    scores_tally = np.array([np.sum(a), np.sum(b)])
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * scores_tally
    z_tally = z_tally - np.max(z_tally)
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # Dynamic Mixture Weight
    # w_ttb depends on a base rate and scales with the validity of the discriminating cue
    w_base = float(parameters["w_base"])
    alpha = float(parameters["alpha"])
    
    w_ttb = w_base + alpha * (v_disc - 0.5)
    w_ttb = np.clip(w_ttb, 0.0, 1.0)
    
    p_mix = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # Lapse rate
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final
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
[0] rationale: The Competing theory (Dynamic Mixture) predicts that choice probabilities depend only on the best discriminating cue and the unweighted sum of cues. Thus, it predicts identical choice probabilities for Trial 1 vs Trial 4, and Trial 5 vs Trial 7. In contrast, the Advocated theory (Rank-Weighted Additive) integrates all cues with exponentially decaying weights based on rank. Since the secondary cues in B shift to lower validities in Trials 4 and 7 compared to Trials 1 and 5, B's overall value decreases. This metric calculates the increase in the probability of choosing A as B's secondary cues weaken, which should be positive for the Advocated theory and near zero for the Competing theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    T1_A = (1, 0, 0, 0, 0)
    T1_B = (0, 1, 1, 1, 0)
    
    T4_A = (1, 0, 0, 0, 0)
    T4_B = (0, 0, 1, 1, 1)
    
    T5_A = (0, 1, 0, 0, 0)
    T5_B = (0, 0, 1, 1, 0)
    
    T7_A = (0, 1, 0, 0, 0)
    T7_B = (0, 0, 0, 1, 1)
    
    def get_pa(A, B):
        mask = (data['A_tuple'] == A) & (data['B_tuple'] == B)
        if mask.sum() == 0:
            return 0.5
        return (data.loc[mask, 'response'] == 0).mean()
        
    p_a_t1 = get_pa(T1_A, T1_B)
    p_a_t4 = get_pa(T4_A, T4_B)
    p_a_t5 = get_pa(T5_A, T5_B)
    p_a_t7 = get_pa(T7_A, T7_B)
    
    return (p_a_t4 - p_a_t1) + (p_a_t7 - p_a_t5)

outcome: self_sim=0.1143 (var=0.1265) adversary_sim=-0.0171 (var=0.0846) welch_t=+1.430 p=0.1594 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data = data.copy()\n    data['A_tuple'] = data['option_a_ratings'].apply(tuple)\n    data['B_tuple'] = data['option_b_ratings'].apply(tuple)\n    \n    high_pairs = {\n        ((1, 0, 0, 0, 0), (0, 0, 1, 1, 1)),\n        ((1, 0, 1, 0, 0), (0, 0, 0, 1, 1)),\n        ((0, 1, 1, 0, 0), (0, 0, 0, 1, 1))\n    }\n    \n    low_pairs = {\n        ((1, 0, 0, 0, 0), (0, 1, 1, 1, 0)),\n        ((1, 0, 0, 0, 1), (0, 1, 1, 0, 0)),\n        ((0, 1, 0, 0, 1), (0, 0, 1, 1, 0))\n    }\n    \n    data['pair'] = list(zip(data['A_tuple'], data['B_tuple']))\n    \n    is_high = data['pair'].isin(high_pairs)\n    is_low = data['pair'].isin(low_pairs)\n    \n    p_high = (data.loc[is_high, 'response'] == 0).mean() if is_high.sum() > 0 else 0.5\n    p_low = (data.loc[is_low, 'response'] == 0).mean() if is_low.sum() > 0 else 0.5\n    \n    return float(p_high - p_low)",
  "rationale": "To maximize discrimination while minimizing subject-level variance, we aggregate trials into two groups ('high' and 'low') based on the Rank-Weighted Additive theory's predicted difference in option values, while perfectly controlling for the Dynamic Mixture theory's components. In the 'high' pairs, Option B's positive cues are shifted to lower validity ranks compared to the 'low' pairs, heavily penalizing B under Rank-Weighted Additive integration and thus increasing the probability of choosing A. However, for the Dynamic Mixture model, each 'high' pair is structurally identical to its corresponding 'low' pair: the best discriminating cue is exactly the same, and the unweighted sum of cues for both options is identical. Thus, the Dynamic Mixture model predicts a difference of exactly zero between these two groups, whereas the Rank-Weighted Additive model predicts a robust positive difference. Averaging over 21 trials per group stabilizes the metric against decision noise."
}
```

## Usage

```json
{
  "prompt_token_count": 4423,
  "candidates_token_count": 653,
  "total_token_count": 11053
}
```
