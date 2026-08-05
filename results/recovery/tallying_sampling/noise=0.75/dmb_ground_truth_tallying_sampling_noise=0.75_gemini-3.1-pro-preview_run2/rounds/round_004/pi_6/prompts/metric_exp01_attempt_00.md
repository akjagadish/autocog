# metric_exp01_attempt_00

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
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 7: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 8: A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  trial 9: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 10: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Environment-Adaptive Strategy Selection Model (EASSM) from Probabilistic Search Take-The-Best (PS-TTB), we exploit their structural differences in strategy execution. EASSM posits a trial-by-trial mixture between strict TTB (which deterministically follows the single best discriminating cue) and Tallying (which counts positive features regardless of validity). PS-TTB performs a probabilistic validity-weighted search and stops at the first discriminator. By using a 5-feature design with a clear validity gradient, we construct trials where the top cue is tied (forcing TTB to the second cue) and Tallying is tied, but the remaining cues are distributed such that PS-TTB's probabilistic search favors the opposite option due to the combined weight of middle-validity cues. We also include classic conflict trials (TTB favors A, Tallying favors B) and agreement trials to robustly estimate the mixture proportions versus search temperature.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
[0] rationale: This metric calculates the difference in the probability of choosing Option A between Trial 3 and Trial 1. In Trial 1, Option A is supported only by the highest-validity cue (cue 1), while B is supported by all other cues. In Trial 3, cue 1 is tied, Option A is supported by cue 2, and B is supported by the remaining cues. For the Environment-Adaptive Strategy Selection Model (EASSM), TTB chooses A in both trials, but Tallying is less strongly opposed to A in Trial 3 than in Trial 1 (because A has 2 positive cues instead of 1). Thus, EASSM predicts a higher probability of choosing A in Trial 3 (metric > 0). In contrast, Probabilistic Search Take-The-Best (PS-TTB) predicts a lower probability of choosing A in Trial 3 because the best cue supporting A is now cue 2 instead of cue 1, making it less likely to be sampled before the cues supporting B. Thus, PS-TTB predicts a lower probability of choosing A in Trial 3 (metric < 0). This qualitative reversal provides a strong discriminatory signal.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t1_mask = data['option_a_ratings'].apply(lambda x: list(x) == [1, 0, 0, 0, 0]) & \
              data['option_b_ratings'].apply(lambda x: list(x) == [0, 1, 1, 1, 1])
    t3_mask = data['option_a_ratings'].apply(lambda x: list(x) == [1, 1, 0, 0, 0]) & \
              data['option_b_ratings'].apply(lambda x: list(x) == [1, 0, 1, 1, 1])
    
    t1_data = data[t1_mask]
    t3_data = data[t3_mask]
    
    if len(t1_data) == 0 or len(t3_data) == 0:
        return 0.0
        
    p_a_t1 = (t1_data['response'] == 0).mean()
    p_a_t3 = (t3_data['response'] == 0).mean()
    
    return float(p_a_t3 - p_a_t1)
outcome: self_sim=0.0022 (var=0.0259) adversary_sim=-0.0267 (var=0.0556) welch_t=+0.506 p=0.6155 (N=25, alpha=0.01) -> reject

[1] rationale: In both Trial 2 and Trial 6, Tallying predicts a tie (Option A and B both have 2 positive cues) and TTB predicts Option A. Specifically, in Trial 2 the highest validity cue favors A, while in Trial 6 the highest validity cue is tied and the second highest favors A. According to the Environment-Adaptive Strategy Selection Model (EASSM), the probability of choosing A should be identical in both trials because the mixture of TTB and Tallying yields the exact same probabilities. In contrast, Probabilistic Search Take-The-Best (PS-TTB) predicts a substantially higher probability of choosing A in Trial 2 than in Trial 6, because Trial 2 features the most valid cue overall favoring A, which is more likely to be sampled first during the probabilistic search than the second-best cue in Trial 6. Thus, the difference in the probability of choosing A between Trial 2 and Trial 6 will be exactly zero under EASSM but strictly positive under PS-TTB.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t2_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 0)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0))
    t6_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 0, 1, 0)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 0, 1))
    
    t2_data = data[t2_mask]
    t6_data = data[t6_mask]
    
    if len(t2_data) == 0 or len(t6_data) == 0:
        return 0.0
        
    p_a_t2 = (t2_data['response'] == 0).mean()
    p_a_t6 = (t6_data['response'] == 0).mean()
    
    return float(p_a_t2 - p_a_t6)
outcome: self_sim=0.0222 (var=0.0607) adversary_sim=-0.0111 (var=0.0564) welch_t=+0.487 p=0.6285 (N=25, alpha=0.01) -> reject

[2] rationale: We are looking for a robust invariant that distinguishes the functional form of the theories. In Trial 1, TTB favors A while Tallying strongly favors B. In Trial 8, TTB favors B while Tallying favors A. The sum of the probability of choosing A in these two trials behaves fundamentally differently in the Environment-Adaptive Strategy Selection Model (which uses a linear trial-by-trial mixture of a strict non-compensatory and a strict compensatory strategy) compared to Probabilistic Search Take-The-Best (which uses a probabilistic validity-weighted search). By summing these probabilities, we cancel out specific parameter-dependent main effects and isolate the structural non-linearity of the probabilistic search against the linear mixture.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Isolate Trial 1 and Trial 8 to exploit the structural differences in how the models handle extreme cue distributions.
    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 1, 1))
    t8_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    
    t1_data = data[t1_mask]
    t8_data = data[t8_mask]
    
    if len(t1_data) == 0 or len(t8_data) == 0:
        return 0.0
        
    p_a_t1 = (t1_data['response'] == 0).mean()
    p_a_t8 = (t8_data['response'] == 0).mean()
    
    # We compute the sum of probabilities of choosing A in T1 and T8.
    # In EASSM, T1 has a strong Tallying pull towards B (diff -3), and T8 has a Tallying pull towards A (diff +1).
    # The specific mixture of TTB and Tallying in EASSM guarantees this sum is bounded tightly in the upper quadrant.
    # In PS-TTB, the probability dynamically scales with tau, producing a structurally different sum.
    return float(p_a_t1 + p_a_t8)
outcome: self_sim=1.0000 (var=0.0509) adversary_sim=0.9800 (var=0.0769) welch_t=+0.280 p=0.7809 (N=25, alpha=0.01) -> reject

[3] rationale: In Trial 1, TTB favors A while Tallying strongly favors B. In Trial 8, TTB favors B while Tallying favors A. Because EASSM dynamically weights TTB vs Tallying based on validity dispersion, and the dispersion in this experiment often falls below the uniformly-sampled threshold, EASSM frequently collapses toward Tallying. This makes P(A|T1) low and P(A|T8) high, yielding a negative difference. By contrast, PS-TTB searches probabilistically but is heavily anchored by the highest-validity cues. Since cue 1 strongly favors A in T1 and B in T8, PS-TTB will reliably produce a high P(A|T1) and a low P(A|T8), yielding a strongly positive difference. While their sum (T1 + T8) is structurally constrained to ~1 in both models (which is why a previous metric failed), their difference (T1 - T8) cleanly separates the mixture-driven sign flip of EASSM from the consistent validity-driven search of PS-TTB.
metric_source:
def metric(data: pd.DataFrame) -> float:
    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 1, 1))
    t8_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    
    t1_data = data[t1_mask]
    t8_data = data[t8_mask]
    
    if len(t1_data) == 0 or len(t8_data) == 0:
        return 0.0
        
    p_a_t1 = (t1_data['response'] == 0).mean()
    p_a_t8 = (t8_data['response'] == 0).mean()
    
    return float(p_a_t1 - p_a_t8)
outcome: self_sim=-0.1511 (var=0.2962) adversary_sim=-0.1844 (var=0.0689) welch_t=+0.276 p=0.7843 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Trial 2: A is supported by cues 1,2; B by 3,4,5\n    t2_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0)) & \\\n              data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))\n    # Trial 9: A is supported by cues 2,3,4; B by 1,5\n    t9_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 1, 0)) & \\\n              data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 1))\n    \n    t2_data = data[t2_mask]\n    t9_data = data[t9_mask]\n    \n    if len(t2_data) == 0 or len(t9_data) == 0:\n        return 1.0\n        \n    p_a_t2 = (t2_data['response'] == 0).mean()\n    p_a_t9 = (t9_data['response'] == 0).mean()\n    \n    return float(p_a_t2 + p_a_t9)",
  "rationale": "In Trial 2, TTB favors A while Tallying favors B (score 2 vs 3). In Trial 9, TTB favors B while Tallying favors A (score 3 vs 2). Because the Environment-Adaptive Strategy Selection Model (EASSM) uses a linear trial-by-trial mixture of these two strategies, the probability of choosing A in Trial 2 and the probability of choosing B in Trial 9 are perfectly symmetric. Thus, EASSM structurally bounds the sum P(A|T2) + P(A|T9) to exactly 1.0 across all parameter combinations. In contrast, Probabilistic Search Take-The-Best (PS-TTB) searches based on individual cue validities. In Trial 2, A is supported by the top two cues (1 and 2), making P(A|T2) very high. In Trial 9, B is supported by cue 1 and cue 5. Because cue 2 is much more valid than cue 5, PS-TTB will choose A in T2 much more frequently than it chooses B in T9. This breaks the symmetry, causing PS-TTB to yield a sum significantly greater than 1.0."
}
```

## Usage

```json
{
  "prompt_token_count": 6144,
  "candidates_token_count": 631,
  "total_token_count": 13922
}
```
