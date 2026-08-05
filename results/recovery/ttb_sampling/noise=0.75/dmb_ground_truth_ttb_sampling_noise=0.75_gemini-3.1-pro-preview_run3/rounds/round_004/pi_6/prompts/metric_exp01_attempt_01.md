# metric_exp01_attempt_01

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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=14):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 6: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 8: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 9: A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 10: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 11: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 12: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 13: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 14: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate Probabilistic Stopping Sequential Search (PSSS) from Threshold-based Sequential Search (TSS), we exploit the functional difference in their stopping rules. TSS employs a hard confidence threshold: search terminates abruptly when accumulated evidence crosses this specific value, leading to step-like changes in choice probabilities across trials. PSSS, on the other hand, evaluates a logistic stopping probability at each step, predicting a smoother, graded change in choices as evidence magnitude varies. We use a 5-feature design with linearly descending validities and construct trials that systematically manipulate the peak accumulated evidence at early steps (e.g., strong initial evidence for A that is later contradicted by B, versus a tie on the first cue followed by moderate evidence for A). TSS predicts stark, qualitative reversals depending on whether the hard threshold is met at step 1 or step 2, whereas PSSS predicts a continuous shift in choice probabilities reflecting the soft stopping mechanism.

**Computed schedule:** 14 unique pairs × 6 reps = 84 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Stopping Sequential Search

**Parameters:**
- threshold: [0.0, 1.5]
- slope: [0.1, 5.0]
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    evidence = 0.0
    
    threshold = float(parameters["threshold"])
    slope = float(parameters["slope"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    p_continue = 1.0
    p_A = 0.0
    
    # Sequential search with probabilistic stopping
    for i, idx in enumerate(order):
        diff = a[idx] - b[idx]
        evidence += diff * validities[idx]
        
        # Determine stopping probability at this step
        if i == len(order) - 1:
            p_stop = 1.0
        else:
            # Logistic function for stopping probability
            z = -slope * (abs(evidence) - threshold)
            z = np.clip(z, -50, 50)  # Prevent overflow
            p_stop = 1.0 / (1.0 + np.exp(z))
            
        p_stop_here = p_continue * p_stop
        p_continue *= (1.0 - p_stop)
        
        # Softmax choice probability if search stops at this step
        z_choice = -beta * evidence
        z_choice = np.clip(z_choice, -50, 50)
        p_A_given_stop = 1.0 / (1.0 + np.exp(z_choice))
        
        p_A += p_stop_here * p_A_given_stop
        
    p_B = 1.0 - p_A
    probs = np.array([p_A, p_B])
    
    # Blend in uniform lapse
    return (1.0 - epsilon) * probs + epsilon * 0.5
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
**Description:** Threshold-based Sequential Search integrates the fast-and-frugal nature of Take-The-Best with the compensatory evidence accumulation of Weighted Additive (WADD) models. Decision-makers search through cues in descending order of their validity, maintaining a running tally of the evidence (weighted by each cue's validity). Instead of stopping at the very first discriminating cue, search terminates only when the absolute accumulated evidence exceeds an internal confidence threshold. If all cues are exhausted without reaching this threshold, the option with the higher accumulated evidence is chosen. This allows for fast, one-reason decisions when a highly valid cue strongly favors one option, while enabling compensatory behavior when early cues provide weak or conflicting evidence.

**Parameters:**
- threshold: [0.0, 1.25]
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.25]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues in descending order of validity
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    evidence = 0.0
    threshold = float(parameters["threshold"])
    
    # Sequential search with evidence accumulation
    for idx in order:
        diff = a[idx] - b[idx]
        evidence += diff * validities[idx]
        
        # Stop search if the confidence threshold is met or exceeded
        if abs(evidence) >= threshold:
            break
            
    # Convert accumulated evidence into discrete choice scores
    scores = np.zeros(2)
    if evidence > 0:
        scores[0] = 1.0
    elif evidence < 0:
        scores[1] = 1.0
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend in uniform lapse
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
[0] rationale: This metric contrasts the probability of choosing Option A in Trial 1 versus Trial 9. In Trial 1, Option A is favored by the most valid cue, but contradicted by all subsequent cues. In Trial 9, Option B is favored by the most valid cue, but Option A accumulates more evidence across later cues. Under Threshold-based Sequential Search (TSS), the decision fundamentally flips based on whether the hard threshold is below or above 0.95, causing a stark bimodal difference between these trials. Probabilistic Stopping Sequential Search (PSSS), with its soft logistic stopping rule, evaluates these evidence trajectories much more continuously, leading to distinct predicted choice proportions.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    t1_mask = (data['A_tuple'] == (1, 0, 0, 0, 0)) & (data['B_tuple'] == (0, 1, 1, 1, 1))
    t9_mask = (data['A_tuple'] == (0, 1, 0, 1, 0)) & (data['B_tuple'] == (1, 0, 0, 0, 1))
    
    p_A_t1 = 1.0 - data.loc[t1_mask, 'response'].mean()
    p_A_t9 = 1.0 - data.loc[t9_mask, 'response'].mean()
    
    # Handle cases where a subject might not have these trials (though by design they should)
    if pd.isna(p_A_t1):
        p_A_t1 = 0.5
    if pd.isna(p_A_t9):
        p_A_t9 = 0.5
        
    return float(p_A_t1 - p_A_t9)
outcome: self_sim=0.2511 (var=0.1214) adversary_sim=0.3533 (var=0.2537) welch_t=-0.835 p=0.4086 (N=25, alpha=0.01) -> reject

[1] rationale: This metric evaluates the difference in the probability of choosing Option B between Trial 1 and Trial 3. In both trials, Option A is favored by the first cue, and Option B is favored by the second cue, leading to identical evidence trajectories up to step 2. In Threshold-based Sequential Search (TSS), a hard threshold determines the choice: if the threshold is low, both trials stop at step 1 and choose A; if the threshold is higher, both trials continue to the end and eventually choose B (since the overall sum favors B). Thus, TSS predicts almost identical choices for both trials, yielding a difference near zero. In contrast, Probabilistic Stopping Sequential Search (PSSS) evaluates stopping probabilistically at each step. In Trial 3, Option A is favored by the third cue, creating a significant positive evidence spike (+0.85) that increases the probability of stopping and choosing A at that step. Trial 1 lacks this spike, leading to a much higher probability of choosing B. Consequently, PSSS predicts a significantly positive difference, perfectly discriminating the soft versus hard stopping mechanisms.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t1_mask = (data['A_str'] == '10000') & (data['B_str'] == '01111')
    t3_mask = (data['A_str'] == '10100') & (data['B_str'] == '01011')
    
    resp_t1 = data.loc[t1_mask, 'response'].mean()
    resp_t3 = data.loc[t3_mask, 'response'].mean()
    
    if pd.isna(resp_t1):
        resp_t1 = 0.5
    if pd.isna(resp_t3):
        resp_t3 = 0.5
        
    return float(resp_t1 - resp_t3)
outcome: self_sim=0.0933 (var=0.0412) adversary_sim=0.0400 (var=0.0300) welch_t=+1.000 p=0.3226 (N=25, alpha=0.01) -> reject

[2] rationale: This metric measures the difference in the probability of choosing Option B between Trial 2 and Trial 6. In both trials, the first two cues favor Option A, accumulating an evidence sum of +1.80 by step 2. Under Threshold-based Sequential Search (TSS), the maximum possible threshold is 1.25. Therefore, TSS will always stop searching by step 2 (or step 1) for both trials and confidently choose Option A, predicting identical choice probabilities and a difference of exactly zero. In contrast, Probabilistic Stopping Sequential Search (PSSS) allows search to continue probabilistically even when evidence is high. If search continues past step 2, Trial 2's subsequent cues strongly favor Option B (driving the final evidence sum to -0.15), while Trial 6's subsequent cues maintain strong evidence for Option A. Thus, PSSS predicts a significantly higher probability of choosing Option B in Trial 2 than in Trial 6, yielding a positive difference that clearly discriminates the theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t2_mask = (data['A_str'] == '11000') & (data['B_str'] == '00111')
    t6_mask = (data['A_str'] == '11100') & (data['B_str'] == '00011')
    
    resp_t2 = data.loc[t2_mask, 'response'].mean()
    resp_t6 = data.loc[t6_mask, 'response'].mean()
    
    if pd.isna(resp_t2):
        resp_t2 = 0.0
    if pd.isna(resp_t6):
        resp_t6 = 0.0
        
    return float(resp_t2 - resp_t6)
outcome: self_sim=0.0400 (var=0.0265) adversary_sim=0.0133 (var=0.0211) welch_t=+0.611 p=0.5441 (N=25, alpha=0.01) -> reject

[3] rationale: We compare the probability of choosing Option B in Trial 7 versus Trial 3. Both trials share the exact same first two cues, leaving the accumulated evidence at +0.10. In Trial 3, the third cue strongly favors Option A (evidence spikes to +0.85), whereas in Trial 7, the third cue favors Option B (evidence drops to -0.65). Both trials ultimately end with negative total evidence (-0.35 and -0.55, respectively) favoring B.

Under Threshold-based Sequential Search (TSS), a subject either stops at the first cue (if threshold <= 0.95) and chooses A, or their threshold is > 0.95. If the threshold > 0.95, the evidence in both trials never exceeds the threshold during intermediate steps (max absolute evidence is 0.85 for T3 and 0.65 for T7). Thus, search always proceeds to the end, where the final negative evidence triggers a choice of B. Therefore, TSS predicts mathematically identical choice probabilities for T3 and T7, yielding an expected difference of exactly zero.

Under Probabilistic Stopping Sequential Search (PSSS), the soft logistic stopping rule evaluates evidence at each step. In Trial 3, the evidence spike to +0.85 at step 3 creates a substantial probability of stopping early and choosing Option A. Trial 7 lacks this positive spike, instead having negative evidence that leans towards B. Consequently, PSSS predicts a significantly higher probability of choosing Option B in Trial 7 than in Trial 3, cleanly dissociating the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t3_mask = (data['A_str'] == '10100') & (data['B_str'] == '01011')
    t7_mask = (data['A_str'] == '10010') & (data['B_str'] == '01101')
    
    resp_t3 = data.loc[t3_mask, 'response'].mean()
    resp_t7 = data.loc[t7_mask, 'response'].mean()
    
    if pd.isna(resp_t3):
        resp_t3 = 0.5
    if pd.isna(resp_t7):
        resp_t7 = 0.5
        
    return float(resp_t7 - resp_t3)
outcome: self_sim=0.1067 (var=0.0459) adversary_sim=0.0022 (var=0.0249) welch_t=+1.962 p=0.05608 (N=25, alpha=0.01) -> reject

[4] rationale: This metric contrasts the probability of choosing Option B in Trial 10 versus Trial 12. Both trials share the exact same first cue (favoring B, evidence -0.9) and both ultimately end with negative total evidence favoring B (-0.7 for T10 and -0.5 for T12). Crucially, in neither trial does the absolute evidence ever exceed 0.9. 

Under Threshold-based Sequential Search (TSS), if a subject's threshold is <= 0.9, search stops at step 1 for both trials. If the threshold is > 0.9, search proceeds to the final step for both trials, where the negative evidence triggers identical discrete choice scores [0, 1]. Because TSS applies its softmax temperature to these discrete scores rather than the continuous evidence margin, it predicts the exact same choice probability for Option B in both trials, yielding a difference of zero.

Under Probabilistic Stopping Sequential Search (PSSS), choice confidence scales continuously with the magnitude of evidence. If search goes to the end, T10's stronger final evidence (-0.7 vs -0.5) results in a higher probability of choosing B. Furthermore, T12 has a positive evidence spike at step 3 (+0.6 favoring A), which creates a chance for search to stop early and erroneously choose A, whereas T10's evidence remains negative (-0.8 at step 3). Both the intermediate stopping probabilities and the final continuous evidence evaluation compound to make PSSS predict a significantly higher probability of choosing B in T10 than in T12, cleanly discriminating the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 10: A=[0, 1, 0, 1, 0], B=[1, 0, 1, 0, 1]
    t10_mask = (data['A_str'] == '01010') & (data['B_str'] == '10101')
    # Trial 12: A=[0, 1, 1, 0, 0], B=[1, 0, 0, 1, 1]
    t12_mask = (data['A_str'] == '01100') & (data['B_str'] == '10011')
    
    # Calculate probability of choosing Option B (response == 1)
    p_B_t10 = data.loc[t10_mask, 'response'].mean()
    p_B_t12 = data.loc[t12_mask, 'response'].mean()
    
    if pd.isna(p_B_t10):
        p_B_t10 = 0.5
    if pd.isna(p_B_t12):
        p_B_t12 = 0.5
        
    return float(p_B_t10 - p_B_t12)
outcome: self_sim=0.0933 (var=0.0568) adversary_sim=0.0167 (var=0.0458) welch_t=+1.196 p=0.2375 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data = data.copy()\n    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))\n    \n    # Strong A trials: Trial 6 and Trial 14\n    strong_mask = ((data['A_str'] == '11100') & (data['B_str'] == '00011')) | \\\n                  ((data['A_str'] == '11110') & (data['B_str'] == '00001'))\n                  \n    # Weak A trials: Trial 11 and Trial 4\n    weak_mask = ((data['A_str'] == '10101') & (data['B_str'] == '01010')) | \\\n                ((data['A_str'] == '10011') & (data['B_str'] == '01100'))\n                \n    # Calculate probability of choosing Option A (response == 0)\n    p_A_strong = 1.0 - data.loc[strong_mask, 'response'].mean()\n    p_A_weak = 1.0 - data.loc[weak_mask, 'response'].mean()\n    \n    if pd.isna(p_A_strong):\n        p_A_strong = 0.5\n    if pd.isna(p_A_weak):\n        p_A_weak = 0.5\n        \n    return float(p_A_strong - p_A_weak)",
  "rationale": "This metric contrasts the probability of choosing Option A in \"Strong A\" trials (Trials 6 and 14) versus \"Weak A\" trials (Trials 4 and 11). In all four of these trials, Threshold-based Sequential Search (TSS) predicts exactly the same underlying probability of choosing Option A. This is because in the Strong A trials, evidence quickly exceeds any possible threshold, stopping search early with positive evidence. In the Weak A trials, the initial evidence is positive (+0.9), but subsequent contradictory evidence never exceeds 0.9 in magnitude. Thus, if a subject's threshold is low (<=0.9), they stop at step 1 with positive evidence; if their threshold is high (>0.9), search proceeds to the final step, where the accumulated evidence is again positive. In all cases, TSS assigns a discrete choice score of 1.0 to Option A, yielding an identical choice probability across all four trials (expected difference = 0). In contrast, Probabilistic Stopping Sequential Search (PSSS) evaluates the continuous magnitude of evidence. The Strong A trials accumulate massive positive evidence (+1.3 to +3.0), leading to near-certainty in choosing A. The Weak A trials end with much weaker positive evidence (+0.5 to +0.7) and experience intermediate drops in evidence, resulting in a significantly lower probability of choosing A. Thus, PSSS predicts a strong positive difference, perfectly discriminating the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 6954,
  "candidates_token_count": 733,
  "total_token_count": 16943
}
```
