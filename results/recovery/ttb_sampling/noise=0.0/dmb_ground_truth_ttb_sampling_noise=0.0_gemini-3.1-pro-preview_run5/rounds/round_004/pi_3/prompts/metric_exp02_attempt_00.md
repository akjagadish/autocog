# metric_exp02_attempt_00

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

**Trial pairs (n=7):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0]

**Rationale:** To cleanly dissociate pure Take The Best (TTB) from Probabilistic Cue Retrieval TTB, we use a 5-feature environment with a linear decline in validities and manipulate both the absolute validity of the highest discriminating cue and the alignment of the subsequent cues. Pure TTB strictly stops at the first discriminating cue, predicting a constant choice probability for the favored option regardless of the cue's validity or the alignment of subsequent cues. In contrast, Probabilistic Cue Retrieval TTB assumes a skip probability that increases as cue validity decreases. By creating trials where the first discriminating cue favors Option A but all subsequent cues favor Option B, and progressively tying the higher-validity cues (forcing reliance on less valid cues), Probabilistic TTB predicts a sharp drop in Option A choices. We also include control trials where subsequent cues align with the first discriminating cue, where Probabilistic TTB predicts a recovery in Option A choices, whereas pure TTB predicts no difference.

**Computed schedule:** 7 unique pairs × 13 reps = 91 trials per subject.



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

[1] rationale: Instead of computing the raw difference in TTB adherence between early and late cues, this metric computes the point-biserial correlation between the index of the first discriminating cue and whether the choice adhered to TTB. For pure TTB, adherence is constant across all trials (plus binomial noise), yielding an expected correlation of 0. For Probabilistic Cue Retrieval TTB, adherence systematically decreases as the first discriminating cue index increases (due to higher skip rates for lower validity cues), yielding a negative correlation. Using a correlation coefficient normalizes the slope against trial-level variance, magnifying the effect size of the systematic trend.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    cue_indices = []
    ttb_adherences = []
    
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
                
        if disc_cue != -1:
            cue_indices.append(disc_cue)
            ttb_adherences.append(1 if resp == ttb_choice else 0)
            
    if len(cue_indices) == 0:
        return 0.0
        
    x = np.array(cue_indices, dtype=float)
    y = np.array(ttb_adherences, dtype=float)
    
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
        
    r = np.corrcoef(x, y)[0, 1]
    if np.isnan(r):
        return 0.0
    return float(r)
outcome: self_sim=-0.0241 (var=0.0091) adversary_sim=-0.0579 (var=0.0085) welch_t=+1.278 p=0.2073 (N=25, alpha=0.01) -> reject

[2] rationale: By isolating the trials where the first discriminating cue is the highest validity cue (cue 0) and comparing them strictly to trials where the first discriminating cue is the lowest validity cue (cue 3), we maximize the theoretical divergence between Pure TTB and Probabilistic Cue Retrieval TTB. Pure TTB predicts a constant choice probability (difference ~ 0), whereas Probabilistic TTB predicts a much higher skip rate for cue 3 due to its lower validity, leading to a measurable drop in TTB adherence. Previous attempts diluted this effect by averaging across intermediate cues.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    adherence_0 = []
    adherence_3 = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        disc_cues = [i for i in range(len(a)) if a[i] != b[i]]
        if not disc_cues:
            continue
            
        first_disc = disc_cues[0]
        ttb_choice = 0 if a[first_disc] > b[first_disc] else 1
        
        if first_disc == 0:
            adherence_0.append(1 if resp == ttb_choice else 0)
        elif first_disc == 3:
            adherence_3.append(1 if resp == ttb_choice else 0)
            
    mean_0 = np.mean(adherence_0) if adherence_0 else 0.0
    mean_3 = np.mean(adherence_3) if adherence_3 else 0.0
    
    return float(mean_0 - mean_3)
outcome: self_sim=0.0233 (var=0.0064) adversary_sim=0.0619 (var=0.0070) welch_t=-1.664 p=0.1026 (N=25, alpha=0.01) -> reject

[3] rationale: To maximize statistical power and minimize binomial trial-level noise, this metric computes the unstandardized linear covariance between the index of the first discriminating cue (x) and TTB adherence (y) for each subject. Pure TTB predicts a flat adherence across all cues (expected covariance = 0). Probabilistic TTB predicts a systematic drop in adherence as the cue index increases due to higher skip probabilities for lower-validity cues (expected covariance < 0). Using the unstandardized covariance rather than the correlation coefficient avoids dividing by the variance of y, which is heavily inflated by binomial noise, thereby significantly stabilizing the per-subject metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    covariances = []
    for subj, subj_df in data.groupby('subject_id'):
        x = []
        y = []
        for _, row in subj_df.iterrows():
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
                    
            if disc_cue != -1:
                x.append(disc_cue)
                y.append(1 if resp == ttb_choice else 0)
                
        if len(x) > 1:
            cov = np.cov(x, y)[0, 1]
            covariances.append(cov)
            
    return float(np.mean(covariances)) if covariances else 0.0
outcome: self_sim=-0.0088 (var=0.0010) adversary_sim=-0.0224 (var=0.0012) welch_t=+1.457 p=0.1517 (N=25, alpha=0.01) -> reject

[4] rationale: This metric leverages the exact dissociation intended by the experimental design. Pure TTB strictly stops at the first discriminating cue, meaning adherence to the TTB-favored option will be identical regardless of whether subsequent cues align with or oppose the first cue. Probabilistic Cue Retrieval TTB, however, assumes the first cue might be skipped. If skipped, the choice depends on subsequent cues. If subsequent cues align with the first cue, the model still likely chooses the TTB-favored option; if they oppose it, the choice probability for the TTB option drops significantly. By calculating the within-subject difference in TTB adherence between 'Align' and 'Oppose' trials, we expect a value of ~0 for Pure TTB and a significantly positive value for Probabilistic TTB.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        align_adherence = []
        oppose_adherence = []
        
        for _, row in subj_df.iterrows():
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
                    
            if disc_cue == -1:
                continue
                
            next_choice = -1
            for i in range(disc_cue + 1, len(a)):
                if a[i] != b[i]:
                    next_choice = 0 if a[i] > b[i] else 1
                    break
                    
            if next_choice != -1:
                if next_choice == ttb_choice:
                    align_adherence.append(1 if resp == ttb_choice else 0)
                else:
                    oppose_adherence.append(1 if resp == ttb_choice else 0)
                    
        if align_adherence and oppose_adherence:
            diffs.append(np.mean(align_adherence) - np.mean(oppose_adherence))
            
    return float(np.mean(diffs)) if diffs else 0.0
outcome: self_sim=0.0050 (var=0.0037) adversary_sim=0.0325 (var=0.0047) welch_t=-1.497 p=0.141 (N=25, alpha=0.01) -> reject

[5] rationale: Previous attempts diluted the signal by averaging the 'Align vs Oppose' difference across all cues. Pure TTB predicts this difference is 0 everywhere. Probabilistic Cue Retrieval TTB predicts this difference is proportional to the skip rate of the first discriminating cue. Because cues 0 and 1 have very high validities, their skip rates are near 0, making them behave almost identically to Pure TTB and diluting the metric. By restricting the Align vs Oppose comparison strictly to trials where the first discriminating cue has lower validity (cues 2 and 3), we isolate the trials with the highest theoretical skip rates. This maximizes the expected gap between the two theories while retaining enough trials (48 per subject) to keep the between-subject variance low.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        align_adherence = []
        oppose_adherence = []
        
        for _, row in subj_df.iterrows():
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
                    
            # Focus only on trials where the first discriminating cue has lower validity (index >= 2)
            if disc_cue >= 2:
                next_choice = -1
                for i in range(disc_cue + 1, len(a)):
                    if a[i] != b[i]:
                        next_choice = 0 if a[i] > b[i] else 1
                        break
                
                if next_choice != -1:
                    if next_choice == ttb_choice:
                        align_adherence.append(1 if resp == ttb_choice else 0)
                    else:
                        oppose_adherence.append(1 if resp == ttb_choice else 0)
                        
        if align_adherence and oppose_adherence:
            diffs.append(np.mean(align_adherence) - np.mean(oppose_adherence))
            
    return float(np.mean(diffs)) if diffs else 0.0
outcome: self_sim=0.0092 (var=0.0082) adversary_sim=0.0400 (var=0.0122) welch_t=-1.078 p=0.2864 (N=25, alpha=0.01) -> reject

[6] rationale: To maximize statistical power and minimize binomial trial-level noise, this metric computes the covariance between a theoretical 'propensity to deviate' (X) and the actual choice deviation from TTB (Y) across all 96 trials for each subject. 

For Probabilistic Cue Retrieval TTB, the probability of skipping the first discriminating cue increases linearly with its index (lower validity). If skipped, the choice depends on the remaining cues. Thus, deviations from TTB should be most frequent when the first cue has a high index AND the remaining cues favor the opponent ('Oppose' trials), and least frequent when the remaining cues align with TTB ('Align' trials). We encode this by setting X = (idx + 1) for Oppose trials and X = -(idx + 1) for Align trials. 

Pure TTB predicts a flat deviation rate (driven purely by epsilon noise) across all trials, yielding an expected covariance of 0. Probabilistic TTB predicts a strong positive covariance. By aggregating a continuous contrast over all 96 trials rather than splitting the data into smaller subsets, this metric drastically reduces between-subject variance while maintaining a large effect size.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    covs = []
    for subj, subj_df in data.groupby('subject_id'):
        X = []
        Y = []
        for _, row in subj_df.iterrows():
            a = row['option_a_ratings']
            b = row['option_b_ratings']
            resp = row['response']
            
            idx = -1
            for i in range(len(a)):
                if a[i] != b[i]:
                    idx = i
                    break
            if idx == -1: continue
            
            if a[idx] > b[idx]:
                opp_choice = 1
                score_opp = sum(1 for i in range(idx+1, len(a)) if b[i] > a[i])
                score_ttb = sum(1 for i in range(idx+1, len(a)) if a[i] > b[i])
            else:
                opp_choice = 0
                score_opp = sum(1 for i in range(idx+1, len(a)) if a[i] > b[i])
                score_ttb = sum(1 for i in range(idx+1, len(a)) if b[i] > a[i])
                
            if score_opp > score_ttb:
                x_val = idx + 1
            elif score_ttb > score_opp:
                x_val = -(idx + 1)
            else:
                x_val = 0
                
            y_val = 1 if resp == opp_choice else 0
            
            X.append(x_val)
            Y.append(y_val)
            
        if len(X) > 1:
            covs.append(np.cov(X, Y)[0, 1])
            
    return float(np.mean(covs)) if covs else 0.0
outcome: self_sim=0.0112 (var=0.0068) adversary_sim=0.0478 (var=0.0107) welch_t=-1.383 p=0.1733 (N=25, alpha=0.01) -> reject

[7] rationale: By comparing the proportion of choices that deviate from Take-The-Best (TTB) across different trial types, we can perfectly isolate the probabilistic cue retrieval mechanism. Pure TTB predicts a constant deviation rate (driven purely by response noise epsilon) across all trials. Probabilistic TTB predicts that deviations will occur only when the first discriminating cue is skipped AND the remaining cues favor the alternative option ('Oppose' trials). Furthermore, the skip rate increases linearly with 1 - validity. By applying mathematically optimal matched-filter weights to the 8 trial types—positive weights for Oppose trials proportional to their theoretical skip rates, and negative weights for Align trials to cancel out the subject's baseline epsilon noise—we maximize the expected signal for Probabilistic TTB while strictly enforcing an expected value of 0 for Pure TTB. This linear combination minimizes between-subject variance and maximizes the discriminative t-statistic.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Mathematically optimal matched-filter weights to maximize the signal-to-noise ratio.
    # Signal is proportional to (1 - validity) for Oppose trials, and 0 for Align trials.
    # Weights are proportional to (Signal - mean(Signal)) to ensure they sum to 0,
    # which perfectly cancels out the baseline epsilon noise for Pure TTB.
    weights = {
        "Oppose_0": -1,
        "Oppose_1": 1,
        "Oppose_2": 3,
        "Oppose_3": 5,
        "Align_0": -2,
        "Align_1": -2,
        "Align_2": -2,
        "Align_3": -2
    }
    
    subject_scores = []
    
    for subj, subj_df in data.groupby('subject_id'):
        type_counts = {k: 0 for k in weights}
        type_b_choices = {k: 0 for k in weights}
        
        for _, row in subj_df.iterrows():
            a = row['option_a_ratings']
            b = row['option_b_ratings']
            resp = row['response']
            
            diff_idx = -1
            for i in range(len(a)):
                if a[i] != b[i]:
                    diff_idx = i
                    break
            if diff_idx == -1:
                continue
                
            score_a = sum(1 for i in range(diff_idx+1, len(a)) if a[i] > b[i])
            score_b = sum(1 for i in range(diff_idx+1, len(a)) if b[i] > a[i])
            
            if a[diff_idx] > b[diff_idx]:
                is_oppose = score_b > score_a
                deviation = 1 if resp == 1 else 0
            else:
                is_oppose = score_a > score_b
                deviation = 1 if resp == 0 else 0
                
            t_type = f"{'Oppose' if is_oppose else 'Align'}_{diff_idx}"
            
            if t_type in type_counts:
                type_counts[t_type] += 1
                type_b_choices[t_type] += deviation
                    
        score = 0.0
        for k in weights:
            if type_counts[k] > 0:
                mean_dev = type_b_choices[k] / type_counts[k]
                score += weights[k] * mean_dev
                
        subject_scores.append(score)
        
    return float(np.mean(subject_scores)) if subject_scores else 0.0
outcome: self_sim=0.1000 (var=0.3819) adversary_sim=0.4067 (var=0.6074) welch_t=-1.542 p=0.1301 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    diffs = []\n    for subj, subj_df in data.groupby('subject_id'):\n        hs_dev = 0\n        hs_count = 0\n        bl_dev = 0\n        bl_count = 0\n        \n        for _, row in subj_df.iterrows():\n            a = row['option_a_ratings']\n            b = row['option_b_ratings']\n            resp = row['response']\n            \n            disc_cue = -1\n            ttb_choice = -1\n            for i in range(len(a)):\n                if a[i] != b[i]:\n                    disc_cue = i\n                    ttb_choice = 0 if a[i] > b[i] else 1\n                    break\n                    \n            if disc_cue == -1:\n                continue\n                \n            score_a = sum(1 for i in range(disc_cue+1, len(a)) if a[i] > b[i])\n            score_b = sum(1 for i in range(disc_cue+1, len(a)) if b[i] > a[i])\n            \n            if a[disc_cue] > b[disc_cue]:\n                is_oppose = score_b > score_a\n            else:\n                is_oppose = score_a > score_b\n                \n            is_deviation = 1 if resp != ttb_choice else 0\n            \n            # High Skip trials: first discriminating cue has low validity (idx 2 or 3)\n            # AND subsequent cues strongly favor the opposing option.\n            if is_oppose and disc_cue in [2, 3]:\n                hs_dev += is_deviation\n                hs_count += 1\n            # Baseline trials: either subsequent cues align with TTB (no effect of skipping),\n            # or it's an oppose trial but the first cue has extremely high validity (idx 0).\n            elif (is_oppose and disc_cue == 0) or (not is_oppose):\n                bl_dev += is_deviation\n                bl_count += 1\n                \n        if hs_count > 0 and bl_count > 0:\n            diffs.append((hs_dev / hs_count) - (bl_dev / bl_count))\n            \n    return float(np.mean(diffs)) if diffs else 0.0",
  "rationale": "To cleanly separate Pure TTB from Probabilistic Cue Retrieval TTB, we must isolate the trials where the theoretical skip rate is highest and its consequence is a choice reversal, while carefully controlling for baseline response noise (epsilon) to minimize between-subject variance. We define 'High Skip' trials as those where the first discriminating cue has low validity (indices 2 and 3) AND the subsequent cues favor the opposing option. We contrast the deviation rate on these trials against a broad 'Baseline' set of trials where the skip effect is near zero\u2014either because subsequent cues align with TTB (so skipping doesn't change the choice) or because the first cue has such high validity (index 0) that skipping is exceedingly rare. Pure TTB predicts a difference of 0, as deviations are driven purely by epsilon uniformly across all trials. Probabilistic TTB predicts a significantly positive difference. By pooling 52 trials into the baseline and 26 into the high-skip condition, we drastically reduce the binomial variance compared to previous attempts, maximizing the t-statistic."
}
```

## Usage

```json
{
  "prompt_token_count": 8406,
  "candidates_token_count": 843,
  "total_token_count": 16701
}
```
