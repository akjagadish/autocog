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
**Validities (n_features=5):** [0.98, 0.95, 0.6, 0.57, 0.54]

**Trial pairs (n=9):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 9: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** This design tests the fundamental weighting mechanism of the Dynamic Dual Process Strategy Mixture (Advocated) against the Rank-Weighted Additive Theory (Competing). The validities are chosen to have a highly non-linear drop in cardinal values: a small drop between Rank 0 and Rank 1 (0.98 to 0.95), a massive drop between Rank 1 and Rank 2 (0.95 to 0.60), and another small drop between Rank 2 and Rank 3. Across trials, we shift which cue is the best discriminator (Rank 0, 1, or 2) while keeping the Tallying advantage for Option B constant. The Advocated theory predicts that the mixture weight on TTB will remain high and roughly constant for Rank 0 and Rank 1 discriminators, but will fall off a cliff for Rank 2, leading to a highly non-linear shift towards Tallying (Option B). In contrast, the Competing theory assumes weights decay exponentially by rank alone, totally ignoring the cardinal validity gap. It strictly dictates that the choice logits will decay by a constant proportional factor across these rank shifts, mathematically unable to produce the 'cliff' predicted by the Mixture theory.

**Computed schedule:** 9 unique pairs × 10 reps = 90 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
[0] rationale: This metric calculates the interaction between the number of opposing cues (Tallying difference) and the validity of the best discriminating cue. It contrasts the difference in P(Choose A) when shifting the best cue downwards by one rank in a neutral condition (Set A: Trials 1 and 2, Tally diff = 0) versus a strong opposing condition (Set C: Trials 8 and 9, Tally diff = -2). The Dynamic Mixture theory predicts a steeper slope (larger difference) in Set C compared to Set A due to probabilistic strategy switching, whereas the Rank-Weighted Additive theory predicts parallel lines in log-logit space, leading to a markedly different probability interaction.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    def get_trial_type(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        sum_a = sum(a)
        sum_b = sum(b)
        first_a = a.index(1) if 1 in a else -1
        
        if sum_a == 1 and sum_b == 1 and first_a == 0:
            return 'T1'
        elif sum_a == 1 and sum_b == 1 and first_a == 1:
            return 'T2'
        elif sum_a == 1 and sum_b == 3 and first_a == 0:
            return 'T8'
        elif sum_a == 1 and sum_b == 3 and first_a == 1:
            return 'T9'
        return 'Other'
        
    data['trial_type'] = data.apply(get_trial_type, axis=1)
    
    # P(Choose A) is 1.0 - mean(response) since response=0 means A and response=1 means B
    p_a = data[data['trial_type'] != 'Other'].groupby('trial_type')['response'].apply(lambda x: 1.0 - x.mean())
    
    t1 = p_a.get('T1', 0.5)
    t2 = p_a.get('T2', 0.5)
    t8 = p_a.get('T8', 0.5)
    t9 = p_a.get('T9', 0.5)
    
    # Interaction contrast between the slope in the presence of strong opposing cues (Set C) 
    # and the slope with neutral opposing cues (Set A)
    interaction = (t8 - t9) - (t1 - t2)
    return float(interaction)
outcome: self_sim=0.0556 (var=0.0564) adversary_sim=0.0044 (var=0.0839) welch_t=+0.682 p=0.4985 (N=25, alpha=0.01) -> reject

[1] rationale: This metric calculates the within-subject difference in the probability of choosing Option B between Trial 10 (where Tallying strongly favors B by 3 cues) and Trial 5 (where Tallying favors B by 1 cue). In both trials, the best discriminating cue is Cue 0 (favoring A). The Rank-Weighted Additive theory predicts a substantial increase in P(Choose B) as more secondary cues are added to B, leading to a large difference. In contrast, the Dynamic Mixture theory predicts that because the best cue's validity is very high (0.95), the probability of relying on Take-The-Best is near its ceiling, effectively dampening the influence of the additional Tallying cues and resulting in a much smaller difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Identify trial 10 and trial 5 based on option_b_ratings sum
    # T10: B has 4 positive cues. T5: B has 2 positive cues (and A has cue 0).
    # We calculate the difference in P(Choose B) between T10 and T5.
    
    def is_t10(row):
        return sum(row['option_b_ratings']) == 4
        
    def is_t5(row):
        return sum(row['option_b_ratings']) == 2 and row['option_a_ratings'][0] == 1
        
    t10_mask = data.apply(is_t10, axis=1)
    t5_mask = data.apply(is_t5, axis=1)
    
    p_b_t10 = data[t10_mask].groupby('subject_id')['response'].mean()
    p_b_t5 = data[t5_mask].groupby('subject_id')['response'].mean()
    
    # Fill missing subjects with 0.5
    all_subjects = data['subject_id'].unique()
    p_b_t10 = p_b_t10.reindex(all_subjects, fill_value=0.5)
    p_b_t5 = p_b_t5.reindex(all_subjects, fill_value=0.5)
    
    # Return the mean within-subject difference
    return float((p_b_t10 - p_b_t5).mean())

outcome: self_sim=0.0111 (var=0.0268) adversary_sim=0.0178 (var=0.0367) welch_t=-0.132 p=0.8953 (N=25, alpha=0.01) -> reject

[2] rationale: This metric calculates the within-subject difference in the probability of choosing Option A between Trial 1 (where the best discriminating cue is Rank 0 vs Rank 1) and Trial 4 (where the best discriminating cue is Rank 3 vs Rank 4). In the Rank-Weighted Additive theory, the difference in utility between the options decays exponentially with rank, meaning the choice probability heavily regresses towards 0.5 in Trial 4 compared to Trial 1, yielding a strong positive difference. In the Dynamic Mixture theory, the reliance on TTB scales linearly with the validity of the best cue. Because the slope parameter 'alpha' is uniformly distributed around zero in the prior, the expected difference in choice probabilities between these two trials across the population is approximately zero.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    def is_t1(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        return a[0] == 1 and sum(a) == 1 and b[1] == 1 and sum(b) == 1
        
    def is_t4(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        return a[3] == 1 and sum(a) == 1 and b[4] == 1 and sum(b) == 1
        
    t1_mask = data.apply(is_t1, axis=1)
    t4_mask = data.apply(is_t4, axis=1)
    
    p_a_t1 = 1.0 - data[t1_mask].groupby('subject_id')['response'].mean()
    p_a_t4 = 1.0 - data[t4_mask].groupby('subject_id')['response'].mean()
    
    all_subs = data['subject_id'].unique()
    p_a_t1 = p_a_t1.reindex(all_subs, fill_value=0.5)
    p_a_t4 = p_a_t4.reindex(all_subs, fill_value=0.5)
    
    return float((p_a_t1 - p_a_t4).mean())

outcome: self_sim=0.0133 (var=0.0349) adversary_sim=0.1111 (var=0.0711) welch_t=-1.502 p=0.1405 (N=25, alpha=0.01) -> reject

[3] rationale: This metric exploits a double dissociation between the theories. We compare the drop in P(Choose A) when an extra opposing cue is added to Option B. In T2->T6, the best cue is Rank 1 (validity 0.85) and the added cue is Rank 3. In T3->T7, the best cue is Rank 2 (validity 0.75) and the added cue is Rank 4. Under the Rank-Weighted Additive theory, the added penalty in T2->T6 (w3) is strictly larger than in T3->T7 (w4) due to exponential decay, predicting a more negative metric. Under the Dynamic Mixture theory, the reliance on TTB is higher for T2->T6 because the best cue is stronger. Thus, the shift towards Tallying (which pulls choice towards B) is less suppressed in T3->T7, predicting a larger drop in P(Choose A) for T3->T7 and therefore a positive metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    def get_trial(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == (0, 1, 0, 0, 0) and b == (0, 0, 1, 0, 0):
            return 'T2'
        if a == (0, 1, 0, 0, 0) and b == (0, 0, 1, 1, 0):
            return 'T6'
        if a == (0, 0, 1, 0, 0) and b == (0, 0, 0, 1, 0):
            return 'T3'
        if a == (0, 0, 1, 0, 0) and b == (0, 0, 0, 1, 1):
            return 'T7'
        return 'Other'
        
    data['trial_type'] = data.apply(get_trial, axis=1)
    
    # Calculate P(Choose A) = 1 - mean(response)
    p_a = 1.0 - data[data['trial_type'] != 'Other'].groupby('trial_type')['response'].mean()
    
    t2 = p_a.get('T2', 0.5)
    t6 = p_a.get('T6', 0.5)
    t3 = p_a.get('T3', 0.5)
    t7 = p_a.get('T7', 0.5)
    
    # Double dissociation contrast
    return float((t3 - t7) - (t2 - t6))
outcome: self_sim=0.0533 (var=0.0885) adversary_sim=-0.0378 (var=0.1302) welch_t=+0.974 p=0.335 (N=25, alpha=0.01) -> reject

[4] rationale: This metric isolates the non-linear 'cliff' in choice probabilities predicted by the Dynamic Mixture theory, using squared differences to bypass the zero-mean symmetric prior of the alpha parameter. The best discriminating cue validities for Trials 1, 2, and 3 are 0.98, 0.95, and 0.60 respectively. The Advocated theory predicts a negligible change in P(Choose A) between T1 and T2, but a massive drop between T2 and T3, yielding a large positive value for `(P(A|T2) - P(A|T3))^2 - (P(A|T1) - P(A|T2))^2`. The Competing theory's exponential rank decay dictates that choice probability differences shrink as rank drops, meaning the shift from T1 to T2 is generally larger than from T2 to T3, resulting in a negative or near-zero metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    def get_trial(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == (1, 0, 0, 0, 0) and b == (0, 1, 1, 0, 0):
            return 'T1'
        if a == (0, 1, 0, 0, 0) and b == (0, 0, 1, 1, 0):
            return 'T2'
        if a == (0, 0, 1, 0, 0) and b == (0, 0, 0, 1, 1):
            return 'T3'
        return 'Other'
        
    data['trial'] = data.apply(get_trial, axis=1)
    df = data[data['trial'] != 'Other'].copy()
    
    # Calculate P(Choose A) = 1 - response
    df['p_A'] = 1.0 - df['response']
    
    subj_means = df.groupby(['subject_id', 'trial'])['p_A'].mean().unstack(fill_value=0.5)
    
    for t in ['T1', 'T2', 'T3']:
        if t not in subj_means.columns:
            subj_means[t] = 0.5
            
    # Metric: (P(A|T2) - P(A|T3))^2 - (P(A|T1) - P(A|T2))^2
    var2 = (subj_means['T2'] - subj_means['T3']) ** 2
    var1 = (subj_means['T1'] - subj_means['T2']) ** 2
    
    subj_metric = var2 - var1
    
    return float(subj_metric.mean())
outcome: self_sim=0.0082 (var=0.0069) adversary_sim=-0.0090 (var=0.0094) welch_t=+0.674 p=0.5036 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n    \n    def get_trial(row):\n        a = tuple(row['option_a_ratings'])\n        b = tuple(row['option_b_ratings'])\n        if a == (1, 0, 0, 0, 0) and b == (0, 1, 1, 0, 0):\n            return 'T1'\n        if a == (1, 0, 0, 0, 0) and b == (0, 1, 1, 1, 0):\n            return 'T4'\n        if a == (0, 1, 0, 0, 0) and b == (0, 0, 1, 1, 0):\n            return 'T2'\n        if a == (0, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):\n            return 'T5'\n        return 'Other'\n        \n    data['trial'] = data.apply(get_trial, axis=1)\n    df = data[data['trial'] != 'Other'].copy()\n    \n    # P(Choose A) = 1 - response\n    df['p_A'] = 1.0 - df['response']\n    \n    subj_means = df.groupby(['subject_id', 'trial'])['p_A'].mean().unstack(fill_value=0.5)\n    \n    for t in ['T1', 'T4', 'T2', 'T5']:\n        if t not in subj_means.columns:\n            subj_means[t] = 0.5\n            \n    # Drop in P(A) when adding a cue to B at rank 3 (T1 -> T4)\n    drop1 = subj_means['T1'] - subj_means['T4']\n    # Drop in P(A) when adding a cue to B at rank 4 (T2 -> T5)\n    drop2 = subj_means['T2'] - subj_means['T5']\n    \n    return float((drop1 - drop2).mean())",
  "rationale": "This metric evaluates the difference in the impact of adding a single opposing cue to Option B at different ranks. In T1->T4, the added cue is at Rank 3. In T2->T5, the added cue is at Rank 4. Under the Competing (Rank-Weighted Additive) theory, the weight of cues decays exponentially, meaning the penalty for adding a Rank 3 cue is strictly and substantially larger than adding a Rank 4 cue, leading to drop1 > drop2 (a positive metric value). Under the Advocated (Dynamic Mixture) theory, Tallying is unit-weighted, so the Tallying difference shifts by exactly -1 in both cases. Because the best discriminating cue validities for these pairs are nearly identical (0.98 vs 0.95), the probability of using Take-The-Best remains practically unchanged, meaning both drops should be nearly identical, resulting in a metric value close to zero."
}
```

## Usage

```json
{
  "prompt_token_count": 7129,
  "candidates_token_count": 744,
  "total_token_count": 12067
}
```
