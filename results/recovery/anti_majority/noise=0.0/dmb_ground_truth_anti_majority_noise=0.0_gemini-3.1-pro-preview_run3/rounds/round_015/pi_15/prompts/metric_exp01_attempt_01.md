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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  trial 3: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 6: A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** This design quantitatively dissociates the Sequential Cue Evaluation with Probabilistic Stopping model (Advocated) from the Strategy Mixture model (Competing) by exploiting the Competing model's invariance to the absolute validity of the highest discriminating cue. In the Strategy Mixture model, choice probabilities depend exclusively on the identity of the single best discriminating cue (which determines the TTB prediction) and the total count of winning cues (which determines the Tallying prediction). By artificially tying the highest-validity cues across a sequence of trials, we can systematically shift the 'best discriminating cue' down the validity hierarchy (e.g., from Cue 1 to Cue 2 to Cue 3) while holding the Tallying score constant. The Strategy Mixture model is mathematically forced to predict identical choice probabilities across these trials because the TTB winner and Tallying winner remain unchanged. In stark contrast, the Advocated model accumulates validity-weighted evidence and stops probabilistically. Tying the top cues forces the Advocated model to evaluate deeper into the cue array, changing both the accumulated evidence margins and the stopping distribution, thus predicting a systematic shift in choice probabilities.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
[0] rationale: This metric calculates the difference in the probability of choosing Option B when its single winning cue is positioned early (cues 2 or 3) versus late (cues 4 or 5) in the validity hierarchy, holding Option A's win on the most valid cue constant. The Competing model (Strategy Mixture) evaluates choice probabilities based only on the identity of the best discriminating cue (TTB) and the total cue count (Tallying), predicting exactly zero difference across these trials. In contrast, the Advocated model (Sequential Cue Evaluation) evaluates cues sequentially and stops probabilistically when a cue discriminates. When Option B's cue appears later, the model is more likely to stop early based on Option A's initial advantage before Option B accumulates any evidence, resulting in a systematically lower probability of choosing Option B in late-cue trials. Thus, the Advocated model predicts a positive difference, while the Competing model predicts zero.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    early_mask = (a_str == '10000') & (b_str.isin(['01000', '00100']))
    late_mask = (a_str == '10000') & (b_str.isin(['00010', '00001']))
    
    early_prob_b = data.loc[early_mask, 'response'].mean()
    late_prob_b = data.loc[late_mask, 'response'].mean()
    
    if pd.isna(early_prob_b) or pd.isna(late_prob_b):
        return 0.0
        
    return float(early_prob_b - late_prob_b)
outcome: self_sim=0.0344 (var=0.0212) adversary_sim=-0.0189 (var=0.0271) welch_t=+1.213 p=0.2312 (N=25, alpha=0.01) -> reject

[1] rationale: This metric contrasts trials where Option B has exactly two winning cues, but their positions are shifted from early/middle (trials 5 and 7) to late (trials 6 and 8), while Option A's winning cues remain fixed at the very beginning. The Competing model relies purely on strict TTB (which only looks at the first cue and favors A) and strict Tallying (which only counts total cues and favors B or ties). Because the cue counts and the best discriminating cue are identical across these trial pairs, the Competing model predicts an exact difference of zero. In contrast, the Advocated model evaluates sequentially and stops probabilistically. When Option B's cues appear earlier, it can accumulate evidence and trigger stopping before the end, leading to a higher probability of choosing B. When Option B's cues are pushed to the end, the model is more likely to stop early while Option A is still leading. Thus, the Advocated model predicts a strongly positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t7 = (a_str == '10000') & (b_str == '01100')
    t8 = (a_str == '10000') & (b_str == '00011')
    t5 = (a_str == '11000') & (b_str == '00110')
    t6 = (a_str == '11000') & (b_str == '00011')
    
    p_b_t7 = data.loc[t7, 'response'].mean()
    p_b_t8 = data.loc[t8, 'response'].mean()
    p_b_t5 = data.loc[t5, 'response'].mean()
    p_b_t6 = data.loc[t6, 'response'].mean()
    
    diff_78 = (p_b_t7 - p_b_t8) if not pd.isna(p_b_t7) and not pd.isna(p_b_t8) else 0.0
    diff_56 = (p_b_t5 - p_b_t6) if not pd.isna(p_b_t5) and not pd.isna(p_b_t6) else 0.0
    
    return float(diff_78 + diff_56)
outcome: self_sim=0.0756 (var=0.0817) adversary_sim=0.0622 (var=0.0988) welch_t=+0.157 p=0.876 (N=25, alpha=0.01) -> reject

[2] rationale: This metric tests the sensitivity of choice probabilities to the magnitude of evidence differences. It compares trials where the TTB-favored option has strong, multi-cue evidence (Trials 5 and 6: top 2 cues vs. 2 weaker cues) against trials where it has weak, single-cue evidence (Trials 1 and 9: cue 1 vs. cue 2). In the Competing model (Strategy Mixture), strict TTB only looks at the first discriminating cue, and strict Tallying only counts total cues. Because both sets of trials feature a tie in total cues (2vs2 or 1vs1) and a win for the TTB-favored option on cue 1, the Competing model predicts identical choice probabilities for the TTB-favored option across all these trials, yielding a difference of exactly zero. In contrast, the Advocated model (Sequential Cue Evaluation) accumulates evidence sequentially. In the strong evidence trials, the TTB-favored option accumulates weight from the top 2 cues before the other option gets any, leading to a near-certain choice. In the weak evidence trials, the opposing option's 2nd-best cue provides strong counter-evidence if the model doesn't stop immediately, significantly reducing the choice probability. Thus, the Advocated model predicts a robust positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Project sequence columns to strings for safe hashing and comparison
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Strong evidence trials: TTB-favored option has the top 2 cues (Trials 5 and 6)
    t5_mask = (a_str == '11000') & (b_str == '00110')
    t6_mask = (a_str == '11000') & (b_str == '00011')
    strong_mask = t5_mask | t6_mask
    
    # Weak evidence trials: TTB-favored option has cue 1, opposing option has cue 2 (Trials 1 and 9)
    t1_mask = (a_str == '10000') & (b_str == '01000')
    t9_mask = (a_str == '01000') & (b_str == '10000')
    
    # Calculate the probability of choosing the TTB-favored option
    # In T5, T6, and T1, TTB favors Option A (response == 0)
    p_strong = (data.loc[strong_mask, 'response'] == 0).mean()
    p_t1 = (data.loc[t1_mask, 'response'] == 0).mean()
    
    # In T9, TTB favors Option B (response == 1)
    p_t9 = (data.loc[t9_mask, 'response'] == 1).mean()
    
    if pd.isna(p_strong) or pd.isna(p_t1) or pd.isna(p_t9):
        return 0.0
        
    p_weak = (p_t1 + p_t9) / 2.0
    
    return float(p_strong - p_weak)
outcome: self_sim=0.0511 (var=0.0249) adversary_sim=0.0144 (var=0.0147) welch_t=+0.921 p=0.3617 (N=25, alpha=0.01) -> reject

[3] rationale: This metric contrasts the probability of choosing Option A when it is strongly favored by the most valid cue (Trials 1-4) versus when Option B is strongly favored by the most valid cue (Trials 9-10). In the Competing model (Strategy Mixture), Tallying predicts a perfect tie (0.5) for both groups, while strict TTB predicts 1.0 for Group 1 and 0.0 for Group 2. This creates a massive, symmetric swing in choice probabilities purely driven by the TTB weight, yielding a large expected difference (mean ~0.375 after execution noise). In the Advocated model (Sequential Cue Evaluation), the decision is dominated by the first cue, meaning the probability of choosing the option with cue 1 is extremely high (often >0.85). Consequently, P(A) swings from very high in Group 1 to very low in Group 2, producing an even larger expected difference (mean ~0.56 after noise). This significant qualitative gap in the magnitude of the swing successfully discriminates the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Group 1: Trials 1 to 4 (A has cue 1, B has one later cue)
    # TTB favors A, Tallying is a tie
    g1_mask = (a_str == '10000') & b_str.isin(['01000', '00100', '00010', '00001'])
    
    # Group 2: Trials 9 and 10 (B has cue 1, A has one later cue)
    # TTB favors B, Tallying is a tie
    g2_mask = b_str.isin(['10000']) & a_str.isin(['01000', '00100'])
    
    p_a_g1 = (data.loc[g1_mask, 'response'] == 0).mean()
    p_a_g2 = (data.loc[g2_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_g1) or pd.isna(p_a_g2):
        return 0.0
        
    # Calculate the difference in probability of choosing A
    return float(p_a_g1 - p_a_g2)
outcome: self_sim=0.5450 (var=0.0728) adversary_sim=0.4361 (var=0.0608) welch_t=+1.490 p=0.1429 (N=25, alpha=0.01) -> reject

[4] rationale: This metric calculates the difference in the probability of choosing the TTB-favored option when the highest discriminating cue is Cue 1 (Trials 1 and 4) versus when it is pushed down to Cue 3 by tying the first two cues (Trials 3 and 6). Across all these trials, the TTB winner is opposed by the Tallying winner. The Strategy Mixture model predicts that the probability of choosing the TTB winner relies exclusively on the mixture parameter 'w_ttb', which is invariant to the absolute validity of the best cue. Thus, it predicts a difference of exactly zero. In contrast, the Sequential Cue Evaluation model accumulates evidence and stops probabilistically; when the discriminating cue is weaker and encountered later, the accumulated evidence margin is smaller and the probability of stopping is shifted, predicting a substantially lower probability of choosing the TTB winner in Trials 3 and 6 compared to Trials 1 and 4. This creates a large, theoretically driven divergence between the models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    t1_mask = (a_str == '10001') & (b_str == '01110')
    t4_mask = (a_str == '01110') & (b_str == '10001')
    
    t3_mask = (a_str == '11100') & (b_str == '11011')
    t6_mask = (a_str == '11011') & (b_str == '11100')
    
    p_ttb_t1 = (data.loc[t1_mask, 'response'] == 0).mean()
    p_ttb_t4 = (data.loc[t4_mask, 'response'] == 1).mean()
    
    p_ttb_t3 = (data.loc[t3_mask, 'response'] == 0).mean()
    p_ttb_t6 = (data.loc[t6_mask, 'response'] == 1).mean()
    
    if pd.isna(p_ttb_t1) or pd.isna(p_ttb_t4) or pd.isna(p_ttb_t3) or pd.isna(p_ttb_t6):
        return 0.0
        
    p_ttb_early = (p_ttb_t1 + p_ttb_t4) / 2.0
    p_ttb_late = (p_ttb_t3 + p_ttb_t6) / 2.0
    
    return float(p_ttb_early - p_ttb_late)
outcome: self_sim=0.0900 (var=0.0200) adversary_sim=0.0150 (var=0.0124) welch_t=+2.083 p=0.04292 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    def subject_metric(sub_df):\n        # Project sequence columns to strings for safe matching\n        a_s = sub_df['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n        b_s = sub_df['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n        \n        # Pair 1 & 4: TTB cue is Cue 1\n        p1_m = (a_s == '10001') & (b_s == '01110')\n        p4_m = (a_s == '01110') & (b_s == '10001')\n        \n        # Pair 2 & 5: TTB cue is Cue 2\n        p2_m = (a_s == '11000') & (b_s == '10110')\n        p5_m = (a_s == '10110') & (b_s == '11000')\n        \n        # Pair 3 & 6: TTB cue is Cue 3\n        p3_m = (a_s == '11100') & (b_s == '11011')\n        p6_m = (a_s == '11011') & (b_s == '11100')\n        \n        # Calculate probability of choosing the TTB winner for each pair group\n        # In p1, p2, p3: Option A is the TTB winner (response == 0)\n        # In p4, p5, p6: Option B is the TTB winner (response == 1)\n        ttb_1 = (sub_df.loc[p1_m, 'response'] == 0).sum() + (sub_df.loc[p4_m, 'response'] == 1).sum()\n        n_1 = p1_m.sum() + p4_m.sum()\n        p_ttb_1 = ttb_1 / n_1 if n_1 > 0 else 0.5\n        \n        ttb_2 = (sub_df.loc[p2_m, 'response'] == 0).sum() + (sub_df.loc[p5_m, 'response'] == 1).sum()\n        n_2 = p2_m.sum() + p5_m.sum()\n        p_ttb_2 = ttb_2 / n_2 if n_2 > 0 else 0.5\n        \n        ttb_3 = (sub_df.loc[p3_m, 'response'] == 0).sum() + (sub_df.loc[p6_m, 'response'] == 1).sum()\n        n_3 = p3_m.sum() + p6_m.sum()\n        p_ttb_3 = ttb_3 / n_3 if n_3 > 0 else 0.5\n        \n        # Overall preference for the TTB winner\n        M = (p_ttb_1 + p_ttb_2 + p_ttb_3) / 3.0\n        \n        # The interaction metric\n        return float((M - 0.5) * (p_ttb_1 - p_ttb_3))\n        \n    if 'subject_id' in data.columns and data['subject_id'].nunique() > 1:\n        return float(data.groupby('subject_id').apply(subject_metric).mean())\n    else:\n        return float(subject_metric(data))",
  "rationale": "This metric capitalizes on a parameter-dependent crossing effect in the Sequential Cue Evaluation (Advocated) model. In the Advocated model, the probability of choosing the TTB winner in early-cue trials (Pairs 1 & 4) versus late-cue trials (Pairs 3 & 6) depends heavily on the probabilistic stopping parameter. Early stoppers (overall P(TTB) > 0.5) choose the TTB winner more often in early-cue trials because the initial validity advantage is stronger (+0.95 vs +0.75). Conversely, late stoppers (overall P(TTB) < 0.5) choose the TTB winner LESS often in early-cue trials because the opposing Tallying advantage at the end of the sequence is stronger (-0.75 vs -0.45). By multiplying the subject's overall TTB preference `(M - 0.5)` by the difference in TTB probability `(P_early - P_late)`, we align these two divergent effects into a consistently positive metric for the Advocated model. In contrast, the Strategy Mixture (Competing) model predicts identical TTB choice probabilities across all pairs, meaning `P_early - P_late` is purely binomial noise with an expectation of zero, uncorrelated with `M`. Thus, the Competing model yields exactly 0, while the Advocated model yields a robustly positive value with low between-subject variance."
}
```

## Usage

```json
{
  "prompt_token_count": 7045,
  "candidates_token_count": 1226,
  "total_token_count": 25093
}
```
