# metric_exp02_attempt_03

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
  trial 1: A=[1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 1]  B=[0, 0, 1, 0, 0]
  trial 4: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 0]
  trial 5: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1]
  trial 6: A=[0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 7: A=[1, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 8: A=[1, 1, 0, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 0]
  trial 10: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0]

**Rationale:** This design quantitatively dissociates the Advocated model (Take-The-Best with Compensatory Verification) from the Competing model (WADD-PWR) by exploiting a strict invariance in the Advocated model. The Advocated model evaluates the single best discriminating cue to determine the 'TTB winner' (assigning it a fixed weight, completely ignoring its actual validity), and then tallies the remaining cues evaluated *after* the discriminating cue. By systematically shifting a single tied cue or a single winning cue across the validity hierarchy, we can hold both the TTB winner and the tally difference strictly constant. For example, if Option A is all 1s and Option B has a single 1, shifting Option B's 1 from the highest-validity cue down to lower-validity cues changes the identity of the discriminating cue but perfectly preserves the tally difference (remaining_a - remaining_b = 3 in all cases). Consequently, the Advocated model is mathematically forced to predict identical choice probabilities across these trials. In stark contrast, the Competing model (WADD-PWR) integrates all cues using a power function of their specific validities. Shifting the location of the cues fundamentally alters the accumulated evidence, leading the Competing model to predict systematically shifting choice probabilities.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Take-The-Best with Compensatory Verification (Sub-linear Tallying)

**Parameters:**
- beta_ttb: [0.0, 10.0]
- beta_tally: [0.0, 10.0]
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
    
    ttb_winner = -1
    disc_idx = -1
    for i, j in enumerate(cue_order):
        if a[j] != b[j]:
            ttb_winner = 0 if a[j] > b[j] else 1
            disc_idx = i
            break
            
    if ttb_winner == -1:
        return np.array([0.5, 0.5])
        
    # Tally remaining cues (those evaluated after the discriminating cue)
    remaining_a = 0.0
    remaining_b = 0.0
    for j in cue_order[disc_idx+1:]:
        remaining_a += a[j]
        remaining_b += b[j]
        
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    
    tally_diff = remaining_a - remaining_b
    compressed_diff = np.sign(tally_diff) * np.sqrt(np.abs(tally_diff))
    
    score_a = beta_ttb * (1.0 if ttb_winner == 0 else 0.0) + beta_tally * compressed_diff
    score_b = beta_ttb * (1.0 if ttb_winner == 1 else 0.0)
    
    scores = np.array([score_a, score_b])
    z = scores - np.max(scores)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    epsilon = float(parameters["epsilon"])
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
**Description:** Weighted Additive with Power-Function Validity Scaling (WADD-PWR): Decision-makers integrate all available cues but weight them non-linearly using a power function of their stated validities. A single parameter, gamma, controls the weighting scheme: positive values approximate Take-The-Best by heavily favoring the most valid cues, zero yields Tallying by weighting all cues equally, and negative values approximate Reverse TTB by favoring less valid cues. The power function naturally anchors a validity of 1.0 to a weight of 1.0, providing built-in numerical stability across the full spectrum of strategies without requiring artificial normalization.

**Parameters:**
- gamma: [-20.0, 20.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Apply power-function weighting scheme to capture TTB, Tallying, and Reverse TTB
    # Validities are in [0.5, 1.0], so val ** gamma is numerically stable.
    w = val ** gamma
    
    # Accumulate evidence based on discriminating cues
    diff = a - b
    ev_a = np.sum(w[diff > 0])
    ev_b = np.sum(w[diff < 0])
    
    # Convert to choice probabilities via softmax
    scores = np.array([ev_a, ev_b])
    z = beta * scores
    e = np.exp(z - np.max(z))
    p_core = e / np.sum(e)
    
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
[0] rationale: This metric calculates the difference in choice probability for the option [1, 0, 0, 0, 0] when paired against [0, 0, 0, 1, 1] (Trial 2) versus when paired against [0, 1, 1, 0, 0] (Trial 1). The Advocated model resolves both trials using the first cue (validity 0.95), and then tallies the remaining cues. Since the alternative option wins exactly two remaining cues in both cases, the tally difference is identical, meaning the Advocated model predicts a difference of exactly 0. The Competing model (WADD-PWR), however, evaluates the specific validities of the cues. Because cues 2 and 3 are stronger than cues 4 and 5, the alternative option is much stronger in Trial 1 than in Trial 2, leading to a systematically positive difference in choice probabilities.
metric_source:
def metric(data: pd.DataFrame) -> float:
    def get_choice_10000(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        t1 = (1, 0, 0, 0, 0)
        if a == t1 and b == (0, 1, 1, 0, 0):
            return 1 if row['response'] == 0 else 0, 't1'
        elif b == t1 and a == (0, 1, 1, 0, 0):
            return 1 if row['response'] == 1 else 0, 't1'
        elif a == t1 and b == (0, 0, 0, 1, 1):
            return 1 if row['response'] == 0 else 0, 't2'
        elif b == t1 and a == (0, 0, 0, 1, 1):
            return 1 if row['response'] == 1 else 0, 't2'
        else:
            return None, None
    
    res = data.apply(get_choice_10000, axis=1)
    t1_choices = [x[0] for x in res if x[1] == 't1']
    t2_choices = [x[0] for x in res if x[1] == 't2']
    
    p_t1 = sum(t1_choices) / len(t1_choices) if t1_choices else 0.5
    p_t2 = sum(t2_choices) / len(t2_choices) if t2_choices else 0.5
    
    return p_t2 - p_t1
outcome: self_sim=0.0067 (var=0.0208) adversary_sim=0.0450 (var=0.0487) welch_t=-0.727 p=0.4712 (N=25, alpha=0.01) -> reject

[1] rationale: This metric aggregates three distinct trial contrasts (T5-T4, T2-T1, T7-T6) designed to exploit the same structural invariance in the Advocated model. In each pair, the Advocated model uses the exact same discriminating cue and faces the exact same tally difference for the remaining cues, predicting a difference of exactly 0 in choice probabilities. By contrast, the Competing model evaluates specific cue validities, which systematically shift between the pairs (e.g., facing weaker cues in T5 vs T4), thus predicting a positive difference. Averaging across these three pairs stabilizes the metric, reducing between-subject variance and sharpening the statistical dissociation between the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    def get_trial_info(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        a1 = (1, 0, 0, 0, 0)
        a6 = (0, 1, 0, 0, 0)
        
        if (a == a1 and b == (0, 1, 1, 0, 0)) or (b == a1 and a == (0, 1, 1, 0, 0)):
            return ('t1', 1 if (a == a1 and resp == 0) or (b == a1 and resp == 1) else 0)
        if (a == a1 and b == (0, 0, 0, 1, 1)) or (b == a1 and a == (0, 0, 0, 1, 1)):
            return ('t2', 1 if (a == a1 and resp == 0) or (b == a1 and resp == 1) else 0)
        if (a == a1 and b == (0, 1, 0, 0, 0)) or (b == a1 and a == (0, 1, 0, 0, 0)):
            return ('t4', 1 if (a == a1 and resp == 0) or (b == a1 and resp == 1) else 0)
        if (a == a1 and b == (0, 0, 0, 0, 1)) or (b == a1 and a == (0, 0, 0, 0, 1)):
            return ('t5', 1 if (a == a1 and resp == 0) or (b == a1 and resp == 1) else 0)
        if (a == a6 and b == (0, 0, 1, 1, 0)) or (b == a6 and a == (0, 0, 1, 1, 0)):
            return ('t6', 1 if (a == a6 and resp == 0) or (b == a6 and resp == 1) else 0)
        if (a == a6 and b == (0, 0, 0, 1, 1)) or (b == a6 and a == (0, 0, 0, 1, 1)):
            return ('t7', 1 if (a == a6 and resp == 0) or (b == a6 and resp == 1) else 0)
            
        return (None, None)

    res = data.apply(get_trial_info, axis=1)
    import pandas as pd
    df = pd.DataFrame(res.tolist(), columns=['trial', 'chose_a']).dropna()
    if df.empty:
        return 0.0
    means = df.groupby('trial')['chose_a'].mean()
    
    val = 0.0
    if 't5' in means and 't4' in means:
        val += means['t5'] - means['t4']
    if 't2' in means and 't1' in means:
        val += means['t2'] - means['t1']
    if 't7' in means and 't6' in means:
        val += means['t7'] - means['t6']
        
    return float(val) / 3.0
outcome: self_sim=0.0056 (var=0.0073) adversary_sim=0.0117 (var=0.0150) welch_t=-0.205 p=0.8389 (N=25, alpha=0.01) -> reject

[2] rationale: Previous attempts failed because they took the signed difference between trials. The Competing model (WADD-PWR) uses a power-function weighting scheme where the parameter gamma can be either positive or negative. A positive gamma makes higher-validity cues stronger, while a negative gamma makes lower-validity cues stronger. Thus, the signed difference in choice probabilities between T1 and T2 flips sign depending on the subject's gamma, causing the average difference across subjects to cancel out to near zero. By instead calculating the sum of squared differences between these matched trial pairs, we capture the absolute magnitude of the divergence. For the Advocated model, the true difference is exactly 0, so the squared difference only reflects binomial noise. For the Competing model, the squared difference will be systematically inflated by the varying gamma parameter, robustly discriminating the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def get_a_choice(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        a_1 = (1, 0, 0, 0, 0)
        a_6 = (0, 1, 0, 0, 0)
        
        if a == a_1 and b == (0, 1, 1, 0, 0): return 'T1', (1 if resp == 0 else 0)
        if b == a_1 and a == (0, 1, 1, 0, 0): return 'T1', (1 if resp == 1 else 0)
        
        if a == a_1 and b == (0, 0, 0, 1, 1): return 'T2', (1 if resp == 0 else 0)
        if b == a_1 and a == (0, 0, 0, 1, 1): return 'T2', (1 if resp == 1 else 0)
        
        if a == a_1 and b == (0, 1, 0, 0, 0): return 'T4', (1 if resp == 0 else 0)
        if b == a_1 and a == (0, 1, 0, 0, 0): return 'T4', (1 if resp == 1 else 0)
        
        if a == a_1 and b == (0, 0, 0, 0, 1): return 'T5', (1 if resp == 0 else 0)
        if b == a_1 and a == (0, 0, 0, 0, 1): return 'T5', (1 if resp == 1 else 0)
        
        if a == a_6 and b == (0, 0, 1, 1, 0): return 'T6', (1 if resp == 0 else 0)
        if b == a_6 and a == (0, 0, 1, 1, 0): return 'T6', (1 if resp == 1 else 0)
        
        if a == a_6 and b == (0, 0, 0, 1, 1): return 'T7', (1 if resp == 0 else 0)
        if b == a_6 and a == (0, 0, 0, 1, 1): return 'T7', (1 if resp == 1 else 0)
        
        return None, None

    mapped = data.apply(get_a_choice, axis=1)
    df = pd.DataFrame(mapped.tolist(), columns=['trial', 'chose_a']).dropna()
    
    if df.empty:
        return 0.0
        
    means = df.groupby('trial')['chose_a'].mean()
    
    val = 0.0
    if 'T1' in means and 'T2' in means:
        val += (means['T1'] - means['T2']) ** 2
    if 'T4' in means and 'T5' in means:
        val += (means['T4'] - means['T5']) ** 2
    if 'T6' in means and 'T7' in means:
        val += (means['T6'] - means['T7']) ** 2
        
    return float(val)
outcome: self_sim=0.0006 (var=0.0034) adversary_sim=0.0023 (var=0.0166) welch_t=-0.060 p=0.9525 (N=25, alpha=0.01) -> reject

[3] rationale: Across trials 1, 2, 3, 6, 7, and 8, the Advocated model (TTB-CV) always selects exactly one discriminating cue and then faces an identical tally difference (-2) for the remaining cues. Because TTB-CV ignores the specific validities of both the discriminating and tallied cues, it predicts the exact same underlying choice probability for all 6 of these trials. Therefore, the variance of the observed choice proportions across these 6 trials will simply reflect binomial sampling noise. By contrast, the Competing model (WADD-PWR) weights cues non-linearly by their validities, producing 5 uniquely different evidence differences across these 6 trials. This systematic spread in true choice probabilities substantially inflates the variance of the observed choice proportions. By computing the variance of the choice proportions across these 6 trials, we create a robust, non-negative metric that avoids cancellation across subjects with different parameters.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def get_a_choice(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Identify the 6 trials where TTB-CV predicts the exact same choice probability
        trials = {
            ((1,0,0,0,0), (0,1,1,0,0)): 'T1',
            ((1,0,0,0,0), (0,0,0,1,1)): 'T2',
            ((1,0,0,0,0), (0,1,0,1,0)): 'T3',
            ((0,1,0,0,0), (0,0,1,1,0)): 'T6',
            ((0,1,0,0,0), (0,0,0,1,1)): 'T7',
            ((1,1,0,0,0), (1,0,1,1,0)): 'T8'
        }
        
        if (a, b) in trials:
            return trials[(a, b)], 1 if resp == 0 else 0
        elif (b, a) in trials:
            return trials[(b, a)], 1 if resp == 1 else 0
            
        return None, None

    mapped = data.apply(get_a_choice, axis=1)
    df = pd.DataFrame(mapped.tolist(), columns=['trial', 'chose_a']).dropna()
    
    if df.empty:
        return 0.0
        
    means = df.groupby('trial')['chose_a'].mean()
    if len(means) < 2:
        return 0.0
        
    return float(means.var())
outcome: self_sim=0.0002 (var=0.0001) adversary_sim=0.0007 (var=0.0003) welch_t=-0.130 p=0.8969 (N=25, alpha=0.01) -> reject

[4] rationale: This metric exploits a structural divergence between the Advocated model (TTB-CV) and the Competing model (WADD-PWR) by using a cross-multiplication of choice probability differences. For TTB-CV, trials T1 and T2 have identical discriminating cues and tally differences, so P(A|T1) == P(A|T2). The same holds for T3 vs T4, and T5 vs T6. Thus, the differences S1 = (T2-T1) + (T4-T3) and S2 = (T6-T5) are both exactly zero in expectation. Because S1 and S2 are computed from disjoint sets of trials, their empirical estimation errors are independent, guaranteeing their product is exactly 0 in expectation, completely eliminating positive bias from binomial noise. For WADD-PWR, these differences are non-zero and always share the same sign (determined by the gamma parameter's weighting of higher vs. lower validity cues). Multiplying these two independent components yields a metric that is strictly positive for WADD-PWR (as the signs cancel out) regardless of the subject's specific parameters, cleanly discriminating the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def get_trial(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        t1 = ((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))
        t2 = ((1, 0, 0, 0, 0), (0, 0, 0, 1, 1))
        t3 = ((1, 1, 0, 0, 0), (1, 0, 1, 1, 0))
        t4 = ((1, 1, 0, 0, 0), (1, 0, 0, 1, 1))
        t5 = ((1, 0, 0, 0, 0), (0, 1, 0, 0, 0))
        t6 = ((1, 0, 0, 0, 0), (0, 0, 0, 0, 1))
        
        if (a, b) == t1: return 'T1', 1 if row['response'] == 0 else 0
        if (b, a) == t1: return 'T1', 1 if row['response'] == 1 else 0
        if (a, b) == t2: return 'T2', 1 if row['response'] == 0 else 0
        if (b, a) == t2: return 'T2', 1 if row['response'] == 1 else 0
        if (a, b) == t3: return 'T3', 1 if row['response'] == 0 else 0
        if (b, a) == t3: return 'T3', 1 if row['response'] == 1 else 0
        if (a, b) == t4: return 'T4', 1 if row['response'] == 0 else 0
        if (b, a) == t4: return 'T4', 1 if row['response'] == 1 else 0
        if (a, b) == t5: return 'T5', 1 if row['response'] == 0 else 0
        if (b, a) == t5: return 'T5', 1 if row['response'] == 1 else 0
        if (a, b) == t6: return 'T6', 1 if row['response'] == 0 else 0
        if (b, a) == t6: return 'T6', 1 if row['response'] == 1 else 0
        
        return None, None

    mapped = data.apply(get_trial, axis=1)
    df = pd.DataFrame(mapped.tolist(), columns=['trial', 'chose_a'], index=data.index)
    df['subject_id'] = data['subject_id']
    df = df.dropna()
    
    if df.empty:
        return 0.0
        
    means = df.groupby(['subject_id', 'trial'])['chose_a'].mean().unstack(fill_value=0.5)
    
    for t in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']:
        if t not in means.columns:
            means[t] = 0.5
            
    S1 = (means['T2'] - means['T1']) + (means['T4'] - means['T3'])
    S2 = means['T6'] - means['T5']
    
    val = (S1 * S2).mean()
    
    return float(val)
outcome: self_sim=0.0019 (var=0.0004) adversary_sim=0.0052 (var=0.0023) welch_t=-0.327 p=0.7455 (N=25, alpha=0.01) -> reject

[5] rationale: To overcome the cancellation effect caused by symmetric parameters (like gamma in WADD-PWR, which can be positive or negative) and to minimize binomial noise, this metric groups trials into two macro-categories. Group 'H' contains trials where the alternative option wins higher-validity cues (T1, T3, T5), and Group 'L' contains trials where it wins lower-validity cues (T2, T4, T6). 

For the Advocated model (TTB-CV), the tally mechanism ignores specific validities, meaning the expected choice probability is identical between Group H and Group L. Thus, the absolute difference in choice proportions per subject is driven purely by binomial noise. By averaging 48 trials per group per subject, this noise is heavily suppressed, yielding a very small and stable metric value.

For the Competing model (WADD-PWR), the non-linear weighting of cues means the true choice probabilities systematically diverge between Group H and Group L. By computing the absolute difference per subject *before* averaging across subjects, we prevent subjects with positive and negative gamma values from canceling each other out. This produces a robust, strictly positive metric for the Competing model, cleanly and reliably discriminating the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    def get_group(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        chose_a = 1 if resp == 0 else 0
        
        # H group: Alternative option wins higher-validity cues
        if (a == (1,0,0,0,0) and b == (0,1,1,0,0)): return 'H', chose_a
        if (b == (1,0,0,0,0) and a == (0,1,1,0,0)): return 'H', 1 - chose_a
        if (a == (1,1,0,0,0) and b == (1,0,1,1,0)): return 'H', chose_a
        if (b == (1,1,0,0,0) and a == (1,0,1,1,0)): return 'H', 1 - chose_a
        if (a == (1,0,0,0,0) and b == (0,1,0,0,0)): return 'H', chose_a
        if (b == (1,0,0,0,0) and a == (0,1,0,0,0)): return 'H', 1 - chose_a
        
        # L group: Alternative option wins lower-validity cues
        if (a == (1,0,0,0,0) and b == (0,0,0,1,1)): return 'L', chose_a
        if (b == (1,0,0,0,0) and a == (0,0,0,1,1)): return 'L', 1 - chose_a
        if (a == (1,1,0,0,0) and b == (1,0,0,1,1)): return 'L', chose_a
        if (b == (1,1,0,0,0) and a == (1,0,0,1,1)): return 'L', 1 - chose_a
        if (a == (1,0,0,0,0) and b == (0,0,0,0,1)): return 'L', chose_a
        if (b == (1,0,0,0,0) and a == (0,0,0,0,1)): return 'L', 1 - chose_a
        
        return None, None

    mapped = data.apply(get_group, axis=1)
    df = pd.DataFrame(mapped.tolist(), columns=['group', 'chose_a'])
    df['subject_id'] = data['subject_id'].values
    df = df.dropna()
    
    if df.empty:
        return 0.0
        
    subj_means = df.groupby(['subject_id', 'group'])['chose_a'].mean().unstack()
    if 'H' not in subj_means.columns: subj_means['H'] = 0.5
    if 'L' not in subj_means.columns: subj_means['L'] = 0.5
    subj_means = subj_means.fillna(0.5)
    
    abs_diff = (subj_means['H'] - subj_means['L']).abs()
    return float(abs_diff.mean())
outcome: self_sim=0.0562 (var=0.0020) adversary_sim=0.0833 (var=0.0054) welch_t=-1.570 p=0.1243 (N=25, alpha=0.01) -> reject

[6] rationale: For the Advocated model (TTB-CV), trials T1, T2, T3, and T4 all share an identical underlying structure: the model selects exactly one discriminating cue in favor of option A, and then tallies exactly 2 remaining cues in favor of option B. Because the tally mechanism ignores specific validities, TTB-CV predicts the exact same choice probability for all four of these trials. Thus, the variance of the observed choice proportions across these 4 trials per subject will be strictly limited to binomial sampling noise. By contrast, the Competing model (WADD-PWR) weights cues non-linearly by their validities, producing 4 systematically distinct evidence differences. This systematic spread in true choice probabilities substantially inflates the variance of the observed choice proportions, reliably discriminating the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def get_trial(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        t1 = ((1, 0, 0, 0, 0), (0, 1, 1, 0, 0))
        t2 = ((1, 0, 0, 0, 0), (0, 0, 0, 1, 1))
        t3 = ((1, 1, 0, 0, 0), (1, 0, 1, 1, 0))
        t4 = ((1, 1, 0, 0, 0), (1, 0, 0, 1, 1))
        
        if (a, b) == t1: return 'T1', 1 if row['response'] == 0 else 0
        if (b, a) == t1: return 'T1', 1 if row['response'] == 1 else 0
        if (a, b) == t2: return 'T2', 1 if row['response'] == 0 else 0
        if (b, a) == t2: return 'T2', 1 if row['response'] == 1 else 0
        if (a, b) == t3: return 'T3', 1 if row['response'] == 0 else 0
        if (b, a) == t3: return 'T3', 1 if row['response'] == 1 else 0
        if (a, b) == t4: return 'T4', 1 if row['response'] == 0 else 0
        if (b, a) == t4: return 'T4', 1 if row['response'] == 1 else 0
        
        return None, None

    mapped = data.apply(get_trial, axis=1)
    df = pd.DataFrame(mapped.tolist(), columns=['trial', 'chose_a'])
    df['subject_id'] = data['subject_id'].values
    df = df.dropna()
    
    if df.empty:
        return 0.0
        
    means = df.groupby(['subject_id', 'trial'])['chose_a'].mean().unstack(fill_value=0.5)
    
    for t in ['T1', 'T2', 'T3', 'T4']:
        if t not in means.columns:
            means[t] = 0.5
            
    var_per_subj = means[['T1', 'T2', 'T3', 'T4']].var(axis=1)
    return float(var_per_subj.mean())
outcome: self_sim=0.0072 (var=0.0000) adversary_sim=0.0134 (var=0.0003) welch_t=-1.712 p=0.0968 (N=25, alpha=0.01) -> reject

[7] rationale: By computing the absolute difference in choice probability for each matched pair of trials (T1 vs T2, T3 vs T4, T5 vs T6) at the individual subject level, we avoid cancellation effects from opposite parameter values (like positive vs negative gamma in WADD-PWR). For the Advocated model (TTB-CV), the true choice probabilities within each pair are identical, meaning the absolute differences are purely driven by binomial noise. For the Competing model (WADD-PWR), the nonlinear weighting of cue validities causes these paired trials to systematically diverge. Summing these three absolute differences maximizes the signal while keeping between-subject variance low, robustly discriminating the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    def get_trial(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        chose_a = 1 if resp == 0 else 0
        
        t1 = ((1,0,0,0,0), (0,1,1,0,0))
        t2 = ((1,0,0,0,0), (0,0,0,1,1))
        t3 = ((1,1,0,0,0), (1,0,1,1,0))
        t4 = ((1,1,0,0,0), (1,0,0,1,1))
        t5 = ((1,0,0,0,0), (0,1,0,0,0))
        t6 = ((1,0,0,0,0), (0,0,0,0,1))
        
        if (a, b) == t1: return 'T1', chose_a
        if (b, a) == t1: return 'T1', 1 - chose_a
        if (a, b) == t2: return 'T2', chose_a
        if (b, a) == t2: return 'T2', 1 - chose_a
        if (a, b) == t3: return 'T3', chose_a
        if (b, a) == t3: return 'T3', 1 - chose_a
        if (a, b) == t4: return 'T4', chose_a
        if (b, a) == t4: return 'T4', 1 - chose_a
        if (a, b) == t5: return 'T5', chose_a
        if (b, a) == t5: return 'T5', 1 - chose_a
        if (a, b) == t6: return 'T6', chose_a
        if (b, a) == t6: return 'T6', 1 - chose_a
        
        return None, None

    mapped = data.apply(get_trial, axis=1)
    df = pd.DataFrame(mapped.tolist(), columns=['trial', 'chose_a'])
    df['subject_id'] = data['subject_id'].values
    df = df.dropna()
    
    if df.empty: return 0.0
    
    means = df.groupby(['subject_id', 'trial'])['chose_a'].mean().unstack(fill_value=0.5)
    
    for t in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']:
        if t not in means.columns:
            means[t] = 0.5

    d1 = (means['T1'] - means['T2']).abs()
    d2 = (means['T3'] - means['T4']).abs()
    d3 = (means['T5'] - means['T6']).abs()
    
    return float((d1 + d2 + d3).mean())
outcome: self_sim=0.2737 (var=0.0219) adversary_sim=0.3550 (var=0.0535) welch_t=-1.480 p=0.1466 (N=25, alpha=0.01) -> reject

[8] rationale: This metric utilizes a cross-multiplication of two independent contrasts (C_X and C_Y) to perfectly isolate the structural divergence between the theories while completely eliminating positive bias from binomial noise. For the Advocated model (TTB-CV), all 8 of these trials result in the model selecting exactly one discriminating cue and tallying a difference of exactly 3 for the remaining cues. Since it ignores specific validities, TTB-CV predicts the exact same choice probability across all 8 trials. Consequently, the expected values of both C_X and C_Y are exactly 0. Because C_X and C_Y are computed from completely disjoint sets of trials, their estimation errors are independent, guaranteeing their product is exactly 0 in expectation. By contrast, the Competing model (WADD-PWR) weights cues non-linearly, causing C_X and C_Y to systematically diverge from 0 in the same direction (both positive if gamma > 0, both negative if gamma < 0). Multiplying them yields a strictly positive signal for the Competing model, robustly discriminating the two theories without being inflated by sampling noise.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def get_trial(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        chose_a = 1 if resp == 0 else 0
        
        t1 = ((1,1,1,1,1), (1,0,0,0,0))
        t2 = ((1,1,1,1,1), (0,1,0,0,0))
        t4 = ((1,1,1,1,1), (0,0,0,1,0))
        t5 = ((1,1,1,1,1), (0,0,0,0,1))
        
        t6 = ((0,1,1,1,1), (0,0,0,0,0))
        t7 = ((1,0,1,1,1), (0,0,0,0,0))
        t9 = ((1,1,1,0,1), (0,0,0,0,0))
        t10 = ((1,1,1,1,0), (0,0,0,0,0))
        
        if (a, b) == t1: return 'T1', chose_a
        if (b, a) == t1: return 'T1', 1 - chose_a
        if (a, b) == t2: return 'T2', chose_a
        if (b, a) == t2: return 'T2', 1 - chose_a
        if (a, b) == t4: return 'T4', chose_a
        if (b, a) == t4: return 'T4', 1 - chose_a
        if (a, b) == t5: return 'T5', chose_a
        if (b, a) == t5: return 'T5', 1 - chose_a
        
        if (a, b) == t6: return 'T6', chose_a
        if (b, a) == t6: return 'T6', 1 - chose_a
        if (a, b) == t7: return 'T7', chose_a
        if (b, a) == t7: return 'T7', 1 - chose_a
        if (a, b) == t9: return 'T9', chose_a
        if (b, a) == t9: return 'T9', 1 - chose_a
        if (a, b) == t10: return 'T10', chose_a
        if (b, a) == t10: return 'T10', 1 - chose_a
        
        return None, None

    mapped = data.apply(get_trial, axis=1)
    df = pd.DataFrame(mapped.tolist(), columns=['trial', 'chose_a'])
    df['subject_id'] = data['subject_id'].values
    df = df.dropna()
    
    if df.empty:
        return 0.0
        
    means = df.groupby(['subject_id', 'trial'])['chose_a'].mean().unstack(fill_value=0.5)
    
    for t in ['T1', 'T2', 'T4', 'T5', 'T6', 'T7', 'T9', 'T10']:
        if t not in means.columns:
            means[t] = 0.5
            
    C_X = 2 * means['T5'] + means['T4'] - means['T2'] - 2 * means['T1']
    C_Y = 2 * means['T10'] + means['T9'] - means['T7'] - 2 * means['T6']
    
    return float((C_X * C_Y).mean())
outcome: self_sim=-0.0244 (var=0.0162) adversary_sim=0.0025 (var=0.0182) welch_t=-0.725 p=0.4719 (N=25, alpha=0.01) -> reject

[9] rationale: This metric utilizes a cross-covariance approach to perfectly isolate the structural divergence between the theories while completely eliminating positive bias from binomial noise. For the Advocated model (TTB-CV), all 10 trials result in the model selecting exactly one discriminating cue and tallying a difference of exactly 3 for the remaining cues. Thus, TTB-CV predicts the exact same underlying choice probability for all 10 trials. By computing the sample covariance between the X trials (T1-T5) and Y trials (T6-T10) per subject, we get an unbiased estimator of the variance of the true choice probabilities. Because the true probabilities are constant under TTB-CV, the expected covariance is exactly 0. For the Competing model (WADD-PWR), the non-linear weighting of cues causes the true choice probabilities to systematically vary across the 5 pairs (e.g., X1 and Y1 share the same probability, which is different from X5 and Y5). This shared variance between X and Y results in a strictly positive covariance, robustly discriminating the two theories without being inflated by sampling noise.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    def get_trial(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        chose_a = 1 if resp == 0 else 0
        
        t_dict = {
            ((1,1,1,1,1), (1,0,0,0,0)): 'X1',
            ((1,1,1,1,1), (0,1,0,0,0)): 'X2',
            ((1,1,1,1,1), (0,0,1,0,0)): 'X3',
            ((1,1,1,1,1), (0,0,0,1,0)): 'X4',
            ((1,1,1,1,1), (0,0,0,0,1)): 'X5',
            ((0,1,1,1,1), (0,0,0,0,0)): 'Y1',
            ((1,0,1,1,1), (0,0,0,0,0)): 'Y2',
            ((1,1,0,1,1), (0,0,0,0,0)): 'Y3',
            ((1,1,1,0,1), (0,0,0,0,0)): 'Y4',
            ((1,1,1,1,0), (0,0,0,0,0)): 'Y5',
        }
        
        if (a, b) in t_dict:
            return t_dict[(a, b)], chose_a
        elif (b, a) in t_dict:
            return t_dict[(b, a)], 1 - chose_a
            
        return None, None

    mapped = data.apply(get_trial, axis=1)
    df = pd.DataFrame(mapped.tolist(), columns=['trial', 'chose_a'])
    df['subject_id'] = data['subject_id'].values
    df = df.dropna()
    
    if df.empty:
        return 0.0
        
    means = df.groupby(['subject_id', 'trial'])['chose_a'].mean().unstack(fill_value=0.5)
    
    for t in ['X1', 'X2', 'X3', 'X4', 'X5', 'Y1', 'Y2', 'Y3', 'Y4', 'Y5']:
        if t not in means.columns:
            means[t] = 0.5
            
    covs = []
    for subj in means.index:
        x = means.loc[subj, ['X1', 'X2', 'X3', 'X4', 'X5']].values.astype(float)
        y = means.loc[subj, ['Y1', 'Y2', 'Y3', 'Y4', 'Y5']].values.astype(float)
        cov = np.cov(x, y)[0, 1]
        covs.append(cov)
        
    return float(np.mean(covs))
outcome: self_sim=-0.0009 (var=0.0001) adversary_sim=-0.0007 (var=0.0001) welch_t=-0.046 p=0.9633 (N=25, alpha=0.01) -> reject

[10] rationale: The Advocated model uses Take-The-Best with Compensatory Verification. Across all 10 trials, the model always selects exactly one discriminating cue in favor of the dominant option, and then tallies exactly the same difference for the remaining cues. Since it ignores specific validities, it predicts the exact same choice probability across all trials. The Competing model (WADD-PWR) integrates all cues using a power function of their validities. The 10 trials systematically vary which cue is tied (from the most valid cue 0 to the least valid cue 4). Thus, WADD-PWR predicts a monotonic trend in choice probabilities across these 5 levels of tied cues.

To robustly discriminate the theories, we compute a linear contrast of the choice proportions across the 5 levels (C = -2*Y0 - 1*Y1 + 1*Y3 + 2*Y4). Because the gamma parameter can be positive or negative, C can be positive or negative. We compute C^2 to capture the magnitude of the trend. However, C^2 is positively biased by binomial sampling noise. We subtract the exact unbiased sample variance estimate of the contrast to completely eliminate this bias. The resulting metric is exactly 0 in expectation for the Advocated model, and strictly positive for the Competing model, providing a highly powerful and unbiased statistical test.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    def get_level_and_choice(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Find the single tied cue index
        tied_indices = [i for i in range(len(a)) if a[i] == b[i]]
        if len(tied_indices) != 1:
            return pd.Series({'level': -1, 'chose_A': np.nan})
        level = tied_indices[0]
        
        sum_a = sum(a)
        sum_b = sum(b)
        if sum_a > sum_b:
            chose_A = 1 if resp == 0 else 0
        elif sum_b > sum_a:
            chose_A = 1 if resp == 1 else 0
        else:
            return pd.Series({'level': -1, 'chose_A': np.nan})
            
        return pd.Series({'level': level, 'chose_A': chose_A})

    mapped = data.apply(get_level_and_choice, axis=1)
    df = pd.DataFrame({
        'subject_id': data['subject_id'].values,
        'level': mapped['level'].values,
        'chose_A': mapped['chose_A'].values
    }).dropna()
    df = df[df['level'] != -1]
    
    def subject_M(sub_df):
        stats = sub_df.groupby('level')['chose_A'].agg(['mean', 'count'])
        
        Y = {}
        n = {}
        for k in range(5):
            if k in stats.index and stats.loc[k, 'count'] > 1:
                Y[k] = stats.loc[k, 'mean']
                n[k] = stats.loc[k, 'count']
            else:
                return 0.0
                
        # Linear contrast C measuring the trend across the 5 levels
        C = -2 * Y[0] - 1 * Y[1] + 0 * Y[2] + 1 * Y[3] + 2 * Y[4]
        
        # Unbiased estimate of the variance of the mean Y[k]
        def var_est(k):
            return Y[k] * (1.0 - Y[k]) / (n[k] - 1.0)
            
        # Variance of the linear contrast
        Var_C = 4 * var_est(0) + 1 * var_est(1) + 0 * var_est(2) + 1 * var_est(3) + 4 * var_est(4)
        
        # Unbiased estimator of the squared expected contrast
        M = C**2 - Var_C
        return float(M)

    subj_M = df.groupby('subject_id').apply(subject_M)
    if subj_M.empty:
        return 0.0
    return float(subj_M.mean())
outcome: self_sim=-0.0019 (var=0.0155) adversary_sim=-0.0028 (var=0.0121) welch_t=+0.026 p=0.979 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n\n    def get_trial(row):\n        a = tuple(row['option_a_ratings'])\n        b = tuple(row['option_b_ratings'])\n        resp = row['response']\n        # In all 10 trials, one option has 4 or 5 cues, the other has 0 or 1.\n        # We identify the dominant option (the one with more 1s).\n        if sum(a) > sum(b):\n            return str(a) + str(b), 1 if resp == 0 else 0\n        else:\n            return str(b) + str(a), 1 if resp == 1 else 0\n\n    mapped = data.apply(get_trial, axis=1)\n    df = pd.DataFrame(mapped.tolist(), columns=['trial', 'chose_dom'])\n    df['subject_id'] = data['subject_id'].values\n    \n    counts = df.groupby(['subject_id', 'trial'])['chose_dom'].agg(['sum', 'count'])\n    \n    def calc_M(sub_df):\n        valid = sub_df[sub_df['count'] > 1]\n        if len(valid) < 2:\n            return np.nan\n        \n        X = valid['sum'].values.astype(float)\n        R = valid['count'].values.astype(float)\n        Y = X / R\n        \n        # S2_Y is the sample variance of the observed choice proportions across the 10 trials\n        S2_Y = np.var(Y, ddof=1)\n        \n        # W_t is the exact unbiased estimator of the binomial variance for trial t: p_t(1-p_t)/R_t\n        W = X * (R - X) / (R**2 * (R - 1.0))\n        mean_W = np.mean(W)\n        \n        # M is the unbiased estimator of the variance of the true underlying choice probabilities\n        return S2_Y - mean_W\n\n    M_per_subj = counts.groupby('subject_id').apply(calc_M).dropna()\n    if M_per_subj.empty:\n        return 0.0\n        \n    return float(M_per_subj.mean())",
  "rationale": "Across all 10 trials, the Advocated model (Take-The-Best with Compensatory Verification) always selects exactly one discriminating cue in favor of the dominant option, and then tallies exactly the same difference (+3) for the remaining cues. Because it ignores the specific validities of both the discriminating and tallied cues, it predicts the exact same underlying choice probability for all 10 trials. Therefore, the true variance of the choice probabilities across the 10 trials is exactly 0. \n\nBy contrast, the Competing model (WADD-PWR) weights cues non-linearly by their validities. Since the trials systematically vary which cues are won by the dominant option, the true choice probabilities systematically vary across the 10 trials, meaning the true variance of the probabilities is strictly positive.\n\nTo measure this variance without being inflated by binomial sampling noise, we compute `M`, the exact unbiased estimator of the variance of the true probabilities. `M` subtracts the unbiased estimate of the expected binomial noise (`mean_W`) from the sample variance of the observed proportions (`S2_Y`). For the Advocated model, the expected value of `M` is exactly 0 regardless of the subject's parameters. For the Competing model, the expected value of `M` is strictly positive. Averaging `M` across subjects yields a highly stable, unbiased test statistic that cleanly discriminates the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 14821,
  "candidates_token_count": 887,
  "total_token_count": 24436
}
```
