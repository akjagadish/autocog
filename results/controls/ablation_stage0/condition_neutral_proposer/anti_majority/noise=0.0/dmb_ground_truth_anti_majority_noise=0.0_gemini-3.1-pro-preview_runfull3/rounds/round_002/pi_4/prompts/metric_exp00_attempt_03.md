# metric_exp00_attempt_03

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
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.55, 0.5]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 3: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 7: A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 8: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 9: A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 10: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]

**Rationale:** To maximally distinguish the TTB/Tallying mixture model from WADD, we must exploit WADD's sensitivity to intermediate validity values, which the mixture model ignores. The mixture model is constrained to be a linear combination of exactly two strategies: TTB (which only considers the single highest-validity discriminating cue) and Tallying (which weighs all winning cues equally). WADD, however, integrates all cues according to their subjectively weighted validities. By using a 6-feature environment with linearly spaced validities, we can construct pairs where the Tallying difference is held constant but the WADD score difference varies dramatically (because the specific validities of the matching features change). For example, pitting cue 2 against cues 3, 4, 5 vs. pitting cue 3 against cues 4, 5, 6. The mixture model will predict identical choice probabilities across these pairs (since the TTB and Tallying signals are identical structurally), whereas WADD will predict systematically different probabilities due to the varying absolute weights of the cues involved.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** People possess an adaptive toolbox of decision heuristics, primarily Take The Best (TTB) and Tallying, and select among them probabilistically. TTB is a non-compensatory, one-reason decision rule that relies solely on the most valid discriminating cue, while Tallying is an equal-weight compensatory rule that simply counts the number of cues favoring each option. By probabilistically mixing these distinct strategies on a trial-by-trial basis (or reflecting a population-level mixture of strategy users), decision-makers exhibit intermediate compensatory behavior. This avoids the need for a unified, complex non-linear cue weighting mechanism like Weighted Additive (WADD) models. Decision stochasticity is accounted for by the probabilistic strategy selection, strategy-specific softmax response noise, and an overarching lapse rate. The mixture is heavily biased toward TTB, reflecting the empirical dominance of one-reason decision making in these environments.

**Parameters:**
- p_ttb: [0.5, 1.0]
- beta_ttb: [0.1, 20.0]
- beta_tally: [0.1, 20.0]
- epsilon: [0.0, 0.2]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # --- Take The Best (TTB) --- 
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb_core = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb_core = e_ttb / np.sum(e_ttb)
        
    # --- Tallying ---
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally_core = e_tally / np.sum(e_tally)
    
    # --- Strategy Mixture ---
    p_ttb_weight = float(parameters["p_ttb"])
    p_mixed = p_ttb_weight * p_ttb_core + (1.0 - p_ttb_weight) * p_tally_core
    
    # --- Lapse Noise ---
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
    
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
**Description:** People use a Weighted Additive (WADD) strategy to make decisions, where each feature is weighted by its subjective importance (a non-linear transformation of its validity). The total score for an option is the sum of the weighted features it possesses. This compensatory mechanism allows multiple lower-validity cues to sometimes outweigh a single high-validity cue, naturally interpolating between Take The Best (when validity differences are heavily magnified) and Tallying (when validities are ignored). Response noise and lapses account for stochasticity in choice.

**Parameters:**
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 30.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Transform validities into subjective weights using a power function.
    # gamma = 0 yields equal weights (Tallying); gamma -> inf yields lexicographic weights (Take The Best).
    weights = val ** gamma
    
    # Compute weighted additive scores for each option
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
[0] rationale: The TTB/Tallying mixture model relies entirely on two signals: the highest validity cue that discriminates (TTB) and the unweighted sum of winning cues (Tallying). Trial 6 and Trial 9 share identical TTB and Tallying profiles, as do Trial 8 and Trial 10. Thus, the mixture model strictly predicts a difference of 0 in choice probability for these pairs. WADD, however, integrates all validities; because Trial 6 involves higher absolute validities than Trial 9, and Trial 8 higher than Trial 10, WADD predicts a systematically positive difference. Measuring this difference cleanly isolates WADD's compensatory integration from the mixture's heuristic approach.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Project lists to tuples for filtering
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    # Identify trials 6, 9, 8, 10 by their unique Option A ratings
    t6_A = (0, 1, 1, 0, 0, 0)
    t9_A = (0, 1, 0, 1, 0, 0)
    t8_A = (1, 0, 1, 0, 0, 0)
    t10_A = (1, 0, 0, 1, 0, 0)
    
    # Calculate probability of choosing A (response == 0)
    def get_pA(target_A):
        mask = (a_tuples == target_A)
        if not mask.any():
            return 0.0
        return (data.loc[mask, 'response'] == 0).mean()
    
    pA_t6 = get_pA(t6_A)
    pA_t9 = get_pA(t9_A)
    pA_t8 = get_pA(t8_A)
    pA_t10 = get_pA(t10_A)
    
    # Metric is the sum of differences where Mixture model predicts 0, but WADD predicts > 0
    diff = (pA_t6 - pA_t9) + (pA_t8 - pA_t10)
    
    return float(diff)
outcome: self_sim=-0.0467 (var=0.0549) adversary_sim=0.0800 (var=0.0968) welch_t=-1.626 p=0.1109 (N=25, alpha=0.01) -> reject

[1] rationale: This metric measures the overall probability of choosing Option A in the classic 'compensatory vs. non-compensatory' test trials (Trials 1, 2, and 3). In these trials, Option A possesses only a single cue, but it is the highest validity cue available. Option B possesses all the remaining lower-validity cues. 

Under the TTB/Tallying Mixture model, Take-The-Best (TTB) will strictly select Option A because it holds the most valid discriminating cue. Since the mixture model heavily biases towards TTB (p_ttb in [0.5, 1.0]), it consistently predicts a very high probability of choosing A (mean ~0.75), with low between-subject variance.

In contrast, the Weighted Additive (WADD) model integrates all cues. For most of its parameter space (low to moderate gamma), the sheer number of cues in Option B outweighs the single high-validity cue in Option A, leading WADD to strongly prefer B. While extreme gamma values can make WADD mimic TTB, the population average probability of choosing A under WADD is drastically lower (mean ~0.25). The massive mean difference (~0.50) easily overwhelms any between-subject variance, ensuring a highly significant t-test.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Project option A ratings to tuples for hashable comparison
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    # Identify the three classic "compensatory vs non-compensatory" test trials
    # In these trials, Option A has only a single cue (but it is the most valid one),
    # while Option B has all the remaining less valid cues.
    t1_A = (1, 0, 0, 0, 0, 0)
    t2_A = (0, 1, 0, 0, 0, 0)
    t3_A = (0, 0, 1, 0, 0, 0)
    
    mask = a_tuples.isin([t1_A, t2_A, t3_A])
    
    # If no such trials exist (fallback), return 0.5
    if not mask.any():
        return 0.5
        
    # Calculate the overall empirical probability of choosing Option A
    # response == 0 means Option A was chosen
    p_A = (data.loc[mask, 'response'] == 0).mean()
    
    return float(p_A)
outcome: self_sim=0.7067 (var=0.0323) adversary_sim=0.5987 (var=0.0566) welch_t=+1.811 p=0.07685 (N=25, alpha=0.01) -> reject

[2] rationale: In Trial 7, Option A and B both have exactly two cues, meaning the Tallying strategy sees a perfect tie. Under the TTB/Tallying Mixture model, choices on Trial 7 are driven either by TTB (which prefers A) or random guessing from Tallying, yielding a very high probability of choosing A. In Trial 1, Tallying strongly prefers B (5 cues to 1), so the Mixture model's probability of choosing A drops, but only proportionally to the Tallying weight (which is small, as p_ttb >= 0.5). Thus, the Mixture model predicts a relatively small difference in P(A) between Trial 7 and Trial 1.

In contrast, WADD integrates all cues. For low to moderate values of gamma, the sheer number of cues for B in Trial 1 (summing to 3.3 vs A's 0.95) makes WADD strongly prefer B. However, in Trial 7, A's cues simply outscore B's cues (e.g., 0.75+0.65 > 0.55+0.5), making WADD strongly prefer A. This leads WADD to predict a massive shift in preference between these two trials, far exceeding the modest shift predicted by the Mixture model.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    # Trial 1: A has only the best cue, B has all 5 other cues
    t1_A = (1, 0, 0, 0, 0, 0)
    # Trial 7: A has 2 intermediate cues, B has 2 worst cues (Tallying is tied)
    t7_A = (0, 0, 1, 1, 0, 0)
    
    mask_t1 = (a_tuples == t1_A)
    mask_t7 = (a_tuples == t7_A)
    
    if not mask_t1.any() or not mask_t7.any():
        return 0.0
        
    p_A_t1 = (data.loc[mask_t1, 'response'] == 0).mean()
    p_A_t7 = (data.loc[mask_t7, 'response'] == 0).mean()
    
    return float(p_A_t7 - p_A_t1)
outcome: self_sim=0.1122 (var=0.0318) adversary_sim=0.0044 (var=0.2101) welch_t=+1.096 p=0.2816 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    a_tuples = data['option_a_ratings'].apply(tuple)\n    b_tuples = data['option_b_ratings'].apply(tuple)\n    \n    # Trials where TTB picks A, Tallying picks B (diff -1)\n    # Set 1: A has 2 cues, B has 3 cues\n    t6_A, t6_B = (0, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1)\n    t9_A, t9_B = (0, 1, 0, 1, 0, 0), (0, 0, 1, 0, 1, 1)\n    \n    # Set 2: A has 1 cue, B has 2 cues\n    t4_A, t4_B = (1, 0, 0, 0, 0, 0), (0, 1, 1, 0, 0, 0)\n    t5_A, t5_B = (0, 1, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0)\n    \n    mask_6 = (a_tuples == t6_A) & (b_tuples == t6_B)\n    mask_9 = (a_tuples == t9_A) & (b_tuples == t9_B)\n    mask_4 = (a_tuples == t4_A) & (b_tuples == t4_B)\n    mask_5 = (a_tuples == t5_A) & (b_tuples == t5_B)\n    \n    def get_p(mask):\n        if not mask.any():\n            return 0.5\n        return (data.loc[mask, 'response'] == 0).mean()\n        \n    p6 = get_p(mask_6)\n    p9 = get_p(mask_9)\n    p4 = get_p(mask_4)\n    p5 = get_p(mask_5)\n    \n    return float((p6 + p9) / 2.0 - (p4 + p5) / 2.0)\n",
  "rationale": "This metric exploits a structural equivalence in the TTB/Tallying Mixture model that WADD strictly breaks. In Trials 6, 9, 4, and 5, Option A wins on the highest discriminating cue (so TTB strictly prefers A), while Option B has exactly one more cue than A (so Tallying strictly prefers B by a score difference of 1). Because the Mixture model relies entirely on these two invariant signals, it predicts identical probabilities of choosing A across all four trials. Therefore, the expected value of `(P6 + P9)/2 - (P4 + P5)/2` is exactly 0 under the Mixture model. Conversely, WADD integrates the specific validities of the cues. The negative weight of the extra cues for B is much harsher in Trials 4 and 5 (where B holds very high-validity cues) than in Trials 6 and 9 (where B's extra cues are the lowest-validity ones). Thus, WADD consistently predicts a higher probability of choosing A in Trials 6 and 9 than in Trials 4 and 5, yielding a strictly positive difference."
}
```

## Usage

```json
{
  "prompt_token_count": 5798,
  "candidates_token_count": 816,
  "total_token_count": 18552
}
```
