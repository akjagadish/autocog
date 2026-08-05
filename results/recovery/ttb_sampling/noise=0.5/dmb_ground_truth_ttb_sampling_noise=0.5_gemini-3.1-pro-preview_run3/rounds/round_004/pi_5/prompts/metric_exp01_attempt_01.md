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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.75, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 3: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 4: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 6: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 7: A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  trial 8: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]

**Rationale:** This design quantitatively dissociates Probabilistic Cue Search (PCS) from Strategy Mixture Theory (SMT) by manipulating the sequence of intermediate evidence states while holding both the highest-validity cue and the final compensatory sum constant. By using two features with identical validities (V3 and V4), we create pairs of trials where the Weighted Additive (WADD) difference and the Take-The-Best (TTB) prediction are exactly identical, meaning SMT predicts the exact same choice probability for both trials. However, PCS evaluates cues sequentially and has a probability of stopping at each step. In one trial, the intermediate step strongly favors Option B before recovering to favor Option A; in the matched trial, the intermediate step strongly favors Option A. Because PCS can stop during these intermediate steps, it predicts significantly different choice probabilities between the matched trials, directly exposing its sequential accumulation mechanism.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Cue Search (Sequential Evidence Accumulation): Decision-makers evaluate features sequentially in descending order of validity. However, instead of strictly stopping at the first discriminating cue (as in pure Take-The-Best) or exhaustively accumulating all cues (as in Weighted Additive), they exhibit a probabilistic stopping rule. After evaluating each discriminating cue and updating their internal evidence, they stop searching and make a choice with probability `theta`. If they do not stop, they continue to the next discriminating cue, accumulating its evidence. This naturally produces a graded interpolation between non-compensatory and compensatory decision-making without relying on a discrete mixture of distinct strategies.

**Parameters:**
- theta: [0.0, 1.0]
- beta: [0.1, 20.0]
- gamma: [0.0, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    theta = float(parameters["theta"])
    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    a, b = stim[0], stim[1]
    
    # Subjective weights normalized so the sum of weights is 1.0
    w = validities ** gamma
    if np.sum(w) > 0:
        w = w / np.sum(w)
    
    # Sort all features by validity descending
    order = np.argsort(validities)[::-1]
    
    # Find discriminating cues in order of validity
    discrim_indices = [i for i in order if a[i] != b[i]]
    
    if len(discrim_indices) == 0:
        p_core = np.array([0.5, 0.5])
    else:
        p_A_total = 0.0
        p_B_total = 0.0
        p_continue = 1.0
        
        score_A = 0.0
        score_B = 0.0
        
        for j, i in enumerate(discrim_indices):
            # Accumulate evidence from the current discriminating cue
            score_A += w[i] * a[i]
            score_B += w[i] * b[i]
            
            # Determine stopping probability
            if j < len(discrim_indices) - 1:
                p_stop = theta
            else:
                p_stop = 1.0  # Must stop at the last discriminating cue
                
            # Softmax over accumulated scores so far
            z_A = beta * score_A
            z_B = beta * score_B
            max_z = max(z_A, z_B)
            e_A = np.exp(z_A - max_z)
            e_B = np.exp(z_B - max_z)
            p_A_given_stop = e_A / (e_A + e_B)
            p_B_given_stop = e_B / (e_A + e_B)
            
            # Marginalize over the stopping probability
            p_A_total += p_continue * p_stop * p_A_given_stop
            p_B_total += p_continue * p_stop * p_B_given_stop
            
            # Update the probability of continuing to the next cue
            p_continue *= (1.0 - p_stop)
            
        p_core = np.array([p_A_total, p_B_total])
        
    # Apply uniform lapse
    n_opts = 2
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
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
**Description:** Strategy Mixture Theory: Decision-makers do not uniformly apply a single choice rule. Instead, they possess a repertoire of strategies and flexibly draw from them. On any given trial, a subject acts as a mixture model, choosing to apply a non-compensatory heuristic (Take-The-Best) with probability alpha, and a compensatory rule (Weighted Additive / Tallying) with probability 1 - alpha. The compensatory rule weights features by its subjective validities, naturally subsuming Tallying and WADD. Crucially, the compensatory scores are normalized to the [0, 1] scale to perfectly match the scale of the heuristic's discrete scores, allowing a single temperature parameter to symmetrically control the determinism of both strategies without numerical compromise.

**Parameters:**
- alpha: [0.0, 1.0]
- beta: [0.1, 20.0]
- gamma: [0.0, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Mixture model expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # --- Take-The-Best (TTB) Prediction ---
    order = np.argsort(validities)[::-1]
    a, b = stim[0], stim[1]
    ttb_scores = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            ttb_scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_scores[1] = 1.0
            break
            
    z_ttb = beta * (ttb_scores - ttb_scores.max())
    e_ttb = np.exp(z_ttb)
    p_ttb = e_ttb / e_ttb.sum()
    
    # --- Compensatory (WADD/Tallying) Prediction ---
    # Subjective validities: gamma=0 yields Tallying, gamma=1 yields strict WADD
    subjective_weights = validities ** gamma
    wadd_scores = stim @ subjective_weights
    
    # Normalize WADD scores to [0, 1] scale to match TTB scores
    wadd_scores = wadd_scores / np.sum(subjective_weights)
    
    z_wadd = beta * (wadd_scores - wadd_scores.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # --- Strategy Mixture ---
    p_core = alpha * p_ttb + (1.0 - alpha) * p_wadd
    
    # --- Uniform Lapse ---
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
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
[0] rationale: The metric calculates the difference in choice probability for Option A between 'Expanded' trials (where intermediate cues V1 and V2 are present but perfectly cancel out in weighted additive terms) and 'Base' trials (where V1 and V2 are tied at 0). Strategy Mixture Theory predicts this difference is exactly zero, because neither the Take-The-Best component nor the Weighted Additive component changes between these specific pairs. Probabilistic Cue Search predicts a robust positive difference, because the presence of discriminating intermediate cues artificially elongates the search sequence, increasing the cumulative probability of stopping early on cues that favor Option A.
metric_source:
def metric(data: pd.DataFrame) -> float:
    base_trials = {
        ((1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 1, 0)),
        ((1, 0, 0, 0, 0, 1), (0, 0, 0, 1, 1, 0)),
        ((1, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 1))
    }
    expanded_trials = {
        ((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 0)),
        ((1, 1, 0, 0, 0, 1), (0, 0, 1, 1, 1, 0)),
        ((1, 1, 0, 1, 0, 0), (0, 0, 1, 0, 1, 1))
    }
    
    A_tuples = data['option_a_ratings'].apply(tuple)
    B_tuples = data['option_b_ratings'].apply(tuple)
    responses = data['response'].values
    
    base_res = []
    exp_res = []
    
    for a, b, r in zip(A_tuples, B_tuples, responses):
        pair = (a, b)
        if pair in base_trials:
            base_res.append(r)
        elif pair in expanded_trials:
            exp_res.append(r)
            
    if not base_res or not exp_res:
        return 0.0
        
    p_a_base = 1.0 - (sum(base_res) / len(base_res))
    p_a_expanded = 1.0 - (sum(exp_res) / len(exp_res))
    
    return float(p_a_expanded - p_a_base)
outcome: self_sim=0.0072 (var=0.0069) adversary_sim=0.0028 (var=0.0066) welch_t=+0.191 p=0.8492 (N=25, alpha=0.01) -> reject

[1] rationale: Instead of relying on a single a priori pair of trials which might suffer from ceiling/floor effects depending on the parameter distributions (e.g., if choices are heavily deterministic), this metric calculates the variance of the choice probabilities across all 8 unique trial types. Strategy Mixture Theory and Probabilistic Cue Search process the accumulation of evidence fundamentally differently: SMT mixes two static rules (one single-cue, one all-cues), whereas PCS evaluates cues sequentially and can stop at any intermediate step. This structural difference means PCS often produces a more graded set of choice probabilities across the trials (as the sequence length and early cue leads vary), while SMT tends to produce more polarized or clustered probabilities. The variance of the trial means robustly captures this difference in the dispersion of predictions.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Create a unique string identifier for each trial type based on the options
    data['trial_id'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x))) + '_' + data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Calculate the proportion of times Option B (response=1) was chosen for each trial type
    trial_means = data.groupby('trial_id')['response'].mean()
    
    # Return the variance of these choice probabilities across the 8 unique trial types
    # This captures how extremely the choice probabilities vary across the different stimulus pairs
    return float(trial_means.var())

outcome: self_sim=0.0038 (var=0.0003) adversary_sim=0.0042 (var=0.0002) welch_t=-0.088 p=0.9306 (N=25, alpha=0.01) -> reject

[2] rationale: This metric measures the difference in the rate of choosing Option B between two sets of trials: those where the compensatory WADD rule favors B (despite TTB favoring A) and those where both WADD and TTB favor A. Under Strategy Mixture Theory, the probability of choosing B drops to near zero in the second set because both underlying strategies agree on A, creating a massive contrast between the two sets. Under Probabilistic Cue Search, because evidence is accumulated sequentially and B possesses multiple trailing cues in the second set, there remains a substantial probability of choosing B if the search does not stop early, leading to a much smaller contrast between the two sets.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # WADD favors B in these trials, while TTB favors A in all trials
    wadd_b_trials = {
        ((1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 1, 0)),
        ((1, 1, 0, 0, 0, 0), (0, 0, 1, 1, 1, 0)),
        ((0, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 1)),
        ((0, 1, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1))
    }
    
    # WADD and TTB both favor A in these trials
    wadd_a_trials = {
        ((1, 0, 0, 0, 0, 1), (0, 0, 0, 1, 1, 0)),
        ((1, 1, 0, 0, 0, 1), (0, 0, 1, 1, 1, 0)),
        ((1, 0, 0, 1, 0, 0), (0, 0, 0, 0, 1, 1)),
        ((1, 1, 0, 1, 0, 0), (0, 0, 1, 0, 1, 1))
    }
    
    A_tuples = data['option_a_ratings'].apply(tuple)
    B_tuples = data['option_b_ratings'].apply(tuple)
    responses = data['response'].values
    
    b_choices_wadd_b = []
    b_choices_wadd_a = []
    
    for a, b, r in zip(A_tuples, B_tuples, responses):
        pair = (a, b)
        if pair in wadd_b_trials:
            b_choices_wadd_b.append(r)
        elif pair in wadd_a_trials:
            b_choices_wadd_a.append(r)
            
    if not b_choices_wadd_b or not b_choices_wadd_a:
        return 0.0
        
    mean_b_wadd_b = sum(b_choices_wadd_b) / len(b_choices_wadd_b)
    mean_b_wadd_a = sum(b_choices_wadd_a) / len(b_choices_wadd_a)
    
    return float(mean_b_wadd_b - mean_b_wadd_a)
outcome: self_sim=0.0754 (var=0.0152) adversary_sim=0.1017 (var=0.0145) welch_t=-0.762 p=0.45 (N=25, alpha=0.01) -> reject

[3] rationale: This metric isolates trials where the most predictive cue (cue 0) is tied (Trials 5 and 6) and evaluates the choice rate for Option B. In these trials, Option A leads on the next best cue (cue 1, validity 0.85), but Option B has a strong coalition of three weaker cues. Strategy Mixture Theory (SMT) handles this by either strictly choosing A (via Take-The-Best) or heavily favoring B (via Weighted Additive). Because the WADD component strongly favors B in Trial 5, the mixture predicts a substantial rate of choosing B. Probabilistic Cue Search (PCS), however, evaluates the cues sequentially. It first sees cue 1 favoring A, and has a probability of stopping immediately. If it continues, it faces a long sequence of cues favoring B, meaning early stopping strongly biases the choice back toward A. The models differ significantly in how they resolve this specific conflict between one strong early cue and a long tail of weaker cues.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Identify trials where the most valid cue (cue 0) is tied at 0.
    # These are trials 5 and 6.
    is_tied_cue0 = data['option_a_ratings'].apply(lambda x: x[0] == 0)
    
    # Return the proportion of times Option B was chosen in these specific trials.
    # In these trials, Option A relies on cue 1 (validity 0.85) while Option B 
    # has a coalition of multiple weaker cues (cues 3, 4, 5).
    if is_tied_cue0.sum() == 0:
        return 0.0
    return float(data.loc[is_tied_cue0, 'response'].mean())
outcome: self_sim=0.3617 (var=0.0238) adversary_sim=0.3192 (var=0.0325) welch_t=+0.896 p=0.3749 (N=25, alpha=0.01) -> reject

[4] rationale: This metric leverages the exact design rationale of the experiment by splitting the 8 unique trials into two matched groups (Group X and Group Y). In Group X, Option A possesses cue 3 and Option B possesses cue 2; in Group Y, this is perfectly reversed. Because cues 2 and 3 have identical validities (0.75), Strategy Mixture Theory (SMT) evaluates both Weighted Additive and Take-The-Best scores as strictly identical between the matched pairs, predicting a mean difference of exactly zero. Probabilistic Cue Search (PCS), however, evaluates cues sequentially. Due to the deterministic sorting of identical values in the search array, one of these cues is consistently evaluated before the other. This sequential bias causes the search to stop more frequently at the earlier cue, generating a systematic, non-zero difference in choice probabilities between the two groups.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Create a hashable key for Option A to identify trials
    data['a_key'] = data['option_a_ratings'].apply(tuple)
    
    # Group X: Trials where Option A has cue 3 and Option B has cue 2
    group_x_keys = {
        (1, 0, 0, 1, 0),
        (1, 0, 0, 1, 1),
        (0, 1, 0, 1, 0),
        (1, 1, 0, 1, 0)
    }
    
    # Group Y: Matched trials where Option A has cue 2 and Option B has cue 3
    group_y_keys = {
        (1, 0, 1, 0, 0),
        (1, 0, 1, 0, 1),
        (0, 1, 1, 0, 0),
        (1, 1, 1, 0, 0)
    }
    
    mask_x = data['a_key'].isin(group_x_keys)
    mask_y = data['a_key'].isin(group_y_keys)
    
    mean_x = data.loc[mask_x, 'response'].mean()
    mean_y = data.loc[mask_y, 'response'].mean()
    
    if pd.isna(mean_x) or pd.isna(mean_y):
        return 0.0
        
    return float(mean_x - mean_y)
outcome: self_sim=-0.0338 (var=0.0065) adversary_sim=-0.0013 (var=0.0074) welch_t=-1.379 p=0.1742 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Project the list of ratings to a hashable tuple\n    a_tuples = data['option_a_ratings'].apply(tuple)\n    \n    # Trials where Option A lacks the final cue (V4=0)\n    trials_1_2 = {\n        (1, 0, 0, 1, 0),  # Trial 1\n        (1, 0, 1, 0, 0)   # Trial 2\n    }\n    \n    # Matched trials where Option A possesses the final cue (V4=1)\n    trials_3_4 = {\n        (1, 0, 0, 1, 1),  # Trial 3\n        (1, 0, 1, 0, 1)   # Trial 4\n    }\n    \n    mask_1_2 = a_tuples.isin(trials_1_2)\n    mask_3_4 = a_tuples.isin(trials_3_4)\n    \n    # Calculate the proportion of times Option B was chosen (response = 1)\n    p_b_1_2 = data.loc[mask_1_2, 'response'].mean()\n    p_b_3_4 = data.loc[mask_3_4, 'response'].mean()\n    \n    if pd.isna(p_b_1_2) or pd.isna(p_b_3_4):\n        return 0.0\n        \n    # Return the difference in P(B) caused by adding V4 to Option A\n    return float(p_b_1_2 - p_b_3_4)",
  "rationale": "This metric isolates the effect of adding the lowest-validity cue (V4) to Option A by comparing Trials 1 & 2 (where A lacks V4) against their exact matches, Trials 3 & 4 (where A possesses V4). Under Strategy Mixture Theory (SMT), the compensatory WADD rule integrates all cues simultaneously. Adding V4 significantly boosts Option A's WADD score, shifting the WADD component from nearly tied to strongly favoring A, which produces a large drop in the probability of choosing B. Under Probabilistic Cue Search (PCS), cues are evaluated sequentially. Because V4 is the least valid cue, it is evaluated last. The probabilistic stopping rule means the search almost always terminates at one of the first four cues before ever reaching V4. Thus, the addition of V4 is rarely observed, resulting in a negligibly small change in choice probability. The metric will be large and positive for SMT, but near zero for PCS."
}
```

## Usage

```json
{
  "prompt_token_count": 7184,
  "candidates_token_count": 613,
  "total_token_count": 28949
}
```
