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

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 4: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** This design quantitatively dissociates Take The Best (TTB) from the Weighted Additive (WADD) model. TTB is a non-compensatory lexicographic heuristic: it makes a decision based entirely on the most valid discriminating feature, ignoring all others. WADD is fully compensatory: it integrates all features weighted by their validities. By selecting a spread of validities (0.9, 0.8, 0.7, 0.6, 0.5), we construct critical trials where the single highest-validity feature points to one option, but the sum of the remaining lower-validity features strongly favors the other option. In these trials, TTB and WADD make diametrically opposed predictions, allowing us to pinpoint which decision rule subjects are employing.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People employ a probabilistic 'Take The Best' (TTB) heuristic to make binary choices. They order features by their subjective validity, which is informed by objective validities but subject to random fluctuations (noise). They compare the options sequentially based on this subjective ordering. The first feature that discriminates between the two options determines the choice, and all subsequent features are ignored. This introduces stochasticity into the cue hierarchy, explaining why choices sometimes deviate from strict reliance on the single highest-validity cue.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- sigma: [0.0, 5.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    sigma = float(parameters["sigma"])
    
    # Add Gaussian noise to validities to model subjective fluctuations in cue hierarchy
    noisy_validities = validities + np.random.normal(0, sigma, size=validities.shape)
    
    # Order features by noisy validity in descending order
    order = np.argsort(noisy_validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    # Iterate through features in order of validity
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            scores[0] = 1.0
            break
        elif stim[1, idx] > stim[0, idx]:
            scores[1] = 1.0
            break
            
    # If no features discriminate, both options are tied
    if scores[0] == 0.0 and scores[1] == 0.0:
        scores = np.array([0.5, 0.5])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
**Description:** People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson
    # 1993). Stimulus is the pair of option feature vectors for the
    # current trial: array-like of shape (2, n_features), row 0 =
    # option A, row 1 = option B. Each option's score is the dot
    # product of its feature vector with the (subjective) validity
    # weights. History is ignored: the choice depends only on the
    # current stimulus.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"weights length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ (validities * w)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax (standard choice-with-lapse
    # formulation; epsilon=0.5 yields equiprobable options, matching
    # the paper's maximum-noise condition).
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
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
[0] rationale: In the chosen experimental design, the first feature (index 0) has the highest validity (0.9). For all 5 trial types, this feature perfectly discriminates between the two options. The Take The Best (TTB) heuristic relies almost entirely on this single most valid feature, predicting choices that align with it. In contrast, the Weighted Additive (WADD) model integrates all features, and the design is specifically constructed such that in 4 out of 5 trial types, the sum of the lower-validity features outweighs the single best feature, leading WADD to predict the opposite choice. Therefore, the proportion of trials where the subject chooses the option favored by the most valid feature will be very high for TTB and very low for WADD, cleanly discriminating the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a_ratings = row['option_a_ratings']
        b_ratings = row['option_b_ratings']
        resp = row['response']
        
        # Feature 0 is the most valid feature (validity 0.9)
        if a_ratings[0] > b_ratings[0]:
            ttb_choice = 0
        elif b_ratings[0] > a_ratings[0]:
            ttb_choice = 1
        else:
            ttb_choice = -1
            
        if ttb_choice == resp:
            matches.append(1.0)
        elif ttb_choice != -1:
            matches.append(0.0)
            
    return float(np.mean(matches)) if matches else 0.5
outcome: self_sim=0.4478 (var=0.0074) adversary_sim=0.3983 (var=0.0415) welch_t=+1.118 p=0.2717 (N=25, alpha=0.01) -> reject

[1] rationale: Measures the proportion of trials where the subject chooses the option with fewer positive features. WADD integrates all features and thus strongly favors the option with more positive features (compensatory). TTB, relying on a single discriminating feature, is much more likely to choose an option with fewer features if it happens to possess the highest-ranked cue.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    fewer_chosen = []
    for _, row in data.iterrows():
        a_ratings = row['option_a_ratings']
        b_ratings = row['option_b_ratings']
        resp = row['response']
        
        sum_a = sum(a_ratings)
        sum_b = sum(b_ratings)
        
        if sum_a < sum_b:
            fewer_chosen.append(1.0 if resp == 0 else 0.0)
        elif sum_b < sum_a:
            fewer_chosen.append(1.0 if resp == 1 else 0.0)
            
    return float(np.mean(fewer_chosen)) if fewer_chosen else 0.5
outcome: self_sim=0.4203 (var=0.0089) adversary_sim=0.3361 (var=0.0472) welch_t=+1.779 p=0.0846 (N=25, alpha=0.01) -> reject

[2] rationale: By restricting the analysis strictly to trials where one option is a 'singleton' (has exactly 1 positive feature) and the other has multiple positive features (Trials 1 and 5), we maximize the contrast between TTB and WADD. In these specific trials, the singleton option is always the one possessing the highest-validity cue (0.9). TTB will choose this singleton option frequently because it only relies on the top-ranked feature. WADD, however, will strongly penalize the singleton option because the summed weights of the 3 or 4 opposing features will almost always eclipse the single feature's weight, leading WADD to reliably choose the multi-feature option. This stark divergence on a targeted subset of trials produces a much larger mean gap and lower variance than aggregating over all trials.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    singleton_chosen = []
    for _, row in data.iterrows():
        a_ratings = row['option_a_ratings']
        b_ratings = row['option_b_ratings']
        resp = row['response']
        
        sum_a = sum(a_ratings)
        sum_b = sum(b_ratings)
        
        # Identify trials where one option has exactly 1 positive feature and the other has many
        if sum_a == 1 and sum_b > 1:
            singleton_chosen.append(1.0 if resp == 0 else 0.0)
        elif sum_b == 1 and sum_a > 1:
            singleton_chosen.append(1.0 if resp == 1 else 0.0)
            
    return float(np.mean(singleton_chosen)) if len(singleton_chosen) > 0 else 0.5
outcome: self_sim=0.3832 (var=0.0159) adversary_sim=0.2742 (var=0.0423) welch_t=+2.257 p=0.02958 (N=25, alpha=0.01) -> reject

[3] rationale: This metric calculates the difference in the rate of choosing Option A between two specific sets of trials. In Trials 3 and 5, Option A has more positive features overall, but Option B possesses the single most valid feature (feature 0). In Trials 1 and 2, the situation is reversed: Option B has more features, but Option A possesses feature 0. The WADD model is compensatory and integrates all features, so it will strongly favor Option A in T3/T5 and Option B in T1/T2, leading to a highly positive difference (High - Low). Conversely, the TTB heuristic relies heavily on the single best feature, meaning it will favor Option B in T3/T5 and Option A in T1/T2, leading to a negative difference (Low - High). This stark positive-versus-negative contrast directly pits the core mechanisms of the two theories against each other.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    t3_t5_choose_A = []
    t1_t2_choose_A = []
    
    for _, row in data.iterrows():
        a_ratings = row['option_a_ratings']
        b_ratings = row['option_b_ratings']
        resp = row['response']
        
        sum_a = sum(a_ratings)
        sum_b = sum(b_ratings)
        
        choose_a = 1.0 if resp == 0 else 0.0
        
        # T3 & T5: Option A has more features overall, but Option B has the best feature (index 0)
        if a_ratings[0] == 0 and b_ratings[0] == 1 and sum_a > sum_b:
            t3_t5_choose_A.append(choose_a)
        # T1 & T2: Option B has more features overall, but Option A has the best feature (index 0)
        elif a_ratings[0] == 1 and b_ratings[0] == 0 and sum_a < sum_b:
            t1_t2_choose_A.append(choose_a)
            
    if not t3_t5_choose_A or not t1_t2_choose_A:
        return 0.0
        
    return float(np.mean(t3_t5_choose_A) - np.mean(t1_t2_choose_A))
outcome: self_sim=0.1595 (var=0.0354) adversary_sim=0.3279 (var=0.1887) welch_t=-1.779 p=0.0846 (N=25, alpha=0.01) -> reject

[4] rationale: In Trials 3 and 4, Option B possesses the single most valid discriminating feature (feature 0 in Trial 3, feature 1 in Trial 4), while Option A possesses a greater number of lower-validity features. The Take The Best (TTB) heuristic, which relies on the highest-ranking discriminating cue, will systematically prefer Option B. Conversely, the Weighted Additive (WADD) model integrates all features, and the sum of the lower-validity cues for Option A strongly outweighs the single cue for Option B, leading WADD to systematically prefer Option A. By calculating the proportion of Option B choices specifically on these two trials, we obtain a metric that should be close to 1 for TTB and close to 0 for WADD.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    b_choices = []
    for _, row in data.iterrows():
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 3: A=[0, 1, 1, 1, 0], B=[1, 0, 0, 0, 0]
        # Trial 4: A=[0, 0, 1, 1, 1], B=[0, 1, 0, 0, 0]
        if (a == [0, 1, 1, 1, 0] and b == [1, 0, 0, 0, 0]) or \
           (a == [0, 0, 1, 1, 1] and b == [0, 1, 0, 0, 0]):
            b_choices.append(1.0 if resp == 1 else 0.0)
            
    return float(np.mean(b_choices)) if b_choices else 0.5
outcome: self_sim=0.3406 (var=0.0168) adversary_sim=0.2894 (var=0.0313) welch_t=+1.168 p=0.2491 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    subject_scores = []\n    for subj, subj_df in data.groupby('subject_id'):\n        t1_a = []\n        t2_a = []\n        t4_a = []\n        t6_a = []\n        \n        for _, row in subj_df.iterrows():\n            a = tuple(row['option_a_ratings'])\n            b = tuple(row['option_b_ratings'])\n            resp = row['response']\n            choose_a = 1.0 if resp == 0 else 0.0\n            \n            # Trial 1\n            if a == (1, 0, 0, 0, 0) and b == (0, 1, 1, 1, 1):\n                t1_a.append(choose_a)\n            elif b == (1, 0, 0, 0, 0) and a == (0, 1, 1, 1, 1):\n                t1_a.append(1.0 - choose_a)\n            # Trial 2\n            elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):\n                t2_a.append(choose_a)\n            elif b == (1, 1, 0, 0, 0) and a == (0, 0, 1, 1, 1):\n                t2_a.append(1.0 - choose_a)\n            # Trial 4\n            elif a == (0, 0, 1, 1, 1) and b == (0, 1, 0, 0, 0):\n                t4_a.append(choose_a)\n            elif b == (0, 0, 1, 1, 1) and a == (0, 1, 0, 0, 0):\n                t4_a.append(1.0 - choose_a)\n            # Trial 6\n            elif a == (0, 1, 0, 0, 0) and b == (1, 0, 0, 0, 0):\n                t6_a.append(choose_a)\n            elif b == (0, 1, 0, 0, 0) and a == (1, 0, 0, 0, 0):\n                t6_a.append(1.0 - choose_a)\n                \n        p1 = float(np.mean(t1_a)) if t1_a else 0.5\n        p2 = float(np.mean(t2_a)) if t2_a else 0.5\n        p4 = float(np.mean(t4_a)) if t4_a else 0.5\n        p6 = float(np.mean(t6_a)) if t6_a else 0.5\n        \n        # Linear combination of probabilities\n        subject_scores.append(p1 + p2 + 2 * p4 + 2 * p6)\n        \n    return float(np.mean(subject_scores))",
  "rationale": "By examining the feature difference vectors (A - B) for the trials, we can find an exact linear dependency: (A-B)_T1 + (A-B)_T2 + 2*(A-B)_T4 + 2*(A-B)_T6 = [0, 0, 0, 0, 0]. Because the Weighted Additive (WADD) model evaluates options using a linear combination of features, the sum of the underlying scores for these trials is exactly zero, regardless of the random subjective weights the participant uses. This strictly anchors WADD's expected sum of choice probabilities P(A|T1) + P(A|T2) + 2*P(A|T4) + 2*P(A|T6) around 3.0, yielding extremely low between-subject variance. In contrast, the Take The Best (TTB) heuristic is non-compensatory and driven by the probability of each feature being the most valid. For TTB, this same sum algebraically simplifies to 2 + P(Feature 1 is highest), which reliably evaluates to ~2.2. This structural difference creates a robust, low-variance gap between the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 6014,
  "candidates_token_count": 1051,
  "total_token_count": 16115
}
```
