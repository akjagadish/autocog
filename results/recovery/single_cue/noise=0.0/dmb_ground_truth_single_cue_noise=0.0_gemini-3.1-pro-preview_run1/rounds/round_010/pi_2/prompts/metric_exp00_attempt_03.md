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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 2: A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Advocated Theory (Pure Tallying) from the Competing Theory (Tallying with Strict Validity Tie-Breaking), we focus on 'Tally Tie' trials. Both models use Tallying as their primary decision mechanism, so they make identical predictions when one option strictly wins on more features. However, when the feature-wise wins are equal, Pure Tallying predicts a strict 50/50 guess, while the Competing Theory resolves the tie deterministically using the explicitly provided validities of the winning features. By including trials where both options win on an equal number of features but differ in the validities of those features, we isolate the secondary validity tie-breaker. We also include 'Strict Win' trials where the option with more wins has lower overall validity, confirming that the validity component in the Competing Theory is strictly bounded and does not override the primary Tallying mechanism.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Tallying with Strict Validity Tie-Breaking: Decision-makers primarily evaluate options by counting feature-wise wins (Tallying). To resolve ties in the tally counts, they employ a secondary, compensatory evaluation based on the explicitly provided validities of the winning features. Crucially, this secondary validity component is strictly bounded so that it can never override a strict tallying win, ensuring that the decision is lexicographic: Tallying always dominates, and validities only influence the choice when tally counts are exactly equal.

**Parameters:**
- validities: validities
- w_val: [0.0, 0.99]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    
    w_val = float(parameters["w_val"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Identify winning features for each option
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    
    # Primary component: Tallying (count of wins)
    tally_a = np.sum(a_wins)
    tally_b = np.sum(b_wins)
    
    # Secondary component: Validity score of winning features
    # Normalize validities so the secondary component is strictly bounded
    norm_validities = validities / np.sum(validities)
    val_a = np.sum(norm_validities * a_wins)
    val_b = np.sum(norm_validities * b_wins)
    
    # Compute final scores
    # Because w_val < 1.0 and the max difference in val is <= 1.0,
    # the validity component can never exceed 1.0, preserving Tallying dominance.
    score_a = tally_a + w_val * val_a
    score_b = tally_b + w_val * val_b
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice probability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    probs = e / np.sum(e)
    
    # Apply random lapse rate
    return (1.0 - epsilon) * probs + epsilon * np.array([0.5, 0.5])
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
[0] rationale: This metric isolates the 'Tally Tie' trials, where both options have an equal number of feature-wise wins. According to the Advocated Theory (Pure Tallying), the choice probability on these trials is exactly 50/50, regardless of the validities of the winning features or the softmax temperature. The Competing Theory, however, uses the validities of the winning features to break these ties deterministically before applying noise. By computing the proportion of tie trials where the subject chose the option with the higher validity sum for its winning features, we expect a value of ~0.5 for Pure Tallying and a value significantly > 0.5 for the Competing Theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validity_favored_choices = 0
    tie_trials_count = 0
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins == b_wins:
            val_a = np.sum(validities * (a > b))
            val_b = np.sum(validities * (b > a))
            
            if val_a > val_b:
                favored = 0
            elif val_b > val_a:
                favored = 1
            else:
                continue
                
            if row['response'] == favored:
                validity_favored_choices += 1
            tie_trials_count += 1
            
    if tie_trials_count == 0:
        return 0.5
    return float(validity_favored_choices / tie_trials_count)

outcome: self_sim=0.5167 (var=0.0062) adversary_sim=0.5421 (var=0.0061) welch_t=-1.146 p=0.2575 (N=25, alpha=0.01) -> reject

[1] rationale: Instead of looking at Tally Tie trials where the baseline choice probability is 50%, we compare two 'Strict Win' trials where the Tallying model predicts exactly the same choice probability. In Trial 4, the winning option wins on the 3 most valid features. In Trial 6, the winning option wins on the 3 least valid features. The Advocated Theory (Pure Tallying) ignores validities and only sees a 3-to-2 win in both cases, predicting a difference of exactly 0 between the choice probabilities. The Competing Theory uses validities as a secondary score component, meaning the softmax score difference will be much larger in Trial 4 than in Trial 6, leading to a systematically positive difference. Because this metric compares two trials within the same subject (sharing the exact same temperature and lapse rate), it strongly isolates the validity effect while minimizing between-subject variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    p4_wins = 0
    p4_n = 0
    p6_wins = 0
    p6_n = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = (a > b)
        b_wins = (b > a)
        
        if np.sum(a_wins) == 3 and np.sum(b_wins) == 2:
            if a_wins[0] and a_wins[1] and a_wins[2]:
                p4_wins += (row['response'] == 0)
                p4_n += 1
            elif a_wins[2] and a_wins[3] and a_wins[4]:
                p6_wins += (row['response'] == 0)
                p6_n += 1
        elif np.sum(b_wins) == 3 and np.sum(a_wins) == 2:
            if b_wins[0] and b_wins[1] and b_wins[2]:
                p4_wins += (row['response'] == 1)
                p4_n += 1
            elif b_wins[2] and b_wins[3] and b_wins[4]:
                p6_wins += (row['response'] == 1)
                p6_n += 1
                
    p4 = p4_wins / p4_n if p4_n > 0 else 0.5
    p6 = p6_wins / p6_n if p6_n > 0 else 0.5
    
    return float(p4 - p6)
outcome: self_sim=-0.0187 (var=0.0114) adversary_sim=0.0088 (var=0.0152) welch_t=-0.843 p=0.4035 (N=25, alpha=0.01) -> reject

[2] rationale: To drastically lower between-subject variance and isolate the true theoretical difference, this metric computes the product of two deviations from uniform guessing: (1) the proportion of validity-favored choices on 'Tally Tie' trials minus 0.5, and (2) the proportion of tally-favored choices on 'Strict Win' trials minus 0.5. Under the Advocated Theory (Pure Tallying), tie trials are strictly 50/50 regardless of the subject's determinism on non-tie trials. Thus, the expected value of the first term is exactly 0, and the expected value of the product is 0. Crucially, multiplying by the second term acts as a variance-reducer for the null theory because the second term is centered below 1.0. Under the Competing Theory, the validity tie-breaker pushes the first term above 0, and this effect is positively correlated with the second term (since low lapse rates and high beta increase determinism on both tie and non-tie trials). This yields a strongly positive mean for the Competing Theory while keeping the variance for Pure Tallying exceptionally tight, maximizing the t-statistic.
metric_source:
def metric(data: pd.DataFrame) -> float:
    tie_favored_choices = 0
    tie_count = 0
    non_tie_tally_choices = 0
    non_tie_count = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 0):
            tie_favored_choices += (1 if resp == 0 else 0)
            tie_count += 1
        elif a == (0, 0, 1, 1, 0) and b == (1, 1, 0, 0, 0):
            tie_favored_choices += (1 if resp == 1 else 0)
            tie_count += 1
            
        # Trial 2
        elif a == (1, 0, 0, 1, 0) and b == (0, 1, 0, 0, 1):
            tie_favored_choices += (1 if resp == 0 else 0)
            tie_count += 1
        elif a == (0, 1, 0, 0, 1) and b == (1, 0, 0, 1, 0):
            tie_favored_choices += (1 if resp == 1 else 0)
            tie_count += 1
            
        # Trial 5
        elif a == (0, 1, 0, 1, 0) and b == (1, 0, 1, 0, 0):
            tie_favored_choices += (1 if resp == 1 else 0)
            tie_count += 1
        elif a == (1, 0, 1, 0, 0) and b == (0, 1, 0, 1, 0):
            tie_favored_choices += (1 if resp == 0 else 0)
            tie_count += 1
            
        # Trial 3
        elif a == (0, 1, 1, 1, 0) and b == (1, 0, 0, 0, 1):
            non_tie_tally_choices += (1 if resp == 0 else 0)
            non_tie_count += 1
        elif a == (1, 0, 0, 0, 1) and b == (0, 1, 1, 1, 0):
            non_tie_tally_choices += (1 if resp == 1 else 0)
            non_tie_count += 1
            
        # Trial 4
        elif a == (1, 1, 1, 0, 0) and b == (0, 0, 0, 1, 1):
            non_tie_tally_choices += (1 if resp == 0 else 0)
            non_tie_count += 1
        elif a == (0, 0, 0, 1, 1) and b == (1, 1, 1, 0, 0):
            non_tie_tally_choices += (1 if resp == 1 else 0)
            non_tie_count += 1
            
        # Trial 6
        elif a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            non_tie_tally_choices += (1 if resp == 0 else 0)
            non_tie_count += 1
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            non_tie_tally_choices += (1 if resp == 1 else 0)
            non_tie_count += 1

    p_tie = tie_favored_choices / tie_count if tie_count > 0 else 0.5
    p_non_tie = non_tie_tally_choices / non_tie_count if non_tie_count > 0 else 0.5
    
    return float((p_tie - 0.5) * (p_non_tie - 0.5))
outcome: self_sim=0.0061 (var=0.0010) adversary_sim=0.0146 (var=0.0009) welch_t=-0.964 p=0.3398 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    tie_favored = 0\n    tie_total = 0\n    nontie_correct = 0\n    nontie_total = 0\n    \n    for _, row in data.iterrows():\n        a = tuple(row['option_a_ratings'])\n        b = tuple(row['option_b_ratings'])\n        resp = row['response']\n        \n        # T1 (Double weight because validity difference is exactly twice that of T2/T5)\n        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 0):\n            tie_favored += 2 * (1 if resp == 0 else 0)\n            tie_total += 2\n        elif a == (0, 0, 1, 1, 0) and b == (1, 1, 0, 0, 0):\n            tie_favored += 2 * (1 if resp == 1 else 0)\n            tie_total += 2\n            \n        # T2\n        elif a == (1, 0, 0, 1, 0) and b == (0, 1, 0, 0, 1):\n            tie_favored += (1 if resp == 0 else 0)\n            tie_total += 1\n        elif a == (0, 1, 0, 0, 1) and b == (1, 0, 0, 1, 0):\n            tie_favored += (1 if resp == 1 else 0)\n            tie_total += 1\n            \n        # T5\n        elif a == (0, 1, 0, 1, 0) and b == (1, 0, 1, 0, 0):\n            tie_favored += (1 if resp == 1 else 0)\n            tie_total += 1\n        elif a == (1, 0, 1, 0, 0) and b == (0, 1, 0, 1, 0):\n            tie_favored += (1 if resp == 0 else 0)\n            tie_total += 1\n            \n        # T3\n        elif a == (0, 1, 1, 1, 0) and b == (1, 0, 0, 0, 1):\n            nontie_correct += (1 if resp == 0 else 0)\n            nontie_total += 1\n        elif a == (1, 0, 0, 0, 1) and b == (0, 1, 1, 1, 0):\n            nontie_correct += (1 if resp == 1 else 0)\n            nontie_total += 1\n            \n        # T4\n        elif a == (1, 1, 1, 0, 0) and b == (0, 0, 0, 1, 1):\n            nontie_correct += (1 if resp == 0 else 0)\n            nontie_total += 1\n        elif a == (0, 0, 0, 1, 1) and b == (1, 1, 1, 0, 0):\n            nontie_correct += (1 if resp == 1 else 0)\n            nontie_total += 1\n            \n        # T6\n        elif a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):\n            nontie_correct += (1 if resp == 0 else 0)\n            nontie_total += 1\n        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):\n            nontie_correct += (1 if resp == 1 else 0)\n            nontie_total += 1\n\n    if nontie_total == 0 or tie_total == 0:\n        return 0.5\n        \n    acc = nontie_correct / nontie_total\n    \n    # Only subjects with clear tallying behavior (>75% accuracy on strict wins)\n    # provide a high-signal measure of the validity tie-breaker.\n    if acc > 0.75:\n        return float(tie_favored / tie_total)\n    else:\n        return 0.5",
  "rationale": "Previous attempts to measure the validity effect on tie trials failed because the between-subject variance (driven by binomial sampling noise and lapse rates) overwhelmed the small mean difference. This metric solves the problem by using the non-tie trials to filter out subjects who are mostly guessing (low beta / high epsilon). By forcing the metric to exactly 0.5 for subjects with <75% accuracy on strict tally wins, we drastically reduce the binomial noise variance for Pure Tallying, while isolating the high-signal subjects where the Competing Theory predicts the strongest validity effect. Furthermore, Trial 1 is weighted twice as heavily as Trials 2 and 5, perfectly matching the mathematical fact that Trial 1's validity difference is exactly twice as large, optimizing the signal-to-noise ratio."
}
```

## Usage

```json
{
  "prompt_token_count": 6094,
  "candidates_token_count": 1276,
  "total_token_count": 26619
}
```
