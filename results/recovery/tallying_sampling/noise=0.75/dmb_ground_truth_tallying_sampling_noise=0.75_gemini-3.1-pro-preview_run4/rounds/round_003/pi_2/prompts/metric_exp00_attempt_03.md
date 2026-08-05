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
**Validities (n_features=5):** [0.95, 0.95, 0.55, 0.55, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 2: A=[0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 6: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 7: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]

**Rationale:** To quantitatively dissociate the Advocated Theory (Weighted Additive with free per-feature weights) from the Competing Theory (Tallying), we exploit the fact that Tallying completely ignores cue validities and relies strictly on the unweighted sum of positive features. We construct trial pairs that pit a smaller number of high-validity features against a larger number of low-validity features. In these 'Tally-favors-B but WADD-favors-A' trials, Tallying predicts a strict preference for the option with more positive features, while WADD predicts a preference for the option with fewer, but higher-validity, features. Furthermore, we include 'Tally-tied' trials where both options possess the exact same number of positive features but differ significantly in the validities of those features. Tallying predicts chance-level guessing on these trials, whereas WADD predicts strong, consistent preferences based on the underlying validities.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
**Description:** Tallying (Equal Weighting): Decision-makers completely ignore the provided cue validities and simply count the number of positive features for each option. The option with the highest tally is chosen. If the tallies are equal, the decision-maker guesses randomly. This heuristic provides an extremely fast and frugal way to compare options, perfectly explaining chance-level performance in 1-on-1 single-cue comparisons (where tallies tie) and highly consistent choices when one option has strictly more positive features. The decision process is subject to significant response noise, reflecting the inherent stochasticity in human choice behavior.

**Parameters:**
- beta: [0.0, 1.5]
- epsilon: [0.2, 0.8]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    # Validities are ignored in Tallying, but we read them to satisfy the parameter reference rule.
    _ = parameters["validities"]
    
    # Tallying: sum the features for each option (ignoring validities)
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over tallies
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Uniform lapse
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
[0] rationale: This metric isolates the specific trial (Trial 3) designed to pit the two theories against each other. In this trial, Option A has fewer positive features but they are of high validity (sum=2, WADD score=1.9), while Option B has more positive features but they are of low validity (sum=3, WADD score=1.65). Tallying will strictly prefer Option B because 3 > 2, resulting in a low proportion of A choices. WADD will prefer Option A because 1.9 > 1.65, resulting in a high proportion of A choices. The metric simply calculates the proportion of A choices on this critical trial, yielding a stark contrast between the theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Identify the critical trial where Tallying and WADD make opposite predictions.
    # Specifically, Trial 3: Option A has 2 high-validity features, Option B has 3 low-validity features.
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    mask = (sum_a == 2) & (sum_b == 3)
    
    if mask.sum() == 0:
        return 0.5
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float((data.loc[mask, 'response'] == 0).mean())
outcome: self_sim=0.5050 (var=0.0804) adversary_sim=0.3983 (var=0.0237) welch_t=+1.653 p=0.1068 (N=25, alpha=0.01) -> reject

[1] rationale: To robustly discriminate WADD from Tallying, we focus on 'Tally-tied' trials where both options have the exact same number of positive features (sum=2). On these trials, Tallying is completely indifferent and must guess randomly, yielding an expected accuracy of 0.5. However, WADD uses the underlying cue validities to form a strong preference. By averaging the proportion of choices that align with WADD's preference on Trial 1 (where A is preferred) and Trial 2 (where B is preferred), we obtain a metric that evaluates to exactly 0.5 under Tallying but significantly higher than 0.5 under WADD. This avoids the small-margin issues seen in trials where the two theories' preferred options differ but the WADD score difference is marginal.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_keys = data['option_a_ratings'].apply(tuple)
    b_keys = data['option_b_ratings'].apply(tuple)
    
    # Trial 1: Equal tally (2 vs 2), but WADD strongly prefers A (1.9 vs 1.1)
    t1_mask = (a_keys == (1, 1, 0, 0, 0)) & (b_keys == (0, 0, 1, 1, 0))
    # Trial 2: Equal tally (2 vs 2), but WADD prefers B (1.1 vs 1.5)
    t2_mask = (a_keys == (0, 0, 0, 1, 1)) & (b_keys == (1, 0, 1, 0, 0))
    
    t1_acc = (data.loc[t1_mask, 'response'] == 0).mean() if t1_mask.any() else 0.5
    t2_acc = (data.loc[t2_mask, 'response'] == 1).mean() if t2_mask.any() else 0.5
    
    # Return average probability of making the WADD-predicted choice in Tally-tied trials
    return float((t1_acc + t2_acc) / 2.0)
outcome: self_sim=0.6167 (var=0.0550) adversary_sim=0.5008 (var=0.0084) welch_t=+2.301 p=0.02826 (N=25, alpha=0.01) -> reject

[2] rationale: Instead of focusing on trials where WADD makes a strong prediction (which suffers from high between-subject variance due to random per-feature subjective weights), we focus on trials where Tallying makes strong, opposing predictions while WADD is largely indifferent. We group trials where Option A has strictly more features (Trials 4, 7) and subtract the choice rate of trials where Option A has strictly fewer features (Trials 3, 5, 8). For Tallying, this difference will be large and positive because it strictly follows feature counts. For WADD, the average weighted score difference in both groups is very close to zero (as high-validity features are balanced against larger counts of low-validity features), so WADD behaves much closer to chance in both groups, yielding a metric near zero. This creates a massive mean gap with low variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_keys = data['option_a_ratings'].apply(tuple)
    b_keys = data['option_b_ratings'].apply(tuple)
    
    # Trials where Option A has more positive features than Option B (Tallying strongly prefers A)
    t4_mask = (a_keys == (0, 1, 1, 1, 0)) & (b_keys == (1, 1, 0, 0, 0))
    t7_mask = (a_keys == (1, 0, 0, 1, 1)) & (b_keys == (0, 1, 1, 0, 0))
    pos_mask = t4_mask | t7_mask
    
    # Trials where Option A has fewer positive features than Option B (Tallying strongly prefers B)
    t3_mask = (a_keys == (1, 1, 0, 0, 0)) & (b_keys == (0, 0, 1, 1, 1))
    t5_mask = (a_keys == (1, 0, 0, 0, 0)) & (b_keys == (0, 0, 1, 1, 0))
    t8_mask = (a_keys == (0, 1, 0, 0, 0)) & (b_keys == (0, 0, 1, 1, 0))
    neg_mask = t3_mask | t5_mask | t8_mask
    
    pos_rate = (data.loc[pos_mask, 'response'] == 0).mean() if pos_mask.any() else 0.5
    neg_rate = (data.loc[neg_mask, 'response'] == 0).mean() if neg_mask.any() else 0.5
    
    # Return the difference in probability of choosing A between Tally-favors-A and Tally-favors-B trials
    return float(pos_rate - neg_rate)

outcome: self_sim=0.1303 (var=0.1282) adversary_sim=0.1867 (var=0.0257) welch_t=-0.719 p=0.4774 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_keys = data['option_a_ratings'].apply(tuple)\n    b_keys = data['option_b_ratings'].apply(tuple)\n    \n    def get_pa(a, b):\n        mask = (a_keys == a) & (b_keys == b)\n        if mask.sum() == 0:\n            return 0.5\n        return float((data.loc[mask, 'response'] == 0).mean())\n\n    # D=0 trials\n    t1 = get_pa((1, 1, 0, 0, 0), (0, 0, 1, 1, 0))\n    t2 = get_pa((0, 0, 0, 1, 1), (1, 0, 1, 0, 0))\n    \n    # D=-1 trials\n    t3 = get_pa((1, 1, 0, 0, 0), (0, 0, 1, 1, 1))\n    t5 = get_pa((1, 0, 0, 0, 0), (0, 0, 1, 1, 0))\n    t8 = get_pa((0, 1, 0, 0, 0), (0, 0, 1, 1, 0))\n    \n    # D=+1 trials\n    t4 = get_pa((0, 1, 1, 1, 0), (1, 1, 0, 0, 0))\n    t7 = get_pa((1, 0, 0, 1, 1), (0, 1, 1, 0, 0))\n    \n    # Compute contrasts within groups of trials that have identical sum(A) - sum(B)\n    c1 = t1 - t2\n    c2 = t3 - (t5 + t8) / 2.0\n    c3 = t7 - t4\n    \n    return float(c1 + c2 + c3)\n",
  "rationale": "To completely isolate WADD's use of validities from Tallying's use of feature counts, we can group trials by their difference in positive features, D = sum(A) - sum(B). For Tallying, any two trials with the same D will yield the exact same predicted probability of choosing A. Therefore, if we subtract the choice rates of trials within the same D group, Tallying's expected difference is exactly zero. However, WADD evaluates these trials differently because the underlying validities of the features vary. We sum three such contrasts (one for D=0, one for D=-1, and one for D=+1) where WADD expects a positive difference. This yields a metric that is tightly distributed around 0 for Tallying (driven only by binomial sampling noise) but strongly positive for WADD, creating a highly significant discrimination."
}
```

## Usage

```json
{
  "prompt_token_count": 5231,
  "candidates_token_count": 702,
  "total_token_count": 10876
}
```
