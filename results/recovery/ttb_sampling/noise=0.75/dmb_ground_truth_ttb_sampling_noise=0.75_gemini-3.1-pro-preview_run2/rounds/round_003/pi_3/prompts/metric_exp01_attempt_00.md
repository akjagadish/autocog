# metric_exp01_attempt_00

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
  trial 1: A=[1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 3: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 6: A=[1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1]
  trial 7: A=[1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1]
  trial 8: A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1]
  trial 9: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 10: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Rationale:** To quantitatively dissociate Take-The-Best (TTB) from Probabilistic Cue Selection (Stochastic TTB), we manipulate the number and validities of subordinate discriminating features while holding the single best discriminating feature constant. Under pure TTB, the decision is strictly based on the most valid discriminating feature; the presence, number, or validities of other lower-ranking discriminating features have no effect on the predicted choice probabilities (the pre-noise score is always 1 for the TTB winner and 0 for the loser). In contrast, Stochastic TTB samples among all discriminating features with probabilities proportional to exp(gamma * validity). Thus, Stochastic TTB predicts that the choice probability for the option favored by the best cue will decrease as more subordinate cues favor the opposing option, whereas pure TTB predicts a constant choice probability across these conditions. By systematically varying the opposing subordinate cues, we can directly test whether choice probabilities are invariant to subordinate cues (TTB) or sensitive to them (Stochastic TTB).

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Take-The-Best (TTB) heuristic: People make decisions by ranking features according to their validities and choosing the option that is favored by the single most valid discriminating feature. If no feature discriminates, they guess. This is a lexicographic, non-compensatory strategy. However, human execution of this strategy is highly noisy, so choice probabilities are heavily tempered by response noise (low beta) and random guessing lapses (high epsilon).

**Parameters:**
- beta: [0.0, 2.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    # We use a stable sort to preserve the original feature order in case of ties.
    ranked_features = np.argsort(validities, kind='stable')[::-1]
    
    scores = np.array([0.0, 0.0])
    # Find the first feature that discriminates between the two options
    for idx in ranked_features:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    # If no feature discriminates, the core preference is uniform
    if scores[0] == 0.0 and scores[1] == 0.0:
        p_core = np.array([0.5, 0.5])
    else:
        beta = float(parameters["beta"])
        # Softmax over the scores to introduce response noise
        z = beta * scores
        z = z - np.max(z)
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Probabilistic Cue Selection (Stochastic TTB): Human decision-makers use a non-compensatory, one-reason heuristic but are stochastic in their cue retrieval. Instead of strictly ranking features by validity, subjects sample a feature to evaluate with a probability proportional to its validity (via a softmax). If the sampled feature discriminates between the options, they base their choice entirely on that feature. If it does not discriminate, they sample again. This maintains the non-compensatory nature of the decision while naturally introducing variability in which cue is selected, offering a mechanistic explanation for choice noise without relying on compensatory tallying.

**Parameters:**
- gamma: [0.0, 50.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Identify which features discriminate between A and B
    discrim = (a != b)
    
    if not np.any(discrim):
        # If no features discriminate, the decision is a random guess
        p_core = np.array([0.5, 0.5])
    else:
        # The subject repeatedly samples features proportional to exp(gamma * validity)
        # until a discriminating feature is found. The probability that the first
        # discriminating feature found is feature i is equivalent to a softmax over
        # the validities restricted to the set of discriminating features.
        z = gamma * validities[discrim]
        z = z - np.max(z)  # For numerical stability
        w = np.exp(z)
        w = w / np.sum(w)
        
        # The chosen discriminating feature dictates the choice entirely.
        # Sum the probabilities of sampling a feature that favors A vs B.
        favor_a = (a[discrim] > b[discrim])
        favor_b = (b[discrim] > a[discrim])
        
        p_a = np.sum(w[favor_a])
        p_b = np.sum(w[favor_b])
        
        p_core = np.array([p_a, p_b])
        
    # Blend with a uniform lapse rate for general response noise/inattention
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
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
[0] rationale: Under strict Take-The-Best (TTB), the decision is determined entirely by the single most valid discriminating feature. In all selected trials, the best feature (index 0) favors Option A, meaning TTB predicts the exact same choice probability for A regardless of the opposing cues (difference ≈ 0). Under Probabilistic Cue Selection (Stochastic TTB), the probability of choosing A depends on the relative validities of all discriminating features. When the opposing features are weak (e.g., indices 3 and 4), Stochastic TTB predicts a higher probability of choosing A compared to when the opposing features are strong (e.g., indices 1 and 2). Thus, the difference in the rate of choosing A between these two sets of trials will be substantially greater than zero for Stochastic TTB, but near zero for TTB.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Convert rating lists to string representations for easy matching
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    # Trials where the best cue (index 0) is opposed by weak cues (indices 3, 4)
    weak_opp = ((a_str == '10000') & (b_str == '00001')) | ((a_str == '10000') & (b_str == '00011'))
    
    # Trials where the best cue (index 0) is opposed by strong cues (indices 1, 2)
    strong_opp = ((a_str == '10000') & (b_str == '01000')) | ((a_str == '10000') & (b_str == '01100'))
    
    # Calculate the probability of choosing Option A (which is favored by the best cue in both sets)
    p_a_weak = (data.loc[weak_opp, 'response'] == 0).mean() if weak_opp.sum() > 0 else 0.5
    p_a_strong = (data.loc[strong_opp, 'response'] == 0).mean() if strong_opp.sum() > 0 else 0.5
    
    return float(p_a_weak - p_a_strong)
outcome: self_sim=0.0067 (var=0.0228) adversary_sim=0.0578 (var=0.0328) welch_t=-1.084 p=0.2842 (N=25, alpha=0.01) -> reject

[1] rationale: Under strict Take-The-Best (TTB), the decision is determined entirely by the single most valid discriminating feature. In all selected trials, the best feature clearly favors one option, meaning TTB predicts the exact same probability of choosing the favored option regardless of the opposing cues. Thus, the error rate (choosing the option not favored by the best cue) should be identical across all these trials (difference ≈ 0). Under Probabilistic Cue Selection (Stochastic TTB), the probability of choosing an option depends on the relative validities of all discriminating features. When the opposing features are strong or numerous (Trials 7, 10, 8, 1), Stochastic TTB predicts a substantially higher error rate compared to when opposing features are weak and few (Trials 4, 6, 3, 2). By subtracting the error rate on these 'easy' trials from the 'hard' trials, we expect a value near 0 for TTB and a significantly positive value for Stochastic TTB. Pooling across 4 trial types per condition minimizes binomial variance, maximizing the statistical distinguishability of the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Convert rating lists to string representations for easy matching
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    # Hard trials: STTB predicts higher error rate because opposing cues are strong/numerous
    # Trial 7
    m7 = (a_str == '10000') & (b_str == '01100')
    e7 = (data.loc[m7, 'response'] == 1).sum()
    n7 = m7.sum()
    
    # Trial 10
    m10 = (a_str == '01100') & (b_str == '10000')
    e10 = (data.loc[m10, 'response'] == 0).sum()
    n10 = m10.sum()
    
    # Trial 8
    m8 = (a_str == '10000') & (b_str == '00011')
    e8 = (data.loc[m8, 'response'] == 1).sum()
    n8 = m8.sum()
    
    # Trial 1
    m1 = (a_str == '10000') & (b_str == '01000')
    e1 = (data.loc[m1, 'response'] == 1).sum()
    n1 = m1.sum()
    
    hard_err = (e7 + e10 + e8 + e1) / max(1, n7 + n10 + n8 + n1)
    
    # Easy trials: STTB predicts lower error rate because opposing cues are weak/few
    # Trial 4
    m4 = (a_str == '10000') & (b_str == '00001')
    e4 = (data.loc[m4, 'response'] == 1).sum()
    n4 = m4.sum()
    
    # Trial 6
    m6 = (a_str == '01000') & (b_str == '00001')
    e6 = (data.loc[m6, 'response'] == 1).sum()
    n6 = m6.sum()
    
    # Trial 3
    m3 = (a_str == '10000') & (b_str == '00010')
    e3 = (data.loc[m3, 'response'] == 1).sum()
    n3 = m3.sum()
    
    # Trial 2
    m2 = (a_str == '10000') & (b_str == '00100')
    e2 = (data.loc[m2, 'response'] == 1).sum()
    n2 = m2.sum()
    
    easy_err = (e4 + e6 + e3 + e2) / max(1, n4 + n6 + n3 + n2)
    
    return float(hard_err - easy_err)
outcome: self_sim=0.0072 (var=0.0124) adversary_sim=0.0600 (var=0.0148) welch_t=-1.600 p=0.1162 (N=25, alpha=0.01) -> reject

[2] rationale: By restricting the metric to the most extreme 'easy' (opposed by a single very weak cue) and 'hard' (opposed by multiple or very strong cues) trials, we maximize the mean difference between the theories. Under pure TTB, the choice probability relies entirely on the single best discriminating feature, so the TTB-consistent choice rate will be identical across these sets, yielding a metric near 0. Under Probabilistic Cue Selection, the probability of sampling the best cue is heavily diluted in the 'hard' trials compared to the 'easy' trials, yielding a significantly positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    # Easy trials: TTB choice is opposed by a single weak feature
    m4 = (a_str == '10000') & (b_str == '00001')
    m6 = (a_str == '01000') & (b_str == '00001')
    m3 = (a_str == '10000') & (b_str == '00010')
    easy_mask = m4 | m6 | m3
    easy_ttb_rate = (data.loc[easy_mask, 'response'] == 0).mean() if easy_mask.sum() > 0 else 0.5
    
    # Hard trials: TTB choice is opposed by strong or multiple features
    m7 = (a_str == '10000') & (b_str == '01100')
    m10 = (a_str == '01100') & (b_str == '10000')
    m1 = (a_str == '10000') & (b_str == '01000')
    
    hard_ttb_7 = (data.loc[m7, 'response'] == 0).sum()
    hard_ttb_10 = (data.loc[m10, 'response'] == 1).sum()
    hard_ttb_1 = (data.loc[m1, 'response'] == 0).sum()
    
    hard_n = m7.sum() + m10.sum() + m1.sum()
    hard_ttb_rate = (hard_ttb_7 + hard_ttb_10 + hard_ttb_1) / hard_n if hard_n > 0 else 0.5
    
    return float(easy_ttb_rate - hard_ttb_rate)
outcome: self_sim=0.0259 (var=0.0160) adversary_sim=0.0689 (var=0.0249) welch_t=-1.062 p=0.294 (N=25, alpha=0.01) -> reject

[3] rationale: Under the strict Take-The-Best (TTB) model, the decision score is always determined entirely by the single most valid discriminating feature. As a result, the core probability of choosing the TTB-favored option is exactly the same across all 10 trial types, meaning the variance of the TTB-consistent choice proportions across trial types should be very low (driven only by binomial noise). In contrast, Stochastic TTB predicts that the probability of choosing the TTB-favored option varies heavily depending on the validities and number of opposing cues. Therefore, the variance of the TTB-consistent choice proportions across the 10 trial types will be significantly higher for Stochastic TTB than for TTB.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, map(int, x))))
    
    trial_type = a_str + "_" + b_str
    
    def get_ttb_favored(a_s, b_s):
        for i in range(len(a_s)):
            if a_s[i] > b_s[i]: return 0
            if b_s[i] > a_s[i]: return 1
        return 0
        
    ttb_favored = [get_ttb_favored(a, b) for a, b in zip(a_str, b_str)]
    is_ttb_consistent = (data['response'] == ttb_favored).astype(float)
    
    df = pd.DataFrame({'trial_type': trial_type, 'is_ttb_consistent': is_ttb_consistent})
    props = df.groupby('trial_type')['is_ttb_consistent'].mean()
    
    return float(np.var(props))
outcome: self_sim=0.0007 (var=0.0001) adversary_sim=0.0013 (var=0.0001) welch_t=-0.194 p=0.8466 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    # Convert lists of binary ratings into 2D numpy arrays\n    a_ratings = np.stack(data['option_a_ratings'].values)\n    b_ratings = np.stack(data['option_b_ratings'].values)\n    \n    # Find trials with exactly 1 discriminating feature (Hamming distance == 1)\n    hamming_dist = np.sum(a_ratings != b_ratings, axis=1)\n    mask = hamming_dist == 1\n    \n    if not np.any(mask):\n        return 0.5\n        \n    # Determine the favored option on these trials.\n    # Since they differ by exactly one feature, the option with the '1' \n    # on that feature will have a strictly greater sum of ratings.\n    a_sum = np.sum(a_ratings[mask], axis=1)\n    b_sum = np.sum(b_ratings[mask], axis=1)\n    \n    # favored_option is 0 (Option A) if A has the higher sum, else 1 (Option B)\n    favored_option = (b_sum > a_sum).astype(int)\n    responses = data.loc[mask, 'response'].values\n    \n    # Return the absolute choice rate of the favored option on these specific trials\n    return float(np.mean(responses == favored_option))",
  "rationale": "Under Probabilistic Cue Selection (STTB), the decision-maker samples a discriminating feature proportional to its validity. On trials where exactly one feature discriminates between the options (Hamming distance of 1), STTB will sample that single feature with probability 1.0, yielding a pre-lapse choice probability of exactly 1.0 for the favored option, regardless of the gamma parameter. In contrast, under the strict Take-The-Best (TTB) model, the pre-lapse choice probability is determined by a softmax over the feature scores, tempered by the beta parameter. Since beta is bounded in the parameter space (max 2.0), TTB's pre-lapse probability is strictly bounded (max ~0.88). Because both models are subject to the same epsilon lapse rates, the absolute observed choice rate for the favored option on these single-discriminating-feature trials will be substantially and reliably higher for STTB than for TTB, providing a clean separation between the theories."
}
```

## Usage

```json
{
  "prompt_token_count": 6643,
  "candidates_token_count": 562,
  "total_token_count": 23430
}
```
