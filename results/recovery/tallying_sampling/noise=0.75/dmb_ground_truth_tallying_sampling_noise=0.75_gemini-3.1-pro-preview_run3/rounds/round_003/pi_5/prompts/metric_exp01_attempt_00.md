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

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Rationale:** To robustly dissociate the Weighted Additive (WADD) model with a flexible sigmoid transformation from the Tallying heuristic, we employ a 5-feature design with a linear spread of validities. Tallying simply counts the total number of positive features, completely ignoring validities. WADD, however, weights features by their subjective validities. By using a sigmoid transformation, WADD can heavily emphasize the top validities while suppressing the lower ones. The critical dissociation trials pit an option with fewer but highest-validity features (favored by WADD) against an option with more but lower-validity features (favored by Tallying). We also include trials where Tallying predicts a tie (equal number of positive features) but WADD has a strong preference, to expose the validity-weighting mechanism.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Weighted Additive (WADD) Decision Theory with Flexible Sigmoid Subjective Validity Transformation

**Parameters:**
- beta: [0.01, 10.0]
- epsilon: [0.0, 1.0]
- gamma: [0.1, 5.0]
- delta: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    delta = float(parameters["delta"])
    
    # Transform raw probabilities into subjective weights using a sigmoid function
    w = 1.0 / (1.0 + np.exp(-gamma * (validities - delta)))
    
    # Calculate the overall score for each option by multiplying cue values by subjective weights
    scores = stim @ w
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice based on the weighted scores
    z = beta * scores
    z = z - np.max(z)  # for numerical stability
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate for uniform guessing
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


## COMPETING THEORY
**Description:** Decision-makers use a 'Tallying' (Equal-Weight) heuristic, a compensatory strategy that ignores cue validities. They simply count the number of positive features (or advantages) each option has and choose the option with the highest total count. If the counts are equal, they guess. Because pure Tallying makes choices that strongly oppose Take The Best on compensatory trials, high levels of choice stochasticity (noise) are needed to pull the predicted consistency up toward the observed ~0.40-0.42 range, reflecting uncertainty or lapses in applying the heuristic.

**Parameters:**
- beta: [0.01, 1.0]
- epsilon: [0.3, 0.8]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    
    # Tallying: sum the unweighted feature values for each option.
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
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
[0] rationale: This metric calculates the proportion of choices that align with the Weighted Additive (WADD) model on 'strict conflict' trials. By using a steep weighting vector that approximates WADD's subjective validity transformation, we can identify trials where WADD strongly prefers one option (due to high-validity features) while Tallying strongly prefers the other (due to a higher total count of features). On these trials, Tallying will only pick the WADD-preferred option by chance (lapse), resulting in a low score. In contrast, the WADD model will systematically pick the WADD-preferred option, yielding a significantly higher score.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Steep weights to approximate WADD's subjective validity transformation
    w = np.array([10.0, 5.0, 1.0, 1.0, 1.0])
    
    conflict_matches = 0
    conflict_trials = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        w_a = np.dot(a, w)
        w_b = np.dot(b, w)
        wadd_choice = 0 if w_a > w_b else (1 if w_b > w_a else -1)
        
        t_a = np.sum(a)
        t_b = np.sum(b)
        tally_choice = 0 if t_a > t_b else (1 if t_b > t_a else -1)
        
        # We only care about strict conflicts where the two models prefer OPPOSITE options
        if wadd_choice != -1 and tally_choice != -1 and wadd_choice != tally_choice:
            conflict_trials += 1
            if row['response'] == wadd_choice:
                conflict_matches += 1
                
    if conflict_trials == 0:
        return 0.5
    return conflict_matches / conflict_trials
outcome: self_sim=0.3983 (var=0.0187) adversary_sim=0.4633 (var=0.0038) welch_t=-2.170 p=0.03724 (N=25, alpha=0.01) -> reject

[1] rationale: This metric isolates trials where Tallying predicts a perfect tie (because both options have the same total number of positive features). On these trials, the Tallying heuristic is forced to guess, resulting in a 50% match rate with any systematic preference. In contrast, the Weighted Additive (WADD) model evaluates the subjective validities of the features; since the design features options with strictly dominating high-validity cues on these tie trials, WADD will systematically prefer the option with the higher-validity features. By measuring the proportion of choices that align with the higher-validity option strictly on Tallying-tie trials, we cleanly separate WADD's validity-weighting mechanism from Tallying's random guessing.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.98, 0.92, 0.65, 0.58, 0.52])
    match_count = 0
    tie_count = 0
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        if np.sum(a) == np.sum(b):
            a_val = np.dot(a, validities)
            b_val = np.dot(b, validities)
            if a_val != b_val:
                expected = 0 if a_val > b_val else 1
                if row['response'] == expected:
                    match_count += 1
                tie_count += 1
    if tie_count == 0:
        return 0.5
    return match_count / tie_count
outcome: self_sim=0.5322 (var=0.0089) adversary_sim=0.4889 (var=0.0070) welch_t=+1.716 p=0.09265 (N=25, alpha=0.01) -> reject

[2] rationale: This metric measures the subject's reliance on the single most valid feature (validity = 0.98). We isolate all trials where exactly one option possesses this feature. For the Tallying heuristic, which ignores validities and simply counts positive features, the option possessing feature 1 is actually at a numerical disadvantage across the experiment's specific trial design (Tallying prefers the option without feature 1 on trials 1 and 8, prefers the option with feature 1 on trial 7, and ties on the rest). Consequently, Tallying predicts that subjects will choose the option with feature 1 less than 50% of the time. In contrast, the WADD model heavily weights the 0.98 validity feature, predicting a choice rate systematically greater than 50%. This creates a clean, robust divergence in the expected proportion.
metric_source:
def metric(data: pd.DataFrame) -> float:
    count = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        # Check if exactly one option has the most valid feature (index 0)
        if a[0] != b[0]:
            total += 1
            # Did the subject choose the option with feature 1?
            if a[0] == 1 and row['response'] == 0:
                count += 1
            elif b[0] == 1 and row['response'] == 1:
                count += 1
    if total == 0:
        return 0.5
    return count / total
outcome: self_sim=0.5050 (var=0.0033) adversary_sim=0.4922 (var=0.0031) welch_t=+0.796 p=0.4302 (N=25, alpha=0.01) -> reject

[3] rationale: This metric isolates Trial 1, which pits an option (A) with fewer but highly valid features (validities 0.98 and 0.92) against an option (B) with more but lower-validity features (validities 0.65, 0.58, 0.52). The Tallying heuristic strictly counts the number of features, scoring B (3 features) over A (2 features), and thus predicts a choice proportion for A below 0.5. In contrast, the Weighted Additive (WADD) model weights features by their subjective validities. Because the validities of A's features are much higher, WADD generally scores A higher than B, predicting a choice proportion for A above 0.5. By simply measuring the choice proportion for Option A on this specific trial, we obtain a clean, robust divergence between the two models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_target = (1, 1, 0, 0, 0)
    b_target = (0, 0, 1, 1, 1)
    
    chosen_a = 0
    total = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        if a == a_target and b == b_target:
            total += 1
            if row['response'] == 0:
                chosen_a += 1
        elif a == b_target and b == a_target:
            total += 1
            if row['response'] == 1:
                chosen_a += 1
                
    if total == 0:
        return 0.5
        
    return chosen_a / total
outcome: self_sim=0.4300 (var=0.0359) adversary_sim=0.5000 (var=0.0125) welch_t=-1.590 p=0.1198 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    t1_a = (1, 1, 0, 0, 0)\n    t1_b = (0, 0, 1, 1, 1)\n    \n    t3_a = (0, 1, 1, 0, 0)\n    t3_b = (1, 0, 0, 1, 1)\n    \n    t1_choices = []\n    t3_choices = []\n    \n    for _, row in data.iterrows():\n        a = tuple(row['option_a_ratings'])\n        b = tuple(row['option_b_ratings'])\n        \n        if a == t1_a and b == t1_b:\n            t1_choices.append(1 if row['response'] == 0 else 0)\n        elif a == t1_b and b == t1_a:\n            t1_choices.append(1 if row['response'] == 1 else 0)\n            \n        elif a == t3_a and b == t3_b:\n            t3_choices.append(1 if row['response'] == 0 else 0)\n        elif a == t3_b and b == t3_a:\n            t3_choices.append(1 if row['response'] == 1 else 0)\n            \n    p_t1 = sum(t1_choices) / len(t1_choices) if t1_choices else 0.5\n    p_t3 = sum(t3_choices) / len(t3_choices) if t3_choices else 0.5\n    \n    return p_t1 - p_t3\n",
  "rationale": "This metric exploits a structural invariance in the Tallying model that the Weighted Additive (WADD) model breaks. On both Trial 1 and Trial 3, Option A has exactly 2 positive features and Option B has exactly 3 positive features. Because Tallying relies exclusively on the unweighted sum of features, it predicts the exact same choice probability for Option A across both trials. Consequently, the difference in choice proportions (P(A|Trial 1) - P(A|Trial 3)) will be 0 in expectation for Tallying subjects. In contrast, WADD weights features by their validities. Since the features comprising Option A in Trial 1 (validities 0.95, 0.85) are strictly superior to those in Trial 3 (validities 0.85, 0.75), WADD evaluates Option A much more favorably on Trial 1 than on Trial 3. Thus, WADD predicts a strictly positive difference. This within-subject contrast effectively cancels out baseline noise and isolates the validity-weighting mechanism."
}
```

## Usage

```json
{
  "prompt_token_count": 4891,
  "candidates_token_count": 658,
  "total_token_count": 14197
}
```
