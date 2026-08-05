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

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 2: A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 6: A=[0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 7: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1]
  trial 8: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]

**Rationale:** To robustly dissociate a pure Tallying model from a Tallying/Take-The-Best (TTB) mixture model, we utilize a 5-feature design and manipulate the alignment between the two heuristics. Pure Tallying choices depend strictly on the difference in the total count of positive features between the options. Crucially, on trials where the feature counts are tied, pure Tallying predicts exactly 50/50 guessing. In contrast, the mixture model predicts a systematic deviation from 50/50 towards the option favored by TTB (the one with the highest-validity discriminating cue). Furthermore, on trials where Tallying favors one option but TTB favors the other, pure Tallying predicts the same choice probability as it would if TTB agreed, whereas the mixture model predicts an attenuated choice probability due to the conflicting TTB component.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Decision-makers probabilistically sample between a compensatory Equal-Weight (Tallying) heuristic and a non-compensatory Lexicographic (Take-The-Best) heuristic on a trial-by-trial basis, but they heavily favor Tallying. Furthermore, when they do use Take-The-Best, the application of the rule is subject to decision noise (modeled via a softmax temperature), reflecting uncertainty or stochasticity in identifying the most valid cue rather than a perfectly deterministic lexicographic choice.

**Parameters:**
- beta: [0.01, 5.0]
- beta_ttb: [0.01, 5.0]
- w_tally: [0.5, 1.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Tallying: sum the unweighted feature values for each option
    scores_tally = stim.sum(axis=1)
    beta = float(parameters["beta"])
    z = beta * (scores_tally - scores_tally.max())
    e = np.exp(z)
    p_tally = e / e.sum()
    
    # Take-The-Best (TTB): find the first discriminating cue ordered by validity
    order = np.argsort(validities)[::-1]
    diff = stim[0, order] - stim[1, order]
    non_zero = np.where(diff != 0)[0]
    
    p_ttb = np.array([0.5, 0.5])
    if len(non_zero) > 0:
        first_diff = diff[non_zero[0]]
        score_ttb = np.array([1.0, 0.0]) if first_diff > 0 else np.array([0.0, 1.0])
        beta_ttb = float(parameters["beta_ttb"])
        z_ttb = beta_ttb * score_ttb
        e_ttb = np.exp(z_ttb - np.max(z_ttb))
        p_ttb = e_ttb / e_ttb.sum()
            
    # Mixture of Tallying and TTB
    w_tally = float(parameters["w_tally"])
    p_mix = w_tally * p_tally + (1.0 - w_tally) * p_ttb
    
    # Global lapse rate
    epsilon = float(parameters["epsilon"])
    n_opts = len(p_mix)
    p_final = (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
    
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
[0] rationale: This metric isolates trials where the Tallying heuristic is completely indifferent (the sum of features is equal for both options). On these tie trials, the pure Tallying model predicts exactly 50/50 guessing, whereas the Mixture model predicts a systematic deviation toward the option favored by the Take-The-Best (TTB) heuristic. By calculating the proportion of choices that align with TTB on these specific tie trials, we can directly measure the presence of the TTB heuristic, strongly discriminating the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # Check for Tallying tie (equal number of positive features)
        if np.sum(a) == np.sum(b):
            diff = a - b
            non_zero = np.where(diff != 0)[0]
            if len(non_zero) > 0:
                first_diff = diff[non_zero[0]]
                ttb_favors_a = (first_diff > 0)
                chose_a = (row['response'] == 0)
                
                if chose_a == ttb_favors_a:
                    ttb_match.append(1.0)
                else:
                    ttb_match.append(0.0)
                    
    if len(ttb_match) == 0:
        return 0.5
    return float(np.mean(ttb_match))
outcome: self_sim=0.5600 (var=0.0170) adversary_sim=0.5108 (var=0.0091) welch_t=+1.523 p=0.1348 (N=25, alpha=0.01) -> reject

[1] rationale: By focusing on trials where the Tallying heuristic favors one option over another, we can categorize them into two groups: trials where Take-The-Best (TTB) agrees with Tallying, and trials where TTB opposes Tallying. For a pure Tallying model, the probability of choosing the Tally-favored option depends strictly on the difference in the sum of features, meaning it predicts no difference in choice probability whether TTB agrees or opposes. In contrast, the Mixture model incorporates a TTB component that increases the likelihood of choosing the Tally-favored option when both heuristics agree, and decreases it when they oppose. Thus, calculating the difference in the proportion of Tally-favored choices between these two sets of trials will yield exactly 0 for pure Tallying, and a strictly positive value for the Mixture model.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    agree_tally_choices = []
    oppose_tally_choices = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        
        # Skip Tallying tie trials
        if sum_a == sum_b:
            continue
            
        diff = a - b
        non_zero = np.where(diff != 0)[0]
        ttb_favors_a = diff[non_zero[0]] > 0
        
        tally_favors_a = sum_a > sum_b
        
        chose_a = (row['response'] == 0)
        chose_tally_favored = chose_a if tally_favors_a else not chose_a
        
        if tally_favors_a == ttb_favors_a:
            agree_tally_choices.append(1.0 if chose_tally_favored else 0.0)
        else:
            oppose_tally_choices.append(1.0 if chose_tally_favored else 0.0)
            
    if len(agree_tally_choices) == 0 or len(oppose_tally_choices) == 0:
        return 0.0
        
    return float(np.mean(agree_tally_choices) - np.mean(oppose_tally_choices))
outcome: self_sim=0.0939 (var=0.0222) adversary_sim=0.0189 (var=0.0099) welch_t=+2.094 p=0.04239 (N=25, alpha=0.01) -> reject

[2] rationale: The experimental design is perfectly balanced such that a pure Tallying model (which ignores validities and relies only on feature sums) has an expected Take-The-Best (TTB) match rate of exactly 50% across the full set of trials. The trials where Tallying opposes TTB perfectly cancel out the trials where Tallying agrees with TTB. Therefore, for pure Tallying, the overall proportion of choices matching the TTB prediction will be exactly 0.5 in expectation, with very low between-subject variance (only irreducible binomial noise). In contrast, the Mixture model explicitly incorporates the TTB heuristic, meaning its overall TTB match rate will be systematically higher than 50%. By calculating the overall proportion of TTB-consistent choices across all 96 trials, we maximize the sample size per subject, minimizing binomial variance and cleanly discriminating the two models with high statistical power.
metric_source:
def metric(data: pd.DataFrame) -> float:
    matches = 0
    total = len(data)
    if total == 0:
        return 0.5
        
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB prediction: find the first feature where they differ
        # (Validities are strictly decreasing from index 0 to 4)
        ttb_favors_a = True
        for v_a, v_b in zip(a, b):
            if v_a != v_b:
                ttb_favors_a = (v_a > v_b)
                break
                
        chose_a = (row['response'] == 0)
        if chose_a == ttb_favors_a:
            matches += 1
            
    return float(matches / total)
outcome: self_sim=0.5502 (var=0.0056) adversary_sim=0.5098 (var=0.0021) welch_t=+2.302 p=0.02668 (N=25, alpha=0.01) -> reject

[3] rationale: Pure Tallying relies exclusively on the difference in feature sums. On tie trials (sum difference = 0), it predicts exactly 50% guessing. On 'oppose' trials where Tallying and Take-The-Best (TTB) disagree (sum difference > 0), pure Tallying predicts a choice probability strictly greater than 50% for the Tally-favored option. Thus, for pure Tallying, the TTB match rate on tie trials (0.5) minus the Tally match rate on oppose trials (>0.5) is strictly and strongly negative. In contrast, the Mixture model incorporates TTB. On tie trials, TTB breaks the tie, pushing the TTB match rate well above 50%. On oppose trials, TTB pulls the choice probability away from the Tally-favored option, significantly depressing the Tally match rate. Consequently, for the Mixture model, this difference is near zero or even positive. This qualitative divergence (a strict negative vs. near zero/positive value) reliably discriminates the two theories with high statistical power.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tie_ttb = []
    oppose_tally = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        
        diff = a - b
        non_zero = np.where(diff != 0)[0]
        if len(non_zero) == 0:
            continue
            
        # Features are ordered by decreasing validity, so the first non-zero difference is the TTB cue
        ttb_favors_a = (diff[non_zero[0]] > 0)
        chose_a = (row['response'] == 0)
        
        if sum_a == sum_b:
            # Tie trials: measure how often choices align with TTB
            tie_ttb.append(1.0 if chose_a == ttb_favors_a else 0.0)
        else:
            tally_favors_a = (sum_a > sum_b)
            if tally_favors_a != ttb_favors_a:
                # Oppose trials: measure how often choices align with Tallying
                oppose_tally.append(1.0 if chose_a == tally_favors_a else 0.0)
                
    mean_tie = float(np.mean(tie_ttb)) if tie_ttb else 0.5
    mean_oppose = float(np.mean(oppose_tally)) if oppose_tally else 0.5
    
    return mean_tie - mean_oppose
outcome: self_sim=-0.0361 (var=0.0448) adversary_sim=-0.0908 (var=0.0169) welch_t=+1.102 p=0.2773 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    agree_tally = []\n    oppose_tally = []\n    \n    for _, row in data.iterrows():\n        a = np.array(row['option_a_ratings'])\n        b = np.array(row['option_b_ratings'])\n        \n        sum_a = np.sum(a)\n        sum_b = np.sum(b)\n        \n        # Isolate trials where the Tallying sum difference is exactly 1\n        if abs(sum_a - sum_b) == 1:\n            diff = a - b\n            non_zero = np.where(diff != 0)[0]\n            if len(non_zero) == 0:\n                continue\n                \n            ttb_favors_a = (diff[non_zero[0]] > 0)\n            tally_favors_a = (sum_a > sum_b)\n            \n            chose_a = (row['response'] == 0)\n            chose_tally = (chose_a == tally_favors_a)\n            \n            if tally_favors_a == ttb_favors_a:\n                agree_tally.append(1.0 if chose_tally else 0.0)\n            else:\n                oppose_tally.append(1.0 if chose_tally else 0.0)\n                \n    if not agree_tally or not oppose_tally:\n        return 0.0\n        \n    return float(np.mean(agree_tally) - np.mean(oppose_tally))",
  "rationale": "This metric isolates only the trials where the difference in feature sums is exactly 1. By holding the Tallying difference constant, we perfectly control for the compensatory signal. We then split these trials into those where Take-The-Best (TTB) agrees with Tallying, and those where TTB opposes it. For a pure Tallying model, the choice probability depends strictly on the sum difference, meaning the Tally match rate will be identical across both sets of trials (yielding an expected difference of exactly 0). In contrast, the Mixture model will exhibit a significantly higher Tally match rate when TTB agrees than when it opposes, yielding a robustly positive difference."
}
```

## Usage

```json
{
  "prompt_token_count": 5670,
  "candidates_token_count": 562,
  "total_token_count": 10025
}
```
