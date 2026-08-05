# metric_exp00_attempt_00

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
**Validities (n_features=5):** [0.85, 0.82, 0.8, 0.6, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 4: A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** To maximally distinguish Theory 1 (Strategy Selection: mixture of Take The Best and Tallying) from Theory 2 (Weighted Additive Model), this design uses a specific set of validities where the top cue is strong, but the next two cues combined strongly outweigh it in log-odds, while the bottom two cues have very low log-odds weights. In key trials (e.g., Trial 1), Option A possesses the single best cue and the two worst cues, giving it a Tallying advantage (3 winning cues to 2) and the TTB advantage. Theory 1 will thus unequivocally prefer Option A, regardless of its mixture parameter. However, Option B possesses the second and third best cues, whose combined log-odds weight heavily exceeds Option A's total, causing Theory 2 (WADD) to unequivocally prefer Option B. Other trials vary the tally difference and cue distribution to further map out the mixture dynamics of Theory 1 against WADD's compensatory integration.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Strategy Selection (Mixed Heuristics): Decision-makers probabilistically sample between a non-compensatory heuristic (Take The Best) and a simple compensatory heuristic (Tallying) on a trial-by-trial basis. The probability of using Tallying increases with the ease of the choice, defined by the absolute difference in the number of features favoring each option. By restricting the sensitivity parameter of this mixture, decision-makers preserve a baseline probability of using TTB even when Tallying discriminates, matching empirical reliance on dominant cues while pulling highly conflicting trials toward chance.

**Parameters:**
- beta: [0.1, 10.0]
- epsilon: [0.0, 1.0]
- gamma: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take The Best (TTB) score
    ttb_score = np.array([0.5, 0.5])
    for j in cue_order:
        if a[j] > b[j]:
            ttb_score = np.array([1.0, 0.0])
            break
        elif b[j] > a[j]:
            ttb_score = np.array([0.0, 1.0])
            break
            
    # Tallying (Equal-Weights) score
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    if a_wins > b_wins:
        tally_score = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        tally_score = np.array([0.0, 1.0])
    else:
        tally_score = np.array([0.5, 0.5])
        
    # Difficulty defined by tally difference
    diff = abs(a_wins - b_wins)
    
    # Probability of using Tallying over TTB
    gamma = float(parameters["gamma"])
    w_tally = 1.0 - np.exp(-gamma * diff)
    w_ttb = 1.0 - w_tally
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for TTB
    z_ttb = beta * ttb_score
    e_ttb = np.exp(z_ttb - np.max(z_ttb))
    p_ttb = e_ttb / np.sum(e_ttb)
    
    # Softmax for Tallying
    z_tally = beta * tally_score
    e_tally = np.exp(z_tally - np.max(z_tally))
    p_tally = e_tally / np.sum(e_tally)
    
    # Mixture of probabilities
    p_core = w_ttb * p_ttb + w_tally * p_tally
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
**Description:** Weighted Additive Model (WADD). Decision-makers integrate all available information by computing a sum of each option's features weighted by their respective validities, transformed into log-odds. The option with the higher weighted sum is chosen, providing a fully compensatory decision rule. Response noise enters through a softmax over the weighted sums with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick.

**Parameters:**
- beta: [0.0, 20.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Convert validities to log-odds weights to act as normative Bayesian evidence
    val_clipped = np.clip(val, 0.5001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    scores = np.array([np.sum(a * weights), np.sum(b * weights)])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over weighted sums with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    wadd_consistent = 0\n    relevant_trials = 0\n    for _, row in data.iterrows():\n        a = row['option_a_ratings']\n        b = row['option_b_ratings']\n        # Check if one option has [1, 0, 0] on the first 3 features and the other has [0, 1, 1]\n        a_top3 = [a[0], a[1], a[2]]\n        b_top3 = [b[0], b[1], b[2]]\n        if a_top3 == [1, 0, 0] and b_top3 == [0, 1, 1]:\n            relevant_trials += 1\n            if row['response'] == 1:\n                wadd_consistent += 1\n        elif a_top3 == [0, 1, 1] and b_top3 == [1, 0, 0]:\n            relevant_trials += 1\n            if row['response'] == 0:\n                wadd_consistent += 1\n    if relevant_trials == 0:\n        return 0.5\n    return wadd_consistent / relevant_trials",
  "rationale": "This metric calculates the proportion of times the subject chooses the option favored by the Weighted Additive Model (WADD) on trials where the top cue (feature 0) is pitted against the second and third best cues (features 1 and 2). In these specific trials, Theory 1 (Strategy Selection) strongly favors the option with the top cue because it wins both Take-The-Best and Tallying (having more total positive cues). In contrast, Theory 2 (WADD) strongly favors the option with the second and third cues, because their combined log-odds weights exceed that of the single top cue. The metric isolates these trials and will yield values close to 0 for Theory 1 and close to 1 for Theory 2."
}
```

## Usage

```json
{
  "prompt_token_count": 3444,
  "candidates_token_count": 473,
  "total_token_count": 5553
}
```
