# metric_exp00_attempt_01

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

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 7: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 8: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 9: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 10: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 11: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 12: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Rationale:** To maximally distinguish Take-The-Best (TTB) from Probabilistic Cue Search (PCS), we systematically vary the number and validity of lower-validity cues that oppose the highest-validity discriminating cue. TTB is strictly non-compensatory: its choice is determined entirely by the top discriminating cue, meaning any lower-validity cues favoring the alternative option are ignored and have no effect on choice probabilities (beyond fixed lapse/softmax noise). In contrast, PCS uses a Plackett-Luce choice rule over all discriminating cues. Thus, as more lower-validity cues stack up against the top cue, PCS predicts a graded decrease in the probability of choosing the option favored by the top cue. By including trials where the top cue is opposed by one, two, three, or four lower-validity cues, we create a strong quantitative dissociation between the step-function prediction of TTB and the graded-integration prediction of PCS.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Cue Search (PCS): Decision makers evaluate options by searching through cues sequentially. However, unlike the strict deterministic order of Take-The-Best, the search order is probabilistic. The probability of retrieving and evaluating a cue is proportional to an exponential function of its validity (a Plackett-Luce choice process). Because non-discriminating cues simply prompt the decision maker to continue searching, the probability that the decision is ultimately determined by a specific discriminating cue is exactly proportional to its weight relative to all other discriminating cues. This model elegantly bridges random search (Tallying) and deterministic search (TTB) through a single search-temperature parameter.

**Parameters:**
- gamma: [0.0, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Find discriminating cues
    diff = stim[0] - stim[1]
    discrim_mask = diff != 0
    
    if not np.any(discrim_mask):
        p_core = np.array([0.5, 0.5])
    else:
        # Due to the Plackett-Luce property of sampling without replacement,
        # the probability that a particular discriminating cue is encountered 
        # before any other discriminating cue is simply its proportional weight
        # among the set of discriminating cues. Non-discriminating cues drop out.
        discrim_validities = validities[discrim_mask]
        max_v = np.max(discrim_validities)
        
        # Calculate weights for discriminating cues with numerical stability
        weights = np.zeros_like(validities)
        weights[discrim_mask] = np.exp(gamma * (validities[discrim_mask] - max_v))
        
        w_A = np.sum(weights[diff > 0])
        w_B = np.sum(weights[diff < 0])
        
        total_w = w_A + w_B
        if total_w == 0:
            p_core = np.array([0.5, 0.5])
        else:
            p_core = np.array([w_A / total_w, w_B / total_w])
            
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
**Description:** People compare two options using the Take-The-Best (TTB) heuristic. TTB is a fast-and-frugal lexicographic strategy that searches through cues in descending order of their validity. The first cue that discriminates between the two options completely determines the choice, and all remaining lower-validity cues are ignored. If no cue discriminates, the decision maker guesses randomly. Response noise enters through a softmax over the binary TTB outcome with inverse temperature beta, plus an independent lapse rate epsilon.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.zeros(2)
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            scores[0] = 1.0
            break
        elif stim[1, idx] > stim[0, idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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
[0] rationale: This metric calculates the difference in the probability of choosing the option favored by the most valid discriminating cue when it is opposed by exactly 1 lower-validity cue versus when it is opposed by 3 or 4 lower-validity cues. Take-The-Best (TTB) makes a non-compensatory decision based solely on the top cue, so the number of opposing cues should not systematically affect choice probabilities, yielding a metric near 0. Probabilistic Cue Search (PCS) integrates over all cues, meaning an accumulation of opposing cues will substantially reduce the probability of choosing the top-cue-favored option, resulting in a significantly positive metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    t1_choices = []
    t3_choices = []
    
    for _, row in data.iterrows():
        A = row['option_a_ratings']
        B = row['option_b_ratings']
        
        top_favors_A = None
        top_idx = -1
        for i in range(len(A)):
            if A[i] != B[i]:
                top_favors_A = (A[i] > B[i])
                top_idx = i
                break
                
        if top_favors_A is None:
            continue
            
        if top_favors_A:
            n_opposing = sum(1 for j in range(top_idx+1, len(A)) if B[j] > A[j])
            chosen_top = (row['response'] == 0)
        else:
            n_opposing = sum(1 for j in range(top_idx+1, len(A)) if A[j] > B[j])
            chosen_top = (row['response'] == 1)
            
        if n_opposing == 1:
            t1_choices.append(chosen_top)
        elif n_opposing >= 3:
            t3_choices.append(chosen_top)
            
    if not t1_choices or not t3_choices:
        return 0.0
        
    return float(np.mean(t1_choices) - np.mean(t3_choices))
outcome: self_sim=0.0627 (var=0.0175) adversary_sim=0.0150 (var=0.0125) welch_t=+1.379 p=0.1745 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    compensatory_choices = []\n    \n    for _, row in data.iterrows():\n        A = row['option_a_ratings']\n        B = row['option_b_ratings']\n        \n        diff = np.array(A) - np.array(B)\n        discrim_mask = diff != 0\n        if not np.any(discrim_mask):\n            continue\n            \n        # Top cue is the first non-zero in diff (since validities are monotonically decreasing)\n        top_idx = np.where(discrim_mask)[0][0]\n        top_favors_A = (diff[top_idx] > 0)\n        \n        count_A = np.sum(diff > 0)\n        count_B = np.sum(diff < 0)\n        \n        # Identify trials where the top cue opposes the simple majority of discriminating cues\n        if top_favors_A and count_B > count_A:\n            chosen_majority = (row['response'] == 1)\n            compensatory_choices.append(chosen_majority)\n        elif (not top_favors_A) and count_A > count_B:\n            chosen_majority = (row['response'] == 0)\n            compensatory_choices.append(chosen_majority)\n            \n    if not compensatory_choices:\n        return 0.0\n        \n    return float(np.mean(compensatory_choices))",
  "rationale": "This metric calculates the proportion of times a subject chooses the option favored by the simple majority of discriminating cues, specifically on trials where this majority is opposed by the single most valid discriminating cue. Take-The-Best (TTB) is strictly non-compensatory and will virtually never choose the majority option in these trials (only doing so via random noise). Probabilistic Cue Search (PCS), however, evaluates cues probabilistically and will integrate the multiple opposing cues, leading to a significantly higher rate of choosing the majority option."
}
```

## Usage

```json
{
  "prompt_token_count": 4164,
  "candidates_token_count": 488,
  "total_token_count": 7780
}
```
