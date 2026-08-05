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

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 0, 0, 1]  B=[1, 1, 0, 1, 0]
  trial 5: A=[0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  trial 7: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 8: A=[1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Rationale:** To quantitatively dissociate TTB with Probabilistic Stopping from Strict TTB with Uniform Lapse, this design varies the validity of the first discriminating feature across trials. Under the competing Strict TTB model, the choice probability for the TTB-favored option is determined by a uniform lapse rate and thus remains constant across all trials, regardless of whether the deciding feature is highly valid or barely better than chance. In contrast, the advocated Probabilistic Stopping model scales the choice probability via a softmax function based on the validity of the discriminating feature. By creating a set of trials where the highest-validity discriminating feature systematically shifts from the most valid cue down to the least valid cue, we can observe whether subjects' choice consistency degrades as the deciding cue's validity decreases, which is uniquely predicted by the advocated model.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Take The Best (TTB) with Probabilistic Stopping: Decision-makers use a lexicographic heuristic, ranking features by subjective validity and stopping at the first discriminating feature. However, rather than making a strictly deterministic choice based on this feature, the decision is probabilistic. The probability of choosing the winning option scales with the validity of that discriminating feature via a softmax function with a highly regularized inverse temperature (beta). This allows confidence to vary depending on how valid the deciding feature is, capturing empirical noise without relying entirely on a global random lapse rate.

**Parameters:**
- beta: [0.0, 2.5]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    beta = float(parameters["beta"])
    
    a, b = stim[0], stim[1]
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.array([0.0, 0.0])
    
    # Iterate through sorted features to find the first discriminator
    for f in order:
        if a[f] > b[f]:
            scores = np.array([validities[f], 0.0])
            break
        elif b[f] > a[f]:
            scores = np.array([0.0, validities[f]])
            break
            
    # If no feature discriminates, default to uniform guessing
    if scores[0] == scores[1]:
        p_core = np.array([0.5, 0.5])
    else:
        # Probabilistic choice scaling with the validity of the discriminating feature
        z = beta * (scores - scores.max())
        e = np.exp(z)
        p_core = e / e.sum()
        
    # Apply lapse rate
    n_opts = 2
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Strict Take-The-Best with Uniform Lapse: Decision-makers rely on a strict lexicographic heuristic, ranking cues by subjective validity and making a deterministic choice based solely on the highest-validity discriminating cue. To account for empirical noise and inattention, decisions are subject to a uniform lapse rate, where the decision-maker simply guesses randomly on a fixed proportion of trials rather than scaling their confidence by the cue's validity.

**Parameters:**
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    
    # Rank features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    
    # Iterate through sorted features to find the first discriminator
    for f in order:
        if a[f] > b[f]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[f] > a[f]:
            p_core = np.array([0.0, 1.0])
            break
            
    # Apply uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
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
[0] rationale: The Probabilistic Stopping theory predicts that decision consistency scales with the validity of the first discriminating feature, meaning subjects will be more consistent with the TTB prediction when the deciding feature has high validity compared to low validity. In contrast, the Strict TTB theory assumes a uniform lapse rate, predicting equal consistency across all trials regardless of the deciding feature's validity. This metric calculates the difference in the proportion of TTB-consistent choices between trials decided by high-validity features (the first two) and trials decided by low-validity features (the last two). It should be near zero for Strict TTB and significantly positive for Probabilistic Stopping TTB.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    high_val_consistent = []
    low_val_consistent = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        disc_f = -1
        ttb_favors_b = -1
        for f in range(len(a)):
            if a[f] > b[f]:
                disc_f = f
                ttb_favors_b = 0
                break
            elif b[f] > a[f]:
                disc_f = f
                ttb_favors_b = 1
                break
                
        if disc_f == -1:
            continue
            
        is_consistent = 1.0 if resp == ttb_favors_b else 0.0
        
        if disc_f in [0, 1]:
            high_val_consistent.append(is_consistent)
        elif disc_f in [3, 4]:
            low_val_consistent.append(is_consistent)
            
    if not high_val_consistent or not low_val_consistent:
        return 0.0
        
    return float(np.mean(high_val_consistent) - np.mean(low_val_consistent))
outcome: self_sim=0.0279 (var=0.0146) adversary_sim=0.0108 (var=0.0090) welch_t=+0.556 p=0.5809 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    # Convert ratings to 2D numpy arrays\n    A = np.array(data['option_a_ratings'].tolist())\n    B = np.array(data['option_b_ratings'].tolist())\n    resp = data['response'].values\n    \n    # Find the first discriminating feature for each trial\n    # Validities are sorted descending, so the first feature with a difference is the deciding one\n    diff = A - B\n    is_diff = diff != 0\n    first_diff_idx = np.argmax(is_diff, axis=1)\n    has_diff = np.any(is_diff, axis=1)\n    \n    # Extract the sign of the first difference to determine TTB prediction\n    row_indices = np.arange(len(diff))\n    first_diff_sign = diff[row_indices, first_diff_idx]\n    \n    # TTB favors Option A (0) if A > B (sign == 1)\n    # TTB favors Option B (1) if B > A (sign == -1)\n    ttb_favors_b = (first_diff_sign == -1).astype(int)\n    \n    # Calculate consistency\n    consistent = (resp == ttb_favors_b)\n    \n    if not np.any(has_diff):\n        return 0.5\n        \n    return float(np.mean(consistent[has_diff]))",
  "rationale": "Strict TTB makes a deterministic core choice (100% confidence) based on the highest-validity discriminating cue, relying entirely on a uniform lapse rate to generate noise. In contrast, Probabilistic Stopping scales its core confidence by the validity of the deciding cue using a softmax function. Because the deciding cue's validity is always less than 1.0, Probabilistic Stopping's internal confidence is structurally diluted compared to Strict TTB. When subjected to the same uniform lapse rate distribution, this dilution results in a significantly lower overall proportion of TTB-consistent choices for Probabilistic Stopping. Thus, the overall TTB-consistency rate serves as a highly robust discriminator with minimal binomial variance."
}
```

## Usage

```json
{
  "prompt_token_count": 3746,
  "candidates_token_count": 519,
  "total_token_count": 12930
}
```
