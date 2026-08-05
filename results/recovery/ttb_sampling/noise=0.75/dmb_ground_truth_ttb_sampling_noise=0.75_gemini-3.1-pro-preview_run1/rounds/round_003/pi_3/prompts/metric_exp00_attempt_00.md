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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=9):**
  trial 1: A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 5: A=[1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1]
  trial 6: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 7: A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 8: A=[1, 1, 1, 0, 1]  B=[1, 1, 0, 1, 1]
  trial 9: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]

**Rationale:** To quantitatively dissociate TTB with Probabilistic Stopping (advocated) from Probabilistic Cue Search (competing), this design holds the highest-validity discriminating cue constant across subsets of trials while varying the number and distribution of lower-validity discriminating cues. Under the advocated TTB model, choice probability depends exclusively on the validity of the first (highest-validity) discriminating cue; thus, it predicts identical choice probabilities across all trials sharing the same top discriminating cue, regardless of other cue values. In contrast, the competing Probabilistic Cue Search model samples all discriminating cues proportionally to their validities. Therefore, adding lower-validity cues that favor the opposing option will systematically pull the choice probability away from the top cue's favored option under PCS, while leaving the TTB prediction completely unaffected.

**Computed schedule:** 9 unique pairs × 10 reps = 90 trials per subject.



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
**Description:** Probabilistic Cue Search: Decision-makers rely on a strict one-reason heuristic but search through cues probabilistically rather than in a fixed deterministic order. The probability of inspecting a cue is proportional to its subjective validity (via a softmax function). The decision-maker inspects cues one by one without replacement until a discriminating cue is found, which then deterministically dictates the choice. Mathematically (via the Independence of Irrelevant Alternatives in the Plackett-Luce model), the probability that a specific discriminating cue is the first one encountered is exactly its softmax weight divided by the sum of the softmax weights of all discriminating cues. This preserves the non-compensatory nature of the process while introducing cognitively plausible noise through stochastic attention and memory retrieval.

**Parameters:**
- beta: [1.0, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Identify which cues discriminate between the two options
    disc = (a != b)
    
    if not np.any(disc):
        # If no cues discriminate, guess randomly
        p_core = np.array([0.5, 0.5])
    else:
        # Compute softmax weights for discriminating cues
        w = np.zeros_like(validities)
        max_v = np.max(validities[disc])
        w[disc] = np.exp(beta * (validities[disc] - max_v))
        
        sum_w = np.sum(w)
        
        # The probability of choosing an option is the sum of the weights 
        # of the discriminating cues that favor it, normalized by the sum
        # of weights of all discriminating cues.
        p_a = np.sum(w[a > b]) / sum_w
        p_b = np.sum(w[b > a]) / sum_w
        
        p_core = np.array([p_a, p_b])
        
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    data = data.copy()\n    data['A_tuple'] = data['option_a_ratings'].apply(tuple)\n    data['B_tuple'] = data['option_b_ratings'].apply(tuple)\n    data['chose_A'] = (data['response'] == 0).astype(float)\n    \n    t1_a, t1_b = (1, 0, 1, 1, 1), (0, 1, 1, 1, 1)\n    t3_a, t3_b = (1, 0, 0, 0, 0), (0, 1, 1, 1, 1)\n    \n    t5_a, t5_b = (1, 1, 0, 1, 1), (1, 0, 1, 1, 1)\n    t6_a, t6_b = (1, 1, 0, 0, 0), (1, 0, 1, 1, 1)\n    \n    t8_a, t8_b = (1, 1, 1, 0, 1), (1, 1, 0, 1, 1)\n    t9_a, t9_b = (1, 1, 1, 0, 0), (1, 1, 0, 1, 1)\n    \n    def get_pa(a, b):\n        mask = (data['A_tuple'] == a) & (data['B_tuple'] == b)\n        if mask.sum() == 0:\n            return 0.5\n        return data.loc[mask, 'chose_A'].mean()\n        \n    diff1 = get_pa(t1_a, t1_b) - get_pa(t3_a, t3_b)\n    diff2 = get_pa(t5_a, t5_b) - get_pa(t6_a, t6_b)\n    diff3 = get_pa(t8_a, t8_b) - get_pa(t9_a, t9_b)\n    \n    return float(diff1 + diff2 + diff3)",
  "rationale": "This metric calculates the drop in the probability of choosing Option A when additional lower-validity cues are introduced that favor Option B, while the highest-validity discriminating cue (which favors Option A) remains constant. Under the advocated Take The Best (TTB) model, choice probability depends only on the highest-validity discriminating cue, so adding lower-validity cues favoring B has zero effect (expected metric ~ 0). Under the competing Probabilistic Cue Search (PCS) model, all discriminating cues are sampled proportionally to their validity, so adding cues favoring B substantially pulls the choice probability toward B (expected metric > 0)."
}
```

## Usage

```json
{
  "prompt_token_count": 3433,
  "candidates_token_count": 697,
  "total_token_count": 6707
}
```
