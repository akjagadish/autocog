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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 2: A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 4: A=[0, 0, 1, 0, 1]  B=[0, 0, 0, 1, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Probabilistic Tallying) from the Competing Theory (Mixture of Deterministic Tallying + Probabilistic WADD), this design holds the Tallying score difference constant across multiple trials while varying the WADD score difference. Under the Advocated Theory, choice probabilities are entirely determined by the difference in the number of winning cues (the Tally difference). Thus, any two trials where Option A wins on 2 cues and Option B wins on 1 cue will yield the exact same choice probability. Under the Competing Theory, Tallying acts as a deterministic baseline, but the final choice probability is heavily modulated by the WADD score difference. By constructing pairs that all have a Tally difference of +1 for Option A but widely varying WADD differences (e.g., +1.0 vs +0.2), the Advocated Theory predicts a flat consistency curve across these pairs, whereas the Competing Theory predicts varying choice probabilities. We also include 'Tally tie' trials where the Advocated Theory predicts exactly 50% guessing, but the Competing Theory predicts a preference driven by WADD.

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
**Description:** Strategy Selection (Mixture of Deterministic Tallying and Probabilistic WADD): Decision-makers probabilistically select between a frugal, unweighted strategy (Tallying) and a fully compensatory, validity-weighted strategy (WADD). Critically, Tallying operates as a deterministic rule (choosing the option with more winning cues, or guessing on ties) rather than a probabilistic score-based process. This breaks the assumption that Tallying consistency scales with the absolute difference in cue counts, allowing the model to capture high consistency in scenarios with small cue count differences (e.g., Exp 4) and lower consistency in scenarios with large cue count differences (e.g., Exp 2). The WADD strategy remains probabilistic and tempers the extremeness of the Tallying predictions.

**Parameters:**
- beta: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- w_tally: [0.5, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")

    a, b = stim[0], stim[1]
    
    # 1. Deterministic Tallying strategy
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    if a_wins > b_wins:
        p_tally = np.array([1.0, 0.0])
    elif b_wins > a_wins:
        p_tally = np.array([0.0, 1.0])
    else:
        p_tally = np.array([0.5, 0.5])
        
    # 2. WADD strategy scores
    val = np.asarray(parameters["validities"], dtype=float)
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    scores_wadd = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_tally = float(parameters["w_tally"])
    
    # Softmax for WADD
    z_wadd = beta * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # Mixture of the two strategies
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_wadd
    
    # Apply lapse rate
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    val = np.array([0.9, 0.8, 0.7, 0.6, 0.5])\n    \n    tally_diffs = []\n    wadd_diffs = []\n    chose_A = []\n    \n    for _, row in data.iterrows():\n        a = np.array(row['option_a_ratings'])\n        b = np.array(row['option_b_ratings'])\n        \n        t_diff = np.sum(a > b) - np.sum(b > a)\n        w_diff = np.round(np.dot(a, val) - np.dot(b, val), 5)\n        \n        tally_diffs.append(t_diff)\n        wadd_diffs.append(w_diff)\n        chose_A.append(1 if row['response'] == 0 else 0)\n        \n    df = pd.DataFrame({\n        'tally_diff': tally_diffs,\n        'wadd_diff': wadd_diffs,\n        'chose_A': chose_A\n    })\n    \n    diff_t1 = 0.0\n    df_t1 = df[df['tally_diff'] == 1]\n    if len(df_t1) > 0:\n        grp = df_t1.groupby('wadd_diff')['chose_A'].mean()\n        if len(grp) > 1:\n            diff_t1 = grp.iloc[-1] - grp.iloc[0]\n            \n    diff_t0 = 0.0\n    df_t0 = df[df['tally_diff'] == 0]\n    if len(df_t0) > 0:\n        grp = df_t0.groupby('wadd_diff')['chose_A'].mean()\n        if len(grp) > 1:\n            diff_t0 = grp.iloc[-1] - grp.iloc[0]\n            \n    return float(diff_t1 + diff_t0)",
  "rationale": "This metric evaluates the degree to which choices are sensitive to the Weighted Additive (WADD) score difference when the Tally score difference is held constant. The Advocated Theory (Probabilistic Tallying) predicts that choice probabilities depend entirely on the difference in Tally scores, meaning the probability of choosing Option A should be identical across trials with the same Tally difference (e.g., Tally diff = +1 or Tally diff = 0), yielding a metric near 0. The Competing Theory (Strategy Selection) posits a mixture of Deterministic Tallying and Probabilistic WADD, meaning the choice probability will covary positively with the underlying WADD score difference even when the Tally difference is held constant. This metric computes the change in the rate of choosing Option A from the minimum to the maximum WADD difference within the Tally diff = +1 and Tally diff = 0 conditions, respectively, and sums them."
}
```

## Usage

```json
{
  "prompt_token_count": 3593,
  "candidates_token_count": 740,
  "total_token_count": 7644
}
```
