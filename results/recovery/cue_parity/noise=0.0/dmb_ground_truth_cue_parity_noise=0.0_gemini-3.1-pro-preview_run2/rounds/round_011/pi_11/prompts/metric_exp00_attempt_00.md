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
**Validities (n_features=5):** [1.0, 0.75, 0.65, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 7: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 8: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate the advocated WADD theory (with Zero-Anchored Soft Validity Transformation) from the competing Two-Stage Tally-Threshold Theory, we exploit their divergent responses to varying tally differences when pitted against a single highly valid cue. The competing theory proposes a discrete threshold (ranging from 1 to 4) for the tally difference; if the difference meets or exceeds this threshold, the decision-maker strictly follows the tally (ignoring validities), leading to a sharp, categorical reversal in choice. The advocated WADD theory applies an exponential transformation to the validities, meaning the highest validity cue can exponentially dominate multiple lower-validity cues regardless of the raw tally difference. By presenting trials where Option A possesses the single highest-validity cue but Option B has a progressively larger tally advantage (from 1 to 4), the competing theory predicts a step-function reversal to Option B once the threshold is crossed. WADD, in contrast, predicts a continuous, graded shift where Option A may remain strongly preferred even at large tally differences due to the exponential weight of its top cue. We also include tally-tied trials to dissociate the competing theory's fallback strategies (TTB vs. unweighted WADD) from the advocated theory's exponentially weighted WADD.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Weighted Additive (WADD) Integration with Zero-Anchored Soft Validity Transformation: Decision-makers compute a subjective value for each option by summing its features, weighted by a zero-anchored exponential transformation of their validities. This transformation (exp(gamma * val) - 1) ensures that non-predictive cues receive no weight, preventing the artificial inflation of tallies by low-validity cues while allowing the highest validity cues to exponentially dominate when necessary. This naturally bridges compensatory and non-compensatory decision-making without heuristic switching.

**Parameters:**
- gamma: [0.0, 10.0]
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Subjective transformation of validities
    # Subtracting 1.0 ensures that a zero-validity cue would receive exactly 0 weight,
    # preventing artificial inflation of low-validity cues and allowing the highest
    # validity cues to dominate when necessary.
    w = np.exp(gamma * val) - 1.0
    
    # Compute subjective values (Weighted Additive sum)
    v_a = np.sum(w * a)
    v_b = np.sum(w * b)
    
    scores = np.array([v_a, v_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Two-Stage Tally-Threshold Theory: Decision-makers evaluate options using a fast, unweighted tallying heuristic as a first pass. If the difference in the number of positive features between the options meets or exceeds a subjective threshold, the decision is made based on this tally difference. However, if the options are tied or the difference is too small to be discriminative (below the threshold), the decision-maker shifts to a second, more effortful stage. In this fallback stage, they evaluate the options using either a weighted additive (WADD) or a lexicographic (Take-The-Best) approach based on cue validities. Evidence scores are normalized across strategies to share a single sensitivity parameter.

**Parameters:**
- threshold: {1, 2, 3, 4}
- fallback: {"WADD", "TTB"}
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    threshold = int(parameters["threshold"])
    fallback = str(parameters["fallback"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    n_features = len(a)
    
    if abs(tally_a - tally_b) >= threshold:
        # Normalize tally scores to [0, 1] to calibrate with fallback scales
        scores = np.array([tally_a, tally_b]) / n_features
    else:
        if fallback == "WADD":
            # Normalize validities so WADD scores are in [0, 1]
            val_norm = val / np.sum(val)
            scores = np.array([np.sum(val_norm * a), np.sum(val_norm * b)])
        else:  # TTB
            cue_order = np.argsort(-val, kind="stable").tolist()
            winner = -1
            for j in cue_order:
                if a[j] > b[j]:
                    winner = 0
                    break
                elif b[j] > a[j]:
                    winner = 1
                    break
            if winner == 0:
                scores = np.array([1.0, 0.0])
            elif winner == 1:
                scores = np.array([0.0, 1.0])
            else:
                scores = np.array([0.0, 0.0])
                
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a = np.array(data['option_a_ratings'].tolist())\n    b = np.array(data['option_b_ratings'].tolist())\n    resp = data['response'].values\n    \n    sum_a = a.sum(axis=1)\n    sum_b = b.sum(axis=1)\n    \n    # HVO (High Validity Option) is the one with the best cue (index 0)\n    # We filter for trials where the HVO is outnumbered in simple tally\n    cond_a = (a[:, 0] == 1) & (b[:, 0] == 0) & (sum_a < sum_b)\n    cond_b = (b[:, 0] == 1) & (a[:, 0] == 0) & (sum_b < sum_a)\n    \n    hvo_chosen = np.sum((cond_a & (resp == 0)) | (cond_b & (resp == 1)))\n    total = np.sum(cond_a | cond_b)\n    \n    return float(hvo_chosen / total) if total > 0 else 0.5",
  "rationale": "This metric calculates the proportion of times a subject chooses the 'High Validity Option' (the option possessing the single most predictive cue) when it is strictly outnumbered by the competing option in terms of total positive features (tally). The Advocated WADD theory, which relies on a zero-anchored exponential transformation of validities, predicts that the single best cue will often exponentially dominate a larger tally of weaker cues, resulting in a high choice rate for the High Validity Option. In contrast, the Competing Two-Stage Tally-Threshold Theory prioritizes the tally difference first; it will only choose the High Validity Option if the tally difference happens to fall below the subject's threshold AND the subject uses a Take-The-Best fallback strategy. Consequently, the Advocated theory predicts a significantly higher mean on this metric compared to the Competing theory."
}
```

## Usage

```json
{
  "prompt_token_count": 3544,
  "candidates_token_count": 483,
  "total_token_count": 8345
}
```
