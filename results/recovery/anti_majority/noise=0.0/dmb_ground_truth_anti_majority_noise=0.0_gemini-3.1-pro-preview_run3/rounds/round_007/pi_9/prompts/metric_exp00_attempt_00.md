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
**Validities (n_features=5):** [0.6, 0.7, 0.95, 0.8, 0.65]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 2: A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** This design creates a strict qualitative dissociation between the Recency-Biased Evidence Accumulation (Advocated) model and the Conflict-Driven Strategy Selection (Competing) model. The Competing model relies on 'diff_cues' (absolute difference in total positive cues) to arbitrate between Take The Best (TTB) and Tallying. Crucially, when diff_cues == 0, the Competing model relies 100% on Tallying. Because the features are binary, any trial where sum(a) == sum(b) mathematically guarantees a tied Tallying score, forcing the Competing model to predict exactly 50/50 regardless of the cues' validities. The Advocated model, however, completely ignores instructed validities and weights cues exponentially based on their left-to-right sequential position. For trials with diff_cues == 0 (e.g., A has the outermost cues, B has the inner cues), the Advocated model predicts strong preferences depending on its primacy/recency parameter (lambda). Furthermore, in high-conflict trials (large diff_cues), we pit the single highest-validity cue (placed in the middle of the array) against all other cues. The Competing model's TTB component will reliably pick the option with the highest-validity cue, whereas the Advocated model's exponential position-weighting mathematically guarantees that a middle cue can never outweigh the sum of the outer cues, forcing it to choose the opposite option.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



## ADVOCATED THEORY
**Description:** Recency-Biased Evidence Accumulation: Decision-makers process cue arrays sequentially (e.g., from left to right) and exhibit a recency or primacy effect, largely overriding instructed objective validities. The subjective weight of a cue grows or shrinks exponentially with its position in the sequence. By framing this exponential growth/decay factor on a log scale, the population symmetrically exhibits both strong primacy and extreme recency, allowing the model to capture diverse behavioral patterns across different experimental contexts.

**Parameters:**
- log_lambda: [-3.5, 2.5]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    # Use log_lambda to symmetrically balance primacy and recency
    log_lam = float(parameters["log_lambda"])
    lam = np.exp(log_lam)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Calculate position-biased weights based on sequential position
    # Position 0 is the first (leftmost) cue, n_features-1 is the last (rightmost)
    weights = np.array([lam ** j for j in range(n_features)])
    
    # Normalize weights to prevent overflow and keep scale consistent for beta
    weights = weights / np.sum(weights)
    
    # Accumulate evidence for each option
    ev_a = np.sum(a * weights)
    ev_b = np.sum(b * weights)
    
    scores = np.array([ev_a, ev_b])
    
    # Convert evidence to choice probabilities via softmax
    z = beta * scores
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.sum(p)
    
    # Incorporate decision noise (epsilon)
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## COMPETING THEORY
**Description:** Conflict-Driven Strategy Selection: Decision-makers adaptively select their decision strategy based on the dissimilarity of the options' total cue counts. When options are highly dissimilar in the number of positive cues (indicating high conflict or complexity), individuals abandon compensatory processing and fall back entirely on a simple non-compensatory heuristic (Take The Best). Conversely, when the total cue counts are similar, individuals attempt to integrate all available information using a compensatory strategy (Tallying). This is modeled as a probabilistic mixture of TTB and Tallying, where the probability of using TTB scales directly with the absolute difference in total cue counts.

**Parameters:**
- beta_ttb: [0.1, 10.0]
- beta_tally: [0.1, 10.0]
- gamma: [0.0, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np

    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take The Best (TTB) Strategy
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        scores_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        scores_ttb = np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.5, 0.5])
        
    # Tallying Strategy
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins]) / max(1.0, float(n_features))
    
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    
    # TTB Probabilities
    z_ttb = beta_ttb * scores_ttb
    e_ttb = np.exp(z_ttb - np.max(z_ttb))
    p_ttb_dist = e_ttb / e_ttb.sum()
    
    # Tallying Probabilities
    z_tally = beta_tally * scores_tally
    e_tally = np.exp(z_tally - np.max(z_tally))
    p_tally_dist = e_tally / e_tally.sum()
    
    # Conflict-Driven Weight
    gamma = float(parameters["gamma"])
    
    # Dissimilarity in total cue counts
    diff_cues = abs(np.sum(a) - np.sum(b))
    
    # Probability of using TTB increases linearly with diff_cues
    w_ttb = min(1.0, gamma * diff_cues / max(1.0, float(n_features)))
    
    epsilon = float(parameters["epsilon"])
    
    p_core = w_ttb * p_ttb_dist + (1.0 - w_ttb) * p_tally_dist
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    # Identify trials where the total number of cues is perfectly tied (diff_cues == 0) \n    # and the options are spatially symmetric (outer vs inner cues).\n    # Trial 3: A=[1, 0, 0, 0, 0], B=[0, 0, 0, 0, 1]\n    # Trial 4: A=[1, 1, 0, 0, 0], B=[0, 0, 0, 1, 1]\n    mask = data['option_a_ratings'].apply(tuple).isin([(1, 0, 0, 0, 0), (1, 1, 0, 0, 0)])\n    df_trial = data[mask]\n    if len(df_trial) == 0:\n        return 0.0\n    \n    # For the Competing model, diff_cues == 0 means 100% reliance on Tallying. \n    # Since the sum of cues is equal, Tallying predicts exactly 50/50, so subject means will be ~0.5.\n    # For the Advocated model, extreme primacy or recency will drive choices deterministically \n    # towards A or B, so subject means will be near 0.0 or 1.0.\n    # Measuring the absolute deviation from 0.5 captures this structural divergence.\n    subj_means = df_trial.groupby('subject_id')['response'].mean()\n    return float(np.mean(np.abs(subj_means - 0.5)))",
  "rationale": "In trials where both options have an equal number of positive cues (diff_cues == 0), the Competing model is forced to rely entirely on Tallying. Because the cue sums are tied, Tallying yields a perfect 50/50 choice probability, leading to subject choice rates around 0.5 (plus small binomial noise). In contrast, the Advocated model uses exponentially decaying/growing positional weights. On trials where cues are spatially segregated (e.g., A has the left-most cues, B has the right-most cues), the Advocated model will strongly prefer one option over the other depending on its lambda parameter (primacy vs. recency). Thus, subjects governed by the Advocated model will exhibit choice proportions near 0 or 1. By computing the mean absolute deviation of each subject's choice rate from 0.5 on these specific trials, we expect a value near 0 for the Competing model and a value near 0.5 for the Advocated model."
}
```

## Usage

```json
{
  "prompt_token_count": 3705,
  "candidates_token_count": 610,
  "total_token_count": 6242
}
```
