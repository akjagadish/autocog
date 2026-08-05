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
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 5: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 7: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 8: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 9: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 10: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 11: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 12: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** To quantitatively dissociate the Mixture of Subjective WADD and Tallying (Advocated) from the Mixture of TTB and Tallying (Competing), this design uses 5 features with a linear spread of validities. The Competing theory restricts non-compensatory choices to a strict lexicographic rule (Take-The-Best) governed by objective validities, mixed with equal-weighting (Tallying). In contrast, the Advocated theory's WADD component uses free subjective weights, allowing it to capture compensatory choices or non-standard lexicographic orders that deviate from the objective validities. By including trials where Tallying predicts a tie but TTB rigidly prefers one option, and trials where Tallying favors one option while TTB strictly favors the other, we create a space of choices where human data often shows graded, compensatory trade-offs. The Competing model is forced to predict either strict TTB choices, Tallying choices, or a rigid linear mixture of the two, while the Advocated model can flexibly accommodate idiosyncratic weighting profiles through its subjective WADD component.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Mixture of Subjective WADD and Tallying: Decision-makers probabilistically mix between a simple equal-weight heuristic (Tallying) and a weighted additive rule that uses subjective, free-varying feature weights rather than objective validities. The subjective weights allow the WADD component to capture non-compensatory, Take-The-Best-like behavior (by assigning heavily skewed weights to features), while the Tallying component accounts for the strong equal-weighting pull observed when individuals fall back on simply counting positive features. Choice probabilities are a mixture of the softmax probabilities derived from each strategy, further blended with a uniform lapse rate to account for response errors. Increased choice noise bounds allow the model to better match human sub-optimal choice frequencies.

**Parameters:**
- beta: [0.01, 5.0]
- epsilon: [0.0, 1.0]
- w_tally: [0.0, 1.0]
- weights: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    w_tally = float(parameters["w_tally"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # --- Subjective WADD ---
    # Uses free subjective weights instead of objective validities
    scores_wadd = stim @ w
    z_wadd = beta * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
        
    # --- Tallying (Equal Weight) ---
    scores_tally = stim.sum(axis=1)
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # --- Mixture ---
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_wadd
    
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
**Description:** Mixture of Take-The-Best (TTB) and Tallying: Decision-makers probabilistically mix between two cognitively simple heuristics. Take-The-Best (TTB) is a strict lexicographic rule that inspects cues in order of their validity and chooses based entirely on the first discriminating cue. Tallying is an equal-weight rule that simply counts the number of positive features for each option. The mixture is biased towards Tallying to account for the strong equal-weighting pull observed in human data, while still allowing TTB to explain some non-compensatory choices. A small uniform lapse rate accounts for general response errors.

**Parameters:**
- beta: [0.01, 10.0]
- epsilon: [0.0, 0.3]
- w_tally: [0.5, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    w_tally = float(parameters["w_tally"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # --- Take-The-Best (TTB) ---
    # Order cues by validity (descending)
    cue_order = np.argsort(val)[::-1]
    diff = stim[0] - stim[1]
    
    p_ttb = np.array([0.5, 0.5])
    for cue in cue_order:
        if diff[cue] > 0:
            p_ttb = np.array([1.0, 0.0])
            break
        elif diff[cue] < 0:
            p_ttb = np.array([0.0, 1.0])
            break
            
    # --- Tallying (Equal Weight) ---
    scores_tally = stim.sum(axis=1)
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # --- Mixture ---
    p_core = w_tally * p_tally + (1.0 - w_tally) * p_ttb
    
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
[0] rationale: In Trials 8, 9, and 10, Option A always has 3 lower-validity cues and Option B always has 1 higher-validity cue. For the Competing theory (TTB + Tallying), TTB always chooses B and Tallying always chooses A. Thus, the Competing theory predicts the exact same choice probability for A across all three trials, resulting in near-zero variance within each subject (save for sampling noise). In contrast, the Advocated theory (Subjective WADD + Tallying) uses free subjective weights, meaning the sum of weights for A's specific cues will vary across these trials. This breaks the symmetry, leading to a significantly higher variance in the choice probabilities across these three trials for each subject.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trials 8, 9, and 10 all feature A having 3 positive cues and B having 1 positive cue.
    # In all three trials, TTB strictly prefers B (because B's single cue is more valid than any of A's),
    # and Tallying strictly prefers A (3 vs 1). 
    # Because the Competing model mixes TTB and Tallying, its predicted choice probability 
    # is identical across all three trials.
    # The Advocated model uses Subjective WADD, which assigns random idiosyncratic weights to cues.
    # Thus, it will evaluate A differently across these three trials, breaking the symmetry.
    target_trials = [
        ('01110', '10000'),
        ('00111', '01000'),
        ('00111', '10000')
    ]
    
    def subj_var(df):
        means = []
        for a, b in target_trials:
            mask = (df['a_str'] == a) & (df['b_str'] == b)
            if mask.sum() > 0:
                means.append((df.loc[mask, 'response'] == 0).mean())
        if len(means) > 1:
            return np.var(means)
        return 0.0
        
    return float(data.groupby('subject_id').apply(subj_var).mean())
outcome: self_sim=0.0173 (var=0.0003) adversary_sim=0.0149 (var=0.0003) welch_t=+0.533 p=0.5968 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))\n    \n    # Trials where Tallying ties (equal number of positive cues) \n    # and Take-The-Best strictly prefers Option A.\n    target_trials = [\n        ('10010', '01100'), # Trial 2\n        ('10001', '01010'), # Trial 3\n        ('01001', '00110'), # Trial 5\n        ('10000', '01000')  # Trial 6\n    ]\n    \n    def subj_score(df):\n        scores = []\n        for a, b in target_trials:\n            mask1 = (df['a_str'] == a) & (df['b_str'] == b)\n            mask2 = (df['a_str'] == b) & (df['b_str'] == a)\n            \n            responses = []\n            if mask1.any():\n                responses.extend((df.loc[mask1, 'response'] == 0).tolist())\n            if mask2.any():\n                responses.extend((df.loc[mask2, 'response'] == 1).tolist())\n                \n            if responses:\n                scores.append(np.mean(responses))\n                \n        if len(scores) == 0:\n            return np.nan\n        return np.mean(scores)\n        \n    return float(data.groupby('subject_id').apply(subj_score).mean())",
  "rationale": "To cleanly discriminate the theories, we isolate trials where Tallying predicts a tie (Trials 2, 3, 5, 6). Because both options have the same number of positive cues, Tallying's choice probability is exactly 50% regardless of the softmax temperature (beta). In the Competing theory, the Take-The-Best (TTB) component consistently breaks these ties in favor of Option A (which always possesses the highest valid discriminating cue). Consequently, the Competing theory predicts a robust aggregate preference for A (>50%). In contrast, the Advocated theory breaks these ties using Subjective WADD with random idiosyncratic weights. Across these four trials, the random weights favor A and B equally. Averaging the choice proportion of A across these specific trials yields a metric that is strictly ~50% for the Advocated theory (with low between-subject variance due to averaging across different cue combinations) and significantly >50% for the Competing theory."
}
```

## Usage

```json
{
  "prompt_token_count": 4324,
  "candidates_token_count": 679,
  "total_token_count": 26654
}
```
