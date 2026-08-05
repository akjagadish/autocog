# metric_exp00_attempt_02

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

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 3: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Single-Cue Focus model (Advocated) from the Weak Weighted Additive model (Competing), we exploit the fact that the Advocated theory relies exclusively on the single highest-validity cue, completely ignoring all other features, while the Competing theory integrates all features via a weighted sum. We design 'compensatory' trials where the highest-validity cue points to Option A, but the sum of the remaining lower-validity cues strongly favors Option B. The Advocated theory predicts a preference for Option A, whereas the Competing theory predicts a preference for Option B. Crucially, we also include 'tie-breaker' trials where both options share the exact same value for the highest-validity cue. On these trials, the Advocated theory is completely blind to any differences and predicts an exact 50/50 random guess, whereas the Competing theory still predicts a clear preference driven by the remaining cues.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



## ADVOCATED THEORY
**Description:** Extreme Cognitive Noise / Single-Cue Focus: In complex multi-cue decision environments, subjects attempt to rely on the provided cue validities but are quickly overwhelmed by cognitive friction. Instead of integrating multiple cues (like Weighted Additive) or falling back to simple counting (Tallying), they occasionally fixate solely on the single most valid cue to make their decision. However, this fragile single-cue heuristic is heavily masked by an overwhelmingly high baseline guessing rate (lapse), meaning that on the vast majority of trials, subjects simply guess randomly. This explains why choice behavior hovers very close to 0.5 across various conflict and agreement metrics, while allowing for slight, systematic deviations driven by the top validity cue rather than overall cue counts.

**Parameters:**
- beta: [0.1, 10.0]
- epsilon: [0.9, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Identify the single highest-validity cue
    best_cue = np.argmax(validities)
    
    # Evaluate options based only on this single cue
    scores = stim[:, best_cue]
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the single-cue scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Overwhelmingly high uniform lapse blended in
    n_opts = len(p_core)
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
**Description:** Weak Weighted Additive with Extreme Noise: In complex multi-cue decision environments, subjects attempt to integrate all available information by weighting each cue according to its provided validity (a compensatory Weighted Additive strategy). However, the cognitive demands of multiplying and summing multiple cues lead to severe cognitive overload and disengagement. Consequently, subjects exhibit an overwhelmingly high lapse rate, effectively guessing on the vast majority of trials. The underlying Weighted Additive evaluation only weakly shines through the noise, explaining why choice behavior hovers very close to random guessing (0.5) across conflict metrics while maintaining a slight systematic preference for higher-validity cue combinations.

**Parameters:**
- beta: [0.1, 10.0]
- epsilon: [0.9, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters['validities'], dtype=float)
    
    # Weighted Additive (WADD): sum of cues weighted by their validities
    scores = stim @ validities
    
    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    
    # Softmax over the WADD scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Overwhelmingly high uniform lapse blended in
    n_opts = len(p_core)
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
[0] rationale: This metric isolates trials where the two theories make strictly opposing predictions despite the overwhelming noise. In trials where Option A possesses the single highest-validity cue but Option B possesses a strong combination of lower-validity cues (Trials 1 and 4), the Single-Cue Focus model will systematically favor Option A (>0.5), whereas the Weak Weighted Additive model will systematically favor Option B (<0.5). By computing the choice proportion specifically on these conflict trials, we maximize the directional divergence between the two models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Identify 'conflict' trials where the Advocated theory (Single-Cue Focus)
    # predicts Option A (because it has the highest-validity cue, index 0),
    # but the Competing theory (Weak WADD) predicts Option B (because the sum
    # of the remaining weighted cues outweighs the first cue).
    # In this design, these correspond to Trial 1 and Trial 4.
    
    is_conflict = data.apply(lambda row: row['option_a_ratings'][0] == 1 and row['option_b_ratings'][0] == 0, axis=1)
    df_conflict = data[is_conflict]
    
    if len(df_conflict) == 0:
        return 0.5
        
    # Return the proportion of times the subject chose Option A in these conflict trials
    return float((df_conflict['response'] == 0).mean())
outcome: self_sim=0.5179 (var=0.0045) adversary_sim=0.4968 (var=0.0077) welch_t=+0.954 p=0.345 (N=25, alpha=0.01) -> reject

[1] rationale: To overcome the extreme noise (epsilon up to 1.0), we compute a specific linear contrast of preferences across the different trial types. We assign positive weights to trials where Advocated predicts Option A but Competing predicts Option B (Trials 1 and 4) and to trials where Advocated is indifferent (Trial 2) but Competing predicts Option B. We assign a negative weight to Trial 3, where Advocated is indifferent but Competing predicts Option A. This maximally separates the theories by aggregating the small systematic deviations into a single robust metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_trial_weight(row):
        A = row['option_a_ratings']
        B = row['option_b_ratings']
        
        # T1: A=[1,0,0,0,0], B=[0,1,1,1,0]
        if A[0] == 1 and B[0] == 0 and A[4] == 0 and B[1] == 1:
            return 2.0
        # T2: A=[1,1,0,0,0], B=[1,0,1,1,1]
        elif A[0] == 1 and B[0] == 1 and A[1] == 1:
            return 1.0
        # T3: A=[0,1,1,0,0], B=[0,0,0,1,1]
        elif A[0] == 0 and B[0] == 0 and A[1] == 1:
            return -2.0
        # T4: A=[1,0,0,0,1], B=[0,1,1,0,0]
        elif A[0] == 1 and B[0] == 0 and A[4] == 1:
            return 2.0
        # T5: A=[0,1,0,0,0], B=[1,0,0,0,0]
        return 0.0
        
    weights = data.apply(get_trial_weight, axis=1)
    
    # Convert response to A=1, B=-1
    choices = 1 - 2 * data['response']
    
    # Sum of weighted choices per subject, then average across subjects
    subject_scores = (weights * choices).groupby(data['subject_id']).mean()
    return float(subject_scores.mean())
outcome: self_sim=0.0291 (var=0.0176) adversary_sim=0.0013 (var=0.0313) welch_t=+0.628 p=0.533 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Convert response to 1 if Option A was chosen, -1 if Option B\n    choice = 1 - 2 * data['response']\n    \n    # Calculate the difference in the most valid cue (Cue 0)\n    # If A has it and B doesn't, this is 1. If B has it and A doesn't, this is -1. If tied, 0.\n    cue0_diff = data.apply(lambda row: row['option_a_ratings'][0] - row['option_b_ratings'][0], axis=1)\n    \n    # Calculate the subject-level alignment with the most valid cue\n    subject_alignment = (choice * cue0_diff).groupby(data['subject_id']).mean()\n    \n    return float(subject_alignment.mean())",
  "rationale": "This metric calculates the degree to which a subject's choices align with the single highest-validity cue (Cue 0). The Advocated theory (Single-Cue Focus) bases its non-lapse decisions entirely on this cue, yielding a strictly positive alignment. By contrast, the Competing theory (Weak WADD) integrates all cues. In this specific experimental design, the trials where Cue 0 differentiates the options (Trials 1, 4, and 5) are carefully constructed such that the sum of the lower-validity cues often strongly overrides Cue 0 (especially in Trial 1, where the WADD difference heavily favors the option lacking Cue 0). Consequently, the Competing theory predicts a net negative alignment with Cue 0 on these differentiating trials. This produces a robust directional split (positive for Advocated, negative for Competing) while being completely invariant to any random swapping of left/right option presentation."
}
```

## Usage

```json
{
  "prompt_token_count": 4033,
  "candidates_token_count": 406,
  "total_token_count": 10849
}
```
