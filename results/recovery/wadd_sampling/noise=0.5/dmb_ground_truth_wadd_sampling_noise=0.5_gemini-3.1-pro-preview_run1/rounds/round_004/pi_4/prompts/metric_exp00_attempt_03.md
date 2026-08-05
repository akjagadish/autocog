# metric_exp00_attempt_03

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
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.7, 0.65, 0.6]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 3: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 4: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  trial 5: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate the Advocated model (Tallying + WADD Mixture) from the Competing model (Thresholded WADD), we use a 6-feature design with a substantial spread in validities. The Competing model transitions between equal-weighting and lexicographic choice either by scaling validities non-linearly (gamma) or by discarding low-validity features entirely (theta). It cannot, however, simultaneously exhibit strong reliance on feature counting (Tallying) while maintaining a compensatory integration of all features, because ignoring features via a threshold destroys the tally of those features. The Advocated model, by contrast, explicitly mixes a pure feature count (normalized to [0,1]) with a fully compensatory WADD score (also normalized to [0,1]). By including trials where Tallying predicts a tie but WADD predicts a strong preference, and trials where a threshold would completely reverse the WADD preference compared to an unthresholded WADD, we create a regime where the Mixture model can capture choice patterns that the Thresholded WADD model strictly cannot fit.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Decision makers employ a dual-process or strategy mixture approach when evaluating multi-attribute options. Rather than relying entirely on a single strategy, choices are generated by a probabilistic mixture of a simple, unweighted Tallying heuristic (which counts the number of strictly winning features) and a compensatory Weighted Additive (WADD) strategy (which integrates all features weighted by their subjective validities). To ensure equitable application of choice determinism, the evidence scores for both strategies are normalized to a common [0, 1] scale before applying a shared inverse temperature parameter. The mixture parameter 'alpha' dictates the reliance on Tallying versus WADD, allowing the model to capture exact chance-level responding in scenarios where features tie while maintaining sensitivity to cue validities in general.

**Parameters:**
- beta: [0.01, 10.0]
- gamma: [0.0, 5.0]
- alpha: [0.0, 1.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    alpha = float(parameters["alpha"])
    epsilon = float(parameters["epsilon"])
    
    # WADD Component: Weighted sum using non-linearly scaled validities, normalized to [0, 1]
    subjective_weights = val ** gamma
    sum_weights = np.sum(subjective_weights)
    score_a_wadd = np.sum(a * subjective_weights) / sum_weights
    score_b_wadd = np.sum(b * subjective_weights) / sum_weights
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    z_wadd = beta * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # Tallying Component: Count of strict feature-wise wins, normalized to [0, 1]
    a_wins = float(np.sum(a > b)) / n_features
    b_wins = float(np.sum(b > a)) / n_features
    scores_tally = np.array([a_wins, b_wins])
    
    z_tally = beta * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # Mixture of the two strategies
    p_mixed = alpha * p_tally + (1.0 - alpha) * p_wadd
    
    # Incorporate response noise (lapse rate)
    return (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
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
**Description:** Decision makers employ a Thresholded Weighted Additive (WADD) strategy. Instead of integrating all available information or relying entirely on a single cue, subjects impose a depth-of-processing limit by ignoring features whose validities fall below a subjective threshold. Features that exceed this threshold are integrated into a compensatory sum weighted by a non-linear transformation of their validities. By allowing the threshold to be arbitrarily low and constraining the non-linear scaling, the model captures bounded rationality while avoiding an over-reliance on a single best cue.

**Parameters:**
- beta: [0.01, 10.0]
- gamma: [0.0, 3.0]
- theta: [0.0, 1.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    
    # Thresholded WADD Component: Only integrate features whose validities are >= theta
    mask = val >= theta
    subjective_weights = np.zeros_like(val)
    if np.any(mask):
        subjective_weights[mask] = val[mask] ** gamma
        
    sum_weights = np.sum(subjective_weights)
    if sum_weights > 0:
        score_a_wadd = np.sum(a * subjective_weights) / sum_weights
        score_b_wadd = np.sum(b * subjective_weights) / sum_weights
    else:
        score_a_wadd, score_b_wadd = 0.5, 0.5
        
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    z_wadd = beta * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # Incorporate response noise (lapse rate)
    p_final = (1.0 - epsilon) * p_wadd + epsilon * np.array([0.5, 0.5])
    
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
[0] rationale: In Trial 1, Option A has the two most valid features but loses on the remaining four, whereas Option B has the four least valid features. The Advocated model (Mixture of Tallying and WADD) strongly predicts choosing Option B, because Tallying overwhelmingly favors B (4 wins to 2) and full WADD also favors B (sum of validities is higher). Consequently, the probability of choosing B is very high. By contrast, the Competing model (Thresholded WADD) frequently drops the less valid features. For many threshold values, it evaluates Option A as superior because it only considers the top features, leading to a much lower average rate of choosing Option B. Measuring the proportion of B choices on this specific trial cleanly discriminates the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    # Identify Trial 1 where A=[1, 1, 0, 0, 0, 0] and B=[0, 0, 1, 1, 1, 1]
    is_t1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0, 0))
    if not is_t1.any():
        return 0.5
    # Response 1 means choosing B. 
    # The Advocated model strongly prefers B (both Tallying and WADD favor B).
    # The Competing model (Thresholded WADD) will often prefer A if the threshold is high (ignoring lower validities).
    return float(data[is_t1]['response'].mean())
outcome: self_sim=0.6112 (var=0.0266) adversary_sim=0.4700 (var=0.0522) welch_t=+2.515 p=0.01567 (N=25, alpha=0.01) -> reject

[1] rationale: By contrasting the choice rates of Option B between Trial 1 and Trial 3, we maximize the divergence between the Advocated and Competing theories. In Trial 1, the Advocated model (Mixture) strongly favors B due to Tallying (4 wins to 2) and WADD, whereas the Competing model (Thresholded WADD) often drops B's lower-validity features and favors A. In Trial 3, the Advocated model favors A (WADD prefers A and Tallying ties), but the Competing model frequently favors B when its threshold is set high (since B has the single most valid cue, 0.95). Thus, the difference (Trial 1 B rate - Trial 3 B rate) will be substantially positive for the Advocated theory and much lower (or negative) for the Competing theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    # Trial 1: A=[1, 1, 0, 0, 0, 0], B=[0, 0, 1, 1, 1, 1]
    is_t1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0, 0))
    # Trial 3: A=[0, 1, 1, 1, 0, 0], B=[1, 0, 0, 0, 1, 1]
    is_t3 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 1, 0, 0))
    
    b_t1 = data[is_t1]['response'].mean() if is_t1.any() else 0.5
    b_t3 = data[is_t3]['response'].mean() if is_t3.any() else 0.5
    
    return float(b_t1 - b_t3)
outcome: self_sim=0.1800 (var=0.0485) adversary_sim=0.0200 (var=0.0569) welch_t=+2.463 p=0.01743 (N=25, alpha=0.01) -> reject

[2] rationale: We contrast the rate of choosing Option B on Trial 4 versus Trial 3. On both trials, the Tallying strategy predicts an exact tie (3 vs 3 features). Thus, for the Advocated model (Mixture), the choice direction is entirely driven by the compensatory WADD component. WADD consistently favors B on Trial 4 and A on Trial 3 across all non-linear scaling (gamma) values, yielding a positive difference (Trial 4 B rate > Trial 3 B rate). Conversely, the Competing model (Thresholded WADD) frequently drops lower-validity features. Because Option A has the single highest validity feature (0.95) on Trial 4, high thresholds make the Competing model strongly favor A. On Trial 3, Option B has the 0.95 feature, so high thresholds make it strongly favor B. This thresholding flips the preference direction compared to the unthresholded WADD, driving the Competing model's difference (Trial 4 B rate - Trial 3 B rate) to be negative or near zero on average. This creates a clean, robust divergence between the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Trial 4: A=[1, 0, 0, 1, 1, 0], B=[0, 1, 1, 0, 0, 1]
    is_t4 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 1, 0))
    # Trial 3: A=[0, 1, 1, 1, 0, 0], B=[1, 0, 0, 0, 1, 1]
    is_t3 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 1, 0, 0))
    
    b_t4 = data[is_t4]['response'].mean() if is_t4.any() else 0.5
    b_t3 = data[is_t3]['response'].mean() if is_t3.any() else 0.5
    
    return float(b_t4 - b_t3)
outcome: self_sim=0.1012 (var=0.0387) adversary_sim=0.0987 (var=0.0922) welch_t=+0.035 p=0.9726 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    # Trial 1: A=[1, 1, 0, 0, 0, 0]\n    is_t1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0, 0))\n    # Trial 3: A=[0, 1, 1, 1, 0, 0]\n    is_t3 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 1, 0, 0))\n    # Trial 4: A=[1, 0, 0, 1, 1, 0]\n    is_t4 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 1, 0))\n    \n    # Calculate the proportion of \"threshold-driven\" choices\n    # For Trial 1, thresholding favors A, but WADD/Tallying favors B. (response 0 = A)\n    t1_a = 1.0 - data[is_t1]['response'].mean() if is_t1.any() else 0.5\n    \n    # For Trial 3, thresholding favors B (B has the 0.95 cue), but WADD favors A.\n    t3_b = data[is_t3]['response'].mean() if is_t3.any() else 0.5\n    \n    # For Trial 4, thresholding favors A (A has the 0.95 cue), but WADD favors B.\n    t4_a = 1.0 - data[is_t4]['response'].mean() if is_t4.any() else 0.5\n    \n    return float((t1_a + t3_b + t4_a) / 3.0)",
  "rationale": "To robustly discriminate the two theories and minimize between-subject variance, this metric aggregates choice rates across three critical trials (Trials 1, 3, and 4) where the Thresholded WADD model frequently diverges from the Advocated model (Tallying + WADD Mixture). In all three trials, the Compensatory/Tallying components of the Advocated model favor one option (B, A, and B respectively), yielding a very low probability for the opposite choices. However, the Competing model, by applying a high threshold and ignoring lower-validity cues, strongly favors the opposite choices (A, B, and A respectively) because those options possess the highest-validity cues. By averaging the rates of these 'threshold-driven' choices, we create a composite score that will be significantly higher for the Competing theory than for the Advocated theory, while smoothing out trial-specific noise to keep the per-subject variance low."
}
```

## Usage

```json
{
  "prompt_token_count": 5318,
  "candidates_token_count": 685,
  "total_token_count": 9774
}
```
