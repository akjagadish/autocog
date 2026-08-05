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
**Validities (n_features=4):** [0.95, 0.85, 0.75, 0.65]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0]  B=[0, 0, 1, 0]
  trial 3: A=[1, 0, 0, 0]  B=[0, 0, 0, 1]
  trial 4: A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 5: A=[0, 1, 0, 0]  B=[0, 0, 0, 1]
  trial 6: A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  trial 7: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 8: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 9: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 10: A=[1, 1, 1, 0]  B=[0, 0, 0, 1]

**Rationale:** While previous designs tested extreme versus moderate differences to detect a WADD signal under high noise, this experiment employs a rigorous pairwise trade-off and additivity test. We isolate individual cue differences (e.g., Cue 1 vs Cue 2, Cue 1 vs Cue 3) and their specific combinations. The advocated Pure Random Guessing theory predicts a completely flat 0.50 choice probability across all pairs, with no underlying structure. In contrast, the competing WADD with Extreme Noise theory predicts that despite the heavy regression to 0.50, the residual small deviations will strictly obey linear additivity (e.g., the preference margin in a combined trial will correspond to the sum of the margins in its component trials). This structural test allows us to detect even minuscule compensatory signals by checking for additive consistency in the noise.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Pure Random Guessing (Complete Cognitive Overload): Subjects completely ignore the cues and validities due to the high cognitive demand and artificial nature of the fictitious task. This results in choices that are strictly driven by a 50/50 binomial coin flip, without any underlying compensatory or non-compensatory signal.

**Parameters:**
- guessing_rate: {0.5}

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    p = float(parameters['guessing_rate'])
    return np.array([p, 1.0 - p])
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
**Description:** Weighted Additive (WADD) with Extreme Noise Theory: Subjects attempt to integrate all available cues by weighting them according to their provided validities. However, the cognitive demand of integrating multiple conflicting fictitious validities is overwhelming. This results in extreme decision conflict and distraction, leading to a near-total reliance on guessing (a very high lapse rate). Consequently, choice probabilities are pulled almost entirely toward chance (0.50), masking the underlying compensatory process in the aggregate behavioral data.

**Parameters:**
- beta: [0.0, 2.0]
- epsilon: [0.95, 1.0]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    
    # Weighted sum per option (dot product with per-feature weights)
    scores = stim @ (validities * w)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the WADD scores with max-subtraction for stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with extreme uniform lapse (guessing)
    n_opts = p_core.shape[0]
    p_final = (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
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
[0] rationale: To maximize the signal-to-noise ratio in detecting a minuscule compensatory signal hidden under extreme noise, we apply a matched filter approach. For each trial, we compute the objective difference in validity scores between Option A and Option B. We then compute the covariance between this objective difference and the subject's choice (mapped to 0.5 for A and -0.5 for B). Pure Random Guessing predicts exactly 0 for this metric. WADD with Extreme Noise predicts a strictly positive value, because even under extreme uniform lapse, the residual softmax probabilities will tilt linearly in favor of the option with the higher validity sum. Weighting by the magnitude of the difference ensures that trials with the strongest structural contrast (like Trial 10) contribute the most to the signal, optimizing statistical power.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.95, 0.85, 0.75, 0.65])
    
    A = np.array(data['option_a_ratings'].tolist())
    B = np.array(data['option_b_ratings'].tolist())
    
    # Calculate the objective difference in validity scores for each trial
    diffs = np.sum((A - B) * validities, axis=1)
    
    # Map response to direction: 0.5 if A was chosen, -0.5 if B was chosen
    choices_A_dir = 0.5 - data['response'].values
    
    # Matched filter: correlation between objective difference and choice direction
    return float(np.mean(diffs * choices_A_dir))
outcome: self_sim=0.0012 (var=0.0011) adversary_sim=0.0013 (var=0.0012) welch_t=-0.009 p=0.9926 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    # Objective validities for the 4 cues\n    validities = np.array([0.95, 0.85, 0.75, 0.65])\n    \n    # Extract ratings arrays\n    A = np.array(data['option_a_ratings'].tolist())\n    B = np.array(data['option_b_ratings'].tolist())\n    \n    # Calculate the objective score difference for each trial\n    # Assuming average weights w = [0.5, 0.5, 0.5, 0.5]\n    diffs = 0.5 * np.sum((A - B) * validities, axis=1)\n    \n    # Compute the expected choice probability under WADD with Extreme Noise\n    # Assuming expected beta = 1.0 and expected epsilon = 0.975 (so 1-epsilon = 0.025)\n    z = 1.0 * diffs\n    p_core = np.exp(z) / (1.0 + np.exp(z))\n    p_A = 0.025 * p_core + 0.975 * 0.5\n    \n    # Convert responses: 1 if A was chosen, 0 if B was chosen\n    y = 1.0 - data['response'].values\n    \n    # Return the log-likelihood of the choices under the expected WADD model\n    # By Neyman-Pearson lemma, this is the most powerful test statistic.\n    ll = y * np.log(p_A) + (1.0 - y) * np.log(1.0 - p_A)\n    \n    return float(np.sum(ll))",
  "rationale": "Given the extreme noise in the WADD model (epsilon uniformly drawn between 0.95 and 1.0), the maximum possible deviation from random guessing in any trial is bounded to ~2.5%. To maximize statistical power, we apply the Neyman-Pearson lemma and compute the log-likelihood of the subject's choices under the expected marginal WADD model. This optimally weights every trial according to its structural signal-to-noise ratio, accumulating all available minuscule compensatory signals across the entire experiment into a single optimal scalar."
}
```

## Usage

```json
{
  "prompt_token_count": 3262,
  "candidates_token_count": 553,
  "total_token_count": 27227
}
```
