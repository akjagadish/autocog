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

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 5: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 6: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]

**Rationale:** To quantitatively dissociate the Advocated Theory (a mixture of Tallying and Take-The-Best) from the Competing Theory (Tallying with an Inverse Validity Tie-Breaker), we focus on two critical trial types: 'Tally Ties' and 'Compensatory' trials. In 'Tally Tie' trials, both options have an equal number of feature-wise wins. The Advocated Theory resolves these ties naturally via its TTB component, favoring the option that wins on the highest-validity discriminating cue. In stark contrast, the Competing Theory uses an inverse validity tie-breaker, strongly favoring the option that wins on the lowest-validity cues. By pitting high-validity wins against low-validity wins in tie scenarios, we create diametrically opposed predictions. Furthermore, we include 'Compensatory' trials where the option with fewer wins is favored by the highest-validity cue. Here, the Competing Theory strictly follows Tallying (as its tie-breaker cannot override a strict win), whereas the Advocated Theory's TTB mixture component induces a measurable shift in choice probabilities away from the pure Tallying prediction.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Decision-making in binary choice tasks is driven by a probabilistic mixture of two primary heuristics: Tallying and Take-The-Best (TTB). On any given trial, a subject may evaluate the options by counting the number of winning features (Tallying) or by relying solely on the single highest-validity discriminating cue (TTB). When Tallying results in a tie, its choice probabilities become uniform, allowing the TTB preference to naturally act as a tie-breaker without requiring a separate, contrived tie-breaking mechanism. To maintain the strong dominance of Tallying observed in non-tie scenarios, the prior probability of using TTB is constrained to be relatively low.

**Parameters:**
- validities: validities
- p_ttb: [0.0, 0.3]
- beta_ttb: [0.1, 20.0]
- beta_tally: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    p_ttb = float(parameters["p_ttb"])
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Take-The-Best (TTB): rely on the highest validity discriminating cue
    # Use stable sort to preserve left-to-right order for equal validities
    order = np.argsort(-validities, kind='stable')
    ttb_a, ttb_b = 0.0, 0.0
    for idx in order:
        if a[idx] > b[idx]:
            ttb_a = 1.0
            break
        elif b[idx] > a[idx]:
            ttb_b = 1.0
            break
            
    # Tallying: count of strict feature-wise wins
    tally_a = float(np.sum(a > b))
    tally_b = float(np.sum(b > a))
    
    # Compute probabilities for each heuristic
    z_ttb = beta_ttb * np.array([ttb_a, ttb_b])
    probs_ttb = np.exp(z_ttb - np.max(z_ttb))
    probs_ttb /= np.sum(probs_ttb)
    
    z_tally = beta_tally * np.array([tally_a, tally_b])
    probs_tally = np.exp(z_tally - np.max(z_tally))
    probs_tally /= np.sum(probs_tally)
    
    # Mix the strategies
    mixed_probs = p_ttb * probs_ttb + (1.0 - p_ttb) * probs_tally
    
    # Apply lapse rate
    final_probs = (1.0 - epsilon) * mixed_probs + epsilon * np.array([0.5, 0.5])
    
    return final_probs
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
**Description:** Inverse Validity Tie-Breaking with Non-linear Scaling: Decision-makers primarily rely on a Tallying heuristic. When this primary mechanism results in a tie, subjects do not guess randomly, but exhibit a recency bias or systematically misinterpret the cue validities, breaking the tie by heavily weighting the lower-validity (or right-most) features. A non-linear scaling parameter exaggerates this inverse-validity preference to better capture the magnitude of the recency effect.

**Parameters:**
- validities: validities
- w_tie: [0.0, 0.95]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 10.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    validities = np.asarray(parameters["validities"], dtype=float)
    w_tie = float(parameters["w_tie"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    
    # Primary mechanism: Tallying (count of strict wins)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    # Secondary mechanism: Inverse Validity Tie-Breaker
    # Weight lower-validity features more heavily, with a non-linear scaling (gamma)
    tie_weights = (1.0 - validities) ** gamma
    if np.sum(tie_weights) > 0:
        tie_weights /= np.sum(tie_weights)
    else:
        tie_weights = np.ones_like(tie_weights) / len(tie_weights)
        
    tie_score_a = np.sum(a * tie_weights)
    tie_score_b = np.sum(b * tie_weights)
    
    # Combine scores. Since w_tie < 1.0 and tie_score difference is <= 1.0,
    # the tie-breaker will never override a strict Tallying win (difference >= 1.0).
    score_a = a_wins + w_tie * tie_score_a
    score_b = b_wins + w_tie * tie_score_b
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice probabilities
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    probs = e / np.sum(e)
    
    # Apply lapse rate
    final_probs = (1.0 - epsilon) * probs + epsilon * np.array([0.5, 0.5])
    
    return final_probs
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a = np.vstack(data['option_a_ratings'].values)\n    b = np.vstack(data['option_b_ratings'].values)\n    \n    a_wins = np.sum(a > b, axis=1)\n    b_wins = np.sum(b > a, axis=1)\n    \n    tie_trials = a_wins == b_wins\n    \n    if np.sum(tie_trials) == 0:\n        return 0.5\n        \n    return float(np.mean(data['response'].values[tie_trials]))",
  "rationale": "Focuses exclusively on 'Tally Tie' trials, where the two theories make diametrically opposed predictions. The Advocated Theory's Take-The-Best mechanism breaks ties using the highest-validity cue (which always favors option A in these specific ties), leading to a low proportion of B choices. In contrast, the Competing Theory's inverse validity tie-breaker heavily weights the lowest-validity cues (which favor option B), leading to a high proportion of B choices. Measuring the rate of B choices on these tie trials will thus yield a massive, stable gap between the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 3713,
  "candidates_token_count": 301,
  "total_token_count": 5621
}
```
