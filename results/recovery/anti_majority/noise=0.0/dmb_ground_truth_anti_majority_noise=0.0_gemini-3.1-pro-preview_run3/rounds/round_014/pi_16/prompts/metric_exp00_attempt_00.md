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
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.55, 0.5]

**Trial pairs (n=11):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  trial 6: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 7: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 10: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]

**Rationale:** Both the Advocated model (CDLA) and the Competing model (Strategy Mixture) share a remarkable structural invariance: their predicted choice probabilities depend exclusively on the identity of the highest-validity discriminating cue (the TTB winner) and the absolute difference in the total number of winning cues (the Tally difference). However, they map this Tally difference to choice probabilities using fundamentally different functional forms. The Competing Mixture model uses a linear mixture of a constant TTB probability and a symmetric logistic Tallying curve. In contrast, the Advocated CDLA model modulates the log-odds evidence of the first cue via an exponential function of the 'contradiction' (which is strictly linearly related to the Tally difference). By holding the TTB winner constant (Option A always wins the highest validity cue) and systematically sweeping the Tally difference from extreme consonance (+5) to extreme contradiction (-5), we map out the response curves. The Mixture model is constrained to a symmetric, asymptoting logistic mixture, whereas CDLA's exponential modulation can produce highly asymmetric, non-linear, and even non-monotonic response curves as contradiction increases.

**Computed schedule:** 11 unique pairs × 8 reps = 88 trials per subject.



## ADVOCATED THEORY
**Description:** Context-Dependent Lexicographic Accumulator (CDLA) with Symmetric Dispersion Effect: Decision-makers process information lexicographically, but the weight assigned to the first discriminating cue is contextually modulated by the dispersion of remaining cues in a symmetric manner. Both contradiction and consonance from secondary cues adjust the primary cue's weight. Furthermore, both primary and secondary cues can be weighted negatively, capturing baseline avoidance and perverse context effects.

**Parameters:**
- w_first: [-10.0, 10.0]
- w_remaining: [-5.0, 5.0]
- dispersion_effect: [-2.0, 2.0]
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
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    first_disc_idx = -1
    for i in cue_order:
        if a[i] != b[i]:
            first_disc_idx = i
            break
            
    if first_disc_idx == -1:
        return np.array([0.5, 0.5])
        
    w_first = float(parameters["w_first"])
    w_remaining = float(parameters["w_remaining"])
    dispersion_effect = float(parameters["dispersion_effect"])
    
    remaining_a = 0
    remaining_b = 0
    passed_first = False
    for i in cue_order:
        if passed_first:
            remaining_a += a[i]
            remaining_b += b[i]
        if i == first_disc_idx:
            passed_first = True
            
    first_diff = a[first_disc_idx] - b[first_disc_idx]
    
    if first_diff > 0:
        contradiction = remaining_b - remaining_a
    else:
        contradiction = remaining_a - remaining_b
        
    # The weight of the first cue is modulated symmetrically by the contradiction/consonance
    adj_first = w_first * np.exp(dispersion_effect * contradiction)
    
    ev_a = 0.0
    ev_b = 0.0
    
    if first_diff > 0:
        ev_a += adj_first
    else:
        ev_b += adj_first
        
    ev_a += w_remaining * remaining_a
    ev_b += w_remaining * remaining_b
    
    beta = float(parameters["beta"])
    z = beta * np.array([ev_a, ev_b])
    z = z - np.max(z)
    p = np.exp(z)
    p = p / np.sum(p)
    
    epsilon = float(parameters["epsilon"])
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
**Description:** Probabilistic Strategy Mixture (Soft TTB and Soft Tallying)

**Parameters:**
- p_ttb: [0.0, 1.0]
- beta_ttb: [0.1, 10.0]
- beta_tally: [0.1, 5.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # 1. Take-The-Best (TTB) Prediction (Probabilistic)
    ttb_winner = 0.5
    for j in cue_order:
        if a[j] > b[j]:
            ttb_winner = 1.0
            break
        elif b[j] > a[j]:
            ttb_winner = 0.0
            break
            
    ttb_scores = np.array([ttb_winner, 1.0 - ttb_winner])
    beta_ttb = float(parameters["beta_ttb"])
    z_ttb = beta_ttb * ttb_scores
    z_ttb = z_ttb - np.max(z_ttb)
    p_ttb_dist = np.exp(z_ttb) / np.sum(np.exp(z_ttb))
    
    # 2. Tallying Prediction
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * np.array([tally_a, tally_b])
    z_tally = z_tally - np.max(z_tally)
    p_tally_dist = np.exp(z_tally) / np.sum(np.exp(z_tally))
    
    # 3. Strategy Mixture
    p_ttb = float(parameters["p_ttb"])
    p_core = p_ttb * p_ttb_dist + (1.0 - p_ttb) * p_tally_dist
    
    return p_core
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data = data.copy()\n    data['chose_A'] = 1 - data['response']\n    data['tally_A'] = data['option_a_ratings'].apply(sum)\n    data['tally_B'] = data['option_b_ratings'].apply(sum)\n    data['tally_diff'] = data['tally_A'] - data['tally_B']\n    \n    means = data.groupby('tally_diff')['chose_A'].mean()\n    \n    # Measure the asymmetry of the marginal effect of cues at the extremes.\n    # Effect of moving from tally_diff -2 to -4 (adding opposing cues)\n    eff_neg = means.get(-2, 0.5) - means.get(-4, 0.5)\n    \n    # Effect of moving from tally_diff +4 to +6 (adding supporting cues)\n    eff_pos = means.get(6, 0.5) - means.get(4, 0.5)\n    \n    return float(eff_neg - eff_pos)\n",
  "rationale": "The Competing Theory (Strategy Mixture) relies on a logistic function of the tally difference, which has a symmetric S-shape. At the extremes of the tally difference (-4 and +6), the marginal effect of adding more cues flattens out symmetrically. The Advocated Theory (CDLA) modulates the primary cue weight via an exponential function of the contradiction. This exponential functional form is inherently asymmetric: it can cause the choice probabilities to sharply drop on the negative side while saturating quickly on the positive side (or vice versa), leading to a strong difference in marginal effects between the two extremes."
}
```

## Usage

```json
{
  "prompt_token_count": 3852,
  "candidates_token_count": 405,
  "total_token_count": 7383
}
```
