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
**Validities (n_features=5):** [0.55, 0.6, 0.65, 0.9, 0.95]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 2: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 5: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 7: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 8: A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]

**Rationale:** To quantitatively dissociate the Advocated Theory (Context-Dependent Dual-Process Mixture of Recency-Boosted WADD and Tallying) from the Competing Theory (Context-Dependent Dual-Process Mixture of TTB and Tallying with Inverse Validity Tie-Breaking), we exploit two key structural differences: the direction of feature evaluation and the treatment of validities in tie-breaking. The Competing Theory uses a Left-to-Right TTB mechanism and an Inverse Validity tie-breaker (which strongly favors lower-validity cues), whereas the Advocated Theory relies on a Recency-Boosted WADD mechanism (which favors right-most cues) combined with explicit validities. By setting up an environment with monotonically increasing validities (lowest on the left, highest on the right), we create direct diametric conflicts. In 'Tally Tie' trials, the Competing Theory's LTR TTB and Inverse Validity tie-breaker strongly favor the left-most options, while the Advocated Theory's Recency-Boosted WADD strongly favors the right-most options. In 'Compensatory' trials, we pit a left-most tallying advantage against a right-most WADD advantage to further separate the mixture components.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Context-Dependent Dual-Process Mixture of Recency-Boosted WADD and Tallying (Score-Mixed, Unnormalized WADD): Decision-makers use a dual-process strategy mixing a Validity-Weighted Additive (WADD) strategy and a Tallying heuristic based on the dispersion of cue validities. The mixture occurs at the level of decision values (scores). To allow WADD to break Tallying ties effectively even when the mixture weight heavily favors Tallying, the WADD cue weights are left unnormalized. This allows the raw WADD scores to scale up naturally with a wider recency parameter, providing a strong enough signal to break ties smoothly without requiring an extreme global softmax temperature.

**Parameters:**
- validities: validities
- disp_slope: [0.0, 200.0]
- disp_threshold: [0.0, 0.5]
- recency: [0.0, 50.0]
- beta: [0.1, 500.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    validities = np.asarray(parameters["validities"], dtype=float)
    disp_slope = float(parameters["disp_slope"])
    disp_threshold = float(parameters["disp_threshold"])
    recency = float(parameters["recency"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Calculate mixture weight based on dispersion of validities (standard deviation)
    dispersion = float(np.std(validities))
    w_wadd_mix = 1.0 / (1.0 + np.exp(-disp_slope * (dispersion - disp_threshold)))
    
    # --- Strategy 1: Tallying ---
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    # Normalize to [0, 1] so it is on a comparable scale to WADD scores
    scores_tally_norm = scores_tally / max(1, n_features)
    
    # --- Strategy 2: Recency-Boosted WADD ---
    ranks = np.arange(n_features)
    # Right-most features get a higher recency boost. Weights are left unnormalized 
    # so that WADD differences can naturally scale up to break Tallying ties.
    w_wadd_cue = validities + recency * (ranks / max(1, n_features - 1))
        
    score_a_wadd = np.sum(a * w_wadd_cue)
    score_b_wadd = np.sum(b * w_wadd_cue)
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    # --- Mixture Model (Mixing Scores) ---
    mixed_scores = (1.0 - w_wadd_mix) * scores_tally_norm + w_wadd_mix * scores_wadd
    
    # Softmax on mixed scores
    z = beta * (mixed_scores - np.max(mixed_scores))
    e = np.exp(z)
    probs = e / np.sum(e)
    
    # --- Lapse Rate ---
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


## COMPETING THEORY
**Description:** Context-Dependent Dual-Process Mixture of TTB and Tallying with Inverse Validity Tie-Breaking: Decision-makers rely on a mixture of Take-The-Best (TTB) and Tallying, but the mixture weight is dynamically determined by the environment. When cue validities are highly dispersed (measured by the standard deviation of the validities), subjects predominantly use TTB; when validities are similar, they rely on Tallying. When Tallying results in a tie, subjects resolve it using an inverse-validity weighting mechanism, heavily favoring options with positive features among the lower-validity (or more recently processed) cues.

**Parameters:**
- validities: validities
- disp_slope: [0.0, 100.0]
- disp_threshold: [0.0, 1.0]
- w_tie: [0.0, 0.95]
- gamma: [0.1, 10.0]
- beta_tally: [0.1, 20.0]
- beta_ttb: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    validities = np.asarray(parameters["validities"], dtype=float)
    disp_slope = float(parameters["disp_slope"])
    disp_threshold = float(parameters["disp_threshold"])
    w_tie = float(parameters["w_tie"])
    gamma = float(parameters["gamma"])
    beta_tally = float(parameters["beta_tally"])
    beta_ttb = float(parameters["beta_ttb"])
    epsilon = float(parameters["epsilon"])
    
    # Calculate mixture weight based on dispersion of validities (standard deviation)
    dispersion = float(np.std(validities))
    w_ttb = 1.0 / (1.0 + np.exp(-disp_slope * (dispersion - disp_threshold)))
    
    # --- Strategy 1: Tallying with Inverse Validity Tie-Breaker ---
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    tie_weights = (1.0 - validities) ** gamma
    if np.sum(tie_weights) > 0:
        tie_weights /= np.sum(tie_weights)
    else:
        tie_weights = np.ones_like(tie_weights) / len(tie_weights)
        
    tie_score_a = np.sum(a * tie_weights)
    tie_score_b = np.sum(b * tie_weights)
    
    # w_tie < 1.0 ensures the tie-breaker only dictates choice when a_wins == b_wins
    score_a_tally = a_wins + w_tie * tie_score_a
    score_b_tally = b_wins + w_tie * tie_score_b
    
    scores_tally = np.array([score_a_tally, score_b_tally])
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    probs_tally = e_tally / np.sum(e_tally)
    
    # --- Strategy 2: Take-The-Best (TTB) ---
    ttb_a = 0.0
    ttb_b = 0.0
    for i in range(n_features):
        if a[i] > b[i]:
            ttb_a = 1.0
            break
        elif b[i] > a[i]:
            ttb_b = 1.0
            break
            
    scores_ttb = np.array([ttb_a, ttb_b])
    z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
    e_ttb = np.exp(z_ttb)
    probs_ttb = e_ttb / np.sum(e_ttb)
    
    # --- Mixture Model ---
    mixed_probs = w_ttb * probs_ttb + (1.0 - w_ttb) * probs_tally
    
    # --- Lapse Rate ---
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    scores = []\n    for _, row in data.iterrows():\n        a = row['option_a_ratings']\n        b = row['option_b_ratings']\n        resp = row['response']\n        \n        # Find right-most differing feature\n        right_favors_b = False\n        right_favors_a = False\n        for i in range(len(a)-1, -1, -1):\n            if a[i] != b[i]:\n                if b[i] > a[i]:\n                    right_favors_b = True\n                else:\n                    right_favors_a = True\n                break\n                \n        if right_favors_b and resp == 1:\n            scores.append(1)\n        elif right_favors_a and resp == 0:\n            scores.append(1)\n        else:\n            scores.append(0)\n            \n    return float(np.mean(scores))",
  "rationale": "This metric calculates the proportion of trials where the subject chooses the option favored by the right-most differing feature. The Advocated Theory (Recency-Boosted WADD) heavily weighs the right-most features due to both the recency parameter and the higher validities on the right. In contrast, the Competing Theory relies on Left-to-Right Take-The-Best (TTB) and an Inverse Validity Tie-Breaker, both of which strongly favor the left-most features. Because the experimental design pits the right-most and left-most features against each other in almost all trials, this metric will yield a high value (~1) for the Advocated Theory and a very low value (~0) for the Competing Theory."
}
```

## Usage

```json
{
  "prompt_token_count": 4330,
  "candidates_token_count": 428,
  "total_token_count": 7054
}
```
