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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 6: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 7: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  trial 8: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]

**Rationale:** To quantitatively dissociate pure Tallying from Tallying with a Conditional Take-The-Best (TTB) Fallback, we focus on trials where the tally difference is zero or small, but the highest-validity discriminating cue strongly points in a specific direction. Pure Tallying strictly evaluates the options based on the difference in the raw count of positive features, predicting exactly equal preference (50/50) when tallies are tied, regardless of cue validities. The Competing Theory posits that when tallies are tied or close (difference <= threshold), subjects fall back on TTB. By including 'tally-tied' trials where the most valid cue favors one option, and 'tally-close' trials where the most valid cue opposes the tally winner, we can detect the localized validity-driven asymmetries predicted by the Competing Theory. If pure Tallying is correct, choice probabilities will depend only on the tally differences and remain completely blind to the identity of the highest-validity cue.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Tallying (Equal-Weight) Heuristic: People evaluate options by simply counting the number of positive features (or cues favoring each option) and choosing the option with the higher total count. This compensatory strategy ignores the differential validities or subjective importance of different cues, treating all pieces of evidence equally. The choice probability is determined by a softmax over the total feature tallies for each option, combined with a uniform lapse rate. Crucially, the softmax temperature is constrained to produce softer choice probabilities, reflecting that humans do not apply the tallying rule completely deterministically.

**Parameters:**
- beta: [0.0, 1.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tally the number of positive cues for each option
    score_a = np.sum(stim[0])
    score_b = np.sum(stim[1])
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tallies
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## COMPETING THEORY
**Description:** Tallying with Conditional Take-The-Best (TTB) Fallback: Decision makers primarily rely on a compensatory, equal-weight heuristic (Tallying). However, when the options are difficult to distinguish based on tallies alone (i.e., when the tallies are tied or very close), subjects fall back on a non-compensatory strategy, checking the single most valid discriminating cue (Take-The-Best). Importantly, this fallback is weak, meaning that even when tallies are tied, subjects often guess rather than deterministically applying TTB. This predicts localized, minor validity-driven asymmetries specifically when tallies fail to provide a strong signal, avoiding the overprediction of global validity effects characteristic of full WADD models.

**Parameters:**
- threshold: {0, 1}
- w_close: [0.0, 0.3]
- w_far: [0.0, 0.5]
- beta_tally: [0.0, 5.0]
- beta_ttb: [0.0, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    # Strategy 1: Tallying
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    diff = abs(tally_a - tally_b)
    
    # Strategy 2: Take-The-Best (TTB)
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable")
    
    ttb_a = 0.5
    ttb_b = 0.5
    for j in cue_order:
        if a[j] > b[j]:
            ttb_a, ttb_b = 1.0, 0.0
            break
        elif b[j] > a[j]:
            ttb_a, ttb_b = 0.0, 1.0
            break
            
    # Conditional reliance on TTB based on tally closeness
    threshold = float(parameters["threshold"])
    if diff <= threshold:
        w_ttb = float(parameters["w_close"])
    else:
        w_ttb = float(parameters["w_far"])
        
    # Softmax for Tallying
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * np.array([tally_a, tally_b])
    z_tally -= np.max(z_tally)
    p_tally = np.exp(z_tally)
    p_tally /= np.sum(p_tally)
    
    # Softmax for TTB
    beta_ttb = float(parameters["beta_ttb"])
    z_ttb = beta_ttb * np.array([ttb_a, ttb_b])
    z_ttb -= np.max(z_ttb)
    p_ttb = np.exp(z_ttb)
    p_ttb /= np.sum(p_ttb)
    
    # Mixture
    p_core = w_ttb * p_ttb + (1.0 - w_ttb) * p_tally
    
    # Lapse rate
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
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
[0] rationale: The metric isolates trials where the tallies of positive features are exactly tied. Pure Tallying predicts choices will be completely random (50%) on these trials, as it cannot distinguish the options. The Competing Theory (Conditional TTB) predicts subjects will fall back to evaluating the most valid discriminating cue, leading to a choice proportion greater than 50% for the TTB-favored option. This cleanly separates the models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_arr = np.array(data['option_a_ratings'].tolist())
    b_arr = np.array(data['option_b_ratings'].tolist())
    
    sum_a = a_arr.sum(axis=1)
    sum_b = b_arr.sum(axis=1)
    
    tied = sum_a == sum_b
    if not np.any(tied):
        return 0.5
        
    a_tied = a_arr[tied]
    b_tied = b_arr[tied]
    resp_tied = data['response'].values[tied]
    
    diffs = a_tied - b_tied
    
    ttb_chosen = 0
    valid_tied = 0
    
    for i in range(len(a_tied)):
        diff = diffs[i]
        nonzero = np.nonzero(diff)[0]
        if len(nonzero) > 0:
            first_idx = nonzero[0]
            ttb_favors_a = diff[first_idx] > 0
            chose_a = (resp_tied[i] == 0)
            if (ttb_favors_a and chose_a) or (not ttb_favors_a and not chose_a):
                ttb_chosen += 1
            valid_tied += 1
            
    return float(ttb_chosen / valid_tied) if valid_tied > 0 else 0.5
outcome: self_sim=0.4928 (var=0.0091) adversary_sim=0.5328 (var=0.0087) welch_t=-1.499 p=0.1403 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_arr = np.array(data['option_a_ratings'].tolist())\n    b_arr = np.array(data['option_b_ratings'].tolist())\n    \n    sum_a = a_arr.sum(axis=1)\n    sum_b = b_arr.sum(axis=1)\n    diffs = np.abs(sum_a - sum_b)\n    resp = data['response'].values\n    \n    effects = []\n    # Evaluate for each magnitude of tally difference separately to control for diff-based probabilities\n    for d_val in [1, 2]:\n        mask = diffs == d_val\n        if not np.any(mask):\n            continue\n            \n        a_sub = a_arr[mask]\n        b_sub = b_arr[mask]\n        sum_a_sub = sum_a[mask]\n        sum_b_sub = sum_b[mask]\n        resp_sub = resp[mask]\n        \n        aligned = []\n        opposed = []\n        \n        for i in range(len(a_sub)):\n            tally_favors_a = sum_a_sub[i] > sum_b_sub[i]\n            \n            # Determine TTB prediction (first differing cue)\n            d_vec = a_sub[i] - b_sub[i]\n            nonzero = np.nonzero(d_vec)[0]\n            if len(nonzero) == 0:\n                continue\n            ttb_favors_a = d_vec[nonzero[0]] > 0\n            \n            chose_a = (resp_sub[i] == 0)\n            tally_consistent = float((tally_favors_a and chose_a) or (not tally_favors_a and not chose_a))\n            \n            if tally_favors_a == ttb_favors_a:\n                aligned.append(tally_consistent)\n            else:\n                opposed.append(tally_consistent)\n                \n        if len(aligned) > 0 and len(opposed) > 0:\n            effects.append(np.mean(aligned) - np.mean(opposed))\n            \n    if len(effects) == 0:\n        return 0.0\n    return float(np.mean(effects))",
  "rationale": "To cleanly dissociate Pure Tallying from TTB Fallback, we examine trials where the tallies are not tied, but are close (difference of 1 or 2). We split these trials into 'Aligned' (where the highest-validity cue points in the same direction as the tally winner) and 'Opposed' (where the highest-validity cue points to the tally loser). By computing the difference in the rate of tally-consistent choices between Aligned and Opposed trials\u2014separately for diff=1 and diff=2 to perfectly control for baseline probabilities\u2014Pure Tallying will predict a difference of exactly zero, as it is blind to cue identity. The TTB Fallback theory predicts a positive difference, as TTB acts as a secondary force that boosts the tally winner in Aligned trials and hurts the tally winner in Opposed trials."
}
```

## Usage

```json
{
  "prompt_token_count": 4129,
  "candidates_token_count": 778,
  "total_token_count": 9083
}
```
