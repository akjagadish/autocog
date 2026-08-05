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
**Validities (n_features=5):** [0.95, 0.9, 0.85, 0.8, 0.75]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 5: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Rationale:** The Competing model uses unweighted sequential tallying, where stopping is triggered when the absolute difference in won cues exceeds a threshold. Trials where a leading option wins consecutive cues (e.g., A wins cues 1 and 2) will rapidly trigger this threshold, causing strong preferences. When wins are alternating, the evidence difference remains near zero, preventing early stopping and pushing the decision to the final unweighted tally. The Advocated model, however, weights evidence by validity. It does not rely on an unweighted tally threshold, but probabilistically stops at any discriminating cue, making choices based on the accumulated validity-weighted evidence.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Sequential Cue Evaluation with Probabilistic Stopping: Decision-makers evaluate cues sequentially in descending order of validity. Upon finding a discriminating cue, they stop with a certain probability and choose based on accumulated evidence. If they continue, they integrate further cues, naturally blending non-compensatory (TTB) and compensatory (Tallying/WADD) behaviors.

**Parameters:**
- p_stop: [0.0, 1.0]
- beta: [0.1, 20.0]
- kappa: [0.0, 10.0]
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
    
    p_stop = float(parameters["p_stop"])
    beta = float(parameters["beta"])
    kappa = float(parameters["kappa"])
    epsilon = float(parameters["epsilon"])
    
    ev_A = 0.0
    ev_B = 0.0
    
    w_reach = 1.0
    p_A_total = 0.0
    p_B_total = 0.0
    
    for i, j in enumerate(cue_order):
        weight = val[j] ** kappa
        ev_A += a[j] * weight
        ev_B += b[j] * weight
        
        is_last = (i == len(cue_order) - 1)
        
        # Stop probabilistically only if the cue discriminates
        if a[j] != b[j]:
            p_s = p_stop
        else:
            p_s = 0.0
            
        # Must stop at the last cue
        if is_last:
            p_s = 1.0
            
        w_stop = w_reach * p_s
        
        # Choice probabilities if stopping at this step
        z = beta * np.array([ev_A, ev_B])
        e = np.exp(z - np.max(z))
        p_choice = e / np.sum(e)
        
        p_A_total += w_stop * p_choice[0]
        p_B_total += w_stop * p_choice[1]
        
        # Update probability of reaching the next step
        w_reach *= (1.0 - p_s)
        
    p_final = np.array([p_A_total, p_B_total])
    return (1.0 - epsilon) * p_final + epsilon * np.array([0.5, 0.5])
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
**Description:** Sequential Bounded Evidence Accumulation: Decision-makers evaluate cues sequentially in descending order of validity, maintaining a running tally of the evidence difference between options. At each step, if the absolute evidence difference exceeds a noisy, subject-specific threshold, evaluation stops and the leading option is chosen. If all cues are exhausted without the threshold being crossed, the final choice is made based on the accumulated evidence. This mechanism naturally bridges non-compensatory (Take-The-Best) and compensatory (Tallying) behaviors.

**Parameters:**
- theta: [0.0, 1.5]
- alpha: [0.0, 20.0]
- beta: [0.0, 20.0]
- epsilon: [0.0, 0.25]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    theta = float(parameters["theta"])
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    p_a_total = 0.0
    p_b_total = 0.0
    p_active = 1.0
    
    E = 0.0
    for j in cue_order:
        E += (a[j] - b[j])
        
        if E != 0:
            z = alpha * (np.abs(E) - theta)
            # Numerically stable sigmoid for stopping probability
            if z > 20:
                p_stop = 1.0
            elif z < -20:
                p_stop = 0.0
            else:
                p_stop = 1.0 / (1.0 + np.exp(-z))
                
            prob_stop_now = p_active * p_stop
            
            if E > 0:
                p_a_total += prob_stop_now
            else:
                p_b_total += prob_stop_now
                
            p_active -= prob_stop_now
            
    # If cues are exhausted, choice is based on final evidence E
    if p_active > 0:
        z_final = beta * E
        if z_final > 20:
            p_a_exhaust = 1.0
        elif z_final < -20:
            p_a_exhaust = 0.0
        else:
            p_a_exhaust = 1.0 / (1.0 + np.exp(-z_final))
            
        p_a_total += p_active * p_a_exhaust
        p_b_total += p_active * (1.0 - p_a_exhaust)
        
    p_core = np.array([p_a_total, p_b_total])
    
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
[0] rationale: This metric isolates 'alternating' trials where the option that wins the most valid cue (Cue 1) ultimately loses the unweighted tally (e.g., winning 2 cues vs 3), and where the running evidence difference never exceeds 1. In the Competing model, the tally threshold is never triggered, forcing it to fall back on the final unweighted tally, thus choosing the Cue 1 loser. In the Advocated model, probabilistic stopping heavily favors the early leader, and validity weighting strongly favors Cue 1, so it predominantly chooses the Cue 1 winner. The proportion of times the Cue 1 winner is chosen on these trials will be high for the Advocated model and low for the Competing model.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    cue1_winner_chosen = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        E = 0
        max_abs_E = 0
        for x, y in zip(a, b):
            E += (x - y)
            if abs(E) > max_abs_E:
                max_abs_E = abs(E)
                
        if max_abs_E <= 1:
            a_sum = sum(a)
            b_sum = sum(b)
            
            if a[0] == 1 and b[0] == 0 and a_sum < b_sum:
                cue1_winner_chosen.append(1 if row['response'] == 0 else 0)
            elif b[0] == 1 and a[0] == 0 and b_sum < a_sum:
                cue1_winner_chosen.append(1 if row['response'] == 1 else 0)
                
    if not cue1_winner_chosen:
        return 0.5
        
    return float(np.mean(cue1_winner_chosen))
outcome: self_sim=0.7567 (var=0.0252) adversary_sim=0.6733 (var=0.0914) welch_t=+1.220 p=0.2303 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    t1_t5_choices = []\n    t2_t6_choices = []\n    \n    for _, row in data.iterrows():\n        a = tuple(row['option_a_ratings'])\n        b = tuple(row['option_b_ratings'])\n        \n        if a[0] == 1 and b[0] == 0:\n            cue1_winner = a\n            chose_cue1_winner = (row['response'] == 0)\n        elif b[0] == 1 and a[0] == 0:\n            cue1_winner = b\n            chose_cue1_winner = (row['response'] == 1)\n        else:\n            continue\n            \n        if cue1_winner == (1, 1, 0, 0, 0):\n            t1_t5_choices.append(1 if chose_cue1_winner else 0)\n        elif cue1_winner == (1, 0, 1, 0, 0):\n            t2_t6_choices.append(1 if chose_cue1_winner else 0)\n            \n    if not t1_t5_choices or not t2_t6_choices:\n        return 0.0\n        \n    return float(np.mean(t1_t5_choices) - np.mean(t2_t6_choices))",
  "rationale": "This metric measures the difference in the probability of choosing the Cue 1 winner between trials where the Cue 1 winner also wins Cue 2 (Trials 1 & 5) and trials where it loses Cue 2 but wins Cue 3 (Trials 2 & 6). In the Competing model, winning consecutive cues rapidly builds the unweighted evidence difference to 2, heavily triggering the stopping threshold and resulting in a high probability of choosing the Cue 1 winner. When wins alternate, the evidence difference fluctuates between 0 and 1, rarely triggering the threshold and forcing a fallback to the final tally (which heavily favors the Cue 1 loser), resulting in a massive drop in choosing the Cue 1 winner. The Advocated model, however, probabilistically stops at any discriminating cue regardless of the running tally; thus, the difference in choice probabilities between these two trial types is positive but much smaller."
}
```

## Usage

```json
{
  "prompt_token_count": 4109,
  "candidates_token_count": 558,
  "total_token_count": 13679
}
```
