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
**Validities (n_features=5):** [0.92, 0.85, 0.78, 0.71, 0.65]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 4: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 6: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 7: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 8: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Rationale:** To quantitatively dissociate Probabilistic Sequential Search Theory (PSS) from Strategy Mixture Theory (SMT), we exploit PSS's step-by-step evidence accumulation mechanism versus SMT's fixed mixture of global heuristics (WADD, Tallying, TTB). In PSS, evidence is updated cue-by-cue, and the decision-maker can stop at any discriminating cue with a constant probability. This means that if early cues strongly favor one option, the accumulated evidence might remain in favor of that option at every possible stopping point, leading to a very high choice probability. In contrast, SMT computes the overall WADD and Tallying scores across all cues simultaneously. By designing trials where the first cue favors Option B, but subsequent cues favor Option A such that the running evidence sum in PSS only flips to Option A late in the sequence, PSS predicts a higher probability of choosing Option B (due to early stopping) than SMT, which sees a clear victory for Option A in both WADD and Tallying. We systematically vary the position and validities of the opposing cues to create a spectrum of divergences between early-stopping accumulation and global heuristic mixtures.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Sequential Search Theory posits that decision-makers evaluate cues sequentially in order of their validity. Instead of adopting a strict stopping rule like Take-The-Best or exhaustively evaluating all cues like WADD, individuals accumulate evidence dynamically. When a cue discriminates between options, there is a constant probability (stop_rate) of stopping the search and deciding based on the accumulated evidence. This creates a flexible, cue-by-cue evidence accumulation process that naturally blends lexicographic and compensatory behaviors, explaining the softer choice probabilities and context-dependent trade-offs seen in human decision-making without forcing an overly aggressive stopping rule.

**Parameters:**
- stop_rate: [0.0, 1.0]
- beta: [0.0, 10.0]
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    
    # Sort cues by validity in descending order
    cue_order = np.argsort(-val_clipped, kind="stable").tolist()
    
    stop_rate = float(parameters["stop_rate"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    P_A = 0.0
    P_B = 0.0
    P_continue = 1.0
    E = 0.0
    
    n_features = len(val)
    
    for i, idx in enumerate(cue_order):
        diff = a[idx] - b[idx]
        w = np.log(val_clipped[idx] / (1.0 - val_clipped[idx]))
        E += w * diff
        
        # Determine stopping probability at this step
        if i == n_features - 1:
            S_t = 1.0
        else:
            if diff != 0:
                # Cue discriminates: stop probability is constant stop_rate
                S_t = stop_rate
            else:
                # Cue does not discriminate: always continue
                S_t = 0.0
                
        # Probability of choosing A if search stops here (using stable sigmoid)
        z = beta * E
        if z > 20:
            p_A_if_stop = 1.0
        elif z < -20:
            p_A_if_stop = 0.0
        else:
            p_A_if_stop = 1.0 / (1.0 + np.exp(-z))
            
        p_B_if_stop = 1.0 - p_A_if_stop
        
        # Accumulate overall choice probabilities
        P_A += P_continue * S_t * p_A_if_stop
        P_B += P_continue * S_t * p_B_if_stop
        
        # Update probability of continuing to the next cue
        P_continue *= (1.0 - S_t)
        
    # Apply lapse rate (epsilon)
    P_A = (1.0 - epsilon) * P_A + epsilon * 0.5
    P_B = (1.0 - epsilon) * P_B + epsilon * 0.5
    
    return np.array([P_A, P_B])
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
**Description:** Strategy Mixture Theory with Independent Scaling: Individuals use a probabilistic mixture of distinct heuristics (WADD, Tallying, and Take-The-Best), but because the internal evidence scales of these heuristics vary dramatically (log-odds sums vs. integer counts vs. binary indicators), each heuristic applies its own independent temperature parameter to properly calibrate its choice probabilities before mixing.

**Parameters:**
- beta_wadd: [0.1, 10.0]
- beta_tally: [0.1, 10.0]
- beta_ttb: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- w_wadd: [0.0, 1.0]
- w_tally: [0.0, 1.0]
- w_ttb: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    w = np.log(val_clipped / (1.0 - val_clipped))
    scores_wadd = np.dot(stim, w)
    
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
    if winner is None:
        scores_ttb = np.array([0.0, 0.0])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        
    beta_wadd = float(parameters["beta_wadd"])
    beta_tally = float(parameters["beta_tally"])
    beta_ttb = float(parameters["beta_ttb"])
    
    def get_probs(scores, beta):
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        return e / np.sum(e)
        
    p_wadd = get_probs(scores_wadd, beta_wadd)
    p_tally = get_probs(scores_tally, beta_tally)
    p_ttb = get_probs(scores_ttb, beta_ttb)
    
    w1 = float(parameters["w_wadd"])
    w2 = float(parameters["w_tally"])
    w3 = float(parameters["w_ttb"])
    w_sum = w1 + w2 + w3 + 1e-9
    
    p_mix = (w1 * p_wadd + w2 * p_tally + w3 * p_ttb) / w_sum
    
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    \n    t1_mask = data['A_str'] == '01111'\n    t2_mask = data['A_str'] == '10000'\n    t3_mask = data['A_str'] == '01100'\n    t4_mask = data['A_str'] == '10011'\n    \n    p_cue1_t1 = (data.loc[t1_mask, 'response'] == 1).mean()\n    p_cue1_t2 = (data.loc[t2_mask, 'response'] == 0).mean()\n    p_cue1_t3 = (data.loc[t3_mask, 'response'] == 1).mean()\n    p_cue1_t4 = (data.loc[t4_mask, 'response'] == 0).mean()\n    \n    p_12 = np.nanmean([p_cue1_t1, p_cue1_t2])\n    p_34 = np.nanmean([p_cue1_t3, p_cue1_t4])\n    \n    if np.isnan(p_12): p_12 = 0.5\n    if np.isnan(p_34): p_34 = 0.5\n    \n    return float(p_34 - p_12)",
  "rationale": "The metric computes the difference in the probability of choosing the option that is favored by the most valid cue (Cue 1) between two sets of trials. In Trials 1 and 2, the Cue 1 winner is opposed by all other cues, meaning WADD and Tallying strongly favor the opposite option. In Trials 3 and 4, the Cue 1 winner is also supported by WADD and Tallying. Strategy Mixture Theory predicts a large difference in choice probabilities here, because Trials 1 and 2 rely solely on the Take-The-Best heuristic to choose the Cue 1 winner, whereas Trials 3 and 4 have all heuristics aligned. Probabilistic Sequential Search, on the other hand, accumulates evidence step-by-step and may show a different magnitude of this alignment effect due to its early stopping mechanism."
}
```

## Usage

```json
{
  "prompt_token_count": 4104,
  "candidates_token_count": 580,
  "total_token_count": 8370
}
```
