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
**Validities (n_features=5):** [0.95, 0.94, 0.93, 0.92, 0.91]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 4: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Rationale:** To quantitatively dissociate the Strategy Mixture Theory (WADD, Tallying, TTB) from the Rank-based Weighting Theory, we exploit their fundamentally different treatments of cue validities. By selecting a set of high, closely-spaced validities (e.g., 0.95 to 0.91), the log-odds weights used by WADD (a core component of the Strategy Mixture) remain relatively flat and large. In contrast, Rank-based Weighting ignores the absolute validity values and strictly enforces a steep decay based on ordinal rank (e.g., proportional to 1/rank). We design critical trials where Option A possesses the 1st-ranked cue and the 5th-ranked cue, while Option B possesses the 2nd, 3rd, and 4th-ranked cues. WADD heavily favors Option B because the sum of three large log-odds easily outweighs the sum of two. Tallying also favors Option B (3 cues vs 2). Thus, the Strategy Mixture Theory strongly predicts a preference for Option B. However, for a moderate to high decay parameter, Rank-based Weighting predicts that the 1st-ranked cue alone carries enough weight to make Option A the winner (e.g., 1 + 1/5 > 1/2 + 1/3 + 1/4). This creates a sharp divergence in the predicted choice probabilities between the two theories.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
**Description:** Rank-based Weighting Theory posits that decision-makers do not use complex mathematical transformations like log-odds to weigh evidence. Instead, they rely on the simple ordinal ranking of cue validities. Cues are weighted according to an inverse function of their rank (proportional to 1/rank^rho). By restricting the decay parameter rho to a moderate range, the theory maintains a highly compensatory mechanism that prevents over-reliance on the most valid cue, capturing the softer probability matching observed in human multi-attribute decision making.

**Parameters:**
- beta: [0.01, 15.0]
- epsilon: [0.0, 0.5]
- rho: [0.0, 2.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute ranks (1 is the highest validity)
    # Using stable sort for consistent tie-breaking if validities are equal
    order = np.argsort(-val, kind='stable')
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(val) + 1)
    
    # Apply rank-based weighting
    rho = float(parameters["rho"])
    w = 1.0 / (ranks ** rho)
    
    # Compute weighted sum of features for each option
    scores = np.dot(stim, w)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
[0] rationale: This metric calculates the proportion of times Option B is chosen on critical trials where Option A possesses the highest-ranked cue but Option B possesses a strictly greater number of less-valid cues. Under the Strategy Mixture Theory, because the validities are closely spaced, the log-odds weights for WADD are relatively flat, meaning WADD and Tallying heavily favor Option B (which has more cues). Thus, the Strategy Mixture Theory predicts a high proportion of Option B choices. Conversely, Rank-based Weighting Theory applies a steep decay based strictly on ordinal rank, meaning the 1st cue alone can often outweigh several lower-ranked cues, predicting a much lower proportion of Option B choices (i.e., a preference for Option A).
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    b_chosen = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        # Select trials where Option A has the most valid cue (cue 1),
        # Option B lacks it, but Option B has a greater total number of cues.
        if a[0] == 1 and b[0] == 0 and sum(b) > sum(a):
            b_chosen.append(row['response'])
            
    if len(b_chosen) == 0:
        return 0.5
    return float(np.mean(b_chosen))
outcome: self_sim=0.5903 (var=0.0233) adversary_sim=0.4831 (var=0.0915) welch_t=+1.582 p=0.1225 (N=25, alpha=0.01) -> reject

[1] rationale: This metric calculates the difference in the probability of choosing the option with the 1st cue when it also has the 4th cue (Trial 3) versus when it stands alone (Trial 2), against a constant alternative (cues 2, 3, and 5). Under the Strategy Mixture Theory, all three heuristics yield identical categorical predictions for both trials: TTB strictly favors the 1st cue, while Tallying and WADD strictly favor the alternative (since 3 cues beat 2, and the sum of validities for 2,3,5 heavily outweighs 1,4). Because the margins are large, the softmax probabilities are saturated, predicting a difference very close to 0. In contrast, Rank-based Weighting is highly sensitive to the addition of the 4th cue (which adds 1/4^rho to the value), meaningfully shifting the choice probability and predicting a strictly positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    t3_a = []
    t2_a = []
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 3: A has cues 1, 4; B has cues 2, 3, 5
        if a == (1, 0, 0, 1, 0) and b == (0, 1, 1, 0, 1):
            t3_a.append(1 if resp == 0 else 0)
        elif b == (1, 0, 0, 1, 0) and a == (0, 1, 1, 0, 1):
            t3_a.append(1 if resp == 1 else 0)
            
        # Trial 2: A has cue 1; B has cues 2, 3, 5
        if a == (1, 0, 0, 0, 0) and b == (0, 1, 1, 0, 1):
            t2_a.append(1 if resp == 0 else 0)
        elif b == (1, 0, 0, 0, 0) and a == (0, 1, 1, 0, 1):
            t2_a.append(1 if resp == 1 else 0)
            
    m3 = np.mean(t3_a) if len(t3_a) > 0 else 0.5
    m2 = np.mean(t2_a) if len(t2_a) > 0 else 0.5
    return float(m3 - m2)
outcome: self_sim=0.0275 (var=0.0267) adversary_sim=0.0950 (var=0.0380) welch_t=-1.326 p=0.1912 (N=25, alpha=0.01) -> reject

[2] rationale: We construct a metric `P(A|Trial 4) - 0.5 * P(A|Trial 1)` to perfectly cancel out the high variance introduced by the mixture weight of TTB in the Strategy Mixture theory. On Trial 4, WADD yields a near-tie (difference of 0.036 in log-odds) and Tallying yields an exact tie, meaning their choice probabilities are ~0.5. TTB strongly favors A. Thus, P(A|T4) ≈ 0.5 + 0.5 * w_ttb. On Trial 1, WADD and Tallying strongly favor B, so their choice probabilities for A are ~0. TTB again strongly favors A. Thus, P(A|T1) ≈ w_ttb. By computing `P(A|T4) - 0.5 * P(A|T1)`, the `w_ttb` term is mathematically eliminated, yielding a theoretical constant of ~0.5 for the Strategy Mixture theory (slightly lowered by epsilon, yielding ~0.44). This results in an extremely low between-subject variance. In contrast, the Rank-based Weighting theory predicts a significantly lower value for this metric (~0.30), and this value is remarkably stable across its parameter space (both `rho` and `beta`), ensuring low variance for the competing theory as well. The large mean difference combined with tiny variances guarantees strong discriminability.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    t4_a = []
    t1_a = []
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 4: A=[0, 1, 0, 0, 1], B=[0, 0, 1, 1, 0]
        if a == (0, 1, 0, 0, 1) and b == (0, 0, 1, 1, 0):
            t4_a.append(1 if resp == 0 else 0)
        elif b == (0, 1, 0, 0, 1) and a == (0, 0, 1, 1, 0):
            t4_a.append(1 if resp == 1 else 0)
            
        # Trial 1: A=[1, 0, 0, 0, 1], B=[0, 1, 1, 1, 0]
        if a == (1, 0, 0, 0, 1) and b == (0, 1, 1, 1, 0):
            t1_a.append(1 if resp == 0 else 0)
        elif b == (1, 0, 0, 0, 1) and a == (0, 1, 1, 1, 0):
            t1_a.append(1 if resp == 1 else 0)
            
    p_t4 = np.mean(t4_a) if len(t4_a) > 0 else 0.5
    p_t1 = np.mean(t1_a) if len(t1_a) > 0 else 0.5
    
    return float(p_t4 - 0.5 * p_t1)
outcome: self_sim=0.4581 (var=0.0140) adversary_sim=0.4050 (var=0.0336) welch_t=+1.217 p=0.2304 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    high_trials = []\n    low_trials = []\n    \n    for _, row in data.iterrows():\n        a = row['option_a_ratings']\n        b = row['option_b_ratings']\n        resp = row['response']\n        \n        # Find the first cue where the two options differ\n        for i in range(len(a)):\n            if a[i] != b[i]:\n                # Identify if the subject chose the option possessing this highest available cue\n                chose_opt1 = (resp == 0) if a[i] == 1 else (resp == 1)\n                \n                # 'Low' trials: the highest available cue is the 1st cue (index 0)\n                if i == 0:\n                    low_trials.append(1 if chose_opt1 else 0)\n                # 'High' trials: the highest available cue is the 2nd cue (index 1)\n                elif i == 1:\n                    high_trials.append(1 if chose_opt1 else 0)\n                break\n                \n    p_high = np.mean(high_trials) if len(high_trials) > 0 else 0.5\n    p_low = np.mean(low_trials) if len(low_trials) > 0 else 0.5\n    \n    return float(p_high - p_low)",
  "rationale": "This metric contrasts the probability of choosing the option favored by the Take-The-Best (TTB) heuristic on two distinct sets of trials. On 'low' trials (T1, T2, T3, T5), the best available distinguishing cue is the 1st cue, but the alternative option possesses a dense cluster of less-valid cues. Under the Strategy Mixture Theory (SMT), the WADD and Tallying heuristics strongly favor the alternative option on these trials, meaning the TTB-favored option is chosen almost exclusively when the TTB heuristic is explicitly sampled. Thus, P(low) is relatively low (~0.4). On 'high' trials (T4, T6), the best available cue is the 2nd cue, and WADD/Tallying now favor (or tie) the TTB-favored option, leading to a massive surge in its choice probability (P(high) ~0.8). SMT therefore predicts a large, strictly positive difference (P_high - P_low ~ +0.4). In contrast, Rank-based Weighting (RBW) evaluates options holistically and places immense weight on the 1st cue. It often favors the 1st-cue option on 'low' trials (especially for rho > 0), leading to a high P(low). On 'high' trials, the 2nd cue is less dominant, yielding a moderate P(high). Consequently, RBW predicts this difference to be near zero or even negative. By aggregating across 4 'low' and 2 'high' trials, binomial noise is heavily suppressed, ensuring the gap is highly discriminable."
}
```

## Usage

```json
{
  "prompt_token_count": 5727,
  "candidates_token_count": 719,
  "total_token_count": 37895
}
```
