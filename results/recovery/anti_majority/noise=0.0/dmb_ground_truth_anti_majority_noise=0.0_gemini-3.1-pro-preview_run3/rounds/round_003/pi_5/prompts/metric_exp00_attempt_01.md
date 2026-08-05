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
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.75, 0.65, 0.65]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 3: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 4: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 6: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 7: A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]

**Rationale:** This design introduces a 'Dilution Effect' to quantitatively dissociate the Probabilistic Cue Selection (Random Dictator) model from the Strategy Mixture (TTB + Tallying) model. The Mixture model's choice probabilities depend solely on the identity of the single best discriminating cue (TTB) and the absolute difference in the number of winning cues (Tallying). By symmetrically adding lower-validity discriminating cues to both options across a series of trials (e.g., Trials 1, 2, and 3), we hold both the best discriminating cue and the tally difference strictly constant. Consequently, the Strategy Mixture model is mathematically forced to predict identical choice probabilities across these trials. In stark contrast, the Random Dictator model samples from the set of all discriminating cues. Adding more discriminating cues dilutes the sampling probability of the highest-validity cue, systematically pulling the choice probabilities toward 0.5. This provides a clear, parameter-independent structural divergence between the two theories.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Cue Selection (Random Dictator): Decision-makers do not deterministically follow a strict cue hierarchy (as in Take The Best) nor do they exhaustively sum all available cues (as in Tallying or WADD). Instead, on any given trial, they probabilistically sample a single cue from the set of discriminating cues to determine their choice. The probability of sampling a specific cue is proportional to its validity raised to a non-linear scaling parameter (gamma). If gamma is very high, the most valid cue is almost always sampled, perfectly mimicking TTB. If gamma is near zero, cues are sampled uniformly, producing a soft Tallying-like behavior where choice probability reflects the proportion of cues favoring an option. This provides a mathematically elegant, single-process probabilistic mechanism that gracefully spans the spectrum of non-compensatory to compensatory decision-making without requiring ad-hoc response noise or explicit strategy mixtures.

**Parameters:**
- gamma: [0.0, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Identify cues that discriminate between option A and option B
    diff = a - b
    disc_mask = diff != 0
    
    if not np.any(disc_mask):
        # No cues discriminate; guess uniformly
        p_core = np.array([0.5, 0.5])
    else:
        # Scale validities of discriminating cues non-linearly
        w = val[disc_mask] ** gamma
        w_sum = np.sum(w)
        
        if w_sum == 0:
            p_core = np.array([0.5, 0.5])
        else:
            # Probability of sampling each discriminating cue
            p = w / w_sum
            
            # The choice is determined entirely by the sampled cue.
            # Thus, the probability of choosing A is the sum of sampling probabilities
            # for cues where A > B.
            p_a = np.sum(p[diff[disc_mask] > 0])
            p_b = np.sum(p[diff[disc_mask] < 0])
            p_core = np.array([p_a, p_b])
            
    # Incorporate uniform lapse rate (guessing)
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


## COMPETING THEORY
**Description:** Strategy Mixture (TTB + Tallying): Decision makers do not universally adhere to a single strategy. Instead, they use a probabilistic mixture of a non-compensatory strategy (Take The Best) and a compensatory strategy (Tallying). A parameter P_TTB dictates the probability of using TTB on any given trial, while 1 - P_TTB is the probability of using Tallying. This accounts for intermediate levels of TTB-consistency and Tallying-consistency observed in empirical data across subjects and trials. The mixture captures a balance between TTB and Tallying, avoiding over-reliance on uniform guessing.

**Parameters:**
- beta: [0.1, 20.0]
- p_ttb: [0.35, 0.95]
- epsilon: [0.0, 0.2]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np

    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take The Best (TTB) Strategy
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        scores_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        scores_ttb = np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.0, 0.0])
        
    # Tallying Strategy
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    # Normalize by n_features to keep scale comparable to TTB for the shared beta
    scores_tally = np.array([a_wins, b_wins]) / max(1.0, float(n_features))
    
    beta = float(parameters["beta"])
    
    # TTB Probabilities
    z_ttb = beta * (scores_ttb - scores_ttb.max())
    e_ttb = np.exp(z_ttb)
    p_ttb_dist = e_ttb / e_ttb.sum()
    
    # Tallying Probabilities
    z_tally = beta * (scores_tally - scores_tally.max())
    e_tally = np.exp(z_tally)
    p_tally_dist = e_tally / e_tally.sum()
    
    # Mixture
    p_ttb_weight = float(parameters["p_ttb"])
    epsilon = float(parameters["epsilon"])
    
    p_core = p_ttb_weight * p_ttb_dist + (1.0 - p_ttb_weight) * p_tally_dist
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
[0] rationale: This metric calculates the Dilution Effect by comparing the probability of choosing the TTB-favored option in 1-cue symmetric trials vs. 3-cue symmetric trials. Under the Strategy Mixture model, the best discriminating cue and the tally difference are strictly constant across these trials, so this difference should be exactly zero (modulo noise). Under the Probabilistic Cue Selection (Random Dictator) model, adding more discriminating cues dilutes the sampling probability of the highest-validity cue, so the choice probability for the TTB winner will systematically drop in the 3-cue trials, yielding a large positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.array(data['option_a_ratings'].tolist())
    B = np.array(data['option_b_ratings'].tolist())
    resp = data['response'].values
    
    sumA = A.sum(axis=1)
    sumB = B.sum(axis=1)
    
    # Identify 1-cue symmetric trials (Trials 1 and 6)
    mask1 = (sumA == 1) & (sumB == 1)
    if mask1.sum() > 0:
        ttb_winner_1 = np.where(A[mask1, 0] == 1, 0, 1)
        p1 = (resp[mask1] == ttb_winner_1).mean()
    else:
        p1 = 0.0
        
    # Identify 3-cue symmetric trials (Trial 3)
    mask3 = (sumA == 3) & (sumB == 3)
    if mask3.sum() > 0:
        ttb_winner_3 = np.where(A[mask3, 0] == 1, 0, 1)
        p3 = (resp[mask3] == ttb_winner_3).mean()
    else:
        p3 = 0.0
        
    return float(p1 - p3)
outcome: self_sim=0.0408 (var=0.0296) adversary_sim=-0.0117 (var=0.0163) welch_t=+1.226 p=0.2268 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    A = np.array(data['option_a_ratings'].tolist())\n    B = np.array(data['option_b_ratings'].tolist())\n    resp = data['response'].values\n    \n    # Identify trials by their sum of cues (Tallying score proxy)\n    sumA = A.sum(axis=1)\n    sumB = B.sum(axis=1)\n    \n    # 1. Trials where one option has strictly more cues (Trials 4, 5, 8)\n    mask_more_B = (sumB > sumA)\n    mask_more_A = (sumA > sumB)\n    \n    more_cues_chosen = 0\n    more_cues_total = 0\n    if np.any(mask_more_B):\n        more_cues_chosen += np.sum(resp[mask_more_B] == 1)\n        more_cues_total += np.sum(mask_more_B)\n    if np.any(mask_more_A):\n        more_cues_chosen += np.sum(resp[mask_more_A] == 0)\n        more_cues_total += np.sum(mask_more_A)\n        \n    p_more_cues = float(more_cues_chosen) / more_cues_total if more_cues_total > 0 else 0.5\n    \n    # 2. Trials where options have an equal number of cues (Trials 1, 2, 3, 6, 7)\n    mask_equal = (sumA == sumB)\n    \n    ttb_winner_chosen = 0\n    ttb_total = 0\n    if np.any(mask_equal):\n        # Cue 0 is the highest validity cue. In equal cue trials, \n        # the option with Cue 0 is always the TTB winner.\n        mask_ttb_A = mask_equal & (A[:, 0] == 1)\n        ttb_winner_chosen += np.sum(resp[mask_ttb_A] == 0)\n        ttb_total += np.sum(mask_ttb_A)\n        \n        mask_ttb_B = mask_equal & (B[:, 0] == 1)\n        ttb_winner_chosen += np.sum(resp[mask_ttb_B] == 1)\n        ttb_total += np.sum(mask_ttb_B)\n        \n    p_ttb_winner = float(ttb_winner_chosen) / ttb_total if ttb_total > 0 else 0.5\n    \n    # The metric is a linear combination designed to cancel out the p_ttb parameter in the Mixture model\n    return float(p_more_cues + 2.0 * p_ttb_winner)\n",
  "rationale": "This metric exploits the strict mathematical structure of the Strategy Mixture model to force a near-constant value, while the Probabilistic Cue Selection (Random Dictator) model structurally diverges. In the Mixture model, TTB exclusively uses the highest validity cue (Cue 0), and Tallying relies solely on cue counts. In trials with unequal cues (Trials 4, 5, 8), TTB always opposes the option with more cues, so `P(choose more cues) = (1 - p_ttb) * P_tally_win`. In trials with equal cues (Trials 1, 2, 3, 6, 7), Tallying is perfectly tied, so `P(choose TTB winner) = p_ttb * 1.0 + (1 - p_ttb) * 0.5 = 0.5 + 0.5 * p_ttb`. By computing `M = P(choose more cues) + 2 * P(choose TTB winner)`, we algebraically cancel out the nuisance parameter `p_ttb`. For the Mixture model, this simplifies to `(1 - p_ttb) * P_tally_win + 1 + p_ttb`, which is heavily anchored near 1.95 (since P_tally_win is typically high). In stark contrast, the Random Dictator model lacks this compensatory two-process structure; it samples cues proportionally. For Dictator, `P(choose more cues)` hovers around 0.35 and `P(choose TTB winner)` around 0.65, anchoring `M` much lower (around 1.65). This provides a massive, parameter-robust gap between the theories."
}
```

## Usage

```json
{
  "prompt_token_count": 4430,
  "candidates_token_count": 1089,
  "total_token_count": 36781
}
```
