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
**Validities (n_features=6):** [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate Continuous Weighted Parity from Cue-Parity, we manipulate whether option A wins on high-validity or low-validity features while holding the total number of A-wins constant. Cue-Parity strictly calculates the unweighted integer count of A-wins, meaning it makes identical, purely deterministic predictions (ignoring response noise) for any two trials where A wins on the same number of features, regardless of their validities. In contrast, Continuous Weighted Parity weights each win by its validity. When A wins on high-validity cues, the weighted sum of wins deviates upward from the integer count; when it wins on low-validity cues, it deviates downward. This pushes the parity score toward the 'dead zones' (half-integers) of the cosine function, systematically attenuating the choice probabilities in Continuous Weighted Parity. By comparing trials with the same number of A-wins but different validities, we can isolate this continuous, validity-driven attenuation predicted by Continuous Weighted Parity but entirely absent in Cue-Parity.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Continuous Weighted Parity (Restricted Parameter): Decision makers compute a parity-like signal, but the contribution of each cue is weighted by its validity. The evidence is integrated into a continuous sum of validity-weighted wins, which is then passed through a periodic parity function (negative cosine). To avoid chaotic phase shifts that destroy the parity signal, the influence of validity weights is kept small, allowing them to gently perturb the parity score without pushing the sum into the 'dead zones' (half-integers) of the cosine function.

**Parameters:**
- validities: validities
- w_validity: [0.0, 0.5]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Continuous Weighted Parity expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    w_validity = float(parameters["w_validity"])
    
    # Calculate weights for each cue based on validity
    mean_val = np.mean(validities)
    weights = 1.0 + w_validity * (validities - mean_val)
    
    # Compute weighted sum of strict A wins
    a_wins_mask = (a > b).astype(float)
    weighted_wins = np.sum(weights * a_wins_mask)
    
    # Continuous parity score: maps even integers to -1 (favoring B) and odd integers to +1 (favoring A)
    parity_score = -np.cos(np.pi * weighted_wins)
    
    # Scores for A and B
    score_a = parity_score
    score_b = -parity_score
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax conversion with numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** People decide by the PARITY of the evidence rather than its weight or direction. They count the number of cues on which option A strictly beats option B, and prefer A when that count is ODD and B when it is EVEN (zero counts as even). This is a deliberately non-monotone, XOR-like rule: adding one more cue in A's favor flips the preference rather than strengthening it, so option dominance does NOT imply choice — an option that wins on every cue (an even count, when the cue number is even) is rejected. The rule uses no validities and no magnitudes, only the parity of feature-wise wins, which makes it an adversarially hard recovery target: it is uncorrelated with any single cue and with the validity-weighted sum, yet perfectly deterministic, much like the anti-majority ensemble. Response noise enters through a softmax over the binary score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Cue-Parity: prefer A iff the number of cues where A strictly beats B is
    # ODD, else prefer B (zero A-wins is even -> B). Non-monotone in the
    # evidence; uses neither validities nor magnitudes. Stimulus is the pair
    # of option feature vectors: (2, n_features), row 0 = A, row 1 = B.
    # History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Cue-Parity expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    a_wins = int(np.sum(a > b))
    # Odd number of A-favoring cues -> A (0); even (including zero) -> B (1).
    winner = 0 if (a_wins % 2 == 1) else 1

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the binary
    # score this collapses to sigmoid(beta) for the winner (beta=0 ↔ 50/50;
    # beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    # response == 0 means A was chosen\n    chose_a = (data['response'] == 0).astype(float)\n    \n    # Extract A and B ratings as 2D arrays\n    a_ratings = np.stack(data['option_a_ratings'].values)\n    b_ratings = np.stack(data['option_b_ratings'].values)\n    \n    # Calculate strict A wins and high-validity A wins (first 3 features)\n    a_wins_mask = (a_ratings > b_ratings).astype(int)\n    a_wins = a_wins_mask.sum(axis=1)\n    high_wins = a_wins_mask[:, :3].sum(axis=1)\n    \n    # Group 1: Trials where CWP score is slightly attenuated (magnitude = cos(0.25*pi*w))\n    # These are trials with 1 win total OR 3 wins with mixed validities.\n    g1_mask = (a_wins == 1) | ((a_wins == 3) & ((high_wins == 1) | (high_wins == 2)))\n    \n    # Group 2: Trials where CWP score is heavily attenuated (magnitude = cos(0.75*pi*w))\n    # These are trials with 3 wins all of the same validity type.\n    g2_mask = (a_wins == 3) & ((high_wins == 0) | (high_wins == 3))\n    \n    if not np.any(g1_mask) or not np.any(g2_mask):\n        return 0.0\n        \n    p_a_g1 = chose_a[g1_mask].mean()\n    p_a_g2 = chose_a[g2_mask].mean()\n    \n    return float(p_a_g1 - p_a_g2)",
  "rationale": "Continuous Weighted Parity predicts that the choice probabilities for A will be systematically attenuated based on the distance of the weighted sum to the nearest odd integer. For trials in Group 1, the deviation is 0.25 * w_validity, leading to a mild attenuation. For trials in Group 2, the deviation is 0.75 * w_validity, pushing the sum much closer to the 'dead zones' (half-integers) and causing severe attenuation. Thus, CWP predicts P(A | Group 1) > P(A | Group 2). Cue-Parity, however, only cares about the unweighted count of A-wins. Since both groups only contain trials with an odd number of A-wins (1 or 3), Cue-Parity produces identical, deterministic max-score differences for both groups, predicting a difference of roughly 0."
}
```

## Usage

```json
{
  "prompt_token_count": 3646,
  "candidates_token_count": 671,
  "total_token_count": 7990
}
```
