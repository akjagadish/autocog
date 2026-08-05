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
**Validities (n_features=6):** [0.95, 0.95, 0.55, 0.55, 0.55, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 2: A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 4: A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate Tallying-Biased Parity from pure Cue-Parity, we exploit the alignment between the primary Cue-Parity rule and the secondary validity-weighted tallying fallback. Pure Cue-Parity predicts that the probability of choosing A depends exclusively on whether the number of A-wins is odd or even, plus a constant uniform lapse rate. It is completely blind to the validities of the winning features. Tallying-Biased Parity, however, mixes the strict parity rule with a small probability of validity-weighted tallying. By designing pairs of trials where the unweighted parity of A-wins is identical (e.g., A wins 2 features, which is Even -> Parity favors B) but the validity-weighted tally is reversed (Trial A: A wins on 2 high-validity cues vs B's 3 low-validity cues; Trial B: A wins on 2 low-validity cues vs B's 2 high-validity cues), we can observe a systematic shift. Tallying-Biased Parity predicts a larger choice share for the parity-disfavored option when the tallying fallback supports it. Pure Cue-Parity predicts perfectly flat choice probabilities across these paired trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Tallying-Biased Parity (Strict Parity Dominance): Decision makers primarily rely on a strict Cue-Parity rule (preferring the option with an odd number of winning cues). However, instead of lapsing into purely uniform random noise, their secondary considerations consist of occasionally falling back to a standard validity-weighted tallying strategy. To ensure that tallying acts only as a weak bias and does not override parity in cases of option dominance, the fallback probability is strictly bounded to a low range.

**Parameters:**
- validities: validities
- beta_parity: [0.1, 20.0]
- beta_tally: [0.1, 10.0]
- epsilon: [0.0, 0.3]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying-Biased Parity expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Core Strategy: Strict Cue-Parity
    a_wins = int(np.sum(a > b))
    parity_winner = 0 if (a_wins % 2 == 1) else 1
    parity_scores = np.array([1.0, 0.0]) if parity_winner == 0 else np.array([0.0, 1.0])
    
    # Fallback Strategy: Validity-weighted Tallying
    val_a = np.sum(validities * (a > b))
    val_b = np.sum(validities * (b > a))
    tally_scores = np.array([val_a, val_b])
    
    beta_parity = float(parameters["beta_parity"])
    beta_tally = float(parameters["beta_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Parity probabilities via softmax
    z_p = beta_parity * (parity_scores - parity_scores.max())
    e_p = np.exp(z_p)
    p_parity = e_p / e_p.sum()
    
    # Tallying probabilities via softmax
    z_t = beta_tally * (tally_scores - tally_scores.max())
    e_t = np.exp(z_t)
    p_tally = e_t / e_t.sum()
    
    # Mixture of the two strategies
    return (1.0 - epsilon) * p_parity + epsilon * p_tally
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Identify the trial types based on option A's features.\n    # T1: sum=1, A[0]=1 (Parity favors A, Tally favors A)\n    # T2: sum=1, A[0]=0 (Parity favors A, Tally favors B)\n    # T3: sum=2, A[0]=1 (Parity favors B, Tally favors A)\n    # T4: sum=2, A[0]=0 (Parity favors B, Tally favors B)\n    # T5: sum=3, A[0]=1 (Parity favors A, Tally favors A)\n    # T6: sum=3, A[0]=0 (Parity favors A, Tally favors B)\n    \n    def get_trial_type(row):\n        a = row['option_a_ratings']\n        s = sum(a)\n        if s == 1:\n            return 'T1' if a[0] == 1 else 'T2'\n        elif s == 2:\n            return 'T3' if a[0] == 1 else 'T4'\n        elif s == 3:\n            return 'T5' if a[0] == 1 else 'T6'\n        return 'Other'\n        \n    data = data.copy()\n    data['trial_type'] = data.apply(get_trial_type, axis=1)\n    data['chose_A'] = (data['response'] == 0).astype(float)\n    \n    # Calculate mean P(choose A) for each trial type\n    means = data.groupby('trial_type')['chose_A'].mean()\n    \n    def get_mean(t):\n        return means.get(t, 0.0)\n        \n    # Difference in P(choose A) when Tally favors A vs when Tally favors B,\n    # holding Parity constant.\n    diff_1 = get_mean('T1') - get_mean('T2')\n    diff_2 = get_mean('T3') - get_mean('T4')\n    diff_3 = get_mean('T5') - get_mean('T6')\n    \n    return float(diff_1 + diff_2 + diff_3)\n",
  "rationale": "This metric calculates the difference in the probability of choosing option A between pairs of trials that have the exact same Cue-Parity status but opposite Tallying (validity-weighted) status. Pure Cue-Parity predicts these differences will be exactly zero on average, as it ignores validities. Tallying-Biased Parity predicts a strictly positive value because the fallback tallying strategy will systematically increase the probability of choosing A when the high-validity cues favor A compared to when they favor B."
}
```

## Usage

```json
{
  "prompt_token_count": 3736,
  "candidates_token_count": 684,
  "total_token_count": 6710
}
```
