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

**Trial pairs (n=12):**
  trial 1: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 3: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 5: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 6: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 7: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 8: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 9: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 10: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]

**Rationale:** This design dissociates the advocated inverse weighted-tally plus validity-ordered tie-break theory from plain inverse tallying. Six validities are spread as [0.95, 0.85, 0.75, 0.65, 0.55, 0.50], so the advocated weak validity weighting (1.9-2.1) makes high-validity cues roughly 2.5 times as influential as low-validity cues. Six tie trials hold the raw inverse tally at exactly 0 (one A-clean and one B-clean discriminating feature). Competing inverse tallying must predict p(A)=0.5 for all six because the tallies are equal, whereas the advocated tie-break predicts p(A) near 0.71 when the first discriminating cue in descending validity order is A=0/B=1 and near 0.29 when it is A=1/B=0; the first discriminator is varied across features 0, 1, and 2. Six conflict trials hold the raw inverse tally at +1 or -1 but place the high-validity wins on the opposite side, so the advocated validity-weighted score reverses sign. For raw +1 trials the weighted scores are about -0.62, -0.42, and -0.22, predicting p(A) around 0.38-0.46, while competing inverse tallying predicts high p(A) around 0.65-0.85. For raw -1 trials the mirror arrangement gives advocated p(A) around 0.54-0.62, while competing inverse tallying predicts low p(A) around 0.15-0.35. Twelve unique pairs repeated K=8 gives 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People invert the valence of binary expert ratings: a 1 is treated as a costly defect and a 0 as a clean, favorable signal. Across all features they compute an inverse-tally advantage for A, weakly weighting the inverse cues by the experts' stated validities. When the raw inverse tally is nonzero, choice is a noisy softmax over the two weakly validity-weighted inverse tallies. When the raw inverse tally is exactly tied, people do not guess uniformly; instead they break the tie lexicographically by consulting features in descending validity order and choosing the option that has a clean 0 on the first discriminating feature. Response noise is captured by a softmax plus an independent uniform lapse. This differs from Take The Best because it aggregates all cues rather than stopping at the first discriminator, and from plain inverse tallying because ties are resolved by validity-ordered clean-cue use instead of guessing.

**Parameters:**
- beta_tally: [0.43, 0.47]
- beta_tiebreak: [1.00, 1.08]
- epsilon: [0.09, 0.11]
- validity_weight: [1.90, 2.10]
- validities: validities

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Expected a (2, n_features) stimulus, got {stim.shape}.')

    a = stim[0]
    b = stim[1]
    val = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    beta_tally = float(parameters['beta_tally'])
    beta_tiebreak = float(parameters['beta_tiebreak'])
    epsilon = float(parameters['epsilon'])
    validity_weight = float(parameters['validity_weight'])

    clean_a = (b > a)
    clean_b = (a > b)
    raw_s = float(np.sum(clean_a) - np.sum(clean_b))

    if raw_s != 0:
        if validity_weight > 0.0 and n_features > 1:
            weights = 1.0 + validity_weight * (val - np.mean(val))
        else:
            weights = np.ones(n_features)
        s = float(np.sum(weights * clean_a) - np.sum(weights * clean_b))
        scores = np.array([s, -s])
        beta = beta_tally
    else:
        cue_order = np.argsort(-val, kind='stable')
        winner = None
        for j in cue_order:
            if a[j] != b[j]:
                winner = 0 if (a[j] == 0 and b[j] == 1) else 1
                break
        if winner is None:
            return np.ones(2) / 2.0
        scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        beta = beta_tiebreak

    logits = beta * scores
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core = p_core / p_core.sum()

    n_opts = 2
    return (1.0 - epsilon) * p_core + epsilon * np.ones(n_opts) / n_opts
```

**`policy source code`:**
```python
import numpy as np

def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return 0
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** People invert the meaning of binary expert ratings: a rating of 1 is treated as a defect or costly negative signal, and a rating of 0 is treated as a clean or favorable signal. Across all features, they tally how often each option is defect-free relative to the other option, and choose the option with the larger defect-free tally. Ties lead to guessing. Choice probability is a noisy softmax over the two defect tallies with inverse temperature beta, plus an independent lapse process that produces uniform guessing with probability epsilon. The heuristic uses all features equally, ignores validities and past trials, and differs from Take The Best by aggregating rather than stopping at the first discriminating cue, and from standard Tallying by reversing the valence of the binary ratings.

**Parameters:**
- beta: [1.0, 5.0]
- epsilon: [0.30, 0.40]

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f'Inverse-tallying expects a (2, n_features) stimulus; got shape {stim.shape}.')

    a, b = stim[0], stim[1]

    # Defect-free wins: A has a 0 where B has a 1.
    a_wins_defects = float(np.sum(b > a))
    # Defect-free wins: B has a 0 where A has a 1.
    b_wins_defects = float(np.sum(a > b))

    scores = np.array([a_wins_defects, b_wins_defects])

    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])

    # Numerically stable softmax over inverse-cue tallies.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
import numpy as np

def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\nPAIR_SIGNS = {\n    ((0, 0, 0, 0, 0, 1), (1, 0, 0, 0, 0, 0)): 1.0,\n    ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)): -1.0,\n    ((0, 0, 0, 0, 0, 1), (0, 1, 0, 0, 0, 0)): 1.0,\n    ((0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)): -1.0,\n    ((0, 0, 0, 0, 0, 1), (0, 0, 1, 0, 0, 0)): 1.0,\n    ((0, 0, 1, 0, 0, 0), (0, 0, 0, 0, 0, 1)): -1.0,\n    ((1, 1, 0, 0, 0, 0), (0, 0, 0, 1, 1, 1)): -1.0,\n    ((1, 0, 0, 0, 0, 0), (0, 0, 0, 0, 1, 1)): -1.0,\n    ((1, 0, 1, 0, 0, 0), (0, 0, 0, 1, 1, 1)): -1.0,\n    ((0, 0, 0, 1, 1, 1), (1, 1, 0, 0, 0, 0)): 1.0,\n    ((0, 0, 0, 0, 1, 1), (1, 0, 0, 0, 0, 0)): 1.0,\n    ((0, 0, 0, 1, 1, 1), (1, 0, 1, 0, 0, 0)): 1.0,\n}\n\ndef metric(data: pd.DataFrame) -> float:\n    contributions = []\n    for _, row in data.iterrows():\n        key = (tuple(row['option_a_ratings']), tuple(row['option_b_ratings']))\n        sign = PAIR_SIGNS.get(key)\n        if sign is None:\n            continue\n        y = 1.0 if row['response'] == 0 else -1.0\n        contributions.append(y * sign)\n    return float(np.mean(contributions)) if contributions else 0.0",
  "rationale": "Each trial is assigned a directional sign from the advocated theory: +1 where the inverse validity-weighted tally, or the validity-ordered tie break, predicts choice A with probability above 0.5, and -1 where it predicts B. The metric scores each A response as +1 and each B response as -1, then averages y*sign. Under the advocated theory all 12 trial types have positive expected contributions: the six raw-tally-tie trials contribute roughly 0.41-0.45, and the six conflict trials contribute smaller positive values. Under the competing unweighted inverse tallying, tie trials have zero expected contribution, while all six conflict trials contribute negatively because unweighted tallying prefers the side opposite the high-validity advantage. With 96 trials per subject the expected value is around +0.30 for the advocated theory and roughly -0.15 to -0.35 for the competing theory, with a within-subject standard deviation of about 0.10, so the between-theory gap is many times the between-subject noise at N=25."
}
```

## Usage

```json
{
  "input_tokens": 4277,
  "output_tokens": 22733,
  "reasoning_tokens": 21830,
  "provider": "Baidu"
}
```
