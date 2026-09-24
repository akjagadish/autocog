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
**Validities (n_features=7):** [0.95, 0.9, 0.85, 0.8, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  trial 3: A=[1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 0, 0]
  trial 4: A=[1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1, 0]
  trial 5: A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0]
  trial 6: A=[1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 1, 1, 0]
  trial 7: A=[1, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0]
  trial 8: A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0]

**Rationale:** Seven experts with a wide validity spread. Every unique pair is a 3-vs-3 split with one tied expert, so the unweighted feature tally is exactly zero for all trials. The validity-weighted difference U = sum_j v_j^gamma (A_j - B_j) is varied across trials by moving the three A-won experts from a mixed set to the three highest-validity experts. NWA depends only on U and predicts p(A) = sigmoid(-beta U) plus lapse, so p(A) must decrease monotonically as U increases. VWEA-CG has conflict = |U|/(1+|tally|) = |U|, so its effective inverse temperature shrinks at large U; the negative-valence majority drives p(A) down for moderate U, but conflict attenuation pushes p(A) back toward chance for large U. Under central parameters the predicted p(A) values for U = 0.134, 0.247, 0.325, 0.437, 0.620, 0.824, 1.014, 1.191 are approximately: NWA .411, .341, .296, .238, .161, .102, .065, .043; VWEA-CG .391, .300, .251, .204, .179, .209, .268, .326. Thus NWA is monotone decreasing while VWEA-CG is U-shaped, a rank-order dissociation that no NWA beta/gamma/epsilon can reproduce.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Validity-weighted evidence accumulation with conflict-gated confidence (VWEA-CG). People integrate binary feature differences by weighting each difference by its communicated validity raised to a subject-specific power. A majority of subjects use a negative-valence or contra-indicator rule, so a feature advantage counts against an option, while a minority use positive-valence trust. Critically, the effective decisiveness of that validity-weighted evidence is attenuated when the evidence strongly conflicts with the simple feature tally: when the validity-weighted score is large but the number of feature-wise wins is nearly tied, the decision is treated as less diagnostic. Choices are generated by a softmax over this conflict-gated signed evidence score, plus a uniform lapse.

**Parameters:**
- p_positive: [0.03, 0.07]
- valence_draw: [0.0, 1.0]
- gamma: [2.0, 2.4]
- beta: [4.2, 4.8]
- epsilon: [0.04, 0.08]
- bias_a: [0.0, 0.03]
- kappa: [2.3, 2.7]
- conflict_power: [3.8, 4.2]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # Conflict-gated heterogeneous signed-validity weighting.
    # state is expected to be array-like of shape (2, n_features),
    # with row 0 = option A and row 1 = option B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "Conflict-gated signed-validity model expects a (2, n_features) stimulus; "
            f"got shape {stim.shape}."
        )

    a = stim[0]
    b = stim[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.size != a.size:
        raise ValueError(
            f"validities length {v.size} does not match n_features {a.size}."
        )

    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    bias_a = float(parameters["bias_a"])
    kappa = float(parameters["kappa"])
    conflict_power = float(parameters["conflict_power"])

    # Latent strategy draw: most subjects use negative-valence weighting,
    # while a minority use positive-valence trust.
    p_positive = float(parameters["p_positive"])
    valence_draw = float(parameters["valence_draw"])
    sign = 1.0 if valence_draw < p_positive else -1.0

    # Signed validity-weighted evidence and simple feature tally.
    weighted_diff = float(np.dot(v ** gamma, a - b))
    tally = float(np.sum(a > b) - np.sum(b > a))

    # Conflict between validity-weighted evidence and simple tally.
    # When validities push strongly toward one option but the tally is
    # nearly tied, confidence falls and the effective softmax gain shrinks.
    conflict = abs(weighted_diff) / (1.0 + abs(tally))
    attenuation = 1.0 / (1.0 + kappa * (conflict ** conflict_power))
    beta_eff = beta * attenuation

    score_a = sign * weighted_diff + bias_a
    z = beta_eff * score_a

    # Numerically stable sigmoid for p(A | score).
    if z >= 0.0:
        p_a_core = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        p_a_core = ez / (1.0 + ez)

    # Uniform lapse over the two options.
    p_a = (1.0 - epsilon) * p_a_core + epsilon / 2.0
    p_a = float(np.clip(p_a, 0.0, 1.0))
    return np.array([p_a, 1.0 - p_a], dtype=float)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=float)
    p = p / p.sum()
    return int(np.random.choice(len(p), p=p))
```


## COMPETING THEORY
**Description:** Negative Validity-Weighted Additive (NWA): people combine all binary feature differences linearly, but assign negative weight to a feature advantage, and the absolute weight magnitude increases with the feature's stated validity. Formally, choice is based on S = sum_j w_j * (A_j - B_j), with w_j = -v_j^gamma (gamma > 0), so an option's cue advantage makes that option less likely to be chosen. Choice probability uses a softmax/sigmoid over this score with inverse temperature beta, plus a small uniform lapse epsilon.

**Parameters:**
- beta: [2.65, 2.75]
- gamma: [2.22, 2.28]
- epsilon: [0.0, 0.02]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # NWA: negative validity-weighted additive integration of all cues.
    # state is shape (2, n_features): row 0 = option A, row 1 = option B.
    # history is ignored because there is no trial-by-trial feedback.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"NWA expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a = stim[0]
    b = stim[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.size != a.size:
        raise ValueError(
            f"validities length {v.size} does not match n_features {a.size}."
        )

    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Negative validity-scaled weights: an advantage on a cue is evidence
    # against that option, with stronger validity cues carrying more weight.
    weights = -(v ** gamma)

    score = float(np.dot(weights, a - b))

    # Stable sigmoid for p(A | score).
    z = beta * score
    if z >= 0:
        p_a_core = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        p_a_core = ez / (1.0 + ez)

    # Uniform lapse mixture.
    p_a = (1.0 - epsilon) * p_a_core + epsilon / 2.0
    return np.array([p_a, 1.0 - p_a], dtype=float)
```

**`policy source code`:**
```python
def policy(probs):
    p = np.asarray(probs, dtype=float)
    p = p / p.sum()
    return int(np.random.choice(len(p), p=p))
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
  "metric_source": "import numpy as np\n\ndef metric(data):\n    v = np.array([0.95, 0.9, 0.85, 0.8, 0.6, 0.55, 0.5], dtype=float)\n\n    a = np.asarray(data['option_a_ratings'].tolist(), dtype=float)\n    b = np.asarray(data['option_b_ratings'].tolist(), dtype=float)\n    z = (a - b).dot(v ** 2)\n\n    resp = np.asarray(data['response'], dtype=float)\n\n    low_mask = z < 0.5\n    high_mask = z > 0.9\n\n    lows = resp[low_mask]\n    highs = resp[high_mask]\n\n    if lows.size == 0 or highs.size == 0:\n        return float('nan')\n\n    p_a_low = float(np.mean(lows == 0))\n    p_a_high = float(np.mean(highs == 0))\n\n    return float(p_a_low - p_a_high)\n",
  "rationale": "Trials are ordered by a validity-weighted conflict score computed from the squared communicated validities. The first four low-score pair types and the last two high-score pair types are used. NWA predicts that the probability of choosing A decreases monotonically as this score increases, so p(A in low-score trials) - p(A in high-score trials) should be large and positive. VWEA-CG predicts that conflict attenuation pushes choices on the largest-score trials back toward chance, so the same contrast should be near zero or slightly negative. Each subject sees all pair types repeatedly, so the contrast is stable at the single-subject level and has low binomial variance."
}
```

## Usage

```json
{
  "input_tokens": 4294,
  "output_tokens": 17090,
  "reasoning_tokens": 16700,
  "provider": "Baidu"
}
```
