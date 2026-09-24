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
**Validities (n_features=9):** [0.99, 0.98, 0.97, 0.95, 0.93, 0.9, 0.81, 0.79, 0.78]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0]
  trial 2: A=[1, 1, 0, 1, 1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0, 0, 1, 0, 0]
  trial 3: A=[0, 1, 1, 1, 1, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 1, 1, 0, 0]
  trial 4: A=[1, 1, 1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0]
  trial 5: A=[0, 1, 1, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 1, 1, 1, 0, 0]
  trial 6: A=[0, 0, 1, 1, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 1, 1, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 1, 1, 1, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 0, 0]
  trial 9: A=[1, 1, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1]
  trial 10: A=[1, 1, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 1, 1]
  trial 11: A=[0, 1, 1, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0, 1, 1]
  trial 12: A=[1, 1, 1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 1, 1]
  trial 13: A=[0, 1, 1, 1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1, 0, 1, 1]
  trial 14: A=[0, 0, 1, 1, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1, 0, 1, 1]
  trial 15: A=[1, 1, 0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0, 1, 1]
  trial 16: A=[0, 0, 0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0, 1, 1]

**Rationale:** Nine experts are used: three near-chance experts (0.53, 0.52, 0.51) and six higher-validity experts (0.99 down to 0.90). The three near-chance experts are deliberately close together. In the competing theory they become the top three normalized weights, each about 0.32 to 0.33, so the top3-similarity gate is near 1 and the effective-n gate is near 3; this holds the context-gated beta roughly constant. All six higher validities together contribute less than about 0.02 of normalized weight, so the competing theory's evidence is set almost entirely by which option wins more of the three low-validity experts. Block A has A win two low experts and B win one, giving competing evidence about +0.33 for A on every trial, so p(A) stays near 0.65 to 0.70 regardless of the high-feature pattern. Block B mirrors this, giving competing evidence about -0.33 and p(A) near 0.30 to 0.35. The advocated theory does not normalize and weights each expert by max(1 - v, 0)^gamma. Its low-expert weights are near 0.80, but its high-expert weights remain substantial, from about 0.50 for the 0.90 expert down to about 0.25 for the 0.99 expert. Eight high-feature patterns are chosen so that the B-winning high-feature weight total S equals 0, 0.349, 0.752, 0.951, 1.202, 1.511, 1.707, or 2.267. In the two-low A block, advocated evidence for A therefore runs +3.08, +2.38, +1.58, +1.18, +0.68, +0.06, -0.33, -1.45, giving p(A) from near ceiling to near floor and crossing chance between the last two intermediate patterns. In the mirrored block it runs +1.45, +0.76, -0.05, -0.45, -0.95, -1.57, -1.96, -3.08. Thus the competitor predicts a flat line within each low-split block, while the advocated theory predicts a steep psychometric function of the high-validity pattern; four pairs are directionally opposite. The 16 unique pairs repeated 6 times give roughly 96 trials.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People treat advertised expert validity as a cue to ignore rather than trust: cues with low validity, especially those near chance, receive the largest decision weight, while very high-validity cues are discounted because they are seen as redundant or overstated. Choices are formed by a continuous reversed-validity weighted additive comparison of the two options, with signed feature differences weighted by an inverse diagnosticity function of the advertised validities. Ties on a feature contribute zero evidence. The accumulated evidence then passes through a softmax choice rule with inverse temperature beta, plus a small uniform lapse probability epsilon that captures attentional or motor noise.

**Parameters:**
- theta: {1.0}
- gamma: [0.3, 0.4]
- beta: [4.8, 5.5]
- epsilon: [0.15, 0.25]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Reversed-diagnosticity weighted additive integration.
    # state is expected to be array-like of shape (2, n_features),
    # with row 0 = option A and row 1 = option B.
    # History is ignored because validities are advertised in the
    # instructions and no trial-by-trial feedback is provided.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Reversed-validity integration expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    if validities.ndim != 1:
        validities = validities.reshape(-1)
    if validities.shape[0] != stim.shape[1]:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {stim.shape[1]}."
        )

    theta = float(parameters["theta"])
    gamma = float(parameters["gamma"])

    # Reversed diagnosticity weights. Low-validity cues receive large
    # weights; high-validity cues are discounted. The max guard keeps
    # weights non-negative even for perfect validities.
    weights = np.maximum(theta - validities, 0.0) ** gamma

    # Signed feature-wise evidence: positive favors A, negative favors B.
    # Feature ties contribute zero.
    diff = stim[0] - stim[1]
    evidence_A = float(np.dot(weights, diff))

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Numerically stable softmax over the two options [A, B].
    logits = np.array([evidence_A, 0.0])
    z = beta * (logits - logits.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))
```


## COMPETING THEORY
**Description:** People normalize each advertised cue validity relative to the current task ensemble, giving salience only to cues whose validity falls below the ensemble median. Low-validity outliers dominate the signed feature tally, with ties contributing zero. The decisiveness of the softmax choice rule is itself context-gated: when a single cue becomes a near-monopoly in normalized decision weight, the response temperature is quenched toward chance; when a small set of low-validity cues is nearly co-equal, the response is sharpened. This keeps equal-weight tallying as a limiting case while allowing reversed-validity-like and regime-sensitive choice behavior.

**Parameters:**
- w0: [0.005, 0.04]
- kappa: [1.5, 4.0]
- gamma: [1.0, 1.5]
- c: [0.005, 0.02]
- beta: [1.7, 2.3]
- epsilon: [0.12, 0.18]
- share_hi_center: [0.84, 0.92]
- share_hi_width: [0.012, 0.035]
- gap_hi_center: [0.42, 0.58]
- gap_hi_width: [0.03, 0.08]
- beta_floor_hi: [0.25, 0.35]
- eff_n_center: [2.95, 3.10]
- eff_n_width: [0.10, 0.18]
- top3_sim_center: [0.92, 0.98]
- top3_sim_width: [0.04, 0.10]
- beta_boost_lo: [0.8, 1.4]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus array.')

    validities = np.asarray(parameters['validities'], dtype=float).reshape(-1)
    if validities.shape[0] != stim.shape[1]:
        raise ValueError('validities length must match n_features.')

    median = np.median(validities)
    mad = np.median(np.abs(validities - median))
    c = float(parameters['c'])

    denom = mad + c
    if denom <= 0.0:
        denom = 1e-12
    deviance = np.maximum(0.0, median - validities) / denom

    w0 = float(parameters['w0'])
    kappa = float(parameters['kappa'])
    gamma = float(parameters['gamma'])
    weights = w0 + kappa * np.power(deviance, gamma)

    total = weights.sum()
    if total <= 0.0:
        weights = np.ones_like(weights) / max(weights.size, 1)
    else:
        weights = weights / total

    diff = stim[0] - stim[1]
    evidence_A = float(np.dot(weights, diff))

    sorted_w = np.sort(weights)[::-1]
    nw = sorted_w.shape[0]
    top1 = sorted_w[0]
    top2 = sorted_w[1] if nw > 1 else top1
    top3 = sorted_w[2] if nw > 2 else (sorted_w[-1] if nw > 0 else 0.0)
    top3_similarity = 1.0 - float(top1 - top3)
    effective_n = 1.0 / max(float(np.sum(weights ** 2)), 1e-12)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-float(np.clip(x, -500.0, 500.0))))

    share_hi_center = float(parameters['share_hi_center'])
    share_hi_width = float(parameters['share_hi_width'])
    gap_hi_center = float(parameters['gap_hi_center'])
    gap_hi_width = float(parameters['gap_hi_width'])
    beta_floor_hi = float(parameters['beta_floor_hi'])

    share_gate = sigmoid((top1 - share_hi_center) / share_hi_width)
    gap_gate = sigmoid(((top1 - top2) - gap_hi_center) / gap_hi_width)
    high_gate = share_gate * gap_gate

    eff_n_center = float(parameters['eff_n_center'])
    eff_n_width = float(parameters['eff_n_width'])
    top3_sim_center = float(parameters['top3_sim_center'])
    top3_sim_width = float(parameters['top3_sim_width'])
    beta_boost_lo = float(parameters['beta_boost_lo'])

    eff_gate = np.exp(-0.5 * ((effective_n - eff_n_center) / eff_n_width) ** 2)
    balance_gate = np.exp(-0.5 * ((top3_similarity - top3_sim_center) / top3_sim_width) ** 2)
    low_gate = eff_gate * balance_gate

    beta = float(parameters['beta'])
    epsilon = float(parameters['epsilon'])
    beta_eff = beta + beta_boost_lo * low_gate
    beta_eff = beta_eff * (1.0 - high_gate) + beta_floor_hi * high_gate
    beta_eff = float(np.clip(beta_eff, 0.05, 15.0))

    logits = np.array([evidence_A, 0.0])
    z = beta_eff * (logits - np.max(logits))
    e = np.exp(z)
    core = e / e.sum()

    n_opts = core.shape[0]
    return (1.0 - epsilon) * core + epsilon * np.ones(n_opts) / n_opts
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    return int(np.random.choice(probs.shape[0], p=probs))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    # High-validity experts are the first six rating positions.\n    # Low-validity experts are positions 6,7,8; their split identifies the block.\n    blocks = {}\n    for row in data.itertuples(index=False):\n        a = np.asarray(row.option_a_ratings, dtype=float)\n        b = np.asarray(row.option_b_ratings, dtype=float)\n        high_diff = a[:6] - b[:6]\n        h = float(np.sum(high_diff))\n        key = tuple(int(x) for x in high_diff)\n        low_sign = int((a[6] - b[6]) + (a[7] - b[7]) + (a[8] - b[8]))\n        yA = 1.0 if int(row.response) == 0 else 0.0\n        d = blocks.setdefault(low_sign, {})\n        entry = d.setdefault(key, [0.0, 0, h])\n        entry[0] += yA\n        entry[1] += 1\n    vals = []\n    for d in blocks.values():\n        if len(d) < 2:\n            continue\n        entries = list(d.values())\n        hs = np.array([e[2] for e in entries], dtype=float)\n        ps = np.array([e[0] / e[1] if e[1] > 0 else 0.5 for e in entries], dtype=float)\n        z = hs - hs.mean()\n        denom = float(np.sqrt(np.mean(z * z)))\n        if denom <= 0.0:\n            continue\n        vals.append(float(np.mean((z / denom) * ps)))\n    return float(np.mean(vals)) if vals else 0.0",
  "rationale": "Within each block the three low-validity feature slots are fixed: in one block A wins two and B wins one, and the mirrored block reverses that split. Therefore within-block variation isolates the six high-validity features. For each block and each unique high-validity pattern I compute the A-choice proportion, then compute its covariance with the net number of high-validity experts favoring A. The metric averages this within-block standardized covariance across the two blocks. The advocated reversed-validity theory still gives the six high-validity experts substantial positive weight, so choices should move strongly with this high-validity score. The competing normalization theory assigns those high-validity experts near-zero normalized weight, so its within-block covariance should stay near chance. Demeaning within block removes the block-level low-validity split, and aggregating over repeated patterns yields a stable per-subject score."
}
```

## Usage

```json
{
  "input_tokens": 5661,
  "output_tokens": 27529,
  "reasoning_tokens": 26895,
  "provider": "Baidu"
}
```
