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
**Validities (n_features=8):** [0.95, 0.95, 0.95, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 1, 1, 1, 1, 0]
  trial 3: A=[0, 0, 1, 0, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  trial 5: A=[0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  trial 7: A=[0, 1, 1, 0, 0, 1, 1, 1]  B=[1, 0, 0, 1, 1, 0, 0, 0]
  trial 8: A=[1, 1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]

**Rationale:** Eight features are used; the three highest-validity experts are tied at 0.95 and the five remaining experts are at 0.50. Every trial uses all eight features as discriminating cues, so the advocated theory is forced into its large tied-top conflict regime: there are eight discriminating cues in an eight-feature stimulus and no unique highest-validity expert, so the raw inverse-clean tally D is multiplied by large_tie_scale. Thus within each fixed-D block the advocated theory predicts one flat choice probability: all D=+2 pairs have the same p(A) and all D=-2 pairs have the mirrored p(A), with no effect of how the high-validity versus low-validity cues are arranged. The competing theory includes each cue with a validity-dependent probability, about 0.96 for the 0.95 cues and about 0.50 for the 0.50 cues at median parameters, and then tallies included cues equally. Its expected included tally is s_hat = p_high * h + p_low * l, where h is the high-group clean-cue advantage and l is the low-group clean-cue advantage. Each block fixes D = h + l at +2 or -2 while h sweeps from -3 to +3. For the D=+2 block the advocated prediction is flat near p(A)=0.69, whereas the competing expected p(A) rises from about 0.33 for h=-3,l=+5, to 0.73, 0.89, and 0.92. The first pair therefore gives opposite directional predictions. The D=-2 block is the mirror: advocated stays flat near p(A)=0.31, while competing falls from about 0.67 for h=+3,l=-5, to 0.27, 0.11, and 0.08, again giving an opposite direction on the first pair. The critical signature is a flat composition effect coupled only with a D sign effect for the advocated theory, versus a large composition gradient and crossover for the competing theory. Eight unique pairs repeated K=12 gives 96 trials.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Inverse-valence majority with gated unique-expert precedence and tied-top large-conflict softness. Subjects treat a binary rating of 0 as a clean/favorable cue and 1 as a defect, so every discriminating expert contributes approximately +1 for A or -1 for B to a raw clean-cue tally. The default choice is a softmax over this near-equal-weight tally. When exactly one discriminating expert has strictly highest communicated validity, that unique expert receives a gap-scaled precedence bonus inside a limited raw-tally window. In large multi-cue conflicts, this unique-expert reliance is separately and more strongly applied. When two or more top experts are tied, unique-expert precedence is suppressed, but the conflict is still processed with reduced decision sensitivity to the raw tally, preventing an unrealistically sharp majority effect while never letting tied top experts override a non-zero low-validity majority.

**Parameters:**
- gain: [1.20, 1.80]
- epsilon: [0.20, 0.36]
- p_override: [0.30, 0.80]
- p_large: [0.35, 0.98]
- tie_p: [0.55, 0.95]
- window: {1, 2}
- large_tie_scale: [0.05, 0.75]
- validities: validities

**`predict source code`:**
```python
import math
import numpy as np

def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')
    a = stim[0]
    b = stim[1]
    val = np.asarray(parameters['validities'], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError('validities length mismatch')

    changed = a != b
    disc = np.flatnonzero(changed)
    if disc.size == 0:
        return np.array([0.5, 0.5])

    diff = (b > a).astype(float) - (a > b).astype(float)
    D = float(np.sum(diff))

    gain = float(parameters['gain'])
    epsilon = float(parameters['epsilon'])
    p_override = float(parameters['p_override'])
    p_large = float(parameters['p_large'])
    tie_p = float(parameters['tie_p'])
    window = int(round(float(parameters['window'])))

    is_large = disc.size >= 6 and 7 <= stim.shape[1] <= 9
    window_eff = window
    if is_large:
        window_eff = max(2, window)
        p_apply = p_large
    else:
        p_apply = p_override

    E = D
    unique_top = False
    sign_top = 0.0
    gap = 0.0

    if disc.size == 1:
        j_top = int(disc[0])
        sign_top = 1.0 if diff[j_top] > 0.0 else -1.0
        gap = 0.04
        unique_top = True
    else:
        vals_disc = val[disc]
        top_val = float(np.max(vals_disc))
        top_hits = disc[np.isclose(vals_disc, top_val, rtol=0.0, atol=1e-9)]
        unique_top = top_hits.size == 1
        if unique_top:
            j_top = int(top_hits[0])
            sign_top = 1.0 if diff[j_top] > 0.0 else -1.0
            others = vals_disc[~np.isclose(vals_disc, top_val, rtol=0.0, atol=1e-9)]
            if others.size == 0:
                gap = 0.04
            else:
                gap = float(top_val - float(np.max(others)))

    if unique_top:
        if abs(D) <= 1e-12:
            if np.random.rand() < tie_p:
                tie_bonus = 0.72 + 0.50 * math.tanh(gap / 0.05)
                E = sign_top * tie_bonus
            else:
                E = 0.0
        elif abs(D) <= window_eff and np.random.rand() < p_apply:
            bonus = 0.95 + 2.0 * math.tanh(gap / 0.10)
            if is_large:
                bonus = 1.00 + 2.40 * math.tanh(gap / 0.10)
                bonus = bonus / (1.0 + 0.04 * max(0.0, float(disc.size - 4)))
            E = D + sign_top * bonus
    elif abs(D) <= 1e-12:
        E = 0.0

    # Tied-top large conflicts: unique-expert precedence is suppressed by the
    # theory, but raw-tally sensitivity is reduced so the low-validity majority
    # does not drive an unrealistically sharp choice.
    if is_large and not unique_top and abs(D) > 1e-12:
        E = D * float(parameters['large_tie_scale'])

    logits = gain * np.array([E, 0.0], dtype=float)
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core = p_core / p_core.sum()

    return (1.0 - epsilon) * p_core + epsilon * 0.5
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
**Description:** People invert the valence of binary expert ratings: a 1 is treated as a defect and a 0 as a clean, favorable signal. Each feature j probabilistically enters an inverse clean-cue tally with inclusion probability p_include_j = logistic(alpha * validity_j + gamma); excluded cues receive exactly zero weight. The included clean-cue tally drives a noisy softmax choice, and exact zero tallies are resolved by consulting the highest-validity included discriminating cue and choosing the option with the clean 0 there. If no included cue discriminates, people guess. Subject-level alpha, gamma, beta_tally, beta_tiebreak, and epsilon create individual differences, with heavy-tailed/logit-transformed subject-level distributions so that some subjects are nearly deterministic while others remain noisy, preserving the pooled inverse-valence pattern and realistically large between-subject variance.

**Parameters:**
- alpha: [5.0, 9.0]
- gamma_logit: [-2.50, 2.50]
- beta_tally_log: [-0.30, 0.50]
- beta_tiebreak_log: [0.40, 1.10]
- epsilon_logit: [-3.00, -0.40]
- validities: validities

**`predict source code`:**
```python
import numpy as np
from scipy.special import expit

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

    # Heavy-tailed subject-level transforms of the sampled raw parameters.
    beta_tally = float(np.exp(float(parameters['beta_tally_log'])))
    beta_tiebreak = float(np.exp(float(parameters['beta_tiebreak_log'])))
    epsilon = float(expit(float(parameters['epsilon_logit'])))
    alpha = float(parameters['alpha'])
    gamma = -4.5 + 2.0 * float(expit(float(parameters['gamma_logit'])))

    p_include = expit(alpha * val + gamma)
    include = np.random.random(n_features) < p_include

    clean_a = ((b > a) & include).astype(float)
    clean_b = ((a > b) & include).astype(float)
    s = float(np.sum(clean_a) - np.sum(clean_b))

    if s != 0.0:
        p_a = expit(2.0 * beta_tally * s)
        p_core = np.array([p_a, 1.0 - p_a])
    else:
        winner = None
        cue_order = np.argsort(-val, kind='stable')
        for j in cue_order:
            if include[j] and a[j] != b[j]:
                winner = 0 if (a[j] == 0 and b[j] == 1) else 1
                break
        if winner is None:
            return np.ones(2) / 2.0
        p_a = expit(beta_tiebreak) if winner == 0 else expit(-beta_tiebreak)
        p_core = np.array([p_a, 1.0 - p_a])

    return (1.0 - epsilon) * p_core + epsilon * np.ones(2) / 2.0
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\n\ndef metric(data: pd.DataFrame) -> float:\n    a = np.array([list(v) for v in data['option_a_ratings']], dtype=float)\n    b = np.array([list(v) for v in data['option_b_ratings']], dtype=float)\n    y = 1.0 - data['response'].to_numpy(dtype=float)\n    sid = data['subject_id'].to_numpy()\n\n    clean_a_high = ((a[:, :3] == 0) & (b[:, :3] == 1)).sum(axis=1)\n    clean_b_high = ((a[:, :3] == 1) & (b[:, :3] == 0)).sum(axis=1)\n    h = clean_a_high - clean_b_high\n\n    clean_a_all = ((a == 0) & (b == 1)).sum(axis=1)\n    clean_b_all = ((a == 1) & (b == 0)).sum(axis=1)\n    D = clean_a_all - clean_b_all\n\n    slopes = []\n    for subj in np.unique(sid):\n        m = sid == subj\n        subj_slopes = []\n        for D_val in (2.0, -2.0):\n            dm = m & (D == D_val)\n            if np.sum(dm) < 2:\n                continue\n            hh = h[dm].astype(float)\n            yy = y[dm]\n            hh_c = hh - hh.mean()\n            yy_c = yy - yy.mean()\n            denom = float(np.sum(hh_c * hh_c))\n            if denom > 1e-12:\n                subj_slopes.append(float(np.sum(hh_c * yy_c) / denom))\n        slopes.append(float(np.mean(subj_slopes)) if subj_slopes else 0.0)\n\n    if not slopes:\n        return 0.0\n    return float(np.mean(slopes))",
  "rationale": "The metric estimates the within-subject composition gradient: inside each D sign block (D = +2 and D = -2) it regresses choose-A (1 - response) on the high-validity clean-cue advantage h. The advocated theory forces identical choice probability for every pair with the same raw D when top experts are tied, so this slope is zero up to binomial sampling. The competing theory includes high-validity cues with probability p_high > p_low, giving expected included tally p_low*D + (p_high - p_low)*h, so choose-A increases with h within both D blocks. Averaging the D=+2 and D=-2 slopes uses all 96 trials and keeps between-subject noise small."
}
```

## Usage

```json
{
  "input_tokens": 5116,
  "output_tokens": 27360,
  "reasoning_tokens": 26708,
  "provider": "Baidu"
}
```
