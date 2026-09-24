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
**Validities (n_features=9):** [0.95, 0.95, 0.95, 0.95, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 1, 1, 1, 0]
  trial 3: A=[0, 0, 1, 1, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 1, 1, 0, 0]
  trial 4: A=[0, 0, 0, 1, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0, 0, 0]
  trial 8: A=[0, 0, 1, 1, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0, 0]
  trial 9: A=[0, 1, 1, 1, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 1, 1, 1, 0, 0]
  trial 10: A=[1, 1, 1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0]
  trial 11: A=[1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0]
  trial 12: A=[0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0]

**Rationale:** This design isolates the tied-top large-conflict divergence between the two theories. Nine experts are split into four tied highest-validity experts (0.95) and five low-validity experts (0.50). Every unique pair is coded by D, the raw inverse clean-cue tally, and h, the clean-cue advantage within the tied top group. In the D = +1 block, h sweeps -4, -2, 0, +2, +4 while the low group is set to D - h; the D = -1 block is the mirror. Because the four top experts are tied and there are 9 features with at least 8 discriminating cues, the advocated theory enters its large tied-top conflict regime: it suppresses tied-top precedence and only scales D by large_tie_scale, so all five D = +1 pairs must produce exactly the same p(A), all D = -1 pairs produce the mirrored p(A), and the D = 0 pairs must produce p(A) = 0.5. The competing theory instead uses a damped validity-weighted tally plus a tied-top clean-cue fallback when the weighted evidence is near zero. Its p(A) therefore varies with h within a fixed D. In the critical D = +1, h = -4 pair the four top cues all favor B while five low cues favor A; the competing fallback flips choice below 0.5, roughly p(A) around 0.41, whereas the advocated theory stays above 0.5. The D = -1, h = +4 mirror gives the opposite reversal. The two D = 0 pairs provide a second lever: advocated is forced to 0.5, while competing follows the tied top clean-cue advantage, roughly 0.26 versus 0.74. This produces both directional reversals and a flat-versus-gradient interaction. The 12 unique pairs repeated 8 times give about 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



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
**Description:** Subjects treat a binary expert rating of 0 as a clean/favorable cue and 1 as a defect, so each discriminating expert contributes an inverse-valence direction. The final evidence is a damped validity-weighted tally: each discriminating cue is weighted by 1 + lambda*(v_j - mean(v)), with lambda modest and further reduced in large multi-cue conflicts, so raw clean-cue majority dominates while high-validity cues receive a moderate extra weight. When exactly one discriminating expert has strictly highest communicated validity, a probabilistic, validity-gap-scaled precedence bonus is added to that expert's direction. When top experts are tied and the weighted evidence is near zero, subjects follow the sign of the tied top group's clean-cue advantage with moderate sensitivity instead of guessing. Choice is a softmax over the resulting evidence with subject-specific gain and lapse. Subject-level gain, lambda, precedence probability, large-conflict raw-tally softening, tie parameters, and epsilon are drawn from Student-t heavy-tailed/logit-Student-t distributions with medians anchored near the accepted base, providing the between-subject variance observed in the data without shifting the aggregate pattern too far.

**Parameters:**
- gain_raw: [0.02, 0.98]
- epsilon_raw: [0.02, 0.98]
- lambda_raw: [0.02, 0.98]
- prec_p_raw: [0.02, 0.98]
- prec_large_p_raw: [0.02, 0.98]
- tie_p_raw: [0.02, 0.98]
- prec_strength_raw: [0.02, 0.98]
- tie_strength_raw: [0.02, 0.98]
- tie_tol_raw: [0.02, 0.98]
- lam_large_scale_raw: [0.02, 0.98]
- large_d_scale_raw: [0.02, 0.98]
- window: {1, 2}
- validities: validities

**`predict source code`:**
```python
import math
import numpy as np
from scipy.stats import norm, t as student_t


def _tqt(u, mu, sigma, lo, hi, df):
    u = float(np.clip(u, 1e-8, 1.0 - 1e-8))
    q_lo = student_t.cdf((lo - mu) / sigma, df)
    q_hi = student_t.cdf((hi - mu) / sigma, df)
    q = q_lo + u * (q_hi - q_lo)
    return mu + sigma * student_t.ppf(q, df)


def _logit(p):
    p = float(np.clip(p, 1e-8, 1.0 - 1e-8))
    return math.log(p / (1.0 - p))


def _ilogit(z):
    return 1.0 / (1.0 + math.exp(-z))


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')
    a = stim[0]
    b = stim[1]
    n_features = int(stim.shape[1])

    val = np.asarray(parameters['validities'], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError('validities length mismatch')

    # ---- Heavy-tailed subject-level transforms -------------------------------
    def u_raw(x):
        return float(np.clip((float(x) - 0.02) / 0.96, 0.0, 1.0))

    gain = float(np.exp(_tqt(
        u_raw(parameters['gain_raw']),
        math.log(1.45), 0.32, math.log(0.60), math.log(2.80), 4)))

    epsilon = float(_ilogit(_tqt(
        u_raw(parameters['epsilon_raw']),
        _logit(0.26), 0.35, _logit(0.07), _logit(0.45), 4)))

    lam = float(np.exp(_tqt(
        u_raw(parameters['lambda_raw']),
        math.log(0.24), 0.55, math.log(0.005), math.log(0.40), 4)))

    p_prec = float(_ilogit(_tqt(
        u_raw(parameters['prec_p_raw']),
        _logit(0.55), 0.50, _logit(0.12), _logit(0.96), 4)))

    p_large = float(_ilogit(_tqt(
        u_raw(parameters['prec_large_p_raw']),
        _logit(0.72), 0.45, _logit(0.20), _logit(0.99), 4)))

    tie_p = float(_ilogit(_tqt(
        u_raw(parameters['tie_p_raw']),
        _logit(0.75), 0.40, _logit(0.25), _logit(0.98), 4)))

    prec_str = float(np.exp(_tqt(
        u_raw(parameters['prec_strength_raw']),
        math.log(0.95), 0.28, math.log(0.40), math.log(1.70), 4)))

    tie_str = float(np.exp(_tqt(
        u_raw(parameters['tie_strength_raw']),
        math.log(0.60), 0.40, math.log(0.15), math.log(1.30), 4)))

    tie_tol = float(np.exp(_tqt(
        u_raw(parameters['tie_tol_raw']),
        math.log(0.60), 0.30, math.log(0.20), math.log(1.30), 4)))

    lam_large_scale = float(_ilogit(_tqt(
        u_raw(parameters['lam_large_scale_raw']),
        _logit(0.45), 0.40, _logit(0.10), _logit(0.85), 4)))

    large_d_scale = float(np.exp(_tqt(
        u_raw(parameters['large_d_scale_raw']),
        math.log(0.50), 0.40, math.log(0.15), math.log(0.90), 4)))

    window = int(round(float(parameters['window'])))

    # ---- Inverse-valence + damped validity weighting -------------------------
    changed = a != b
    disc_idx = np.flatnonzero(changed)
    if disc_idx.size == 0:
        return np.array([0.5, 0.5])

    diff = (b > a).astype(float) - (a > b).astype(float)
    D = float(np.sum(diff))
    disc_size = int(disc_idx.size)

    # pi_8-style large-conflict regime: n_features 7-9 with many discriminators
    is_large_scale = disc_size >= 6 and 7 <= n_features <= 9
    # broader large-conflict flag used only to shrink lambda further
    is_large_weight = disc_size >= 6 and n_features >= 7

    lam_eff = lam * (lam_large_scale if is_large_weight else 1.0)
    mean_v = float(np.mean(val))
    vals_disc = val[disc_idx]
    w_disc = 1.0 + lam_eff * (vals_disc - mean_v)
    E = float(np.sum(w_disc * diff[disc_idx]))

    # ---- Unique top-expert precedence (pi_8-like) ---------------------------
    unique_top = False
    top_tied = False
    top_tied_idx = np.array([], dtype=int)
    top_clean = 0.0
    sign_top = 0.0
    gap = 0.04

    if disc_size == 1:
        unique_top = True
        sign_top = 1.0 if diff[disc_idx[0]] > 0.0 else -1.0
        gap = 0.04
    else:
        top_val = float(np.max(vals_disc))
        top_mask = np.isclose(vals_disc, top_val, rtol=0.0, atol=1e-9)
        top_idx = disc_idx[top_mask]
        if top_idx.size == 1:
            unique_top = True
            top_pos = int(np.flatnonzero(top_mask)[0])
            sign_top = 1.0 if diff[disc_idx[top_pos]] > 0.0 else -1.0
            others = vals_disc[~top_mask]
            if others.size > 0:
                gap = float(top_val - float(np.max(others)))
            else:
                gap = 0.04
        else:
            top_tied = True
            top_tied_idx = top_idx
            top_clean = float(np.sum(diff[top_idx]))

    if unique_top:
        if abs(D) <= 1e-12:
            if np.random.rand() < tie_p:
                tie_bonus = prec_str * (0.70 + 1.70 * math.tanh(gap / 0.05))
                E = sign_top * tie_bonus
        else:
            p_apply = p_large if is_large_scale else p_prec
            if abs(D) <= window and np.random.rand() < p_apply:
                bonus = prec_str * (0.95 + 2.0 * math.tanh(gap / 0.10))
                if is_large_scale:
                    bonus = prec_str * (1.00 + 2.40 * math.tanh(gap / 0.10))
                    bonus = bonus / (1.0 + 0.04 * max(0.0, float(disc_size - 4)))
                E = E + sign_top * bonus
    elif top_tied:
        # Raw-majority-dominant softening in the known large-conflict regime.
        if is_large_scale and abs(D) > 1e-12:
            E = D * large_d_scale + (E - D)

        # Tied-top group clean-cue advantage when the tally is near zero.
        if abs(E) <= tie_tol and abs(top_clean) > 1e-9:
            top_effect = tie_str * math.tanh(top_clean / 2.8)
            E = E + top_effect

    # ---- Softmax choice ------------------------------------------------------
    logits = gain * np.array([E, 0.0], dtype=float)
    logits = logits - np.max(logits)
    p_core = np.exp(logits)
    p_core = p_core / p_core.sum()

    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\n\ndef metric(data: pd.DataFrame) -> float:\n    a = np.array(data['option_a_ratings'].tolist(), dtype=int)\n    b = np.array(data['option_b_ratings'].tolist(), dtype=int)\n    if a.size == 0:\n        return 0.0\n\n    diff = (b > a).astype(int) - (a > b).astype(int)\n    h = diff[:, :4].sum(axis=1).astype(float)\n    d = diff.sum(axis=1).astype(int)\n\n    resp = np.asarray(data['response'], dtype=float)\n    y = 1.0 - resp\n\n    h_c = h.copy()\n    y_c = y.copy()\n    for dval in np.unique(d):\n        mask = d == dval\n        h_c[mask] -= h_c[mask].mean()\n        y_c[mask] -= y_c[mask].mean()\n\n    denom = float(np.sum(h_c * h_c))\n    if denom < 1e-12:\n        return 0.0\n    return float(np.sum(h_c * y_c) / denom)\n",
  "rationale": "The first four experts are the tied highest-validity group. For each trial I compute D as the total clean-cue tally favoring A, where a 0 rating counts for an option and a 1 rating counts against it, and h as the clean-cue tally restricted to the four tied top experts. The advocated theory predicts that choices in these large tied-top conflicts depend only on D and are flat in h within each D value, so the within-D slope of choosing A on h is exactly zero. The competing theory predicts a tied-top clean-cue fallback, making the same within-D slope positive. Because the design balances h within each D value, this dummy-adjusted slope isolates the tied-top clean-cue effect, and it pools all 96 trials per subject to keep the estimate very tight."
}
```

## Usage

```json
{
  "input_tokens": 6998,
  "output_tokens": 24101,
  "reasoning_tokens": 23638,
  "provider": "Baidu"
}
```
