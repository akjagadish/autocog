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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.79]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 7: A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 10: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 11: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 12: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** The design pits a first-discriminating-cue inversion process (GSS-ITTB) against a continuous validity-weighted integration process with saturation (CNVW-tau). With validities [0.90, 0.80, 0.70, 0.60, 0.55], the 0.90 expert discriminates on every unique pair, so GSS-ITTB's attentional gate is zero and no later cue can affect the response. GSS-ITTB therefore predicts only two flat choice probabilities: about 0.27 for A when the 0.90 cue favors A, and about 0.73 for A when the 0.90 cue favors B, with no effect of the lower-cue balance. CNVW-tau instead aggregates D = sum_j v_j^gamma*(A_j-B_j) and most subjects invert its sign. In the six top-A pairs, D takes approximately +0.789, +0.184, +0.472, -0.264, -0.581, -0.841; under central CNVW-tau parameters this gives p(A) near .18, .28, .21, .75, .80, .82, crossing from choosing B to choosing A while GSS-ITTB stays at about .27 throughout. The six top-B mirrors have the complementary D signs and likewise force CNVW-tau to cross from high A choice to low A choice while GSS-ITTB stays at about .73. The critical dissociation is the top-A negative-D trials and the top-B positive-D trials: GSS-ITTB continues choosing opposite the first cue, whereas CNVW-tau reverses because the lower-validity majority flips the sign of D. This sign-reversal signature cannot be produced by a flat first-cue rule regardless of q_90, and cannot be flattened by any CNVW-tau tau/lambda/beta/bias/lapse tuning while preserving the D-dependence.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Validity-gated slipped-start Inverted Take-the-Best (GSS-ITTB). People order cues by stated validity and normally use the first discriminating cue, inverting its recommendation with subject-specific probability q_i. Attentional slips to later ordered cues are not driven by feature count alone; they occur mainly when the first discriminator's validity is below a threshold, and are strongly suppressed for first discriminators with validity 0.90 or higher. When a slip occurs, the later discriminating cue controls the response with a boosted inversion probability, reflecting stronger contra-indicator responding in the low-validity regime. Trials with no discriminating cue are resolved by subject-level bias, and a small lapse adds residual noise. There is no graded validity-weighted accumulation and no continuous logit over a compensatory score.

**Parameters:**
- q_lo: [0.20, 0.40]
- q_mid: [0.55, 0.68]
- q_90: [0.66, 0.82]
- q_92: [0.48, 0.62]
- q_hi: [0.70, 0.86]
- q_floor: [0.45, 0.65]
- bias: [0.30, 0.70]
- epsilon: [0.02, 0.08]
- late2_base: [0.02, 0.08]
- late2_slope: [0.80, 1.10]
- late3_base: [0.02, 0.06]
- late3_slope: [0.10, 0.25]
- gate_low: [0.84, 0.88]
- q_low_boost: [0.10, 0.25]
- max_slip: [0.93, 0.98]
- validities: validities

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    """Validity-gated slipped-start Inverted Take-the-Best (GSS-ITTB).

    Cues are ordered by stated validity. The normative first
    discriminating cue sets the subject's base inversion probability
    q_norm. Attentional slips to later ordered cues are gated by the
    validity of that first discriminator: slips are allowed and
    strengthened when validity is below gate_low, ramp down between
    gate_low and 0.90, and are switched off at 0.90 or above. Slipped
    responses use a boosted inversion strength q_slip. If no cue
    discriminates, a subject-level bias is used. No graded cue
    accumulation enters the model.
    """
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "GSS-ITTB expects a (2, n_features) stimulus; "
            f"got shape {stim.shape}."
        )

    a = stim[0]
    b = stim[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.size != a.size:
        raise ValueError(
            f"validities length {v.size} does not match n_features {a.size}."
        )

    n = int(a.size)

    q_lo = float(parameters["q_lo"])
    q_mid = float(parameters["q_mid"])
    q_90 = float(parameters["q_90"])
    q_92 = float(parameters["q_92"])
    q_hi = float(parameters["q_hi"])
    q_floor = float(parameters["q_floor"])
    bias = float(parameters["bias"])
    epsilon = float(parameters["epsilon"])
    late2_base = float(parameters["late2_base"])
    late2_slope = float(parameters["late2_slope"])
    late3_base = float(parameters["late3_base"])
    late3_slope = float(parameters["late3_slope"])
    gate_low = float(parameters["gate_low"])
    q_low_boost = float(parameters["q_low_boost"])
    max_slip = float(parameters["max_slip"])

    def q_of(vv):
        vv = float(vv)
        if vv > 0.97:
            return float(np.clip(q_floor, 0.0, 1.0))
        return float(np.interp(
            vv,
            np.array([0.50, 0.80, 0.90, 0.92, 0.95], dtype=float),
            np.array([q_lo, q_mid, q_90, q_92, q_hi], dtype=float),
        ))

    cue_order = np.argsort(-v, kind="stable")

    def first_disc(start_idx):
        for r in range(start_idx, n):
            j = int(cue_order[r])
            if a[j] != b[j]:
                return j
        return -1

    j0 = first_disc(0)

    if j0 < 0:
        # No discriminating cue anywhere: subject-level bias.
        p_core = bias
    else:
        # Base inversion strength is calibrated by the normative first
        # discriminating cue.
        q_norm = q_of(v[j0])

        # Feature-count pressure toward late search.
        extra = max(0.0, float(n - 5))
        late2 = float(np.clip(late2_base + late2_slope * extra, 0.0, 0.90))
        late3 = float(np.clip(late3_base + late3_slope * extra, 0.0, 0.60))

        # Validity gate on slip: full slip below gate_low, zero slip at
        # validity 0.90 or higher, linear ramp between.
        vj0 = float(v[j0])
        if vj0 < gate_low:
            gate = 1.0
        elif vj0 >= 0.90:
            gate = 0.0
        else:
            gate = (0.90 - vj0) / (0.90 - gate_low)
        gate = float(np.clip(gate, 0.0, 1.0))

        late2 *= gate
        late3 *= gate

        if late2 + late3 > max_slip:
            scale = max_slip / (late2 + late3)
            late2 *= scale
            late3 *= scale

        p0 = 1.0 - late2 - late3

        # Slipped-to cues are inverted more strongly in the gated regime.
        q_slip = float(np.clip(q_norm + q_low_boost * gate, 0.0, 1.0))

        def direction_prob(start_idx, q_val):
            j = first_disc(start_idx)
            if j < 0:
                return bias
            if a[j] > b[j]:
                # Effective cue favors A, so inverted tendency chooses B.
                return 1.0 - q_val
            # Effective cue favors B, so inverted tendency chooses A.
            return q_val

        p_core = (
            p0 * direction_prob(0, q_norm)
            + late2 * direction_prob(2, q_slip)
            + late3 * direction_prob(3, q_slip)
        )

    p_a_core = float(np.clip(p_core, 0.0, 1.0))

    # Uniform lapse over the two options.
    p_a = (1.0 - epsilon) * p_a_core + epsilon / 2.0
    p_a = float(np.clip(p_a, 0.0, 1.0))

    return np.array([p_a, 1.0 - p_a], dtype=float)
```

**`policy source code`:**
```python
import numpy as np

def policy(probs):
    p = np.asarray(probs, dtype=float)
    if p.sum() <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / p.sum()
    return int(np.random.choice(len(p), p=p))
```


## COMPETING THEORY
**Description:** Compressive Negative Validity-Weighted integration with subject-level saturation (CNVW-tau). Each subject forms D = sum_j v_j^gamma_i * (A_j - B_j), where v_j are the instructed expert validities and gamma_i is a subject-level validity exponent. The weighted difference is passed through a strongly saturating bounded transform psi(D) = D / (tau_i + abs(D)), with a small subject-level half-saturation constant tau_i. Most subjects apply a negative sign, S_i = -lambda_i * psi(D) + bias_i, while a minority apply a positive sign. Choice probability is p(A) = epsilon_i/2 + (1 - epsilon_i) * sigmoid(beta_i * S_i). There is no feature-tally term and no conflict-gating interaction.

**Parameters:**
- gamma: [1.5, 2.5]
- tau: [0.10, 0.25]
- lambda: [0.8, 2.0]
- beta: [0.8, 2.0]
- bias: [-0.25, 0.25]
- epsilon: [0.03, 0.10]
- p_positive: [0.03, 0.10]
- sign_draw: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    """Compressive negative validity-weighted integration with subject-level tau.

    D = sum_j v_j^gamma * (A_j - B_j)
    psi(D) = D / (tau + abs(D))
    S = sign * lambda * psi(D) + bias
    p(A) = lapse/2 + (1-lapse) * sigmoid(beta * S)

    Most subjects use sign = -1; a minority draw sign = +1.
    history is intentionally ignored because there is no feedback.
    """
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "CNVW-tau expects a (2, n_features) stimulus; "
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
    lam = float(parameters["lambda"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    bias = float(parameters["bias"])
    p_positive = float(parameters["p_positive"])
    sign_draw = float(parameters["sign_draw"])
    tau = float(parameters["tau"])

    # Latent sign mixture: most subjects use negative-valence weighting;
    # a minority trust positive-valence evidence and get a positive sign.
    sign = 1.0 if sign_draw < p_positive else -1.0

    # Validity-weighted feature difference.
    d = float(np.dot(v ** gamma, a - b))

    # Subject-level strongly saturating bounded transform. The small
    # half-saturation constant tau forces most non-negligible validity-
    # weighted differences toward the sign bound, so sign and not
    # continuous magnitude drives the main choice contrast.
    psi = d / (tau + abs(d))

    score = sign * lam * psi + bias
    z = beta * score

    # Numerically stable sigmoid for p(A | score).
    if z >= 0.0:
        p_a_core = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        p_a_core = ez / (1.0 + ez)

    # Uniform lapse mixture over the two options.
    p_a = (1.0 - epsilon) * p_a_core + epsilon / 2.0
    p_a = float(np.clip(p_a, 0.0, 1.0))

    return np.array([p_a, 1.0 - p_a], dtype=float)
```

**`policy source code`:**
```python
import numpy as np

def policy(probs):
    p = np.asarray(probs, dtype=float)
    if p.sum() <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n\n    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.79], dtype=float)\n    gamma = 2.0\n    weights = validities ** gamma\n\n    def d_sign(row):\n        a = np.asarray(row['option_a_ratings'], dtype=float)\n        b = np.asarray(row['option_b_ratings'], dtype=float)\n        d = float(np.dot(weights, a - b))\n        if d > 0.0:\n            return 1\n        if d < 0.0:\n            return -1\n        return 0\n\n    signs = data.apply(d_sign, axis=1)\n    p_a = 1.0 - data['response'].astype(float)\n\n    neg_mean = p_a[signs == -1].mean()\n    pos_mean = p_a[signs == 1].mean()\n\n    return float(neg_mean - pos_mean)",
  "rationale": "Score is mean probability of choosing A on trials with negative validity-weighted difference D minus mean p(A) on trials with positive D, where D = sum_j validities_j^2 * (A_j - B_j). The 12 unique pairs split into six negative-D and six positive-D pairs, and each group contains three top-A and three top-B trials. Under GSS-ITTB the 0.90 cue discriminates on every trial and its slip gate is zero, so choices depend only on whether that cue favors A or B. Because both D-sign groups are balanced over top-A and top-B trials, both group means are 0.5 in expectation, giving a score of exactly 0 under GSS-ITTB. Under CNVW-tau most subjects use a negative sign, so negative-D trials produce high p(A) and positive-D trials produce low p(A), yielding a large positive score. Averaging 48 trials per D-sign group keeps the per-subject binomial noise small."
}
```

## Usage

```json
{
  "input_tokens": 5666,
  "output_tokens": 16531,
  "reasoning_tokens": 16071,
  "provider": "Baidu"
}
```
