# metric_exp00_attempt_02

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
**Validities (n_features=6):** [0.707106, 0.707106, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 3: A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]

**Rationale:** CNVW-tau is a scalar-D theory: D = sum_j v_j^gamma * (A_j - B_j), psi(D) = D / (tau + abs(D)), and p(A) cannot depend on how a zero D is built. The validities [0.7071, 0.7071, 0.5, 0.5, 0.5, 0.5] give v^2 weights of 0.5 for the two high-validity cues and 0.25 for the four low-validity cues at gamma = 2, so one high-cue win exactly cancels two low-cue wins. All four unique pairs are engineered to have D = 0 exactly at gamma = 2, while their unweighted tally differences (sum of A ratings minus sum of B ratings) are -2, -1, +1, +2, i.e. T = -1/3, -1/6, +1/6, +1/3. CNVW-tau therefore predicts a flat p(A) across all four rows because psi(0) = 0 and only bias/lapse can move the probability, with no tally term. CMCAV-iv also has a zero saturating term at D = 0, but it adds w_t * T with negative w_t in [-0.32, -0.14], so its score decreases monotonically as T increases. Under central CMCAV parameters (beta = 2.0, w_t = -0.23, bias = 0, lapse = 0.06) the predicted p(A) values are approximately 0.536, 0.518, 0.482, 0.464; the endpoint separation grows with beta and abs(w_t). This creates a flat-versus-tally-graded dissociation: CNVW-tau cannot make D = 0 trials depend on T, while CMCAV-iv cannot remove the tally gradient unless w_t = 0, which is outside its range. The four unique pairs are repeated 24 times each for 96 trials.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
**Description:** Conflict-modulated continuous anti-validity with inverse-validity conflict counting and a tally-regularized response (CMCAV-iv). Subjects form a continuous validity-weighted difference D = sum_j v_j^gamma_i * (A_j - B_j), mostly apply a negative validity-weighting sign, and respond through a fluid evidence-gated continuous transform S_i = -lambda_i * G(C) * D / (kappa_i + abs(D) + r_i) + bias_i + w_t * T. There is no fixed threshold, no dead zone, and no single fixed half-saturation constant. The conflict gain G(C) depends on both concentration of the dominant validity-weighted contribution and an inverse-validity-weighted count of disagreeing features, so conflict from many low-validity cues strongly suppresses responding while concentrated high-validity evidence remains effective. A small unidirectional tally term T = (1/n) * sum_j (A_j - B_j) with negative weight supports count-driven anti-validity gradients.

**Parameters:**
- gamma: [1.6, 2.4]
- lam: [1.3, 2.2]
- beta: [1.5, 2.5]
- bias: [-0.06, 0.06]
- lapse: [0.03, 0.09]
- kappa: [0.02, 0.06]
- r: [0.00, 0.02]
- p_positive: [0.02, 0.10]
- sign_draw: [0.0, 1.0]
- w_t: [-0.32, -0.14]
- rho: [5.0, 8.0]
- eta: [1.5, 2.0]
- conc_exp: [1.4, 1.9]
- nu: [0.5, 1.0]
- xi: [0.0, 0.20]
- g_min: [0.02, 0.06]
- g_range: [2.5, 3.2]
- c_half: [0.4, 0.7]
- validities: validities

**`predict source code`:**
```python
import numpy as np


def predict(parameters, state, history):
    """Conflict-modulated continuous anti-validity with inverse-validity
    conflict counting and a single negative tally term (CMCAV-iv).

    D = sum_j v_j^gamma * (A_j - B_j)
    S = sign * lam * G(C) * D / (kappa + abs(D) + r)
        + bias + w_t * (1/n) * sum_j (A_j - B_j)

    G(C) uses a concentration-style conflict gain whose conflict
    denominator is an inverse-validity weighted count of disagreeing
    features, not a raw disagreement count.
    """
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "CMCAV-iv expects a (2, n_features) stimulus; "
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
    lam = float(parameters["lam"])
    beta = float(parameters["beta"])
    bias = float(parameters["bias"])
    lapse = float(parameters["lapse"])
    kappa = float(parameters["kappa"])
    r = float(parameters["r"])
    p_positive = float(parameters["p_positive"])
    sign_draw = float(parameters["sign_draw"])

    rho = float(parameters["rho"])
    eta = float(parameters["eta"])
    conc_exp = float(parameters["conc_exp"])
    nu = float(parameters["nu"])
    xi = float(parameters["xi"])
    g_min = float(parameters["g_min"])
    g_range = float(parameters["g_range"])
    c_half = float(parameters["c_half"])
    w_t = float(parameters["w_t"])

    # Mostly negative validity-weighting sign.
    sign = 1.0 if sign_draw < p_positive else -1.0

    # Validity-weighted contributions.
    w = np.power(v, gamma)
    diff = a - b
    contrib = w * diff
    D = float(np.sum(contrib))

    mag = np.abs(contrib)
    M = float(np.sum(mag))
    disc_mask = mag > 1e-12

    # Concentration-style conflict gain.
    if M <= 1e-12:
        G = g_min + 0.5 * g_range
    else:
        dom_idx = int(np.argmax(mag))
        max_m = float(mag[dom_idx])
        v_dom = float(v[dom_idx])

        # Dominant share of total validity-weighted evidence.
        cohesion = max_m / M

        # Inverse-validity weighted disagreement count. High-validity
        # disagreeing cues are nearly free, while low-validity disagreeing
        # cues build conflict and suppress the gain.
        inv_validity = np.power(1.0 - v, eta)
        low_conflict = float(np.sum(inv_validity[disc_mask]))
        breadth = 1.0 + rho * low_conflict

        C = (cohesion ** conc_exp) * (v_dom ** nu)
        # Small residual coherence exponent so broad but internally
        # consistent evidence is not fully suppressed.
        C *= ((abs(D) / M) ** xi)
        C /= breadth
        C = float(np.clip(C, 0.0, 20.0))

        G = g_min + g_range * C / (c_half + C)

    G = float(np.clip(G, 0.0, 3.0))

    # Single negative tally term: p(A) is pushed against the count-based
    # validity direction regardless of D's sign.
    T = float(np.mean(diff))

    # Continuous anti-validity response: no fixed threshold and no one
    # fixed half-saturation constant.
    sat = D / (kappa + abs(D) + r)
    score = sign * lam * G * sat + bias + w_t * T
    z = beta * score

    # Numerically stable sigmoid.
    if z >= 0.0:
        p_a_core = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        p_a_core = ez / (1.0 + ez)

    p_a = (1.0 - lapse) * p_a_core + lapse / 2.0
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
[0] rationale: This metric uses the outer-vs-inner pair contrast. Under CNVW-tau, the outer pairs have exactly twice the validity-weighted difference of the inner pairs and pass through a single monotone saturating transform, so the absolute choice difference between row 1 and row 4 should always be larger than the absolute difference between row 2 and row 3. Under CMCAV-iv, the broad outer pairs have much stronger inverse-validity conflict suppression than the narrower inner pairs, and the tally term can further change the signed contrasts; this often reverses the ordering so the inner pair becomes more extreme. The metric therefore codes each subject as +1 when the outer pair difference dominates and -1 when the inner pair difference dominates, then averages over subjects.
metric_source:
def metric(data):
    import numpy as np
    df = data.copy()
    df['a_sum'] = df['option_a_ratings'].apply(lambda x: int(np.sum(x)))
    df['b_sum'] = df['option_b_ratings'].apply(lambda x: int(np.sum(x)))
    df['diff'] = df['a_sum'] - df['b_sum']
    row_map = {-2: 1, -1: 2, 1: 3, 2: 4}
    df['row'] = df['diff'].map(row_map)
    vals = []
    for sid, sdf in df.groupby('subject_id'):
        bprop = sdf.groupby('row')['response'].mean()
        pA = 1.0 - bprop
        p1 = pA.get(1, 0.5)
        p2 = pA.get(2, 0.5)
        p3 = pA.get(3, 0.5)
        p4 = pA.get(4, 0.5)
        X = p1 - p4
        Y = p2 - p3
        ax = abs(X)
        ay = abs(Y)
        if ax > ay:
            vals.append(1.0)
        elif ax < ay:
            vals.append(-1.0)
        else:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0
outcome: self_sim=0.3800 (var=0.8356) adversary_sim=0.1000 (var=0.9300) welch_t=+1.054 p=0.2973 (N=25, alpha=0.01) -> reject

[1] rationale: The four row types have tally differences -1/3, -1/6, +1/6, +1/3. Under CNVW-tau the validity-weighted difference D is monotone across these rows and passes through a monotone psi and sigmoid, so the row p(A) sequence is monotone: the edge steps d12 and d34 have the same sign as the middle step d23 for every subject, regardless of whether the monotone direction is increasing or decreasing. Under CMCAV-iv, the negative tally term and the conflict-gated D term can oppose each other, especially for positive gamma deviations, producing an alternating pattern where the middle step d23 points opposite to the outer edge steps d12 and d34. The metric therefore collapses each subject's four row p(A) estimates into a bounded sign-consistency score: it standardizes the edge contrast d12+d34 and the middle contrast d23 by their binomial standard errors, then applies tanh to their product. A monotone CNVW-tau pattern gives a strong positive score; an alternating CMCAV-iv pattern gives a strong negative score. The score is averaged across subjects, is bounded in [-1, 1], and is insensitive to the overall increasing/decreasing direction.
metric_source:
import numpy as np


def metric(data: pd.DataFrame) -> float:
    df = data.copy()
    df['a_sum'] = df['option_a_ratings'].apply(lambda x: sum(x))
    df['b_sum'] = df['option_b_ratings'].apply(lambda x: sum(x))
    df['tally'] = df['a_sum'] - df['b_sum']
    row_map = {-2: 1, -1: 2, 1: 3, 2: 4}
    df['row'] = df['tally'].map(row_map)

    scores = []
    for sid, sdf in df.groupby('subject_id'):
        p = {}
        n = {}
        for r in range(1, 5):
            mask = sdf['row'] == r
            n[r] = int(mask.sum())
            if n[r] == 0:
                p[r] = 0.5
            else:
                p[r] = 1.0 - float(sdf.loc[mask, 'response'].mean())

        d1 = p[2] - p[1]
        d2 = p[3] - p[2]
        d3 = p[4] - p[3]

        v = {}
        for r in range(1, 5):
            if n[r] > 0:
                v[r] = p[r] * (1.0 - p[r]) / n[r]
            else:
                v[r] = 0.25 / 24.0

        eps = 0.01
        edge_se = float(np.sqrt(v[1] + v[2] + v[3] + v[4] + eps))
        mid_se = float(np.sqrt(v[2] + v[3] + eps))

        edge_z = (d1 + d3) / edge_se if edge_se > 0 else 0.0
        mid_z = d2 / mid_se if mid_se > 0 else 0.0

        score = float(np.tanh(2.0 * edge_z * mid_z))
        scores.append(score)

    if not scores:
        return 0.0
    return float(np.mean(scores))

outcome: self_sim=0.1972 (var=0.5895) adversary_sim=-0.2308 (var=0.3390) welch_t=+2.221 p=0.03146 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "import numpy as np\nimport pandas as pd\n\n\ndef metric(data: pd.DataFrame) -> float:\n    df = data.copy()\n    df['a_sum'] = df['option_a_ratings'].apply(lambda x: int(np.sum(x)))\n    df['b_sum'] = df['option_b_ratings'].apply(lambda x: int(np.sum(x)))\n    df['tally'] = df['a_sum'] - df['b_sum']\n\n    # The four unique pair types are fully determined by their\n    # unweighted tally difference in this design.\n    row_map = {-2: 1, -1: 2, 1: 3, 2: 4}\n    df['row'] = df['tally'].map(row_map)\n\n    subject_scores = []\n    for sid, sdf in df.groupby('subject_id'):\n        logits = []\n        for r in range(1, 5):\n            mask = sdf['row'] == r\n            n = int(mask.sum())\n            if n == 0:\n                logits.append(0.0)\n                continue\n\n            n_a = int((mask & (sdf['response'] == 0)).sum())\n            p_a = (n_a + 1.0) / (n + 2.0)\n            p_a = min(max(p_a, 0.025), 0.975)\n            logits.append(float(np.log(p_a / (1.0 - p_a))))\n\n        subject_scores.append(float(np.var(np.asarray(logits))))\n\n    return float(np.mean(subject_scores)) if subject_scores else 0.0",
  "rationale": "This metric measures the dispersion of a subject's four row-wise choice log-odds. CNVW-tau uses a strongly saturating transform with small tau, so subjects with gamma away from 2 produce large validity-weighted differences and the four log-odds are pushed far apart, regardless of whether the monotone direction is increasing or decreasing. CMCAV-iv instead has a fluid conflict-gated transform and only a weak negative tally term, so the four row log-odds remain close to chance. Averaging subject-level logit variances, rather than pooled row proportions, prevents individual sign flips in CNVW-tau from cancelling and keeps the metric stable per subject."
}
```

## Usage

```json
{
  "input_tokens": 6610,
  "output_tokens": 53326,
  "reasoning_tokens": 52785,
  "provider": "Baidu"
}
```
