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
**Validities (n_features=8):** [0.8, 0.73, 0.78, 0.75, 0.6, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  trial 4: A=[0, 0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 0]

**Rationale:** Design isolates the fragmented-evidence dilution mechanism of TAVU-tn against CNVW-tau's scalar-D architecture. Every row keeps the unweighted tally T=0 exactly, so TAVU's nonzero-tally fallback is never engaged. Rows 1 and 2 are matched on D = 0.80^gamma - 0.73^gamma (≈ +0.107 at gamma=2): row 2 adds one A win and one B win on two duplicate 0.60 experts, which cancels exactly in D and T for any gamma but increases the number of discriminating features n_disc from 2 to 4. The largest possible |D1| across gamma in [1,3] is 0.80^3 - 0.73^3 ≈ 0.123, below every TAVU threshold (minimum 0.15), so all TAVU subjects remain in the zero-tally dead-zone branch. TAVU multiplies that branch's graded score by exp(-rho_disc * max(0, n_disc-2)), so row 2 is strongly diluted toward 0.5 relative to row 1. CNVW-tau is a scalar function of D alone and must predict identical p(A) for rows 1 and 2 for every subject; no beta/lambda/tau/bias/lapse tuning can break this equality. Rows 3 and 4 apply the same 2-vs-4 discrimination contrast to a smaller D2 = 0.78^gamma - 0.75^gamma (≈ +0.046 at gamma=2), also below threshold for all gamma in [1,3]. Rows 5-8 are exact A/B mirrors. Under central illustrative parameters (TAVU: gamma=2, kappa_dead=0.10, psi=1.15, rho_disc=1.15, beta=2.40, lapse=0.02, 5% positive sign; CNVW: gamma=2, tau=0.175, lambda=1.40, beta=1.40, epsilon=0.05, 5% positive sign), TAVU predicts p(A) ≈ [.23,.47,.32,.48,.77,.53,.68,.52] while CNVW predicts ≈ [.35,.35,.41,.41,.65,.65,.59,.59]. The critical result is the within-D n_disc effect: CNVW must be flat within each matched-D pair, whereas TAVU must separate them.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Thresholded validity-weighted anti-validity with a zero-tally dead-zone repair and fragmented-evidence dilution. Each subject forms D = sum_j v_j^{gamma_i} (A_j - B_j). When |D| is below a subject-level ambiguity threshold theta_i, the validity signal is treated as ambiguous. If the unweighted tally is nonzero, choice is driven by that tally through omega_i with a single strong weight, exactly as in the accepted base. If the tally is exactly zero, choice remains graded via a small signed validity-derived anti-validity term psi_i * sign_i * D / (kappa_i + |D|), but this term is now diluted multiplicatively by exp(-rho_disc_i * max(0, n_disc - 2)), where n_disc is the number of discriminating features. This dilution suppresses responding for zero-tally rows with many simultaneously disagreeing features while leaving zero-tally rows with only two discriminating features untouched. When |D| >= theta_i, choice follows the mostly negative anti-validity transform sign_i * lambda_i * D / (tau_i + |D|) + bias_i. Choice probability is p(A) = lapse_i/2 + (1-lapse_i) * sigmoid(beta_i * S_i).

**Parameters:**
- gamma: [1.0, 3.0]
- theta: [0.15, 0.28]
- tau: [0.70, 1.30]
- lam: [1.60, 2.20]
- beta: [2.10, 2.70]
- bias: [-0.05, 0.05]
- lapse: [0.01, 0.04]
- p_positive: [0.03, 0.10]
- sign_draw: [0.0, 1.0]
- omega_mag: [0.45, 0.75]
- p_omega_positive: [0.85, 0.95]
- omega_sign_draw: [0.0, 1.0]
- psi: [0.90, 1.40]
- kappa_dead: [0.06, 0.16]
- rho_disc: [0.9, 1.4]
- validities: validities

**`predict source code`:**
```python
import numpy as np


def predict(parameters, state, history):
    """Thresholded validity-weighted anti-validity with zero-tally dead-zone repair
    and fragmented-evidence dilution.

    D = sum_j v_j^gamma * (A_j - B_j)
    T = sum_j (A_j - B_j)
    n_disc = count of features where A_j != B_j

    If |D| < theta:
        if T == 0:
            dilution = exp(-rho_disc * max(0, n_disc - 2))
            score = psi * sign * D / (kappa_dead + |D|) * dilution
        else:
            score = omega * T
    Else:
        score = sign * lam * D / (tau + |D|) + bias

    p(A) = lapse/2 + (1 - lapse) * sigmoid(beta * score)

    history is intentionally ignored because the task provides no trial-by-trial
    correctness feedback.
    """
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            "Thresholded anti-validity model expects a (2, n_features) stimulus; "
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
    theta = float(parameters["theta"])
    tau = float(parameters["tau"])
    lam = float(parameters["lam"])
    beta = float(parameters["beta"])
    bias = float(parameters["bias"])
    lapse = float(parameters["lapse"])
    omega_mag = float(parameters["omega_mag"])
    p_omega_positive = float(parameters["p_omega_positive"])
    omega_sign_draw = float(parameters["omega_sign_draw"])
    p_positive = float(parameters["p_positive"])
    sign_draw = float(parameters["sign_draw"])
    psi = float(parameters["psi"])
    kappa_dead = float(parameters["kappa_dead"])
    rho_disc = float(parameters["rho_disc"])

    # Mostly negative validity-weighting sign; a minority trust the
    # validity-weighted direction.
    sign = 1.0 if sign_draw < p_positive else -1.0

    # Subject-level tally weight, mostly positive but allowing a minority of
    # subjects to use the opposite tally sign.
    omega = omega_mag if omega_sign_draw < p_omega_positive else -omega_mag

    # Subject-level validity-weighted feature difference.
    w = np.power(v, gamma)
    diff = a - b
    D = float(np.dot(w, diff))

    # Unweighted tally difference used in the dead zone.
    T = float(np.sum(diff))

    # Number of discriminating features, used only in the zero-tally branch.
    n_disc = int(np.count_nonzero(diff))

    if abs(D) < theta:
        # Ambiguous validity signal.
        if abs(T) < 1e-12:
            # Zero-tally repair: equal raw counts no longer force p(A)=0.5.
            # A small signed validity-derived anti-validity term preserves
            # graded responding when the options differ only in which cue is
            # present.  This term is diluted by a factor that shrinks
            # exponentially with the number of discriminating features beyond
            # two, so multi-feature zero-tally rows stay near chance while
            # two-feature rows keep the graded contrast.
            sat_dead = D / (kappa_dead + abs(D))
            dilution = np.exp(-rho_disc * max(0.0, float(n_disc - 2)))
            score = psi * sign * sat_dead * dilution
        else:
            # Nonzero tally drives the ambiguous-regime choice.
            score = omega * T
    else:
        # Clear validity signal: mostly negative anti-validity response through
        # a moderately saturating continuous transform.
        sat = D / (tau + abs(D))
        score = sign * lam * sat + bias

    z = beta * score

    # Numerically stable sigmoid for p(A | score).
    if z >= 0.0:
        p_a_core = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        p_a_core = ez / (1.0 + ez)

    # Uniform lapse mixture over the two options.
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\n\ndef metric(data: pd.DataFrame) -> float:\n    r1 = ((1, 0, 0, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0, 0, 0))\n    r2 = ((1, 0, 0, 0, 1, 0, 0, 0), (0, 1, 0, 0, 0, 1, 0, 0))\n    r3 = ((0, 0, 1, 0, 0, 0, 0, 0), (0, 0, 0, 1, 0, 0, 0, 0))\n    r4 = ((0, 0, 1, 0, 1, 0, 0, 0), (0, 0, 0, 1, 0, 1, 0, 0))\n    r5 = ((0, 1, 0, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0, 0, 0))\n    r6 = ((0, 1, 0, 0, 0, 1, 0, 0), (1, 0, 0, 0, 1, 0, 0, 0))\n    r7 = ((0, 0, 0, 1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0, 0))\n    r8 = ((0, 0, 0, 1, 0, 1, 0, 0), (0, 0, 1, 0, 1, 0, 0, 0))\n\n    wanted = {\n        r1: ('b1', 0),\n        r2: ('f1', 0),\n        r3: ('b2', 0),\n        r4: ('f2', 0),\n        r5: ('b1m', 1),\n        r6: ('f1m', 1),\n        r7: ('b2m', 1),\n        r8: ('f2m', 1),\n    }\n\n    labels = set(v[0] for v in wanted.values())\n\n    def one_subject(df):\n        stats = {label: [0, 0] for label in labels}\n        for _, row in df.iterrows():\n            key = (tuple(row['option_a_ratings']), tuple(row['option_b_ratings']))\n            if key in wanted:\n                label, target_resp = wanted[key]\n                stats[label][0] += 1\n                if row['response'] == target_resp:\n                    stats[label][1] += 1\n\n        p = {}\n        for label, (n, a) in stats.items():\n            p[label] = (a / n) if n > 0 else 0.5\n\n        orient_n = stats['b1'][0] + stats['b1m'][0]\n        orient_a = stats['b1'][1] + stats['b1m'][1]\n        orient_p = (orient_a / orient_n) if orient_n > 0 else 0.5\n        if orient_p == 0.5:\n            s_global = 0.0\n        else:\n            s_global = 1.0 if orient_p < 0.5 else -1.0\n\n        def e(orient_label, basic_test_label, frag_test_label):\n            if p[orient_label] == 0.5:\n                return 0.0\n            s = 1.0 if p[orient_label] < 0.5 else -1.0\n            return s * (p[frag_test_label] - p[basic_test_label])\n\n        d1a = e('b1', 'b1m', 'f1m')\n        d1b = e('b1m', 'b1', 'f1')\n        d2a = s_global * (p['f2'] - p['b2'])\n        d2b = s_global * (p['f2m'] - p['b2m'])\n\n        return 0.25 * (d1a + d1b + d2a + d2b)\n\n    values = [one_subject(g) for _, g in data.groupby('subject_id', sort=False)]\n    return float(np.mean(values)) if values else 0.0",
  "rationale": "This metric measures a cross-validated oriented dilution contrast. The design pairs rows with identical validity-weighted difference D but different numbers of discriminating features: 2 versus 4. CNVW is a function of D alone, so each matched pair must have the same underlying probability of choosing the anchor feature, making the oriented basic-minus-fragmented differences zero in expectation. TAVU instead multiplies the zero-tally branch by exp(-rho_disc * max(0, n_disc - 2)), so the 4-feature version moves closer to chance than the 2-feature version. To avoid cancellation across subjects with positive versus negative validity signs, the metric reorients choices relative to the anchor feature and infers each subject's sign on the stronger D1 basic stimulus pair, then tests the dilution effect on independent mirror pairs and D2 pairs. This keeps per-subject estimates based on 12 repetitions per row and yields a large positive average under TAVU while remaining near zero under CNVW."
}
```

## Usage

```json
{
  "input_tokens": 5688,
  "output_tokens": 40605,
  "reasoning_tokens": 39297,
  "provider": "Baidu"
}
```
