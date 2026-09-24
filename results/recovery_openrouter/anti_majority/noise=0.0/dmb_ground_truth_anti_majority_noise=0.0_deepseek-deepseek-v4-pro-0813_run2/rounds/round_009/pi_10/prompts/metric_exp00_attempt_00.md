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
**Validities (n_features=6):** [0.95, 0.8, 0.79, 0.78, 0.76, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 3: A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 4: A=[1, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 5: A=[1, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 7: A=[1, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 1, 0]  B=[1, 0, 0, 1, 0, 0]

**Rationale:** This design isolates the advocated GCAV theory's negative-exponent tail component against the competing TAVU theory's strictly positive gamma and zero-tally dead-zone behavior. Six experts are used. The highest-validity expert (0.95) and lowest-validity expert (0.50) are tied filler cues: in every row the 0.95 expert gives both products 1, the 0.50 expert gives both products 0, and all non-discriminating middle experts give both products 0. Thus every unique pair is a one-versus-one swap between two middle experts, so the unweighted tally T is exactly zero and the number of discriminating features is exactly two. This turns off TAVU's nonzero-tally fallback and leaves TAVU's fragmented-evidence dilution inactive, forcing TAVU into its zero-tally graded anti-validity branch. In the first four rows the higher-validity middle expert favors A: D values at gamma=2 are approximately 0.062, 0.047, 0.032, and 0.031, all far below TAVU's smallest threshold theta=0.15, so TAVU must predict p(A)<0.5 for all four positive rows. Under central TAVU parameters (gamma=2, kappa_dead=0.10, psi=1.15, beta=2.40, lapse=0.02) the predicted p(A) values are about [0.26, 0.30, 0.34, 0.35]. The GCAV bulk component also has D below its minimum dead-zone threshold of 0.06 (maximum bulk D across gamma 1.3-1.8 is only about 0.059), so the bulk anti-validity signal is rho-attenuated near chance. Crucially, GCAV's tail component may draw a negative validity exponent; with gamma_tail=-1 the same rows have negative tail D values, e.g. 0.80^-1 - 0.76^-1 = -0.066. The near-deterministic tail (beta 4-8) therefore chooses A with high probability, lifting the GCAV mixture above 0.5: approximately [0.56, 0.56, 0.55, 0.55]. This yields a signed dissociation on every positive row: GCAV predicts an A majority while TAVU predicts a B majority. The last four rows are exact A/B mirrors, giving the complementary signed predictions. Across the positive D staircase TAVU's p(A) moves monotonically downward as D increases, whereas GCAV remains roughly flat above chance. TAVU cannot produce this majority reversal because its gamma is constrained to positive values and its positive-sign minority is at most 10%; even at the upper p_positive bound the zero-tally branch aggregate stays below 0.5. The eight unique pairs repeat 12 times for the 96-trial budget.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Gated compressive anti-validity with validity-curvature gain and sign-selectable dead-zone repair. Each subject forms D = sum_j v_j^gamma * (A_j - B_j), passes D through a saturating transform D/(tau + |D|), and mostly applies a negative validity-weighting sign. A subject-level threshold theta defines a mild dead zone around D=0: inside this zone the anti-validity score is attenuated and augmented by a bounded graded repair D/(kappa + |D|) whose sign is anti-validity for most subjects but validity-consistent for a minority; outside the zone the full compressive score plus bias is used. A two-component subject mixture provides moderate bulk responding with stronger validity curvature plus a near-deterministic tail with negative or sub-linear validity exponents. There is no strong unweighted tally switch and no fragmented-evidence dilution; only a small tally leak eta*T is retained.

**Parameters:**
- comp_draw: [0.0, 1.0]
- p_tail: [0.12, 0.20]
- gamma_bulk: [1.30, 1.80]
- theta_bulk: [0.06, 0.28]
- tau_bulk: [0.80, 1.80]
- lambda_bulk: [1.00, 1.80]
- beta_bulk: [1.20, 2.50]
- bias_bulk: [-0.05, 0.05]
- rho_bulk: [0.10, 0.22]
- c_repair_bulk: [0.03, 0.10]
- gamma_tail: [-1.80, 0.50]
- theta_tail: [0.00, 0.08]
- tau_tail: [0.08, 0.30]
- lambda_tail: [1.50, 3.00]
- beta_tail: [4.00, 8.00]
- bias_tail: [-0.04, 0.04]
- rho_tail: [0.50, 0.80]
- c_repair_tail: [0.15, 0.45]
- p_positive: [0.02, 0.08]
- sign_draw: [0.0, 1.0]
- lapse: [0.01, 0.04]
- eta: [0.0, 0.04]
- p_repair_positive: [0.12, 0.25]
- repair_sign_draw: [0.0, 1.0]
- c_repair_pos: [0.20, 0.60]
- kappa_repair: [0.03, 0.10]
- validities: validities

**`predict source code`:**
```python
import numpy as np


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            'GCAV expects a (2, n_features) stimulus; got shape '
            + str(stim.shape) + '.'
        )

    a = stim[0]
    b = stim[1]

    v = np.asarray(parameters['validities'], dtype=float)
    if v.size != a.size:
        raise ValueError('validities length does not match n_features.')

    comp_draw = float(parameters['comp_draw'])
    p_tail = float(parameters['p_tail'])

    if comp_draw < p_tail:
        gamma = float(parameters['gamma_tail'])
        theta = float(parameters['theta_tail'])
        tau = float(parameters['tau_tail'])
        lam = float(parameters['lambda_tail'])
        beta = float(parameters['beta_tail'])
        bias = float(parameters['bias_tail'])
        rho = float(parameters['rho_tail'])
        c_repair = float(parameters['c_repair_tail'])
    else:
        gamma = float(parameters['gamma_bulk'])
        theta = float(parameters['theta_bulk'])
        tau = float(parameters['tau_bulk'])
        lam = float(parameters['lambda_bulk'])
        beta = float(parameters['beta_bulk'])
        bias = float(parameters['bias_bulk'])
        rho = float(parameters['rho_bulk'])
        c_repair = float(parameters['c_repair_bulk'])

    lapse = float(parameters['lapse'])
    eta = float(parameters['eta'])
    p_positive = float(parameters['p_positive'])
    sign_draw = float(parameters['sign_draw'])

    p_repair_positive = float(parameters['p_repair_positive'])
    repair_sign_draw = float(parameters['repair_sign_draw'])
    c_repair_pos = float(parameters['c_repair_pos'])
    kappa_repair = float(parameters['kappa_repair'])

    # Mostly negative validity-weighting sign.
    sign = 1.0 if sign_draw < p_positive else -1.0

    # Dead-zone repair sign is also mostly anti-validity, but a minority
    # receive a validity-consistent repair inside the dead zone.
    repair_sign = 1.0 if repair_sign_draw < p_repair_positive else -1.0

    diff = a - b
    D = float(np.dot(np.power(v, gamma), diff))
    T = float(np.mean(diff))

    sat = D / (tau + abs(D))
    core = sign * lam * sat + bias

    if abs(D) < theta:
        # Bounded graded repair, applied identically for all tallies.
        repair_sat = D / (kappa_repair + abs(D))
        if repair_sign > 0:
            repair = c_repair_pos * repair_sat
        else:
            repair = -c_repair * repair_sat
        score = rho * core + repair + eta * T
    else:
        score = core + eta * T

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
  "metric_source": "import numpy as np\nimport pandas as pd\n\n\ndef metric(data: pd.DataFrame) -> float:\n    validities = np.array([0.95, 0.80, 0.79, 0.78, 0.76, 0.50], dtype=float)\n    rows = data[['option_a_ratings', 'option_b_ratings', 'response']].dropna().copy()\n    if rows.empty:\n        return float('nan')\n\n    a = np.vstack([np.asarray(x, dtype=float) for x in rows['option_a_ratings'].tolist()])\n    b = np.vstack([np.asarray(x, dtype=float) for x in rows['option_b_ratings'].tolist()])\n    weighted = (a - b).dot(validities)\n\n    resp = rows['response'].to_numpy(dtype=float)\n    high_choice = np.where(\n        weighted > 0.0,\n        1.0 - resp,\n        np.where(weighted < 0.0, resp, np.nan)\n    )\n\n    valid = high_choice[np.isfinite(high_choice)]\n    if valid.size == 0:\n        return float('nan')\n    return float(valid.mean())\n",
  "rationale": "This metric is the observed rate of choosing the option supported by the more valid of the two discriminating experts. In every designed pair the 0.95 and 0.50 experts are tied fillers and exactly two middle experts disagree, so this rate isolates whether choices follow or invert the communicated validity ordering of the only informative cues. TAVU's zero-tally anti-validity branch applies a mostly negative sign to D and must therefore favor the lower-validity expert on most trials, producing a high-validity choice rate near 0.31-0.35. GCAV's bulk component stays near chance, but its tail component frequently uses negative validity exponents; with the mostly negative sign, that reverses the anti-validity direction for these rows and pushes the high-validity choice rate above chance, around 0.55. The 12 repetitions of all 8 trial types keep the per-subject estimate tight while preserving the mean dissociation."
}
```

## Usage

```json
{
  "input_tokens": 6233,
  "output_tokens": 22502,
  "reasoning_tokens": 21991,
  "provider": "Baidu"
}
```
