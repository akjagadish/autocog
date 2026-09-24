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
**Validities (n_features=6):** [1.0, 1.0, 1.0, 0.9, 0.9, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 4: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 6: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 7: A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]

**Rationale:** This experiment uses six experts with validities [1.0, 1.0, 1.0, 0.9, 0.9, 0.5]. The mean validity is 0.8833, so the centered validities are [0.1167, 0.1167, 0.1167, 0.0167, 0.0167, -0.3833]. The top tier is exactly the three 1.0 experts; the second highest validity is 0.9, giving a validity gap of 0.10 and a top size of 3, which opens the advocated theory's small-elite tie-break gate.

Every unique pair is a tally tie: d = sum(A - B) = 0. On these pairs the advocated theory's default tallying signal is silent, and its gated tie-break consults only the three top experts. Let T be the top sub-tally, the number of top experts favoring A minus the number favoring B, and let S be the centered-validity score, S = sum c*(A - B). The construction uses the two 0.9 experts and the one 0.5 expert to vary S independently of T; in this validity set S = 0.5*T + 0.4*U, where U is the difference sum of the two 0.9 experts.

Trials 1-6 have T = +1, +1, +1, +2, +2, +3 with S values -0.30, 0.10, 0.50, 0.20, 0.60, 0.70. Trials 7-12 are the exact A/B mirrors with T = -1, -1, -1, -2, -2, -3 and S values 0.30, -0.10, -0.50, -0.20, -0.60, -0.70.

The advocated theory predicts p(A) follows T only: the three T=+1 pairs are indistinguishable around 0.57-0.61, the two T=+2 pairs around 0.645-0.71, and the T=+3 pair around 0.71-0.79, with mirrored values below 0.5. In particular, within T=+1, the large S change from -0.30 to +0.50 should not move p(A).

The competing validity-graded integrator has d = 0 on every trial, so its only advantage term is kappa*S. Because kappa has a symmetric zero-centered prior, the expected p(A) is exactly 0.5 for every one of the 12 pairs, regardless of T. Positive-kappa subjects would order the choices by S, with P1 below 0.5 and P3 above 0.5, while negative-kappa subjects would reverse that order. Thus the design gives a quantitative dissociation: a T-ordered elite sub-tally psychometric function for the advocated theory versus a flat-at-chance S-only pattern for the competing theory. The 12 unique pairs are repeated K = 8 times to give 96 trials.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Tally-first choice with a salience-gated and size-regularized validity tie-break. On every trial the unweighted signed tally of the feature differences is the default choice signal. When the tally is clear, choice follows it through a moderate softmax plus an explicit lapse. When the tally is tied or near-tied, validity is allowed to matter only if the task's validity distribution itself advertises a clear elite: either a small elite block of two to three maximally valid cues separated from the remaining cues, or a large validity gap combined with a bounded elite-block size. In those gated cases people compute a restricted sub-tally over the elite cues and choose its favored option stochastically. Larger elite blocks receive an additional lapse, preventing overconfident validity-following for wide top tiers. In large stimulus arrays, near-zero tallies are treated as low-confidence and are heavily regularized with extra choice noise, so deterministic regression blows-up do not occur.

**Parameters:**
- beta_tally: [2.0, 4.0]
- epsilon_tally: [0.02, 0.08]
- beta_near: [0.3, 0.8]
- epsilon_near: [0.78, 0.97]
- near_tally_bound: {1.5}
- n_low_conf: {15}
- tau_tie: [0.0, 0.5]
- salience_gap_hi: [0.30, 0.45]
- salience_gap_lo: [0.04, 0.06]
- top_tol: {0.01}
- top_size_small_min: {2}
- top_size_small_max: {3}
- top_size_block_max: {6}
- beta_sub: [0.35, 0.50]
- epsilon_sub: [0.08, 0.14]
- epsilon_sub_size: [0.16, 0.20]
- validities: validities

**`predict source code`:**
```python
import numpy as np


def predict(parameters, state, history):
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError('Expected a (2, n_features) stimulus.')

    validities = np.asarray(parameters['validities'], dtype=float)
    n_features = stim.shape[1]
    if validities.shape[0] != n_features:
        raise ValueError('validities length mismatch.')

    a = stim[0]
    b = stim[1]
    diff = a - b
    tally = float(np.sum(diff))

    vmax = float(np.max(validities))
    top_tol = float(parameters['top_tol'])
    top_mask = validities >= (vmax - top_tol)
    top_size = int(np.sum(top_mask))

    if top_size < n_features:
        second_max = float(np.max(validities[~top_mask]))
    else:
        second_max = vmax
    validity_gap = vmax - second_max

    gap_hi = float(parameters['salience_gap_hi'])
    gap_lo = float(parameters['salience_gap_lo'])
    size_min = int(parameters['top_size_small_min'])
    size_max = int(parameters['top_size_small_max'])
    block_max = int(parameters['top_size_block_max'])

    block_clear = (validity_gap >= gap_hi) and (top_size <= block_max)
    small_elite = (size_min <= top_size <= size_max) and (validity_gap >= gap_lo)
    salience = 1.0 if (block_clear or small_elite) else 0.0

    def stable_softmax(scores, inv_temp):
        scores = np.asarray(scores, dtype=float)
        z = inv_temp * (scores - np.max(scores))
        z = np.clip(z, -30.0, 30.0)
        e = np.exp(z)
        return e / np.sum(e)

    tau_tie = float(parameters['tau_tie'])

    if salience > 0.5 and abs(tally) <= tau_tie:
        top_diff = diff[top_mask]
        sub_a = float(np.sum(top_diff > 0.0))
        sub_b = float(np.sum(top_diff < 0.0))
        sub_tally = sub_a - sub_b

        beta_sub = float(parameters['beta_sub'])
        eps_sub = float(parameters['epsilon_sub'])
        eps_sub_size = float(parameters['epsilon_sub_size'])
        eps_sub_eff = eps_sub + eps_sub_size * max(0.0, float(top_size) - 3.0)
        eps_sub_eff = float(np.clip(eps_sub_eff, 0.0, 0.95))

        if abs(sub_tally) > 1e-12:
            core = stable_softmax(np.array([sub_tally, -sub_tally]), beta_sub)
            probs = (1.0 - eps_sub_eff) * core + eps_sub_eff * np.array([0.5, 0.5])
        else:
            probs = np.array([0.5, 0.5])
        return probs

    near_bound = float(parameters['near_tally_bound'])
    n_low_conf = int(parameters['n_low_conf'])

    if abs(tally) <= near_bound and n_features >= n_low_conf:
        beta = float(parameters['beta_near'])
        eps = float(parameters['epsilon_near'])
    else:
        beta = float(parameters['beta_tally'])
        eps = float(parameters['epsilon_tally'])

    core = stable_softmax(np.array([tally, -tally]), beta)
    probs = (1.0 - eps) * core + eps * np.array([0.5, 0.5])
    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if total <= 0.0:
        return np.array([0.5, 0.5])
    return probs / total
```

**`policy source code`:**
```python
import numpy as np


def policy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    total = p.sum()
    if total <= 0.0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / total
    return int(np.random.choice(len(p), p=p))
```


## COMPETING THEORY
**Description:** People use a validity-graded compensatory integration rule. On every choice they inspect all features, compute a signed option advantage by summing each discriminating feature's contribution, and then choose via a softmax over that advantage plus an independent lapse. Unlike Take The Best there is no early stopping. Unlike pure Tallying, each feature's contribution is weighted by a validity-sensitive multiplier. The multiplier is anchored around equal weighting, w_j = 1 + kappa * (v_j - mean(v)), normalized so the average weight is exactly 1. Because kappa has a symmetric prior centered at zero, the population-level expected weight vector is the equal-weight vector, but individual subjects may slightly overweight or underweight higher-validity cues. Tied features contribute nothing.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- kappa: [-0.10, 0.10]
- validities: validities

**`predict source code`:**
```python
import numpy as np

def predict(parameters, state, history):
    # Validity-graded compensatory integration.
    # state is the current pair of option feature vectors, shape (2, n_features).
    # Row 0 = option A, row 1 = option B. History is irrelevant because there is
    # no trial-by-trial feedback and the rule is applied independently each trial.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Validity-graded integration expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    validities = np.asarray(parameters["validities"], dtype=float)
    n_features = stim.shape[1]
    if validities.shape[0] != n_features:
        raise ValueError(
            f"validities length {validities.shape[0]} != n_features {n_features}."
        )

    # Validity-sensitive weights around a strict equal-weight anchor.
    # The centered validities sum to zero, so the raw weights sum to n_features.
    # Normalizing by their mean keeps the scale of the advantage score directly
    # comparable to an unweighted signed tally when kappa is small.
    kappa = float(parameters["kappa"])
    centered_validities = validities - np.mean(validities)
    weights = 1.0 + kappa * centered_validities
    weights = weights / np.mean(weights)

    # Discriminating features contribute according to their weights; ties
    # have difference zero and thus contribute nothing.
    advantage = float(np.dot(weights, stim[0] - stim[1]))

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Stable softmax over the signed advantage versus a zero-advantage boundary.
    # This reduces to logistic choice in the weighted advantage.
    scores = np.array([advantage, 0.0])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)

    n_opts = 2
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
import numpy as np

def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\n\ndef metric(data):\n    # Top three features are the three perfectly valid experts in this design.\n    a_top = np.array([float(r[0] + r[1] + r[2]) for r in data['option_a_ratings']])\n    b_top = np.array([float(r[0] + r[1] + r[2]) for r in data['option_b_ratings']])\n    top_tally = a_top - b_top\n\n    chose_a = 1.0 - np.asarray(data['response'], dtype=float)\n\n    mask_pos3 = top_tally == 3\n    mask_neg3 = top_tally == -3\n\n    if mask_pos3.sum() == 0 or mask_neg3.sum() == 0:\n        return 0.0\n\n    p_a_given_top_plus3 = chose_a[mask_pos3].mean()\n    p_a_given_top_minus3 = chose_a[mask_neg3].mean()\n    return float(p_a_given_top_plus3 - p_a_given_top_minus3)\n",
  "rationale": "For the three elite validity-1.0 experts, define the top sub-tally T = (a0 + a1 + a2) - (b0 + b1 + b2). The metric is mean chose_A on rows with T = +3 minus mean chose_A on rows with T = -3, where chose_A = 1 - response. The advocated theory predicts that the gated tie-break makes A much more likely when all three elite experts favor A, and much less likely when all three favor B, so this signed difference is large and positive. For the competing validity-graded integrator these two rows have centered-validity scores S = +0.7 and S = -0.7; with a symmetric zero-centered kappa prior, expected p(A) is 0.5 for both rows, so the signed contrast averages to zero. The same contrast is computed on pooled and per-subject slices, keeping its scale stable."
}
```

## Usage

```json
{
  "input_tokens": 5419,
  "output_tokens": 21864,
  "reasoning_tokens": 21391,
  "provider": "Baidu"
}
```
