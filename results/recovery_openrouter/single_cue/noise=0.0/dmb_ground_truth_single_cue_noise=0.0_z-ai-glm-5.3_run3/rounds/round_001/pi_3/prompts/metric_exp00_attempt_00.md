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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.6]

**Trial pairs (n=32):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 4: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 8: A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 9: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 10: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 11: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 12: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 13: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 14: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 15: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 16: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 18: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 19: A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 1, 0]
  trial 20: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 21: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 22: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 23: A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]
  trial 24: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  trial 25: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 26: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 27: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 28: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 29: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 30: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 31: A=[0, 0, 0, 1, 0]  B=[1, 1, 1, 0, 0]
  trial 32: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** This design dissociates NVW (validity-weighted, capacity-limited integration with one-reason lapses) from Tallying by exploiting the fact that Tallying's predicted P(A) depends ONLY on the win-count margin: for any beta and epsilon, Tallying assigns exactly ONE probability to every trial with the same (a_wins, b_wins), and exactly 0.5 to every tie trial. NVW instead assigns a graded probability ordered by the validity-weighted evidence difference d = sum(v^gamma over A-won cues) - sum(v^gamma over B-won cues), plus a small one-reason tilt toward the highest-validity discriminating cue. I use n=5 (one cue beyond the working-memory span, so the interference factor delta is active but integration remains informative: beta_eff = beta*delta in [0.9, 3.3]) with a wide validity spread [0.95, 0.85, 0.75, 0.65, 0.55]. The design contains four dissociation axes, each with mirrored (A/B-swapped) versions to absorb side bias: (1) MARGIN-0 VALIDITY GRADIENT (types 1-5): 2-2 tie trials where the validity allocation varies from strongly pro-A (d=+0.60: A wins the two most valid cues) through d=+0.40, +0.20, 0.00, to a top-cue-conflict probe (d=+0.10 with the top cue favoring B). Tallying predicts EXACTLY 0.5 on all five types for every parameter value; NVW predicts a monotone deviation from 0.5 whose sign is parameter-free (because v^gamma preserves validity order for any gamma>0, A winning {f0,f1} vs B winning {f3,f4} yields d>0 at any gamma in [0.7,1.3]), with magnitude sigma(beta_eff*d) roughly 0.61-0.93 at d=0.60. (2) WITHIN-MARGIN-1 SPREAD (types 6-14): nine 3-2 trials spanning d from 1.35 down to 0.15. Tallying predicts an identical probability sigma(beta) on ALL nine; NVW predicts a monotone spread of at least ~0.12 (and up to ~0.5) across the aligned subseries (types 6-10, where the top cue also favors A, so both the integration and one-reason components agree). (3) ORDER REVERSAL: NVW predicts P(type 1; margin 0, d=0.60) > P(type 14; margin 1, d=0.15) strictly for every admissible parameter setting (the d-gap is positive at all gamma, and the one-reason lapse favors A on type 1 but B on type 14), whereas Tallying predicts the strict opposite ordering (0.5 < sigma(beta)). The sign of P(margin-0, high-d) - P(margin-1, low-d) is therefore a decisive, parameter-free test. (4) MATCHED-d ALLEGIANCE FLIP: types 9 vs 11 (both d=0.75) and 10 vs 12 (both d=0.55) hold counts AND the validity-weighted evidence identical while flipping which option wins the top cue; Tallying predicts exactly zero difference, NVW predicts a small positive difference (the one-reason lapse signature, magnitude up to ~0.07). Types 15-16 are margin-2 anchors probing the steepness of the margin gradient (Tallying: sigma(2beta); NVW: sigma(beta_eff*1.90) vs sigma(beta_eff*1.00), itself a within-margin spread Tallying cannot produce). In total Tallying predicts only THREE distinct choice probabilities across all 32 pairs (margin 0, 1, 2 and their complements), while NVW predicts a graded profile ordered by d with allegiance modulations - a sharp structural signature. With 32 unique pairs, K = 96 // 32 = 3, giving exactly 96 trials per subject.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Noisy Validity-weighted evidence integration with capacity-interference and one-reason lapses (NVW). People do not use a single fixed heuristic; they pool all the expert ratings they can, but each cue's contribution is weighted by its validity raised to a steepness exponent gamma (gamma = 0 recovers Tallying, gamma -> infinity recovers Take-The-Best, so both classic heuristics are limiting cases). Integration is capacity-limited: beyond a working-memory span of about four cues, every additional cue that must be pooled multiplies the reliability (effective inverse temperature) of the integrated comparison by an interference factor delta < 1, so choice consistency is design-dependent and falls sharply as feature count grows. Each cue is additionally attended only with probability alpha (lapses of attention). On a minority of trials — with a probability omega that is modulated by how strongly the most valid discriminating cue's validity dominates the validity of the opposing side's cues — the agent abandons integration and makes a one-reason decision based solely on that top cue. The final choice is a softmax (at the capacity-scaled inverse temperature) over the attended, validity-weighted evidence difference, mixed with an independent lapse to uniform guessing.

**Parameters:**
- gamma: [0.7, 1.3]
- alpha: [0.9, 1.0]
- beta: [5.0, 11.0]
- delta: [0.18, 0.30]
- epsilon: [0.0, 0.06]
- omega_base: [0.0, 0.04]
- omega_dom: [-0.1, 0.1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # NVW: Noisy Validity-weighted integration with capacity-interference
    # and one-reason lapses.
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"NVW expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n}."
        )

    gamma = float(parameters["gamma"])
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    delta = float(parameters["delta"])
    epsilon = float(parameters["epsilon"])
    omega_base = float(parameters["omega_base"])
    omega_dom = float(parameters["omega_dom"])

    # (1) Validity-steepness weights: gamma = 0 -> Tallying (equal weights),
    # gamma -> infinity -> Take-The-Best (top cue only).
    w = np.power(v, gamma)

    # (3) Attention: each cue independently attended with probability alpha.
    att = np.random.random(n) < alpha
    a_win = (a > b) & att
    b_win = (b > a) & att
    disc = a_win | b_win

    # (3) Capacity / interference: beyond a working-memory span of ~4 cues,
    # each additional cue that must be pooled multiplies the reliability
    # (effective inverse temperature) of the integrated evidence by delta.
    # This makes decision noise grow with feature count.
    excess = max(0, n - 4)
    beta_eff = beta * (delta ** excess)

    def _sig(z):
        # Numerically stable logistic.
        return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))

    if not np.any(disc):
        # No attended discriminating cue: pure guess.
        p_a = 0.5
    else:
        # (1) Integration: softmax over the weighted evidence difference.
        d = float(np.sum(w[a_win]) - np.sum(w[b_win]))
        p_int = float(_sig(beta_eff * d))

        # (2) One-reason mode: the single most valid attended
        # discriminating cue decides (TTB-like), with confidence scaled
        # by that cue's weight and the effective sensitivity.
        idx = np.where(disc)[0]
        top = int(idx[np.argmax(v[idx])])
        s = 1.0 if a[top] > b[top] else -1.0
        p_1r = float(_sig(beta_eff * w[top] * s))

        # (2) Validity-dominance coupling for the one-reason lapse rate:
        # dominance of the top cue over the opposing side's cue validities.
        opp = b_win if s > 0 else a_win
        if np.any(opp):
            dom = float(v[top] - np.mean(v[opp]))
        else:
            dom = float(v[top] - 0.5)
        omega_t = float(np.clip(omega_base + omega_dom * dom, 0.0, 0.95))

        p_a = (1.0 - omega_t) * p_int + omega_t * p_1r

    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (4) Independent lapse toward uniform guessing.
    probs = (1.0 - epsilon) * probs + epsilon * 0.5
    return probs
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\n\ndef _to_vec(x):\n    \"\"\"Robustly convert a cell (list/tuple/ndarray/str) to a float 1-D array.\"\"\"\n    if isinstance(x, np.ndarray):\n        return x.astype(float)\n    if isinstance(x, (list, tuple)):\n        return np.asarray(list(x), dtype=float)\n    s = str(x).strip().replace(\"[\", \" \").replace(\"]\", \" \").replace(\",\", \" \")\n    return np.asarray(s.split(), dtype=float)\n\n\ndef metric(data: pd.DataFrame) -> float:\n    df = data.reset_index(drop=True)\n    n_rows = len(df)\n    if n_rows == 0:\n        return 0.0\n\n    A = np.vstack([_to_vec(x) for x in df[\"option_a_ratings\"]])\n    B = np.vstack([_to_vec(x) for x in df[\"option_b_ratings\"]])\n    n_feat = A.shape[1]\n\n    # Design validities [0.95, 0.85, 0.75, 0.65, 0.60]; fall back gracefully.\n    v = np.array([0.95, 0.85, 0.75, 0.65, 0.60], dtype=float)\n    if v.shape[0] != n_feat:\n        v = np.linspace(0.95, 0.60, n_feat)\n\n    # Chose-A indicator (response: 0 = A, 1 = B).\n    y = (np.asarray(df[\"response\"], dtype=float) == 0.0).astype(float)\n\n    a_win = A > B\n    b_win = B > A\n\n    # Signed tally margin (Tallying's ONLY explanatory variable).\n    margin = a_win.sum(axis=1).astype(float) - b_win.sum(axis=1).astype(float)\n\n    # Validity-weighted evidence difference (NVW's driving quantity).\n    d = (a_win * v).sum(axis=1) - (b_win * v).sum(axis=1)\n\n    # OLS: y ~ 1 + margin + d. Return the partial coefficient on d.\n    X = np.column_stack([np.ones(n_rows), margin, d])\n    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)\n    return float(beta[2])",
  "rationale": "The metric is the partial OLS coefficient of the validity-weighted evidence difference d = sum(v over A-won cues) - sum(v over B-won cues) in a regression of the chose-A indicator on [intercept, signed tally margin, d], pooled over all trials.\n\nWHY IT IS EXACTLY ZERO UNDER TALLYING (for every beta, epsilon): Tallying's P(A) = sigmoid(beta*(a_wins - b_wins)) (plus a symmetric lapse), i.e., a function of the signed margin m alone. The design's trial set is perfectly mirrored within every margin level: the d-values of all trials sharing a given margin sum to exactly zero (margin 0: +0.55,+0.40,+0.15,0,+0.05,-0.55,-0.40,-0.15,0,-0.05; margin +1: +1.30...+0.20 with the exact negatives in margin -1; margin +2: +1.90,+1.05 with exact negatives). Hence E[d | m] = 0 for each m, so the residual of d after projecting on m is orthogonal to any function of m, including Tallying's (possibly nonlinear) mean choice function. The d-coefficient is therefore exactly zero in expectation under Tallying, for ALL of its parameter values, with only binomial sampling noise around zero.\n\nWHY IT IS ROBUSTLY POSITIVE UNDER NVW (for every admissible parameter): NVW's P(A) is monotone increasing in the validity-weighted evidence difference (v^gamma preserves validity order for gamma in [0.7,1.3], so the sign and rank of d computed with raw v is preserved), and the design provides large within-margin spread of d (margin 0: |d| up to 0.55; margin 1: 0.20 to 1.30; margin 2: 1.05 to 1.90). The within-margin d-slope of the logistic is approximately beta_eff * p(1-p) with beta_eff = beta*delta in [0.9, 3.3], so even at the worst-case corner (beta=5, delta=0.18) the slope is >= ~0.19, and at mid parameters (~2.1) it is ~0.4-0.5. The one-reason lapse component (top-cue direction) is itself aligned with sign(d) on the high-|d| trials and only adds to the slope. The lapse epsilon (<=0.06) shrinks the slope by at most 6%.\n\nPOWER AT N=25: The pooled within-margin residual variance of d is ~0.70 and each subject contributes 96 trials, so the per-subject standard error of the d-coefficient is about sqrt(0.25/(96*0.70)) ~ 0.06 (worst case, Tallying with low beta; NVW's is smaller because its p(1-p) is smaller at extreme d). Welch's t for a mean gap of even 0.19 (absolute worst-case NVW corner) vs 0 (Tallying) is then roughly 0.19 / sqrt((0.0037+0.0022)/25) ~ 12-14, far below alpha=0.01; at typical parameters the gap (~0.4-0.5) gives t > 25. The metric works identically on the pooled DataFrame and on a single subject's 96-trial slice (the regression is well-conditioned in both cases), so the between-subject variance is well defined. The contrast is structural (Tallying literally has no within-margin sensitivity; NVW does), not a parameter tuning, and cannot be trivially satisfied by the advocated theory's noise terms."
}
```

## Usage

```json
{
  "input_tokens": 5532,
  "output_tokens": 16087,
  "reasoning_tokens": 14703,
  "provider": "Sail Research"
}
```
