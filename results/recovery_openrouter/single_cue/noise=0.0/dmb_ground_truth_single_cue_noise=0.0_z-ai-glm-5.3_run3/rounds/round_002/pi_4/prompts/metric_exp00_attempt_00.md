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
**Validities (n_features=7):** [0.6, 0.96, 0.6, 0.6, 0.6, 0.6, 0.9]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  trial 2: A=[0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  trial 4: A=[0, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  trial 6: A=[0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 9: A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0]
  trial 10: A=[0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 0]
  trial 12: A=[0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  trial 13: A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0, 0]
  trial 14: A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  trial 15: A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0]
  trial 16: A=[0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  trial 17: A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0, 0]
  trial 18: A=[0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  trial 19: A=[0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 1, 0]
  trial 20: A=[0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1]
  trial 21: A=[0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]
  trial 22: A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]
  trial 23: A=[0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0]
  trial 24: A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]

**Rationale:** GOAL: dissociate SWC (convex cue-order recency, deterministic attention, no capacity limit) from NVW (validity^gamma weighting, capacity interference, span ~4). Prior design [4] used strictly decreasing validities, making recency and validity ANTI-correlated everywhere. This design takes the opposite, novel route: a NON-MONOTONE validity profile [0.60, 0.96, 0.60, 0.60, 0.60, 0.60, 0.90] with two high-validity peaks at positions 2 and 7 and five equal low-validity cues. This decorrelates validity from recency (rather than anti-correlating it), so each axis can be isolated in pure form, and it lets the CONVEXITY of SWC's recency gradient (eta in [2.5,4.5]: the last cue's weight is ~2x the second-to-last's and ~10x the third-to-last's) fight NVW's validity MASS head-to-head. n_features = 7 also puts three cues beyond NVW's span-4, engaging its capacity interference (beta_eff = beta*delta^3 <= 0.30), while SWC's beta is design-independent. All boxes below were verified at the corners of BOTH parameter spaces.

AXIS 1 - EQUAL-VALIDITY RECENCY LADDER (T4/T5/T6 + mirrors): A wins expert 1 (60%), B wins exactly one equally-valid (60%) expert at position 4, 5, or 6. NVW's evidence is EXACTLY zero for every parameter value (equal validities => equal weights v^gamma, symmetric attention/lapses), so NVW predicts exactly 50/50 on all six trials - parameter-free. SWC's recency gradient predicts a monotone ladder below 0.5 for the early-cue option: P(A) boxes [0.42,0.49] (vs p4), [0.30,0.43] (vs p5), [0.17,0.31] (vs p6). Observed P(later expert's product) > 0.60 on T6 is impossible for NVW at ANY admissible parameter combination, and required by SWC. This is the single cleanest trial family: pure recency with validity perfectly controlled.

AXIS 2 - HEAD-TO-HEAD DUELS (D1/D2 + mirrors): D1: A wins expert 2 (96%), B wins expert 7 (90%). NVW follows validity: P(A) in [0.50, 0.53]. SWC's convex recency makes the LAST cue dominate: P(A) in [0.14, 0.38]. Opposite signs, non-overlapping boxes - the subject prefers the product endorsed by the slightly LESS accurate but LATER expert iff SWC is true. D2 sharpens it: 96% expert 2 vs 60% expert 6 - a huge validity gap that NVW respects (P(A) in [0.50,0.54]) but SWC's recency overrides (P(A) in [0.31,0.46]). These flips survive even if NVW's capacity interference were removed (delta=1): NVW would just become more confident in the validity winner; the sign disagreement is structural.

AXIS 3 - CONVEXITY vs VALIDITY MASS (C1, L1-L4, E2 + mirrors): C1: A wins expert 7 (90%) alone; B wins experts 2+4 (96%+60%). NVW: validity mass favors B, P(A) in [0.43,0.50]; SWC: the last cue's convex weight beats both, P(A) in [0.59,0.80]. The L-ladder measures the crossover count: A wins expert 7 vs B wins k of the mid cues {3,4,5,6}. NVW's validity arithmetic says the 90% cue beats exactly ONE 60% cue (d=+0.23 at k=1, d=-0.47 at k=2), so NVW crosses 0.5 between L1 and L2. SWC's convex gradient says the last cue outweighs three mid cues (recency sum at k=3 is 0.40-0.60 < 1.0), crossing only at k=4 - a measured crossover-count dissociation of 2-3 opposing cues. E2 is the block-level flip: A wins the 3 late cues (incl. only the 90%), B wins 4 early cues (incl. the 96%): NVW says B (P(A) in [0.42,0.50]), SWC says A (P(A) in [0.71,0.94]).

AXIS 4 - CAPACITY CLIFF / CONFIDENCE (E1 + mirror): A wins experts 4-7, B wins 1-3. Both theories favor A in direction, but NVW's beta_eff <= 0.30 caps its P(A) at ~[0.50,0.56] on EVERY trial in this design (max |beta_eff*d| ~ 0.55), while SWC predicts [0.75,0.95]. Observed accuracy > 0.70 on E1 falsifies NVW's entire parameter space; near-chance performance everywhere with a residual validity-ordering falsifies SWC.

Aggregate signature: regressing P(A) on recency-weighted evidence sum((j/6)^3.5 * s_j) versus validity-weighted evidence sum(v^gamma * s_j) - decorrelated here by construction (expert 2: high validity, negligible recency; expert 6: low validity, high recency) - yields opposite-loading profiles under the two theories. Every trial type appears in A/B-mirrored form to absorb side/position bias. 24 unique pairs, K = 96//24 = 4 repetitions in randomized order = exactly 96 trials per subject.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Subjective-Weighted Counting with convex cue-order recency (SWC), deterministic-attention variant. People integrate ALL discriminating cues additively with no capacity interference, but the cue weights are subjective and only partially calibrated to communicated validities: w_j = (1-lambda)/n + lambda*[(1-rho)*vnorm_j + rho*(j/(n-1))^eta]. Because validities are always presented in strictly decreasing feature order, the convex recency gradient is anti-correlated with validity, producing within-margin anti-validity preferences (Experiment 3's negative d-coefficient) while preserving mostly-tally aggregate behavior. Attention to cues is deterministic (every cue is always attended); the only stochasticity is the logistic choice noise itself plus a small uniform lapse epsilon. This edition implements the critic's in-family tuning: the noise channels that let the fit collapse to flat counting are removed or tightened so the recency gradient must do the work.

**Parameters:**
- beta: [0.8, 1.7]
- epsilon: [0.0, 0.03]
- lambda: [0.8, 1.0]
- rho: [0.8, 1.0]
- eta: [2.5, 4.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # SWC: Subjective-Weighted Counting with convex cue-order recency
    # (deterministic-attention variant).
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SWC expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: linearly decreasing validities.
        v = np.linspace(0.95, 0.55, n)

    lam = float(parameters["lambda"])
    rho = float(parameters["rho"])
    eta = float(parameters["eta"])
    beta = float(parameters["beta"])
    eps = float(parameters["epsilon"])

    # (1) Subjective validity component: min-max normalized validities.
    vspan = float(v.max() - v.min())
    if vspan > 1e-12:
        vnorm = (v - v.min()) / vspan
    else:
        vnorm = np.full(n, 0.5)

    # (2) Recency component: position in presentation order, convexly
    # transformed so that late cues can be strongly overweighted.
    if n > 1:
        pos = (np.arange(n, dtype=float) / (n - 1.0)) ** eta
    else:
        pos = np.full(n, 0.5)

    # Subjective weights: mixture of flat counting, validity-following,
    # and recency-following. Because validities are presented in strictly
    # decreasing feature order, the recency part is anti-correlated with
    # validity -- the source of within-margin anti-validity preferences.
    g = (1.0 - rho) * vnorm + rho * pos
    w = (1.0 - lam) / n + lam * g

    # (3) Deterministic attention: every cue is attended on every trial
    # (per-cue attention lapse removed -- real subjects are highly
    # consistent, and stochastic lapses were the dominant noise source
    # inflating between-subject variance).
    s = np.sign(a - b)

    # Additive integration of all discriminating cues.
    # NO capacity interference: beta is design-independent.
    d = float(np.sum(w * s))

    # Numerically stable logistic.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (4) Independent uniform lapse.
    probs = (1.0 - eps) * probs + eps * 0.5
    return probs
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # guard against float drift
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Recency-consistency rate: fraction of choices aligned with a convex\n    # cue-order recency gradient, restricted to trials where that gradient\n    # is strong and its direction is unambiguous across SWC's eta range.\n    #\n    # SWC (advocated): late cues dominate via convex recency -> rate well\n    #   above 0.5 (predicted ~0.65-0.85 across its parameter box).\n    # NVW (competing): validity-weighted integration with capacity\n    #   interference (beta_eff = beta*delta^3 <= ~0.30 at n=7) -> choices\n    #   near chance on nearly every trial, and anti-recency or neutral on\n    #   the conflict trials -> rate ~0.50.\n    df = data.reset_index(drop=True)\n    if len(df) == 0:\n        return 0.5\n\n    a_arrs = [np.asarray(list(x), dtype=float) for x in df['option_a_ratings']]\n    b_arrs = [np.asarray(list(x), dtype=float) for x in df['option_b_ratings']]\n    resp = pd.to_numeric(df['response'], errors='coerce').to_numpy()\n\n    if len(a_arrs) == 0 or a_arrs[0].ndim != 1:\n        return 0.5\n    n = int(a_arrs[0].shape[0])\n    if n < 2:\n        return 0.5\n\n    pos = np.arange(n, dtype=float) / (n - 1.0)\n    w_lo = pos ** 2.5   # SWC's eta lower corner\n    w_mid = pos ** 3.5  # SWC's eta midpoint\n    w_hi = pos ** 4.5   # SWC's eta upper corner\n\n    consistent = []\n    for a, b, r in zip(a_arrs, b_arrs, resp):\n        if a.shape[0] != n or b.shape[0] != n:\n            continue\n        if not np.isfinite(r):\n            continue\n        s = np.sign(a - b)\n        if not np.any(s != 0):\n            continue\n        e_lo = float(np.dot(w_lo, s))\n        e_mid = float(np.dot(w_mid, s))\n        e_hi = float(np.dot(w_hi, s))\n        # Require the recency gradient's direction to be invariant across\n        # SWC's whole eta range (drops ambiguous trials, e.g. 1 late cue vs\n        # 4 mid cues, where the sign flips with eta).\n        if e_lo == 0.0 or e_hi == 0.0 or np.sign(e_lo) != np.sign(e_hi):\n            continue\n        # Require non-trivial gradient magnitude (drops near-degenerate\n        # trials where even SWC predicts ~0.52).\n        if abs(e_mid) < 0.25:\n            continue\n        fav_a = e_mid > 0.0\n        chose_a = (int(r) == 0)  # response: 0 = chose A, 1 = chose B\n        consistent.append(1.0 if (chose_a == fav_a) else 0.0)\n\n    if len(consistent) == 0:\n        return 0.5\n    return float(np.mean(consistent))",
  "rationale": "The metric is the fraction of choices consistent with a convex cue-order recency gradient (position weights (j/(n-1))^eta), computed only on trials where (a) the gradient's preferred option is invariant across SWC's entire eta range [2.5, 4.5] and (b) its mid-range evidence magnitude is at least 0.25. In this design that admits 20 of the 24 trial types (the equal-validity recency ladder T3-T6, the validity-vs-recency duels T7-T12, the late-cue-mass trials T13-T18, and the block-level flips T21-T24), while excluding only the degenerate cue-1-vs-cue-4 trials and the eta-ambiguous 1-late-vs-4-mid trials. Mirroring in the design makes the recency-favored option A on exactly half the included trials, so side bias cancels. Predicted separation: under SWC, the per-trial probability of choosing the late-cue-endorsed product ranges from ~0.55 (worst-case corner beta=0.8, eta=2.5) to ~0.9, averaging ~0.70-0.75, so the metric lands ~0.65-0.85. Under NVW, capacity interference forces beta_eff = beta*delta^3 <= ~0.30 at n_features=7, so its logistic outputs stay within ~0.5 +/- 0.07 of chance on every trial in this design, and on the conflict trials its residual validity-ordering is anti-recency; its expected rate is ~0.49-0.51 with pure binomial noise. The gap (~0.2+ in means) is large relative to per-subject binomial noise over ~80 scored trials (sd ~0.05), so with N=25 the Welch t-test separates the theories by many sigma. The metric works identically on a single subject's slice (same trial filtering), so between-subject variance is well-defined."
}
```

## Usage

```json
{
  "input_tokens": 6295,
  "output_tokens": 12822,
  "reasoning_tokens": 11619,
  "provider": "Sail Research"
}
```
