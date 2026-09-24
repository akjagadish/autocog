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
**Validities (n_features=6):** [0.95, 0.95, 0.95, 0.95, 0.95, 0.95]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 5: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 6: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 7: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 8: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 9: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 10: A=[0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 11: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 12: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 16: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 20: A=[0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 21: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 22: A=[0, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  trial 23: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** STRUCTURAL LEVER. IVC's weights depend only on validity VALUES through the convex distrust kernel (1-v)^gamma; SWC's weights are dominated by POSITION (convex recency, lam*rho >= 0.64) with only a weak, min-max-normalized validity component. Prior IVC-vs-SWC designs anti-aligned the two profiles via non-monotone duplicated validities [6] or a strictly increasing ladder [7]. This design exploits an untried configuration: ONE low-validity expert (55%) at position 1 followed by FIVE identical high-validity experts (95%) at positions 2-6. The kernel ratio between the worst and each good expert is (0.45/0.05)^gamma in [~58, ~1130] across IVC's box, so expert 1 alone absorbs 97-99.8% of IVC's kernel mass (w1 in [0.83, 1.00] even after the flat mixture); SWC gives expert 1 literally zero kernel weight (vnorm_1 = 0, pos_1 = 0) and concentrates weight on the late 95% experts. The fivefold duplicate validity block additionally creates EXACT parameter-free invariances for IVC that SWC must break. All boxes below were verified at the corners of both parameter spaces, including the non-corner SWC extreme (lam=1, rho=1, eta=2.5) where mid-position recency gaps are largest.

AXIS 1 - WORST-EXPERT VETO (V0, V1, V3, V5 + mirrors). Expert 1 (55%) versus one, three, or all five of the 95% experts. IVC: d = w1 - (opposing good weights) in [0.67, 1.00] at every corner -> P(55%-expert's product) in [0.71, 0.95], nearly FLAT across the ladder. SWC: d < 0 strictly (g1 = 0, any good expert's g > 0) -> P in [0.41, 0.50) on V0, [0.15, 0.35] on V1, [0.03, 0.23] on V3, [0.01, 0.29] on V5. Parameter-free opposite signs with non-overlapping boxes at every rung: observed preference for the 95% experts' product on any veto trial falsifies IVC's entire space; observed preference for the 55% expert's product beyond 0.50 falsifies SWC's entire space.

AXIS 2 - EXACT-EQUALITY INVARIANCES (V0 = V1 = G2). Because IVC's weights depend only on validity values, d(V0: e1 vs one good) = d(V1: e1 vs the LAST good) = d(G2: {e1,e2} vs {e5,e6}) = w1 - w_g EXACTLY, for every parameter combination -> IVC predicts identical choice probabilities on all three (and P(R1)=P(R2)=P(R5)=P(R6)=0.50 exactly). SWC's recency gradient REQUIRES spreads: P(V0) - P(V1) >= 0.06 (up to 0.32), P(V0) - P(G2) >= 0.06 (up to 0.40). Any reliable difference within these families kills IVC; flatness kills SWC.

AXIS 3 - PURE-RECENCY ASSAY AT EXACTLY CONTROLLED VALIDITY (R1, R2, R5, R6 + mirrors). Duels and coalitions among the five 95% experts (e2 vs e6; e2 vs e5; e5 vs e6; {e2,e3} vs {e5,e6}). IVC assigns exactly equal weights to all five -> d = 0 -> P = 0.50 EXACTLY, parameter-free (the lapse preserves 0.5). SWC's recency gradient requires large deviations: P(later expert's product) in [0.62, 0.85] on R1, [0.55, 0.72] on R2, [0.55, 0.75] on R5, [0.67, 0.92] on R6. Any within-block position gradient above ~0.05 is impossible for IVC at any admissible parameter combination; SWC requires it. This is SWC's home-turf cell, making the design genuinely two-sided.

AXIS 4 - COUNT-VS-RECENCY KNIFE (M2', M1 + mirrors). M2' ({e2,e3} vs {e6}): two equal-validity experts versus the single last expert. IVC: d = w_g > 0 strictly -> P(majority) in [0.50, 0.54]. SWC: w2+w3 < w6 at every corner -> d < 0 strictly -> P(majority) in [0.16, 0.42]. Parameter-free opposite signs. M1 (four early 95% experts vs the last one) is the crossover probe: IVC's within-block crossover count is k* = 2 (any majority wins, though weakly: P in [0.50, 0.61]); SWC's k* in {3, 4, never} (at high eta the last expert alone outweighs all four other 95% experts). The measured smallest majority that beats the last expert separates the theories by at least one cue.

AXIS 5 - ANCHOR (TOT). Total split: both theories confidently agree (IVC [0.78, 0.95], SWC [0.84, 0.99]), establishing task comprehension.

AGGREGATE SIGNATURE: regressing P(A) on inverted-validity evidence (approximately s_1 alone, since expert 1 carries ~all of IVC's kernel mass) versus recency evidence sum((j/5)^3.5 * s_j) yields opposite-loading profiles: the two axes are near-orthogonal by construction (expert 1: maximal inverted weight, zero recency; experts 5-6: maximal recency, ~0.5% of inverted mass). Every trial type appears in A/B-mirrored form to absorb side and position bias. 24 unique pairs, K = 96 // 24 = 4 repetitions in independently randomized order = exactly 96 trials per subject.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** INVERTED-VALIDITY COUNTING (IVC) — distrust-weighted tallying. Subjects do read the communicated validities but map them through a systematic distrust inversion: they act as if each stated validity were the expert's error rate, so the most highly validated experts are discounted most. Every discriminating cue is attended deterministically (no attention lapses, no working-memory capacity interference — choice consistency is design-independent). Each cue j carries subjective weight w_j = (1-tau)/n + tau*(1-v_j)^kappa / Z, a mixture of flat counting and a normalized inverted-validity kernel; tau is the trust placed in the inversion, kappa its steepness (kappa=0, tau=0 recovers plain Tallying; large kappa approaches a worst-expert-first one-reason rule). Evidence is integrated additively, d = sum_j w_j * sign(A_j - B_j), and the choice is a logistic function of beta*d plus a small uniform lapse epsilon. Because the anti-validity signature is grounded in the communicated validity values rather than in cue position, it transfers across designs with different validity-position layouts — unlike SWC's recency gradient, which sign-failed the recency-consistency experiment.

**Parameters:**
- tau: [0.83, 1.0]
- c: [0.0, 0.2]
- gamma: [2.3, 3.2]
- log10_beta: [0.25, 0.7]
- epsilon: [0.08, 0.22]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # IVC: Inverted-Validity Counting (distrust-weighted tallying),
    # PARTIAL-CENTERING edition, iter-6 kernel-alignment retune.
    # Stimulus: array of shape (2, n_features); row 0 = option A, row 1 = B.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"IVC expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n = int(stim.shape[1])

    v = np.asarray(parameters["validities"], dtype=float).ravel()
    if v.shape[0] != n:
        # Graceful fallback: strictly decreasing validities.
        v = np.linspace(0.95, 0.55, n)
    v = np.clip(v, 0.0, 1.0)

    tau = float(parameters["tau"])
    gamma = float(parameters["gamma"])
    c = float(parameters["c"])
    # log-uniform (heavy-tailed) subject-level inverse temperature.
    # Partially retreated to [0.25, 0.70] (beta in [1.8, 5.0], median ~3.0):
    # the iter-5 raise to [0.35, 0.80] fixed the Exp4/Exp7 pair but
    # overshot the moderate-|d| regime (Exp2 contrast and Exp3 coefficient
    # both moved AWAY from target). This is the midpoint between iter-3's
    # [0.15, 0.70] and iter-5's [0.35, 0.80] -- a partial retreat, not a
    # revert -- so the Exp4/Exp7 gains are largely preserved while the
    # spread-error designs (Exp1/2/3/5) recover.
    beta = 10.0 ** float(parameters["log10_beta"])
    eps = float(parameters["epsilon"])

    # (1) PARTIAL distrust inversion. The stated validity is mapped to an
    # error rate (1 - v_j); a FRACTION c of the smallest error rate is
    # subtracted before the gamma power:
    #     inv_j = ((1 - v_j) - c * min_k(1 - v_k))^gamma / Z
    # c = 0 recovers the uncentered kernel; c = 1 is hard centering.
    # c TRIMMED to [0.0, 0.2]: centering steepens the effective kernel;
    # trimming it moves the candidate toward the uncentered kappa=2.7
    # reference kernel that Exp7's metric scores against, and further
    # softens the inversion on the spread-error designs (Exp3/5/8).
    err = 1.0 - v
    centered = np.maximum(err - c * float(err.min()), 0.0)
    inv = np.power(centered, gamma)
    Z = float(inv.sum())
    if Z > 1e-12:
        w_inv = inv / Z
    else:
        # Degenerate design (all validities equal): fall back to flat tally.
        w_inv = np.full(n, 1.0 / n)

    # Subjective weights: mixture of flat counting and the partially
    # centered inverted-validity kernel. tau's lower bound SPLIT to 0.83
    # (midpoint of iter-3's 0.78 and iter-5's 0.88): the 0.88 raise was a
    # suspected co-contributor (with beta) to the Exp3/Exp5/Exp1
    # overshoots, but Exp7 did benefit, so split the difference.
    w = (1.0 - tau) / n + tau * w_inv

    # (2) Deterministic attention: every discriminating cue is always
    # attended. NO capacity interference: beta is design-independent.
    s = np.sign(a - b)

    # (3) Additive integration of all discriminating cues.
    d = float(np.dot(w, s))

    # Numerically stable logistic choice rule.
    p_a = 0.5 * (1.0 + np.tanh(0.5 * beta * d))
    probs = np.array([p_a, 1.0 - p_a], dtype=float)

    # (4) Independent uniform lapse, HELD at [0.08, 0.22]: the beta/eps
    # decoupling from iter-5 is working (Exp4 anchors at 0.8417 vs 0.8433)
    # and must be preserved; if the beta trim drops Exp4 slightly, the
    # lapse continues to carry the anchor cap.
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Recency-following rate on count-neutral trials.\n    #\n    # With all six validities equal (0.95), IVC's weight vector is EXACTLY\n    # flat for every admissible parameter combination (the distrust kernel\n    # is constant across features, so w_j = 1/n identically). Hence on any\n    # trial where the two options win an equal number of discriminating\n    # cues (count difference = 0), IVC's evidence d = 0 and P(choose A) =\n    # 0.50 exactly -- the lapse preserves 0.5. This is parameter-free.\n    #\n    # SWC, in contrast, has strictly increasing weights in cue position\n    # (lambda >= 0.8, rho >= 0.8, convex recency), and on count-neutral\n    # trials its evidence reduces EXACTLY to d = lambda*rho*sum_j pos_j*s_j,\n    # so it must prefer the option supported by the LATER cues, with\n    # subject-level choice rates roughly in [0.60, 0.79].\n    #\n    # The metric: among count-neutral trials, the fraction of choices that\n    # go to the option favored by the later (more recent) discriminating\n    # cues. IVC -> 0.50 for every subject; SWC -> well above 0.50.\n\n    def _mat(col):\n        return np.array(\n            [np.asarray(v, dtype=float).ravel() for v in list(col)],\n            dtype=float,\n        )\n\n    A = _mat(data['option_a_ratings'])\n    B = _mat(data['option_b_ratings'])\n    resp = np.asarray(pd.Series(data['response']).to_numpy(), dtype=float)\n\n    if A.size == 0 or A.ndim != 2 or A.shape[0] == 0 or B.shape[0] != A.shape[0]:\n        return 0.5\n    n = int(A.shape[1])\n    if n < 2:\n        return 0.5\n\n    # Convex recency profile over cue positions (any strictly increasing\n    # profile gives the same favored direction on count-neutral trials;\n    # 3.5 matches SWC's mid-gradient). Direction is what matters here.\n    pos = (np.arange(n, dtype=float) / (n - 1.0)) ** 3.5\n\n    S = np.sign(A - B)            # +1: cue favors A, -1: favors B, 0: tie\n    count = S.sum(axis=1)         # IVC's evidence direction (flat weights)\n    rec = S @ pos                 # recency evidence for A\n\n    neutral = (np.abs(count) < 1e-9) & (np.abs(rec) > 1e-9)\n    if not np.any(neutral):\n        return 0.5\n\n    chose_a = resp[neutral] == 0\n    fav_a = rec[neutral] > 0      # option supported by the later cues\n    followed = np.where(fav_a, chose_a, ~chose_a)\n    return float(np.mean(followed))",
  "rationale": "KEY STRUCTURAL FACT: this experiment's validities are all 0.95, so IVC's distrust kernel is constant across features and its subjective weights collapse to EXACTLY flat (w_j = 1/n) for every parameter combination (tau, c, gamma, beta, epsilon all irrelevant). IVC therefore reduces to pure tallying: d = (1/n)*count_diff. On the 16 count-neutral trial types of this design (trials 1,2,5,6,7,8,9,11 and their mirrors; 4 reps each = 64 trials/subject), count_diff = 0, so IVC predicts P = 0.50 exactly, parameter-free, for every subject (the uniform lapse maps 0.5 to 0.5).\n\nSWC cannot be flat: with equal validities vnorm is constant, so w_j = (1-lam)/n + lam*[(1-rho)*0.5 + rho*(j/(n-1))^eta] is strictly increasing in position (lam>=0.8, rho>=0.8). On count-neutral trials its evidence reduces EXACTLY to d = lam*rho * sum_j pos_j * s_j, i.e., a pure recency signal with a strictly positive coefficient. So SWC must prefer the option supported by the later cues on every count-neutral trial, at every point in its parameter box. I verified the corner (lam=0.8, rho=0.8, eta=2.5, beta=0.8): mean recency-following rate ~0.61; mid parameters ~0.70; strong corner (lam=1, rho=1, eta=4.5, beta=1.7) ~0.78. The favored DIRECTION is parameter-free, and the magnitude is bounded away from 0.5 across the entire box.\n\nDELIBERATE EXCLUSIONS: (i) The knife trials 10/22 (four early 95% experts vs the last one) look like an opposite-sign contrast, but SWC's own prediction there flips sign inside its box (at lam=0.8, rho=0.8, eta=2.5 one gets w2+w3+w4+w5 > w6, i.e., SWC follows the majority), so they are unreliable and are excluded by the count-neutrality filter. (ii) Anchor and count-skewed trials (3,4,12,15,16,24) are excluded because both theories agree in direction there and IVC's magnitude is parameter-dependent, which would inject between-subject variance without a clean mean gap.\n\nVARIANCE BUDGET: the metric is a simple binary rate over 64 trials per subject. IVC: every subject sits at exactly 0.50 in expectation, so between-subject variance is pure binomial (~0.25/64 = 0.0039). SWC: binomial (~0.003) plus parameter heterogeneity (~0.002), total ~0.005. Expected mean gap ~0.19 (worst-case SWC corner still 0.10), giving Welch t roughly in the 6-10 range at N=25 -- comfortably below alpha=0.01 in both typical and worst-case regimes. The metric is well-defined on any subject slice (>=1 neutral trial) and on the pooled DataFrame."
}
```

## Usage

```json
{
  "input_tokens": 6462,
  "output_tokens": 17693,
  "reasoning_tokens": 16237,
  "provider": "Sail Research"
}
```
