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
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.62, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 3: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 5: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 6: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 7: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 8: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 9: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 10: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 11: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 12: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 13: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 14: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 15: A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  trial 16: A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  trial 17: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 18: A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 19: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 20: A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 21: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  trial 22: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 23: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** STRUCTURAL LEVER. With validities fixed per feature, both theories reduce to a fixed position-indexed weight profile applied to the cue-allegiance pattern s_j = sign(A_j - B_j): NVW uses w_j ∝ v_j^gamma (validity-VALUE based), while SWC uses w_j = (1-lambda)/n + lambda[(1-rho)vnorm_j + rho*(j/(n-1))^eta] (position-RECENCY based, with recency necessarily dominating: lam*rho >= 0.64 versus a validity coefficient <= 0.20 and flat <= 0.20 across SWC's entire box). I therefore present validities in STRICTLY DECREASING order [0.95, 0.90, 0.85, 0.62, 0.56, 0.52] — SWC's own assumed presentation format, i.e., its home turf — with a sharp validity gap between positions 3 and 4 creating a 'high-validity early block' {0.95, 0.90, 0.85} and a 'low-validity late block' {0.62, 0.56, 0.52}. This makes the two weight profiles strongly ANTI-CORRELATED: NVW's weights fall with position, SWC's rise (convexly) with position. n_features = 6 also places two cues beyond NVW's span-4, engaging its capacity-interference signature (beta_eff = beta*delta^2 in [0.16, 0.99], i.e., NVW itself predicts compressed confidence), while SWC's beta is design-independent.

DISSOCIATION AXES (all predicted boxes verified at the corners of BOTH parameter spaces; NVW's d-sign is invariant over gamma in [0.7, 1.3] because v^gamma is monotone in v, and SWC's recency dominance holds over its whole box):

(1) SIGN-FLIP FAMILY (S1, S2, S3, E1, E3, T3, T6, T8 + mirrors; 16 of 24 trials). The early high-validity block favors one option and the late low-validity block the other. NVW predicts the early-block (validity) winner; SWC predicts the late-block (recency) winner — PARAMETER-FREE OPPOSITE SIGNS. Headline boxes for P(A): S1 (A wins only expert 6 vs expert 1): NVW 0.39-0.51, SWC 0.60-0.85. S2 (A wins experts 5-6 vs 1-2): NVW 0.28-0.48, SWC 0.64-0.91. S3 (A wins experts 4-6 vs 1-3): NVW 0.25-0.47, SWC 0.63-0.91. E1 — the most damning trial for SWC: B wins the FOUR MOST VALID experts and A only the two least valid, yet SWC still predicts A (P(A) in [0.55, 0.88]) because recency weight on the last two positions outweighs everything; NVW predicts B (P(A) in [0.10, 0.41]). E3: A wins the two top experts vs B's single bottom expert — SWC predicts B (P(A) in [0.18, 0.41]) because one late cue outweighs two top cues; NVW predicts A (0.55-0.80). T6/T8 extend the flip to mixed patterns (boxes: NVW 0.52-0.71 vs SWC 0.09-0.37; NVW 0.44-0.50 vs SWC 0.62-0.89).

(2) VALIDITY-TIE / RECENCY-ONLY (T1, T1m). A wins experts 1 and 6, B wins experts 3 and 4: the validity-weighted evidence is exactly balanced (|d| <= 0.04 for all gamma), so NVW predicts chance (P(A) in [0.50, 0.55], the tiny upward tilt being its one-reason lapse toward the top cue), while SWC's recency gradient (expert 6 vs experts 3-4) predicts P(A) in [0.58, 0.82]. Non-overlapping boxes: observed P(A) <= 0.55 falsifies SWC's entire parameter space.

(3) FIVE-VS-ONE (E2, E2m). The five most valid experts versus the single least valid expert: NVW predicts the validity winner with P in [0.62, 0.97]; SWC's flat + vnorm components nearly cancel its recency component, so its box STRADDLES chance (P(validity winner) in [0.42, 0.58]). Observed P > 0.6 for the five-expert side is impossible for SWC at any admissible parameter combination.

(4) CUE-MIGRATION PAIR (S3 vs S4). Moving expert 3 (v = 0.85) from the early side to the late side flips NVW's sign (d: -1.00 to +0.70; predicted P(A) crosses 0.5 between adjacent trial types) while leaving SWC's prediction on the SAME side (d approx 1.0 to 1.3, both strongly pro-late-block). A sign-crossing between S3 and S4 is a qualitative signature only NVW produces.

(5) AGREEMENT ANCHORS (A1/A1m total splits; S4/S4m). Both theories agree; they establish that subjects can perform the task and calibrate overall consistency. Note the confidence dissociation: because of capacity interference, NVW's maximum predicted probability on any non-total trial is approx 0.8-0.9, whereas SWC predicts up to 0.93-0.97 on recency-unbalanced trials — the joint profile 'validity-ordered choices at moderate consistency' is uniquely NVW; 'recency-ordered choices at high consistency' is uniquely SWC.

Every conflict type appears in A/B-mirrored form to absorb side and position bias. With 24 unique pairs, K = 96 // 24 = 4 repetitions in randomized order = exactly 96 trials. Summary of decisive, parameter-free signatures: (i) on 16 sign-flip trials the theories predict opposite choices with non-overlapping probability boxes; (ii) on T1 NVW predicts exactly chance where SWC's box excludes chance; (iii) on E2 SWC's box is confined to near-chance where NVW predicts high confidence; (iv) the S3->S4 sign crossing is NVW-only; (v) regressing P(A) on validity-weighted evidence (sum v_j*s_j) versus recency-weighted evidence (sum (j/5)^3*s_j) — anti-correlated by construction — identifies which profile subjects use, with opposite regression signs under the two theories.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Validity-adherence rate on VALIDITY-vs-RECENCY conflict trials.\n    #\n    # For every trial we compute two decision variables from the presented\n    # cue patterns:\n    #   d_v = sum_j v_j * sign(A_j - B_j)   (validity-weighted evidence for A)\n    #   d_r = sum_j (j/(n-1))^3 * sign(A_j - B_j)  (convex cue-order recency\n    #                                              evidence for A)\n    # A trial is a CONFLICT trial when the two point in opposite directions\n    # and both are non-trivial in magnitude. On those trials the metric is\n    # the proportion of choices that follow the VALIDITY-implied option.\n    #\n    # NVW (validity-weighted integration) predicts adherence clearly above\n    # chance (~0.6); SWC (recency-dominated subjective weights) predicts\n    # adherence well below chance (~0.3), because on these trials its\n    # recency gradient favors the opposite option.\n\n    validities = np.array([0.95, 0.90, 0.85, 0.62, 0.50, 0.50], dtype=float)\n    n = int(validities.shape[0])\n    # Convex recency profile over presentation positions (eta ~ 3).\n    recency = (np.arange(n, dtype=float) / float(n - 1)) ** 3.0\n\n    a_col = data[\"option_a_ratings\"]\n    b_col = data[\"option_b_ratings\"]\n    r_col = data[\"response\"]\n\n    hits = []\n    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):\n        a = np.asarray(a_raw, dtype=float).ravel()\n        b = np.asarray(b_raw, dtype=float).ravel()\n        if a.shape[0] != n or b.shape[0] != n:\n            continue\n        s = np.sign(a - b)\n        if not np.any(s != 0):\n            continue\n\n        d_v = float(np.dot(validities, s))\n        d_r = float(np.dot(recency, s))\n\n        # Need a real conflict with non-trivial evidence on both axes.\n        if abs(d_v) < 0.30 or abs(d_r) < 0.50:\n            continue\n        if np.sign(d_v) == np.sign(d_r):\n            continue\n\n        validity_winner_is_a = (d_v > 0.0)\n        chose_a = (int(resp) == 0)\n        hits.append(1.0 if (chose_a == validity_winner_is_a) else 0.0)\n\n    if len(hits) == 0:\n        return 0.0\n    return float(np.mean(hits))",
  "rationale": "CORE CONTRAST. Because validities are presented in strictly decreasing order, the validity-weighted evidence (d_v = sum v_j*s_j) and the convex cue-order recency evidence (d_r = sum (j/5)^3*s_j) are anti-correlated by construction. The 24-trial design contains 18 'conflict' trials where the two decision variables have opposite signs and both are non-trivial (|d_v| >= 0.3, |d_r| >= 0.5): e.g., trial 13 (A wins experts 1-2, B wins expert 6: d_v = +1.35, d_r = -0.99), trial 19 (A wins experts 1,3 vs B's 5,6: d_v = +0.80, d_r = -1.45), trial 3/5/9/11 and all their A/B mirrors. On exactly these trials the two theories make opposite parameter-free predictions: NVW's weights follow communicated validity (its integration term sigmoid(beta_eff*d_v) and its one-reason term, keyed to the top cue, both favor the validity-implied option), while SWC's recency gradient (w_6 >= 0.64 vs w_1 <= 0.19 across its entire box) favors the late-block option, i.e., the anti-validity side.\n\nMETRIC. The metric is simply the proportion of choices that agree with the validity-implied option on those conflict trials (72 observations per subject: 18 trial types x 4 repetitions). It is a plain behavioral summary \u2014 'do subjects follow the communicated validities when validity and recency disagree?' \u2014 not a re-implementation of either model.\n\nPREDICTED VALUES. NVW: per-trial adherence sigmoid(beta_eff*|d_v|) with beta_eff in [0.16, 0.99] gives per-trial values 0.54-0.80, aggregate ~0.60-0.65 (worst-case corner ~0.54). SWC: per-trial adherence 1 - sigmoid(beta*|d_swc|) with d_swc recency-dominated gives per-trial values ~0.10-0.45 on the 2-vs-1 and 3-vs-2 cue trials (aggregate ~0.28-0.35; even its most validity-friendly corner stays ~0.42). The gap in means is ~0.3.\n\nWHY IT SURVIVES THE N=25 TEST. Each subject contributes 72 binary conflict observations, so pure binomial per-subject SD is ~0.055-0.06; adding parameter heterogeneity across simulated subjects keeps per-subject SD ~0.08 (variance ~0.006). With a mean gap of ~0.3, Welch's t at N=25 is roughly 0.3 / sqrt(2*0.006/25) \u2248 9-14, far below the alpha=0.01 threshold, and even the joint worst-case corners of both parameter boxes (noisiest NVW vs most validity-friendly SWC) still leave a gap of ~0.12 with t \u2248 7. Mirror trials cancel side/position bias, and the |d_v| threshold removes the validity-tie trial (d_v \u2248 0) where 'validity winner' is arbitrary. The metric works identically on the pooled DataFrame and on single-subject slices, so between-subject variance is well defined."
}
```

## Usage

```json
{
  "input_tokens": 6375,
  "output_tokens": 24164,
  "reasoning_tokens": 22757,
  "provider": "Sail Research"
}
```
