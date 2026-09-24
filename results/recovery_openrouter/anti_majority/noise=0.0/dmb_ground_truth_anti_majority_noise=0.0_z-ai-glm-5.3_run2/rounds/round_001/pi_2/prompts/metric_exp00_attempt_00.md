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
**Validities (n_features=6):** [1.0, 0.95, 0.65, 0.5, 0.5, 0.5]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 2: A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  trial 3: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 5: A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  trial 6: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 7: A=[1, 1, 0, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0]
  trial 8: A=[1, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  trial 9: A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 10: A=[1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1]
  trial 11: A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 12: A=[1, 1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 13: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 15: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 16: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Rationale:** ADVERSARIAL LOGIC (the 'transpose' of the count-tie trick). On binary cues, Tallying's evidence is the integer count margin m = a_wins - b_wins, while SCWI's evidence is the anti-validity weighted margin D = -sum_j (v_j + tau*v_j^2)*s_j (s_j = +1 if A wins cue j, -1 if B wins). The already-explored design pinned TALLYING at 0.5 (count ties) and let SCWI vary. I do the exact inverse: I engineer WEIGHTED TIES (D ~= 0, robustly across the full tau range [0.25, 0.45]) at NONZERO tally margins, so SCWI is pinned near 0.5 while Tallying is graded — and then hold D at a large fixed negative value while m varies, so SCWI is flat-low while Tallying rises. This yields a two-probe double slope dissociation in the opposite assignment from the prior design.

VALIDITY ENGINEERING. With validities [1.00, 0.95, 0.65, 0.54, 0.51, 0.50], a weighted tie between a cue set X and cue set Y requires sum_X v + tau*sum_X v^2 = sum_Y v + tau*sum_Y v^2. I choose sets where the constant and tau-coefficient nearly cancel simultaneously: e.g., the four low cues {3,4,5,6} (sum v = 2.20, sum v^2 = 1.224) vs the two top cues {1,2} (1.95, 1.9025): residual c(tau) = 0.25 - 0.678*tau, i.e., |c| <= 0.080 for ALL tau in [0.25, 0.45]. Similar cancellations: {4,5} vs {2} (|c| <= 0.058) and {3,6} vs {1} (|c| <= 0.068).

PROBE A (pairs 1-6, weighted ties at nonzero tally margin): pairs 1-2 have m = +1 and pair 3 has m = +2, all with |D| <= 0.08 across the entire SCWI parameter box. SCWI therefore predicts P(A) = 0.500 +/- 0.032 on ALL SIX (including mirrors) for ANY beta in [1.1,1.7], tau in [0.25,0.45], epsilon in [0.05,0.15] — a horizontal line with essentially no free parameters. Tallying predicts a graded ladder: P(A) = sigmoid(m*beta)*(1-eps)+eps/2, with lapse-corrected logits exactly proportional to m (the m=+1 and m=+2 trials must sit in 1:2 logit ratio, symmetric under mirroring). Any reliable, mirror-symmetric departure from 0.5 that grows with |m| falsifies SCWI outright; perfect flatness at 0.5 falsifies Tallying (which must exceed 0.5 at m>0 for any parameters).

PROBE B (pairs 7-12, fixed weighted evidence, varying tally margin): three trials with D pinned in a narrow confident band — pair 7: m=+1, D in [-1.30,-1.13]; pair 8: m=+2, D in [-1.18,-1.02]; pair 9: m=+3, D in [-1.30,-1.26]. SCWI predicts a FLAT-LOW P(A) ~ 0.15-0.26 on all three (mirrors ~0.74-0.85), because its evidence barely changes. Tallying predicts P(A) >= 0.5 always (m>0), strictly increasing in m: ~0.86 -> 0.96 -> 0.97 at beta=2. Huge vertical separation plus opposite slope sign. Note this is the inverse hold of the prior design's fixed-margin probe (there: m fixed, D varied; here: D fixed, m varies).

MINIMAL PAIR (pairs 3 vs 9): identical stimuli except cue 2 changes from a B-win to a tie. Tallying: m goes +2 -> +3, P(A) barely moves (0.96 -> 0.97). SCWI: D jumps from -0.01 to -1.28, P(A) collapses 0.50 -> 0.18. A single-cue manipulation that one theory treats as negligible and the other as decisive.

ANCHORS AND CONTROLS: pairs 13/14 (one option wins all six cues, m = +/-6, |D| ~ 5.0-5.6) are maximum-separation trials — Tallying ~0.97 vs SCWI ~epsilon/2 (~0.05). Pairs 15/16 are full ties (identical vectors, one all-0 and one mixed) where BOTH theories predict exactly 0.500, providing a pure lapse/side-bias baseline and verifying that tie values (0 vs 1) are inert. All critical pairs are mirrored to absorb side/response bias. 16 unique pairs, K = 96//16 = 6 repetitions = 96 trials.

DECISION RULES: (i) If P(A) on pairs 1-6 deviates from 0.5 symmetrically and grows with |m| (logit ratio ~1:2 for m=1 vs m=2), Tallying is supported and SCWI is falsified — SCWI cannot exceed 0.532 or dip below 0.468 there for any allowed parameters. (ii) If pairs 1-6 are flat at 0.5 while pairs 7-12 sit flat at ~0.2 (mirrors ~0.8) and pair 13 shows P(A) ~ 0.05 (anti-following the all-1s option), SCWI is supported and Tallying is falsified — Tallying cannot go below 0.5 - eps/2 at m>0. (iii) Flat 0.5 on Probe A combined with a Tallying-like increase on Probe B (or any preference for the tally winner graded by validity weights) falsifies both pure forms. Bonus: pair 9 also separates TTB (which picks B via cue 1) from Tallying, so a TTB-like subject is identifiable rather than confounded with SCWI.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
**Description:** Signed-Cue Weighted Integration (SCWI). When choosing between two options described by binary expert ratings, subjects do not use a lexicographic cascade (TTB) and do not merely count wins (Tallying). Instead they integrate ALL cues additively into a single subjective value per option — but the mapping from a rating to evidence carries a systematic SIGN INVERSION: a rating of 1 is treated as a defect/criticism rather than an endorsement (a comprehension inversion of the rating direction). Moreover, the inversion is not uniform across cues: the subjective weight of cue j is w_j = -(v_j + tau * v_j^2), i.e. an anti-validity weighting that grows SUPRALINEARLY with the instructed validity — the endorsements of the most trusted experts are avoided most strongly, while low-validity experts are nearly ignored (their anti-weight is small in magnitude). The option with the higher (less negative) signed value is preferred, with softmax noise (inverse temperature beta) and an independent lapse (epsilon) capturing response variability. This single task-invariant mechanism predicts both experimental signatures simultaneously: in Experiment 1, the top-cue-favored option carries more 1s and hence more 'defects', so P(choose the top-cue-favored option) DECREASES with its tally margin (negative slope); in Experiment 2, on margin-1 trials where TTB and the tally disagree the two options' signed values are close together (so noise frequently flips the subject onto the tally winner), whereas when they agree the value gap is large (so the subject reliably anti-follows) — yielding p_disagree > p_agree, a positive metric.

**Parameters:**
- beta: [1.1, 1.7]
- tau: [0.25, 0.45]
- epsilon: [0.05, 0.15]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Signed-Cue Weighted Integration (SCWI).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Each cue contributes signed evidence w_j = -(v_j + tau * v_j^2)
    # per unit rating (anti-validity weighting, supralinear in the
    # instructed validity). Option values are the cue-weighted sums;
    # choice is a numerically stable softmax over beta * values, mixed
    # with a symmetric lapse. History is ignored (no feedback).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SCWI expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    raw = parameters.get("validities")
    if isinstance(raw, str):
        s = raw.strip().strip('[]()')
        val = np.array([float(t) for t in s.replace(',', ' ').split()])
    else:
        val = np.asarray(raw, dtype=float).ravel()
    if val.size != n_features:
        # Defensive fallback: uniform mid-range validities.
        val = np.full(n_features, 0.75)

    beta = float(parameters["beta"])
    tau = float(parameters["tau"])
    epsilon = float(parameters["epsilon"])

    # Signed (anti-validity) cue weights: a rating of 1 is evidence
    # AGAINST an option, and the inversion strength grows supralinearly
    # with the instructed validity of the expert.
    w = -(val + tau * np.square(val))

    scores = np.array([float(np.dot(w, stim[0])), float(np.dot(w, stim[1]))])

    # Numerically stable softmax over the two option values.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Guard against float drift.
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Conflict-set tally-following rate.\n    #\n    # Selection rule: keep only trials where (i) the tally is non-zero (one\n    # option strictly wins more feature-wise comparisons) and (ii) a\n    # validity-weighted signed integrator with anti-validity weights\n    # w_j = v_j + tau * v_j^2 (tau = 0.35, the midpoint of SCWI's range)\n    # STRONGLY prefers the tally LOSER (|evidence| > 0.8).\n    # In this design that selects exactly Probe B (pairs 7-12, |D| in\n    # [1.0, 1.3]) plus the anchors (pairs 13-14, |D| ~ 5.2), i.e. 8 of the\n    # 16 unique pairs = 48 of 96 trials per subject.\n    #\n    # Metric value = proportion of choices that follow the TALLY WINNER on\n    # those conflict trials.\n    #   Tallying: P >= sigmoid(beta*|m|) with |m| in {1,3,6}  -> > 0.5 for\n    #             ANY beta > 0, epsilon < 1; ~0.95+ for typical beta.\n    #   SCWI:     P = (1-eps)*(1-sigmoid(beta*|D|)) + eps/2 ~ 0.16-0.20 on\n    #             Probe B and ~ eps/2 ~ 0.075 on anchors -> far below 0.5.\n    # The two theories are separated by ~0.4-0.8 in mean with tight\n    # per-subject binomial variance (48 trials/subject).\n\n    import numpy as np\n\n    VALIDITIES = np.array([1.0, 0.95, 0.65, 0.5, 0.5, 0.5])\n    TAU = 0.35      # midpoint of SCWI's tau box [0.25, 0.45]\n    THRESH = 0.8    # cleanly above Probe A's |D| <= 0.08, below Probe B's |D| >= 1.0\n\n    w = VALIDITIES + TAU * np.square(VALIDITIES)  # positive anti-validity magnitudes\n\n    a_col = data[\"option_a_ratings\"].tolist()\n    b_col = data[\"option_b_ratings\"].tolist()\n    r_col = data[\"response\"].tolist()\n\n    n_sel = 0\n    n_match = 0\n    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):\n        a = np.asarray(a_raw, dtype=float).ravel()\n        b = np.asarray(b_raw, dtype=float).ravel()\n        if a.size == 0 or a.size != b.size or a.size != w.size:\n            continue\n\n        a_wins = int(np.sum(a > b))\n        b_wins = int(np.sum(b > a))\n        if a_wins == b_wins:\n            # Full-tie trials (pairs 15/16): both theories say 0.5, uninformative.\n            continue\n\n        # Signed evidence for option A under a weighted anti-validity\n        # integrator: value_A - value_B = sum_j w_j * (b_j - a_j).\n        ev = float(np.dot(w, b - a))\n\n        tally_a = a_wins > b_wins\n        if tally_a:\n            # A is the tally winner; keep only if the integrator strongly\n            # prefers B (the tally loser).\n            if ev > -THRESH:\n                continue\n        else:\n            # B is the tally winner; keep only if the integrator strongly\n            # prefers A (the tally loser).\n            if ev < THRESH:\n                continue\n\n        r = int(round(float(resp)))  # 0 = chose A, 1 = chose B\n        chose_tally_winner = (r == 0) if tally_a else (r == 1)\n        n_sel += 1\n        if chose_tally_winner:\n            n_match += 1\n\n    if n_sel == 0:\n        return 0.5  # degenerate fallback (no conflict trials found)\n    return float(n_match) / float(n_sel)",
  "rationale": "The metric isolates the trials the design engineered as maximal-conflict: Probe B (pairs 7-12) and the anchors (pairs 13-14), where the tally margin is positive (|m| in {1,3,6}) but the SCWI signed evidence strongly favors the tally loser (|D| between ~1.0 and ~5.2). Selection is done from the raw ratings themselves: I recompute the tally margin and the anti-validity weighted evidence with tau = 0.35 (the midpoint of SCWI's box; the selection is insensitive to tau because Probe B's |D| stays in [1.0, 1.3] and Probe A's stays <= 0.08 across the whole tau range, so the 0.8 threshold separates them robustly). Full-tie trials (pairs 15/16) are dropped since both theories predict exactly 0.5 there. On the selected 48 trials per subject, Tallying predicts P(follow tally winner) = (1-eps)*sigmoid(beta*|m|) + eps/2, which is strictly above 0.5 for any beta > 0 and epsilon < 1 (worst case ~0.53 at beta=0.1, eps=0.5; ~0.95+ for typical beta). SCWI predicts (1-eps)*(1-sigmoid(beta*|D|)) + eps/2, i.e. ~0.16-0.20 on Probe B and ~eps/2 (~0.075) on the anchors, averaging ~0.16 \u2014 far below 0.5, and below Tallying's prediction in every corner of either parameter box. So the two theories produce visibly different numbers (Tallying > 0.5 always, SCWI ~ 0.16) with the sign of the gap guaranteed, while per-subject variance stays tight (48 Bernoulli trials per subject, p near 0 or 1 under both theories for typical parameters, giving SD ~ 0.04-0.07). This makes Welch's t-test at N=25 overwhelmingly significant. The metric is also a genuine behavioral signature rather than a tautology: it asks whether human subjects side with the count-based winner precisely when a weighted integrator would strongly oppose it \u2014 the single cleanest dissociation the design provides (e.g., the pair-3 vs pair-9 minimal contrast: changing cue 2 from a B-win to a tie barely moves Tallying but collapses SCWI's preference)."
}
```

## Usage

```json
{
  "input_tokens": 5218,
  "output_tokens": 12802,
  "reasoning_tokens": 11263,
  "provider": "Sail Research"
}
```
