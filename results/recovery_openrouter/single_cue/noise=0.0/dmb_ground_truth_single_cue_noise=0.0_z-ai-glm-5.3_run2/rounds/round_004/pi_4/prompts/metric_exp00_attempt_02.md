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
**Validities (n_features=6):** [0.9, 0.65, 0.6, 0.8, 0.95, 0.7]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 2: A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 4: A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  trial 5: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 6: A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  trial 7: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 8: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1]
  trial 9: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 10: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 11: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 12: A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 13: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 14: A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  trial 15: A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 1]
  trial 16: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Rationale:** GOAL: Quantitatively dissociate Weakest-Expert-First Frugality (advocated; value-based ascending-validity cascade) from Reversed-Hierarchy Frugality (competing; positional misbinding, w = val[::-1]). The two theories are behaviorally IDENTICAL whenever the validity list is monotone in position, so the design rests on a scrambled arrangement that turns them into two genuinely different cue permutations — but with a novel twist relative to design [8]: here BOTH hierarchies are 'weak-expert dictatorships', so the design tests the MECHANISM of the anti-validity claim (value-sort vs positional misbinding) without the confound that subjects simply follow strong experts. SETUP: 6 features, all-distinct validities [0.55, 0.90, 0.65, 0.95, 0.60, 0.80] (f0..f5). Distinctness strips the advocated model's free tie_break parameter of any role. ADVOCATED hierarchy (ascending validity): f0 (55%) -> f4 (60%) -> f2 (65%) -> f5 (80%) -> f1 (90%) -> f3 (95%). COMPETING hierarchy (w = val[::-1] = [0.80, 0.60, 0.95, 0.65, 0.90, 0.55], descending): f2 (65%) -> f4 (60%) -> f0 (55%) -> f3 (95%) -> f1 (90%) -> f5 (80%). Note the competing model's TOP cue is the 65% expert f2 and the advocated model's top cue is the 55% expert f0 — both models trust weak experts first, but DIFFERENT weak experts, and both consult the 95% expert late. The two hierarchies disagree on exactly the cue pairs (f0,f2), (f0,f4), (f2,f4), (f1,f3), (f1,f5), (f3,f5); f4 and f1 are 'fixed points' sitting at the SAME cascade depth in both hierarchies (depth 2 and depth 5 respectively). SIGNATURE 1 — THE PURE PERMUTATION DISSOCIATION GRID AT THE TALLY KNIFE-EDGE (T1-T12): twelve cells, one per dissociating cue pair per direction, each with EXACTLY two discriminating cues in opposition and all others tied (tally margin exactly 0). On every one of these cells the advocated model follows its earlier cue and the competing model follows the OPPOSITE cue, so their predicted intervals are DISJOINT CELL-BY-CELL: advocated p(A) in [0.54, 0.80] toward its decider vs competing p(A) in [0.20, 0.43] — a direction flip on all twelve cells (T1-T12 come in six exact A/B mirror pairs, ruling out side bias). Because every cell sits at margin 0, ANY margin-integrating strategy (e.g. Tallying) is pinned to EXACTLY 0.50 on all twelve — a parameter-free knife-edge — so any consistent preference on any cell simultaneously falsifies the entire integration family while the two one-reason theories are the only candidates predicting systematic deviation: a three-way separation no prior design combined with the permutation axis. SIGNATURE 2 — WEAK-VS-WEAK DICTATORSHIP (novel axis vs design [8], which pitted strong against weak experts): the advocated decider sweeps 55% (f0: T1-T4), 60% (f4: T5-T6), 80% (f5: T9-T12), 90% (f1: T7-T8); the competing decider sweeps 65% (f2: T1,T2,T5,T6), 95% (f3: T7,T8,T11,T12), 90% (f1: T9,T10). If subjects follow the 95% expert on T7/T11 or the 90% expert on T9, BOTH anti-validity theories fail — a shared falsification probe that design [8] could not mount, since there the advocated model itself followed the strong experts. SIGNATURE 3 — CASCADE-DEPTH INVARIANCE WITHIN EACH MODEL: the deciding rung moves from depth 1 (T1-T4 for advocated; T1,T2,T5,T6 for competing) through depths 4-5 (T7-T12), and each model predicts the SAME follow probability p = (1-eps)*sigmoid(beta)+eps/2 regardless of whether its decider is consulted first or fifth — testable flatness across the grid, with the mirror structure balancing depth across options. SIGNATURE 4 — COALITION CELLS WITH ROLE REVERSAL (T13, T14): both at tally margin +1 for A. On T13 the advocated top cue f0 leads a 3-vs-2 majority for A while the competing top cue f2 dissents to B; on T14 the roles reverse (f0 dissents to B against a 3-2 majority, f2 leads the majority to A). Each model gets exactly one majority-siding and one minority-siding coalition cell, so a 'follow-the-majority' heuristic cannot mimic either theory's profile; both models also predict p(T13) = p(T1) = p(T3) (same decider f0) and p(T14) = p(T5) (same decider f2) — context invariance across margin and coalition size. SIGNATURE 5 — FIXED-POINT AND TIE CONTROLS: T15 has only f4 discriminating, a cue at depth 2 in BOTH hierarchies — an agreement cell where the two theories concur in choice AND in cascade depth, anchoring the follow rate and testing both models' depth-invariance claims where they cannot differ; any deviation of p(T15) from the dissociation-cell follow rates falsifies the flat-follow assumption of whichever model mispredicts it. T16 is a full tie (both theories: exactly 0.50), anchoring lapse rate and side bias. GLOBAL PROFILES: the advocated model emits p(A) ≈ [0.54, 0.80] on T1, T3, T6, T7, T10, T11, T13, T15 and ≈ [0.20, 0.46] on T2, T4, T5, T8, T9, T12, T14, with 0.50 on T16; the competing model emits the EXACT TRANSPOSE — a near-perfect sign-flipped anti-correlation on 14 of 16 cells (agreement only on T15 and T16). Because the intervals are disjoint cell-by-cell on all twelve permutation cells, model comparison on the full 16-cell response profile separates the theories even after fitting beta, epsilon, and tie_break to each — and even the aggregate direction rate across the twelve mirror cells (advocated: its decider's direction ~70%; competing: the opposite direction ~70%) is diagnostic. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Weakest-Expert-First Frugality (value-based anti-validity TTB). People are one-reason decision makers: they consult cues one at a time and let the first strictly discriminating cue decide, ignoring all lower cues and all magnitude/tally information. However, the subjective cue hierarchy is anti-correlated in VALUE with the instructed validities: subjects systematically consult the cue with the LOWEST advertised validity first, then the next-lowest, and so on (equivalently, subjective weights w_j = -validity_j). Because no integration across cues occurs, the model predicts a flat choice profile across tally margins (zero psychometric slope). When no cue discriminates, the model guesses uniformly. Validity ties are broken by a free per-subject tie-breaking parameter (early-position vs late-position preference among equally weak cues). Response noise enters through a softmax with inverse temperature beta over the binary winner score, plus an independent lapse epsilon mixing in a uniform choice, giving an implied cascade-follow probability p = (1-eps)*sigmoid(beta) + eps/2 that spans roughly 0.54 to 0.86 across subjects with a mean near 0.70 — high enough to produce the steep contrasts in the non-monotone-validity experiments while moderate values cover the monotone ones.

**Parameters:**
- beta: [0.5, 1.5]
- epsilon: [0.06, 0.18]
- tie_break: {0, 1}
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Weakest-Expert-First Frugality (value-based anti-validity TTB).
    # One-reason decision making: cues are consulted one at a time and
    # the first strictly discriminating cue decides. The subjective
    # hierarchy is anti-correlated in VALUE with the instructed
    # validities: the cue with the LOWEST advertised validity is
    # consulted FIRST, then the next-lowest, etc. (subjective weights
    # w_j = -validity_j). This is a VALUE-based sort (ascending
    # validity), not a positional reversal of the validity list, so in
    # experiments where the validity list is not monotone in feature
    # position the two hierarchies genuinely differ. No integration
    # across cues occurs -> flat psychometric profile across tally
    # margins. Noise: softmax (inverse temperature beta) over the
    # binary winner score, plus an independent lapse epsilon mixing in
    # a uniform choice. History is ignored (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    tie_break = float(parameters["tie_break"])

    # Weakest-expert-first hierarchy: sort cues by ASCENDING validity
    # value. Ties in validity are broken by the free tie_break
    # parameter: 0 -> earlier feature position first (stable),
    # 1 -> later feature position first among equally weak cues.
    # np.lexsort uses the LAST key as primary: primary = val ascending,
    # secondary = position (forward or reversed).
    pos = np.arange(n_features)
    secondary = pos if tie_break < 0.5 else (n_features - 1 - pos)
    cue_order = np.lexsort((secondary, val))

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    n_opts = 2
    if winner is None:
        # No discriminating cue — pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta). The implied cascade-follow
    # probability is p = (1-eps)*sigmoid(beta) + eps/2, which over the
    # sampled (beta, epsilon) ranges spans ~[0.54, 0.86] across
    # subjects with a mean near 0.70.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / float(n_opts))
```

**`policy source code`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Reversed-Hierarchy Frugality (misbinding TTB). People are one-reason decision makers: they consult cues one at a time and let the first strictly discriminating cue decide, ignoring all lower cues and all magnitude information. However, the subjective cue hierarchy is ANTI-correlated with the instructed validities: subjects misbind the communicated validity list to feature positions (binding the highest validity to the last-presented feature), so the cascade effectively runs in ascending true validity (kappa = 1, full misbinding). When no cue discriminates, the model guesses uniformly. Response noise enters through a softmax with inverse temperature beta over the binary winner score, plus an independent lapse epsilon mixing in a uniform choice. Because no integration across cues occurs, the model predicts a flat choice profile across tally margins (zero psychometric slope), in direct opposition to Tallying's graded curve. The noise regime is moderate: the cascade-follow probability p = (1-eps)*sigmoid(beta) + eps/2 sits near 0.60, matching the human follow rate implied by both experiments.

**Parameters:**
- beta: [0.6, 1.2]
- epsilon: [0.08, 0.18]
- kappa: {1.0}
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Reversed-Hierarchy Frugality (misbinding TTB). One-reason
    # decision making: cues are consulted one at a time and the first
    # strictly discriminating cue decides. The subjective hierarchy is
    # ANTI-correlated with the instructed validities (kappa = 1: the
    # communicated validity list is misbound to feature positions,
    # highest validity bound to the last-presented feature), so the
    # cascade runs in ascending true validity. No integration across
    # cues occurs -> flat psychometric profile across tally margins.
    # Noise: softmax (inverse temperature beta) over the binary winner
    # score, plus an independent lapse epsilon mixing in a uniform
    # choice. History is ignored (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # Subjective validities: mixture of the instructed vector and its
    # end-to-end reversal. kappa = 1 -> fully reversed hierarchy
    # (full positional misbinding), per the theory's fixed claim.
    kappa = float(parameters["kappa"])
    w = (1.0 - kappa) * val + kappa * val[::-1]

    # Descending SUBJECTIVE validity; stable argsort so ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-w, kind="stable")

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    n_opts = 2
    if winner is None:
        # No discriminating cue — pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta). The implied cascade-follow
    # probability is p = (1-eps)*sigmoid(beta) + eps/2, which over the
    # sampled (beta, epsilon) ranges averages ~0.68 with per-subject
    # values spanning ~[0.62, 0.75].
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / float(n_opts))
```

**`policy source code`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
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
[0] rationale: The two theories are one-reason (TTB) models that differ ONLY in their cue hierarchy. With the design's scrambled validities [0.9, 0.65, 0.6, 0.8, 0.95, 0.7], the advocated hierarchy (ascending validity) is [f2, f1, f5, f3, f0, f4] while the competing hierarchy (descending w = val[::-1]) is [f1, f5, f3, f0, f4, f2]. These two orders differ ONLY in the placement of feature f2 (first for advocated, last for competing); all other pairwise cue orders coincide. Therefore the theories make OPPOSITE predictions exactly on trials where f2 discriminates and so does some other cue that precedes f2 in the non-f2 order — in this design, trials T1, T2, T5, T6 (4 unique pairs x 6 reps = 24 trials per subject). On those trials the advocated cascade follows f2 (choice toward f2's direction with probability p ~ 0.72) while the competing cascade follows the other cue (choice toward the OPPOSITE direction with the same p). The metric computes, over exactly these theory-dissociating trials, the fraction of responses matching the ADVOCATED model's predicted choice. Under advocated-simulated data this is ~0.72; under competing-simulated data it is ~0.28 — a mean gap of ~0.44. Per-subject estimates are based on 24 Bernoulli trials, giving sampling sd ~0.09 plus between-subject spread in the follow probability p (sd ~0.07), for a total per-subject sd ~0.11; with N=25 the Welch t is ~13 (p ~ 1e-15), far below alpha = 0.01. The metric works identically on the pooled DataFrame and on single-subject slices (each subject has all 24 dissociating trials), so the between-subject variance is well defined. Trials where the theories agree (including the tie trial T16 and the f4-only control T15) are excluded, which sharpens the contrast rather than diluting it toward noise.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Fixed design parameters (n_features = 6, scrambled validities)
    validities = np.array([0.9, 0.65, 0.6, 0.8, 0.95, 0.7])
    n_features = len(validities)

    # Advocated hierarchy (Weakest-Expert-First): cues sorted by ASCENDING
    # true validity -> [2, 1, 5, 3, 0, 4] (f2=0.60 first, f4=0.95 last).
    adv_order = np.argsort(validities, kind="stable")

    # Competing hierarchy (Reversed-Hierarchy misbinding): subjective
    # weights w = val[::-1] = [0.70, 0.95, 0.60, 0.80, 0.65, 0.90],
    # consulted in DESCENDING subjective validity -> [1, 5, 3, 0, 4, 2].
    w = validities[::-1].copy()
    comp_order = np.argsort(-w, kind="stable")

    def first_discriminant(a, b, order):
        for j in order:
            if a[j] > b[j]:
                return 0  # option A wins the cascade
            if b[j] > a[j]:
                return 1  # option B wins the cascade
        return None

    hits = []
    for ra, rb, resp in zip(
        data["option_a_ratings"], data["option_b_ratings"], data["response"]
    ):
        a = np.asarray(ra, dtype=int).ravel()
        b = np.asarray(rb, dtype=int).ravel()
        if a.shape[0] != n_features or b.shape[0] != n_features:
            continue

        wa = first_discriminant(a, b, adv_order)
        wc = first_discriminant(a, b, comp_order)

        # Only keep trials where the two hierarchies genuinely disagree
        # (first discriminating cue differs between the two cascades).
        if wa is None or wc is None or wa == wc:
            continue

        hits.append(1.0 if int(resp) == wa else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
outcome: self_sim=0.5000 (var=0.0000) adversary_sim=0.5000 (var=0.0000) welch_t=+nan p=1 (N=25, alpha=0.01) -> reject

[1] rationale: The two theories are one-reason (TTB) models that differ ONLY in their cue hierarchy. With the design's validities [0.9, 0.65, 0.6, 0.8, 0.95, 0.7], the advocated hierarchy (ascending validity) is [f2, f1, f5, f3, f0, f4] while the competing hierarchy (descending w = val[::-1]) is [f1, f5, f3, f0, f4, f2]; the two orders differ only in the placement of f2 (first for advocated, last for competing). Hence the theories make OPPOSITE predictions exactly on trials where f2 discriminates and so does some other cue — in this design trial types T1, T2, T5, T6, T13, T14 (6 types x 6 reps = 36 trials per subject). On those trials the advocated cascade follows f2's direction with probability p ~ 0.70 while the competing cascade follows the OPPOSITE direction with the same p. The metric computes, over exactly these theory-dissociating trials (identified stimulus-by-stimulus by simulating both cascades, not by hard-coded trial indices), the fraction of responses matching the ADVOCATED model's predicted choice: ~0.72 under advocated-simulated data vs ~0.28 under competing-simulated data, a mean gap of ~0.44. Per-subject estimates rest on 36 Bernoulli trials (binomial sd ~0.08) plus between-subject spread in follow probability p (sd ~0.07), giving total per-subject sd ~0.11; with N=25 the Welch t is ~14 (p << 1e-10), far below alpha = 0.01. The mirror-pair structure of the dissociating cells (T1/T2, T5/T6, T13/T14) cancels side bias, and excluding agreement trials (T3, T4, T7-T12, T15, T16) removes pure noise that would dilute the contrast. The metric works identically on the pooled DataFrame and on single-subject slices (every subject completes all 96 trials, so 36 dissociating trials each), so the between-subject variance is well defined. It is also sign-robust to the design's validity-list ambiguity: even if the simulation used the alternative scrambled list from the design rationale, the two theories still fall on opposite sides of this metric (~0.30 vs ~0.70), because the dissociation is defined by the two cascades disagreeing, and only the sign (not the existence or magnitude) of the gap changes. Finally, this version hardens the cell parsing that likely caused the previous attempt to fail: rating cells are accepted as lists, tuples, numpy arrays, or any string encoding ('[1, 0, ...]', '1 0 ...', '100000', quoted digits), responses as 0/1 ints/floats, booleans, or 'A'/'B' strings, and column names are resolved with fallbacks — so no data-format quirk can silently reduce the metric to a constant 0.5.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # ---------------- fixed design constants ----------------
    validities = np.array([0.9, 0.65, 0.6, 0.8, 0.95, 0.7])
    n_features = int(validities.size)

    # Advocated hierarchy (Weakest-Expert-First Frugality): cues sorted
    # by ASCENDING true validity  -> [2, 1, 5, 3, 0, 4].
    adv_order = np.argsort(validities, kind="stable").tolist()
    # Competing hierarchy (Reversed-Hierarchy misbinding): subjective
    # weights w = validities[::-1], consulted in DESCENDING subjective
    # validity -> [1, 5, 3, 0, 4, 2].
    comp_order = np.argsort(-validities[::-1], kind="stable").tolist()

    # ---------------- robust cell parsers ----------------
    def parse_vec(x):
        # Accept lists, tuples, ndarrays, and any string encoding
        # ("[1, 0, ...]", "1 0 ...", "100000", "['1','0',...]").
        try:
            if isinstance(x, np.ndarray):
                return np.asarray(x, dtype=int).ravel()
            if isinstance(x, (list, tuple)):
                return np.asarray(list(x), dtype=int).ravel()
            s = str(x).strip()
            core = s[1:-1] if (s.startswith("[") and s.endswith("]")) else s
            toks = core.replace(",", " ").replace(";", " ").split()
            if (
                len(toks) == 1
                and len(toks[0]) == n_features
                and all((ch in "01") for ch in toks[0])
            ):
                return np.asarray([int(ch) for ch in toks[0]], dtype=int)
            vals = [int(float(t)) for t in toks]
            return np.asarray(vals, dtype=int).ravel()
        except Exception:
            try:
                digits = [int(ch) for ch in str(x) if ch in "01"]
                return np.asarray(digits, dtype=int).ravel()
            except Exception:
                return np.asarray([], dtype=int)

    def parse_resp(x):
        # Accept 0/1 ints/floats, booleans, 'A'/'B', '0'/'1' strings.
        try:
            if isinstance(x, (bool, np.bool_)):
                return int(x)
            if isinstance(x, (int, np.integer)):
                r = int(x)
                return r if r in (0, 1) else None
            if isinstance(x, (float, np.floating)):
                r = int(round(float(x)))
                return r if r in (0, 1) else None
            s = str(x).strip().upper()
            if s == "A":
                return 0
            if s == "B":
                return 1
            r = int(float(s))
            return r if r in (0, 1) else None
        except Exception:
            return None

    def first_disc(a, b, order):
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return None

    # ---------------- column resolution ----------------
    cols = [str(c) for c in list(data.columns)]
    lookup = {}
    for c in list(data.columns):
        lookup[str(c)] = c

    a_col = None
    for c in ["option_a_ratings", "optionA_ratings", "ratings_a", "a_ratings"]:
        if c in lookup:
            a_col = lookup[c]
            break
    if a_col is None:
        for c in cols:
            lc = c.lower()
            if ("a" in lc) and ("rating" in lc):
                a_col = lookup[c]
                break

    b_col = None
    for c in ["option_b_ratings", "optionB_ratings", "ratings_b", "b_ratings"]:
        if c in lookup:
            b_col = lookup[c]
            break
    if b_col is None:
        for c in cols:
            lc = c.lower()
            if ("b" in lc) and ("rating" in lc):
                b_col = lookup[c]
                break

    r_col = None
    for c in ["response", "choice", "chosen", "answer", "pick", "selected"]:
        if c in lookup:
            r_col = lookup[c]
            break
    if r_col is None:
        for c in cols:
            lc = c.lower()
            if ("respon" in lc) or ("choic" in lc) or ("select" in lc):
                r_col = lookup[c]
                break

    if a_col is None or b_col is None or r_col is None:
        return 0.5

    # ---------------- core computation ----------------
    # Keep exactly the trials on which the two hierarchies render
    # OPPOSITE choices (their first strictly discriminating cues
    # differ), and score agreement with the advocated cascade.
    hits = []
    for ra, rb, resp in zip(data[a_col], data[b_col], data[r_col]):
        a = parse_vec(ra)
        b = parse_vec(rb)
        if a.size != n_features or b.size != n_features:
            continue
        if (not np.all((a == 0) | (a == 1))) or (not np.all((b == 0) | (b == 1))):
            continue
        r = parse_resp(resp)
        if r is None:
            continue
        wa = first_disc(a, b, adv_order)
        wc = first_disc(a, b, comp_order)
        if wa is None or wc is None or wa == wc:
            continue
        hits.append(1.0 if (r == wa) else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
outcome: self_sim=0.5000 (var=0.0000) adversary_sim=0.5000 (var=0.0000) welch_t=+nan p=1 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # ---------------- fixed design constants ----------------\n    NF = 6  # n_features\n    # Instructed validities (design of record): [0.9, 0.65, 0.6, 0.8, 0.95, 0.7]\n    # Advocated (Weakest-Expert-First) hierarchy = ASCENDING validity:\n    #   f2 (.60) -> f1 (.65) -> f5 (.70) -> f3 (.80) -> f0 (.90) -> f4 (.95)\n    ADV_ORDER = (2, 1, 5, 3, 0, 4)\n    # Competing (Reversed-Hierarchy) hierarchy: w = validities[::-1]\n    #   = [.70, .95, .60, .80, .65, .90], consulted DESCENDING:\n    #   f1 -> f5 -> f3 -> f0 -> f4 -> f2\n    COMP_ORDER = (1, 5, 3, 0, 4, 2)\n\n    SEQ = (list, tuple, np.ndarray)\n    NAME_SKIP = (\"subject\", \"subj\", \"trial\", \"block\", \"participant\")\n\n    # ---------------- robust parsing ----------------\n    def _ints_from_str(s):\n        s = s.strip()\n        if s.startswith(\"[\") and s.endswith(\"]\"):\n            s = s[1:-1]\n        if s.startswith(\"(\") and s.endswith(\")\"):\n            s = s[1:-1]\n        s = s.replace(\",\", \" \").replace(\";\", \" \").replace(\"|\", \" \")\n        toks = s.split()\n        if len(toks) == 1:\n            t0 = toks[0].strip(\"'\\\"\")\n            if 2 <= len(t0) <= 2 * NF and all(ch in \"01\" for ch in t0):\n                return [int(ch) for ch in t0]\n        out = []\n        for t in toks:\n            t2 = t.strip(\"'\\\"\")\n            try:\n                out.append(int(round(float(t2))))\n            except Exception:\n                out.extend(int(ch) for ch in t2 if ch in \"01\")\n        return out\n\n    def parse_vec(x):\n        try:\n            if isinstance(x, np.ndarray):\n                flat = [int(round(float(v))) for v in x.ravel()]\n            elif isinstance(x, (list, tuple)):\n                flat = []\n                for e in x:\n                    if isinstance(e, (bool, np.bool_, int, np.integer, float, np.floating)):\n                        flat.append(int(round(float(e))))\n                    else:\n                        s = str(e).strip().strip(\"'\\\"\")\n                        flat.append(int(round(float(s))))\n            elif isinstance(x, (bool, np.bool_, int, np.integer, float, np.floating)):\n                return None  # scalar: not a rating vector\n            else:\n                flat = _ints_from_str(str(x))\n        except Exception:\n            try:\n                flat = [int(ch) for ch in str(x) if ch in \"01\"]\n            except Exception:\n                return None\n        if not flat:\n            return None\n        arr = np.asarray(flat, dtype=int)\n        if arr.size == 0 or not np.all((arr == 0) | (arr == 1)):\n            return None\n        return arr\n\n    def first_disc(av, bv, order):\n        for j in order:\n            if av[j] > bv[j]:\n                return 0\n            if bv[j] > av[j]:\n                return 1\n        return None\n\n    # ---------------- response decoding ----------------\n    # Generic: map the two distinct response values (whatever their type /\n    # coding: 0/1, 1/2, -1/1, 'A'/'B', 'option_a'/'option_b', bools, ...) to\n    # 0/1 by lexicographic order. A globally flipped coding only flips the\n    # sign of the metric consistently for both theories, which the two-sided\n    # Welch test tolerates. Sequence-valued responses (the chosen option's\n    # rating vector) are matched against the A/B vectors directly.\n    def decode_responses(raw_r, pairs):\n        n = len(raw_r)\n        bits = [None] * n\n        n_seq = 0\n        for v in raw_r:\n            if isinstance(v, SEQ):\n                n_seq += 1\n        if n_seq > 0 and 2 * n_seq >= n:\n            for i in range(n):\n                v = raw_r[i]\n                if not isinstance(v, SEQ) or pairs[i] is None:\n                    continue\n                try:\n                    vv = tuple(int(round(float(t))) for t in np.asarray(v).ravel())\n                    ta = tuple(int(t) for t in pairs[i][0])\n                    tb = tuple(int(t) for t in pairs[i][1])\n                except Exception:\n                    continue\n                if vv == ta:\n                    bits[i] = 0\n                elif vv == tb:\n                    bits[i] = 1\n            return bits\n        counts = {}\n        for v in raw_r:\n            if isinstance(v, SEQ) or v is None:\n                continue\n            if isinstance(v, float):\n                try:\n                    if np.isnan(v):\n                        continue\n                except Exception:\n                    pass\n            k = str(v).strip()\n            if k == \"\" or k.lower() in (\"nan\", \"none\", \"null\", \"<na>\"):\n                continue\n            counts[k] = counts.get(k, 0) + 1\n        if not counts:\n            return bits\n        if len(counts) == 1:\n            for i in range(n):\n                v = raw_r[i]\n                if not isinstance(v, SEQ) and v is not None:\n                    bits[i] = 0\n            return bits\n        top2 = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:2]\n        k1, k2 = sorted((top2[0][0], top2[1][0]))\n        mp = {k1: 0, k2: 1}\n        for i in range(n):\n            v = raw_r[i]\n            if isinstance(v, SEQ) or v is None:\n                continue\n            kk = str(v).strip()\n            if kk in mp:\n                bits[i] = mp[kk]\n        return bits\n\n    # ---------------- core analysis ----------------\n    def analyze(a, b, r):\n        if a is None or b is None or r is None:\n            return None, None\n        try:\n            raw_a = list(data[a])\n            raw_b = list(data[b])\n            raw_r = list(data[r])\n        except Exception:\n            return None, None\n        pairs = []\n        for ra, rb in zip(raw_a, raw_b):\n            av = parse_vec(ra)\n            bv = parse_vec(rb)\n            if av is not None and bv is not None and av.size == NF and bv.size == NF:\n                pairs.append((av, bv))\n            elif av is not None and av.size == 2 * NF:\n                # one cell holding the full flattened 2 x NF state\n                pairs.append((av[:NF], av[NF:]))\n            elif bv is not None and bv.size == 2 * NF:\n                pairs.append((bv[:NF], bv[NF:]))\n            else:\n                pairs.append(None)\n        if sum(1 for p in pairs if p is not None) == 0:\n            return None, None\n        bits = decode_responses(raw_r, pairs)\n        scores = []   # dissociation trials only (sharp contrast)\n        fb = []       # all discriminating trials (fallback contrast)\n        for i in range(len(pairs)):\n            if pairs[i] is None or bits[i] is None:\n                continue\n            av, bv = pairs[i]\n            wa = first_disc(av, bv, ADV_ORDER)\n            if wa is None:\n                continue\n            wc = first_disc(av, bv, COMP_ORDER)\n            if wc is not None and wc != wa:\n                scores.append(1.0 if bits[i] == wa else -1.0)\n            fb.append(1.0 if bits[i] == wa else -1.0)\n        return scores, fb\n\n    # ---------------- column resolution ----------------\n    cols = list(data.columns)\n    lmap = {}\n    for c in cols:\n        lmap[str(c).lower()] = c\n\n    def name_resolve():\n        a = b = r = None\n        for k in (\"option_a_ratings\", \"optiona_ratings\", \"ratings_a\",\n                  \"a_ratings\", \"option_a\", \"opt_a\", \"stim_a\",\n                  \"features_a\", \"a\"):\n            if k in lmap:\n                a = lmap[k]\n                break\n        for k in (\"option_b_ratings\", \"optionb_ratings\", \"ratings_b\",\n                  \"b_ratings\", \"option_b\", \"opt_b\", \"stim_b\",\n                  \"features_b\", \"b\"):\n            if k in lmap:\n                b = lmap[k]\n                break\n        for k in (\"response\", \"choice\", \"chosen\", \"answer\", \"pick\",\n                  \"selected\", \"resp\"):\n            if k in lmap:\n                r = lmap[k]\n                break\n        if a is None:\n            for c in cols:\n                lc = str(c).lower()\n                if (\"a\" in lc) and (\"b\" not in lc) and any(\n                    kw in lc for kw in (\"rat\", \"feat\", \"opt\", \"cue\", \"expert\", \"stim\", \"item\")\n                ):\n                    a = c\n                    break\n        if b is None:\n            for c in cols:\n                lc = str(c).lower()\n                if (c != a) and (\"b\" in lc) and any(\n                    kw in lc for kw in (\"rat\", \"feat\", \"opt\", \"cue\", \"expert\", \"stim\", \"item\")\n                ):\n                    b = c\n                    break\n        if r is None:\n            for c in cols:\n                lc = str(c).lower()\n                if any(k in lc for k in (\"respon\", \"choic\", \"chose\", \"select\", \"answer\", \"pick\")):\n                    r = c\n                    break\n        return a, b, r\n\n    def detect_vec_cols():\n        found = []\n        for c in cols:\n            lc = str(c).lower()\n            if any(k in lc for k in NAME_SKIP) or lc == \"id\":\n                continue\n            try:\n                s = data[c].dropna()\n            except Exception:\n                continue\n            if len(s) == 0:\n                continue\n            sample = list(s.head(10))\n            good = 0\n            for v in sample:\n                p = parse_vec(v)\n                if p is not None and p.size >= 2:\n                    good += 1\n            if good > 0 and 2 * good >= len(sample):\n                found.append(c)\n        return found\n\n    def detect_resp_col(exclude):\n        two_cols = []\n        seq_cols = []\n        for c in cols:\n            if c in exclude:\n                continue\n            lc = str(c).lower()\n            if any(k in lc for k in NAME_SKIP) or lc == \"id\":\n                continue\n            is_kw = any(k in lc for k in (\"respon\", \"choic\", \"chose\", \"select\", \"answer\", \"pick\"))\n            try:\n                s = data[c].dropna()\n                if len(s) > 0 and isinstance(s.iloc[0], SEQ):\n                    seq_cols.append(c)\n                    continue\n                nu = len(s.unique())\n            except Exception:\n                continue\n            if nu == 2:\n                if is_kw:\n                    return c\n                two_cols.append(c)\n        if two_cols:\n            return two_cols[0]\n        if seq_cols:\n            return seq_cols[0]\n        for c in cols:\n            lc = str(c).lower()\n            if any(k in lc for k in (\"respon\", \"choic\", \"chose\", \"select\", \"answer\", \"pick\")):\n                return c\n        return None\n\n    # ---------------- main flow ----------------\n    a, b, r = name_resolve()\n    scores, fb = analyze(a, b, r)\n\n    if scores is None or len(scores) < max(1, len(data) // 8):\n        vc = detect_vec_cols()\n        excl = list(vc)\n        for x in (a, b, r):\n            if x is not None and x not in excl:\n                excl.append(x)\n        r_cands = []\n        if r is not None:\n            r_cands.append(r)\n        r_alt = detect_resp_col(excl)\n        if r_alt is not None and r_alt not in r_cands:\n            r_cands.append(r_alt)\n        if len(vc) >= 2:\n            ca, cb = vc[0], vc[1]\n            if a is not None and b is not None and a in vc and b in vc:\n                ca, cb = a, b\n            best = scores if scores is not None else []\n            best_fb = fb if fb is not None else []\n            for rc in r_cands:\n                s2, f2 = analyze(ca, cb, rc)\n                if s2 is not None and len(s2) > len(best):\n                    best = s2\n                    best_fb = f2\n            scores, fb = best, best_fb\n\n    if scores:\n        return float(np.mean(scores))\n    if fb:\n        return float(np.mean(fb))\n    return 0.0",
  "rationale": "CORE CONTRAST. With the design's validities [0.9, 0.65, 0.6, 0.8, 0.95, 0.7], the advocated hierarchy (ascending validity) is [f2, f1, f5, f3, f0, f4] and the competing hierarchy (descending w = val[::-1]) is [f1, f5, f3, f0, f4, f2]; these orders differ only in f2's placement (first vs last). The theories therefore render OPPOSITE choices exactly on trials where f2 discriminates together with some other cue \u2014 in this design T1, T2, T5, T6 (4 types x 6 reps = 24 trials per subject). The metric detects these dissociation trials stimulus-by-stimulus (no hard-coded trial indices) and returns the signed agreement rate with the ADVOCATED cascade: +1 when the response matches the advocated cascade's choice, -1 when it matches the competing cascade's (on dissociation trials these are the only two possibilities). Expected values: ~ +0.40 under advocated-simulated data (follow probability p ~ 0.70 gives 2p-1 ~ 0.4) and ~ -0.40 under competing-simulated data \u2014 a mean gap of ~0.8. Per-subject sd ~ 0.22 (binomial noise over 24 trials plus between-subject spread in p), so between-subject variance ~ 0.05; with N = 25 the Welch t is ~ 0.8 / sqrt(2*0.05/25) ~ 12.7, p ~ 1e-13, far below alpha = 0.01. The metric works identically on the pooled DataFrame and on single-subject slices (every subject sees all 24 dissociation trials), so the between-subject variance is well defined. SIGN ROBUSTNESS: if the response coding or A/B labeling is flipped anywhere in the pipeline, both theory conditions flip sign together, which the two-sided test tolerates; and even if the simulation used the alternative scrambled validity list from the design rationale, the same 24 trials remain dissociating (the two hierarchies disagree there under either list) and the gap magnitude is preserved (only its sign may flip). WHY THE PREVIOUS ATTEMPTS FAILED AND WHAT CHANGED. Both prior metrics returned EXACTLY 0.5000 with EXACTLY zero between-subject variance on both simulated datasets \u2014 the signature of the metric falling through to its default because every row was discarded (unparseable cells, unfound columns, or undecodable responses), not of a weak statistical contrast. This version therefore (a) parses rating cells as lists, tuples, ndarrays, or any string encoding ('[1, 0, ...]', '100000', '(1;0;...)', quoted digits, bytes-reprs), (b) decodes responses with a GENERIC two-value mapping inferred from the data (handles 0/1, 1/2, -1/1, 'A'/'B', 'option_a'/'option_b', booleans, floats, strings), or by direct vector matching if responses are the chosen option's rating vector, (c) resolves columns by exact name, then name heuristics, then CONTENT-based detection (columns whose cells parse as binary vectors of length >= 2; response = the non-rating column with exactly two distinct values), (d) handles cells that hold the full flattened 2x6 state by splitting them in half, and (e) falls back \u2014 only if no dissociation trial survives \u2014 to the signed agreement with the advocated cascade over ALL discriminating trials (self ~ +0.40 vs competing ~ +0.20, still a significant contrast: t ~ 4.7 at N = 25). The primary dissociation-only statistic is retained because it maximizes the standardized between-theory gap."
}
```

## Usage

```json
{
  "input_tokens": 9047,
  "output_tokens": 45973,
  "reasoning_tokens": 41896,
  "provider": "Sail Research"
}
```
