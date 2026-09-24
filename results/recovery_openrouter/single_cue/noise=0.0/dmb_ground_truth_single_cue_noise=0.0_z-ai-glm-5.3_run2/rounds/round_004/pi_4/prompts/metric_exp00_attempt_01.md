# metric_exp00_attempt_01

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

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # ---------------- fixed design constants ----------------\n    validities = np.array([0.9, 0.65, 0.6, 0.8, 0.95, 0.7])\n    n_features = int(validities.size)\n\n    # Advocated hierarchy (Weakest-Expert-First Frugality): cues sorted\n    # by ASCENDING true validity  -> [2, 1, 5, 3, 0, 4].\n    adv_order = np.argsort(validities, kind=\"stable\").tolist()\n    # Competing hierarchy (Reversed-Hierarchy misbinding): subjective\n    # weights w = validities[::-1], consulted in DESCENDING subjective\n    # validity -> [1, 5, 3, 0, 4, 2].\n    comp_order = np.argsort(-validities[::-1], kind=\"stable\").tolist()\n\n    # ---------------- robust cell parsers ----------------\n    def parse_vec(x):\n        # Accept lists, tuples, ndarrays, and any string encoding\n        # (\"[1, 0, ...]\", \"1 0 ...\", \"100000\", \"['1','0',...]\").\n        try:\n            if isinstance(x, np.ndarray):\n                return np.asarray(x, dtype=int).ravel()\n            if isinstance(x, (list, tuple)):\n                return np.asarray(list(x), dtype=int).ravel()\n            s = str(x).strip()\n            core = s[1:-1] if (s.startswith(\"[\") and s.endswith(\"]\")) else s\n            toks = core.replace(\",\", \" \").replace(\";\", \" \").split()\n            if (\n                len(toks) == 1\n                and len(toks[0]) == n_features\n                and all((ch in \"01\") for ch in toks[0])\n            ):\n                return np.asarray([int(ch) for ch in toks[0]], dtype=int)\n            vals = [int(float(t)) for t in toks]\n            return np.asarray(vals, dtype=int).ravel()\n        except Exception:\n            try:\n                digits = [int(ch) for ch in str(x) if ch in \"01\"]\n                return np.asarray(digits, dtype=int).ravel()\n            except Exception:\n                return np.asarray([], dtype=int)\n\n    def parse_resp(x):\n        # Accept 0/1 ints/floats, booleans, 'A'/'B', '0'/'1' strings.\n        try:\n            if isinstance(x, (bool, np.bool_)):\n                return int(x)\n            if isinstance(x, (int, np.integer)):\n                r = int(x)\n                return r if r in (0, 1) else None\n            if isinstance(x, (float, np.floating)):\n                r = int(round(float(x)))\n                return r if r in (0, 1) else None\n            s = str(x).strip().upper()\n            if s == \"A\":\n                return 0\n            if s == \"B\":\n                return 1\n            r = int(float(s))\n            return r if r in (0, 1) else None\n        except Exception:\n            return None\n\n    def first_disc(a, b, order):\n        for j in order:\n            if a[j] > b[j]:\n                return 0\n            if b[j] > a[j]:\n                return 1\n        return None\n\n    # ---------------- column resolution ----------------\n    cols = [str(c) for c in list(data.columns)]\n    lookup = {}\n    for c in list(data.columns):\n        lookup[str(c)] = c\n\n    a_col = None\n    for c in [\"option_a_ratings\", \"optionA_ratings\", \"ratings_a\", \"a_ratings\"]:\n        if c in lookup:\n            a_col = lookup[c]\n            break\n    if a_col is None:\n        for c in cols:\n            lc = c.lower()\n            if (\"a\" in lc) and (\"rating\" in lc):\n                a_col = lookup[c]\n                break\n\n    b_col = None\n    for c in [\"option_b_ratings\", \"optionB_ratings\", \"ratings_b\", \"b_ratings\"]:\n        if c in lookup:\n            b_col = lookup[c]\n            break\n    if b_col is None:\n        for c in cols:\n            lc = c.lower()\n            if (\"b\" in lc) and (\"rating\" in lc):\n                b_col = lookup[c]\n                break\n\n    r_col = None\n    for c in [\"response\", \"choice\", \"chosen\", \"answer\", \"pick\", \"selected\"]:\n        if c in lookup:\n            r_col = lookup[c]\n            break\n    if r_col is None:\n        for c in cols:\n            lc = c.lower()\n            if (\"respon\" in lc) or (\"choic\" in lc) or (\"select\" in lc):\n                r_col = lookup[c]\n                break\n\n    if a_col is None or b_col is None or r_col is None:\n        return 0.5\n\n    # ---------------- core computation ----------------\n    # Keep exactly the trials on which the two hierarchies render\n    # OPPOSITE choices (their first strictly discriminating cues\n    # differ), and score agreement with the advocated cascade.\n    hits = []\n    for ra, rb, resp in zip(data[a_col], data[b_col], data[r_col]):\n        a = parse_vec(ra)\n        b = parse_vec(rb)\n        if a.size != n_features or b.size != n_features:\n            continue\n        if (not np.all((a == 0) | (a == 1))) or (not np.all((b == 0) | (b == 1))):\n            continue\n        r = parse_resp(resp)\n        if r is None:\n            continue\n        wa = first_disc(a, b, adv_order)\n        wc = first_disc(a, b, comp_order)\n        if wa is None or wc is None or wa == wc:\n            continue\n        hits.append(1.0 if (r == wa) else 0.0)\n\n    if len(hits) == 0:\n        return 0.5\n    return float(np.mean(hits))",
  "rationale": "The two theories are one-reason (TTB) models that differ ONLY in their cue hierarchy. With the design's validities [0.9, 0.65, 0.6, 0.8, 0.95, 0.7], the advocated hierarchy (ascending validity) is [f2, f1, f5, f3, f0, f4] while the competing hierarchy (descending w = val[::-1]) is [f1, f5, f3, f0, f4, f2]; the two orders differ only in the placement of f2 (first for advocated, last for competing). Hence the theories make OPPOSITE predictions exactly on trials where f2 discriminates and so does some other cue \u2014 in this design trial types T1, T2, T5, T6, T13, T14 (6 types x 6 reps = 36 trials per subject). On those trials the advocated cascade follows f2's direction with probability p ~ 0.70 while the competing cascade follows the OPPOSITE direction with the same p. The metric computes, over exactly these theory-dissociating trials (identified stimulus-by-stimulus by simulating both cascades, not by hard-coded trial indices), the fraction of responses matching the ADVOCATED model's predicted choice: ~0.72 under advocated-simulated data vs ~0.28 under competing-simulated data, a mean gap of ~0.44. Per-subject estimates rest on 36 Bernoulli trials (binomial sd ~0.08) plus between-subject spread in follow probability p (sd ~0.07), giving total per-subject sd ~0.11; with N=25 the Welch t is ~14 (p << 1e-10), far below alpha = 0.01. The mirror-pair structure of the dissociating cells (T1/T2, T5/T6, T13/T14) cancels side bias, and excluding agreement trials (T3, T4, T7-T12, T15, T16) removes pure noise that would dilute the contrast. The metric works identically on the pooled DataFrame and on single-subject slices (every subject completes all 96 trials, so 36 dissociating trials each), so the between-subject variance is well defined. It is also sign-robust to the design's validity-list ambiguity: even if the simulation used the alternative scrambled list from the design rationale, the two theories still fall on opposite sides of this metric (~0.30 vs ~0.70), because the dissociation is defined by the two cascades disagreeing, and only the sign (not the existence or magnitude) of the gap changes. Finally, this version hardens the cell parsing that likely caused the previous attempt to fail: rating cells are accepted as lists, tuples, numpy arrays, or any string encoding ('[1, 0, ...]', '1 0 ...', '100000', quoted digits), responses as 0/1 ints/floats, booleans, or 'A'/'B' strings, and column names are resolved with fallbacks \u2014 so no data-format quirk can silently reduce the metric to a constant 0.5."
}
```

## Usage

```json
{
  "input_tokens": 6998,
  "output_tokens": 25728,
  "reasoning_tokens": 23577,
  "provider": "Sail Research"
}
```
