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
**Validities (n_features=5):** [0.9, 0.86, 0.84, 0.6, 0.52]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 6: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 9: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 10: A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  trial 11: A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 12: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  trial 14: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 16: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** I advocate PWEA (validity-weighted, margin-graded signed integration) against FC-LEX (validity-blind flag counting with a single lexicographic tiebreak). Both models agree that a '1' is a warning flag, so raw direction is often non-diagnostic; the decisive tests are (a) FC-LEX's magnitude invariance, (b) its composition invariance, and (c) the fact that FC-LEX must use ONE subject-level tie_bias, expressed in the frame of the top discriminating expert, on every equal-count pair.

Validities (0.90, 0.86, 0.84, 0.60, 0.52) are deliberately clustered high-high-high / low-low. At gamma=1 the mean-1-normalised PWEA weights are w=[1.639, 1.475, 1.393, 0.410, 0.082] (a 20:1 top-to-bottom ratio); at gamma=0.1 they flatten to ~[1.09, 1.08, 1.07, 0.95, 0.81]. Crucially the *three-cue clustering* means several of my diagnostic contrasts survive the entire admissible gamma range, so they are not just gamma-identification items.

CLASS 1 - THE TIEBREAK-FRAME CONTRADICTION (base pairs 1-3, the novel core of this design). All three have EQUAL flag counts, so FC-LEX's counting rule is silent and it must fall back on the highest discriminating expert, giving p(option holding the top-cue '1') = tie_bias, the SAME number on all three pairs for a given subject.
  P1: [1,0,0,0,1] vs [0,1,1,0,0]. Top discriminating expert is #1, owned by A. But A's weighted mass (1.639+0.082=1.72) is SMALLER than B's (1.475+1.393=2.87), so PWEA chooses A. This inequality holds for every gamma in [0.1,1.15] (at gamma=0.1: 1.70 vs 1.93; at gamma=1.15: 0.80 vs 1.32 in raw units), so it is parameter-robust.
  P2: [1,0,0,0,0] vs [0,0,0,0,1]. Top discriminating expert is again #1, again owned by A, but now A carries the larger mass (1.639 vs 0.082), so PWEA chooses B decisively.
  Therefore PWEA predicts OPPOSITE directions on P1 and P2 while FC-LEX is mathematically forced to the identical p(A)=tie_bias on both. No FC-LEX parameter setting can produce this reversal; it is the single cleanest falsification test in the design.
  P3: [0,1,0,0,0] vs [0,0,1,0,0]. Weighted masses are 1.475 vs 1.393 - essentially tied for all gamma - so PWEA is pinned at ~50/50 for any beta and |s|, whereas FC-LEX descends to expert #2 (owner A) and predicts p(A)=tie_bias, which may be as extreme as 0.10 or 0.70. Near-chance responding here combined with decisive, oppositely-signed responding on P1 and P2 is uniquely PWEA.

CLASS 2 - COMPOSITION INVARIANCE (P4 vs P5). Both are 0-flags-versus-1-flag, so FC-LEX predicts identical accuracy (validity-blind counting). PWEA's margin is 1.639 (flag on the 90% expert) versus 0.082 (flag on the 52% expert) at gamma=1 - a 20:1 evidence ratio, i.e. near-deterministic versus near-chance. Direction is the same for both models here, so a reliable gradient cannot be attributed to any strategy reversal.

CLASS 3 - MARGIN LADDER / PLATEAU TEST (P5, P4, P6: margins 0.08, 1.64, 5.00 with the count difference always in the same direction). FC-LEX predicts one flat plateau accuracy (1-eps)*sigmoid(beta)+eps/2 across all three; PWEA must produce a monotone psychometric function. P6 ([0,0,0,0,0] vs [1,1,1,1,1]) also anchors beta*|s| and epsilon for PWEA and beta/epsilon for FC-LEX.

CLASS 4 - COUNT-vs-WEIGHT REVERSAL (P7: [1,0,0,0,0] vs [0,0,0,1,1]; P8: [1,1,0,0,0] vs [0,0,1,1,1]). Here MORE flags coincide with LESS weighted mass (1.639 vs 0.492; 3.114 vs 1.885 at gamma=1). FC-LEX must choose the fewer-flag option A at its flat rate; PWEA chooses B for every gamma above about 0.35 and agrees with FC-LEX only as gamma->0.1. Any reliable preference for the many-flag option is impossible for FC-LEX at any parameter setting, and the graded difference between P7 and P8 further pins gamma.

All 8 base pairs appear in both A/B mirror orders (16 unique pairs, K=6 -> 96 trials), so side bias cancels exactly.

PREDICTIONS. PWEA: A chosen on P1 but B chosen on P2 (a within-subject sign flip that FC-LEX's single tie_bias cannot express); chance on P3; steeply graded accuracy across P5 < P4 < P6; a large P4-vs-P5 composition gap; reversals toward the many-flag option on P7-P8 for moderate/high gamma. FC-LEX: one flat above-chance rate toward the fewer-flag option on P4-P8 irrespective of margin size or which expert carries the flag, and the SAME, magnitude-invariant p(top-cue-flag option)=tie_bias on all three equal-count pairs P1-P3.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** **Polarised Weighted Evidence Accumulation (PWEA).**

People do NOT stop at the first discriminating cue. On every trial they integrate *all* expert ratings for each option into a single scalar impression, and then choose between the two impressions with a graded (margin-sensitive) rule.

Three claims:

1. **Compensatory integration with validity-graded weights.** Each cue j contributes w_j = ((v_j - 0.5)/0.5)^gamma to the option's score, where the weights are rescaled to have mean 1 so that the total evidence scale grows with the number of cues. The steepness exponent gamma is a free, subject-level parameter: gamma -> 0 gives equal weighting (pure tallying), gamma ~ 1 gives classic validity-proportional weighting, and large gamma approaches a lexicographic, Take-The-Best-like regime in which the top cue can outvote all the others. The family therefore *nests* both incumbent heuristics as special cases rather than opposing them.

2. **Polarity of endorsement is a free interpretive parameter s.** A binary rating of 1 is not intrinsically 'good'. It is a symbol whose desirability sign the subject must infer from the instructions/response mapping. In this task population the sign is systematically *negative*: an expert '1' is read as a flag/warning (or the A/B response mapping is inverted), so the option carrying the greater weighted mass of 1s is *rejected*. score_i = s * sum_j w_j x_ij with s < 0.

3. **Graded choice on the evidence margin, with genuinely heterogeneous evidence sensitivity.** Choice is a softmax over the two scores with inverse temperature beta, so the probability of choosing an option increases smoothly with the *magnitude* of the weighted difference (a strict prediction that one-reason stopping rules cannot make), plus an independent attention-lapse epsilon that replaces the choice with a coin flip. Exact ties in weighted score produce exactly 50/50. Critically, the effective evidence sensitivity k = beta*|s| is only *moderate* in this population and varies widely across subjects: many subjects are near-threshold on small-margin pairs, so their choices on narrow-margin conflict items are only weakly above chance, whereas the same subjects are near-deterministic on wide-margin (dominance-like) items. This within-family heterogeneity in k and gamma — not lapse alone — is what produces the between-subject dispersion of the data and keeps the pooled conflict-hit rate well below ceiling.

Because s < 0 and gamma stays in the compensatory range, the model reproduces both signature results with one mechanism: in conflict pairs the option carrying the single highest-validity 1 also carries the *smallest* weighted mass of 1s, so it is chosen (looking exactly like TTB), whereas in agreement/dominance pairs the option that both TTB and tallying favour carries the *largest* mass of 1s and is therefore rejected (looking anti-TTB). Because Exp-1 conflict margins are small and Exp-2 conflict/dominance margins are large, moderate k depresses the Exp-1 hit rate while leaving the Exp-2 contrast saturated — exactly the asymmetry the human data show.

**Parameters:**
- polarity: [-1.0, -0.35]
- gamma: [0.1, 1.15]
- beta: [2.0, 9.0]
- epsilon: [0.10, 0.55]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---------- 1. parse the stimulus into two rating vectors ----------
    a = b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            # flat concatenation of the two options
            if stim.size % 2 != 0:
                return np.ones(2) / 2.0
            half = stim.size // 2
            a, b = stim[:half], stim[half:]
        else:
            stim = stim.reshape(stim.shape[0], -1)
            if stim.shape[0] < 2:
                return np.ones(2) / 2.0
            a, b = stim[0], stim[1]

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or a.size != b.size:
        return np.ones(2) / 2.0
    n_features = a.size

    # ---------- 2. validity-derived weights ----------
    val = parameters.get("validities", None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = np.array([])
    if val.size != n_features:
        # graceful fallback: linearly descending validities
        if n_features == 1:
            val = np.array([0.9])
        else:
            val = np.linspace(0.9, 0.55, n_features)

    gamma = float(parameters["gamma"])
    r = np.clip((val - 0.5) / 0.5, 1e-6, 1.0)          # normalised validity strength
    w = np.power(r, max(gamma, 0.0))
    m = float(np.mean(w))
    if not np.isfinite(m) or m <= 0.0:
        w = np.ones(n_features)
    else:
        w = w / m                                       # mean weight == 1 (scale grows with n cues)

    # ---------- 3. signed compensatory integration ----------
    s = float(parameters["polarity"])                   # negative: a '1' counts AGAINST the option
    score_a = s * float(np.dot(w, a))
    score_b = s * float(np.dot(w, b))
    scores = np.array([score_a, score_b], dtype=float)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # ---------- 4. margin-graded softmax + lapse ----------
    z = beta * scores
    z = z - np.max(z)                                    # numerically stable
    e = np.exp(z)
    p_core = e / e.sum()
    if not np.all(np.isfinite(p_core)):
        p_core = np.ones(2) / 2.0

    p = (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        probs = np.ones(len(probs)) / len(probs)
    else:
        probs = probs / total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** **Flag Counting with Lexicographic Tiebreak (FC-LEX).**

Subjects in this task do not read a binary expert rating of '1' as an endorsement; they read it as a *warning flag*. And, crucially, they do not weight those flags by the validities they were told about in the instructions. On every trial the subject performs a single, extremely cheap operation: count how many 1s each option carries, and reject the option carrying more flags.

Three claims:

1. **Negative polarity.** A '1' is a strike against an option (a flag/complaint/warning), so an option's *badness* is monotone in its number of 1s. The option with the strictly smaller flag count is chosen.

2. **Validity-blind, magnitude-blind counting.** All experts are counted equally: the validities are acknowledged in the instructions but are not converted into cue weights. Furthermore, only the *sign* of the count difference matters. Choice probability is therefore a single flat accuracy h = (1-eps)*sigmoid(beta) + eps/2 on every trial where the counts differ, regardless of whether the difference is 1 flag or 5 flags, and regardless of which experts contributed the flags. There is no psychometric function over a validity-weighted margin: a 1-vs-0 comparison and a 1-vs-5 comparison produce the same accuracy.

3. **Lexicographic tiebreak on equal counts.** When the two options carry exactly the same number of flags the counting rule is silent. The subject then falls back to a one-reason rule: locate the highest-validity expert who discriminates and let that single expert decide. Because the subject is unsure whether that expert's '1' should attract or repel (the counting rule just told them 1s are bad, but the tiebreak is invoked precisely because badness is balanced), the direction of this tiebreak is a free subject-level bias tie_bias, which sits near chance and is mildly repelled-by-1 on average. Equal-count pairs therefore yield near-chance performance, not the strongly graded reversals a weighted-mass model predicts.

The theory is deliberately non-compensatory in *magnitude* while being fully compensatory in *counting*: it agrees with a weighted-evidence account on every design tested so far (because in those designs the option with the single top-validity 1 also happens to carry the fewest flags), but it diverges sharply on equal-count pairs such as [1,0,0,0,0] vs [0,0,0,0,1], on pairs where more 1s coincides with smaller weighted mass, and on margin ladders, where FC-LEX predicts a flat accuracy plateau conditional on the sign of the count difference.

**Parameters:**
- beta: [1.3, 2.7]
- epsilon: [0.05, 0.25]
- tie_bias: [0.10, 0.70]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---------- 1. parse the stimulus into two rating vectors ----------
    a = b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            if stim.size % 2 != 0:
                return np.ones(2) / 2.0
            half = stim.size // 2
            a, b = stim[:half], stim[half:]
        else:
            stim = stim.reshape(stim.shape[0], -1)
            if stim.shape[0] < 2:
                return np.ones(2) / 2.0
            a, b = stim[0], stim[1]

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or a.size != b.size:
        return np.ones(2) / 2.0
    n_features = a.size

    beta = float(parameters["beta"])
    eps = float(parameters["epsilon"])
    tie_bias = float(parameters["tie_bias"])
    eps = min(max(eps, 0.0), 1.0)
    tie_bias = min(max(tie_bias, 0.0), 1.0)

    # ---------- 2. validity-blind flag count (negative polarity) ----------
    count_a = float(np.sum(a))
    count_b = float(np.sum(b))
    d = count_a - count_b   # positive => A carries MORE warning flags => reject A

    if abs(d) > 1e-9:
        # Sign-only rule: magnitude of the count difference is ignored.
        winner = 1 if d > 0 else 0   # fewer flags wins
        s = 1.0 / (1.0 + np.exp(-beta))   # sigmoid(beta * sign) for the winner
        p_core = np.zeros(2, dtype=float)
        p_core[winner] = s
        p_core[1 - winner] = 1.0 - s
    else:
        # ---------- 3. lexicographic tiebreak on equal flag counts ----------
        val = parameters.get("validities", None)
        try:
            val = np.asarray(val, dtype=float).ravel()
        except Exception:
            val = np.array([])
        if val.size != n_features:
            if n_features == 1:
                val = np.array([0.9])
            else:
                val = np.linspace(0.9, 0.55, n_features)

        order = np.argsort(-val, kind="stable")
        top1_owner = None   # index of the option carrying the '1' on the top discriminating cue
        for j in order:
            if a[j] > b[j]:
                top1_owner = 0
                break
            if b[j] > a[j]:
                top1_owner = 1
                break

        if top1_owner is None:
            p_core = np.ones(2) / 2.0
        else:
            p_core = np.zeros(2, dtype=float)
            p_core[top1_owner] = tie_bias
            p_core[1 - top1_owner] = 1.0 - tie_bias

    if not np.all(np.isfinite(p_core)):
        p_core = np.ones(2) / 2.0

    # ---------- 4. attention lapse ----------
    p = (1.0 - eps) * p_core + eps * (np.ones(2) / 2.0)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        probs = np.ones(len(probs)) / len(probs)
    else:
        probs = probs / total
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    tgt1 = tuple(sorted([(1, 0, 0, 0, 1), (0, 1, 1, 0, 0)]))\n    tgt2 = tuple(sorted([(1, 0, 0, 0, 0), (0, 0, 0, 0, 1)]))\n\n    x1 = []\n    x2 = []\n\n    for _, row in data.iterrows():\n        try:\n            a = tuple(int(v) for v in row[\"option_a_ratings\"])\n            b = tuple(int(v) for v in row[\"option_b_ratings\"])\n            r = int(row[\"response\"])\n        except Exception:\n            continue\n        if len(a) != 5 or len(b) != 5:\n            continue\n        key = tuple(sorted([a, b]))\n        if key != tgt1 and key != tgt2:\n            continue\n        chosen = a if r == 0 else b\n        other = b if r == 0 else a\n        # feature 0 (validity 0.90) must discriminate on these pairs\n        if chosen[0] == other[0]:\n            continue\n        hit = 1.0 if chosen[0] == 1 else 0.0\n        if key == tgt1:\n            x1.append(hit)\n        else:\n            x2.append(hit)\n\n    if len(x1) == 0 or len(x2) == 0:\n        return 0.0\n    return float(np.mean(x1) - np.mean(x2))",
  "rationale": "Both critical pairs are EQUAL-flag-count pairs in which the single highest-validity expert (v=0.90) discriminates and is owned by the same side within each pair. Define x = 1 if the subject picks the option carrying the top expert's '1'.\n\nPair P1 = {[1,0,0,0,1] vs [0,1,1,0,0]}: the top-cue owner also carries the SMALLEST validity-weighted mass of flags (w1+w5 < w2+w3 for every admissible gamma), so PWEA (negative polarity, weighted integration) picks it -> x high.\nPair P2 = {[1,0,0,0,0] vs [0,0,0,0,1]}: the top-cue owner now carries the LARGEST weighted mass (w1 >> w5), so PWEA rejects it -> x low.\n\nHence PWEA predicts a large positive value of mean(x|P1) - mean(x|P2) (roughly 0.4-0.9 depending on gamma and beta*|s|).\n\nFC-LEX cannot express this at all: on both pairs the counts are tied, so the lexicographic fallback fires with the SAME top discriminating expert owned by the SAME (top-cue) option, yielding p(x) = tie_bias identically on P1 and P2. Its expected metric value is exactly 0 for every parameter setting. Crucially, taking the difference cancels the subject-level tie_bias entirely, so FC-LEX's between-subject variance reduces to pure binomial noise over 12+12 trials, keeping the competing theory's distribution tight around zero. Mirror (A/B swapped) presentations are pooled automatically because x is defined by cue ownership rather than by screen side, so any side bias cancels."
}
```

## Usage

```json
{
  "input_tokens": 10324,
  "output_tokens": 6916
}
```
