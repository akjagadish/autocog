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
**Validities (n_features=5):** [0.97, 0.66, 0.6, 0.56, 0.52]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 4: A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 6: A=[0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 8: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 9: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 10: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 11: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 12: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 13: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 14: A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  trial 15: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 16: A=[0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]

**Rationale:** I advocate FC-LEX (validity-blind flag counting with a lexicographic tiebreak) against PWEA (validity-weighted, margin-graded signed integration). Both models agree that a '1' is a warning flag, so direction alone is often non-diagnostic; the decisive dissociations are (a) MAGNITUDE INVARIANCE and (b) COMPOSITION INVARIANCE of the count rule, plus (c) the equal-count class.

Validities are deliberately skewed: (0.97, 0.66, 0.60, 0.56, 0.52). At gamma=1 the mean-1-normalised PWEA weights are w=[2.90, 0.99, 0.62, 0.37, 0.12] (a ~24:1 top-to-bottom ratio); at gamma=0.1 they are nearly flat (~[1.19,1.04,1.02,1.00,0.94]). This skew makes PWEA's weighted mass diverge sharply from raw flag counts for moderate/high gamma, while low gamma collapses PWEA onto tallying (which is direction-identical to counting) - so the design simultaneously separates the families and identifies gamma.

CLASS 1 - FLAT PLATEAU / MARGIN LADDER (pairs 1-2, 5-6, 7-8). A=[0,0,0,0,0] against B with 1, 4, or 5 top-loaded flags. All three have a count difference in the same direction, so FC-LEX predicts EXACTLY the same accuracy (1-eps)*sigmoid(beta)+eps/2 on all of them - no parameter of FC-LEX can make them differ. PWEA's weighted margins are 2.90, 1.80, 5.00 at gamma=1 and 1.19, 4.00, 5.19 at gamma=0.1: for every admissible gamma the [0,1,1,1,1] and [1,1,1,1,1] items carry a much larger margin than a single low-weight flag, so PWEA must show a monotone psychometric gradient.

CLASS 2 - COMPOSITION INVARIANCE (pairs 1-2 vs 3-4). Both are 0-flags-versus-1-flag, but the flag sits on the 97% expert in one case and on the 52% expert in the other. FC-LEX: identical accuracy (counting is validity-blind). PWEA at gamma=1: margin 2.90 versus 0.12, i.e. near-deterministic versus essentially chance - a ~24:1 evidence ratio on trials that FC-LEX must treat as identical. This is the single cleanest quantitative test in the design, because direction is the same for both models, so no strategy-reversal confound can explain a gradient.

CLASS 3 - COUNT-vs-WEIGHT REVERSAL (pairs 9-10: [1,0,0,0,0] vs [0,1,1,1,1]; pairs 11-12: [1,1,0,0,0] vs [0,0,1,1,1]). Here more flags coincides with LESS weighted mass. FC-LEX chooses the fewer-flag option (A in 9 and 11) at its usual flat rate. PWEA with gamma near 1 chooses the OPPOSITE option (masses 2.90 vs 1.80 and 3.89 vs 1.11), while with gamma near 0.1 it agrees with FC-LEX. A reliable preference for the many-flags option here is impossible for FC-LEX at any parameter setting and pins gamma for PWEA.

CLASS 4 - EQUAL-COUNT TIEBREAK (pairs 13-14: [1,0,0,0,0] vs [0,0,0,0,1]; pairs 15-16: [0,0,1,0,0] vs [0,0,0,1,0]). Counts are tied, so the counting rule is silent and FC-LEX falls back on the top discriminating expert with a subject-level bias tie_bias near chance - and, critically, the SAME p(top-flag option) in both pairs, because both are decided by whichever expert discriminates first. PWEA instead predicts a strongly graded reversal: margin 2.78 for pair 13 versus 0.25 for pair 15 at gamma=1 (a >10:1 ratio), collapsing toward chance for both as gamma->0. So Class 4 gives a within-subject magnitude contrast on trials where FC-LEX is committed to invariance.

All 8 base pairs appear in both A/B mirror orders (16 unique pairs, K=6 -> 96 trials), so side bias cancels exactly, and pairs 7-8 (dominance) plus pairs 1-2 provide clean identification of beta/epsilon for FC-LEX and of beta*|s|/epsilon for PWEA.

PREDICTIONS. FC-LEX: one flat above-chance accuracy toward the fewer-flag option across pairs 1-12 regardless of margin size (1 vs 4 vs 5 flags) and regardless of which expert carries the flag; near-chance, magnitude-invariant, subject-idiosyncratic responding on pairs 13-16. PWEA: monotone accuracy growth with weighted margin in Class 1, a large accuracy gap between top-cue and bottom-cue single-flag items in Class 2, direction reversals in Class 3 for moderate/high gamma, and a strongly graded (not invariant) split between pairs 13-14 and 15-16.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # design-time validities (fixed by the experiment)\n    val_default = np.array([0.97, 0.66, 0.60, 0.56, 0.52], dtype=float)\n\n    def weights(n):\n        if n == len(val_default):\n            v = val_default.copy()\n        elif n == 1:\n            v = np.array([0.9])\n        else:\n            v = np.linspace(0.9, 0.55, n)\n        r = np.clip((v - 0.5) / 0.5, 1e-6, 1.0)\n        return r / float(np.mean(r))\n\n    hits_strong = []\n    hits_any = []\n\n    for _, row in data.iterrows():\n        try:\n            a = np.asarray(row[\"option_a_ratings\"], dtype=float).ravel()\n            b = np.asarray(row[\"option_b_ratings\"], dtype=float).ravel()\n            resp = int(row[\"response\"])\n        except Exception:\n            continue\n        if a.size == 0 or a.size != b.size:\n            continue\n        w = weights(a.size)\n        ca = float(a.sum())\n        cb = float(b.sum())\n        if abs(ca - cb) < 1e-9:\n            continue  # equal-count pairs are not informative here\n        ma = float(np.dot(w, a))\n        mb = float(np.dot(w, b))\n        fewer = 0 if ca < cb else 1\n        # does the option with FEWER flags carry the LARGER validity-weighted mass?\n        conflict = (ma > mb) if fewer == 0 else (mb > ma)\n        if not conflict:\n            continue\n        hit = 1.0 if resp == fewer else 0.0\n        hits_any.append(hit)\n        if abs(ma - mb) > 1.5:\n            hits_strong.append(hit)\n\n    if len(hits_strong) > 0:\n        return float(np.mean(hits_strong))\n    if len(hits_any) > 0:\n        return float(np.mean(hits_any))\n    return float(\"nan\")\n",
  "rationale": "The decisive dissociation in this design is Class 3, the count-vs-weight reversal pairs 11/12 ([1,1,0,0,0] vs [0,0,1,1,1]). Both theories agree that a '1' is a warning flag, so no polarity confound is possible; they simply disagree about WHAT is counted. FC-LEX counts flags validity-blind and must therefore choose the two-flag option [1,1,0,0,0] at its single flat plateau accuracy h = (1-eps)*sigmoid(beta)+eps/2, which for the admissible ranges (beta 1.3-2.7, eps 0.05-0.25) lies tightly around 0.78-0.85 with very little between-subject spread (parameter-induced sd ~0.04 plus binomial noise over the 12 trials of that pair). PWEA, by contrast, integrates validity-weighted mass: with the deliberately skewed validities the mass of [1,1,0,0,0] is 3.89 versus 1.11 for [0,0,1,1,1] at gamma=1, and the ordering already reverses at gamma ~ 0.25, so for the large majority of the admissible gamma range PWEA must REJECT the fewer-flag option, pushing this proportion down toward the lapse floor (~0.16-0.30). Integrating over gamma in [0.1,1.15] gives an expected value near 0.27, i.e. roughly a 0.55 gap from FC-LEX's ~0.81, with per-subject variance dominated by binomial noise plus a modest gamma-driven component. The metric selects these trials generically (counts differ, but the fewer-flag option carries the larger mean-1-normalised weighted mass, with a large weighted margin) rather than by hard-coded row indices, and falls back to all count/weight-conflict trials if the strong-margin set is empty, so it is well defined on both the pooled frame and any single subject's 12-trial slice."
}
```

## Usage

```json
{
  "input_tokens": 10152,
  "output_tokens": 12642
}
```
