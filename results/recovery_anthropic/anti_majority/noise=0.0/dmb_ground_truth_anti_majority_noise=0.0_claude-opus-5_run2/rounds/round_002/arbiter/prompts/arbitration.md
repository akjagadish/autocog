# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
**Flag Counting with Lexicographic Tiebreak (FC-LEX).**

Subjects in this task do not read a binary expert rating of '1' as an endorsement; they read it as a *warning flag*. And, crucially, they do not weight those flags by the validities they were told about in the instructions. On every trial the subject performs a single, extremely cheap operation: count how many 1s each option carries, and reject the option carrying more flags.

Three claims:

1. **Negative polarity.** A '1' is a strike against an option (a flag/complaint/warning), so an option's *badness* is monotone in its number of 1s. The option with the strictly smaller flag count is chosen.

2. **Validity-blind, magnitude-blind counting.** All experts are counted equally: the validities are acknowledged in the instructions but are not converted into cue weights. Furthermore, only the *sign* of the count difference matters. Choice probability is therefore a single flat accuracy h = (1-eps)*sigmoid(beta) + eps/2 on every trial where the counts differ, regardless of whether the difference is 1 flag or 5 flags, and regardless of which experts contributed the flags. There is no psychometric function over a validity-weighted margin: a 1-vs-0 comparison and a 1-vs-5 comparison produce the same accuracy.

3. **Lexicographic tiebreak on equal counts.** When the two options carry exactly the same number of flags the counting rule is silent. The subject then falls back to a one-reason rule: locate the highest-validity expert who discriminates and let that single expert decide. Because the subject is unsure whether that expert's '1' should attract or repel (the counting rule just told them 1s are bad, but the tiebreak is invoked precisely because badness is balanced), the direction of this tiebreak is a free subject-level bias tie_bias, which sits near chance and is mildly repelled-by-1 on average. Equal-count pairs therefore yield near-chance performance, not the strongly graded reversals a weighted-mass model predicts.

The theory is deliberately non-compensatory in *magnitude* while being fully compensatory in *counting*: it agrees with a weighted-evidence account on every design tested so far (because in those designs the option with the single top-validity 1 also happens to carry the fewest flags), but it diverges sharply on equal-count pairs such as [1,0,0,0,0] vs [0,0,0,0,1], on pairs where more 1s coincides with smaller weighted mass, and on margin ladders, where FC-LEX predicts a flat accuracy plateau conditional on the sign of the count difference.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
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

## THEORY 2 — pi_3
**Polarised Weighted Evidence Accumulation (PWEA).**

People do NOT stop at the first discriminating cue. On every trial they integrate *all* expert ratings for each option into a single scalar impression, and then choose between the two impressions with a graded (margin-sensitive) rule.

Three claims:

1. **Compensatory integration with validity-graded weights.** Each cue j contributes w_j = ((v_j - 0.5)/0.5)^gamma to the option's score, where the weights are rescaled to have mean 1 so that the total evidence scale grows with the number of cues. The steepness exponent gamma is a free, subject-level parameter: gamma -> 0 gives equal weighting (pure tallying), gamma ~ 1 gives classic validity-proportional weighting, and large gamma approaches a lexicographic, Take-The-Best-like regime in which the top cue can outvote all the others. The family therefore *nests* both incumbent heuristics as special cases rather than opposing them.

2. **Polarity of endorsement is a free interpretive parameter s.** A binary rating of 1 is not intrinsically 'good'. It is a symbol whose desirability sign the subject must infer from the instructions/response mapping. In this task population the sign is systematically *negative*: an expert '1' is read as a flag/warning (or the A/B response mapping is inverted), so the option carrying the greater weighted mass of 1s is *rejected*. score_i = s * sum_j w_j x_ij with s < 0.

3. **Graded choice on the evidence margin, with genuinely heterogeneous evidence sensitivity.** Choice is a softmax over the two scores with inverse temperature beta, so the probability of choosing an option increases smoothly with the *magnitude* of the weighted difference (a strict prediction that one-reason stopping rules cannot make), plus an independent attention-lapse epsilon that replaces the choice with a coin flip. Exact ties in weighted score produce exactly 50/50. Critically, the effective evidence sensitivity k = beta*|s| is only *moderate* in this population and varies widely across subjects: many subjects are near-threshold on small-margin pairs, so their choices on narrow-margin conflict items are only weakly above chance, whereas the same subjects are near-deterministic on wide-margin (dominance-like) items. This within-family heterogeneity in k and gamma — not lapse alone — is what produces the between-subject dispersion of the data and keeps the pooled conflict-hit rate well below ceiling.

Because s < 0 and gamma stays in the compensatory range, the model reproduces both signature results with one mechanism: in conflict pairs the option carrying the single highest-validity 1 also carries the *smallest* weighted mass of 1s, so it is chosen (looking exactly like TTB), whereas in agreement/dominance pairs the option that both TTB and tallying favour carries the *largest* mass of 1s and is therefore rejected (looking anti-TTB). Because Exp-1 conflict margins are small and Exp-2 conflict/dominance margins are large, moderate k depresses the Exp-1 hit rate while leaving the Exp-2 contrast saturated — exactly the asymmetry the human data show.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
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

## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
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



### METRIC
Rationale:
The decisive dissociation in this design is Class 3, the count-vs-weight reversal pairs 11/12 ([1,1,0,0,0] vs [0,0,1,1,1]). Both theories agree that a '1' is a warning flag, so no polarity confound is possible; they simply disagree about WHAT is counted. FC-LEX counts flags validity-blind and must therefore choose the two-flag option [1,1,0,0,0] at its single flat plateau accuracy h = (1-eps)*sigmoid(beta)+eps/2, which for the admissible ranges (beta 1.3-2.7, eps 0.05-0.25) lies tightly around 0.78-0.85 with very little between-subject spread (parameter-induced sd ~0.04 plus binomial noise over the 12 trials of that pair). PWEA, by contrast, integrates validity-weighted mass: with the deliberately skewed validities the mass of [1,1,0,0,0] is 3.89 versus 1.11 for [0,0,1,1,1] at gamma=1, and the ordering already reverses at gamma ~ 0.25, so for the large majority of the admissible gamma range PWEA must REJECT the fewer-flag option, pushing this proportion down toward the lapse floor (~0.16-0.30). Integrating over gamma in [0.1,1.15] gives an expected value near 0.27, i.e. roughly a 0.55 gap from FC-LEX's ~0.81, with per-subject variance dominated by binomial noise plus a modest gamma-driven component. The metric selects these trials generically (counts differ, but the fewer-flag option carries the larger mean-1-normalised weighted mass, with a large weighted margin) rather than by hard-coded row indices, and falls back to all count/weight-conflict trials if the strong-margin set is empty, so it is well defined on both the pooled frame and any single subject's 12-trial slice.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # design-time validities (fixed by the experiment)
    val_default = np.array([0.97, 0.66, 0.60, 0.56, 0.52], dtype=float)

    def weights(n):
        if n == len(val_default):
            v = val_default.copy()
        elif n == 1:
            v = np.array([0.9])
        else:
            v = np.linspace(0.9, 0.55, n)
        r = np.clip((v - 0.5) / 0.5, 1e-6, 1.0)
        return r / float(np.mean(r))

    hits_strong = []
    hits_any = []

    for _, row in data.iterrows():
        try:
            a = np.asarray(row["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(row["option_b_ratings"], dtype=float).ravel()
            resp = int(row["response"])
        except Exception:
            continue
        if a.size == 0 or a.size != b.size:
            continue
        w = weights(a.size)
        ca = float(a.sum())
        cb = float(b.sum())
        if abs(ca - cb) < 1e-9:
            continue  # equal-count pairs are not informative here
        ma = float(np.dot(w, a))
        mb = float(np.dot(w, b))
        fewer = 0 if ca < cb else 1
        # does the option with FEWER flags carry the LARGER validity-weighted mass?
        conflict = (ma > mb) if fewer == 0 else (mb > ma)
        if not conflict:
            continue
        hit = 1.0 if resp == fewer else 0.0
        hits_any.append(hit)
        if abs(ma - mb) > 1.5:
            hits_strong.append(hit)

    if len(hits_strong) > 0:
        return float(np.mean(hits_strong))
    if len(hits_any) > 0:
        return float(np.mean(hits_any))
    return float("nan")


### RESULTS
- Predicted under pi_4 (simulated): 0.8367 (var=0.0105)
- Predicted under pi_3 (simulated): 0.2383 (var=0.0433)
- Observed on real data: 0.5167 (var=0.1333)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
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



### METRIC
Rationale:
Both critical pairs are EQUAL-flag-count pairs in which the single highest-validity expert (v=0.90) discriminates and is owned by the same side within each pair. Define x = 1 if the subject picks the option carrying the top expert's '1'.

Pair P1 = {[1,0,0,0,1] vs [0,1,1,0,0]}: the top-cue owner also carries the SMALLEST validity-weighted mass of flags (w1+w5 < w2+w3 for every admissible gamma), so PWEA (negative polarity, weighted integration) picks it -> x high.
Pair P2 = {[1,0,0,0,0] vs [0,0,0,0,1]}: the top-cue owner now carries the LARGEST weighted mass (w1 >> w5), so PWEA rejects it -> x low.

Hence PWEA predicts a large positive value of mean(x|P1) - mean(x|P2) (roughly 0.4-0.9 depending on gamma and beta*|s|).

FC-LEX cannot express this at all: on both pairs the counts are tied, so the lexicographic fallback fires with the SAME top discriminating expert owned by the SAME (top-cue) option, yielding p(x) = tie_bias identically on P1 and P2. Its expected metric value is exactly 0 for every parameter setting. Crucially, taking the difference cancels the subject-level tie_bias entirely, so FC-LEX's between-subject variance reduces to pure binomial noise over 12+12 trials, keeping the competing theory's distribution tight around zero. Mirror (A/B swapped) presentations are pooled automatically because x is defined by cue ownership rather than by screen side, so any side bias cancels.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    tgt1 = tuple(sorted([(1, 0, 0, 0, 1), (0, 1, 1, 0, 0)]))
    tgt2 = tuple(sorted([(1, 0, 0, 0, 0), (0, 0, 0, 0, 1)]))

    x1 = []
    x2 = []

    for _, row in data.iterrows():
        try:
            a = tuple(int(v) for v in row["option_a_ratings"])
            b = tuple(int(v) for v in row["option_b_ratings"])
            r = int(row["response"])
        except Exception:
            continue
        if len(a) != 5 or len(b) != 5:
            continue
        key = tuple(sorted([a, b]))
        if key != tgt1 and key != tgt2:
            continue
        chosen = a if r == 0 else b
        other = b if r == 0 else a
        # feature 0 (validity 0.90) must discriminate on these pairs
        if chosen[0] == other[0]:
            continue
        hit = 1.0 if chosen[0] == 1 else 0.0
        if key == tgt1:
            x1.append(hit)
        else:
            x2.append(hit)

    if len(x1) == 0 or len(x2) == 0:
        return 0.0
    return float(np.mean(x1) - np.mean(x2))

### RESULTS
- Predicted under pi_4 (simulated): -0.0117 (var=0.0319)
- Predicted under pi_3 (simulated): 0.5383 (var=0.0564)
- Observed on real data: 0.0667 (var=0.0622)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    def ttb_winner(a, b):
        # features are ordered by descending validity (index 0 = 0.9, ... index 4 = 0.55)
        for j in range(len(a)):
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return -1

    def tally_winner(a, b):
        a_wins = sum(1 for x, y in zip(a, b) if x > y)
        b_wins = sum(1 for x, y in zip(a, b) if y > x)
        if a_wins > b_wins:
            return 0
        if b_wins > a_wins:
            return 1
        return -1

    hits = 0
    n = 0
    for _, row in data.iterrows():
        try:
            a = [int(v) for v in row["option_a_ratings"]]
            b = [int(v) for v in row["option_b_ratings"]]
        except Exception:
            continue
        if len(a) != len(b) or len(a) == 0:
            continue
        w_ttb = ttb_winner(a, b)
        w_tal = tally_winner(a, b)
        # keep only trials where the two heuristics make opposite, well-defined predictions
        if w_ttb < 0 or w_tal < 0 or w_ttb == w_tal:
            continue
        resp = row["response"]
        if resp is None or (isinstance(resp, float) and np.isnan(resp)):
            continue
        n += 1
        if int(resp) == w_ttb:
            hits += 1

    if n == 0:
        return 0.5
    return float(hits) / float(n)
```

**Observed (real) value:** 0.7744 (var=0.0262)
**Predicted under pi_4:** 0.8083 (var=0.0076)
**Predicted under pi_3:** 0.7611 (var=0.0116)

### Experiment 4
**Design**
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    agree_hits = []
    conflict_hits = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        diff = a - b
        nz = np.nonzero(diff)[0]
        if nz.size == 0:
            continue
        # features are listed in descending validity order in this design
        ttb_choice = 0 if diff[nz[0]] > 0 else 1
        margin = int(np.sum(a > b) - np.sum(b > a))
        if margin == 0:
            continue
        tally_choice = 0 if margin > 0 else 1
        try:
            resp = int(row['response'])
        except (TypeError, ValueError):
            continue
        chose_ttb = 1.0 if resp == ttb_choice else 0.0
        if tally_choice == ttb_choice and abs(margin) >= 3:
            agree_hits.append(chose_ttb)
        elif tally_choice != ttb_choice and abs(margin) >= 2:
            conflict_hits.append(chose_ttb)
    if len(agree_hits) == 0 or len(conflict_hits) == 0:
        return 0.0
    return float(np.mean(agree_hits) - np.mean(conflict_hits))
```

**Observed (real) value:** -0.6339 (var=0.0508)
**Predicted under pi_4:** -0.6325 (var=0.0187)
**Predicted under pi_3:** -0.6858 (var=0.0254)

### Experiment 5
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def key(x):
        return ''.join([str(int(v)) for v in list(x)])

    # Class I / II 'polarity-reversal' pairs: the option favoured by the
    # top-validity expert ALSO carries the larger weighted mass of 1s.
    rev = {
        ('11111', '00000'), ('00000', '11111'),
        ('11100', '00011'), ('00011', '11100'),
        ('10000', '00000'), ('00000', '10000'),
        ('10000', '00010'), ('00010', '10000'),
    }
    # Class IV 'agreement' pairs: top-cue option carries the SMALLER mass of 1s.
    agr = {
        ('10000', '01111'), ('01111', '10000'),
        ('10000', '01100'), ('01100', '10000'),
    }

    hits_rev = []
    hits_agr = []
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        try:
            ka = key(a)
            kb = key(b)
        except Exception:
            continue
        pair = (ka, kb)
        in_rev = pair in rev
        in_agr = pair in agr
        if not (in_rev or in_agr):
            continue
        aa = int(list(a)[0])
        bb = int(list(b)[0])
        if aa == bb:
            continue
        winner = 0 if aa == 1 else 1
        hit = 1.0 if int(r) == winner else 0.0
        if in_rev:
            hits_rev.append(hit)
        else:
            hits_agr.append(hit)

    if len(hits_rev) == 0 or len(hits_agr) == 0:
        return 0.0
    return float(np.mean(hits_rev) - np.mean(hits_agr))
```

**Observed (real) value:** -0.5767 (var=0.0708)
**Predicted under pi_4:** -0.5608 (var=0.0168)
**Predicted under pi_3:** -0.6121 (var=0.0311)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def as_str(x):
        return ''.join([str(int(v)) for v in list(x)])

    dom_hits = []
    conf_hits = []

    for _, row in data.iterrows():
        a = [int(v) for v in list(row['option_a_ratings'])]
        b = [int(v) for v in list(row['option_b_ratings'])]
        if len(a) != len(b) or len(a) == 0:
            continue
        # Cue order = feature index order (validities are strictly descending by design)
        winner = None
        for j in range(len(a)):
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            continue
        try:
            resp = int(row['response'])
        except Exception:
            continue
        hit = 1.0 if resp == winner else 0.0

        sa = as_str(a)
        sb = as_str(b)
        key = tuple(sorted([sa, sb]))

        # DOMINANCE base pair: [1,0,0,0,0] vs [0,0,0,0,0]  (trials 1 and 9)
        if key == ('00000', '10000'):
            dom_hits.append(hit)
        # CONFLICT base pair: [1,0,0,0,0] vs [0,1,1,1,1]  (trials 4 and 12)
        elif key == ('01111', '10000'):
            conf_hits.append(hit)

    if len(dom_hits) == 0 or len(conf_hits) == 0:
        return 0.0

    return float(np.mean(conf_hits) - np.mean(dom_hits))
```

**Observed (real) value:** 0.6600 (var=0.0633)
**Predicted under pi_4:** 0.6133 (var=0.0274)
**Predicted under pi_3:** 0.6550 (var=0.0311)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the six experiments the data are dominated by one robust regularity: a rating of '1' is read as a warning flag and the option carrying FEWER flags is chosen. Both theories encode that regularity, which is why Exps 3-6 (all of which reduce to 'fewer flags wins' contrasts) are fit almost equally well by both (pi_4: 0.808/-0.633/-0.561/0.613; pi_3: 0.761/-0.686/-0.612/0.655; observed: 0.774/-0.634/-0.577/0.660). These four experiments are therefore essentially non-diagnostic between FC-LEX and PWEA.\n\nThe two new experiments are diagnostic, and they cut in opposite directions.\n\nExp 2 (equal-count tiebreak pairs P1=[1,0,0,0,1] vs [0,1,1,0,0] and P2=[1,0,0,0,0] vs [0,0,0,0,1]): PWEA is committed to a large positive difference (+0.54) because the weighted mass of flags reverses between the two pairs; FC-LEX is mathematically pinned at 0 because both pairs are decided by the same top discriminating expert owned by the same side. Observed = 0.067 (var 0.062), i.e. statistically indistinguishable from zero. This is a clean, strong falsification of graded validity-weighted integration on equal-count pairs: subjects show essentially NO sensitivity to which experts carry the flags once the raw counts are tied. pi_4 wins decisively here.\n\nExp 1 (count-vs-weight reversal on [1,1,0,0,0] vs [0,0,1,1,1], validities 0.97/0.66/0.60/0.56/0.52): FC-LEX predicts its flat plateau 0.84, PWEA predicts a reversal to 0.24; observed = 0.517 with a very large between-subject variance (0.133, far above the ~0.02 binomial floor over 12 trials). So neither theory is right: pooled performance is exactly at chance and subject-level performance is close to bimodal. FC-LEX's central commitment \\u2014 that a count difference of one flag yields the SAME accuracy whether the comparison is 0-vs-1 or 2-vs-3 \\u2014 is contradicted, because in Exp 6 the 0-vs-1 dominance pair is decided reliably (dominance hit ~0.15-0.17, conflict ~0.78, difference 0.66) while the 2-vs-3 pair here is at chance.\n\nNet: pi_4 (FC-LEX) explains 5 of 6 experiments and fails only on the magnitude/ratio-invariance prediction of Exp 1; pi_3 (PWEA) fails Exp 1 in the opposite direction AND fails Exp 2 badly. PWEA is the weaker, and its core mechanism (graded weighted-mass integration) is directly refuted by Exp 2. Rather than re-fitting PWEA's parameters (no gamma/beta setting can produce a ~0 difference on Exp 2 while still reversing Exp 1), it should be replaced by a theory that keeps validity-blind flag counting but adds the one thing FC-LEX lacks: ratio-dependent (Weber-law) discriminability of the flag counts.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace PWEA (pi_3) with a NEW theory: 'Noisy Numerosity Comparison of Warning Flags (Weber Counting, WC)'.\n\nCore claims:\n1. Negative polarity and validity-blindness are retained (as in FC-LEX, and as forced by Exps 2-6): the subject forms the raw number of 1s for each option; the option with the smaller perceived count is chosen. Which experts carry the flags does not enter the comparison.\n2. The counts are represented with scalar-variability / Weber noise, so discriminability depends on the RATIO (or scaled difference) of the two counts, not on the raw difference and not on a flat plateau. Concretely: p(choose fewer-flag option) = (1-eps)*Phi( (c_more - c_less) / (w * sqrt(c_more^2 + c_less^2 + c0)) ) + eps/2, with a subject-level Weber fraction w and a small additive constant c0 so that 0-vs-1 remains easy. This yields: 0-vs-1, 0-vs-5, 1-vs-4, 2-vs-4 easy (~0.8-0.9, matching Exps 3-6), but 2-vs-3 hard (~0.5-0.6, matching Exp 1's observed 0.517), which no parameterisation of FC-LEX can produce and which PWEA only reproduces by an overshoot to 0.24.\n3. Exact count ties are undecidable by the counting mechanism and are resolved by a subject-level, cue-INDEPENDENT idiosyncratic bias (side/order/first-cue habit) that is the same across all equal-count pairs. This predicts exactly the ~0 Exp-2 metric (observed 0.067) that killed PWEA, while remaining agnostic about the absolute tie-choice level.\n4. Optional heterogeneity term to capture Exp 1's bimodal between-subject variance (var=0.133): a subject-level probability that near-chance experts (validity below a subject cutoff, e.g. < ~0.6) are discarded before counting, so a minority of subjects reverse on 2-high-flag vs 3-low-flag pairs. Keep this fraction modest so it does not induce a large Exp-2 P1/P2 asymmetry, and make the filter discrete (all-or-none per expert), NOT graded weighting \\u2014 gradedness is what Exp 2 rules out.\n\nParameter ranges should be chosen so that the model reproduces, in simulation: Exp1 metric ~0.5 with high between-subject spread; Exp2 metric ~0.0-0.1; Exp3 ~0.75-0.80; Exp4 ~ -0.63; Exp5 ~ -0.58; Exp6 ~ +0.66.\n\nThis new theory is a sharp competitor to FC-LEX: they agree on polarity and validity-blindness but disagree on magnitude invariance. Future experiments should therefore contrast count-ratio ladders holding the count difference fixed (0-vs-1, 1-vs-2, 2-vs-3, 3-vs-4) against difference ladders (0-vs-1, 0-vs-2, 0-vs-4), where FC-LEX predicts one flat plateau and WC predicts a monotone ratio-driven gradient."
}
```

## Usage

```json
{
  "input_tokens": 20035,
  "output_tokens": 16146
}
```
