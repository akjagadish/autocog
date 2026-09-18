# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


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

## THEORY 2 — pi_5
**Weber Counting with Severity Tiebreak and Expert Triage (WC-ST).**

A binary expert rating of '1' is read as a *warning flag*, not an endorsement, so an option's badness grows with the number of 1s it carries and the subject picks the option that seems to carry FEWER flags. Three mechanistic claims, plus one heterogeneity claim:

1. **Negative polarity, validity-blind numerosity.** The decision variable is the raw *count* of flags per option, not a validity-weighted mass. Which experts contributed the flags is irrelevant to the count; the stated validities are acknowledged but never converted into cue weights. This is what makes the model look anti-TTB on dominance/agreement pairs and pro-TTB on conflict pairs, with a single mechanism.

2. **Scalar variability (Weber noise) on the counts.** Perceived numerosity is noisy with noise that scales with magnitude, so discriminability depends on the RATIO-scaled difference, not the raw difference:  s = (c_hi - c_lo) / sqrt(c_hi^2 + c_lo^2 + 1). The small additive constant 1 keeps 0-vs-1 easy. Crucially, the psychometric function over s is *threshold-like* (Weibull with slope exponent kappa > 1), i.e. accelerating near zero: comparisons whose count ratio falls below the subject's Weber threshold theta stay essentially at chance, while comparisons above it jump quickly to a lapse-limited ceiling. This is what produces near-chance 2-vs-3 (~0.53) at the same time as high 1-vs-2 (~0.73), 2-vs-4 (~0.78) and near-ceiling 0-vs-1 / 1-vs-4 / 0-vs-5 (~0.85) — a pattern that neither a flat sign-only rule (FC-LEX) nor a linear/Gaussian margin rule can generate.

3. **Severity tiebreak on exact count ties.** When the two options carry exactly the same number of flags the counting mechanism is silent. The subject then asks which single flag is *worst*: the most reliable expert that discriminates decides, and the option carrying THAT expert's flag is (usually) rejected. The strength/direction of this fallback is a subject-level bias tie_bias = P(choose the option flagged by the top discriminating expert), sitting below 0.5 (mildly repelled). Because this rule keys on the top discriminating expert in exactly the same way for every equal-count pair, it predicts *no* differential across equal-count pairs that share the same top discriminating expert.

4. **Discrete expert triage (minority).** A minority of subjects do not count everyone: they apply an all-or-none trust filter, ignoring the one or two least-valid experts entirely before counting (never a graded weighting — gradedness is ruled out by the equal-count data). Triage subjects therefore *reverse* on pairs whose flag surplus lives entirely on the discarded experts (e.g. 2 high-validity flags vs 3 low-validity flags), which injects genuine bimodal between-subject dispersion exactly where the human data show it, and creates the small positive asymmetry among equal-count pairs whose tie is broken by triage.

WC-ST is a sharp competitor to sign-only flag counting: they agree on polarity and validity-blindness but disagree about magnitude invariance. WC-ST predicts a monotone *ratio-driven* accuracy gradient (0v1 > 1v2 > 2v3 > 3v4 at constant difference; 0v1 < 0v2 < 0v4 at constant smaller count), whereas the sign-only rule predicts one flat plateau.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ---------------- 1. parse stimulus ----------------
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
    n = a.size

    # ---------------- 2. validities ----------------
    val = parameters.get("validities", None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = np.array([])
    if val.size != n:
        val = np.array([0.9]) if n == 1 else np.linspace(0.9, 0.55, n)

    # ---------------- 3. subject-level parameters ----------------
    theta = float(parameters["theta"])
    kappa = float(parameters["kappa"])
    eps = float(np.clip(float(parameters["epsilon"]), 0.0, 1.0))
    tie_bias = float(np.clip(float(parameters["tie_bias"]), 0.0, 1.0))
    ip = float(np.clip(float(parameters["ignore_prone"]), 0.0, 1.0))
    theta = max(theta, 1e-3)
    kappa = max(kappa, 0.5)

    # ---------------- 4. discrete expert triage (minority of subjects) ----
    # most subjects count every expert; a minority discard the 1 (or 2)
    # least-valid experts outright before counting.
    if ip < 0.84:
        n_drop = 0
    elif ip < 0.96:
        n_drop = 1
    else:
        n_drop = 2
    mask = np.ones(n, dtype=bool)
    if n_drop > 0 and n > 2:
        order_asc = np.argsort(val, kind="stable")   # ascending validity
        dropped = 0
        for idx in order_asc:
            if dropped >= n_drop or (n - dropped) <= 2:
                break
            if val[idx] < 0.65:                       # only near-chance experts
                mask[idx] = False
                dropped += 1
    af = a[mask]
    bf = b[mask]
    if af.size == 0:
        af, bf, mask = a, b, np.ones(n, dtype=bool)

    # ---------------- 5. Weber-noisy numerosity comparison --------------
    ca = float(np.sum(af))
    cb = float(np.sum(bf))

    if abs(ca - cb) > 1e-9:
        c_hi = max(ca, cb)
        c_lo = min(ca, cb)
        denom = np.sqrt(c_hi * c_hi + c_lo * c_lo + 1.0)
        s = (c_hi - c_lo) / denom               # ratio-scaled evidence
        x = (s / theta) ** kappa
        x = float(np.clip(x, 0.0, 50.0))
        core = 1.0 - 0.5 * np.exp(-x)           # Weibull psychometric fn
        core = float(np.clip(core, 0.5, 1.0))
        winner = 0 if ca < cb else 1            # fewer flags is chosen
        p_core = np.empty(2, dtype=float)
        p_core[winner] = core
        p_core[1 - winner] = 1.0 - core
    else:
        # ------------- 6. severity tiebreak on exact count ties ----------
        idxs = np.where(mask)[0]
        owner = None
        order = idxs[np.argsort(-val[idxs], kind="stable")]
        for j in order:
            if a[j] > b[j]:
                owner = 0
                break
            if b[j] > a[j]:
                owner = 1
                break
        if owner is None:                        # fall back to full cue set
            order_all = np.argsort(-val, kind="stable")
            for j in order_all:
                if a[j] > b[j]:
                    owner = 0
                    break
                if b[j] > a[j]:
                    owner = 1
                    break
        if owner is None:
            p_core = np.ones(2) / 2.0
        else:
            p_core = np.empty(2, dtype=float)
            p_core[owner] = tie_bias             # option carrying the worst flag
            p_core[1 - owner] = 1.0 - tie_bias

    if not np.all(np.isfinite(p_core)):
        p_core = np.ones(2) / 2.0

    # ---------------- 7. attention lapse ----------------
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

## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=6):** [0.92, 0.86, 0.78, 0.68, 0.6, 0.54]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 2: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 4: A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 5: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  trial 6: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  trial 8: A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  trial 9: A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 1, 0]
  trial 10: A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1]
  trial 11: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 12: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 14: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 15: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 16: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]

**Rationale:** I advocate FC-LEX (sign-only, validity-blind flag counting with a lexicographic tiebreak) against WC-ST (Weber-scaled flag counting with a threshold-like Weibull psychometric plus discrete expert triage). The two models agree completely on POLARITY (a '1' is a warning flag, fewer flags wins) and on VALIDITY-BLINDNESS of the count, and they use the identical severity/lexicographic tiebreak on exact ties. So direction-only tests are useless here. The whole design is built on the ONE place they disagree: FC-LEX says only the SIGN of the count difference matters (one flat plateau accuracy on every unequal-count trial), while WC-ST says discriminability is ratio-scaled, s=(c_hi-c_lo)/sqrt(c_hi^2+c_lo^2+1), pushed through an accelerating Weibull, so accuracy must collapse to chance when the count ratio is small.

Six experts with well-spread validities (0.92, 0.86, 0.78, 0.68, 0.60, 0.54). Exactly two experts fall below the 0.65 'triage' threshold (experts 5 and 6), so WC-ST's minority triage mechanism is well defined and testable. All 8 base pairs appear in both A/B mirror orders (16 unique pairs, K=6 -> 96 trials), so side bias cancels exactly.

CLASS 1 - CONSTANT-DIFFERENCE MARGIN LADDER (P1 0-vs-1, P2 1-vs-2, P3 2-vs-3, P4 3-vs-4). Every pair has a count difference of exactly ONE flag in the same direction (A always carries fewer). FC-LEX is mathematically forced to the SAME accuracy (1-eps)*sigmoid(beta)+eps/2 on all four; no FC-LEX parameter can make them differ. WC-ST's ratio evidence is s = 0.707, 0.408, 0.267, 0.196, which through the Weibull (theta in [0.34,0.46], kappa in [4.5,7.5]) gives roughly 1.00, 0.84, 0.54, 0.51 before lapses: a steep monotone collapse from near-ceiling to essentially chance. A reliable within-subject accuracy drop from 0-vs-1 to 3-vs-4 falsifies FC-LEX; a flat plateau across the ladder cannot be produced by WC-ST at any theta/kappa, because its psychometric function is strictly increasing in s.

CLASS 2 - CONSTANT-SMALLER-COUNT CONTRAST (P3 2-vs-3 versus P5 2-vs-4). Same smaller count, different difference. FC-LEX: identical accuracy. WC-ST: s = 0.267 vs 0.436, i.e. near-chance versus ~0.90 core. This is a second, orthogonal magnitude test that shares direction between models, so no strategy-reversal confound can explain a gradient.

CLASS 3 - TRIAGE REVERSAL (P6: A=[1,1,0,0,0,0] two flags on the 92%/86% experts vs B=[0,0,0,1,1,1] three flags on the 68%/60%/54% experts). Counting subjects choose A (2<3); WC-ST triage subjects discard experts 5-6 and read 2-vs-1, choosing B, and n_drop=1 subjects read 2-vs-2 and fall to the severity tiebreak, which also pushes them toward B. FC-LEX must choose A at its usual flat rate for every subject. Bimodal between-subject dispersion, or a group mean pulled to/below chance on P6 while P1 stays near ceiling, is uniquely WC-ST.

CLASS 4 - EQUAL-COUNT PAIRS (P7: [0,0,0,0,1,1] vs [1,1,0,0,0,0], 2-vs-2 with the flags on opposite ends of the validity scale; P8: [1,0,0,0,0,0] vs [0,1,0,0,0,0], 1-vs-1 control). Both models are silent on counting and fall back on the top discriminating expert, so P8 cleanly identifies tie_bias for both. P7 is diagnostic only for WC-ST's triage subgroup (dropping experts 5-6 turns it into 0-vs-2, a decisive A choice), whereas FC-LEX must give exactly the same p(top-flag option)=tie_bias on P7 and P8. A systematically more extreme response on P7 than P8, in a subset of subjects, is a triage signature.

Ladder pairs P1-P3 carry no flags on the droppable experts (triage-immune), and P4-P5 are constructed so that triage preserves the direction of the count difference, so the magnitude ladder is not confounded by the triage mechanism.

PREDICTIONS. FC-LEX: one flat, above-chance accuracy toward the fewer-flag option on P1-P6 regardless of whether the comparison is 0-vs-1 or 3-vs-4, plus the same magnitude-invariant tie_bias on P7 and P8. WC-ST: a steep monotone ratio-driven gradient (P1 near ceiling > P5 > P2 >> P3 ~ P4 ~ chance), and a bimodal, partly reversed response on P6 with an accompanying P7-vs-P8 asymmetry.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The design's core disagreement is magnitude invariance. I restrict scoring to the two 'hard' constant-difference-of-one pairs where the base count is large (2-vs-3 in trials 5/6 and 3-vs-4 in trials 7/8), and I explicitly exclude the triage pair (2 high-validity flags vs 3 low-validity flags) by requiring that the sign of the count difference survives dropping the two least-valid experts. This leaves 24 clean trials per subject.

FC-LEX (advocated) is mathematically forced to its single flat plateau accuracy h = (1-eps)*sigmoid(beta)+eps/2 on these trials, exactly as on 0-vs-1: with beta in [1.3,2.7] and eps in [0.05,0.25] this lands around 0.78-0.85 with small per-subject spread (binomial sd over 24 trials ~0.08).

WC-ST predicts ratio-scaled evidence s = 0.267 (2v3) and 0.196 (3v4), which through its accelerating Weibull (theta in [0.34,0.46], kappa in [4.5,7.5]) gives cores of ~0.51-0.64, and its much larger lapse range (eps up to 0.48) compresses this further toward chance: roughly 0.50-0.58. Excluding the triage pair removes the bimodal reversal subgroup, so WC-ST's per-subject variance also stays tight (it is pinned near the chance floor).

Expected contrast: ~0.81 (FC-LEX) vs ~0.53 (WC-ST), a ~0.28 gap with per-subject sd ~0.10 on both sides, easily significant at N=25. It is not trivially true for my theory: if humans show the ratio-driven collapse, this metric will read near chance and falsify FC-LEX.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0
    tot = 0
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()
            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()
        except Exception:
            continue
        if a.size == 0 or a.size != b.size:
            continue
        n = a.size
        ca = float(a.sum())
        cb = float(b.sum())
        d = ca - cb
        # keep only constant-difference-of-one comparisons with a LARGE base count
        if abs(abs(d) - 1.0) > 1e-9:
            continue
        if min(ca, cb) < 2.0 - 1e-9:
            continue
        # drop triage-sensitive pairs: the sign of the count difference must be
        # preserved when the two least-valid experts are discarded
        k = n - 2
        if k < 1:
            continue
        dh = float(a[:k].sum()) - float(b[:k].sum())
        if abs(dh) < 1e-9:
            continue
        if np.sign(dh) != np.sign(d):
            continue
        winner = 0 if d < 0 else 1  # fewer flags wins
        try:
            resp = int(row['response'])
        except Exception:
            continue
        if resp not in (0, 1):
            continue
        tot += 1
        if resp == winner:
            hits += 1

    if tot == 0:
        return 0.5
    return float(hits) / float(tot)

### RESULTS
- Predicted under pi_4 (simulated): 0.8217 (var=0.0065)
- Predicted under pi_5 (simulated): 0.5408 (var=0.0203)
- Observed on real data: 0.7733 (var=0.0327)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=8):** [0.93, 0.88, 0.82, 0.75, 0.7, 0.62, 0.57, 0.52]

**Trial pairs (n=18):**
  trial 1: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  trial 4: A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0]
  trial 8: A=[0, 1, 1, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 0, 0]
  trial 10: A=[0, 0, 1, 1, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  trial 11: A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 0]
  trial 12: A=[0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  trial 13: A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1]
  trial 14: A=[0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  trial 15: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  trial 18: A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1]

**Rationale:** I advocate WC-ST (Weber-scaled flag counting with a threshold-like Weibull, severity tiebreak, and discrete expert triage) against FC-LEX (sign-only, validity-blind flag counting with a lexicographic tiebreak). The two models are identical in polarity (a '1' is a warning flag; the option with fewer flags is chosen), identical in validity-blindness of the count, and identical in their tiebreak on exact ties. So direction tests are worthless. Every pair here targets the single point of disagreement: FC-LEX says only the SIGN of the count difference matters (one flat plateau accuracy on every unequal-count trial, for any beta/eps), whereas WC-ST says discriminability is RATIO-scaled, s=(c_hi-c_lo)/sqrt(c_hi^2+c_lo^2+1), pushed through an accelerating Weibull, so accuracy must fall to chance when the count ratio is small and must be INVARIANT across pairs with equal ratio but very different raw differences.

Eight experts, well spread (0.93, 0.88, 0.82, 0.75, 0.70, 0.62, 0.57, 0.52). Exactly the two lowest (0.57, 0.52 at indices 6 and 7) are the ones WC-ST's triage subjects can discard (val<0.65 and only the 1-2 least valid are dropped), so triage is sharply defined. All 9 base pairs appear in both A/B mirror orders (18 unique pairs, K=5 -> 90 trials), so side bias cancels exactly.

CLASS 1 - CONSTANT-DIFFERENCE LADDER (P1 0v1, P2 1v2, P3 2v3, P4 3v4; all flags confined to experts 1-6 so triage cannot touch them). The count difference is exactly ONE flag in the same direction on all four, so FC-LEX is mathematically forced to the identical accuracy (1-eps)*sigmoid(beta)+eps/2 on all four. WC-ST's ratio evidence is s = 0.707, 0.408, 0.267, 0.196, giving core accuracies of about 1.00, 0.84, 0.54, 0.51 for mid-range theta=0.40, kappa=6: a steep, monotone collapse from ceiling to chance. A within-subject accuracy drop from 0v1 to 3v4 is impossible for FC-LEX at any parameters; a flat plateau is impossible for WC-ST because its psychometric function is strictly increasing in s.

CLASS 2 - RATIO INVARIANCE AT GROWING RAW DIFFERENCE (P2 1v2, P5 2v4, P6 3v6; ratios all 1:2, raw differences 1, 2, 3). WC-ST's s = 0.408, 0.436, 0.442 -- essentially identical, so it predicts these three to be statistically indistinguishable and all clearly ABOVE P3 (2v3, difference 1, s=0.267) and P4 (3v4, difference 1, s=0.196), even though P3/P4 involve larger counts and P4 has the same raw difference as P2. FC-LEX predicts all five of P2-P6 equal. A linear/Gaussian margin account would predict P6>P5>P2 and P4~P2. Only WC-ST predicts the exact pattern {P2 ~ P5 ~ P6} >> {P3 > P4 ~ chance}. This is the core quantitative signature and is measured on trials where both theories agree in DIRECTION, so no strategy-reversal confound can explain a gradient.

CLASS 3 - TRIAGE REVERSAL (P7: A=[1,1,0,0,0,0,0,0], two flags on the 93%/88% experts, vs B=[0,0,0,0,0,1,1,1], three flags on the 62%/57%/52% experts). Plain counters (both models) choose A because 2<3. WC-ST triage subjects who discard experts 7 and 8 read 2-vs-1 and choose B at s=0.408 (~0.84 core); subjects who discard only expert 8 read 2-vs-2, fall to the severity tiebreak keyed on expert 1 (owned by A) and, since tie_bias<0.5, are also pushed toward B. FC-LEX must choose A at its flat plateau rate for every subject. Bimodal between-subject dispersion on P7, or a group mean pulled toward/below chance on P7 while P1 stays near ceiling, is uniquely WC-ST.

CLASS 4 - EQUAL-COUNT PAIRS (P8: [1,0,...] vs [0,1,0,...], a clean 1-vs-1 tie; P9: A=[0,0,0,0,0,0,1,1] (2 flags on the two droppable experts) vs B=[1,1,0,0,0,0,0,0] (2 flags on the top two experts)). Both models are silent on counting and fall back on the top discriminating expert, so P8 identifies tie_bias for both models simultaneously. FC-LEX is then committed to |p-0.5| being EXACTLY the same on P9 as on P8 (the tiebreak is invoked identically and keys on expert 1 in both). WC-ST agrees for non-triage subjects, but triage subjects read P9 as 0-vs-2 (s=0.894, ceiling) and choose A decisively. A reliably more extreme, less variable response on P9 than P8 -- concentrated in the same subjects who reverse on P7 -- is a triage signature that FC-LEX cannot produce with its single tie_bias.

PREDICTIONS. WC-ST: near-ceiling on P1; ~0.84-0.91 and mutually indistinguishable on P2, P5, P6; near chance on P3 and especially P4; partly reversed / bimodal on P7; and a P9-vs-P8 asymmetry in the triage subgroup. FC-LEX: one flat above-chance accuracy toward the fewer-flag option on P1-P7 irrespective of whether the comparison is 0v1, 3v4 or 3v6 and irrespective of which experts carry the flags, plus the same magnitude-invariant tie_bias-driven deviation on P8 and P9.

**Computed schedule:** 18 unique pairs × 5 reps = 90 trials per subject.



### METRIC
Rationale:
This is a within-subject Weber-ratio contrast that both theories can be scored on with identical trial-direction coding (choose the option with fewer flags), so it cannot be trivially true for my theory.

EASY group = count pairs with high ratio-scaled evidence s = (c_hi-c_lo)/sqrt(c_hi^2+c_lo^2+1): 0v1 (s=.71), 1v2 (s=.41), 2v4 (s=.44), 3v6 (s=.44). HARD group = count pairs with low ratio evidence but comparable or identical raw differences: 2v3 (s=.27) and 3v4 (s=.20, same raw difference of 1 as 0v1 and 1v2). I exclude the 2v3 pair whose surplus flags sit on the two droppable low-validity experts (trials 13/14) so that discrete triage subjects do not add bimodal dispersion to the hard group; likewise the equal-count tie pairs are automatically excluded (ca==cb skipped).

FC-LEX is sign-only and magnitude-blind: every trial in both groups has a strictly unequal flag count, so its accuracy is the SAME constant h = (1-eps)*sigmoid(beta)+eps/2 on every trial in both groups, for any beta and eps. Its expected metric value is exactly 0, with only binomial sampling noise across 40 easy / 20 hard trials per subject (var ~0.011), and the beta/eps heterogeneity cancels within subject.

WC-ST has an accelerating Weibull over s with threshold theta in [.34,.46] and kappa in [4.5,7.5]: easy-group core accuracy runs ~0.72-1.00 (mid ~0.91) while hard-group core is pinned near chance (~0.50-0.64, mid ~0.52). After lapses the expected gap is roughly 0.25-0.30, and the gap is positively driven by the same subject-level theta so it stays comfortably above zero for essentially every subject.

Expected: ~0.27 for WC-ST vs ~0.00 for FC-LEX, with per-subject variance ~0.02 and ~0.01, giving Welch t ~7 at N=25. Any flat plateau (FC-LEX) or a linear/Gaussian margin rule (which would push 2v4 and 3v6 up but keep 3v4 ~ 1v2) is distinguished from the pure ratio signature.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    if data is None or len(data) == 0:
        return 0.0

    easy_pairs = {(0, 1), (1, 2), (2, 4), (3, 6)}
    hard_pairs = {(2, 3), (3, 4)}

    easy_hits = []
    hard_hits = []

    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row["option_a_ratings"]), dtype=float).ravel()
            b = np.asarray(list(row["option_b_ratings"]), dtype=float).ravel()
            resp = int(row["response"])
        except Exception:
            continue
        if a.size == 0 or a.size != b.size:
            continue
        ca = int(round(float(np.sum(a))))
        cb = int(round(float(np.sum(b))))
        if ca == cb:
            continue
        cp = (min(ca, cb), max(ca, cb))
        # winner = option with FEWER flags (both theories agree on this direction)
        winner = 0 if ca < cb else 1
        hit = 1.0 if resp == winner else 0.0

        n = a.size
        if n >= 3:
            low_flag = bool(np.any(a[-2:] > 0.5) or np.any(b[-2:] > 0.5))
        else:
            low_flag = False

        if cp in easy_pairs:
            easy_hits.append(hit)
        elif cp in hard_pairs and not low_flag:
            hard_hits.append(hit)

    if len(easy_hits) == 0 or len(hard_hits) == 0:
        return 0.0

    return float(np.mean(easy_hits) - np.mean(hard_hits))

### RESULTS
- Predicted under pi_4 (simulated): 0.0065 (var=0.0132)
- Predicted under pi_5 (simulated): 0.2835 (var=0.0155)
- Observed on real data: 0.1660 (var=0.0789)

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
**Predicted under pi_5:** 0.7900 (var=0.0075)

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
**Predicted under pi_5:** -0.6361 (var=0.0184)

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
**Predicted under pi_5:** -0.5296 (var=0.0160)

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
**Predicted under pi_5:** 0.6633 (var=0.0347)

### Experiment 7
**Design**
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]

**Metric**
```python
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

```

**Observed (real) value:** 0.5167 (var=0.1333)
**Predicted under pi_4:** 0.8367 (var=0.0105)
**Predicted under pi_5:** 0.5350 (var=0.0250)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
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
```

**Observed (real) value:** 0.0667 (var=0.0622)
**Predicted under pi_4:** -0.0117 (var=0.0319)
**Predicted under pi_5:** 0.1433 (var=0.1053)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all eight experiments the two theories agree on the big, robust facts of this domain: '1' is read as a warning flag (negative polarity), and the option with fewer flags is generally chosen. That shared core is why both fit Experiments 3, 4, 5 and 6 almost perfectly (predicted values within ~0.05 of observed). The arbitration therefore turns entirely on the two places where they differ: magnitude scaling and validity sensitivity.\n\nMagnitude. Experiment 1 (pi_4's design) scored accuracy on hard constant-difference pairs (2v3, 3v4) with triage-sensitive pairs excluded: observed 0.773. FC-LEX (pi_4) predicted 0.822 \u2014 close; WC-ST (pi_5) predicted 0.541 \u2014 a 0.23 miss, because its accelerating Weibull pins low-ratio comparisons at chance. Experiment 2 (pi_5's design) scored the easy-minus-hard accuracy gap: observed 0.166. FC-LEX predicted 0.007 (flat plateau, a 0.16 miss); WC-ST predicted 0.284 (a 0.12 miss). So the truth is intermediate: humans DO show a real, reliable magnitude/ratio gradient, but a shallow one \u2014 hard pairs stay well above chance (~0.70-0.77) rather than collapsing. FC-LEX's mathematically-forced flat plateau is falsified by Exp 2; WC-ST's threshold collapse is falsified by Exp 1.\n\nValidity sensitivity. Experiment 7 is the decisive discriminator and it hurts pi_4 badly. On the strong-conflict pair [1,1,0,0,0] vs [0,0,1,1,1] with extremely skewed validities (0.97, 0.66, 0.60, 0.56, 0.52), the observed rate of choosing the fewer-flag option was 0.517 \u2014 pure chance \u2014 with an enormous between-subject variance of 0.133 (clear bimodality). FC-LEX must choose the fewer-flag option at its flat plateau: it predicted 0.837, a 0.32 miss, its worst failure anywhere. WC-ST's discrete triage subgroup produced 0.535 \u2014 essentially exact. Note that in Experiment 1, where the validities were only mildly spread (0.92...0.54), the same 2-vs-3 structure yielded 0.77, i.e. counting won. So flag counting is NOT strictly validity-blind: when one expert is enormously more valid than the rest, subjects (or a large subgroup of them) discount the low-validity flags and the count advantage evaporates. That is graded/partial validity weighting with large between-subject heterogeneity, which pi_5 approximates crudely (all-or-none triage) and pi_4 cannot produce at all.\n\nSummed absolute error across the eight experiments: pi_4 ~0.70 (driven by Exp 7 = 0.32 and Exp 2 = 0.16), pi_5 ~0.51 (driven almost entirely by Exp 1 = 0.23). pi_5 is the more task-invariant of the two and should be kept as the incumbent. pi_4's two defining commitments \u2014 sign-only magnitude invariance and strict validity-blindness \u2014 are each directly contradicted by data, so it is the theory to replace. Rather than re-tuning FC-LEX's parameters (its flat plateau is a structural, parameter-free prediction, so no re-parameterisation can fix Exp 2 or Exp 7), it should be replaced by a genuinely new competitor that can beat pi_5 on Exp 1 while matching it on Exp 7.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_4 (FC-LEX) with a graded negative-evidence account that keeps the domain's robust polarity finding but abandons both sign-only invariance and strict validity-blindness. Sketch: 'Partially Weighted Flag Mass with Divisive Compression (PW-FM)'.\n\n1. Polarity (shared, keep): a '1' is a warning flag; the option with the smaller badness score is chosen. This preserves the excellent fits on Exps 3-6.\n\n2. Partial validity weighting with subject-level heterogeneity (the new mechanism, replacing both validity-blind counting and all-or-none triage). Each cue gets weight w_j = 1 + gamma * (r_j - mean(r)), where r_j = (v_j - 0.5)/0.5 and gamma is a SUBJECT-level parameter spanning 0 (pure counter) to ~1.3 (strongly validity-weighted). Badness = sum of w_j over that option's flags. This is a continuum, not a discrete filter, so the model naturally produces bimodal-looking dispersion when the design puts counting and weighting in conflict. Critically it is design-adaptive in exactly the way the data demand: with the mildly spread validities of Exp 1 (0.92...0.54) the weighted and count-based winners COINCIDE on the 2v3 and 3v4 pairs, so accuracy stays high (~0.77 target); with the extremely skewed validities of Exp 7 (0.97 vs 0.52-0.66) the weighted score for [1,1,0,0,0] roughly equals that for [0,0,1,1,1], so the group mean lands at chance with huge between-subject variance (matching 0.517, var 0.133). This single mechanism explains what neither incumbent could explain jointly.\n\n3. Mild divisive compression instead of a threshold Weibull (fixing pi_5's Exp 1 failure and pi_4's Exp 2 failure). Evidence s = (mass_hi - mass_lo) / (1 + c * (mass_hi + mass_lo)), with c small (roughly 0.1-0.3), passed through a plain logistic p = sigmoid(beta * s) with lapse eps. This yields a SHALLOW monotone decline with base count (0v1 > 1v2 > 2v3 > 3v4) rather than a collapse to chance, so an easy-minus-hard gap near the observed 0.17 (Exp 2) coexists with ~0.75-0.80 accuracy on 2v3/3v4 (Exp 1). Set beta ranges so that the 0v1 core sits near 0.90 and the 3v4 core near 0.68-0.72 for a mid-range subject.\n\n4. Exact ties (equal weighted mass, e.g. gamma=0 counters on 2v2 pairs) fall back to a top-discriminating-expert rule with a subject-level tie_bias slightly below 0.5 (repelled by the worst flag), reproducing Exp 5, Exp 6 and the small positive Exp 8 asymmetry (weighted subjects break the [1,0,0,0,1] vs [0,1,1,0,0] pair differently from the [1,0,0,0,0] vs [0,0,0,0,1] pair, giving a small positive metric ~+0.07-0.15).\n\nThe new theory must consume the design validities from state/parameters (they vary per experiment: n_features 5, 6 and 8 have all appeared) and must be checked to reproduce, at minimum: Exp1 ~0.77, Exp2 ~0.17, Exp3 ~0.77, Exp4 ~-0.63, Exp5 ~-0.58, Exp6 ~0.66, Exp7 ~0.52 with large subject variance, Exp8 ~0.07. Keep pi_5 (WC-ST) unchanged as the incumbent competitor; the sharp future test between them will be whether hard-ratio accuracy is threshold-like (WC-ST) or shallow-graded (PW-FM), and whether the Exp-7-style reversal is all-or-none in a minority (triage) or continuously graded with validity spread (partial weighting)."
}
```

## Usage

```json
{
  "input_tokens": 24356,
  "output_tokens": 7474
}
```
