# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
**Limited-Sample Noisy Cue Counting (LSNC).** People neither integrate all cues (Tallying) nor consult a single best cue (Take-The-Best). Instead, on each choice they inspect only a *limited, randomly selected subset* of the presented expert ratings, tally feature-wise wins **within that sample only**, and pick the option that is ahead in the sample; a within-sample tie (including a sample containing no discriminating cue) forces a guess.

Three commitments define the theory:
1. **Equal-probability sampling.** The inspected subset is drawn uniformly at random from the whole profile — it is *not* validity-ordered and *not* validity-weighted. Stated validities are too weakly represented to steer the search order, so no cue enjoys priority. Consequently, on globally tally-tied, side-counterbalanced profiles the process is exactly symmetric and choice is 50/50, with no tendency to follow the most valid discriminating cue.
2. **Sample-size limitation with tie dilution.** The number of cues actually inspected, K, is a random variable with mean k < n (implemented as K = 1 + Binomial(n−1, q), so at least one cue is always read). Because the subset is drawn from *all* features, non-discriminating (tied) features occupy sampling slots and dilute the evidence: two profiles with the same raw tally margin M produce different adherence depending on how many uninformative cues pad the profile, and on the proportion (not the raw count) of informative cues won. This is the signature that dissociates LSNC from softmax Tallying, whose predictions depend only on M.
3. **Encoding noise + lapse.** Each inspected feature-wise comparison is registered with the wrong sign with probability nu (attention/encoding noise), and with probability epsilon the whole decision is replaced by a coin flip.

Because a small, noisy sample can easily reverse the sign of a large global margin, adherence to the full-tally winner is *capped well below 1* even on high-margin conflict trials — the systematic attenuation that full Tallying cannot produce without degenerate temperature — while symmetry guarantees exactly chance behaviour on tally-tied pairs.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    from math import comb

    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("LSNC expects a (2, n_features) stimulus; got shape %s." % (stim.shape,))
    a, b = stim[0], stim[1]
    n = int(a.shape[0])

    nA = int(np.sum(a > b))          # cues on which A wins
    nB = int(np.sum(b > a))          # cues on which B wins
    nT = n - nA - nB                 # uninformative (tied) cues

    k = float(parameters["k"])            # mean number of cues inspected
    nu = float(parameters["nu"])          # per-comparison encoding error
    eps = float(parameters["epsilon"])    # lapse rate
    nu = float(min(max(nu, 0.0), 0.5))
    eps = float(min(max(eps, 0.0), 1.0))

    if n == 0:
        return np.ones(2) / 2.0

    # ---- distribution of the inspected sample size K in {1..n} -------------
    if n == 1:
        pK = np.array([1.0])
    else:
        q = (k - 1.0) / (n - 1.0)
        q = float(min(max(q, 0.0), 1.0))
        pK = np.array([comb(n - 1, s - 1) * (q ** (s - 1)) * ((1.0 - q) ** (n - s))
                       for s in range(1, n + 1)], dtype=float)
        tot = pK.sum()
        pK = pK / tot if tot > 0 else np.ones(n) / float(n)

    # ---- exact enumeration over sample compositions ------------------------
    pA = 0.0
    for si, s in enumerate(range(1, n + 1)):
        w = float(pK[si])
        if w <= 0.0:
            continue
        denom = float(comb(n, s))
        for ai in range(0, min(nA, s) + 1):
            for bi in range(0, min(nB, s - ai) + 1):
                ti = s - ai - bi
                if ti < 0 or ti > nT:
                    continue
                cnt = comb(nA, ai) * comb(nB, bi) * comb(nT, ti)
                if cnt == 0:
                    continue
                ph = float(cnt) / denom          # multivariate hypergeometric
                m = ai + bi                      # informative cues in sample
                if m == 0:
                    pA += w * ph * 0.5           # nothing discriminates -> guess
                    continue
                # X = number of sampled comparisons REGISTERED as A-wins
                d1 = np.array([comb(ai, x) * ((1.0 - nu) ** x) * (nu ** (ai - x))
                               for x in range(ai + 1)], dtype=float)
                d2 = np.array([comb(bi, x) * (nu ** x) * ((1.0 - nu) ** (bi - x))
                               for x in range(bi + 1)], dtype=float)
                dist = np.convolve(d1, d2)
                xs = np.arange(m + 1)
                p_win = float(dist[2 * xs > m].sum())
                p_tie = float(dist[2 * xs == m].sum())
                pA += w * ph * (p_win + 0.5 * p_tie)

    pA = float(min(max(pA, 0.0), 1.0))
    p = np.array([pA, 1.0 - pA], dtype=float)
    p = (1.0 - eps) * p + eps * 0.5
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum()

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_4
**Count-Direction with Conflict-Graded Confidence (CDCC).**

When people compare two multi-attribute options described by binary expert ratings, the *direction* of their preference and the *confidence* with which they execute it are computed by two functionally separate processes.

1. **Direction is purely count-based.** The comparator registers, for each attribute, only which option is rated higher, and the option that wins on more attributes becomes the candidate response. Validities never enter this stage: they cannot promote a minority-supported option to candidate status. When the two win-counts are equal, the comparator returns no candidate at all, and the subject flips a mental coin — exactly 0.5, with **no** validity-ordered tie-break. This is a hard, parameter-free prediction: on tally-tied, side-counterbalanced profiles CDCC predicts chance adherence to the most-valid discriminating cue, regardless of any parameter setting.

2. **Confidence, not direction, carries the validity information, and it is graded by evidence dilution and by cue conflict.** The probability of actually executing the count-winner is a saturating (tanh) function of a scalar confidence signal built from three ingredients: (i) the raw win margin M = |nA − nB|; (ii) *dilution* — the margin is discounted by the total number of discriminating cues raised to a power gamma, so that a 6-vs-2 split feels much weaker than a 4-vs-0 split even though both have M = 4 (evidence spread thin across many mutually contradicting attributes is subjectively less compelling than a clean, sparse, unanimous split); and (iii) *coherence* — the signed, normalised balance of validity weight between the cues the count-winner wins and the cues the loser wins, where the subjective weight of a cue is a convex (super-linear) function of its stated validity advantage, w_j = (v_j − 0.5)^rho. Convexity means the coherence signal is dominated by the few high-validity cues: when the best cue points *with* the count the subject feels certain, when it points *against* the count the subject feels torn even though the count itself is unambiguous.

3. **Conflict erodes confidence but never reverses direction.** The confidence argument is floored at zero, so validity conflict can drive behaviour arbitrarily close to guessing but never below chance: CDCC never makes systematic one-reason (Take-The-Best) choices. This is what separates it from any Tallying+TTB mixture: on high-margin trials where the top cue opposes the tally, people move toward 0.5 from above, not past it.

4. **Lapses.** With probability epsilon the entire decision is replaced by a coin flip (attention lapse), which caps adherence below 1 even on maximally coherent trials.

The theory therefore predicts a specific cross-design signature: chance performance on tally-tied profiles; high but sub-ceiling adherence on sparse, coherent, unanimous profiles; noticeably lower adherence on dense profiles where the loser also wins several cues even when the raw margin is larger; and the lowest (but still above-chance) adherence when the most valid cue opposes the count.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 1:
        stim = stim.reshape(2, -1)
    if stim.ndim != 2 or stim.shape[0] != 2:
        stim = stim.reshape(2, -1)
    a = stim[0]
    b = stim[1]
    n = int(a.shape[0])

    if n == 0:
        return np.array([0.5, 0.5], dtype=float)

    a_win = a > b
    b_win = b > a
    nA = int(np.sum(a_win))
    nB = int(np.sum(b_win))
    D = nA + nB

    # --- Commitment 1: direction is purely count-based; ties -> exact guess ---
    if D == 0 or nA == nB:
        return np.array([0.5, 0.5], dtype=float)

    winner = 0 if nA > nB else 1
    M = float(abs(nA - nB))

    # --- parameters -----------------------------------------------------
    beta = float(parameters.get("beta", 2.6))
    gamma = float(parameters.get("gamma", 1.5))
    kappa = float(parameters.get("kappa", 0.7))
    rho = float(parameters.get("rho", 3.0))
    eps = float(parameters.get("epsilon", 0.10))

    beta = min(max(beta, 0.0), 20.0)
    gamma = min(max(gamma, 0.0), 3.0)
    kappa = min(max(kappa, 0.0), 5.0)
    rho = min(max(rho, 0.5), 8.0)
    eps = min(max(eps, 0.0), 0.6)

    # --- validity weights (convex in validity advantage) ------------------
    v_raw = parameters.get("validities", None)
    v = None
    if v_raw is not None:
        try:
            v = np.asarray(v_raw, dtype=float).ravel()
        except Exception:
            v = None
    if v is None or v.shape[0] != n:
        # fallback: assume features are ordered by descending validity
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75])

    w = np.clip(v - 0.5, 1e-9, None) ** rho

    if winner == 0:
        W_win = float(np.sum(w[a_win]))
        W_lose = float(np.sum(w[b_win]))
    else:
        W_win = float(np.sum(w[b_win]))
        W_lose = float(np.sum(w[a_win]))

    tot = W_win + W_lose
    C = 0.0 if tot <= 0.0 else (W_win - W_lose) / tot   # in [-1, 1]

    # --- Commitment 2: dilution-discounted margin, coherence-gated --------
    S = M / (float(D) ** gamma)
    gate = (1.0 + kappa * C) / (1.0 + kappa)             # 1.0 when fully coherent
    arg = beta * S * gate

    # --- Commitment 3: conflict erodes confidence, never reverses it ------
    if arg < 0.0:
        arg = 0.0
    if arg > 30.0:
        arg = 30.0

    conf = float(np.tanh(arg))

    # --- Commitment 4: lapse ---------------------------------------------
    p_winner = 0.5 + 0.5 * (1.0 - eps) * conf
    p_winner = float(min(max(p_winner, 0.5), 1.0 - 1e-9))

    p = np.empty(2, dtype=float)
    p[winner] = p_winner
    p[1 - winner] = 1.0 - p_winner

    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum()

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=6):** [0.94, 0.57, 0.86, 0.62, 0.7, 0.78]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  trial 2: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  trial 3: A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 5: A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 0, 1, 0]
  trial 6: A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1]
  trial 7: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 8: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  trial 9: A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  trial 10: A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  trial 11: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 12: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]

**Rationale:** GOAL: dissociate Limited-Sample Noisy Cue Counting (LSNC, advocated) from Count-Direction with Conflict-Graded Confidence (CDCC, competing). Both theories agree on the count-based DIRECTION of choice and both cap adherence below 1, so the classic tally-vs-TTB contrasts are useless here. They differ on exactly two things, and this design isolates both.

KEY DISSOCIATION 1 (parameter-free for LSNC): COHERENCE. LSNC's predicted probability is a function of the composition (nA, nB, nT) ALONE 
cue identity/validity is irrelevant because the inspected subset is drawn uniformly at random. CDCC's confidence is gated by the coherence term C = (W_win - W_lose)/(W_win + W_lose) with convex weights w = (v-0.5)^rho, so it must show a large drop when the highest-validity cue is won by the count LOSER, holding (nA, nB, nT) exactly fixed. Every composition therefore appears in a COHERENT version (count winner also wins the top-validity cues, loser wins only the weakest ones) and a CONFLICT version (count winner wins only the weakest cues, loser wins the top cue(s)), with the identical (nA, nB, nT).

KEY DISSOCIATION 2 (composition ordering reversal): with n = 6 fixed, CDCC's dilution term is S = M/D^gamma (D = nA+nB, ties irrelevant), whereas LSNC's dilution comes from tied cues occupying sampling slots and from the PROPORTION of sampled informative cues won. These give opposite orderings for two of the three cells: LSNC ranks (4,2,0) ABOVE (2,1,3) (0.630 vs 0.589), while CDCC ranks (2,1,3) ABOVE (4,2,0) (0.633 vs 0.602 averaged over coherence; 0.707 vs 0.651 in the coherent versions), because CDCC penalises D = 6 heavily via D^1.5 while LSNC penalises the 3 tied slots.

DESIGN: 6 experts, scrambled validity spread [0.94, 0.57, 0.86, 0.62, 0.70, 0.78] (descending rank f0 > f2 > f5 > f4 > f3 > f1, so validity rank is decoupled from screen position and positional heuristics cannot mimic validity-based coherence). One very high (0.94) and two low (0.57, 0.62) validities make the convex weighting in CDCC bite hard (w = (v-.5)^3 gives f0 = .085 vs f1 = .0003, a 250:1 ratio). 3 compositions x 2 coherence levels x 2 side mirrors = 12 unique pairs x 8 repetitions = 96 trials.

CELLS (A-favouring version listed; the mirror swaps A and B):
- T1/T2  C1-COHERENT (nA,nB,nT) = (2,1,3): A wins f0(.94), f2(.86); B wins f1(.57); f3,f4,f5 tied.
- T3/T4  C1-CONFLICT (2,1,3): A wins f1(.57), f3(.62); B wins f0(.94); f2,f4,f5 tied.
- T5/T6  C2-COHERENT (3,1,2): A wins f0,f2,f5; B wins f1; f3,f4 tied.
- T7/T8  C2-CONFLICT (3,1,2): A wins f1,f3,f4; B wins f0; f2,f5 tied.
- T9/T10 C3-COHERENT (4,2,0): A wins f0,f2,f4,f5; B wins f1,f3.
- T11/T12 C3-CONFLICT (4,2,0): A wins f1,f3,f4,f5; B wins f0,f2.

QUANTITATIVE PREDICTIONS. LSNC (exact enumeration, k = 2.5, nu = 0.20, eps = 0): P(count winner) = 0.589 for (2,1,3), 0.656 for (3,1,2), 0.630 for (4,2,0) 
 IDENTICAL for the coherent and conflict member of each pair (a strictly zero coherence effect, for every k, nu, eps). CDCC (beta = 2.6, gamma = 1.5, kappa = 0.7, rho = 3.0, eps = 0.10): coherent 0.707 / 0.757 / 0.651 versus conflict 0.544 / 0.576 / 0.553 
 a 10-18 point coherence effect that cannot be switched off (kappa > 0.4 by hypothesis) and that is largest for the sparse (2,1,3) and (3,1,2) cells where the loser's single win is the top expert.

Each subject contributes 48 coherent and 48 conflict trials, so the coherence contrast is estimated within subject with high power. INTERPRETATION: adherence that depends only on the (nA,nB,nT) composition, is flat across the coherence manipulation, and shows the (3,1,2) > (4,2,0) > (2,1,3) ordering with a ceiling near 0.65, uniquely supports LSNC. A systematic coherent > conflict gap (never dropping below 0.5) together with the (3,1,2) > (2,1,3) > (4,2,0) ordering uniquely supports CDCC. Below-chance adherence in the conflict cells would falsify BOTH theories in favour of validity-weighted / one-reason integration, so the design is also self-diagnostic.

CONTROLS: (i) every configuration appears in an A-favouring and a B-favouring mirror, so side bias cancels exactly; (ii) tied cues are encoded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s does not covary with composition; (iii) validity order is scrambled relative to display position; (iv) no pair is degenerate 
 every trial has a strict count winner and every trial requires reading more than one rating.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
The design holds the win-count composition (nA,nB,nT) exactly fixed while flipping whether the highest-validity expert (0.94) sides with or against the count winner. LSNC (advocated) samples cues uniformly at random, so its predicted adherence depends only on (nA,nB,nT) and is mathematically invariant to which cues the winner owns: the coherent-minus-conflict adherence difference is exactly 0 for every parameter setting (k, nu, epsilon). CDCC gates confidence through the convex validity-weighted coherence term C, with kappa >= 0.4 by hypothesis, so it must show a large positive gap (~0.15, i.e. ~0.71 coherent vs ~0.56 conflict). Each subject supplies 48 coherent and 48 conflict trials, so the within-subject difference has SE ~0.10, giving a very tight per-subject estimate and a between-theory separation of many standard errors at N=25. Side mirrors within each cell cancel any side bias, and the coherence label is derived generically (winner owns the most valid discriminating cue), so no trial-index bookkeeping is required.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    validities = np.array([0.94, 0.57, 0.86, 0.62, 0.70, 0.78], dtype=float)

    coh_hit = 0.0
    coh_n = 0.0
    con_hit = 0.0
    con_n = 0.0

    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(list(a_raw), dtype=float)
        b = np.asarray(list(b_raw), dtype=float)
        n = a.shape[0]
        if n == 0:
            continue
        if validities.shape[0] == n:
            v = validities
        else:
            v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75])

        a_win = a > b
        b_win = b > a
        nA = int(a_win.sum())
        nB = int(b_win.sum())
        if nA == nB:
            continue  # no count winner; excluded
        winner = 0 if nA > nB else 1

        disc = a_win | b_win
        if not disc.any():
            continue
        # highest-validity discriminating cue
        idx = int(np.argmax(np.where(disc, v, -np.inf)))
        top_owner = 0 if a_win[idx] else 1
        coherent = (top_owner == winner)

        try:
            r = int(resp)
        except Exception:
            continue
        hit = 1.0 if r == winner else 0.0

        if coherent:
            coh_hit += hit
            coh_n += 1.0
        else:
            con_hit += hit
            con_n += 1.0

    if coh_n == 0.0 or con_n == 0.0:
        return 0.0
    return float(coh_hit / coh_n - con_hit / con_n)


### RESULTS
- Predicted under pi_3 (simulated): -0.0042 (var=0.0075)
- Predicted under pi_4 (simulated): 0.1438 (var=0.0077)
- Observed on real data: -0.7633 (var=0.0200)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=7):** [0.94, 0.58, 0.86, 0.63, 0.72, 0.8, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0, 1]
  trial 2: A=[0, 1, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 1, 0]
  trial 4: A=[1, 0, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 1, 1]
  trial 5: A=[1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0, 1]
  trial 6: A=[0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 1]
  trial 7: A=[0, 1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0]
  trial 8: A=[1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1]
  trial 9: A=[1, 0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 0, 1]
  trial 10: A=[1, 0, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1]
  trial 11: A=[1, 1, 1, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0]
  trial 12: A=[0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1]

**Rationale:** GOAL: dissociate Count-Direction with Conflict-Graded Confidence (CDCC, advocated) from Limited-Sample Noisy Cue Counting (LSNC, competing). Both theories agree that the DIRECTION of choice is the count winner and both cap adherence below 1, so tally-vs-TTB style contrasts are useless. They differ on exactly what modulates adherence: LSNC's probability is a function of the composition (nA, nB, nT) ALONE (uniform sampling makes cue identity/validity irrelevant), whereas CDCC's confidence is (i) gated by validity COHERENCE C = (W_win - W_lose)/(W_win + W_lose) with convex weights w = (v-0.5)^rho, and (ii) diluted by S = M/D^gamma where D = nA+nB, so tied cues are completely irrelevant to CDCC but occupy sampling slots for LSNC. This design attacks both differences and, crucially, contains two ORDERING REVERSALS.

DESIGN: 7 experts, scrambled validity spread [0.94, 0.58, 0.86, 0.63, 0.72, 0.80, 0.55] (descending rank f0 > f2 > f5 > f4 > f3 > f1 > f6, so validity rank is decoupled from screen position and positional heuristics cannot mimic validity coherence). One very high (0.94) and two very low (0.58, 0.55) validities make the convex weighting bite: w = (v-.5)^3 gives f0 = .0852 vs f6 = .000125, a ~680:1 ratio. 6 configurations x 2 side mirrors = 12 unique pairs x 8 repetitions = 96 trials.

CELLS (A-favouring version listed; the even-numbered mirror swaps A and B):
- T1/T2  SPARSE-COHERENT (nA,nB,nT) = (2,1,4): A wins f0(.94), f2(.86); B wins f6(.55); rest tied.
- T3/T4  SPARSE-CONFLICT  (2,1,4): A wins f1(.58), f6(.55); B wins f0(.94); rest tied. IDENTICAL composition to T1/T2.
- T5/T6  DENSE-COHERENT   (4,2,1): A wins f0,f2,f5,f4; B wins f3,f1; f6 tied.
- T7/T8  DENSE-CONFLICT   (4,2,1): A wins f1,f3,f4,f6; B wins f0,f2; f5 tied. IDENTICAL composition to T5/T6.
- T9/T10 UNANIMOUS-SPARSE (2,0,5): A wins f2,f4; five tied cues, no cue opposes.
- T11/T12 UNANIMOUS-DENSE (4,0,3): A wins f0,f3,f5,f6; three tied cues, no cue opposes.

QUANTITATIVE PREDICTIONS.
CDCC (beta=2.6, gamma=1.5, kappa=0.7, rho=3.0, eps=0.10): (2,1,4) coherent 0.708 vs conflict 0.541; (4,2,1) coherent 0.651 vs conflict 0.540; (2,0,5) 0.928; (4,0,3) 0.888.
LSNC (exact enumeration, k=2.5, nu=0.20, eps=0): (2,1,4) 0.579 for BOTH coherent and conflict; (4,2,1) 0.621 for BOTH; (2,0,5) 0.680; (4,0,3) 0.770.

THREE ORTHOGONAL DIAGNOSTICS:
(1) COHERENCE (parameter-free for LSNC): holding (nA,nB,nT) exactly fixed, CDCC predicts a 11-17 point coherent-minus-conflict gap that cannot be switched off (kappa >= 0.4 by hypothesis), while LSNC predicts EXACTLY ZERO for every k, nu, epsilon. Each subject contributes 32 coherent and 32 conflict trials on matched compositions, so this contrast is estimated within subject with high power.
(2) SPARSITY REVERSAL among the coherent cells: CDCC's D^gamma dilution makes (2,1,4) [D=2 winners vs 1 loser, D=3] MORE confident than (4,2,1) [D=6] despite the smaller raw margin (0.708 > 0.651); LSNC's tie dilution predicts the OPPOSITE order (0.579 < 0.621) because the sparse cell wastes 4 of 7 sampling slots on tied cues. Same reversal for the unanimous cells: CDCC predicts (2,0,5) > (4,0,3) (0.928 > 0.888, and 0.911 > 0.826 at gamma=1.75), LSNC predicts (4,0,3) > (2,0,5) (0.770 > 0.680). A margin-2 profile beating a margin-4 profile is impossible for LSNC under any parameters.
(3) CEILING: on the clean unanimous-sparse cell CDCC allows 0.90-0.95 adherence, whereas LSNC is capped near 0.68 (and at most ~0.79 even at k=3.5, nu=0.12) because a small noisy sample easily reverses the sign. Adherence above ~0.85 on T9/T10 is outside LSNC's reachable range.

CONTROLS: (i) every configuration appears in an A-favouring and a B-favouring mirror, so side bias cancels exactly; (ii) tied cues are encoded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s does not covary with composition or coherence; (iii) validity rank is scrambled relative to display position; (iv) no pair is degenerate: every trial has a strict count winner and requires reading several ratings; (v) systematic BELOW-chance adherence in the conflict cells (i.e. following the single best cue against the tally) would falsify BOTH theories in favour of one-reason/WADD integration, so the design is self-diagnostic.

INTERPRETATION: a large coherent > conflict gap that never dips below 0.5, together with sparse > dense ordering at matched or even smaller margins and near-0.9 adherence on unanimous sparse profiles, uniquely supports CDCC. Flat adherence across the coherence manipulation with a (4,0,3) > (2,0,5) > (4,2,1) > (2,1,4) composition ordering and a ceiling near 0.7 uniquely supports LSNC.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
Two structurally signed contrasts are summed, each of which has an opposite (or null) sign under the two theories, so the composite separates them with a large mean gap while each component is estimated from 32/32 and 16/16 trials per subject (tight per-subject noise).

D1 = adherence(count-winner) on validity-COHERENT conflicting profiles minus adherence on validity-INCOHERENT profiles, with the (nA,nB,nT) composition held EXACTLY matched across the two sets by design (T1/T2 vs T3/T4 both (2,1,4); T5/T6 vs T7/T8 both (4,2,1)). LSNC samples cues uniformly and ignores validity entirely, so its predicted adherence depends only on (nA,nB,nT): D1 = 0 exactly, for every k, nu, epsilon. CDCC gates confidence by the convex validity-weighted coherence C, so D1 is strictly positive (~+0.11 to +0.17, larger for larger kappa) and cannot be turned off.

D2 = adherence on the UNANIMOUS-SPARSE profile (2 winning cues, 0 opposing, 5 tied) minus adherence on the UNANIMOUS-DENSE profile (4 winning cues, 0 opposing, 3 tied). For CDCC the dilution term S = M/D^gamma gives 2^(1-gamma) vs 4^(1-gamma) = (2^(1-gamma))^2, and since gamma > 1 the sparse profile ALWAYS yields the larger confidence: D2 >= 0 for every admissible parameter vector. For LSNC tied cues waste sampling slots, so the dense profile (4/7 informative slots) always beats the sparse one (2/7): D2 < 0 (~ -0.06 to -0.12) for every admissible parameter vector.

Hence the composite D1 + D2 is bounded below by 0 under CDCC (expected ~ +0.18) and is expected to be clearly negative under LSNC (~ -0.09), a gap of roughly 0.27 with per-subject variance dominated by binomial sampling over 96 trials (sd ~ 0.17), which gives a Welch t well above 4 at N=25. It is not trivially true for CDCC: a flat coherence effect or an ordering with dense > sparse on unanimous profiles would push the metric to zero/negative and falsify it.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V = np.array([0.94, 0.58, 0.86, 0.63, 0.72, 0.80, 0.55], dtype=float)

    coh_hit = 0.0; coh_n = 0
    con_hit = 0.0; con_n = 0
    us_hit = 0.0; us_n = 0
    ud_hit = 0.0; ud_n = 0

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(a_raw, dtype=float).ravel()
            b = np.asarray(b_raw, dtype=float).ravel()
        except Exception:
            continue
        n = int(a.size)
        if n == 0 or int(b.size) != n:
            continue
        if n == int(V.size):
            w = V - 0.5
        elif n > 1:
            w = np.linspace(0.95, 0.55, n) - 0.5
        else:
            w = np.array([0.25])

        aw = a > b
        bw = b > a
        nA = int(np.sum(aw))
        nB = int(np.sum(bw))
        if nA == nB:
            continue
        winner = 0 if nA > nB else 1
        try:
            r = int(resp)
        except Exception:
            continue
        if r not in (0, 1):
            continue
        hit = 1.0 if r == winner else 0.0
        D = nA + nB
        if winner == 0:
            Ww = float(np.sum(w[aw])); Wl = float(np.sum(w[bw])); n_lose = nB
        else:
            Ww = float(np.sum(w[bw])); Wl = float(np.sum(w[aw])); n_lose = nA

        if n_lose == 0:
            # unanimous profiles: sparse (few discriminating cues) vs dense
            if D <= 2:
                us_hit += hit; us_n += 1
            else:
                ud_hit += hit; ud_n += 1
        else:
            if Ww > Wl:
                coh_hit += hit; coh_n += 1
            elif Ww < Wl:
                con_hit += hit; con_n += 1

    if coh_n > 0 and con_n > 0:
        d1 = coh_hit / coh_n - con_hit / con_n
    else:
        d1 = 0.0
    if us_n > 0 and ud_n > 0:
        d2 = us_hit / us_n - ud_hit / ud_n
    else:
        d2 = 0.0

    return float(d1 + d2)


### RESULTS
- Predicted under pi_3 (simulated): -0.0675 (var=0.0417)
- Predicted under pi_4 (simulated): 0.1587 (var=0.0237)
- Observed on real data: -0.9375 (var=0.0693)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = [0.95, 0.85, 0.75, 0.62, 0.55]
    hits = []
    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(list(a_raw), dtype=float)
        b = np.asarray(list(b_raw), dtype=float)
        n = a.shape[0]
        if b.shape[0] != n or n == 0:
            continue
        if n == len(validities):
            v = np.asarray(validities, dtype=float)
        else:
            # fall back: assume features already ordered by descending validity
            v = np.arange(n, 0, -1).astype(float)
        order = np.argsort(-v, kind='stable')

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins != b_wins:
            continue  # only tally-tie trials

        winner = None
        for j in order:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            continue  # no discriminating cue at all

        try:
            resp = int(r)
        except Exception:
            continue
        hits.append(1.0 if resp == winner else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4738 (var=0.0073)
**Predicted under pi_3:** 0.5125 (var=0.0055)
**Predicted under pi_4:** 0.4913 (var=0.0090)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    val = np.array([0.93, 0.57, 0.79, 0.86, 0.62, 0.71], dtype=float)
    hits = []
    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(list(a_raw), dtype=float)
        b = np.asarray(list(b_raw), dtype=float)
        n = a.shape[0]
        if b.shape[0] != n or n == 0:
            continue
        if val.shape[0] == n:
            order = np.argsort(-val, kind='stable')
        else:
            order = np.arange(n)
        # tally margin (A wins minus B wins)
        margin = int(np.sum(a > b) - np.sum(b > a))
        if abs(margin) < 2:
            continue
        tally_winner = 0 if margin > 0 else 1
        # take-the-best winner
        ttb_winner = None
        for j in order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue
        if ttb_winner == tally_winner:
            continue  # keep only conflict trials
        try:
            resp = int(r)
        except Exception:
            continue
        hits.append(1.0 if resp == tally_winner else 0.0)
    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.6538 (var=0.0092)
**Predicted under pi_3:** 0.6669 (var=0.0098)
**Predicted under pi_4:** 0.6469 (var=0.0093)

### Experiment 5
**Design**
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    hits = []
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()
    for a, b, r in zip(a_col, b_col, r_col):
        av = np.asarray(a, dtype=float).ravel()
        bv = np.asarray(b, dtype=float).ravel()
        if av.size == 0 or av.size != bv.size:
            continue
        nA = float(np.sum(av > bv))
        nB = float(np.sum(bv > av))
        if nA == nB:
            continue
        winner = 0 if nA > nB else 1
        try:
            resp = int(r)
        except Exception:
            continue
        hits.append(1.0 if resp == winner else 0.0)
    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))

```

**Observed (real) value:** 0.7937 (var=0.0071)
**Predicted under pi_3:** 0.6379 (var=0.0030)
**Predicted under pi_4:** 0.7781 (var=0.0023)

### Experiment 6
**Design**
  A=[1, 1, 0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 1]  B=[0, 0, 1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Adherence to the full-profile tally winner.

    For every trial we recompute the global feature-wise win counts
    nA = #(A > B) and nB = #(B > A) over ALL features, identify the
    option favoured by the complete tally, and score whether the subject
    chose it. The returned scalar is the proportion of such trials on
    which the subject followed the complete-tally winner.

    Works identically on the pooled DataFrame and on a single subject's
    slice.
    """
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    hits = []
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        nA = float(np.sum(a > b))
        nB = float(np.sum(b > a))
        if nA == nB:
            continue  # undiagnostic (no complete-tally winner)
        winner = 0 if nA > nB else 1
        try:
            resp = int(r)
        except (TypeError, ValueError):
            continue
        if resp not in (0, 1):
            continue
        hits.append(1.0 if resp == winner else 0.0)

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.7546 (var=0.0064)
**Predicted under pi_3:** 0.6565 (var=0.0039)
**Predicted under pi_4:** 0.7438 (var=0.0061)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Looking across all six experiments, the two theories share a hard structural commitment: choice direction is set by the raw feature-win count, so adherence to the count winner can never fall below 0.5 (pi_3 by symmetry of uniform sampling, pi_4 by the floored tanh argument). Experiments 3-6 are largely compatible with that commitment: exp3 shows chance (0.474) on tally-tied profiles, exp4 shows above-chance (0.654) adherence to the tally even when the best cue opposes it, and exps 5-6 show 0.75-0.79 overall tally adherence. On these, pi_4 (CDCC) is clearly the better calibrated of the two: it hits 0.778 vs 0.794 (exp5) and 0.744 vs 0.755 (exp6), whereas pi_3 (LSNC) is structurally capped too low (0.638, 0.657) because a small noisy sample keeps reversing large margins \\u2014 LSNC's ceiling is a systematic misfit on any design with clean, high-margin (often dominance) profiles.\n\nExperiments 1 and 2, however, break BOTH theories, and in the same direction. The exp1 metric (coherent-minus-conflict adherence to the count winner) is observed at -0.763 (var 0.020, i.e. highly consistent across subjects) where pi_3 predicts exactly 0 (parameter-free) and pi_4 predicts +0.144. Exp2's composite is -0.938 vs pi_3 -0.068 and pi_4 +0.159. Decomposing exp1: adherence to the count winner must be roughly 0.15 in the coherent cells and roughly 0.91 in the conflict cells. In every A-favouring cell of exps 1 and 2 the option that subjects actually chose is the one that wins the low-validity cue f1 (0.57/0.58) and loses the top cue f0 (0.94), regardless of the count. In other words: (i) a single cue \\u2014 not the stated most-valid one \\u2014 dominated the choice, and (ii) adherence to the numeric count fell far BELOW chance in a whole, well-populated condition. Both facts are outside the reachable range of both theories: pi_3 forbids any cue-identity effect whatsoever (its coherence effect is exactly 0 for all k, nu, epsilon), and pi_4 both forbids sub-chance count adherence and predicts the opposite sign of the identity effect (its coherence gate is monotone in stated validity).\n\nSo the current pair is: pi_4 = good on the four generic experiments, qualitatively wrong (wrong sign, and structurally unable to go below 0.5) on the two cue-identity experiments; pi_3 = wrong on the identity experiments (predicts a null) AND miscalibrated on the generic ones (adherence ceiling too low). pi_3 is dominated: it is never the better account of any experiment. It is therefore the theory to replace. Replacing it with a genuinely different competitor \\u2014 one that permits a single heavily weighted focal cue to override a numeric majority, with the cue's weight only loosely (and possibly non-monotonically) tied to the stated validity \\u2014 is much more valuable than re-parameterising LSNC, because that is exactly the degree of freedom that the observed data demand and that neither incumbent possesses.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_3 (LSNC) with a theory built around ONE-REASON OVERRIDE ON A NOISY TALLY BASELINE with subjectively (mis)assigned cue weights. Sketch:\n\n1) Weighted evidence, not raw counting. On each trial the subject computes a weighted difference E = sum_j w_j * (a_j - b_j) and chooses via a softmax/logistic on E with a lapse term. Because the w_j are unequal, a single strongly weighted cue CAN outvote a numeric majority \\u2014 i.e. adherence to the feature-count winner is allowed to fall BELOW 0.5. This is the crucial capability both incumbents lack (exp1 coherent cells ~0.15, exp2 D1 ~ -0.8).\n\n2) The subjective weights are only weakly and imperfectly bound to the STATED validities. Model w_j = exp(alpha*(v_j - vbar)/s_v + phi_j), where alpha is a free validity-sensitivity coefficient that is allowed to be near zero or NEGATIVE, and phi_j is a display-position salience profile (free, e.g. a small primacy/serial-position vector). Motivation: when the validity list is scrambled relative to display order (exps 1, 2, 4) subjects mis-bind validity values to expert positions, so the cue they treat as decisive need not be the objectively most valid one; when validities are presented in descending order (exp3) binding is coherent. This single mechanism can produce (a) chance on tally-tied, mirrored profiles when the weighted evidence approximately cancels (exp3 = 0.474), (b) ~0.75-0.80 adherence on generic/near-dominance profiles where all weightings agree with the count (exps 5, 6), (c) intermediate ~0.65 tally adherence on TTB-conflict trials with large margins (exp4), and (d) strongly sub-chance count adherence in exps 1/2 where, by construction, the count winner owns f0/f2 while the option that subjects actually chose owns f1/f3.\n\n3) Add a mixture/override component if useful: with probability lambda the decision follows the single highest-subjective-weight discriminating cue (one-reason), otherwise the weighted sum; plus a lapse epsilon. Ties in the weighted evidence -> exact coin flip, preserving the exp3 chance result.\n\nParameter ranges should let alpha span negative-through-positive values, keep the position profile small but non-degenerate, and keep the softmax slope high enough to reach 0.9+ adherence on dominance trials (exps 5, 6) while a lapse caps it below 1. The decisive novel prediction versus pi_4 is that adherence to the count winner can go BELOW 0.5 whenever a single subjectively-dominant cue opposes the count \\u2014 exactly the pattern in exps 1 and 2 \\u2014 while pi_4 remains floored at chance."
}
```

## Usage

```json
{
  "input_tokens": 20058,
  "output_tokens": 28752
}
```
