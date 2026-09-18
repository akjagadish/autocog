# experiment_attempt_00

## System Prompt

You are a renowned cognitive scientist designing an experiment in the Decision Making (Binary Features) domain.

Your goal is to be an adversarial collaborator: propose a design whose outcomes would be predicted by your advocated theory but NOT by the competing theory. Both are provided below.

A useful proposal targets a *quantitative* dissociation between the two theories — how they respond differently to specific stimuli in addition to differences in overall performance.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

A multi-attribute decision-making experiment. On each trial the subject sees two options (A, B), each described by `n_features` integer expert ratings (`n_features` is set by the length of `validities` you propose). Choose `validities` — one per feature, each in [0.5, 1.0], order free — to fix each expert's advertised accuracy; subjects are told these values up front. Then choose `trial_a_ratings/trial_b_ratings` (each rating value in [0, 1]) so that the intended decision-making theories make distinguishable predictions: avoid degenerate pairs where every theory agrees, and prefer pairs that maximally dissociate the candidate theories. Validities and the trial ratings together define the design; they are fixed across all trials in this experiment. No trial-by-trial correctness feedback. The total number of trials per subject is held at roughly 96: each unique pair is repeated K = max(1, 96 // n_unique_pairs) times in an independently-randomized order per subject.

Subjects see the following instructions:
In this experiment you will repeatedly choose between two fictitious products, A and B. On every trial you will see `n_features` expert ratings for each product (the number of experts is fixed across all trials and is set by the length of `validities`).

Each rating is an integer in [0, 1]. The ratings are displayed as a horizontal filled bar with the numeric value (e.g. "0/1") shown next to it. Higher = more positive.

The same experts (in the same order) provide ratings for both products on every trial. Each expert's accuracy (their validity expressed as a percentage, e.g. "Expert 1 (80%)") is shown next to their rating on every trial AND is also listed up front in an "Expert accuracies" panel.

On each trial, decide which product is of higher quality and press A for product A or B for product B. There is no time limit and no feedback. Note that for the first ~`min_rt_ms` of each trial the answer prompt is hidden and the keys are locked, so subjects first see the full ratings and can answer once the A / B prompt appears — design pairs that actually require comparing the ratings.

Total trials per subject is roughly `MAX_TRIALS`: each unique pair you propose is repeated `K = max(1, MAX_TRIALS // n_unique_pairs)` times in an independently-randomized order per subject.

## ADVOCATED THEORY
**Description:** **Count-Direction with Conflict-Graded Confidence (CDCC).**

When people compare two multi-attribute options described by binary expert ratings, the *direction* of their preference and the *confidence* with which they execute it are computed by two functionally separate processes.

1. **Direction is purely count-based.** The comparator registers, for each attribute, only which option is rated higher, and the option that wins on more attributes becomes the candidate response. Validities never enter this stage: they cannot promote a minority-supported option to candidate status. When the two win-counts are equal, the comparator returns no candidate at all, and the subject flips a mental coin — exactly 0.5, with **no** validity-ordered tie-break. This is a hard, parameter-free prediction: on tally-tied, side-counterbalanced profiles CDCC predicts chance adherence to the most-valid discriminating cue, regardless of any parameter setting.

2. **Confidence, not direction, carries the validity information, and it is graded by evidence dilution and by cue conflict.** The probability of actually executing the count-winner is a saturating (tanh) function of a scalar confidence signal built from three ingredients: (i) the raw win margin M = |nA − nB|; (ii) *dilution* — the margin is discounted by the total number of discriminating cues raised to a power gamma, so that a 6-vs-2 split feels much weaker than a 4-vs-0 split even though both have M = 4 (evidence spread thin across many mutually contradicting attributes is subjectively less compelling than a clean, sparse, unanimous split); and (iii) *coherence* — the signed, normalised balance of validity weight between the cues the count-winner wins and the cues the loser wins, where the subjective weight of a cue is a convex (super-linear) function of its stated validity advantage, w_j = (v_j − 0.5)^rho. Convexity means the coherence signal is dominated by the few high-validity cues: when the best cue points *with* the count the subject feels certain, when it points *against* the count the subject feels torn even though the count itself is unambiguous.

3. **Conflict erodes confidence but never reverses direction.** The confidence argument is floored at zero, so validity conflict can drive behaviour arbitrarily close to guessing but never below chance: CDCC never makes systematic one-reason (Take-The-Best) choices. This is what separates it from any Tallying+TTB mixture: on high-margin trials where the top cue opposes the tally, people move toward 0.5 from above, not past it.

4. **Lapses.** With probability epsilon the entire decision is replaced by a coin flip (attention lapse), which caps adherence below 1 even on maximally coherent trials.

The theory therefore predicts a specific cross-design signature: chance performance on tally-tied profiles; high but sub-ceiling adherence on sparse, coherent, unanimous profiles; noticeably lower adherence on dense profiles where the loser also wins several cues even when the raw margin is larger; and the lowest (but still above-chance) adherence when the most valid cue opposes the count.

**Parameters:**
- beta: [1.6, 3.6]
- gamma: [1.25, 1.75]
- kappa: [0.4, 1.0]
- rho: [2.2, 3.8]
- epsilon: [0.02, 0.18]
- validities: validities

**`predict source code`:**
```python
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
```

**`policy source code`:**
```python
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
```


## COMPETING THEORY
**Description:** **Limited-Sample Noisy Cue Counting (LSNC).** People neither integrate all cues (Tallying) nor consult a single best cue (Take-The-Best). Instead, on each choice they inspect only a *limited, randomly selected subset* of the presented expert ratings, tally feature-wise wins **within that sample only**, and pick the option that is ahead in the sample; a within-sample tie (including a sample containing no discriminating cue) forces a guess.

Three commitments define the theory:
1. **Equal-probability sampling.** The inspected subset is drawn uniformly at random from the whole profile — it is *not* validity-ordered and *not* validity-weighted. Stated validities are too weakly represented to steer the search order, so no cue enjoys priority. Consequently, on globally tally-tied, side-counterbalanced profiles the process is exactly symmetric and choice is 50/50, with no tendency to follow the most valid discriminating cue.
2. **Sample-size limitation with tie dilution.** The number of cues actually inspected, K, is a random variable with mean k < n (implemented as K = 1 + Binomial(n−1, q), so at least one cue is always read). Because the subset is drawn from *all* features, non-discriminating (tied) features occupy sampling slots and dilute the evidence: two profiles with the same raw tally margin M produce different adherence depending on how many uninformative cues pad the profile, and on the proportion (not the raw count) of informative cues won. This is the signature that dissociates LSNC from softmax Tallying, whose predictions depend only on M.
3. **Encoding noise + lapse.** Each inspected feature-wise comparison is registered with the wrong sign with probability nu (attention/encoding noise), and with probability epsilon the whole decision is replaced by a coin flip.

Because a small, noisy sample can easily reverse the sign of a large global margin, adherence to the full-tally winner is *capped well below 1* even on high-margin conflict trials — the systematic attenuation that full Tallying cannot produce without degenerate temperature — while symmetry guarantees exactly chance behaviour on tally-tied pairs.

**Parameters:**
- k: [1.5, 3.5]
- nu: [0.12, 0.32]
- epsilon: [0.0, 0.12]

**`predict source code`:**
```python
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
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## ALREADY-EXPLORED EXPERIMENTS (do not repeat)
[0] Five experts with a wide validity spread (0.95, 0.85, 0.75, 0.62, 0.55) make the cue hierarchy salient, so Take The Best (TTB) has an unambiguous search order, while Tallying is indifferent to that order. The 12 unique pairs (repeated 8x each, ~96 trials) fall into three families, all of which dissociate the two models.

(1) Direct-conflict pairs (T1-T6, T11-T12): one option wins the highest discriminating cue while the other wins the majority of the remaining lower-validity cues. TTB predicts choice of the top-cue winner on every one of these; Tallying predicts the opposite option. Critically, the tally margin is parametrically varied: margin 3 (T1/T2: 1 vs 4 cue wins), margin 2 (T5/T6: 1 vs 3; T11/T12: 1 vs 3 with the top cue tied so TTB's deciding cue is expert 2), and margin 1 (T3/T4: 2 vs 3). This yields the key *quantitative* dissociation: TTB's softmax operates on a binary 1/0 score, so its predicted choice probability for the top-cue winner is identical (sigmoid(beta), lapse-corrected) across all conflict trials regardless of tally margin, whereas Tallying's softmax probability grows monotonically with the tally margin (exp(beta*1) vs exp(beta*2) vs exp(beta*3) contrasts). Thus a flat, margin-invariant preference for the top-cue winner is uniquely TTB, while a graded, margin-increasing preference for the majority-cue winner is uniquely Tallying.

(2) Tally-tie pairs (T7-T10): the two options split cue wins evenly (1-1 or 2-2), so Tallying is forced to guess at exactly 50/50 irrespective of its parameters, while TTB still decides confidently via the highest discriminating cue. Systematic above-chance choice of the top-cue winner here cannot be produced by Tallying under any beta/epsilon, making these trials a parameter-free falsification test.

(3) Left/right and direction counterbalancing: every conflict and tie configuration appears in both A-favoring and B-favoring versions, so any side bias cancels and cannot masquerade as evidence for either heuristic.

T11/T12 additionally probe whether the deciding cue must be the very best cue: expert 1 ties, so TTB's decision rests on expert 2 while the tally strongly favors the other option; this separates TTB from any 'only-the-single-best-cue-matters' caricature as well as from Tallying.

[1] Goal: dissociate Tallying (advocated) from Take The Best by factorially crossing (i) which option wins the highest-validity discriminating cue \u2014 the only thing TTB is sensitive to \u2014 with (ii) the tally margin, the only thing Tallying is sensitive to. Six experts with a wide, deliberately scrambled validity spread [0.93, 0.57, 0.79, 0.86, 0.62, 0.71] give the descending cue order f0 > f3 > f2 > f5 > f4 > f1, so validity rank is decoupled from screen position (unlike a monotone left-to-right layout, this prevents a positional-ordering confound from mimicking TTB).

Core manipulation (trials 1,3,5,7,9 and their exact A/B mirrors 2,4,6,8,10): option A always wins expert f0 (validity .93), so TTB's deciding cue and predicted winner are IDENTICAL across all five cells; its predicted P(choose A) is the constant (1-eps)*sigmoid(beta) + eps/2 regardless of the rest of the profile. Meanwhile the tally margin is parametrically swept M = -3, -1, 0, +2, +4 by re-allocating the five lower-validity cues. Tallying's softmax gives P(A) = sigmoid(beta*M) (lapse-corrected), i.e. a strictly monotone, sign-crossing psychometric function of M: strong B-preference at M=-3, weak B at M=-1, exactly 50/50 at M=0, and increasingly strong A at +2 and +4. A flat line across the five margin cells is a unique TTB signature; a monotone increasing line crossing chance at M=0 is a unique Tallying signature. Crucially the design contains both conflict cells (M=-3, -1: TTB says A, Tallying says B) and agreement cells (M=+2, +4: both say A), so the same subject's data pin down beta from the agreement cells and then test whether the conflict cells reverse (Tallying) or not (TTB) \u2014 a within-subject, parameter-anchored test rather than a mere direction contrast.

Tally-tie cells (trials 5, 6): three cue wins each. Tallying MUST be exactly 50/50 for every possible beta and epsilon, while TTB confidently picks the f0 winner. Systematic above-chance choice here parameter-free falsifies Tallying; chance performance parameter-free falsifies TTB.

Low-rank deciding cue (trials 11, 12): the two best experts (f0=.93 and f3=.86) are tied, so TTB must descend to f2 (.79), which favors A, while the tally favors B by 2 (B wins f5, f4, f1). This checks whether the one-reason rule really cascades down the validity hierarchy or whether subjects instead aggregate the three lower experts \u2014 an independent conflict test whose TTB prediction is the same magnitude as in trials 1/3 (binary score) but whose Tallying prediction is intermediate (|M|=2).

Every configuration appears in both an A-favoring and a B-favoring mirror, so side bias cancels exactly and cannot masquerade as support for either heuristic. Tie cues are encoded sometimes as 0-0 and sometimes as 1-1 to prevent a 'count the 1s' surface strategy from being confounded with either model. 12 unique pairs \u00d7 8 repetitions \u2248 96 trials.

[2] GOAL: dissociate Limited-Sample Noisy Cue Counting (LSNC, advocated) from softmax Tallying (competing). Tallying's choice probability is a function of the tally margin M = nA - nB ALONE: any two pairs with the same M must produce exactly the same P(choose A) for every possible beta and epsilon. LSNC instead depends on the full composition (nA, nB, nT): tied/uninformative cues occupy sampling slots (tie dilution) and only the PROPORTION of sampled informative cues won matters, so two profiles with identical M but different numbers of tied cues yield different adherence. The design is therefore a 3 (margin: M = 1, 2, 3) x 2 (informative density: SPARSE, all discriminating cues won by one option but many tied cues; DENSE, no/few tied cues but the loser also wins several cues) factorial, fully mirrored for side. Tallying predicts a pure main effect of margin and EXACTLY ZERO density effect (parameter-free); LSNC predicts sparse > dense at every margin.

CELLS (A-favouring version; the mirror swaps A and B):
- S1 sparse M=+1: (nA,nB,nT) = (1,0,5)
- D1 dense  M=+1: (3,2,1)
- S2 sparse M=+2: (2,0,4)
- D2 dense  M=+2: (4,2,0)
- S3 sparse M=+3: (3,0,3)
- D3 dense  M=+3: (4,1,1)

QUANTITATIVE LSNC PREDICTIONS (k=2.5, nu=0.20, epsilon=0, exact enumeration of the hypergeometric sample compositions): M=1: 0.625 (sparse) vs 0.572 (dense); M=2: 0.702 vs 0.630; M=3: 0.754 vs 0.707. So LSNC predicts a ~5-7 point sparse-minus-dense advantage at EVERY margin (and the gap grows with k), while adherence stays capped well below 1 even at M=3. Tallying predicts 0 at every margin. Aggregated over the 3 margins each subject contributes 48 sparse and 48 dense trials, so the parameter-free density contrast is powerfully estimated within subject.

CONFOUND CONTROL (critical): (i) In every A-favouring cell option A wins the highest-validity discriminating cue f0 (0.92), which also discriminates in every cell, so Take-The-Best (or any 'best cue' strategy) predicts a CONSTANT P(A) across all six cells and cannot generate a density effect. (ii) Validity-weighted additive evidence (log-odds weights w = [2.44, 0.41, 1.59, 0.90, 0.20, 1.21]) was deliberately matched between the sparse and dense member of each margin pair: M=1: +2.44 vs +2.13; M=2: +2.64 vs +2.53; M=3: +3.05 vs +3.16. Hence a WADD-type subject also cannot produce the sparse>dense pattern; only sample-limited counting with tie dilution can. (iii) Validity order is scrambled relative to screen position (descending order is f0, f2, f5, f3, f1, f4) so positional heuristics do not mimic validity-ordered search. (iv) Tied cues are encoded sometimes as 1-1 and sometimes as 0-0 so that the number of displayed 1s is not a constant cue to density. (v) Every configuration appears in both an A-favouring and a B-favouring mirror (12 unique pairs x 8 repetitions = 96 trials), so side bias cancels exactly.

INTERPRETATION: a graded, margin-monotone function that is IDENTICAL for sparse and dense cells supports Tallying; a margin-monotone function that is systematically shifted upward in the sparse (tie-padded, proportion-1.0) cells relative to the dense cells with the same margin, together with adherence ceilings around 0.70-0.80 at M=3, uniquely supports LSNC and cannot be produced by Tallying under any beta/epsilon.

[3] GOAL: dissociate softmax Tallying (advocated) from Limited-Sample Noisy Cue Counting (LSNC, competing). Tallying's choice probability is a function of the tally margin M = nA - nB ALONE: P(A) = (1-eps)*sigmoid(beta*M) + eps/2. Therefore any two profiles with the same M MUST yield exactly the same P(A) for every beta and epsilon -- a parameter-free invariance. LSNC instead depends on the full composition (nA, nB, nT): because the inspected subset is drawn uniformly from ALL features, tied cues occupy sampling slots (tie dilution) and only the PROPORTION of sampled informative cues won matters, so adherence must fall monotonically as a fixed margin is realized with fewer ties and more loser-wins.

DESIGN: 8 experts, scrambled validity spread [0.93, 0.55, 0.78, 0.86, 0.60, 0.71, 0.66, 0.82] (descending rank f0 > f3 > f7 > f2 > f5 > f6 > f4 > f1, so validity order is decoupled from screen position). 2 (margin: M = 2, 4) x 3 (composition: SPARSE / MID / DENSE) factorial, each cell fully mirrored for side => 12 unique pairs x 8 repetitions ~ 96 trials.

Cells (A-favouring version; mirror swaps A and B):
- S2 sparse M=2: (nA,nB,nT) = (2,0,6)
- M2 mid    M=2: (3,1,4)
- D2 dense  M=2: (5,3,0)
- S4 sparse M=4: (4,0,4)
- M4 mid    M=4: (5,1,2)
- D4 dense  M=4: (6,2,0)

QUANTITATIVE PREDICTIONS. Tallying: a pure margin main effect and EXACTLY ZERO composition effect -- the three M=2 cells must be statistically identical, the three M=4 cells must be statistically identical, and the M=4 level must exceed M=2 by the sigmoid gap sigmoid(4b) - sigmoid(2b). LSNC (exact enumeration, k=2.5, nu=0.20, eps=0): M=2 -> 0.661 (sparse) vs ~0.62 (mid) vs 0.598 (dense); M=4 -> 0.750 (sparse) vs ~0.72 (mid) vs 0.689 (dense). I.e. LSNC predicts a monotone ~6-point decline across the composition ladder at BOTH margins, plus adherence capped near 0.75-0.80 even at M=4. Note that at sample size K=1 all same-margin cells coincide (P = 0.5 + 0.3*M/n), so any composition effect in the data is diagnostic of multi-cue but sub-total sampling; a flat ladder is diagnostic of full integration (Tallying).

CONFOUND CONTROLS. (i) In every A-favouring cell option A wins the highest-validity cue f0 (0.93), which discriminates in all six cells, so Take-The-Best / any 'best cue' rule predicts a CONSTANT P(A) across all cells and cannot generate either the margin effect or a composition ladder. (ii) Validity-weighted additive evidence (log-odds weights) was deliberately matched WITHIN each margin level: M=2 cells give net +2.99 (sparse), +3.04 (mid), +3.12 (dense); M=4 cells give +4.55, +4.78, +4.39. Hence WADD-type integration also predicts no composition effect, so a sparse>mid>dense ordering can only come from sample-limited counting with tie dilution, while a flat ladder plus a clear margin effect uniquely supports Tallying. (iii) Tied cues are encoded sometimes 1-1 and sometimes 0-0 so that display density of 1s is not itself a cue to composition. (iv) Every configuration appears in both an A-favouring and a B-favouring mirror, so side bias cancels exactly.

INTERPRETATION: margin-graded but composition-invariant adherence (with the M=4 cells approaching the ceiling implied by the beta estimated from the M=2 cells) supports Tallying; a systematic sparse > mid > dense decline at each margin, with adherence capped around 0.75 at M=4, supports LSNC.

[4] GOAL: dissociate Limited-Sample Noisy Cue Counting (LSNC, advocated) from Count-Direction with Conflict-Graded Confidence (CDCC, competing). Both theories agree on the count-based DIRECTION of choice and both cap adherence below 1, so the classic tally-vs-TTB contrasts are useless here. They differ on exactly two things, and this design isolates both.

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

## RESPONSE FORMAT
Return a JSON object with the following fields:
- rationale: Reasoning behind this experimental design; used when interpreting results.
- validities: Per-expert validities, one per feature. Each must be in [0.5, 1.0]; order is free (no descending requirement — theories that depend on validity order re-sort internally). Length determines n_features for the whole experiment and must match the rating-list length in every trial pair. Pick a spread (at least one high validity and at least one low one) — uniform validities (e.g. all 0.7) make many candidate theories indistinguishable, so spread them out.
- trial_a_ratings: List of option-A rating vectors, one per trial. Each inner list has length equal to len(validities); each value is binary (0 or 1). `trial_a_ratings[i]` pairs with `trial_b_ratings[i]` on trial i.
- trial_b_ratings: List of option-B rating vectors, one per trial. Same length and shape constraints as `trial_a_ratings`. Pick trials that let the intended theories be dissociated; avoid trials where every theory agrees.


## Response

```json
{
  "rationale": "GOAL: dissociate Count-Direction with Conflict-Graded Confidence (CDCC, advocated) from Limited-Sample Noisy Cue Counting (LSNC, competing). Both theories agree that the DIRECTION of choice is the count winner and both cap adherence below 1, so tally-vs-TTB style contrasts are useless. They differ on exactly what modulates adherence: LSNC's probability is a function of the composition (nA, nB, nT) ALONE (uniform sampling makes cue identity/validity irrelevant), whereas CDCC's confidence is (i) gated by validity COHERENCE C = (W_win - W_lose)/(W_win + W_lose) with convex weights w = (v-0.5)^rho, and (ii) diluted by S = M/D^gamma where D = nA+nB, so tied cues are completely irrelevant to CDCC but occupy sampling slots for LSNC. This design attacks both differences and, crucially, contains two ORDERING REVERSALS.\n\nDESIGN: 7 experts, scrambled validity spread [0.94, 0.58, 0.86, 0.63, 0.72, 0.80, 0.55] (descending rank f0 > f2 > f5 > f4 > f3 > f1 > f6, so validity rank is decoupled from screen position and positional heuristics cannot mimic validity coherence). One very high (0.94) and two very low (0.58, 0.55) validities make the convex weighting bite: w = (v-.5)^3 gives f0 = .0852 vs f6 = .000125, a ~680:1 ratio. 6 configurations x 2 side mirrors = 12 unique pairs x 8 repetitions = 96 trials.\n\nCELLS (A-favouring version listed; the even-numbered mirror swaps A and B):\n- T1/T2  SPARSE-COHERENT (nA,nB,nT) = (2,1,4): A wins f0(.94), f2(.86); B wins f6(.55); rest tied.\n- T3/T4  SPARSE-CONFLICT  (2,1,4): A wins f1(.58), f6(.55); B wins f0(.94); rest tied. IDENTICAL composition to T1/T2.\n- T5/T6  DENSE-COHERENT   (4,2,1): A wins f0,f2,f5,f4; B wins f3,f1; f6 tied.\n- T7/T8  DENSE-CONFLICT   (4,2,1): A wins f1,f3,f4,f6; B wins f0,f2; f5 tied. IDENTICAL composition to T5/T6.\n- T9/T10 UNANIMOUS-SPARSE (2,0,5): A wins f2,f4; five tied cues, no cue opposes.\n- T11/T12 UNANIMOUS-DENSE (4,0,3): A wins f0,f3,f5,f6; three tied cues, no cue opposes.\n\nQUANTITATIVE PREDICTIONS.\nCDCC (beta=2.6, gamma=1.5, kappa=0.7, rho=3.0, eps=0.10): (2,1,4) coherent 0.708 vs conflict 0.541; (4,2,1) coherent 0.651 vs conflict 0.540; (2,0,5) 0.928; (4,0,3) 0.888.\nLSNC (exact enumeration, k=2.5, nu=0.20, eps=0): (2,1,4) 0.579 for BOTH coherent and conflict; (4,2,1) 0.621 for BOTH; (2,0,5) 0.680; (4,0,3) 0.770.\n\nTHREE ORTHOGONAL DIAGNOSTICS:\n(1) COHERENCE (parameter-free for LSNC): holding (nA,nB,nT) exactly fixed, CDCC predicts a 11-17 point coherent-minus-conflict gap that cannot be switched off (kappa >= 0.4 by hypothesis), while LSNC predicts EXACTLY ZERO for every k, nu, epsilon. Each subject contributes 32 coherent and 32 conflict trials on matched compositions, so this contrast is estimated within subject with high power.\n(2) SPARSITY REVERSAL among the coherent cells: CDCC's D^gamma dilution makes (2,1,4) [D=2 winners vs 1 loser, D=3] MORE confident than (4,2,1) [D=6] despite the smaller raw margin (0.708 > 0.651); LSNC's tie dilution predicts the OPPOSITE order (0.579 < 0.621) because the sparse cell wastes 4 of 7 sampling slots on tied cues. Same reversal for the unanimous cells: CDCC predicts (2,0,5) > (4,0,3) (0.928 > 0.888, and 0.911 > 0.826 at gamma=1.75), LSNC predicts (4,0,3) > (2,0,5) (0.770 > 0.680). A margin-2 profile beating a margin-4 profile is impossible for LSNC under any parameters.\n(3) CEILING: on the clean unanimous-sparse cell CDCC allows 0.90-0.95 adherence, whereas LSNC is capped near 0.68 (and at most ~0.79 even at k=3.5, nu=0.12) because a small noisy sample easily reverses the sign. Adherence above ~0.85 on T9/T10 is outside LSNC's reachable range.\n\nCONTROLS: (i) every configuration appears in an A-favouring and a B-favouring mirror, so side bias cancels exactly; (ii) tied cues are encoded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s does not covary with composition or coherence; (iii) validity rank is scrambled relative to display position; (iv) no pair is degenerate: every trial has a strict count winner and requires reading several ratings; (v) systematic BELOW-chance adherence in the conflict cells (i.e. following the single best cue against the tally) would falsify BOTH theories in favour of one-reason/WADD integration, so the design is self-diagnostic.\n\nINTERPRETATION: a large coherent > conflict gap that never dips below 0.5, together with sparse > dense ordering at matched or even smaller margins and near-0.9 adherence on unanimous sparse profiles, uniquely supports CDCC. Flat adherence across the coherence manipulation with a (4,0,3) > (2,0,5) > (4,2,1) > (2,1,4) composition ordering and a ceiling near 0.7 uniquely supports LSNC.",
  "validities": [
    0.94,
    0.58,
    0.86,
    0.63,
    0.72,
    0.8,
    0.55
  ],
  "trial_a_ratings": [
    [
      1,
      1,
      1,
      0,
      1,
      0,
      0
    ],
    [
      0,
      1,
      0,
      0,
      1,
      0,
      1
    ],
    [
      0,
      1,
      1,
      0,
      0,
      1,
      1
    ],
    [
      1,
      0,
      1,
      0,
      0,
      1,
      0
    ],
    [
      1,
      0,
      1,
      0,
      1,
      1,
      1
    ],
    [
      0,
      1,
      0,
      1,
      0,
      0,
      1
    ],
    [
      0,
      1,
      0,
      1,
      1,
      0,
      1
    ],
    [
      1,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      0,
      1,
      1,
      1,
      0,
      1
    ],
    [
      1,
      0,
      0,
      1,
      0,
      0,
      1
    ],
    [
      1,
      1,
      1,
      1,
      0,
      1,
      1
    ],
    [
      0,
      1,
      1,
      0,
      0,
      0,
      0
    ]
  ],
  "trial_b_ratings": [
    [
      0,
      1,
      0,
      0,
      1,
      0,
      1
    ],
    [
      1,
      1,
      1,
      0,
      1,
      0,
      0
    ],
    [
      1,
      0,
      1,
      0,
      0,
      1,
      0
    ],
    [
      0,
      1,
      1,
      0,
      0,
      1,
      1
    ],
    [
      0,
      1,
      0,
      1,
      0,
      0,
      1
    ],
    [
      1,
      0,
      1,
      0,
      1,
      1,
      1
    ],
    [
      1,
      0,
      1,
      0,
      0,
      0,
      0
    ],
    [
      0,
      1,
      0,
      1,
      1,
      0,
      1
    ],
    [
      1,
      0,
      0,
      1,
      0,
      0,
      1
    ],
    [
      1,
      0,
      1,
      1,
      1,
      0,
      1
    ],
    [
      0,
      1,
      1,
      0,
      0,
      0,
      0
    ],
    [
      1,
      1,
      1,
      1,
      0,
      1,
      1
    ]
  ]
}
```

## Usage

```json
{
  "input_tokens": 14229,
  "output_tokens": 15005
}
```
