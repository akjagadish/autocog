# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


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

## THEORY 2 — pi_2
People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
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


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=6):** [0.92, 0.6, 0.83, 0.71, 0.55, 0.77]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  trial 2: A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  trial 3: A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 4: A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  trial 5: A=[1, 0, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  trial 6: A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1]
  trial 7: A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 8: A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]
  trial 9: A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  trial 10: A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 1, 1]
  trial 11: A=[1, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 12: A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]

**Rationale:** GOAL: dissociate Limited-Sample Noisy Cue Counting (LSNC, advocated) from softmax Tallying (competing). Tallying's choice probability is a function of the tally margin M = nA - nB ALONE: any two pairs with the same M must produce exactly the same P(choose A) for every possible beta and epsilon. LSNC instead depends on the full composition (nA, nB, nT): tied/uninformative cues occupy sampling slots (tie dilution) and only the PROPORTION of sampled informative cues won matters, so two profiles with identical M but different numbers of tied cues yield different adherence. The design is therefore a 3 (margin: M = 1, 2, 3) x 2 (informative density: SPARSE, all discriminating cues won by one option but many tied cues; DENSE, no/few tied cues but the loser also wins several cues) factorial, fully mirrored for side. Tallying predicts a pure main effect of margin and EXACTLY ZERO density effect (parameter-free); LSNC predicts sparse > dense at every margin.

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

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This is the overall adherence rate to the full-tally winner (the option that wins on strictly more features), pooled over the 96 trials of the design (every trial in this design has a non-zero tally margin, so no trial is dropped).

Why it separates the theories: LSNC's core commitment is that only a small, randomly drawn subset of cues (mean k between 1.5 and 3.5 of the 6 features) is ever inspected, that tied cues occupy sampling slots (tie dilution), and that each inspected comparison is mis-registered with probability nu = .12-.32. A small noisy sample frequently reverses or fails to recover the sign of the global margin, so LSNC's adherence is structurally CAPPED: exact enumeration for this design gives ~0.57-0.75 across the six cells, i.e. an aggregate near 0.65-0.68, and it cannot approach 1 for any admissible parameter setting. Tallying, in contrast, reads every feature and passes the exact margin (M = 1, 2 or 3 here, never 0) through a softmax; over the bulk of its inverse-temperature range the softmax is essentially deterministic, so its adherence sits near 1 - epsilon/2, i.e. roughly 0.85-0.90 on average and never structurally attenuated by the composition of the profile.

This is not trivially true of my theory: Tallying is perfectly free to produce low adherence (small beta, large epsilon), and LSNC is free to produce moderately high adherence (large k, small nu) - the metric measures a behavioural ceiling, not a theory label. The prediction that humans in a 6-cue, feedback-free, high-conflict task fall far short of margin-consistent responding (adherence well below 0.8 even at margin 3) is the substantive LSNC claim; Tallying, which by construction integrates all cues, has to attribute any such shortfall to a degenerate temperature that would then also flatten the margin gradient. The metric is a simple proportion over ~96 trials per subject, so its per-subject sampling noise is small (sd ~0.04-0.05), keeping the between-subject variance tight relative to the ~0.2 mean gap.

Source:
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


### RESULTS
- Predicted under pi_3 (simulated): 0.6379 (var=0.0030)
- Predicted under pi_2 (simulated): 0.8494 (var=0.0069)
- Observed on real data: 0.7937 (var=0.0071)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=8):** [0.93, 0.55, 0.78, 0.86, 0.6, 0.71, 0.66, 0.82]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 1, 0]
  trial 2: A=[0, 1, 0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0, 1, 0]
  trial 3: A=[1, 1, 0, 0, 1, 1, 0, 1]  B=[0, 0, 1, 0, 1, 1, 0, 0]
  trial 4: A=[0, 0, 1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1, 0, 1]
  trial 5: A=[1, 1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  trial 6: A=[0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 1, 1]
  trial 7: A=[1, 0, 1, 0, 1, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 1, 1]
  trial 9: A=[1, 1, 1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0]
  trial 10: A=[0, 0, 1, 0, 0, 1, 0, 0]  B=[1, 1, 1, 1, 1, 0, 1, 0]
  trial 11: A=[1, 1, 1, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0, 0, 1, 0]
  trial 12: A=[0, 0, 0, 1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 1, 1, 0, 1]

**Rationale:** GOAL: dissociate softmax Tallying (advocated) from Limited-Sample Noisy Cue Counting (LSNC, competing). Tallying's choice probability is a function of the tally margin M = nA - nB ALONE: P(A) = (1-eps)*sigmoid(beta*M) + eps/2. Therefore any two profiles with the same M MUST yield exactly the same P(A) for every beta and epsilon -- a parameter-free invariance. LSNC instead depends on the full composition (nA, nB, nT): because the inspected subset is drawn uniformly from ALL features, tied cues occupy sampling slots (tie dilution) and only the PROPORTION of sampled informative cues won matters, so adherence must fall monotonically as a fixed margin is realized with fewer ties and more loser-wins.

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

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
The single sharpest quantitative divergence between the two accounts is the ATTAINABLE LEVEL of adherence to the complete-profile tally winner on clearly discriminating pairs. Every trial in this design has a strict global tally winner with margin M = 2 or M = 4, and (by construction) the option favoured by the tally also wins the most valid cue, so the metric is not confounded by cue-order strategies.

Under softmax Tallying the decision variable is the FULL margin: P(winner) = (1-eps)*sigmoid(beta*M) + eps/2. Since M >= 2 on every trial, sigmoid(beta*M) is already essentially at ceiling for most of the beta range, so adherence is pinned near 1 - eps/2, i.e. roughly 0.85-0.90 on average and, crucially, only weakly dispersed (its across-subject spread comes from the lapse parameter alone, since 48 + 48 trials wash out binomial noise).

Under LSNC the inspected subset averages ~2.5 of 8 cues and each comparison is mis-registered with prob nu ~ 0.2, so a large global margin is routinely reversed or diluted by tied cues occupying sampling slots. Exact enumeration for this design gives adherence ~0.60-0.66 at M=2 and ~0.69-0.75 at M=4, i.e. a pooled value around 0.66-0.70 that is CAPPED well below what full integration produces, and again tightly distributed across subjects because the parameter ranges (k in [1.5,3.5], nu in [0.12,0.32]) move it only a few points.

So the expected contrast is roughly 0.87 vs 0.68 with per-subject variances of order 0.005-0.008 in both cases, which at N=25 is a very large Welch t. This is not trivially true of Tallying: had subjects been strongly noise-limited, Tallying's own epsilon/beta would drag adherence down toward the LSNC band; the prediction that adherence sits high above the sample-limited ceiling is a substantive commitment of full-cue integration.

Source:
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

### RESULTS
- Predicted under pi_3 (simulated): 0.6565 (var=0.0039)
- Predicted under pi_2 (simulated): 0.8538 (var=0.0068)
- Observed on real data: 0.7546 (var=0.0064)

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
**Predicted under pi_2:** 0.4988 (var=0.0068)

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
**Predicted under pi_2:** 0.8525 (var=0.0139)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Pooling all four experiments, neither theory is task-invariant, but they fail in opposite and informative ways.\n\nTallying (pi_2) predicts essentially the same high adherence (~0.85) in every design with a non-zero margin. It matches Exp1 reasonably (0.849 vs 0.794 observed), is off by ~0.10 in Exp2 (0.854 vs 0.755), correctly predicts chance on tally-tie trials in Exp3 (0.499 vs 0.474 observed), but fails catastrophically in Exp4 (0.853 vs 0.654 observed). Its structural problem is that adherence is a function of margin M alone: it cannot produce (a) the systematic sub-ceiling attenuation seen in Exp1/Exp2 without a degenerate beta that would flatten everything, nor (b) the 0.14\ndrop in Exp4 relative to Exp1 even though the Exp4 conflict trials have LARGER margins (2 and 3) than the average Exp1 trial.\n\nLSNC (pi_3) has the right qualitative shape \\u2014 capped adherence, exact chance on tally ties (0.513 vs 0.474 in Exp3), and it nails the Exp4 conflict cell (0.667 vs 0.654) \\u2014 but it is systematically too pessimistic where the tally is easy: 0.638 vs 0.794 (Exp1) and 0.657 vs 0.755 (Exp2). Worse, its composition/dilution mechanism gets the CROSS-experiment ordering backwards: the Exp4 conflict trials have a higher informative-win proportion (0.75\\u20130.80) than the average Exp1/Exp2 trial (~0.65\\u20130.69), so uniform-subset sampling predicts Exp4 > Exp1, whereas the data show Exp4 (0.654) well BELOW Exp1 (0.794). Raising k / lowering nu to fix Exp1/Exp2 would push Exp4 to ~0.80\\u2013.85 and break it. So LSNC's single free mechanism cannot cover all four datasets either, though its aggregate error (mean |err| \\u2248 0.077) is smaller than Tallying's (\\u2248 0.095) and its errors are all in one direction (uniform under-prediction), which is a repairable calibration issue rather than a structural one.\n\nThe empirical regularities that any winning theory must reproduce are: (i) EXACT chance on globally tally-tied profiles even when a highly valid cue (0.95 vs 0.85) discriminates \\u2014 no Take-The-Best or WADD tie-break at all (Exp3 = 0.474); (ii) high but clearly sub-ceiling adherence (0.75\\u20130.79) when the most valid cue AGREES with the count (Exp1, Exp2); (iii) a large drop toward (but not below) chance, ~0.65, when the most valid cue OPPOSES the count, even at margin 2\\u20133 (Exp4). Together (i) and (iii) rule out both pure margin-only Tallying and any additive validity-weighted (WADD/TTB-mixture) account: a TTB mixture strong enough to produce (iii) would push Exp3 to ~0.63, contradicting (i). What is needed is a theory in which the DIRECTION of choice is purely count-based (hence exact chance on ties, no validity tie-break) but the RELIABILITY/confidence with which that direction is executed is degraded by conflict with high-validity cues and by evidence dilution.\n\nSince Tallying is the weaker and structurally rigid account (it cannot produce attenuation or the conflict effect at all), it should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Tallying (pi_2) with a new theory: **Count-Direction with Conflict-Graded Confidence (CDCC)** \\u2014 a noisy tally whose SIGN is purely count-based but whose reliability is modulated by cue-validity coherence and evidence dilution.\n\nCore commitments (state them in prose, distinct from both current theories):\n1. Direction is decided only by the sign of the feature-wise win count (nA vs nB). When the counts tie, the subject perceives no basis for preference and guesses \\u2014 exactly 0.5, with NO validity-ordered tie-break. This reproduces Exp3 (0.474) and is a hard, parameter-free prediction that separates CDCC from Take-The-Best/WADD.\n2. Confidence, not direction, carries validity information. The probability of executing the count-winner is a saturating function of a scalar confidence signal that increases with the margin M, decreases with dilution (number of cues the loser also wins, i.e. it depends on M/(nA+nB) as well as M), and is ATTENUATED when high-validity cues point against the count winner. Concretely: p = 0.5 + 0.5*(1-eps)*tanh( beta * (|M| / (nA+nB)^gamma) * (1 + kappa*C) ), where C is a normalised coherence term, e.g. C = (sum of w_j over cues won by the count winner \\u2212 sum over cues won by the loser)/(sum of all discriminating w_j) with w_j a monotone function of stated validity (e.g. v_j\\u22120.5). Crucially p is clipped at >= 0.5: validity conflict drives behaviour toward guessing but NEVER toward systematic reversal (no TTB choices), which is what distinguishes CDCC from any Tallying+TTB mixture and is required jointly by Exp3 and Exp4.\n3. A lapse eps replaces the decision with a coin flip.\n\nCalibration targets the parameter ranges must be able to hit simultaneously: Exp1 \\u2248 0.79, Exp2 \\u2248 0.75, Exp3 = 0.50 exactly, Exp4 conflict cell \\u2248 0.65. Suggested ranges: beta in [0.8, 3.0], gamma in [0.2, 0.8] (dilution exponent), kappa in [0.5, 2.0] (coherence gain), eps in [0.0, 0.15]. Note the design signature: in Exp1/Exp2 the count winner always also wins the top cue (C > 0, confidence boosted -> ~0.75\\u20130.80), while in the Exp4 conflict trials the top cue opposes (C small or negative -> ~0.60\\u20130.70), and pair 11/12 of Exp4 (nA=1, nB=3, M=2) has near-zero coherence and should land near 0.55\\u20130.60 \\u2014 exactly the pattern that neither margin-only Tallying nor uniform-sampling LSNC can produce. CDCC should also yield a small sparse>dense composition effect (via the dilution exponent gamma) so it remains a genuine competitor to LSNC on its own home-turf designs, while beating it on cross-experiment level ordering."
}
```

## Usage

```json
{
  "input_tokens": 14473,
  "output_tokens": 18141
}
```
