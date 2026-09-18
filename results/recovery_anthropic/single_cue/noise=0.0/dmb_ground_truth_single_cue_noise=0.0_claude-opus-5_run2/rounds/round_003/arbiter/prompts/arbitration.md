# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
**Redundancy-Discounted Tiered Integration with Majority Discounting, Panel-Size Capacity Dilution, and Graded Redundancy Relaxation (RDTI-MCR).**

All previous commitments are retained. (i) Stated validities are encoded only *coarsely and ordinally*: a median split into a "star" tier and a "marginal" tier, with no gradations inside a tier. (ii) The star tier is treated as **redundant** — the very accurate experts are construed as merely restating the obvious consensus quality of the product — so their ratings enter integration with only a small residual weight, while the idiosyncratic marginal experts carry full weight. Evidence is the tier-weighted signed sum E = Σ_j w_j (a_j − b_j), and choice is logistic in that scalar plus an attention lapse. Because the weights take only two values, tier-matched profiles cancel *exactly* and give an exact coin flip, with no validity tie-break and no take-the-best fallback. (iii) The same redundancy logic is applied to the majority itself: a feature-count majority is the most "obvious" statistic in the display, so subjects apply a *contrarian consensus discount* proportional to the count margin. (iv) Integration is capacity-limited over the *whole displayed panel*, so the effective slope is divided by a power of the panel size n, in addition to a milder within-trial spread discount over the number of discriminating cues D.

**The new commitment is that the redundancy discount is itself a comparative, graded judgement rather than a fixed property of a cue.** A star expert is discounted precisely *because other experts are seen as restating what he says*; the discount is therefore a function of how much of the panel is actually speaking on this trial. When almost nothing in the display discriminates — in the limit, when a single expert is the only one who differentiates the two products — there is no chorus for that expert to be redundant with, so his rating is taken at (nearly) full face value. The discount relaxes continuously back toward full weight as the number of discriminating cues D falls toward one: eps_eff = eps_w + (1 − eps_w)·ψ·exp(−(D−1)/κ_D), with ψ controlling how completely redundancy can be released and κ_D controlling how fast it releases. Because κ_D is small, the release is essentially confined to genuinely sparse displays. This predicts a sharp, design-level signature that no fixed-weight tiering model produces: **profiles in which exactly one high-validity expert discriminates are followed at high rates (≈0.78–0.85), even though profiles in which that same expert discriminates alongside several others are followed near chance**, since only in the latter case is the star rating construed as redundant restatement. Tally-tied, tier-matched profiles remain at exactly chance (E = 0 and margin = 0 regardless of eps_eff), and all dense profiles (D ≥ 2) are left effectively untouched.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ---------- parse the stimulus into two rating vectors ----------
    a = None
    b = None
    try:
        if isinstance(state, dict):
            if 'option_a_ratings' in state and 'option_b_ratings' in state:
                a = np.asarray(list(state['option_a_ratings']), dtype=float).ravel()
                b = np.asarray(list(state['option_b_ratings']), dtype=float).ravel()
    except Exception:
        a = None
        b = None

    if a is None or b is None:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            stim = stim.reshape(2, -1)
        elif stim.ndim > 2:
            stim = stim.reshape(2, -1)
        if stim.shape[0] != 2:
            stim = stim.reshape(2, -1)
        a = np.asarray(stim[0], dtype=float).ravel()
        b = np.asarray(stim[1], dtype=float).ravel()

    n = int(a.shape[0])
    if n == 0 or int(b.shape[0]) != n:
        return np.array([0.5, 0.5], dtype=float)

    # ---------- parameters ----------
    def _get(name, default):
        try:
            return float(parameters.get(name, default))
        except Exception:
            return float(default)

    eps_w = _get('eps_w', 0.18)          # residual weight of the "star" (redundant) tier
    beta = _get('beta', 2.9)             # logistic slope at the reference panel size
    delta = _get('delta', 0.075)         # within-trial evidence-spread discount exponent
    tau = _get('tau', 0.9)               # panel-size (capacity) dilution exponent
    lapse = _get('lapse', 0.115)         # attention lapse
    count_bias = _get('count_bias', 0.45)  # contrarian consensus (majority) discount
    psi = _get('psi', 0.60)              # max release of the redundancy discount
    kappa_d = _get('kappa_d', 0.30)      # how fast redundancy releases as D -> 1

    eps_w = float(min(max(eps_w, 0.0), 1.0))
    beta = float(min(max(beta, 0.0), 20.0))
    delta = float(min(max(delta, -1.0), 2.0))
    tau = float(min(max(tau, 0.0), 3.0))
    lapse = float(min(max(lapse, 0.0), 0.6))
    count_bias = float(min(max(count_bias, 0.0), 1.5))
    psi = float(min(max(psi, 0.0), 1.0))
    kappa_d = float(min(max(kappa_d, 1e-6), 3.0))

    # ---------- stated validities ----------
    v = None
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is not None:
        try:
            v = np.asarray(v_raw, dtype=float).ravel()
        except Exception:
            v = None
    if v is None or v.shape[0] != n:
        # fallback: assume the display order is descending in validity
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)

    # ---------- signed feature-wise comparison ----------
    diff = np.sign(a - b)                            # +1 A wins, -1 B wins, 0 tie
    D = int(np.sum(diff != 0))
    if D == 0:
        return np.array([0.5, 0.5], dtype=float)

    # ---------- graded redundancy relaxation ----------
    # A star expert is discounted only because OTHER experts restate him; when
    # the discriminating panel shrinks toward a single voice there is nothing
    # left to be redundant with, so the discount releases back toward 1.
    relax = psi * float(np.exp(-(float(D) - 1.0) / kappa_d))
    eps_eff = eps_w + (1.0 - eps_w) * relax
    eps_eff = float(min(max(eps_eff, 0.0), 1.0))

    # ---------- coarse credibility tiering + redundancy discount ----------
    k = int(n // 2)                                  # size of the "star" tier
    order = np.argsort(-v, kind='stable')            # most valid first
    w = np.ones(n, dtype=float)
    if k > 0:
        w[order[:k]] = eps_eff                       # star tier == treated as redundant

    # ---------- tier-weighted additive evidence ----------
    E = float(np.sum(w * diff))

    # spread discount: evidence thinly spread over many contradicting cues feels weaker
    denom = float(D) ** delta
    if not np.isfinite(denom) or denom <= 0.0:
        denom = 1.0

    # ---------- panel-size capacity dilution ----------
    # a bounded comparison budget is shared across the WHOLE displayed panel,
    # so integration fidelity falls as the number of experts n grows.
    # 6.0 is a reference panel size (a pure reparametrisation of beta).
    cap = (float(n) / 6.0) ** tau
    if not np.isfinite(cap) or cap <= 0.0:
        cap = 1.0
    denom = denom * cap

    z = beta * E / denom

    # ---------- contrarian consensus (majority-redundancy) discount ----------
    nA = int(np.sum(diff > 0))
    nB = int(np.sum(diff < 0))
    margin = float(np.clip(nA - nB, -6, 6))
    z = z - count_bias * margin

    if z > 30.0:
        z = 30.0
    elif z < -30.0:
        z = -30.0

    p_a = 1.0 / (1.0 + np.exp(-z))

    # ---------- lapse ----------
    p_a = (1.0 - lapse) * p_a + lapse * 0.5
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    return p / p.sum()

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / float(len(probs))
    else:
        probs = probs / s
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

## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=6):** [0.95, 0.57, 0.88, 0.6, 0.66, 0.83]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  trial 2: A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  trial 3: A=[1, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  trial 4: A=[1, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0]
  trial 5: A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1]
  trial 6: A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  trial 7: A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  trial 8: A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 9: A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  trial 10: A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  trial 11: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  trial 12: A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 1]

**Rationale:** GOAL: dissociate RDTI-MCR (advocated) from Count-Direction with Conflict-Graded Confidence (CDCC, competing). The two models agree on nothing important here: CDCC makes DIRECTION purely count-based (ties => exactly 0.5, parameter-free) and lets validity affect only confidence, never sign, and never drops adherence to the count-winner below 0.5. RDTI-MCR instead (i) median-splits the panel into a discounted 'star' tier and a full-weight 'marginal' tier, (ii) applies a contrarian discount proportional to the count margin, and (iii) releases the star discount only when the discriminating set is essentially a single voice (D=1).

PANEL: 6 experts, validities [0.95, 0.57, 0.88, 0.60, 0.66, 0.83] (screen order deliberately scrambled). Descending rank f0(.95) > f2(.88) > f5(.83) > f4(.66) > f3(.60) > f1(.57), so the RDTI median split is unambiguous: STAR = {f0,f2,f5}, MARGINAL = {f4,f3,f1}, with a wide .83/.66 gap at the split. CDCC's convex weights w=(v-.5)^3 are .091/.066/.036 for the stars versus .0041/.0010/.0003 for the marginals (a >20:1 ratio), so its coherence gate is driven almost entirely by the stars. n=6 makes RDTI's panel-size dilution term exactly 1, removing a nuisance factor. 6 configurations x 2 side mirrors = 12 unique pairs x 8 repetitions = 96 trials.

CELLS (A-favouring version listed; the following trial is the exact A/B mirror):
C1 STAR-SOLO: only f0(.95) discriminates, A wins it (D=1).
C2 MARGINAL-SOLO: only f1(.57) discriminates, A wins it (D=1).
C3 TIE-1v1 MISMATCHED: A wins f1(.57), B wins f0(.95); tally tied.
C4 TIE-2v2 MISMATCHED: A wins f3(.60) and f4(.66), B wins f0(.95) and f2(.88); tally tied.
C5 DENSE STAR-MAJORITY: A wins f0,f2,f5 (all stars), B wins f3,f4 (marginals); nA=3, nB=2, f1 tied.
C6 DENSE MARGINAL-MAJORITY: A wins f1,f3,f4 (all marginals), B wins f0,f2 (stars); nA=3, nB=2, f5 tied.

QUANTITATIVE PREDICTIONS, P(choose A) at nominal parameters.
RDTI-MCR (eps_w=.18, beta=2.9, delta=.075, tau=.9, lapse=.115, count_bias=.45, psi=.60, kappa_d=.30): C1 = 0.79; C2 = 0.87; C3 = 0.86; C4 = 0.93; C5 = 0.07; C6 = 0.94.
CDCC (beta=2.6, gamma=1.5, kappa=0.7, rho=3.0, eps=0.10): C1 = 0.945; C2 = 0.945; C3 = 0.500; C4 = 0.500; C5 = 0.600; C6 = 0.521.

THREE ORTHOGONAL DIAGNOSTICS:
(1) TALLY-TIED, TIER-MISMATCHED CELLS (C3, C4) are parameter-free for CDCC: nA = nB so it MUST be exactly 0.5 for every beta, gamma, kappa, rho, epsilon. RDTI predicts strong, systematic choice of the option backed only by the LOW-validity marginal experts (0.78-0.93 across its whole parameter box for C3, 0.85-0.97 for C4). Each subject contributes 32 such trials, so even a 10-point deviation from 0.5 is decisive. Note the sign is also diagnostic against WADD/TTB, which would predict the opposite (below 0.5), so the cell is self-diagnostic three ways.
(2) DIRECTION REVERSAL ON DENSE CONFLICT (C5 vs C6). Both cells have the same count margin (+1 for A) and the same D=5, so CDCC must put BOTH above 0.5 for A (0.52-0.70 depending on gamma/beta; conflict can only erode confidence toward 0.5, never past it). RDTI's redundancy discount plus contrarian majority discount makes C5 collapse to ~0.07 while C6 rises to ~0.94 -- an ~0.87 swing between two profiles CDCC treats as same-direction. Observing sub-chance adherence to the count winner in C5 falsifies CDCC outright; observing C5 and C6 both mildly above 0.5 with C5 > C6 falsifies RDTI.
(3) GRADED REDUNDANCY RELEASE (C1 vs C2). Both are single-discriminating-cue trials with the count winner also winning the only cue, so CDCC's S = 1, C = +1 and its prediction is EXACTLY IDENTICAL for the two cells regardless of parameters (~0.94). RDTI predicts C2 > C1 by 0.08-0.23 (the solo star is only partially released from its redundancy discount: eps_eff = eps_w + (1-eps_w)*psi < 1), and predicts C1 to sit noticeably below CDCC's ~0.94 ceiling. Comparing C1 (0.79-ish, star alone) with C5 (0.07, same star embedded in a dense display) also isolates RDTI's claim that a star is discounted only when there is a chorus to be redundant with.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror so side bias cancels exactly; tied cues are coded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s does not track composition; validity rank is decoupled from screen position; no pair is degenerate (every trial requires reading several ratings and no cell has the two models making the same prediction).

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
The design contains exactly four cells in which the high-validity 'star' tier (f0=.95, f2=.88, f5=.83) and the low-validity 'marginal' tier (f1=.57, f3=.60, f4=.66) point in OPPOSITE directions: C3 (1 star vs 1 marginal, tally-tied), C4 (2 stars vs 2 marginals, tally-tied), C5 (3 stars vs 2 marginals, count favours the star side) and C6 (3 marginals vs 2 stars, count favours the marginal side). The metric is the overall rate of choosing the option backed by the LOW-validity marginal tier on exactly those trials (64 of the 96 trials per subject, mirrored so side bias cancels exactly).

RDTI-MCR treats the star tier as redundant restatement (residual weight eps_w) and additionally applies a contrarian discount proportional to the count margin. In all four cells the discriminating panel is dense (D>=2), so the graded redundancy release is essentially zero and the star tier stays discounted. Across the entire declared parameter box this yields marginal-tier adherence of roughly .73-.93 (C3), .82-.97 (C4), .82-.97 (C5, i.e. sub-chance adherence to the count winner) and ~.88-.95 (C6) — an overall metric near .85-.95.

CDCC is pinned near .5: on C3 and C4 the tallies are equal, so it must be EXACTLY .5 for every parameter setting; on C5 the count winner is the star-backed option and its coherence gate is near 1, giving marginal-tier adherence of only ~.29-.40; on C6 the coherence gate collapses (best cues oppose the count) so it barely exceeds chance, ~.50-.55. Averaging gives ~.44-.51.

So the predicted gap is ~0.40-0.45 with each subject contributing 64 trials (binomial SD ~0.06) and both theories being parametrically tight in this contrast, giving a very large Welch t. Crucially the metric is not trivially true of RDTI: any tallying, WADD, take-the-best, or validity-weighted account predicts values at or below .5 here (it requires systematically siding with the least valid experts), so the contrast is a genuine, falsifiable behavioural signature rather than a restatement of the model.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design validities (fixed by the experiment): [0.95, 0.57, 0.88, 0.60, 0.66, 0.83]
    # Median split -> STAR tier (high validity)  = features 0, 2, 5
    #              -> MARGINAL tier (low validity) = features 1, 3, 4
    validities = np.array([0.95, 0.57, 0.88, 0.60, 0.66, 0.83], dtype=float)

    hits = 0.0
    n = 0.0

    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()
            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()
        except Exception:
            continue
        if a.size != b.size or a.size == 0:
            continue
        m = a.size
        if validities.size == m:
            v = validities
        else:
            v = np.linspace(0.95, 0.55, m)
        order = np.argsort(-v, kind='stable')
        k = m // 2
        star_idx = order[:k]
        marg_idx = order[k:]

        d = np.sign(a - b)  # +1 A wins the cue, -1 B wins the cue, 0 tie
        star_score = float(np.sum(d[star_idx]))
        marg_score = float(np.sum(d[marg_idx]))

        # keep only trials where BOTH tiers discriminate and they DISAGREE
        if star_score == 0.0 or marg_score == 0.0:
            continue
        if np.sign(star_score) == np.sign(marg_score):
            continue

        resp = row['response']
        try:
            resp = int(resp)
        except Exception:
            continue
        if resp not in (0, 1):
            continue

        marginal_preferred = 0 if marg_score > 0 else 1  # 0 = A, 1 = B
        hits += 1.0 if resp == marginal_preferred else 0.0
        n += 1.0

    if n == 0.0:
        return 0.5
    return float(hits / n)

### RESULTS
- Predicted under pi_5 (simulated): 0.9059 (var=0.0023)
- Predicted under pi_4 (simulated): 0.4641 (var=0.0042)
- Observed on real data: 0.6606 (var=0.0038)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=8):** [0.95, 0.56, 0.85, 0.6, 0.65, 0.9, 0.53, 0.8]

**Trial pairs (n=12):**
  trial 1: A=[0, 1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 1, 0]
  trial 2: A=[1, 0, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0, 1, 0]
  trial 3: A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0]
  trial 4: A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0]
  trial 5: A=[1, 1, 1, 0, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 1, 0, 1]
  trial 7: A=[0, 1, 1, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0, 0]
  trial 8: A=[1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 0, 1, 0]
  trial 9: A=[1, 0, 1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0, 1]
  trial 10: A=[0, 0, 1, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1, 0, 1]
  trial 11: A=[1, 1, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0]
  trial 12: A=[0, 1, 0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 0, 1, 1, 0]

**Rationale:** GOAL: dissociate Count-Direction with Conflict-Graded Confidence (CDCC, advocated) from Redundancy-Discounted Tiered Integration with Majority Discounting (RDTI-MCR, competing). CDCC makes DIRECTION purely count-based (exact 0.5 on tally ties, parameter-free; never sub-chance adherence to the count winner) and lets validity enter only as a confidence gate, with dilution S = M/D^gamma. RDTI-MCR median-splits the panel into a discounted 'star' tier and a full-weight 'marginal' tier, adds a contrarian discount proportional to the count margin, and releases the star discount only for essentially solo displays (D=1). This design attacks the three points where their signs, not just their magnitudes, diverge.

PANEL: 8 experts, validities in scrambled screen order [0.95, 0.56, 0.85, 0.60, 0.65, 0.90, 0.53, 0.80]. Descending rank f0(.95) > f5(.90) > f2(.85) > f7(.80) >> f4(.65) > f3(.60) > f1(.56) > f6(.53), so RDTI's median split is unambiguous (STAR = {f0,f5,f2,f7}, MARGINAL = {f4,f3,f1,f6}) with a wide .80/.65 gap at the split, and CDCC's convex weights w=(v-.5)^3 are .091/.064/.043/.027 for the stars versus .0034/.0010/.0002/.00003 for the marginals (>25:1), so its coherence gate is star-driven. n=8 (rather than 6) also puts RDTI's capacity term (n/6)^tau above 1, which shrinks its evidence term relative to the count-margin discount and sharpens the predicted sign reversals. 6 configurations x 2 side mirrors = 12 unique pairs x 8 repetitions = 96 trials.

CELLS (A-favouring version listed; the next trial is its exact A/B mirror):
C1 (T1/T2) TALLY-TIED 3v3, TIER-MISMATCHED: A wins marginals f4,f3,f1; B wins stars f0,f5,f2; f6,f7 tied.
C2 (T3/T4) TALLY-TIED 2v2, TIER-MISMATCHED: A wins the two weakest experts f6(.53), f1(.56); B wins the two best f0(.95), f5(.90); f2,f3,f4,f7 tied.
C3 (T5/T6) DENSE STAR-MAJORITY (D=6, M=+2): A wins all four stars f0,f5,f2,f7; B wins marginals f4,f3; f1,f6 tied.
C4 (T7/T8) DENSE MARGINAL-MAJORITY (D=6, M=+2): A wins all four marginals f4,f3,f1,f6; B wins stars f0,f5; f2,f7 tied.
C5 (T9/T10) SPARSE UNANIMOUS STAR (D=2, M=+2): A wins only f0(.95) and f5(.90); six cues tied.
C6 (T11/T12) MID STAR-MAJORITY (D=4, M=+2): A wins stars f0,f5,f2; B wins marginal f4; f1,f3,f6,f7 tied.

QUANTITATIVE PREDICTIONS, P(choose A).
CDCC (beta=2.6, gamma=1.5, kappa=0.7, rho=3.0, eps=0.10): C1 = 0.500, C2 = 0.500, C3 = 0.650, C4 = 0.532, C5 = 0.928, C6 = 0.754.
RDTI-MCR (eps_w=.18, beta=2.9, delta=.075, tau=.9, lapse=.115, count_bias=.45, psi=.60, kappa_d=.30): C1 = 0.935, C2 = 0.911, C3 = 0.086, C4 = 0.941, C5 = 0.487, C6 = 0.180.

THREE ORTHOGONAL DIAGNOSTICS:
(1) TALLY-TIED TIER-MISMATCHED CELLS (C1, C2). CDCC MUST be exactly 0.5 for every beta, gamma, kappa, rho, epsilon (nA = nB returns the coin flip before any validity computation). RDTI predicts strong, systematic choice of the option backed only by the LOW-validity marginal experts (0.85-0.97 across its whole parameter box, since E = k(1 - eps_w) > 0 with eps_w <= 0.34 and the count-margin term is exactly zero). Each subject contributes 32 such trials, so even a 10-point deviation from 0.5 is decisive; note the sign also separates RDTI from WADD/TTB, which would predict below 0.5 here, so the cell is self-diagnostic three ways.
(2) SIGN REVERSAL ON STAR-BACKED MAJORITIES (C3 and C6 versus C4). C3, C4 and C6 all have count margin +2 for A. CDCC must put ALL THREE above 0.5 (conflict can only erode confidence toward chance, never past it) and predicts the coherent star-backed cells ABOVE the conflict cell: C3 0.65 and C6 0.75 versus C4 0.53. RDTI predicts C3 = 0.09 and C6 = 0.18 (sub-chance: 4*eps_w <= 1.36 < 2 so the tier-weighted evidence is negative whenever the count winner's wins are the stars, and the contrarian majority discount pushes it further down), while C4 = 0.94 -- an ~0.85 swing between profiles CDCC treats as same-direction and similarly signed. Observing systematic sub-chance adherence to the count winner in C3/C6 falsifies CDCC outright; observing C3 > C4 > 0.5 falsifies RDTI.
(3) DILUTION LADDER AT CONSTANT MARGIN AND CONSTANT COHERENCE (C5, C6, C3: D = 2, 4, 6 with M = +2 and the count winner always holding the stars). CDCC's S = M/D^gamma predicts a steep monotone decline 0.93 -> 0.75 -> 0.65, with near-ceiling (0.90-0.95) adherence on the sparse unanimous star cell C5 -- the classic CDCC signature that sparse clean evidence beats dense larger-D evidence. RDTI predicts the same ordering but hugging or crossing below chance (0.49 -> 0.18 -> 0.09), because its graded redundancy release is essentially confined to D = 1 (relax = psi*exp(-(D-1)/kappa_d) is already <= 0.03 at D = 2) while the count-margin discount of 0.45*2 applies in all three cells. Thus C5 is the cleanest magnitude test: CDCC needs >= 0.85, RDTI at nominal parameters sits at chance and only reaches ~0.8 at the extreme corner of its box.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror, so side bias cancels exactly; tied cues are coded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s does not track composition or tier; validity rank is scrambled relative to screen position so positional heuristics cannot mimic tiering; no pair is degenerate (every trial requires reading several ratings) and no cell has the two models making the same prediction.

INTERPRETATION: exact chance on C1/C2, above-chance adherence to the count winner in every one of C3-C6 with the coherent cells highest, and a steep sparse-to-dense decline 0.93/0.75/0.65 at fixed margin uniquely supports CDCC. Strong marginal-tier-driven preferences on the tally-tied cells plus sub-chance choice of the star-backed majority in C3/C6 uniquely supports RDTI-MCR.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This is the sign-reversal diagnostic (design cells C3 and C6): dense/mid displays (D>=4) with a feature-count majority whose supporting cues are exactly the high-validity 'star' experts while the count loser is supported only by low-validity marginal experts. The metric is the proportion of such trials on which the subject chooses the count winner (32 of 96 trials per subject, so binomial noise per subject is ~0.008 in variance).

CDCC makes direction purely count-based and floors the confidence argument at zero, so adherence must be strictly above 0.5 here, and because coherence is maximal (all star weight on the winner's side) the gate is near 1: across its whole parameter box the predicted adherence is roughly 0.57-0.92, centred near 0.70-0.75.

RDTI-MCR discounts precisely the star tier (residual weight eps_w<=0.34, and its graded redundancy release is inert at D>=2) and additionally applies a contrarian discount proportional to the count margin, so the tier-weighted evidence 4*eps_w-2 (C3) and 3*eps_w-1 (C6) is negative and the majority discount pushes further down: predicted adherence is roughly 0.02-0.35, centred near 0.13.

So the two theories are separated not just in magnitude but in sign relative to chance (~0.73 vs ~0.15), with tight per-subject estimates (32 trials each and a narrow parameter-induced spread), which should yield a very large Welch t at N=25. It is not trivially true of CDCC: any model that lets high-validity cues drive direction (WADD/TTB) would also score high here, and any model that discounts them scores low; the metric measures a directional signature, not a parameter of my model.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design-time validities (fixed by the experiment), used only to define the
    # median-split 'star' tier structurally.
    v = np.array([0.95, 0.56, 0.85, 0.60, 0.65, 0.90, 0.53, 0.80], dtype=float)
    n = v.shape[0]
    order = np.argsort(-v, kind='stable')
    star = set(int(i) for i in order[: n // 2])

    hits = []
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        try:
            a = np.asarray(list(a_raw), dtype=float).ravel()
            b = np.asarray(list(b_raw), dtype=float).ravel()
        except Exception:
            continue
        if a.shape[0] != n or b.shape[0] != n:
            continue
        d = np.sign(a - b)
        nA = int(np.sum(d > 0))
        nB = int(np.sum(d < 0))
        D = nA + nB
        # dense / mid displays only (D >= 4), with a strict count winner
        if D < 4 or nA == nB:
            continue
        winner = 0 if nA > nB else 1
        if winner == 0:
            win_idx = set(int(i) for i in np.where(d > 0)[0])
            lose_idx = set(int(i) for i in np.where(d < 0)[0])
        else:
            win_idx = set(int(i) for i in np.where(d < 0)[0])
            lose_idx = set(int(i) for i in np.where(d > 0)[0])
        # keep only STAR-BACKED majorities: the count winner's wins are all
        # high-validity (star) experts, the count loser's wins are all
        # low-validity (marginal) experts.
        if not win_idx.issubset(star):
            continue
        if len(lose_idx & star) > 0:
            continue
        try:
            r = int(resp)
        except Exception:
            continue
        hits.append(1.0 if r == winner else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))

### RESULTS
- Predicted under pi_5 (simulated): 0.1506 (var=0.0044)
- Predicted under pi_4 (simulated): 0.7144 (var=0.0105)
- Observed on real data: 0.4850 (var=0.0085)

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
**Predicted under pi_5:** 0.5169 (var=0.0089)
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
**Predicted under pi_5:** 0.9281 (var=0.0024)
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
**Predicted under pi_5:** 0.8167 (var=0.0020)
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
**Predicted under pi_5:** 0.7783 (var=0.0043)
**Predicted under pi_4:** 0.7438 (var=0.0061)

### Experiment 7
**Design**
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]

**Metric**
```python
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

```

**Observed (real) value:** -0.7633 (var=0.0200)
**Predicted under pi_5:** -0.7671 (var=0.0062)
**Predicted under pi_4:** 0.1438 (var=0.0077)

### Experiment 8
**Design**
  A=[1, 1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1, 1]

**Metric**
```python
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

```

**Observed (real) value:** -0.9375 (var=0.0693)
**Predicted under pi_5:** -0.8519 (var=0.0288)
**Predicted under pi_4:** 0.1587 (var=0.0237)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Pooling all eight experiments, the data show a coherent but unusual profile. (a) On count-dominated designs (Exp 5: 0.794, Exp 6: 0.755) subjects follow the feature tally at roughly 0.75-0.80 \ncomparable to a hard behavioural ceiling implied by ~20-25% noise. (b) On tally-tied trials (Exp 3: 0.474) they are exactly at chance with no validity tie-break. (c) Crucially, in Exps 7 and 8 the coherent-minus-conflict adherence differences are hugely NEGATIVE (-0.763 and -0.938): subjects abandon the count winner precisely when the most-valid expert backs it, and follow the count winner enthusiastically when the most-valid expert opposes it. Exp 1 (0.661 adherence to the LOW-validity tier on tier-conflict trials) points the same way. So the population is systematically anti-validity: cues with low stated validity carry more subjective weight than cues with high stated validity.\n\npi_4 (CDCC) is structurally incapable of this. Its confidence argument is floored at zero, so adherence to the count winner can never fall below 0.5 and coherent trials must always be >= conflict trials. It therefore predicts +0.14 and +0.16 where the data show -0.76 and -0.94 \ncatastrophic sign errors \nand it also gets Exp 1 on the wrong side of chance (0.464 vs 0.661). Its successes (Exps 3, 4, 5, 6) come from getting the count-driven magnitudes and the ~0.8 ceiling right, which is a real strength, but it is falsified on the validity contrasts by the sign of the effect, not merely by magnitude.\n\npi_5 (RDTI-MCR) gets the sign right essentially everywhere (errors of 0.004 on Exp 7, 0.086 on Exp 8, correct side of chance in Exps 1 and 4) and total absolute error ~1.03 versus ~2.48 for pi_4. But it is systematically too extreme wherever the effect is graded: Exp 1 (0.906 vs 0.661), Exp 2 (0.151 vs 0.485), Exp 4 (0.928 vs 0.654). Its discrete median-split tiering, its contrarian count-margin discount, and its D=1 'redundancy release' patch are the machinery producing this overshoot; the tiering also only reproduces Exp 3's chance result by accident (the design happened to be tier-matched). So pi_5 has the right qualitative commitment (high-validity cues are discounted, sometimes to the point of reversing the tally) but the wrong functional form and far too much rigidity.\n\nNeither theory is task-invariant. The obvious missing model is a smooth, graded version of the anti-validity commitment with strong noise: one that can flip the tally when the validity gradient is steep (Exps 7, 8), sit near chance when the two forces roughly cancel (Exps 2, 3), and land at intermediate values (0.65-0.75) in Exps 1 and 4, while still reproducing 0.75-0.80 tally adherence in Exps 5-6. Since pi_4 is the one that is falsified in SIGN (a structural, unfixable failure given its zero-floor commitment), it should be replaced with that stronger competitor rather than merely re-parameterised.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_4 (CDCC) with a smooth 'inverted-informativeness' integration theory \ncall it Graded Validity-Inversion Weighted Tallying (GVIT) \nthat is a genuinely stronger competitor to pi_5 by fixing pi_5's systematic overshoot while keeping the anti-validity sign.\n\nCore claims:\n1. SUBJECTIVE NEWS VALUE IS INVERSELY GRADED IN STATED VALIDITY. A rating from a near-perfect expert is predictable and construed as restating the obvious, so it carries little decision weight; a rating from a barely-diagnostic expert is surprising and treated as genuine private information. Formally, weights are a CONTINUOUS, panel-relative function of validity: s_j = (v_j - v_min)/(v_max - v_min) computed within the displayed panel, w_j = 1 - lambda * s_j, with lambda in roughly [0.6, 1.6] so that at high lambda the single best expert can carry a small NEGATIVE weight (active contrarianism) but at moderate lambda the ordering is merely flattened or mildly reversed. No median split, no discrete tiers, no special-case release at D=1, and no count-margin contrarian term \nevery cue enters through one smooth function.\n2. EVIDENCE AND CHOICE. E = sum_j w_j * sign(a_j - b_j); p(A) = logistic(beta * E / D^gamma) where D is the number of discriminating cues (dilution: evidence spread over many mutually contradicting cues is subjectively weaker; gamma ~ 0.3-0.8). When E = 0 exactly (including all tally-tied, weight-matched profiles) the model returns exactly 0.5 with no validity tie-break \nthis reproduces Exp 3 (0.474) without pi_5's accidental tier-matching.\n3. LARGE LAPSE / CEILING. epsilon in ~[0.15, 0.35], so even maximally coherent profiles top out near 0.75-0.88, matching the empirical ceiling in Exps 5 (0.794) and 6 (0.755) and preventing the 0.90+ / 0.10- extremes that made pi_5 miss Exps 1, 2 and 4.\n\nTarget quantitative behaviour to check while building it: Exp 7 coherent-minus-conflict about -0.5 to -0.8; Exp 8 about -0.7 to -0.95; Exp 1 marginal-tier adherence about 0.65-0.75 (NOT 0.9); Exp 2 star-backed-majority adherence about 0.42-0.55 (the +2 count margin and the inverted weights should roughly cancel once dilution at D=6 is applied); Exp 3 about 0.40-0.50; Exp 4 about 0.65-0.80; Exps 5-6 about 0.75-0.82. Choose parameter ranges (lambda, beta, gamma, epsilon) so that this whole envelope is reachable inside the box, and make sure the box does NOT contain settings that force sub-0.2 or above-0.9 predictions on dense mixed profiles.\n\nHow it differs from both incumbents: unlike pi_4 it allows adherence to the count winner to go below chance (the sign reversal actually observed), and unlike pi_5 it does so continuously and with bounded strength, with no tiering, no majority-margin discount, and no sparse-display patch \nso it should dominate pi_5 on Exps 1, 2 and 4 while matching it on Exps 7 and 8."
}
```

## Usage

```json
{
  "input_tokens": 24843,
  "output_tokens": 29546
}
```
