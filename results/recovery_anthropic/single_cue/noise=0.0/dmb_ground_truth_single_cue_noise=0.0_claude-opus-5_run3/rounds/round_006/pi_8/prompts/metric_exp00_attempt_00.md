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
**Validities (n_features=8):** [0.82, 0.6, 0.95, 0.88, 0.72, 0.58, 0.68, 0.76]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1, 1, 1, 0]
  trial 2: A=[0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1, 0, 1]
  trial 3: A=[1, 0, 1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 1, 0, 0]
  trial 5: A=[0, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  trial 6: A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 1, 1, 1]
  trial 7: A=[1, 1, 1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 1, 1, 0]
  trial 8: A=[0, 1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 1, 0, 1]
  trial 9: A=[1, 0, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0, 1, 0]
  trial 10: A=[0, 0, 0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0, 0, 1]
  trial 11: A=[1, 1, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1, 1]
  trial 13: A=[1, 1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 1, 1]
  trial 15: A=[1, 0, 1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 0, 0, 0, 1]

**Rationale:** GOAL: dissociate the ADVOCATED attention-lottery model (asymmetric screen salience with tau~0.85 so only rows 0,1,6,7 are really visible; authority-contingent first-mismatch capture gated by the most diagnostic DISCRIMINATING row only; limited-sample comparison of k=2-3 rows; panel-volume confidence beta_eff=beta*(n/6)^zeta; and above all the NEW commitment that the panel-informativeness judgement is CATEGORICAL, a step at f0~0.388 with kappa~100, with mutual endorsements counted at omega~0.84) from the COMPETING reason-counting model (near-linear validity vote over ALL discriminating experts, categorical unanimity bonus ups~1.5, saturating majority bonus lam~0.67, symmetric first/last-row salience entering at the constant residual g_res~0.25 whenever the vote is not deadlocked, a SHALLOW silence sigmoid sil_k~38 centred at 0.36, floor c_min~0.10, and a first-cue commitment whose authority cap is computed over ALL later rows including silent ones).

n=8, validities by SCREEN position [.82,.60,.95,.88,.72,.58,.68,.76]: the two heavyweights (.95,.88) sit mid-list at pos2/pos3 where the advocated model's salience is ~0.10/0.03 of pos0's, the screen extremes are mid-high (.82 at pos0, .76 at pos7), and the weakest experts (.60,.58) fill pos1/pos5. All displays are deliberately DENSE (>=4 discriminating rows): with n=8 any panel with 5+ mute rows is voided by BOTH theories, so sparse cells are uninformative by construction. Because no cell is vote-deadlocked, the competitor's salience term always sits at its constant residual g_res, which makes its predictions tight. 8 base cells, each mirrored across sides (16 unique pairs, K=6 -> 96 trials); mirroring forces an exact 8A/8B split for both theories.

(1) EXT (pairs 1-2): A=[1,1,0,0,0,1,0,1] vs B=[0,1,1,1,1,1,1,0]. The two screen extremes (.82,.76) favour A; the four mid-list rows (.95,.88,.72,.68) favour B; 2 mute rows are mutual endorsements (f_sil=.21, both theories at full confidence). Advocated: att(pos0)=1.57 and att(pos7)=0.32 against 0.13 for ALL four mid rows together, so the 3-row lottery almost never sees the mid-list -> P(A)~.93. Competitor: V=-1.35, majority bonus -0.67, residual salience +0.62 -> P(A)~.22. ~70-point DIRECTION disagreement over both full parameter boxes.

(2) AUTH (pairs 3-4): A=[1,0,1,1,0,1,0,0] vs B=[0,1,1,1,1,1,1,1]. pos0 (.82) alone favours A against four lower-validity rows (.60,.72,.68,.76); crucially the .95 and .88 experts are MUTE (mutual endorsements, f_sil=.315). This isolates the two theories' different authority gates: the advocated model's first-mismatch capture asks only whether pos0 is the most diagnostic DISCRIMINATING row (it is: ratio 1.16 -> q_auth~.83, capture factor ~6.5, att(pos0)=4.7 vs 0.5 for all four opponents) -> P(A)~.90; the competitor's commitment gate is computed against the max validity of ALL later rows, i.e. the silent .95 expert, so q_com~.07 and it falls back on the vote (V=-0.94, M=-0.67, salience -0.37) -> P(A)~.18. Again ~70 points, opposite direction, and the mechanism is the silent-vs-speaking composition of the gate.

(3) TAIL (pairs 5-6): A=[0,1,1,1,0,1,1,1] vs B=[1,1,1,1,1,0,0,0]. pos0 (.82) and pos4 (.72) favour B; the tail rows pos5/pos6/pos7 (.58,.68,.76) favour A; f_sil=.315. Validities were chosen so the competitor's vote is almost exactly null (V=-0.02 to +0.01 across rho in [.88,1.02]) and only its residual salience tilts it (+0.18 to +0.30), giving P(A)~.51-.55, i.e. a coin flip. The advocated model captures on pos0 (q_auth~.83, att 4.7 vs 0.41 for the whole tail) and predicts P(A)~.08. This cell also reverses the side that pos0 favours relative to cells 1-2, so 'follow the top row' cannot be confounded with a side bias.

(4-5) SILENCE-COMPOSITION CONTRAST at fixed d, SIL-LO (pairs 7-8) vs SIL-HI (pairs 9-10). Identical difference vector (pos0,pos2,pos7 favour A; pos4,pos6 favour B); the three mute rows are mutual ENDORSEMENTS in SIL-LO (f_sil=.84*3/8=.315) and mutual ABSTENTIONS in SIL-HI (f_sil=.375). Both values lie BELOW the advocated step (f0 in [.380,.396]) but straddle the competitor's shallow sigmoid centre (.36): advocated c goes 1.00 -> .81 (P .95 -> .92, and >=.87 anywhere in its box), whereas competitor c goes .85 -> .37 (P .88 -> .71). A ~17-point graded drop vs a ~3-point near-null, produced entirely by ratings that discriminate nothing, is the direct test of categorical-vs-graded panel informativeness.

(6-7) INVISIBLE-DISSENTER / UNANIMITY CONTRAST at fixed silence, UNI41 (pairs 11-12) vs UNI50 (pairs 13-14). Both have the same three mute both-0 rows (f_sil=.375) and the same five speaking experts (pos0,pos1,pos2,pos6,pos7); the ONLY change is the .95 expert at pos2 flipping from favouring B (4-1) to favouring A (5-0). The advocated model's lottery samples pos2 with probability ~.005 (its salience is 0.096 while pos0 carries the capture bonus), and its silence/vbar terms are identical because the speaker set is identical, so it is FORCED to predict the same probability in both cells (P(A)~.94 in each, difference <1 point). The competitor gains the categorical unanimity bonus (+1.5) plus ~1.9 of vote, moving from P(A)~.76 to ~.91. A 15-point within-subject jump where the advocated theory predicts an exact null.

(8) VOID (pairs 15-16): A=[1,0,1,1,0,0,0,1] vs B=[0,0,0,0,0,0,0,0]. Four experts (incl. the .95 and .88) unanimously endorse A, four endorse neither (f_sil=.50), i.e. the strongest verdict a half-mute panel can deliver. Both theories judge the panel uninformative, but the advocated model voids it essentially completely (c floored at .015-.055) and therefore predicts a LITERAL coin flip, P(A)=.51-.55 across its whole box, despite a 4-0 landslide, whereas the competitor's floor c_min~.10 multiplied by a large drive (V=2.89 + unanimity 1.5 + majority .67 + salience 1.01) leaves a reliable residual tilt, P(A)~.62-.72. Paired with UNI50 (same all-zero competitor, one more mute row) this is a 41-point cliff for the advocated model vs 26 points for the competitor, and the absolute level at VOID (.53 vs .67) is the decisive statistic.

DIAGNOSTIC YIELD: three full direction/level disagreements (EXT, AUTH ~70 points each; TAIL, near-determinism vs chance, ~43 points); one forced null for the competitor's silence sigmoid turned into a 17-point graded drop (SIL-LO vs SIL-HI); one forced null for the advocated lottery turned into a 15-point unanimity jump (UNI41 vs UNI50); and a coin-flip-vs-residual-tilt test of the two floors (VOID). Every pair has at least one discriminating cue and no cell is degenerate.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** ATTENTION LOTTERY WITH LIMITED-SAMPLE COMPARISON, AUTHORITY-CONTINGENT CAPTURE, PANEL-VOLUME CONFIDENCE, AND A CATEGORICAL (STEP-LIKE) PANEL-INFORMATIVENESS GATE. (1) Each display row enters an attention lottery with weight att_j = s_j * x_j * pop_j, where s_j = exp(-j/tau) + gamma*exp(-(n-1-j)/tau) is literal screen salience (asymmetric: top of panel stronger than bottom), x_j = ((v_j-.5)/.5)^rho is mildly compressed stated diagnosticity (rho ~ .55-.85), and pop_j is a pop-out factor: rows on which the two columns are identical do not pop out and keep only a residual share delta, while the FIRST row at which the columns disagree gets a bonus. (2) That first-mismatch bonus is AUTHORITY-CONTINGENT: a row captures attention outright only if it is both the first thing that discriminates AND the most diagnostic discriminating row on the screen. Position is always active, never gated by a deadlock rule. (3) The subject does not integrate the panel: only k = clip(ceil(k_frac*m), 2, 3) rows are actually compared, drawn without replacement with probability proportional to att. Choice = the attention-weighted sign of the sampled differences plus a small within-sample count term; a 1-1 sample is decided by the more salient sampled row. Numerosity helps only because more rows on one side makes the sample more likely to be dominated by that side; there is no unanimity bonus, no additive majority bonus, no serial stop rule. Because the sample is small and random, the map from true evidence to P(choice) is compressive with a ceiling near .85-.90 even for landslides. (4) NEW COMMITMENT (this iteration): THE READER'S JUDGEMENT OF WHETHER THE PANEL IS INFORMATIVE AT ALL IS CATEGORICAL, NOT GRADED. Silence is read as sensitivity, but the subject does not compute a continuous 'how much of the panel was mute' discount; they make a yes/no assessment — 'did most of this panel have nothing to say?' — with the boundary at about two-fifths of the panel (f0 ~ .39) and an essentially step-like transition (kappa ~ 100). Below the boundary the sampled verdict is taken at full face value (c ~ 1); above it the verdict is essentially voided and the choice approaches a coin flip (c floored near .03). There is no intermediate regime in which a panel is 'somewhat' informative. Mutual endorsements (both columns say yes) count toward silence at weight omega ~ .84, i.e. they are nearly, but not quite, as uninformative as mutual abstentions; because the gate is now a step, this small composition weight decides on which SIDE of the categorical boundary a mixed panel falls, which is where the whole behavioural signature lives. (5) Confidence also scales with the VOLUME of the panel, not with the amount actually read: beta_eff = beta*(n/6)^zeta (illusion of knowledge — short panels are systematically less decisive than structurally identical cells on wide panels). (6) Individual differences are unimodal jitter in k_frac, tau, gamma, rho, phi, phi_auth, s_auth, beta, zeta, eta, omega, c_min and a lapse rate; no strategy subpopulations, no learning.

**Parameters:**
- tau: [0.72, 1.00]
- gamma_rec: [0.38, 0.62]
- rho: [0.55, 0.85]
- delta_sil: [0.04, 0.11]
- phi_first: [0.30, 0.70]
- phi_auth: [4.0, 9.0]
- s_auth: [6.0, 16.0]
- lam_w: [0.82, 0.95]
- beta: [2.10, 2.70]
- zeta_n: [1.0, 2.0]
- eta: [1.30, 1.60]
- f0: [0.380, 0.396]
- kap_s: [70.0, 130.0]
- omega: [0.80, 0.88]
- c_min: [0.015, 0.055]
- k_frac: [0.40, 0.55]
- epsilon: [0.02, 0.06]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    from itertools import combinations, permutations

    # ------------------------------------------------------------------
    # 1. parse the trial stimulus into two rating vectors (screen order)
    # ------------------------------------------------------------------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(list(state['option_a_ratings']), dtype=float)
            b = np.asarray(list(state['option_b_ratings']), dtype=float)
    if a is None:
        arr = np.asarray(state, dtype=float)
        arr = arr.reshape(2, -1)
        a = arr[0].astype(float)
        b = arr[1].astype(float)
    a = np.ravel(a).astype(float)
    b = np.ravel(b).astype(float)
    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.array([0.5, 0.5])
    a = a[:n]
    b = b[:n]
    d = a - b
    disc = np.abs(d) > 1e-12
    if not np.any(disc):
        return np.array([0.5, 0.5])

    # ------------------------------------------------------------------
    # 2. communicated validities -> diagnosticity
    # ------------------------------------------------------------------
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])
    val = np.clip(val, 0.5 + 1e-6, 1.0 - 1e-9)
    vs = np.clip((val - 0.5) / 0.5, 1e-6, 1.0)

    # ------------------------------------------------------------------
    # 3. parameters
    # ------------------------------------------------------------------
    def _pget(key, dflt, lo, hi):
        try:
            v = float(parameters.get(key, dflt))
        except Exception:
            v = float(dflt)
        if not np.isfinite(v):
            v = float(dflt)
        return float(np.clip(v, lo, hi))

    tau = _pget('tau', 0.85, 0.2, 8.0)
    gam = _pget('gamma_rec', 0.50, 0.0, 1.5)
    rho = _pget('rho', 0.70, 0.05, 4.0)
    dlt = _pget('delta_sil', 0.07, 0.0, 1.0)
    phi = _pget('phi_first', 0.50, 0.0, 3.0)
    phi_auth = _pget('phi_auth', 6.0, 0.0, 20.0)
    s_auth = _pget('s_auth', 10.0, 0.5, 60.0)
    lam = _pget('lam_w', 0.88, 0.0, 1.0)
    beta = _pget('beta', 2.40, 0.05, 40.0)
    zet = _pget('zeta_n', 1.50, 0.0, 4.0)
    eta = _pget('eta', 1.45, 0.0, 5.0)
    f0 = _pget('f0', 0.388, 0.0, 1.0)
    kap = _pget('kap_s', 100.0, 1.0, 400.0)
    ome = _pget('omega', 0.84, 0.0, 1.0)
    cmin = _pget('c_min', 0.03, 0.0, 0.5)
    kfrac = _pget('k_frac', 0.47, 0.05, 1.0)
    eps = _pget('epsilon', 0.035, 0.0, 0.5)

    def sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    # ------------------------------------------------------------------
    # 4. attention weights: screen salience x compressed diagnosticity,
    #    matched rows do not pop out (residual share delta), first
    #    mismatching row of the top-down scan gets a pop-out bonus that
    #    is AUTHORITY-CONTINGENT: it captures attention outright only if
    #    no more diagnostic expert further down the panel also speaks.
    # ------------------------------------------------------------------
    pos = np.arange(n, dtype=float)
    s = np.exp(-pos / tau) + gam * np.exp(-(n - 1.0 - pos) / tau)
    x = np.power(vs, rho)
    att = s * x
    att = np.where(disc, att, dlt * att)
    idxs = np.flatnonzero(disc)
    att = np.array(att, dtype=float)
    j1 = int(idxs[0])
    if idxs.shape[0] > 1:
        x_rest = float(np.max(x[idxs[1:]]))
        ratio = float(x[j1]) / max(x_rest, 1e-9)
        q_auth = float(sig(s_auth * (ratio - 1.0)))
    else:
        q_auth = 1.0
    att[j1] = att[j1] * (1.0 + phi + phi_auth * q_auth)
    att = np.clip(att, 1e-12, None)

    # ------------------------------------------------------------------
    # 5. CATEGORICAL panel-informativeness gate (silence-as-sensitivity).
    #    The reader makes a yes/no judgement 'did most of the panel have
    #    nothing to say?' with an essentially step-like boundary at f0;
    #    below it the verdict is taken at face value, above it it is
    #    voided (floored near zero).  Mutual endorsements count toward
    #    silence at weight omega, which decides which SIDE of the
    #    boundary a mixed panel falls on.
    # ------------------------------------------------------------------
    both1 = (~disc) & (a > 0.5) & (b > 0.5)
    n_both0 = float(np.count_nonzero((~disc) & (~both1)))
    n_both1 = float(np.count_nonzero(both1))
    f_sil = float(np.clip((n_both0 + ome * n_both1) / float(n), 0.0, 1.0))
    xw = x[idxs]
    sxw = float(np.sum(xw))
    vbar = float(np.sum(xw * vs[idxs]) / sxw) if sxw > 1e-12 else 0.5
    c = 1.0 - eta * float(sig(kap * (f_sil - f0))) * vbar
    c = float(np.clip(c, cmin, 1.0))

    # ------------------------------------------------------------------
    # 5b. PANEL-VOLUME CONFIDENCE: the sampled comparison stays the same
    #     size, but the felt informativeness of its verdict grows with
    #     how many experts were displayed (illusion of knowledge).
    # ------------------------------------------------------------------
    beta_eff = beta * float(np.power(float(n) / 6.0, zet))

    # ------------------------------------------------------------------
    # 6. the lottery: draw k rows without replacement, prob ~ attention
    # ------------------------------------------------------------------
    m = int(idxs.shape[0])
    k = int(np.clip(int(np.ceil(kfrac * m)), 2, 3))
    k = int(min(k, n))
    if k < 1:
        k = 1

    p = att / float(np.sum(att))
    p = 0.998 * p + 0.002 / float(n)
    p = p / float(np.sum(p))

    tot = 0.0
    pa = 0.0
    for S in combinations(range(n), k):
        ps = 0.0
        for perm in permutations(S):
            q = 1.0
            rem = 1.0
            good = True
            for t in perm:
                if rem <= 1e-9:
                    good = False
                    break
                q *= p[t] / rem
                rem -= p[t]
            if good:
                ps += q
        if ps <= 1e-14:
            continue
        sel = [j for j in S if disc[j]]
        if len(sel) == 0:
            E = 0.0
        else:
            ws = att[sel]
            sg = np.sign(d[sel])
            wsum = float(np.sum(ws))
            Ew = float(np.sum(ws * sg) / wsum) if wsum > 1e-12 else 0.0
            nA = float(np.sum(sg > 0))
            nB = float(np.sum(sg < 0))
            Ec = (nA - nB) / (nA + nB) if (nA + nB) > 0 else 0.0
            E = lam * Ew + (1.0 - lam) * Ec
        pa += ps * float(sig(beta_eff * c * E))
        tot += ps

    p_a = (pa / tot) if tot > 1e-12 else 0.5

    pr = np.array([p_a, 1.0 - p_a], dtype=float)
    pr = (1.0 - eps) * pr + eps * 0.5
    pr = np.clip(pr, 1e-12, None)
    pr = pr / pr.sum()
    return pr
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
**Description:** **Reason-counting with a TWO-REGIME display-salience gate (full force only on a deadlocked vote, a constant residual otherwise), near-linear diagnosticity weighting, a saturating majority bonus, unanimity boost, sharp silence-to-confidence collapse (mutual endorsement ≈ silence), and a non-zero floor on read-out gain.**

This keeps every mechanism of the accepted base and changes exactly three calibration commitments (one of them structural-but-local: the shape of the salience gate).

1. **Direction is a vote over discriminating experts, graded near-linearly by stated validity.** Each discriminating expert contributes x_j = ((v_j-0.5)/0.5)^rho with rho ~0.88-1.02. At this exponent a single .95 expert is worth about two upper-middling experts (.72 + .70), so numerosity and diagnosticity are *balanced* rather than one dominating: three mid experts still outvote one top expert, but two mid experts only tie it. Screen position plays no role in this term.

2. **Display salience is a TIE-BREAKER WITH A RESIDUAL, not a quantity that decays with the size of the majority.** s_j = exp(-j/tau)+exp(-(n-1-j)/tau) over the literal display row, multiplied by a gate g = max(exp(-(k_A-k_B)^2/2 sigma^2), g_res) with sigma ~0.60 and g_res ~0.25. The gate has two regimes: when the reason count is exactly deadlocked the first/last rows of the panel decide the choice outright (g = 1); as soon as one side has more reasons, position stops being decisive but does NOT disappear — it persists as a constant, modest attention bonus (g = g_res ≈ 0.25) that is the SAME for a one-reason edge and for a five-against-one landslide. The previous Gaussian-only gate made position vanish geometrically with the majority size, which is unmotivated: there is no reason why a larger majority should erase the extra attention paid to the top and bottom of the screen. The empirical signature is that a three-against-one display whose majority occupies both screen ends is markedly more decisive (~.88) than a three-against-one display whose lone dissenter sits at a screen end (~.82), even when the validities are matched.

3. **Clear majorities are categorical and saturating:** M = lambda*sign(k_A-k_B)*min(max(|k_A-k_B|-1,0),1) with lambda ~0.67. Once one side has at least two more reasons, 'more experts back this one' is itself a discrete reason, and it does not grow further with the size of the majority.

4. **Unanimity among the speaking experts is categorical**: if every expert who noticed a difference points the same way, a constant bonus is added.

5. **Gap inference from the panel is a sharp threshold on how much of the panel was informative, and 'both products are good' is nearly as uninformative as 'neither is'.** Validity is read as sensitivity: a difference only sensitive experts notice implies a marginal true gap. Mutual endorsements count at weight omega ~0.8. c = 1 - eta*sigmoid(k*(f_sil - f0))*vbar.

6. **The collapse of confidence is bounded below:** c is floored at c_min ~0.10, so a panel judged uninformative leaves a small residual tilt rather than a literal coin flip.

7. **A small, display-contingent first-cue commitment survives**: people commit to the first expert who actually discriminates only if no expert further down the list outranks him in stated validity. Individual differences are unimodal jitter in rho, mu, sigma, g_res, lambda, eta, omega, c_min, beta; no strategy subpopulations, no serial stop rule, no learning.

**Parameters:**
- rho: [0.88, 1.02]
- mu: [1.70, 1.95]
- lam: [0.60, 0.74]
- sigma: [0.55, 0.66]
- g_res: [0.22, 0.28]
- tau: [0.85, 1.25]
- ups: [1.25, 1.80]
- beta: [1.05, 1.40]
- eta: [1.45, 1.65]
- sil_k: [30.0, 45.0]
- sil_f0: [0.345, 0.375]
- omega: [0.72, 0.92]
- c_min: [0.07, 0.13]
- phi: [0.95, 1.12]
- s_stop: [6.0, 10.0]
- d_com: [2.20, 2.90]
- epsilon: [0.01, 0.06]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ------------------------------------------------------------------
    # 1. parse the trial stimulus into two rating vectors (screen order)
    # ------------------------------------------------------------------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(list(state['option_a_ratings']), dtype=float)
            b = np.asarray(list(state['option_b_ratings']), dtype=float)
    if a is None:
        arr = np.asarray(state, dtype=float)
        arr = arr.reshape(2, -1)
        a = arr[0].astype(float)
        b = arr[1].astype(float)

    a = np.ravel(a).astype(float)
    b = np.ravel(b).astype(float)
    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.array([0.5, 0.5])
    a = a[:n]
    b = b[:n]
    d = a - b
    disc = np.abs(d) > 1e-12
    if not np.any(disc):
        return np.array([0.5, 0.5])

    # ------------------------------------------------------------------
    # 2. communicated validities -> diagnosticity (= sensitivity)
    # ------------------------------------------------------------------
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])
    val = np.clip(val, 0.5 + 1e-6, 1.0 - 1e-9)
    vs = np.clip((val - 0.5) / 0.5, 1e-6, 1.0)

    # ------------------------------------------------------------------
    # 3. parameters
    # ------------------------------------------------------------------
    rho = float(np.clip(parameters.get('rho', 0.95), 0.05, 4.0))
    mu = float(np.clip(parameters.get('mu', 1.82), 0.0, 8.0))
    lam = float(np.clip(parameters.get('lam', 0.67), 0.0, 3.0))
    sigma = float(np.clip(parameters.get('sigma', 0.60), 0.15, 3.0))
    g_res = float(np.clip(parameters.get('g_res', 0.25), 0.0, 0.6))
    tau = float(np.clip(parameters.get('tau', 1.0), 0.2, 8.0))
    ups = float(np.clip(parameters.get('ups', 1.50), 0.0, 4.0))
    beta = float(np.clip(parameters.get('beta', 1.20), 0.05, 40.0))
    eta = float(np.clip(parameters.get('eta', 1.55), 0.0, 5.0))
    sil_k = float(np.clip(parameters.get('sil_k', 38.0), 1.0, 80.0))
    sil_f0 = float(np.clip(parameters.get('sil_f0', 0.36), 0.05, 0.9))
    omega = float(np.clip(parameters.get('omega', 0.82), 0.0, 1.0))
    c_min = float(np.clip(parameters.get('c_min', 0.10), 0.0, 0.5))
    phi = float(np.clip(parameters.get('phi', 1.05), 0.3, 3.0))
    s_stop = float(np.clip(parameters.get('s_stop', 8.0), 0.5, 60.0))
    d_com = float(np.clip(parameters.get('d_com', 2.50), 0.5, 8.0))
    eps = float(np.clip(parameters.get('epsilon', 0.035), 0.0, 0.5))

    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    # ------------------------------------------------------------------
    # 4. reason count, graded by stated validity (no position here)
    # ------------------------------------------------------------------
    x = np.power(vs, rho)
    idxs = np.flatnonzero(disc)
    m = int(idxs.shape[0])
    sgn = np.sign(d[idxs])
    xs = x[idxs]
    V = float(np.sum(sgn * xs))

    kA = int(np.count_nonzero(sgn > 0))
    kB = int(np.count_nonzero(sgn < 0))

    # ------------------------------------------------------------------
    # 5. display salience over ABSOLUTE SCREEN POSITION (first & last
    #    rows of the panel), deployed as a TWO-REGIME tie-breaker:
    #    FULL force when the reason count is deadlocked, and a CONSTANT
    #    residual fraction g_res once either side has more reasons --
    #    position stops deciding, but it does not evaporate with the
    #    size of the majority.
    # ------------------------------------------------------------------
    pos = np.arange(n, dtype=float)
    e_sal_full = np.exp(-pos / tau) + np.exp(-(n - 1.0 - pos) / tau)
    e_sal = e_sal_full[idxs]
    S = float(np.sum(sgn * e_sal))
    dk = float(kA - kB)
    g = float(np.exp(-np.clip((dk * dk) / (2.0 * sigma * sigma), 0.0, 60.0)))
    g = float(max(g, g_res))

    # ------------------------------------------------------------------
    # 5b. saturating majority bonus (fires only at |dk| >= 2)
    # ------------------------------------------------------------------
    maj = min(max(abs(dk) - 1.0, 0.0), 1.0)
    M = lam * (1.0 if dk > 0 else (-1.0 if dk < 0 else 0.0)) * maj

    # ------------------------------------------------------------------
    # 6. unanimity among the speakers is categorical
    # ------------------------------------------------------------------
    uni = 0.0
    if kA == 0 or kB == 0:
        uni = ups * (1.0 if kA > 0 else -1.0)

    drive = V + mu * g * S + uni + M

    # ------------------------------------------------------------------
    # 7. gap inference from panel silence (validity = sensitivity).
    #    An expert who endorsed BOTH products has positively asserted the
    #    two are alike on his dimension: he is ALMOST as uninformative
    #    about the quality gap as one who endorsed neither (weight omega).
    #    SHARP collapse, but BOUNDED BELOW at c_min.
    # ------------------------------------------------------------------
    both1 = (~disc) & (a > 0.5) & (b > 0.5)
    both0 = (~disc) & (~both1)
    n_sil_eff = float(np.count_nonzero(both0)) + omega * float(np.count_nonzero(both1))
    f_sil = n_sil_eff / float(n)
    f_sil = float(np.clip(f_sil, 0.0, 1.0))
    tot_x = float(np.sum(xs))
    if tot_x <= 0:
        return np.array([0.5, 0.5])
    vbar = float(np.sum(xs * vs[idxs]) / tot_x)
    c = 1.0 - eta * float(_sig(sil_k * (f_sil - sil_f0))) * vbar
    c = float(np.clip(c, c_min, 1.0))

    p_int = _sig(beta * c * drive)

    # ------------------------------------------------------------------
    # 8. authority-capped first-cue commitment (display-contingent)
    # ------------------------------------------------------------------
    j1 = int(idxs[0])
    if j1 >= n - 1:
        q_com = 1.0
    else:
        v_rest = float(np.max(vs[j1 + 1:]))
        v_rest = max(v_rest, 1e-6)
        ratio = vs[j1] / (phi * v_rest)
        q_com = _sig(s_stop * (ratio - 1.0))
    q_com = float(np.clip(q_com, 0.0, 1.0))

    p_dec = _sig(beta * c * d_com)
    p_com_a = p_dec if d[j1] > 0 else (1.0 - p_dec)

    p_a = q_com * p_com_a + (1.0 - q_com) * p_int

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = (1.0 - eps) * p + eps * 0.5
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return p
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # The six target displays (EXT, AUTH, TAIL cells and their mirrors).\n    targets = {\n        # EXT\n        (\"11000101\", \"01111110\"),\n        (\"01111110\", \"11000101\"),\n        # AUTH\n        (\"10110100\", \"01111111\"),\n        (\"01111111\", \"10110100\"),\n        # TAIL\n        (\"01110111\", \"11111000\"),\n        (\"11111000\", \"01110111\"),\n    }\n\n    def as_str(x):\n        try:\n            return \"\".join([str(int(round(float(v)))) for v in x])\n        except Exception:\n            return \"\"\n\n    hits = 0\n    tot = 0\n    for _, row in data.iterrows():\n        try:\n            a = [int(round(float(v))) for v in row[\"option_a_ratings\"]]\n            b = [int(round(float(v))) for v in row[\"option_b_ratings\"]]\n        except Exception:\n            continue\n        ka = \"\".join([str(v) for v in a])\n        kb = \"\".join([str(v) for v in b])\n        if (ka, kb) not in targets:\n            continue\n        # first (top-most) screen row on which the two columns disagree\n        j = -1\n        for i in range(min(len(a), len(b))):\n            if a[i] != b[i]:\n                j = i\n                break\n        if j < 0:\n            continue\n        try:\n            resp = int(row[\"response\"])\n        except Exception:\n            continue\n        if resp not in (0, 1):\n            continue\n        favors_a = a[j] > b[j]\n        followed = (favors_a and resp == 0) or ((not favors_a) and resp == 1)\n        hits += 1 if followed else 0\n        tot += 1\n\n    if tot == 0:\n        return 0.5\n    return float(hits) / float(tot)",
  "rationale": "The design's three 'direction-disagreement' cells (EXT, AUTH, TAIL) were constructed so that the advocated attention-lottery model, via asymmetric screen salience plus authority-contingent first-mismatch capture, follows the TOP screen row (position 0, validity .82) almost deterministically (P ~ .90-.93 per cell), while the competing reason-counting model integrates the full validity-weighted vote over all discriminating experts and therefore goes the OTHER way in EXT and AUTH (P(top-row option) ~ .18-.22) and is essentially at chance in TAIL (~ .47). The metric is simply the proportion of choices, restricted to those six displays (3 cells x 2 mirror orders, 36 trials per subject), that agree with the option favoured by the first discriminating screen row. Predicted values: ~0.90 under the advocated theory, ~0.29 under the competitor -- a ~60-point gap. Because it pools 36 trials per subject and both theories make near-extreme, parameter-robust predictions in these cells (the design rationale states the direction holds across both full parameter boxes), the per-subject sampling variance is small (binomial var ~0.002-0.007 plus modest parameter jitter), so the contrast should survive Welch's test at N=25. Mirroring A/B across the six displays removes any side-bias confound, and the metric is well defined on both pooled and single-subject slices."
}
```

## Usage

```json
{
  "input_tokens": 16259,
  "output_tokens": 2497
}
```
