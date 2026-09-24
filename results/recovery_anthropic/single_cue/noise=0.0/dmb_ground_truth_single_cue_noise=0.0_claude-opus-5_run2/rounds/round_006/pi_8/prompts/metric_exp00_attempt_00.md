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
**Validities (n_features=6):** [0.88, 0.55, 0.72, 0.95, 0.63, 0.8]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 0]
  trial 2: A=[0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0]
  trial 3: A=[0, 1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  trial 4: A=[0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 1, 1]
  trial 5: A=[1, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 1, 1]
  trial 6: A=[0, 0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1]
  trial 7: A=[1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 8: A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 1]
  trial 9: A=[0, 0, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  trial 10: A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 11: A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 12: A=[0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  trial 13: A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 14: A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0]
  trial 15: A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0]
  trial 16: A=[1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 1]

**Rationale:** GOAL: dissociate CG-VI-2 (advocated) from GVIT-CAP-S4 (competing). The two models are nearly identical in spirit (hard count gate -> exact 0.5 on tally ties, inverse-validity 'news value' weights with a strongly saturating exponent s^omega, dilution by D^gamma, capacity term above 7 experts), so tally-tied cells and generic conflict cells are useless \u2014 they agree exactly. They differ on exactly TWO structural commitments, and every cell here is built to turn one of them into a parameter-free equality or a sign reversal.

DIFFERENCE 1 \u2014 THE ZERO FLOOR. CG-VI-2 bounds news value at zero: w = max(0, 1 - lam_eff*s^omega); a redundant expert is IGNORED. GVIT lets w go NEGATIVE, so a high-validity expert actively counts for the opposite product. Consequence: on displays in which every discriminating voice sits appreciably above the panel floor, CG-VI-2's evidence is EXACTLY ZERO (all weights clipped to 0) and it must return EXACTLY 0.500 for every lam, omega, q, kappa, beta, gamma, epsilon, whereas GVIT accumulates negative weights and REJECTS the count winner.

DIFFERENCE 2 \u2014 THE CHORUS CRITERION. CG-VI-2 reads the chorus off the bloc structure, CH = max(mA,mB) + |mA-mB| - 2, with a steep saturating gate CH^q/(CH^q+kappa), q~4. A 2-vs-1 display has CH = 1, so lam_eff = lam/(1+kappa) ~ 0.65 \u2014 only a WEAK discount. GVIT's chorus is the raw number of discriminating cues, lam_eff = lam*(D-1)/((D-1)+kappa_d) with kappa_d <= 0.45, so the same 2-vs-1 display (D=3) gets lam_eff ~ 1.38 \u2014 a NEAR-MAXIMAL discount. Hence 2-vs-1 displays are the sign-reversal engine.

PANEL: 6 experts, validities in deliberately scrambled screen order [f0=.88, f1=.55, f2=.72, f3=.95, f4=.63, f5=.80]; descending rank f3 > f0 > f5 > f2 > f4 > f1. Panel-relative seniority s = (v-.55)/.40 = f3 1.00, f0 .825, f5 .625, f2 .425, f4 .200, f1 .000. One lone floor voice (.55, s=0, full weight 1 in BOTH models), a dense upper cluster (.95/.88/.80/.72) all of whose s^omega exceed the clipping threshold for CG-VI-2 whenever the chorus is >= 2, and one buffer expert (.63). n = 6 keeps both models below their capacity terms, removing panel-size nuisance. 8 configurations x 2 exact A/B mirrors = 16 unique pairs x K = 96//16 = 6 repetitions = 96 trials. Tied cues are coded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s never tracks composition; every cell has a strict count winner (no tally ties, where both models trivially predict .5).

CELLS (A-favouring version listed; the next trial is its exact mirror). Nominal predictions P(choose A): CG-VI-2 (lam=1.75, omega=.34, q=3.95, kappa=1.70, beta=11, gamma=.90, eps=.12) vs GVIT (lam=1.45, omega=.34, beta=6.7, gamma=.90, kappa_d=.10, eps=.17).
A1 (T1/T2) UNOPPOSED STAR-PAIR (mA=2, mB=0, D=2, CH=2): A wins f3(.95), f0(.88); four cues tied. CG-VI-2: both weights clip to 0 -> E = 0 -> EXACTLY 0.500 for every parameter setting (verified at the corners lam=1.6/q=3.4/kappa=2.0/omega=.40, where lam_eff=1.345 and the clipping threshold is s>=.48 < .825). GVIT: w = -0.318 and -0.238 -> P(A) = 0.185, i.e. 80%+ choice of the WEAKLY DOMINATED product. Flagship cell.
A2 (T3/T4) UNOPPOSED BOTTOM-PAIR (2,0, CH=2): A wins f4(.63), f1(.55). Matched composition to A1, opposite identity. CG 0.94 (range .91-.96), GVIT 0.91 (range .79-.96). Lapse/slope anchor, and the A1-vs-A2 within-subject gap at IDENTICAL (nA,nB,nT) is pure identity-driven: CG predicts a drop to exactly chance, GVIT a drop to well below chance.
A3 (T5/T6) STAR-PAIR vs LONE MID (2 v 1, D=3, CH=1): A wins f3(.95), f0(.88); B wins f2(.72). CG: weak discount (lam_eff=.648) leaves w = .352/.392 vs .513 -> E=+0.23 -> 0.694 (range .57-.78 over its whole box: ALWAYS above chance). GVIT: strong discount (lam_eff=1.381) makes the stars counter-indicative -> E=-0.64 -> 0.225 (range .23-.46: never above chance). SIGN REVERSAL, ~0.47 at nominal parameters.
A4 (T7/T8) MID-PAIR vs LONE FLOOR (2 v 1, D=3, CH=1): A wins f5(.80), f2(.72); B wins f1(.55). CG: E = .449+.513-1 = -0.04 -> 0.484, essentially CHANCE (its stated signature for 2-vs-1 mid-versus-floor). GVIT: E = -1.21 -> 0.123, decisive rejection. Gap ~0.36 on a display with a clear count winner.
A5 (T9/T10) MID-TRIO vs LONE FLOOR (3 v 1, D=4, CH=3): A wins f5, f2, f4; B wins f1(.55). CG: the chorus gate now saturates (lam_eff=1.71), the three supporters clip to ~0, the floor voice keeps weight 1 -> 0.097 (range .06-.15). GVIT: lam_eff barely changes from A4 -> 0.181. The DIAGNOSTIC IS THE A4-MINUS-A5 CONTRAST: CG predicts a large POSITIVE swing (+0.39: adding a third supporter makes adherence COLLAPSE from chance to 0.10, because bloc size, not cue count, drives the redundancy construal), GVIT predicts a flat or slightly NEGATIVE swing (-0.06). Non-monotonicity in majority size is the unique CG-VI-2 chorus signature.
A6 (T11/T12) ALL-UPPER 3 v 1 (D=4, CH=3): A wins f3(.95), f0(.88), f5(.80); B wins f2(.72); no floor voice discriminates. CG: every discriminating weight clips to 0 (threshold s>=.655^(1/omega) <= .43 at the worst corner, and s2=.425 with s2^.40=.710 already exceeds 1/lam_eff) -> E = 0 -> EXACTLY 0.500 despite a +2 count margin. GVIT: E = -0.86 -> 0.219. Second parameter-free cell, and the cleanest possible demonstration that 'redundant' means MUTE (CG) rather than COUNTER-INDICATIVE (GVIT).
A7 (T13/T14) MAVERICK-IN-MAJORITY (3 v 2, D=5, CH=2): A wins f3(.95), f0(.88) AND the floor voice f1(.55); B wins f5(.80), f2(.72). CG: all four upper voices are mute, the maverick speaks for the majority -> E=+1 -> 0.858 (near-ceiling following of a majority that contains the floor voice). GVIT: the two stars on A's side subtract while the two mids on B's side add back -> 0.662. Gap ~0.20; combined with A6 it separates 'stars are mute' from 'stars vote against their own side'.
A8 (T15/T16) MID-PAIR vs LONE STAR (2 v 1, D=3, CH=1): A wins f5(.80), f2(.72); B wins f3(.95). CG 0.873, GVIT 0.586. Same majority as A4 with only the identity of the lone dissenter changed: the OPPONENT-IDENTITY contrast A8-minus-A4 is +0.39 for CG and +0.46 for GVIT, but the absolute levels differ by ~0.29-0.36 in both cells, which pins each model's lam_eff at CH=1 / D=3 \u2014 exactly the quantity on which the two chorus rules disagree.

SUMMARY OF PREDICTED PROFILES (A1..A8): CG-VI-2 = .500, .94, .69, .48, .10, .500, .86, .87 ; GVIT = .19, .91, .23, .12, .18, .22, .66, .59. The models agree only on the anchor A2, and disagree in SIGN relative to chance on A1, A3, A4 and A6.

INTERPRETATION. Exact chance on A1 and A6 (two cells with strict count winners, one with a +2 margin), above-chance following of the star pair in A3, chance in A4 collapsing to ~0.10 in A5 when a third mid supporter is added, and near-ceiling in A7 uniquely support CG-VI-2. Systematic BELOW-chance responding in A1, A3, A4 and A6 (i.e. choosing the weakly dominated product in the unopposed star display) with a flat A4-A5 contrast uniquely supports GVIT-CAP-S4.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror so side bias cancels exactly; validity rank is decoupled from screen position so positional heuristics cannot mimic validity-based weighting; tie coding alternates 1-1 / 0-0; no cell is tally-tied or otherwise degenerate, and no cell has the two models making the same prediction except the deliberate lapse anchor A2.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** **Consensus-Graded Validity Inversion with a Non-Negative News Floor (CG-VI-2).**

People told the stated accuracies of a panel of binary expert ratings do not weight experts *by* accuracy; they weight them by how much *news* a rating carries. Six commitments.

**(1) The count comparator is a gate, not a weight.** Before any expert identity is consulted, an unweighted feature-wise comparison asks a purely ordinal question: does one product win on strictly more attributes? If not — equal win counts, including wholly non-discriminating displays — the display is construed as non-diagnostic and the subject flips a mental coin: exactly 0.5, with NO validity tie-break and NO take-the-best fallback. Parameter-free at every panel composition (Exp 1, Exp 3's tied cells, Exp 7's tied cells, Exp 10's C3 cell).

**(2) When a count winner exists, weights are inverse in stated validity, measured against the panel's least-accurate voice and strongly SATURATING.** s_j = (v_j - v_min)/(v_max - v_min); news value w_j = 1 - lambda_eff * s_j^omega with omega well below 1 (~0.3-0.4). Because the transform is strongly concave, *any* expert appreciably above the panel floor is already heard as "one of the accurate ones" — merely restating the obvious consensus quality of the product — while the panel's genuinely idiosyncratic bottom voice is heard as private, surprising information and keeps full credit.

**(3) THE NEW COMMITMENT — news value is bounded below at ZERO: a redundant expert is IGNORED, never counted against his own side.** w_j = max(0, 1 - lambda_eff*s_j^omega). Redundancy is a reason to stop listening, not a reason to believe the opposite. This is what generates the signature pattern that no signed-inversion model produces: *the side holding the panel's least-accurate voice wins, and when neither side holds it the plain tally decides*. A majority made of stars plus the maverick is followed near ceiling (the stars are mute, the maverick speaks for the majority), whereas the numerically identical majority made of mid-panel experts is rejected when the maverick sits in the minority. It also stops dense, unanimous, high-validity displays from being actively rejected (they merely lose force and drift toward chance), which preserves plain-tally adherence.

**(4) The redundancy construal is graded by how big and how lopsided the visible consensus is, and it saturates sharply.** The subjective chorus is read off the display as CH = max(m_A, m_B) + |m_A - m_B| - 2, i.e. the size of the larger bloc *plus* the net excess of same-direction voices. lambda_eff = lambda * CH^q/(CH^q + kappa) with q ~ 3.4-4.5 and kappa ~ 1.4-2.0. Hence: a solitary discriminating voice (CH = 0) is heard at full face value; a 2-vs-1 display (CH = 1) is only weakly discounted, so a mid pair opposed by a lone floor voice lands essentially at chance; a 3-vs-2, 2-vs-0 or 3-vs-1 display (CH >= 2) is discounted almost maximally, so the floor voice takes over; and everything denser is saturated.

**(5) Confidence is compressed by the sheer number of discriminating cues (D^gamma) and capped by an attention lapse epsilon.** Evidence spread across many mutually contradicting attributes is subjectively weaker than the same net weight carried by a sparse clean split.

**(6) The inversion needs the whole validity ordering in mind, so it degrades past working-memory span (~7 voices).** For n > 7 the discount shrinks (lambda scaled by 1 - psi*(n-7)) and integration gets noisier (divisor multiplied by 1 + chi*(n-7)), so eight-expert star-backed majorities sit at chance rather than being reversed, while plain tally adherence there is preserved.

**Cross-design signature:** exact chance on tally ties at any validity composition; the panel's least-accurate discriminating voice decides whenever it is present on exactly one side, at every panel size up to seven; near-chance for 2-vs-1 mid-versus-floor displays but decisive rejection of 3-cue majorities opposed by the same floor voice; near-ceiling following of majorities that themselves contain the floor voice; lone voices taken at face value; unanimous all-star displays drifting to chance rather than reversing; and markedly attenuated inversion on eight-expert panels.

**Parameters:**
- lam: [1.60, 1.90]
- omega: [0.28, 0.40]
- q_m: [3.40, 4.50]
- kappa_m: [1.40, 2.00]
- beta: [9.5, 12.5]
- gamma: [0.80, 1.00]
- psi: [0.42, 0.58]
- chi: [0.60, 1.10]
- epsilon: [0.08, 0.18]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---------------- parse the stimulus ----------------
    a = None
    b = None
    if isinstance(state, dict):
        for ka, kb in (('option_a_ratings', 'option_b_ratings'),
                       ('a_ratings', 'b_ratings'),
                       ('A', 'B'), ('a', 'b')):
            if ka in state and kb in state:
                try:
                    a = np.asarray(list(state[ka]), dtype=float).ravel()
                    b = np.asarray(list(state[kb]), dtype=float).ravel()
                except Exception:
                    a = None
                    b = None
                break
    if a is None or b is None:
        try:
            arr = np.asarray(state, dtype=float)
            arr = arr.reshape(2, -1)
            a = np.asarray(arr[0], dtype=float).ravel()
            b = np.asarray(arr[1], dtype=float).ravel()
        except Exception:
            return np.array([0.5, 0.5], dtype=float)

    if a is None or b is None or a.size == 0 or a.size != b.size:
        return np.array([0.5, 0.5], dtype=float)

    n = int(a.size)

    # ------------- Commitment 1: parameter-free count gate -------------
    d = np.sign(a - b)
    iA = np.where(d > 0)[0]
    iB = np.where(d < 0)[0]
    mA = int(iA.size)
    mB = int(iB.size)
    if mA == mB:                       # includes wholly non-discriminating displays
        return np.array([0.5, 0.5], dtype=float)

    D = float(mA + mB)
    M = float(abs(mA - mB))            # net consensus margin
    c_side = float(max(mA, mB))        # size of the larger bloc
    CH = c_side + M - 2.0              # visible chorus argument
    if (not np.isfinite(CH)) or CH < 0.0:
        CH = 0.0

    # ---------------- parameters ----------------
    def _g(name, default, lo, hi):
        try:
            x = float(parameters.get(name, default))
        except Exception:
            x = float(default)
        if not np.isfinite(x):
            x = float(default)
        return float(min(max(x, lo), hi))

    lam = _g('lam', 1.75, 0.0, 4.0)
    omega = _g('omega', 0.34, 0.10, 1.50)
    q_m = _g('q_m', 3.95, 0.50, 8.00)
    kappa_m = _g('kappa_m', 1.70, 0.05, 8.00)
    beta = _g('beta', 11.0, 0.0, 40.0)
    gamma = _g('gamma', 0.90, 0.0, 2.5)
    psi = _g('psi', 0.50, 0.0, 1.0)
    chi = _g('chi', 0.85, 0.0, 4.0)
    eps = _g('epsilon', 0.12, 0.0, 0.6)

    # ---------------- stated validities ----------------
    v = None
    try:
        vr = parameters.get('validities', None) if hasattr(parameters, 'get') else None
        if vr is not None:
            v = np.asarray(vr, dtype=float).ravel()
    except Exception:
        v = None
    if v is None or v.size != n:
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)

    # ------------- Commitment 2: saturating inverse-validity news value -------------
    v_min = float(np.min(v))
    v_max = float(np.max(v))
    span = v_max - v_min
    if (not np.isfinite(span)) or span <= 1e-9:
        s = np.zeros(n, dtype=float)
    else:
        s = np.clip((v - v_min) / span, 0.0, 1.0)
    f = np.power(np.clip(s, 0.0, 1.0), omega)

    # ------------- Commitment 6: capacity-limited construal (n > 7) -------------
    over = max(0.0, float(n) - 7.0)
    lam_n = lam * max(0.0, 1.0 - psi * over)
    if (not np.isfinite(lam_n)) or lam_n < 0.0:
        lam_n = 0.0

    # ------------- Commitment 4: chorus(size+margin)-graded redundancy -------------
    try:
        CHq = float(CH) ** q_m
    except Exception:
        CHq = float(CH)
    if (not np.isfinite(CHq)) or CHq < 0.0:
        CHq = float(CH)
    lam_eff = lam_n * CHq / (CHq + kappa_m)
    if (not np.isfinite(lam_eff)) or lam_eff < 0.0:
        lam_eff = 0.0

    # ------------- Commitment 3: news value is floored at ZERO -------------
    w = np.maximum(0.0, 1.0 - lam_eff * f)

    E = float(np.sum(w[iA]) - np.sum(w[iB]))
    if not np.isfinite(E):
        E = 0.0

    # ------------- Commitment 5: dilution by cue count, plus overload noise -------------
    den = float(D) ** gamma
    if (not np.isfinite(den)) or den <= 0.0:
        den = 1.0
    den = den * (1.0 + chi * over)
    if (not np.isfinite(den)) or den <= 0.0:
        den = 1.0

    z = beta * E / den
    if not np.isfinite(z):
        z = 0.0
    z = float(min(max(z, -30.0), 30.0))

    p_a = 1.0 / (1.0 + np.exp(-z))

    # attention lapse
    p_a = (1.0 - eps) * p_a + eps * 0.5
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))

    p = np.array([p_a, 1.0 - p_a], dtype=float)
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
    if (not np.isfinite(s)) or s <= 0:
        probs = np.ones_like(probs) / float(len(probs))
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** **Capacity-Gated Strongly-Saturating Redundancy Validity-Inversion Tallying (GVIT-CAP-S4).**

Five commitments describe how people combine binary expert ratings whose stated validities they have been told.

**(1) The count comparator is a gate, not a weight.** Before any evaluation of *whose* rating is whose, the subject asks a purely ordinal question: does one product win on more attributes than the other? This is answered with unweighted feature-wise comparisons. If the answer is "no" (equal numbers of winning attributes, including displays where nothing discriminates), the display is construed as *non-diagnostic*, no candidate is generated, and the subject flips a mental coin. Stated validities are **never** used as a tie-break. This is a hard, parameter-free prediction: on tally-tied, side-counterbalanced profiles adherence to the most-valid discriminating cue is exactly chance whatever the validity vector.

**(2) When a majority exists, integration uses inverse-informativeness weights.** The subjective news value of a rating is a smooth, panel-relative *decreasing* function of stated validity: s_j = (v_j - v_min)/(v_max - v_min) within the displayed panel, w_j = 1 - lambda_eff*f(s_j). A near-perfect expert is heard as merely restating the obvious consensus quality of the product, so his rating adds almost nothing (and, when lambda_eff*f > 1, is mildly counter-indicative); a barely-diagnostic expert is surprising and is treated as genuine private information carrying full weight. Evidence is the single scalar E = sum_j w_j*sign(a_j - b_j).

**(3) The redundancy construal saturates VERY STRONGLY just above the panel floor.** Redundancy is not judged on a linear scale of stated accuracy: "being one of the accurate ones" is a nearly categorical construal. An expert only has to be slightly above the panel's least-accurate voice before he is already heard as echoing the consensus. Formally the discount is a strongly concave function of panel-relative validity, f(s) = s^omega with omega well below 1 (~0.26-0.42), so the discount rises steeply just above the panel floor and then flattens. Two consequences follow that neither a linear nor a mildly concave inversion produces. (i) A majority carried by *middling* experts (the 2nd-5th most accurate voices) carries almost no subjective evidence: mid-panel support is discounted nearly as heavily as star support, so a mid-backed majority facing a lone star is followed only weakly, close to chance. (ii) Conversely, when the count loser is backed by the panel's genuinely idiosyncratic bottom voices, the anti-count evidence is *strengthened*, sharpening the reversal. So the model separates three displays that a linear inversion renders nearly identical: majority-by-mid-experts (followed barely above chance), majority-by-bottom-voices (followed near ceiling), and star-backed majorities (reversed toward floor).

**(4) Redundancy is comparative; evidence spread over many contradicting cues is compressed.** An expert is discounted only because other experts are heard as echoing him, so the inversion scales with how many voices actually speak on the trial: lambda_eff = lambda_n*(D-1)/((D-1)+kappa_D), where D is the number of discriminating cues; when exactly one expert differentiates the products there is no chorus and his rating is taken at face value. Confidence in E is then compressed by the number of mutually contradicting cues, D^gamma.

**(5) The inversion is a capacity-limited construal: it collapses when the panel exceeds working-memory span.** Discounting the star requires holding the validity ordering of the whole panel in mind, which is possible only up to roughly seven experts. Beyond capacity the ordering blurs, so the redundancy construal washes out toward equal weighting (lambda_n = lambda*(1 - psi*o)) and integration becomes noisier (slope divided by 1 + chi*o), where o = logistic((n - 7.5)/0.2). Finally an attention lapse epsilon replaces the decision by a coin flip.

Cross-design signature: exact chance on tally-tied profiles; strong systematic reversal of star-backed majorities on panels of <=7 experts; near-chance for the same star-backed profiles on panels of 8+ experts, with preserved tally adherence there for broad-mixture majorities; near-ceiling following of a lone discriminating expert; only weakly above-chance following of mid-panel-backed majorities that oppose a lone star; and markedly attenuated confidence whenever the weighted evidence is spread across many mutually contradicting cues.

**Parameters:**
- lam: [1.10, 1.80]
- beta: [5.8, 7.6]
- gamma: [0.74, 1.06]
- kappa_d: [0.01, 0.45]
- psi: [0.32, 0.48]
- chi: [0.75, 1.15]
- epsilon: [0.07, 0.27]
- omega: [0.26, 0.42]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---------- parse the stimulus ----------
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
        try:
            stim = np.asarray(state, dtype=float)
            if stim.ndim == 1:
                stim = stim.reshape(2, -1)
            elif stim.ndim > 2:
                stim = stim.reshape(2, -1)
            if stim.shape[0] != 2:
                stim = stim.reshape(2, -1)
            a = np.asarray(stim[0], dtype=float).ravel()
            b = np.asarray(stim[1], dtype=float).ravel()
        except Exception:
            return np.array([0.5, 0.5], dtype=float)

    n = int(a.shape[0])
    if n == 0 or int(b.shape[0]) != n:
        return np.array([0.5, 0.5], dtype=float)

    # ---------- feature-wise comparison ----------
    d = np.sign(a - b)                       # +1 A wins cue, -1 B wins cue, 0 tie
    nA = int(np.sum(d > 0))
    nB = int(np.sum(d < 0))
    D = nA + nB

    # ---- Commitment 1: the count comparator is a gate. No majority -> guess.
    if D == 0 or nA == nB:
        return np.array([0.5, 0.5], dtype=float)

    # ---------- parameters ----------
    def _get(name, default):
        try:
            return float(parameters.get(name, default))
        except Exception:
            return float(default)

    lam = _get('lam', 1.45)
    beta = _get('beta', 6.7)
    gamma = _get('gamma', 0.90)
    kappa_d = _get('kappa_d', 0.10)
    psi = _get('psi', 0.40)
    chi = _get('chi', 0.95)
    eps = _get('epsilon', 0.17)
    omega = _get('omega', 0.34)              # saturating-redundancy exponent

    lam = float(min(max(lam, 0.0), 2.5))
    beta = float(min(max(beta, 0.0), 30.0))
    gamma = float(min(max(gamma, 0.0), 2.5))
    kappa_d = float(min(max(kappa_d, 1e-6), 3.0))
    psi = float(min(max(psi, 0.0), 1.0))
    chi = float(min(max(chi, 0.0), 5.0))
    eps = float(min(max(eps, 0.0), 0.6))
    omega = float(min(max(omega, 0.2), 1.5))

    # ---------- stated validities ----------
    v = None
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is not None:
        try:
            v = np.asarray(v_raw, dtype=float).ravel()
        except Exception:
            v = None
    if v is None or v.shape[0] != n:
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)

    v_min = float(np.min(v))
    v_max = float(np.max(v))
    rng = v_max - v_min
    if not np.isfinite(rng) or rng <= 1e-9:
        s = np.zeros(n, dtype=float)          # no validity contrast in the panel
    else:
        s = (v - v_min) / rng                 # panel-relative informativeness, in [0,1]

    # ---- Commitment 3: the redundancy construal SATURATES above the floor ----
    # strongly concave transform: being appreciably above the panel's least-accurate
    # voice is already enough to be heard as "one of the accurate ones".
    s = np.power(np.clip(s, 0.0, 1.0), omega)

    # ---- Commitment 5: capacity overload of the panel (~7 experts) --------
    N_CAP = 7.5
    W_CAP = 0.20
    arg_o = (float(n) - N_CAP) / W_CAP
    arg_o = float(min(max(arg_o, -60.0), 60.0))
    o = 1.0 / (1.0 + np.exp(-arg_o))

    lam_n = lam * (1.0 - psi * o)
    if not np.isfinite(lam_n) or lam_n < 0.0:
        lam_n = 0.0

    # ---- Commitment 4: redundancy is comparative (needs a chorus) -------
    lam_eff = lam_n * (float(D) - 1.0) / ((float(D) - 1.0) + kappa_d)
    if not np.isfinite(lam_eff) or lam_eff < 0.0:
        lam_eff = 0.0

    # ---- Commitment 2: inverse-informativeness weights ------------------
    w = 1.0 - lam_eff * s
    E = float(np.sum(w * d))

    # ---- Commitment 4b: compression by the number of contradicting cues --
    denom = float(D) ** gamma
    if not np.isfinite(denom) or denom <= 0.0:
        denom = 1.0
    overload_noise = 1.0 + chi * o
    if not np.isfinite(overload_noise) or overload_noise <= 0.0:
        overload_noise = 1.0
    denom = denom * overload_noise

    z = beta * E / denom
    if z > 30.0:
        z = 30.0
    elif z < -30.0:
        z = -30.0

    p_a = 1.0 / (1.0 + np.exp(-z))

    # ---- attention lapse ------------------------------------------------
    p_a = (1.0 - eps) * p_a + eps * 0.5
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))

    p = np.array([p_a, 1.0 - p_a], dtype=float)
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
        probs = np.ones_like(probs) / float(len(probs))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # stated validities of the six experts in screen order (design-time constant)\n    V = np.array([0.88, 0.55, 0.72, 0.95, 0.63, 0.80], dtype=float)\n    THR = 0.70  # 'upper panel' voices: 0.72, 0.80, 0.88, 0.95\n\n    hits = 0\n    tot = 0\n\n    a_col = data['option_a_ratings'].tolist()\n    b_col = data['option_b_ratings'].tolist()\n    r_col = data['response'].tolist()\n\n    for a_raw, b_raw, r_raw in zip(a_col, b_col, r_col):\n        try:\n            a = np.asarray([float(x) for x in a_raw], dtype=float)\n            b = np.asarray([float(x) for x in b_raw], dtype=float)\n            r = int(r_raw)\n        except Exception:\n            continue\n        if a.size == 0 or a.size != b.size:\n            continue\n        if a.size == V.size:\n            v = V\n        else:\n            v = np.full(a.size, 1.0, dtype=float)\n\n        d = np.sign(a - b)\n        idx = np.nonzero(d)[0]\n        if idx.size == 0:\n            continue\n        nA = int(np.sum(d > 0))\n        nB = int(np.sum(d < 0))\n        if nA == nB:\n            continue  # no strict count winner: both theories predict chance\n        # keep only displays where EVERY discriminating voice is an upper-panel expert\n        if np.any(v[idx] < THR):\n            continue\n\n        winner = 0 if nA > nB else 1  # 0 == chose A, 1 == chose B\n        tot += 1\n        if r == winner:\n            hits += 1\n\n    if tot == 0:\n        return 0.5\n    return float(hits) / float(tot)",
  "rationale": "The two models differ structurally only in whether a 'redundant' (high-validity) expert is MUTE (CG-VI-2: news value floored at zero) or COUNTER-INDICATIVE (GVIT: weights go negative). That difference can only show up on displays in which the discriminating voices are ALL appreciably above the panel floor, i.e. where no low-validity maverick (0.55 or 0.63) speaks. This metric selects exactly those trials (all discriminating features have validity >= 0.72, and a strict count winner exists) and scores adherence to the plain count winner. In this design those are the A1 (T1/T2, unopposed star pair, 2-0), A3 (T5/T6, star pair vs lone mid, 2-1), A6 (T11/T12, all-upper 3-1) and A8 (T15/T16, mid pair vs lone star) cells \\u2014 48 of the 96 trials, so the binomial component of per-subject noise is small.\n\nPredictions: under CG-VI-2 all discriminating weights clip to zero in A1 and A6, giving EXACTLY 0.500 there for every parameter setting (zero between-subject parameter variance), plus ~0.68 in A3 and ~0.87 in A8, so the pooled adherence lands near 0.63-0.65. Under GVIT the same experts carry negative weight, so the count winner is actively rejected in A1 (~0.25), A3 (~0.21) and A6 (~0.19) and only weakly followed in A8 (~0.59), giving pooled adherence near 0.30-0.33. Predicted gap ~0.30 in a quantity bounded in [0,1], with tight per-subject distributions (two of the four cells are parameter-free under CG, and the deliberately excluded floor-voice cells \\u2014 A2, A4, A5, A7 \\u2014 are the ones that carry most of each model's parameter-driven spread). Crucially it is not trivially true for CG: it would come out near chance or below if humans actually treat high-validity ratings as counter-indicative."
}
```

## Usage

```json
{
  "input_tokens": 15261,
  "output_tokens": 11092
}
```
