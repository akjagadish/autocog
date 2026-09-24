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
**Validities (n_features=7):** [0.82, 0.53, 0.95, 0.78, 0.91, 0.72, 0.87]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  trial 2: A=[1, 0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 0, 1, 0]
  trial 3: A=[0, 1, 1, 1, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  trial 4: A=[0, 1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1, 0, 1]
  trial 5: A=[1, 1, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1, 0]
  trial 6: A=[1, 0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1, 0]
  trial 7: A=[1, 0, 1, 1, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1]
  trial 8: A=[0, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1]
  trial 9: A=[0, 1, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1, 0]
  trial 10: A=[0, 1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1, 0]
  trial 11: A=[1, 0, 0, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 0, 0]
  trial 12: A=[0, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 1, 0]
  trial 13: A=[1, 1, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0, 1]
  trial 14: A=[1, 1, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1]
  trial 15: A=[0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 1]
  trial 16: A=[0, 0, 1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0, 0]

**Rationale:** GOAL: dissociate MBM-PG7 (advocated) from CG-VI-2 (competing). The two models share a parameter-free count gate (tally ties -> exactly 0.5), panel-relative seniority s=(v-vmin)/(vmax-vmin), an inverse-validity 'news value' logic, dilution by D^gamma, a lapse, and a capacity term past 7 voices. They also AGREE on the classic maverick reversals, so those cells are wasted except as anchors. They differ on three structural points, and every cell below converts one of them into a parameter-free equality, a large magnitude gap, or a sign flip.

DIFFERENCE 1 - SOLO CUES (D=1). CG's chorus argument CH = max(mA,mB)+|mA-mB|-2 is 0 for a solo discriminating cue, so lam_eff=0, EVERY weight is exactly 1, E=1, den=1, z=beta in [9.5,12.5] -> P = 1-eps/2 in [0.91,0.96] IDENTICALLY for a solo .95, a solo .78 and a solo .53 expert (a three-way parameter-free equality at near ceiling). MBM has no such release: a solo star is outside the mute band and contributes only the thin residual eta -> z = beta*eta in [0.13,0.25] -> P in [0.53,0.56]; a solo low-upper (s=.595) contributes eta+0.405c -> P ~ 0.63-0.72; a solo floor voice contributes 1+eta+c -> P ~ 0.83-0.93. So MBM must produce a steep monotone inverse-validity LADDER (~.54/.68/.89) where CG must produce a FLAT line (~.94/.94/.94). Cell C1 alone has DISJOINT parameter boxes (CG >= .91 vs MBM <= .56).

DIFFERENCE 2 - MUTE MEANS EXACTLY ZERO (CG) VS A GRADED RESIDUAL GRADIENT (MBM). CG clips w = max(0, 1-lam_eff*s^omega): once the bloc chorus is CH>=2 every voice appreciably above the panel floor is EXACTLY mute, so any display in which no floor voice discriminates has E=0 and returns EXACTLY 0.500 for every lam, omega, q, kappa, beta, gamma, epsilon - even a zero-conflict dominance display (C4, C5) or a +2 tally margin (C6). MBM's muted experts keep r = eta + c(1-s), so its prediction tracks the seniority composition of the display: above chance and graded. Clipping was verified at every corner of CG's box (at CH=2 clipping needs s >= .476 at the worst corner; the cues used have s = .595/.690/.905/1.00; at CH=3 it needs s >= .345 and the lowest cue used is s = .452).

DIFFERENCE 3 - THE 2-vs-1 REGIME. CG's chorus gate is steep in BLOC structure: a 2-vs-1 display has CH=1, so lam_eff = lam/(1+kappa) in [0.53,0.79] - only a WEAK discount, all weights stay positive, and with beta ~ 11 the weighted tally is followed decisively. MBM has no bloc-size release for out-of-band voices, only the residual gradient and a small beta (0.7-0.9), so two senior voices (residual .23+.42) lose to one low-upper voice (residual 1.04): C7 is a SIGN FLIP with disjoint boxes (CG in [0.59,0.80] vs MBM in [0.455,0.49]).

PANEL: 7 experts, validities in deliberately scrambled screen order [f0=.82, f1=.53, f2=.95, f3=.78, f4=.91, f5=.72, f6=.87]; vmin=.53, vmax=.95, span=.42, so s = .690/.000/1.000/.595/.905/.452/.810. EXACTLY ONE voice sits at the panel floor (f1=.53, s=0: a maverick for both models, weight 1 in CG and inside MBM's band for every theta), and every other voice has s >= .452, far outside MBM's widest possible band (theta <= .20) and inside CG's clipping region whenever CH >= 2. This removes the band-width nuisance completely and makes the dissociation rest only on the three commitments above. n=7 keeps both models at their capacity threshold (over = 0), removing panel-size nuisance.

DESIGN: 8 configurations x 2 exact A/B mirrors = 16 unique pairs x K = 96//16 = 6 repetitions = 96 trials. No cell is tally-tied (both models would trivially return 0.5) and every trial requires reading several ratings.

CELLS (A-favouring version listed first; the next trial is its exact mirror). Nominal P(choose A): MBM-PG7 (theta=.16, rho=2.1, eta=.23, c=2.0, beta=.80, gamma=.95, alpha=52, kappa=205, p=6.4, lam_own=.36, eps=.09) vs CG-VI-2 (lam=1.75, omega=.34, q=3.95, kappa=1.70, beta=11, gamma=.90, eps=.12).
C1 (T1/T2) SOLO STAR: only f2(.95) discriminates. MBM 0.54 [.53,.56] vs CG 0.94 [.91,.96]. FLAGSHIP - disjoint boxes.
C2 (T3/T4) SOLO LOW-UPPER: only f3(.78) discriminates. MBM 0.68 vs CG 0.94.
C3 (T5/T6) SOLO FLOOR: only f1(.53) discriminates. MBM 0.89 vs CG 0.94. Deliberate near-agreement: lapse/ceiling anchor and third rung of the ladder. CG must make C1=C2=C3 exactly; MBM must make them a steep monotone ladder.
C4 (T7/T8) UNOPPOSED LOW PAIR (mA=2, mB=0, CH=2): A wins f3(.78) and f0(.82), five ties. CG EXACTLY 0.500 at every box corner (both weights clip) vs MBM 0.67 [.62,.73]. Chance responding on a weakly dominant display is CG's signature.
C5 (T9/T10) UNOPPOSED SENIOR PAIR (2,0, CH=2): A wins f4(.91) and f2(.95). CG EXACTLY 0.500 (a parameter-free equality with C4) vs MBM 0.56. The C4-minus-C5 contrast at IDENTICAL composition is pure seniority: CG must give exactly 0 (at exactly chance), MBM ~ +0.11 with both above chance.
C6 (T11/T12) ALL-UPPER 3v1: A wins f5(.72), f3(.78), f0(.82); B wins f2(.95). CG EXACTLY 0.500 despite a +2 count margin; MBM 0.64.
C7 (T13/T14) 2v1 SENIOR PAIR vs LONE LOW-UPPER (CH=1): A wins f2(.95), f4(.91); B wins f3(.78). CG 0.72 [.59,.80] vs MBM 0.475 [.455,.49]. SIGN FLIP with disjoint boxes: CG follows the weighted majority decisively, MBM lets the single low-upper's fat residual cancel two thin senior residuals.
C8 (T15/T16) MAVERICK vs SENIOR TRIO: A wins f1(.53) alone; B wins f2(.95), f4(.91), f6(.87). MBM 0.93 vs CG 0.90. Deliberate agreement anchor AND self-diagnostic: WADD/TTB predict ~0.05 here, so systematic choice of B falsifies both models.

PREDICTED PROFILE (C1..C8): MBM-PG7 = .54, .68, .89, .67, .56, .64, .475, .93 ; CG-VI-2 = .94, .94, .94, .500, .500, .500, .72, .90. They agree only on the deliberate anchors C3 and C8.

INTERPRETATION: a steep solo ladder (~.54/.68/.89), systematically ABOVE-chance responding on the all-upper cells C4/C5/C6 with the low-seniority display exceeding the senior display, and at-or-below-chance responding on C7 uniquely support MBM-PG7. Flat near-ceiling adherence across all three solo cells, exact chance on C4, C5 and C6 (a dominance display, a matched dominance display and a +2-margin display), and decisive above-chance following of the senior pair in C7 uniquely support CG-VI-2.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror so side bias cancels exactly; validity rank is decoupled from screen position so positional heuristics cannot mimic validity-based weighting; tied cues are coded sometimes 1-1 and sometimes 0-0 within and across cells so the number of displayed 1s never tracks composition or seniority; every trial has a strict count winner.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** **Mute-Band Maverick Arbitration with a HARD (near-step) Band and a Stronger Capacity Noise Term (MBM-PG7).**

People who are told the accuracies of a panel of binary expert ratings do not weight experts *by* accuracy; they weight them by how much *news* a rating carries, and news is a steeply thresholded, panel-relative quantity. Six commitments.

**(1) The count comparator is a parameter-free gate.** Before any expert identity is consulted, an unweighted feature-wise comparison asks a purely ordinal question: does one product win on strictly more attributes? If not — equal win counts, including wholly non-discriminating displays — the display is construed as non-diagnostic and the subject flips a mental coin: exactly 0.5, with NO validity tie-break and NO take-the-best fallback, at every panel composition and size.

**(2) HARD, NEAR-STEP MUTE BAND, CHORUS-INDEPENDENT.** Seniority is read off the *displayed panel*: s_j = (v_j − v_min)/(v_max − v_min). Only voices lying within a narrow band theta (≈12-20% of the panel span) above the panel floor are heard as carrying private, surprising information, and *within* that band the listening function is now strongly convex: m_j = max(0, 1 − s_j/theta)^rho with rho ≈ 1.7-2.5. The maverick construal is therefore close to categorical — only the panel's genuinely idiosyncratic BOTTOM voice speaks with full force; a voice that merely sits low-ish (a fifth to a half of the way into the band) is already nearly as mute as a star. Muting is unconditional: a lone star cue is as mute as a star inside a bloc.

**(3) GRADED, STRONGLY ASYMMETRIC CHORUS AMPLIFICATION WITH A LATE, STEEP KNEE — and the post-knee amplification is DEEP.** A maverick's voice is loud only in proportion to the redundancy it is heard against, and redundancy is *asymmetric*: the bloc a cue CONTRADICTS is the chorus proper, while the bloc a cue sits INSIDE echoes it only weakly and therefore counts at a discount lam_own. For cue j: C_j = (m_opp − 1) + lam_own·(m_own − 1) when the cue is opposed, and C_j = (m_own − 1) when the display is unanimous. Amplification G_j = 1 + alpha·C_j^p/(C_j^p + kappa) with a LATE, STEEP knee (p ≈ 6.4, kappa ≈ 205) and a LARGE post-knee asymptote (alpha ≈ 52). Because (2) is now near-categorical, this large amplification is applied essentially only to true floor voices: a partial, mid-band voice can no longer borrow a 20-30x chorus gain and masquerade as a maverick.

**(4) THE PANEL GRADIENT: the mute band is quiet, not empty, and the listening floor is THIN.** Muted experts do not vanish; each discriminating cue contributes a residual r_j = eta + c·(1 − s_j) — a thin flat listening floor plus an inverse-seniority gradient. Displays in which no voice is inside the mute band hover near chance at any count margin, while the preserved slope still generates the anti-validity contrasts on conflict displays.

**(5) DILUTION AND LAPSE.** Net evidence E = Σ_j dir_j·(G_j·m_j + r_j) is compressed by the number of mutually contradicting cues, D^gamma, and mixed with an attention lapse epsilon.

**(6) CAPACITY IS BOTH A FLATTENING AND A SUBSTANTIAL NOISE SOURCE.** Holding the whole seniority ordering in mind is possible only up to about seven voices. For n > 7 the amplification collapses toward 1 AND integration becomes markedly noisier (divisor × (1 + chi·(n−7)) with chi now appreciably larger), so eight-expert star-backed majorities sit at chance and eight-expert plain tally adherence is systematically LOWER than the six-expert equivalent even though the tally still governs direction.

**Cross-design signature:** exact chance on tally ties at any composition; chance on displays whose every discriminating voice is upper-band, at ANY count margin; only the panel's bottom voice (not the second- or third-from-bottom) can reverse a majority; a lone floor voice opposed by a mere PAIR lands at chance while the same voice opposed by a three-cue bloc reverses it to ≈.15; near-ceiling following of dense unanimous displays and of majorities containing the floor voice; and visibly attenuated, noisy integration on eight-expert panels.

**Parameters:**
- theta: [0.12, 0.20]
- rho: [1.7, 2.5]
- eta: [0.18, 0.28]
- c_res: [1.7, 2.3]
- beta: [0.70, 0.90]
- gamma: [0.88, 1.02]
- alpha: [46, 58]
- kappa: [180, 230]
- p_ch: [6.1, 6.7]
- lam_own: [0.28, 0.45]
- psi: [0.10, 0.35]
- chi: [0.26, 0.52]
- epsilon: [0.04, 0.14]
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
            arr = np.asarray(state, dtype=float).reshape(2, -1)
            a = np.asarray(arr[0], dtype=float).ravel()
            b = np.asarray(arr[1], dtype=float).ravel()
        except Exception:
            return np.array([0.5, 0.5], dtype=float)

    if a is None or b is None or a.size == 0 or a.size != b.size:
        return np.array([0.5, 0.5], dtype=float)

    n = int(a.size)

    # ---------- Commitment 1: parameter-free count gate ----------
    d = np.sign(a - b)
    iA = np.where(d > 0)[0]
    iB = np.where(d < 0)[0]
    mA = int(iA.size)
    mB = int(iB.size)
    if mA == mB:                      # includes wholly non-discriminating displays
        return np.array([0.5, 0.5], dtype=float)

    D = float(mA + mB)

    # ---------------- parameters ----------------
    def _g(name, default, lo, hi):
        try:
            x = float(parameters.get(name, default))
        except Exception:
            x = float(default)
        if not np.isfinite(x):
            x = float(default)
        return float(min(max(x, lo), hi))

    theta = _g('theta', 0.16, 0.02, 0.90)
    rho = _g('rho', 2.1, 0.20, 4.00)
    eta = _g('eta', 0.23, 0.00, 1.00)
    c_res = _g('c_res', 2.0, 0.00, 6.00)
    beta = _g('beta', 0.80, 0.00, 5.00)
    gamma = _g('gamma', 0.95, 0.00, 2.50)
    alpha = _g('alpha', 52.0, 0.00, 400.0)
    kappa = _g('kappa', 205.0, 0.05, 2000.0)
    p_ch = _g('p_ch', 6.4, 0.50, 12.0)
    lam_own = _g('lam_own', 0.36, 0.00, 1.00)
    psi = _g('psi', 0.22, 0.00, 1.00)
    chi = _g('chi', 0.39, 0.00, 3.00)
    eps = _g('epsilon', 0.09, 0.00, 0.60)

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

    v_min = float(np.min(v))
    v_max = float(np.max(v))
    span = v_max - v_min
    if (not np.isfinite(span)) or span <= 1e-9:
        s = np.zeros(n, dtype=float)
    else:
        s = np.clip((v - v_min) / span, 0.0, 1.0)

    # ---------- Commitment 2: hard, near-step, chorus-independent mute band ----------
    band = np.clip(1.0 - s / max(theta, 1e-6), 0.0, 1.0)
    mav = np.power(band, rho)                      # maverick strength, 0 outside the band

    # ---------- Commitment 3: GRADED chorus amplification (cue-specific) ----------
    def _G(base):
        base = float(max(0.0, base))
        try:
            bp = base ** p_ch
        except Exception:
            bp = base
        if (not np.isfinite(bp)) or bp < 0.0:
            bp = 0.0
        g = 1.0 + alpha * bp / (bp + kappa)
        if (not np.isfinite(g)) or g < 1.0:
            g = 1.0
        return g

    # the bloc a cue OPPOSES counts fully; the bloc it sits INSIDE counts at
    # weight lam_own; an unopposed cue hears only the consensus it belongs to.
    if mB > 0:
        base_A = (float(mB) - 1.0) + lam_own * (float(mA) - 1.0)
    else:
        base_A = (float(mA) - 1.0)
    if mA > 0:
        base_B = (float(mA) - 1.0) + lam_own * (float(mB) - 1.0)
    else:
        base_B = (float(mB) - 1.0)

    G_A = _G(base_A)
    G_B = _G(base_B)

    # ---------- Commitment 6: capacity limit past ~7 voices ----------
    over = max(0.0, float(n) - 7.0)
    if over > 0.0:
        shrink = max(0.0, 1.0 - psi * over)
        G_A = 1.0 + (G_A - 1.0) * shrink
        G_B = 1.0 + (G_B - 1.0) * shrink
    noise = 1.0 + chi * over
    if (not np.isfinite(noise)) or noise <= 0.0:
        noise = 1.0

    # ---------- Commitment 4: the panel gradient (quiet, not empty) ----------
    resid = eta + c_res * (1.0 - s)

    w = np.array(resid, dtype=float, copy=True)
    if iA.size > 0:
        w[iA] = G_A * mav[iA] + resid[iA]
    if iB.size > 0:
        w[iB] = G_B * mav[iB] + resid[iB]

    E = float(np.sum(w[iA]) - np.sum(w[iB]))
    if not np.isfinite(E):
        E = 0.0

    # ---------- Commitment 5: dilution by cue count, then lapse ----------
    den = float(D) ** gamma
    if (not np.isfinite(den)) or den <= 0.0:
        den = 1.0
    den = den * noise

    z = beta * E / den
    if not np.isfinite(z):
        z = 0.0
    z = float(min(max(z, -30.0), 30.0))

    p_a = 1.0 / (1.0 + np.exp(-z))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n\n    # cell accumulators: hits / total\n    hits = {}\n    tot = {}\n\n    for _, row in data.iterrows():\n        try:\n            a = [int(x) for x in row['option_a_ratings']]\n            b = [int(x) for x in row['option_b_ratings']]\n        except Exception:\n            continue\n        if len(a) != len(b) or len(a) == 0:\n            continue\n        winsA = [i for i in range(len(a)) if a[i] > b[i]]\n        winsB = [i for i in range(len(a)) if b[i] > a[i]]\n        mA, mB = len(winsA), len(winsB)\n        if mA == mB:\n            continue\n        if mA > mB:\n            win_idx, lose_idx, winner_is_A = set(winsA), set(winsB), True\n        else:\n            win_idx, lose_idx, winner_is_A = set(winsB), set(winsA), False\n        D = mA + mB\n\n        cell = None\n        if D == 1:\n            j = list(win_idx)[0]\n            if j == 2:\n                cell = 'C1'      # solo star (v=.95)\n            elif j == 3:\n                cell = 'C2'      # solo low-upper (v=.78)\n        elif D == 2 and len(lose_idx) == 0:\n            if win_idx == {0, 3}:\n                cell = 'C4'      # unopposed low pair\n            elif win_idx == {2, 4}:\n                cell = 'C5'      # unopposed senior pair\n        elif D == 4 and win_idx == {0, 3, 5} and lose_idx == {2}:\n            cell = 'C6'          # all-upper 3v1\n\n        if cell is None:\n            continue\n\n        try:\n            resp = int(row['response'])\n        except Exception:\n            continue\n        chose_A = (resp == 0)\n        follow = 1.0 if (chose_A == winner_is_A) else 0.0\n\n        hits[cell] = hits.get(cell, 0.0) + follow\n        tot[cell] = tot.get(cell, 0.0) + 1.0\n\n    def rate(c):\n        if tot.get(c, 0.0) <= 0:\n            return None\n        return hits[c] / tot[c]\n\n    g1 = [r for r in (rate('C1'), rate('C2')) if r is not None]\n    g2 = [r for r in (rate('C4'), rate('C5'), rate('C6')) if r is not None]\n\n    if len(g1) == 0 or len(g2) == 0:\n        return 0.0\n    return float(np.mean(g1) - np.mean(g2))",
  "rationale": "The two models make structurally opposite predictions about WHEN the count winner is followed. CG-VI-2 releases all discounting for a solitary discriminating cue (chorus argument CH=0, so every weight = 1, z = beta), forcing near-ceiling adherence (~.94) on the solo-star and solo-low-upper cells; but for CH>=2 displays in which no panel-floor voice discriminates, its news values clip to exactly zero, giving E=0 and EXACTLY chance (.50) on the unopposed low pair (C4), the unopposed senior pair (C5) and the all-upper 3-vs-1 (C6), regardless of parameters. MBM-PG7 has no solo release (a lone star is still mute, contributing only the thin residual, ~.54; a lone low-upper ~.68) but its residual gradient keeps all-upper multi-cue displays clearly above chance (~.56-.67). The contrast [mean follow-rate on solo cells C1,C2] minus [mean follow-rate on all-upper cells C4,C5,C6] is therefore ~+0.44 under CG-VI-2 and ~-0.01 under MBM-PG7 -- a ~0.45 gap. Each cell contributes 12 trials per subject, the cells involved sit in the tightest parameter regions of both models (CG's C4/C5/C6 are parameter-free exact .50), so per-subject variance is dominated by binomial noise (~0.010-0.016), giving a Welch t of roughly 14 at N=25. Both theories' predictions here follow directly from their stated commitments rather than being trivially true of either."
}
```

## Usage

```json
{
  "input_tokens": 15328,
  "output_tokens": 4912
}
```
