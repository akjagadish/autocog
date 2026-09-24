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
**Validities (n_features=6):** [0.74, 0.94, 0.54, 0.76, 0.92, 0.72]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 3: A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 1, 1]
  trial 4: A=[0, 1, 0, 1, 1, 1]  B=[0, 0, 1, 1, 0, 1]
  trial 5: A=[0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  trial 6: A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 7: A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0, 1]
  trial 8: A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 1, 1]
  trial 9: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  trial 10: A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 1]
  trial 11: A=[0, 1, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0, 1]
  trial 12: A=[0, 0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1]
  trial 13: A=[1, 1, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 14: A=[1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 1, 0]
  trial 15: A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 16: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0, 1]

**Rationale:** GOAL: dissociate CTN-BG (advocated) from MBM-PG7 (competing). The two models are near-twins: parameter-free count gate (tally ties -> exactly 0.5), panel-relative seniority s=(v-vmin)/(vmax-vmin), a near-categorical bottom-voice 'maverick/novelty' weight, chorus-gated amplification of that voice, dilution by D^gamma, a lapse, and a capacity term past 7 voices. With one lone floor expert and everything else far above the band, their novelty/mute functions AGREE (nov=1 vs mav=1 at s=0; both ~0 for every s>=0.45, for every theta<=0.20 and every kappa in [14,16.5]). So band-shape cells are wasted. They differ on exactly ONE structural commitment that this design attacks from four directions, plus a magnitude difference in the residual gradient.

THE CRITICAL DIFFERENCE. CTN-BG measures the chorus in MID-EXPERT EQUIVALENTS: Q_j = sum_opp u_i + lam_own*sum_own u_i with u=(1+tau*s)/(1+tau/2) (mid=1.00, star~1.33-1.43, floor~0.6), gated by a narrow logistic at q0~2.45 units. MBM-PG7 measures it in RAW COUNTS: base = (m_opp-1) + lam_own*(m_own-1), gated by a late steep knee (p~6.4, kappa~205) that needs base >~2.3. Hence: TWO STARS are a chorus for CTN (Q=2.63-2.86 > q0) but not for MBM (base=1 -> G=1.20-1.32); THREE MIDS are a chorus for both (Q=3.0; base=2 -> G=11.6-22.2); an UNOPPOSED 3-cue bloc containing the floor voice is a chorus for MBM (base=2) but not for CTN (own-side echo lam_own*(u_star+u_star)=1.32-2.00 < q0).

PANEL (screen order deliberately scrambled so validity rank is decoupled from position): [f0=.74, f1=.94, f2=.54, f3=.76, f4=.92, f5=.72]. vmin=.54, vmax=.94, span=.40 -> s = .50/1.00/.00/.55/.95/.45. Exactly ONE floor voice (f2=.54, s=0: nov=1 in CTN, inside MBM's band for every theta), TWO stars (s=1.00, .95: u=1.34-1.43 each), THREE mids straddling s=0.5 (.45/.50/.55, so any two of them sum to exactly 2.00 mid-equivalents and all three to exactly 3.00). n=6 keeps both models below their capacity terms, removing panel-size nuisance. 8 configurations x 2 exact A/B mirrors = 16 unique pairs x K=96//16=6 repetitions = 96 trials. Tied cues are coded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s never tracks condition. No cell is tally-tied (both models would trivially return 0.5).

CELLS (A-favouring version listed; the next trial is its exact mirror). Nominal P(choose A): CTN-BG (eta=.04, c_res=.12, nu=1.88, kappa=15, tau=1.2, lam_own=.60, gmax=1.9, q0=2.45, wid=.14, c_n=2.9, beta=41, gamma=.96, eps=.12) vs MBM-PG7 (theta=.16, rho=2.1, eta=.23, c=2.0, beta=.80, gamma=.95, alpha=52, kappa=205, p=6.4, lam_own=.36, eps=.09).
C1 (T1/T2) FLOOR vs MID-PAIR (1 v 2, D=3; Q=2.00 units): A wins f2; B wins f5,f3. CTN 0.58, MBM 0.56. Low rung of the chorus ladder; both near chance.
C2 (T3/T4) FLOOR vs STAR-PAIR (1 v 2, D=3; Q=2.71 units, SAME COUNT as C1): A wins f2; B wins f1,f4. CTN 0.93, box [0.86,0.96]; MBM 0.68, box [0.62,0.75]. FLAGSHIP: the parameter boxes are DISJOINT, and the within-subject contrast C2-C1 is +0.25..+0.40 for CTN but only +0.04..+0.21 for MBM (whose only lever here is the thin star residual).
C3 (T5/T6) FLOOR vs MID-TRIO (1 v 3, D=4; Q=3.00 units, only ~0.3 units more than C2 but one more COUNT): A wins f2; B wins f5,f0,f3. CTN 0.90, MBM 0.92. SECOND KEY CONTRAST: C3-C2 is ~0.00 (range -0.05..+0.03) for CTN, because 2 stars and 3 mids are the same chorus in mid-equivalents, but +0.08..+0.33 (nominal +0.24) for MBM, whose knee sits between base=1 and base=2. C2 and C3 together are the crossed dissociation: seniority-driven (CTN) versus count-driven (MBM) amplification.
C4 (T7/T8) FLOOR+STAR-PAIR vs LONE MID (3 v 1, D=4): A wins f2,f1,f4; B wins f0. CTN 0.90 (box [0.66,0.93]); MBM 0.62 (box [0.58,0.65]). CTN's own-side echo of two stars (0.6*2.71=1.63) plus the opposing mid crosses q0; MBM's own-side echo is 0.36*2=0.72 counts, far below its knee.
C5 (T9/T10) FLOOR+MID-PAIR vs LONE STAR (3 v 1, D=4): A wins f2,f5,f3; B wins f1. CTN 0.90 (equal to C4, since 1.375+0.6*2.00 ~ 1.00+0.6*2.71), MBM 0.75. CTN predicts C4=C5; MBM predicts C5-C4 ~ +0.13 (its thick residual gradient c~2.0 makes mid companions worth far more than star companions).
C6 (T11/T12) UNANIMOUS FLOOR+STAR (2 v 0, D=2): A wins f2,f1; four ties. CTN 0.76, MBM 0.79. Deliberate near-agreement anchor that identifies each model's lapse/slope.
C7 (T13/T14) UNANIMOUS FLOOR+STAR+STAR (3 v 0, D=3): A wins f2,f1,f4; three ties. CTN 0.72 (box [0.63,0.89]) vs MBM 0.95 (box [0.93,0.98]). THIRD KEY CONTRAST, and it runs in the OPPOSITE DIRECTION from C2/C4 (so no generic 'one model is more decisive' confound can explain the data): adding a second star to an unopposed bloc containing the floor voice leaves CTN flat or slightly lower (C7-C6 in [-0.06,+0.03], D-dilution offsets the sub-threshold echo) but pushes MBM from base=1 to base=2, i.e. across its knee (C7-C6 = +0.08..+0.26).
C8 (T15/T16) ALL-UPPER MID-TRIO vs LONE STAR (3 v 1, D=4, no floor voice discriminates): A wins f5,f0,f3; B wins f1. CTN 0.63, MBM 0.665. Deliberate agreement cell that is SELF-DIAGNOSTIC: TTB/one-reason would give <0.2 here and near-1.0 in C2/C7, so systematic star-following falsifies BOTH models; it also verifies that the residual gradient alone yields only moderate adherence on displays with no bottom voice.

PREDICTED PROFILE (C1..C8): CTN-BG = .58, .93, .90, .90, .90, .76, .72, .63 ; MBM-PG7 = .56, .92-ambiguous? no: .56, .68, .92, .62, .75, .79, .95, .665. They agree only on the deliberate anchors C1, C3, C6 and C8.

INTERPRETATION. A large C2-C1 jump (~+0.35) with C2 at or above 0.86 and C3 no higher than C2, C4 = C5 near 0.90, and a FLAT C6->C7 (unanimous bloc growth does nothing) uniquely support CTN-BG's mid-expert-equivalent chorus. A flat C2-C1 (~+0.1) with C2 capped near 0.70, a large C3-C2 jump (~+0.24), C5 clearly above C4 with both near 0.6-0.75, and a big C6->C7 jump to ~0.95 uniquely support MBM-PG7's count-based knee.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror, so side bias cancels exactly; validity rank is scrambled relative to screen position; tie coding alternates 1-1/0-0 within and across cells; every trial has a strict count winner and requires reading several ratings; n=6 removes all capacity-term nuisance.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** **Chorus-Threshold Novelty Weighting, Balanced-Gradient variant (CTN-BG).**

People told the accuracies of a panel of binary expert ratings do not weight a rating by its accuracy; they weight it by how much *news* it carries, and news is loud only when it stands against a genuine CHORUS. Six commitments, all smooth and continuous (no mute band, no solo release, no discrete count switch).

**(1) The count comparator is a parameter-free gate.** Before any expert identity is consulted, an unweighted feature-wise comparison asks: does one product win on strictly more attributes? If not — equal win counts, including wholly non-discriminating displays — the display is non-diagnostic and the subject flips a mental coin: exactly 0.5, with NO validity tie-break and NO take-the-best fallback, at every panel composition and size.

**(2) Seniority is panel-relative; novelty is a smooth convex, monotonically DECREASING function of it, and the descent is STEEP.** s_j = (v_j − v_min)/(v_max − v_min); nov_j = (1 − s_j)^kappa with kappa ≈ 14–16.5. Nothing is muted by fiat and nothing is released by fiat, but the curve is sharp: only the panel's genuinely idiosyncratic BOTTOM voice speaks with nov ≈ 1; the second-from-bottom voice retains a real but small share; a mid voice essentially nothing; a star nothing at all. Novelty is never negative — a star is ignored, never counted against its own side.

**(3) THE CORE CLAIM — a novel voice is loud only in proportion to the CHORUS it speaks into, and a chorus is measured in *mid-expert equivalents*.** Each other discriminating voice contributes a subjective unit u_i = (1 + tau·s_i)/(1 + tau/2): a mid-panel voice counts as exactly one unit, a star ≈ 1.4, a bottom voice ≈ 0.6. The chorus a cue j faces is Q_j = Σ_{opposing} u_i + lam_own·Σ_{own side, i≠j} u_i — voices it contradicts counted in full, voices it merely stands beside echoed at a discount. Loudness is a smooth logistic of that chorus, g_j = Gmax·σ((Q_j − q0)/wid), with a narrow but continuous threshold at roughly two-and-a-half mid-equivalents. BOTH how many voices and how senior they are matter, in one scalar: two mid voices are not yet a chorus, two stars just barely are, three upper voices fully are.

**(4) The ungated panel gradient is THIN and only MILDLY convex — a bare ordinal pull that does not, by itself, give the panel's bottom voice an edge over a pair of mid voices.** Every discriminating cue contributes r_j = eta + c·(1 − s_j)^nu with small eta, small c and a moderate exponent nu ≈ 1.7–2.05. The scale is small enough that all-upper displays drift near chance at any count margin, and the exponent is flat enough that a single floor cue's residual is worth about the same as two mid cues' residuals — so a floor voice opposed by a mid PAIR is a dead heat (it needs the chorus gate, not the gradient, to win), while a floor voice opposed by a three-cue upper bloc is carried past chance by the gate.

**(5) Integration is Luce-like and saturating.** w_j = r_j + nov_j·g_j; E = Σ dir_j w_j; z = beta·E/((c_n + Σ_j w_j)·D^gamma). Evidence is judged relative to the total weight of evidence on the table plus a constant of indecision, and is diluted by the number of mutually discriminating attributes with gamma ≈ 1, so a lone cue and an unopposed pair of the same tier feel about equally compelling. An attention lapse epsilon mixes the result with a coin flip.

**(6) Capacity is a near-total collapse of the chorus construal past ~7 voices, with only mild extra noise.** For n > 7 the amplification essentially vanishes (Gmax × (1 − psi·(n−7))) while integration noise grows only slightly, so eight-expert star-backed majorities sit near chance rather than being reversed, and eight-expert plain tally adherence survives on the residual gradient.

**Cross-design signature:** exact chance on tally ties at any composition; near-chance on displays whose every discriminating voice is upper-band, at every margin; moderate (not ceiling) adherence to majorities containing no bottom voice; a bottom voice facing two MID voices lands at chance, facing two STARS reverses substantially, facing a three-cue upper bloc reverses to ≈.12–.18; near-ceiling following of majorities that contain the bottom voice; attenuated integration on eight-expert panels.

**Parameters:**
- eta: [0.030, 0.048]
- c_res: [0.10, 0.14]
- nu: [1.70, 2.05]
- kappa: [14.0, 16.5]
- tau: [1.0, 1.5]
- lam_own: [0.50, 0.70]
- gmax: [1.70, 2.10]
- q0: [2.38, 2.52]
- wid: [0.11, 0.17]
- c_n: [2.70, 3.05]
- beta: [34, 48]
- gamma: [0.88, 1.05]
- epsilon: [0.09, 0.15]
- psi: [0.80, 0.95]
- chi: [0.02, 0.08]
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

    eta = _g('eta', 0.040, 0.0, 0.5)
    c_res = _g('c_res', 0.12, 0.0, 1.5)
    nu = _g('nu', 1.88, 0.5, 5.0)
    kappa = _g('kappa', 15.0, 1.0, 30.0)
    tau = _g('tau', 1.2, 0.0, 4.0)
    lam_own = _g('lam_own', 0.60, 0.0, 1.0)
    gmax = _g('gmax', 1.90, 0.0, 5.0)
    q0 = _g('q0', 2.45, 0.20, 8.0)
    wid = _g('wid', 0.14, 0.02, 1.0)
    c_n = _g('c_n', 2.9, 0.10, 12.0)
    beta = _g('beta', 41.0, 0.0, 150.0)
    gamma = _g('gamma', 0.96, 0.0, 2.0)
    eps = _g('epsilon', 0.12, 0.0, 0.6)
    psi = _g('psi', 0.88, 0.0, 1.0)
    chi = _g('chi', 0.05, 0.0, 3.0)

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

    one_minus = np.clip(1.0 - s, 0.0, 1.0)

    # ---------- Commitment 2: smooth convex novelty, steep descent ----------
    nov = np.power(one_minus, kappa)
    nov = np.where(np.isfinite(nov), nov, 0.0)

    # ---------- Commitment 4: THIN, mildly convex panel gradient ----------
    resid = eta + c_res * np.power(one_minus, nu)
    resid = np.where(np.isfinite(resid), resid, eta)

    # ---------- Commitment 3: chorus measured in MID-EXPERT EQUIVALENTS ----------
    denom_u = 1.0 + 0.5 * tau
    if (not np.isfinite(denom_u)) or denom_u <= 0.0:
        denom_u = 1.0
    u = (1.0 + tau * s) / denom_u        # mid voice == 1 unit, star ~1.4, floor ~0.6
    u = np.where(np.isfinite(u), u, 1.0)

    U_A = float(np.sum(u[iA])) if mA > 0 else 0.0
    U_B = float(np.sum(u[iB])) if mB > 0 else 0.0

    Q = np.zeros(n, dtype=float)
    if mA > 0:
        Q[iA] = U_B + lam_own * (U_A - u[iA])
    if mB > 0:
        Q[iB] = U_A + lam_own * (U_B - u[iB])
    Q = np.clip(Q, 0.0, None)

    # ---------- Commitment 6: capacity limit past ~7 voices ----------
    over = max(0.0, float(n) - 7.0)
    gmax_eff = gmax * max(0.0, 1.0 - psi * over)
    noise = 1.0 + chi * over
    if (not np.isfinite(noise)) or noise <= 0.0:
        noise = 1.0

    # smooth logistic threshold in the chorus size
    xg = (Q - q0) / max(wid, 1e-6)
    xg = np.clip(xg, -60.0, 60.0)
    gate = gmax_eff / (1.0 + np.exp(-xg))
    gate = np.where(np.isfinite(gate), gate, 0.0)

    # ---------- weights ----------
    w = resid + nov * gate
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w, 0.0, None)

    E = 0.0
    if mA > 0:
        E += float(np.sum(w[iA]))
    if mB > 0:
        E -= float(np.sum(w[iB]))
    if not np.isfinite(E):
        E = 0.0

    W_tot = 0.0
    if mA > 0:
        W_tot += float(np.sum(w[iA]))
    if mB > 0:
        W_tot += float(np.sum(w[iB]))
    if (not np.isfinite(W_tot)) or W_tot < 0.0:
        W_tot = 0.0

    # ---------- Commitment 5: Luce-like saturating integration ----------
    den = (c_n + W_tot) * (float(D) ** gamma) * noise
    if (not np.isfinite(den)) or den <= 0.0:
        den = 1.0

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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    FLOOR = 2          # validity 0.54, unique panel floor\n    # cells in which the floor voice sides with a set F against opposition O\n    POS = {\n        (frozenset({2}), frozenset({1, 4})),        # C2: floor vs STAR-PAIR\n        (frozenset({1, 2, 4}), frozenset({0})),     # C4: floor+2 stars vs lone mid\n        (frozenset({2, 3, 5}), frozenset({1})),     # C5: floor+2 mids vs lone star\n    }\n    NEG = {\n        (frozenset({1, 2, 4}), frozenset()),        # C7: unanimous floor+star+star\n    }\n\n    pos_hits = 0\n    pos_n = 0\n    neg_hits = 0\n    neg_n = 0\n\n    a_col = data['option_a_ratings'].tolist()\n    b_col = data['option_b_ratings'].tolist()\n    r_col = data['response'].tolist()\n\n    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):\n        try:\n            a = np.asarray(list(a_raw), dtype=float).ravel()\n            b = np.asarray(list(b_raw), dtype=float).ravel()\n        except Exception:\n            continue\n        if a.size != b.size or a.size == 0:\n            continue\n        winA = frozenset(int(i) for i in np.where(a > b)[0])\n        winB = frozenset(int(i) for i in np.where(b > a)[0])\n        if FLOOR in winA:\n            F, O = winA, winB\n            follow = 1 if int(resp) == 0 else 0\n        elif FLOOR in winB:\n            F, O = winB, winA\n            follow = 1 if int(resp) == 1 else 0\n        else:\n            continue\n        key = (F, O)\n        if key in POS:\n            pos_n += 1\n            pos_hits += follow\n        elif key in NEG:\n            neg_n += 1\n            neg_hits += follow\n\n    if pos_n == 0 or neg_n == 0:\n        return float('nan')\n\n    return float(pos_hits) / pos_n - float(neg_hits) / neg_n\n",
  "rationale": "Both models agree that the panel's unique floor expert (f2, v=.54) is the only 'news' carrier, and both are parameter-free coin flips on tally ties; they differ on ONE thing: how the amplifying chorus is measured. CTN-BG measures it in mid-expert EQUIVALENTS (two stars already constitute a chorus; an unopposed own-side bloc does not), whereas MBM-PG7 measures it in RAW COUNTS with a late knee between base=1 and base=2 (two stars are NOT a chorus, but any three-cue bloc, including an unopposed one, is).\n\nThe metric is a within-subject DOUBLE contrast on the rate of choosing the option the floor voice favours:\n  POS cells (opposed floor voice, chorus in seniority units but low raw count):\n    C2 floor vs star-pair, C4 floor+2 stars vs lone mid, C5 floor+2 mids vs lone star.\n  NEG cell (unopposed 3-cue bloc containing the floor voice, raw count 2 but only a discounted own-side echo in units):\n    C7 unanimous floor+star+star.\nScore = P(follow floor side | POS) - P(follow floor side | C7).\n\nPredicted values from the design's nominal parameterisations: CTN-BG ~ 0.91 - 0.72 = +0.19; MBM-PG7 ~ 0.68 - 0.95 = -0.27. A gap of roughly 0.45-0.5 with opposite SIGNS. Crucially the two components move in opposite directions across the theories, so no generic 'one model is more decisive/noisier' confound can mimic it, and differencing two within-subject rates cancels overall lapse rate and side bias (each cell appears in exact A/B mirrors). POS pools 36 trials and NEG 12 trials per subject, keeping the per-subject binomial noise small (sd ~0.15), so with N=25 subjects the Welch contrast should be far beyond alpha=0.01."
}
```

## Usage

```json
{
  "input_tokens": 16072,
  "output_tokens": 5140
}
```
