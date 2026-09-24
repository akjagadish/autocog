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
**Validities (n_features=7):** [0.75, 0.95, 0.71, 0.89, 0.55, 0.79, 0.92]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  trial 2: A=[1, 0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0]
  trial 3: A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1]
  trial 4: A=[0, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  trial 5: A=[0, 1, 1, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1, 0]
  trial 6: A=[0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1, 0]
  trial 7: A=[1, 0, 1, 1, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0, 1]
  trial 8: A=[0, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1, 1]
  trial 9: A=[1, 0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 0]
  trial 10: A=[0, 1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0, 1, 0]
  trial 11: A=[0, 1, 0, 1, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 0]
  trial 12: A=[0, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1]
  trial 13: A=[1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 1, 0]
  trial 14: A=[0, 1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 1, 0]
  trial 15: A=[0, 1, 0, 1, 0, 1, 1]  B=[0, 0, 1, 1, 0, 1, 0]
  trial 16: A=[0, 0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 1]

**Rationale:** GOAL: dissociate CG-VI-2 (advocated) from MBM-PG7 (competing). The two models share almost everything: a parameter-free count gate (tally ties -> exactly 0.5), panel-relative seniority s=(v-vmin)/(vmax-vmin), inverse-validity 'news value', dilution by D^gamma, a lapse, and a capacity term past 7 voices. They also AGREE on the classic maverick reversals (a lone floor voice opposed by a 3-cue bloc reverses it in both models), so those cells are wasted. They differ on exactly three things, and every cell below turns one of them into a parameter-free equality, a large magnitude gap, or a sign flip.

DIFFERENCE 1 - WHAT HAPPENS WHEN NO CHORUS EXISTS (D=1). CG's chorus argument is CH = max(mA,mB)+|mA-mB|-2, so a SOLO discriminating cue has CH=0 -> lam_eff=0 -> EVERY weight is exactly 1 -> E=1, den=1, z=beta in [9.5,12.5] -> P = 1-eps/2 in [0.91,0.96], IDENTICALLY for a solo .95 expert, a solo .75 expert and a solo .55 expert (a three-way parameter-free equality at near ceiling). MBM has no such release: a solo star is still outside the mute band, so its whole contribution is the thin residual r=eta (0.18-0.28), giving z=beta*eta in [0.13,0.25] -> P in [0.53,0.56]; a solo mid contributes eta+c/2 -> P ~ 0.65-0.77; a solo floor voice contributes 1+eta+c -> P ~ 0.86-0.93. So MBM must produce a steep monotone inverse-validity LADDER across the three solo cells (~0.54 / 0.71 / 0.89) whereas CG must produce a FLAT near-ceiling line (~0.94 / 0.94 / 0.94). C1 (solo star) alone is decisive: the two models' whole parameter boxes are disjoint (CG >= 0.91 vs MBM <= 0.56).

DIFFERENCE 2 - MUTE MEANS ZERO (CG) VS MUTE MEANS A GRADED RESIDUAL (MBM). CG clips w = max(0, 1-lam_eff*s^omega): once the chorus is >= 2 every voice appreciably above the panel floor is EXACTLY mute, so a display in which no floor voice discriminates has E = 0 and returns EXACTLY 0.500 for every lam, omega, q, kappa, beta, gamma, epsilon - even when one product is weakly DOMINANT or wins the tally 3-to-1. MBM's muted experts keep r = eta + c(1-s), an inverse-seniority gradient, so MBM's prediction on such displays tracks the seniority composition. Cells C4 (unopposed .79+.75 pair, mA=2/mB=0, CH=2), C5 (low trio .79/.75/.71 beats lone .95, CH=3) and C6 (senior trio .95/.92/.89 beats lone .71, CH=3) are all built so that every discriminating voice clips at EVERY corner of CG's box (at CH=2 clipping needs s>=0.48 at the worst corner - the cues used are s=0.50 and 0.60; at CH=3 it needs s>=0.35 - the lowest cue used is s=0.40). CG therefore predicts exactly 0.500 in C4, C5 and C6, including a +2 tally margin in C5/C6 and a zero-conflict dominance display in C4. MBM predicts ~0.70 (C4), ~0.66 (C5) and ~0.49 (C6). C5 vs C6 is the cleanest form of this test: IDENTICAL composition (3 vs 1, D=4, CH=3), only the seniority band of the majority is swapped; CG must give an exact null difference at exactly chance, MBM must give a ~0.17 swing that straddles chance.

DIFFERENCE 3 - THE 2-vs-1 REGIME AND OVERALL DECISIVENESS. CG's chorus gate is steep in bloc structure: a 2-vs-1 display has CH=1, so lam_eff = lam/(1+kappa) in [0.53,0.79] - only a WEAK discount, all weights stay positive, and with beta in [9.5,12.5] the weighted tally is followed decisively. MBM has no bloc-size gate for out-of-band voices and a small beta (0.7-0.9), so its 2-vs-1 predictions are always mid-range. C7 (A wins .75 and .71; B wins .95): CG in [0.82,0.93] vs MBM in [0.59,0.72] - nearly disjoint magnitude prediction at the same sign. C8 (A wins .95 and .92; B wins .71): CG in [~0.50,0.78] (its weighted tally still favours the pair) vs MBM in [0.42,0.47] (the single low voice's residual outweighs two star residuals) - a sign contrast around chance. C8 also pairs with C6: CG predicts a NON-MONOTONICITY in majority size (senior pair over lone low = ~0.66, but ADDING a third senior supporter collapses it to exactly 0.500, because bloc size drives the redundancy construal), whereas MBM predicts both cells flat and slightly below chance (~0.45 / ~0.49).

PANEL: 7 experts, validities in deliberately scrambled screen order [f0=.75, f1=.95, f2=.71, f3=.89, f4=.55, f5=.79, f6=.92]; vmin=.55, vmax=.95, span=.40, so s = .50/1.00/.40/.85/.00/.60/.925. Exactly ONE voice sits at the panel floor (f4=.55, s=0: maverick for BOTH models, weight 1 in CG and inside MBM's band for every theta), and every other voice sits at s>=0.40 - far outside MBM's widest possible band (theta<=0.20) and inside CG's clipping region whenever CH>=2. This removes the band-width nuisance entirely and makes the dissociation rest only on the three structural commitments above. n=7 keeps both models at/below their capacity thresholds (over = 0), so no panel-size nuisance.

DESIGN: 8 configurations x 2 exact A/B mirrors = 16 unique pairs x K = 96//16 = 6 repetitions = 96 trials. No cell is tally-tied (both models trivially return 0.5 there) and no cell is fully degenerate.

CELLS (A-favouring version listed first; the next trial is its exact mirror). Nominal predictions P(choose A): CG-VI-2 (lam=1.75, omega=.34, q=3.95, kappa=1.70, beta=11, gamma=.90, eps=.12) vs MBM-PG7 (theta=.16, rho=2.1, eta=.23, c=2.0, beta=.80, gamma=.95, alpha=52, kappa=205, p=6.4, lam_own=.36, eps=.09).
C1 (T1/T2) SOLO STAR: only f1(.95) discriminates. CG 0.94 [0.91,0.96] vs MBM 0.54 [0.53,0.56]. FLAGSHIP - disjoint boxes.
C2 (T3/T4) SOLO MID: only f0(.75) discriminates. CG 0.94 vs MBM 0.71.
C3 (T5/T6) SOLO FLOOR: only f4(.55) discriminates. CG 0.94 vs MBM 0.89. Deliberate near-agreement: the lapse/ceiling anchor that identifies each model's epsilon, and the third rung of the solo ladder.
C4 (T7/T8) UNOPPOSED LOW-UPPER PAIR (2,0,CH=2): A wins f5(.79), f0(.75). CG EXACTLY 0.500 (both clip at every box corner) vs MBM 0.70. Chance responding on a weakly dominant display is CG's signature.
C5 (T9/T10) ALL-UPPER 3v1, LOW TRIO vs LONE STAR: A wins f5,f0,f2; B wins f1(.95). CG EXACTLY 0.500 despite a +2 margin; MBM 0.66.
C6 (T11/T12) ALL-UPPER 3v1, SENIOR TRIO vs LONE LOW: A wins f1,f6,f3; B wins f2(.71). CG EXACTLY 0.500 (identical to C5, parameter-free equality); MBM 0.49.
C7 (T13/T14) 2v1 LOW PAIR vs LONE STAR (CH=1): A wins f0(.75), f2(.71); B wins f1(.95). CG 0.88 [0.82,0.93] vs MBM 0.65 [0.59,0.72].
C8 (T15/T16) 2v1 SENIOR PAIR vs LONE LOW (CH=1): A wins f1(.95), f6(.92); B wins f2(.71). CG 0.66 (>=~0.5 over essentially its whole box) vs MBM 0.45 [0.42,0.47].

PREDICTED PROFILE (C1..C8): CG-VI-2 = .94, .94, .94, .500, .500, .500, .88, .66 ; MBM-PG7 = .54, .71, .89, .70, .66, .49, .65, .45. They agree only on the deliberate anchor C3.

INTERPRETATION: flat near-ceiling adherence across all three solo cells, exact chance on C4/C5/C6 (a dominance display and two +2-margin displays), decisive following of the low pair in C7 and above-chance following of the senior pair in C8 uniquely support CG-VI-2. A steep solo ladder (.54/.71/.89), above-chance following of the tally in C4/C5 with a systematic C5-minus-C6 gap, mid-range (never extreme) adherence in C7, and sub-chance C8 uniquely support MBM-PG7. If instead subjects simply follow validity (WADD/TTB), C6 and C8 will be HIGH and C5/C7 LOW, which falsifies both models - so the design is self-diagnostic.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror, so side bias cancels exactly; validity rank is decoupled from screen position so positional heuristics cannot mimic validity-based weighting; tied cues are coded sometimes 1-1 and sometimes 0-0 within each cell so the number of displayed 1s never tracks composition or seniority; every trial has a strict count winner and requires reading several ratings.

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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Canonical (order-free) keys for the four diagnostic cells.\n    # SOLO cells: exactly one discriminating expert, and it is NOT the panel floor\n    #   C1 = solo .95 star, C2 = solo .75 mid\n    SOLO = {\n        (\"1001010\", \"1101010\"),   # C1 (trials 1/2)\n        (\"0010101\", \"1010101\"),   # C2 (trials 3/4)\n    }\n    # FLAT cells: CH >= 2 displays whose every discriminating voice is upper-band\n    #   C4 = unopposed .79/.75 pair (2 vs 0), C5 = low trio vs lone .95 (3 vs 1)\n    FLAT = {\n        (\"0011001\", \"1011011\"),   # C4 (trials 7/8)\n        (\"0101000\", \"1011010\"),   # C5 (trials 9/10)\n    }\n\n    solo_hits = 0.0\n    solo_n = 0.0\n    flat_hits = 0.0\n    flat_n = 0.0\n\n    for _, row in data.iterrows():\n        try:\n            a = [int(x) for x in list(row['option_a_ratings'])]\n            b = [int(x) for x in list(row['option_b_ratings'])]\n        except Exception:\n            continue\n        if len(a) != len(b) or len(a) == 0:\n            continue\n        ka = ''.join([str(x) for x in a])\n        kb = ''.join([str(x) for x in b])\n        key = tuple(sorted((ka, kb)))\n\n        if key in SOLO:\n            grp = 0\n        elif key in FLAT:\n            grp = 1\n        else:\n            continue\n\n        mA = sum(1 for i in range(len(a)) if a[i] > b[i])\n        mB = sum(1 for i in range(len(a)) if b[i] > a[i])\n        if mA == mB:\n            continue\n        winner = 0 if mA > mB else 1\n\n        try:\n            resp = int(row['response'])\n        except Exception:\n            continue\n        hit = 1.0 if resp == winner else 0.0\n\n        if grp == 0:\n            solo_hits += hit\n            solo_n += 1.0\n        else:\n            flat_hits += hit\n            flat_n += 1.0\n\n    if solo_n == 0.0 or flat_n == 0.0:\n        return float('nan')\n\n    return float(solo_hits / solo_n - flat_hits / flat_n)",
  "rationale": "The two models disagree most sharply on two structurally opposite kinds of display, and in OPPOSITE directions, so a difference score amplifies the contrast while cancelling generic per-subject response noise.\n\n(a) SOLO cells (C1 solo .95 star, C2 solo .75 mid; 24 trials/subject). CG-VI-2's chorus argument is CH = max(m)+|\\u0394m|-2, so a lone discriminating voice gives CH=0, lam_eff=0, every weight exactly 1, z = beta (9.5-12.5): near-ceiling adherence ~0.94 regardless of the lone expert's validity. MBM-PG7 mutes any voice outside its narrow band unconditionally, so a lone star contributes only the thin residual eta -> ~0.54, and a lone mid ~0.71; mean ~0.63.\n\n(b) FLAT cells (C4 unopposed .79/.75 pair, C5 low trio vs lone .95; 24 trials/subject). Here CH>=2 and every discriminating voice clips to exactly zero news value at every corner of CG's parameter box, so CG returns EXACTLY 0.500 even on a weakly dominant display and on a +2 tally margin. MBM keeps an inverse-seniority residual for muted voices and therefore follows the tally at ~0.70 / ~0.66.\n\nHence the composite (SOLO adherence - FLAT adherence) is predicted to be ~ +0.44 under CG-VI-2 but ~ -0.05 under MBM-PG7 (a near-zero/slightly negative value, since MBM actually follows the tally somewhat MORE in the dense cells than in the solo-star cell). Both terms are tight per subject: CG's solo term sits at ceiling (binomial var ~0.002 over 24 trials, parameter box only [0.91,0.96]) and its flat term is a parameter-free 0.5; MBM's terms are mid-range but its parameter box is narrow in both cells. The predicted between-theory gap (~0.5) is an order of magnitude larger than the expected per-subject SD (~0.11-0.14), giving a large Welch t at N=25. It is not trivially true of CG: any model that simply weights by validity, or that follows the plain tally, or that has graded (non-clipped) weights would produce a small or negative value here, since tally-following models score HIGH on the 2-vs-0 and 3-vs-1 cells."
}
```

## Usage

```json
{
  "input_tokens": 15868,
  "output_tokens": 5355
}
```
