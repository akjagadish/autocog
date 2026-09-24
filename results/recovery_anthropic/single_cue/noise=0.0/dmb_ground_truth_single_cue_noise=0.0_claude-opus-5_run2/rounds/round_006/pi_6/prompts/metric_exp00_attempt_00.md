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
**Validities (n_features=6):** [0.92, 0.6, 0.84, 0.53, 0.76, 0.68]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0, 1]
  trial 2: A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  trial 3: A=[1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0]
  trial 5: A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 6: A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  trial 7: A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  trial 8: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  trial 9: A=[0, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 10: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  trial 11: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1]
  trial 12: A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1, 1]
  trial 13: A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  trial 14: A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 15: A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 16: A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]

**Rationale:** GOAL: dissociate GVIT-CAP-S4 (advocated) from CG-VI-2 (competing). The two models share a hard count gate (tally ties -> exactly 0.5), an inverse-validity 'news value' weighting with a strongly saturating exponent s^omega, dilution by D^gamma and a capacity term above 7 experts. They differ on exactly TWO structural commitments, and every cell below turns one of them into a sign reversal or a parameter-free equality. (1) THE ZERO FLOOR: CG clips w = max(0, 1 - lam_eff*s^omega), so a redundant expert is MUTE; GVIT lets w go NEGATIVE, so a high-validity expert votes against his own product. (2) THE CHORUS CRITERION: CG reads the chorus off bloc structure, CH = max(mA,mB)+|mA-mB|-2 with a steep gate CH^q/(CH^q+kappa) (q~4), so a 2-vs-1 display (CH=1) is only WEAKLY discounted (lam_eff ~ 0.53-0.79); GVIT uses the raw number of discriminating cues, lam_eff = lam*(D-1)/((D-1)+kappa_d), so the same 2-vs-1 display (D=3) is NEAR-MAXIMALLY discounted (lam_eff ~ 0.90-1.79). Hence 2-vs-1 displays are the sign-reversal engine and unopposed/all-upper displays are the zero-floor engine.

PANEL: 6 experts, validities in scrambled screen order [f0=.92, f1=.60, f2=.84, f3=.53, f4=.76, f5=.68]; descending rank f0 > f2 > f4 > f5 > f1 > f3. v_min=.53, v_max=.92, span=.39, so panel-relative seniority s = 1.00/.795/.590/.385/.179/0 for f0/f2/f4/f5/f1/f3, and s^.34 = 1.00/.925/.836/.723/.558/0. One lone floor voice (f3=.53, full weight 1 in BOTH models), a dense upper cluster (.92/.84) whose weights clip to zero for CG whenever CH>=2 but go clearly negative for GVIT, a mid band (.76/.68) sitting exactly where GVIT's concave discount nullifies it, and a low-but-not-floor voice (.60). n=6 keeps both models below their capacity terms, removing panel-size nuisance. 8 configurations x 2 exact A/B mirrors = 16 unique pairs x K = 96//16 = 6 repetitions = 96 trials. Tied cues are coded sometimes 1-1 and sometimes 0-0 so the displayed number of 1s never tracks composition; no cell is tally-tied (both models would trivially return 0.5).

CELLS (A-favouring version listed; the next trial is its exact mirror). Nominal P(choose A): GVIT (lam=1.45, omega=.34, beta=6.7, gamma=.90, kappa_d=.10, eps=.17) vs CG-VI-2 (lam=1.75, omega=.34, q=3.95, kappa=1.70, beta=11, gamma=.90, eps=.12).
C1 (T1/T2) UNOPPOSED UPPER PAIR (mA=2, mB=0, D=2, CH=2): A wins f0(.92), f2(.84); four ties. CG: both weights clip to 0 -> E = 0 -> EXACTLY 0.500 at every corner of its box (worst corner lam=1.6,q=3.4,kappa=2.0 gives lam_eff=1.345 and even omega=.40 leaves s_f2^om=.912 > 1/1.345). GVIT: w = -0.318, -0.219 -> P(A) = 0.19, i.e. 80% choice of the weakly DOMINATED product. FLAGSHIP: chance vs strong reversal on a zero-conflict display.
C2 (T3/T4) UNOPPOSED BOTTOM PAIR (2,0, CH=2): A wins f1(.60), f3(.53). GVIT 0.91, CG 0.94. Matched composition to C1 but opposite identity: the C1-minus-C2 within-subject drop is 0.44 for CG (ceiling -> exact chance) and 0.72 for GVIT (ceiling -> far below chance). Also the lapse/slope anchor.
C3 (T5/T6) UPPER-PAIR vs LONE MID (2v1, D=3, CH=1): A wins f0, f2; B wins f5(.68). E = 1 - lam_eff*1.2024 in BOTH models, so the cell isolates lam_eff at 2-vs-1: CG's weak chorus gate keeps E > 0 across its entire box (+0.05 to +0.36) -> P(A) = 0.69 (>0.5 always); GVIT's D-based gate makes E <= 0 across its box -> P(A) = 0.22. SIGN REVERSAL.
C4 (T7/T8) UPPER-PAIR vs LONE LOW (2v1): same A side, dissenter is f1(.60). GVIT 0.17, CG 0.60. Replication of the sign reversal with a different lone opponent; the C3/C4 pair also traces how far down the panel the 'lone dissenter beats the star pair' effect extends.
C5 (T9/T10) MID-PAIR vs LONE STAR (2v1): A wins f4(.76), f5(.68); B wins f0(.92). E = 1 - 0.5582*lam_eff. GVIT: the mids are discounted almost as hard as the star, so the majority is followed only WEAKLY (0.62; 0.50-0.70 over its box) - its commitment-3(i) signature. CG: lam_eff at CH=1 is small, all weights stay large and positive, so the majority is followed strongly (0.88).
C6 (T11/T12) BOTTOM-PAIR vs LONE STAR (2v1): A wins f1(.60), f3(.53); B wins f0(.92). GVIT 0.90, CG 0.94. KEY GRADED CONTRAST C6 - C5 (identical composition, only the seniority band of the majority changes): GVIT predicts a LARGE gap (~0.28, because mid support is nearly mute while bottom support keeps full weight), CG predicts a SMALL gap (~0.06, because at CH=1 nothing is heavily discounted). 32 trials per subject on this contrast.
C7 (T13/T14) ALL-UPPER 3v1 (mA=3, mB=1, D=4, CH=3): A wins f0, f2, f4; B wins f5(.68). CG: the chorus gate saturates, every discriminating weight clips to 0 -> E = 0 -> EXACTLY 0.500 despite a +2 count margin (robust at all box corners). GVIT: E = -0.86 -> 0.22. Second parameter-free cell for CG and the cleanest demonstration that 'redundant' means COUNTER-INDICATIVE (GVIT) rather than MUTE (CG).
C8 (T15/T16) MAVERICK-IN-MAJORITY 3v2 (D=5, CH=2): A wins f0(.92), f2(.84) AND the floor voice f3(.53); B wins f4(.76), f5(.68). CG: all four upper voices are mute, the maverick speaks for the majority -> near-ceiling 0.88. GVIT: A's two stars subtract while B's two mids add back, leaving only a modest positive -> 0.65. Gap ~0.23; combined with C7 it separates 'stars are silent' from 'stars vote against their own side'.

PREDICTED PROFILE (C1..C8): GVIT = .19, .91, .22, .17, .62, .90, .22, .65 ; CG-VI-2 = .500, .94, .69, .60, .88, .94, .500, .88. The models agree only on the anchors C2 and C6, and disagree in SIGN relative to chance on C1, C3, C4 and C7.

INTERPRETATION: systematic BELOW-chance responding on C1 and C7 (choosing the product that no expert prefers / that loses the 3-vs-1 count), below-chance on the 2-vs-1 star-pair cells C3/C4, only weakly-above-chance following of the mid-backed majority in C5 with a large C6-minus-C5 gap, uniquely supports GVIT-CAP-S4. Exact chance on C1 and C7, above-chance following of the star pair in C3/C4, strong adherence in C5 with a flat C6-C5 contrast, and near-ceiling in C8 uniquely supports CG-VI-2.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror so side bias cancels exactly; validity rank is decoupled from screen position so positional heuristics cannot mimic validity-based weighting; tie coding alternates 1-1 / 0-0; no cell is tally-tied or otherwise degenerate; every trial requires reading several ratings.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Target cells: displays where the TALLY-MAJORITY side is carried by the two\n    # most valid experts (features 0 = .92 and 2 = .84).\n    # C1: majority wins {0,2}, minority wins {}        (unopposed star pair, 2v0)\n    # C3: majority wins {0,2}, minority wins {5}       (star pair vs lone mid, 2v1)\n    # C4: majority wins {0,2}, minority wins {1}       (star pair vs lone low, 2v1)\n    # C7: majority wins {0,2,4}, minority wins {5}     (all-upper 3v1)\n    targets = {\n        (frozenset([0, 2]), frozenset()),\n        (frozenset([0, 2]), frozenset([5])),\n        (frozenset([0, 2]), frozenset([1])),\n        (frozenset([0, 2, 4]), frozenset([5])),\n    }\n\n    hits = []\n    for _, row in data.iterrows():\n        try:\n            a = [int(x) for x in row['option_a_ratings']]\n            b = [int(x) for x in row['option_b_ratings']]\n        except Exception:\n            continue\n        if len(a) != len(b) or len(a) == 0:\n            continue\n        idxA = frozenset([i for i in range(len(a)) if a[i] > b[i]])\n        idxB = frozenset([i for i in range(len(a)) if b[i] > a[i]])\n        if len(idxA) == len(idxB):\n            continue\n        if len(idxA) > len(idxB):\n            maj_set, min_set, maj_resp = idxA, idxB, 0  # A is count winner\n        else:\n            maj_set, min_set, maj_resp = idxB, idxA, 1  # B is count winner\n        if (maj_set, min_set) not in targets:\n            continue\n        try:\n            r = int(row['response'])\n        except Exception:\n            continue\n        hits.append(1.0 if r == maj_resp else 0.0)\n\n    if len(hits) == 0:\n        return 0.5\n    return float(np.mean(hits))\n",
  "rationale": "The two theories differ most sharply on displays whose tally-majority is carried by the panel's two star experts (f0=.92, f2=.84). CG-VI-2 floors news value at zero, so those stars are merely MUTE: for the unopposed star pair (C1, 2v0) and the all-upper 3v1 (C7) every discriminating weight clips to zero and P(choose the count winner) is exactly 0.500 by construction, while for the 2v1 star-pair cells (C3, C4) its weak chorus gate leaves positive evidence, so the majority is still followed above chance (~.60-.69). GVIT-CAP-S4 lets the weights go NEGATIVE, so on exactly these cells the star-backed majority is actively rejected (~.17-.22). Pooling only these four cells (48 trials per subject, mirrored A/B so side bias cancels) yields a predicted adherence of ~0.20 under GVIT versus ~0.57 under CG-VI-2 \u2014 a large mean gap, while the per-subject binomial noise over 48 trials is small and two of the four cells are parameter-free under CG, keeping between-subject variance tight. The metric is a simple proportion, well defined on both pooled and single-subject slices, and is not trivially true of my theory: it is a plain measure of following the unweighted cue count, which most standard accounts predict well above chance."
}
```

## Usage

```json
{
  "input_tokens": 14766,
  "output_tokens": 2742
}
```
