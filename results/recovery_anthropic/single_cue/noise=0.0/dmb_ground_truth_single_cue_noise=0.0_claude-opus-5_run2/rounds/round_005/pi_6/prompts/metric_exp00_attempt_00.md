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
**Validities (n_features=6):** [0.72, 0.95, 0.58, 0.88, 0.52, 0.68]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  trial 2: A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1]
  trial 3: A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  trial 4: A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0, 1]
  trial 5: A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  trial 6: A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 0, 1]
  trial 7: A=[1, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  trial 8: A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1]
  trial 9: A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  trial 10: A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  trial 11: A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 12: A=[1, 0, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1, 0]

**Rationale:** GOAL: dissociate GVIT-CAP-S4 (advocated) from SR-DT-6b (competing). The two models share a hard count gate (tally ties -> exactly 0.5), both invert stated validity, both release the discount for solo voices, and both compress evidence by the number of discriminating cues, so tally-tied cells and generic conflict cells are useless (they agree exactly). They differ on exactly three things: (i) the SHAPE of the redundancy discount (GVIT: strongly concave s^omega, omega~.26-.42, so low-MID experts are discounted almost as hard as stars; SR-DT: superlinear s^omega, omega>1, so low-mid experts keep large positive news value and only the top one or two voices go negative, bounded by w_floor); (ii) the CHORUS criterion (GVIT's lam_eff uses the TOTAL number of discriminating cues D, so a lone dissenting star inside a D=3 display is still counter-indicative, w = 1-lam_eff < 0; SR-DT's redundancy is SAME-SIDE, so any side with exactly one cue is BOOSTED above face value, w = phi0+(phi-phi0)*s >= 1.15 for a lone star); (iii) DISSENT dilution (SR-DT divides by (1+m_lose)^gamma_c so unopposed majorities approach ceiling; GVIT has no dissent term and simply divides by D^gamma).

PANEL (screen order deliberately scrambled so validity rank is decoupled from position): [f0=.72, f1=.95, f2=.58, f3=.88, f4=.52, f5=.68]; descending rank f1 > f3 > f0 > f5 > f2 > f4. Three structural choices are essential. (a) The floor is at .52, essentially chance, so BOTH models' seniority anchors (GVIT: v_min; SR-DT: 0.5) nearly coincide -- the dissociation is then driven purely by the exponent and the chorus rule, not by an anchoring artefact. (b) There is a LOW-MID band (.72, .68, .58) sitting at panel-relative s = .47/.37/.14, exactly where a concave and a convex exponent diverge maximally: GVIT weights them -0.09 / -0.01 / +0.28 (at lam_eff=1.38) while SR-DT weights them +0.54 / +0.67 / +0.92. (c) n = 6 keeps GVIT's overload term o ~ 0 and SR-DT's size_fac = (6/6)^tau = 1, removing all panel-size nuisance. Only the floor voice f4 (.52) gets full weight 1.0 in GVIT (s = 0 exactly, parameter-free) -- so GVIT is, in this panel, 'follow the least accurate expert, mildly opposed by everyone else', whereas SR-DT is 'everyone below .80 speaks with near-full voice, and a solitary voice speaks louder still'.

DESIGN: 6 configurations x 2 exact A/B mirrors = 12 unique pairs x 8 repetitions = 96 trials. Every cell has a strict count winner (no tally ties, where both models trivially return 0.5). Tied cues are coded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s does not track composition.

CELLS (A-favouring version listed; the next trial is its exact mirror). Nominal predictions P(choose A): GVIT (lam=1.45, beta=6.7, gamma=.90, kappa_d=.10, omega=.34, eps=.17) vs SR-DT (lam=1.6, omega=1.7, beta=13.5, gamma_d=.68, gamma_c=.66, kappa=.05, phi=1.40, phi0=1.08, w_floor=-.30, eps=.09).
K1 (T1/T2) MID-PAIR vs LONE STAR (mA=2, mB=1, D=3): A wins f0(.72), f5(.68); B wins f1(.95) alone. GVIT E=+0.33 -> 0.67; SR-DT E=-0.17 -> 0.35. SIGN REVERSAL.
K2 (T3/T4) MID-PAIR vs LONE FLOOR (2 vs 1, D=3): identical A side, the single dissenter is now f4(.52). GVIT E=-1.05 -> 0.16; SR-DT E=+0.13 -> 0.62. SIGN REVERSAL, in the OPPOSITE direction from K1.
K3 (T5/T6) MID-TRIO vs LONE STAR (3 vs 1, D=4): A wins f0(.72), f5(.68), f2(.58); B wins f1(.95). GVIT 0.72; SR-DT 0.88.
K4 (T7/T8) MID-TRIO vs LONE FLOOR (3 vs 1, D=4): same A side, dissenter f4(.52). GVIT 0.22; SR-DT 0.92. LARGEST SINGLE-CELL GAP (~0.70) and the most parameter-robust: GVIT's E = 2 - lam_eff*(s72^om+s68^om+s58^om) is negative over almost its entire box (P(A) <= ~0.5), while SR-DT stays >= ~0.72 over its entire box.
K5 (T9/T10) UNOPPOSED MID-PAIR (mA=2, mB=0, D=2): A wins f0(.72), f5(.68); nothing opposes. GVIT 0.53 (near chance on a dominance display!); SR-DT 0.955.
K6 (T11/T12) UNOPPOSED BOTTOM-PAIR (mA=2, mB=0, D=2): A wins f2(.58), f4(.52); nothing opposes. GVIT 0.93; SR-DT 0.955.

FOUR ORTHOGONAL DIAGNOSTICS.
(1) LONE-OPPONENT IDENTITY CONTRAST, majority held exactly fixed (K1-K2 and K3-K4). GVIT's chorus criterion is TOTAL-D based, so the lone star carries negative weight while the lone floor voice carries +1: it predicts star-minus-floor = +0.51 (2-cue majority) and +0.50 (3-cue majority) -- i.e. people follow two middling experts against one .95 expert but abandon them against one .52 expert. SR-DT's same-side redundancy boosts BOTH lone voices above face value, more so the senior one, so it must predict a NON-POSITIVE contrast (-0.27 and -0.04). The SIGN of this within-subject contrast, replicated at two majority sizes with 32 trials each, is the single cleanest diagnostic.
(2) MAJORITY-SIZE GROWTH AGAINST THE FLOOR VOICE (K2 -> K4). SR-DT: 0.62 -> 0.92 (adding a third low-mid supporter converts a marginal display into a confident one). GVIT: 0.16 -> 0.22, i.e. still firmly sub-chance, because each added mid contributes ~0 while the floor voice keeps weight 1. A 3-versus-1 count majority being REJECTED is impossible for SR-DT anywhere in its box.
(3) UNOPPOSED-DOMINANCE GAP (K5 vs K6). Two structurally identical zero-conflict displays (2 wins, 0 losses, 4 ties). SR-DT has no dissent term and all sub-.80 experts retain large news value, so it must put BOTH at ceiling (0.955 / 0.955, gap ~0.00, and its eps <= .13 forces >= .935). GVIT's saturating inversion nullifies the mid pair but leaves the floor voice at full weight, predicting a ~0.40 gap (0.53 vs 0.93) -- near-chance responding on a display where one product is weakly dominant. K5 also serves as the unopposed control for K1: GVIT predicts P(A) goes UP (0.53 -> 0.67) when an opposing .95 expert is ADDED, a non-monotonicity SR-DT cannot produce (0.955 -> 0.35).
(4) CEILING/LAPSE ANCHOR (K6). Both models put this cell high (0.93 vs 0.955), which separately identifies each model's lapse rate so that slope and ceiling cannot be traded off when fitting the diagnostic cells.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror, so side bias cancels exactly; validity rank is scrambled relative to screen position; tie coding alternates 1-1 / 0-0; no cell is degenerate and no cell is tally-tied (where both models trivially predict 0.5).

INTERPRETATION: a large POSITIVE lone-star-minus-lone-floor contrast at fixed majority (following two/three middling experts over one .95 expert while abandoning them against one .52 expert), sub-chance adherence in K2/K4, and near-chance responding on the unopposed mid-pair K5 with near-ceiling on the unopposed bottom-pair K6 uniquely support GVIT-CAP-S4. A NEGATIVE or null opponent-identity contrast, above-chance adherence to the count winner in all of K1-K4 with K4 near 0.9, and equal ceiling performance on K5 and K6 uniquely support SR-DT-6b.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



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
**Description:** **Superlinear Redundancy-Discounted Tallying with a Bounded Anti-Count Floor, Baselined Seniority-Scaled Dissent Salience, Dissent-Graded Evidence Dilution and Two-Regime Panel-Size Precision Loss (SR-DT-6b).**

People told the validities of a panel of binary expert ratings do not weight experts *by* validity; they weight them by how much *news* a rating carries, which is a decreasing function of how senior the expert is inside the displayed panel. Eight commitments.

**(1) The count comparator is a gate, not a weight.** Before any expert is evaluated, an unweighted feature-wise comparison asks: does one product win on strictly more attributes? If not (equal win-counts, including wholly non-discriminating displays), the display is construed as non-diagnostic: choice is exactly 0.5, with NO validity tie-break and NO take-the-best fallback. Parameter-free: on tally-tied, side-counterbalanced profiles adherence to the most-valid discriminating cue is exactly chance whatever the validity vector.

**(2) Seniority is measured against the chance baseline and the panel's best voice, and the discount is SUPERLINEAR in seniority.** s_j = (v_j - 0.5)/(v_max - 0.5), clipped to [0,1]; news value w_j = 1 - lambda * s_j^omega with omega > 1. The discount is negligible across the bottom half of the panel, about half weight for genuine mid-panel experts, and only for the top one or two voices does it drive weight to zero or below. Two mid-panel experts therefore roughly offset one floor-level voice, while a single top expert is worth nothing or less.

**(2b) The anti-count is BOUNDED: news value has a mild negative floor.** w_j = max(w_floor, 1 - lambda * s_j^omega) with a small negative w_floor. Endorsement by the panel's star can subtract a little, never a lot.

**(3) Redundancy requires same-side corroboration.** lambda_side = lambda_n * (m_side - 1)/((m_side - 1) + kappa). A lone voice cannot be redundant with anybody.

**(4) A lone voice is heard at MORE than face value, with a seniority-graded bonus on top of a constant dissent baseline.** w = phi0 + (phi - phi0) * s_j for a solitary discriminating cue.

**(5) Confidence is diluted by DISSENT more than by sheer cue count, but only MILDLY so.** Two functionally distinct things can make a display 'busy': many attributes discriminating at all, and some of those attributes pointing the other way. Only the second is genuinely undermining, but its undermining force is *modest and graded*, not overwhelming. Formally the divisor is D^gamma_d * (1 + m_lose)^gamma_c with a sub-linear cue-count exponent gamma_d and a mild-to-moderate dissent exponent gamma_c that is bounded well below 1.0, where m_lose is the number of cues backing the count loser. Consequences: unopposed majorities grow in subjective strength with their size (dense-unanimous adherence exceeds sparse-unanimous adherence), while internally contradicted displays are pulled somewhat — but never all the way — toward indifference. Crucially the *asymmetry* of this dilution is empirically constrained: displays with a single dissenting voice (lone-star dissent, the cells that generate the large negative coherence contrasts) must retain nearly full evidential force, whereas displays with two or more dissenters are noticeably softened. A strong dissent exponent would flatten the former and is ruled out.

**(6) Precision also falls, in two regimes, with panel size:** divisor multiplied by (n/6)^tau * (1 + eta * max(0, n - 7)).

**(7) Ordering fidelity decays smoothly with panel span:** lambda_n = lambda * exp(-rho * max(0, n - 7)).

**(8) The decision stage is low-lapse but only moderately steep.** The lapse rate epsilon alone fixes the ceiling attainable on evidence-saturated displays; the slope beta alone fixes how far intermediate, internally conflicted displays are driven from indifference. Stimulus-driven evidence strength and stimulus-independent attentional failure are separate sources of stochasticity.

**Cross-design signature:** exact chance on tally-tied displays of any validity composition; strong but bounded reversal of chorus-backed star majorities on six/seven-expert panels; near-chance for the same structure on eight-expert panels with visibly reduced pure-tally adherence there; near-ceiling and *increasing-with-size* following of unopposed majorities; distinctly middling adherence whenever a numeric majority is actively contradicted by several cues; and mid-panel-backed majorities facing a lone floor voice sitting essentially at chance.

**Parameters:**
- lam: [1.35, 1.90]
- omega: [1.45, 2.05]
- beta: [11.0, 16.0]
- gamma_d: [0.50, 0.85]
- gamma_c: [0.42, 0.90]
- kappa: [0.02, 0.15]
- rho: [0.55, 0.90]
- tau: [0.0, 0.4]
- eta: [1.6, 2.8]
- phi: [1.15, 1.75]
- phi0: [1.00, 1.15]
- w_floor: [-0.45, -0.10]
- epsilon: [0.05, 0.13]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---------- parse the stimulus ----------
    def _pair(st):
        a = None
        b = None
        if isinstance(st, dict):
            for ka, kb in (('option_a_ratings', 'option_b_ratings'),
                           ('a_ratings', 'b_ratings'),
                           ('A', 'B'), ('a', 'b')):
                if ka in st and kb in st:
                    try:
                        a = np.asarray(list(st[ka]), dtype=float).ravel()
                        b = np.asarray(list(st[kb]), dtype=float).ravel()
                    except Exception:
                        a = None
                        b = None
                    break
        if a is None:
            try:
                arr = np.asarray(st, dtype=float)
                arr = arr.reshape(2, -1)
                a = np.asarray(arr[0], dtype=float).ravel()
                b = np.asarray(arr[1], dtype=float).ravel()
            except Exception:
                return None, None
        return a, b

    a, b = _pair(state)
    if a is None or b is None:
        return np.array([0.5, 0.5], dtype=float)
    if a.size == 0 or a.size != b.size:
        return np.array([0.5, 0.5], dtype=float)

    n = int(a.size)

    # ---------- Commitment 1: unweighted count gate ----------
    d = np.sign(a - b)
    iA = np.where(d > 0)[0]
    iB = np.where(d < 0)[0]
    mA = int(iA.size)
    mB = int(iB.size)
    if mA == mB:                       # includes the wholly non-discriminating case
        return np.array([0.5, 0.5], dtype=float)
    D = float(mA + mB)
    m_lose = float(min(mA, mB))        # cues backing the count LOSER (dissent)

    # ---------- parameters ----------
    def _g(name, default, lo, hi):
        try:
            x = float(parameters.get(name, default))
        except Exception:
            x = float(default)
        if not np.isfinite(x):
            x = float(default)
        return float(min(max(x, lo), hi))

    lam = _g('lam', 1.60, 0.0, 3.0)
    omega = _g('omega', 1.70, 0.5, 3.5)
    beta = _g('beta', 13.5, 0.0, 40.0)
    gamma_d = _g('gamma_d', 0.68, 0.0, 2.5)   # sub-linear dilution by sheer cue count
    gamma_c = _g('gamma_c', 0.66, 0.0, 2.5)   # mild dilution by DISSENT
    kappa = _g('kappa', 0.05, 1e-6, 1.0)
    rho = _g('rho', 0.70, 0.0, 2.0)
    tau = _g('tau', 0.20, 0.0, 3.0)
    eta = _g('eta', 2.10, 0.0, 6.0)
    phi = _g('phi', 1.40, 0.5, 2.5)
    phi0 = _g('phi0', 1.08, 0.5, 2.0)
    eps = _g('epsilon', 0.09, 0.0, 0.6)
    w_floor = _g('w_floor', -0.30, -1.5, 0.0)   # bounded anti-count (Commitment 2b)

    # ---------- stated validities ----------
    v = None
    try:
        vr = parameters.get('validities', None) if hasattr(parameters, 'get') else None
        if vr is not None:
            v = np.asarray(vr, dtype=float).ravel()
    except Exception:
        v = None
    if v is None or v.size != n:
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)

    # ---------- Commitment 2: seniority anchored at chance & the panel's best voice ----------
    vmax = float(np.max(v))
    span = vmax - 0.5
    if (not np.isfinite(span)) or span <= 1e-9:
        s = np.zeros(n, dtype=float)
    else:
        s = np.clip((v - 0.5) / span, 0.0, 1.0)
    s_pow = np.power(np.clip(s, 0.0, 1.0), omega)   # superlinear discount profile

    # ---------- Commitment 7: span-graded ordering fidelity ----------
    over = max(0.0, float(n) - 7.0)
    lam_n = lam * float(np.exp(-rho * over))
    if (not np.isfinite(lam_n)) or lam_n < 0.0:
        lam_n = 0.0

    # ---------- Commitment 3: redundancy needs same-side corroboration ----------
    def _side_lam(m):
        mm = float(m)
        if mm <= 1.0:
            return 0.0                       # a lone voice cannot be redundant
        val = lam_n * (mm - 1.0) / ((mm - 1.0) + kappa)
        if (not np.isfinite(val)) or val < 0.0:
            return 0.0
        return float(val)

    lam_A = _side_lam(mA)
    lam_B = _side_lam(mB)

    # ---------- Commitment 2b: news value is bounded below by a mild negative floor ----------
    if mA > 0:
        wA = np.maximum(w_floor, 1.0 - lam_A * s_pow[iA])
    else:
        wA = np.array([], dtype=float)
    if mB > 0:
        wB = np.maximum(w_floor, 1.0 - lam_B * s_pow[iB])
    else:
        wB = np.array([], dtype=float)

    # ---------- Commitment 4: baselined, seniority-scaled lone-dissent salience ----------
    phi_eff = phi if phi > phi0 else phi0
    if mA == 1:
        wA = phi0 + (phi_eff - phi0) * s[iA]
    if mB == 1:
        wB = phi0 + (phi_eff - phi0) * s[iB]

    E = float(np.sum(wA) - np.sum(wB))
    if not np.isfinite(E):
        E = 0.0

    # ---------- Commitment 5: DISSENT-graded dilution (sub-linear in sheer cue count) ----------
    den = (float(D) ** gamma_d) * ((1.0 + m_lose) ** gamma_c)
    if (not np.isfinite(den)) or den <= 0.0:
        den = 1.0

    # ---------- Commitment 6: panel-size precision loss (two regimes) ----------
    size_fac = ((float(n) / 6.0) ** tau) * (1.0 + eta * max(0.0, float(n) - 7.0))
    if (not np.isfinite(size_fac)) or size_fac <= 0.0:
        size_fac = 1.0
    den = den * size_fac

    z = beta * E / den
    if not np.isfinite(z):
        z = 0.0
    z = float(min(max(z, -30.0), 30.0))

    p_a = 1.0 / (1.0 + np.exp(-z))

    # ---------- Commitment 8: attention lapse (ceiling) is separate from slope ----------
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    hits = 0\n    tot = 0\n    a_col = data['option_a_ratings'].tolist()\n    b_col = data['option_b_ratings'].tolist()\n    r_col = data['response'].tolist()\n\n    for a_raw, b_raw, r in zip(a_col, b_col, r_col):\n        try:\n            a = np.asarray([float(x) for x in list(a_raw)])\n            b = np.asarray([float(x) for x in list(b_raw)])\n        except Exception:\n            continue\n        if a.size != b.size or a.size < 5:\n            continue\n        d = np.sign(a - b)\n        iA = np.where(d > 0)[0]\n        iB = np.where(d < 0)[0]\n        try:\n            resp = int(r)\n        except Exception:\n            continue\n        # K4-type displays: a 3-cue majority opposed by exactly ONE cue,\n        # and that lone opposing cue is the panel's floor expert (index 4, v=0.52).\n        if iA.size == 3 and iB.size == 1 and int(iB[0]) == 4:\n            tot += 1\n            if resp == 0:\n                hits += 1\n        elif iB.size == 3 and iA.size == 1 and int(iA[0]) == 4:\n            tot += 1\n            if resp == 1:\n                hits += 1\n\n    if tot == 0:\n        return 0.5\n    return float(hits) / float(tot)\n",
  "rationale": "This isolates the single most diagnostic cell of the design (K4, trials 7/8 and their exact mirrors): a three-cue majority carried by the low-mid experts (.72, .58, .68) opposed by exactly ONE dissenting cue, and that dissenter is the panel's floor voice (.52). The metric is the proportion of those 16 trials on which the subject follows the 3-cue count majority; because both A-favouring and B-favouring mirrors are pooled, side bias cancels exactly.\n\nGVIT-CAP-S4 and SR-DT-6b make opposite-sided predictions here, for structural reasons rather than parameter fine-tuning. Under GVIT, the redundancy discount is strongly concave (omega<1) and its chorus criterion uses the TOTAL number of discriminating cues, so all three mid-level supporters are discounted almost to zero news value while the lone .52 dissenter (panel-relative s = 0 exactly, weight 1.0, parameter-free) retains full weight: E is negative across essentially the whole parameter box, giving sub-chance adherence (~0.2). Under SR-DT, seniority is anchored at chance, the discount is superlinear (so sub-.80 experts keep most of their voice), redundancy is same-side only, and the lone floor dissenter gets only ~phi0 weight; unopposed-ish majorities are further sharpened by the dissent divisor, yielding near-ceiling adherence (~0.9). A 3-vs-1 count majority being REJECTED is impossible anywhere in SR-DT's box.\n\nEach subject contributes 16 trials to the estimate, so within-subject binomial noise is small (<=0.016 variance) and the predicted between-theory gap (~0.7) is many standard errors wide at N=25. I advocate GVIT and predict a value near 0.2; the competitor must predict near 0.9."
}
```

## Usage

```json
{
  "input_tokens": 15336,
  "output_tokens": 3031
}
```
