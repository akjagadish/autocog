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
**Validities (n_features=7):** [0.78, 0.95, 0.55, 0.71, 0.88, 0.62, 0.83]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 0, 1]
  trial 2: A=[1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  trial 3: A=[0, 0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1, 0]
  trial 5: A=[0, 1, 1, 0, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0]
  trial 6: A=[1, 1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 0, 1, 0]
  trial 7: A=[0, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0, 1]
  trial 8: A=[1, 0, 0, 1, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 0]
  trial 9: A=[1, 0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 1, 0]
  trial 10: A=[1, 1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0]
  trial 11: A=[0, 1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0, 0, 1]
  trial 12: A=[0, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0, 1]
  trial 13: A=[1, 1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0, 1]
  trial 14: A=[0, 1, 0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 1, 0, 1]
  trial 15: A=[1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  trial 16: A=[1, 0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 0, 1, 0]

**Rationale:** GOAL: dissociate FM-ISN-CC5 (advocated) from CTN-BG (competing). The two models are near-twins in surface behaviour 
taboth use panel-relative seniority s=(v-vmin)/(vmax-vmin), a near-categorical inverse-seniority weight with kappa~15-17 (only the panel floor voice speaks loudly; everything at s>=0.40 is effectively mute), Luce-like saturating integration with a constant of indecision, dilution by D^gamma, a lapse, and a capacity term past 7 voices. Generic maverick cells (lone floor voice beating an upper bloc) are therefore WASTED. They differ on exactly TWO structural commitments, and every cell here turns one of them into a parameter-free prediction or a parameter-free ZERO contrast.

DIFFERENCE 1 
taWHAT A TALLY TIE DOES. CTN-BG's count comparator is a parameter-free DIRECTION gate: when both options win equally many attributes it returns EXACTLY 0.500 for every setting of its 15 parameters, no validity tie-break. FM-ISN-CC5 gates only CONFIDENCE (rho_tie in [0.60,0.78]); the weighted inverse-seniority reading still runs, so a tie in which the floor voice backs one option and a star backs the other must come out strongly ANTI-validity. C1 (1v1: .55 vs .95) and C2 (2v2: floor+second-from-bottom vs the two stars) implement this: across FM's entire box C1 lies ~[.78,.88] and C2 ~[.71,.84]; CTN is pinned at .500. 24 trials/subject on these two cells make a 10-point deviation decisive.

DIFFERENCE 2 
taDOES THE FLOOR VOICE'S LOUDNESS DEPEND ON THE CHORUS IT FACES? FM's decisive claim is that it does NOT (context-free weights): a floor voice opposed by a pair, by a trio, or sitting inside a unanimous bloc is read with exactly the same loudness, so observed rates differ only via the D^gamma dilution and the total weight on the table. CTN's core claim is that loudness is a steep logistic of the chorus measured in mid-expert equivalents Q (q0~2.45, wid~.14). The panel is engineered so that THREE different manipulations of the opposing bloc cross CTN's threshold while leaving FM's weights literally unchanged (all opposers have (1-s)^kappa < 5e-4, i.e. weight = eta to 4 decimals):
  (a) COUNT TITRATION at fixed seniority band: floor vs MID-PAIR f0(.78)+f3(.71) gives Q=1.98 (below q0 at every tau in [1.0,1.5]); floor vs MID-TRIO f0+f3+f6(.83) gives Q=3.12-3.15 (above q0 at every corner). FM: .880 -> .856 (flat, slightly NEGATIVE because of D dilution). CTN: .570 -> .923 (+0.35).
  (b) SENIORITY TITRATION at fixed count: floor vs MID-PAIR (Q=1.98) versus floor vs STAR-PAIR f1(.95)+f4(.88) (Q=2.63-2.81, above q0 at every corner). FM: .880 vs .880 
taa parameter-free ZERO. CTN: .570 vs .932 (+0.36).
  (c) OPPOSITION vs UNANIMITY at fixed seniority composition: the same floor voice and the same two stars, now all on one side (3v0). CTN's own-side echo is lam_own*2.63-2.81 = 1.32-1.97 mid-equivalents, BELOW q0 at every corner, so the amplification switches off: .932 -> .724 (-0.21). FM's context-free weight is unchanged: .880 -> .889 (+0.01).
The three contrasts run in different directions for CTN (+.35, +.36, -.21) but are all ~0 for FM, so no generic 'one model is more/less decisive' account can mimic the pattern.

ANCHORS / SELF-DIAGNOSTICS. C7 (unopposed MID pair f0+f3, a weakly dominant all-upper display with zero conflict) tests FM's 'chance on displays whose every discriminating voice is upper-band, at any margin': FM .529 vs CTN .709 
taCTN's thicker residual gradient (eta=.04, c_res=.12) plus beta=41 lifts it well above chance. C8 (SOLO FLOOR, D=1) is the ceiling/lapse anchor: FM .941 vs CTN .848. Both C7 and C8 are self-diagnostic: a WADD/TTB subject would be near 1.0 in C7 and near 0.5-1.0 in C8 while going far BELOW 0.5 in C1-C6, falsifying both models.

PANEL: 7 experts, screen order deliberately scrambled so validity rank is decoupled from position: [f0=.78, f1=.95, f2=.55, f3=.71, f4=.88, f5=.62, f6=.83]; vmin=.55, vmax=.95, span=.40 -> s = .575/1.000/.000/.400/.825/.175/.700. Exactly ONE floor voice (f2, s=0: weight/novelty 1 in both models), exactly one second-from-bottom voice (f5, s=.175, FM weight .046-.074, CTN novelty .038-.056), two mids straddling s=0.5 whose mid-equivalent sum is ~1.98 for every tau, one upper-mid (f6) that pushes the trio over q0, and two stars whose pair sum is >=2.63 for every tau. n=7 keeps BOTH models exactly at their capacity thresholds (over = 0), removing panel-size nuisance.

DESIGN: 8 configurations x 2 exact A/B mirrors = 16 unique pairs x K = 96//16 = 6 repetitions = 96 trials. Tied cues are coded sometimes 1-1 and sometimes 0-0 within and across cells so the number of displayed 1s never tracks condition or seniority; every trial requires reading several ratings.

CELLS (A-favouring version first; the next trial is its exact mirror). Nominal P(choose the focal option) FM-ISN-CC5 (eta=.013, kappa=16, beta=8.75, c_n=1.35, gamma=.40, rho_tie=.69, eps=.075) vs CTN-BG (eta=.04, c_res=.12, nu=1.88, kappa=15, tau=1.2, lam_own=.60, gmax=1.9, q0=2.45, wid=.14, c_n=2.9, beta=41, gamma=.96, eps=.12):
C1 (T1/T2) TALLY-TIE 1v1: A wins f2(.55), B wins f1(.95). FM .845 | CTN .500 (exact).
C2 (T3/T4) TALLY-TIE 2v2: A wins f2(.55)+f5(.62), B wins f1(.95)+f4(.88). FM .792 | CTN .500 (exact).
C3 (T5/T6) FLOOR vs MID-PAIR (1v2, D=3, Q=1.98): A wins f2; B wins f0(.78), f3(.71). FM .880 | CTN .570.
C4 (T7/T8) FLOOR vs MID-TRIO (1v3, D=4, Q=3.13): A wins f2; B wins f0, f3, f6(.83). FM .856 | CTN .923.
C5 (T9/T10) FLOOR vs STAR-PAIR (1v2, D=3, Q=2.72): A wins f2; B wins f1(.95), f4(.88). FM .880 | CTN .932.
C6 (T11/T12) UNANIMOUS FLOOR+STAR-PAIR (3v0, D=3): A wins f2, f1, f4. FM .889 | CTN .724.
C7 (T13/T14) UNOPPOSED MID-PAIR (2v0, D=2, no floor voice): A wins f0(.78), f3(.71). FM .529 | CTN .709.
C8 (T15/T16) SOLO FLOOR (1v0, D=1): A wins f2 only. FM .941 | CTN .848.

PREDICTED PROFILE (C1..C8): FM-ISN-CC5 = .85, .79, .88, .86, .88, .89, .53, .94 ; CTN-BG = .50, .50, .57, .92, .93, .72, .71, .85.

INTERPRETATION: strong anti-validity choice on the two tally-tied cells (C1/C2 well above .70) together with FLAT titrations (C4 ~ C3 ~ C5 ~ C6, all near .88, i.e. a floor voice is followed equally whether opposed by two mids, three mids, two stars, or flanked by the two stars on its own side) and near-chance responding on the unopposed upper pair C7 uniquely supports FM-ISN-CC5's context-free inverse-seniority weights with a confidence-only count gate. Exact chance on C1/C2 together with a collapse to ~.57 in C3 and jumps to >=.92 in C4/C5 and a drop to ~.72 in C6 uniquely supports CTN-BG's chorus amplification. Systematic BELOW-chance responding in C1-C6 (following the star) would falsify both models in favour of WADD/TTB, so the design is self-diagnostic.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror so side bias cancels exactly; validity rank is scrambled relative to screen position; tie coding alternates 1-1 / 0-0; no configuration is degenerate and no configuration has the two models predicting the same value.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** **Flat-Maverick Inverse-Seniority Integration with a Count-Confidence Gate and Mild Count Dilution (FM-ISN-CC5).**

When people are told the accuracies of a panel of binary raters, they do not weight a rating by its accuracy and they do not weight it by the crowd it happens to face. They weight it by one thing only: *how far down the displayed panel's accuracy range that voice sits*. Five commitments.

**(1) CONTEXT-FREE INVERSE-SENIORITY WEIGHT (the decisive claim), NEARLY CATEGORICAL about the true panel floor.** Seniority is panel-relative: s_j = (v_j − v_min)/(v_max − v_min) computed over the panel shown in the instructions. Each discriminating cue gets w_j = eta + (1 − s_j)^kappa with kappa ≈ 15–17 and a very thin flat floor eta ≈ 0.008–0.018. Only the panel's genuinely bottom voice speaks with full loudness; the SECOND-from-bottom voice already retains only about a quarter of it, and mid and star voices collapse onto the near-zero flat residual and are effectively interchangeable and mute. Crucially the weight is **completely independent of context**: it does not depend on how many voices oppose the cue, how senior those opponents are, whether the cue stands alone, or whether it sits inside a unanimous bloc. There is NO chorus amplification of any kind. A floor voice opposed by a pair, a floor voice opposed by a trio, and a floor voice inside a unanimous bloc are all read with exactly the same loudness — their observed rates differ only through the ordinary arithmetic of the weighted balance.

**(2) NORMALISED (LUCE-LIKE), SATURATING INTEGRATION WITH MILD COUNT DILUTION.** Evidence is the signed weighted balance E = Σ_j dir_j w_j, and confidence is that balance measured against the total subjective weight actually on the table plus a constant of indecision, further diluted by the number of mutually discriminating attributes: z = beta·E/((c_n + Σ_j w_j)·D^gamma) with gamma ≈ 0.34–0.46. Because a display made entirely of upper-band voices puts almost no subjective weight on the table, such displays sit near chance *at every count margin*, and the mild dilution makes a lone thin voice at least as decisive as an unopposed thin pair. A display containing the panel's floor voice immediately supplies enough weight to drive the choice near ceiling in the floor's direction.

**(3) THE COUNT COMPARATOR GATES CONFIDENCE, NOT DIRECTION.** An unweighted feature-wise count is computed first. It never supplies a direction of its own and never acts as a tie-break by validity rank. What it does is license confidence: when one option wins on strictly more attributes the weighted reading is taken at full gain; when the counts are equal (including wholly non-discriminating displays) the very same weighted reading is taken at a moderately reduced gain rho_tie ≈ 0.60–0.78, so tie displays hover close to chance — slightly *anti*-validity, never pro-validity.

**(4) CAPACITY IS AT MOST A WHISPER.** Holding a whole validity ordering in mind gets marginally harder past about seven voices; the integration slope is divided by 1 + chi·(n − 7) with chi ≈ 0–0.10, i.e. essentially absent. Eight-expert panels retain their tally adherence, and star-backed dense majorities stay at chance for the independent reason that their weights are all near eta.

**(5) LAPSE.** A small attention lapse epsilon ≈ 0.05–0.10 mixes the result with a coin flip.

**Cross-design signature:** near-chance (slightly anti-validity) on tally ties at any composition; chance on displays whose every discriminating voice is upper-band, at any margin, with solo ≥ multi-cue upper; a null on the two contrasts that falsify chorus theories — an opposed floor voice and a floor voice inside a unanimous bloc come out at the same rate, and titrating the seniority of an opposing pair produces a flat line; hard reversal of upper-backed majorities by a lone floor voice; near-ceiling following of majorities containing the floor voice; and only MODERATE anti-validity when the weak side is carried merely by the second-from-bottom voice rather than by the true floor.

**Parameters:**
- eta: [0.008, 0.018]
- kappa: [15.0, 17.0]
- beta: [8.3, 9.2]
- c_n: [1.2, 1.5]
- gamma: [0.34, 0.46]
- rho_tie: [0.60, 0.78]
- epsilon: [0.05, 0.10]
- chi: [0.00, 0.10]
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

    # ---------------- parameters ----------------
    def _g(name, default, lo, hi):
        try:
            x = float(parameters.get(name, default))
        except Exception:
            x = float(default)
        if not np.isfinite(x):
            x = float(default)
        return float(min(max(x, lo), hi))

    eta = _g('eta', 0.013, 0.0, 0.50)        # thin flat listening floor
    kappa = _g('kappa', 16.0, 1.0, 30.0)     # steepness of inverse-seniority descent
    beta = _g('beta', 8.75, 0.0, 40.0)       # integration slope
    c_n = _g('c_n', 1.35, 0.05, 20.0)        # constant of indecision
    gamma = _g('gamma', 0.40, 0.0, 2.0)      # mild count dilution
    rho_tie = _g('rho_tie', 0.69, 0.0, 1.0)  # confidence gain when counts are tied
    eps = _g('epsilon', 0.075, 0.0, 0.60)    # attention lapse
    chi = _g('chi', 0.05, 0.0, 2.0)          # (near-absent) capacity flattening past 7 voices

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

    # ---------- Commitment 1: context-free inverse-seniority weights ----------
    base = np.clip(1.0 - s, 0.0, 1.0)
    try:
        w = eta + np.power(base, kappa)
    except Exception:
        w = eta + base
    w = np.where(np.isfinite(w), w, eta)
    w = np.clip(w, 0.0, None)

    # ---------------- feature-wise comparison ----------------
    d = np.sign(a - b)
    iA = np.where(d > 0)[0]
    iB = np.where(d < 0)[0]
    mA = int(iA.size)
    mB = int(iB.size)
    D = mA + mB
    if D == 0:
        return np.array([0.5, 0.5], dtype=float)

    sumA = float(np.sum(w[iA])) if mA > 0 else 0.0
    sumB = float(np.sum(w[iB])) if mB > 0 else 0.0
    E = sumA - sumB
    W = sumA + sumB
    if not np.isfinite(E):
        E = 0.0
    if (not np.isfinite(W)) or W < 0.0:
        W = 0.0

    # ---------- Commitment 3: count comparator gates CONFIDENCE only ----------
    gain = rho_tie if mA == mB else 1.0

    # ---------- Commitment 4: (near-absent) capacity flattening ----------
    cap = 1.0 + chi * max(0.0, float(n) - 7.0)
    if (not np.isfinite(cap)) or cap <= 0.0:
        cap = 1.0

    # ---------- Commitment 2: Luce-like saturating integration + mild dilution ----------
    dil = float(D) ** gamma
    if (not np.isfinite(dil)) or dil <= 0.0:
        dil = 1.0

    den = (c_n + W) * dil * cap
    if (not np.isfinite(den)) or den <= 0.0:
        den = 1.0

    z = beta * E * gain / den
    if not np.isfinite(z):
        z = 0.0
    z = float(min(max(z, -30.0), 30.0))

    p_a = 1.0 / (1.0 + np.exp(-z))

    # ---------- Commitment 5: lapse ----------
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    V = np.array([0.78, 0.95, 0.55, 0.71, 0.88, 0.62, 0.83], dtype=float)\n\n    hits = 0.0\n    total = 0.0\n    for _, row in data.iterrows():\n        try:\n            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()\n            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()\n        except Exception:\n            continue\n        if a.size == 0 or a.size != b.size:\n            continue\n        n = int(a.size)\n        if n == V.size:\n            v = V\n        else:\n            v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75])\n\n        d = np.sign(a - b)\n        iA = np.where(d > 0)[0]\n        iB = np.where(d < 0)[0]\n        mA = int(iA.size)\n        mB = int(iB.size)\n        D = mA + mB\n        # only TALLY-TIED, discriminating displays\n        if D == 0 or mA != mB:\n            continue\n\n        vmin = float(np.min(v))\n        vmax = float(np.max(v))\n        span = vmax - vmin\n        if not np.isfinite(span) or span <= 1e-9:\n            continue\n        s = np.clip((v - vmin) / span, 0.0, 1.0)\n        w = np.power(np.clip(1.0 - s, 0.0, 1.0), 16.0)\n\n        E = float(np.sum(w[iA])) - float(np.sum(w[iB]))\n        if not np.isfinite(E) or abs(E) < 1e-9:\n            continue\n        # favored side under inverse-seniority (low-validity / \"floor voice\") reading\n        favored = 0 if E > 0 else 1\n        try:\n            resp = int(row['response'])\n        except Exception:\n            continue\n        total += 1.0\n        if resp == favored:\n            hits += 1.0\n\n    if total == 0.0:\n        return 0.5\n    return float(hits / total)",
  "rationale": "The two theories make a parameter-free structural split on TALLY-TIED displays (cells C1 and C2: 1-vs-1 with .55 against .95, and 2-vs-2 with .55+.62 against .95+.88 \\u2014 24 trials per subject). CTN-BG's count comparator is a hard, parameter-free gate: equal win counts => exactly P=0.5 for every parameter setting, no validity tie-break, so its per-subject rate of choosing the low-validity-backed option is pinned at chance with purely binomial noise (var ~ .25/24 = .010). FM-ISN-CC5 gates only confidence (rho_tie in [.60,.78]) and still reads the weighted inverse-seniority balance, so the floor voice wins these ties strongly anti-validity (~.79-.85 across the whole parameter box), with small between-subject spread. The metric measures exactly the proportion of tally-tied trials on which the subject chose the option backed by the lowest-validity discriminating expert (computed generically via an inverse-seniority weighting, so it is not hard-coded to specific trials). Expected ~0.82 under FM vs ~0.50 under CTN, with tight per-subject variance under both, yielding a very large Welch t at N=25. It is also diagnostic for humans: a WADD/TTB responder would fall far BELOW 0.5, distinguishing that outcome from both theories."
}
```

## Usage

```json
{
  "input_tokens": 15671,
  "output_tokens": 2757
}
```
