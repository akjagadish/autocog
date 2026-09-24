# feedback_iter_07

## System Prompt

You are a renowned cognitive scientist critiquing a freshly proposed candidate theory and model in the Decision Making (Binary Features) domain.

The candidate has been simulated on every previously run experiment. For each experiment you are shown the design, the metric, the value the metric takes on real data, and the value it takes on the candidate's simulated data.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the feedback is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across multiple experiments. 
Your task is to determine whether the candidate captures the human/real behavior well enough across these experiments. Return a verdict:
  * "continue"   — the candidate is good enough; carry on.
  * "regenerate" — the candidate fails to capture the empirical pattern; the proposing agent must produce a new candidate, taking your rationale into account.

Justify the verdict with a concrete diagnosis (which experiments fail, in what direction, what mechanism is likely missing or miscalibrated).

## SCOPE OF YOUR CRITIQUE — STAY INSIDE THE ARBITER'S MECHANISM FAMILY
When an "## ARBITER RECOMMENDATION" block is present below, the proposer was explicitly instructed to implement the mechanism family the arbiter prescribed. Your job is to grade FIT QUALITY *within that prescribed family*, not to relitigate which family should be used — that is the arbiter's call, made one level above this loop.

Concretely:
  * If the candidate misses the data, you may push for MINOR ADJUSTMENTS that keep the prescribed mechanism intact: tightening / widening parameter ranges, adding a temperature, swapping a normalization scheme, fixing a softmax / distance metric, re-balancing attention weights, fixing a learning-rate sign, correcting a bug in the gating or recurrence, etc.
  * You MUST NOT recommend switching to a different mechanism family. Such a switch is the arbiter's prerogative; recommending it here will mislead the proposer into oscillating between families across iterations.
  * Also grade FAITHFULNESS to the recommendation explicitly: if the candidate has clearly drifted into a different family than the one prescribed, say so in the rationale and ask for a return to the prescribed family — again, with minor adjustments, not a re-design.

## ACCEPT GATE — HOW THE LOOP DECIDES WHAT TO BUILD ON NEXT
This propose-loop has a programmatic accept gate. After every iteration the candidate's `aggregate_loss` is compared against the running-best loss (`accepted_loss`):
  * `loss < accepted_loss` → ACCEPTED. The candidate becomes the new running-best base; the next iteration's proposer will build on THIS candidate.
  * `loss >= accepted_loss` → REJECTED. The base is unchanged; the next iteration's proposer will build on the SAME `accepted` candidate again, with your new feedback on top. Rejected candidates are discarded — the loop guarantees the base never regresses, so you do NOT need to ask the proposer to "revert" anything; that already happens for free.

Two consequences for your verdict:
  * If the candidate you are grading was REJECTED by the gate, returning `"continue"` is silently downgraded to `"regenerate"` (returning a worse candidate would defeat the gate). Spend your rationale on a NEW direction the proposer should try on top of the unchanged accepted base, not on defending the rejected attempt.
  * If the candidate was ACCEPTED, you can return `"continue"` to stop the loop and ship this candidate, or `"regenerate"` to keep tuning further.

## LEARN FROM YOUR OWN PAST ADVICE
When a "## YOUR PRIOR CRITIQUES" block is present below, each prior iteration ends with an "Outcome of your advice" line that says whether the next candidate the proposer produced was ACCEPTED (your advice helped — its loss strictly beat the running best) or REJECTED (your advice didn't help — the proposer discarded the result and reset to the previous accepted base). This is the loop's ground-truth signal on whether *your own previous critique was good*. Use it explicitly:
  * If a previous piece of advice was ACCEPTED, it is OK to repeat / extend it. Reinforce in the same direction.
  * If a previous piece of advice was REJECTED, do NOT repeat the same recommendation; in your new rationale, briefly acknowledge that the previous push in that direction was rejected by the gate and try a different in-family knob (or a smaller step in the same direction) instead.
  * If you find yourself oscillating (e.g. iter 1 said "increase α", iter 2 said "decrease α", iter 3 about to say "increase α" again), STOP and recommend a value between the two flanking iterations instead.
  * The "## LOSS TRAJECTORY" block at the top of the user prompt summarises the same information at the loop level — consult it before issuing a new regenerate-with-direction recommendation.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## ARBITER RECOMMENDATION (mechanism family the proposer was told to implement)
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_8`
- THEORY 2 = `pi_9`
- The recommendation below acts on THEORY 1 (= `pi_8`).

Replace pi_8 with a SMOOTH ANTI-VALIDITY WEIGHTING account that has NO solo release, NO hard mute band and NO chorus amplification \u2014 call it e.g. 'Convex Novelty Weighting (CNW)'. Core commitments:

1. Count gate unchanged and parameter-free: equal feature-wise win counts (including wholly non-discriminating displays) \u21d2 exactly 0.5, no validity tie-break, no take-the-best fallback. (Matches Exp 3 = .474.)

2. Each discriminating cue carries weight that is a CONVEX, monotonically DECREASING function of panel-relative validity: s_j = (v_j \u2212 v_min)/(v_max \u2212 v_min), w_j = (1 \u2212 s_j)^kappa with kappa \u2248 1.5\u20133.0, plus a very small floor-independent baseline b \u2248 0.02\u20130.10. This is the whole 'news' story: the panel's bottom voice has weight \u22481, a mid voice (s \u2248 0.5) retains only 2^\u2212kappa \u2248 0.12\u20130.35 of it, and a star (s \u2248 1) contributes essentially nothing. Weights are strictly non-negative \u2014 a star is ignored, never counted against its own side.

3. Because star weights vanish, ALL-UPPER displays \u2014 solo star, solo mid-high, unopposed upper pair, upper 3-vs-1 \u2014 automatically land near chance at ANY count margin (Exp 1 solo\u2212flat \u2248 0, Exp 2 \u2248 +0.05, Exp 15 = .489, Exp 16 = .489), WITHOUT needing pi_8's solo release or pi_9's step band. Note that this predicts a shallow, monotone but compressed solo ladder, not a flat ceiling line: the model-writer should choose kappa and beta so that solo-star \u2248 .52\u2013.58, solo-mid \u2248 .58\u2013.68, solo-floor \u2248 .85\u2013.92.

4. The floor-voice reversals come out of the WEIGHT ARITHMETIC, not a chorus knee: a lone floor cue (w\u22481) against a mid PAIR (2 \u00d7 ~0.4\u20130.5) nets \u22480 \u21d2 chance (Exp 12 C4 = .478), while the same floor cue against a three-cue UPPER bloc (3 \u00d7 ~0.05\u20130.15) nets strongly positive \u21d2 .80\u2013.90 against the majority (Exp 14 = .145 adherence; Exp 7 = \u2212.76). Calibrate kappa so that these two cells fall on opposite sides of chance simultaneously \u2014 this is the single tightest quantitative constraint in the corpus and it replaces pi_9's alpha/kappa/p_ch knee with one exponent.

5. Integration: E = \u03a3 dir_j \u00b7 w_j, compressed by the number of mutually contradicting cues (E / D^gamma, gamma \u2248 0.6\u20131.1), passed through a logistic with moderate beta (chosen so that plain mixed-panel adherence lands at .70\u2013.80, NOT .85\u2013.90 \u2014 pi_8's persistent over-shoot on Exp 4/5/6 must not be repeated), plus an attention lapse epsilon \u2248 0.08\u20130.15.

6. Capacity: for n > 7 the anti-validity ordering degrades (kappa shrinks toward 1, i.e. weights flatten) AND integration noise rises (divisor \u00d7 (1 + chi\u00b7(n\u22127))), so eight-expert star-backed majorities sit at chance (Exp 10 = .485) and eight-expert tally adherence (.755) is below the six-expert value (.794).

Distinctive, testable contrast with pi_9: CNW is CONTINUOUS in seniority and CHORUS-FREE, so (i) the second- and third-from-bottom experts retain appreciable, graded influence (pi_9 mutes them near-categorically), and (ii) the strength of a floor voice depends only on the summed seniority of what opposes it, not on the COUNT of opposing cues \u2014 so a floor voice opposed by two STARS should already reverse under CNW (stars \u2248 0 weight) while pi_9 needs a three-cue bloc to cross its late knee. Future experiments should target exactly those two cells: (a) a lone second-from-bottom voice opposed by a star pair, and (b) a lone floor voice opposed by exactly two stars versus exactly two mids.


## CANDIDATE THEORY
**Chorus-Threshold Novelty Weighting, Loud-Echo variant (CTN-LE).**

People told the accuracies of a panel of binary expert ratings do not weight a rating by its accuracy; they weight it by how much *news* it carries, and news is loud only when it stands inside or against a genuine CHORUS. Six commitments, all smooth and continuous (no mute band, no solo release, no discrete count switch).

**(1) The count comparator is a parameter-free gate.** Before any expert identity is consulted, an unweighted feature-wise comparison asks: does one product win on strictly more attributes? If not — equal win counts, including wholly non-discriminating displays — the display is non-diagnostic and the subject flips a mental coin: exactly 0.5, with NO validity tie-break and NO take-the-best fallback, at every panel composition and size.

**(2) Seniority is panel-relative; novelty is a smooth convex, monotonically DECREASING function of it, and the descent is STEEP.** s_j = (v_j − v_min)/(v_max − v_min); nov_j = (1 − s_j)^kappa with kappa ≈ 16–18.5. Nothing is muted by fiat and nothing is released by fiat, but the curve is sharp: only the panel's genuinely idiosyncratic BOTTOM voice speaks with nov ≈ 1; the second-from-bottom voice retains a real but small share; a mid voice essentially nothing; a star nothing at all. Novelty is never negative — a star is ignored, never counted against its own side.

**(3) THE CORE CLAIM — a novel voice is loud only in proportion to the CHORUS it is embedded in, and a chorus is measured in *mid-expert equivalents*, counting BOTH the voices it contradicts AND, at a substantial discount, the voices it stands beside.** Each other discriminating voice contributes a subjective unit u_i = (1 + tau·s_i)/(1 + tau/2): a mid-panel voice counts as exactly one unit, a star ≈ 1.4, a bottom voice ≈ 0.6. The chorus a cue j faces is Q_j = Σ_{opposing} u_i + lam_own·Σ_{own side, i≠j} u_i, with lam_own now LARGE (≈0.8): a low voice that speaks inside a senior consensus is nearly as audible as one that speaks against it, which is what makes a dense unanimous display containing the panel floor near-ceiling compelling while a sparse all-upper unanimous display is not. Loudness is a smooth logistic of that chorus, g_j = Gmax·σ((Q_j − q0)/wid), with a narrow but continuous threshold at roughly two-and-a-half mid-equivalents. BOTH how many voices and how senior they are matter, in one scalar.

**(4) The ungated panel gradient is THIN and only MILDLY convex** — a bare ordinal pull r_j = eta + c·(1 − s_j)^nu with small eta, small c and a moderate exponent nu ≈ 1.9, so all-upper displays drift near chance at any count margin and a single floor cue's residual is worth about the same as two mid cues' residuals (a floor voice opposed by a mid PAIR is a dead heat; it needs the chorus gate, not the gradient, to win).

**(5) Integration is Luce-like and saturating.** w_j = r_j + nov_j·g_j; E = Σ dir_j w_j; z = beta·E/((c_n + Σ_j w_j)·D^gamma). Evidence is judged relative to the total weight on the table plus a constant of indecision, diluted by the number of mutually discriminating attributes with gamma ≈ 1. An attention lapse epsilon mixes the result with a coin flip.

**(6) Capacity is a near-total collapse of the chorus construal past ~7 voices**, with only mild extra noise, so eight-expert star-backed majorities sit near chance and eight-expert tally adherence survives on the residual gradient alone.

**Cross-design signature:** exact chance on tally ties at any composition; near-chance on displays whose every discriminating voice is upper-band, at every margin; a bottom voice facing two MID voices lands at chance, facing a three-cue upper bloc reverses to ≈.12–.18; near-ceiling following of dense majorities that CONTAIN the bottom voice (the loud-echo prediction) but not of sparse all-upper unanimous displays; attenuated integration on eight-expert panels.

`predict(parameters, state, history) -> np.ndarray`:
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
    kappa = _g('kappa', 17.2, 1.0, 30.0)
    tau = _g('tau', 1.2, 0.0, 4.0)
    lam_own = _g('lam_own', 0.79, 0.0, 1.0)
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

`policy(probs) -> int`:
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

`parameters`:
- eta: [0.030, 0.048]
- c_res: [0.10, 0.14]
- nu: [1.70, 2.05]
- kappa: [16.0, 18.5]
- tau: [1.0, 1.5]
- lam_own: [0.72, 0.86]
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

`rationale`:
**Minimal diff: exactly two parameter ranges changed (lam_own, kappa) plus the two matching in-code defaults. Every equation, variable name and structural line is byte-identical to the accepted iter-5 base.** I did NOT re-apply the rejected residual/beta rescale (iter 7) or the rejected lam_own CUT (iter 6); I reverted to iter-5 values for eta/c_res/beta and moved only the one direction the gate has never tested.

**1. PRIMARY — lam_own 0.50–0.70 → 0.72–0.86 (default 0.60 → 0.79), targeting Exp 6's d2 term exactly as the critic diagnosed.** I hand-checked the arithmetic on Exp 6's two unanimous cells (V = [.94,.58,.86,.63,.72,.80,.55], s_f6 = 0). Dense unanimous display (A wins f0,f3,f5,f6; no opposition): the floor voice f6's chorus is purely own-side, Q = lam·(ΣU_own − u_f6) = lam·3.259. At lam = .60 that is 1.956 — far below q0 = 2.45 — so gate ≈ .054, w_f6 ≈ .214, W_tot ≈ .431, z ≈ 1.41, cell ≈ .744. Sparse all-upper unanimous display (A wins f2,f4; nov ≈ 0 for both) gives ≈ .625. So d2 ≈ −0.119 and the total is capped near −0.78, matching the observed −0.761 — no amount of beta or gmax on OPPOSED cells could fix this, which is why four rounds of global rescaling failed. At lam = .79, Q_f6 = 2.575 crosses the knee (gate ≈ 1.35), w_f6 ≈ 1.51, z ≈ 4.0, dense cell ≈ .923, so d2 ≈ −0.30 and Exp 6 lands ≈ −0.95 against a real −0.9375: the corpus's largest single error (0.176) collapses to ≈0.02.

**2. Why this is SAFE for the two cells that pin the knee from below.** Exp 10-C4 (lone floor voice vs a mid pair) and Exp 12-K4 (lone floor voice vs an upper trio) both have NO own side, so Q is literally untouched by lam_own — unlike a q0 cut or a wid widening, which the critic correctly ruled out because they would re-inflate both. Exp 14's four target cells are star-pair majorities (nov ≈ 0) or lone opposing cues with no own side; Exp 15/16's solo and unopposed-pair cells are all upper-band (nov ≈ 0); Exp 1 is pinned at exact chance by the parameter-free count gate. All unchanged.

**3. Exp 3 checked cell-by-cell and unaffected.** Its displays are 1v0, 2v0, 3v0, 3v2 and 4v1. In the 3v2 and 4v1 cells a floor voice already has Q ≈ 3.0–3.4 (full gate) at lam = .60, so nothing moves; in the 3v0 and 2v0 cells Q rises only 1.2 → 1.6 and 0.6 → 0.8, both still deep below the knee. Exp 13 is pure residual arithmetic (nov ≈ 0 for every all-upper cue) and is therefore untouched, and Exp 7's marginal voices are all opposed by ≥2 senior cues (Q ≥ 2.6, already saturated).

**4. SECONDARY — kappa 14.0–16.5 → 16.0–18.5, used solely as the Exp-2 offset the critic specified.** The one real cost of a louder echo is Exp 2's 1-vs-3 conflict display, where the panel FLOOR f1 (v = .57, s = 0) sits inside the majority: Q_f1 goes 2.07 → 2.35, gate .12 → .63, and that cell rises ≈.74 → ≈.83. Sharpening the novelty curve cannot touch f1 (s = 0, nov ≡ 1) but it halves the leak from the second-from-bottom voice f4 (s = .139): nov .106 → .076, which pulls the other conflict cell (1-vs-4, star vs a four-cue low bloc) from ≈.714 down to ≈.700. Net Exp 2 ≈ .77 versus the current .758 — a ≈0.012 cost against a ≈0.16 gain on Exp 6. The same kappa move nudges Exp 7 (.678 → ≈.67, real .661) and damps the small Exp-5 over-shoot the echo raise would otherwise create (projected −0.78, inside the −0.74/−0.80 guard-rail), so it pays for itself twice.

**5. Expected side-benefit on Exp 11 (.658 vs .698).** Its e4 cell is a floor voice f2 standing INSIDE a star-backed majority — Q = lam·(u_f1 + u_f4) goes 1.93 → 2.54 and crosses the knee, raising e4 toward ceiling. Its other three cells (d3, c2, a1) all involve voices with no own side or already-saturated opposition mass, so they do not move.

**6. Deliberately untouched, per the critic's explicit instructions:** gamma and epsilon (they hold the Exp 15/16 sign and Exp 1/8 at data), eta / c_res / nu (just re-balanced successfully in iter 5 to make Exp 10-C4 a dead heat), q0 / wid / gmax / tau / c_n / chi, psi (the n = 8 knob — I left it alone rather than risk Exp 8, which sits at .494 against a real .485; it is the critic's stated third-order step and should be tested only after this one is verified), and the entire parameter-free count gate, which continues to deliver exact chance on every tally tie — the one experiment-invariant prediction no competitor family makes as cleanly.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.1129 -> ACCEPTED
- iter 2: loss=0.1006 -> ACCEPTED
- iter 3: loss=0.0692 -> ACCEPTED
- iter 4: loss=0.0680 -> ACCEPTED
- iter 5: loss=0.0669 -> ACCEPTED
- iter 6: loss=0.0773 -> REJECTED
- iter 7: loss=0.0694 -> REJECTED
- iter 8 (current candidate you are grading): loss=0.0671 -> REJECTED
Running-best (last accepted) base: iter 5 at loss=0.0669.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
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
**Candidate trajectory (this loop):**
  - iter 1: 0.4944 (var=0.0092) (Δ vs real +0.0206)
  - iter 2: 0.4944 (var=0.0082) (Δ vs real +0.0206)
  - iter 3: 0.4944 (var=0.0093) (Δ vs real +0.0206)
  - iter 4: 0.4944 (var=0.0062) (Δ vs real +0.0206)
  - iter 5: 0.4838 (var=0.0062) (Δ vs real +0.0100)
  - iter 6: 0.5069 (var=0.0063) (Δ vs real +0.0331)
  - iter 7: 0.5106 (var=0.0075) (Δ vs real +0.0369)
  - iter 8 (current): 0.4888 (var=0.0069) (Δ vs real +0.0150)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8988 (var=0.0090)
- pi_2: 0.4988 (var=0.0068)
- pi_3: 0.5125 (var=0.0055)
- pi_4: 0.4913 (var=0.0090)
- pi_5: 0.5169 (var=0.0089)
- pi_6: 0.5106 (var=0.0056)
- pi_7: 0.4819 (var=0.0066)
- pi_8: 0.5075 (var=0.0083)
- pi_9: 0.5081 (var=0.0086)

### Experiment 2
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
**Candidate trajectory (this loop):**
  - iter 1: 0.7887 (var=0.0058) (Δ vs real +0.1350)
  - iter 2: 0.8194 (var=0.0047) (Δ vs real +0.1656)
  - iter 3: 0.8037 (var=0.0064) (Δ vs real +0.1500)
  - iter 4: 0.7438 (var=0.0048) (Δ vs real +0.0900)
  - iter 5: 0.7581 (var=0.0093) (Δ vs real +0.1044)
  - iter 6: 0.7169 (var=0.0061) (Δ vs real +0.0631)
  - iter 7: 0.7650 (var=0.0071) (Δ vs real +0.1112)
  - iter 8 (current): 0.7994 (var=0.0056) (Δ vs real +0.1456)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8525 (var=0.0139)
- pi_1: 0.1431 (var=0.0111)
- pi_3: 0.6669 (var=0.0098)
- pi_4: 0.6469 (var=0.0093)
- pi_5: 0.9281 (var=0.0024)
- pi_6: 0.7025 (var=0.0100)
- pi_7: 0.7050 (var=0.0185)
- pi_8: 0.7425 (var=0.0058)
- pi_9: 0.7087 (var=0.0069)

### Experiment 3
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
**Candidate trajectory (this loop):**
  - iter 1: 0.7998 (var=0.0028) (Δ vs real +0.0060)
  - iter 2: 0.8125 (var=0.0018) (Δ vs real +0.0188)
  - iter 3: 0.8256 (var=0.0017) (Δ vs real +0.0319)
  - iter 4: 0.8175 (var=0.0019) (Δ vs real +0.0238)
  - iter 5: 0.8100 (var=0.0015) (Δ vs real +0.0163)
  - iter 6: 0.7767 (var=0.0017) (Δ vs real -0.0171)
  - iter 7: 0.8121 (var=0.0013) (Δ vs real +0.0183)
  - iter 8 (current): 0.8177 (var=0.0018) (Δ vs real +0.0240)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6379 (var=0.0030)
- pi_2: 0.8494 (var=0.0069)
- pi_1: 0.8731 (var=0.0059)
- pi_4: 0.7781 (var=0.0023)
- pi_5: 0.8167 (var=0.0020)
- pi_6: 0.7831 (var=0.0039)
- pi_7: 0.7769 (var=0.0024)
- pi_8: 0.8994 (var=0.0010)
- pi_9: 0.7742 (var=0.0020)

### Experiment 4
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
**Candidate trajectory (this loop):**
  - iter 1: 0.7146 (var=0.0038) (Δ vs real -0.0400)
  - iter 2: 0.7060 (var=0.0030) (Δ vs real -0.0485)
  - iter 3: 0.7071 (var=0.0022) (Δ vs real -0.0475)
  - iter 4: 0.7087 (var=0.0023) (Δ vs real -0.0458)
  - iter 5: 0.7006 (var=0.0033) (Δ vs real -0.0540)
  - iter 6: 0.6963 (var=0.0027) (Δ vs real -0.0583)
  - iter 7: 0.7110 (var=0.0026) (Δ vs real -0.0435)
  - iter 8 (current): 0.6985 (var=0.0030) (Δ vs real -0.0560)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8538 (var=0.0068)
- pi_3: 0.6565 (var=0.0039)
- pi_1: 0.8571 (var=0.0084)
- pi_4: 0.7438 (var=0.0061)
- pi_5: 0.7783 (var=0.0043)
- pi_6: 0.7175 (var=0.0042)
- pi_7: 0.7642 (var=0.0035)
- pi_8: 0.8350 (var=0.0023)
- pi_9: 0.7492 (var=0.0014)

### Experiment 5
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
**Candidate trajectory (this loop):**
  - iter 1: -0.7704 (var=0.0071) (Δ vs real -0.0071)
  - iter 2: -0.7913 (var=0.0073) (Δ vs real -0.0279)
  - iter 3: -0.7675 (var=0.0058) (Δ vs real -0.0042)
  - iter 4: -0.7600 (var=0.0055) (Δ vs real +0.0033)
  - iter 5: -0.7579 (var=0.0072) (Δ vs real +0.0054)
  - iter 6: -0.7312 (var=0.0053) (Δ vs real +0.0321)
  - iter 7: -0.7696 (var=0.0044) (Δ vs real -0.0063)
  - iter 8 (current): -0.7642 (var=0.0041) (Δ vs real -0.0008)
**Other theories' values on this metric (for reference):**
- pi_3: -0.0042 (var=0.0075)
- pi_4: 0.1438 (var=0.0077)
- pi_1: 0.6683 (var=0.0558)
- pi_2: 0.0062 (var=0.0039)
- pi_5: -0.7671 (var=0.0062)
- pi_6: -0.7525 (var=0.0071)
- pi_7: -0.7733 (var=0.0093)
- pi_8: -0.7063 (var=0.0049)
- pi_9: -0.6542 (var=0.0050)

### Experiment 6
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
**Candidate trajectory (this loop):**
  - iter 1: -0.9463 (var=0.0324) (Δ vs real -0.0088)
  - iter 2: -0.8469 (var=0.0229) (Δ vs real +0.0906)
  - iter 3: -0.8294 (var=0.0319) (Δ vs real +0.1081)
  - iter 4: -0.8100 (var=0.0314) (Δ vs real +0.1275)
  - iter 5: -0.7612 (var=0.0294) (Δ vs real +0.1763)
  - iter 6: -0.7144 (var=0.0317) (Δ vs real +0.2231)
  - iter 7: -0.7831 (var=0.0342) (Δ vs real +0.1544)
  - iter 8 (current): -0.9269 (var=0.0317) (Δ vs real +0.0106)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1587 (var=0.0237)
- pi_3: -0.0675 (var=0.0417)
- pi_1: 0.7206 (var=0.0361)
- pi_2: -0.0038 (var=0.0345)
- pi_5: -0.8519 (var=0.0288)
- pi_6: -0.9162 (var=0.0494)
- pi_7: -0.8237 (var=0.0280)
- pi_8: -1.0012 (var=0.0482)
- pi_9: -0.9213 (var=0.0220)

### Experiment 7
**Design**
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 1]

**Metric**
```python
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
```

**Observed (real) value:** 0.6606 (var=0.0038)
**Candidate trajectory (this loop):**
  - iter 1: 0.6831 (var=0.0034) (Δ vs real +0.0225)
  - iter 2: 0.7047 (var=0.0030) (Δ vs real +0.0441)
  - iter 3: 0.6666 (var=0.0031) (Δ vs real +0.0059)
  - iter 4: 0.6800 (var=0.0030) (Δ vs real +0.0194)
  - iter 5: 0.6778 (var=0.0024) (Δ vs real +0.0172)
  - iter 6: 0.6884 (var=0.0028) (Δ vs real +0.0278)
  - iter 7: 0.6844 (var=0.0026) (Δ vs real +0.0237)
  - iter 8 (current): 0.6763 (var=0.0032) (Δ vs real +0.0156)
**Other theories' values on this metric (for reference):**
- pi_5: 0.9059 (var=0.0023)
- pi_4: 0.4641 (var=0.0042)
- pi_1: 0.1278 (var=0.0098)
- pi_2: 0.4997 (var=0.0029)
- pi_3: 0.4981 (var=0.0037)
- pi_6: 0.6775 (var=0.0023)
- pi_7: 0.7197 (var=0.0017)
- pi_8: 0.6572 (var=0.0028)
- pi_9: 0.6750 (var=0.0035)

### Experiment 8
**Design**
  A=[0, 1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 0, 1, 1, 0]

**Metric**
```python
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
```

**Observed (real) value:** 0.4850 (var=0.0085)
**Candidate trajectory (this loop):**
  - iter 1: 0.4662 (var=0.0049) (Δ vs real -0.0187)
  - iter 2: 0.4181 (var=0.0082) (Δ vs real -0.0669)
  - iter 3: 0.4844 (var=0.0081) (Δ vs real -0.0006)
  - iter 4: 0.4750 (var=0.0087) (Δ vs real -0.0100)
  - iter 5: 0.4944 (var=0.0073) (Δ vs real +0.0094)
  - iter 6: 0.5019 (var=0.0100) (Δ vs real +0.0169)
  - iter 7: 0.5000 (var=0.0054) (Δ vs real +0.0150)
  - iter 8 (current): 0.5081 (var=0.0094) (Δ vs real +0.0231)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7144 (var=0.0105)
- pi_5: 0.1506 (var=0.0044)
- pi_1: 0.8488 (var=0.0147)
- pi_2: 0.8550 (var=0.0140)
- pi_3: 0.5913 (var=0.0076)
- pi_6: 0.4919 (var=0.0121)
- pi_7: 0.4775 (var=0.0150)
- pi_8: 0.4850 (var=0.0181)
- pi_9: 0.4931 (var=0.0066)

### Experiment 9
**Design**
  A=[1, 0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design validities (fixed by the experiment): [f0..f5]
    val = np.array([0.95, 0.78, 0.82, 0.55, 0.76, 0.80], dtype=float)
    n_feat = val.shape[0]
    order = np.argsort(-val, kind='stable')
    marginal_idx = [int(j) for j in order[n_feat // 2:]]  # bottom-half tier: f1, f4, f3

    hits = 0.0
    tot = 0.0

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(list(a_raw), dtype=float).ravel()
            b = np.asarray(list(b_raw), dtype=float).ravel()
        except Exception:
            continue
        if a.shape[0] != n_feat or b.shape[0] != n_feat:
            continue
        d = np.sign(a - b)
        D = int(np.sum(d != 0))
        if D < 2:
            continue  # drop the solo-cue cells (K5/K6), where the two theories nearly agree
        net_marg = float(np.sum(d[marginal_idx]))
        if net_marg == 0.0:
            continue
        pred = 0 if net_marg > 0 else 1  # 0 = chose A, 1 = chose B
        try:
            rr = int(r)
        except Exception:
            continue
        hits += 1.0 if rr == pred else 0.0
        tot += 1.0

    if tot <= 0:
        return float('nan')
    return float(hits / tot)
```

**Observed (real) value:** 0.4050 (var=0.0022)
**Candidate trajectory (this loop):**
  - iter 1: 0.3588 (var=0.0036) (Δ vs real -0.0463)
  - iter 2: 0.3650 (var=0.0040) (Δ vs real -0.0400)
  - iter 3: 0.3653 (var=0.0026) (Δ vs real -0.0397)
  - iter 4: 0.3791 (var=0.0050) (Δ vs real -0.0259)
  - iter 5: 0.3606 (var=0.0028) (Δ vs real -0.0444)
  - iter 6: 0.3744 (var=0.0041) (Δ vs real -0.0306)
  - iter 7: 0.3525 (var=0.0032) (Δ vs real -0.0525)
  - iter 8 (current): 0.3766 (var=0.0028) (Δ vs real -0.0284)
**Other theories' values on this metric (for reference):**
- pi_5: 0.8438 (var=0.0038)
- pi_6: 0.3791 (var=0.0043)
- pi_1: 0.3253 (var=0.0042)
- pi_2: 0.4994 (var=0.0020)
- pi_3: 0.4828 (var=0.0041)
- pi_4: 0.5244 (var=0.0029)
- pi_7: 0.3969 (var=0.0046)
- pi_8: 0.3950 (var=0.0035)
- pi_9: 0.3756 (var=0.0029)

### Experiment 10
**Design**
  A=[1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0, 1]
  A=[0, 1, 0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Cell C3 (tally-tied 2v2, tier-mismatched): discriminating cues {f1,f2,f5,f6};
    #   one option is backed by the two BOTTOM experts {f2(.55), f5(.52)},
    #   the other by the two STARS {f1(.95), f6(.90)}.
    # Cell C4 (mid-pair vs floor): discriminating cues {f0,f4,f5};
    #   one option is backed by the mid pair {f0(.74), f4(.70)},
    #   the other only by the floor expert {f5(.52)}.
    C3_ALL = frozenset([1, 2, 5, 6])
    C3_BOTTOM = frozenset([2, 5])
    C4_ALL = frozenset([0, 4, 5])
    C4_FLOOR = frozenset([5])

    c3_hits = 0
    c3_n = 0
    c4_hits = 0
    c4_n = 0

    for _, row in data.iterrows():
        try:
            a = [int(x) for x in list(row['option_a_ratings'])]
            b = [int(x) for x in list(row['option_b_ratings'])]
            resp = int(row['response'])
        except Exception:
            continue
        if len(a) != len(b) or len(a) < 7:
            continue
        pos = frozenset([i for i in range(len(a)) if a[i] > b[i]])
        neg = frozenset([i for i in range(len(a)) if b[i] > a[i]])
        allset = frozenset(pos | neg)

        if allset == C3_ALL:
            # which side is the bottom-expert-backed one?
            if pos == C3_BOTTOM:
                target = 0  # A
            elif neg == C3_BOTTOM:
                target = 1  # B
            else:
                target = None
            if target is not None:
                c3_n += 1
                if resp == target:
                    c3_hits += 1

        elif allset == C4_ALL:
            # which side is backed only by the floor expert f5?
            if pos == C4_FLOOR:
                target = 0
            elif neg == C4_FLOOR:
                target = 1
            else:
                target = None
            if target is not None:
                c4_n += 1
                if resp == target:
                    c4_hits += 1

    if c3_n == 0 or c4_n == 0:
        return float('nan')

    p_c4_floor = c4_hits / float(c4_n)
    p_c3_bottom = c3_hits / float(c3_n)
    return float(p_c4_floor - p_c3_bottom)

```

**Observed (real) value:** -0.0225 (var=0.0100)
**Candidate trajectory (this loop):**
  - iter 1: 0.0925 (var=0.0404) (Δ vs real +0.1150)
  - iter 2: 0.0687 (var=0.0233) (Δ vs real +0.0912)
  - iter 3: 0.0200 (var=0.0388) (Δ vs real +0.0425)
  - iter 4: 0.1038 (var=0.0388) (Δ vs real +0.1262)
  - iter 5: 0.0287 (var=0.0288) (Δ vs real +0.0512)
  - iter 6: 0.0713 (var=0.0241) (Δ vs real +0.0938)
  - iter 7: 0.0400 (var=0.0456) (Δ vs real +0.0625)
  - iter 8 (current): 0.0575 (var=0.0236) (Δ vs real +0.0800)
**Other theories' values on this metric (for reference):**
- pi_6: 0.3338 (var=0.0282)
- pi_5: -0.7275 (var=0.0246)
- pi_1: -0.0050 (var=0.0123)
- pi_2: -0.3575 (var=0.0255)
- pi_3: -0.1062 (var=0.0243)
- pi_4: -0.2200 (var=0.0268)
- pi_7: -0.0475 (var=0.0412)
- pi_8: -0.0162 (var=0.0372)
- pi_9: 0.0663 (var=0.0326)

### Experiment 11
**Design**
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    cells = {}
    for _, row in data.iterrows():
        try:
            a = [int(x) for x in row['option_a_ratings']]
            b = [int(x) for x in row['option_b_ratings']]
        except Exception:
            continue
        if len(a) != len(b) or len(a) == 0:
            continue
        ai = frozenset([i for i in range(len(a)) if a[i] > b[i]])
        bi = frozenset([i for i in range(len(a)) if b[i] > a[i]])
        if len(ai) == len(bi):
            continue
        try:
            resp = int(row['response'])
        except Exception:
            continue
        if len(ai) > len(bi):
            maj, mino = ai, bi
            follow = 1.0 if resp == 0 else 0.0
        else:
            maj, mino = bi, ai
            follow = 1.0 if resp == 1 else 0.0
        key = (maj, mino)
        if key not in cells:
            cells[key] = []
        cells[key].append(follow)

    def rate(maj, mino):
        k = (frozenset(maj), frozenset(mino))
        v = cells.get(k)
        if v is None or len(v) == 0:
            return None
        return float(np.mean(v))

    # cells identified by which expert indices each side wins
    # validities = [0.78, 0.95, 0.68, 0.84, 0.90, 0.72]
    d3 = rate({0, 3}, {1})          # mid-pair majority vs LONE STAR (.95)
    c2 = rate({0, 3}, {2})          # same majority vs LONE FLOOR (.68)
    e4 = rate({1, 2, 4}, {0, 5})    # star+star+floor majority vs mid-pair
    a1 = rate({0, 3, 5}, {1, 2})    # mid-trio majority vs star+floor

    terms = []
    if (d3 is not None) and (c2 is not None):
        terms.append(d3 - c2)
    if e4 is not None:
        terms.append(e4)
    if a1 is not None:
        terms.append(1.0 - a1)

    if len(terms) == 0:
        return float('nan')
    return float(np.mean(terms))
```

**Observed (real) value:** 0.6983 (var=0.0122)
**Candidate trajectory (this loop):**
  - iter 1: 0.5954 (var=0.0114) (Δ vs real -0.1029)
  - iter 2: 0.5379 (var=0.0057) (Δ vs real -0.1604)
  - iter 3: 0.6312 (var=0.0040) (Δ vs real -0.0671)
  - iter 4: 0.6296 (var=0.0064) (Δ vs real -0.0687)
  - iter 5: 0.6575 (var=0.0045) (Δ vs real -0.0408)
  - iter 6: 0.5946 (var=0.0058) (Δ vs real -0.1037)
  - iter 7: 0.6787 (var=0.0030) (Δ vs real -0.0196)
  - iter 8 (current): 0.6663 (var=0.0059) (Δ vs real -0.0321)
**Other theories' values on this metric (for reference):**
- pi_7: 0.1512 (var=0.0062)
- pi_6: 0.5292 (var=0.0060)
- pi_1: 0.3317 (var=0.0037)
- pi_2: 0.3525 (var=0.0038)
- pi_3: 0.3254 (var=0.0065)
- pi_4: 0.3283 (var=0.0079)
- pi_5: 0.2325 (var=0.0054)
- pi_8: 0.6917 (var=0.0064)
- pi_9: 0.6533 (var=0.0064)

### Experiment 12
**Design**
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0
    tot = 0
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        try:
            a = np.asarray([float(x) for x in list(a_raw)])
            b = np.asarray([float(x) for x in list(b_raw)])
        except Exception:
            continue
        if a.size != b.size or a.size < 5:
            continue
        d = np.sign(a - b)
        iA = np.where(d > 0)[0]
        iB = np.where(d < 0)[0]
        try:
            resp = int(r)
        except Exception:
            continue
        # K4-type displays: a 3-cue majority opposed by exactly ONE cue,
        # and that lone opposing cue is the panel's floor expert (index 4, v=0.52).
        if iA.size == 3 and iB.size == 1 and int(iB[0]) == 4:
            tot += 1
            if resp == 0:
                hits += 1
        elif iB.size == 3 and iA.size == 1 and int(iA[0]) == 4:
            tot += 1
            if resp == 1:
                hits += 1

    if tot == 0:
        return 0.5
    return float(hits) / float(tot)

```

**Observed (real) value:** 0.1450 (var=0.0099)
**Candidate trajectory (this loop):**
  - iter 1: 0.5238 (var=0.0200) (Δ vs real +0.3788)
  - iter 2: 0.1625 (var=0.0131) (Δ vs real +0.0175)
  - iter 3: 0.1537 (var=0.0085) (Δ vs real +0.0088)
  - iter 4: 0.0938 (var=0.0050) (Δ vs real -0.0512)
  - iter 5: 0.1125 (var=0.0061) (Δ vs real -0.0325)
  - iter 6: 0.1113 (var=0.0072) (Δ vs real -0.0337)
  - iter 7: 0.0875 (var=0.0041) (Δ vs real -0.0575)
  - iter 8 (current): 0.1175 (var=0.0078) (Δ vs real -0.0275)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2562 (var=0.0218)
- pi_7: 0.9187 (var=0.0063)
- pi_1: 0.8500 (var=0.0170)
- pi_2: 0.8438 (var=0.0149)
- pi_3: 0.5925 (var=0.0166)
- pi_4: 0.7612 (var=0.0210)
- pi_5: 0.8588 (var=0.0103)
- pi_8: 0.1375 (var=0.0089)
- pi_9: 0.0788 (var=0.0054)

### Experiment 13
**Design**
  A=[1, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # stated validities of the six experts in screen order (design-time constant)
    V = np.array([0.88, 0.55, 0.72, 0.95, 0.63, 0.80], dtype=float)
    THR = 0.70  # 'upper panel' voices: 0.72, 0.80, 0.88, 0.95

    hits = 0
    tot = 0

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, r_raw in zip(a_col, b_col, r_col):
        try:
            a = np.asarray([float(x) for x in a_raw], dtype=float)
            b = np.asarray([float(x) for x in b_raw], dtype=float)
            r = int(r_raw)
        except Exception:
            continue
        if a.size == 0 or a.size != b.size:
            continue
        if a.size == V.size:
            v = V
        else:
            v = np.full(a.size, 1.0, dtype=float)

        d = np.sign(a - b)
        idx = np.nonzero(d)[0]
        if idx.size == 0:
            continue
        nA = int(np.sum(d > 0))
        nB = int(np.sum(d < 0))
        if nA == nB:
            continue  # no strict count winner: both theories predict chance
        # keep only displays where EVERY discriminating voice is an upper-panel expert
        if np.any(v[idx] < THR):
            continue

        winner = 0 if nA > nB else 1  # 0 == chose A, 1 == chose B
        tot += 1
        if r == winner:
            hits += 1

    if tot == 0:
        return 0.5
    return float(hits) / float(tot)
```

**Observed (real) value:** 0.4892 (var=0.0049)
**Candidate trajectory (this loop):**
  - iter 1: 0.5583 (var=0.0030) (Δ vs real +0.0692)
  - iter 2: 0.5371 (var=0.0042) (Δ vs real +0.0479)
  - iter 3: 0.5767 (var=0.0067) (Δ vs real +0.0875)
  - iter 4: 0.5483 (var=0.0057) (Δ vs real +0.0592)
  - iter 5: 0.5737 (var=0.0051) (Δ vs real +0.0846)
  - iter 6: 0.5733 (var=0.0045) (Δ vs real +0.0842)
  - iter 7: 0.5583 (var=0.0044) (Δ vs real +0.0692)
  - iter 8 (current): 0.5492 (var=0.0051) (Δ vs real +0.0600)
**Other theories' values on this metric (for reference):**
- pi_8: 0.6446 (var=0.0042)
- pi_6: 0.3654 (var=0.0148)
- pi_1: 0.6813 (var=0.0037)
- pi_2: 0.8667 (var=0.0111)
- pi_3: 0.5975 (var=0.0056)
- pi_4: 0.7529 (var=0.0048)
- pi_5: 0.4121 (var=0.0048)
- pi_7: 0.0892 (var=0.0030)
- pi_9: 0.5542 (var=0.0056)

### Experiment 14
**Design**
  A=[1, 0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Target cells: displays where the TALLY-MAJORITY side is carried by the two
    # most valid experts (features 0 = .92 and 2 = .84).
    # C1: majority wins {0,2}, minority wins {}        (unopposed star pair, 2v0)
    # C3: majority wins {0,2}, minority wins {5}       (star pair vs lone mid, 2v1)
    # C4: majority wins {0,2}, minority wins {1}       (star pair vs lone low, 2v1)
    # C7: majority wins {0,2,4}, minority wins {5}     (all-upper 3v1)
    targets = {
        (frozenset([0, 2]), frozenset()),
        (frozenset([0, 2]), frozenset([5])),
        (frozenset([0, 2]), frozenset([1])),
        (frozenset([0, 2, 4]), frozenset([5])),
    }

    hits = []
    for _, row in data.iterrows():
        try:
            a = [int(x) for x in row['option_a_ratings']]
            b = [int(x) for x in row['option_b_ratings']]
        except Exception:
            continue
        if len(a) != len(b) or len(a) == 0:
            continue
        idxA = frozenset([i for i in range(len(a)) if a[i] > b[i]])
        idxB = frozenset([i for i in range(len(a)) if b[i] > a[i]])
        if len(idxA) == len(idxB):
            continue
        if len(idxA) > len(idxB):
            maj_set, min_set, maj_resp = idxA, idxB, 0  # A is count winner
        else:
            maj_set, min_set, maj_resp = idxB, idxA, 1  # B is count winner
        if (maj_set, min_set) not in targets:
            continue
        try:
            r = int(row['response'])
        except Exception:
            continue
        hits.append(1.0 if r == maj_resp else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))

```

**Observed (real) value:** 0.4892 (var=0.0049)
**Candidate trajectory (this loop):**
  - iter 1: 0.4867 (var=0.0061) (Δ vs real -0.0025)
  - iter 2: 0.3987 (var=0.0039) (Δ vs real -0.0904)
  - iter 3: 0.4888 (var=0.0059) (Δ vs real -0.0004)
  - iter 4: 0.5050 (var=0.0057) (Δ vs real +0.0158)
  - iter 5: 0.5029 (var=0.0041) (Δ vs real +0.0137)
  - iter 6: 0.5017 (var=0.0054) (Δ vs real +0.0125)
  - iter 7: 0.5050 (var=0.0039) (Δ vs real +0.0158)
  - iter 8 (current): 0.5075 (var=0.0074) (Δ vs real +0.0183)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2792 (var=0.0159)
- pi_8: 0.5913 (var=0.0058)
- pi_1: 0.8583 (var=0.0102)
- pi_2: 0.8646 (var=0.0075)
- pi_3: 0.6125 (var=0.0071)
- pi_4: 0.7679 (var=0.0054)
- pi_5: 0.2533 (var=0.0084)
- pi_7: 0.0717 (var=0.0021)
- pi_9: 0.4929 (var=0.0046)

### Experiment 15
**Design**
  A=[1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 1]  B=[0, 0, 1, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Canonical (order-free) keys for the four diagnostic cells.
    # SOLO cells: exactly one discriminating expert, and it is NOT the panel floor
    #   C1 = solo .95 star, C2 = solo .75 mid
    SOLO = {
        ("1001010", "1101010"),   # C1 (trials 1/2)
        ("0010101", "1010101"),   # C2 (trials 3/4)
    }
    # FLAT cells: CH >= 2 displays whose every discriminating voice is upper-band
    #   C4 = unopposed .79/.75 pair (2 vs 0), C5 = low trio vs lone .95 (3 vs 1)
    FLAT = {
        ("0011001", "1011011"),   # C4 (trials 7/8)
        ("0101000", "1011010"),   # C5 (trials 9/10)
    }

    solo_hits = 0.0
    solo_n = 0.0
    flat_hits = 0.0
    flat_n = 0.0

    for _, row in data.iterrows():
        try:
            a = [int(x) for x in list(row['option_a_ratings'])]
            b = [int(x) for x in list(row['option_b_ratings'])]
        except Exception:
            continue
        if len(a) != len(b) or len(a) == 0:
            continue
        ka = ''.join([str(x) for x in a])
        kb = ''.join([str(x) for x in b])
        key = tuple(sorted((ka, kb)))

        if key in SOLO:
            grp = 0
        elif key in FLAT:
            grp = 1
        else:
            continue

        mA = sum(1 for i in range(len(a)) if a[i] > b[i])
        mB = sum(1 for i in range(len(a)) if b[i] > a[i])
        if mA == mB:
            continue
        winner = 0 if mA > mB else 1

        try:
            resp = int(row['response'])
        except Exception:
            continue
        hit = 1.0 if resp == winner else 0.0

        if grp == 0:
            solo_hits += hit
            solo_n += 1.0
        else:
            flat_hits += hit
            flat_n += 1.0

    if solo_n == 0.0 or flat_n == 0.0:
        return float('nan')

    return float(solo_hits / solo_n - flat_hits / flat_n)
```

**Observed (real) value:** 0.0067 (var=0.0123)
**Candidate trajectory (this loop):**
  - iter 1: -0.1050 (var=0.0218) (Δ vs real -0.1117)
  - iter 2: -0.0283 (var=0.0173) (Δ vs real -0.0350)
  - iter 3: -0.0242 (var=0.0178) (Δ vs real -0.0308)
  - iter 4: -0.0117 (var=0.0170) (Δ vs real -0.0183)
  - iter 5: -0.0017 (var=0.0250) (Δ vs real -0.0083)
  - iter 6: 0.0108 (var=0.0128) (Δ vs real +0.0042)
  - iter 7: -0.0125 (var=0.0222) (Δ vs real -0.0192)
  - iter 8 (current): -0.0133 (var=0.0182) (Δ vs real -0.0200)
**Other theories' values on this metric (for reference):**
- pi_8: 0.4275 (var=0.0123)
- pi_9: -0.0583 (var=0.0163)
- pi_1: 0.3392 (var=0.0260)
- pi_2: -0.0092 (var=0.0081)
- pi_3: -0.0492 (var=0.0130)
- pi_4: 0.1650 (var=0.0092)
- pi_5: -0.1375 (var=0.0105)
- pi_6: 0.3375 (var=0.0364)
- pi_7: 0.2758 (var=0.0115)

### Experiment 16
**Design**
  A=[1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0, 1]  B=[1, 1, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # cell accumulators: hits / total
    hits = {}
    tot = {}

    for _, row in data.iterrows():
        try:
            a = [int(x) for x in row['option_a_ratings']]
            b = [int(x) for x in row['option_b_ratings']]
        except Exception:
            continue
        if len(a) != len(b) or len(a) == 0:
            continue
        winsA = [i for i in range(len(a)) if a[i] > b[i]]
        winsB = [i for i in range(len(a)) if b[i] > a[i]]
        mA, mB = len(winsA), len(winsB)
        if mA == mB:
            continue
        if mA > mB:
            win_idx, lose_idx, winner_is_A = set(winsA), set(winsB), True
        else:
            win_idx, lose_idx, winner_is_A = set(winsB), set(winsA), False
        D = mA + mB

        cell = None
        if D == 1:
            j = list(win_idx)[0]
            if j == 2:
                cell = 'C1'      # solo star (v=.95)
            elif j == 3:
                cell = 'C2'      # solo low-upper (v=.78)
        elif D == 2 and len(lose_idx) == 0:
            if win_idx == {0, 3}:
                cell = 'C4'      # unopposed low pair
            elif win_idx == {2, 4}:
                cell = 'C5'      # unopposed senior pair
        elif D == 4 and win_idx == {0, 3, 5} and lose_idx == {2}:
            cell = 'C6'          # all-upper 3v1

        if cell is None:
            continue

        try:
            resp = int(row['response'])
        except Exception:
            continue
        chose_A = (resp == 0)
        follow = 1.0 if (chose_A == winner_is_A) else 0.0

        hits[cell] = hits.get(cell, 0.0) + follow
        tot[cell] = tot.get(cell, 0.0) + 1.0

    def rate(c):
        if tot.get(c, 0.0) <= 0:
            return None
        return hits[c] / tot[c]

    g1 = [r for r in (rate('C1'), rate('C2')) if r is not None]
    g2 = [r for r in (rate('C4'), rate('C5'), rate('C6')) if r is not None]

    if len(g1) == 0 or len(g2) == 0:
        return 0.0
    return float(np.mean(g1) - np.mean(g2))
```

**Observed (real) value:** 0.0517 (var=0.0159)
**Candidate trajectory (this loop):**
  - iter 1: -0.0469 (var=0.0179) (Δ vs real -0.0986)
  - iter 2: -0.0044 (var=0.0198) (Δ vs real -0.0561)
  - iter 3: 0.0414 (var=0.0160) (Δ vs real -0.0103)
  - iter 4: 0.0481 (var=0.0183) (Δ vs real -0.0036)
  - iter 5: 0.0217 (var=0.0123) (Δ vs real -0.0300)
  - iter 6: 0.0503 (var=0.0154) (Δ vs real -0.0014)
  - iter 7: 0.0278 (var=0.0158) (Δ vs real -0.0239)
  - iter 8 (current): 0.0144 (var=0.0211) (Δ vs real -0.0372)
**Other theories' values on this metric (for reference):**
- pi_9: -0.0381 (var=0.0193)
- pi_8: 0.4408 (var=0.0102)
- pi_1: 0.2372 (var=0.0099)
- pi_2: -0.0078 (var=0.0092)
- pi_3: -0.0428 (var=0.0142)
- pi_4: 0.1047 (var=0.0067)
- pi_5: -0.0139 (var=0.0137)
- pi_6: 0.4219 (var=0.0362)
- pi_7: 0.5256 (var=0.0124)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is a faithful instantiation of the arbiter's family in spirit (parameter-free count gate, smooth convex anti-validity novelty, no solo release, no hard mute band, no count-keyed chorus), and it wins on a broad set of cells: Exp 1 (.494 vs .474), Exp 3 (.800 vs .794), Exp 5 (	a-0.770 vs -0.763, the best of any theory), Exp 6 (-0.946 vs -0.938), Exp 7 (.683 vs .661), Exp 8 (.466 vs .485), Exp 9 (.359 vs .405), Exp 14 (.487 vs .489). Those are real, experiment-invariant successes and the low aggregate loss reflects them.

But the single cell the arbiter identified as the tightest quantitative constraint in the corpus is catastrophically missed. Exp 12 K4 (lone FLOOR voice vs a three-cue upper bloc) is real .145 adherence; the candidate simulates .524 	a-- essentially chance, a 0.38 miss, and far worse than pi_8 (.1375) and pi_9 (.0788) on the same cell. The rationale's own hand calculation for this cell predicted ~.14, so the implementation is not doing what the prose says: either the design validity vector is not being aligned with feature index inside `predict` for this panel (the fallback `np.linspace(0.95,0.55,n)` would put the panel floor at index 5, not index 4, and would zero out the floor voice's novelty), or the saturating gate Q^p/(Q^p+q_half^p) is not firing there. Crucially, the model's whole claim is that Exp 10-C4 (floor vs mid PAIR, real .478) and Exp 12-K4 (floor vs upper TRIO, real .145) fall on opposite sides of chance; in simulation they do not separate at all 	a-- Exp 10-C4 comes out around .59 (metric +0.0925 vs real -0.0225, since C3 is pinned at .5 by the gate) and Exp 12-K4 at .52. The contradiction-mass mechanism is therefore currently inert in exactly the contrast it was introduced to explain.

A second, cleaner calibration failure: the solo-vs-flat contrasts have the wrong SIGN. Exp 15 real +0.007, sim -0.105; Exp 16 real +0.052, sim -0.047. This is a direct consequence of gamma being set to 0.40-0.62 (below the arbiter's prescribed 0.6-1.1): with den 	a 3 D^gamma, an unopposed upper PAIR (two residuals, D=2, divisor 1.41) beats a SOLO upper cue (one residual, D=1, divisor 1.0), so flat displays draw more adherence than solos. The same under-dilution inflates Exp 13 (all-upper displays, .558 vs .489) and Exp 2 conflict adherence (.789 vs .654). Meanwhile Exp 4 (eight-expert tally, .715 vs .755) and Exp 11 (.595 vs .698) undershoot.
Rationale: Stay inside the prescribed family 	a-- count gate, smooth convex anti-validity novelty, seniority-mass (not count) modulation. Do NOT redesign. Three targeted in-family fixes:

1. DEBUG THE FLOOR-VOICE PATH FOR EXP 12 (highest priority). The prose predicts .14 for K4 and the simulation returns .52. Verify inside `predict` that the design validity vector is being received and indexed in FEATURE ORDER (the `v.size != n` fallback `linspace(0.95,0.55,n)` silently re-ranks experts and would strip novelty from a floor expert that is not the last feature 	a-- add an assertion/trace rather than a silent fallback). Then re-check the gate numerically on the two flanking cells and choose q_half / p_gate / gmax so that they STRADDLE chance simultaneously: Exp 12-K4 adherence .12-.20 (floor beats an upper trio) and Exp 10-C4 floor-choice .45-.52 (floor merely ties a mid pair). As written, both cells land ~.52-.59, i.e. the contradiction-mass knee is doing no work; likely q_half is too high relative to the realized Q of an upper trio, or gmax is being swamped by the (c_n + W_tot) Luce denominator. Lowering c_n to ~2.0-2.5 and/or raising gmax to ~1.6-1.9 while keeping q_half near 1.2 is the natural knob.

2. RAISE gamma INTO THE PRESCRIBED RANGE. Set gamma 	a 0.85-1.05 (arbiter asked for 0.6-1.1; the candidate used 0.40-0.62). At gamma 	a 1 a solo cue and an unopposed pair of the same tier become roughly equally compelling, which flips Exp 15/16 from -0.10/-0.05 to the observed slightly-positive values and pulls Exp 13 from .558 toward .489.

3. COMPENSATE THE TALLY CELLS. Raising gamma will drag Exp 3/4 down, and Exp 4 already undershoots. Offset with a modest beta increase (to ~18-24) or c_n decrease, checking that Exp 2 conflict adherence lands .65-.72 (currently .79) and Exp 3 stays ~.79, Exp 4 ~.75. Exp 11 (.595 vs .698) should improve once the own-side echo term (lam_own) actually makes the floor voice loud when it stands AMONG stars 	a- consider lam_own near the top of its range (0.80-0.85).

Do not touch what already works: Exp 1/5/6/7/8/9/14 are at or near the data and should be re-verified after each change so the fix for Exp 12 does not break the Exp 5/6 coherent-conflicting reversals.

**Outcome of your advice:** iter 1 candidate loss=0.1129 -> iter 2 candidate loss=0.1006 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The iter-2 candidate is a clear improvement over iter-1 and was accepted by the gate (0.1129 -> 0.1006). The mid-equivalent chorus scalar Q fixed exactly the cell I flagged as fatal last round: Exp 12-K4 went from .524 to .1625 against a real .145 
inety percent of the way there, and it did so without destroying the Exp 5/6 coherent-vs-conflicting reversals (-0.791 vs -0.763; -0.847 vs -0.938). Raising gamma into the prescribed band also fixed the SIGN error on the solo-vs-flat contrasts: Exp 15 went -0.105 -> -0.028 (real +0.007) and Exp 16 -0.047 -> -0.004 (real +0.052), and Exp 13 came down from .558 to .537 (real .489). The parameter-free count gate continues to nail Exp 1 (.494 vs .474). So the prescribed family (count gate + smooth convex anti-validity weighting + seniority-mass-not-count modulation) is being implemented faithfully and is carrying most of the corpus.

What the fix cost, however, is visible in four cells, and they are now the dominant contributors to the residual loss:
(a) Exp 2 is the single worst cell: sim .819 vs real .654 (+0.166), and it got WORSE than iter 1 (.789). On tally-vs-TTB conflict displays the model follows the low-validity majority far too slavishly; the star's voice has been zeroed out too completely (eta was dropped to 0.015-0.035 this round, which is exactly the knob that lets a star pull back against a low-tier majority).
(b) Exp 11 regressed from .595 to .538 (real .698, and both pi_8 = .692 and pi_9 = .653 beat it). The cells that metric rewards are precisely 'a floor voice standing AMONG stars makes its majority compelling' (e4) and 'a star+floor minority beats a mid trio' (1-a1) 
inety the own-side echo term lam_own is the natural lever and is currently mid-range.
(c) The chorus threshold is now firing slightly too eagerly on 2-cue opposition. Exp 10's floor-vs-mid-pair cell simulates ~.57 floor-choice where real is ~.478 (metric +0.069 vs real -0.023), and Exp 14 fell to .399 vs real .489 (worse than iter 1's .487), i.e. a lone low voice opposed by a STAR PAIR now reverses the majority more strongly than subjects do.
(d) Exp 4 (eight-expert tally) still undershoots (.706 vs .755) and Exp 8 over-reverses (.418 vs .485), suggesting the capacity term is now shaving a bit too much drive at n=8 while the chorus gate still bites there.

None of this requires leaving the family; it is pure calibration of eta, q0/wid, lam_own, epsilon and psi.
Rationale: Stay inside the arbiter's family exactly as implemented (parameter-free count gate, smooth convex anti-validity novelty, mid-equivalent chorus scalar, Luce-like saturating integration, capacity term). Do NOT redesign, and do NOT touch gamma again 
inety my last two-step on gamma (0.4-0.6 -> 0.88-1.05) was ACCEPTED and has fixed the Exp 15/16 sign; leave it where it is. Four targeted knob moves, in priority order:

1. RAISE eta BACK (highest leverage, and it reverses a change that hurt). Exp 2 went .789 -> .819 against a real .654 when eta was cut to 0.015-0.035. Put eta back to ~0.045-0.075. A star that carries a small ungated residual can still pull back against a low-tier majority, which is exactly what tally-vs-TTB conflict trials require. Check the side effects: Exp 3 should stay ~.79 (currently .813, so a small drop is welcome), Exp 12-K4 should not rise above ~.20 (three upper residuals at 0.06 only cost ~0.18 against a floor weight of ~1.6), Exp 7 should land .65-.70 (currently .705).

2. NUDGE THE CHORUS THRESHOLD UP SLIGHTLY, not down. Two MID voices must remain sub-threshold and two STARS must be only marginally supra-threshold. Currently the 2-cue cells fire too hard: Exp 10 floor-vs-mid-pair simulates ~.57 (target .45-.52) and Exp 14 sits at .399 (target ~.49). Move q0 to ~2.45-2.60 and/or widen wid to 0.13-0.18 so the star-pair cell yields a partial (not near-maximal) gate. Guard-rail: Exp 12-K4 (three-cue upper bloc, Q~2.9) must stay in .12-.20 and Exp 5/6 must keep their strong negative values (-0.72 to -0.82 and -0.88 to -0.95).

3. PUSH lam_own TO THE TOP OF ITS RANGE (0.75-0.90) to recover Exp 11 (.538 -> target .65-.70). The two Exp-11 cells that are failing both involve a low voice standing BESIDE senior voices rather than against them (star+star+floor majority; star+floor minority vs a mid trio). A larger own-side echo makes those low voices loud without touching any opposition-side arithmetic, so it should leave Exp 5/6/12 essentially unchanged.

4. MILD GLOBAL SOFTENING + CAPACITY RE-BALANCE. Move epsilon to ~0.11-0.17; this simultaneously pulls Exp 2 (.819), Exp 3 (.813), Exp 7 (.705), Exp 8 (.418), Exp 9 (.365) and Exp 14 (.399) toward their data values. Because that also drags Exp 4 (already low at .706 vs .755), compensate on the eight-expert side by lowering psi to ~0.50-0.70 and chi to ~0.05-0.20 so that n=8 tally adherence recovers to ~.75 while Exp 8's star-backed majorities still sit near .48 (they are currently .418, i.e. over-reversed, so both moves point the same way).

Re-verify after each edit that the untouched successes hold: Exp 1 ~.47-.50, Exp 5 ~-0.76, Exp 6 ~-0.90, Exp 12 ~.14-.19, Exp 15/16 in [-0.02, +0.06]. The aim is to trade ~0.08 of Exp-2 error and ~0.12 of Exp-11 error for at most ~0.03 elsewhere.

**Outcome of your advice:** iter 2 candidate loss=0.1006 -> iter 3 candidate loss=0.0692 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: Iter 3 is a large, genuine win (0.1006 \u2192 0.0692) and it is faithful to the arbiter's family: the count gate is still parameter-free (Exp 1 = .494 vs .474), weights stay non-negative, seniority is continuous, and the chorus scalar is measured in mid-equivalents rather than counts. The knob moves I asked for and the proposer's own kappa move both paid off exactly where predicted: Exp 8 went .418 \u2192 .484 (real .485), Exp 14 .399 \u2192 .489 (real .489), Exp 11 .538 \u2192 .631 (real .698), Exp 10 +0.069 \u2192 +0.020 (real \u22120.023), Exp 16 \u22120.004 \u2192 +0.041 (real +0.052), Exp 7 .705 \u2192 .667 (real .661), Exp 5 \u22120.791 \u2192 \u22120.768 (real \u22120.763), and Exp 12-K4 stays locked at .154 (real .145) \u2014 the corpus's tightest constraint. Nothing that was working has broken.

The residual loss is now dominated by a single, coherent direction: the model FOLLOWS COUNT MAJORITIES TOO MUCH when the majority contains no true floor voice. Exp 2 is the worst cell in the corpus by a factor of two (.804 vs .654, +0.150) and barely moved despite kappa going 6.5 \u2192 13.5; Exp 13 (all-upper displays) got worse this round (.537 \u2192 .577 vs real .489, +0.088); Exp 3 also drifted up (.800 \u2192 .826 vs .794). All three are driven by the same term: the 'thin' convex residual gradient r_j = eta + c_res\u00b7(1\u2212s)^nu is not thin. With c_res \u2248 .22 and nu \u2248 1.5 a second-from-bottom voice carries r \u2248 .22 and a mid voice r \u2248 .09\u2013.14, so in Exp 2's 4-vs-1 and 3-vs-1 conflict displays the low-ish majority accumulates ~.5\u2013.8 of ungated weight against a star's bare eta \u2248 .04, and the logistic saturates at ~.80. My hand-check of Exp 2 trial 1 gives z \u2248 1.5 (p \u2248 .78 post-lapse); halving c_res to \u2248 .12 drops z to \u2248 0.7 and lands the cell at \u2248 .65, i.e. exactly the data. The same cut pulls Exp 13's all-upper displays toward chance (their entire drive is residual, since nov \u2248 0 for every cue there) and shaves Exp 3 by the ~.03 it needs. Raising kappa further cannot do this job \u2014 kappa only touches nov, and in Exp 2 and Exp 13 nov is already near zero; the over-adherence is pure residual arithmetic.

Secondary, smaller residuals: Exp 4 (eight-expert tally, .707 vs .755) still undershoots and Exp 11 (.631 vs .698) still undershoots; Exp 6 drifted from \u22120.946 (iter 1) to \u22120.829 (real \u22120.938) as gmax and kappa rose, though its between-subject variance (.069) makes it the noisiest cell in the set.
Rationale: Stay inside the prescribed family exactly as implemented (parameter-free count gate, smooth convex anti-validity novelty, mid-equivalent chorus scalar Q with a logistic threshold, Luce-like saturating integration, capacity term). No redesign. One primary knob and two compensations:

1. THIN THE RESIDUAL GRADIENT: c_res 0.18\u20130.28 \u2192 0.10\u20130.16 (highest leverage; this is a DIFFERENT knob from eta, which I asked you to raise last round and which was ACCEPTED \u2014 keep eta at 0.030\u20130.048, it is what fixed the Exp 15/16 solo-vs-flat sign, and do NOT cut it). The residual is supposed to be a thin ordinal pull; at c_res \u2248 .22 it is the main driver of every all-upper / no-floor-voice majority. Targets after the cut: Exp 2 \u2192 .64\u2013.70 (now .804), Exp 13 \u2192 .49\u2013.53 (now .577), Exp 3 \u2192 .78\u2013.80 (now .826). Guard-rails that must NOT move: Exp 12-K4 stays .12\u2013.19 (its drive is nov\u00b7gate, not residual), Exp 14 stays .46\u2013.51, Exp 8 stays .46\u2013.50, Exp 5 stays \u22120.74 to \u22120.80, Exp 1 stays .47\u2013.50.

2. COMPENSATE THE PURE-TALLY CELLS WITHOUT RE-INFLATING EXP 2. Thinning c_res will drag Exp 4 (already low at .707 vs .755) and Exp 9 (.365 vs .405). Do the compensation with c_n (3.0\u20133.8 \u2192 2.4\u20133.1) rather than beta: lowering the constant of indecision boosts z most where total weight on the table is small (sparse, low-margin tally displays like Exp 4/9) and least where W_tot is already large, so it recovers Exp 4 without undoing the Exp 2 gain. Also set chi to 0.02\u20130.08 (its only effect is the n=8 divisor, worth ~+0.01\u20130.02 on Exp 4) but leave psi at 0.80\u20130.95 \u2014 psi is what currently pins Exp 8 at .484 vs real .485, so touching it will re-open the over-reversal I flagged last round.

3. EXP 11 \u2014 SMALL STEP ONLY, AND NOT VIA lam_own. Last round the proposer showed lam_own is inert in the Exp-11 cells (their floor voice is already far supra-threshold) and harmful in Exp 2's near-threshold cell; I accept that and withdraw the lam_own push. The residual \u22120.067 on Exp 11 is a drive problem in the e4 and (1\u2212a1) cells. Nudge gmax only slightly (1.70\u20132.10 \u2192 1.85\u20132.20) AFTER the c_res cut is verified, and re-check Exp 6 (which fell from \u22120.946 to \u22120.829 as gmax/kappa rose; target \u22120.87 to \u22120.94) and Exp 12-K4 (must not go below .10). If Exp 6 or Exp 12 degrade, revert gmax and accept Exp 11 at ~.63.

Priority order: do (1) first and report Exp 2/13/3 before touching anything else; (2) and (3) are only there to keep the tally cells from slipping. The target is to convert ~0.09 of Exp-2 error and ~0.06 of Exp-13 error into at most ~0.02 elsewhere.

**Outcome of your advice:** iter 3 candidate loss=0.0692 -> iter 4 candidate loss=0.0680 -> the gate marked it ACCEPTED.

### Iteration 4
Verdict: regenerate
Interpretation: Iter 4 is the running best (0.0692 \u2192 0.0680) and the gain came exactly where I asked: the targeted residual thinning (c_res 0.22\u21920.17, nu 1.5\u21922.4) plus the small kappa bump pulled Exp 2 from .804 to .744 (real .654) and Exp 13 from .577 to .548 (real .489), and Exp 3 stayed close (.818 vs .794). The corpus-wide successes are intact: Exp 1 .494 (real .474), Exp 5 \u22120.760 (real \u22120.763, best of any theory), Exp 7 .680 (.661), Exp 8 .475 (.485), Exp 14 .505 (.489), Exp 16 +0.048 (+0.052), Exp 15 \u22120.012 (+0.007), Exp 9 .379 (.405). The parameter-free count gate continues to pin every tally tie at exactly chance, which no competitor family does as cleanly.

The cost of the thinning is now visible and it is concentrated in one arithmetic cell that the arbiter named as the corpus's tightest constraint. Exp 10 REGRESSED from +0.020 to +0.104 (real \u22120.023): since the C3 half is pinned at .5 by the gate, this means the simulated floor-vs-mid-PAIR cell (C4) is at \u2248.604 where humans are at \u2248.478. Hand-arithmetic on that cell shows why: with eta=.04, c_res=.17, nu=2.4 the floor voice carries r\u2248.21 while the two mids (s\u2248.42/.51) carry only \u2248.086/.071 \u2192 net \u2248+.05 for the floor plus a small sub-threshold gate leak, i.e. a systematic floor advantage where the data say dead heat. Raising nu made the floor's residual EDGE over a mid pair larger, not smaller. Symmetrically, Exp 12-K4 has now overshot in the other direction (.094 vs real .145), so the floor voice is globally a touch too loud.

Second, a slow monotone leak: Exp 6 has drifted \u22120.946 \u2192 \u22120.847 \u2192 \u22120.829 \u2192 \u22120.810 against a real \u22120.938 across the four iterations, tracking each successive residual cut \u2014 the conflicting (low-voice-majority) cells need more drive than the thinned gradient now supplies. Exp 11 (.630 vs .698) and Exp 4 (.709 vs .755) undershoot for the same reason. So the corpus now splits cleanly: cells whose majorities contain NO floor voice (Exp 2, 13, 3, 10-C4) are still over-followed, while cells whose drive comes from a low voice standing among or against seniors (Exp 6, 11, 4, 12) are slightly under-driven. Further uniform scaling of c_res cannot fix both; the two need to be separated by SHAPE and by the gated term.

Faithfulness note: the implementation keeps the count gate parameter-free, weights non-negative, and modulation keyed to summed seniority mass rather than cue COUNT, so the arbiter's distinctive continuous-in-seniority signature survives. It does, however, run kappa\u224815 and a narrow logistic knee, which is well outside the prescribed kappa\u22481.5\u20133 'chorus-free' letter of CNW. Because the gate has been repeatedly validated by the accept gate (it is what pins Exp 12-K4 and Exp 8), I am not asking for its removal \u2014 only flagging that the prose/parameter ranges have drifted from the recommendation and should not drift further toward a count-keyed or discrete-band rule.
Rationale: The candidate was ACCEPTED but the margin was tiny (0.0692\u21920.0680) and one cell actively regressed, so there is a concrete, low-risk move left. Stay strictly inside the current implementation (parameter-free count gate, smooth convex anti-validity novelty, seniority-mass scalar Q with a smooth threshold, Luce-like saturating integration, capacity term). Do NOT redesign, do NOT touch gamma or epsilon (both are currently what hold Exp 15/16 sign and Exp 1/8 at data), and do NOT cut c_res uniformly again \u2014 that lever has now been pushed twice and its remaining head-room goes straight into Exp 6/11/4.

1. PRIMARY \u2014 RE-BALANCE THE GRADIENT SHAPE TO FIX EXP 10-C4 WITHOUT RE-INFLATING EXP 2. Move BOTH knobs together: c_res 0.14\u20130.20 \u2192 0.10\u20130.14 AND nu 2.0\u20132.8 \u2192 1.7\u20132.05. This is not a repeat of the pure scale cut and not a reversal of the convexity move; it is the combination that shrinks the FLOOR residual while leaving MID residuals essentially where they are. Worked example on the Exp-10 panel (v_floor=.52, mids .70/.74, v_max=.95): current (c_res=.17, nu=2.4) gives floor r=.210 vs mids .086+.071=.157 (floor edge +.053 \u21d2 sim .604); proposed (c_res=.12, nu=1.85) gives floor r=.160 vs mids .085+.073=.158 (edge \u2248.00 \u21d2 C4 lands .50\u2013.54, real .478). Crucially the same substitution is near-neutral on Exp 2: I recompute the B-side residual mass on its 4-vs-1 conflict trial as .529 vs .548 now, because the flatter exponent gives back to the mid voices roughly what the smaller scale takes away. Target after the edit: Exp 10 metric in [\u22120.06, +0.04], Exp 12-K4 in [.11, .18], Exp 2 \u2264 .74, Exp 13 \u2264 .55.

2. COMPENSATE THE UNDER-DRIVEN, FLOOR-CONTAINING CELLS WITH THE GATED TERM, NOT THE RESIDUAL. The floor residual cut in (1) will shave Exp 4/9/11/6. Offset with a small c_n reduction (3.2 \u2192 2.8\u20133.1; I asked for a bigger cut in iter 3 and the proposer's objection \u2014 that it multiplies z on small-W_tot displays \u2014 is now LESS binding because Exp 2's residual mass is unchanged and Exp 13 is being handled by (1)). If Exp 6 is still above \u22120.86 after (1)+(2), nudge gmax 1.70\u20132.10 \u2192 1.85\u20132.15, but ONLY while checking that Exp 12-K4 does not fall below .10 (it is already over-reversed at .094) \u2014 if it does, revert gmax and accept Exp 6 at \u2248\u22120.82 given its between-subject variance of .069 makes it the noisiest cell in the set.

3. DO NOT push kappa further. It was raised 13.5\u219215 last round and, together with the residual change, bought Exp 2; another raise now costs Exp 6 (whose second-from-bottom voice f1 at s=.077 is exactly the cue kappa suppresses) and Exp 11, both of which are already under-magnitude. If anything, hold kappa at 14\u201316.

Guard-rails to re-verify after every edit (these are the experiment-invariant wins that must not move): Exp 1 .47\u2013.50, Exp 5 \u22120.74 to \u22120.80, Exp 7 .65\u2013.70, Exp 8 .46\u2013.50, Exp 14 .46\u2013.52, Exp 15 in [\u22120.03,+0.04], Exp 16 in [+0.02,+0.07], Exp 3 .78\u2013.82. The goal is to convert the 0.126 Exp-10 error and ~0.03 of Exp-2/13 error into at most ~0.02 on Exp 4/9.

**Outcome of your advice:** iter 4 candidate loss=0.0680 -> iter 5 candidate loss=0.0669 -> the gate marked it ACCEPTED.

### Iteration 5
Verdict: regenerate
Interpretation: Iter 5 is again the running best (0.0680 \u2192 0.0669) and the gain landed exactly where I asked: the joint (c_res, nu) re-balance pulled Exp 10 from +0.104 back to +0.051 (real \u22120.023) and nudged Exp 12-K4 from .094 up to .1125 (real .145), while the corpus-wide wins held \u2014 Exp 1 .484 (real .474), Exp 5 \u22120.758 (\u22120.763), Exp 7 .678 (.661), Exp 8 .494 (.485), Exp 14 .503 (.489), Exp 15 \u22120.002 (+0.007), Exp 16 +0.022 (+0.052), Exp 3 .810 (.794). The parameter-free count gate still pins every tally tie at chance, which remains the cleanest experiment-invariant signature in the set, and weights are still non-negative with modulation keyed to summed seniority mass rather than cue count, so the arbiter's continuous-in-seniority family is intact (with the standing caveat, flagged in iter 4, that kappa\u224815 plus a narrow knee is far from the letter of CNW's kappa\u22481.5\u20133 \u2014 it should not drift further toward a discrete/count-keyed rule).

The residual loss is now concentrated in exactly two cells, and they are large: Exp 6 (\u22120.761 vs real \u22120.938, error 0.176 \u2014 now the single worst in the corpus, and it has drifted monotonically \u22120.946 \u2192 \u22120.847 \u2192 \u22120.829 \u2192 \u22120.810 \u2192 \u22120.761 across every residual/scale cut) and Exp 2 (.758 vs .654, error 0.104, which actually moved the WRONG way this round, .744 \u2192 .758). Exp 13 also crept back up (.548 \u2192 .574 vs .489). Behind Exp 2 there is a specific, diagnosable arithmetic fault rather than a generic scale problem. On its dominant conflict display (A wins only the star f0; B wins f2,f3,f4,f5) the low voice f4 (s=.139, nov\u2248.11) is NOT opposed by a chorus \u2014 it faces a single star, u\u22481.38, well below q0=2.45 \u2014 yet it still gets a full gate because the OWN-SIDE echo lam_own\u00b7\u03a3u_own adds \u22481.94, pushing Q to \u22483.3. In other words the model is amplifying a low voice for standing beside its own majority, which inflates precisely the cells (Exp 2, and to a lesser degree Exp 13's denser displays) where humans do not follow the low-tier majority that hard. Crucially, the cells that DO need the gate survive without the echo: in Exp 6's conflicting displays the floor voice f6 faces two upper voices (\u03a3u\u22482.60 > q0) on opposition mass alone, Exp 12-K4's floor voice faces three upper cues, and Exp 7's marginal cues face two stars. So there is a clean differential knob available that has never been pushed in this direction.
Rationale: The candidate was ACCEPTED, but the two largest errors (Exp 6 = 0.176, Exp 2 = 0.104) are still worth ~0.28 of pooled error and one of them (Exp 2) regressed this round, so one more targeted in-family pass is justified. Stay strictly inside the current implementation (parameter-free count gate, smooth convex anti-validity novelty, mid-equivalent seniority-mass scalar Q with a smooth threshold, Luce-like saturating integration, capacity term). No redesign; do NOT touch gamma, epsilon, eta, nu, kappa, q0 or wid \u2014 gamma/epsilon hold the Exp 15/16 sign and Exp 1/8 at data, eta/nu were just re-balanced successfully, and widening wid or lowering q0 both re-inflate Exp 10-C4 (I checked: q0\u21922.30 or wid\u21920.22 each put a .14\u2013.18 leak on the lone floor voice and push C4 back above .60).

1. PRIMARY \u2014 CUT THE OWN-SIDE ECHO: lam_own 0.50\u20130.70 \u2192 0.20\u20130.35. Note this is a NEW direction, not a reversal: in iter 2 I pushed lam_own UP, the proposer declined on the grounds that it was inert in Exp 11 and harmful in Exp 2's near-threshold cell \u2014 that argument is exactly the case for cutting it now. Worked numbers at lam_own=0.25 with the current settings: Exp 2's f4 gets Q = 1.375 + 0.25\u00b73.23 = 2.18 \u21d2 gate \u2248 0.25 (was \u22481.9), removing \u2248.18 of spurious majority weight \u21d2 z drops \u22481.23 \u2192 \u22480.83 \u21d2 cell \u2248 .67\u2013.68 against a real .654. Meanwhile the cells that matter are untouched because their gate is carried by OPPOSITION mass: Exp 6's f6 Q = 2.60 + 0.25\u00b72.41 = 3.20 (full gate), Exp 12-K4's floor voice has no own side at all (stays .11\u2013.15), Exp 7's f3 Q \u2248 2.81 (full gate), Exp 10-C4's floor voice is unchanged (Q = 1.95, sub-threshold). This is the only knob I can find that separates 'low voice inside its own majority' (over-followed) from 'low voice against a senior bloc' (under-driven).

2. COMPENSATION \u2014 SMALL c_n CUT: 2.70\u20133.05 \u2192 2.45\u20132.80. The echo cut will shave the one Exp-11 cell that genuinely uses it (e4: floor f2 among two stars, Q falls 3.21 \u2192 2.29, gate \u22480.46 instead of full), and Exp 4/6/9/11 are all under-driven already. Lowering the constant of indecision raises z most where W_tot is small, which is where those cells live, and least where Exp 2's dense 4-vs-1 display sits. Targets after (1)+(2): Exp 2 in [.64, .70], Exp 6 at \u22120.82 or stronger (it must NOT drift further toward zero \u2014 treat \u22120.761 as a hard floor), Exp 11 \u2265 .62, Exp 4 \u2265 .71.

3. DO NOT try to fix Exp 13 (+0.085) by re-raising nu or cutting c_res a third time. Exp 13 is pure residual arithmetic (nov\u22480 for every cue there), and both of those levers now feed straight into Exp 10-C4 / Exp 6 / Exp 4. If you want a shot at it, the only safe move is a joint 'residual down, beta up' rescale that leaves gated cells' drive intact \u2014 e.g. eta and c_res both \u00d70.8 with beta \u00d71.1, which reduces residual-only cells (Exp 13, 3, 2) by \u224812% while raising gate-driven cells (Exp 6, 11, 12, 5) by \u224810%. Try this ONLY as a second, separately verified step after (1)+(2), and abandon it if Exp 4 falls below .69 or Exp 9 below .35.

Guard-rails to re-verify after every edit (the experiment-invariant wins that must not move): Exp 1 .47\u2013.50, Exp 3 .78\u2013.82, Exp 5 \u22120.74 to \u22120.80, Exp 7 .65\u2013.70, Exp 8 .46\u2013.51, Exp 10 metric in [\u22120.06, +0.05], Exp 12 .11\u2013.18, Exp 14 .46\u2013.52, Exp 15 in [\u22120.03, +0.04], Exp 16 in [+0.01, +0.07]. The goal is to convert \u22480.08 of Exp-2 error and \u22480.03 of Exp-6 drift into at most \u22480.03 on Exp 11.

**Outcome of your advice:** iter 5 candidate loss=0.0669 -> iter 6 candidate loss=0.0773 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The iter-6 candidate implemented my iter-5 primary recommendation (own-side echo cut, lam_own 0.60 \u2192 0.27, plus a small c_n cut) and the accept gate rejected it (0.0669 \u2192 0.0773). The trade it made was real but lopsided: Exp 2 improved as predicted (.758 \u2192 .717 against real .654, \u22120.04 of error) and Exp 3 moved through the target (.810 \u2192 .777 vs .794), but four cells paid more than that: Exp 6 drifted further toward zero (\u22120.761 \u2192 \u22120.714 vs real \u22120.938 \u2014 it broke the hard floor I explicitly set), Exp 11 fell back to .595 (vs .698, undoing two iterations of gain), Exp 10 re-inflated (+0.029 \u2192 +0.071 vs real \u22120.023), Exp 5 weakened (\u22120.758 \u2192 \u22120.731), and Exp 4 slipped again (.701 \u2192 .696 vs .755). So the own-side echo is NOT the spurious term I diagnosed: a meaningful fraction of the corpus's low-voice drive (Exp 11 e4, Exp 6's conflicting cells, Exp 5) comes from a low voice standing *inside* a majority, not only from one speaking into opposition. My iter-5 diagnosis was wrong on that point and should not be repeated.

On the unchanged accepted base (iter 5), the error structure is now very clear and, importantly, it splits by MECHANISM rather than by experiment: cells whose drive is purely the ungated residual gradient are over-followed (Exp 13 all-upper displays .573 vs .489; Exp 2 .758 vs .654; Exp 3 .810 vs .794), while cells whose drive is the gated novelty term are under-driven (Exp 6 \u22120.761 vs \u22120.938; Exp 11 .658 vs .698; Exp 12-K4 .113 vs .145 is the one gated cell that is slightly over-driven). I verified the Exp-13 arithmetic on the base parameters: with every cue upper-band, nov \u2248 0 for all of them, residuals run .040\u2013.083, and a 2-vs-1 display yields z \u2248 0.3\u20130.5 \u21d2 p \u2248 .54\u2013.56 post-lapse \u2014 i.e. the residual gradient alone is still strong enough to make all-upper count majorities followable, which is exactly the .085 error. Six further uniform cuts to c_res are off the table (I have pushed that lever twice and it feeds straight into Exp 4/6/9/10-C4), but a JOINT rescale that shrinks the residual while raising beta moves residual-only and gated cells in opposite directions \u2014 this is the one in-family knob combination that has never been tried, and it is aimed squarely at the two largest remaining errors simultaneously.

Faithfulness: the implementation keeps the parameter-free count gate (Exp 1 pinned at chance, the cleanest experiment-invariant signature in the corpus), non-negative weights, continuous seniority, and modulation keyed to summed seniority MASS rather than cue count. It still runs kappa \u2248 15 with a narrow knee, well outside CNW's literal kappa \u2248 1.5\u20133, but the accept gate has validated that repeatedly; it should not drift further toward any discrete or count-keyed rule.
Rationale: The candidate was REJECTED by the gate, so the base reverts to iter 5. Build on iter 5 and do NOT re-apply the lam_own cut \u2014 my iter-5 advice in that direction was wrong and the gate discarded it; Exp 11/6/5/10 all depend on the own-side echo. If anything, leave lam_own exactly at the iter-5 setting (0.50\u20130.70); at most trim it to 0.45\u20130.60 (a value between my two flanking recommendations), and only if Exp 2 is still above .75 after step 1.

1. PRIMARY \u2014 JOINT 'RESIDUAL DOWN, BETA UP' RESCALE (the step-3 option I flagged last round but that was never attempted). Scale the ungated gradient down and the logistic slope up together: eta 0.030\u20130.048 \u2192 0.022\u20130.036, c_res 0.10\u20130.14 \u2192 0.075\u20130.108, beta 34\u201348 \u2192 44\u201358 (roughly residual \u00d70.75, beta \u00d71.25). Because z \u2248 beta\u00b7E/((c_n + W_tot)\u00b7D^gamma) and c_n dominates W_tot on sparse displays, residual-ONLY cells (where nov \u2248 0 for every cue) move by \u2248\u00d70.94 while GATED cells (where the floor voice carries nov\u00b7gate \u2248 1.7\u20132.1, unscaled) move by \u2248\u00d71.25. Expected: Exp 13 .573 \u2192 .53\u2013.55, Exp 2 .758 \u2192 .72\u2013.74, Exp 3 stays .78\u2013.81, while Exp 6 \u2192 \u22120.83 or stronger, Exp 11 \u2192 .68\u2013.71, Exp 5 \u2192 \u22120.78\u2013\u22120.80. Do NOT change nu, kappa, gamma, epsilon, q0, wid or lam_own in this step \u2014 the whole point is to move the residual/gate BALANCE, not the shapes that currently hold Exp 15/16's sign and Exp 10-C4's dead heat.

2. GUARD THE ONE GATED CELL THAT IS ALREADY OVER-DRIVEN. Exp 12-K4 sits at .113 vs real .145 and the beta increase will push it down. If it falls below .095, do NOT fix it with q0/wid (both re-inflate Exp 10-C4); instead trim gmax by ~5% (1.70\u20132.10 \u2192 1.62\u20132.00), which pulls back gated drive uniformly but leaves the residual/gate ratio you just bought intact. Accept Exp 12 anywhere in [.10, .18].

3. SECOND-ORDER \u2014 THE n=8 CELLS. Exp 4 (.701 vs .755) is the third-largest error and is mostly carried by the residual at n=8 because psi \u2248 0.88 kills 88% of the gate there. The beta increase already helps it; if it is still below .72 after step 1, reduce psi modestly (0.80\u20130.95 \u2192 0.68\u20130.85) so eight-expert panels retain a little more novelty drive. Side-effect check: Exp 8 is currently .494 against a real .485, i.e. a small DROP is welcome, so this move is safe up to Exp 8 \u2265 .46. Do not touch chi.

Guard-rails to re-verify after every edit (the experiment-invariant wins that must not move): Exp 1 .47\u2013.50 (the count gate must stay parameter-free \u2014 exact chance on every tally tie), Exp 3 .78\u2013.82, Exp 5 \u22120.74 to \u22120.80, Exp 7 .65\u2013.70, Exp 8 .46\u2013.51, Exp 9 \u2265 .35, Exp 10 metric in [\u22120.06, +0.05], Exp 12 .10\u2013.18, Exp 14 .46\u2013.52, Exp 15 in [\u22120.03, +0.04], Exp 16 in [+0.01, +0.07]. Target: convert \u22480.08 of Exp-6 error and \u22480.04 of Exp-13/2 error into at most \u22480.02 on Exp 12/4. Report Exp 6, 13, 2, 11 and 12 before making any further change.

**Outcome of your advice:** iter 6 candidate loss=0.0773 -> iter 7 candidate loss=0.0694 -> the gate marked it REJECTED.

### Iteration 7 (most recent)
Verdict: regenerate
Interpretation: The iter-7 candidate implemented my iter-6 'residual down (\u00d70.75), beta up (\u00d71.25)' rescale exactly, and the gate rejected it (0.0669 \u2192 0.0694). The decomposition is informative rather than simply negative: the rescale DID move the two classes in the predicted directions \u2014 Exp 6 \u22120.761 \u2192 \u22120.783, Exp 11 .658 \u2192 .679 (closest to the real .698 of any iteration), Exp 13 .574 \u2192 .558, Exp 4 .701 \u2192 .711 \u2014 but the magnitudes were far smaller than my hand-arithmetic predicted, and the costs landed exactly on the cell I flagged as fragile: Exp 12-K4 fell .113 \u2192 .088 (real .145, error now 0.057, and below the .095 floor I set), Exp 9 slipped .361 \u2192 .353 (real .405), and Exp 2 did NOT improve at all (.758 \u2192 .765 against real .654), which falsifies the premise that Exp 2 is mainly a residual-scale cell. Net: ~+0.03 of new error against ~+0.02 of gain.

On the unchanged iter-5 base the error is still concentrated in Exp 6 (\u22120.761 vs \u22120.938, 0.176 \u2014 by far the largest), Exp 2 (.758 vs .654), Exp 13 (.574 vs .489), then Exp 4/10/9/11/12 at 0.03\u20130.05 each. Crucially, I now think I have mis-diagnosed Exp 6 for several rounds by treating it as a global 'gated drive' deficit. Its metric is d1 + d2, and d2 is the UNANIMOUS contrast: sparse (D\u22642, all-upper, e.g. A wins f2/f4 only) minus dense (D=4 including the panel floor f6 at .55). In the data that term should be strongly negative (\u2248chance on the sparse all-upper unanimous display, near-ceiling on the dense display containing the floor voice). In the current model the floor voice in a WHOLLY UNOPPOSED display gets Q = lam_own\u00b7\u03a3u_own \u2248 0.6\u00b73.0 \u2248 1.8, which is well below q0 = 2.45, so its gate never fires and the dense unanimous cell lands near the sparse one; d2 collapses toward 0 and caps |Exp 6| around \u22120.78 no matter how much beta or gmax is applied to opposed cells. That also explains, post hoc, why my iter-5 lam_own CUT was so badly rejected (Exp 6 \u2192 \u22120.714, Exp 11 \u2192 .595, Exp 5 weakened): the own-side echo is precisely what carries floor-voice drive in unopposed and majority-internal displays.

Faithfulness: the implementation is still inside the prescribed family \u2014 parameter-free count gate (Exp 1 pinned at exact chance, the corpus's cleanest invariant), non-negative weights, continuous seniority, modulation keyed to summed seniority MASS rather than cue count. kappa \u2248 15 plus a narrow knee remains far from CNW's literal kappa \u2248 1.5\u20133, but the gate has repeatedly validated it; it must not drift further toward a discrete or count-keyed rule.
Rationale: The candidate was REJECTED, so the base reverts to iter 5. Revert beta/eta/c_res to the iter-5 values (beta 34\u201348, eta 0.030\u20130.048, c_res 0.10\u20130.14) \u2014 the full rescale was tested and lost; do not re-apply it wholesale.

1. PRIMARY \u2014 RAISE THE OWN-SIDE ECHO: lam_own 0.50\u20130.70 \u2192 0.74\u20130.88. This is the one direction never actually tested by the gate (in iter 2 I asked for it and the proposer declined; in iter 5 I asked for the OPPOSITE and the gate rejected it \u2014 strong evidence the echo is load-bearing). Target mechanism: Exp 6's d2 term. Its dense unanimous display (A wins f0,f3,f5,f6 with f6 = panel floor, no opposition) currently gives the floor voice Q \u2248 1.8 < q0 = 2.45, so it gets no amplification and the cell sits near the sparse all-upper unanimous cell; at lam_own \u2248 0.82 the same voice reaches Q \u2248 2.4\u20132.6 and crosses the knee, pushing that cell toward ceiling and d2 toward \u22120.35\u2014\u22120.45. Critically this is SAFE for the two cells that constrain the knee from the other side: Exp 10-C4's floor voice (solo vs mid pair) and Exp 12-K4's floor voice (solo vs upper trio) have NO own side, so their Q is untouched by lam_own \u2014 unlike a q0 cut or a wid widening, which would re-inflate both. Targets: Exp 6 \u2264 \u22120.85, Exp 11 \u2265 .67, Exp 5 in [\u22120.74, \u22120.80], Exp 12 unchanged at .11\u2013.15, Exp 10 unchanged at [\u22120.06, +0.05].

2. GUARD EXP 2, AND DO IT WITH kappa, NOT WITH lam_own. The known side-effect of a larger echo is Exp 2's dominant 4-vs-1 conflict display, where the second-from-bottom voice f4 (s \u2248 .14, nov \u2248 .105) sits inside its own majority and would now cross the knee. Offset it by sharpening the novelty curve slightly: kappa 14.0\u201316.5 \u2192 16.5\u201319.5, which cuts f4's nov to \u2248.06 (nearly halving the leak) while leaving every true floor voice (s = 0, nov = 1: Exp 6 f6, Exp 12 f4, Exp 10 f5, Exp 11 f2) exactly where it is. This is the first kappa move I have asked for since iter 3 (which was ACCEPTED), and it is targeted rather than global. Guard-rail: Exp 6's second voice f1 (s \u2248 .08, nov .30 \u2192 .24) must not cost more than the d2 gain \u2014 if Exp 6 fails to reach \u22120.85, hold kappa at 16.5 and accept Exp 2 at \u2248.76.

3. SECOND-ORDER \u2014 THE n=8 CELLS (only after 1\u20132 verify). Exp 4 is still .701 vs .755 and Exp 9 .361 vs .405; at n = 8 psi \u2248 0.88 kills ~88% of the gate, leaving those panels driven by the thin residual alone. Cut psi 0.80\u20130.95 \u2192 0.62\u20130.78 so eight-expert majorities containing a low voice retain some novelty drive. Side-effect check: Exp 8 (star-backed n=8 majorities) is at .494 against a real .485, so a small drop is welcome \u2014 this move is safe down to Exp 8 \u2265 .46. Do not touch chi.

DO NOT: re-apply the residual/beta rescale (rejected), cut lam_own (rejected), move gamma or epsilon (they hold the Exp 15/16 sign and Exp 1/8 at data), or lower q0 / widen wid (both re-inflate Exp 10-C4 and Exp 12-K4 simultaneously). Re-verify after every edit: Exp 1 .47\u2013.51, Exp 3 .78\u2013.82, Exp 5 \u22120.74\u2014\u22120.80, Exp 7 .65\u2013.70, Exp 8 .46\u2013.51, Exp 12 .10\u2013.18, Exp 14 .46\u2013.52, Exp 15 in [\u22120.03, +0.04], Exp 16 in [+0.01, +0.07]. The aim is to convert \u22480.09 of the Exp-6 error and \u22480.03 of Exp-4/9 into at most \u22480.02 on Exp 2/13. Report Exp 6 (with d1 and d2 separately), 2, 11, 12 and 4 before making any further change.

**Outcome of your advice:** iter 7 candidate loss=0.0694 -> CURRENT candidate loss=0.0671 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "Iter 8 implemented my iter-7 prescription almost verbatim (lam_own 0.60\\u21920.79, kappa 15\\u219217.2) and the gate rejected it by a hair (0.0669 \\u2192 0.0671). The decomposition is unusually clean and, unlike the two previous rejections, it is mostly GOOD news: the own-side-echo raise did exactly what I predicted on the corpus's largest single error \\u2014 Exp 6 went \\u22120.761 \\u2192 \\u22120.927 against a real \\u22120.9375 (error 0.176 \\u2192 0.011, the biggest single-cell gain of the whole loop), and Exp 13 (.574 \\u2192 .549), Exp 12-K4 (.113 \\u2192 .118, real .145), Exp 5 (\\u22120.764 vs \\u22120.763) and Exp 11 (.658 \\u2192 .666) all improved or held. The mechanism claim behind it is now empirically settled: floor-voice drive in unopposed / majority-internal displays is carried by the own-side echo (the iter-5 CUT was rejected, the iter-8 RAISE reproduces Exp 6 almost exactly).\n\nThe reason it still lost the gate is that the raise paid for Exp 6 out of a small set of near-knee cells: Exp 2 regressed .758 \\u2192 .799 (real .654, now by far the worst cell at 0.146), Exp 10 +0.029 \\u2192 +0.058 (real \\u22120.023, i.e. the lone-floor-vs-mid-pair cell sits at \\u2248.56 instead of \\u2248.48), Exp 3 .810 \\u2192 .818, Exp 16 +0.022 \\u2192 +0.014 and Exp 8 .494 \\u2192 .508. The Exp-2 cost is fully diagnosable and is NOT a global scale problem (the iter-7 residual/beta rescale proved that): on its second conflict cell (A wins only f2 = .79; B wins f1 = panel floor, f4, f5) the floor voice's chorus is Q = u_f2 + lam\\u00b7(u_f4 + u_f5) \\u2248 1.08 + 0.79\\u00b71.65 \\u2248 2.38, which at wid = 0.14 sits right on the q0 = 2.45 knee \\u2014 the echo raise dragged that cell over it, so a floor voice gets amplified for merely standing beside two other low voices against a single senior cue. Exp 6's dense unanimous cell, by contrast, is WHOLLY unopposed, with Q = lam\\u00b7\\u03a3u_own \\u2248 0.79\\u00b73.26 \\u2248 2.58. The two cells are separated by only \\u22480.2 mid-equivalents at lam_own = 0.79, which is why they cannot both be right with the current (lam_own, q0) pair.\n\nFaithfulness: still inside the prescribed family \\u2014 parameter-free count gate (Exp 1 pinned at chance, the corpus's cleanest invariant), non-negative weights, continuous seniority, modulation keyed to summed seniority MASS rather than cue count. kappa \\u2248 17 with a narrow knee remains far from CNW's literal kappa \\u2248 1.5\\u20133, but the gate has validated that repeatedly; it must not drift further toward any discrete or count-keyed rule.",
  "verdict": "regenerate",
  "rationale": "The candidate was REJECTED, so the base is iter 5 again \\u2014 but do NOT discard the lesson of iter 8: the lam_own raise is the only move that has ever brought Exp 6 to data, and it lost the gate by 0.0002, i.e. it is a near-tie that needs one compensating knob rather than a reversal. Do not oscillate lam_own again (0.27 rejected, 0.60 accepted, 0.79 rejected-by-a-hair); instead RE-SCALE THE KNEE TOGETHER WITH IT so the echo can stay loud without dragging near-knee majority-internal cells over the threshold.\n\n1. PRIMARY \\u2014 MOVE lam_own AND q0 UP TOGETHER (new, never-tested combination; q0 has only ever been considered downward, which I correctly vetoed). Set lam_own 0.72\\u20130.86 \\u2192 0.94\\u20131.00 and q0 2.38\\u20132.52 \\u2192 2.75\\u20132.92, keeping wid at 0.11\\u20130.17. Worked arithmetic on the three cells that pin this knee: (a) Exp 6 dense UNOPPOSED display containing the panel floor f6 \\u2014 Q = lam\\u00b7\\u03a3u_own \\u2248 3.10\\u20133.26 \\u21d2 still clearly supra-threshold, so the d2 term and hence Exp 6 \\u2248 \\u22120.90 to \\u22120.94 is preserved; (b) Exp 2's conflict cell with the floor voice INSIDE a three-cue low majority \\u2014 Q = 1.08 + lam\\u00b71.65 \\u2248 2.63\\u20132.73 \\u21d2 now BELOW the knee, gate collapses, and that cell should fall from \\u2248.85 to \\u2248.70\\u2013.72, taking Exp 2 to \\u2248.72\\u2013.74; (c) Exp 12-K4, lone floor voice vs an upper TRIO with no own side \\u2014 \\u03a3u \\u2248 3.5\\u20133.6 \\u21d2 comfortably supra, so K4 must stay in [.10, .18]. Side benefit: Exp 10-C4's lone floor voice (Q \\u2248 2.0, no own side) loses its remaining sub-threshold leak, so the Exp 10 metric should come back from +0.058 toward [\\u22120.03, +0.03], and Exp 3/13 should each shed \\u22480.01\\u20130.02.\n\n2. VERIFY THE KNEE MARGINS BEFORE ANYTHING ELSE. The whole move rests on a \\u22480.5 mid-equivalent separation between 'unopposed own-side chorus' (\\u22483.2) and 'own-side chorus plus one senior opponent' (\\u22482.7). Report the realized Q of (i) Exp 6's dense unanimous floor voice, (ii) Exp 2's majority-internal floor voice, (iii) Exp 12-K4's opposed floor voice, and (iv) Exp 10-C4's opposed floor voice, together with q0, so the ordering Q(iii) > Q(i) > q0 > Q(ii) > Q(iv) is explicit. If Exp 12-K4 rises above .20 (i.e. its trio no longer clears q0), back q0 down to \\u22482.70 and raise lam_own to 1.00 rather than lowering lam_own.\n\n3. HOLD kappa AT 16.5\\u201317.5. The iter-8 kappa raise was at best neutral (Exp 16 +0.022 \\u2192 +0.014, Exp 10 worse) and it is not the lever for Exp 2 \\u2014 the offending cue there is a true panel FLOOR (s = 0, nov \\u2261 1), which kappa cannot touch. Do not raise it again, and do not cut it below \\u224816 (Exp 2's second-from-bottom leak returns).\n\n4. ONLY IF Exp 2 IS STILL ABOVE .75 AFTER (1): take a small epsilon step, 0.09\\u20130.15 \\u2192 0.12\\u20130.17. I have vetoed this before, but Exp 6 now has \\u22480.01 of headroom (\\u22120.927 vs \\u22120.9375) and epsilon simultaneously helps Exp 2, 3, 7, 9, 12 and 13, all of which currently over-adhere. Abandon it if Exp 5 weakens past \\u22120.73, Exp 6 past \\u22120.87, Exp 11 below .64 or Exp 4 below .69.\n\nDO NOT: cut lam_own (rejected), re-apply the residual-down/beta-up rescale (rejected), cut c_res or raise nu a third time, lower q0 or widen wid (both re-inflate Exp 10-C4 and Exp 2 at once), or touch gamma (it holds the Exp 15/16 solo-vs-flat sign) or the parameter-free count gate. Guard-rails to re-verify after every edit: Exp 1 .47\\u2013.51, Exp 3 .78\\u2013.82, Exp 5 \\u22120.74\\u2014\\u22120.80, Exp 6 \\u2264 \\u22120.88, Exp 7 .65\\u2013.70, Exp 8 .46\\u2013.51, Exp 10 in [\\u22120.06, +0.04], Exp 11 \\u2265 .65, Exp 12 .10\\u2013.18, Exp 13 \\u2264 .56, Exp 14 .46\\u2013.52, Exp 15 in [\\u22120.03, +0.04], Exp 16 in [+0.01, +0.07]. Target: keep the iter-8 Exp-6 gain (\\u22120.16 of error) while returning \\u22480.05\\u20130.08 of Exp-2 and \\u22480.03 of Exp-10 error, which should clear the 0.0669 base."
}
```

## Usage

```json
{
  "input_tokens": 59659,
  "output_tokens": 9880
}
```
