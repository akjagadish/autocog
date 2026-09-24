# feedback_iter_01

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
- THEORY 2 = `pi_6`
- The recommendation below acts on THEORY 2 (= `pi_6`).

Replace pi_6 with a 'Mute-Band Maverick Arbitration' theory (working name MBM). Core commitments:

(1) UNWEIGHTED COUNT GATE (kept, parameter-free): equal feature-wise win counts (including wholly non-discriminating displays) \u2192 exactly 0.5, no validity tie-break, no take-the-best fallback. This is required by Exp 3 (.474) and by the tied cells of Exp 7/10/12.

(2) HARD MUTE BAND, CHORUS-INDEPENDENT (the key departure from BOTH pi_8 and pi_6). Panel-relative seniority s_j = (v_j - v_min)/(v_max - v_min). News weight u(s) = max(0, 1 - s/theta)^rho + eta, with theta ~ 0.25-0.40 and rho ~ 0.7-1.5; eta is a tiny residual (0-0.08) common to every discriminating cue. Any expert more than theta of the panel span above the floor is essentially SILENT \u2014 not negative (pi_6's error) and not conditionally silent only when a big bloc is visible (pi_8's error). Immediate consequence: displays in which every discriminating voice is upper-band (Exp 1 cells A1/A3/A6/A8, Exp 2 cells C1/C3/C4/C7) give E \u2248 0 and therefore ~0.5 adherence at ANY count margin \u2014 which is exactly the observed 0.489 in both new experiments, where pi_8 over-predicts (.59-.64) and pi_6 under-predicts (.28-.37). The small eta lets 2-vs-0 and 3-vs-1 upper-only displays sit a hair above chance without ever reaching pi_8's .69-.87.

(3) CHORUS AMPLIFICATION OF THE MAVERICK (replaces pi_6's D-based discount with a bloc-based one, matching the evidence that favoured pi_8's CH rule). The weight of a below-theta voice is multiplied by an increasing, saturating function of the size of the bloc it opposes: g = 1 + alpha * (m_opp - 1)^p / ((m_opp - 1)^p + kappa), m_opp = size of the larger (count-winning) bloc. Calibrate so that a lone floor voice against a 2-cue majority yields only ~0.4-0.5 adherence (Exp 12 \u2248 0 contrast; Exp 13 c2 \u2248 .3-.4) while a lone floor voice against a 3-cue majority drives adherence down to ~0.15 (Exp 14 = .145).

(4) DILUTION AND LAPSE: divide the logit by D^gamma (D = number of discriminating cues) and mix with an attention lapse epsilon, as before.

(5) CAPACITY MODULE for n > 7: shrink theta's effect (weights flatten toward equal) AND inflate noise, tuned so that 8-expert star-backed majorities sit at chance (Exp 10 = .485) while overall tally adherence on 8-expert mixed displays stays ~0.75 (Exp 6).

Calibration targets the new theory must hit simultaneously (use these to set parameter ranges): Exp 1 = .49, Exp 2 = .49, Exp 3 = .47, Exp 4 \u2248 .65, Exp 5 \u2248 .79, Exp 6 \u2248 .75, Exp 7 \u2248 -.76, Exp 8 \u2248 -.94, Exp 9 \u2248 .66, Exp 10 \u2248 .49, Exp 11 \u2248 .41, Exp 12 \u2248 -.02, Exp 13 \u2248 .70, Exp 14 \u2248 .15. The decisive future test against pi_8 is any 2-vs-1 or 2-vs-0 display whose discriminating voices are all upper-band: MBM says exactly chance, pi_8 says .69-.88.


## CANDIDATE THEORY
**Mute-Band Maverick Arbitration with a Panel Gradient and a Graded Chorus (MBM-PG2).**

People who are told the accuracies of a panel of binary expert ratings do not weight experts *by* accuracy; they weight them by how much *news* a rating carries, and news is a steeply thresholded, panel-relative quantity. Six commitments.

**(1) The count comparator is a parameter-free gate.** Before any expert identity is consulted, an unweighted feature-wise comparison asks a purely ordinal question: does one product win on strictly more attributes? If not — equal win counts, including wholly non-discriminating displays — the display is construed as non-diagnostic and the subject flips a mental coin: exactly 0.5, with NO validity tie-break and NO take-the-best fallback, at every panel composition and size.

**(2) HARD MUTE BAND, CHORUS-INDEPENDENT.** Seniority is read off the *displayed panel*: s_j = (v_j − v_min)/(v_max − v_min). Only voices lying within a narrow band theta (≈12-20% of the panel span) above the panel floor are heard as carrying private, surprising information; their maverick strength is m_j = max(0, 1 − s_j/theta)^rho. Every expert more than theta above the floor is essentially SILENT as a maverick — not negative (a signed inversion's error) and not conditionally silent only when a big bloc is visible. Muting is unconditional: a lone star cue is as mute as a star inside a bloc.

**(3) GRADED CHORUS AMPLIFICATION OF THE MAVERICK.** A maverick's voice is loud only in proportion to the redundancy it is heard against, and redundancy is *asymmetric*: the bloc a cue CONTRADICTS is the primary chorus, while the bloc a cue sits INSIDE echoes it only partly and therefore counts at a discount lam_own (≈0.4-0.6). Formally, for cue j: C_j = (m_opp − 1) + lam_own·(m_own − 1) when the cue is opposed, and C_j = (m_own − 1) when the display is unanimous (no opposition: the only chorus is the consensus he stands in). Amplification G_j = 1 + alpha·C_j^p/(C_j^p + kappa) with p ≈ 5, so amplification is negligible for a maverick facing a single opponent, mild against a bare pair, and several-fold once a three-or-more-voice bloc is in play. This asymmetry is what separates a floor voice that *opposes* a big consensus (heard as decisive news, reversing a 3-cue majority to ≈.15 adherence) from a floor voice that *belongs* to a large winning bloc (heard as an ordinary vote, so majorities carried by low-validity experts are followed only tepidly, ≈.7-.8, never at ceiling).

**(4) THE PANEL GRADIENT: the mute band is quiet, not empty.** Muted experts do not vanish; each discriminating cue contributes a residual r_j = eta + c·(1 − s_j) — a flat listening floor plus a mild inverse-seniority gradient. This residual decides displays in which no voice is inside the mute band: they hover at chance at any count margin (a two-star unopposed majority a hair above .5, a star pair opposed by a lone mid-panel voice a hair below). It also separates *star-backed* majorities (tiny residual, easily overturned by a maverick) from *mid-backed* majorities (substantial residual, which survives a lone maverick), and lets tier comparisons come out the marginal tier's way without any signed inversion.

**(5) DILUTION AND LAPSE.** Net evidence E = Σ_j dir_j·(G_j·m_j + r_j) is compressed by the number of mutually contradicting cues, D^gamma, and the resulting choice is mixed with an attention lapse epsilon.

**(6) CAPACITY.** Holding the whole seniority ordering in mind is possible only up to about seven voices. For n > 7 the amplification collapses toward 1 (weights flatten to the panel gradient) and integration gets noisier, so eight-expert star-backed majorities sit at chance while plain tally adherence on eight-expert mixed displays is preserved.

**Cross-design signature:** exact chance on tally ties at any composition; chance on displays whose every discriminating voice is upper-band, at ANY count margin; decisive reversal (≈.15) of a 3-cue majority opposed by the floor voice but only near-chance rejection of a 2-cue majority opposed by the same voice; near-ceiling following of dense unanimous displays that contain the floor voice; tepid (≈.7-.8, not ceiling) following of majorities whose low-validity members merely swell the winning bloc; mid-backed majorities surviving a lone star; and attenuated inversion on eight-expert panels.

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

    theta = _g('theta', 0.16, 0.02, 0.90)
    rho = _g('rho', 1.2, 0.20, 4.00)
    eta = _g('eta', 0.26, 0.00, 1.00)
    c_res = _g('c_res', 2.4, 0.00, 6.00)
    beta = _g('beta', 0.80, 0.00, 5.00)
    gamma = _g('gamma', 0.95, 0.00, 2.50)
    alpha = _g('alpha', 46.0, 0.00, 400.0)
    kappa = _g('kappa', 92.0, 0.05, 2000.0)
    p_ch = _g('p_ch', 5.0, 0.50, 12.0)
    lam_own = _g('lam_own', 0.50, 0.00, 1.00)
    psi = _g('psi', 0.22, 0.00, 1.00)
    chi = _g('chi', 0.15, 0.00, 3.00)
    eps = _g('epsilon', 0.08, 0.00, 0.60)

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

    # ---------- Commitment 2: hard, chorus-independent mute band ----------
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
- theta: [0.12, 0.20]
- rho: [0.9, 1.5]
- eta: [0.18, 0.32]
- c_res: [2.1, 2.7]
- beta: [0.70, 0.90]
- gamma: [0.88, 1.02]
- alpha: [38, 54]
- kappa: [78, 110]
- p_ch: [4.6, 5.4]
- lam_own: [0.38, 0.60]
- psi: [0.10, 0.35]
- chi: [0.05, 0.25]
- epsilon: [0.04, 0.12]
- validities: validities

`rationale`:
**Minimal edit, one mechanism line.** Everything is re-emitted verbatim from the accepted base except the chorus term in commitment 3 (and the retuned constants it implies). The critic's primary fix was to replace the display-global `M = max(nA,nB)` with the size of the bloc a cue OPPOSES. I implemented exactly that, but with two necessary refinements that a literal implementation would have broken:

(1) **Asymmetric, graded chorus instead of a purely opposing one.** C_j = (m_opp − 1) + lam_own·(m_own − 1), with lam_own ≈ 0.5. A pure m_opp rule fixes Exp 2 (.83 → .76) but, by hand-simulation over the actual design matrices, it also destroys three cells that the base got right: Exp 5/6's *conflict* cells (a floor voice inside the count-winning bloc, which real data follow at ≈.85-.95 — the contrast would have gone from −.87 to −.69 on Exp 6, err .07 → .25), Exp 6's dense-unanimous cell, and Exp 11's e4 (star+star+floor majority vs a mid pair). The lam_own discount keeps a floor voice inside a large bloc audible (so Exp 5 ≈ −.70, Exp 6 ≈ −.96, Exp 11 e4 ≈ .85) while removing the pathological order-of-magnitude inflation that put Exp 2 at ceiling (now ≈ .76) and Exp 3/4 above the real tally level (now ≈ .81/.79 vs .794/.755).

(2) **Unopposed cues hear only their own bloc** (C = m_own − 1), which reproduces the base model exactly on unanimous displays — this is what carries Exp 6's d2 term (dense unanimous ≈ .96 vs sparse ≈ .69); a literal m_opp = 0 rule would have collapsed d2 from −.28 to −.09.

(3) **Re-calibration per critic items (3)-(4).** The chorus exponent is steepened (p ≈ 5) and alpha/kappa re-solved so that G ≈ 1.5 for a maverick facing a bare pair (keeps Exp 10's c4 near chance and Exp 11's c2 at ≈ .43, in the requested .35-.45 window) but G ≈ 13-35 once a three-or-more-voice bloc is opposed — this deepens the Exp 12 reversal from .236 to ≈ .11 (real .145) and Exp 11's a1 rejection to ≈ .04. The residual gradient is left essentially as-is (c_res 2.1-2.7, eta 0.18-0.32) because it is what holds all-upper-band displays at chance (Exp 13/14) and star-backed 8-expert majorities at chance (Exp 8 ≈ .46 by pure gradient, no amplification involved).

Hand-computed cell-by-cell predictions for the new configuration: Exp1 .50, Exp2 .76, Exp3 .81, Exp4 .79, Exp5 −.70, Exp6 −.96, Exp7 .71, Exp8 .46, Exp9 .36, Exp10 +.06, Exp11 .69, Exp12 .11, Exp13 .55, Exp14 .47 — a squared-error total roughly 40% of the accepted base's, with the two largest residuals (Exp 2 and Exp 11) both more than halved. The parameter-free tie gate, the unconditional mute band, the panel gradient, the dilution/lapse stage, the capacity module and the policy are untouched.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.0962 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.0819 -> ACCEPTED
Running-best (last accepted) base: iter 2 at loss=0.0819.

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
  - iter 1: 0.4931 (var=0.0090) (Δ vs real +0.0194)
  - iter 2 (current): 0.5181 (var=0.0062) (Δ vs real +0.0444)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8988 (var=0.0090)
- pi_2: 0.4988 (var=0.0068)
- pi_3: 0.5125 (var=0.0055)
- pi_4: 0.4913 (var=0.0090)
- pi_5: 0.5169 (var=0.0089)
- pi_6: 0.5106 (var=0.0056)
- pi_7: 0.4819 (var=0.0066)
- pi_8: 0.5075 (var=0.0083)

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
  - iter 1: 0.8269 (var=0.0050) (Δ vs real +0.1731)
  - iter 2 (current): 0.7662 (var=0.0069) (Δ vs real +0.1125)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8525 (var=0.0139)
- pi_1: 0.1431 (var=0.0111)
- pi_3: 0.6669 (var=0.0098)
- pi_4: 0.6469 (var=0.0093)
- pi_5: 0.9281 (var=0.0024)
- pi_6: 0.7025 (var=0.0100)
- pi_7: 0.7050 (var=0.0185)
- pi_8: 0.7425 (var=0.0058)

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
  - iter 1: 0.8213 (var=0.0018) (Δ vs real +0.0275)
  - iter 2 (current): 0.8235 (var=0.0018) (Δ vs real +0.0298)
**Other theories' values on this metric (for reference):**
- pi_3: 0.6379 (var=0.0030)
- pi_2: 0.8494 (var=0.0069)
- pi_1: 0.8731 (var=0.0059)
- pi_4: 0.7781 (var=0.0023)
- pi_5: 0.8167 (var=0.0020)
- pi_6: 0.7831 (var=0.0039)
- pi_7: 0.7769 (var=0.0024)
- pi_8: 0.8994 (var=0.0010)

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
  - iter 1: 0.8096 (var=0.0027) (Δ vs real +0.0550)
  - iter 2 (current): 0.8179 (var=0.0025) (Δ vs real +0.0633)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8538 (var=0.0068)
- pi_3: 0.6565 (var=0.0039)
- pi_1: 0.8571 (var=0.0084)
- pi_4: 0.7438 (var=0.0061)
- pi_5: 0.7783 (var=0.0043)
- pi_6: 0.7175 (var=0.0042)
- pi_7: 0.7642 (var=0.0035)
- pi_8: 0.8350 (var=0.0023)

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
  - iter 1: -0.6842 (var=0.0087) (Δ vs real +0.0792)
  - iter 2 (current): -0.7112 (var=0.0044) (Δ vs real +0.0521)
**Other theories' values on this metric (for reference):**
- pi_3: -0.0042 (var=0.0075)
- pi_4: 0.1438 (var=0.0077)
- pi_1: 0.6683 (var=0.0558)
- pi_2: 0.0062 (var=0.0039)
- pi_5: -0.7671 (var=0.0062)
- pi_6: -0.7525 (var=0.0071)
- pi_7: -0.7733 (var=0.0093)
- pi_8: -0.7063 (var=0.0049)

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
  - iter 1: -0.8662 (var=0.0266) (Δ vs real +0.0713)
  - iter 2 (current): -0.9206 (var=0.0188) (Δ vs real +0.0169)
**Other theories' values on this metric (for reference):**
- pi_4: 0.1587 (var=0.0237)
- pi_3: -0.0675 (var=0.0417)
- pi_1: 0.7206 (var=0.0361)
- pi_2: -0.0038 (var=0.0345)
- pi_5: -0.8519 (var=0.0288)
- pi_6: -0.9162 (var=0.0494)
- pi_7: -0.8237 (var=0.0280)
- pi_8: -1.0012 (var=0.0482)

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
  - iter 1: 0.6628 (var=0.0035) (Δ vs real +0.0022)
  - iter 2 (current): 0.6925 (var=0.0025) (Δ vs real +0.0319)
**Other theories' values on this metric (for reference):**
- pi_5: 0.9059 (var=0.0023)
- pi_4: 0.4641 (var=0.0042)
- pi_1: 0.1278 (var=0.0098)
- pi_2: 0.4997 (var=0.0029)
- pi_3: 0.4981 (var=0.0037)
- pi_6: 0.6775 (var=0.0023)
- pi_7: 0.7197 (var=0.0017)
- pi_8: 0.6572 (var=0.0028)

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
  - iter 1: 0.4888 (var=0.0072) (Δ vs real +0.0038)
  - iter 2 (current): 0.4512 (var=0.0073) (Δ vs real -0.0338)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7144 (var=0.0105)
- pi_5: 0.1506 (var=0.0044)
- pi_1: 0.8488 (var=0.0147)
- pi_2: 0.8550 (var=0.0140)
- pi_3: 0.5913 (var=0.0076)
- pi_6: 0.4919 (var=0.0121)
- pi_7: 0.4775 (var=0.0150)
- pi_8: 0.4850 (var=0.0181)

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
  - iter 1: 0.3769 (var=0.0034) (Δ vs real -0.0281)
  - iter 2 (current): 0.3572 (var=0.0029) (Δ vs real -0.0478)
**Other theories' values on this metric (for reference):**
- pi_5: 0.8438 (var=0.0038)
- pi_6: 0.3791 (var=0.0043)
- pi_1: 0.3253 (var=0.0042)
- pi_2: 0.4994 (var=0.0020)
- pi_3: 0.4828 (var=0.0041)
- pi_4: 0.5244 (var=0.0029)
- pi_7: 0.3969 (var=0.0046)
- pi_8: 0.3950 (var=0.0035)

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
  - iter 1: 0.0613 (var=0.0321) (Δ vs real +0.0837)
  - iter 2 (current): 0.0500 (var=0.0234) (Δ vs real +0.0725)
**Other theories' values on this metric (for reference):**
- pi_6: 0.3338 (var=0.0282)
- pi_5: -0.7275 (var=0.0246)
- pi_1: -0.0050 (var=0.0123)
- pi_2: -0.3575 (var=0.0255)
- pi_3: -0.1062 (var=0.0243)
- pi_4: -0.2200 (var=0.0268)
- pi_7: -0.0475 (var=0.0412)
- pi_8: -0.0162 (var=0.0372)

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
  - iter 1: 0.5738 (var=0.0074) (Δ vs real -0.1246)
  - iter 2 (current): 0.6558 (var=0.0041) (Δ vs real -0.0425)
**Other theories' values on this metric (for reference):**
- pi_7: 0.1512 (var=0.0062)
- pi_6: 0.5292 (var=0.0060)
- pi_1: 0.3317 (var=0.0037)
- pi_2: 0.3525 (var=0.0038)
- pi_3: 0.3254 (var=0.0065)
- pi_4: 0.3283 (var=0.0079)
- pi_5: 0.2325 (var=0.0054)
- pi_8: 0.6917 (var=0.0064)

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
  - iter 1: 0.2362 (var=0.0121) (Δ vs real +0.0912)
  - iter 2 (current): 0.1675 (var=0.0119) (Δ vs real +0.0225)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2562 (var=0.0218)
- pi_7: 0.9187 (var=0.0063)
- pi_1: 0.8500 (var=0.0170)
- pi_2: 0.8438 (var=0.0149)
- pi_3: 0.5925 (var=0.0166)
- pi_4: 0.7612 (var=0.0210)
- pi_5: 0.8588 (var=0.0103)
- pi_8: 0.1375 (var=0.0089)

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
  - iter 1: 0.5558 (var=0.0045) (Δ vs real +0.0667)
  - iter 2 (current): 0.5408 (var=0.0033) (Δ vs real +0.0517)
**Other theories' values on this metric (for reference):**
- pi_8: 0.6446 (var=0.0042)
- pi_6: 0.3654 (var=0.0148)
- pi_1: 0.6813 (var=0.0037)
- pi_2: 0.8667 (var=0.0111)
- pi_3: 0.5975 (var=0.0056)
- pi_4: 0.7529 (var=0.0048)
- pi_5: 0.4121 (var=0.0048)
- pi_7: 0.0892 (var=0.0030)

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
  - iter 1: 0.5154 (var=0.0059) (Δ vs real +0.0262)
  - iter 2 (current): 0.4883 (var=0.0037) (Δ vs real -0.0008)
**Other theories' values on this metric (for reference):**
- pi_6: 0.2792 (var=0.0159)
- pi_8: 0.5913 (var=0.0058)
- pi_1: 0.8583 (var=0.0102)
- pi_2: 0.8646 (var=0.0075)
- pi_3: 0.6125 (var=0.0071)
- pi_4: 0.7679 (var=0.0054)
- pi_5: 0.2533 (var=0.0084)
- pi_7: 0.0717 (var=0.0021)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The candidate (MBM-PG) is clearly the strongest instantiation of the arbiter's prescribed family so far: it is faithful to all six commitments (parameter-free count gate, unconditional mute band, chorus amplification, inverse-seniority residual gradient, dilution+lapse, capacity module) and it hits the hard calibration targets that discriminated the family from its predecessors. Exp 1 (.493 vs .474), Exp 7 (.663 vs .661), Exp 8 (.489 vs .485), Exp 9 (.377 vs .405), Exp 14 (.515 vs .489) are all essentially exact, and Exp 5/6 (-.68/-.87 vs -.76/-.94) are the right sign and near-right magnitude. The decisive all-upper-band prediction is broadly borne out (Exp 14 .515, Exp 13 .556 vs .489) — far better than pi_8's .59-.64 and pi_6's .28-.37.

The residual misfit is systematic and points at one localized implementation choice rather than at the mechanism family. The largest single error is Exp 2 (.827 sim vs .654 real, +.17). Those conflict cells are 1-vs-3 and 1-vs-4 displays in which the LOSING lone cue is the panel star (correctly muted) but the WINNING majority happens to contain the panel floor expert (f1=.57) and a near-floor expert (f4=.62). Because the code sets M = max(nA,nB) globally and applies G to every cue, the floor voice *inside the majority* gets the full 3-/4-bloc amplification (G≈10-19), inflating the majority evidence by an order of magnitude and driving adherence to ceiling. Conceptually, commitment 3 says the maverick is amplified by the size of the bloc it OPPOSES; a floor voice that is part of the winning bloc opposes only 1 cue and should get G≈1. The same bug pushes Exp 3 (.821 vs .794) and Exp 4 (.810 vs .755) a few points above the real tally-adherence level.

Secondary misfits: Exp 11 (.574 vs .698; pi_8 reached .692) — the composite under-separates d3 (mid-pair vs lone star, should be high) from c2 (mid-pair vs lone floor, should be ~.35-.45), and/or a1 (mid-trio vs star+floor) is not being rejected strongly enough. Exp 12 (.236 vs .145) — the 3-bloc-vs-lone-floor reversal is not quite deep enough. Exp 13 (.556 vs .489) and Exp 10 (+.061 vs -.022) show that all-upper-band and mid-pair-vs-floor cells still sit a few points too far above chance, i.e. the residual gradient c_res*(1-s) plus beta is slightly too steep/hot for those cells.
Rationale: The candidate was ACCEPTED (loss 0.0962, new running best), so the base is good and should be kept and tuned, not redesigned. Stay strictly inside the Mute-Band Maverick family; the only changes requested are implementation/calibration knobs:

(1) PRIMARY FIX \u2014 make the chorus amplification cue-specific. Replace the global `M = max(nA,nB)` with a per-cue opposing-bloc size: for a cue on side A use m_opp = nB, for a cue on side B use m_opp = nA, and G_j = 1 + alpha*(m_opp-1)^p/((m_opp-1)^p + kappa). This is exactly what commitment 3 states in prose ('the size of the bloc it opposes') and it removes the pathological amplification of floor voices that sit INSIDE the count-winning bloc. Expected effect: Exp 2 drops from .83 toward ~.65-.70 (the single largest residual), Exp 3 toward ~.80, Exp 4 toward ~.76, while Exp 12/14 (lone floor voice facing a 3-bloc, which still gets the full G) are untouched.

(2) After (1), re-check the two cells that depended on the old inflation: Exp 11's e4 (star+star+floor majority vs mid-pair) must stay high \u2014 it should now be carried by the floor cue's large residual (eta + c_res) rather than by amplification, so keep c_res near the upper end (~2.3-2.6) if e4 sags.

(3) Deepen the 3-bloc reversal slightly for Exp 12 (.236 \u2192 ~.15): raise alpha a little (e.g. 26-34) or lower kappa (15-25) while keeping p_ch\u22484 so the 2-bloc case stays near chance (Exp 11 c2 \u2248 .35-.45, Exp 10 c4 \u2248 .48).

(4) Cool the all-upper-band cells: Exp 13 (.556) and Exp 10 (+.061) are ~.06-.08 too high. Since these cells are decided purely by the residual gradient, either shrink c_res's effective slope relative to eta (raise eta toward .30-.40 / lower c_res's lower bound) or shrink beta's range slightly (e.g. .50-.72) and/or allow epsilon up to ~.15. Do this only after (1), since (1) already removes some of the over-adherence; do not over-cool or Exp 3/4/11 will undershoot.

(5) Keep everything else exactly as is: the parameter-free tie gate, unconditional muting, capacity module and policy are all working (Exp 1, 7, 8, 9, 14 are near-exact) \u2014 do not touch them.

**Outcome of your advice:** iter 1 candidate loss=0.0962 -> CURRENT candidate loss=0.0819 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate (MBM-PG2) is faithful to the arbiter's prescribed Mute-Band Maverick family: parameter-free count gate, unconditional (chorus-independent) mute band, bloc-based amplification of the maverick, residual panel gradient, dilution + lapse, and the n>7 capacity module. Nothing has drifted into a different family. The asymmetric chorus (C_j = (m_opp-1) + lam_own*(m_own-1)) I asked for in iteration 1 clearly worked: aggregate loss fell 0.0962 -> 0.0819 (ACCEPTED), and it bought improvements on exactly the cells I diagnosed \\u2014 Exp 2 .827 -> .766 (real .654), Exp 11 .574 -> .656 (real .698), Exp 12 .236 -> .168 (real .145), Exp 6 -.866 -> -.921 (real -.938), Exp 14 .515 -> .488 (real .489), Exp 5 -.684 -> -.711 (real -.763) \\u2014 without breaking the cells that were already exact (Exp 7 .693 vs .661, Exp 8 .451 vs .485).\n\nRemaining residuals are small and systematic, and they all point the same direction: the model is still slightly too confident on displays whose evidence comes from the winning bloc rather than from a true maverick. (a) Exp 2 is still the single biggest error (+.113): in those 1-vs-3/1-vs-4 conflict cells the floor/near-floor experts sitting INSIDE the count-winning bloc are still receiving ~half the full bloc amplification via lam_own = 0.5, so the tally side is over-weighted and adherence sits at .77 instead of .65. (b) Exp 3 (+.030) and Exp 4 (+.063) show the same mild global over-adherence on plain tally displays. (c) Exp 13 (+.052) and Exp 10 (+.073) are decided purely by the residual gradient eta + c_res*(1-s); both sit a few points too far above chance, i.e. the residual slope is still marginally too steep / the logit too hot for all-upper-band and mid-pair-vs-floor cells. (d) Exp 9 (-.048) and Exp 11 (-.043) sit slightly low, so any global cooling must be modest or those will overshoot downward. Exp 1 (.518 vs .474) is NOT a real miss \\u2014 the gate returns exactly 0.5 on every scored trial there, so the deviation is pure sampling noise and should not be chased.",
  "verdict": "regenerate",
  "rationale": "The candidate was ACCEPTED and is the right family, so keep it as the base and make only small in-family calibration edits. Concretely, in priority order:\n\n(1) REINFORCE THE ACCEPTED DIRECTION, ONE MORE SMALL STEP. The lam_own discount was the change that bought most of the improvement; it is still too generous. Lower the own-bloc echo range from [0.38, 0.60] to roughly [0.28, 0.45] (do NOT go to 0 \\u2014 the proposer's hand-check showed a pure m_opp rule collapses Exp 5/6 conflict cells and Exp 11 e4). Expected: Exp 2 .766 -> ~.70, Exp 4 -> ~.79, with Exp 5/6 contrasts essentially unchanged because those cells lean on the residual c_res rather than on amplification once lam_own is small.\n\n(2) FLATTEN THE RESIDUAL GRADIENT SLIGHTLY, NOT THE WHOLE LOGIT. Exp 13 (+.052) and Exp 10 (+.073) are governed only by eta + c_res*(1-s). Trade slope for floor: raise eta to ~[0.24, 0.38] and drop c_res to ~[1.7, 2.3], holding eta + c_res roughly constant so that dense/unanimous cells (Exp 6 d2, Exp 11 e4) keep their total residual mass while the star-vs-mid separation that inflates all-upper-band cells shrinks. This should bring Exp 13 toward ~.51 and Exp 10 toward ~.01 without pulling Exp 11 down further.\n\n(3) DO NOT apply a large global cooling (big beta cut or large epsilon increase). Exp 9 (-.048) and Exp 11 (-.043) are already on the low side, so a blanket temperature drop would fix Exp 2/3/4 at the cost of those two. At most allow a marginal widening of epsilon's upper bound to ~0.14 and leave beta's range where it is.\n\n(4) LEAVE UNTOUCHED: the parameter-free tie gate (Exp 1/3 tie cells are exact by construction), the unconditional mute band, the chorus exponent/alpha/kappa calibration (Exp 12 .168 vs .145 and Exp 14 .488 vs .489 are now essentially on target \\u2014 further deepening risks over-reversing), the capacity module (Exp 8 .451 vs .485 is fine), and the policy.\n\nGrading faithfulness: the candidate implements the arbiter's family correctly; these are calibration knobs only, not a redesign."
}
```

## Usage

```json
{
  "input_tokens": 34180,
  "output_tokens": 3154
}
```
