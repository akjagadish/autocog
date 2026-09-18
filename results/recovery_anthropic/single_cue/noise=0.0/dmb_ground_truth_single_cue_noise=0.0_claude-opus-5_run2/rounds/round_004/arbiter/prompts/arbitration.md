# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
**Redundancy-Discounted Tiered Integration with Majority Discounting, Panel-Size Capacity Dilution, and Graded Redundancy Relaxation (RDTI-MCR).**

All previous commitments are retained. (i) Stated validities are encoded only *coarsely and ordinally*: a median split into a "star" tier and a "marginal" tier, with no gradations inside a tier. (ii) The star tier is treated as **redundant** — the very accurate experts are construed as merely restating the obvious consensus quality of the product — so their ratings enter integration with only a small residual weight, while the idiosyncratic marginal experts carry full weight. Evidence is the tier-weighted signed sum E = Σ_j w_j (a_j − b_j), and choice is logistic in that scalar plus an attention lapse. Because the weights take only two values, tier-matched profiles cancel *exactly* and give an exact coin flip, with no validity tie-break and no take-the-best fallback. (iii) The same redundancy logic is applied to the majority itself: a feature-count majority is the most "obvious" statistic in the display, so subjects apply a *contrarian consensus discount* proportional to the count margin. (iv) Integration is capacity-limited over the *whole displayed panel*, so the effective slope is divided by a power of the panel size n, in addition to a milder within-trial spread discount over the number of discriminating cues D.

**The new commitment is that the redundancy discount is itself a comparative, graded judgement rather than a fixed property of a cue.** A star expert is discounted precisely *because other experts are seen as restating what he says*; the discount is therefore a function of how much of the panel is actually speaking on this trial. When almost nothing in the display discriminates — in the limit, when a single expert is the only one who differentiates the two products — there is no chorus for that expert to be redundant with, so his rating is taken at (nearly) full face value. The discount relaxes continuously back toward full weight as the number of discriminating cues D falls toward one: eps_eff = eps_w + (1 − eps_w)·ψ·exp(−(D−1)/κ_D), with ψ controlling how completely redundancy can be released and κ_D controlling how fast it releases. Because κ_D is small, the release is essentially confined to genuinely sparse displays. This predicts a sharp, design-level signature that no fixed-weight tiering model produces: **profiles in which exactly one high-validity expert discriminates are followed at high rates (≈0.78–0.85), even though profiles in which that same expert discriminates alongside several others are followed near chance**, since only in the latter case is the star rating construed as redundant restatement. Tally-tied, tier-matched profiles remain at exactly chance (E = 0 and margin = 0 regardless of eps_eff), and all dense profiles (D ≥ 2) are left effectively untouched.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ---------- parse the stimulus into two rating vectors ----------
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
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            stim = stim.reshape(2, -1)
        elif stim.ndim > 2:
            stim = stim.reshape(2, -1)
        if stim.shape[0] != 2:
            stim = stim.reshape(2, -1)
        a = np.asarray(stim[0], dtype=float).ravel()
        b = np.asarray(stim[1], dtype=float).ravel()

    n = int(a.shape[0])
    if n == 0 or int(b.shape[0]) != n:
        return np.array([0.5, 0.5], dtype=float)

    # ---------- parameters ----------
    def _get(name, default):
        try:
            return float(parameters.get(name, default))
        except Exception:
            return float(default)

    eps_w = _get('eps_w', 0.18)          # residual weight of the "star" (redundant) tier
    beta = _get('beta', 2.9)             # logistic slope at the reference panel size
    delta = _get('delta', 0.075)         # within-trial evidence-spread discount exponent
    tau = _get('tau', 0.9)               # panel-size (capacity) dilution exponent
    lapse = _get('lapse', 0.115)         # attention lapse
    count_bias = _get('count_bias', 0.45)  # contrarian consensus (majority) discount
    psi = _get('psi', 0.60)              # max release of the redundancy discount
    kappa_d = _get('kappa_d', 0.30)      # how fast redundancy releases as D -> 1

    eps_w = float(min(max(eps_w, 0.0), 1.0))
    beta = float(min(max(beta, 0.0), 20.0))
    delta = float(min(max(delta, -1.0), 2.0))
    tau = float(min(max(tau, 0.0), 3.0))
    lapse = float(min(max(lapse, 0.0), 0.6))
    count_bias = float(min(max(count_bias, 0.0), 1.5))
    psi = float(min(max(psi, 0.0), 1.0))
    kappa_d = float(min(max(kappa_d, 1e-6), 3.0))

    # ---------- stated validities ----------
    v = None
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is not None:
        try:
            v = np.asarray(v_raw, dtype=float).ravel()
        except Exception:
            v = None
    if v is None or v.shape[0] != n:
        # fallback: assume the display order is descending in validity
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75], dtype=float)

    # ---------- signed feature-wise comparison ----------
    diff = np.sign(a - b)                            # +1 A wins, -1 B wins, 0 tie
    D = int(np.sum(diff != 0))
    if D == 0:
        return np.array([0.5, 0.5], dtype=float)

    # ---------- graded redundancy relaxation ----------
    # A star expert is discounted only because OTHER experts restate him; when
    # the discriminating panel shrinks toward a single voice there is nothing
    # left to be redundant with, so the discount releases back toward 1.
    relax = psi * float(np.exp(-(float(D) - 1.0) / kappa_d))
    eps_eff = eps_w + (1.0 - eps_w) * relax
    eps_eff = float(min(max(eps_eff, 0.0), 1.0))

    # ---------- coarse credibility tiering + redundancy discount ----------
    k = int(n // 2)                                  # size of the "star" tier
    order = np.argsort(-v, kind='stable')            # most valid first
    w = np.ones(n, dtype=float)
    if k > 0:
        w[order[:k]] = eps_eff                       # star tier == treated as redundant

    # ---------- tier-weighted additive evidence ----------
    E = float(np.sum(w * diff))

    # spread discount: evidence thinly spread over many contradicting cues feels weaker
    denom = float(D) ** delta
    if not np.isfinite(denom) or denom <= 0.0:
        denom = 1.0

    # ---------- panel-size capacity dilution ----------
    # a bounded comparison budget is shared across the WHOLE displayed panel,
    # so integration fidelity falls as the number of experts n grows.
    # 6.0 is a reference panel size (a pure reparametrisation of beta).
    cap = (float(n) / 6.0) ** tau
    if not np.isfinite(cap) or cap <= 0.0:
        cap = 1.0
    denom = denom * cap

    z = beta * E / denom

    # ---------- contrarian consensus (majority-redundancy) discount ----------
    nA = int(np.sum(diff > 0))
    nB = int(np.sum(diff < 0))
    margin = float(np.clip(nA - nB, -6, 6))
    z = z - count_bias * margin

    if z > 30.0:
        z = 30.0
    elif z < -30.0:
        z = -30.0

    p_a = 1.0 / (1.0 + np.exp(-z))

    # ---------- lapse ----------
    p_a = (1.0 - lapse) * p_a + lapse * 0.5
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
    if not np.isfinite(s) or s <= 0:
        probs = np.ones_like(probs) / float(len(probs))
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_6
**Capacity-Gated Strongly-Saturating Redundancy Validity-Inversion Tallying (GVIT-CAP-S4).**

Five commitments describe how people combine binary expert ratings whose stated validities they have been told.

**(1) The count comparator is a gate, not a weight.** Before any evaluation of *whose* rating is whose, the subject asks a purely ordinal question: does one product win on more attributes than the other? This is answered with unweighted feature-wise comparisons. If the answer is "no" (equal numbers of winning attributes, including displays where nothing discriminates), the display is construed as *non-diagnostic*, no candidate is generated, and the subject flips a mental coin. Stated validities are **never** used as a tie-break. This is a hard, parameter-free prediction: on tally-tied, side-counterbalanced profiles adherence to the most-valid discriminating cue is exactly chance whatever the validity vector.

**(2) When a majority exists, integration uses inverse-informativeness weights.** The subjective news value of a rating is a smooth, panel-relative *decreasing* function of stated validity: s_j = (v_j - v_min)/(v_max - v_min) within the displayed panel, w_j = 1 - lambda_eff*f(s_j). A near-perfect expert is heard as merely restating the obvious consensus quality of the product, so his rating adds almost nothing (and, when lambda_eff*f > 1, is mildly counter-indicative); a barely-diagnostic expert is surprising and is treated as genuine private information carrying full weight. Evidence is the single scalar E = sum_j w_j*sign(a_j - b_j).

**(3) The redundancy construal saturates VERY STRONGLY just above the panel floor.** Redundancy is not judged on a linear scale of stated accuracy: "being one of the accurate ones" is a nearly categorical construal. An expert only has to be slightly above the panel's least-accurate voice before he is already heard as echoing the consensus. Formally the discount is a strongly concave function of panel-relative validity, f(s) = s^omega with omega well below 1 (~0.26-0.42), so the discount rises steeply just above the panel floor and then flattens. Two consequences follow that neither a linear nor a mildly concave inversion produces. (i) A majority carried by *middling* experts (the 2nd-5th most accurate voices) carries almost no subjective evidence: mid-panel support is discounted nearly as heavily as star support, so a mid-backed majority facing a lone star is followed only weakly, close to chance. (ii) Conversely, when the count loser is backed by the panel's genuinely idiosyncratic bottom voices, the anti-count evidence is *strengthened*, sharpening the reversal. So the model separates three displays that a linear inversion renders nearly identical: majority-by-mid-experts (followed barely above chance), majority-by-bottom-voices (followed near ceiling), and star-backed majorities (reversed toward floor).

**(4) Redundancy is comparative; evidence spread over many contradicting cues is compressed.** An expert is discounted only because other experts are heard as echoing him, so the inversion scales with how many voices actually speak on the trial: lambda_eff = lambda_n*(D-1)/((D-1)+kappa_D), where D is the number of discriminating cues; when exactly one expert differentiates the products there is no chorus and his rating is taken at face value. Confidence in E is then compressed by the number of mutually contradicting cues, D^gamma.

**(5) The inversion is a capacity-limited construal: it collapses when the panel exceeds working-memory span.** Discounting the star requires holding the validity ordering of the whole panel in mind, which is possible only up to roughly seven experts. Beyond capacity the ordering blurs, so the redundancy construal washes out toward equal weighting (lambda_n = lambda*(1 - psi*o)) and integration becomes noisier (slope divided by 1 + chi*o), where o = logistic((n - 7.5)/0.2). Finally an attention lapse epsilon replaces the decision by a coin flip.

Cross-design signature: exact chance on tally-tied profiles; strong systematic reversal of star-backed majorities on panels of <=7 experts; near-chance for the same star-backed profiles on panels of 8+ experts, with preserved tally adherence there for broad-mixture majorities; near-ceiling following of a lone discriminating expert; only weakly above-chance following of mid-panel-backed majorities that oppose a lone star; and markedly attenuated confidence whenever the weighted evidence is spread across many mutually contradicting cues.

`predict(parameters, state, history) -> np.ndarray`:
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

`policy(probs) -> int`:
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

## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=6):** [0.95, 0.78, 0.82, 0.55, 0.76, 0.8]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 2: A=[1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  trial 3: A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  trial 4: A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 5: A=[1, 1, 0, 0, 1, 1]  B=[1, 0, 0, 1, 0, 1]
  trial 6: A=[1, 0, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1, 1]
  trial 7: A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  trial 8: A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 0, 1]
  trial 9: A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 10: A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  trial 11: A=[1, 1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0, 1]
  trial 12: A=[1, 1, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 1]

**Rationale:** GOAL: dissociate RDTI-MCR (advocated) from GVIT-CAP-S4 (competing). Both models discount high-validity experts as 'redundant', both release the discount when only one cue discriminates, and both compress evidence by D, so generic conflict trials are useless. They differ on exactly three things, and the panel is engineered so that each difference produces a SIGN reversal or a parameter-free contrast.

PANEL: 6 experts, validities in scrambled screen order [f0=.95, f1=.78, f2=.82, f3=.55, f4=.76, f5=.80]. Descending rank f0>f2>f5>f1>f4>f3. Two structural properties are deliberate. (a) The median split falls INSIDE a tight cluster (.80 vs .78), so RDTI's coarse ordinal tiering (STAR={f0,f2,f5} weight eps_w; MARGINAL={f1,f4,f3} weight 1) cuts sharply between two nearly identical stated validities. (b) There is one lone floor expert (f3=.55) far below a dense upper cluster, so GVIT's strongly saturating inversion f(s)=s^omega (omega .26-.42) gives ALL of f0,f2,f5,f1,f4 nearly the same (and, for lam>1.1, negative) weight while f3 alone keeps weight +1. Hence GVIT ~ 'follow the least accurate expert, mildly opposed by everyone else' whereas RDTI ~ 'follow the bottom-half tier, stars count 0.18'. Validity rank is decoupled from screen position; n=6 keeps RDTI's capacity term exactly 1 and is below GVIT's 7.5 overload threshold, removing nuisance factors. 6 configurations x 2 exact A/B mirrors = 12 unique pairs x 8 repetitions = 96 trials. Tied cues are coded sometimes 1-1 and sometimes 0-0.

CELLS (A-favouring version listed; the following trial is its exact mirror):
K1 (T1/T2) TALLY-TIED, MEDIAN-BOUNDARY: A wins f5(.80, star tier), B wins f1(.78, marginal tier); 4 cues tied.
K2 (T3/T4) TALLY-TIED 2v2, WIDE MISMATCH: A wins f4(.76)+f3(.55) [marginals], B wins f0(.95)+f2(.82) [stars].
K3 (T5/T6) MID-CLUSTER vs FLOOR: A wins f1(.78)+f4(.76), B wins f3(.55); f0,f2,f5 tied (D=3, margin +1).
K4 (T7/T8) DENSE MIXED (D=6, margin +2): A wins f0,f2,f5 (all stars) AND f3 (floor); B wins f1(.78),f4(.76).
K5 (T9/T10) SOLO STAR: only f0(.95) discriminates, A wins it (D=1).
K6 (T11/T12) SOLO FLOOR: only f3(.55) discriminates, A wins it (D=1).

QUANTITATIVE PREDICTIONS, P(choose A) at nominal parameters.
RDTI-MCR (eps_w=.18, beta=2.9, delta=.075, tau=.9, lapse=.115, count_bias=.45, psi=.60, kappa_d=.30): K1=0.14, K2=0.93, K3=0.81, K4=0.16, K5=0.77, K6=0.85.
GVIT-CAP-S4 (lam=1.45, beta=6.7, gamma=.90, kappa_d=.10, psi=.40, chi=.95, eps=.17, omega=.34): K1=0.500, K2=0.500, K3=0.12, K4=0.62, K5=0.914, K6=0.914.

FOUR ORTHOGONAL DIAGNOSTICS:
(1) TALLY-TIED CELLS K1, K2 are parameter-free for GVIT: nA=nB triggers the count gate and returns EXACTLY 0.5 for every lam, beta, gamma, kappa_d, psi, chi, eps, omega. RDTI predicts strong, systematic preference for the marginal-tier-backed option: across its entire parameter box K1 gives P(A)=0.08-0.27 (i.e. 73-92% choice of the f1-backed option) and K2 gives 0.82-0.95. Each subject contributes 32 such trials, so even a 10-point deviation from 0.5 is decisive. K1 additionally localises the tier boundary to the .80/.78 crack: a subject who tiers at all but places the cut elsewhere will be at chance on K1 while still strong on K2.
(2) SIGN REVERSAL K3 (mid cluster vs floor expert). RDTI: all three discriminating cues are in the marginal tier, so E=+1 exactly regardless of eps_w; P(A)=0.71-0.93 over the whole box. GVIT: the saturating inversion makes f1 and f4 (.78/.76) almost as 'redundant' as the stars while f3 keeps full weight, so E<0 for every admissible lam/omega; P(A)=0.10-0.25. An ~0.6-0.8 swing on identical stimuli. Observing that subjects follow the two middling experts over the single worst expert falsifies GVIT's saturating-inversion commitment; following the .55 expert against two .76-.78 experts falsifies RDTI.
(3) SIGN REVERSAL K4 (dense, margin +2 for A). RDTI: E=3*eps_w+1-2 <= 0.02, and the contrarian consensus discount subtracts count_bias*2, giving P(A)=0.10-0.40 (always sub-chance) - the star-backed majority is REJECTED. GVIT: the floor expert f3 favours A with weight +1 while the opposing mids carry negative weight, so E>0 and P(A)=0.50-0.85. This is the cleanest test of RDTI's majority-discount plus fixed two-valued tiering against GVIT's graded inverse-validity weights.
(4) SOLO CELLS K5 vs K6 (both D=1). For GVIT, lam_eff = lam*(D-1)/((D-1)+kappa_D) = 0 exactly at D=1, so ALL weights are 1 and the two cells have IDENTICAL predictions (~0.86-0.99) for every parameter setting - a parameter-free equality. RDTI releases the star discount only partially (eps_eff = eps_w+(1-eps_w)*psi < 1), so it predicts solo-floor > solo-star by 0.05-0.25 (0.85 vs 0.77 nominally) and predicts the solo-star cell to sit well below GVIT's >=0.86 floor. Comparing K5 (star alone, ~0.77) with K4 (same star embedded in a chorus, ~0.16) also isolates RDTI's graded-relaxation claim.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror so side bias cancels exactly; the number of displayed 1s does not track composition; validity rank is scrambled relative to screen position; no pair is degenerate and no cell has the two models predicting the same thing.

INTERPRETATION: exact chance on K1/K2 plus following the .55 expert in K3 and the floor-backed option in K4, with equal near-ceiling adherence on both solo cells, uniquely supports GVIT-CAP-S4. Strong marginal-tier-driven choices on the tally-tied cells, adherence to the two middling experts in K3, sub-chance choice of the star-backed majority in K4, and a lower solo-star than solo-floor adherence uniquely support RDTI-MCR.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
The metric is the proportion of dense-display trials (D>=2 discriminating cues) on which the subject chooses the option favoured by the net vote of the LOW-validity (bottom-half) experts f1(.78), f4(.76), f3(.55). This single number is defined stimulus-wise (no trial indices needed), applies to 64 of the 96 trials per subject, and is well defined for a single subject's slice, so it is tight per subject (mostly binomial noise over 64 trials plus modest parameter spread).

Why it discriminates: RDTI-MCR gives the whole bottom tier weight 1 and the star tier weight eps_w<=0.34, and additionally subtracts a contrarian majority discount, so on all four dense cells it follows the marginal-tier-backed option: K1 ~0.86, K2 ~0.93, K3 ~0.81, K4 ~0.84 (adherence 0.75-0.95 over its whole parameter box) => pooled value ~0.86.

GVIT-CAP-S4 behaves very differently on exactly these trials. On the two tally-tied cells (K1, K2, 32 trials) its count gate returns EXACTLY 0.5 for every parameter setting, so adherence is chance. On K3 its strongly saturating inversion makes f1/f4 nearly as discounted as the stars while the lone floor expert f3 keeps full weight, reversing the choice (adherence ~0.12). On K4 the floor expert again pulls toward the star-backed side (adherence ~0.38). Pooled value ~0.37-0.40.

Expected gap ~0.48 with per-subject variance on the order of 0.005 for both theories, giving a Welch t of roughly 20 at N=25. Crucially the metric is not trivially true for RDTI: any model that weights cues monotonically in stated validity (the normative/classic weighted-additive account) would score far BELOW 0.5, and GVIT's own inverse-validity account scores below 0.5 too; only RDTI's coarse two-tier split at the .80/.78 boundary plus majority discounting produces a high value.

Source:
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

### RESULTS
- Predicted under pi_5 (simulated): 0.8438 (var=0.0038)
- Predicted under pi_6 (simulated): 0.3791 (var=0.0043)
- Observed on real data: 0.4050 (var=0.0022)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=7):** [0.74, 0.95, 0.55, 0.78, 0.7, 0.52, 0.9]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1, 0]
  trial 2: A=[0, 1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1, 0]
  trial 3: A=[0, 0, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0, 1]
  trial 4: A=[0, 1, 0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0, 1, 1]
  trial 5: A=[1, 0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0, 1]
  trial 6: A=[1, 1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 1, 0]
  trial 7: A=[1, 1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  trial 8: A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0, 0]
  trial 9: A=[0, 1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1, 0]
  trial 10: A=[0, 0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 1, 0]
  trial 11: A=[1, 0, 0, 1, 1, 1, 1]  B=[1, 0, 0, 1, 1, 0, 1]
  trial 12: A=[1, 0, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate GVIT-CAP-S4 (advocated) from RDTI-MCR (competing). Both models discount high-validity experts as 'redundant', both release the discount when a single cue discriminates, and both compress evidence by the number of discriminating cues, so ordinary conflict trials are useless. They differ on exactly three points, and the panel is engineered so each difference produces either a parameter-free equality that only one model can honour, or a sign reversal.

PANEL: 7 experts, validities in deliberately scrambled screen order [f0=.74, f1=.95, f2=.55, f3=.78, f4=.70, f5=.52, f6=.90]. Descending rank f1(.95) > f6(.90) > f3(.78) > f0(.74) > f4(.70) > f2(.55) > f5(.52). Two structural properties are deliberate. (a) RDTI's median split (k = 7//2 = 3) makes STAR = {f1,f6,f3} (weight eps_w) and MARGINAL = {f0,f4,f2,f5} (weight 1), and CRUCIALLY the marginal tier contains BOTH a middling pair (.74/.70) and a genuine bottom pair (.55/.52); because RDTI's weights take only two values, these two pairs are interchangeable for it. (b) GVIT's panel-relative saturating inversion f(s)=s^omega (omega .26-.42) with vmin=.52, vmax=.95 assigns s^.34 = 1.00/.959/.843/.796/.744/.405/0 to f1/f6/f3/f0/f4/f2/f5, so the middling experts (.74,.70) are discounted almost as heavily as the stars (weights ~ -0.07 and ~ +0.00 at lam_eff=1.34) while the two bottom voices keep large positive weight (+0.46 and +1.00). Hence GVIT sharply separates a mid-backed majority from a bottom-backed one, whereas RDTI cannot. n=7 is below GVIT's 7.5 capacity threshold (o=.076), so the inversion is intact and only mildly noisy. 6 configurations x 2 exact A/B mirrors = 12 unique pairs x 8 repetitions = 96 trials; tied cues are coded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s does not track composition.

CELLS (A-favouring version listed; the following trial is its exact mirror):
C1 (T1/T2) MID-MAJORITY vs LONE STAR (D=3, margin +1): A wins f0(.74) and f4(.70); B wins f1(.95); f2,f3,f5,f6 tied.
C2 (T3/T4) BOTTOM-MAJORITY vs LONE STAR (D=3, margin +1): A wins f2(.55) and f5(.52); B wins f1(.95); f0,f3,f4,f6 tied. IDENTICAL tier composition to C1 for RDTI.
C3 (T5/T6) TALLY-TIED 2v2, TIER-MISMATCHED: A wins f2(.55), f5(.52); B wins f1(.95), f6(.90); f0,f3,f4 tied.
C4 (T7/T8) MID-PAIR vs FLOOR (D=3, margin +1): A wins f0(.74), f4(.70); B wins f5(.52); f1,f2,f3,f6 tied.
C5 (T9/T10) SOLO STAR (D=1): only f1(.95) discriminates, A wins it.
C6 (T11/T12) SOLO FLOOR (D=1): only f5(.52) discriminates, A wins it.

QUANTITATIVE PREDICTIONS, P(choose A) at nominal parameters.
GVIT-CAP-S4 (lam=1.45, beta=6.7, gamma=.90, kappa_d=.10, psi=.40, chi=.95, eps=.17, omega=.34): C1=0.63, C2=0.90, C3=0.500 (exact), C4=0.15, C5=0.91, C6=0.91.
RDTI-MCR (eps_w=.18, beta=2.9, delta=.075, tau=.9, lapse=.115, count_bias=.45, psi=.60, kappa_d=.30): C1=0.92, C2=0.92 (exactly equal), C3=0.92, C4=0.83, C5=0.75, C6=0.84.

FOUR ORTHOGONAL DIAGNOSTICS:
(1) MID vs BOTTOM MAJORITY (C1 vs C2) is PARAMETER-FREE for RDTI: both cells have two marginal-tier winners, one star-tier loser, D=3, margin +1, same n, so E = 2 - eps_w and the predicted P(A) is IDENTICAL for every eps_w, beta, delta, tau, lapse, count_bias, psi, kappa_d. GVIT's saturating inversion predicts a large gap: E(C1) = 1 - lam_eff*(s0^om + s4^om - 1) ranges 0.06-0.53 over its whole box (and can go slightly negative at omega=.26, high lam), while E(C2) = 1 + lam_eff*(1 - s2^om) ranges 1.5-2.0, giving C1 ~ 0.52-0.82 versus C2 ~ 0.86-0.96, i.e. a C2-minus-C1 gap of about 0.15-0.40 that RDTI must set to exactly 0. Each subject contributes 16 trials per cell, so this within-subject contrast is decisive.
(2) TALLY-TIED TIER-MISMATCHED CELL (C3) is parameter-free for GVIT: nA = nB triggers the count gate and returns EXACTLY 0.5 for every lam, beta, gamma, kappa_d, psi, chi, eps, omega, with no validity tie-break. RDTI predicts a strong, systematic preference for the option backed only by the two WORST experts (E = 2(1-eps_w) >= 1.32, margin = 0 so the contrarian discount is inert): 0.85-0.96 across its entire box. Note the sign is also opposite to WADD/TTB (which would go below 0.5 here), so the cell is self-diagnostic three ways.
(3) SIGN REVERSAL ON C4 (mid pair against the floor expert). RDTI: all three discriminating cues are marginal-tier, so E = +1 EXACTLY regardless of eps_w, and P(A) = 0.72-0.92 over the box. GVIT: E = 1 - lam_eff*(s0^om + s4^om) is negative for every admissible lam and omega (-0.33 to -1.9), so P(A) = 0.10-0.35 - subjects should follow the single .52 expert AGAINST two .70-.74 experts. Comparing C4 with C1, which is the same A-side profile with only the opponent changed from the floor expert to the star, isolates GVIT's claim that a lone .95 expert is heard as counter-indicative while a lone .52 expert is heard as decisive: GVIT predicts a ~0.48 swing (0.63 vs 0.15), RDTI predicts both high and close (0.92 vs 0.83).
(4) SOLO CELLS C5 vs C6 (both D=1). For GVIT lam_eff = lam_n*(D-1)/((D-1)+kappa_d) = 0 exactly at D=1, so all weights equal 1 and the two cells have IDENTICAL predictions (0.865-0.965) for every parameter setting - a parameter-free equality, and a high floor. RDTI releases the star discount only partially (eps_eff = eps_w + (1-eps_w)*psi < 1), so it predicts solo-floor > solo-star (nominally 0.84 vs 0.75) and allows the solo-star cell to fall as low as ~0.5, below GVIT's reachable floor.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror so side bias cancels exactly; validity rank is decoupled from screen position so positional heuristics cannot mimic validity-based weighting; tied cues alternate between 1-1 and 0-0 coding; no pair is degenerate (every trial requires reading several ratings) and no cell has the two models predicting the same thing.

INTERPRETATION: exact chance on C3, a large bottom-backed-minus-mid-backed gap (C2 >> C1) at matched composition, sub-chance choice of the mid-backed majority in C4, and equal near-ceiling adherence on the two solo cells uniquely support GVIT-CAP-S4. Equal adherence on C1 and C2, strong marginal-tier-driven choice on the tally-tied C3, above-chance adherence in C4, and solo-floor exceeding solo-star uniquely support RDTI-MCR.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This is a within-subject double contrast built from the two cells where the models make opposite-signed, near-parameter-free commitments, so the individual gaps ADD rather than cancel.

Cell C3 (trials 5/6): exactly two cues favour each option (bottom pair f2=.55,f5=.52 vs stars f1=.95,f6=.90). GVIT-CAP-S4's count gate fires on nA=nB and returns EXACTLY 0.5 with no validity tie-break, for every parameter setting: P(choose bottom-backed) = 0.50. RDTI-MCR weights the marginal tier fully and the star tier at eps_w, with margin=0 so the contrarian discount is inert, giving a strong pull toward the bottom-backed option (~0.85-0.96 across its whole parameter box).

Cell C4 (trials 7/8): the mid pair (f0=.74,f4=.70) opposes the lone floor expert (f5=.52). GVIT's strongly saturating inversion makes the mid pair nearly worthless (or counter-indicative) while the floor voice keeps full weight, so subjects should follow the single .52 expert: P(follow floor) ~ 0.65-0.90. RDTI puts all three cues in the marginal tier, so E = +1 exactly and, despite the count discount, it follows the mid-pair majority: P(follow floor) ~ 0.08-0.28.

Taking (follow-floor in C4) minus (bottom-backed in C3) makes the two signs oppose: GVIT ~ +0.35, RDTI ~ -0.75, a gap of ~1.1 on a bounded [-1,1] scale. Each subject contributes 16 trials to each cell, so the per-subject sampling variance of the difference is at most ~0.03 (SD ~0.18), which is tiny relative to the gap. The metric is computed identically on a single subject's slice and on the pooled frame, and mirrored A/B versions of each configuration are pooled by identifying the target option structurally (which side wins the bottom pair / the floor cue), so any side bias cancels exactly.

Source:
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


### RESULTS
- Predicted under pi_5 (simulated): -0.7275 (var=0.0246)
- Predicted under pi_6 (simulated): 0.3338 (var=0.0282)
- Observed on real data: -0.0225 (var=0.0100)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
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
**Predicted under pi_5:** 0.5169 (var=0.0089)
**Predicted under pi_6:** 0.5106 (var=0.0056)

### Experiment 4
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
**Predicted under pi_5:** 0.9281 (var=0.0024)
**Predicted under pi_6:** 0.7025 (var=0.0100)

### Experiment 5
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
**Predicted under pi_5:** 0.8167 (var=0.0020)
**Predicted under pi_6:** 0.7831 (var=0.0039)

### Experiment 6
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
**Predicted under pi_5:** 0.7783 (var=0.0043)
**Predicted under pi_6:** 0.7175 (var=0.0042)

### Experiment 7
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
**Predicted under pi_5:** -0.7671 (var=0.0062)
**Predicted under pi_6:** -0.7525 (var=0.0071)

### Experiment 8
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
**Predicted under pi_5:** -0.8519 (var=0.0288)
**Predicted under pi_6:** -0.9162 (var=0.0494)

### Experiment 9
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
**Predicted under pi_5:** 0.9059 (var=0.0023)
**Predicted under pi_6:** 0.6775 (var=0.0023)

### Experiment 10
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
**Predicted under pi_5:** 0.1506 (var=0.0044)
**Predicted under pi_6:** 0.4919 (var=0.0121)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Pooling all ten experiments, pi_6 (GVIT-CAP-S4) dominates pi_5 (RDTI-MCR) almost everywhere, and pi_5's misses are systematic rather than noise.\n\nWhere they are both fine: Exp3 (tally-tied trials at chance: obs 0.474, pi_5 0.517, pi_6 0.511), Exp5 (tally adherence 0.794 vs 0.817/0.783), Exp6 (0.755 vs 0.778/0.718), Exp7 (coherent\\conflict contrast -0.763 vs -0.767/-0.753), Exp8 (-0.938 vs -0.852/-0.916). These establish the robust, task-invariant facts: (a) subjects follow the unweighted count roughly 75-80% of the time on dense displays, (b) tally ties yield exact chance with no validity tie-break, and (c) there is a strong *inverse*-validity signature \\, the top-validity discriminating cue backing the count winner sharply lowers adherence.\n\nWhere they diverge, pi_5 is consistently too extreme in the anti-star / majority-discount direction. Exp1: obs 0.405, pi_6 0.379, pi_5 0.844 \\, a 0.44 miss. Exp9: obs 0.661, pi_6 0.678, pi_5 0.906. Exp10: obs 0.485 (star-backed majorities on an 8-expert panel are at *chance*, not reversed), pi_6 0.492, pi_5 0.151. Exp4: obs 0.654, pi_6 0.703, pi_5 0.928. Four decisive failures, all traceable to pi_5's two-valued median-split tiering plus its unconditional contrarian count discount, which forces near-deterministic rejection of star-backed majorities and near-deterministic marginal-tier following on tally-tied displays \\, exactly what the data refute (tally ties are at chance; star-backed dense majorities on large panels are at chance).\n\npi_6 is not perfect either. Its one clear miss is Exp2: observed -0.023 versus predicted +0.334 (pi_5: -0.728). Decomposed, C3 (tally tied) is at chance as its gate demands, so the miss lies in C4: subjects followed the lone floor expert (.52) against the mid pair (.74/.70) only about half the time, whereas pi_6's strongly saturating inversion (omega ~0.34, which nearly zeroes mid-panel experts) predicts ~0.83. A *linear* inverse-validity discount with lam ~1.0-1.1 makes two mid weights sum to about one floor weight and lands C4 near 0.5 \\, i.e. commitment (3) of pi_6 (very strong saturation just above the panel floor) is the wrong part, while its gate, its inverse weighting, its D-compression and its (ad hoc but empirically useful) large-panel washout are all supported.\n\nSo pi_5 is the degenerate member of the pair and should be replaced. Since pi_6 already explains nine of ten experiments well, the useful move is not a re-parameterisation of pi_5's tiering story (its qualitative commitments are falsified, not merely mis-tuned) but a genuinely new theory that keeps the empirically supported invariants and repairs the exact place where pi_6 overshoots, so that the next round is a sharp contest on the shape of the validity discount and on the mechanism behind the large-panel washout.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_5 with a *graded* inverse-validity tallying account that removes coarse tiering and the unconditional majority discount, and that explains the large-panel washout by smooth dilution rather than by a hard capacity cliff. Sketch:\n\n**Gated Linear Inverse-Validity Integration with Panel-Size Dilution (GLIV-D).**\n1. Count gate (shared with pi_6, strongly supported by Exp3, Exp1's tied cells and Exp2's C3): if nA == nB the display is non-diagnostic and P(A)=0.5 exactly, with no validity tie-break and no take-the-best fallback.\n2. When a strict count winner exists, evidence is E = sum_j w_j * sign(a_j - b_j) with a *linear* (or at most mildly concave, exponent 0.8-1.2) panel-relative inverse-validity weight w_j = 1 - lam * (v_j - v_min)/(v_max - v_min), lam in ~0.85-1.20. This is the crucial departure from pi_6: mid-panel experts must retain roughly half-weight so that two mids (~.74/.70) exactly offset one floor voice (~.52) \\, forcing Exp2's C4 to ~0.50 rather than 0.83, while still making the top expert near-zero/slightly negative (preserving Exp7/Exp8's large negative contrasts and Exp9's 0.66).\n3. Graded dilution instead of a 7.5-expert step: slope beta is divided by D^gamma * (n/6)^tau (tau ~0.6-1.2, gamma ~0.7-1.0). This must reproduce, without any threshold, that star-backed dense majorities on n=6/7 panels are reversed or well below chance (Exp1, Exp7, Exp8) while the same structural profiles on n=8 panels regress to chance (Exp10 obs 0.485), and that overall tally adherence on mixed dense displays stays ~0.75-0.80 (Exp5 0.794, Exp6 0.755).\n4. Only a mild release of the discount when a single cue discriminates (lam_eff scaled by (D-1)/((D-1)+kappa), kappa small), plus a lapse ~0.10-0.20 keeping predictions off the 0/1 rails \\, the observed values are all intermediate (0.40-0.79, or -0.02), so parameter ranges should be chosen so that no cell is pushed past ~0.90 or below ~0.10.\n\nCalibration targets the new theory must hit simultaneously: Exp1 ~0.40-0.45; Exp2 ~-0.10 to +0.05; Exp3 ~0.48-0.52; Exp4 ~0.62-0.70; Exp5 ~0.78-0.82; Exp6 ~0.73-0.78; Exp7 ~-0.75; Exp8 ~-0.90; Exp9 ~0.65-0.70; Exp10 ~0.45-0.52. It should be designed to be *distinguishable* from pi_6 by two clean signatures: (i) mid-panel-backed majorities facing a lone star should be clearly above chance under GLIV-D but near chance under pi_6's saturating inversion, and (ii) the n=7 -> n=8 transition should be smooth under GLIV-D but a step under pi_6's capacity gate."
}
```

## Usage

```json
{
  "input_tokens": 29990,
  "output_tokens": 7889
}
```
