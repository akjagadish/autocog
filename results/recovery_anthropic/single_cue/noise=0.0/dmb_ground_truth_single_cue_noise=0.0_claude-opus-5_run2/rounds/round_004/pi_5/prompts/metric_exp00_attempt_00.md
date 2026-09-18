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



## ADVOCATED THEORY
**Description:** **Redundancy-Discounted Tiered Integration with Majority Discounting, Panel-Size Capacity Dilution, and Graded Redundancy Relaxation (RDTI-MCR).**

All previous commitments are retained. (i) Stated validities are encoded only *coarsely and ordinally*: a median split into a "star" tier and a "marginal" tier, with no gradations inside a tier. (ii) The star tier is treated as **redundant** — the very accurate experts are construed as merely restating the obvious consensus quality of the product — so their ratings enter integration with only a small residual weight, while the idiosyncratic marginal experts carry full weight. Evidence is the tier-weighted signed sum E = Σ_j w_j (a_j − b_j), and choice is logistic in that scalar plus an attention lapse. Because the weights take only two values, tier-matched profiles cancel *exactly* and give an exact coin flip, with no validity tie-break and no take-the-best fallback. (iii) The same redundancy logic is applied to the majority itself: a feature-count majority is the most "obvious" statistic in the display, so subjects apply a *contrarian consensus discount* proportional to the count margin. (iv) Integration is capacity-limited over the *whole displayed panel*, so the effective slope is divided by a power of the panel size n, in addition to a milder within-trial spread discount over the number of discriminating cues D.

**The new commitment is that the redundancy discount is itself a comparative, graded judgement rather than a fixed property of a cue.** A star expert is discounted precisely *because other experts are seen as restating what he says*; the discount is therefore a function of how much of the panel is actually speaking on this trial. When almost nothing in the display discriminates — in the limit, when a single expert is the only one who differentiates the two products — there is no chorus for that expert to be redundant with, so his rating is taken at (nearly) full face value. The discount relaxes continuously back toward full weight as the number of discriminating cues D falls toward one: eps_eff = eps_w + (1 − eps_w)·ψ·exp(−(D−1)/κ_D), with ψ controlling how completely redundancy can be released and κ_D controlling how fast it releases. Because κ_D is small, the release is essentially confined to genuinely sparse displays. This predicts a sharp, design-level signature that no fixed-weight tiering model produces: **profiles in which exactly one high-validity expert discriminates are followed at high rates (≈0.78–0.85), even though profiles in which that same expert discriminates alongside several others are followed near chance**, since only in the latter case is the star rating construed as redundant restatement. Tally-tied, tier-matched profiles remain at exactly chance (E = 0 and margin = 0 regardless of eps_eff), and all dense profiles (D ≥ 2) are left effectively untouched.

**Parameters:**
- eps_w: [0.03, 0.34]
- beta: [2.2, 3.6]
- delta: [-0.10, 0.25]
- tau: [0.5, 1.3]
- lapse: [0.05, 0.18]
- count_bias: [0.28, 0.62]
- psi: [0.35, 0.85]
- kappa_d: [0.15, 0.45]
- validities: validities

**`predict source code`:**
```python
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Design validities (fixed by the experiment): [f0..f5]\n    val = np.array([0.95, 0.78, 0.82, 0.55, 0.76, 0.80], dtype=float)\n    n_feat = val.shape[0]\n    order = np.argsort(-val, kind='stable')\n    marginal_idx = [int(j) for j in order[n_feat // 2:]]  # bottom-half tier: f1, f4, f3\n\n    hits = 0.0\n    tot = 0.0\n\n    a_col = data['option_a_ratings'].tolist()\n    b_col = data['option_b_ratings'].tolist()\n    r_col = data['response'].tolist()\n\n    for a_raw, b_raw, r in zip(a_col, b_col, r_col):\n        try:\n            a = np.asarray(list(a_raw), dtype=float).ravel()\n            b = np.asarray(list(b_raw), dtype=float).ravel()\n        except Exception:\n            continue\n        if a.shape[0] != n_feat or b.shape[0] != n_feat:\n            continue\n        d = np.sign(a - b)\n        D = int(np.sum(d != 0))\n        if D < 2:\n            continue  # drop the solo-cue cells (K5/K6), where the two theories nearly agree\n        net_marg = float(np.sum(d[marginal_idx]))\n        if net_marg == 0.0:\n            continue\n        pred = 0 if net_marg > 0 else 1  # 0 = chose A, 1 = chose B\n        try:\n            rr = int(r)\n        except Exception:\n            continue\n        hits += 1.0 if rr == pred else 0.0\n        tot += 1.0\n\n    if tot <= 0:\n        return float('nan')\n    return float(hits / tot)",
  "rationale": "The metric is the proportion of dense-display trials (D>=2 discriminating cues) on which the subject chooses the option favoured by the net vote of the LOW-validity (bottom-half) experts f1(.78), f4(.76), f3(.55). This single number is defined stimulus-wise (no trial indices needed), applies to 64 of the 96 trials per subject, and is well defined for a single subject's slice, so it is tight per subject (mostly binomial noise over 64 trials plus modest parameter spread).\n\nWhy it discriminates: RDTI-MCR gives the whole bottom tier weight 1 and the star tier weight eps_w<=0.34, and additionally subtracts a contrarian majority discount, so on all four dense cells it follows the marginal-tier-backed option: K1 ~0.86, K2 ~0.93, K3 ~0.81, K4 ~0.84 (adherence 0.75-0.95 over its whole parameter box) => pooled value ~0.86.\n\nGVIT-CAP-S4 behaves very differently on exactly these trials. On the two tally-tied cells (K1, K2, 32 trials) its count gate returns EXACTLY 0.5 for every parameter setting, so adherence is chance. On K3 its strongly saturating inversion makes f1/f4 nearly as discounted as the stars while the lone floor expert f3 keeps full weight, reversing the choice (adherence ~0.12). On K4 the floor expert again pulls toward the star-backed side (adherence ~0.38). Pooled value ~0.37-0.40.\n\nExpected gap ~0.48 with per-subject variance on the order of 0.005 for both theories, giving a Welch t of roughly 20 at N=25. Crucially the metric is not trivially true for RDTI: any model that weights cues monotonically in stated validity (the normative/classic weighted-additive account) would score far BELOW 0.5, and GVIT's own inverse-validity account scores below 0.5 too; only RDTI's coarse two-tier split at the .80/.78 boundary plus majority discounting produces a high value."
}
```

## Usage

```json
{
  "input_tokens": 13547,
  "output_tokens": 5530
}
```
