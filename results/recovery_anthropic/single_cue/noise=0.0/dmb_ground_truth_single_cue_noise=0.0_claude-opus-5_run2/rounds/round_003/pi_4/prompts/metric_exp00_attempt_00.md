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
**Validities (n_features=8):** [0.95, 0.56, 0.85, 0.6, 0.65, 0.9, 0.53, 0.8]

**Trial pairs (n=12):**
  trial 1: A=[0, 1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 1, 0]
  trial 2: A=[1, 0, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0, 1, 0]
  trial 3: A=[0, 1, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1, 0, 0]
  trial 4: A=[1, 0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0, 1, 0]
  trial 5: A=[1, 1, 1, 0, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 1, 0, 1]
  trial 7: A=[0, 1, 1, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 1, 0, 0]
  trial 8: A=[1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 0, 1, 0]
  trial 9: A=[1, 0, 1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 0, 0, 1]
  trial 10: A=[0, 0, 1, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1, 0, 1]
  trial 11: A=[1, 1, 1, 0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1, 0]
  trial 12: A=[0, 1, 0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 0, 1, 1, 0]

**Rationale:** GOAL: dissociate Count-Direction with Conflict-Graded Confidence (CDCC, advocated) from Redundancy-Discounted Tiered Integration with Majority Discounting (RDTI-MCR, competing). CDCC makes DIRECTION purely count-based (exact 0.5 on tally ties, parameter-free; never sub-chance adherence to the count winner) and lets validity enter only as a confidence gate, with dilution S = M/D^gamma. RDTI-MCR median-splits the panel into a discounted 'star' tier and a full-weight 'marginal' tier, adds a contrarian discount proportional to the count margin, and releases the star discount only for essentially solo displays (D=1). This design attacks the three points where their signs, not just their magnitudes, diverge.

PANEL: 8 experts, validities in scrambled screen order [0.95, 0.56, 0.85, 0.60, 0.65, 0.90, 0.53, 0.80]. Descending rank f0(.95) > f5(.90) > f2(.85) > f7(.80) >> f4(.65) > f3(.60) > f1(.56) > f6(.53), so RDTI's median split is unambiguous (STAR = {f0,f5,f2,f7}, MARGINAL = {f4,f3,f1,f6}) with a wide .80/.65 gap at the split, and CDCC's convex weights w=(v-.5)^3 are .091/.064/.043/.027 for the stars versus .0034/.0010/.0002/.00003 for the marginals (>25:1), so its coherence gate is star-driven. n=8 (rather than 6) also puts RDTI's capacity term (n/6)^tau above 1, which shrinks its evidence term relative to the count-margin discount and sharpens the predicted sign reversals. 6 configurations x 2 side mirrors = 12 unique pairs x 8 repetitions = 96 trials.

CELLS (A-favouring version listed; the next trial is its exact A/B mirror):
C1 (T1/T2) TALLY-TIED 3v3, TIER-MISMATCHED: A wins marginals f4,f3,f1; B wins stars f0,f5,f2; f6,f7 tied.
C2 (T3/T4) TALLY-TIED 2v2, TIER-MISMATCHED: A wins the two weakest experts f6(.53), f1(.56); B wins the two best f0(.95), f5(.90); f2,f3,f4,f7 tied.
C3 (T5/T6) DENSE STAR-MAJORITY (D=6, M=+2): A wins all four stars f0,f5,f2,f7; B wins marginals f4,f3; f1,f6 tied.
C4 (T7/T8) DENSE MARGINAL-MAJORITY (D=6, M=+2): A wins all four marginals f4,f3,f1,f6; B wins stars f0,f5; f2,f7 tied.
C5 (T9/T10) SPARSE UNANIMOUS STAR (D=2, M=+2): A wins only f0(.95) and f5(.90); six cues tied.
C6 (T11/T12) MID STAR-MAJORITY (D=4, M=+2): A wins stars f0,f5,f2; B wins marginal f4; f1,f3,f6,f7 tied.

QUANTITATIVE PREDICTIONS, P(choose A).
CDCC (beta=2.6, gamma=1.5, kappa=0.7, rho=3.0, eps=0.10): C1 = 0.500, C2 = 0.500, C3 = 0.650, C4 = 0.532, C5 = 0.928, C6 = 0.754.
RDTI-MCR (eps_w=.18, beta=2.9, delta=.075, tau=.9, lapse=.115, count_bias=.45, psi=.60, kappa_d=.30): C1 = 0.935, C2 = 0.911, C3 = 0.086, C4 = 0.941, C5 = 0.487, C6 = 0.180.

THREE ORTHOGONAL DIAGNOSTICS:
(1) TALLY-TIED TIER-MISMATCHED CELLS (C1, C2). CDCC MUST be exactly 0.5 for every beta, gamma, kappa, rho, epsilon (nA = nB returns the coin flip before any validity computation). RDTI predicts strong, systematic choice of the option backed only by the LOW-validity marginal experts (0.85-0.97 across its whole parameter box, since E = k(1 - eps_w) > 0 with eps_w <= 0.34 and the count-margin term is exactly zero). Each subject contributes 32 such trials, so even a 10-point deviation from 0.5 is decisive; note the sign also separates RDTI from WADD/TTB, which would predict below 0.5 here, so the cell is self-diagnostic three ways.
(2) SIGN REVERSAL ON STAR-BACKED MAJORITIES (C3 and C6 versus C4). C3, C4 and C6 all have count margin +2 for A. CDCC must put ALL THREE above 0.5 (conflict can only erode confidence toward chance, never past it) and predicts the coherent star-backed cells ABOVE the conflict cell: C3 0.65 and C6 0.75 versus C4 0.53. RDTI predicts C3 = 0.09 and C6 = 0.18 (sub-chance: 4*eps_w <= 1.36 < 2 so the tier-weighted evidence is negative whenever the count winner's wins are the stars, and the contrarian majority discount pushes it further down), while C4 = 0.94 -- an ~0.85 swing between profiles CDCC treats as same-direction and similarly signed. Observing systematic sub-chance adherence to the count winner in C3/C6 falsifies CDCC outright; observing C3 > C4 > 0.5 falsifies RDTI.
(3) DILUTION LADDER AT CONSTANT MARGIN AND CONSTANT COHERENCE (C5, C6, C3: D = 2, 4, 6 with M = +2 and the count winner always holding the stars). CDCC's S = M/D^gamma predicts a steep monotone decline 0.93 -> 0.75 -> 0.65, with near-ceiling (0.90-0.95) adherence on the sparse unanimous star cell C5 -- the classic CDCC signature that sparse clean evidence beats dense larger-D evidence. RDTI predicts the same ordering but hugging or crossing below chance (0.49 -> 0.18 -> 0.09), because its graded redundancy release is essentially confined to D = 1 (relax = psi*exp(-(D-1)/kappa_d) is already <= 0.03 at D = 2) while the count-margin discount of 0.45*2 applies in all three cells. Thus C5 is the cleanest magnitude test: CDCC needs >= 0.85, RDTI at nominal parameters sits at chance and only reaches ~0.8 at the extreme corner of its box.

CONTROLS: every configuration appears in an A-favouring and a B-favouring mirror, so side bias cancels exactly; tied cues are coded sometimes 1-1 and sometimes 0-0 so the number of displayed 1s does not track composition or tier; validity rank is scrambled relative to screen position so positional heuristics cannot mimic tiering; no pair is degenerate (every trial requires reading several ratings) and no cell has the two models making the same prediction.

INTERPRETATION: exact chance on C1/C2, above-chance adherence to the count winner in every one of C3-C6 with the coherent cells highest, and a steep sparse-to-dense decline 0.93/0.75/0.65 at fixed margin uniquely supports CDCC. Strong marginal-tier-driven preferences on the tally-tied cells plus sub-chance choice of the star-backed majority in C3/C6 uniquely supports RDTI-MCR.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** **Count-Direction with Conflict-Graded Confidence (CDCC).**

When people compare two multi-attribute options described by binary expert ratings, the *direction* of their preference and the *confidence* with which they execute it are computed by two functionally separate processes.

1. **Direction is purely count-based.** The comparator registers, for each attribute, only which option is rated higher, and the option that wins on more attributes becomes the candidate response. Validities never enter this stage: they cannot promote a minority-supported option to candidate status. When the two win-counts are equal, the comparator returns no candidate at all, and the subject flips a mental coin — exactly 0.5, with **no** validity-ordered tie-break. This is a hard, parameter-free prediction: on tally-tied, side-counterbalanced profiles CDCC predicts chance adherence to the most-valid discriminating cue, regardless of any parameter setting.

2. **Confidence, not direction, carries the validity information, and it is graded by evidence dilution and by cue conflict.** The probability of actually executing the count-winner is a saturating (tanh) function of a scalar confidence signal built from three ingredients: (i) the raw win margin M = |nA − nB|; (ii) *dilution* — the margin is discounted by the total number of discriminating cues raised to a power gamma, so that a 6-vs-2 split feels much weaker than a 4-vs-0 split even though both have M = 4 (evidence spread thin across many mutually contradicting attributes is subjectively less compelling than a clean, sparse, unanimous split); and (iii) *coherence* — the signed, normalised balance of validity weight between the cues the count-winner wins and the cues the loser wins, where the subjective weight of a cue is a convex (super-linear) function of its stated validity advantage, w_j = (v_j − 0.5)^rho. Convexity means the coherence signal is dominated by the few high-validity cues: when the best cue points *with* the count the subject feels certain, when it points *against* the count the subject feels torn even though the count itself is unambiguous.

3. **Conflict erodes confidence but never reverses direction.** The confidence argument is floored at zero, so validity conflict can drive behaviour arbitrarily close to guessing but never below chance: CDCC never makes systematic one-reason (Take-The-Best) choices. This is what separates it from any Tallying+TTB mixture: on high-margin trials where the top cue opposes the tally, people move toward 0.5 from above, not past it.

4. **Lapses.** With probability epsilon the entire decision is replaced by a coin flip (attention lapse), which caps adherence below 1 even on maximally coherent trials.

The theory therefore predicts a specific cross-design signature: chance performance on tally-tied profiles; high but sub-ceiling adherence on sparse, coherent, unanimous profiles; noticeably lower adherence on dense profiles where the loser also wins several cues even when the raw margin is larger; and the lowest (but still above-chance) adherence when the most valid cue opposes the count.

**Parameters:**
- beta: [1.6, 3.6]
- gamma: [1.25, 1.75]
- kappa: [0.4, 1.0]
- rho: [2.2, 3.8]
- epsilon: [0.02, 0.18]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 1:
        stim = stim.reshape(2, -1)
    if stim.ndim != 2 or stim.shape[0] != 2:
        stim = stim.reshape(2, -1)
    a = stim[0]
    b = stim[1]
    n = int(a.shape[0])

    if n == 0:
        return np.array([0.5, 0.5], dtype=float)

    a_win = a > b
    b_win = b > a
    nA = int(np.sum(a_win))
    nB = int(np.sum(b_win))
    D = nA + nB

    # --- Commitment 1: direction is purely count-based; ties -> exact guess ---
    if D == 0 or nA == nB:
        return np.array([0.5, 0.5], dtype=float)

    winner = 0 if nA > nB else 1
    M = float(abs(nA - nB))

    # --- parameters -----------------------------------------------------
    beta = float(parameters.get("beta", 2.6))
    gamma = float(parameters.get("gamma", 1.5))
    kappa = float(parameters.get("kappa", 0.7))
    rho = float(parameters.get("rho", 3.0))
    eps = float(parameters.get("epsilon", 0.10))

    beta = min(max(beta, 0.0), 20.0)
    gamma = min(max(gamma, 0.0), 3.0)
    kappa = min(max(kappa, 0.0), 5.0)
    rho = min(max(rho, 0.5), 8.0)
    eps = min(max(eps, 0.0), 0.6)

    # --- validity weights (convex in validity advantage) ------------------
    v_raw = parameters.get("validities", None)
    v = None
    if v_raw is not None:
        try:
            v = np.asarray(v_raw, dtype=float).ravel()
        except Exception:
            v = None
    if v is None or v.shape[0] != n:
        # fallback: assume features are ordered by descending validity
        v = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.75])

    w = np.clip(v - 0.5, 1e-9, None) ** rho

    if winner == 0:
        W_win = float(np.sum(w[a_win]))
        W_lose = float(np.sum(w[b_win]))
    else:
        W_win = float(np.sum(w[b_win]))
        W_lose = float(np.sum(w[a_win]))

    tot = W_win + W_lose
    C = 0.0 if tot <= 0.0 else (W_win - W_lose) / tot   # in [-1, 1]

    # --- Commitment 2: dilution-discounted margin, coherence-gated --------
    S = M / (float(D) ** gamma)
    gate = (1.0 + kappa * C) / (1.0 + kappa)             # 1.0 when fully coherent
    arg = beta * S * gate

    # --- Commitment 3: conflict erodes confidence, never reverses it ------
    if arg < 0.0:
        arg = 0.0
    if arg > 30.0:
        arg = 30.0

    conf = float(np.tanh(arg))

    # --- Commitment 4: lapse ---------------------------------------------
    p_winner = 0.5 + 0.5 * (1.0 - eps) * conf
    p_winner = float(min(max(p_winner, 0.5), 1.0 - 1e-9))

    p = np.empty(2, dtype=float)
    p[winner] = p_winner
    p[1 - winner] = 1.0 - p_winner

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
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Design-time validities (fixed by the experiment), used only to define the\n    # median-split 'star' tier structurally.\n    v = np.array([0.95, 0.56, 0.85, 0.60, 0.65, 0.90, 0.53, 0.80], dtype=float)\n    n = v.shape[0]\n    order = np.argsort(-v, kind='stable')\n    star = set(int(i) for i in order[: n // 2])\n\n    hits = []\n    for a_raw, b_raw, resp in zip(data['option_a_ratings'],\n                                  data['option_b_ratings'],\n                                  data['response']):\n        try:\n            a = np.asarray(list(a_raw), dtype=float).ravel()\n            b = np.asarray(list(b_raw), dtype=float).ravel()\n        except Exception:\n            continue\n        if a.shape[0] != n or b.shape[0] != n:\n            continue\n        d = np.sign(a - b)\n        nA = int(np.sum(d > 0))\n        nB = int(np.sum(d < 0))\n        D = nA + nB\n        # dense / mid displays only (D >= 4), with a strict count winner\n        if D < 4 or nA == nB:\n            continue\n        winner = 0 if nA > nB else 1\n        if winner == 0:\n            win_idx = set(int(i) for i in np.where(d > 0)[0])\n            lose_idx = set(int(i) for i in np.where(d < 0)[0])\n        else:\n            win_idx = set(int(i) for i in np.where(d < 0)[0])\n            lose_idx = set(int(i) for i in np.where(d > 0)[0])\n        # keep only STAR-BACKED majorities: the count winner's wins are all\n        # high-validity (star) experts, the count loser's wins are all\n        # low-validity (marginal) experts.\n        if not win_idx.issubset(star):\n            continue\n        if len(lose_idx & star) > 0:\n            continue\n        try:\n            r = int(resp)\n        except Exception:\n            continue\n        hits.append(1.0 if r == winner else 0.0)\n\n    if len(hits) == 0:\n        return 0.5\n    return float(np.mean(hits))",
  "rationale": "This is the sign-reversal diagnostic (design cells C3 and C6): dense/mid displays (D>=4) with a feature-count majority whose supporting cues are exactly the high-validity 'star' experts while the count loser is supported only by low-validity marginal experts. The metric is the proportion of such trials on which the subject chooses the count winner (32 of 96 trials per subject, so binomial noise per subject is ~0.008 in variance).\n\nCDCC makes direction purely count-based and floors the confidence argument at zero, so adherence must be strictly above 0.5 here, and because coherence is maximal (all star weight on the winner's side) the gate is near 1: across its whole parameter box the predicted adherence is roughly 0.57-0.92, centred near 0.70-0.75.\n\nRDTI-MCR discounts precisely the star tier (residual weight eps_w<=0.34, and its graded redundancy release is inert at D>=2) and additionally applies a contrarian discount proportional to the count margin, so the tier-weighted evidence 4*eps_w-2 (C3) and 3*eps_w-1 (C6) is negative and the majority discount pushes further down: predicted adherence is roughly 0.02-0.35, centred near 0.13.\n\nSo the two theories are separated not just in magnitude but in sign relative to chance (~0.73 vs ~0.15), with tight per-subject estimates (32 trials each and a narrow parameter-induced spread), which should yield a very large Welch t at N=25. It is not trivially true of CDCC: any model that lets high-validity cues drive direction (WADD/TTB) would also score high here, and any model that discounts them scores low; the metric measures a directional signature, not a parameter of my model."
}
```

## Usage

```json
{
  "input_tokens": 12302,
  "output_tokens": 5195
}
```
