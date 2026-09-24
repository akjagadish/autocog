# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_7" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_7" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_7
**MDSC-NV (sharpened) — Maximum-Doubt Single-Cue selection with Null-Verdict capture.**

Four claims; the mechanism family is unchanged from the accepted base, but one quantitative commitment is sharpened.

**(1) One-reason choice by MAXIMUM DOUBT (no summation).** On each trial the subject settles A-vs-B with the verdict of a SINGLE expert. Which expert is consulted is decided by *doubt*: a disagreement voiced by a near-chance expert (v≈.53) is a strange, newsworthy fact that has to be adjudicated, whereas a disagreement voiced by a near-certain expert (v≈.97) is discounted as a mere echo of the impression the subject already has. Salience is log u_j = delta*log(1-v_j) + lam*j and the consulted row is drawn from softmax(u). **The doubt gain is LARGER than previously assumed (delta ≈ 13-17, not 8-12): the doubt rule is essentially argmax, not a graded share.** This is the sharpened commitment. It matters wherever two moderately-doubtful rows are within ~0.4 log-validity of one another (e.g. v=.55 vs v=.70 rows two columns apart): a share-based rule lets the second-most-doubtful row steal ~18% of the verdict and blunts the choice, whereas the near-argmax rule leaves the display fully decided by the single most doubtful row. Coalitions still buy nothing: five agreeing rows never outvote one more-doubtful row, which is why choice is systematically anti-validity in every design where the printed labels are usable.

**(2) Reading position is a WEAK tie-breaker.** lam = pos_ratio*delta with pos_ratio ≈ 0.09-0.14 — an order of magnitude below the doubt term's dynamic range. Position therefore only arbitrates between rows whose stated validities are (nearly) *equal* — exactly the .955/.950/.945 and .958/.954/.950 designs, where it makes the last low-validity row win. Whenever validities genuinely differ, doubt beats recency: a low-validity EARLY column defeats a late .93/.99 column. Because delta is raised and pos_ratio held fixed, the *ratio* of doubt-range to positional gain is preserved, so all orderings in the corpus are unchanged.

**(3) NULL-VERDICT CAPTURE.** Doubt is a property of the *expert*, not of the disagreement, so attention is captured pre-attentively by the most doubtful expert **in the whole display**, before the subject checks whether that expert actually discriminates. With probability q (≈ .41, slightly higher than previously estimated) the subject simply goes with whoever captured attention; if that expert gave both products the same rating it returns no verdict, the subject has nothing to adjudicate and guesses. With probability 1-q the subject re-targets deliberately, restricting the doubt competition to the rows that differ. Sharp prediction: *holding the discriminating rows fixed, adding agreement on a MORE doubtful (lower-validity) row pushes choice toward chance*, whereas adding agreement on a high-validity row does nothing. Displays in which only two high/mid-validity rows disagree while the display's most doubtful expert stays silent are therefore near chance — no compensatory or one-reason model predicts that non-discriminating rows matter at all.

**(4) LABEL-DEGENERACY GATE (all-or-none) and LOAD LAPSE.** The doubt rule presupposes that each printed validity picks out one expert. If any label is printed twice (tolerance .002 — it must NOT fire for the .004/.005-gap designs) the experts are interchangeable, the doubt ordering is unavailable, and choice collapses to a graded unweighted row tally (exact coin flip on 1-1 splits). Independently, the more rows disagree the more often the subject loses the thread: p(guess) = 1-(1-eps0)*rho^(m-2), which compresses conflict displays toward chance and keeps even unanimous 6-row displays well below ceiling. Individual subjects differ mainly in how leaky (eps0, rho) and how capture-prone (q) they are, not in the doubt rule itself.

No feedback is given, so the rule is stationary across the block.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # MDSC-NV: Maximum-Doubt Single-Cue selection with Null-Verdict capture
    import numpy as np

    # ---------- parse the (2, n_features) stimulus robustly ----------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(state['option_a_ratings'], dtype=float).ravel()
            b = np.asarray(state['option_b_ratings'], dtype=float).ravel()
        else:
            vals = list(state.values())
            if len(vals) >= 2:
                a = np.asarray(vals[0], dtype=float).ravel()
                b = np.asarray(vals[1], dtype=float).ravel()
    if a is None or b is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 2:
            a, b = arr[0].ravel(), arr[1].ravel()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            a, b = arr[:, 0].ravel(), arr[:, 1].ravel()
        elif arr.ndim == 1:
            h = arr.shape[0] // 2
            a, b = arr[:h], arr[h:2 * h]
        else:
            f = np.asarray(arr, dtype=float).reshape(2, -1)
            a, b = f[0].ravel(), f[1].ravel()

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float)[:n]
    b = np.asarray(b, dtype=float)[:n]

    # ---------- communicated validities (the printed labels) ----------
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is None:
        v = np.linspace(0.95, 0.55, n)
    else:
        v = np.asarray(v_raw, dtype=float).ravel()
        if v.size < n:
            v = np.concatenate([v, np.full(n - v.size, 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5, 0.9975)

    # ---------- parameters ----------
    delta = float(parameters['doubt_gain'])
    lam = float(parameters['pos_ratio']) * delta        # weak positional gain
    q = float(np.clip(float(parameters['capture']), 0.0, 1.0))
    eps0 = float(np.clip(float(parameters['eps0']), 0.0, 1.0))
    rho = float(np.clip(float(parameters['rho']), 1e-6, 1.0))
    kap = float(parameters['tally_kappa'])

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])
    fav_a = diff[disc] > 0.0
    n_a = float(np.sum(fav_a))
    n_b = float(m) - n_a

    # ---------- label-degeneracy gate (all-or-none, exact repetition) ----------
    degenerate = False
    if n > 1:
        vs = np.sort(v)
        if bool(np.any(np.diff(vs) < 2.0e-3)):
            degenerate = True

    if degenerate:
        # experts cannot be individuated -> unweighted graded row tally
        z = float(np.clip(kap * (n_a - n_b), -60.0, 60.0))
        p_core_a = 1.0 / (1.0 + np.exp(-z))
    else:
        # doubt salience over ALL rows
        logu = delta * np.log(np.clip(1.0 - v, 1e-12, None)) + lam * np.arange(n, dtype=float)

        lg = logu - np.max(logu)
        s = np.exp(lg)
        ssum = float(np.sum(s))
        s = s / ssum if (np.isfinite(ssum) and ssum > 0.0) else np.full(n, 1.0 / n)

        ld = logu[disc]
        ld = ld - np.max(ld)
        t = np.exp(ld)
        tsum = float(np.sum(t))
        t = t / tsum if (np.isfinite(tsum) and tsum > 0.0) else np.full(m, 1.0 / m)

        # (i) deliberate re-targeting: doubt competition among disagreeing rows
        p_rest_a = float(np.sum(t[fav_a]))

        # (ii) null-verdict capture: attention grabbed by the most doubtful
        #      expert in the whole display; if it is silent -> guess
        mask = np.zeros(n, dtype=bool)
        mask[disc] = True
        s_nondisc = float(np.sum(s[~mask]))
        s_disc = s[disc]
        p_glob_a = float(np.sum(s_disc[fav_a])) + 0.5 * s_nondisc

        p_core_a = q * p_glob_a + (1.0 - q) * p_rest_a

    # ---------- load-dependent lapse ----------
    expo = max(m - 2, 0)
    g = 1.0 - (1.0 - eps0) * (rho ** expo)
    g = float(min(max(g, 0.0), 1.0))

    p_a = 0.5 * g + (1.0 - g) * p_core_a
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_6
SSI-DS+ (Strategy Selection by Label-Code Injectivity, with Distinctive-Cue Summation and Competition-Scaled Interference).

(1) STRATEGY SWITCH. Before the block the subject reads the printed validity list as a naming code. If all labels are distinct printed symbols (0.955 vs 0.950 counts as distinct) the experts are individuable and the subject adopts a WEIGHTED mode; if any label is repeated the experts are interchangeable and the subject adopts an unweighted MAJORITY TALLY mode. The switch is all-or-none at the design level (only a small per-trial slip probability omega), not a graded function of numeric gap size.

(2) WEIGHTED MODE = COMPENSATORY SUMMATION over all discriminating rows: E = sum_j sign(a_j-b_j) * (1-v_j)^gamma * exp(lam*j), rescaled so the strongest discriminating row has magnitude 1, then p(A) = sigmoid(beta*E). Validity enters with INVERTED sign (near-certain experts are redundant echoes of the general impression; near-chance disagreements are the news that must be adjudicated) and position enters as a reading-order recency gain. Coalitions ADD, so several early distinctive rows can outvote a single later row.

(3) TALLY MODE = graded majority p(A) = sigmoid(kappa*(n_A - n_B)): 1-1 splits give exactly 0.50, 2-1 splits ~0.77 (harder than probability matching, softer than a hard majority rule).

(4) CONFLICT INTERFERENCE IS COMPETITION-SCALED (the refinement). Guessing is driven by the load of holding several discrepancies in mind while ADJUDICATING BETWEEN THEM. The load is therefore not the raw number of discriminating rows but the number of rows that actually have to be pitted against one another: when the display is unanimous (min(n_A,n_B) = 0, i.e. every discrepancy points the same way) there is nothing to adjudicate and the extra rows impose only a discounted reading load eta*(m-2) rather than the full (m-2). Formally g = 1 - (1-eps0) * rho^L with L = (m-2) on opposed displays and L = eta*(m-2) on unanimous displays, and the core probability is mixed toward 0.5 by g. This predicts that dominance accuracy degrades with the number of agreeing experts only mildly, whereas conflict displays degrade steeply with the number of opposed experts - the two are dissociated, not a single monotone function of display size.

No feedback, hence no learning; the rule is stationary across the block.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np

    # ---------- parse the (2, n_features) stimulus robustly ----------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(state['option_a_ratings'], dtype=float).ravel()
            b = np.asarray(state['option_b_ratings'], dtype=float).ravel()
        else:
            vals = list(state.values())
            if len(vals) >= 2:
                a = np.asarray(vals[0], dtype=float).ravel()
                b = np.asarray(vals[1], dtype=float).ravel()
    if a is None or b is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 2:
            a, b = arr[0].ravel(), arr[1].ravel()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            a, b = arr[:, 0].ravel(), arr[:, 1].ravel()
        elif arr.ndim == 1:
            h = arr.shape[0] // 2
            a, b = arr[:h], arr[h:2 * h]
        else:
            flat = np.asarray(arr, dtype=float).reshape(2, -1)
            a, b = flat[0].ravel(), flat[1].ravel()

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float)[:n]
    b = np.asarray(b, dtype=float)[:n]

    # ---------- communicated validities (the printed label code) ----------
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is None:
        v = np.linspace(0.95, 0.55, n)
    else:
        v = np.asarray(v_raw, dtype=float).ravel()
        if v.size < n:
            v = np.concatenate([v, np.full(n - v.size, 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5, 0.9995)

    # ---------- parameters ----------
    gamma = float(parameters['gamma'])
    lam = float(parameters['lam'])
    beta = float(parameters['beta'])
    kappa = float(parameters['kappa'])
    eps0 = float(np.clip(float(parameters['eps0']), 0.0, 1.0))
    rho = float(np.clip(float(parameters['rho']), 1e-6, 1.0))
    omega = float(np.clip(float(parameters['omega']), 0.0, 0.5))
    eta = float(np.clip(float(parameters['eta']), 0.0, 1.0))

    # ---------- STRATEGY SWITCH: injectivity of the printed label code ----------
    LABEL_TOL = 2.5e-3   # two printed labels closer than this are the same symbol
    if n <= 1:
        injective = True
    else:
        vs = np.sort(v)
        gaps = np.diff(vs)
        injective = bool(np.all(gaps > LABEL_TOL))

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])
    signs = np.sign(diff[disc])

    # ---------- weighted mode: distinctive-cue SUMMATION ----------
    # log weight = gamma*log(1 - v_j) + lam*j ; rescaled so strongest row = 1
    logw = gamma * np.log(np.clip(1.0 - v[disc], 1e-9, None)) + lam * disc.astype(float)
    logw = logw - np.max(logw)
    w = np.exp(logw)
    E = float(np.sum(signs * w))
    z = float(np.clip(beta * E, -60.0, 60.0))
    p_weighted_a = 1.0 / (1.0 + np.exp(-z))

    # ---------- tally mode: graded majority ----------
    n_a = float(np.sum(signs > 0.0))
    n_b = float(m) - n_a
    zt = float(np.clip(kappa * (n_a - n_b), -60.0, 60.0))
    p_tally_a = 1.0 / (1.0 + np.exp(-zt))

    if injective:
        p_core_a = (1.0 - omega) * p_weighted_a + omega * p_tally_a
    else:
        p_core_a = (1.0 - omega) * p_tally_a + omega * p_weighted_a

    # ---------- conflict interference / lapse (competition-scaled load) ----------
    # load counts only rows that must be ADJUDICATED against one another:
    # unanimous (dominance) displays impose only a discounted reading load.
    conflict = float(min(n_a, n_b))
    expo = float(max(m - 2, 0))
    if conflict <= 0.0:
        expo = eta * expo
    g = 1.0 - (1.0 - eps0) * (rho ** expo)
    g = float(min(max(g, 0.0), 1.0))

    p_a = 0.5 * g + (1.0 - g) * p_core_a
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_7)

### DESIGN
**Validities (n_features=6):** [0.55, 0.9, 0.75, 0.95, 0.68, 0.85]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 2: A=[1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1]
  trial 3: A=[1, 0, 1, 0, 0, 1]  B=[0, 0, 0, 0, 0, 1]
  trial 4: A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  trial 5: A=[0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 6: A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  trial 7: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  trial 9: A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1, 0]
  trial 10: A=[1, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  trial 11: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1]
  trial 12: A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  trial 13: A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 14: A=[1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 16: A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]

**Rationale:** GOAL: separate MDSC-NV (advocated) from SSI-DS+ (competitor). Both models invert validity (near-chance experts carry the decision) and both add a reading-position gain, so they can only be separated on three axes: (A) DOUBT GAIN MAGNITUDE — MDSC-NV uses (1-v)^delta with delta=13-17, SSI-DS+ uses (1-v)^gamma with gamma~0.9, i.e. ~17x weaker, so with comparable positional gains (lam_MDSC = pos_ratio*delta ~ 1.2-2.4 vs lam_SSI ~ 1.5) the two models put their weight on DIFFERENT rows: MDSC is governed by the lowest-validity row wherever it sits, SSI is governed by the LAST row unless it is extremely non-distinctive; (B) SINGLE-CUE vs SUMMATION — MDSC draws one row and cannot let a coalition add up, SSI sums and passes the max-normalised sum through a sigmoid; (C) NULL-VERDICT CAPTURE — MDSC says non-discriminating rows matter (with prob q~.41 attention is captured by the most doubtful expert in the WHOLE display, and if that expert is silent the subject guesses), whereas SSI (like every compensatory or one-reason model) says silent rows are strictly irrelevant.

VALIDITIES [0.55, 0.90, 0.75, 0.95, 0.68, 0.85]: scrambled so validity does not track column order; the two most doubtful experts sit at column 0 (.55, FIRST) and column 4 (.68, late), the two most certain at columns 1 (.90) and 3 (.95). All pairwise gaps >= .05, far above both models' label tolerances (.002 / .0025), so neither model is in tally/degenerate mode — the contrast is purely mechanistic. MDSC log-salience (delta=15, lam=1.725): col4 -10.2 > col0 -12.0 > col2 -17.3 > col5 -19.8 > col1 -32.8 > col3 -39.8. SSI log-weight (gamma=.9, lam=1.5): col5 5.79 > col4 4.98 > col3 1.80 ~ col2 1.75 >> col1 -0.57 ~ col0 -0.72. Note the near-perfect crossing of the two orderings at the extremes (col0 is MDSC's #2 and SSI's last; col5 is SSI's #1 and MDSC's #4).

8 base structures, each with its exact A/B mirror (16 unique pairs, K=6 -> 96 trials), so both models' predicted responses are perfectly key-balanced; non-discriminating rows carry matched 1s AND matched 0s so displays are dense.

BLOCK 1 — CROWN JEWEL: NULL-VERDICT CAPTURE ON UNANIMOUS m=2 DISPLAYS (bases 1-3). All three are 2-0 dominance displays that differ ONLY in which rows are silent. SSI must predict the same saturated value (~.88) on all three, because E is computed over discriminating rows only and a unanimous 2-row display always gives E in (1,2). MDSC predicts a graded decline set by how much global doubt mass sits on SILENT rows: base 1 (disc = cols 0 & 4, the two most doubtful rows -> s_nondisc ~ 0) ~ .88; base 3 (disc = cols 0 & 2, so the globally most doubtful expert col4 is silent -> s_nondisc ~ .86) ~ .75; base 2 (disc = cols 1 & 3, the two highest-validity rows, BOTH doubtful experts silent -> s_nondisc ~ 1) ~ .72. A flat .88/.88/.88 profile falsifies null-verdict capture; a 15-16 point drop from base 1 to base 2 with identical 2-0 dominance structure is producible by no compensatory or standard one-reason model. Base 1 simultaneously pins eps0 for both models.

BLOCK 2 — DOUBT vs RECENCY DIRECTIONAL REVERSALS (bases 4-5, m=2 splits).
- Base 4: A wins col0 (.55, first column), B wins col5 (.85, last column). MDSC: the .55 row wins the doubt competition by ~8 log units -> p(A) ~ .75. SSI: col5's weight is ~670x col0's -> E ~ -1 -> p(A) ~ .14. ~60-point reversal.
- Base 5: A wins col1 (.90), B wins col3 (.95), with both doubtful rows silent. MDSC: doubt still picks col1 (.90 is 2x as doubtful as .95) but capture drags it toward chance -> p(A) ~ .72. SSI: position dominates -> p(A) ~ .15. ~57-point reversal. Bases 4 and 5 jointly estimate the delta/lam ratio: only a doubt gain above ~10 can make an EARLY .55 (or a mid-display .90) row beat a late row.

BLOCK 3 — COALITIONS BUY NOTHING (bases 6, 8).
- Base 6 (m=6): A wins col4 alone (.68, the globally most doubtful expert); B wins the other five rows. MDSC: one more-doubtful row outvotes five -> p(A) ~ .62 (above chance) despite being outvoted 1-vs-5. SSI: the summed coalition (dominated by col5) wins -> p(A) ~ .39 (below chance). A clean directional split at a 1-vs-5 row count.
- Base 8 (m=4, 2-vs-2): A wins cols 0 & 4 (the two most doubtful), B wins cols 3 & 5 (SSI's heavy late rows). MDSC ~ .75; SSI ~ .32 (its summed weight is dominated by col5). ~43-point reversal at an exact row-count tie, which no tally account can produce either way.

BLOCK 4 — UNANIMITY DECAY / eta TEST (base 7): A wins all six rows. MDSC's raw-m interference gives ~.66; SSI's competition-scaled load (eta discount, nothing to adjudicate) gives ~.69. Together with base 1 (~.88 for both) this pins the lapse parameters free of the contrast and gives a mild test of the eta discount.

INTERPRETATION: if unanimous 2-0 displays vary by ~15 points purely as a function of WHICH experts are silent (base1 > base3 > base2), if the early .55 row and the .90 row beat the late .85/.95 rows (bases 4-5 above chance), and if the lone doubtful row beats a five-row coalition (base 6 above chance) and the doubt pair beats the late pair (base 8 ~ .75), MDSC-NV is confirmed and SSI-DS+ is falsified. If instead all three unanimous m=2 displays sit flat at ~.88, and bases 4, 5, 6, 8 all land in the .14-.39 range (the option holding the LATE rows being chosen), compensatory summation over discriminating rows with an all-or-none injectivity switch wins and the null-verdict-capture claim is dead.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The metric is a pure choice-key agreement score restricted to the trial types on which the two theories make OPPOSITE binary predictions. For each trial I recompute (a) MDSC-NV's near-argmax maximum-doubt winner (the verdict of the single most doubtful discriminating expert, log u = 15*log(1-v) + 1.725*j) and (b) SSI-DS+'s compensatory summation winner (sign of sum of sign*(1-v)^0.9*exp(1.5*j)). On this design the two orderings cross at the extremes (col0, v=.55, is MDSC's #2 but SSI's last; col5, v=.85, is SSI's #1 but MDSC's #4), so 8 of the 16 pairs (trials 4,5,6,8 and their exact mirrors 12,13,14,16 => 48 of 96 trials per subject) are diagnostic and key-balanced (4 favour A, 4 favour B), so response bias cannot inflate the score. Hand-computing the generative probabilities: MDSC-NV gives p(its own key) ~ .75 (t4/t12), ~.72 (t5/t13), ~.62 (t6/t14, the lone doubtful row beating a 5-row coalition), ~.75 (t8/t16), i.e. an expected metric value ~.71; SSI-DS+ gives ~.14, ~.15, ~.39, ~.32, i.e. ~.25. Gap ~.46. Because the score averages 48 trials per subject and the free parameters (q, eps0, rho, beta, gamma) only shift these probabilities by a few points, per-subject SD should be ~.07-.08 (var ~.006), giving a Welch t of roughly 15-20 at N=25. The metric also mirrors the two key qualitative claims: doubt beats recency (early .55 / mid .90 rows outvoting late .85/.95 rows) and coalitions buy nothing (1 row beating 5).

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design-fixed printed validities for this experiment
    v = np.array([0.55, 0.90, 0.75, 0.95, 0.68, 0.85], dtype=float)
    n = v.size
    j = np.arange(n, dtype=float)

    # MDSC-NV salience: near-argmax doubt rule (delta=15, lam=0.115*15)
    logu = 15.0 * np.log(np.clip(1.0 - v, 1e-12, None)) + 1.725 * j
    # SSI-DS+ compensatory summation weights (gamma=0.9, lam=1.5)
    w_ssi = np.power(np.clip(1.0 - v, 1e-12, None), 0.9) * np.exp(1.5 * j)

    hits = 0
    tot = 0
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(a_raw, dtype=float).ravel()
            b = np.asarray(b_raw, dtype=float).ravel()
        except Exception:
            continue
        if a.size != n or b.size != n:
            continue
        d = a - b
        disc = np.nonzero(d != 0.0)[0]
        if disc.size == 0:
            continue

        # MDSC-NV predicted winner: verdict of the single most doubtful
        # discriminating expert
        k = disc[int(np.argmax(logu[disc]))]
        pred_mdsc = 0 if d[k] > 0 else 1

        # SSI-DS+ predicted winner: sign of the weighted sum over all
        # discriminating rows (coalitions add, late rows dominate)
        E = float(np.sum(np.sign(d[disc]) * w_ssi[disc]))
        if E == 0.0:
            continue
        pred_ssi = 0 if E > 0 else 1

        if pred_ssi == pred_mdsc:
            continue  # non-diagnostic trial type

        tot += 1
        try:
            resp = int(r)
        except Exception:
            continue
        if resp == pred_mdsc:
            hits += 1

    if tot == 0:
        return 0.5
    return float(hits) / float(tot)

### RESULTS
- Predicted under pi_7 (simulated): 0.6833 (var=0.0038)
- Predicted under pi_6 (simulated): 0.2583 (var=0.0034)
- Observed on real data: 0.5925 (var=0.0026)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=6):** [0.51, 0.72, 0.8, 0.9, 0.95, 0.985]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  trial 2: A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0]
  trial 3: A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1]
  trial 4: A=[1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  trial 5: A=[0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  trial 6: A=[0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 7: A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 8: A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 9: A=[0, 1, 1, 0, 0, 1]  B=[0, 0, 1, 0, 1, 1]
  trial 10: A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0, 1]
  trial 11: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 12: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 13: A=[0, 1, 0, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 14: A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 15: A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 16: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  trial 17: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 19: A=[0, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 20: A=[0, 1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  trial 21: A=[1, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 22: A=[1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 23: A=[1, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 24: A=[1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]

**Rationale:** GOAL: separate SSI-DS+ (advocated) from MDSC-NV (competitor). Both invert validity and both add a reading-position gain, so the separation has to be built on three mechanistic axes: (A) MAGNITUDE OF THE VALIDITY INVERSION \u2014 SSI-DS+ uses (1-v)^gamma with gamma~0.9 so position (exp(lam*j), lam~1.5) dominates and the LATE rows carry the decision, whereas MDSC-NV uses (1-v)^delta with delta=13-17 so the single LOWEST-validity expert carries the decision wherever it sits; (B) SUMMATION vs SINGLE-CUE \u2014 SSI-DS+ sums all discriminating rows and passes the max-normalised sum through a sigmoid (coalitions add, and the sum can be near-zero even at a lopsided row count), MDSC-NV draws ONE row so coalitions buy nothing; (C) NULL-VERDICT CAPTURE \u2014 MDSC-NV says NON-discriminating rows matter (with prob q~.41 attention is captured by the most doubtful expert in the whole display; if that expert is silent the subject guesses), while SSI-DS+ says silent rows are strictly irrelevant.

VALIDITIES [0.51, 0.72, 0.80, 0.90, 0.95, 0.985]: validity RISES monotonically with column index, so anti-validity (doubt) and recency pull in exactly OPPOSITE directions on every trial \u2014 the cleanest possible geometry for axis (A). All pairwise gaps are >= 0.035, far above both models' label tolerances (.0025 / .002), so neither model can retreat into tally/degenerate mode; the contrast is purely mechanistic. SSI weights (gamma=.9, lam=1.5): w = [0.53, 1.43, 4.72, 11.33, 27.21, 41.27] \u2014 late-dominated but with w2+w3+w4 = 43.3 just barely exceeding w5 = 41.3, which makes a genuine coalition-summation test possible. MDSC log-salience (delta=15, lam=1.725): [-10.7, -17.4, -20.7, -29.4, -38.0, -54.4] \u2014 column 0 (v=.51) is the global doubt anchor by ~6.7 log units and stays the argmax over the whole parameter box (delta 13-17 x pos_ratio .09-.14), so MDSC is effectively 'whoever wins column 0; if column 0 is silent, guess with prob q and otherwise follow the lowest-validity discriminating row'.

12 base structures, each with its exact A/B mirror (24 unique pairs, K=4 -> 96 trials), so both models' predicted responses are perfectly key-balanced and side bias cannot mimic either. Non-discriminating rows carry matched 1s AND matched 0s so displays are dense and cannot be solved by counting 1s. Predictions use SSI-DS+ (gamma=.90, lam=1.50, beta=3.6, eps0=.24, rho=.78, eta=.73) and MDSC-NV (delta=15, pos_ratio=.115, q=.41, eps0=.24, rho=.81).

BLOCK 1 \u2014 CROWN JEWEL: NULL-VERDICT CAPTURE ON UNANIMOUS m=2 DISPLAYS (bases A,B,C = pairs 1-6). Three 2-0 dominance displays that differ ONLY in WHICH experts are silent. SSI-DS+ must predict the SAME saturated value on all three (E is computed over discriminating rows only and always lies in (1,2); unanimous m=2 also means zero load): .875 / .877 / .880 \u2014 a flat profile, and the base A minus base B/C difference must be ZERO for every parameter setting. MDSC-NV predicts a step: base A (disc = cols 0,1, so the globally most doubtful expert speaks) ~.88, base B (disc = cols 2,3, col0 silent) ~.72, base C (disc = cols 4,5, col0 silent) ~.72. A ~15-point drop between displays with identical 2-0 dominance structure is producible by no compensatory or standard one-reason account; flatness kills null-verdict capture. Base A also pins eps0 for both models.

BLOCK 2 \u2014 DOUBT vs RECENCY DIRECTIONAL REVERSALS AT m=2 (bases D,E = pairs 7-10).
\u2022 Base D: A wins col0 (.51, first) vs B wins col5 (.985, last). SSI: w5/w0 = 78 -> E ~ -.99 -> p(A) ~ .14. MDSC: the .51 row wins the doubt competition by ~44 log units and it is also the capture target -> p(A) ~ .88. A ~74-POINT REVERSAL, the single largest contrast in the design.
\u2022 Base E: A wins col1 (.72) vs B wins col4 (.95), with col0 SILENT. SSI: p(A) ~ .14 (same late-row logic). MDSC: doubt still picks col1, but capture on the silent col0 drags it back -> p(A) ~ .72. ~58-point reversal, and the D-vs-E comparison within MDSC (.88 vs .72) is a second, independent read-out of capture that SSI must call identical (.141 vs .144).

BLOCK 3 \u2014 COALITIONS ADD vs COALITIONS BUY NOTHING (bases F,G,H,K = pairs 11-16, 21-22).
\u2022 Base F (m=6, 1-vs-5): A wins col0 alone; B wins cols 1-5. SSI: summed weight -2.07 max-normalised -> p(A) ~ .36 (majority/late coalition wins). MDSC: one more-doubtful row outvotes five -> p(A) ~ .66. Clean directional split at a 1-vs-5 row count.
\u2022 Base G (m=3, 2-vs-1): A wins cols 4,5; B wins col0. SSI ~ .80; MDSC ~ .19. ~60-point reversal.
\u2022 Base H (m=4, exact 2-2 tie): A wins cols 0,1; B wins cols 4,5. SSI ~ .27; MDSC ~ .75. ~48-point reversal at a row-count tie, which no tally account can produce in either direction.
\u2022 Base K (m=4, 1-vs-3): A wins col5 alone; B wins cols 1,2,3. SSI ~ .68 (the lone heavy late row outweighs three earlier rows); MDSC ~ .35 (lowest-validity discriminating row is col1, favouring B; capture adds a pull to chance). ~33-point reversal.

BLOCK 4 \u2014 SUM NEAR ZERO AT A LOPSIDED ROW COUNT (base J = pairs 19-20). A wins cols 2,3,4 (summed weight 43.3), B wins col5 (41.3). SSI: E = +.05 -> p(A) ~ .52, i.e. essentially CHANCE despite a 3-vs-1 row count \u2014 the signature 'big count margin, cancelling weighted margin' prediction. MDSC: lowest-validity discriminating row is col2 (favouring A) -> p(A) ~ .65. A 13-point gap in which SSI is at chance and MDSC is clearly above it.

BLOCK 5 \u2014 UNANIMITY DECAY / eta AND CAPTURE-MONOTONICITY (bases I,L = pairs 17-18, 23-24). Base L: unanimous m=4 on cols 2-5 (col0 silent). Base I: unanimous m=6 (all rows, col0 speaks). SSI, whose load is discounted by eta on unanimous displays and which ignores silent rows, must produce a STRICTLY MONOTONE decline with display size: .88 (m=2) > .76 (m=4) > .68 (m=6). MDSC must produce a NON-MONOTONE profile because capture depends on whether col0 speaks: .88 (m=2, base A) > .65 (m=4, base L) ~ .66 (m=6, base I) \u2014 the m=6 display is no worse than the smaller m=4 one, because at m=6 the doubt anchor is guaranteed to discriminate.

INTERPRETATION: if the three unanimous m=2 displays sit flat at ~.87 regardless of which experts are silent, if the option holding the LATE high-validity rows is chosen on bases D, E, G, H (~.14-.27 for the early/low-validity holder), if the lone late row beats three earlier rows (base K ~.68) and the 3-vs-1 base J sits at chance, then compensatory max-normalised summation with a weak validity inversion and competition-scaled interference (SSI-DS+) is confirmed and MDSC-NV's near-argmax doubt rule plus null-verdict capture is falsified. If instead unanimous 2-0 accuracy drops ~15 points purely as a function of whether the .51 expert happens to speak, the early .51 row beats the late .985 row (base D ~.88), and one doubtful row outvotes a five-row coalition (base F above chance), MDSC-NV wins. No pair is degenerate: on 8 of the 12 bases the models disagree in direction or by more than 30 points, and the unanimous bases double as the shared lapse baseline.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
The design contains six base structures (D, E, F, G, H, K) on which SSI-DS+ and MDSC-NV point to OPPOSITE options: SSI-DS+ sums max-normalised weights that are dominated by the late, high-validity rows, while MDSC-NV lets the single most doubtful (lowest-validity) discriminating row decide, with coalitions buying nothing. The metric selects exactly those conflict trials online (by recomputing the two decision rules per trial from the stimulus, using each theory's central parameter values) and reports the proportion of choices that went to the SSI-DS+ favoured option. Unanimous displays, base J (where both rules agree), and any trial where the rules coincide are excluded automatically, so the metric is not trivially true for either account: it is a pure directional read-out of 'weighted late-row summation' vs 'anti-validity single cue'. Expected value under SSI-DS+ is roughly 0.75-0.78 (average of .86, .86, .64, .80, .73, .68 across bases D, E, F, G, H, K) and under MDSC-NV roughly 0.25 (average of .12, .28, .34, .19, .25, .35) \u2014 a ~0.5 gap. Because ~48 trials per subject enter the average, binomial noise is only ~0.06 SD and parameter heterogeneity is modest (both models' lapse terms are bounded), so the per-subject spread should be far below the between-theory gap.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V_DESIGN = np.array([0.51, 0.72, 0.80, 0.90, 0.95, 0.985], dtype=float)

    GAMMA = 0.90
    LAM = 1.50
    DELTA = 15.0
    LAM_M = 1.725

    hits = 0.0
    tot = 0.0

    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        try:
            a = np.asarray(list(a_raw), dtype=float).ravel()
            b = np.asarray(list(b_raw), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]

        if n <= V_DESIGN.size:
            v = V_DESIGN[:n]
        else:
            v = np.concatenate([V_DESIGN, np.full(n - V_DESIGN.size, 0.985)])
        v = np.clip(v, 0.5, 0.9995)

        diff = a - b
        disc = np.nonzero(diff != 0.0)[0]
        if disc.size == 0:
            continue
        signs = np.sign(diff[disc])

        # --- SSI-DS+ side: max-normalised weighted SUM, late/high-validity dominated
        logw = GAMMA * np.log(np.clip(1.0 - v[disc], 1e-12, None)) + LAM * disc.astype(float)
        logw = logw - np.max(logw)
        w = np.exp(logw)
        E = float(np.sum(signs * w))
        if abs(E) < 1e-9:
            continue
        ssi_side = 1.0 if E > 0 else -1.0

        # --- MDSC-NV side: single most-doubtful DISCRIMINATING row decides
        sal = DELTA * np.log(np.clip(1.0 - v[disc], 1e-12, None)) + LAM_M * disc.astype(float)
        k = int(np.argmax(sal))
        mdsc_side = float(signs[k])
        if mdsc_side == 0.0:
            continue

        # keep only trials where the two accounts point to OPPOSITE options
        if mdsc_side == ssi_side:
            continue

        try:
            r = int(resp)
        except Exception:
            continue
        chose_a = (r == 0)
        ssi_favours_a = (ssi_side > 0)
        hits += 1.0 if (chose_a == ssi_favours_a) else 0.0
        tot += 1.0

    if tot == 0.0:
        return 0.5
    return float(hits / tot)

### RESULTS
- Predicted under pi_7 (simulated): 0.2592 (var=0.0042)
- Predicted under pi_6 (simulated): 0.7412 (var=0.0040)
- Observed on real data: 0.2608 (var=0.0079)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """One-reason adherence on model-disagreement trials.

    For every trial we compute (a) the option favoured by the FIRST
    discriminating cue in validity order (features are listed in
    descending validity: 0.95, 0.85, 0.75, 0.65, 0.55) and (b) the option
    favoured by the unit-weight tally of feature-wise wins.  We keep only
    trials where these two disagree in direction OR where the tally is tied
    (so a counting rule must guess).  The metric is the proportion of such
    trials on which the subject picked the option favoured by the first
    discriminating cue.
    """
    hits = []
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = min(a.shape[0], b.shape[0])
        a = a[:n]
        b = b[:n]

        # first discriminating cue (features already in descending validity)
        first_winner = None
        for j in range(n):
            if a[j] > b[j]:
                first_winner = 0
                break
            if b[j] > a[j]:
                first_winner = 1
                break
        if first_winner is None:
            continue  # no cue discriminates: uninformative trial

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = None  # counting rule must guess

        if tally_winner is not None and tally_winner == first_winner:
            continue  # agreement control trial: not diagnostic

        try:
            r = int(resp)
        except (TypeError, ValueError):
            continue
        hits.append(1.0 if r == first_winner else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.2820 (var=0.0035)
**Predicted under pi_7:** 0.2878 (var=0.0023)
**Predicted under pi_6:** 0.2560 (var=0.0021)

### Experiment 4
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.6, 0.93, 0.55, 0.87, 0.7, 0.78], dtype=float)

    def analyze(row):
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        n = min(a.shape[0], b.shape[0])
        a = a[:n]
        b = b[:n]
        val = validities[:n] if validities.shape[0] >= n else np.arange(n, 0, -1, dtype=float)
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        margin = a_wins - b_wins
        if margin > 0:
            tally_winner = 0
        elif margin < 0:
            tally_winner = 1
        else:
            tally_winner = -1
        order = np.argsort(-val, kind='stable')
        ttb_winner = -1
        for j in order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        return margin, tally_winner, ttb_winner

    info = [analyze(r) for _, r in data.iterrows()]
    margins = np.array([x[0] for x in info], dtype=float)
    tally_w = np.array([x[1] for x in info], dtype=int)
    ttb_w = np.array([x[2] for x in info], dtype=int)
    resp = data['response'].to_numpy().astype(int)

    # Group 1: tally-tie trials where TTB is decisive (parameter-free for Tallying)
    tie_mask = (tally_w == -1) & (ttb_w >= 0)
    # Group 2: trials where tally margin is >= 2 AND TTB points the other way
    dis_mask = (np.abs(margins) >= 2) & (tally_w >= 0) & (ttb_w >= 0) & (tally_w != ttb_w)

    if tie_mask.sum() == 0 and dis_mask.sum() == 0:
        return float('nan')

    if dis_mask.sum() > 0:
        p_tally = float(np.mean(resp[dis_mask] == tally_w[dis_mask]))
    else:
        p_tally = 0.5
    if tie_mask.sum() > 0:
        p_ttb_tie = float(np.mean(resp[tie_mask] == ttb_w[tie_mask]))
    else:
        p_ttb_tie = 0.5

    return p_tally - p_ttb_tie

```

**Observed (real) value:** 0.1744 (var=0.0127)
**Predicted under pi_7:** 0.2300 (var=0.0149)
**Predicted under pi_6:** 0.1639 (var=0.0160)

### Experiment 5
**Design**
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V = np.array([0.68, 0.96, 0.52, 0.88, 0.58, 0.78], dtype=float)

    hits = 0.0
    n = 0.0
    for _, row in data.iterrows():
        try:
            a = np.asarray([float(x) for x in row['option_a_ratings']], dtype=float)
            b = np.asarray([float(x) for x in row['option_b_ratings']], dtype=float)
        except Exception:
            continue
        k = int(min(a.shape[0], b.shape[0]))
        if k == 0:
            continue
        a = a[:k]
        b = b[:k]
        if V.shape[0] >= k:
            v = V[:k]
        else:
            v = np.concatenate([V, np.full(k - V.shape[0], 0.55)])[:k]

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        # keep only unit-tally TIES with at least one discriminating cue
        if a_wins != b_wins or a_wins == 0:
            continue

        w = v ** 2.0
        d = float(np.sum(w * (a - b)))
        # discard tie trials whose validity-weighted margin is itself
        # near zero (no clear "higher-validity side" to predict)
        if abs(d) < 0.15:
            continue

        resp = row['response']
        try:
            resp = int(resp)
        except Exception:
            continue
        if resp not in (0, 1):
            continue

        chose_a = (resp == 0)
        picked_high_validity = chose_a if d > 0 else (not chose_a)
        hits += 1.0 if picked_high_validity else 0.0
        n += 1.0

    if n == 0:
        return 0.5
    return float(hits / n)
```

**Observed (real) value:** 0.2856 (var=0.0160)
**Predicted under pi_7:** 0.2778 (var=0.0058)
**Predicted under pi_6:** 0.2928 (var=0.0057)

### Experiment 6
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    v = np.array([0.58, 0.97, 0.53, 0.80, 0.66, 0.90], dtype=float)
    w = v ** 2.0  # AVAI-style amplified weights (gamma=2)

    num = 0.0
    den = 0.0
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = int(min(a.shape[0], b.shape[0], w.shape[0]))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        ww = w[:n]
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins != b_wins:
            continue  # not a tally tie
        d = float(np.sum(ww * (a - b)))
        if abs(d) < 0.30:
            continue  # tie trials whose weighted margin is negligible
        den += 1.0
        if d > 0:
            num += 1.0 if int(resp) == 0 else 0.0
        else:
            num += 1.0 if int(resp) == 1 else 0.0

    if den == 0:
        return 0.5
    return num / den
```

**Observed (real) value:** 0.1500 (var=0.0194)
**Predicted under pi_7:** 0.1500 (var=0.0133)
**Predicted under pi_6:** 0.1450 (var=0.0086)

### Experiment 7
**Design**
  A=[1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    correct = []
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()
            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        d = a[:n] - b[:n]
        nz = d[d != 0.0]
        if nz.size == 0:
            continue
        # dominance trial: every discriminating cue favours the same option
        if not (np.all(nz > 0) or np.all(nz < 0)):
            continue
        dominant_is_a = bool(np.all(nz > 0))
        try:
            resp = int(row['response'])
        except Exception:
            continue
        chose_a = (resp == 0)
        correct.append(1.0 if (chose_a == dominant_is_a) else 0.0)

    if len(correct) == 0:
        return 0.5
    return float(np.mean(correct))
```

**Observed (real) value:** 0.7427 (var=0.0173)
**Predicted under pi_7:** 0.7467 (var=0.0048)
**Predicted under pi_6:** 0.7773 (var=0.0049)

### Experiment 8
**Design**
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    # design validities (fixed by the experimental design)
    v_design = np.array([0.95, 0.70, 0.55, 0.95, 0.70, 0.55], dtype=float)
    hits = 0
    n = 0
    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        k = int(min(a.size, b.size))
        if k == 0:
            continue
        if v_design.size >= k:
            vv = v_design[:k]
        else:
            vv = np.linspace(0.95, 0.55, k)
        d = a[:k] - b[:k]
        idx = np.nonzero(d != 0.0)[0]
        if idx.size != 2:
            continue
        j1 = int(idx[0])
        j2 = int(idx[1])
        # the two discriminating cues must point to OPPOSITE options
        if d[j1] * d[j2] > 0:
            continue
        # and the LATER cue must be the LESS valid one
        if not (vv[j2] < vv[j1]):
            continue
        late_choice = 0 if d[j2] > 0 else 1  # 0 = A, 1 = B
        try:
            resp = int(r)
        except Exception:
            continue
        n += 1
        if resp == late_choice:
            hits += 1
    if n == 0:
        return float('nan')
    return float(hits) / float(n)
```

**Observed (real) value:** 0.5067 (var=0.0112)
**Predicted under pi_7:** 0.4817 (var=0.0130)
**Predicted under pi_6:** 0.5033 (var=0.0076)

### Experiment 9
**Design**
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float).ravel()
        b = np.asarray(row['option_b_ratings'], dtype=float).ravel()
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        d = a[:n] - b[:n]
        disc = np.nonzero(d != 0.0)[0]
        m = int(disc.size)
        if m < 2:
            continue
        fav_a = d[disc] > 0.0
        # skip unanimous (dominance) displays: both theories agree there
        if fav_a.all() or (~fav_a).all():
            continue
        # option favoured by the LAST-READ (highest-index) discriminating row
        last_idx = disc[-1]
        late_option = 0 if d[last_idx] > 0.0 else 1
        try:
            resp = int(row['response'])
        except Exception:
            continue
        hits.append(1.0 if resp == late_option else 0.0)

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4383 (var=0.0023)
**Predicted under pi_7:** 0.4508 (var=0.0038)
**Predicted under pi_6:** 0.4419 (var=0.0028)

### Experiment 10
**Design**
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design validities (fixed by the experiment); fallback handles other lengths.
    V_DESIGN = np.array([0.955, 0.545, 0.950, 0.540, 0.945, 0.535], dtype=float)
    LAM = 1.6
    DELTA = 0.9

    hits = []
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()
            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]

        if V_DESIGN.size >= n:
            v = V_DESIGN[:n]
        else:
            v = np.concatenate([V_DESIGN, np.full(n - V_DESIGN.size, 0.545)])[:n]

        d = a - b
        disc = np.nonzero(d != 0.0)[0]
        m = int(disc.size)
        if m < 2:
            continue

        # salience of each discriminating row: recency x (1 - validity)
        j = disc.astype(float)
        logu = LAM * j + DELTA * np.log(np.clip(1.0 - v[disc], 1e-9, None))
        logu = logu - np.max(logu)
        u = np.exp(logu)
        top = disc[int(np.argmax(u))]
        top_owner_a = bool(d[top] > 0.0)  # True -> the salient row favours A

        n_a = int(np.sum(d[disc] > 0.0))
        n_b = m - n_a

        # include only trials where the row-count tally does NOT strictly
        # agree with the most salient (late / low-validity) discriminating row
        if n_a > n_b:
            majority_a = True
            tie = False
        elif n_b > n_a:
            majority_a = False
            tie = False
        else:
            majority_a = None
            tie = True
        if (not tie) and (majority_a == top_owner_a):
            continue

        try:
            resp = int(row['response'])
        except Exception:
            continue
        chose_a = (resp == 0)
        hits.append(1.0 if (chose_a == top_owner_a) else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.7697 (var=0.0057)
**Predicted under pi_7:** 0.7686 (var=0.0034)
**Predicted under pi_6:** 0.7540 (var=0.0026)

### Experiment 11
**Design**
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V_DESIGN = np.array([0.958, 0.546, 0.954, 0.542, 0.950, 0.538], dtype=float)
    LAM = 1.5      # reading-order recency gain (shared by both accounts)
    GAM = 0.9      # oddness exponent on (1 - validity) (shared by both accounts)

    hits = 0.0
    tot = 0.0

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(a_raw, dtype=float).ravel()
            b = np.asarray(b_raw, dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]

        if n == V_DESIGN.size:
            v = V_DESIGN.copy()
        else:
            v = np.linspace(0.95, 0.55, n)
        v = np.clip(v, 0.5, 0.995)

        diff = a - b
        disc = np.nonzero(diff != 0.0)[0]
        m = int(disc.size)
        if m < 2:
            continue
        signs = np.sign(diff[disc])
        # keep only CONFLICT displays (both options win at least one row)
        if np.all(signs > 0) or np.all(signs < 0):
            continue

        logw = GAM * np.log(np.clip(1.0 - v[disc], 1e-9, None)) + LAM * disc.astype(float)
        k = int(np.argmax(logw))
        favored_a = signs[k] > 0.0

        try:
            r = int(resp)
        except Exception:
            continue
        chose_a = (r == 0)
        hits += 1.0 if (chose_a == favored_a) else 0.0
        tot += 1.0

    if tot == 0.0:
        return float('nan')
    return float(hits / tot)
```

**Observed (real) value:** 0.8189 (var=0.0074)
**Predicted under pi_7:** 0.7714 (var=0.0022)
**Predicted under pi_6:** 0.7364 (var=0.0031)

### Experiment 12
**Design**
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # Design validities (fixed by the experiment)
    v_default = np.array([0.53, 0.72, 0.86, 0.93, 0.99], dtype=float)

    def weights(n):
        if n <= v_default.size:
            v = v_default[:n]
        else:
            v = np.concatenate([v_default, np.full(n - v_default.size, 0.99)])
        v = np.clip(v, 0.5, 0.9995)
        logw = 0.9 * np.log(1.0 - v) + 1.5 * np.arange(n, dtype=float)
        logw = logw - np.max(logw)
        return np.exp(logw)

    wcache = {}
    hits = []
    for a_raw, b_raw, resp in zip(data['option_a_ratings'].values,
                                  data['option_b_ratings'].values,
                                  data['response'].values):
        try:
            a = np.asarray(list(a_raw), dtype=float).ravel()
            b = np.asarray(list(b_raw), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        d = a - b
        nA = int(np.sum(d > 0))
        nB = int(np.sum(d < 0))
        # keep only CONFLICT displays (both options favoured by at least one expert)
        if nA == 0 or nB == 0:
            continue
        if n not in wcache:
            wcache[n] = weights(n)
        w = wcache[n]
        S = float(np.sum(np.sign(d) * w))
        if S == 0.0:
            continue
        fav = 0 if S > 0 else 1  # response coding: 0 = A, 1 = B
        try:
            r = int(resp)
        except Exception:
            continue
        hits.append(1.0 if r == fav else 0.0)

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4178 (var=0.0051)
**Predicted under pi_7:** 0.3878 (var=0.0029)
**Predicted under pi_6:** 0.7386 (var=0.0031)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all twelve experiments the anti-validity signature is overwhelming and consistent: on tally-tie designs subjects pick the side favoured by the LOW-validity cue (Exp 5 = .286, Exp 6 = .150), TTB adherence is far below chance (Exp 3 = .282, Exp 4's tally-minus-TTB gap = +.17), and on the alternating-label designs adherence to the salient low-validity row is very high (Exp 10 = .770, Exp 11 = .819). Both theories encode validity inversion, so they agree on those. The discriminating tests are the ones where the MAGNITUDE of the inversion and summation-vs-single-cue matter, and there pi_6 (SSI-DS+) fails badly. Exp 2, pi_6's OWN crown-jewel design (validity rising monotonically so doubt and recency oppose on every trial), gave observed .261 against pi_6's prediction of .741 and pi_7's .259 \ru2014 a near-perfect hit for the near-argmax doubt rule and a 48-point miss for compensatory late-row-dominated summation. Exp 12 (ascending validities again) repeats the verdict: observed .418 vs pi_7 .388 vs pi_6 .739. Exp 1 also favours pi_7 (.593 observed vs .683 / .258), though pi_7 over-predicts by ~9 points, and Exp 11 (.819 vs .771/.736) and Exp 7/10/3 all lean pi_7. pi_6 is only trivially better on Exp 4, 5, 8, 9, all of which are low-discrimination designs where the two models predict within ~.03 of each other. So pi_6's core quantitative commitment (gamma~0.9, recency-dominated exp(lam*j) weights, coalitions add) is decisively falsified wherever it is testable: recency does NOT beat doubt, and late high-validity coalitions do NOT win. pi_7 is the clear survivor and should be kept. It is not perfect \ru2014 the Exp 1 shortfall (.593 vs .683) and the Exp 11 under-prediction (.771 vs .819) suggest the mixture of near-argmax doubt plus null-verdict capture is slightly mis-calibrated: subjects look more graded (coalition-sensitive / noisier) on the scrambled-validity design but MORE deterministic on the alternating-label design than pi_7 allows. That residual is exactly where a sharper competitor should attack, rather than resurrecting SSI-DS+ with new parameters (no parameter setting inside its description can reverse the sign of the Exp 2 / Exp 12 predictions, since the description hard-commits to weak inversion with recency-dominant exponential position weights).",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_6 entirely. Do NOT re-tune SSI-DS+: its description commits to gamma~0.9 with exp(lam*j) recency dominance, which is sign-wrong on every monotone-validity design (Exp 2, Exp 12) and cannot be rescued by parameters.\n\nProposed replacement \ru2014 'TTL-RD: Take-The-Least-Valid with RANK-based geometric discounting and conflict-scaled noise'. Core claims:\n(1) Subjects order the printed experts by validity and treat the LEAST valid discriminating expert as the decisive voice (the 'newsworthy dissent' intuition, same direction as MDSC-NV, so it reproduces Exp 2/3/5/6/12 correctly).\n(2) CRUCIAL DIFFERENCE 1 \ru2014 weighting is by RANK, not by the numeric value of (1-v): w for the r-th least-valid discriminating cue is phi^r with phi ~ 0.30-0.50. Hence only the ORDER of the printed validities matters, not the size of the gaps. This makes sharp, opposite predictions to MDSC-NV in designs where two doubtful rows are numerically close but rank-separated (e.g. .55 vs .68, or the .945/.950/.955 families), and it predicts identical behaviour for designs that differ only in how far apart the labels are printed.\n(3) CRUCIAL DIFFERENCE 2 \ru2014 evidence SUMS over ranks with the geometric discount: E = sum_r sign_r * phi^r, p(A) = sigmoid(beta*E). With phi < 0.5 the top-ranked cue normally wins, but a large coalition of lower-ranked cues can overturn it when the margin is big (1-vs-5 displays). This is exactly the regime where the real Exp 1 value (.593) sat BELOW pi_7's near-argmax prediction (.683), so the new theory should fit Exp 1 better while still matching Exp 2.\n(4) CRUCIAL DIFFERENCE 3 \ru2014 NO null-verdict capture: non-discriminating rows are strictly irrelevant. All the pull toward chance comes from a conflict-scaled lapse g = 1-(1-eps0)*rho^min(nA,nB) that depends on the number of OPPOSED cues only (not on total display size), which lets dominance displays stay high (Exp 7 = .743) while conflict displays compress. This is a direct, testable clash with MDSC-NV claim (3).\n(5) Ties in the printed labels (gap < .002) collapse adjacent experts into a single rank, broken by reading position (later = decisive); this reproduces the exact-chance 1-1 result of Exp 8 and the high late-low-validity adherence of Exp 10/11 (target ~.78-.82, i.e. slightly more deterministic than pi_7 achieved).\nParameter ranges: phi 0.25-0.55, beta 2-6, eps0 0.05-0.30, rho 0.55-0.90, plus a small position tiebreak weight. The next experiment can then be designed to pit rank-based vs magnitude-based doubt (numerically close but rank-distinct doubtful rows) and to test null-verdict capture directly by holding the discriminating rows fixed while silencing/unsilencing the globally most doubtful expert."
}
```

## Usage

```json
{
  "input_tokens": 34366,
  "output_tokens": 5364
}
```
