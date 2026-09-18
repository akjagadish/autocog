# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_1
People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_4
**Readability-limited validity weighting with a strict-unanimity consensus premium (RVW-DE+/U), with broad determinacy heterogeneity.**

A single stationary, fully compensatory integration rule underlies choice in all cue-based binary-feature environments; the apparent switching between 'take-the-best-like' and 'tallying-like' behaviour across environments is produced by modulators that are properties of the *display and the evidence pattern*, not of the strategy, plus wide between-person variation in decision determinacy.

(1) **Graded validity weighting.** Each expert's subjective importance is a power transform of that expert's diagnosticity on the log-odds scale, w_j ∝ logit(v_j)^gamma, gamma spread over ~1.05–2.0 across people. At the population mean this is steep enough that a clearly best expert usually survives a coalition of three or four weak experts, yet weak enough that a near-peer plus a couple of moderate cues can overturn the top cue.

(2) **Binding readability, not position.** Stated validities must be bound to displayed columns. When the display order is *monotone* in validity (ascending OR descending) this binding is trivial and the validity weights are used essentially intact. When the order is scrambled, binding fails partially and the weights are flattened toward EQUAL weighting: w_eff = (1-kappa)·w + kappa/n, with kappa = k0 + k1·(1-|Kendall tau(column index, validity)|) + k2·[best expert not in an extreme column]. Direction-free: reversing a sorted display costs nothing, permuting it costs a lot.

(3) **Readability-gated reading-order fallback.** The left-to-right attention gradient is a fallback recruited only when validity-to-column binding fails: phi_eff = phi·(r0 + r1·kappa). It only *scales* weights, never re-sorts them.

(4) **Strict-unanimity consensus premium (dissent aversion).** The premium attaches to *consensus itself*, and consensus is an essentially all-or-none perceptual property: a panel is 'agreed' only when it is unanimous or one voice away from it. Subjects add a direction-blind premium rho·|2·mean(x)-1|^q with a HIGH exponent q (~5–11, person-specific), so a 5/5 or 0/5 panel earns almost the whole premium while a 4/5 or 1/5 panel earns almost none. The magnitude rho of this premium is substantial and heterogeneous (population mean ~0.39): an option unanimously rated 0 can be *preferred* to one carrying a single weak endorsement (sub-chance accuracy on one-endorsement-vs-nothing pairs), while options that are merely lopsided (4-of-5 endorsements) receive essentially no credibility bonus and therefore cannot dilute genuine weighted-evidence conflicts. The premium is a *unanimity* effect, not a mean-rating effect — that is the theory's sharpest falsifiable commitment (a strongly non-linear step at exactly 0 and n dissenters, not a smooth gradient in endorsement proportion).

(5) **Endorsement discount / pull.** Each positive endorsement carries a small credibility adjustment theta (faint-praise skepticism for most, mild endorsement-count pull for some), inert whenever the two options carry equal endorsement counts.

(6) **Population structure.** Everyone holds the same rule; heterogeneity is broad and continuous — in determinacy beta, in gamma, in lapse, in the modulator magnitudes rho, q, theta, phi and the readability coefficients. No discrete strategy switching.

Evidence for A: D = Σ_j w_eff_j (x_Aj - x_Bj) - theta·(Σx_A - Σx_B) + rho·(coh(x_A) - coh(x_B)); p(A) = (1-eps)·logistic(beta·D) + eps/2.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Readability-limited validity weighting with dissent aversion and
    # endorsement discount (RVW-DE+/U).
    #   w_j        proportional to logit(v_j) ** gamma
    #   w_eff      = (1-kappa)*w + kappa/n          (readability flattening)
    #   w_eff     *= (1 + phi_eff * reading-order gradient)
    #                with phi_eff = phi*(r0 + r1*kappa): the reading-order
    #                fallback is recruited only when binding fails.
    #   D          = w_eff.(a-b) - theta*(sum a - sum b) + rho*(coh(a)-coh(b))
    #   p(A)       = (1-eps)*logistic(beta*D) + eps/2
    # kappa is driven by how incongruent the column order is with the
    # stated validity order (direction-free: monotone displays are cheap,
    # scrambled displays are expensive).  coh(x) = |2*mean(x)-1|**q with a
    # HIGH person-specific exponent q: the consensus premium is a strict
    # unanimity effect, not a smooth function of endorsement proportion.
    # History is unused: no feedback is given, the rule is stationary.
    import numpy as np

    # ---------------- unpack the two rating vectors --------------------
    a = None
    b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 1:
            half = arr.shape[0] // 2
            a = arr[:half].astype(float)
            b = arr[half:half + half].astype(float)
        else:
            arr = arr.reshape(arr.shape[0], -1)
            a = arr[0].astype(float)
            b = arr[1].astype(float)

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float).ravel()[:n]
    b = np.asarray(b, dtype=float).ravel()[:n]

    # ---------------- validities --------------------------------------
    val = parameters.get("validities", None)
    if val is None:
        v = np.linspace(0.9, 0.55, n)
    else:
        v = np.asarray(val, dtype=float).ravel().astype(float)
        if v.shape[0] < n:
            v = np.concatenate([v, np.full(n - v.shape[0], 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-6)

    # ---------------- parameters --------------------------------------
    gamma = float(parameters.get("gamma", 1.5))
    beta = float(parameters.get("beta", 7.0))
    eps = float(np.clip(float(parameters.get("eps", 0.09)), 0.0, 1.0))
    k0 = float(parameters.get("k0", 0.08))
    k1 = float(parameters.get("k1", 0.78))
    k2 = float(parameters.get("k2", 0.35))
    phi = float(parameters.get("phi", 0.20))
    theta = float(parameters.get("theta", 0.02))
    rho = float(parameters.get("rho", 0.39))
    q = float(parameters.get("q", 8.0))

    # readability-gating of the reading-order fallback (fixed structure)
    r0 = 0.35
    r1 = 1.15

    # ---------------- validity weights --------------------------------
    L = np.log(v / (1.0 - v))
    L = np.maximum(L, 1e-9)
    with np.errstate(over="ignore", invalid="ignore"):
        w = np.power(L, gamma)
    w = np.nan_to_num(w, nan=0.0, posinf=1e12, neginf=0.0)
    s = float(np.sum(w))
    if (not np.isfinite(s)) or s <= 0.0:
        w = np.ones(n, dtype=float) / float(n)
    else:
        w = w / s

    # ---------------- readability flattening --------------------------
    # Kendall tau between column index and stated validity (ties skipped).
    conc = 0
    disc = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = v[j] - v[i]
            if d > 0:
                conc += 1
            elif d < 0:
                disc += 1
    tot = conc + disc
    tau = 0.0 if tot == 0 else float(conc - disc) / float(tot)
    incong = 1.0 - abs(tau)

    jmax = int(np.argmax(v))
    off_extreme = 0.0 if (jmax == 0 or jmax == n - 1) else 1.0

    kappa = k0 + k1 * incong + k2 * off_extreme
    kappa = float(np.clip(kappa, 0.0, 1.0))

    w_eff = (1.0 - kappa) * w + kappa * (1.0 / float(n))

    # ------- reading-order (primacy) gradient, gated by readability ----
    if n > 1 and phi != 0.0:
        phi_eff = phi * (r0 + r1 * kappa)
        phi_eff = float(np.clip(phi_eff, 0.0, 0.45))
        idx = np.arange(n, dtype=float)
        grad = 1.0 + phi_eff * ((float(n - 1) - 2.0 * idx) / float(n - 1))
        grad = np.maximum(grad, 1e-6)
        w_eff = w_eff * grad
    ssum = float(np.sum(w_eff))
    if (not np.isfinite(ssum)) or ssum <= 0.0:
        w_eff = np.ones(n, dtype=float) / float(n)
    else:
        w_eff = w_eff / ssum

    # ---------------- evidence ----------------------------------------
    core = float(np.dot(w_eff, a - b))

    # endorsement discount: credibility adjustment per positive rating
    cost = -theta * (float(np.sum(a)) - float(np.sum(b)))

    # dissent aversion: strict-unanimity consensus premium (high exponent q)
    ma = float(np.mean(a))
    mb = float(np.mean(b))
    ua = abs(2.0 * ma - 1.0)
    ub = abs(2.0 * mb - 1.0)
    coh_a = float(np.power(ua, q))
    coh_b = float(np.power(ub, q))
    cons = rho * (coh_a - coh_b)

    D = core + cost + cons

    # ---------------- logistic choice with lapse -----------------------
    z = np.array([beta * D, 0.0], dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)

    p = (1.0 - eps) * p_core + eps * 0.5
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64).ravel()
    p = np.clip(p, 0.0, None)
    tot = p.sum()
    if (not np.isfinite(tot)) or tot <= 0.0:
        p = np.ones_like(p) / float(len(p))
    else:
        p = p / tot
    return int(np.random.choice(len(p), p=p))

## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 5: A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 7: A=[1, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0]
  trial 8: A=[1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 1]
  trial 9: A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  trial 10: A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1]
  trial 11: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 12: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 13: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 14: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 15: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate Take-The-Best (advocated) from RVW-DE+/U (competing). The competitor's sharpest falsifiable commitment is the STRICT-UNANIMITY consensus premium: rho*|2*mean(x)-1|^q with high q (5-11), so a panel that is unanimous (0/5 or 5/5) collects almost the whole premium (rho ~ .19-.59, mean .39) while a merely lopsided 4/5 or 1/5 panel collects essentially none. TTB has no such term: on every discriminating trial it sides with the highest-validity discriminating expert at one flat level sigmoid(beta) (plus lapse), regardless of endorsement counts, coherence, or lower cues.

DISPLAY: five experts shown in strictly DESCENDING validity (.95,.85,.75,.65,.55). This is deliberately the competitor's *best case*: Kendall tau = -1 so incongruence = 0, the best expert sits in an extreme column, hence kappa = k0 (.02-.16) and the reading-order fallback phi_eff = phi*(0.35+1.15*k0) is nearly inert. Readability nuisance is therefore removed and the competitor's predictions are driven almost entirely by its graded validity weights and by the unanimity premium. Normalised weights across its gamma box (1.05-1.95) are approximately [.46-.64, .23-.26, .09-.16, .03-.09, .003-.03], i.e. cue1 >> cue2 > cue3 > cue4 > cue5 for every subject.

16 unique pairs (K=6, 96 trials), every pair mirrored so neither response key is systematically correct. TTB predicts A on pairs 1,3,5,7,9,11,13,15 and B on their mirrors, always at the SAME probability.

FAMILY S - SINGLE ENDORSEMENT vs AN EMPTY (UNANIMOUS-ZERO) PANEL (pairs 1-6). One option has exactly one '1' (on cue1, cue3 or cue5) and the other is all zeros. TTB: the endorsed option wins in all three cases at an identical level (~.99 at moderate beta). Competitor: D(endorsed) = w_j - rho*(1 - 0.6^q) ~ w_j - .38, so with beta=7, gamma=1.5, rho=.39, q=8 it predicts p(endorsed) ~ .70 (cue1), .13 (cue3), .06 (cue5) - a steep monotone gradient that CROSSES CHANCE between cue1 and cue3, i.e. explicit sub-chance accuracy when only a weak expert speaks against an all-zero panel. Flat-at-ceiling versus sign-reversing-gradient is the core directional dissociation.

FAMILY P - UNANIMITY STEP WITH WEIGHTED EVIDENCE HELD EXACTLY CONSTANT (pair 5/6 reused as the baseline, plus pairs 7-10). In all three cells the two options differ on cue5 and only cue5, so the weighted-evidence term is identical (w5 ~ .01-.03) and TTB must give the SAME probability in all three. Only the coherence of the panels changes: (i) [0,0,0,0,1] vs [0,0,0,0,0] - loser panel unanimous, full premium; (ii) [1,1,0,0,1] vs [1,1,0,0,0] - both panels middling (2/5 vs 3/5), premium ~ 0; (iii) [1,1,1,1,1] vs [1,1,1,1,0] - winner panel unanimous, full premium in the same direction as the cue. Competitor predicts p(cue5-winner) ~ .06 / .51 / .94 respectively (a non-monotone step at exactly 0 and n dissenters), TTB predicts ~.99 / .99 / .99. This triple is the sharpest quantitative signature in the design and cannot be mimicked by any re-weighting of cues.

FAMILY D - COMPENSATION / GAMMA IDENTIFICATION (pairs 11-12): [1,0,0,0,0] vs [0,1,1,1,1]. Coherence cancels exactly (both panels 1/5 vs 4/5, |2m-1| identical), so this is a pure weighted-integration test. TTB stops at cue1 and is at ceiling for the top-cue option; the competitor is near-cancelling and its sign flips inside its own gamma box (core = +.05 at gamma=1.5, -.08 at gamma=1.05, +.15 at gamma=1.95), so it must predict the LARGEST between-subject spread of any family here and an overall level close to chance.

FAMILY E - DEEP CASCADE vs COALITION (pairs 13-14): [1,1,0,0,0] vs [1,0,1,1,1]. Cue1 ties, cue2 (.85) favours one option, cues 3-5 favour the other. TTB stops at cue2 and is at ceiling; the competitor computes core ~ .02 and lands near chance (p ~ .59). Confirms the cascade (not merely the single best cue) drives TTB, while remaining maximally diagnostic.

FAMILY F - DOMINANCE ANCHOR (pairs 15-16): [1,1,1,1,1] vs [0,0,0,0,0]. Both panels unanimous so the premium cancels exactly; both models are at ceiling. This pins beta and epsilon for BOTH models, so every other family is an out-of-sample prediction rather than a free fit.

DECISION RULE. If TTB is right: a single flat probability level (calibrated by Family F) accounts for Families S, P, D and E alike; accuracy on the single-endorsement pairs is high and IDENTICAL for cue1, cue3 and cue5; the three cue5-only cells (S5, P2, P3) are indistinguishable; and no coherence/unanimity effect appears anywhere. If RVW-DE+/U is right: accuracy on Family S falls steeply with cue validity and goes SUB-CHANCE for cue3 and cue5 (the all-zero panel is preferred to a weakly endorsed one), the cue5-only ladder shows a large non-monotone step (~.06 / ~.5 / ~.94) driven purely by unanimity, Family D shows near-chance means with maximal individual spread, and Family E hovers near chance.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Within-subject contrast designed to cancel every free noise parameter of Take-The-Best while maximally loading the competitor's strict-unanimity consensus premium.

Cell HIGH = pairs [1,1,1,1,1] vs [1,1,1,1,0] (and mirror): the two options differ on the LOWEST-validity cue only, and the winner's panel is unanimous. Cell LOW = pairs where one option carries a single weak endorsement (cue3 or cue5) and the other is the all-zero (also unanimous) panel; here the weakly endorsed option is the TTB winner but its panel is maximally incoherent while its rival is unanimous.

TTB: in both cells the first discriminating cue favours the target option, so the choice probability is exactly (1-eps)*sigmoid(beta)+eps/2 in BOTH cells, whatever beta and eps are. The contrast is therefore identically 0 in expectation for every subject, regardless of that subject's determinacy or lapse rate — the metric is immune to TTB's (very wide) parameter heterogeneity, leaving only small binomial sampling noise (12 + 24 trials per subject).

RVW-DE+/U: the weighted-evidence term is essentially the same tiny w5 (or w3) in both cells, but the direction-blind premium rho*|2m-1|^q with high q flips sign between cells: in HIGH the premium (~+rho) adds to the cue, in LOW it (~-rho) opposes the single endorsement, driving predicted accuracy from ~.94 down to ~.06-.13. Its predicted contrast is therefore large and positive (~.7-.9), with only moderate spread from rho/q/beta heterogeneity.

So: TTB predicts ~0, the competitor predicts a large positive step, and neither prediction is trivially guaranteed by the metric's construction (both cells are ordinary accuracy proportions on distinct trial types).

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    def key(x):
        return tuple(int(v) for v in list(x))

    A = [key(x) for x in data['option_a_ratings']]
    B = [key(x) for x in data['option_b_ratings']]
    R = [int(r) for r in data['response']]

    ones5 = (1, 1, 1, 1, 1)
    four = (1, 1, 1, 1, 0)
    zeros = (0, 0, 0, 0, 0)
    weak_single = [(0, 0, 1, 0, 0), (0, 0, 0, 0, 1)]

    hi_n = 0
    hi_d = 0
    lo_n = 0
    lo_d = 0

    for a, b, r in zip(A, B, R):
        if len(a) != 5 or len(b) != 5:
            continue
        # HIGH-coherence cell: 5/5 unanimous panel vs 4/5 panel, differ on cue5 only.
        if a == ones5 and b == four:
            hi_d += 1
            if r == 0:
                hi_n += 1
        elif b == ones5 and a == four:
            hi_d += 1
            if r == 1:
                hi_n += 1
        # LOW-coherence cell: single weak endorsement vs unanimous all-zero panel.
        elif (a in weak_single) and b == zeros:
            lo_d += 1
            if r == 0:
                lo_n += 1
        elif (b in weak_single) and a == zeros:
            lo_d += 1
            if r == 1:
                lo_n += 1

    if hi_d == 0 or lo_d == 0:
        return float('nan')

    p_hi = hi_n / float(hi_d)
    p_lo = lo_n / float(lo_d)
    return float(p_hi - p_lo)


### RESULTS
- Predicted under pi_1 (simulated): 0.0150 (var=0.0203)
- Predicted under pi_4 (simulated): 0.7083 (var=0.0508)
- Observed on real data: -0.0133 (var=0.0143)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.62, 0.83, 0.93, 0.54, 0.72]

**Trial pairs (n=24):**
  trial 1: A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  trial 5: A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 6: A=[1, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  trial 7: A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 8: A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  trial 9: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 11: A=[1, 1, 0, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 12: A=[1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 0]
  trial 13: A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0]
  trial 14: A=[1, 1, 1, 0, 0]  B=[1, 1, 1, 1, 0]
  trial 15: A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1]
  trial 16: A=[1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1]
  trial 17: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 18: A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  trial 19: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 20: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 21: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 22: A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 23: A=[1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 24: A=[0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate RVW-DE+/U (advocated) from Take-The-Best (competing) on three orthogonal signatures: (i) READABILITY FLATTENING, (ii) the STRICT-UNANIMITY consensus premium, (iii) flat-vs-graded choice probability. TTB's core commitment is that on every discriminating trial it sides with the highest-validity discriminating expert at ONE flat level sigmoid(beta) (plus lapse), regardless of the other cues, of endorsement counts, of coherence, and of where the expert is displayed.

DISPLAY (deliberately scrambled): col0=.62, col1=.83, col2=.93, col3=.54, col4=.72. Kendall tau between column index and validity is exactly 0 (5 concordant / 5 discordant pairs) and the best expert (.93) sits in a MIDDLE column, so under RVW kappa = k0 + k1*1 + k2*1 lies in [0.67, 1.0] across the whole parameter box: validity weights are strongly flattened toward equal weighting and the reading-order fallback (phi_eff = phi*(0.35+1.15*kappa)) is maximally recruited. Effective weights therefore run roughly [.18-.26, .22-.24, .20-.34, .12-.17, .13-.16] by column (mid-parameter ~[.223,.237,.253,.148,.138]) - the .93 expert never commands more than ~1/3 of the weight, and two or three other columns always outweigh it. TTB's cascade, by contrast, is col2 > col1 > col4 > col0 > col3, bound to experts, not columns.

12 base pairs, each mirrored (A/B swapped) so neither key is systematically correct: 24 unique pairs, K = 4, 96 trials. TTB predicts A on 11 of the 12 base pairs (and B on P1) at IDENTICAL probability throughout.

FAMILY L - COALITION LADDER (pairs 1-4 + mirrors). A carries only the .93 endorsement (col2); B carries a growing coalition of other columns: {col0}, {col0,col3}, {col0,col3,col4}, {col0,col1,col3,col4}. TTB stops at col2 and picks A at ceiling in all four, with zero modulation. RVW predicts a steep monotone ladder that CROSSES CHANCE: mid-parameter cores +.03 / -.12 / -.26 / -.49, i.e. p(A) ~ .55 / .30 / .14 / .03 at beta=7. Coherence is exactly cancelling in L1 and L4 (1/5 vs 1/5 and 1/5 vs 4/5 give identical |2m-1|) and negligible elsewhere, so this family is a pure weighted-integration/flattening test. L1 additionally is the near-cancelling cell where the RVW sign flips inside its own parameter box, so it should show the LARGEST between-subject spread of any family.

FAMILY U - UNANIMITY STEP WITH WEIGHTED EVIDENCE HELD EXACTLY CONSTANT (pairs 5-8 + mirrors). All four cells differ on col3 and ONLY col3, so the weighted-evidence term is identical and TTB must give the same ceiling probability in all four. Only panel coherence changes: (U1) 1/5 vs 0/5 - the LOSER is unanimous, so it collects nearly the whole premium; (U2) 3/5 vs 2/5 - neither panel coherent; (U3) 4/5 vs 3/5 - still no premium under high q even though mean rating rose; (U4) 5/5 vs 4/5 - the WINNER is unanimous. RVW predicts p(A) ~ .15 / .74 / .75 / .97: a sub-chance reversal when a lone weak endorsement faces an all-zero panel, a flat plateau across U2-U3 (which a smooth, low-q coherence rule could not produce - it would make U3 clearly exceed U2), and a jump at exactly 0 dissenters. This non-monotone step at the unanimity boundary, with weighted evidence pinned, is the sharpest falsifiable signature and is impossible under any re-weighting of cues.

FAMILY V - PREMIUM ADDITIVITY AT A HEAVIER COLUMN (pairs 11-12 + mirrors). Same manipulation on col0 (weight ~.22 rather than ~.15): [1,0,0,0,0] vs all-zeros, and [1,1,1,1,1] vs [0,1,1,1,1]. RVW predicts p(A) ~ .22 and ~.98 - the same rho-sized step, shifted upward by the larger cue weight (V1 > U1, V2 > U4). TTB again predicts one flat ceiling.

FAMILY P - POSITION VERSUS VALIDITY (pair 9 + mirror). Only col0 (.62, leftmost) and col4 (.72, rightmost) discriminate, endorsement counts and coherence identical, so theta and rho are inert. TTB decisively picks the col4 (.72) option. RVW, with flattened weights plus the recruited reading-order gradient, predicts the OPPOSITE (col0) option at p ~ .64 mid-parameter, ranging from chance to clearly reversed. Any failure to prefer the higher-validity expert here is fatal for TTB.

FAMILY D - DOMINANCE ANCHOR (pair 10 + mirror). [1,1,1,1,1] vs [0,0,0,0,0]: both panels unanimous so the premium cancels exactly, all weights point the same way. Both models are at ceiling; this pins beta and epsilon for BOTH models, making every other family an out-of-sample prediction rather than a free fit.

DECISION RULE. If TTB is right: one flat probability level, calibrated by Family D, accounts for L1-L4, U1-U4, V1-V2 alike (all choosing the option favoured by the first discriminating cue), no ladder gradation, no coherence effect, and Family P shows decisive choice of the .72 option. If RVW-DE+/U is right: the L ladder falls monotonically and crosses chance, the U ladder shows the non-monotone unanimity step (.15 / .74 / .75 / .97) with U2 = U3, V mirrors U at a higher level, Family P reverses or sits at chance, and between-subject spread peaks in L1 and P.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
Every term is a WITHIN-subject contrast between two cells in which Take-The-Best makes the *same* one-reason decision at the *same* flat probability sigmoid(beta)(1-eps)+eps/2, so each contrast has expectation exactly 0 under TTB regardless of that subject's beta and lapse rate (only binomial noise remains, ~8 trials/cell). (i) L1-L4: TTB stops at the .93 cue in both, picking the col2 option identically whether the rival carries one weak endorsement or four; RVW's flattened weights let the growing coalition overturn the top cue, so the col2 option falls from ~.55 to ~.03. (ii) U4-U1: both cells differ on col3 ALONE, so TTB's discriminating cue and its winner are identical; RVW's strict-unanimity premium makes the all-zero panel beat a lone weak endorsement (~.15) while a 5/5 panel beats a 4/5 panel (~.97). (iii) V2-V1: same manipulation at col0, again a single flat TTB level but a ~.22 vs ~.98 split under RVW. Averaging the three contrasts cuts per-subject binomial noise while preserving a mean gap of roughly 0.7 (RVW) versus 0.0 (TTB); because the TTB side is pinned at zero with only sampling variance, and the RVW side is driven by shared, strongly-signed effects, the between-subject spread stays small relative to the gap.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    df = data
    if len(df) == 0:
        return 0.0

    def as_tup(x):
        return tuple(int(v) for v in x)

    A = df['option_a_ratings'].apply(as_tup).values
    B = df['option_b_ratings'].apply(as_tup).values
    R = df['response'].values

    def frac_choose(target, other):
        """Fraction of trials with pair {target, other} on which the
        subject picked `target`. Collapses across A/B display order."""
        num = 0
        den = 0
        for a, b, r in zip(A, B, R):
            if a == target and b == other:
                den += 1
                if int(r) == 0:
                    num += 1
            elif a == other and b == target:
                den += 1
                if int(r) == 1:
                    num += 1
        if den == 0:
            return None
        return float(num) / float(den)

    # ---- cells -------------------------------------------------------
    # Family L (coalition ladder): does the lone top-validity (col2)
    # endorsement keep winning as the opposing coalition grows?
    L1 = frac_choose((0, 0, 1, 0, 0), (1, 0, 0, 0, 0))
    L4 = frac_choose((0, 0, 1, 0, 0), (1, 1, 0, 1, 1))

    # Family U (unanimity step, weighted evidence pinned to col3 only):
    # winner-unanimous cell vs loser-unanimous cell.
    U4 = frac_choose((1, 1, 1, 1, 1), (1, 1, 1, 0, 1))
    U1 = frac_choose((0, 0, 0, 1, 0), (0, 0, 0, 0, 0))

    # Family V (same step at a heavier column, col0).
    V2 = frac_choose((1, 1, 1, 1, 1), (0, 1, 1, 1, 1))
    V1 = frac_choose((1, 0, 0, 0, 0), (0, 0, 0, 0, 0))

    contrasts = []
    if L1 is not None and L4 is not None:
        contrasts.append(L1 - L4)
    if U4 is not None and U1 is not None:
        contrasts.append(U4 - U1)
    if V2 is not None and V1 is not None:
        contrasts.append(V2 - V1)

    if len(contrasts) == 0:
        return 0.0
    return float(np.mean(contrasts))


### RESULTS
- Predicted under pi_1 (simulated): -0.0092 (var=0.0076)
- Predicted under pi_4 (simulated): 0.4600 (var=0.0383)
- Observed on real data: -0.1500 (var=0.0125)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0.0
    n = 0.0
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float).ravel()
        b = np.asarray(row['option_b_ratings'], dtype=float).ravel()
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        # Take-The-Best winner: first discriminating cue in validity order.
        # Feature columns are already ordered by descending validity in this design.
        ttb_winner = None
        for j in range(a.shape[0]):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue
        # Tallying winner: majority of feature-wise wins (ties excluded).
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins == b_wins:
            continue
        tally_winner = 0 if a_wins > b_wins else 1
        if tally_winner == ttb_winner:
            continue  # not a conflict trial
        resp = row['response']
        if resp is None or (isinstance(resp, float) and np.isnan(resp)):
            continue
        hits += 1.0 if int(resp) == ttb_winner else 0.0
        n += 1.0

    if n == 0.0:
        return 0.5
    return float(hits / n)
```

**Observed (real) value:** 0.7533 (var=0.0228)
**Predicted under pi_1:** 0.8375 (var=0.0108)
**Predicted under pi_4:** 0.7196 (var=0.0166)

### Experiment 4
**Design**
  A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Fixed design validities: f0=.93, f1=.58, f2=.86, f3=.51, f4=.74, f5=.65
    val = np.array([0.93, 0.58, 0.86, 0.51, 0.74, 0.65], dtype=float)

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    hits = 0
    n = 0
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)
        if a.shape[0] != b.shape[0]:
            continue
        if a.shape[0] == val.shape[0]:
            order = np.argsort(-val, kind='stable')
        else:
            order = np.arange(a.shape[0])

        tally = int(np.sum(a > b)) - int(np.sum(b > a))

        winner = None
        for j in order:
            j = int(j)
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            continue  # no cue discriminates: uninformative for both models

        ttb_sign = 1 if winner == 0 else -1
        # Keep only trials where the tally does NOT favour the TTB winner:
        # either the tally is exactly tied (Family D) or it points the
        # other way (Family E, directional conflict).
        if tally * ttb_sign > 0:
            continue

        n += 1
        try:
            resp = int(r)
        except Exception:
            continue
        if resp == winner:
            hits += 1

    if n == 0:
        return 0.5
    return float(hits) / float(n)
```

**Observed (real) value:** 0.5100 (var=0.0136)
**Predicted under pi_1:** 0.8204 (var=0.0123)
**Predicted under pi_4:** 0.5204 (var=0.0106)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    validities = [0.55, 0.62, 0.70, 0.85, 0.93]

    def key(v):
        return ''.join([str(int(round(float(x)))) for x in v])

    # conflict families: TTB (top-validity cue) and a left-to-right positional
    # weighting scheme point to OPPOSITE options
    conflict_keys = {
        '10000|00001', '00001|10000',      # family A
        '01000|00010', '00010|01000',      # family B
        '11000|00011', '00011|11000',      # family E
        '00100|11000', '11000|00100',      # family F
    }
    # agreement families: both accounts favour the same option
    agree_keys = {
        '11111|00000', '00000|11111',      # family D (dominance)
        '00001|00000', '00000|00001',      # family G
        '10000|00000', '00000|10000',      # family H
    }

    def ttb_winner(a, b, order):
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return -1

    n_conf = 0
    hit_conf = 0
    n_agree = 0
    hit_agree = 0

    for _, row in data.iterrows():
        try:
            a = [int(round(float(x))) for x in row['option_a_ratings']]
            b = [int(round(float(x))) for x in row['option_b_ratings']]
            resp = int(row['response'])
        except Exception:
            continue
        n = len(a)
        if len(b) != n or n == 0:
            continue
        if n == len(validities):
            v = np.asarray(validities, dtype=float)
        else:
            # fallback: assume columns are displayed in ascending validity
            v = np.linspace(0.55, 0.93, n)
        order = list(np.argsort(-v, kind='stable'))
        w = ttb_winner(a, b, order)
        if w < 0:
            continue
        k = key(a) + '|' + key(b)
        if k in conflict_keys:
            n_conf += 1
            hit_conf += 1 if resp == w else 0
        elif k in agree_keys:
            n_agree += 1
            hit_agree += 1 if resp == w else 0

    if n_conf == 0 or n_agree == 0:
        return 0.0

    return float(hit_agree) / n_agree - float(hit_conf) / n_conf

```

**Observed (real) value:** -0.2081 (var=0.0368)
**Predicted under pi_1:** -0.0092 (var=0.0059)
**Predicted under pi_4:** -0.2069 (var=0.0176)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # Design validities (display order): col0..col5
    v_default = np.array([0.60, 0.93, 0.55, 0.86, 0.68, 0.78], dtype=float)

    def build(n):
        v = v_default.copy()
        if v.shape[0] < n:
            v = np.concatenate([v, np.full(n - v.shape[0], 0.55)])
        v = v[:n]
        v = np.clip(v, 0.5 + 1e-6, 1 - 1e-6)
        L = np.log(v / (1.0 - v))
        w = np.power(L, 1.5)
        wpos = np.sort(w)[::-1]
        weff = 0.2 * w + 0.8 * wpos          # lam = 0.8, mid of advocated box
        s = weff.sum()
        weff = weff / s if s > 0 else np.ones(n) / n
        cascade = list(np.argsort(-v, kind="stable"))
        return weff, cascade

    cache = {}
    num = 0.0
    den = 0.0

    for a_raw, b_raw, resp in zip(data["option_a_ratings"],
                                  data["option_b_ratings"],
                                  data["response"]):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = int(min(a.shape[0], b.shape[0]))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        if n not in cache:
            cache[n] = build(n)
        weff, cascade = cache[n]

        d = float(np.dot(weff, a - b))

        winner = None
        for j in cascade:
            if a[j] > b[j]:
                winner = 0
                break
            if b[j] > a[j]:
                winner = 1
                break
        if winner is None:
            continue
        if abs(d) < 0.20:                    # ambiguous / near-cancelling for WADD-pos
            continue
        pos_choice = 0 if d > 0 else 1
        if pos_choice == winner:             # models agree -> uninformative
            continue

        den += 1.0
        try:
            r = int(resp)
        except Exception:
            continue
        num += 1.0 if r == pos_choice else 0.0

    if den == 0:
        return 0.5
    return float(num / den)
```

**Observed (real) value:** 0.6587 (var=0.0379)
**Predicted under pi_1:** 0.1800 (var=0.0087)
**Predicted under pi_4:** 0.6633 (var=0.0149)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Pooling all six experiments gives a very clear, and internally consistent, picture in which each theory owns a disjoint half of the evidence.\n\nWhere TTB (pi_1) wins: Experiments 1 and 2 were engineered so that every cell involves the *same* first discriminating cue, making the metric identically zero under any one-reason rule but hugely positive under RVW's strict-unanimity premium. Observed: -0.013 (Exp 1) and -0.150 (Exp 2) versus RVW's +0.708 and +0.460. This is a decisive, two-experiment falsification of component (4) of pi_4: there is no direction-blind, high-exponent consensus/unanimity premium. A 5/5 panel does not beat a 4/5 panel any more than a lone weak endorsement beats an empty panel; if anything the ordering runs slightly the other way (both Exp 1 and Exp 2 observed values sit at or just below zero, hinting at a small *monotone* endorsement-count penalty rather than a coherence step).\n\nWhere TTB loses, and badly: Exp 4 (six scrambled-validity cues, conflict trials) observed 0.510 versus TTB's 0.820 \\u2014 subjects are exactly at chance when the tally opposes the top cue, i.e. behaviour is compensatory, not one-reason. Exp 5 observed -0.208 versus TTB's -0.009 (RVW -0.207). Exp 6 observed 0.659 versus TTB's 0.180 (RVW 0.663) \\u2014 with a scrambled display, subjects follow a *column-position*-weighted integration rather than the validity cascade on two-thirds of the disagreement trials. Exp 3 is roughly a tie (obs 0.753; TTB 0.838; RVW 0.720). So the display-binding/readability + graded compensatory-integration machinery of pi_4 (components 1\\u20133) is strongly supported, and TTB's frugality claim is refuted whenever the display is scrambled or a coalition opposes the top cue.\n\nNet: pi_4 has the right architecture but one clearly wrong module (the unanimity premium, and probably an over-large rho); pi_1 has the right *null* on coherence but the wrong integration rule. Neither is task-invariant, and no re-parameterisation of TTB can ever produce Exp 4/5/6 (its predictions there are structural, not parametric). The productive move is to replace TTB with a theory that shares pi_4's compensatory + readability-gated position-weighting core but swaps the all-or-none consensus premium for a smooth, monotone endorsement-count effect \\u2014 that variant is, on my reading, capable of hitting all six observations simultaneously and would be a genuinely sharp competitor to pi_4.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace Take-The-Best entirely with a compensatory, display-bound weighted-additive theory WITHOUT any coherence/unanimity term. Sketch:\n\n**Name:** Position-Bound Weighted Additive integration with faint-praise endorsement discounting (PWA-FP).\n\n**Core:** D = \\u03a3_j w_eff_j (a_j - b_j) - \\u03b8\\u00b7(\\u03a3a - \\u03a3b); p(A) = (1-\\u03b5)\\u00b7logistic(\\u03b2 D) + \\u03b5/2. All cues are always integrated (no stopping rule) \\u2014 this is what buys Exp 4's chance-level conflict performance and Exp 3's ~0.75.\n\n**Weights:** w_j is a convex mixture of (i) a validity-derived weight (e.g. \\u221d logit(v_j)^\\u03b3) bound to the correct expert, and (ii) a purely positional weight that decays left-to-right over displayed columns, w_pos \\u221d (n - j)^\\u03bb. The mixture coefficient is a *free* readability parameter that should be allowed to depend on how easy the validity-to-column mapping is (near-pure validity weighting when the display is monotone in validity, near-pure positional weighting when it is scrambled, best cue in a middle column). This is deliberately close to pi_4's kappa mechanism (which the data support: Exp 5 obs -0.208, Exp 6 obs 0.659) but formulated as an explicit position-vs-validity mixture rather than a flattening-toward-equal plus a small gradient, so it can generate the strong positional dominance seen in Exp 6 (0.66) while still letting the .93 cue win in Exp 5's monotone display.\n\n**Crucially, no consensus/coherence premium.** The rho\\u00b7|2\\u00b7mean-1|^q term must be dropped outright. In its place put a small, strictly MONOTONE endorsement-count discount \\u03b8 (0 to ~0.05 of total weight): each extra '1' slightly lowers an option's appeal (faint-praise skepticism). This reproduces (a) Exp 1's ~0 contrast (both the 5/5-vs-4/5 cell and the single-weak-endorsement-vs-empty cell are shifted in the *same* direction, so the difference stays near zero, and can be very slightly negative), (b) Exp 2's small negative -0.15 (U4-U1 and V2-V1 near zero or slightly negative because \\u03b8 penalises the higher-endorsement option equally in both cells, while L1-L4 is small once flattened weights let the 4-cue coalition partly offset the .93 cue), and (c) Exp 5's negative agree-minus-conflict value via near-chance/sub-chance responding on the weak-single-endorsement-vs-empty pair, without ever predicting the huge unanimity steps that Exps 1 and 2 refuted. Constrain \\u03b8 so that the dominance anchor ([1..1] vs [0..0]) stays at ceiling.\n\n**Parameter ranges:** \\u03b2 wide (2\\u201312), \\u03b3 ~0.8\\u20132.0, positional-mixture weight wide (0\\u20130.9) and tied to display monotonicity, \\u03b8 small (0\\u20130.06), \\u03b5 0\\u20130.15. No stopping/frugality parameter, no q, no rho.\n\nThis new theory competes with pi_4 on exactly the mechanism the data can still adjudicate \\u2014 smooth endorsement-count effects versus all-or-none unanimity, and flattening-toward-equal versus position-sorted re-weighting \\u2014 while both remain compensatory, so future experiments will be genuinely diagnostic rather than re-testing frugality, which is now settled."
}
```

## Usage

```json
{
  "input_tokens": 22579,
  "output_tokens": 8890
}
```
