# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
**RD-WEI: Relative (Normalized) Directional Weighted Evidence Integration.**

This keeps the two claims that carried the previous candidate and adds a third, which is the theoretical content of this revision.

(1) COMPRESSED, COMPENSATORY INTEGRATION. Subjects integrate ALL expert ratings. Each stated validity v_i is turned into normative evidence d_i = log(v_i/(1-v_i)), which enters the comparison after a strongly *compressive* transform w_i = d_i^gamma with gamma well below 1 (~0.2-0.5). The weight profile is therefore only mildly validity-ordered - much closer to a tally than to log-odds weighting - so no single expert can lexicographically out-weigh three or four others, yet validity still decides 1-vs-1 comparisons.

(2) SUBJECT-LEVEL POLARITY FRAME. The direction in which a positive binary rating pushes preference is a latent subject-level frame s in {-1,+1}: the decision variable is s * E. When the frame is inverted (s = -1, the overwhelming majority in these environments) a '1' is read as a mark/flag/non-selective endorsement and the subject prefers the option carrying the SMALLER validity-weighted count of positive ratings. Because in these designs the option with fewer marks is usually also the option favoured by the single most valid discriminating cue, reversed integration *masquerades* as one-reason (Take-The-Best) decision making on cue-conflict trials while producing strongly NEGATIVE weighted-margin contrasts on dominance/unanimity pairs - a signature no forward integrator and no lexicographic rule can produce.

(3) NEW: EVIDENCE IS READ IN RELATIVE, NOT ABSOLUTE, UNITS (divisive normalization). The margin is not evaluated in absolute log-odds units; it is evaluated as a *share of the evidence actually in play* on that trial. Only cues that discriminate enter the comparison, and the margin is divided by the total weight of those discriminating cues raised to a power kappa (~0.4-0.6):

    E' = s * [ sum_i w_i (a_i - b_i) ] / ( sum_{i: a_i != b_i} w_i )^kappa ,   P(A) = sigmoid(beta * E'), then lapse.

Psychologically this is a Weber-like normalization of the comparison: what governs confidence is how *lopsided* the discriminating evidence is, not how much of it there is. Two consequences distinguish RD-WEI from the un-normalized version: (a) a one-cue-versus-one-cue disagreement (few cues in play) is resolved almost as decisively as a 1-versus-5 rout, so single-cue and small-set discriminations stay sharp; (b) trials in which many cues discriminate but the split is nearly even (e.g. 4 marks vs 6 marks, or 2 vs 3 across a 7-cue array) become markedly noisier than their absolute margin would imply, and even unanimous/dominance pairs stop being perfectly deterministic once every cue is in play. This is exactly the graded-ness pattern the data show: near-ceiling reversed responding on lopsided splits, sub-ceiling responding on many-cue near-even splits, and sizeable between-subject disagreement precisely on the near-zero-share pairs.

The family still nests the classics: s=+1, gamma->0, kappa=0 = Tallying; s=+1, gamma=1, kappa=0 = Franklin's rule / naive Bayes; s=+1, gamma->inf = TTB; kappa=1 = pure average-diagnosticity-per-discriminating-cue; s=-1 gives the reversed-frame regime these environments demand.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Relative Directional Weighted Evidence Integration (RD-WEI).
    #   d_i = log(v_i/(1-v_i))                    normative cue evidence
    #   w_i = d_i ** gamma,  gamma < 1            COMPRESSED (near-tally) weights
    #   E   = sum_i w_i (a_i - b_i)               un-normalised weighted margin
    #   S   = sum of w_i over DISCRIMINATING cues (a_i != b_i)
    #   E'  = E / S**kappa                        relative (share-of-evidence) margin
    #   s   = -1 with prob p_reverse, else +1     subject-level polarity frame
    #   P(A) = sigmoid(s * beta * E'), then lapse epsilon toward uniform.
    # History is ignored (no feedback in this domain).
    import numpy as np

    stim = np.asarray(state, dtype=float)
    if stim.ndim == 1:
        if stim.shape[0] % 2 != 0:
            return np.ones(2) / 2.0
        half = stim.shape[0] // 2
        stim = np.vstack([stim[:half], stim[half:]])
    if stim.ndim != 2 or stim.shape[0] != 2:
        return np.ones(2) / 2.0

    n_features = int(stim.shape[1])
    if n_features == 0:
        return np.ones(2) / 2.0

    # ---- validities ------------------------------------------------
    try:
        v = np.asarray(parameters.get("validities", None), dtype=float).ravel()
    except Exception:
        v = np.array([])
    if v.size != n_features or not np.all(np.isfinite(v)):
        v = np.linspace(0.90, 0.55, n_features)
    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-6)

    # ---- compressed diagnosticity weights (un-normalised) ----------
    gamma = float(parameters["gamma"])
    d = np.maximum(np.log(v / (1.0 - v)), 1e-12)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        w = np.power(d, gamma)
    w = np.where(np.isfinite(w), w, 0.0)
    if not np.isfinite(np.sum(w)) or np.sum(w) <= 0.0:
        w = np.ones(n_features, dtype=float)

    # ---- weighted margin (forward frame) ---------------------------
    diff = stim[0] - stim[1]
    E = float(np.dot(w, diff))
    if not np.isfinite(E):
        E = 0.0

    # ---- divisive normalization by the evidence actually in play ----
    kappa = float(parameters["kappa"])
    disc = np.abs(diff) > 0.0
    S = float(np.sum(w[disc])) if np.any(disc) else 0.0
    if S > 1e-12 and kappa != 0.0:
        denom = S ** kappa
        if np.isfinite(denom) and denom > 1e-12:
            E = E / denom
    if not np.isfinite(E):
        E = 0.0

    # ---- subject-level polarity frame ------------------------------
    u = float(parameters["polarity_u"])
    q = float(parameters["p_reverse"])
    s = -1.0 if u < q else 1.0

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    z = np.array([s * beta * E, 0.0], dtype=float)
    z = np.clip(z, -500.0, 500.0)
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)

    p = (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64).ravel()
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        p = np.ones(p.shape[0]) / p.shape[0]
    else:
        p = p / s
    return int(np.random.choice(len(p), p=p))

## THEORY 2 — pi_5
**RFS-C+D: Reversed Flag Screening under Limited Capacity, with a Duel (severity-ordered) Mode for Sparse Difference Sets.**

Claims (1)-(3) are unchanged from the accepted RFS-C base:

(1) POLARITY: a binary '1' is read as a warning mark against an option, not an endorsement; the option carrying FEWER encoded marks wins. This is the empirically robust reversal seen on dominance, single-cue and cue-conflict pairs alike.

(2) VALIDITY GATES *READING*, NOT *WEIGHTING*: expert i is consulted with probability q_i = q_max * (2v_i - 1)^rho; once consulted it contributes exactly ONE mark, no heavier than any other consulted expert.

(3) ATTENTION IS AN ABSOLUTE BUDGET: if sum_i q_i exceeds capacity C, every q_i is rescaled by C / sum_i q_i, so larger expert panels are noisier per expert.

NEW CLAIM (4) — MODE SWITCH BY DIFFERENCE-SET SPARSITY. Counting marks is what people do when the two products differ on MANY experts and the identities of the flagging experts are lost in the tally. But when the two displays differ on only a couple of experts, the comparison degenerates into a *duel between two named experts*, and the reader can then compare how DAMNING each flag is: the flag issued by the more credible expert is the more serious warning, so the reader rejects the option flagged by the highest-validity consulted discriminating expert (a reversed, consultation-limited Take-The-Best). The probability of engaging this duel mode falls off steeply — as a Gaussian in the number of discriminating experts beyond a two-expert duel, p_duel = omega * exp(-lam * (n_disc - 2)^2) — because holding individual expert identities in the comparison is only feasible for very sparse difference sets. For n_disc = 1 the two modes are behaviourally identical (both reject the lone flagged option, and both guess when the lone expert goes unread), and for two flags on the SAME option they are also nearly identical; the two modes diverge only on genuine sparse *conflicts* (one flag on each side) and, weakly, on three-cue displays.

This reconciles the two facts that no pure counter and no pure lexicographic rule can hold together: on many-cue conflicts (one flag on the best expert vs four or five flags on weaker experts) people go with the FEWER-mark option ~80% of the time (so counting, not rank, decides there), yet on a one-flag-versus-one-flag duel (.85-expert flags A, .78-expert flags B) they reject the option flagged by the MORE credible expert ~75-80% of the time (so severity, not count, decides there). The family still nests: omega -> 0 recovers pure reversed flag counting; lam -> 0 with omega -> 1 recovers a fully reversed one-reason rule; q_max -> 1, C -> infinity, beta -> infinity gives deterministic reversed tallying.

It remains cleanly separable from RD-WEI: RD-WEI predicts a share-driven graded tilt on top-heavy near-even splits and a flat single-cue ladder, whereas this account predicts near-chance on top-heavy near-even MANY-cue splits (counting is blind to cue identity), a steep single-cue ladder tracking consultation probability, panel-size-graded noise, and a sharp qualitative discontinuity between two-cue duels (rank-resolved) and four-plus-cue conflicts (count-resolved).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # RFS-C+D: Reversed Flag Screening under limited attentional capacity,
    # plus a severity-ordered DUEL mode for sparse difference sets.
    #   q_i = q_max * (2 v_i - 1) ** rho          probability expert i is consulted
    #   if sum_i q_i > capacity: q *= capacity / sum(q)   (attention is divided)
    #   COUNT MODE: consulted experts contribute ONE mark each; fewer marks wins
    #       P(A) = E[ sigmoid( beta * (c_B - c_A) ) ]
    #   DUEL MODE (weight p_duel = omega * exp(-lam*(n_disc-2)^2)):
    #       scan discriminating experts in descending validity; the FIRST consulted
    #       one issues the most damning flag -> reject the option it flags;
    #       if none of them is consulted -> guess.
    #   then lapse epsilon toward uniform.
    # History is ignored (no feedback in this domain).
    import numpy as np

    # ---------------- unpack stimulus ----------------
    stim = None
    if isinstance(state, dict):
        a_raw = state.get("option_a_ratings", None)
        b_raw = state.get("option_b_ratings", None)
        if a_raw is not None and b_raw is not None:
            try:
                a_v = np.asarray(a_raw, dtype=float).ravel()
                b_v = np.asarray(b_raw, dtype=float).ravel()
                if a_v.size == b_v.size and a_v.size > 0:
                    stim = np.vstack([a_v, b_v])
            except Exception:
                stim = None
    if stim is None:
        try:
            stim = np.asarray(state, dtype=float)
        except Exception:
            return np.ones(2) / 2.0
        if stim.ndim == 1:
            if stim.shape[0] % 2 != 0:
                return np.ones(2) / 2.0
            h = stim.shape[0] // 2
            stim = np.vstack([stim[:h], stim[h:]])
    if stim.ndim != 2 or stim.shape[0] != 2 or stim.shape[1] == 0:
        return np.ones(2) / 2.0

    n_features = int(stim.shape[1])

    # ---------------- validities ----------------
    try:
        v = np.asarray(parameters.get("validities", None), dtype=float).ravel()
    except Exception:
        v = np.array([])
    if v.size != n_features or not np.all(np.isfinite(v)):
        v = np.linspace(0.90, 0.55, n_features)
    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-6)

    rho = float(parameters["rho"])
    q_max = float(parameters["q_max"])
    cap = float(parameters["capacity"])
    beta = float(parameters["beta"])
    eps = float(parameters["epsilon"])
    omega = float(parameters["omega"])
    lam = float(parameters["lam"])

    # ------- validity-graded consultation probabilities -------
    strength = np.clip(2.0 * v - 1.0, 1e-9, 1.0)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        q = q_max * np.power(strength, rho)
    q = np.where(np.isfinite(q), q, q_max)
    q = np.clip(q, 1e-6, 1.0)

    # ------- absolute capacity: attention is divided, not added -------
    tot = float(np.sum(q))
    if np.isfinite(tot) and tot > cap and tot > 1e-12 and cap > 0:
        q = q * (cap / tot)
    q = np.clip(q, 1e-6, 1.0)

    # ------- which experts mark which option -------
    diff = stim[0] - stim[1]
    q_mark_a = q[diff > 0.0]   # a marked (1), b not -> a mark AGAINST A
    q_mark_b = q[diff < 0.0]   # b marked, a not  -> a mark AGAINST B
    # experts that mark both options contribute one mark to each -> cancel out

    def _pb(probs):
        # Poisson-binomial pmf of the number of consulted marks
        d = np.array([1.0], dtype=float)
        for p in probs:
            p = float(min(max(p, 0.0), 1.0))
            d = np.convolve(d, np.array([1.0 - p, p], dtype=float))
        s = d.sum()
        if not np.isfinite(s) or s <= 0:
            return np.array([1.0])
        return d / s

    da = _pb(q_mark_a)
    db = _pb(q_mark_b)

    ka = np.arange(da.size, dtype=float)
    kb = np.arange(db.size, dtype=float)
    # positive difference (more marks against B) favours A
    D = kb[None, :] - ka[:, None]
    z = np.clip(beta * D, -500.0, 500.0)
    sig = 1.0 / (1.0 + np.exp(-z))
    joint = da[:, None] * db[None, :]
    p_a = float(np.sum(joint * sig))
    if not np.isfinite(p_a):
        p_a = 0.5
    p_a = min(max(p_a, 0.0), 1.0)

    # ------- DUEL mode: severity-ordered screening on sparse difference sets -------
    disc = np.nonzero(diff)[0]
    n_disc = int(disc.size)
    if n_disc >= 2:
        p_mode = omega * float(np.exp(-lam * float(n_disc - 2) ** 2))
        if not np.isfinite(p_mode):
            p_mode = 0.0
        p_mode = min(max(p_mode, 0.0), 1.0)
        if p_mode > 1e-6:
            order = disc[np.argsort(-v[disc], kind="stable")]
            sg = 1.0 / (1.0 + np.exp(-float(np.clip(beta, -500.0, 500.0))))
            remain = 1.0
            p_lex = 0.0
            for j in order:
                qj = float(q[j])
                if diff[j] > 0.0:
                    # this expert flags A -> reject A
                    p_lex += remain * qj * (1.0 - sg)
                else:
                    # this expert flags B -> reject B
                    p_lex += remain * qj * sg
                remain *= (1.0 - qj)
            p_lex += remain * 0.5   # no discriminating expert was consulted -> guess
            if np.isfinite(p_lex):
                p_lex = min(max(p_lex, 0.0), 1.0)
                p_a = (1.0 - p_mode) * p_a + p_mode * p_lex

    p_core = np.array([p_a, 1.0 - p_a], dtype=float)
    p = (1.0 - eps) * p_core + eps * (np.ones(2) / 2.0)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64).ravel()
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        p = np.ones(p.shape[0]) / p.shape[0]
    else:
        p = p / s
    return int(np.random.choice(len(p), p=p))

## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=6):** [0.72, 0.95, 0.55, 0.9, 0.6, 0.8]

**Trial pairs (n=18):**
  trial 1: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 2: A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 4: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  trial 5: A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 7: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 8: A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 9: A=[1, 0, 0, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  trial 10: A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 11: A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 1, 1]
  trial 12: A=[1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 1, 1]
  trial 13: A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 14: A=[0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 16: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 17: A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  trial 18: A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]

**Rationale:** GOAL: dissociate RD-WEI (compressed near-tally weights, reversed polarity s=-1, divisive normalization of the margin by the weight of the DISCRIMINATING cues) from RFS-C+D (reversed flag COUNTING under an absolute attention budget, plus a rank-lexicographic DUEL mode on sparse difference sets). Both theories share the reversed polarity claim (fewer marks = better), so overall direction cannot separate them; the design therefore targets three quantitative signatures where they diverge: (i) how a 1-flag-vs-1-flag duel is resolved (graded in the WEIGHT GAP for RD-WEI vs rank-lexicographic and graded in CONSULTATION LEVEL for RFS-C+D), (ii) the height and slope of the single-cue ladder (normalization makes RD-WEI near-ceiling and almost flat; consultation failure makes RFS-C+D much lower and steeper), and (iii) a count-vs-weight magnitude test on a 2-vs-2 top-heavy conflict.

ENVIRONMENT: six experts in SCRAMBLED display order [0.72, 0.95, 0.55, 0.90, 0.60, 0.80] so no 'first-listed expert' rule can imitate either model. Log-odds d = (0.944, 2.944, 0.201, 2.197, 0.405, 1.386). At RD-WEI's gamma~0.33 the weights are w = (0.98, 1.43, 0.59, 1.30, 0.74, 1.11) (only ~2.4:1 spread). At RFS-C+D's rho~0.35, q_max~0.85, capacity~3.1 the consultation probabilities are q ~ (0.52, 0.665, 0.31, 0.64, 0.39, 0.58). Crucially the panel is built with TWO near-equal high experts (0.95/0.90), TWO near-equal mid experts (0.80/0.72), TWO near-equal weak experts (0.60/0.55) and one maximal gap (0.95 vs 0.55).

NINE TYPES, each mirrored across A/B so no side bias can mimic either model (18 unique pairs, K=5, 90 trials). Predicted p(A) at representative parameters (RD-WEI: gamma=0.33, kappa=0.52, beta=2.1, s=-1; RFS-C+D: rho=0.35, q_max=0.85, C=3.1, beta=4.5, omega=0.92, lam=1.4):

(TA, pairs 1-2) DUEL, small gap / HIGH level: 0.95 flags A, 0.90 flags B. RD-WEI ~0.46 (the compressed weights 1.43 vs 1.30 differ by almost nothing, so after normalization by S the trial is essentially a coin flip). RFS-C+D ~0.29 (duel mode is RANK-lexicographic and magnitude-blind: the 0.95 expert is consulted first with q=0.665 and its flag condemns A).
(TB, pairs 3-4) DUEL, small gap / LOW level: 0.60 flags A, 0.55 flags B. RD-WEI ~0.43 (again near chance, essentially the SAME value as TA: RD-WEI's duel resolution depends on the weight GAP, not on the validity level). RFS-C+D ~0.40 (both experts are rarely consulted, so the duel usually ends in a guess). TA vs TB is the cleanest interaction test: RD-WEI predicts NO level effect (0.46 vs 0.43), RFS-C+D predicts a large one (0.29 vs 0.40).
(TC, pairs 5-6) DUEL, small gap / MID level: 0.80 flags A, 0.72 flags B. RD-WEI ~0.45; RFS-C+D ~0.33.
(TD, pairs 7-8) DUEL, MAXIMAL gap: 0.95 flags A, 0.55 flags B. Both models ~0.23 (they agree here). Combined with TA/TB/TC this yields the decisive ladder: RD-WEI predicts strong GAP-gradedness (0.46, 0.45, 0.43 -> 0.23), RFS-C+D predicts a nearly FLAT duel profile (0.29, 0.33, 0.40 -> 0.23) that is ordered by consultation probability instead. No reparameterisation lets one model produce the other's ladder: RFS-C+D's duel mode cannot return to chance for two high-validity, near-equal experts, and RD-WEI cannot be decisive when the two duelling weights are nearly identical.
(TE, pairs 9-10) SINGLE CUE 0.55 (other cues tie, two of them tied at 1 so the displays are non-degenerate). RD-WEI ~0.84 (after normalization a 1-vs-0 discrimination gives E' = w^(1-kappa) ~ 0.78 even for the worst expert). RFS-C+D ~0.655 (the flag is read only ~31% of the time; otherwise a guess).
(TF, pairs 11-12) SINGLE CUE 0.72. RD-WEI ~0.89; RFS-C+D ~0.76.
(TG, pairs 13-14) SINGLE CUE 0.95. RD-WEI ~0.92; RFS-C+D ~0.83. TE-TG give the second orthogonal handle: RD-WEI predicts a HIGH, nearly FLAT single-cue ladder (0.84, 0.89, 0.92; span ~0.09) because divisive normalization cancels most of the cue-strength difference, whereas RFS-C+D predicts a much LOWER, twice-as-steep ladder (0.655, 0.76, 0.83; span ~0.18) tracking q_i = q_max*(2v-1)^rho under the capacity budget.
(TH, pairs 15-16) UNANIMOUS 6-0 dominance. Both models ~0.99 for the all-0 option. This pair is included as a LAPSE ANCHOR: it pins epsilon for both models so the TE-TG levels cannot be mimicked by simply raising the lapse rate (a large epsilon that pushes RD-WEI down to RFS-C+D's 0.655 on TE would also push dominance well below ceiling).
(TI, pairs 17-18) 2-vs-2 TOP-HEAVY CONFLICT: A flagged by 0.95+0.90, B flagged by 0.60+0.55 (four cues discriminate, so RFS-C+D's duel mode is switched off, p_duel ~ 0.003). RD-WEI ~0.20 (weighted marks 2.73 vs 1.33, decisive against A); RFS-C+D ~0.28 (counts are EQUAL, 2 vs 2; only the asymmetry in reading probability tilts it). A pure magnitude test in a regime where both models agree in direction.

DIAGNOSTIC SUMMARY of p(A) across TA..TI: RD-WEI = (0.46, 0.43, 0.45, 0.23, 0.84, 0.89, 0.92, 0.99, 0.20); RFS-C+D = (0.29, 0.40, 0.33, 0.23, 0.655, 0.76, 0.83, 0.99, 0.28). The vectors agree in sign everywhere (both are reversed-polarity accounts) but differ in SHAPE in ways neither can reparameterise away: RD-WEI's duel resolution is gap-graded and level-independent while RFS-C+D's is level-graded and gap-independent; RD-WEI's single-cue profile is high and flat while RFS-C+D's is low and steep, with the dominance pair fixing the lapse rate that would otherwise trade off against that level difference.

**Computed schedule:** 18 unique pairs × 5 reps = 90 trials per subject.



### METRIC
Rationale:
Two orthogonal signatures that the two accounts cannot reparameterise into each other are combined into one contrast with opposite signs, so the gap adds while each component stays reasonably tight (30 trials each per subject).

(1) SINGLE-CUE LEVEL (n_disc == 1, trial types TE/TF/TG). RD-WEI's divisive normalization means a lone discriminating cue is evaluated as a share of the (tiny) evidence in play, so E' = w^(1-kappa) is large even for the weakest expert: reversed responding is HIGH and nearly flat (~0.84-0.92, mean ~0.88). RFS-C+D reads the lone expert only with probability q_i under an absolute attention budget (q ~ 0.23-0.70 after capacity rescaling) and otherwise guesses, so its single-cue reversal rate is much LOWER (~0.65-0.83, mean ~0.72-0.76). The unanimous-dominance pairs in the design pin the lapse rate, so RD-WEI cannot be pushed down to that level without wrecking dominance performance.

(2) NEAR-EQUAL-VALIDITY DUELS (n_disc == 2, one flag on each side, |v_hi - v_lo| <= 0.20; types TA/TB/TC, with the maximal-gap TD pair excluded because both theories agree there). RD-WEI resolves duels by the WEIGHT GAP; with compressed weights (0.95 vs 0.90 gives 1.43 vs 1.30) these trials are close to a coin flip, so the rate of rejecting the option flagged by the more credible expert is only ~0.55. RFS-C+D's duel mode is rank-lexicographic and magnitude-blind: the higher-validity discriminating expert is consulted first and its flag condemns that option, giving ~0.60-0.71.

Difference score = single-cue reversal - duel 'reject-more-credible' rate. Predicted ~0.33 for RD-WEI vs ~0.09 for RFS-C+D, a ~0.24 gap, while per-subject values stay tight because each component averages 30 trials and both components are near-deterministic functions of the model parameters.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # design validities (display order)
    V = np.array([0.72, 0.95, 0.55, 0.90, 0.60, 0.80], dtype=float)

    sc_n = 0; sc_k = 0        # single-cue trials (n_disc == 1)
    du_n = 0; du_k = 0        # small-gap 1-vs-1 duels (n_disc == 2, one flag each side)

    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        try:
            a = np.asarray(list(a_raw), dtype=float)
            b = np.asarray(list(b_raw), dtype=float)
        except Exception:
            continue
        if a.size != b.size or a.size == 0:
            continue
        n = a.size
        if V.size >= n:
            v = V[:n]
        else:
            v = np.linspace(0.90, 0.55, n)
        diff = a - b
        disc = np.nonzero(diff)[0]
        try:
            r = int(resp)
        except Exception:
            continue
        if r not in (0, 1):
            continue

        if disc.size == 1:
            j = int(disc[0])
            # A carries the lone extra mark -> "reversed" (fewer-marks) choice is B
            chose_unflagged = (r == 1) if diff[j] > 0 else (r == 0)
            sc_n += 1
            sc_k += 1 if chose_unflagged else 0

        elif disc.size == 2 and (np.sum(diff[disc] > 0) == 1) and (np.sum(diff[disc] < 0) == 1):
            vv = v[disc]
            gap = float(abs(vv[0] - vv[1]))
            if gap <= 0.20:   # near-equal validity duel (excludes the 0.95 vs 0.55 pair)
                hi = int(disc[int(np.argmax(vv))])
                # option flagged by the MORE credible expert is rejected?
                rejected_hi = (r == 1) if diff[hi] > 0 else (r == 0)
                du_n += 1
                du_k += 1 if rejected_hi else 0

    sc = (sc_k / sc_n) if sc_n > 0 else 0.5
    du = (du_k / du_n) if du_n > 0 else 0.5
    return float(sc - du)

### RESULTS
- Predicted under pi_4 (simulated): 0.3260 (var=0.0118)
- Predicted under pi_5 (simulated): 0.0600 (var=0.0133)
- Observed on real data: 0.1453 (var=0.0133)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=8):** [0.7, 0.97, 0.55, 0.88, 0.62, 0.78, 0.93, 0.66]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1, 0, 1]
  trial 2: A=[1, 0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1, 0, 1]
  trial 3: A=[0, 0, 1, 1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0, 1, 1, 0]
  trial 4: A=[0, 1, 1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 0, 1, 1, 0]
  trial 5: A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 1, 0]
  trial 6: A=[0, 0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1, 1, 0]
  trial 8: A=[1, 0, 0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0]
  trial 9: A=[0, 0, 0, 0, 1, 1, 0, 1]  B=[0, 0, 1, 0, 0, 1, 0, 1]
  trial 10: A=[0, 0, 1, 0, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 1, 0, 1]
  trial 11: A=[1, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  trial 12: A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  trial 14: A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 1, 1, 0]
  trial 16: A=[0, 1, 1, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0]

**Rationale:** GOAL: dissociate RFS-C+D (reversed flag COUNTING under an absolute attention budget, with a rank-lexicographic DUEL mode on sparse difference sets) from RD-WEI (compressed near-tally weights, reversed polarity, divisive normalization by the discriminating-cue mass). Both are reversed-polarity accounts, so direction cannot separate them; earlier work (experiment [6]) already probed a 6-expert panel. Here I exploit two handles that a 6-expert panel cannot deliver and that no reparameterisation can erase.

HANDLE 1 - PANEL-SIZE / CAPACITY DILUTION. I use EIGHT experts, displayed in SCRAMBLED order [0.70, 0.97, 0.55, 0.88, 0.62, 0.78, 0.93, 0.66] (idx 0..7), so no 'first-listed expert' rule can mimic anything. Raw consultation probabilities q_i = q_max(2v-1)^rho (q_max=.85, rho=.35) sum to ~5.19, far above capacity C~3.1, so every q is rescaled by .60: q = (.369, .497, .227, .461, .308, .415, .482, .341). Consequently RFS-C+D is HARD-CAPPED on sparse trials: a single flag can be read at most ~50% of the time, so the best possible single-cue accuracy is ~0.75 and even two top-expert flags cap near 0.86 - while unanimity is still ~0.99 (some flag is almost surely read). RD-WEI has no such bound: after divisive normalization a 1-vs-0 discrimination gives E' = w^(1-kappa) regardless of how many experts are on the panel, so single-cue trials sit at 0.84-0.93 and 2-flag trials at 0.92-0.97. The dominance pair pins the lapse rate, so RD-WEI cannot buy RFS's low sparse-trial level by raising epsilon or lowering beta without also sinking the dominance anchor.

HANDLE 2 - DUEL LEVEL x GAP CROSSOVER (the headline, parameter-robust interaction). Three 1-flag-vs-1-flag duels are matched in structure but placed at three validity LEVELS whose compressed-weight GAPS run in the OPPOSITE direction: H = .97 vs .93 (q .497/.482, w gap small, large S), M = .88 vs .78, L = .62 vs .55 (q .308/.227, w gap relatively large, tiny S). RFS-C+D's duel is rank-lexicographic and magnitude-blind, so its decisiveness tracks CONSULTATION probability and must FALL as validity level falls; RD-WEI's duel resolution tracks the normalized weight gap (w1-w2)/S^kappa, which here RISES as level falls. Deviations from chance: RFS (.115, .109, .071) versus RD-WEI (.042, .059, .088) - a full crossover that holds across the whole stated parameter ranges (gamma .18-.48 all keep RD-WEI's ordering; any rho/C keep RFS's ordering).

EIGHT TYPES, each mirrored across A/B so no side bias can imitate either model (16 unique pairs, K=6, 96 trials). Predicted p(A) at representative parameters (RFS-C+D: rho=.35, q_max=.85, C=3.1, beta=4.5, omega=.92, lam=1.4, eps=0; RD-WEI: gamma=.33, kappa=.52, beta=2.1, s=-1, eps=0):
(T1, pairs 1-2) SINGLE CUE 0.55 (idx2), all else tied: RFS .611 | RD-WEI .836.
(T2, pairs 3-4) SINGLE CUE 0.97 (idx1): RFS .746 | RD-WEI .928. T1/T2 give the single-cue ladder: RFS LOW and STEEP (.611->.746, span .135, exactly q-graded), RD-WEI HIGH and FLAT (.836->.928, span .09).
(T3, pairs 5-6) DUEL H: 0.97 flags A, 0.93 flags B: RFS .385 | RD-WEI .458.
(T4, pairs 7-8) DUEL M: 0.88 flags A, 0.78 flags B: RFS .391 | RD-WEI .441.
(T5, pairs 9-10) DUEL L: 0.62 flags A, 0.55 flags B: RFS .429 | RD-WEI .412. T3-T5 is the crossover: RFS predicts monotonically WEAKER duel resolution as level drops, RD-WEI predicts monotonically STRONGER.
(T6, pairs 11-12) TWO WEAK FLAGS vs ZERO (0.55+0.62 on one side): RFS .727 | RD-WEI .921.
(T7, pairs 13-14) UNANIMOUS 8-0 dominance: both ~.99 - the LAPSE/BETA ANCHOR that blocks a global-noise mimicry of the sparse-trial level difference.
(T8, pairs 15-16) TWO STRONG FLAGS vs ZERO (0.97+0.93 on one side): RFS .862 | RD-WEI .970. T6 vs T8 isolates the consultation bound: RFS predicts a large weak-vs-strong flag-pair spread (.135) far below ceiling; RD-WEI predicts both near ceiling with a small spread (.049).

DIAGNOSTIC SUMMARY of p(A) across T1..T8: RFS-C+D = (.611, .746, .385, .391, .429, .727, .991, .862); RD-WEI = (.836, .928, .458, .441, .412, .921, .997, .970). The vectors agree in sign everywhere (both reversed-polarity) but differ in SHAPE in two independent, non-reparameterisable ways: (i) sparse-trial decisiveness capped well below ceiling and steeply q-graded versus near-ceiling and flat, with dominance fixed at ceiling for both; (ii) an outright crossover in how a two-expert duel is resolved as a function of validity LEVEL versus weight GAP.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Both theories are reversed-polarity accounts, so the SIGN of choices cannot separate them; what separates them is the DECISIVENESS on sparse, non-conflicting displays (one or two flags all on the same option, i.e. trial types T1, T2, T6, T8 of the design: 48 of the 96 trials).

RFS-C+D is hard-capped there: a flag can only influence the choice if the issuing expert is actually consulted, and the absolute attention budget (sum q ~5.19 rescaled to C~3.1 across 8 experts) forces q_i <= ~0.50. A single 0.55-validity flag therefore moves p(reversed) only to ~.61, a single 0.97 flag to ~.75, two weak flags to ~.73, two strong flags to ~.86 -- an average of about .74, well below ceiling and steeply graded by consultation probability. RD-WEI has no consultation bottleneck: after divisive normalization a 1-vs-0 discrimination yields E' = w^(1-kappa) irrespective of panel size, putting all four of these cells at .84-.97, average ~.91.

The metric is the absolute deviation from chance of the fraction of 'fewer-marks' choices restricted to those sparse non-conflicting trials. Predicted values: ~0.24 (RFS-C+D) vs ~0.39-0.41 (RD-WEI). Taking the absolute deviation (rather than the raw proportion) makes the statistic invariant to RD-WEI's latent subject-level polarity frame (p_reverse < 1 occasionally yields a forward-frame subject), which would otherwise inject enormous between-subject variance into the competitor and kill the t-test. Restricting to 48 trials per subject keeps binomial noise small (sd ~.06), and both theories' parameter ranges leave their respective cells on the same side of the gap, so the contrast should be both large in mean and tight per subject.

Crucially, this is not trivially true of my theory: it is a quantitative capacity-dilution prediction (sparse-trial decisiveness far below the ceiling fixed by the unanimity anchor), which a pure reversed integrator with any beta/epsilon cannot reproduce without also destroying its dominance-pair performance.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    hits = []
    for a_raw, b_raw, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        try:
            a = np.asarray(list(a_raw), dtype=float).ravel()
            b = np.asarray(list(b_raw), dtype=float).ravel()
        except Exception:
            continue
        if a.size == 0 or a.size != b.size:
            continue
        d = a - b
        nz = np.nonzero(d)[0]
        nd = int(nz.size)
        # sparse, NON-conflicting difference sets only: 1 or 2 discriminating
        # cues, all favouring the same side (all flags on one option)
        if nd < 1 or nd > 2:
            continue
        signs = np.sign(d[nz])
        if not np.all(signs == signs[0]):
            continue
        try:
            r = int(resp)
        except Exception:
            continue
        if signs[0] > 0:
            # A carries the extra 1s -> 'reversed'/fewer-marks choice is B (response 1)
            hits.append(1.0 if r == 1 else 0.0)
        else:
            # B carries the extra 1s -> reversed choice is A (response 0)
            hits.append(1.0 if r == 0 else 0.0)
    if len(hits) == 0:
        return 0.0
    p = float(np.mean(hits))
    return float(abs(p - 0.5))

### RESULTS
- Predicted under pi_4 (simulated): 0.3908 (var=0.0031)
- Predicted under pi_5 (simulated): 0.2092 (var=0.0056)
- Observed on real data: 0.3567 (var=0.0085)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    a_all = data['option_a_ratings'].tolist()
    b_all = data['option_b_ratings'].tolist()
    resp = data['response'].tolist()

    hits = 0
    n = 0
    for a, b, r in zip(a_all, b_all, resp):
        a = np.asarray(a, dtype=float).ravel()
        b = np.asarray(b, dtype=float).ravel()
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        # TTB winner: features are ordered by descending validity in this design,
        # so scan left to right for the first discriminating cue.
        ttb = None
        for j in range(a.shape[0]):
            if a[j] > b[j]:
                ttb = 0
                break
            if b[j] > a[j]:
                ttb = 1
                break
        if ttb is None:
            continue
        # Tally winner: count strict feature-wise wins.
        aw = float(np.sum(a > b))
        bw = float(np.sum(b > a))
        if aw == bw:
            continue
        tal = 0 if aw > bw else 1
        # Keep only trials where the two heuristics point to OPPOSITE options.
        if tal == ttb:
            continue
        n += 1
        try:
            rr = int(r)
        except Exception:
            continue
        if rr == ttb:
            hits += 1

    if n == 0:
        return float('nan')
    return float(hits) / float(n)
```

**Observed (real) value:** 0.7367 (var=0.0369)
**Predicted under pi_4:** 0.7429 (var=0.0044)
**Predicted under pi_5:** 0.7229 (var=0.0050)

### Experiment 4
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    validities = np.array([0.60, 0.93, 0.55, 0.80, 0.68, 0.57], dtype=float)

    def ttb_winner(a, b, val):
        order = np.argsort(-val, kind="stable")
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return -1

    hits = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        n = a.shape[0]
        val = validities if validities.shape[0] == n else np.linspace(0.9, 0.55, n)

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        margin = a_wins - b_wins
        if margin > 0:
            tally_win = 0
        elif margin < 0:
            tally_win = 1
        else:
            continue  # tally undecided: excluded

        tw = ttb_winner(a, b, val)
        if tw < 0 or tw == tally_win:
            continue  # only trials where the two rules point opposite ways
        if abs(margin) < 3:
            continue  # keep only large-margin conflicts (strong tally signal)

        hits.append(1.0 if int(row["response"]) == tally_win else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.1889 (var=0.0124)
**Predicted under pi_4:** 0.1622 (var=0.0193)
**Predicted under pi_5:** 0.1711 (var=0.0033)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    if len(data) == 0:
        return 0.0

    A = np.array([[int(x) for x in r] for r in data['option_a_ratings']], dtype=float)
    B = np.array([[int(x) for x in r] for r in data['option_b_ratings']], dtype=float)
    resp = np.asarray(data['response'], dtype=int)
    n_feat = A.shape[1]

    val = np.array([0.72, 0.85, 0.55, 0.78, 0.60, 0.66], dtype=float)
    if val.shape[0] != n_feat:
        val = np.linspace(0.90, 0.55, n_feat)
    val = np.clip(val, 0.5 + 1e-6, 1.0 - 1e-6)

    order = np.argsort(-val, kind='stable')

    # log-odds diagnosticity weights with a mid-range gamma (reference AEWI weights)
    d = np.log(val / (1.0 - val))
    w = np.power(np.maximum(d, 1e-12), 1.4)
    beta_ref = 2.0

    diff = A - B  # +1 favours A on that cue, -1 favours B

    # Take-The-Best winner per trial: first discriminating cue in validity order
    n = A.shape[0]
    ttb = np.full(n, -1, dtype=int)  # 0 = A, 1 = B, -1 = no discrimination
    undecided = np.ones(n, dtype=bool)
    for j in order:
        dj = diff[:, j]
        pickA = undecided & (dj > 0)
        pickB = undecided & (dj < 0)
        ttb[pickA] = 0
        ttb[pickB] = 1
        undecided = undecided & (dj == 0)

    valid = ttb >= 0
    if not np.any(valid):
        return 0.0

    # Signed absolute-evidence margin IN FAVOUR OF the TTB-chosen option
    E = diff.dot(w)
    sign_ttb = np.where(ttb == 0, 1.0, -1.0)
    margin = E * sign_ttb
    p_aewi = 1.0 / (1.0 + np.exp(-beta_ref * margin))

    agree = (resp == ttb).astype(float)

    high = valid & (p_aewi > 0.65)   # integration and one-reason agree strongly
    low = valid & (p_aewi < 0.35)    # integration reverses the one-reason choice

    if high.sum() == 0 or low.sum() == 0:
        return 0.0

    return float(agree[high].mean() - agree[low].mean())
```

**Observed (real) value:** -0.6733 (var=0.0205)
**Predicted under pi_4:** -0.6700 (var=0.0086)
**Predicted under pi_5:** -0.6817 (var=0.0102)

### Experiment 6
**Design**
  A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 1]  B=[0, 1, 0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Within-subject contrast: TTB-consistency on large-margin AGREEMENT pairs
    minus TTB-consistency on large-margin CONFLICT pairs (where the weighted
    log-odds evidence points opposite to the single best discriminating cue).
    TTB must give the same choice probability in both sets (=> ~0);
    AEWI predicts ~0.99 vs ~0.15 (=> ~0.85)."""
    v = np.array([0.74, 0.90, 0.55, 0.70, 0.82, 0.62, 0.78], dtype=float)
    d = np.log(v / (1.0 - v))
    w = np.power(d, 1.4)
    order = list(np.argsort(-v, kind='stable'))

    agree = []
    conflict = []

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(a_raw, dtype=float).ravel()
            b = np.asarray(b_raw, dtype=float).ravel()
        except Exception:
            continue
        if a.size != 7 or b.size != 7:
            continue
        diff = a - b
        E = float(np.dot(w, diff))

        winner = None
        for j in order:
            if diff[j] > 0:
                winner = 0
                break
            if diff[j] < 0:
                winner = 1
                break
        if winner is None:
            continue

        try:
            resp = int(r)
        except Exception:
            continue
        cons = 1.0 if resp == winner else 0.0

        sgn = 1.0 if winner == 0 else -1.0
        signed = E * sgn
        if signed > 0 and abs(E) >= 1.5:
            agree.append(cons)
        elif signed < 0 and abs(E) >= 0.5:
            conflict.append(cons)

    if len(agree) == 0 or len(conflict) == 0:
        return 0.0
    return float(np.mean(agree) - np.mean(conflict))

```

**Observed (real) value:** -0.4100 (var=0.0833)
**Predicted under pi_4:** -0.4683 (var=0.0405)
**Predicted under pi_5:** -0.4500 (var=0.0151)

### Experiment 7
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Fixed design validities (communicated in instructions)
    v = np.array([0.70, 0.92, 0.55, 0.84, 0.62, 0.76], dtype=float)
    d = np.log(v / (1.0 - v))

    # Representative parameter settings for the two competing accounts
    w_rd = d ** 0.33          # compressed (near-tally) weights
    w_ae = d ** 1.41          # absolute log-odds^gamma weights
    beta_rd, kappa = 2.1, 0.52
    beta_ae = 2.0

    def _p_rd_A(a, b):
        diff = a - b
        E = float(np.dot(w_rd, diff))
        disc = np.abs(diff) > 0.0
        S = float(w_rd[disc].sum()) if np.any(disc) else 0.0
        if S > 1e-12:
            E = E / (S ** kappa)
        z = -beta_rd * E      # reversed polarity frame s = -1
        z = float(np.clip(z, -500.0, 500.0))
        return 1.0 / (1.0 + np.exp(-z))

    def _p_ae_A(a, b):
        E = float(np.dot(w_ae, a - b))
        z = float(np.clip(beta_ae * E, -500.0, 500.0))
        return 1.0 / (1.0 + np.exp(-z))

    memo = {}

    def _target(a_list, b_list):
        key = (tuple(int(x) for x in a_list), tuple(int(x) for x in b_list))
        if key in memo:
            return memo[key]
        a = np.asarray(key[0], dtype=float)
        b = np.asarray(key[1], dtype=float)
        out = None
        if a.shape[0] == v.shape[0] and b.shape[0] == v.shape[0]:
            pr = _p_rd_A(a, b)
            pa = _p_ae_A(a, b)
            # keep only pairs where the two accounts point in OPPOSITE
            # directions and BOTH are reasonably decisive there
            if (pr - 0.5) * (pa - 0.5) < 0 and abs(pa - 0.5) >= 0.15 and abs(pr - 0.5) >= 0.25:
                out = 0 if pr > 0.5 else 1   # option favoured by RD-WEI
        memo[key] = out
        return out

    hits = 0
    n = 0
    for a_list, b_list, resp in zip(data['option_a_ratings'],
                                    data['option_b_ratings'],
                                    data['response']):
        try:
            t = _target(a_list, b_list)
        except Exception:
            t = None
        if t is None:
            continue
        try:
            r = int(resp)
        except Exception:
            continue
        n += 1
        if r == t:
            hits += 1

    if n == 0:
        return 0.5
    return float(hits) / float(n)

```

**Observed (real) value:** 0.8458 (var=0.0103)
**Predicted under pi_4:** 0.8321 (var=0.0476)
**Predicted under pi_5:** 0.8446 (var=0.0038)

### Experiment 8
**Design**
  A=[1, 0, 1, 1, 0, 1, 0]  B=[1, 0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0, 1]
  A=[0, 1, 0, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    V = np.array([0.66, 0.95, 0.55, 0.80, 0.60, 0.88, 0.72], dtype=float)
    hits = 0
    n = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        try:
            av = np.asarray(list(a), dtype=float)
            bv = np.asarray(list(b), dtype=float)
        except Exception:
            continue
        if av.shape[0] != bv.shape[0] or av.shape[0] == 0:
            continue
        d = av - bv
        nz = np.nonzero(d)[0]
        if nz.size == 0:
            continue
        include = False
        # (1) unanimous / total-dominance pairs: every cue discriminates, same direction
        if nz.size == d.shape[0] and (np.all(d > 0) or np.all(d < 0)):
            include = True
        # (2) single-cue pairs where the lone discriminating cue is highly valid
        elif nz.size == 1 and d.shape[0] == V.shape[0] and V[nz[0]] >= 0.85:
            include = True
        if not include:
            continue
        # target = the option carrying the positive rating(s) on the discriminating cue(s)
        target = 0 if d[nz[0]] > 0 else 1
        try:
            chosen = int(r)
        except Exception:
            continue
        if chosen == target:
            hits += 1
        n += 1
    if n == 0:
        return 0.5
    return float(hits) / float(n)
```

**Observed (real) value:** 0.1383 (var=0.0102)
**Predicted under pi_4:** 0.0958 (var=0.0153)
**Predicted under pi_5:** 0.1317 (var=0.0030)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the eight experiments the two accounts are indistinguishable on everything that only tests POLARITY. Experiments 3-8 all essentially confirm the shared claim that a '1' is read as a warning mark (reversed integration): TTB-consistency 0.74 on TTB-vs-tally conflicts (both predict ~0.73), tally-consistency 0.19 on large-margin conflicts (both ~0.16-0.17), -0.67 and -0.41 on the AEWI-style contrasts (both within noise), 0.85 on the RD-vs-AEWI discrimination metric (both ~0.83-0.84), and 0.14 on the 'choose the positively-rated option' metric (both 0.10-0.13). Neither theory is falsified there and neither gains an edge.\n\nThe two DISCRIMINATIVE experiments split the verdict cleanly, and each theory is falsified on exactly one of its distinguishing claims.\n\nExperiment 2 (sparse, non-conflicting displays, |p-0.5|): observed 0.357, i.e. subjects pick the fewer-marks option ~86% of the time on one- and two-flag displays with an 8-expert panel. pi_4 (RD-WEI) predicts 0.391 - essentially right. pi_5 (RFS-C+D) predicts 0.209 - badly wrong. This is a direct falsification of RFS-C's central NEW claim: the absolute attention budget (sum q rescaled to C~3.1 over 8 experts) hard-caps sparse-trial decisiveness at ~0.75, and the data show no such cap. Panel-size dilution of consultation probability is not happening; people evidently register sparse differences near ceiling.\n\nExperiment 1 (single-cue reversal minus duel 'reject-the-more-credible-flag' rate): observed 0.145. pi_4 predicts 0.326, pi_5 predicts 0.060. Given that Experiment 2 pins the single-cue level near 0.87, the observed contrast implies a duel resolution rate of roughly 0.70-0.75 - i.e. on 1-flag-vs-1-flag conflicts subjects robustly reject the option flagged by the MORE credible expert, even when the two validities are near-equal (0.95 vs 0.90, 0.60 vs 0.55). That is exactly RFS-C+D's severity-ordered duel mode and exactly what RD-WEI's strongly compressed weights (gamma~0.2-0.5) cannot deliver: with w = 1.43 vs 1.30 the duel is a coin flip for RD-WEI. Note that simply raising RD-WEI's gamma or beta does not fix this: with gamma~1 and beta raised, single-cue trials go to ceiling while duels only reach ~0.65-0.75, and the contrast stays near 0.20-0.23, still above the observed 0.145. What the data demand is a rank/severity component that operates specifically when the difference set is sparse, on top of near-ceiling sparse detection with NO consultation bottleneck.\n\nSo: pi_4's compressive, magnitude-only duel resolution is wrong; pi_5's capacity-limited consultation (which is what generates its low sparse-trial ceiling) is wrong. Each holds half of the truth. Since neither is a task-invariant account, the productive move is not to re-fit either one but to replace the more clearly falsified mechanism (pi_5's absolute attention budget, which is contradicted by a 0.15 gap on its OWN designed metric) with a hybrid that keeps its duel/severity insight and drops the bottleneck - which will then be a genuinely stronger competitor to RD-WEI on precisely the axis where RD-WEI fails.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_5 with a hybrid reversed-polarity account that removes the falsified consultation/capacity bottleneck but keeps (and generalises) the sparsity-dependent severity ordering. Sketch - 'R-SED: Reversed Evidence with Sparsity-Gated Severity Dominance':\n\n(1) POLARITY (retain, it is the one robustly confirmed claim): a '1' is a warning mark; the option with the smaller weighted mark load is preferred. Allow a small minority of forward-frame subjects (p_reverse ~0.85-0.95) to reproduce the observed between-subject variance.\n\n(2) NO CONSULTATION BOTTLENECK. Every displayed rating is registered; panel size does not dilute the reading of a lone difference. Sparse non-conflicting displays (1-2 flags on one side) must land at ~0.85-0.90 fewer-marks choices for a 6- AND an 8-expert panel (Exp 2 observed |p-0.5| = 0.357; Exp 1 dominance pairs near ceiling). Achieve this by comparing the margin RELATIVE to the evidence in play (divisive normalisation, as in RD-WEI) or by a fixed per-difference detection term - but NOT by any q_i that shrinks with panel size.\n\n(3) SPARSITY-GATED SEVERITY DOMINANCE (the new theoretical content). The effective weight of the single most valid DISCRIMINATING cue is amplified by a factor that decays with the size of the difference set: w_eff(top) = w(top) * (1 + phi * exp(-lam * (n_disc - 1))). When only one or two experts disagree, the comparison is a duel between named experts and the more credible flag dominates almost lexicographically (predicting ~0.72-0.80 rejection of the option flagged by the higher-validity expert even for near-equal pairs such as 0.95 vs 0.90 and 0.60 vs 0.55, i.e. gap-INSENSITIVE and level-insensitive); when four or more experts disagree, identities are lost and the decision reverts to a near-tally weighted count (reproducing Exp 3/4/6/8 and Exp 1's 2-vs-2 top-heavy cell). Base weights should be mildly compressed (gamma ~0.3-0.7) so that many-cue conflicts stay count-driven.\n\n(4) Quantitative targets the new model must hit simultaneously: Exp 1 metric (single-cue minus duel) ~= 0.10-0.18 with single-cue ~0.85-0.90 and duel ~0.70-0.78; Exp 2 metric ~= 0.34-0.39; Exp 3 ~0.74; Exp 4 ~0.19; Exp 5 ~ -0.67; Exp 6 ~ -0.41; Exp 7 ~0.85; Exp 8 ~0.14. Parameter ranges should be chosen so that the whole range keeps sparse-trial decisiveness above 0.82 and duel dominance above 0.68 - these are the two signatures on which it must beat RD-WEI, whose compressed magnitude-only duels are stuck near chance. Future experiments should then target the boundary: how duel resolution changes as n_disc goes 2 -> 3 -> 4 at matched weighted margin, where R-SED predicts a sharp drop toward count-driven chance and RD-WEI predicts smooth share-graded behaviour."
}
```

## Usage

```json
{
  "input_tokens": 27938,
  "output_tokens": 5617
}
```
