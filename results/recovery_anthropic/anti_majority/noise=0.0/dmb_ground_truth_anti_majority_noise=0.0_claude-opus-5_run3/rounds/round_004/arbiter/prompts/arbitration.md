# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


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

## THEORY 2 — pi_6
**R-SED: Reversed Evidence with Sparsity-Gated Severity Dominance.**

Four claims.

(1) POLARITY (retained; the one robustly confirmed claim). In these instruction-only, feedback-free product-rating environments a binary '1' is read as a WARNING MARK / flag against an option rather than as an endorsement. The decision variable is therefore the *negated* weighted mark load: the option carrying the SMALLER validity-weighted number of 1s is preferred. A small minority of subjects keep the forward frame (subject-level latent s in {-1,+1}, P(s=-1)=p_reverse ~0.94-1.0), which is what generates the observed between-subject spread on dominance/unanimity pairs.

(2) NO CONSULTATION BOTTLENECK. Every displayed rating is registered; there is no per-expert reading probability and no capacity that shrinks with panel size. A lone flag on a 6-expert panel is read exactly as reliably as on a 4-expert panel. This kills the falsified panel-size dilution of RFS-C: sparse non-conflicting displays land at ~0.86 fewer-marks choices regardless of n_features.

(3) EVIDENCE IS RELATIVE, WITH MILDLY COMPRESSED BASE WEIGHTS. Stated validities become log-odds d_i = log(v_i/(1-v_i)), compressed to w_i = d_i^gamma with gamma ~0.3 so that many-cue conflicts stay essentially count-driven (no single expert can lexicographically beat four others). The comparison is evaluated in *relative* units: the signed weighted margin is divided by the total weight of the cues actually in play raised to kappa ~0.7 (Weber-like divisive normalisation). Because kappa is close to (but below) 1, decisiveness saturates: 1-cue, 2-cue-same-side and 7-cue unanimous displays all sit at roughly the same ~0.86 reversed-choice level rather than becoming deterministic as the panel grows, while lopsided many-cue conflicts remain only slightly less decisive (~0.82).

(4) NEW THEORETICAL CONTENT — SPARSITY-GATED SEVERITY DOMINANCE. When only a couple of experts disagree, the comparison stops being a tally and becomes a *duel between named experts*: the reader asks which flag is the more damning, and the single most valid DISCRIMINATING cue acquires an amplified effective weight,

    w_eff(top) = w(top) * (1 + phi * exp(-lam * (n_disc - 2))),  applied only when n_disc >= 2,

so the more credible flag nearly dominates the comparison. Crucially the amplification enters BOTH the margin and the normaliser, so duels resolve at a roughly constant ~0.72-0.78 rejection of the option flagged by the more credible expert, essentially INDEPENDENT of the validity gap and of the validity level (0.95-vs-0.90 behaves like 0.60-vs-0.55). As the difference set grows the identities of the flagging experts are lost, the boost decays geometrically (already ~1.07 at n_disc = 4, ~1.01 at n_disc = 6) and behaviour reverts to a near-tally weighted count of marks, which is why one flag from the best expert loses to four flags from weak experts and why top-heavy near-even splits (2-vs-3, 2-vs-2) sit near chance.

The family nests the classics: s=+1, gamma->0, kappa=0, phi=0 = Tallying; s=+1, gamma=1, kappa=0, phi=0 = Franklin's rule; phi->inf with lam->0 = (reversed) Take-The-Best; kappa=1 = pure share-of-evidence. Its sharp, falsifiable signature relative to RD-WEI is the n_disc ladder at matched weighted margin: R-SED predicts a steep drop in duel resolution as n_disc goes 2 -> 3 -> 4 (severity gate closing), whereas RD-WEI predicts smooth share-graded behaviour and near-chance duels throughout.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # R-SED: Reversed Evidence with Sparsity-Gated Severity Dominance.
    #   d_i = log(v_i/(1-v_i));  w_i = d_i ** gamma          (mild compression)
    #   if n_disc >= 2: w_eff[top discriminating cue] *= 1 + phi*exp(-lam*(n_disc-2))
    #   E  = sum_i w_eff_i (a_i - b_i)      (positive = A carries more marks)
    #   S  = sum_{i in disc} w_eff_i
    #   E' = E / S**kappa                    (relative / divisive normalisation)
    #   s  = -1 with prob p_reverse (mark-reading frame), else +1
    #   P(A) = sigmoid(s * beta * E'), then lapse epsilon toward uniform.
    # No consultation bottleneck: every rating is registered. History unused
    # (there is no feedback in this domain).
    import numpy as np

    # ---------------- unpack stimulus ----------------
    stim = None
    if isinstance(state, dict):
        a_raw = state.get("option_a_ratings", None)
        b_raw = state.get("option_b_ratings", None)
        if a_raw is not None and b_raw is not None:
            try:
                a_v = np.asarray(list(a_raw), dtype=float).ravel()
                b_v = np.asarray(list(b_raw), dtype=float).ravel()
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

    gamma = float(parameters["gamma"])
    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    phi = float(parameters["phi"])
    lam = float(parameters["lam"])
    eps = float(parameters["epsilon"])
    p_rev = float(parameters["p_reverse"])
    u = float(parameters["polarity_u"])

    # ---------------- compressed diagnosticity weights ----------------
    d = np.maximum(np.log(v / (1.0 - v)), 1e-12)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        w = np.power(d, gamma)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 1.0)

    diff = stim[0] - stim[1]
    disc = np.nonzero(np.abs(diff) > 0.0)[0]
    n_disc = int(disc.size)
    if n_disc == 0:
        return np.ones(2) / 2.0

    # ---------------- sparsity-gated severity amplification ----------------
    w_eff = w.astype(float).copy()
    if n_disc >= 2:
        # most valid DISCRIMINATING cue
        top = int(disc[int(np.argmax(v[disc]))])
        expo = -lam * float(n_disc - 2)
        expo = float(np.clip(expo, -50.0, 50.0))
        amp = 1.0 + phi * float(np.exp(expo))
        if not np.isfinite(amp) or amp < 1.0:
            amp = 1.0
        w_eff[top] = w_eff[top] * amp

    # ---------------- relative (normalised) margin ----------------
    E = float(np.dot(w_eff, diff))          # >0 : A carries more weighted marks
    S = float(np.sum(w_eff[disc]))
    if not np.isfinite(E):
        E = 0.0
    if np.isfinite(S) and S > 1e-12:
        denom = S ** kappa
        if np.isfinite(denom) and denom > 1e-12:
            E = E / denom
    if not np.isfinite(E):
        E = 0.0

    # ---------------- subject-level polarity frame ----------------
    s = -1.0 if u < p_rev else 1.0

    z = np.array([s * beta * E, 0.0], dtype=float)
    z = np.clip(z, -500.0, 500.0)
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)

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
**Validities (n_features=8):** [0.75, 0.93, 0.55, 0.68, 0.88, 0.6, 0.75, 0.68]

**Trial pairs (n=18):**
  trial 1: A=[0, 1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 1, 0, 0, 1]
  trial 2: A=[0, 0, 1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 0, 0, 1]
  trial 3: A=[1, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0]
  trial 4: A=[0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 1, 0]
  trial 6: A=[0, 0, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0]
  trial 7: A=[0, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  trial 8: A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0, 0, 0]
  trial 9: A=[1, 1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  trial 10: A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1, 0, 0]
  trial 11: A=[1, 1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 1, 1]
  trial 12: A=[0, 0, 0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0, 0, 0]
  trial 13: A=[1, 1, 0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0, 0]
  trial 14: A=[1, 0, 0, 0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 1, 0, 0]
  trial 15: A=[1, 0, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 1, 0, 1, 1, 1]
  trial 17: A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate RD-WEI (compressed near-tally weights gamma~0.33, reversed polarity s=-1, divisive normalisation with kappa~0.52, beta~2.1, small lapse) from R-SED (same reversed-polarity family but kappa~0.70, beta~3.6, large lapse ~0.20, and crucially a SPARSITY-GATED SEVERITY BOOST that multiplies the weight of the single most valid DISCRIMINATING cue by 1+phi*exp(-lam*(n_disc-2)) whenever n_disc>=2). Both are reversed-frame accounts, so direction cannot separate them; the design targets (i) the level of 1-flag-vs-1-flag DUEL resolution, (ii) the n_disc LADDER at matched weighted margin, and (iii) magnitude anchors that pin beta and epsilon so no global-noise reparameterisation can mimic the shape.

ENVIRONMENT: eight experts in scrambled display order [0.75, 0.93, 0.55, 0.68, 0.88, 0.60, 0.75, 0.68] (idx0..idx7), so no 'first-listed expert' rule can imitate either model. Two exact DUPLICATE pairs are built in (idx0/idx6 both 0.75; idx3/idx7 both 0.68) so that cancelling flag pairs add EXACTLY zero to the signed weighted margin while inflating the discriminating mass S. Log-odds d = (1.099, 2.587, 0.201, 0.754, 1.992, 0.406, 1.099, 0.754). At RD-WEI's gamma=0.33 the weights are w = (1.032, 1.368, 0.589, 0.911, 1.255, 0.742, 1.032, 0.911) (only ~2.3:1 spread); R-SED's gamma=0.34 gives virtually the same base profile, so ALL differences below come from the severity gate, kappa and the beta/epsilon regime, not from weight ordering.

NINE TYPES, each mirrored across A/B so no side bias can imitate either model (18 unique pairs, K=5, 90 trials). Predicted p(A) at representative parameters (RD-WEI: gamma=.33, kappa=.52, beta=2.1, eps=.06, s=-1 | R-SED: gamma=.34, kappa=.70, phi=.5, lam=1.0, beta=3.6, eps=.20, s=-1):
(T1, pairs 1-2) DUEL HIGH, small gap: 0.93 flags A, 0.88 flags B (n_disc=2). RD-WEI .466 (compressed weights 1.368 vs 1.255 differ by almost nothing, so after normalisation this is essentially a coin flip) | R-SED .278 (the severity gate amplifies the 0.93 flag by 1.5 and condemns A). PRIMARY DISCRIMINATOR.
(T2, pairs 3-4) DUEL MID, small gap: 0.75 flags A, 0.68 flags B. RD-WEI .457 | R-SED .281.
(T3, pairs 5-6) DUEL LOW, small gap: 0.60 flags A, 0.55 flags B. RD-WEI .435 | R-SED .270. T1-T3 test R-SED's headline claim that duel resolution is roughly constant (~.27-.28) and independent of validity LEVEL, while RD-WEI must sit near chance (.43-.47) throughout. Parameter sweeps over the stated ranges keep the two sets disjoint (RD-WEI .39-.48 vs R-SED .20-.36).
(T4, pairs 7-8) DUEL MAX GAP: 0.93 flags A, 0.55 flags B. RD-WEI .265 | R-SED .150. Contrasted with T1-T3 this gives the GAP-gradedness test: RD-WEI's deviation from chance grows ~7-fold from small-gap to max-gap duels, R-SED's only ~1.6-fold.
(T5, pairs 9-10) MATCHED-MARGIN LADDER, n_disc=4: T1's duel plus an exactly cancelling 0.75/0.75 flag pair (idx0 on A, idx6 on B). Signed margin identical to T1; only S grows and R-SED's boost decays to 1.068. RD-WEI .476 | R-SED .450.
(T6, pairs 11-12) MATCHED-MARGIN LADDER, n_disc=6: T1's duel plus cancelling 0.75/0.75 and 0.68/0.68 pairs. RD-WEI .479 | R-SED .481. THE LADDER T1->T5->T6 is the decisive shape test: at a CONSTANT weighted margin RD-WEI predicts an essentially FLAT profile (.466, .476, .479; deviation ratio ~1.4 across the ladder even at extreme parameters) whereas R-SED predicts a STEEP collapse (.278, .450, .481; deviation ratio ~12) as the severity gate closes. No reparameterisation of either model reproduces the other's ladder.
(T7, pairs 13-14) SINGLE CUE 0.93 (n_disc=1, two tied 1s present so the display is non-degenerate). RD-WEI .095 | R-SED .115. Because R-SED's boost is switched OFF at n_disc=1 and ON at n_disc=2, the T7->T1 step is diagnostic: RD-WEI predicts a huge jump toward chance (+.37) when a 0.88 counter-flag is added, R-SED only +.16.
(T8, pairs 15-16) LOPSIDED 5-vs-1: five weak/mid flags (0.75,0.68,0.60,0.75,0.68) on A against a single 0.93 flag on B (n_disc=6, boost ~1.009, i.e. off). RD-WEI .089 | R-SED .228 - a pure magnitude test isolating the kappa/beta/epsilon regime once the severity gate is inert.
(T9, pairs 17-18) UNANIMOUS 8-0 DOMINANCE. RD-WEI .033 | R-SED .101. LAPSE ANCHOR: it pins epsilon for both models, so R-SED's near-chance ladder tail and RD-WEI's near-chance duels cannot be traded off against global noise.

DIAGNOSTIC SUMMARY of p(A) across T1..T9: RD-WEI = (.466, .457, .435, .265, .476, .479, .095, .089, .033); R-SED = (.278, .281, .270, .150, .450, .481, .115, .228, .101). The vectors agree in sign everywhere (both reversed-polarity), but differ in SHAPE in three independent, non-reparameterisable ways: (i) duels near chance vs strongly resolved; (ii) a flat vs steeply decaying n_disc ladder at matched margin; (iii) different sparse/dominance magnitude anchors that block a noise-only mimicry.

**Computed schedule:** 18 unique pairs × 5 reps = 90 trials per subject.



### METRIC
Rationale:
The two theories agree in polarity everywhere, so the discriminating signal must be a SHAPE contrast between trial classes. RD-WEI (divisive normalisation with small kappa~0.52) says a 1-flag-vs-1-flag duel between two cues of similar validity is essentially a coin flip (p(A)~.44-.47, deviation from chance ~.03-.07), because the compressed near-tally weights of the two duelling cues barely differ and normalisation removes any absolute-magnitude advantage. R-SED's sparsity-gated severity boost instead turns such duels into decisive 'which flag is more damning' judgements (p(A)~.27-.28, deviation ~.22) essentially independent of the validity gap. Conversely, on lopsided/dominance displays (5-vs-1 and 8-vs-0 marks) RD-WEI is near-ceiling (~.91-.97 for the fewer-marks option) while R-SED's larger kappa (~.70) plus large lapse (~.20) caps it around .77-.90.

The metric therefore computes: (deviation-from-chance on high-mark-difference dominance trials) minus (signed deviation on SMALL-VALIDITY-GAP duel trials, sign-aligned to the subject's own dominance-trial polarity). Predicted values: RD-WEI ~0.40, R-SED ~0.11.

Sign-aligning the duel term to the polarity revealed by the dominance trials makes the metric invariant to the rare forward-frame subject (s=+1), preventing outlier subjects from inflating between-subject variance in either simulation. Each component is estimated from 20-30 trials per subject, so per-subject SD is ~0.10 while the between-theory gap is ~0.29 -- comfortably separable at N=25. Restricting the duel class to matched-mark, n_disc=2 pairs with validity gap <= 0.20 excludes the max-gap duel (0.93 vs 0.55), where RD-WEI also deviates strongly from chance, thereby maximising the contrast.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    V = np.array([0.75, 0.93, 0.55, 0.68, 0.88, 0.60, 0.75, 0.68], dtype=float)
    duel = []
    dom = []
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()
            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()
            resp = int(row['response'])
        except Exception:
            continue
        if a.size == 0 or a.size != b.size:
            continue
        if a.size == V.size:
            v = V
        else:
            v = np.linspace(0.90, 0.55, a.size)
        d = a - b
        disc = np.nonzero(np.abs(d) > 0.0)[0]
        sa = float(a.sum())
        sb = float(b.sum())
        if disc.size == 2 and sa == sb:
            vs = v[disc]
            if abs(float(vs[0]) - float(vs[1])) <= 0.20:
                top = int(disc[int(np.argmax(vs))])
                # reference = the option NOT flagged by the more valid duelling cue
                ref = 1 if d[top] > 0 else 0
                duel.append(1.0 if resp == ref else 0.0)
        elif abs(sa - sb) >= 4.0:
            # reference = option carrying fewer marks
            ref = 1 if sa > sb else 0
            dom.append(1.0 if resp == ref else 0.0)
    if len(duel) == 0 or len(dom) == 0:
        return float('nan')
    p_dom = float(np.mean(dom))
    p_duel = float(np.mean(duel))
    s = 1.0 if p_dom >= 0.5 else -1.0
    return float(s * (p_dom - 0.5) - s * (p_duel - 0.5))

### RESULTS
- Predicted under pi_4 (simulated): 0.3750 (var=0.0108)
- Predicted under pi_6 (simulated): 0.1353 (var=0.0085)
- Observed on real data: 0.1693 (var=0.0185)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=7):** [0.72, 0.95, 0.55, 0.85, 0.6, 0.72, 0.6]

**Trial pairs (n=18):**
  trial 1: A=[0, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 1]
  trial 2: A=[0, 0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 0, 0, 1]
  trial 3: A=[0, 0, 1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0]
  trial 4: A=[1, 0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1, 0]
  trial 6: A=[0, 0, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 1, 0]
  trial 7: A=[1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0]
  trial 8: A=[1, 0, 1, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 1, 0]
  trial 10: A=[0, 0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0]
  trial 11: A=[1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1, 1]
  trial 12: A=[0, 0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0, 0]
  trial 13: A=[1, 1, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 1]
  trial 14: A=[1, 0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0, 1]
  trial 15: A=[1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 16: A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1, 1]
  trial 17: A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate R-SED (advocated: reversed polarity, mildly compressed weights gamma~0.34, kappa~0.70, beta~3.6, lapse~0.20, plus the SPARSITY-GATED SEVERITY BOOST that multiplies the weight of the single most valid DISCRIMINATING cue by 1+phi*exp(-lam*(n_disc-2)) when n_disc>=2) from RD-WEI (reversed polarity, gamma~0.33, kappa~0.52, beta~2.1, small lapse, NO severity gate). Both are reversed-frame accounts, so overall direction cannot separate them; the design attacks three orthogonal SHAPE signatures: (i) the LEVEL of 1-flag-vs-1-flag duel resolution and its (in)dependence of the validity GAP, (ii) the n_disc ladder at an EXACTLY matched signed weighted margin, and (iii) sparse/dominance magnitude anchors that pin beta and epsilon so no noise-only reparameterisation can mimic the pattern.

ENVIRONMENT: seven experts in scrambled display order [0.72, 0.95, 0.55, 0.85, 0.60, 0.72, 0.60] (idx0..idx6), so no 'use the first-listed expert' rule can imitate either model. Two exact DUPLICATE pairs are built in (idx0/idx5 = 0.72; idx4/idx6 = 0.60) so that cancelling flag pairs add EXACTLY zero to the signed weighted margin while inflating the discriminating mass S. Log-odds d = (0.944, 2.944, 0.201, 1.735, 0.406, 0.944, 0.406). At the shared compression exponent the base weights are w ~ (0.98, 1.43, 0.59, 1.20, 0.74, 0.98, 0.74) - only a ~2.4:1 spread - so ALL divergences below come from the severity gate and the kappa/beta/epsilon regime, never from weight ordering.

NINE TYPES, each mirrored across A/B so no side bias can imitate either model (18 unique pairs, K=5, 90 trials). Predicted p(A) at representative parameters (R-SED: gamma=.34, kappa=.70, phi=.5, lam=1.0, beta=3.6, eps=.20, s=-1 | RD-WEI: gamma=.33, kappa=.52, beta=2.1, eps=.06, s=-1):
(T1, pairs 1-2) DUEL HIGH, small gap: 0.95 flags A, 0.85 flags B (n_disc=2). R-SED .25 (the severity gate multiplies the 0.95 flag by 1.5 and condemns A) | RD-WEI .44 (compressed weights 1.43 vs 1.20 differ by almost nothing, so after normalisation this is nearly a coin flip). PRIMARY DISCRIMINATOR.
(T2, pairs 3-4) DUEL MID, small gap: 0.85 flags A, 0.72 flags B. R-SED .25 | RD-WEI .43.
(T3, pairs 5-6) DUEL LOW, small gap: 0.60 flags A, 0.55 flags B. R-SED .27 | RD-WEI .44. T1-T3 test R-SED's headline claim that duel resolution is roughly CONSTANT and independent of validity LEVEL, while RD-WEI must hover near chance throughout.
(T4, pairs 7-8) DUEL MAX GAP: 0.95 flags A, 0.55 flags B. Both models ~.245 - they AGREE here. The T1/T4 contrast is the gap test: RD-WEI's deviation from chance must grow ~4-fold from small-gap to max-gap duels, while R-SED's stays flat (.25 -> .245). Neither model can reproduce the other's duel profile at any admissible parameters.
(T5, pairs 9-10) MATCHED-MARGIN LADDER, n_disc=4: T1's duel plus an exactly cancelling 0.72/0.72 flag pair (idx0 on A, idx5 on B). Signed margin identical to T1; only S grows and R-SED's boost decays to 1.068. R-SED .42 | RD-WEI .45.
(T6, pairs 11-12) MATCHED-MARGIN LADDER, n_disc=6: T1's duel plus cancelling 0.72/0.72 and 0.60/0.60 pairs. R-SED .45 | RD-WEI .46. THE LADDER T1->T5->T6 is the decisive shape test: at a CONSTANT weighted margin R-SED predicts a steep collapse toward chance (.25, .42, .45; deviation shrinks ~5-fold as the severity gate closes) whereas RD-WEI predicts an essentially FLAT profile (.44, .45, .46).
(T7, pairs 13-14) SINGLE CUE 0.95, n_disc=1 (a tied 1 at idx0 and idx6 keeps the display non-degenerate). R-SED .11 | RD-WEI .10 - the models nearly coincide. Because R-SED's boost is OFF at n_disc=1 and ON at n_disc=2, the T7->T1 STEP is diagnostic: RD-WEI predicts a large jump toward chance when a 0.85 counter-flag is added (+.34), R-SED only +.14.
(T8, pairs 15-16) LOPSIDED 5-vs-1: five weak/mid flags (0.72,0.72,0.60,0.60,0.55) on A against a single 0.95 flag on B (n_disc=6, boost ~1.01, i.e. inert). R-SED .245 | RD-WEI .13 - a pure magnitude test isolating the kappa/beta/epsilon regime once the severity gate is switched off, and a check that one flag from the best expert loses to four/five weak flags.
(T9, pairs 17-18) UNANIMOUS 7-0 DOMINANCE. R-SED .10 | RD-WEI .033. LAPSE ANCHOR: it pins epsilon for both models, so R-SED's near-chance ladder tail and RD-WEI's near-chance duels cannot be traded off against global noise.

DIAGNOSTIC SUMMARY of p(A) across T1..T9: R-SED = (.25, .25, .27, .245, .42, .45, .11, .245, .10); RD-WEI = (.44, .43, .44, .245, .45, .46, .10, .13, .033). The vectors agree in sign everywhere (both reversed-polarity) but differ in SHAPE in three independent, non-reparameterisable ways: (i) small-gap duels strongly resolved vs near chance, with the two accounts converging exactly at the max-gap duel; (ii) a steeply decaying vs flat n_disc ladder at an exactly matched signed margin; (iii) opposite orderings of the sparse (T7), lopsided (T8) and dominance (T9) anchors, which fix beta and epsilon and block any noise-only mimicry.

**Computed schedule:** 18 unique pairs × 5 reps = 90 trials per subject.



### METRIC
Rationale:
The two accounts are both reversed-polarity, so any overall direction measure is useless. Their sharpest non-reparameterisable divergence is the LEVEL at which a 1-flag-vs-1-flag duel with a SMALL validity gap is resolved. R-SED's sparsity-gated severity boost amplifies the single most valid discriminating cue when n_disc=2, so the option flagged by the more credible expert is rejected at roughly 0.72-0.78 regardless of how small the validity gap is (trials 1-6: .95 vs .85, .85 vs .72, .60 vs .55 all resolve alike, ~0.75). RD-WEI has no such gate: after its compressive gamma transform the two competing weights are nearly equal (e.g. 1.43 vs 1.20), and divisive normalisation cannot rescue a near-zero share, so it must hover near chance (~0.56). Predicted gap ~0.19 in a bounded rate estimated from 30 trials per subject (binomial sd ~0.08), which is tight enough for N=25.

Two design choices keep per-subject variance small. (1) Max-gap duels (trials 7-8, .95 vs .55), where the two theories AGREE (~0.755 both), are excluded by the |v_hi - v_lo| <= 0.20 filter, so only the discriminating small-gap duels contribute. (2) The dominant nuisance source of between-subject spread in both models is the latent polarity frame s (a few percent of subjects run the forward frame and mirror every prediction). I read each subject's frame off the unanimous 7-0 dominance trials \u2014 which BOTH theories predict at >0.90/0.96 accuracy, so frame recovery from 10 trials is essentially error-free \u2014 and align the duel rate to that frame. This removes the bimodality without abs()-style bias, leaving mostly binomial noise. The metric is identically defined on the pooled frame (mean over subject-level rates) and on any single-subject slice.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V = np.array([0.72, 0.95, 0.55, 0.85, 0.60, 0.72, 0.60], dtype=float)

    def _subj(df):
        dom = []   # 1 if subject picked the ALL-ZERO option on 7-0 dominance trials
        duel = []  # 1 if subject picked the option NOT flagged by the more valid expert
        for a_raw, b_raw, r in zip(df['option_a_ratings'], df['option_b_ratings'], df['response']):
            try:
                a = np.asarray(list(a_raw), dtype=float).ravel()
                b = np.asarray(list(b_raw), dtype=float).ravel()
            except Exception:
                continue
            if a.size != b.size or a.size != V.size:
                continue
            try:
                resp = int(r)
            except Exception:
                continue
            n = a.size
            diff = a - b
            disc = np.nonzero(diff != 0.0)[0]

            # --- dominance / unanimity trials: frame anchor ---
            if (a.sum() == n and b.sum() == 0.0) or (b.sum() == n and a.sum() == 0.0):
                zero_side = 0 if a.sum() == 0.0 else 1
                dom.append(1.0 if resp == zero_side else 0.0)
                continue

            # --- small-gap 1-vs-1 duels (exactly two discriminating cues, one per side) ---
            if disc.size == 2:
                i, j = int(disc[0]), int(disc[1])
                if diff[i] * diff[j] < 0.0:
                    vi, vj = V[i], V[j]
                    if vi != vj and abs(vi - vj) <= 0.20:
                        hi = i if vi > vj else j
                        hi_side = 0 if diff[hi] > 0.0 else 1  # option flagged by more valid expert
                        duel.append(1.0 if resp == (1 - hi_side) else 0.0)

        if len(duel) == 0:
            return np.nan
        # subject-level polarity frame, read off the unanimous dominance pairs
        frame = 1.0
        if len(dom) > 0 and float(np.mean(dom)) < 0.5:
            frame = -1.0
        rate = float(np.mean(duel))
        return rate if frame > 0.0 else 1.0 - rate

    vals = []
    for _, g in data.groupby('subject_id', sort=False):
        v = _subj(g)
        if v == v:
            vals.append(v)
    if len(vals) == 0:
        return 0.5
    return float(np.mean(vals))

### RESULTS
- Predicted under pi_4 (simulated): 0.5787 (var=0.0068)
- Predicted under pi_6 (simulated): 0.7347 (var=0.0080)
- Observed on real data: 0.7160 (var=0.0218)

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
**Predicted under pi_6:** 0.7067 (var=0.0082)

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
**Predicted under pi_6:** 0.1750 (var=0.0123)

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
**Predicted under pi_6:** -0.5700 (var=0.1548)

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
**Predicted under pi_6:** -0.4975 (var=0.0146)

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
**Predicted under pi_6:** 0.8450 (var=0.0205)

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
**Predicted under pi_6:** 0.0958 (var=0.0044)

### Experiment 9
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]

**Metric**
```python
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
```

**Observed (real) value:** 0.1453 (var=0.0133)
**Predicted under pi_4:** 0.3260 (var=0.0118)
**Predicted under pi_6:** 0.1467 (var=0.0117)

### Experiment 10
**Design**
  A=[1, 0, 0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 1, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 1]  B=[0, 0, 1, 0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 1, 0, 1]  B=[0, 0, 0, 0, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1, 0, 0]

**Metric**
```python
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
```

**Observed (real) value:** 0.3567 (var=0.0085)
**Predicted under pi_4:** 0.3908 (var=0.0031)
**Predicted under pi_6:** 0.3625 (var=0.0018)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Pooling all ten experiments, the two accounts agree on the one robust qualitative fact of this domain \\u2014 the reversed polarity frame (subjects prefer the option carrying fewer validity-weighted '1's; Exps 4, 5, 6, 8, 10 all confirm this and both models fit them within noise). They differ on the SHAPE of sparse-conflict behaviour, and there the data are decisive against pi_4 (RD-WEI). Exp 1 (dominance-minus-duel contrast): observed 0.169, pi_6 0.135, pi_4 0.375 \\u2014 pi_4 is off by ~0.21, roughly 2 between-subject SDs. Exp 2 (small-gap duel resolution level): observed 0.716, pi_6 0.735, pi_4 0.579 \\u2014 again pi_6 nails it, pi_4 misses badly. Exp 9 (single-cue minus small-gap-duel): observed 0.145, pi_6 0.147, pi_4 0.326 \\u2014 same story. The core theoretical claim of RD-WEI, that after compressive weighting plus divisive normalisation a 1-flag-vs-1-flag duel between similar-validity experts must sit near chance, is falsified three independent times: real subjects reject the option flagged by the more credible expert at ~72-75%, essentially independent of the validity gap and of the validity level. That is exactly R-SED's sparsity-gated severity claim. pi_4's only advantages are marginal and on metrics where both models are essentially indistinguishable (Exp 3: .743 vs .707 vs observed .737; Exp 5: -.67 vs -.57 vs observed -.673; Exp 6: -.468 vs -.498 vs observed -.410; Exp 10 slightly favours pi_6). So pi_6 is currently the task-invariant account: it is within noise on all ten metrics, whereas pi_4 fails systematically on the whole family of sparse-conflict/duel metrics. Since RD-WEI's distinguishing content (kappa-normalised near-chance duels) is the very thing repeatedly refuted, re-fitting its parameters would only push it toward mimicking R-SED without any new theoretical content; it should be replaced by a genuinely different mechanism that can also produce gap-independent, sparsity-gated duel resolution, so that the next round is a real test rather than a foregone conclusion.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace pi_4 (RD-WEI) with a DISCRETE TWO-ROUTE / FOCUS-SAMPLING account of the same reversed-polarity data \\u2014 call it R-FSA (Reversed Focus-Sampling Arbitration). Core claims: (1) retain the reversed polarity frame (a '1' is read as a warning mark; subject-level latent s in {-1,+1} with P(s=-1)~0.9-1.0), since this is confirmed everywhere. (2) NO continuous evidence accumulator with divisive normalisation. Instead, on each trial the subject stochastically ARBITRATES between two discrete routes: (a) a ONE-REASON route \\u2014 attention lands on a single discriminating cue, sampled by a Luce rule over the discriminating cues with sampling weights proportional to v_i^tau (tau moderate, so with two cues the more valid one wins the focus with a probability that is bounded and only weakly gap-dependent, ~0.7-0.8 for any gap, matching Exps 1/2/9), and the option flagged by that cue is rejected; (b) a COUNTING route \\u2014 the subject rejects the option with the larger (near-unweighted) mark count, which produces the anti-tally behaviour of Exps 4, 5, 6, 8, 10. Route probability is a decreasing function of the size of the difference set: P(one-reason) = 1/(1 + c*(n_disc-1)^delta), so sparse displays are duel-like and dense displays are count-like. (3) Choice is then near-deterministic given the sampled route, with only a small lapse epsilon; there is no beta/temperature acting on a graded margin. This differs from R-SED mechanistically: R-SED produces graded, sigmoid-of-normalised-margin behaviour where the duel rate depends on beta and on the amplified weighted margin, whereas R-FSA caps the duel rate at the focus-sampling probability (independent of beta and of validity level), predicts DISCRETE within-subject response distributions on repeated identical pairs (a mixture of two deterministic routes, i.e. over-dispersion relative to binomial), and predicts that on many-cue near-even splits behaviour is set purely by the raw mark COUNT (validity-blind), whereas R-SED still tracks the compressed weighted margin. Good discriminating designs for the next round: (i) repeat-level over-dispersion on a fixed sparse pair; (ii) many-cue trials where the raw count and the validity-weighted count point in opposite directions; (iii) duels at extreme validity levels (0.99 vs 0.97 vs 0.56 vs 0.54) to test whether duel resolution is truly level-invariant."
}
```

## Usage

```json
{
  "input_tokens": 31155,
  "output_tokens": 3520
}
```
