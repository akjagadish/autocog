# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_6" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_6" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_6
Dampened Confidence-Distrust Integration (DCD). Every expert endorsement is treated as a liability whose magnitude grows with the endorser's claimed validity: a highly valid expert's praise is a strong claim, and strong claims attract the most suspicion. The per-endorsement distrust charge is d_j = c0 + c1*(2*v_j - 1)^kappa, where the (2v-1) rescaling DAMPENS the charge near chance validity (a 55% expert's endorsement costs barely more than a coin flip's, i.e. only the flat baseline c0) while remaining steeply increasing toward confident experts. Two additions repair the incumbents' failures: (i) a large flat charge c0 preserves pure endorsement-count avoidance, so when the low-validity option carries MORE endorsements the count signal can overcome validity-distrust (the count-conflict cells where VPS is wrong); (ii) a small residual validity-weighted trust term alpha * sum_j logit(v_j)*(a_j - b_j) captures residual endorsement-following that pulls matched-count choices back toward and slightly past chance in the direction of the higher-validity endorser (softening VPS's extremes). The decision variable is D = (P(B) - P(A)) + alpha * sum_j logit(v_j)*(a_j - b_j), where P(X) = sum_j x_j*d_j is the accumulated suspicion for option X; the subject chooses via P(A) = sigmoid(beta*D) mixed with a uniform lapse epsilon. History is ignored because validities are communicated in the instructions, so distrust weights are fixed for the whole block. DCD nests VPS as a limiting case (alpha -> 0, rescaling absorbed) and SDI as another (c1 -> 0), while filling the empirically human region between them.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Dampened Confidence-Distrust Integration (DCD).
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Per-endorser liability (dampened confidence-distrust charge):
    #   d_j = c0 + c1 * (2*v_j - 1)^kappa
    #     c0            -> flat suspicion floor: every endorsement is a
    #                      defect, and a near-chance expert's praise
    #                      costs essentially only this floor (the
    #                      (2v-1) rescaling dampens distrust near
    #                      v = 0.5 relative to a raw v^kappa scaling)
    #     c1*(2v-1)^kap -> steeply INCREASING in claimed validity:
    #                      a confident expert's endorsement is a big
    #                      suspected defect
    # Accumulated suspicion:  P(X) = sum_j x_j * d_j
    # Residual trust term:    E = sum_j logit(v_j) * (a_j - b_j)
    # Decision variable:      D = (P(B) - P(A)) + alpha * E
    #   (positive favors A: B carries more accumulated suspicion,
    #    plus a small validity-weighted endorsement edge for A)
    # P(A) = sigmoid(beta * D), mixed with a uniform lapse epsilon.
    # History is ignored: validities are in the instructions, so
    # the distrust weights are fixed for the whole block.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"DCD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Guard the logit against v = 1 and v < 0.5. The (2v-1) power
    # is safe for v = 0.5 (charge collapses to the flat floor c0).
    v = np.clip(v, 0.5, 1.0 - 1e-6)

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])
    alpha = float(parameters["alpha"])

    # Dampened confidence-distrust charge per endorser: flat
    # suspicion floor plus a steeply validity-scaled distrust term
    # that vanishes toward chance validity.
    d = c0 + c1 * np.power(2.0 * v - 1.0, kappa)

    # Total accumulated suspicion for each option.
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # Residual validity-weighted endorsement (trust) term: a small
    # tendency to follow the more credibly endorsed option that
    # pulls matched-count choices toward (and slightly past)
    # chance in favor of the higher-validity endorser.
    w = np.log(v / (1.0 - v))
    endorse = float(np.dot(w, a - b))

    # D > 0 favors A.
    D = (pen_b - pen_a) + alpha * endorse

    # Numerically stable sigmoid: 0.5 * (1 + tanh(x/2)).
    # D == 0 (exact tie) -> exactly 0.5.
    x = beta * D
    p_a = 0.5 * (1.0 + np.tanh(0.5 * x))
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)

## THEORY 2 — pi_5
Validity-Proportional Skepticism (VPS), a.k.a. Confidence-Distrust Defect Integration. Subjects treat every expert endorsement not as an asset but as a liability whose magnitude scales with the endorser's claimed validity: a highly valid expert's praise is a strong claim, and strong claims attract the most suspicion. On each trial the decision maker computes a penalty for each option, P(X) = sum_j x_j * (c0 + c1 * v_j^kappa), where c0 is a flat defect-count term (c1 = 0 recovers pure unweighted defect counting) and c1 * v_j^kappa is the confidence-distrust term, with penalty INCREASING in the endorser's validity and kappa controlling how steeply. The subject chooses the lower-penalty option via P(A) = sigmoid(beta * (P(B) - P(A))) mixed with a uniform lapse epsilon. History is ignored because validities are communicated in the instructions, so skepticism weights are fixed for the whole block. VPS reproduces the sign of every observed metric while fixing the one structural blind spot of pure defect counting (SDI): on matched-count trials with a large validity gap (e.g., a single 0.90 endorser vs a single 0.60 endorser), SDI is pinned at 0.5 while VPS predicts a clear majority for the LOWER-validity endorser — exactly what the Experiment 6 composite (observed 0.193) requires.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # Validity-Proportional Skepticism (VPS) /
    # Confidence-Distrust Defect Integration.
    #
    # Stimulus: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    #
    # Each endorsement is a LIABILITY whose magnitude scales with the
    # endorser's claimed validity (a strong claim attracts suspicion):
    #   penalty for option X:  P(X) = sum_j x_j * (c0 + c1 * v_j^kappa)
    #     c0        -> flat defect-count term (c1 = 0 recovers pure
    #                  unweighted endorsement counting)
    #     c1*v^kappa -> confidence-distrust term, INCREASING in validity
    # Decision variable:  D = P(B) - P(A)   (positive favors A)
    # P(A) = sigmoid(beta * D), mixed with a uniform lapse epsilon.
    # History is ignored: validities are in the instructions, so the
    # skepticism weights are fixed for the whole block.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"VPS expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]

    v = np.asarray(parameters["validities"], dtype=float)
    if v.shape[0] != n_features:
        raise ValueError(
            f"validities length {v.shape[0]} != n_features {n_features}."
        )
    # Validities live in [0.5, 1.0]; clip defensively (no logarithms
    # are taken, so v = 1 is harmless here).
    v = np.clip(v, 0.5, 1.0)

    kappa = float(parameters["kappa"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    c0 = float(parameters["c0"])
    c1 = float(parameters["c1"])

    # Per-endorser liability: flat suspicion plus a
    # validity-proportional confidence-distrust charge.
    d = c0 + c1 * (v ** kappa)

    # Total accumulated suspicion for each option.
    pen_a = float(np.dot(d, a))
    pen_b = float(np.dot(d, b))

    # D > 0 favors A (B carries more accumulated suspicion than A).
    D = pen_b - pen_a

    # Numerically stable sigmoid: 0.5 * (1 + tanh(x/2)).
    # D == 0 (exact tie in penalties) -> exactly 0.5.
    x = beta * D
    p_a = 0.5 * (1.0 + np.tanh(0.5 * x))
    p_core = np.array([p_a, 1.0 - p_a], dtype=float)

    # Independent lapse: with probability epsilon pick uniformly.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)

## EXPERIMENT 1 (proposed by pi_6)

### DESIGN
**Validities (n_features=20):** [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]

**Trial pairs (n=32):**
  trial 1: A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 20: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 22: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 25: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 26: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  trial 27: A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 28: A=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 29: A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 30: A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  trial 31: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  trial 32: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

**Rationale:** NOVEL AXIS (vs Exps 0-7): the first head-to-head between the two CONFIDENCE-DISTRUST theories, DCD (advocated) and VPS (competing). Both treat endorsements as liabilities increasing in the endorser's validity, so count-conflict cells and high-validity cells largely produce AGREEMENT; the dissociation lives in the SHAPE of the distrust weight near chance validity plus DCD's residual trust term. Formally, DCD's effective per-endorsement score s(v) = c0 + c1*(2v-1)^kappa - alpha*logit(v) is NON-MONOTONE in v (at mid parameters s(0.55)=0.268 < s(0.50)=0.280 < s(0.60)=0.271... < s(0.65)=0.288: the LEAST suspicious endorser is a 55-60% expert, because the (2v-1)^kappa dampening collapses the distrust charge near chance while alpha*logit(v) actively rewards validity), whereas VPS's penalty p(v) = c0 + c1*v^kappa is STRICTLY increasing (a 0.60 expert's endorsement costs c1*(0.6^kappa - 0.5^kappa) = 0.07-0.17 units more than a coin-flip's; DCD's corresponding charge is <= 0.043 units and is OPPOSED by alpha*logit(0.60) >= 0.032). At MATCHED endorsement counts VPS's flat term c0 cancels exactly, so VPS sign-robustly prefers the LOWER-validity endorser on every matched-count pair across its entire box (P(A) < 0.5 always); DCD prefers the 0.55/0.60 endorser (sign-robustly ABOVE 0.5 for v=0.55; within +/-0.06 of 0.5 for v=0.60). The design therefore concentrates on matched-count composition among near-chance experts and avoids the moderate/high-validity count cells where the two distrust theories agree. Validities (20 experts): 0.50 x4, 0.55 x4, 0.60 x4, 0.65 x3, 0.70 x3, 0.90 x2. 16 base pairs, each exactly A/B-mirrored (32 unique pairs, K = 96//32 = 3, 96 trials, 6 samples per base cell per subject), perfectly side-balanced. Predictions are P(A) (A = higher-validity-composition side) at DCD mid (c0=.28, c1=.78, kappa=2, alpha=.10, beta=2.4, eps=.10) and VPS mid (c0=.10, c1=1.0, kappa=2, beta=2.0, eps=.10); box ranges include the full parameter boxes AND the lapse. FAMILY 1 -- k-GRADIENT AT 0.55 vs 0.50 (pairs 1-4, k=1..4): DCD 0.51, 0.51, 0.52, 0.53 (flat-rising, ABOVE 0.5, sign-robust: -c1*0.1^kappa + alpha*0.2007 >= +0.004 per endorsement); VPS 0.48, 0.45, 0.43, 0.41 (descending; per-unit D = -c1*(0.55^kappa - 0.5^kappa) in [-0.070, -0.034], remarkably stable across kappa). OPPOSITE-SIGNED SLOPES in k; boxes disjoint including lapse (k=4: DCD [0.51, 0.55] vs VPS [0.36, 0.45]). FAMILY 2 -- k-GRADIENT AT 0.60 vs 0.50 (pairs 5-8): DCD 0.51, 0.51, 0.52, 0.52 (flat; per-unit D in [-0.011, +0.027], bounded by the dampening and opposed by the trust term); VPS 0.45, 0.40, 0.36, 0.31 (diving; per-unit in [-0.146, -0.074]). At k=4: DCD [0.48, 0.56] vs VPS [0.24, 0.39] -- disjoint by 9+ points. FAMILY 3 -- VALIDITY LADDER AT k=3 (pairs 3, 7, 9, 10: 3x{0.55, 0.60, 0.65, 0.70} vs 3x0.50): VPS is a monotone steep descent 0.43, 0.36, 0.29, 0.22 (box ranges [0.39,0.46], [0.29,0.42], [0.21,0.37], [0.15,0.33]); DCD is HUMP-SHAPED 0.52, 0.52, 0.49, 0.44 (boxes [0.52,0.53], [0.50,0.55], [0.44,0.54], [0.38,0.51]), crossing 0.5 between v=0.60 and v=0.65 -- DCD's trust-distrust equilibrium validity (~0.63) that VPS structurally cannot exhibit. Max mid separation 22 points at v=0.70; all four rungs box-disjoint. FAMILY 4 -- ONE-STEP GAP CELLS (pairs 13, 11, 14, 12: {2x0.60, 3x0.60, 2x0.65, 3x0.65} vs {2x0.55, 3x0.55}): a single 0.05-validity step among near-chance experts moves VPS 0.45, 0.42, 0.39, 0.35 (boxes down to [0.28,0.41]) but leaves DCD blind at 0.50, 0.50, 0.48, 0.47 (all boxes within [0.43, 0.51]); all four disjoint. FAMILY 5 -- HIGH-VALIDITY ANCHORS (pairs 15, 16): pair 15 (1x0.90 vs 1x0.50): both avoid the 0.90 endorser but DCD is systematically softer (DCD 0.35, box [0.31, 0.39] vs VPS 0.27, box [0.21, 0.34]) -- DCD's softening of VPS's extremes. Pair 16 (1x0.90 vs 2x0.50) is the c0-MAGNITUDE assay: DCD's large flat charge exactly offsets its dampened 0.90 distrust (2*s(0.50) = 0.560 = s(0.90) at mid; P(A) = 0.50), while VPS's small c0 (2*0.35 = 0.70 << 0.91) leaves the validity-distrust gap exposed (P(A) = 0.41) -- a 10-point mid separation testing whether the count floor can rescue the high-validity endorser. UNFAKEABILITY IN BOTH DIRECTIONS: (i) no VPS parameterization can produce P(A) >= 0.5 on ANY matched-count cell (c0 cancels, c1 > 0, v^kappa strictly increasing), so the above-chance preference for the 0.55 endorser (Family 1) or even chance-level performance at k=4 where VPS sits at 0.24-0.31 (Family 2) falsifies VPS outright; (ii) no DCD parameterization can reproduce VPS's low-validity ladder: DCD's matched-count decision variable between 0.60 and 0.50 experts is bounded per endorsement in [-0.011, +0.027] (dampened distrust <= 0.043, opposed by alpha*logit >= 0.032), so at any k <= 4 it cannot leave [0.44, 0.57], while VPS's is <= -0.073 per endorsement (P <= 0.39 at k=4 including lapse); between 0.55 and 0.50 DCD is bounded BELOW by +0.004 per endorsement while VPS is bounded ABOVE by -0.034. The two theories are thus placed in orthogonal sensitivity regimes exactly at the validity range where their weight profiles diverge maximally: VPS is most sensitive precisely where DCD is dampened. 14 of 16 base pairs have fully disjoint predicted intervals across the ENTIRE parameter boxes including lapse; the two high-validity anchors (pairs 15, 16) are secondary magnitude cells (10-11-point mid separations, same direction). All 32 pairs are non-degenerate (every pair has discriminating endorsements, both options carry >= 1 endorsement and >= 1 non-endorsement, no all-tie pairs, no pair where both theories guess -- every cell has at least one theory off-chance by >= 4 mid-points, and the profile statistics use all cells), and every trial requires actually comparing the two rating vectors. Both theories are history-free (validities in instructions), so the K=3 repetition is harmless.

**Computed schedule:** 32 unique pairs × 3 reps = 96 trials per subject.



### METRIC
Rationale:
The design's actual validities are 0.50 (features 0-5) and 0.60 (features 6-19), so the entire DCD-vs-VPS dissociation lives on matched-endorsement-count trials: there the flat count term c0 cancels in BOTH theories and the choice is driven purely by the shape of the distrust weight near chance validity. VPS's penalty c0 + c1*v^kappa is strictly increasing in v, so on every matched-count pair it sign-robustly avoids the 0.60 experts (per-extra-0.60-endorsement decision contribution between -0.146 and -0.074 across its whole box, giving a weighted P(higher-validity side) around 0.38 at mid parameters, <= ~0.45 even at the most forgiving box corner). DCD's dampened charge c1*(2v-1)^kappa nearly vanishes at v=0.6 (0.022-0.043) and is actively opposed by the residual trust term alpha*logit(0.6) (0.032-0.049), so its matched-count decision variable stays within about +/-0.011 per endorsement and P(higher-validity side) stays at/above chance (~0.51 at mid, >= ~0.49 across the box). The metric is the m-weighted proportion of choosing the side with more 0.60-expert endorsements over all matched-count, non-tie trials (78 of 96 trials per subject; weights m = 1,2,3,4 concentrate on the cells with the largest predicted separation, e.g. m=4: DCD ~0.52 vs VPS ~0.31). It is a simple bounded proportion, so per-subject binomial noise is small (SD ~0.06 over 78 trials with these weights) and parameter heterogeneity barely moves DCD off 0.5 while VPS stays pinned below 0.45 — giving a mean gap of ~0.13 with tight per-subject values, far beyond what N=25 needs for a Welch t-test at alpha=0.01. Count-conflict trials (pairs 16/32) and pure validity ties (pairs 1/2/17/18) are excluded because both theories agree there, so including them would only add noise.

Source:
def metric(data: pd.DataFrame) -> float:
    # Weighted proportion of choosing the side whose endorsements come from
    # MORE VALID experts, on matched-endorsement-count trials.
    #
    # Design validities (n_features=20, fixed): indices 0-5 are 0.50,
    # indices 6-19 are 0.60. On every trial where both options carry the
    # SAME number of endorsements, the flat count terms (c0) of both
    # DCD and VPS cancel exactly, so the choice is driven purely by the
    # validity composition of the endorsers:
    #   - VPS: penalty strictly increasing in v  -> always avoids the
    #     0.60 experts -> weighted P(hi-validity side) well below 0.5
    #     (mid ~0.38, box <= ~0.45).
    #   - DCD: dampened distrust (2v-1)^kappa collapses near chance and
    #     is opposed by the residual trust term alpha*logit(v) ->
    #     essentially at/above chance (mid ~0.51, box >= ~0.49).
    # Trials are weighted by m = |#0.60-endorsements difference|, which
    # emphasizes the cells (m=2,3,4) where the two theories diverge most.
    V = np.array([0.5] * 6 + [0.6] * 14, dtype=float)
    hi = (V > 0.55).astype(float)  # indicator of the 0.60-validity experts

    if data is None or len(data) == 0:
        return 0.5

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    resp = data['response'].to_numpy()

    num = 0.0
    den = 0.0
    for i in range(len(data)):
        a = np.asarray(a_col[i], dtype=float)
        b = np.asarray(b_col[i], dtype=float)
        if a.shape[0] != V.shape[0] or b.shape[0] != V.shape[0]:
            continue
        tA = float(np.sum(a))
        tB = float(np.sum(b))
        # Only matched-count trials: the flat c0 terms cancel and the
        # choice isolates the validity-composition sensitivity.
        if tA != tB:
            continue
        hA = float(np.dot(hi, a))  # number of 0.60-expert endorsements on A
        hB = float(np.dot(hi, b))
        if hA == hB:
            continue  # pure tie in validity composition
        m = abs(hA - hB)
        r = int(resp[i])
        chose_hi = (r == 0) if (hA > hB) else (r == 1)
        den += m
        if chose_hi:
            num += m

    if den <= 0:
        return 0.5
    return num / den

### RESULTS
- Predicted under pi_6 (simulated): 0.5269 (var=0.0032)
- Predicted under pi_5 (simulated): 0.3820 (var=0.0036)
- Observed on real data: 0.3729 (var=0.0159)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=20):** [0.92, 0.9, 0.9, 0.9, 0.85, 0.77, 0.77, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.72, 0.72, 0.5, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 10: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 11: A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 20: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 22: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 23: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]

**Rationale:** NOVEL AXIS (vs. Exps 0-8): COALITION COMPOSITION at matched endorsement count with arithmetically CANCELED trust, plus the COIN-FLIP EXCHANGE LOCUS. Exp 8 dissociated VPS from DCD using matched-count gradients among near-chance experts (0.55-0.70 vs 0.50) and two 0.90 anchors. It never pitted a MIXED coalition {v_hi, 0.50} against a HOMOGENEOUS coalition {v_mid, v_mid}, and never mapped how many coin-flip endorsements balance mid-validity endorsements. Both are direct assays of the one structural difference between the two confidence-distrust theories: the GEOMETRY of the distrust charge. VPS's charge c0 + c1*v^kappa uses RAW validity, so a 0.50 endorsement still costs c1*0.25-0.33 distrust units; DCD's charge c0 + c1*(2v-1)^kappa is DAMPENED to the bare floor c0 at v=0.50, and its residual trust term alpha*logit(v) partially refunds validity. Validities (20 experts): [0.92, 0.90x3, 0.85, 0.77x2, 0.75x6, 0.72x2, 0.50x5]. 12 base pairs, each exactly A/B-mirrored (24 unique pairs, K = 96//24 = 4, 96 trials, 8 samples per base cell per subject), perfectly side-balanced. FAMILY 1 -- COMPOSITION SIGN CONFLICT (pairs 1-5): A = {v_hi, 0.50} vs B = {v_mid, v_mid} where v_mid is the LOGIT-MEAN of v_hi and 0.50, so DCD's trust term cancels by arithmetic identity: logit(0.90) = 2*logit(0.75) exactly (ln 9 = 2 ln 3; rungs 0.92/0.77 and 0.85/0.72 have residuals of only +0.003 and -0.015 at mid alpha). With trust gone, the pure charge shapes make OPPOSITE predictions, sign-robustly across the ENTIRE parameter boxes: VPS satisfies 2*v_mid^kappa > v_hi^kappa + 0.5^kappa for every kappa in [1.6, 2.4] at every rung (e.g. 0.90/0.75: 2*0.75^kappa - 0.9^kappa - 0.5^kappa in [+0.037, +0.087]), so the MIXED coalition is LESS suspicious and P(A) > 0.5 always (D >= 0.7*0.037 > 0). DCD satisfies (2v_hi-1)^kappa > 2*(2v_mid-1)^kappa for every kappa in [1.85, 2.15] (e.g. 0.8^kappa - 2*0.5^kappa in [0.122, 0.181]), so the mixed coalition is MORE suspicious and P(A) < 0.5 always (D <= -0.068 with trust residual <= +0.003). Mid-parameter predictions (VPS: kappa=2, beta=2, eps=.10, c0=.1, c1=1; DCD: kappa=2, beta=2.4, eps=.10, c0=.28, c1=.78, alpha=.10), with full-box ranges including lapse: pair 1 {90,50}v{75,75}: VPS 0.53 [0.51,0.57] vs DCD 0.44 [0.44,0.48] -- DISJOINT. Pair 2 (amplified x2): VPS 0.56 [0.52,0.62] vs DCD 0.39 [0.37,0.43] -- DISJOINT. Pair 3 (amplified x3): VPS 0.59 [0.52,0.68] vs DCD 0.33 [0.31,0.35] -- DISJOINT with an 18-point gap between interval edges. The amplification gradient is itself a signature: opposite-signed slopes in k (VPS rises 0.53->0.56->0.59; DCD falls 0.44->0.39->0.33), because both decision variables scale linearly in the number of replicated coalitions. Pairs 4-5 replicate the composition conflict at v_hi = 0.85 (vs 0.72) and 0.92 (vs 0.77): VPS 0.53 [0.51,0.55] and 0.54 [0.52,0.58] vs DCD 0.45 [0.44,0.47] and 0.45 [0.45,0.47] -- both disjoint, showing the effect spans the whole upper-validity range. FAMILY 2 -- EXCHANGE LOCUS (pairs 6-10): A = {75,75} (or {75x3}) vs B = k coin-flip endorsements, k = 2..5. VPS's internal exchange rate is (c0+c1*0.75^kappa)/(c0+0.25c1) ~ 1.9 coin-flips per 0.75 (it must 'pay' the substantial raw-v distrust of a 0.50 endorsement); DCD's is (c0+c1*0.5^kappa-alpha*1.0986)/c0 ~ 1.3 (a 0.50 endorsement costs only the floor, and trust refunds part of the 0.75 charge). The indifference point therefore differs by a full rung: VPS crosses 50% between k=3 and k=4 per two 0.75s; DCD crosses between k=2 and k=3. Pair 6 (k=2): VPS 0.25 [0.22,0.35] vs DCD 0.41 [0.38,0.48] -- disjoint magnitude conflict. Pair 7 (k=3): SIGN CONFLICT, VPS 0.38 [0.26,0.48] vs DCD 0.56 [0.48,0.64]. Pair 8 (k=4): VPS 0.53 [0.41,0.67] vs DCD 0.70 [0.60,0.76] -- DCD robustly prefers TWO 0.75 endorsements over FOUR coin-flips; VPS calls it near-chance. Pair 9 (k=5): agreement baseline (VPS 0.68, DCD 0.80). Pair 10 ({75x3} vs {50x5}, ratio 1.67 -- squarely between the two exchange rates): the strongest cell in the design, SIGN CONFLICT with VPS 0.40 [0.26,0.55] vs DCD 0.66 [0.53,0.71]; DCD's D > 0 by >= 0.076 at every box corner while VPS's mid D = -0.24. FAMILY 3 -- CALIBRATION (pairs 11-12): pair 11 ({90} vs {50}) both avoid the 0.90 endorser with DCD systematically softer (VPS 0.27 [0.20,0.34] vs DCD 0.35 [0.32,0.45]); pair 12 ({50,50} vs {90,90}) is a ceiling agreement cell (VPS 0.86, DCD 0.76) that scales each theory's slope where they concur. UNFAKEABILITY IN BOTH DIRECTIONS: (i) on all five composition cells, no DCD parameterization can reach P(A) = 0.5 (its D is negative by >= 0.068 with the trust residual bounded in [-0.019, +0.003]), and no VPS parameterization can fall to 0.5 (its D is positive by >= 0.026) -- the sign of the majority choice on any composition cell adjudicates the theories with no free parameters able to reverse it, and the k-amplification makes the P3 cell a 26-point mid separation; (ii) on the exchange family, the only VPS corner that reaches a positive decision variable on pair 10 (c0=0.15, c1=0.7, kappa=2.4) simultaneously predicts pair 6 at 0.34 (vs DCD 0.43) and pair 3 at 0.53 (vs DCD 0.31) -- a joint profile across cells that neither theory's box can mimic, since the composition conflict and the exchange conflict are two faces of the same structural difference (raw-v vs dampened charge) and no single reparameterization flips both. As the VPS advocate, the predicted profile is: majority for the MIXED coalition on all composition cells, growing with amplification (0.53 -> 0.59), and coin-flip avoidance that persists through k=3 (0.38) with an indifference point near k=4. A DCD subject instead shows majority-HOMOGENEOUS on all composition cells (falling to 0.33 at k=3), near-chance already at k=3 on the exchange gradient (0.56), and a robust preference for two 0.75s over four coin-flips (0.70). All 24 pairs are non-degenerate (every pair has discriminating endorsements, both options carry >= 1 endorsement and >= 1 non-endorsement, no all-tie pairs, no pair where both theories guess), and every trial requires actually comparing the two rating vectors. Both theories are history-free (validities in the instructions), so K=4 repetition is harmless.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



### METRIC
Rationale:
The metric is the proportion of trials on which the subject chose the option carrying MORE endorsements from the five v=0.50 (coin-flip) experts. Every one of the 24 unique pairs (96 trials/subject, 4 reps each) has a strict coin-flip asymmetry, so all trials contribute, and the metric is a simple Bernoulli proportion that works identically on the pooled data and on a single subject's 96-trial slice. It targets the single structural difference between the theories: VPS's raw-v charge c0+c1*v^kappa makes a 0.50 endorsement cost substantial distrust (exchange rate ~1.9 coin-flips per 0.75 endorsement), while DCD's dampened charge c0+c1*(2v-1)^kappa collapses to the floor c0 at v=0.50 with a trust refund (exchange rate ~1.3). Consequently, on every cell family the VPS subject is more coin-flip tolerant: composition cells (mixed {v_hi,0.50} vs homogeneous mid-validity) give VPS 0.53/0.56/0.59 (pairs 1-3, amplified) vs DCD 0.44/0.39/0.33 — opposite-signed majorities that no parameterization in either box can reverse; exchange cells (75-coalition vs k coin-flips) give VPS 0.75/0.62/0.47/0.32/0.60 vs DCD 0.59/0.44/0.30/0.20/0.34; calibration cells give VPS 0.73/0.86 vs DCD 0.65/0.76. Averaging over all 24 pairs: VPS mid-parameter value ≈ 0.59, DCD ≈ 0.45 — a 0.146 mean gap that is disjoint across the two theories' full parameter boxes (VPS per-cell box ranges stay above DCD's on essentially every cell, with the composition cells sign-robust). Per-subject standard deviation is ~0.05 (binomial over 96 trials) plus modest parameter heterogeneity (~0.02-0.04), giving Welch t ≈ 8-9 at N=25, far below alpha=0.01. Averaging over all cells (rather than only the sign-conflict cells) maximizes the standardized gap because the added trials shrink the binomial variance faster than the moderate cells dilute the mean difference, and it makes the metric a single interpretable behavioral quantity — coin-flip endorsement tolerance — rather than a hand-picked subset of trials.

Source:
def metric(data: pd.DataFrame) -> float:
    # COIN-FLIP ENDORSEMENT TOLERANCE.
    #
    # For every trial, count how many endorsements each option receives
    # from the five chance-validity (v = 0.50) experts (feature indices
    # 15-19 in this experiment's fixed validity vector). Every one of the
    # 24 unique pairs pits an option carrying coin-flip endorsements
    # against an option carrying none, so the trial is scored:
    #   1.0  if the subject chose the option with MORE 0.50-expert
    #        endorsements (the 'coin-flip tolerant' choice),
    #   0.0  otherwise.
    # The metric is the mean of this score over all trials.
    #
    # This is a direct assay of the one structural difference between the
    # two confidence-distrust theories: the geometry of the per-endorsement
    # distrust charge. VPS charges c0 + c1*v^kappa, so a 0.50 endorsement
    # still costs a substantial c1*0.25-0.33 distrust units and the
    # internal exchange rate between mid-validity (0.75) endorsements and
    # coin-flips is ~1.9 coin-flips per 0.75. DCD charges
    # c0 + c1*(2v-1)^kappa, which collapses to the bare floor c0 at
    # v = 0.50 (and refunds part of the 0.75 charge via the residual trust
    # term), yielding an exchange rate of only ~1.3. Hence VPS subjects
    # are systematically MORE willing to accept coin-flip endorsements,
    # across every cell family in the design:
    #   composition cells (mixed {v_hi, 0.50} vs homogeneous mid): VPS
    #     picks the mixed side (0.53-0.59), DCD avoids it (0.33-0.45);
    #   exchange cells (0.75-coalition vs k coin-flips): VPS keeps
    #     choosing the coin-flip side through k=3-4, DCD abandons it;
    #   calibration cells ({0.90} vs {0.50}, {0.50,0.50} vs {0.90,0.90}):
    #     VPS 0.73/0.86 vs DCD 0.65/0.76.
    # Mid-parameter predictions: VPS ~0.59, DCD ~0.45 (disjoint across
    # their full parameter boxes), with per-subject SD ~0.05-0.06 from
    # 96 Bernoulli trials.

    validities = np.array([
        0.92, 0.90, 0.90, 0.90, 0.85, 0.77, 0.77,
        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.72, 0.72,
        0.50, 0.50, 0.50, 0.50, 0.50,
    ])
    coin_mask = np.isclose(validities, 0.5)

    responses = data["response"].to_numpy()
    ratings_a = data["option_a_ratings"].tolist()
    ratings_b = data["option_b_ratings"].tolist()

    scores = []
    for i in range(len(ratings_a)):
        a = np.asarray(ratings_a[i], dtype=float).ravel()
        b = np.asarray(ratings_b[i], dtype=float).ravel()
        n = min(a.shape[0], b.shape[0], validities.shape[0])
        mask = coin_mask[:n]
        ca = float(np.sum(a[:n][mask]))
        cb = float(np.sum(b[:n][mask]))
        if ca == cb:
            # No coin-flip asymmetry on this trial; skip defensively.
            continue
        chose_a = (int(responses[i]) == 0)
        c_chosen = ca if chose_a else cb
        c_other = cb if chose_a else ca
        scores.append(1.0 if c_chosen > c_other else 0.0)

    if len(scores) == 0:
        return 0.5
    return float(np.mean(scores))

### RESULTS
- Predicted under pi_6 (simulated): 0.4429 (var=0.0037)
- Predicted under pi_5 (simulated): 0.6071 (var=0.0030)
- Observed on real data: 0.5096 (var=0.0121)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """P(choose the Take-The-Best winner) on trials where TTB and Tallying
    make OPPOSITE deterministic predictions (i.e., trials where the two
    options win on different numbers of features, so the tallies disagree).

    High value -> TTB behavior; low value -> Tallying behavior.
    Works on the pooled DataFrame and on a single subject's slice.
    """
    validities = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    cue_order = np.argsort(-validities, kind="stable")

    def _to_matrix(col):
        rows = [list(r) for r in col]
        return np.array(rows, dtype=float)

    a = _to_matrix(data["option_a_ratings"])
    b = _to_matrix(data["option_b_ratings"])
    resp = np.asarray(data["response"].to_numpy(), dtype=float)

    n_trials = a.shape[0]

    # --- TTB winner per trial (cue cascade, first discriminating cue) ---
    winners = np.full(n_trials, -1, dtype=int)
    for j in cue_order:
        undecided = winners < 0
        a_better = undecided & (a[:, j] > b[:, j])
        b_better = undecided & (b[:, j] > a[:, j])
        winners[a_better] = 0
        winners[b_better] = 1

    # --- Tallying score per trial (number of strict feature-wise wins) ---
    a_wins = (a > b).sum(axis=1)
    b_wins = (b > a).sum(axis=1)

    # Conflict trials: tallies differ (so Tallying has a strict winner that,
    # in this design, is always the option OPPOSITE to the TTB winner),
    # and TTB also has a strict winner.
    conflict = (a_wins != b_wins) & (winners >= 0)

    if not np.any(conflict):
        # No diagnostic trials -> uninformative, return chance.
        return 0.5

    agree = float(np.mean(resp[conflict] == winners[conflict]))
    return agree
```

**Observed (real) value:** 0.7117 (var=0.0205)
**Predicted under pi_6:** 0.6844 (var=0.0026)
**Predicted under pi_5:** 0.7272 (var=0.0025)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Signature: allegiance to the tally winner on CONFLICT trials
    # (trials where the single top-validity discriminating cue favors
    # the tally LOSER). Tallying predicts P(choose tally winner) > 0.5
    # there (it follows the cue majority against the best cue);
    # Take-The-Best predicts P < 0.5 (it follows the top cue against
    # the majority). We return P(W | conflict) - 0.5, so the sign
    # itself identifies the theory for any noise level beta > 0.
    validities = np.array([0.95, 0.90, 0.85, 0.65, 0.60, 0.50])
    cue_order = np.argsort(-validities, kind="stable")

    a_list = data["option_a_ratings"].tolist()
    b_list = data["option_b_ratings"].tolist()
    resp = data["response"].to_numpy()

    conflict_w = []
    for a_raw, b_raw, r in zip(a_list, b_list, resp):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)

        a_w = float(np.sum(a > b))
        b_w = float(np.sum(b > a))
        if a_w == b_w:
            # Tally is tied -> heuristic guesses; uninformative cell.
            continue
        winner = 0 if a_w > b_w else 1

        # First discriminating cue in descending-validity order.
        top_favors = None
        for j in cue_order:
            if a[j] > b[j]:
                top_favors = 0
                break
            if b[j] > a[j]:
                top_favors = 1
                break
        if top_favors is None:
            continue  # no discriminating cue at all

        if top_favors == winner:
            continue  # congruent trial: both heuristics point the same way

        # Conflict trial: tally winner opposes the top cue's favorite.
        conflict_w.append(1.0 if int(r) == winner else 0.0)

    if len(conflict_w) == 0:
        return 0.0
    return float(np.mean(conflict_w)) - 0.5
```

**Observed (real) value:** -0.2562 (var=0.0128)
**Predicted under pi_6:** -0.1978 (var=0.0023)
**Predicted under pi_5:** -0.2456 (var=0.0038)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Validity-sensitivity contrast for the cue-2 cascade design.

    For every trial we compute (a) the Take-The-Best winner (first
    discriminating cue in validity order) and (b) the VWEI evidence
    E = sum_j [log(v_j/(1-v_j))]^3 * sign(a_j - b_j), expressed
    relative to the TTB winner (Erel = E * (+1 if TTB winner is A
    else -1)).

    Two trial classes:
      HIGH  : Erel >  3   -> pairs 1-4, where the 90% expert decides
             and VWEI strongly agrees (P(pick TTB winner) ~ 0.85).
      LOW   : Erel < -0.5 -> pairs 7-12 and 15-16, where cue 2 (or 3)
             decides but the weighted evidence of the opposing
             coalition dominates (P(pick TTB winner) ~ 0.35-0.47).

    Metric = P(choose TTB winner | HIGH) - P(choose TTB winner | LOW).

    TTB predicts the SAME choice probability on every non-tied pair
    (its score is binary), so this difference is 0 in expectation.
    VWEI predicts a large positive difference (~ +0.42). The contrast
    cancels each subject's overall response rate (beta / epsilon),
    leaving only the stimulus-driven validity sensitivity.
    """
    v = np.array([0.90, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68, 0.66, 0.64])
    lo = np.log(v / (1.0 - v))
    w = lo ** 3.0
    order = np.argsort(-v, kind="stable")

    highs = []
    lows = []

    for a_cell, b_cell, r in zip(
        data["option_a_ratings"], data["option_b_ratings"], data["response"]
    ):
        a = np.asarray(a_cell, dtype=float)
        b = np.asarray(b_cell, dtype=float)
        if a.shape[0] != 9 or b.shape[0] != 9:
            continue

        # VWEI-style evidence for A over B (gamma = 3).
        d = np.sign(a - b)
        E = float(np.dot(w, d))

        # TTB winner: first discriminating cue in validity order.
        win = None
        for j in order:
            if a[j] > b[j]:
                win = 0
                break
            if b[j] > a[j]:
                win = 1
                break
        if win is None:
            continue  # no discriminating cue (does not occur here)

        # Evidence relative to the TTB winner.
        s = 1.0 if win == 0 else -1.0
        Erel = E * s

        r_val = int(r)
        chose_winner = 1.0 if r_val == win else 0.0

        if Erel > 3.0:
            highs.append(chose_winner)
        elif Erel < -0.5:
            lows.append(chose_winner)

    if not highs or not lows:
        return 0.0

    return float(np.mean(highs) - np.mean(lows))
```

**Observed (real) value:** -0.5467 (var=0.0421)
**Predicted under pi_6:** -0.4858 (var=0.0140)
**Predicted under pi_5:** -0.5708 (var=0.0133)

### Experiment 6
**Design**
  A=[1, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """
    Signed-evidence slope.

    For every trial we compute the VWEI evidence
        E = sum_j w_j * sign(a_j - b_j),  w_j = logit(v_j)^3
    and the TTB winner (first discriminating cue in validity order;
    validities are strictly descending so cue order = feature order).

    Define the signed regressor  x = +E if TTB's winner is A, x = -E if
    TTB's winner is B, and the outcome y = 1 iff the subject chose TTB's
    winner.  Return the OLS slope of y on x.

    TTB's probability of choosing its own winner is a single constant
    (softmax(beta) mixed with a lapse) on EVERY trial, so its expected
    slope is exactly 0.  VWEI's probability of choosing TTB's winner is
    sigmoid(beta * x) on every trial, giving a clearly positive slope
    (~0.05 at the design center, sign-stable over the whole parameter
    box).  The two theories are maximally separated on this statistic.
    """
    if data is None or len(data) == 0:
        return 0.0

    validities = np.array([0.74, 0.73, 0.70, 0.69, 0.68,
                           0.67, 0.66, 0.65, 0.64], dtype=float)
    v = np.clip(validities, 0.5 + 1e-9, 1.0 - 1e-6)
    w = np.log(v / (1.0 - v)) ** 3.0

    def _to_vec(cell):
        if isinstance(cell, str):
            return [float(ch) for ch in cell if ch in '01']
        return [float(val) for val in cell]

    a = np.array([_to_vec(r) for r in data['option_a_ratings']], dtype=float)
    b = np.array([_to_vec(r) for r in data['option_b_ratings']], dtype=float)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] != len(w):
        return 0.0

    resp = data['response'].to_numpy(dtype=float)

    d = np.sign(a - b)                     # +1 favors A, -1 favors B
    E = d @ w                              # VWEI evidence for A

    disc = d != 0
    keep = disc.any(axis=1)                # drop degenerate all-tie rows
    d, E, resp, disc = d[keep], E[keep], resp[keep], disc[keep]
    if d.shape[0] == 0:
        return 0.0

    # First discriminating cue (validities strictly descending -> order 0..8)
    first_idx = disc.argmax(axis=1)
    rows = np.arange(d.shape[0])
    d_first = d[rows, first_idx]
    winner_a = d_first > 0                  # TTB's winner is option A

    # y = 1 iff subject chose TTB's winner; x = evidence signed toward winner
    y = np.where(winner_a, 1.0 - resp, resp)
    x = np.where(winner_a, E, -E)

    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 1e-12:
        return 0.0
    yc = y - y.mean()
    slope = float(np.dot(xc, yc)) / denom
    return slope
```

**Observed (real) value:** -0.1938 (var=0.0027)
**Predicted under pi_6:** -0.1987 (var=0.0006)
**Predicted under pi_5:** -0.2142 (var=0.0004)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """Top-cue allegiance under endorsement-majority conflict.

    Selected trials: the 95%-validity expert (feature index 0) endorses
    exactly one option, AND that option carries strictly MORE total
    endorsements (sum of ratings) than its rival.  On these trials the
    metric is the empirical rate of choosing the option endorsed by the
    top expert.

    VWEI (any gamma, beta, epsilon) must follow the top cue at ~0.98
    here: the amplified log-odds weight of v=.95 dominates every
    possible opposing coalition of the remaining cues, and 50%-expert
    endorsements contribute exactly zero weight.
    SDI must AVOID the top-cue option here (~0.15-0.25): every
    endorsement is a liability, and the top-cue option is also the
    endorsement-majority option.
    """
    df = data.reset_index(drop=True)

    a = np.array([list(x) for x in df['option_a_ratings']], dtype=float)
    b = np.array([list(x) for x in df['option_b_ratings']], dtype=float)
    resp = df['response'].to_numpy(dtype=float)

    if a.ndim != 2 or a.shape[0] == 0:
        return 0.5

    chose_a = resp < 0.5

    # Which option does the top (95%) expert endorse?
    top_a = a[:, 0] > 0.5
    top_b = b[:, 0] > 0.5
    # Keep only trials where exactly one option gets the top expert's
    # endorsement (guards against degenerate rows).
    xor_top = np.logical_xor(top_a, top_b)

    # Total endorsement counts per option.
    cnt_a = a.sum(axis=1)
    cnt_b = b.sum(axis=1)

    # Is the top-cue-endorsed option also the endorsement-majority
    # option (strictly more endorsements)?
    top_option_more = np.where(top_a, cnt_a > cnt_b, cnt_b > cnt_a)

    sel = xor_top & top_option_more
    if not np.any(sel):
        return 0.5

    # Did the subject choose the option endorsed by the top expert?
    followed = np.where(top_a[sel], chose_a[sel], ~chose_a[sel])
    return float(np.mean(followed))
```

**Observed (real) value:** 0.1143 (var=0.0077)
**Predicted under pi_6:** 0.1043 (var=0.0024)
**Predicted under pi_5:** 0.0852 (var=0.0016)

### Experiment 8
**Design**
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Double-dissociation composite built on the two strongest contrasts
    # engineered into this design:
    #
    # COMPONENT 1 ("big-gap single endorsers", pairs 6/18, 8 trials/subject):
    #   both options carry exactly ONE endorsement; the endorsers' validities
    #   differ by >= 0.25 (0.90 vs 0.60).  VWEI overwhelmingly follows the
    #   higher-validity endorser (P ~ 0.89 across its whole parameter box);
    #   SDI is structurally near-blind at matched endorsement counts
    #   (P ~ 0.50-0.55 across its whole box).
    #
    # COMPONENT 2 ("ridge-crest coalition", pairs 10/22, 8 trials/subject):
    #   one option carries a single 0.80-validity endorsement, the other a
    #   coalition of four 0.75-validity endorsements.  Score +1 for choosing
    #   the COALITION.  VWEI favors the coalition (P ~ 0.62); SDI strongly
    #   favors the single endorser (coalition P ~ 0.10).
    #
    # Metric = mean of the two rates.  Expected: VWEI ~ 0.75, SDI ~ 0.31.
    # Both components use only 8 trials each but their gaps point the SAME
    # way after sign alignment, so the composite has a large mean gap and
    # modest binomial-only between-subject variance.

    V = np.array([0.90, 0.85, 0.80, 0.80, 0.75, 0.75, 0.75, 0.75, 0.65, 0.60])

    a_cells = data['option_a_ratings'].tolist()
    b_cells = data['option_b_ratings'].tolist()
    resp = data['response'].to_numpy()

    scores_gap = []    # 1 = chose the higher-validity single endorser
    scores_coal = []    # 1 = chose the 4x0.75 coalition over the single 0.80 endorser

    for a_raw, b_raw, r in zip(a_cells, b_cells, resp):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        if a.shape != V.shape or b.shape != V.shape:
            continue
        ca = int(round(a.sum()))
        cb = int(round(b.sum()))
        chose_a = (int(r) == 0)

        if ca == 1 and cb == 1:
            # Single endorsement on each side: who endorsed, and how good
            # is that expert?
            ia = int(np.argmax(a))
            ib = int(np.argmax(b))
            va = V[ia]
            vb = V[ib]
            if abs(va - vb) >= 0.25:
                hi_is_a = (va > vb)
                scores_gap.append(1.0 if (chose_a == hi_is_a) else 0.0)

        elif (ca == 1 and cb == 4) or (ca == 4 and cb == 1):
            # Ridge-crest conflict: verify it is exactly the designed cell
            # (single 0.80 expert vs four 0.75 experts) and score the
            # coalition choice.
            if ca == 1:
                single_vec, coal_vec = a, b
                single_is_a = True
            else:
                single_vec, coal_vec = b, a
                single_is_a = False
            i_single = int(np.argmax(single_vec))
            coal_idx = np.where(coal_vec > 0.5)[0]
            if (
                abs(V[i_single] - 0.80) < 1e-9
                and len(coal_idx) == 4
                and bool(np.all(np.abs(V[coal_idx] - 0.75) < 1e-9))
            ):
                # +1 when the subject chose the coalition side
                scores_coal.append(1.0 if (chose_a != single_is_a) else 0.0)

    r_gap = float(np.mean(scores_gap)) if len(scores_gap) > 0 else 0.5
    r_coal = float(np.mean(scores_coal)) if len(scores_coal) > 0 else 0.5
    return 0.5 * (r_gap + r_coal)
```

**Observed (real) value:** 0.1925 (var=0.0159)
**Predicted under pi_6:** 0.2387 (var=0.0095)
**Predicted under pi_5:** 0.1862 (var=0.0087)

### Experiment 9
**Design**
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # "Validity-following rate in theory-conflict cells"
    #
    # The 15 experts split into a HIGH-validity group (features 0-6:
    # validities 0.95, 0.90, 0.90, 0.90, 0.85, 0.80, 0.65) and a
    # LOW-validity group (features 7-14: validity 0.60). Every trial in
    # this design pits a "pure high" option against a "pure low"
    # option. Two families of cells separate SDI from VPS:
    #
    # Family 1 (matched endorsement counts, validity gap): SDI's
    #   penalty is a near-flat function of validity (delta ~ 0), so it
    #   is pinned at ~0.51 for the high-validity option (its weak
    #   alpha tie-break even pushes slightly ABOVE 0.5). VPS's
    #   penalty rises steeply in validity, so it systematically picks
    #   the LOW-validity option (~0.28).
    #
    # Family 2 (count conflict where the FEWER-endorsed option is the
    #   high-validity one): SDI follows the endorsement count and
    #   picks the high-validity option (~0.72). VPS's validity-scaled
    #   distrust makes the low-validity majority cheaper, so it picks
    #   the high-validity option only ~0.30 of the time.
    #
    # The metric is the average of P(choose the high-validity-group
    #   option) over these two families. SDI predicts ~0.62; VPS
    #   predicts ~0.29.

    V = np.array([0.95, 0.90, 0.90, 0.90, 0.85, 0.80, 0.65,
                  0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60])

    f1_hits = 0.0
    f1_n = 0
    f2_hits = 0.0
    f2_n = 0

    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        m = int(min(a.size, b.size, V.size))
        if m == 0:
            continue
        a = a[:m]
        b = b[:m]

        top = V[:m] >= 0.65  # high-validity expert group
        a_top = float(np.sum(a[top]))
        a_bot = float(np.sum(a[~top]))
        b_top = float(np.sum(b[top]))
        b_bot = float(np.sum(b[~top]))

        # Require each option to be endorsed purely by one group
        # (all trials in this design satisfy this; guards oddities).
        if (a_top > 0 and a_bot > 0) or (b_top > 0 and b_bot > 0):
            continue
        ca = a_top + a_bot
        cb = b_top + b_bot
        if ca == 0 or cb == 0:
            continue

        a_is_top = a_top > 0
        b_is_top = b_top > 0
        if a_is_top == b_is_top:
            continue

        chose_a = (int(round(float(r))) == 0)
        chose_top = chose_a if a_is_top else (not chose_a)

        if ca == cb:
            # Family 1: matched counts, validity gap.
            f1_n += 1
            f2_unused = None
            f1_hits += 1.0 if chose_top else 0.0
        else:
            # Family 2: count conflict; keep only cells where the
            # FEWER-endorsed option is the high-validity one
            # (the cells where SDI and VPS point in opposite
            # directions). Cells where the fewer-endorsed option is
            # the low-validity one are agreed upon by both theories
            # and are excluded.
            fewer_is_top = (a_is_top and ca < cb) or (b_is_top and cb < ca)
            if fewer_is_top:
                f2_n += 1
                f2_hits += 1.0 if chose_top else 0.0

    p1 = f1_hits / f1_n if f1_n > 0 else np.nan
    p2 = f2_hits / f2_n if f2_n > 0 else np.nan
    vals = [p for p in (p1, p2) if not np.isnan(p)]
    if not vals:
        return 0.5
    return float(np.mean(vals))
```

**Observed (real) value:** 0.3815 (var=0.0238)
**Predicted under pi_6:** 0.3286 (var=0.0033)
**Predicted under pi_5:** 0.2844 (var=0.0060)

### Experiment 10
**Design**
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Matched-count validity-reversal index.
    #
    # On trials where A and B carry the SAME number of endorsements (k vs k),
    # the two theories make opposite-sign predictions:
    #   VPS: the coalition of LOWER-validity endorsers is preferred
    #        (a high-validity endorsement is a bigger suspected defect),
    #        with preference strengthening in k.
    #   SDI: at matched count its validity sensitivity has the wrong sign
    #        and is bounded, so it stays at or below chance for the
    #        low-validity coalition on every such trial.
    #
    # The metric is the empirical proportion of choices of the
    # LOW-validity coalition across all matched-count trials
    # (design pairs 1-6 and their A/B mirrors 13-18; 48 of the 96
    # trials per subject). VPS predicts ~0.85; SDI predicts ~0.45.
    # It is well-defined on a single subject's slice (48 Bernoulli
    # trials) and on the pooled data.

    v = np.array([
        0.95, 0.95, 0.95, 0.90, 0.90, 0.90, 0.90, 0.85, 0.85, 0.75,
        0.65, 0.65, 0.65, 0.60, 0.60, 0.60, 0.60, 0.65, 0.65, 0.65,
    ], dtype=float)

    hits = []
    for _, row in data.iterrows():
        a = np.asarray(row["option_a_ratings"], dtype=float)
        b = np.asarray(row["option_b_ratings"], dtype=float)
        if a.shape[0] != v.shape[0] or b.shape[0] != v.shape[0]:
            continue

        ca = float(a.sum())
        cb = float(b.sum())
        # Keep only matched-count trials with at least one endorsement
        # on each side (the validity-reversal assay cells).
        if ca == 0.0 or cb == 0.0 or ca != cb:
            continue

        # Mean validity of the endorsing coalition on each side.
        va = float(np.mean(v[a > 0]))
        vb = float(np.mean(v[b > 0]))
        if va == vb:
            continue

        low_is_a = va < vb
        chose_a = (int(row["response"]) == 0)
        hits.append(1.0 if (chose_a == low_is_a) else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.7808 (var=0.0201)
**Predicted under pi_6:** 0.8054 (var=0.0034)
**Predicted under pi_5:** 0.8263 (var=0.0045)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all ten experiments, pi_5 (VPS) is the clearly superior theory, and pi_6 (DCD) is not merely worse \u2014 its distinctive mechanisms were directly falsified. Scoring absolute prediction error on every experiment: Exp 1 (DCD's own adversarial design): DCD misses by 0.154 (predicted 0.527, observed 0.373) while VPS misses by only 0.009 \u2014 this is a devastating result for DCD, because Experiment 1 was specifically constructed so that DCD's dampened charge c1*(2v-1)^kappa plus its residual trust term alpha*logit(v) would keep matched-count choices at/above chance for the higher-validity endorser. The observed value (0.373) shows humans robustly AVOID the higher-validity endorser even in the near-chance validity range (0.60 vs 0.50), exactly the monotone raw-v distrust VPS predicts and exactly what DCD's dampening-plus-trust structure forbids. Exp 2 (VPS's design): observed 0.510 sits between DCD 0.443 and VPS 0.607 \u2014 DCD is nominally closer (0.067 vs 0.098), but note that DCD's error here is a systematic overshoot in the opposite direction of its Exp 1 error: no single DCD parameterization can be simultaneously too trustful (Exp 1) and too distrustful (Exp 2) on the same matched-count composition axis. On the remaining eight experiments, VPS is closer on Exps 3, 4, 5, and 8; DCD is closer on Exps 6, 7, 9, and 10 \u2014 but in every one of those cases both theories get the sign and rough magnitude right (they share the confidence-distrust core), and the errors are small (mostly < 0.05). Total absolute error: DCD \u2248 0.51, VPS \u2248 0.36. The decisive pattern is that DCD's two innovations over VPS \u2014 the (2v-1)^kappa dampening near chance validity and the alpha*logit(v) trust term \u2014 are precisely what failed in Exp 1, while VPS's raw-v monotone distrust was confirmed. However, VPS is not perfect: its errors are systematically one-directional. On Exps 2, 9, and 10, VPS overpredicts the extremity of distrust (0.607 vs 0.510; 0.284 vs 0.382; 0.826 vs 0.781), i.e., observed behavior is consistently pulled toward 0.5 relative to VPS's predictions. This suggests the true generative process is a 'softer' or heterogeneous version of validity-distrust \u2014 e.g., a mixture of VPS-style skepticism with endorsement-count following and a larger effective lapse \u2014 rather than anything resembling DCD's trust-refund mechanism. Since DCD's core description (dampened distrust + residual trust) is empirically degenerate, it should be replaced, not reparameterized: a new_model verdict would keep a falsified description. The right move is a new theory that retains VPS's confirmed monotone raw-validity distrust as its backbone but adds the moderation mechanisms needed to explain why observed choice proportions are consistently less extreme than pure VPS predicts.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace DCD with a 'Softened Validity-Distrust with Count Mixture' (SVD) theory. Core claims: (1) RETAIN VPS's confirmed backbone \u2014 every endorsement is a liability whose magnitude increases monotonically in the endorser's RAW claimed validity, with NO dampening near v=0.5 and NO trust refund (this is what Exp 1 decisively confirmed: charge d_j = c0 + c1*v_j^kappa, kappa >= 1, c0, c1 > 0). (2) Add a dual-process mixture: on each trial the subject is, with probability w, a 'skeptic' who integrates the distrust difference D = P(B) - P(A) via sigmoid(beta*D), and with probability (1-w) a 'counter' who simply chooses the option with FEWER total endorsements (pure tallying on counts, indifferent at matched counts). This count branch explains the systematic shrinkage of observed proportions toward 0.5 relative to pure VPS: in Exp 2's coin-flip exchange cells it pulls the coin-flip tolerance down from 0.61 toward the observed 0.51 (count-following favors the larger coin-flip coalitions less than VPS's distrust does); in Exp 9's count-conflict cells it pulls the high-validity-option rate up from 0.28 toward the observed 0.38; in Exp 10 it pulls the low-validity-coalition rate down from 0.83 toward 0.78; and in Exp 1's matched-count cells it is exactly neutral (counts equal), so the strong VPS-signed result (0.37) is preserved with w near 0.8-0.9. (3) Allow a larger lapse epsilon (0.10-0.25) and/or between-subject heterogeneity in beta to further soften extremes (Exps 3, 5). Parameter sketch: c0 in [0.05, 0.25], c1 in [0.5, 1.3], kappa in [1.5, 2.5], beta in [1.0, 2.5], w in [0.6, 0.95], epsilon in [0.08, 0.20]. This theory differs from DCD in that its moderation comes from an explicit count-heuristic mixture and noise, not from a trust term that reverses sign near chance validity \u2014 the specific mechanism Exp 1 falsified \u2014 and it differs from VPS by predicting systematically less extreme choice proportions, which VPS over- or under-shoots in Exps 2, 9, and 10. It should be competitive with (and empirically distinguishable from) VPS on cells where counts and validity-distrust conflict at unequal endorsement counts, while agreeing with VPS on the matched-count validity gradient where VPS was strongly confirmed."
}
```

## Usage

```json
{
  "input_tokens": 36991,
  "output_tokens": 3415,
  "reasoning_tokens": 2042,
  "provider": "Sail Research"
}
```
