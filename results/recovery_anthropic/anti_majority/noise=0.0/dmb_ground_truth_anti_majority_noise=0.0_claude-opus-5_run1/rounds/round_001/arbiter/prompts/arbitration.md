# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


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


## THEORY 2 — pi_3
**WADD-γ with Positional Validity Misattribution (WADD-γ + position-default blending), with heterogeneous decision determinacy.**

People integrate cues compensatorily: each expert's subjective importance is a power transform of that expert's diagnosticity on the log-odds scale, L_j = log(v_j/(1−v_j)), w_j ∝ L_j^γ; the decision rests on the normalised evidence margin d = Σ_j w_j (x_Aj − x_Bj), with p(A) logistic in β·d and an attentional lapse ε. γ ≈ 1.15–1.85 places the decision maker in the near-lexicographic but still compensatory regime: the top expert usually survives a coalition of dissenters, but a near-peer second cue plus a couple of moderate cues can cancel or overturn it.

The substantive claim about *where the weights get attached* is retained and is the theory's main falsifiable commitment. Validities are stated verbally in the instructions, but the decision maker must bind each stated validity to a *column* of the displayed rating vector. This binding is costly and error-prone, and people fall back on a powerful display heuristic: **importance decreases from left to right** (the first-listed expert is the best one, the last-listed the worst). Subjective weights are therefore a blend, with mixing weight λ, of the correctly-bound validity weights and the weights implied by the positional default (the same weight multiset re-sorted into descending order across columns):

  w_eff ∝ (1−λ)·L_j^γ + λ·sort_desc(L^γ)_j.

This is an *environmental* prediction, not a per-experiment fudge: conflict-trial accuracy is a property of the alignment between display order and validity order, while the integration rule itself stays a single stationary compensatory WADD-γ. Concretely, (i) scramble the column order of an environment whose cues are currently presented in descending validity and conflict accuracy must fall toward (and, on some families, below) chance; (ii) re-sort the columns of a scrambled environment into descending validity and the same subjects must become sharply more accurate and more top-cue-consistent; (iii) an explicit column-labelling manipulation that makes the validity→column binding visually trivial should abolish the drop. Because the blend is the exact identity whenever columns are already validity-ordered, λ is behaviourally inert in validity-sorted environments and can only be identified by comparing environments — precisely the invariance signature the theory stakes itself on.

The third claim is about **population structure**: the same stationary rule is held by everyone, but decision determinacy is heterogeneous across people. Some subjects convert modest weighted margins into near-deterministic choices, others remain near chance on all but the most lopsided evidence. That heterogeneity is carried by the logistic gain β and by mild variation in the validity-sensitivity exponent γ, not by discrete strategy switching; the theory therefore predicts that between-subject spread in conflict accuracy should be largest in environments with intermediate margins and should shrink towards zero in environments where every conflict family is near-cancelling.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # ------------------------------------------------------------------
    # WADD-gamma with positional validity misattribution.
    #   w_j      proportional to  logit(v_j) ** gamma
    #   w_eff    = (1-lam) * w + lam * sort_desc(w)     (position default)
    #   d        = sum_j w_eff_j * (x_Aj - x_Bj)        in [-1, 1]
    #   p(A)     = sigmoid(beta * d), mixed with lapse epsilon.
    # gamma -> 0 recovers Tallying, gamma -> large recovers Take-The-Best.
    # lam is the degree of reliance on the display heuristic "importance
    # decreases left-to-right"; it is a no-op when the experiment already
    # presents cues in descending validity order.
    # History is not used: the rule is stationary (no feedback is given).
    # ------------------------------------------------------------------
    import numpy as np

    # ---- unpack the stimulus into two rating vectors -------------------
    a = b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            # flat concatenation of the two option vectors
            half = stim.shape[0] // 2
            a, b = stim[:half], stim[half:]
        else:
            stim = stim.reshape(stim.shape[0], -1)
            a, b = stim[0].astype(float), stim[1].astype(float)

    n = int(min(a.shape[0], b.shape[0]))
    a = a[:n]
    b = b[:n]
    if n == 0:
        return np.ones(2) / 2.0

    # ---- subjective importance weights from cue diagnosticity ----------
    val = parameters.get("validities", None)
    if val is None:
        v = np.full(n, 0.75, dtype=float)
    else:
        v = np.asarray(val, dtype=float).ravel().astype(float)
        if v.shape[0] < n:
            v = np.concatenate([v, np.full(n - v.shape[0], 0.55)])
        v = v[:n]

    # validities live in (0.5, 1); clip for numerical safety
    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-6)
    logit = np.log(v / (1.0 - v))          # diagnosticity on evidence scale
    logit = np.maximum(logit, 1e-9)        # strictly positive

    gamma = float(parameters.get("gamma", 1.5))
    beta = float(parameters.get("beta", 7.0))
    epsilon = float(parameters.get("epsilon", 0.0))
    epsilon = float(np.clip(epsilon, 0.0, 1.0))
    lam = float(parameters.get("lam", 0.0))
    lam = float(np.clip(lam, 0.0, 1.0))

    with np.errstate(over="ignore", invalid="ignore"):
        w = np.power(logit, gamma)
    w = np.nan_to_num(w, nan=0.0, posinf=np.finfo(float).max / (10.0 * n), neginf=0.0)

    # ---- positional misattribution: blend with the left-to-right default
    # The default binds the largest importance to the first displayed cue,
    # the next largest to the second, etc.  When the experiment already
    # orders columns by descending validity this blend is the identity.
    if n > 1 and lam > 0.0:
        w_pos = np.sort(w)[::-1]
        w = (1.0 - lam) * w + lam * w_pos

    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0.0:
        w = np.ones(n, dtype=float) / float(n)      # degenerate -> equal weights
    else:
        w = w / s

    # ---- weighted additive evidence margin -----------------------------
    d = float(np.dot(w, a - b))            # in [-1, 1]

    # numerically stable two-alternative softmax == logistic on beta*d
    z = np.array([beta * d, 0.0], dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)

    p = (1.0 - epsilon) * p_core + epsilon * 0.5
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    p = np.asarray(probs, dtype=np.float64).ravel()
    p = np.clip(p, 0.0, None)
    tot = p.sum()
    if not np.isfinite(tot) or tot <= 0:
        p = np.ones_like(p) / len(p)
    else:
        p = p / tot
    return int(np.random.choice(len(p), p=p))

## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=5):** [0.55, 0.62, 0.7, 0.85, 0.93]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 2: A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 4: A=[0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  trial 5: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 6: A=[0, 0, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  trial 7: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 9: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 10: A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 11: A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  trial 12: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 13: A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 15: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** GOAL: dissociate Take-The-Best (advocated) from WADD-\u03b3 with Positional Validity Misattribution (competing). The competitor's central falsifiable commitment is that subjective weights are largely bound to *display columns* (importance assumed to fall left-to-right, mixing weight \u03bb \u2265 0.64), whereas TTB binds validities to experts correctly and consults cues in true validity order, stopping at the first discriminator. The blend is the identity only when columns are already sorted by descending validity, so I present the five experts in strictly ASCENDING validity order (col0=.55, col1=.62, col2=.70, col3=.85, col4=.93). This maximises the mismatch: the positional default assigns the LARGEST weight to col0 (the 55% expert, TTB's very last cue) and the SMALLEST to col4 (the 93% expert, TTB's first cue).\n\nEffective competitor weights (normalised) across its parameter range: at \u03bb=.64,\u03b3=1.15 -> [.31,.22,.13,.15,.19]; at \u03bb=.80,\u03b3=1.50 -> [.44,.25,.10,.10,.12]; at \u03bb=1.0,\u03b3=1.85 -> [.60,.29,.08,.03,.01]. In every case col0 > col1 > {col2,col3,col4}. TTB's order is exactly col4 > col3 > col2 > col1 > col0. This gives sign reversals that hold over the entire competitor parameter box.\n\n16 unique pairs (K=6, 96 trials), every family mirrored so neither key is systematically correct:\n\nFAMILY A (pairs 1-2, EXTREME POSITION CONFLICT): only col0 and col4 discriminate, in opposite directions. TTB decisively picks the 93%-cue option; the competitor picks the col0 option with |d| = .13 to .60 (p up to ~1 at high \u03b2). Opposite signs, robust to \u03bb and \u03b3.\n\nFAMILY B (pairs 3-4, MILD POSITION CONFLICT): only col1 (62%) and col3 (85%) discriminate. TTB again picks the higher-validity (col3) option with EXACTLY the same probability as in Family A (sigmoid(\u03b2) is invariant across all discriminating trials); the competitor again reverses but with a smaller margin (|d| = .06 to .26). The TTB-flat vs competitor-graded profile across A > B is the core quantitative signature.\n\nFAMILY C (pairs 5-6, NEAR-CANCELLING): col2 (70%) vs col3 (85%). Their positional weights are nearly identical, so the competitor is pinned near chance (|d| \u2264 .05, and its sign even flips within the parameter box), while TTB is fully committed to the col3 option at its usual sigmoid(\u03b2) level. Any systematic above-chance choice here is fatal for the competitor's positional binding.\n\nFAMILY D (pairs 7-8, DOMINANCE ANCHOR): all five cues favour one option. Both models agree and near ceiling; this pins \u03b2 and \u03b5 for both models so the conflict families are out-of-sample predictions rather than free fits.\n\nFAMILY E (pairs 9-10, TWO-VS-TWO POSITION CONFLICT): A wins the two leftmost/least valid cues (col0,col1), B wins the two rightmost/most valid cues (col3,col4). TTB \u2192 B via col4; competitor \u2192 A with a large margin (|d| = .19 to .57). This is the strongest directional test.\n\nFAMILY F (pairs 11-12, DEEP-CASCADE vs POSITION): the top two validity cues (col4, col3) tie; col2 favours one option and col0+col1 favour the other. TTB stops at col2 and chooses against a 2-vs-1 vote; the competitor chooses the col0+col1 option strongly. Tests that the cascade, not just the single best cue, governs TTB choice, while remaining maximally diagnostic against positional weighting.\n\nFAMILIES G and H (pairs 13-16, MAGNITUDE-ONLY CONTRAST WITH DIRECTION HELD CONSTANT): in G only col4 (93%) discriminates; in H only col0 (55%) discriminates. Both models pick the option with the single 1, so there is no directional confound. TTB predicts IDENTICAL accuracy in G and H (and identical to Families A-F and D, all sigmoid(\u03b2)). The competitor predicts H \u226b G (weights .44 vs .12 at \u03bb=.8,\u03b3=1.5; .60 vs .01 at \u03bb=1), i.e. subjects should be far more decisive when only the worst-but-leftmost expert speaks than when only the best-but-rightmost expert speaks. That inversion is impossible under TTB.\n\nDECISION RULE: If TTB is right, one flat probability level accounts for all eight families, with choices always tracking the higher-validity discriminating expert (col4 > col3 > col2 > col1 > col0) irrespective of column position, and no G-vs-H asymmetry. If the positional WADD is right, choices reverse in Families A, B, E and F, sit at chance in Family C, and show a large H > G accuracy gap with graded magnitudes ordered D > F > E > H > A > B > G > C.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
Both theories are pinned near the same level on the agreement families (D: full dominance, G: only the 93% expert speaks, H: only the 55% expert speaks) \u2014 in all of these the option carrying the 1(s) is favoured by TTB and by any positive-weight compensatory rule alike. The conflict families (A, B, E, F) are constructed so the true-validity cascade and the left-to-right positional weighting point to opposite options. The metric is the WITHIN-SUBJECT difference: (rate of choosing the TTB-implied option on agreement trials) minus (rate on conflict trials).

Under Take-The-Best the choice probability is sigmoid(beta) mixed with lapse on EVERY discriminating trial regardless of which cue does the discriminating, so both terms have the same expectation and the difference is ~0 for every subject regardless of beta and epsilon; the only spread is binomial sampling noise over 36 and 48 trials (var <= ~0.012). Under WADD-\u03b3 with positional misattribution the conflict trials are answered against the TTB option (weights: col0 \u2248 .44, col1 \u2248 .25 vs col3 \u2248 .10, col4 \u2248 .12 at \u03bb=.8, \u03b3=1.5), giving conflict TTB-consistency well below .5 while agreement consistency stays high; the difference is large and positive (~.2 at the weakest \u03b2, ~.8 at moderate \u03b2). Family C is deliberately excluded because the competitor is at chance there and would dilute the contrast. Because the metric self-normalises each subject against their own agreement-trial determinacy, it is insensitive to the wide beta/epsilon heterogeneity that would otherwise inflate between-subject variance for TTB.

Source:
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


### RESULTS
- Predicted under pi_1 (simulated): -0.0092 (var=0.0059)
- Predicted under pi_3 (simulated): 0.6219 (var=0.0370)
- Observed on real data: -0.2081 (var=0.0368)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=6):** [0.6, 0.93, 0.55, 0.86, 0.68, 0.78]

**Trial pairs (n=18):**
  trial 1: A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 2: A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  trial 3: A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 4: A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  trial 5: A=[0, 1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 1, 1]
  trial 6: A=[0, 1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1]
  trial 7: A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  trial 8: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  trial 9: A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 10: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 0, 1, 1, 0]
  trial 11: A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  trial 12: A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 13: A=[0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  trial 14: A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  trial 15: A=[0, 1, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  trial 16: A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  trial 17: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** GOAL: dissociate WADD-\u03b3 with Positional Validity Misattribution (advocated) from Take-The-Best (competing). TTB binds stated validities to the correct expert and consults cues in true validity order, stopping at the first discriminator, so its choice probability is a single flat level sigmoid(beta) (plus lapse) on EVERY discriminating trial, always siding with the highest-validity discriminating expert. The advocated model claims that the validity->column binding is largely replaced by the display default 'importance falls left to right' (lam >= 0.64), and that the resulting weighted margin d produces GRADED choice probabilities.

DISPLAY ORDER (deliberately scrambled, not merely ascending): col0=.60, col1=.93, col2=.55, col3=.86, col4=.68, col5=.78. TTB's cascade is col1 > col3 > col5 > col4 > col0 > col2. The positional default instead assigns the weight multiset in descending order to columns, so col0 (a 60% expert) receives the LARGEST subjective weight and col5 (a 78% expert) receives nearly none. Effective normalised weights over the whole competitor-free parameter box: lam=.64,gamma=1.15 -> [.271,.312,.119,.157,.065,.076]; lam=.80,gamma=1.50 -> [.374,.309,.128,.112,.037,.039]; lam=1.0,gamma=1.85 -> [.518,.269,.138,.053,.017,.005]. Across the whole box col0 and col1 dominate, col2 > col3 for most of it, and col4/col5 are near zero even though col5's true validity (.78) is third best.

18 unique pairs (9 families, each mirrored so neither key is systematically correct), K=5 -> 90 trials. Shared 1s are added as background so no option is trivially empty.

FAMILY A (pairs 1-2, col0 .60 vs col3 .86 discriminate in opposite directions): TTB -> the col3 option; WADD-pos -> the col0 option, |d| ~ .26-.47. Sign reversal robust over the entire parameter box.
FAMILY B (pairs 3-4, col0 vs col5 .78): TTB -> col5 option with EXACTLY the same probability as in Family A; WADD-pos -> the col0 option again but with a LARGER margin (|d| ~ .34-.51). A>B ordering with direction held constant is the core flat-vs-graded signature.
FAMILY C (pairs 5-6, col2 .55 vs col4 .68): TTB -> the col4 option (col4 precedes col2 in the true cascade); WADD-pos -> the col2 option but only weakly (|d| ~ .05-.12). Reversal with a small, gradable margin.
FAMILY D (pairs 7-8, col0 .60 vs col1 .93): TTB is decisive for the col1 option. WADD-pos is near-cancelling and its sign flips inside the parameter box (col1 wins at lam=.64/gamma=1.15, col0 wins at lam>=.8). This is the lam-identifying family and, per the heterogeneity claim, should show the LARGEST between-subject spread of any family.
FAMILY E (pairs 9-10, coalition): one option wins col1 (the true 93% expert) alone, the other wins col0 + col5. TTB stops at col1; WADD-pos reverses (B wins for every (lam,gamma) in the box, though with a modest margin .04-.25).
FAMILY F (pairs 11-12, deep cascade vs position): col1 ties, col3 (.86) favours A while col0 + col2 favour B. TTB stops at col3 and picks A against a 2-vs-1 vote; WADD-pos picks B strongly (|d| ~ .23-.42). Strongest directional test that also verifies TTB uses the cascade, not just the single best cue.
FAMILIES G and H (pairs 13-16, MAGNITUDE ONLY, NO DIRECTIONAL CONFOUND): in G only col1 (93%) discriminates; in H only col5 (78%) discriminates. Both models favour the same option in both, so TTB must produce IDENTICAL accuracy in G and H (and identical to A-F). WADD-pos predicts G (|d| ~ .27-.31) far above H (|d| ~ .005-.08, essentially chance): subjects should be near-indifferent when only a 78% expert speaks from the rightmost column.
FAMILY I (pairs 17-18, full dominance): all six cues agree; both models near ceiling. This pins beta and epsilon for both models so every conflict family is an out-of-sample prediction rather than a free fit.

DECISION RULE: If TTB is right, one flat probability level fits families A-I, choices always track the highest-validity discriminating column (col1 > col3 > col5 > col4 > col0 > col2), and G = H exactly. If positional WADD-\u03b3 is right, choices REVERSE in A, B, C, E, F, sit near chance in D and H, and the graded ordering of |accuracy - .5| should be I > F > B > A > G > E > C > D > H, with the between-subject spread peaking in the intermediate-margin families (D, E, C) and collapsing in I and H.

**Computed schedule:** 18 unique pairs × 5 reps = 90 trials per subject.



### METRIC
Rationale:
The metric isolates the trials on which the two theories make OPPOSITE directional predictions with a large positional-WADD margin: families A (col0 .60 vs col3 .86), B (col0 vs col5 .78) and F (col3 vs the col0+col2 coalition). Take-The-Best follows the true validity cascade (col1 > col3 > col5 > col4 > col0 > col2) and therefore picks the option that positional WADD rejects on every one of these 30 trials; positional misattribution (weights re-sorted descending across display columns, giving col0 the largest subjective weight) picks the opposite option with |d| ~ .26-.39. The metric is simply the proportion of those conflict trials on which the subject chose the positionally-favoured option. Under TTB this is essentially the flip-noise floor ((1-eps)*sigmoid(-beta)+eps/2, roughly .05-.30), under the advocated model it is well above .5 (roughly .6-.95 depending on beta). Trials with |d| < .20 (families C, D, E) are excluded because the advocated model is near-indifferent there, which would only add between-subject noise; trials where both models agree (G, H, I) are excluded because they carry no directional contrast. This yields a large mean gap (~.5) while keeping each subject's value tightly bounded within its theory, since the sign of the prediction is invariant over both theories' whole parameter boxes.

Source:
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

### RESULTS
- Predicted under pi_1 (simulated): 0.1800 (var=0.0087)
- Predicted under pi_3 (simulated): 0.8380 (var=0.0096)
- Observed on real data: 0.6587 (var=0.0379)

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
**Predicted under pi_3:** 0.7375 (var=0.0102)

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
**Predicted under pi_3:** 0.5371 (var=0.0083)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Neither theory is task-invariant, and the four experiments jointly pin down what a correct account must look like.\n\npi_1 (TTB) wins where the display columns are monotone in validity. In Exp 1 (columns in strictly ASCENDING validity) the metric requires that conflict-trial choices track the true high-validity cue: observed -0.208, TTB predicted -0.009 (error .20), pi_3 predicted +0.622 (error .83). This is the single largest prediction failure in the whole set and it kills pi_3's distinctive commitment: if subjective weights were largely bound left-to-right (lambda >= 0.64), conflict trials in Exp 1 would have been answered against the 93%/85% experts, driving the metric strongly positive. It went NEGATIVE, i.e. subjects were at least as validity-consistent on conflict trials as on the trivial agreement trials. Exp 3 (columns descending) also shows strong top-cue dominance (0.753), which TTB (0.838) and a plain steep WADD (pi_3 with lambda inert, 0.738) both approximate.\n\npi_3 wins where the display is scrambled. In Exp 2 subjects chose the option favoured by column-position/equal weighting on 66% of conflict trials (TTB predicted 0.18). Importantly, a *correct-validity* compensatory rule also fails here: for families A (.60 vs .86), B (.60 vs .78) and F (.86 vs .60+.55) a correctly weighted WADD sides with TTB, giving ~0.15. What fits almost exactly is EQUAL weighting (tallying): A and B become 1-vs-1 ties (0.5) and F becomes a 2-vs-1 majority (~0.9), yielding 0.63-0.67 vs observed 0.659. Exp 4 (also scrambled, but with the .93 expert in the leftmost column) is intermediate: pure tallying predicts 0.25, correct-validity WADD ~0.58, observed 0.510; TTB's flat 0.82 is far off. So accuracy is strongly environment-dependent, which TTB structurally cannot produce (its accuracy is one flat sigmoid(beta) level everywhere), while pi_3 produces environment dependence but with the WRONG mechanism (reverse/positional re-sorting rather than flattening).\n\nTwo further facts constrain the successor. (a) What matters is the READABILITY of the validity->column mapping, not left-to-right direction: Exp 1 is anti-sorted yet subjects used true validities, Exp 3 is sorted and they did too; both scrambled environments (Exp 2, Exp 4) degrade toward equal weighting, more completely in Exp 2 (where the best expert is not even leftmost) than in Exp 4 (where it is). (b) Exp 1's negative metric requires near-chance responding on the 'agreement' families where one option carries a single endorsement and the other carries none ([00001] vs [00000], [10000] vs [00000]) while conflict families stay high. No absolute-margin model reproduces that: it needs determinacy to depend on the margin RELATIVE to the total endorsement evidence in the pair (when both products are barely endorsed, subjects treat the difference as negligible and guess).\n\nBoth theories therefore explain a subset only. pi_1's core failure is diffuse but its description is a clean, still-useful benchmark (best on Exp 1, near-best on Exp 3). pi_3's core falsifiable claim is dead, so its description is the one that should be discarded.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace pi_3 with a compensatory theory that keeps its good fits on Exps 2-4 but drops the (falsified) positional re-sorting and adds relative-margin determinacy. Sketch:\n\n**'Readability-limited validity weighting with relative-margin determinacy'.** (1) Integration is compensatory: p(A) = logistic(beta * D) with lapse epsilon, where the evidence uses subjective weights w_j proportional to logit(v_j)^gamma with gamma ~ 1.2-2.0 (steep enough that in a validity-sorted environment the top cue usually beats a coalition of 3-4 weak cues, reproducing Exp 3 ~0.75 and Exp 1's high conflict consistency).\n(2) **Binding readability, not position.** Weights are flattened toward EQUAL weighting by an amount kappa that depends on how easily the stated validities can be bound to displayed columns: w_eff = (1-kappa)*w + kappa*(1/n). kappa should be a monotone function of the *incongruence/unreadability* of the display, e.g. kappa = k0 + k1*(1 - |Kendall tau(validity rank, column order)|) with an extra increment when the single most valid expert is not in an extreme (first/last) column. Crucially kappa is SMALL for any monotone display, ascending or descending (Exp 1 tau=-1 and Exp 3 tau=+1 both readable -> validity weights preserved, which is exactly where pi_3 died), and LARGE for scrambled displays (Exp 2 -> kappa near 1 gives tallying: A/B ties at 0.5, F at ~0.9, metric ~0.65 vs observed 0.659; Exp 4 -> kappa ~0.4 gives ~0.48 vs observed 0.510). This makes the theory environmentally falsifiable in a direction-free way: it predicts that REVERSING a sorted display costs nothing, while permuting it costs a lot - the opposite of pi_3.\n(3) **Relative-margin determinacy.** The logistic argument is the weighted difference normalised by the total endorsement mass in the pair: D = sum_j w_eff_j (x_Aj - x_Bj) / (c + sum_j w_eff_j (x_Aj + x_Bj)), with c a small constant. This yields near-indifference when both options carry almost no endorsements (Exp 1 families G/H: [00001] vs [00000] and [10000] vs [00000] -> ~chance) while conflict families with two endorsed options stay decisive, which is the only way to get Exp 1's metric NEGATIVE (agreement < conflict, observed -0.208) - something both current theories get wrong in sign or magnitude.\n(4) Population structure: heterogeneous beta and mild gamma variation (no discrete strategy switching), so between-subject variance is largest in intermediate-margin/scrambled environments (matching the observed var ~0.037 in Exps 1-2 vs ~0.01-0.02 in Exps 3-4).\n\nParameter ranges should be set so the model reproduces, without per-experiment refitting: Exp 1 metric in [-0.30, -0.05], Exp 2 in [0.55, 0.75], Exp 3 in [0.70, 0.82], Exp 4 in [0.45, 0.60]. Future discriminating experiments: same validities presented sorted-ascending vs sorted-descending vs scrambled (readability theory predicts ascending = descending >> scrambled; pi_3 predicted ascending << descending), and single-endorsement-vs-zero-endorsement trials versus many-endorsement trials with the same absolute weighted margin (relative-margin determinacy predicts a large accuracy drop in the former; TTB predicts none)."
}
```

## Usage

```json
{
  "input_tokens": 17426,
  "output_tokens": 23876
}
```
