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
**Validities (n_features=6):** [0.85, 0.95, 0.65, 0.75, 0.6, 0.9]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 1, 0, 0, 1]
  trial 3: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1]
  trial 5: A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 0, 0, 1, 0]
  trial 7: A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 8: A=[1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 9: A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 1, 1]
  trial 10: A=[1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 1, 0, 1]
  trial 11: A=[0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 12: A=[0, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  trial 13: A=[1, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 14: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 1, 0]
  trial 15: A=[1, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 1]
  trial 16: A=[1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]

**Rationale:** GOAL: Quantitatively dissociate Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity — momentary-diffusion variant (advocated) from Attenuated Frugality with Position-Validity Mixture Heterogeneity (competing). Both are one-reason models sharing the anti-validity core, so the dissociation must come from their OPPOSITE heterogeneity mechanics: the advocated adds a LARGE MOMENTARY per-trial diffusion (kappa*xi, re-drawn every trial, gated by A^3) that makes each subject's consultation order an essentially fresh random permutation on every trial; the competing adds a FIXED per-subject discrete draw from {primacy, anti, recency, TTB} with primacy mass 0.80*ambiguity. The design engineers a validity list that (i) fully opens the advocated's diffusion channel, (ii) keeps the competing's per-cue zeta too weak to blur its dominant primacy component, and (iii) makes the EARLIEST-listed expert (f0) ALSO the WEAKEST-advertised (0.55), so that whenever f0 discriminates the competing's two major components (primacy + anti, ~93% mass) CONCUR — while the advocated's diffusion is structurally blind to both position and the decider's identity, responding only to the NUMBER of discriminators on each side.

SETUP: 6 features, all-distinct validities [0.55, 0.95, 0.78, 0.86, 0.62, 0.70] (f0..f5). Arrangement statistics (both models' own formulas, average-tie ranks): Spearman(position, validity) = -0.029, so binding ambiguity = 0.971 and A^3 = 0.916 — the advocated's momentary term (kappa in [8,12] times A^3, i.e. +/-[7.3, 11] per cue per trial) fully dominates its stable part (-val <= 0.40, gamma*sal <= 0.06, sigma_h*zeta <= 1.1), so the consultation order is re-randomized essentially every trial. Reversal coherence = |Spearman(val, val[::-1])| = 0.20, so the competing's zeta scale = sigma_h*ambiguity*coherence <= 0.27: it CANNOT flip the primacy component's position gaps (1.0) or recency (1.0), leaving the competing's dominant 58.6% primacy mass fully sharp (it only mildly scrambles the competing's 34% anti mass, slightly diluting but never reversing its predictions). All-distinct validities make the free tie_break parameter inert in BOTH models (no weight ties), so neither gains flexibility; the theories differ only in heterogeneity structure plus (beta, epsilon). PARAMETER-FREE MIXTURE COMMITMENT of the competing at ambiguity 0.971: primacy 0.586, anti 0.340, recency 0.044, TTB 0.030. The advocated commits to NO discrete types at all: everyone is a momentary diffuser whose per-trial decider is ~uniform over the discriminating cues, tilted only slightly by the stable anti-validity core (tilt ~ gap/(kappa*A^3) ~ 0.03 per validity gap).

THE FIVE HIERARCHIES: ANTI: f0(.55)->f4(.62)->f5(.70)->f2(.78)->f3(.86)->f1(.95). PRIMACY: f0->f1->f2->f3->f4->f5. RECENCY: f5->f4->f3->f2->f1->f0. TTB: f1->f3->f2->f5->f4->f0. ADVOCATED: ~random permutation per trial with a small low-validity lean.

SIGNATURE 1 — CONCORDANCE LADDER, FLAT vs GRADED (T1-T3, mirrors T4-T6): f0 (earliest AND weakest) always favors one option while a growing coalition of later, stronger experts (including the strongest f1 and the latest f5, so the competing's recency and TTB components oppose) favors the other. The competing's decider structure is IDENTICAL on all three rungs (primacy->f0, anti->f0, recency->B-cue, TTB->B-cue), so it predicts EXACTLY FLAT p = 0.926*p_f + 0.074*(1-p_f) ~ 0.67 [0.54, 0.80] across T1, T2, T3 — a parameter-free equality. The advocated's momentary diffusion makes the first discriminating cue ~uniformly random among the k discriminators, so p(A) = P(A-cue first)*p_f + (1-P)*(1-p_f) tracks the DISCRIMINATOR COUNT: ~0.51 (T1, 1-vs-1), ~0.45 (T2, 1-vs-2), ~0.42 (T3, 1-vs-3) — strictly GRADED and below 0.5 where the count opposes f0. The paired contrast p(T1)-p(T3) is exactly 0 under the competing and ~ +0.09 under the advocated; on T2/T3 the population intervals are DISJOINT (advocated <= 0.48 vs competing >= 0.54). Mirrors T4-T6 give the symmetric pattern (advocated 0.49/0.55/0.57; competing flat 0.33), ruling out side bias. This inverts the classic one-reason-vs-tally signature: here the DISCRETE one-reason mixture is flat and the DIFFUSE one-reason model is graded by count.

SIGNATURE 2 — PRIMACY-ALONE CELLS WITH DISJOINT INTERVALS (T7/T8): on T7 the earliest discriminator f2 (78%) favors A while the weakest (f4), latest (f5), and strongest (f3) discriminators all favor B (count 1-vs-3). The competing follows its 58.6% primacy mass to A: p(A) ~ 0.55-0.62, ABOVE 0.5 for any parameters. The advocated follows the count: p(A) ~ 0.38 [0.33, 0.45], BELOW 0.5 — the two theories predict OPPOSITE SIGNS of preference on a single cell. T8 is the exact mirror (advocated 0.62, competing 0.45). These cells also estimate the primacy mass m_hat = (p_A-(1-p_f))/(2p_f-1): the competing commits to m ~ 0.586 for THIS arrangement (a number derived entirely from its own ambiguity formula); the advocated commits to m ~ 0 (its bounded gamma tilt, <= 0.06, can never flip a decider and is swamped by the diffusion anyway).

SIGNATURE 3 — COUNT-BALANCED STRUCTURAL CELLS (T13/T14): a 2-vs-2 cell (f0,f4 -> A vs f1,f5 -> B) where the advocated's count-based diffusion is pinned near its knife-edge (~0.51, since the count is balanced) while the competing's deciders (primacy f0, anti f0 vs recency f5, TTB f1) drive p(A) ~ 0.67. Advocated <= 0.53 vs competing >= 0.54 — disjoint at the population level; the mirror T14 gives the transpose.

SIGNATURE 4 — PER-SUBJECT STRUCTURE: THE SHARPEST TEST (T7-T12): on the opposition cells the earliest discriminator and the weakest discriminator point to OPPOSITE options (T9: f2->A vs f4->B; T11: f3->A vs f5->B; T7: f2->A vs f4->B within a 1-vs-3 count). The competing predicts each subject holds ONE fixed hierarchy and is CONSISTENT across all 6 repetitions and all 6 cells: ~59% of subjects are primacy-consistent (always following the earliest discriminator, per-cell p ~ 0.73), ~34% anti-consistent (always following the weakest, p ~ 0.27) — a BIMODAL population with between-subject SD of cell choice proportions ~ 0.22 and strong cross-cell consistency (a primacy subject picks the earliest discriminator's option on T7, T9, AND T11). The advocated predicts NO structural types: momentary re-drawing keeps every subject's per-cell proportion tight around the count-based value (SD ~ 0.10) and choices INDEPENDENT across cells (cross-cell consistency ~ chance). This is a double dissociation in DISPERSION STRUCTURE: the competing predicts between-subject spread exactly where the advocated predicts tightness — the advocated theory's own headline claim (diffuse attention keeps populations tight around chance-level conformity) is tested head-on. With 6 cells x 6 repetitions = 36 classification trials per subject, the per-subject likelihood ratio between 'fixed primacy type' and 'momentary diffuser' is overwhelming.

SIGNATURE 5 — WITHIN-SUBJECT REPETITION FLIP RATE: on opposition cells the competing predicts flip rate ~2*p_f*(1-p_f) ~ 0.40 between repetitions of the same cell; the advocated predicts ~0.47-0.50 (per-trial p near 0.5). Pooled over cells this is estimable per subject.

CONTROLS: T15 (single discriminator f4->A) is an agreement cell where ALL hierarchies concur — both theories predict exactly p_f toward A, pinning the follow rate and lapse without knowing the mixture; any deviation of the dissociation-cell follow rates from p(T15) falsifies the flat-follow assumption of whichever model is otherwise supported. T16 (full tie) pins both theories to exactly 0.50, anchoring lapse and side bias. T4-T6, T8, T10, T12, T14 are exact A/B mirrors ruling out side bias.

GLOBAL PREDICTED PROFILES (population means): ADVOCATED: T1 0.51, T2 0.45, T3 0.42, T4 0.49, T5 0.55, T6 0.57, T7 0.38, T8 0.62, T9 0.49, T10 0.51, T11 0.49, T12 0.51, T13 0.51, T14 0.49, T15 ~0.71, T16 0.50 — a profile ordered by discriminator COUNT (with a small anti-validity leak toward the weakest discriminator), tight across subjects. COMPETING: T1 = T2 = T3 = T13 ~ 0.67 (EXACTLY equal — parameter-free), T4 = T5 = T6 = T14 ~ 0.33, T7 ~ 0.58, T8 ~ 0.42, T9 = T11 ~ 0.60, T10 = T12 ~ 0.40, T15 ~ 0.73, T16 0.50 — a decider-structured step function, bimodal across subjects. The profiles are structurally disjoint (e.g., competing says p(T1) = p(T3); advocated says p(T1) - p(T3) ~ +0.09; competing says p(T7) > 0.5; advocated says p(T7) < 0.45), so model comparison on the 16-cell response profile — plus the between-subject dispersion and within-subject consistency statistics — separates the theories even after fitting beta, epsilon, sigma_h, zeta, kappa, gamma, comp, and tie_break to each. The test is genuinely two-sided: a flat ladder plus primacy-consistent subject types falsifies the momentary-diffusion claim; a count-graded ladder plus a typeless, tight population falsifies the discrete position-mixture claim. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity (momentary-diffusion variant). People are one-reason decision makers: cues are consulted one at a time in descending subjective weight and the first strictly discriminating cue decides; no integration across cues (flat psychometric profile over tally margins). The population-central hierarchy is anti-validity (distrust-the-weakest, w = -validity). Each subject's weights are a continuous perturbation, w_j = -val_j + gamma*salience_j + A^3*(sigma_h*zeta_j + kappa*xi_j), where (i) salience_j is a positional gradient favoring early-listed experts with gamma drawn asymmetrically from [-0.05, +0.12] — a graded tilt, never a discrete primacy rule, bounded so it can flip nothing on the anchored designs that pin the anti-validity core; (ii) binding ambiguity A = 1 - |Spearman(position, validity)| enters CUBED, a steeply convex, gate-free gradient; and (iii) the idiosyncratic distortion is split into a small stable part (zeta_j, fixed per subject) and a large MOMENTARY part (xi_j, re-drawn every trial): under high binding ambiguity the consultation order is not merely idiosyncratic but diffuse — attention fluctuates from trial to trial, which keeps populations tight around chance-level conformity instead of spreading subjects into fixed opposing deciders. All distortion applies only when all advertised validities are distinct (exact ties anchor the distrust order). Noise is softmax(beta) over the binary winner score plus an independent lapse epsilon (implied follow p_f mean ~0.71, SD ~0.08); no discriminating cue yields exactly 0.5; validity ties are broken by a free per-subject tie-break (~80% early position); history is ignored.

**Parameters:**
- beta: [0.5, 1.9]
- epsilon: [0.08, 0.24]
- gamma: [-0.05, 0.12]
- sigma_h: [0.6, 1.2]
- kappa: [8, 12]
- zeta: [(-1, 1)] * n_features
- tie_break: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity
    # (momentary-diffusion variant).
    #
    # One-reason decision making: cues are consulted one at a time in
    # DESCENDING subjective weight and the first strictly discriminating
    # cue decides; no integration across cues (flat psychometric profile
    # over tally margins).
    #
    # Subjective weights (continuous, per subject):
    #     w_j = -val_j + gamma * salience_j
    #            + A^3 * (sigma_h * zeta_j + kappa * xi_j)
    #
    # - ANTI-VALIDITY CORE: -val_j (distrust the weakest advertised
    #   expert; consult ascending validity).
    # - POSITIONAL SALIENCE: salience_j = (n-1-j)/(n-1) - 1/2, favoring
    #   early-listed experts; gamma is drawn ASYMMETRICALLY from
    #   [-0.05, +0.12]: the negative clamp sits above the anchored-design
    #   flip threshold (-0.0625) and the positive cap below the primacy
    #   type threshold (+0.144), so the tilt remains a graded perturbation
    #   that can never manufacture spurious pure types while retaining a
    #   partial early-position lean on anchored designs.
    # - BINDING AMBIGUITY: A = 1 - |Spearman(display position, validity)|,
    #   now entering CUBED. The steeply convex gradient keeps
    #   low/moderate-ambiguity designs anchored (A^3 ~ 0.01-0.02) while
    #   fully diffusing high-ambiguity all-distinct designs (A^3 ~ 0.3-0.6).
    # - DIFFUSE HETEROGENEITY: the idiosyncratic distortion is split into
    #   (a) a SMALL STABLE part, sigma_h * zeta_j (zeta fixed per subject,
    #   zero-centered), and (b) a LARGE MOMENTARY part, kappa * xi_j,
    #   where xi is RE-DRAWN EVERY TRIAL. Momentary consultation diffusion
    #   is what makes high-ambiguity populations tight around chance-level
    #   conformity: with fixed per-subject orders, scrambling MAXIMIZES
    #   between-subject follow-rate dispersion, whereas the observed
    #   population SD (~0.064 on the conformity-dispersion experiment)
    #   sits barely above the binomial floor. Both parts are applied ONLY
    #   when all advertised validities are distinct (exact ties anchor the
    #   distrust order).
    # NO reversal re-encoding, NO sigmoid gates, NO discrete component
    # draws.
    #
    # Noise: softmax(beta) over the binary winner score mixed with an
    # independent lapse epsilon (implied follow probability
    # p_f = (1-eps)*sigmoid(beta) + eps/2, mean ~0.71, SD ~0.08).
    # No discriminating cue -> exactly 0.5. History is ignored.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float).ravel()
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    gamma = float(parameters["gamma"])
    sigma_h = float(parameters["sigma_h"])
    kappa = float(parameters["kappa"])
    tie_break = float(parameters["tie_break"])

    zeta = np.asarray(parameters["zeta"], dtype=float).ravel()
    if zeta.shape[0] != n_features:
        # NO silent zeros fallback: a shape mismatch means the entire
        # heterogeneity mechanism would be silently disabled, so fail
        # loudly instead (per the arbiter's diagnosis request).
        raise ValueError(
            f"zeta length {zeta.shape[0]} != n_features {n_features}."
        )

    # ---- helper: average-tie ranks ----
    def avg_ranks(x):
        n = x.shape[0]
        srt = np.argsort(x, kind="stable")
        ranks = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[srt[j + 1]] == x[srt[i]]:
                j += 1
            ranks[srt[i:j + 1]] = 0.5 * (i + j)  # 0-indexed average rank
            i = j + 1
        return ranks

    # ---- positional-salience gradient (early-listed experts favored) ----
    # Centered so the tilt is a pure graded perturbation: +0.5 at the
    # first listed expert, -0.5 at the last, linear in between.
    if n_features > 1:
        sal = ((n_features - 1 - np.arange(n_features, dtype=float))
               / (n_features - 1.0)) - 0.5
    else:
        sal = np.zeros(1, dtype=float)

    # ---- binding ambiguity: A = 1 - |Spearman(position, validity)| ----
    if n_features > 1:
        ranks = avg_ranks(val)
        pos_r = np.arange(n_features, dtype=float)
        vp = pos_r - pos_r.mean()
        vr = ranks - ranks.mean()
        denom = np.sqrt(float((vp * vp).sum()) * float((vr * vr).sum()))
        if denom > 1e-12:
            rho = float((vp * vr).sum()) / denom
        else:
            # All validities tied: distrust order undefined.
            rho = 0.0
        ambiguity = 1.0 - abs(rho)
    else:
        ambiguity = 0.0

    # ---- continuous subjective weights ----
    w = -val + gamma * sal

    # ---- diffuse idiosyncratic distortion (distinct lists only) ----
    # Exact validity ties anchor the distrust order (no distortion);
    # all-distinct lists degrade with the CUBE of binding ambiguity.
    # The distortion is split into a small STABLE per-subject part
    # (sigma_h * zeta) and a large MOMENTARY per-trial part (kappa * xi,
    # xi re-drawn fresh on every call), so that high-ambiguity designs
    # show tight, chance-centered conformity rather than fixed
    # idiosyncratic deciders that spread the population apart.
    distinct = (np.unique(val).shape[0] == n_features)
    if distinct and ambiguity > 0.0:
        amb3 = ambiguity * ambiguity * ambiguity
        xi = np.random.uniform(-1.0, 1.0, size=n_features)
        w = w + amb3 * (sigma_h * zeta + kappa * xi)

    # Consult cues in DESCENDING subjective weight. np.lexsort uses the
    # LAST key as primary. Exact ties in w are broken by the free
    # per-subject tie_break parameter (population-consistent: ~80% of
    # subjects break ties by early display position).
    pos = np.arange(n_features)
    secondary = pos if tie_break < 0.8 else (n_features - 1 - pos)
    cue_order = np.lexsort((secondary, -w))

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    n_opts = 2
    if winner is None:
        # No discriminating cue: pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta). Implied decider-follow
    # probability p_f = (1-eps)*sigmoid(beta) + eps/2 has mean ~0.71
    # with per-subject SD ~0.08 over the sampled (beta, epsilon) ranges.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / float(n_opts))
```

**`policy source code`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Attenuated Frugality with Position-Validity Mixture Heterogeneity. People are one-reason decision makers: cues are consulted one at a time and the first strictly discriminating cue decides, with no integration across cues (flat psychometric profile over tally margins, matching the margin-0 signatures of the knife-edge experiments). The subjective cue hierarchy is NOT universal: each subject draws a single consultation order from a graded mixture over a small candidate set: (1) anti-validity (distrust-the-weakest: consult the lowest advertised validity first) as the population-central tendency with a constant base weight; (2) position-primacy (consult experts in reading order, earliest-listed first); (3) position-recency (consult later-listed experts first, a minority variant); and (4) a residual standard-TTB component (descending validity). The position components' mixture weights grow smoothly and linearly with binding ambiguity A = 1 - |Spearman(display position, advertised validity)|, with no sigmoid gates of any kind: when the validity-to-position binding is unambiguous (monotone validity lists), the mixture collapses onto the anti-validity order (plus the small TTB residual), producing strong, consistent one-reason contrasts; when the binding is ambiguous (non-monotone lists), the position components enter, the pooled deciders disperse, and hierarchy-dissociation contrasts collapse toward chance. On top of the discrete mixture, a per-cue idiosyncratic weight distortion (zeta, fixed per subject, zero-centered) is applied, smoothly scaled by ambiguity times reversal-coherence and only when all advertised validities are distinct, so that all-distinct ambiguous designs degrade grain-by-grain while exact validity ties anchor the discrete rule mixture. Response discipline is attenuated and heterogeneous: the decider-follow probability p_f = (1-eps)*sigmoid(beta) + eps/2 spans roughly 0.55-0.85 across subjects with mean ~0.73 and SD ~0.06, so pooled contrasts are diluted both by mixture heterogeneity (disagreeing deciders) and by moderate, variable following. Validity ties are broken by a free per-subject tie-break; noise enters as softmax(beta) over the binary winner score plus an independent lapse epsilon; history is ignored (no feedback in this task).

**Parameters:**
- beta: [0.85, 1.8]
- epsilon: [0.10, 0.22]
- sigma_h: [0.4, 1.4]
- zeta: [(-1, 1)] * n_features
- tie_break: {0, 1}
- comp: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Attenuated Frugality with Position-Validity Mixture Heterogeneity.
    #
    # One-reason decision making: cues are consulted one at a time in
    # DESCENDING subjective weight and the first strictly discriminating
    # cue decides; no integration across cues (flat psychometric profile
    # over tally margins).
    #
    # Subjective hierarchy: each subject draws ONE consultation order from
    # a graded mixture over a small candidate set:
    #   - ANTI-VALIDITY (central):  w = -validity  (distrust the weakest
    #     advertised expert; consult ascending validity)
    #   - POSITION-PRIMACY:         w = -position  (reading order;
    #     earliest-listed expert first)
    #   - POSITION-RECENCY:         w = +position  (later-listed first;
    #     minority variant)
    #   - RESIDUAL TTB:             w = +validity  (descending validity)
    # The position components' weights grow SMOOTHLY and LINEARLY with
    # binding ambiguity
    #     A = 1 - |Spearman(display position, advertised validity)|
    # with no sigmoid gates: at A ~ 0 (monotone validity lists) the mixture
    # collapses onto anti-validity plus the small TTB residual; at high A
    # the position components enter and pooled deciders disperse, pulling
    # hierarchy-dissociation contrasts toward chance.
    #
    # Per-cue idiosyncratic distortion (zeta, zero-centered, fixed per
    # subject) is applied only when all advertised validities are DISTINCT,
    # smoothly scaled by ambiguity x reversal-coherence, so all-distinct
    # ambiguous designs degrade grain-by-grain while exact validity ties
    # anchor the discrete rule mixture.
    #
    # Noise: softmax(beta) over the binary winner score mixed with an
    # independent lapse epsilon. History is ignored (no feedback).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float).ravel()
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    sigma_h = float(parameters["sigma_h"])
    tie_break = float(parameters["tie_break"])
    comp = float(parameters["comp"])

    zeta = np.asarray(parameters["zeta"], dtype=float).ravel()
    if zeta.shape[0] != n_features:
        # Graceful fallback: no per-cue distortion.
        zeta = np.zeros(n_features)

    # ---- helper: average-tie ranks ----
    def avg_ranks(x):
        n = x.shape[0]
        srt = np.argsort(x, kind="stable")
        ranks = np.empty(n, dtype=float)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[srt[j + 1]] == x[srt[i]]:
                j += 1
            ranks[srt[i:j + 1]] = 0.5 * (i + j)  # 0-indexed average rank
            i = j + 1
        return ranks

    # ---- binding ambiguity: A = 1 - |Spearman(position, validity)| ----
    # 0 when the validity list is monotone in display position (the
    # validity-to-position binding is unambiguous -> the mixture
    # concentrates on the anti-validity distrust order); growing with
    # position-validity conflict (ambiguous binding -> position-anchored
    # reading orders enter the mixture).
    if n_features > 1:
        ranks = avg_ranks(val)
        pos_r = np.arange(n_features, dtype=float)
        vp = pos_r - pos_r.mean()
        vr = ranks - ranks.mean()
        denom = np.sqrt(float((vp * vp).sum()) * float((vr * vr).sum()))
        if denom > 1e-12:
            rho = float((vp * vr).sum()) / denom
        else:
            # All validities tied: distrust order undefined -> maximal
            # idiosyncrasy.
            rho = 0.0
        ambiguity = 1.0 - abs(rho)
    else:
        ambiguity = 0.0

    # ---- reversal coherence: |Spearman(validity, validity[::-1])| ----
    # Used ONLY as a smooth (gate-free) multiplier on the per-cue zeta
    # distortion: idiosyncratic scrambling is strongest when the binding is
    # ambiguous AND the list structure is reversal-coherent enough that
    # small perturbations reshuffle the subjective order.
    if n_features > 1:
        ranks_v = avg_ranks(val)
        ranks_r = avg_ranks(val[::-1])
        dv = ranks_v - ranks_v.mean()
        dr = ranks_r - ranks_r.mean()
        denom2 = np.sqrt(float((dv * dv).sum()) * float((dr * dr).sum()))
        if denom2 > 1e-12:
            rho_rev = float((dv * dr).sum()) / denom2
        else:
            rho_rev = 0.0
        coherence = abs(rho_rev)
    else:
        coherence = 0.0

    # ---- graded mixture weights over the candidate subjective orders ----
    # Smooth in ambiguity; NO sigmoid gates. The per-subject draw `comp` ~
    # U(0,1) selects one component via the cumulative normalized weights.
    w_anti = 0.45                  # anti-validity: constant central mass
    w_rec = 0.06 * ambiguity       # position-recency: minority variant
    w_prim = 0.80 * ambiguity      # position-primacy: reading-order default
    w_ttb = 0.04                   # residual standard TTB
    z_tot = w_anti + w_rec + w_prim + w_ttb
    c_anti = w_anti / z_tot
    c_rec = c_anti + w_rec / z_tot
    c_prim = c_rec + w_prim / z_tot

    if comp < c_anti:
        # Distrust-the-weakest: anti-validity subjective weights.
        w_base = -val
    elif comp < c_rec:
        # Position-recency: later-listed experts consulted first.
        w_base = np.arange(n_features, dtype=float)
    elif comp < c_prim:
        # Position-primacy: earlier-listed experts consulted first.
        w_base = -np.arange(n_features, dtype=float)
    else:
        # Residual standard TTB: descending advertised validity.
        w_base = val

    # ---- per-cue idiosyncratic distortion (distinct lists only) ----
    # Exact validity ties anchor the discrete rule mixture (no per-cue
    # noise); all-distinct lists degrade grain-by-grain, smoothly scaled
    # by ambiguity x reversal-coherence.
    distinct = (np.unique(val).shape[0] == n_features)
    zeta_scale = sigma_h * ambiguity * coherence if distinct else 0.0

    w = w_base + zeta_scale * zeta

    # Consult cues in DESCENDING subjective weight. np.lexsort uses the
    # LAST key as primary. Exact ties in w are broken by the free
    # per-subject tie_break parameter: 0 -> earlier feature position
    # first, 1 -> later feature position first among equally weighted cues.
    pos = np.arange(n_features)
    secondary = pos if tie_break < 0.5 else (n_features - 1 - pos)
    cue_order = np.lexsort((secondary, -w))

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    n_opts = 2
    if winner is None:
        # No discriminating cue: pure guess (lapse mixing leaves a
        # uniform unchanged).
        return np.ones(n_opts) / float(n_opts)

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    # Numerically stable softmax over the binary winner score: the
    # winner's probability is sigmoid(beta). Implied decider-follow
    # probability p = (1-eps)*sigmoid(beta) + eps/2 has mean ~0.73 with
    # per-subject SD ~0.06 over the sampled (beta, epsilon) ranges:
    # attenuated and variable, but high enough to reproduce the strong
    # margin-0 one-reason signatures of the monotone-validity designs.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / float(n_opts))
```

**`policy source code`:**
```python
def policy(probs):
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return np.random.choice(len(probabilities), p=probabilities)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Count-vs-Primacy planned contrast for one-reason binary-feature choice.\n    #\n    # For every trial we compute, directly from the rating vectors:\n    #   diff[j] = +1 if expert j rates A=1,B=0 (favors A), -1 if B=1,A=0,\n    #   n      = number of discriminating experts,\n    #   d      = sum(diff) = (#A-favoring cues) - (#B-favoring cues),\n    #   e      = diff of the EARLIEST (lowest-index) discriminating cue (+1 -> A).\n    #\n    # The trial weight is  w = d/n - e , i.e. the count-based choice\n    # direction minus the primacy-based choice direction, and the metric is\n    # the mean of w * y with y = +1 for choosing A, -1 for choosing B.\n    #\n    # Under the advocated momentary-diffusion theory the first consulted\n    # discriminating cue is ~uniform over the discriminators, so\n    # E[y] ~ s * d/n  (s = 2*p_f - 1) and the metric concentrates near\n    # +s * 0.26 ~ +0.11 (tight across subjects: only binomial noise plus a\n    # small s-spread; full-tie trials carry no weight and the single-cue\n    # control trial gets w = 0 automatically).\n    #\n    # Under the competing discrete position-mixture theory the dominant\n    # primacy component (mass ~0.54-0.59 at this design's ambiguity, with\n    # anti/recency/TTB minorities) makes E[y] ~ s * e on essentially every\n    # cell, so the metric is strongly negative (population ~ -0.28), with\n    # large between-subject spread from the fixed per-subject type draw.\n    # The contrast is validity-vector robust: the earliest-discriminator and\n    # count structures are fixed by the trial list, and the sign of the gap\n    # holds for either advertised-validity arrangement.\n    df = data\n    vals = []\n    a_col = df[\"option_a_ratings\"]\n    b_col = df[\"option_b_ratings\"]\n    r_col = df[\"response\"]\n    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):\n        a = np.asarray(a_raw, dtype=float)\n        b = np.asarray(b_raw, dtype=float)\n        if a.shape != b.shape or a.ndim != 1 or a.size == 0:\n            continue\n        diff = a - b\n        disc = np.where(diff != 0)[0]\n        n = disc.size\n        if n == 0:\n            # Full tie (no discriminating expert): no structural signal.\n            continue\n        d = float(diff.sum())            # count margin: A-cues minus B-cues\n        e = float(diff[disc[0]])        # earliest discriminator: +1 -> A, -1 -> B\n        w = d / float(n) - e            # count direction minus primacy direction\n        y = 1.0 if int(resp) == 0 else -1.0   # response 0 = chose A\n        vals.append(w * y)\n    if len(vals) == 0:\n        return 0.0\n    return float(np.mean(vals))",
  "rationale": "PLANNED CONTRAST: count-following minus primacy-following. For each trial I compute two one-reason choice directions that require no knowledge of the advertised validities: (i) the COUNT direction d/n (share of discriminating experts favoring A) and (ii) the PRIMACY direction e (the earliest-listed discriminating expert's favored option). The metric is mean over trials of (d/n - e)*y, y=+1 for choosing A.\n\nADVOCATED (momentary diffusion): the per-trial re-drawn kappa*xi noise (SD ~4-7 per cue at this design's ambiguity, vs a stable part <=~1.5) makes the first consulted discriminating cue ~uniform over the n discriminators, so E[y] = s*d/n with s=2p_f-1 ~ 0.42. Hence E[metric] = s * mean((d/n)(d/n - e)) = s * 3.889/15 ~ +0.11. Only the six count-unbalanced cells (T2,T3,T5,T6,T7,T8) carry expectation; the eight balanced cells and the tie contribute ~0. Per-subject spread is essentially binomial (SD ~0.11-0.12) plus a small s-spread -> between-subject variance ~0.015. The single-discriminator control T15 gets weight d/n - e = 0 automatically, and the full tie is skipped, so lapse/side-bias anchoring is built in.\n\nCOMPETING (discrete position mixture): its dominant component is position-primacy (mass 0.54-0.59 depending on which advertised-validity arrangement the simulator uses), with anti/recency/TTB minorities. A primacy follower has E[y] = s*e, giving per-subject metric -1.11*s ~ -0.51; recency +0.51; anti ~ -0.08 to -0.11; TTB ~ +0.27. Population mean ~ -0.28 to -0.30, between-subject variance ~0.09-0.11 (bimodal type spread). Every count-balanced cell contributes negatively in expectation because the primacy direction opposes the count direction there by construction of the ladder/mirror cells.\n\nSEPARATION: mean gap ~0.40 with self SD ~0.12 and competing SD ~0.32; Welch t at N=25 ~ 5.5-6, p ~ 1e-6, comfortably below alpha=0.01. The contrast is robust to (a) which of the two advertised validity vectors the simulator uses (earliest-discriminator and count structure depend only on the trial list; I verified the competing population mean is ~-0.28 under BOTH [0.85,0.95,0.65,0.75,0.6,0.9] and [0.55,0.95,0.78,0.86,0.62,0.70]), (b) per-subject p_f variation (metric scales ~linearly in s for both theories), and (c) the competing's zeta scrambling (it only dilutes the anti/recency minorities toward 0, leaving the dominant primacy mass intact). The metric works identically on the pooled DataFrame and on a single subject's 96-trial slice (it is a per-trial mean), so the between-subject variance is well defined. It is a genuine two-sided diagnostic: a graded count-ladder with a typeless tight population yields ~+0.11, a flat ladder with primacy-consistent subject types yields ~-0.28, and a random responder yields ~0."
}
```

## Usage

```json
{
  "input_tokens": 10098,
  "output_tokens": 35814,
  "reasoning_tokens": 34245,
  "provider": "Sail Research"
}
```
