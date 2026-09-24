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
**Validities (n_features=6):** [0.8, 0.5, 0.85, 0.6, 0.9, 0.95]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 4: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 6: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 1]
  trial 7: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 8: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 11: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 12: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 13: A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 14: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  trial 15: A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]

**Rationale:** GOAL: Quantitatively dissociate Attenuated Frugality with Position-Validity Mixture Heterogeneity (advocated) from Anti-Validity Frugality with Diffuse Positional-Salience Heterogeneity, momentary-diffusion variant (competing). Both are one-reason models sharing the anti-validity core, so the dissociation must come from how their NON-central heterogeneity scales with binding ambiguity: the advocated enters ambiguity LINEARLY (primacy mass = 0.80*A/(0.49+0.86*A)), while the competing enters it CUBED (momentary diffusion amplitude = kappa*A^3) with a bounded gamma tilt that 'can flip nothing on anchored designs'. Experiment [14] tested the same theory pair only in the HIGH-ambiguity corner (A = 0.971, full diffusion vs full primacy). This design occupies the opposite, unexplored corner: MODERATE-LOW ambiguity A = 0.286, chosen as the point of maximal structural divergence — the advocated's linear term already commits 31.1% of subjects to pure position-primacy hierarchies, while the competing's own cubic gradient (A^3 = 0.023, exactly the 'anchored' regime its description cites) commits it to ~0% position-following: everyone is an anchored anti-validity decider with a bounded tilt that mathematically cannot flip any decider in this design.

SETUP: 6 features, all-distinct validities [0.80, 0.50, 0.85, 0.60, 0.90, 0.95] (f0..f5). Arrangement statistics (both models' own formulas): validity ranks by position [2, 0, 3, 1, 4, 5], Spearman(position, validity) = +0.714, so binding ambiguity A = 0.286; reversal coherence = 0.657. ADVOCATED mixture commitment (parameter-free, from its own formulas): anti-validity 0.612, position-primacy 0.311, TTB 0.054, recency 0.023. COMPETING commitment: A^3 = 0.023, so its momentary term kappa*A^3 <= 0.28 and stable term <= 0.028 are BELOW every cross-block validity gap used in the design (f1-f0 = 0.30, f1-f5 = 0.45, f2-f3 = 0.25, f0-f3 = 0.20), and its gamma tilt (max 0.12 * salience-gap, i.e. <= 0.024 for adjacent positions, <= 0.12 for extremes vs the 0.15 minimum strong-block gap) can flip NO decider: the competing is EXACTLY anchored — its decider is always the weakest-advertised discriminator, with flip probability < 1% on every trial. The advocated's own zeta blur (scale <= sigma_h*A*coherence = [0.075, 0.262]) never flips primacy/recency types (position gaps 1.0) and flips anti types on the key f0-vs-f1 cells only ~2-5% at typical sigma_h; all-distinct validities make tie_break inert in both models. DESIGN RULES: the weak block {f1, f3} and the strong block {f0, f2, f4, f5} are never internally opposed (small intra-block gaps would blur both models), except the f0/f5 oppositions in T5-T10 where only the 5.4% TTB mass is mildly blurred (~1.7% population effect); all dissociations pit the blocks against each other across large gaps.

THE FOUR HIERARCHIES: ANTI: f1(.50)->f3(.60)->f0(.80)->f2(.85)->f4(.90)->f5(.95). PRIMACY: f0->f1->f2->f3->f4->f5. RECENCY: f5->f4->f3->f2->f1->f0. TTB: f5->f4->f2->f0->f3->f1. The earliest-listed expert f0 (0.80) and the second-listed weakest expert f1 (0.50) are the core dissociating pair; f2 (pos 2, 0.85) vs f3 (pos 3, 0.60) provides an independent replication pair.

SIGNATURE 1 — THE PRIMACY-MASS KNIFE EDGE (parameter-free equality vs graded swing): T5 (f0->A, f1->B, f5->B) and T6 (f0->B, f1->B, f5->B) have the IDENTICAL competing decider (f1->B) — the competing predicts p(T5) = p(T6) EXACTLY, for any (beta, epsilon, gamma, sigma_h, kappa, zeta): the earliest discriminator's direction is structurally irrelevant to it. The advocated predicts p(T5) - p(T6) = m_prim*(2p_f-1) ~= +0.13 [0.06, 0.24] — on T5 its 31.1% primacy mass follows f0->A against the anti, recency, AND TTB components, against the strongest expert (0.95), and against the tally majority (margin -1). Mirrors: p(T8) - p(T9) = -0.13 (advocated) vs 0 (competing); also p(T1) = p(T6) exactly under the competing (same decider f1->B) vs a +0.16 gap under the advocated. Any reliable deviation from these equalities falsifies the competing's A^3 anchoring claim at any parameter setting; near-zero deviations falsify the advocated's point-predicted 0.311 primacy mass.

SIGNATURE 2 — THE SWING-RATIO POINT PREDICTION: the anti-flip contrast T7 - T5 (identical except f1's direction) measures the anti mass: advocated m_anti*(2p_f-1) ~= 0.26, competing (2p_f-1) ~= 0.42. Combined with Signature 1, the ratio of swings primacy-swing/anti-swing = m_prim/m_anti = 0.8A/0.45 = 0.509 is a PARAMETER-FREE point prediction of the advocated for THIS arrangement (derived entirely from its own mixture formula at A = 0.286), while the competing commits to a ratio of ~0. The two ratios (0.51 vs 0.00) are disjoint — a genuinely two-sided quantitative test of the linear-vs-cubic gradient assumption.

SIGNATURE 3 — GLOBAL PROFILE SHAPE: the competing emits a TWO-LEVEL STEP FUNCTION over the 16 cells: p_f (~0.71) on T2, T4, T7, T8, T9, T11, T13, T15 and 1-p_f (~0.29) on T1, T3, T5, T6, T10, T12, T14, exactly 0.5 on T16 — invariant to margin, coalition size, decider identity, and the earliest/strongest discriminators' directions. The advocated emits a MULTI-LEVEL profile (~0.42 on T5; ~0.44 on T1/T3/T12/T14; ~0.31 on T10; ~0.28 on T6; ~0.58 on T8; ~0.56 on T2/T4/T13/T15; ~0.69 on T7; p_f on T9/T11; 0.5 on T16), pulled toward 0.5 on every conflict cell by its 36.5% primacy+TTB mass. In particular the competing demands p(T5)=p(T1)=p(T6) and p(T8)=p(T9)=p(T7); the advocated demands strict orderings within each set. Model comparison on the 16-cell profile separates the theories even after fitting all free parameters to each.

SIGNATURE 4 — PER-SUBJECT TYPE CLASSIFICATION (sharpest test): 12 conflict cells x 6 repetitions = 72 classification trials per subject. The advocated predicts ~31% PRIMACY-consistent subjects (follow the earliest discriminator: A on T1/T3/T5/T7, B on T8/T10), ~61% anti-consistent (follow the weakest discriminator), ~5% TTB, ~2% recency. The competing predicts ~0% primacy-consistent subjects — its bounded gamma tilt and <1% diffusion cannot produce a subject who reliably follows f0 against f1+f5 on T5. T5/T8/T10 also cleanly separate primacy types from standard validity-respecting TTB strategists (a TTB strategist follows the 0.95 expert: B on T5, A on T8/T10 — the opposite of a primacy type). Observed primacy-consistent fraction ~0.3 supports the advocated's linear-ambiguity mixture; ~0 falsifies its 0.311 point commitment; > 0.1 falsifies the competing outright.

SIGNATURE 5 — SHARED FLATNESS + THREE-WAY SEPARATION FROM INTEGRATION: both theories predict p(T1) = p(T12) = p(T14) (same decider structure, margins 0/-1/0, anti decider at depth 1 vs depth 2) — any margin gradient falsifies the one-reason core of whichever model is otherwise supported, and Tallying (which predicts 0.5 / <0.5 / 0.5 across them) is separately falsified by any consistent preference on the margin-0 conflict cells. T11 (only f1 discriminates) and T9 (unanimous A) pin the follow rate p_f without knowing the mixture; T16 (full tie) anchors lapse and side bias. T2/T4/T8/T10/T13/T15 are exact A/B mirrors ruling out side bias.

HONESTY NOTE: on single cells the population prediction intervals overlap after parameter heterogeneity (both theories' major mass is anti-validity, and both mean p_f ~ 0.71-0.73); the decisive evidence is the paired within-subject contrasts (Signature 1-2), the profile shape (Signature 3), and the between-subject type distribution (Signature 4) — all robust to fitting beta, epsilon, gamma, sigma_h, kappa, zeta, comp, and tie_break to each theory. With 16 unique pairs and K = 6 (96 trials exactly), each cell yields 6 repetitions per subject for stable per-subject choice proportions.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Design-fixed advertised validities (positions f0..f5).\n    VALIDITIES = np.array([0.80, 0.50, 0.85, 0.60, 0.90, 0.95])\n    # A subject whose choice rate on primacy-vs-anti conflict trials exceeds\n    # this threshold is classified as a position-primacy (earliest-listed-\n    # expert) decider. 0.55 sits well above the anti-followers' ceiling\n    # (~0.27-0.36 incl. the competing theory's momentary diffusion) and well\n    # below the primacy-followers' level (~0.73).\n    THRESHOLD = 0.55\n    MIN_CONFLICT_TRIALS = 10\n\n    def to_arr(x):\n        if isinstance(x, str):\n            s = x.strip().strip(\"[]()\")\n            return np.array([float(v) for v in s.replace(\",\", \" \").split()])\n        return np.asarray(x, dtype=float).ravel()\n\n    df = data.reset_index(drop=True)\n    n = len(df)\n    if n == 0:\n        return 0.0\n\n    subj = df[\"subject_id\"].values\n    resp = np.asarray(df[\"response\"].values, dtype=float)\n    a_col = df[\"option_a_ratings\"].values\n    b_col = df[\"option_b_ratings\"].values\n\n    is_conflict = np.zeros(n, dtype=bool)\n    prim_dir = np.zeros(n, dtype=int)\n\n    for i in range(n):\n        a = to_arr(a_col[i])\n        b = to_arr(b_col[i])\n        if a.size != b.size or a.size != VALIDITIES.size:\n            continue\n        disc = a != b\n        if not disc.any():\n            continue\n        d_idx = np.where(disc)[0]\n        # Primacy decider: earliest-listed discriminating expert.\n        e = int(d_idx[0])\n        # Anti-validity decider: weakest advertised discriminating expert.\n        anti = int(d_idx[np.argmin(VALIDITIES[d_idx])])\n        pdir = 0 if a[e] > b[e] else 1\n        adir = 0 if a[anti] > b[anti] else 1\n        if pdir != adir:\n            is_conflict[i] = True\n            prim_dir[i] = pdir\n\n    subjects = pd.unique(df[\"subject_id\"])\n    flags = []\n    for s in subjects:\n        m = (subj == s) & is_conflict\n        cnt = int(m.sum())\n        if cnt < MIN_CONFLICT_TRIALS:\n            flags.append(0.0)\n            continue\n        rate = float(np.mean(resp[m] == prim_dir[m]))\n        flags.append(1.0 if rate > THRESHOLD else 0.0)\n\n    if not flags:\n        return 0.0\n    return float(np.mean(flags))",
  "rationale": "This metric implements Signature 4 (per-subject type classification), the sharpest dissociation in the design. On the 10 conflict trial types (T1-T5, T8, T12-T15; 60 observations per subject) the earliest-listed discriminating expert (f0 or f2) points to the OPPOSITE option from the weakest-advertised discriminating expert (f1 or f3). The metric computes, per subject, the rate of following the earliest-listed expert's direction on exactly those trials, and flags the subject as a position-primacy decider if that rate exceeds 0.55; the returned value is the population fraction of primacy-flagged subjects.\n\nPredicted values: (a) Advocated theory \u2014 its parameter-free mixture at this arrangement's ambiguity (A = 0.286) commits 31.1% of subjects to pure position-primacy orders (w_prim = 0.80*A normalized) plus 5.4% TTB subjects who agree with primacy on 8/10 conflict trials; primacy subjects follow f0/f2 with p_f \u2248 0.73 (never flipped: position weight gaps are 1.0 vs zeta blur \u2264 0.26), so they are flagged with probability \u2248 1, giving \u2248 0.36-0.38. (b) Competing theory \u2014 its ambiguity term enters cubed (A\u00b3 = 0.023), its stable distortion (\u2264 0.028 per cue) and gamma tilt (\u2264 0.06 on f0) are both below every cross-block validity gap (\u2265 0.20), so no subject can stably consult f0 before f1; only the momentary diffusion (kappa*A\u00b3 \u2264 0.28 vs a 0.30 gap) occasionally puts f0 first (~2-10% of trials depending on kappa), yielding a per-subject primacy-follow rate of \u2248 0.30-0.36 \u2014 far below 0.55 \u2014 so the flagged fraction is \u2248 0.01-0.03 (rare binomial overshoots only).\n\nWhy this form maximizes discriminability at N = 25: the advocated theory's heterogeneity is a DISCRETE mixture, so any continuous contrast (e.g., the pooled T5\u2212T6 swing, or the raw mean follow-rate) is bimodal per subject, carrying between-subject variance \u2248 0.05 (SD \u2248 0.22) against a mean gap of only ~0.12 against the competing theory \u2014 insufficient at N = 25. Thresholding collapses each subject to a near-noiseless Bernoulli indicator of their type (within-type SD \u2248 0.06 vs a 0.18-0.23 margin to the 0.55 cut), so the between-subject variance reduces to the irreducible binomial floor p(1\u2212p) \u2248 0.23 while the between-theory mean gap stays large (\u2248 0.35). Welch's t at N = 25 is then \u2248 3.5 (p \u2248 0.001-0.002), comfortably below alpha = 0.01. The metric is computed identically on the pooled data (fraction of flagged subjects) and on single-subject slices (0/1 flag), so the point estimate and between-subject variance are mutually consistent. Trials with no discriminating cue (T16) and trials where primacy and anti deciders agree (T6, T7, T9, T10, T11) are automatically excluded, so lapse/side-bias and shared one-reason flatness cannot contaminate the classification."
}
```

## Usage

```json
{
  "input_tokens": 9771,
  "output_tokens": 24616,
  "reasoning_tokens": 23109,
  "provider": "Sail Research"
}
```
