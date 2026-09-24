# metric_exp00_attempt_02

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
**Validities (n_features=14):** [1.0, 1.0, 1.0, 0.72, 0.7, 0.65, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]

**Trial pairs (n=24):**
  trial 1: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]
  trial 2: A=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
  trial 8: A=[1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
  trial 9: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 10: A=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
  trial 11: A=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 13: A=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
  trial 14: A=[0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  trial 19: A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
  trial 20: A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
  trial 21: A=[0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]
  trial 22: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  trial 23: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  trial 24: A=[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** STRUCTURAL ANALYSIS. RSD-vgp and SGAI-r7 share the identical anti-endorsement core: p(A) = 0.5 + A_eff*(sigmoid(beta*E) - 0.5), E = sum_j v_j^gamma*(b_j - a_j), RAW weights, nearly identical m and epsilon boxes, and per-trial polarity mixture. No single pooled cell can separate them. The theories differ ONLY in population architecture: (i) RSD's gamma is EXACTLY Uniform[0,2] (its two-component mixture has density 0.5 on both segments) INDEPENDENT of beta ~ logU[1.8, 6.0]; (ii) SGAI-r7 is a bimodal mixture in which gamma and beta are perfectly anti-correlated across clusters: an 80% high-precision cluster with a HARD habitual gamma floor of 0.8 (distribution 16%/[0.8,1.3], 38.4%/[1.3,1.7], 19.2%/[1.7,2.0], 6.4%/(2.0,2.2], none above 2.2) and beta >= 1.8, plus a 20% low-precision cluster with gamma ~ U[0,1], beta ~ logU[1.2, 1.8] and diffuse lapse 0.15-0.25, plus a per-trial stochastic attention gate (delta ~ 0.10, gamma_t jittered +/-0.4 around habitual gamma, capped at 2.2). Three consequences are decisive: (a) under RSD, gamma and beta are independent, so LOW-gamma subjects (gamma < 0.8, ~40% of the population) have beta ~ logU[1.8,6] -- STEEP; under SGAI, the ONLY subjects with gamma < 0.8 are the low-precision cluster, ALL with beta <= 1.8 -- SHALLOW. The quadrant {gamma-hat < 0.78, beta-hat > 1.8} holds ~39% of subjects under RSD and is a HARD ZERO under SGAI; the quadrant {beta-hat < 1.74} holds ~18% under SGAI and is a HARD ZERO under RSD. (b) SGAI's gamma distribution stochastically dominates RSD's everywhere (mean 1.34 vs 1.0; density 0.20 below 0.8, 0.32 on [0.8,1.3], 0.96 on [1.3,1.7], 0.64 on [1.7,2.0], 0.32 on (2.0,2.2] vs RSD's flat 0.5), so a flip ladder predicts HIGHER inversion rates at every rung under SGAI. (c) SGAI places 6.4% habitual mass above RSD's hard gamma ceiling of 2.0, plus delta-gate leakage up to 2.2. This design is therefore built around a JOINT (gamma-hat, beta-hat) per-subject readout -- a signature no prior design in this series has exploited -- with a flip ladder whose rungs are placed LOW (0.52, 0.78), where the two populations differ most, plus a gamma-invariant exact-evidence ladder that reads out beta cleanly.

ARCHITECTURE (14 experts): three unit cues at validity 1.00 (power weight EXACTLY 1 for every gamma in [0, 2.2], for both theories, immune even to SGAI's per-trial gamma_t jitter) and eleven low/mid cues (0.72, 0.70, 0.65, 0.60 x3, 0.55 x2, 0.50 x3) used to build flip cells with designed sign-change points. Unanimous total weight S(gamma) >= 6.87 for all gamma <= 2.2, so the unanimous cells saturate every subject of both theories (beta*S >= 1.2*6.87 = 8.2), reading out A_eff = 2*(p_sparse - 0.5).

DISSOCIATION 1 -- FLIP LADDER AT LOW GAMMA (headline population contrast). Each flip cell endorses A on unit cues (weight exactly u for all gamma) and B on a low-validity block with W(gamma) strictly decreasing; E(gamma) = W(gamma) - u has a single sign change at g*, so subjects with gamma < g* prefer the count-sparser unit-cue side and gamma > g* prefer the low-validity-denser side. Flip points: FL1 g* = 0.52 (u=3 vs {0.65, 0.60, 0.55, 0.50}); FL2 g* = 0.78 (u=3 vs {0.55 x2, 0.50 x3}, count margin 2, strong evidence magnitudes E in [-1.65, +2]); FL3 g* = 1.42 (u=2 vs {0.65, 0.60 x3}); FL4 g* = 1.73 (u=2 vs {0.72, 0.70, 0.65, 0.60}); FL5 g* = 2.02 (u=1 vs {0.72, 0.70}). Predicted population inversion rates P(gamma > g*): RSD 0.74 / 0.61 / 0.29 / 0.135 / 0.00 vs SGAI 0.90 / 0.84 / 0.53 / 0.24 / ~0.01 -- SGAI higher at EVERY rung (stochastic dominance). Predicted pooled p(unit side) at mid-box: FL1 0.36 vs 0.25; FL2 0.44 vs 0.24; FL3 0.57 vs 0.40; FL4 0.65 vs 0.48; FL5 0.68 vs 0.50. The survival-curve slope is flat -0.5 under RSD but piecewise under SGAI (only -0.20 below 0.8, -0.96 on [1.3,1.7]): the FL1->FL2 drop (0.13 RSD vs 0.05 SGAI) isolates SGAI's depleted low-gamma region, and the FL4->FL5 stretch tests the 2.0 ceiling.

DISSOCIATION 2 -- THE CONDITIONAL SLOPE TEST (the hard categorical weapon). B1/B2/B3 use only unit cues with exactly canceling shared endorsements on mid cues, giving E = -1, -2, -3 EXACTLY for every subject at every gamma (and every gamma_t). The per-subject ratio r12 = (p_sparse,E2 - 0.5)/(p_sparse,E1 - 0.5) = (sigmoid(2*beta)-0.5)/(sigmoid(beta)-0.5) is exactly A_eff-free and lapse-free and reads out beta alone. Supports: RSD r12 in [1.003, 1.322] (beta in [1.8, 6]); SGAI high-prec the same; SGAI low-prec r12 in [1.322, 1.549] (beta in [1.2, 1.8]). THE TEST: among subjects classified low-gamma by the flip ladder (unit-side preference on FL2, gamma-hat < 0.78), RSD predicts ~80% with r12 < 1.25 (beta > 2.28, steep) because gamma and beta are independent; SGAI predicts EXACTLY 0% -- every low-gamma SGAI subject is low-precision with r12 >= 1.322. Conversely, any subject with r12-hat > 1.35 (beta < 1.74) falsifies RSD categorically (hard beta floor 1.8) while SGAI requires ~18% of subjects there. In the 2D (gamma-hat, beta-hat) plane, RSD predicts a FILLED RECTANGLE [0,2] x [1.8,6] (uniform-in-gamma vertical band, no mass below beta 1.8, hard edge at gamma 2.0); SGAI predicts an L-SHAPE (low-gamma subjects only below beta 1.8; steep subjects only right of gamma 0.8) plus a 6.4% sliver past gamma 2.0.

DISSOCIATION 3 -- THE 2.0 CEILING. On FL5 (g* = 2.02), E(2.0) = 0.49 + 0.5184 - 1 = +0.0084 > 0 and E is strictly decreasing, so since m >= 0.74 > 0.5 makes p >= 0.5 whenever E >= 0, NO RSD subject can prefer the block side on FL5 -- while SGAI's (2.0, 2.2] tail (6.4%) plus delta-gate leakage (habitual 1.7-2.0 subjects drawing gamma_t up to 2.2 on 10% of trials) predicts ~1% population inversion rate and, more usefully, a lower pooled p(unit side) (0.50 vs 0.68). Acknowledged: per-subject tail inversions are shallow (E(2.2) = -0.058) -- this weapon is population-level and one-sided; the joint (gamma, beta) structure is the arbiter.

DISSOCIATION 4 -- DELTA-GATE INCONSISTENCY (tertiary). Both theories carry the per-trial m-mixture, which adds response inconsistency UNIFORMLY across all cells. SGAI's delta gate adds inconsistency ONLY on flip cells near a subject's habitual gamma. Test: per-subject excess trial-to-trial inconsistency (above each subject's own fitted m), regressed on |gamma-hat - g*| across the five flip cells: concentrated at near-flip cells (SGAI) vs uniform (RSD).

CONTROLS. U/Um (unanimous) saturate and read out A_eff (RSD p_U in [0.714, 0.985] vs SGAI [0.725, 0.966]; any subject credibly above 0.97 falsifies SGAI, which caps m at 0.98 -- a weak secondary signature). Uc/Ucm are conflict-style saturated twins (11 endorsements vs 3; |E| >= 5.4), verifying the A_eff anchor is not an empty-bar artifact. C/Cm is a 2-2 endorsement-count tie (A on two unit cues vs B on {0.55, 0.50}): E = 0.55^gamma + 0.50^gamma - 2 < 0 for all gamma > 0, so both theories prefer the low-validity-endorsed side (pooled ~0.81 RSD / ~0.78 SGAI) while pure counting predicts exactly 50% -- ruling out count-based responding. T1/T2 are identical-vector ties where both theories predict exactly 50% (key-bias check). Every informative cell has a mirrored twin (A/B swapped) so side bias cancels; analysis pools by unit-cue side (flip cells), sparse side (B-cells, U), or low-validity side (C). All informative trials carry nonempty, distinct rating vectors on both bars (locked-prompt requirement).

POWER AND ANALYSIS. 24 unique pairs x K = 4 = 96 trials exactly; per subject 40 flip observations (8 per pooled mirrored rung), 24 exact-E observations, 16 saturated observations, 8 count-tie, 8 ties. Pooled per-cell differences are substantial (FL2: 0.44 vs 0.24) but the decisive evidence is SUBJECT-LEVEL: requires N >= 80 subjects with hierarchical Bayesian fitting of per-subject (gamma, beta, A_eff) under each theory's population prior, plus simulation-based model recovery and simple robust cross-checks (e.g., subjects with FL2 unit-rate >= 6/8 classified low-gamma; their r12 distribution tested against the disjoint supports).

PREDICTED OUTCOMES. If RSD-vgp is true: inversion survival curve 0.74/0.61/0.29/0.135/0 across g* = 0.52/0.78/1.42/1.73/2.02 with constant slope 0.5 and a hard zero past 2.0; gamma-hat histogram flat on [0,2] with ~40% of subjects below 0.8; NO subject with r12-hat > 1.35 (hard beta floor); ~30% of subjects in the quadrant {gamma-hat < 0.78, beta-hat > 2.2}; among low-gamma subjects, ~80% steep (r12 < 1.25); FL5 pooled p(unit) ~0.68 with zero inversions. If SGAI-r7 is true: inversion survival 0.90/0.84/0.53/0.24/~0.01, higher at every rung, with a visibly flat segment below 0.8 (density 0.20) and a steep drop across [1.3,1.7]; gamma-hat histogram depleted below 0.8 (16%), bulging on [1.3,1.7] (38%), with ~6% of gamma-hat mass above 2.0 and a hard edge at 2.2; ~18-20% of subjects with r12-hat > 1.35 (bimodal beta population straddling 1.8) and high fitted lapse; ZERO subjects with gamma-hat < 0.8 AND beta-hat > 1.8 (the low-gamma population is entirely shallow); near-flip-concentrated excess inconsistency. Each theory is thus categorically falsified by a population region the other fills: RSD dies if any low-gamma subject is steep... rather, RSD dies if any subject has beta-hat < 1.8 or gamma-hat > 2.05, and -- the sharpest single test -- if the low-gamma subpopulation is not predominantly steep, RSD's independence assumption is refuted; SGAI dies if any low-gamma subject is steep, if no ~20% shallow cluster exists, or if any subject shows gamma-hat > 2.2.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Raw-Evidence Sparse Dominance with Validity-Gated Precision (RSD-vgp). When choosing between two products described by binary expert ratings, subjects read an endorsement (rating = 1) as evidence AGAINST the option carrying it, so the count-sparser option is perceived as higher quality. Within a trial, ALL cues are integrated additively with RAW (unnormalized) validity weights: the evidence for A over B is E = sum_j v_j^gamma (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical vectors give exactly 50%. The population is a continuum of graded polarity (no bimodal trait clusters): each subject carries a per-trial inverted-polarity probability m, a lapse epsilon, and — the new discriminative dimension — validity-gated precision: a per-subject validity exponent gamma dispersed over [0, 2] with a substantial steep-gamma subpopulation, and a steep log-uniformly dispersed raw inverse temperature beta over [1.8, 6.0], calibrated so that essentially every subject's effective normalized psychometric slope beta * sum(v^gamma) exceeds 9.5. The resulting subject family is 'steep slope x moderate asymptote': p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2, with effective attenuation A_eff = (1-eps)(2m-1) concentrated near 0.69. This decomposition is what the data jointly demand: the slope-sensitive diagnostics (psychometric slope posterior, evidence-usage slopes, mixture log-likelihood ratios) require steep normalized slopes, while the extremity/consistency diagnostics (saturated sparse-choice rates) require moderate asymptotic preference — a combination no single shallow-slope, high-extremity or steep-slope, high-extremity parameterization can produce.

**Parameters:**
- gamma_u: [0, 1]
- log_beta: [0.588, 1.792]
- epsilon: [0.03, 0.11]
- m: [0.74, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Raw-Evidence Sparse Dominance with Validity-Gated Precision (RSD-vgp).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"RSD-vgp expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # ---- Validity gating: per-subject gamma via a two-component mixture ----
    # gamma_u ~ U[0, 1]. With probability 0.65 the subject draws gamma
    # uniformly from [0, 1.3] (near-flat validity weighting); with
    # probability 0.35 from [1.3, 2.0] (the steep-gating subpopulation
    # that validity-conflict cells discriminate).
    u = float(parameters["gamma_u"])
    if u < 0.65:
        gamma = 1.3 * (u / 0.65)
    else:
        gamma = 1.3 + 0.7 * ((u - 0.65) / 0.35)

    # ---- Precision: log-uniform raw inverse temperature ----
    # beta = exp(log_beta), log_beta ~ U[ln 1.8, ln 6.0]  =>  beta
    # log-uniform over [1.8, 6.0]. Calibrated against the RAW evidence
    # scale below so that every subject's effective normalized slope
    # beta * sum(v^gamma) comfortably exceeds 9.5 on any plausible
    # validity vector (min ~ 1.8 * 7.4 ~ 13).
    beta = float(np.exp(float(parameters["log_beta"])))
    epsilon = float(parameters["epsilon"])
    m = float(parameters["m"])

    # RAW validity weighting: w_j = v_j^gamma, NO normalization.
    # This preserves the absolute endorsement-margin scale: a margin-2
    # conflict carries twice the evidence of a margin-1 conflict,
    # independent of n_features (the cross-experiment scale separation
    # that per-experiment normalization provably erases).
    w = np.power(val, gamma)

    a, b = stim[0], stim[1]
    # Anti-endorsement (sparse-preferring) evidence for A over B:
    # each endorsement carried by B counts FOR A, each endorsement
    # carried by A counts AGAINST A. E = 0 on identical rating vectors.
    E = float(np.sum(w * (b - a)))

    # Numerically stable logistic (both branches exponentiate a
    # non-positive argument, so no overflow is possible).
    x = beta * E
    if x >= 0.0:
        sig = 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        sig = ex / (1.0 + ex)

    # Graded polarity: with per-trial probability m the subject responds
    # with the INVERTED (sparse-preferring) polarity, with probability
    # (1 - m) with the conventional polarity. Continuous, no empty
    # middle, no mirror cluster.
    p_core = m * sig + (1.0 - m) * (1.0 - sig)

    # Independent lapse to uniform choice. Ties (E = 0) give exactly 50%.
    p_a = (1.0 - epsilon) * p_core + 0.5 * epsilon

    return np.array([p_a, 1.0 - p_a])
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)
```


## COMPETING THEORY
**Description:** Stochastic-Gate Anti-Endorsement Integration, revision 7 (SGAI-r7). Subjects reading a product described by binary expert ratings treat an endorsement (rating = 1) as evidence AGAINST the option carrying it (a 'fewer red flags' reading), so the count-sparser option is perceived as higher quality. Within a trial, all cues are integrated additively with RAW (unnormalized) validity weights: the evidence for A over B is E = sum_j v_j^gamma_t (b_j - a_j), so the absolute endorsement-count margin sets the evidence scale independent of feature count, and identical vectors give exactly 50%. Choice follows p(A) = (1-eps)*[m*sigmoid(beta*E) + (1-m)*(1-sigmoid(beta*E))] + eps/2. The population is bimodal in precision: a dominant high-precision cluster (80%; beta log-uniform on [1.8, 6.0] — the floor restored to the RSD boundary so the majority's beta support no longer spills below BOTH theory boxes of the Exp 13 grid, which is what flipped the RSD-vs-EPAI log Bayes factor negative in r6) whose habitual validity exponent is distributed 20% in [0.8, 1.3], 48% in [1.3, 1.7], 24% in [1.7, 2.0], and an 8% sliver in (2.0, 2.2], none above 2.2 — plus a 20% low-precision cluster (beta log-uniform on [1.2, 1.8], habitual gamma uniform on [0, 1.0], diffuse lapse 0.15-0.25 restored per two converging data points that the diffuse setting maximizes its GSP-ward Bayes-factor pull). On each trial, with probability delta ~ 0.10, the subject re-samples cue attention, drawing an effective per-trial exponent from a symmetric distribution around their OWN habitual gamma (gamma_t ~ U[max(0, gamma-0.4), min(2.2, gamma+0.4)]). Falsifiable signatures: psychometric mid-slopes shallower than saturated asymptotes imply, supra-binomial trial-to-trial inconsistency on repeated identical trials, a bimodal precision population straddling the RSD/GSP beta boundary at 1.8, a mildly positive RSD-vs-EPAI log Bayes factor, and per-subject gamma posteriors that spill above 2.0 for the steep-gamma bins without any habitual mass above 2.2.

**Parameters:**
- gamma_u: [0, 1]
- log_beta: [0.588, 1.792]
- epsilon: [0.03, 0.10]
- m: [0.80, 0.98]
- delta: [0.10, 0.12]
- prec_u: [0, 1]
- lp_log_beta: [0.182, 0.588]
- lp_epsilon: [0.15, 0.25]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Stochastic-Gate Anti-Endorsement Integration, revision 7 (SGAI-r7).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SGAI-r7 expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    # ---- Population: habitual validity exponent gamma (majority) ----
    # gamma_u ~ U[0, 1]. The confirmed anchor distribution (the only
    # setting that has ever put Exp 11 in band while keeping Exp 13 in
    # band): 20% in [0.8, 1.3], 48% in [1.3, 1.7], 24% in [1.7, 2.0],
    # and an 8% sliver in (2.0, 2.2]. NO habitual mass above 2.2
    # (guardrail for Exp 13).
    u = float(parameters["gamma_u"])
    if u < 0.20:
        gamma = 0.8 + 0.5 * (u / 0.20)
    elif u < 0.68:
        gamma = 1.3 + 0.4 * ((u - 0.20) / 0.48)
    elif u < 0.92:
        gamma = 1.7 + 0.3 * ((u - 0.68) / 0.24)
    else:
        gamma = 2.0 + 0.2 * ((u - 0.92) / 0.08)

    # ---- Precision subpopulation (bimodal) ----
    # 20% of subjects are low-precision: beta log-uniform on [1.2, 1.8]
    # (deep inside the GSP box, below the RSD floor of 1.8), DIFFUSE
    # lapse epsilon in [0.15, 0.25] (restored: two converging data
    # points -- iter 3 and iter 5/6 -- show the diffuse setting
    # maximizes this cluster's GSP-ward pull on the Exp 12 tanh BF,
    # while every sharpening regressed it), and habitual gamma forced
    # LOW (uniform on [0, 1.0]). The remaining 80% draw beta
    # log-uniformly over [1.8, 6.0]: the r6 floor of 1.6 sat below
    # BOTH theory boxes of the Exp 13 grid (RSD starts at 1.8, EPAI at
    # 2.0), flipping the RSD-vs-EPAI log Bayes factor negative
    # (-1.88 vs observed +6.03). Restoring the floor to the RSD
    # boundary 1.8 (the midpoint between the in-band iter-4 floor 2.0
    # and the sign-flipping iter-6 floor 1.6) returns the majority to
    # RSD-identified territory while still leaving the lower-gamma
    # majority bins (gamma <= 1.5) with beta in [1.8, 3.2] inside the
    # GSP overlap, preserving part of r6's Exp 1/2 gains.
    if float(parameters["prec_u"]) < 0.20:
        beta = float(np.exp(float(parameters["lp_log_beta"])))
        epsilon = float(parameters["lp_epsilon"])
        gamma = 1.0 * u  # low-precision subjects keep gamma <= 1.0
    else:
        beta = float(np.exp(float(parameters["log_beta"])))
        epsilon = float(parameters["epsilon"])

    m = float(parameters["m"])
    delta = float(parameters["delta"])

    # ---- Trial-level stochastic attention gate (RELATIVE draw) ----
    # With probability delta (~0.10) the subject re-samples cue
    # attention for THIS trial, drawing an effective exponent from a
    # SYMMETRIC distribution AROUND THEIR OWN habitual gamma with a
    # +/-0.4 width. The draw is relative, so the low-precision
    # cluster's shallow gamma is never globally overridden, while
    # steep-gamma subjects still jitter across the 2.0 boundary on a
    # minority of trials. Confirmed dead lever for Exp 14; kept only
    # in its gentlest, least harmful form.
    if np.random.random() < delta:
        lo = max(0.0, gamma - 0.4)
        hi = min(2.2, gamma + 0.4)
        gamma_t = lo + (hi - lo) * np.random.random()
    else:
        gamma_t = gamma

    # RAW validity weighting: w_j = v_j^gamma_t, NO normalization.
    # This preserves the absolute endorsement-margin scale: a margin-2
    # conflict carries twice the evidence of a margin-1 conflict,
    # independent of n_features.
    w = np.power(val, gamma_t)

    a, b = stim[0], stim[1]
    # Anti-endorsement (sparse-preferring) evidence for A over B:
    # each endorsement carried by B counts FOR A, each endorsement
    # carried by A counts AGAINST A. E = 0 on identical rating vectors.
    E = float(np.sum(w * (b - a)))

    # Numerically stable logistic (both branches exponentiate a
    # non-positive argument, so no overflow is possible).
    x = beta * E
    if x >= 0.0:
        sig = 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        sig = ex / (1.0 + ex)

    # Graded polarity: with per-trial probability m the subject responds
    # with the INVERTED (sparse-preferring) polarity, with probability
    # (1 - m) with the conventional polarity.
    p_core = m * sig + (1.0 - m) * (1.0 - sig)

    # Independent lapse to uniform choice. Ties (E = 0) give exactly 50%.
    p_a = (1.0 - epsilon) * p_core + 0.5 * epsilon

    return np.array([p_a, 1.0 - p_a])
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)
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
[0] rationale: The two theories share an identical anti-endorsement core p(A)=0.5+A_eff*(sigmoid(beta*E)-0.5), E=sum_j v_j^gamma*(b_j-a_j), and nearly identical m/epsilon boxes, so no single pooled cell rate separates them (all single-cell mean gaps are <=0.18 with per-subject SDs ~0.25). They differ ONLY in population architecture: (i) RSD draws gamma and beta independently (gamma uniform-density on [0,2], beta logU[1.8,6]), so its ~30-40% low-gamma subjects are STEEP; (ii) SGAI-r7's only low-gamma subjects are its 20% low-precision cluster (beta logU[1.2,1.8], gamma U[0,1], lapse 0.15-0.25), and its high-precision cluster has a hard habitual gamma floor of 0.8. The metric therefore targets the JOINT (gamma,beta) structure, which no previous round exploited.

Implementation: for each subject I fit the shared core model p(A)=0.5+a*(sigmoid(beta*E_gamma)-0.5) by grid MLE (gamma in [0,2.4], beta log-spaced [0.8,6], a in [0.05,0.99]) over all 96 trials — this model is exactly correct in expectation under BOTH theories, so the fit is theory-neutral. From the fit I form the smoothed quadrant contrast J = Phi((0.75-gamma_hat)/0.35) * (2*Phi((beta_hat-1.85)/0.35) - 1): positive for low-gamma STEEP subjects (RSD-unique, ~30-40% of RSD, a hard zero under SGAI), negative for low-gamma SHALLOW subjects (SGAI's low-precision cluster, impossible under RSD), ~0 for high-gamma subjects (where the theories agree). The beta leg works even though beta is only weakly identified for steep subjects, because steep subjects' combination of high E=1 rates with moderate saturated rates cannot be reproduced by beta<1.8 without violating a<=1. To this I add 0.8*M, the block-side (low-validity-endorsed side) choice rate over the strong validity-conflict trials (trials 1,2,3,4,14,15,16,17; 32 obs), which reads the gamma-marginal difference (RSD ~0.55 vs SGAI ~0.40) and is nearly uncorrelated with J, so the combination roughly adds the two gaps in signal-to-noise. metric(data) returns the mean of per-subject scores, so the pooled point estimate is exactly the population mean of the per-subject statistic and the between-subject variance is the variance of the same quantity.

Predicted values: under RSD-vgp, point estimate ~0.65-0.70 (between-subject SD ~0.35): a large low-gamma-steep subpopulation fills the positive quadrant and the flip ladder shows frequent block-side (low-gamma) preferences. Under SGAI-r7, point estimate ~0.30-0.35 (between-subject SD ~0.19): the low-gamma subpopulation is entirely shallow (negative J contribution) and the gamma histogram is depleted below 0.8, lowering M. Expected Welch t at N=25 is ~4 (p<0.001), with comfortable margin under pessimistic assumptions about estimator noise (t>=2.8 in sensitivity checks varying the gamma-hat standard error from 0.1 to 0.35 and the shallow-cluster beta-hat mean from 1.5 to 1.9). The statistic is not trivially true for RSD: it fails if RSD's low-gamma subjects were shallow or if its beta floor were violated, and it degrades gracefully to the flip-ladder marginal contrast if the precision readout is noisy.
metric_source:
import numpy as np
import pandas as pd
from math import log, erf, sqrt

VALIDITIES = np.array([1.0, 1.0, 1.0, 0.72, 0.70, 0.65,
                       0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60])
UNIT_MASK = VALIDITIES > 0.99
LOW_MASK = ~UNIT_MASK


def _parse_vec(x):
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=float)
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=float)
    s = str(x).strip()
    try:
        import ast
        return np.asarray(ast.literal_eval(s), dtype=float)
    except Exception:
        vals = [float(ch) for ch in s if ch in "01"]
        return np.asarray(vals, dtype=float)


def _sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _phi(z):
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _subject_score(A, B, y):
    """A, B: (n,14) rating matrices; y: (n,) with 1 = chose B.
    Returns the per-subject discriminating score."""
    # ---------- per-subject MLE of (gamma, beta, a) ----------
    # Shared anti-endorsement core of both theories:
    #   p(A) = 0.5 + a * (sigmoid(beta * E) - 0.5),
    #   E = sum_j v_j^gamma * (B_j - A_j),  a in (0,1].
    gammas = np.linspace(0.0, 2.4, 49)
    betas = np.exp(np.linspace(np.log(0.8), np.log(6.0), 25))
    alphas = np.linspace(0.05, 0.99, 25)

    diff = B - A                                  # (n,14)
    Wg = VALIDITIES[None, :] ** gammas[:, None]   # (49,14)
    D = diff @ Wg.T                               # (n,49): raw evidence for A

    yb = y[:, None]                               # (n,1)
    best_ll = -np.inf
    best_g, best_b = 1.0, 3.0
    for beta in betas:
        gmat = _sigmoid(beta * D) - 0.5           # (n,49)
        # p(A) for every alpha -> (25, n, 49)
        pA = 0.5 + alphas[:, None, None] * gmat[None, :, :]
        np.clip(pA, 1e-9, 1.0 - 1e-9, out=pA)
        ll = ((1.0 - yb) * np.log(pA) + yb * np.log(1.0 - pA)).sum(axis=1)  # (25,49)
        idx = np.unravel_index(np.argmax(ll), ll.shape)
        if ll[idx] > best_ll:
            best_ll = ll[idx]
            best_g = gammas[idx[1]]
            best_b = beta

    # ---------- joint (gamma, beta) population contrast ----------
    # RSD-vgp: gamma and beta are independent, so LOW-gamma subjects are
    # still STEEP (beta >= 1.8).  SGAI-r7: every low-gamma subject belongs
    # to the low-precision cluster (beta <= 1.8, high lapse).  The
    # smoothed product reads out "low gamma AND steep" minus
    # "low gamma AND shallow".
    f_low = _phi((0.75 - best_g) / 0.35)          # P(subject is low-gamma)
    f_steep = _phi((best_b - 1.85) / 0.35)        # P(subject is steep)
    J = f_low * (2.0 * f_steep - 1.0)

    # ---------- flip-ladder block-side rate (gamma-marginal contrast) ----------
    # Validity-conflict trials: one side endorsed only on unit cues,
    # the other only on low-validity cues (>=3 of them).
    a_unit = (A[:, UNIT_MASK].sum(axis=1) > 0) & (A[:, LOW_MASK].sum(axis=1) == 0)
    b_unit = (B[:, UNIT_MASK].sum(axis=1) > 0) & (B[:, LOW_MASK].sum(axis=1) == 0)
    a_low = (A[:, LOW_MASK].sum(axis=1) >= 3) & (A[:, UNIT_MASK].sum(axis=1) == 0)
    b_low = (B[:, LOW_MASK].sum(axis=1) >= 3) & (B[:, UNIT_MASK].sum(axis=1) == 0)
    sel = (a_unit & b_low) | (a_low & b_unit)
    if sel.sum() > 0:
        block_is_b = (a_unit & b_low)[sel]
        ysel = y[sel]
        chose_block = np.where(block_is_b, ysel, 1.0 - ysel)
        M = float(chose_block.mean())
    else:
        M = 0.5

    return J + 0.8 * M


def metric(data: pd.DataFrame) -> float:
    df = data
    A = np.vstack([_parse_vec(v) for v in df["option_a_ratings"]])
    B = np.vstack([_parse_vec(v) for v in df["option_b_ratings"]])
    y = np.asarray(df["response"].to_numpy(), dtype=float)
    subs = df["subject_id"].to_numpy()
    scores = []
    for s in pd.unique(subs):
        m = subs == s
        scores.append(_subject_score(A[m], B[m], y[m]))
    return float(np.mean(scores))
outcome: self_sim=0.5843 (var=0.1458) adversary_sim=0.4953 (var=0.0767) welch_t=+0.943 p=0.3509 (N=25, alpha=0.01) -> reject

[1] rationale: The two theories share an identical anti-endorsement core p(A)=0.5+a*(sigmoid(beta*E_gamma)-0.5), E=sum_j v_j^gamma*(b_j-a_j), so no single pooled cell rate separates them (all single-cell mean gaps <=0.16 with per-subject SDs ~0.2). My previous attempt estimated per-subject (gamma_hat, beta_hat) by grid MLE and formed a hard quadrant indicator; the point estimates were too noisy, so the indicator fired equally under both theories (t=0.94). The fix is to use the information OPTIMALLY: instead of plugging noisy point estimates into a discontinuous indicator, integrate the likelihood over each theory's full population prior. The metric is therefore the per-subject marginal log-likelihood ratio (log Bayes factor) of RSD-vgp vs SGAI-r7, tanh-compressed and averaged over subjects. Both marginal likelihoods are computed by exact grid quadrature over each theory's true parameter boxes (gamma: 65%/35% two-segment uniform for RSD vs the 80% four-segment high-precision distribution + 20% U[0,1] low-precision cluster for SGAI; beta log-uniform on [1.8,6] vs cluster-specific [1.2,1.8]/[1.8,6]; a=(1-eps)(2m-1) induced by each theory's (eps,m) boxes; SGAI's per-trial delta~0.11 attention gate modeled as the correct per-trial mixture over gamma_t in [gamma-0.4, gamma+0.4]). Why this discriminates: (1) the theories differ in the JOINT population architecture, and the Bayes factor is a function of the subject's whole 96-trial response profile, which localizes gamma via the flip ladder (rungs at gamma=0.56, 1.0, 1.38, 1.75, 2.02), a via the saturated cells, and the gamma-free exact-evidence ladder (E=+/-1,2,3 on unit cues) constrains beta. (2) RSD's low-gamma subjects (~30-40% of its population, gamma<0.6) are steep (beta logU[1.8,6], median 3.3); SGAI's only low-gamma subjects are the low-precision cluster with a hard beta<=1.8 cap and a<=0.816. On mid-evidence cells (|E| in [0.45,1.3]) the steep subject's deviations from 0.5 are ~0.31 while the best low-precision fit can reach only ~0.27 while simultaneously overshooting the saturated cells — an irreducible KL of ~1.5-2 nats; moreover RSD's likelihood is nearly flat in beta over [2.2,6] (prior mass 0.83) whereas SGAI's low-precision support [1.2,1.8] lies entirely outside that region. Each such subject therefore contributes a strongly positive log-BF (~+2), and they are a hard zero under SGAI. (3) Conversely, SGAI's 20% low-precision cluster (shallow, high lapse, gamma in [0,1]) is accommodated by RSD only at the beta=1.8 boundary with systematic misfit on the E=2/E=3/saturated cells, contributing ~-0.8 per such subject — impossible under RSD. (4) The gamma-prior mismatch alone contributes the symmetrized KL between the two gamma distributions (~0.35 nats/subject in the gap). Expected values: mean tanh(logBF) ~ +0.3 under RSD-vgp (SD ~0.55) and ~ -0.35 under SGAI-r7 (SD ~0.4), giving Welch t ~ 3.5-4 at N=25. The tanh compression is essential: it bounds the heavy-tailed log-BF (a few subjects contribute +3 or more), which caps the between-subject variance while preserving the mean gap, and it also compresses per-subject sampling noise (the log-BF of a strongly discriminated subject fluctuates about +-1 nat across data replications). The metric is a symmetric model-comparison statistic — it would favor SGAI on SGAI-generated data — so it is not trivially true for my theory; it is simply the likelihood-based test of the one dimension (the joint gamma-beta population structure) on which the two theories make disjoint predictions. It is well defined on any single subject's slice (returns that subject's log-BF), so the pooled point estimate is the population mean and the between-subject variance is the variance of the same per-subject quantity.
metric_source:
import numpy as np
import pandas as pd

V = np.array([1.0, 1.0, 1.0, 0.72, 0.70, 0.65,
              0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60])

GAM = np.arange(0.0, 2.2 + 1e-9, 0.025)
DGAM = 0.025
LNB = np.linspace(np.log(1.2), np.log(6.0), 19)
BET = np.exp(LNB)
DELTA = 0.11
NJIT = 5


def _parse_vec(x):
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=float)
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=float)
    s = str(x).strip()
    try:
        import ast
        v = np.asarray(ast.literal_eval(s), dtype=float)
        if v.size == V.size:
            return v
    except Exception:
        pass
    vals = [float(ch) for ch in s if ch in "01"]
    return np.asarray(vals, dtype=float)


def _sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _logu_beta_weights(lnb, lo, hi):
    d = (lnb[1] - lnb[0]) / 2.0
    L = hi - lo
    w = np.zeros(lnb.shape[0])
    for i in range(lnb.shape[0]):
        a = max(lnb[i] - d, lo)
        b = min(lnb[i] + d, hi)
        if b > a:
            w[i] = (b - a) / L
    return w


W_BET_R = _logu_beta_weights(LNB, np.log(1.8), np.log(6.0))
W_BET_SL = _logu_beta_weights(LNB, np.log(1.2), np.log(1.8))


def _rsd_gdens(g):
    return 0.5 if (0.0 <= g <= 2.0) else 0.0


def _sgai_high_gdens(g):
    if 0.8 <= g < 1.3:
        return 0.2 / 0.5
    if 1.3 <= g < 1.7:
        return 0.48 / 0.4
    if 1.7 <= g < 2.0:
        return 0.24 / 0.3
    if 2.0 <= g <= 2.2:
        return 0.08 / 0.2
    return 0.0


def _sgai_low_gdens(g):
    return 1.0 if (0.0 <= g <= 1.0) else 0.0


def _gamma_weights(dens):
    w = np.zeros(GAM.shape[0])
    for i in range(GAM.shape[0]):
        pts = np.linspace(GAM[i] - DGAM / 2.0, GAM[i] + DGAM / 2.0, 7)
        vals = np.array([dens(p) for p in pts])
        w[i] = float(vals.mean()) * DGAM
    return w


W_GAM_R = _gamma_weights(_rsd_gdens)
W_GAM_SH = 0.8 * _gamma_weights(_sgai_high_gdens)
W_GAM_SL = 0.2 * _gamma_weights(_sgai_low_gdens)


def _a_grid(eps_lo, eps_hi, m_lo, m_hi, ne, nm):
    eps = np.linspace(eps_lo, eps_hi, ne)
    ms = np.linspace(m_lo, m_hi, nm)
    a = np.array([(1.0 - e) * (2.0 * m - 1.0) for e in eps for m in ms])
    w = np.full(a.shape[0], 1.0 / a.shape[0])
    return a, w


A_R, WA_R = _a_grid(0.03, 0.11, 0.74, 1.00, 7, 9)
A_SH, WA_SH = _a_grid(0.03, 0.10, 0.80, 0.98, 6, 8)
A_SL, WA_SL = _a_grid(0.15, 0.25, 0.80, 0.98, 7, 8)

_CACHE = {}


def _build_cache(D):
    key = np.ascontiguousarray(D, dtype=np.float64).tobytes()
    c = _CACHE.get(key)
    if c is not None:
        return c
    T = D.shape[0]
    G = GAM.shape[0]
    Wg = V[None, :] ** GAM[:, None]
    E = Wg @ D.T
    JG = np.empty((G, NJIT))
    for i in range(G):
        lo = max(0.0, GAM[i] - 0.4)
        hi = min(2.2, GAM[i] + 0.4)
        JG[i] = np.linspace(lo, hi, NJIT)
    Wj = V[None, :] ** JG.ravel()[:, None]
    Ej = (Wj @ D.T).reshape(G, NJIT, T).mean(axis=1)

    configs = (
        ("R", A_R, WA_R, W_GAM_R, W_BET_R, False),
        ("SH", A_SH, WA_SH, W_GAM_SH, W_BET_R, True),
        ("SL", A_SL, WA_SL, W_GAM_SL, W_BET_SL, True),
    )
    c = {}
    for name, a, wa, wg, wb, jit in configs:
        LP = [None] * BET.shape[0]
        LM = [None] * BET.shape[0]
        for bi in range(BET.shape[0]):
            if wb[bi] <= 0.0:
                continue
            if jit:
                sig = (1.0 - DELTA) * _sigmoid(BET[bi] * E) + DELTA * _sigmoid(BET[bi] * Ej)
            else:
                sig = _sigmoid(BET[bi] * E)
            s = sig - 0.5
            P = 0.5 + a[:, None, None] * s[None, :, :]
            np.clip(P, 1e-12, 1.0 - 1e-12, out=P)
            LP[bi] = np.log(P)
            LM[bi] = np.log1p(-P)
        c["LP_" + name] = LP
        c["LM_" + name] = LM
        c["lwa_" + name] = np.log(np.maximum(wa, 1e-300))
        c["lwg_" + name] = np.log(np.maximum(wg, 1e-300))
        c["lwb_" + name] = np.log(np.maximum(wb, 1e-300))
    _CACHE.clear()
    _CACHE[key] = c
    return c


def _logml(nA, nB, c, name):
    LP = c["LP_" + name]
    LM = c["LM_" + name]
    lwa = c["lwa_" + name]
    lwg = c["lwg_" + name]
    lwb = c["lwb_" + name]
    parts = []
    for bi in range(BET.shape[0]):
        if LP[bi] is None:
            continue
        ll = (np.tensordot(LP[bi], nA, axes=([2], [0])) +
              np.tensordot(LM[bi], nB, axes=([2], [0])))
        ll = ll + lwa[:, None] + lwg[None, :] + lwb[bi]
        parts.append(ll.ravel())
    allp = np.concatenate(parts)
    m = float(allp.max())
    return m + float(np.log(np.exp(allp - m).sum()))


def _subject_score(nA, nB, c):
    lr = _logml(nA, nB, c, "R")
    lsh = _logml(nA, nB, c, "SH")
    lsl = _logml(nA, nB, c, "SL")
    ls = np.logaddexp(np.log(0.8) + lsh, np.log(0.2) + lsl)
    return float(lr - ls)


def metric(data: pd.DataFrame) -> float:
    df = data
    sub = df["subject_id"].to_numpy()
    Av = [_parse_vec(v) for v in df["option_a_ratings"]]
    Bv = [_parse_vec(v) for v in df["option_b_ratings"]]
    r = np.asarray(df["response"].to_numpy(), dtype=float)

    keys = []
    for a, b in zip(Av, Bv):
        keys.append((np.asarray(a, dtype=np.int64).tobytes(),
                     np.asarray(b, dtype=np.int64).tobytes()))
    uniq = {}
    for k, a, b in zip(keys, Av, Bv):
        if k not in uniq:
            uniq[k] = (np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    tlist = list(uniq.keys())
    tidx = {k: i for i, k in enumerate(tlist)}
    D = np.vstack([uniq[k][1] - uniq[k][0] for k in tlist])
    T = D.shape[0]
    if T == 0:
        return 0.0

    c = _build_cache(D)

    scores = []
    for s in pd.unique(sub):
        m = sub == s
        idx = np.where(m)[0]
        nA = np.zeros(T)
        nB = np.zeros(T)
        for i in idx:
            j = tidx[keys[i]]
            if r[i] < 0.5:
                nA[j] += 1.0
            else:
                nB[j] += 1.0
        scores.append(_subject_score(nA, nB, c))
    return float(np.mean(np.tanh(np.asarray(scores))))
outcome: self_sim=0.2634 (var=0.3906) adversary_sim=-0.0921 (var=0.2394) welch_t=+2.240 p=0.03005 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "import numpy as np\nimport pandas as pd\n\nV = np.array([1.0, 1.0, 1.0, 0.72, 0.70, 0.65,\n              0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60, 0.60])\n\nGAM = np.round(np.arange(0.0, 2.4001, 0.05), 4)          # 49 gamma points\nBET = np.exp(np.linspace(np.log(1.2), np.log(6.0), 19))  # union beta grid\nALP = np.concatenate([np.array([0.35, 0.40]), np.linspace(0.43, 0.97, 12)])\n\n# Cluster masks (theory boxes):\n#   steep cluster (RSD):  beta ~ logU[1.8, 6], a = (1-eps)(2m-1) in [0.427, 0.97]\n#   shallow cluster (SGAI low-precision): beta ~ logU[1.2, 1.8],\n#       a in [0.45, 0.816]\nSTEEP_B = BET >= 1.79\nSHAL_B = BET <= 1.81\nSTEEP_A = ALP >= 0.425\nSHAL_A = (ALP >= 0.445) & (ALP <= 0.825)\n\nMASK_ST = np.outer(STEEP_B, STEEP_A).reshape(-1)\nMASK_SH = np.outer(SHAL_B, SHAL_A).reshape(-1)\n\n\ndef _parse_vec(x):\n    if isinstance(x, np.ndarray):\n        return np.asarray(x, dtype=float)\n    if isinstance(x, (list, tuple)):\n        return np.asarray(x, dtype=float)\n    s = str(x).strip()\n    try:\n        import ast\n        v = np.asarray(ast.literal_eval(s), dtype=float)\n        if v.size == V.size:\n            return v\n    except Exception:\n        pass\n    vals = [float(ch) for ch in s if ch in \"01\"]\n    return np.asarray(vals, dtype=float)\n\n\ndef _sigmoid(z):\n    out = np.empty_like(z, dtype=float)\n    pos = z >= 0\n    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))\n    ez = np.exp(z[~pos])\n    out[~pos] = ez / (1.0 + ez)\n    return out\n\n\ndef _logsumexp(z):\n    m = float(np.max(z))\n    if not np.isfinite(m):\n        return -np.inf\n    return m + float(np.log(np.sum(np.exp(z - m))))\n\n\ndef _subject_score(A, B, y):\n    \"\"\"A, B: (n,14) rating matrices; y: (n,) with 1 = chose B.\"\"\"\n    Dint = (B - A).astype(np.int64)\n    n = Dint.shape[0]\n    keys = [row.tobytes() for row in Dint]\n    index = {}\n    Dlist = []\n    nAs = []\n    nBs = []\n    for i in range(n):\n        k = keys[i]\n        if k not in index:\n            index[k] = len(Dlist)\n            Dlist.append((B[i] - A[i]).astype(float))\n            nAs.append(0.0)\n            nBs.append(0.0)\n        j = index[k]\n        if y[i] < 0.5:\n            nAs[j] += 1.0\n        else:\n            nBs[j] += 1.0\n    Ds = np.vstack(Dlist)\n    nA = np.asarray(nAs, dtype=float)\n    nB = np.asarray(nBs, dtype=float)\n\n    # Raw-evidence core: E(gamma) = sum_j v_j^gamma * (B_j - A_j)\n    W = V[None, :] ** GAM[:, None]                 # (G,14)\n    E = W @ Ds.T                                   # (G,T)\n    X = BET[None, :, None] * E[:, None, :]         # (G,B,T)\n    DEV = _sigmoid(X) - 0.5                        # (G,B,T)\n\n    # p(A) = 0.5 + a * dev  for every (gamma, beta, a)\n    P = 0.5 + ALP[None, None, :, None] * DEV[:, :, None, :]   # (G,B,Aa,T)\n    np.clip(P, 1e-9, 1.0 - 1e-9, out=P)\n    LP = np.log(P)\n    LQ = np.log1p(-P)\n    ll = (np.tensordot(LP, nA, axes=([3], [0])) +\n          np.tensordot(LQ, nB, axes=([3], [0])))   # (G,B,Aa)\n\n    # ---- L: nuisance-marginalized posterior mass at gamma < 0.8 ----\n    ll2 = ll.reshape(GAM.shape[0], -1)\n    lw = np.array([_logsumexp(r) for r in ll2])\n    w = np.exp(lw - np.max(lw))\n    L = float(np.sum(w[GAM < 0.8]) / np.sum(w))\n\n    # ---- Lambda: steep-cluster vs shallow-cluster log-likelihood ratio\n    #      evaluated at the subject's gamma-hat ----\n    g_hat = int(np.argmax(lw))\n    row = ll[g_hat].reshape(-1)\n    ls = _logsumexp(row[MASK_ST])\n    lh = _logsumexp(row[MASK_SH])\n    lam = ls - lh\n\n    return L * float(np.tanh(lam / 1.4))\n\n\ndef metric(data: pd.DataFrame) -> float:\n    df = data\n    A = np.vstack([_parse_vec(v) for v in df[\"option_a_ratings\"]])\n    B = np.vstack([_parse_vec(v) for v in df[\"option_b_ratings\"]])\n    y = np.asarray(df[\"response\"].to_numpy(), dtype=float)\n    subs = df[\"subject_id\"].to_numpy()\n    scores = []\n    for s in pd.unique(subs):\n        m = subs == s\n        scores.append(_subject_score(A[m], B[m], y[m]))\n    return float(np.mean(scores))",
  "rationale": "THE DISPUTE. Both theories share the anti-endorsement core p(A)=0.5+a*(sigmoid(beta*E_gamma)-0.5), E=sum_j v_j^gamma*(b_j-a_j), so no pooled cell rate separates them. They differ ONLY in the JOINT population architecture: under RSD-vgp, gamma has flat density 0.5 on [0,2] INDEPENDENT of beta~logU[1.8,6] and eps<=0.11, so its ~40% of subjects with gamma<0.8 are STEEP; under SGAI-r7 the ONLY subjects with gamma<0.8 are the 20% low-precision cluster (beta~logU[1.2,1.8], eps 0.15-0.25, hence attenuation a=(1-eps)(2m-1)<=0.82), while its 80% high-precision cluster has a hard habitual gamma floor of 0.8. So the quadrant {gamma<0.8 AND steep} holds ~40% of subjects under RSD and is a HARD ZERO under SGAI, and the quadrant {gamma<0.8 AND shallow} holds 20% under SGAI and is a hard zero under RSD. Every other subject (high-gamma, steep) exists under both theories with nearly identical behavior.\n\nWHY THE PREVIOUS ATTEMPTS FAILED. Attempt 1 plugged noisy per-subject (gamma_hat, beta_hat) point estimates into a hard quadrant indicator: misclassification noise erased the gap (t=0.94). Attempt 2 used the global per-subject log Bayes factor: the ~60-80% of subjects on which the theories agree contributed pure variance (tanh(logBF) spread, var 0.39/0.24), and the gamma-prior-ratio contribution I counted on washed out because gamma is only localized to +/-0.2-0.3, so the observed gap (0.355) carried too little of the true quadrant signal (t=2.24).\n\nTHE METRIC. Per subject I fit the shared core by grid (gamma in [0,2.4], beta log-spaced [1.2,6], a in [0.35,0.97]) -- a theory-neutral model exactly correct in expectation under both theories -- and form the product s = L * tanh(Lambda/1.4), where (i) L = posterior mass at gamma<0.8, obtained by marginalizing the likelihood over (beta,a) with flat nuisance priors (the flip ladder -- FL-A flips at gamma~0.58, FL-B at ~1.45/1.7, count-tie and FL-C above -- identifies gamma from 48 gamma-sensitive trials, SD(gamma_hat)~0.15-0.25); (ii) Lambda = log-likelihood ratio between the STEEP cluster (beta logU[1.8,6], a in [0.43,0.97]) and the SHALLOW cluster (beta logU[1.2,1.8], a in [0.45,0.82]) evaluated at gamma_hat, i.e. the two clusters' true parameter boxes. The metric is the mean of s over subjects, so the pooled point estimate is the population mean of a per-subject score and the between-subject variance is the variance of the same quantity.\n\nWHY IT SEPARATES. The gating by L zeroes out every subject on which the theories agree (high-gamma subjects of both theories get L~0.04 and contribute ~0), removing the variance that killed the global Bayes factor, while retaining both disjoint quadrants as signal of OPPOSITE sign: RSD's low-gamma subjects (40%) are steep, get L~0.9 and Lambda~+1 (the shallow cluster's beta<=1.8 forces an E-ladder curvature -- relative steepening from E=1 to E=3 to the saturated cells -- that no steep parameterization reproduces once the 16 saturated trials pin a, and its a<=0.82 ceiling cannot match high-a steep subjects' saturated sparse-choice rates), giving s~+0.45; SGAI's low-precision subjects (20%) are inside the shallow box and outside the steep box, get L~0.7 and Lambda~-0.6, giving s~-0.25. Unlike attempt 1, no hard thresholding is done: both L and tanh(Lambda) are continuous, bounded, and use the full 96-trial profile, so misclassification costs gap rather than adding Bernoulli noise. Unlike attempt 2, the gamma-prior terms (which washed out) are excluded by conditioning on gamma_hat, and the agreeing majority is gated out. Expected values: RSD-vgp point ~ +0.18 to +0.20 (between-subject var ~0.08); SGAI-r7 point ~ -0.03 (var ~0.03); Welch t ~ 3.0-3.4 at N=25, p<0.005. The statistic is symmetric -- it is a two-cluster likelihood ratio, not a rigged test -- and would strongly favor SGAI on SGAI-generated data (the low-precision cluster yields distinctly negative scores), so it is not trivially true for my theory; it is simply the likelihood-based readout of the one population region (low-gamma x steep vs low-gamma x shallow) in which the two theories make disjoint predictions."
}
```

## Usage

```json
{
  "input_tokens": 16341,
  "output_tokens": 125985,
  "reasoning_tokens": 123278,
  "provider": "Sail Research"
}
```
