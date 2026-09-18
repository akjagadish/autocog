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



## ADVOCATED THEORY
**Description:** **RD-WEI: Relative (Normalized) Directional Weighted Evidence Integration.**

This keeps the two claims that carried the previous candidate and adds a third, which is the theoretical content of this revision.

(1) COMPRESSED, COMPENSATORY INTEGRATION. Subjects integrate ALL expert ratings. Each stated validity v_i is turned into normative evidence d_i = log(v_i/(1-v_i)), which enters the comparison after a strongly *compressive* transform w_i = d_i^gamma with gamma well below 1 (~0.2-0.5). The weight profile is therefore only mildly validity-ordered - much closer to a tally than to log-odds weighting - so no single expert can lexicographically out-weigh three or four others, yet validity still decides 1-vs-1 comparisons.

(2) SUBJECT-LEVEL POLARITY FRAME. The direction in which a positive binary rating pushes preference is a latent subject-level frame s in {-1,+1}: the decision variable is s * E. When the frame is inverted (s = -1, the overwhelming majority in these environments) a '1' is read as a mark/flag/non-selective endorsement and the subject prefers the option carrying the SMALLER validity-weighted count of positive ratings. Because in these designs the option with fewer marks is usually also the option favoured by the single most valid discriminating cue, reversed integration *masquerades* as one-reason (Take-The-Best) decision making on cue-conflict trials while producing strongly NEGATIVE weighted-margin contrasts on dominance/unanimity pairs - a signature no forward integrator and no lexicographic rule can produce.

(3) NEW: EVIDENCE IS READ IN RELATIVE, NOT ABSOLUTE, UNITS (divisive normalization). The margin is not evaluated in absolute log-odds units; it is evaluated as a *share of the evidence actually in play* on that trial. Only cues that discriminate enter the comparison, and the margin is divided by the total weight of those discriminating cues raised to a power kappa (~0.4-0.6):

    E' = s * [ sum_i w_i (a_i - b_i) ] / ( sum_{i: a_i != b_i} w_i )^kappa ,   P(A) = sigmoid(beta * E'), then lapse.

Psychologically this is a Weber-like normalization of the comparison: what governs confidence is how *lopsided* the discriminating evidence is, not how much of it there is. Two consequences distinguish RD-WEI from the un-normalized version: (a) a one-cue-versus-one-cue disagreement (few cues in play) is resolved almost as decisively as a 1-versus-5 rout, so single-cue and small-set discriminations stay sharp; (b) trials in which many cues discriminate but the split is nearly even (e.g. 4 marks vs 6 marks, or 2 vs 3 across a 7-cue array) become markedly noisier than their absolute margin would imply, and even unanimous/dominance pairs stop being perfectly deterministic once every cue is in play. This is exactly the graded-ness pattern the data show: near-ceiling reversed responding on lopsided splits, sub-ceiling responding on many-cue near-even splits, and sizeable between-subject disagreement precisely on the near-zero-share pairs.

The family still nests the classics: s=+1, gamma->0, kappa=0 = Tallying; s=+1, gamma=1, kappa=0 = Franklin's rule / naive Bayes; s=+1, gamma->inf = TTB; kappa=1 = pure average-diagnosticity-per-discriminating-cue; s=-1 gives the reversed-frame regime these environments demand.

**Parameters:**
- gamma: [0.18, 0.48]
- kappa: [0.42, 0.62]
- beta: [1.8, 2.5]
- epsilon: [0.0, 0.12]
- p_reverse: [0.94, 1.0]
- polarity_u: [0, 1]
- validities: validities

**`predict source code`:**
```python
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
```

**`policy source code`:**
```python
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
```


## COMPETING THEORY
**Description:** **RFS-C+D: Reversed Flag Screening under Limited Capacity, with a Duel (severity-ordered) Mode for Sparse Difference Sets.**

Claims (1)-(3) are unchanged from the accepted RFS-C base:

(1) POLARITY: a binary '1' is read as a warning mark against an option, not an endorsement; the option carrying FEWER encoded marks wins. This is the empirically robust reversal seen on dominance, single-cue and cue-conflict pairs alike.

(2) VALIDITY GATES *READING*, NOT *WEIGHTING*: expert i is consulted with probability q_i = q_max * (2v_i - 1)^rho; once consulted it contributes exactly ONE mark, no heavier than any other consulted expert.

(3) ATTENTION IS AN ABSOLUTE BUDGET: if sum_i q_i exceeds capacity C, every q_i is rescaled by C / sum_i q_i, so larger expert panels are noisier per expert.

NEW CLAIM (4) — MODE SWITCH BY DIFFERENCE-SET SPARSITY. Counting marks is what people do when the two products differ on MANY experts and the identities of the flagging experts are lost in the tally. But when the two displays differ on only a couple of experts, the comparison degenerates into a *duel between two named experts*, and the reader can then compare how DAMNING each flag is: the flag issued by the more credible expert is the more serious warning, so the reader rejects the option flagged by the highest-validity consulted discriminating expert (a reversed, consultation-limited Take-The-Best). The probability of engaging this duel mode falls off steeply — as a Gaussian in the number of discriminating experts beyond a two-expert duel, p_duel = omega * exp(-lam * (n_disc - 2)^2) — because holding individual expert identities in the comparison is only feasible for very sparse difference sets. For n_disc = 1 the two modes are behaviourally identical (both reject the lone flagged option, and both guess when the lone expert goes unread), and for two flags on the SAME option they are also nearly identical; the two modes diverge only on genuine sparse *conflicts* (one flag on each side) and, weakly, on three-cue displays.

This reconciles the two facts that no pure counter and no pure lexicographic rule can hold together: on many-cue conflicts (one flag on the best expert vs four or five flags on weaker experts) people go with the FEWER-mark option ~80% of the time (so counting, not rank, decides there), yet on a one-flag-versus-one-flag duel (.85-expert flags A, .78-expert flags B) they reject the option flagged by the MORE credible expert ~75-80% of the time (so severity, not count, decides there). The family still nests: omega -> 0 recovers pure reversed flag counting; lam -> 0 with omega -> 1 recovers a fully reversed one-reason rule; q_max -> 1, C -> infinity, beta -> infinity gives deterministic reversed tallying.

It remains cleanly separable from RD-WEI: RD-WEI predicts a share-driven graded tilt on top-heavy near-even splits and a flat single-cue ladder, whereas this account predicts near-chance on top-heavy near-even MANY-cue splits (counting is blind to cue identity), a steep single-cue ladder tracking consultation probability, panel-size-graded noise, and a sharp qualitative discontinuity between two-cue duels (rank-resolved) and four-plus-cue conflicts (count-resolved).

**Parameters:**
- rho: [0.25, 0.45]
- q_max: [0.78, 0.92]
- capacity: [2.7, 3.5]
- beta: [3.2, 5.2]
- epsilon: [0.0, 0.09]
- omega: [0.85, 1.0]
- lam: [1.0, 1.8]
- validities: validities

**`predict source code`:**
```python
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
```

**`policy source code`:**
```python
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # design validities (display order)\n    V = np.array([0.72, 0.95, 0.55, 0.90, 0.60, 0.80], dtype=float)\n\n    sc_n = 0; sc_k = 0        # single-cue trials (n_disc == 1)\n    du_n = 0; du_k = 0        # small-gap 1-vs-1 duels (n_disc == 2, one flag each side)\n\n    for a_raw, b_raw, resp in zip(data['option_a_ratings'],\n                                  data['option_b_ratings'],\n                                  data['response']):\n        try:\n            a = np.asarray(list(a_raw), dtype=float)\n            b = np.asarray(list(b_raw), dtype=float)\n        except Exception:\n            continue\n        if a.size != b.size or a.size == 0:\n            continue\n        n = a.size\n        if V.size >= n:\n            v = V[:n]\n        else:\n            v = np.linspace(0.90, 0.55, n)\n        diff = a - b\n        disc = np.nonzero(diff)[0]\n        try:\n            r = int(resp)\n        except Exception:\n            continue\n        if r not in (0, 1):\n            continue\n\n        if disc.size == 1:\n            j = int(disc[0])\n            # A carries the lone extra mark -> \"reversed\" (fewer-marks) choice is B\n            chose_unflagged = (r == 1) if diff[j] > 0 else (r == 0)\n            sc_n += 1\n            sc_k += 1 if chose_unflagged else 0\n\n        elif disc.size == 2 and (np.sum(diff[disc] > 0) == 1) and (np.sum(diff[disc] < 0) == 1):\n            vv = v[disc]\n            gap = float(abs(vv[0] - vv[1]))\n            if gap <= 0.20:   # near-equal validity duel (excludes the 0.95 vs 0.55 pair)\n                hi = int(disc[int(np.argmax(vv))])\n                # option flagged by the MORE credible expert is rejected?\n                rejected_hi = (r == 1) if diff[hi] > 0 else (r == 0)\n                du_n += 1\n                du_k += 1 if rejected_hi else 0\n\n    sc = (sc_k / sc_n) if sc_n > 0 else 0.5\n    du = (du_k / du_n) if du_n > 0 else 0.5\n    return float(sc - du)",
  "rationale": "Two orthogonal signatures that the two accounts cannot reparameterise into each other are combined into one contrast with opposite signs, so the gap adds while each component stays reasonably tight (30 trials each per subject).\n\n(1) SINGLE-CUE LEVEL (n_disc == 1, trial types TE/TF/TG). RD-WEI's divisive normalization means a lone discriminating cue is evaluated as a share of the (tiny) evidence in play, so E' = w^(1-kappa) is large even for the weakest expert: reversed responding is HIGH and nearly flat (~0.84-0.92, mean ~0.88). RFS-C+D reads the lone expert only with probability q_i under an absolute attention budget (q ~ 0.23-0.70 after capacity rescaling) and otherwise guesses, so its single-cue reversal rate is much LOWER (~0.65-0.83, mean ~0.72-0.76). The unanimous-dominance pairs in the design pin the lapse rate, so RD-WEI cannot be pushed down to that level without wrecking dominance performance.\n\n(2) NEAR-EQUAL-VALIDITY DUELS (n_disc == 2, one flag on each side, |v_hi - v_lo| <= 0.20; types TA/TB/TC, with the maximal-gap TD pair excluded because both theories agree there). RD-WEI resolves duels by the WEIGHT GAP; with compressed weights (0.95 vs 0.90 gives 1.43 vs 1.30) these trials are close to a coin flip, so the rate of rejecting the option flagged by the more credible expert is only ~0.55. RFS-C+D's duel mode is rank-lexicographic and magnitude-blind: the higher-validity discriminating expert is consulted first and its flag condemns that option, giving ~0.60-0.71.\n\nDifference score = single-cue reversal - duel 'reject-more-credible' rate. Predicted ~0.33 for RD-WEI vs ~0.09 for RFS-C+D, a ~0.24 gap, while per-subject values stay tight because each component averages 30 trials and both components are near-deterministic functions of the model parameters."
}
```

## Usage

```json
{
  "input_tokens": 13209,
  "output_tokens": 6910
}
```
