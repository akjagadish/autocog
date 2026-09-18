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



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    hits = []\n    for a_raw, b_raw, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):\n        try:\n            a = np.asarray(list(a_raw), dtype=float).ravel()\n            b = np.asarray(list(b_raw), dtype=float).ravel()\n        except Exception:\n            continue\n        if a.size == 0 or a.size != b.size:\n            continue\n        d = a - b\n        nz = np.nonzero(d)[0]\n        nd = int(nz.size)\n        # sparse, NON-conflicting difference sets only: 1 or 2 discriminating\n        # cues, all favouring the same side (all flags on one option)\n        if nd < 1 or nd > 2:\n            continue\n        signs = np.sign(d[nz])\n        if not np.all(signs == signs[0]):\n            continue\n        try:\n            r = int(resp)\n        except Exception:\n            continue\n        if signs[0] > 0:\n            # A carries the extra 1s -> 'reversed'/fewer-marks choice is B (response 1)\n            hits.append(1.0 if r == 1 else 0.0)\n        else:\n            # B carries the extra 1s -> reversed choice is A (response 0)\n            hits.append(1.0 if r == 0 else 0.0)\n    if len(hits) == 0:\n        return 0.0\n    p = float(np.mean(hits))\n    return float(abs(p - 0.5))",
  "rationale": "Both theories are reversed-polarity accounts, so the SIGN of choices cannot separate them; what separates them is the DECISIVENESS on sparse, non-conflicting displays (one or two flags all on the same option, i.e. trial types T1, T2, T6, T8 of the design: 48 of the 96 trials).\n\nRFS-C+D is hard-capped there: a flag can only influence the choice if the issuing expert is actually consulted, and the absolute attention budget (sum q ~5.19 rescaled to C~3.1 across 8 experts) forces q_i <= ~0.50. A single 0.55-validity flag therefore moves p(reversed) only to ~.61, a single 0.97 flag to ~.75, two weak flags to ~.73, two strong flags to ~.86 -- an average of about .74, well below ceiling and steeply graded by consultation probability. RD-WEI has no consultation bottleneck: after divisive normalization a 1-vs-0 discrimination yields E' = w^(1-kappa) irrespective of panel size, putting all four of these cells at .84-.97, average ~.91.\n\nThe metric is the absolute deviation from chance of the fraction of 'fewer-marks' choices restricted to those sparse non-conflicting trials. Predicted values: ~0.24 (RFS-C+D) vs ~0.39-0.41 (RD-WEI). Taking the absolute deviation (rather than the raw proportion) makes the statistic invariant to RD-WEI's latent subject-level polarity frame (p_reverse < 1 occasionally yields a forward-frame subject), which would otherwise inject enormous between-subject variance into the competitor and kill the t-test. Restricting to 48 trials per subject keeps binomial noise small (sd ~.06), and both theories' parameter ranges leave their respective cells on the same side of the gap, so the contrast should be both large in mean and tight per subject.\n\nCrucially, this is not trivially true of my theory: it is a quantitative capacity-dilution prediction (sparse-trial decisiveness far below the ceiling fixed by the unanimity anchor), which a pure reversed integrator with any beta/epsilon cannot reproduce without also destroying its dominance-pair performance."
}
```

## Usage

```json
{
  "input_tokens": 12796,
  "output_tokens": 4105
}
```
