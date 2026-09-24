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
**Validities (n_features=8):** [0.99, 0.99, 0.75, 0.75, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  trial 4: A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  trial 6: A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  trial 8: A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  trial 9: A=[1, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  trial 10: A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0, 0, 0]
  trial 11: A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  trial 13: A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  trial 14: A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  trial 15: A=[0, 0, 0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 1, 0]
  trial 16: A=[0, 0, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 1, 0, 0]
  trial 17: A=[0, 0, 0, 0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0, 1]
  trial 18: A=[0, 0, 0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 1, 0]
  trial 19: A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[0, 0, 0, 1, 0, 0, 0, 0]
  trial 20: A=[0, 0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1, 1]
  trial 21: A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0, 0]
  trial 22: A=[0, 0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 0, 0]
  trial 23: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 24: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]

**Rationale:** CORE LOGIC. Both theories share the identical architecture (sign votes x weights x softmax x lapse) and differ ONLY in the weight function: competitor w_j = v_j^gamma (gamma in [0.05, 0.7]); mine w_j = max(log(v_j/(1-v_j)) - tau, 0)^delta (delta in [0.1, 0.2], tau in [0, 0.04]). With no chance-level cues, the only identifiable quantity is the EXCHANGE RATE ER(v_hi, v_lo) = w(v_hi)/w(v_lo) between validity classes, measurable behaviorally from conflict trials. The design engineers validity classes so that the two theories' ER boxes are DISJOINT. Validities: [0.99, 0.99, 0.75, 0.75, 0.51, 0.51, 0.51, 0.51] -- two super-experts, two mid experts, four barely-diagnostic 51% experts (all cues are genuinely diagnostic; NO v=0.5 cues, so this is a pure test of the weighting FORM among diagnostic cues, unlike Exp [4]'s chance-annihilation design, and unlike [2]/[3]'s count-vs-weight decorrelations). The key arithmetic: llr(0.99)=4.595 vs llr(0.51)=0.040, a 115x log-odds gap, so even the heaviest delta-compression (delta=0.1, tau=0) leaves my theory a minimum super-premium ER(0.99,0.51) = (4.595/0.040)^0.1 = 1.61, while the competitor's maximum is (0.99/0.51)^0.7 = 1.59 at its steepest corner (gamma=0.7). DISJOINT INTERVALS: competitor ER(99,51) in [1.03, 1.59], mine in [1.61, ~2.6+] (center delta=0.15, tau=0.01: 2.13). Likewise ER(0.75, 0.51): competitor [1.02, 1.31], mine [1.39, 1.94] -- disjoint with a wider margin. Psychologically: does one 99% expert outweigh TWO 51% experts (my theory: yes, ~2 weak experts per super) or barely one (competitor: at most 1.59, i.e., two weak experts always beat one super)?

BETA-FREE RATIO STATISTICS. Because both theories share the softmax, the logit RATIO between trial types cancels beta and (to first order) epsilon: ER_hat = 1 + logit(K1)/logit(W2), where K1 (A wins one 99% expert, B wins one 51% expert) has margin w99 - w51 and W2 (A wins two 51% experts, B wins one) has margin w51. Competitor box forces ER_hat <= 1.59; my box forces ER_hat >= 1.61 (center: 2.13). The parallel mid-level statistic ER_M = 1 + logit(R1)/logit(W2) (R1: one 75% expert vs one 51% expert) is disjoint with a bigger gap: competitor <= 1.31, mine >= 1.39 (centers 1.15 vs 1.71). These ratios remain disjoint even at the two theories' closest corners (delta=0.1, tau=0 vs gamma=0.7), where single-cell probabilities nearly converge -- the ratio metrics are the robust separator.

ONE-SIDED PARAMETER-FREE KNIVES against the competitor. (i) K2 (one 99% for A vs two 51% for B) and M4 (two 99% vs four 51%): the competitor's margin 0.99^gamma - 2*0.51^gamma (or doubled) is NEGATIVE for every (gamma, beta, epsilon) in its box, so it must predict p(A) < 0.50 strictly (box range roughly [0.26, 0.50) at K2). My center predicts p(A) = 0.51 (K2) and 0.52 (M4), rising to 0.60-0.70 at higher delta/tau. Any reliable p(A) > 0.55 on these cells falsifies the competitor outright. (ii) M3 (two 99% vs three 51%): competitor box max p(A) = 0.536 (at gamma=0.7, beta=0.55); my box min p(A) = 0.521 (at delta=0.1, tau=0, beta=0.27) -- near-disjoint intervals, with centers far apart (0.44 vs 0.62). p(M3) >= 0.60 is essentially impossible for the competitor. (iii) K1: competitor p(A) <= 0.60 across its entire box (center 0.53); mine >= 0.56 (center 0.61). (iv) R3 (two 75% vs three 51%): competitor in [0.40, 0.48], mine in [0.48, 0.69], centers 0.42 vs 0.54.

CROSSOVER DISSOCIATION. In the K family (one super expert for A vs k weak experts for B, k = 1, 2, 3) the competitor's psychometric crosses p = 0.5 strictly between k=1 and k=2 (one super is worth less than two weak experts for EVERY gamma), whereas my theory's crossover sits at k* = ER in [2, 2.6] for the bulk of my box (center: p(K1)=0.605, p(K2)=0.512, p(K3)=0.418 -- crossing between k=2 and k=3). Competitor center: 0.531, 0.413, 0.305. The location of the crossover is a direct, interpretable estimate of the super-expert premium.

HONEST ADVERSARIAL CELLS AGAINST MY OWN THEORY. The competitor's weak-cue weights (0.51^gamma in [0.62, 0.97]) are LARGER than mine at high tau (down to ~0.09 if tau annihilates the 0.51 cues), so on weak-cue-dominated trials the competitor can be far more confident than I can: W3 (three 51% for A vs one for B): my box ceiling ~0.75 (center 0.680), competitor up to 0.89 -- observed p(W3) > 0.78 falsifies my box. W4a (four 51% for A vs one 75% for B): my ceiling ~0.79 (center 0.70), competitor up to 0.92 -- p(W4a) > 0.85 falsifies my theory with no parameters to rescue it. G1 anchors (two supers + one 75% vs one 51%): I predict >= 0.79 across my box (center 0.87; my beta floor 0.27 forces high confidence), the competitor can go as low as 0.53. X1/X1m (the two 99% experts split between options) are tied-evidence controls where BOTH theories must predict exactly 0.50 for every parameter -- they isolate side/display bias that would otherwise contaminate every knife cell.

PREDICTED PATTERN IF MY THEORY IS RIGHT (center params delta=0.15, tau=0.01, beta=0.32): K1 0.605, K2 0.512, K3 0.418, M3 0.616, M4 0.524, R1 0.567, R3 0.540, W2 0.593, W3 0.680, W4a 0.703, G1 0.868, X1 0.500 -- a super-premium of ~2x, near-0.5 choices at the 1-super-vs-2-weak and 2-super-vs-4-weak ratios, moderate (never extreme) confidence on weak-aligned trials. IF THE COMPETITOR IS RIGHT (center gamma=0.35, beta=0.30): K1 0.531, K2 0.413, K3 0.305, M3 0.444, M4 0.332, R1 0.517, R3 0.417, W2 0.616, W3 0.721, W4a 0.795, G1 0.780 -- a super-premium of ~1.26x, consistent weak-side majorities at every 2:1 weak:super ratio, and higher confidence on weak-cue stacks. The patterns diverge most where it matters: every super/mid-vs-weak conflict cell separates the centers by 7-19 points, and the ER ratio statistics separate the entire boxes.

STRUCTURE. 24 unique pairs x K = 96//24 = 4 reps = 96 trials; every family has an exact A/B mirror (12 tilted each way) controlling side bias; no all-tie trials; every trial has 2-5 discriminators among 8 features, requiring genuine scanning; tied features are shown as 0/0 for both options. Display order (descending validity) is irrelevant to both theories, which re-sort internally. Differentiation from prior experiments: no v=0.5 cues and no chance-annihilation logic ([4]); no tally-margin decorrelation or d=0 tied-count knives ([1], [2]); no bounded-box ceiling attack on an unbounded-beta tallying model ([3]) -- here both models are bounded, weighted, and compared on the functional FORM of diagnosticity weighting via exchange rates.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Flattened-Diagnosticity Evidence Integration (the extreme-compression corner of the Bayesian cue-counting family): people integrate ALL binary expert ratings into one evidence score, where each discriminating cue casts a sign vote weighted by w_j = max(log(v_j/(1-v_j)) - tau, 0)^delta — chance cues (v=0.5) are annihilated exactly, but among genuinely diagnostic cues the subjective weight is a HEAVILY flattened (delta ~ 0.15) version of the normative log-odds, and choice is a low-gain softmax (2*beta ~ 0.6) plus a small uniform lapse. The new theoretical claim is that the compression exponent delta is itself the structural margin-compression device: shrinking delta equalizes weights, which GROWS count-dominated conflict margins (where cues split across options) while SHRINKING alignment-dominated cumulative margins (where several strong cues pile onto one side). This differential action — not any global gain knob, and not trial-level normalization — is what lets a single small beta simultaneously produce moderate TTB-leaning conflict-trial behavior (Exp 1), a slightly positive tally-vs-top-cue regression slope (Exp 2), graded ~0.62 tally-following (Exp 3), and near-chance |d|=1 agreement driven by the cancellation between super-expert matches and super-expert reversals (Exp 4).

**Parameters:**
- delta: [0.1, 0.2]
- tau: [0.0, 0.04]
- beta: [0.27, 0.37]
- epsilon: [0.0, 0.05]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Diagnosticity-Weighted Evidence Integration (Bayesian cue-counting).
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings (0/1).
    # Each feature j casts a sign vote sign(a_j - b_j); the vote is
    # weighted by the cue's log-likelihood-ratio diagnosticity
    # llr_j = log(v_j / (1 - v_j)), rectified by an attention floor
    # tau and compressed by an exponent delta:
    #   w_j = (max(llr_j - tau, 0))^delta
    # A v = 0.5 expert has llr = 0 -> exactly zero evidence.
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    delta = float(parameters["delta"])
    tau = float(parameters["tau"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    a, b = stim[0], stim[1]

    # Feature-wise votes: +1 where A wins the feature, -1 where B
    # wins, 0 on ties (a tied cue is uninformative for the pair).
    votes = np.sign(a - b)

    # Normative diagnosticity: log-likelihood-ratio of each expert.
    # Clip away from v = 1.0 to keep the log finite; v = 0.5 maps to 0.
    v = np.clip(val, 0.5, 1.0 - 1e-9)
    llr = np.log(v / (1.0 - v))

    # Limited-attention floor: diagnosticity below tau is discarded;
    # the remainder is compressed by the exponent delta. delta = 1 is
    # the purely normative Bayesian weight; delta < 1 flattens the
    # diagnosticity gradient toward tallying over informative cues
    # (while v = 0.5 cues stay annihilated at any delta, since
    # llr(0.5) = 0 <= tau); delta > 1 sharpens it toward
    # lexicographic/TTB-like use of the best experts.
    w = np.power(np.maximum(llr - tau, 0.0), delta)

    # Weighted evidence score. B's score is the mirror of A's because
    # every vote is zero-sum across the two options.
    s_a = float(np.dot(w, votes))
    scores = np.array([s_a, -s_a])

    # Numerically stable low-temperature softmax. When all attended
    # cues tie (s_a == 0) the softmax is exactly uniform, which is the
    # correct behavior for undiscriminating evidence.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Independent lapse: with probability epsilon output a uniform
    # pick over the two options.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Validity-Weighted Tallying (graded-evidence regime): people integrate evidence from ALL binary expert ratings, casting each feature a sign vote (+1 win / -1 loss / 0 tie) weighted by w_j = validity_j^gamma, summing into a zero-sum evidence score, with choice a softmax (inverse temperature beta) over the two scores plus an independent uniform lapse (epsilon). The theoretical claim added this round is about where humans sit in this family: in the UNSATURATED softmax regime. Because the evidence scores are integer-scale sign tallies, beta >= 3 already drives tanh(beta*m) ~ 1 for every unit margin, making the model behaviorally deterministic and pinning it to the tallying corner; the graded dose-response in tally margin that both experiments demand (Exp 1 metric at 44% of the deterministic-tally value; Exp 2's small positive b_d) lives at beta ~ 0.2-0.3, where p(A) is approximately linear in the tally margin. Gamma stays in a mild, provably sub-lexicographic band so that no subject leaps into the TTB regime.

**Parameters:**
- gamma: [0.05, 0.7]
- beta: [0.03, 0.55]
- epsilon: [0.0, 0.08]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Validity-Weighted Tallying.
    # Stimulus: array of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are binary expert ratings.
    # Each feature j casts a vote sign(a_j - b_j) for A's score and
    # the opposite for B's score; votes are weighted by w_j =
    # validity_j ** gamma. gamma = 0 -> unweighted Tallying;
    # gamma -> large -> lexicographic TTB (top cue dominates).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    n_features = stim.shape[1]
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )

    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    a, b = stim[0], stim[1]

    # Feature-wise votes: +1 where A wins the feature, -1 where B
    # wins, 0 on ties (a tied cue is uninformative for the pair).
    votes = np.sign(a - b)

    # Validity-based weights. Since validities are in [0.5, 1.0],
    # gamma = 0 gives all-ones weights (pure Tallying); increasing
    # gamma increasingly privileges high-validity cues, with the
    # top cue dominating as gamma grows (TTB limit).
    w = np.power(val, gamma)

    # Weighted evidence scores. B's score is the mirror of A's
    # because every vote is zero-sum across the two options.
    s_a = float(np.dot(w, votes))
    scores = np.array([s_a, -s_a])

    # Numerically stable softmax over the two weighted scores.
    # When all cues tie (s_a == 0) the softmax is exactly uniform,
    # which is the correct behavior for an undiscriminating tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Independent lapse: with probability epsilon output a uniform
    # pick over the two options.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # guard against float drift
    return int(np.random.choice(len(probabilities), p=probabilities))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    \"\"\"Strong-Side Choice Rate on Multi-Weak Conflict trials (SSR).\n\n    Selects the five conflict cells in which a high-validity side (99%\n    experts, or 75% experts for R3) faces TWO OR MORE opposing weak\n    experts: K2 (1x99 vs 2 weak), K3 (1x99 vs 3 weak), M3 (2x99 vs 3\n    weak), M4 (2x99 vs 4 weak), R3 (2x75 vs 3 weak) -- 5 unique pairs\n    x 2 mirrors x 4 reps = 40 trials per subject. Returns the\n    proportion of trials on which the subject chose the strong side.\n    \"\"\"\n    hits = 0\n    n = 0\n\n    a_col = data['option_a_ratings']\n    b_col = data['option_b_ratings']\n    r_col = data['response']\n\n    for a, b, r in zip(a_col, b_col, r_col):\n        a = [int(x) for x in a]\n        b = [int(x) for x in b]\n        if len(a) < 8 or len(b) < 8:\n            continue\n\n        # Validity-class vote balances (features 0-1 = 0.99 experts,\n        # 2-3 = 0.75 experts, 4-7 = weak experts).\n        d_hi = (a[0] + a[1]) - (b[0] + b[1])\n        d_mid = (a[2] + a[3]) - (b[2] + b[3])\n        d_lo = (a[4] + a[5] + a[6] + a[7]) - (b[4] + b[5] + b[6] + b[7])\n\n        # K2/K3/M3/M4: super experts alone on one side (no 75s involved)\n        # opposing >= 2 weak experts.\n        if d_mid == 0 and d_hi != 0 and abs(d_lo) >= 2:\n            target_is_a = d_hi > 0\n        # R3: two 75% experts vs three weak experts.\n        elif d_hi == 0 and abs(d_mid) == 2 and abs(d_lo) == 3:\n            target_is_a = d_mid > 0\n        else:\n            # Excludes K1 (|d_lo|=1), R1, W2/W3 (no strong side),\n            # W4a (|d_mid|=1), G1 (mixed 99+75 side), X1 (all tied).\n            continue\n\n        n += 1\n        chose_a = (int(r) == 0)\n        if chose_a == target_is_a:\n            hits += 1\n\n    if n == 0:\n        return 0.5\n    return hits / n\n",
  "rationale": "SSR = P(choose the high-validity side | that side faces >=2 opposing weak experts), computed over the five 'multi-weak conflict' cells K2, K3, M3, M4, R3 (40 trials/subject; mirrors make it side-bias-free). This is the cleanest behavioral readout of the super-expert exchange rate, which is the only thing the two theories disagree on (they share sign-vote architecture, softmax, lapse).\n\nWhy these cells: on them the choice margin is w_strong*k - w_weak*m with m>=2, so SSR is monotone in the strong:weak weight ratio. My theory's ratio w99/w_weak is at least ~1.6 across its whole box (log-odds compression with delta<=0.2 keeps a large super-premium; if weak validities are exactly 0.5 the weak weight is annihilated entirely, making the premium maximal), so it predicts SSR at or above ~0.50 (center ~0.52 with v_weak=0.51; ~0.74 with v_weak=0.50; up to ~0.81 at delta=0.2/tau high). The competitor's ratio v_hi^gamma/v_weak^gamma is at most ~1.6 and typically ~1.2-1.3, so 2+ weak experts always outvote one super expert: its SSR is strictly below 0.50 for EVERY parameter in its box (center ~0.38; range ~0.20-0.49; it only approaches 0.49 when beta~0 collapses all probabilities to chance, which still lies below my theory's typical values). The two predicted intervals are essentially disjoint, and the separation survives parameter heterogeneity: with per-subject parameters drawn from the boxes, my SSR mean is ~0.57 (sd ~0.13) vs competitor ~0.38 (sd ~0.11) under v_weak=0.51, and ~0.77 (sd ~0.07) vs ~0.38 under v_weak=0.50 \u2014 Welch t at N=25 is roughly 5.5-15 in all scenarios, far beyond the 0.01 threshold.\n\nWhy not other cells: single-weak conflicts (K1, R1) have small gaps (~0.05-0.07) and only dilute; weak-stack cells (W2/W3/W4a) have reversed but noisy gaps (8-24 trials, p far from 0/1) and would add variance faster than signal; G1 and the X1 tie are excluded because both theories (or side bias) contaminate them. Concentrating all 40 trials on the five cells with the largest, sign-consistent, box-robust gaps maximizes the mean difference per unit of between-subject variance, which is exactly what the Welch test rewards at N=25."
}
```

## Usage

```json
{
  "input_tokens": 7203,
  "output_tokens": 37152,
  "reasoning_tokens": 35855,
  "provider": "Sail Research"
}
```
