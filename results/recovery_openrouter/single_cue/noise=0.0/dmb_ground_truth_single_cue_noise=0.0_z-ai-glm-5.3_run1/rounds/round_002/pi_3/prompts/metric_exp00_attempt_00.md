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
**Validities (n_features=8):** [0.95, 0.85, 0.75, 0.65, 0.5, 0.5, 0.5, 0.5]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0, 0, 1]
  trial 2: A=[0, 1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0, 0, 1]
  trial 3: A=[1, 1, 0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 0, 0, 0]
  trial 4: A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  trial 5: A=[0, 1, 0, 1, 1, 1, 0, 1]  B=[1, 1, 0, 1, 1, 0, 0, 1]
  trial 6: A=[0, 0, 1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 1, 0, 0, 0]
  trial 7: A=[0, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  trial 8: A=[1, 0, 0, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 1, 0, 1, 0]
  trial 9: A=[1, 1, 0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 1, 1, 1]
  trial 10: A=[1, 0, 1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  trial 11: A=[1, 1, 1, 0, 1, 1, 0, 1]  B=[1, 1, 1, 0, 0, 0, 0, 1]
  trial 12: A=[0, 1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 0, 0, 0]
  trial 13: A=[1, 0, 1, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0, 0, 0]
  trial 14: A=[0, 0, 1, 1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0, 1, 1, 1]
  trial 15: A=[1, 1, 1, 1, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1, 0, 1, 0]
  trial 16: A=[0, 0, 0, 0, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0, 1, 0, 1]

**Rationale:** CORE LOGIC. The two theories share the same architecture (sign votes x weights x softmax) and differ ONLY in the weight function. The one validity where they are structurally disjoint is v = 0.5: the competitor's weight is max(llr - tau, 0)^delta with llr(0.5) = 0, so a 50%-accurate expert is annihilated EXACTLY (weight 0) for every (delta, tau, beta, epsilon) in its box. My advocated theory gives it weight 0.5^gamma in [0.62, 0.97] -- i.e., 62-97% of a super-expert's weight. The design therefore builds everything around three chance experts (v = 0.5) set against one near-chance diagnostic expert (v = 0.55) and one super-expert (v = 0.95), using 8 features: [0.95, 0.85, 0.75, 0.65, 0.55, 0.50, 0.50, 0.50]. DISSOCIATION 1 (parameter-free knife: chance-only trials). Trials Z1-Z3 and mirror Z1m have ONLY v=0.5 cues discriminating. The competitor must predict p(A) = 0.50 EXACTLY on all four (all weights zero -> uniform softmax, and the lapse also injects uniform) for every parameter setting. My theory predicts a strictly graded ladder in the net chance-cue count k: p(A) = sigmoid(2*beta*0.5^gamma*k); at representative (beta=0.25, gamma=0.4): Z3 = 0.59, Z2 = 0.68, Z1 = 0.76 (range across my box: 0.51-0.96, always > 0.5 for k > 0). Any systematic deviation from 0.50 on Z trials, or any nonzero slope of p(A) against k, falsifies the competitor outright with no free parameters. DISSOCIATION 2 (parameter-free flatness + robust sign flip: super-expert vs chance-cue dose). Trials X0-X3 hold the diagnostic structure fixed (B wins the v=0.95 expert; for the competitor the chance cues contribute exactly nothing) while the net chance-cue count for A varies 0, 1, 2, 3. The competitor must predict the IDENTICAL p(A) on X0, X3, X2, X1 for all parameters -- its entire box spans only p(A) in [0.29, 0.36]. My theory predicts a strictly monotone ladder crossing 0.5: at (beta=0.25, gamma=0.4): X0 = 0.38, X3 = 0.47, X2 = 0.57, X1 = 0.66. Critically, on X1 and X2 the SIGN of preference disagrees robustly across BOTH entire parameter boxes: my theory predicts p(A) > 0.5 for every admissible (gamma, beta) (minimum 0.52), the competitor predicts p(A) < 0.5 for every admissible (delta, tau, beta, epsilon) (maximum 0.36). Mirrors Xm1, Xm2 flip sides and control for side bias. DISSOCIATION 3 (threshold structure of the 0.55 expert -- includes an honest knife against my own theory). The competitor treats v=0.55 as a genuinely diagnostic cue with weight 0.72-0.85 (about 60-75% of the super-expert's weight), while my theory treats 0.55 and 0.5 as nearly identical (weight ratio (0.5/0.55)^gamma in [0.94, 1.0]). W1 (A wins the 0.55 cue, B wins one 0.5 cue): my theory predicts p(A) <= 0.52 across its ENTIRE box (the validity difference is at most 0.042 in weighted units, so even beta = 0.55 yields only 0.523); the competitor predicts 0.59-0.65. This cell alone can falsify MY theory -- included deliberately. W2/W3 (A wins 2 or 3 chance cues, B wins the 0.55 cue): my theory predicts an increasing A-ladder (0.59 -> 0.68 at representative params; >= 0.52 across the box); the competitor predicts p(A) = 0.36-0.41, and moreover demands W2 = W3 EXACTLY (both equal -w(0.55) in evidence) -- a second parameter-free flatness knife. Wm2 mirrors W2. So the W family pits the competitor's sharp 0.5/0.55 threshold against my theory's near-equal weighting: the competitor demands W1 = 1 - W2 = 1 - W3 with W2 = W3; my theory demands W1 ~ 0.50 with W2 < W3 strictly. ANCHORS. G1/G1m (four best experts vs the 0.55 expert) are agreement trials pinning beta and epsilon (my theory ~0.81/~0.19; competitor ~0.90/~0.10 at representative params -- even anchor confidence mildly separates them). KEY QUANTITATIVE METRICS: (i) slope of p(A) on net chance-cue wins k at fixed diagnostic opposition (Z family and X family): exactly 0 for the competitor, ~0.08-0.10 per cue for my theory at the claimed unsaturated regime (beta ~ 0.25); (ii) the sign of p(A)-0.5 on X1/X2/W2: positive for all my parameters, negative for all competitor parameters; (iii) p(W1): mine <= 0.52, competitor >= 0.59 -- disjoint intervals, a two-sided knife. PREDICTED PATTERN IF MY THEORY IS RIGHT: graded chance-cue dose-response everywhere (Z ladder 0.59/0.68/0.76; X ladder 0.38/0.47/0.57/0.66; W ladder 0.50/0.59/0.68 at representative params), i.e., subjects treat 50%-accurate experts as carrying most of a super-expert's weight. IF THE COMPETITOR IS RIGHT: exact 0.50 on all Z trials, flat 0.32 across the X family, W1 = 0.62 with W2 = W3 = 0.38, i.e., subjects exactly annihilate chance experts and sharply distinguish 0.55 from 0.50. The patterns are disjoint almost everywhere. 16 unique pairs x K = 6 = 96 trials; sides balanced (8 A-tilted, 7 B-tilted, 1 near-neutral); no all-tie trials; every trial requires scanning all 8 experts to locate discriminators; the three 50% experts are announced up front, so the design is a fair test of whether explicitly chance information is integrated or annihilated.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
  "metric_source": "import numpy as np\nimport pandas as pd\n\n\ndef metric(data: pd.DataFrame) -> float:\n    # Chance-cue sensitivity on \"chance-only\" (Z-family) trials.\n    # Design validities: [0.95, 0.85, 0.75, 0.65, 0.5, 0.5, 0.5, 0.5]\n    # -> features 0-3 are diagnostic, features 4-7 are the four v=0.5 experts.\n    # A Z-family trial is one where ALL diagnostic cues tie, so the pair is\n    # discriminated ONLY by the 50%-accurate experts. On such a trial let\n    # k = net number of chance cues won by A (signed). The competing theory\n    # (llr-rectified weights) annihilates v=0.5 cues exactly, so p(A)=0.5\n    # regardless of k -> subjects match the k-implied side at chance (metric 0).\n    # My theory gives v=0.5 cues weight 0.5^gamma, so choices should follow\n    # sign(k) above chance (metric > 0).\n    DIAG = [0, 1, 2, 3]\n    CHANCE = [4, 5, 6, 7]\n\n    a_ratings = data[\"option_a_ratings\"].apply(lambda x: np.asarray(list(x), dtype=int))\n    b_ratings = data[\"option_b_ratings\"].apply(lambda x: np.asarray(list(x), dtype=int))\n    responses = data[\"response\"].to_numpy()\n\n    correct = []\n    for av, bv, r in zip(a_ratings, b_ratings, responses):\n        if av.shape[0] != 8 or bv.shape[0] != 8:\n            continue\n        # Keep only trials where every diagnostic cue ties (chance-only trials)\n        if not np.array_equal(av[DIAG], bv[DIAG]):\n            continue\n        # Net chance-cue advantage for A\n        k = int(np.sign(av[CHANCE] - bv[CHANCE]).sum())\n        if k == 0:\n            continue\n        chose_a = (int(r) == 0)  # response: 0 = A, 1 = B\n        followed = (k > 0 and chose_a) or (k < 0 and not chose_a)\n        correct.append(1.0 if followed else 0.0)\n\n    if len(correct) == 0:\n        return 0.0\n    # 2 * (accuracy - 0.5): 0 = chance-level following of chance cues,\n    # positive = subjects systematically follow the 50%-accurate experts.\n    return 2.0 * (float(np.mean(correct)) - 0.5)",
  "rationale": "This metric isolates the single structural knife-edge between the two theories: the weight assigned to v=0.5 experts. The competing theory's weight w_j = max(llr_j - tau, 0)^delta is EXACTLY zero for v=0.5 (llr=0 <= tau) for every admissible (delta, tau, beta, epsilon), so on trials where all four diagnostic cues (v in {0.95, 0.85, 0.75, 0.65}) tie, it predicts p(A) = 0.5 exactly regardless of how many chance cues favor A or B. My advocated theory assigns v=0.5 cues weight 0.5^gamma in [0.62, 0.97], producing a graded preference for whichever side wins more chance cues. The metric selects exactly these 'chance-only' trials (8 of the 16 unique pairs: trials 1, 2, 3, 4, 11, 12, 13, 14 \u2014 48 of 96 trials per subject), computes the signed chance-cue advantage k for A, and measures how often the subject's choice follows sign(k), mapped to [-1, 1] around the 0.5 chance baseline. Expected values: competitor -> 0 (pure binomial noise, per-subject variance ~4*0.25/48 = 0.021); my theory -> strongly positive (at representative parameters, p(A|k) = sigmoid(2*beta*0.5^gamma*k) gives accuracies ~0.59/0.68/0.76 for |k|=1/2/3, i.e., a metric around +0.3; even at the lowest beta in my box it stays above 0 for every subject since beta>0 and gamma<inf). The contrast is parameter-free on the competitor side (its entire box predicts exactly 0.5 on every chance-only trial) and sign-consistent across my entire parameter box, so the between-theory gap is large while per-subject variance is tight (48 binary trials per subject), giving a Welch t-test at N=25 a comfortable margin below alpha=0.01. The metric works identically on the pooled DataFrame and on single-subject slices, satisfying the acceptance-test requirements."
}
```

## Usage

```json
{
  "input_tokens": 6115,
  "output_tokens": 6699,
  "reasoning_tokens": 5568,
  "provider": "Sail Research"
}
```
