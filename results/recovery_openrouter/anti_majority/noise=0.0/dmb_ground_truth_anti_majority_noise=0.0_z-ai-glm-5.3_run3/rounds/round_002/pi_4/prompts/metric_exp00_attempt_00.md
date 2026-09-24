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
**Validities (n_features=8):** [0.95, 0.92, 0.62, 0.6, 0.62, 0.6, 0.62, 0.6]

**Trial pairs (n=24):**
  trial 1: A=[1, 0, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1, 1]
  trial 7: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0, 0]
  trial 10: A=[1, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  trial 11: A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1, 1]
  trial 13: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 0, 0]
  trial 14: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 15: A=[0, 0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 16: A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 17: A=[0, 0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 18: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 19: A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0, 0]
  trial 20: A=[0, 1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 21: A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0, 0]
  trial 22: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1, 1]
  trial 23: A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  trial 24: A=[1, 1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1, 1]

**Rationale:** ADVERSARIAL DESIGN: SPI (advocated) vs TTB (competing), built on a two-tier validity architecture with 8 cues: a dominant pair (0.95, 0.92) and a tight near-uniform low block of six (0.62 down to 0.52). This architecture is chosen so that (a) TTB's cascade is unambiguous (cue 1, then cue 2, then the low block), and (b) SPI's near-uniform weighting (gamma in [0,2]) makes the low block act as a coherent 'counting block': at gamma=0 SPI's evidence is exactly the 1s-count difference (n_B - n_A)/8; even at gamma=2 the whole low block (sum weight 0.528) outweighs either top cue, so block-mass sign conclusions are robust across SPI's entire parameter box. TTB NEVER consults the low block whenever cue 1 or cue 2 discriminates — so the low block is invisible to TTB and decisive for SPI. The design sweeps the low-block mass from the TTB winner's side to the loser's side while holding the TTB decision constant, generating a wide monotone psychometric that TTB's single constant p* cannot fit at any (beta, epsilon).

CORE DISSOCIATION 1 (TTB-flat vs SPI-sweep): Trials 1-6 (L1, L4, L5, L7, L8, L11) all have cue 1 (0.95) as the first and only-relevant discriminator favoring A, so TTB predicts ONE constant p(A) = (1-eps)*sigmoid(beta)+eps/2 on all of them. SPI's evidence for A sweeps monotonically: E = -0.875/-0.77 (L1), -0.25/-0.35 (L4), -0.125/-0.24 (L5), 0/-0.04 (L7, exact cancellation at gamma=0), +0.125/+0.05 (L8), +0.625/+0.28 (L11) (values at gamma=0/gamma=2). At gamma=1, beta=6, eps=0.2, rho=0.1, SPI predicts p(TTB winner) of roughly 0.19, 0.27, 0.29, 0.55, 0.66, 0.81 — a ~0.6-wide monotone sweep crossing 50% between L5 and L8, with L7 sitting near 50% (cancellation cell: SPI's weighted evidence nearly vanishes while TTB is at full p*). No constant fits a sweep this wide that straddles 50%.

CORE DISSOCIATION 2 (sign-robust direction conflicts): On X2 (trial 11: unanimous endorsement of A, B all zeros) TTB must pick A with p >= 0.51 at ANY (beta in [0.1,20], eps in [0,0.5]); SPI must pick B: E = -1, and even at its worst case (rho=0.35, eps=0.35, beta=0.5) p(A) <= 0.486 < 0.5. The same sign-robust conflict holds on L1 (E in [-0.875, -0.77]) and on the headline depth-2 cell Y1 (trial 9: cue 1 tied, cue 2 plus four low cues all endorse B, A empty — TTB winner B at p*, SPI p(TTB winner) <= 0.49 across its whole box because E in [+0.60, +0.63] makes empty A the sparse option). These cells dissociate the two theories in DIRECTION with sign robustness over both full parameter boxes.

CORE DISSOCIATION 3 (depth): D1 and D3 (trials 7-8) replicate L5 and L8 with cue 2 (0.92) as first discriminator. TTB predicts EXACTLY the same p* as the depth-1 twins (depth invariance); SPI predicts nearly identical evidence (E differs only by w1-w2), so depth cells double as replication while testing TTB's structural claim that cue identity is irrelevant. Y1 extends depth to a 5-cue mass against the cue-2 winner — the largest direction conflict available at depth 2.

ANCHORS: X1 (trial 10: cue 1 endorses A, all seven other cues endorse B) is an extreme AGREEMENT cell — both theories pick A — but with graded SPI confidence (E in [+0.51, +0.75]); it verifies that SPI subjects are not simply anti-endorsement robots, since here the sparser option coincides with the top-cue option. Controls C1/C2 (trials 23-24: identical vectors, all-0 and all-1) check both theories' exact-50% guessing prediction and catch key bias. Every dissociating trial has a mirrored twin (A/B swapped, trials 12-22) so side bias cancels; analysis pools by TTB-winner side. 24 unique pairs x K=4 = exactly 96 trials, 8 observations per pooled cell per subject.

DECISIVE QUANTITATIVE SIGNATURES: (a) Spearman correlation between cell choice proportion and SPI evidence E: strongly positive if SPI true, ~0 if TTB true; (b) p(TTB winner) < 0.5 on X2, L1, Y1 and their mirrors (sign-robust over both boxes) vs > 0.5 under TTB; (c) monotone ladder L1 < L4 < L5 < ~0.5 at L7 < L8 < L11 vs TTB's flat equality of all discriminating cells at one common p*; (d) L7 near 50% while L1/L11 are extreme — a confidence pattern TTB's two-level step function cannot produce, and conversely a flat p* that SPI's strictly monotone evidence-to-choice map cannot produce. This design differs from prior SPI-vs-TTB work by using the two-tier dominant-pair + counting-block architecture, the depth-2 mass-conflict cell Y1, and a cancellation cell placed exactly where SPI's weighted evidence vanishes.

**Computed schedule:** 24 unique pairs × 4 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Sparse-Option Preference / Polarity-Inverted Evidence Integration (SPI). When choosing between two options described by binary expert ratings, a substantial share of subjects systematically mis-map cue polarity: an endorsement (rating = 1) is treated as evidence AGAINST an option, so the option with FEWER 1s (the sparser option) is perceived as higher quality — consistent with a 'fewer red flags' reading of the display or a rarity-implies-quality inference. Subjects integrate ALL cues additively with near-uniform weights: the evidence for option A over B is E = sum_j w_j * (b_j - a_j), where w_j = v_j^gamma with gamma small (validity plays at most a mild role). Choice is a softmax over [E, -E] with inverse temperature beta, plus an independent lapse epsilon to uniform. A per-subject polarity-mixing weight rho captures heterogeneity: with probability 1-rho the subject responds according to the inverted (sparse-preferring) polarity, with probability rho according to the conventional polarity. This mixture attenuates the extreme pure-sparse predictions toward the moderately negative values observed. The theory is sharply distinguished from both incumbents: unlike Take-The-Best it uses all cues and inverts the evidence sign (so it is not pinned at ~0 on the depth/conflict-invariance and margin-slope metrics where TTB is structurally stuck), and unlike power-weighted integration it places humans at the FLAT end of the weighting ladder with inverted polarity, producing the negative signatures that steep-gamma PWEI cannot generate at any parameter setting.

**Parameters:**
- gamma: [0, 2]
- beta: [0.5, 20]
- epsilon: [0.05, 0.35]
- rho: [0, 0.35]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Sparse-option Preference / Polarity-Inverted Evidence Integration (SPI).
    # Stimulus: array-like of shape (2, n_features); row 0 = option A,
    # row 1 = option B; entries are 0/1 cue values. History is ignored
    # (no feedback in this task).
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"SPI expects a (2, n_features) stimulus; got shape {stim.shape}."
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
    rho = float(parameters["rho"])

    # Near-uniform validity weighting: w_j = v_j^gamma, gamma in [0, 2].
    # Normalization is a per-experiment constant rescale (validities are
    # fixed), absorbed by beta; it keeps the evidence scale O(1) for any
    # gamma and any n_features.
    w = np.power(val, gamma)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.full(n_features, 1.0 / n_features)

    a, b = stim[0], stim[1]
    # Polarity-INVERTED evidence for A over B: each cue where B carries
    # the endorsement (1) counts FOR A; each cue where A carries the
    # endorsement counts AGAINST A. Sparse options accumulate evidence.
    E = float(np.sum(w * (b - a)))

    # Softmax over [E, -E] with max-subtraction for numerical stability.
    # Inverted polarity: positive E favors A (the sparser option).
    scores_inv = np.array([E, -E])
    z = beta * (scores_inv - scores_inv.max())
    e = np.exp(z)
    p_inv = e / e.sum()

    # Conventional polarity (endorsement favors the option carrying it):
    # mirror of the inverted distribution.
    scores_norm = -scores_inv
    z2 = beta * (scores_norm - scores_norm.max())
    e2 = np.exp(z2)
    p_norm = e2 / e2.sum()

    # Per-subject polarity mixture: (1 - rho) inverted, rho conventional.
    p_core = (1.0 - rho) * p_inv + rho * p_norm

    # Independent lapse to uniform choice.
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Guard against float drift.
    return np.random.choice(len(probs), p=probs)
```


## COMPETING THEORY
**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
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
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    # Fixed design constants: 8 cues with the experiment's validities.\n    validities = np.array([0.95, 0.92, 0.62, 0.60, 0.62, 0.60, 0.62, 0.60])\n\n    # Parse per-trial rating vectors (cells are lists -> project safely).\n    a_lists = [list(r) for r in data[\"option_a_ratings\"]]\n    b_lists = [list(r) for r in data[\"option_b_ratings\"]]\n    A = np.asarray(a_lists, dtype=float)\n    B = np.asarray(b_lists, dtype=float)\n    resp = np.asarray(data[\"response\"]).astype(int)\n\n    n_feat = A.shape[1]\n    if n_feat == len(validities):\n        cue_order = np.argsort(-validities, kind=\"stable\")\n    else:\n        cue_order = np.arange(n_feat)\n\n    n_a = A.sum(axis=1)\n    n_b = B.sum(axis=1)\n\n    # TTB winner per trial: first discriminating cue in validity order.\n    winner = np.full(A.shape[0], -1, dtype=int)\n    for j in cue_order:\n        open_rows = winner == -1\n        if not open_rows.any():\n            break\n        wa = open_rows & (A[:, j] > B[:, j])\n        wb = open_rows & (B[:, j] > A[:, j])\n        winner[wa] = 0\n        winner[wb] = 1\n\n    abs_diff = np.abs(n_a - n_b)\n    denser = np.where(n_a > n_b, 0, 1)  # option carrying MORE 1-ratings\n\n    # Extreme-count trials only (|n_A - n_B| >= 5): the sign of SPI's\n    # evidence is robust across its whole parameter box on these cells.\n    valid = (winner >= 0) & (abs_diff >= 5)\n    ttb_picks_denser = (winner == denser)\n\n    chose_winner = (resp == winner)\n\n    grp_d = valid & ttb_picks_denser      # TTB winner is the DENSER option\n    grp_s = valid & (~ttb_picks_denser)    # TTB winner is the SPARSER option\n\n    nd = int(grp_d.sum())\n    ns = int(grp_s.sum())\n    if nd == 0 or ns == 0:\n        return 0.0\n\n    p_denser = float(chose_winner[grp_d].mean())\n    p_sparser = float(chose_winner[grp_s].mean())\n\n    return p_denser - p_sparser",
  "rationale": "METRIC: Among trials where the two options differ in total endorsement count by at least 5 (|n_A - n_B| >= 5), compute p(choose the TTB winner | TTB winner is the DENSER option) minus p(choose the TTB winner | TTB winner is the SPARSER option). These extreme cells are trials 1, 9, 10, 11, 12, 17, 20, 21, 22, 6 (24 conflict trials + 16 agreement trials per subject).\n\nWHY TTB GIVES EXACTLY 0: On every trial where any cue discriminates, TTB's probability of choosing its winner is the same constant p* = (1-eps)*sigmoid(beta) + eps/2, completely independent of the endorsement counts. Both groups (TTB-winner-denser and TTB-winner-sparser) consist only of discriminating trials, so the two conditional proportions have identical expectation p* and the difference is exactly 0 at ANY (beta, epsilon). This is a structural null \u2014 no parameter setting can move TTB off zero.\n\nWHY SPI GIVES A LARGE NEGATIVE VALUE: SPI's evidence for the TTB winner on these cells is sign-robust across its entire parameter box (gamma in [0,2], any beta, rho, eps). On conflict trials (TTB winner is denser: trials 1, 9, 11, 12, 20, 22) SPI's weighted evidence for the TTB winner is between -0.6 and -1 (e.g., trial 11: E = -1; trial 1: E in [-0.875, -0.789]), so p(choose TTB winner) < 0.5. On agreement trials (TTB winner is sparser: trials 6, 10, 17, 21) SPI's evidence is between +0.33 and +0.75, so p(choose TTB winner) > 0.5. Hence the difference is negative, and strongly so for any beta > 0 and rho < 0.5 (at gamma=1, beta=6, eps=0.2, rho=0.1 the metric is about -0.43; even at modest beta=2 it is about -0.15).\n\nWHY THE |count difference| >= 5 THRESHOLD: The 1s-count difference is a gamma-robust proxy for SPI's evidence magnitude. Restricting to extreme cells concentrates the signal where SPI's sigmoid is far from 50% on both groups, which maximizes the mean gap per unit of binomial noise; including weak-evidence cells (count diff 1-2) adds mostly noise since SPI is near 50% there. Per subject the metric uses 40 trials, keeping the between-subject standard deviation around 0.10-0.16 (binomial), while the between-theory mean gap is 0.3-0.5 for typical SPI parameters \u2014 comfortably detectable with Welch's t-test at N=25, alpha=0.01. The metric works identically on the pooled DataFrame and on single-subject slices (every subject sees all 24 unique pairs x4)."
}
```

## Usage

```json
{
  "input_tokens": 6211,
  "output_tokens": 30034,
  "reasoning_tokens": 28685,
  "provider": "Sail Research"
}
```
