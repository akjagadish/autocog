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



## ADVOCATED THEORY
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


## COMPETING THEORY
**Description:** **WADD-γ with Positional Validity Misattribution (WADD-γ + position-default blending), with heterogeneous decision determinacy.**

People integrate cues compensatorily: each expert's subjective importance is a power transform of that expert's diagnosticity on the log-odds scale, L_j = log(v_j/(1−v_j)), w_j ∝ L_j^γ; the decision rests on the normalised evidence margin d = Σ_j w_j (x_Aj − x_Bj), with p(A) logistic in β·d and an attentional lapse ε. γ ≈ 1.15–1.85 places the decision maker in the near-lexicographic but still compensatory regime: the top expert usually survives a coalition of dissenters, but a near-peer second cue plus a couple of moderate cues can cancel or overturn it.

The substantive claim about *where the weights get attached* is retained and is the theory's main falsifiable commitment. Validities are stated verbally in the instructions, but the decision maker must bind each stated validity to a *column* of the displayed rating vector. This binding is costly and error-prone, and people fall back on a powerful display heuristic: **importance decreases from left to right** (the first-listed expert is the best one, the last-listed the worst). Subjective weights are therefore a blend, with mixing weight λ, of the correctly-bound validity weights and the weights implied by the positional default (the same weight multiset re-sorted into descending order across columns):

  w_eff ∝ (1−λ)·L_j^γ + λ·sort_desc(L^γ)_j.

This is an *environmental* prediction, not a per-experiment fudge: conflict-trial accuracy is a property of the alignment between display order and validity order, while the integration rule itself stays a single stationary compensatory WADD-γ. Concretely, (i) scramble the column order of an environment whose cues are currently presented in descending validity and conflict accuracy must fall toward (and, on some families, below) chance; (ii) re-sort the columns of a scrambled environment into descending validity and the same subjects must become sharply more accurate and more top-cue-consistent; (iii) an explicit column-labelling manipulation that makes the validity→column binding visually trivial should abolish the drop. Because the blend is the exact identity whenever columns are already validity-ordered, λ is behaviourally inert in validity-sorted environments and can only be identified by comparing environments — precisely the invariance signature the theory stakes itself on.

The third claim is about **population structure**: the same stationary rule is held by everyone, but decision determinacy is heterogeneous across people. Some subjects convert modest weighted margins into near-deterministic choices, others remain near chance on all but the most lopsided evidence. That heterogeneity is carried by the logistic gain β and by mild variation in the validity-sensitivity exponent γ, not by discrete strategy switching; the theory therefore predicts that between-subject spread in conflict accuracy should be largest in environments with intermediate margins and should shrink towards zero in environments where every conflict family is near-cancelling.

**Parameters:**
- gamma: [1.15, 1.85]
- beta: [1.0, 13.0]
- epsilon: [0.0, 0.14]
- lam: [0.64, 1.00]
- validities: validities

**`predict source code`:**
```python
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
```

**`policy source code`:**
```python
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    validities = [0.55, 0.62, 0.70, 0.85, 0.93]\n\n    def key(v):\n        return ''.join([str(int(round(float(x)))) for x in v])\n\n    # conflict families: TTB (top-validity cue) and a left-to-right positional\n    # weighting scheme point to OPPOSITE options\n    conflict_keys = {\n        '10000|00001', '00001|10000',      # family A\n        '01000|00010', '00010|01000',      # family B\n        '11000|00011', '00011|11000',      # family E\n        '00100|11000', '11000|00100',      # family F\n    }\n    # agreement families: both accounts favour the same option\n    agree_keys = {\n        '11111|00000', '00000|11111',      # family D (dominance)\n        '00001|00000', '00000|00001',      # family G\n        '10000|00000', '00000|10000',      # family H\n    }\n\n    def ttb_winner(a, b, order):\n        for j in order:\n            if a[j] > b[j]:\n                return 0\n            if b[j] > a[j]:\n                return 1\n        return -1\n\n    n_conf = 0\n    hit_conf = 0\n    n_agree = 0\n    hit_agree = 0\n\n    for _, row in data.iterrows():\n        try:\n            a = [int(round(float(x))) for x in row['option_a_ratings']]\n            b = [int(round(float(x))) for x in row['option_b_ratings']]\n            resp = int(row['response'])\n        except Exception:\n            continue\n        n = len(a)\n        if len(b) != n or n == 0:\n            continue\n        if n == len(validities):\n            v = np.asarray(validities, dtype=float)\n        else:\n            # fallback: assume columns are displayed in ascending validity\n            v = np.linspace(0.55, 0.93, n)\n        order = list(np.argsort(-v, kind='stable'))\n        w = ttb_winner(a, b, order)\n        if w < 0:\n            continue\n        k = key(a) + '|' + key(b)\n        if k in conflict_keys:\n            n_conf += 1\n            hit_conf += 1 if resp == w else 0\n        elif k in agree_keys:\n            n_agree += 1\n            hit_agree += 1 if resp == w else 0\n\n    if n_conf == 0 or n_agree == 0:\n        return 0.0\n\n    return float(hit_agree) / n_agree - float(hit_conf) / n_conf\n",
  "rationale": "Both theories are pinned near the same level on the agreement families (D: full dominance, G: only the 93% expert speaks, H: only the 55% expert speaks) \\u2014 in all of these the option carrying the 1(s) is favoured by TTB and by any positive-weight compensatory rule alike. The conflict families (A, B, E, F) are constructed so the true-validity cascade and the left-to-right positional weighting point to opposite options. The metric is the WITHIN-SUBJECT difference: (rate of choosing the TTB-implied option on agreement trials) minus (rate on conflict trials).\n\nUnder Take-The-Best the choice probability is sigmoid(beta) mixed with lapse on EVERY discriminating trial regardless of which cue does the discriminating, so both terms have the same expectation and the difference is ~0 for every subject regardless of beta and epsilon; the only spread is binomial sampling noise over 36 and 48 trials (var <= ~0.012). Under WADD-\\u03b3 with positional misattribution the conflict trials are answered against the TTB option (weights: col0 \\u2248 .44, col1 \\u2248 .25 vs col3 \\u2248 .10, col4 \\u2248 .12 at \\u03bb=.8, \\u03b3=1.5), giving conflict TTB-consistency well below .5 while agreement consistency stays high; the difference is large and positive (~.2 at the weakest \\u03b2, ~.8 at moderate \\u03b2). Family C is deliberately excluded because the competitor is at chance there and would dilute the contrast. Because the metric self-normalises each subject against their own agreement-trial determinacy, it is insensitive to the wide beta/epsilon heterogeneity that would otherwise inflate between-subject variance for TTB."
}
```

## Usage

```json
{
  "input_tokens": 9605,
  "output_tokens": 5173
}
```
