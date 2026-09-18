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



## ADVOCATED THEORY
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n\n    # Design validities (display order): col0..col5\n    v_default = np.array([0.60, 0.93, 0.55, 0.86, 0.68, 0.78], dtype=float)\n\n    def build(n):\n        v = v_default.copy()\n        if v.shape[0] < n:\n            v = np.concatenate([v, np.full(n - v.shape[0], 0.55)])\n        v = v[:n]\n        v = np.clip(v, 0.5 + 1e-6, 1 - 1e-6)\n        L = np.log(v / (1.0 - v))\n        w = np.power(L, 1.5)\n        wpos = np.sort(w)[::-1]\n        weff = 0.2 * w + 0.8 * wpos          # lam = 0.8, mid of advocated box\n        s = weff.sum()\n        weff = weff / s if s > 0 else np.ones(n) / n\n        cascade = list(np.argsort(-v, kind=\"stable\"))\n        return weff, cascade\n\n    cache = {}\n    num = 0.0\n    den = 0.0\n\n    for a_raw, b_raw, resp in zip(data[\"option_a_ratings\"],\n                                  data[\"option_b_ratings\"],\n                                  data[\"response\"]):\n        a = np.asarray(a_raw, dtype=float).ravel()\n        b = np.asarray(b_raw, dtype=float).ravel()\n        n = int(min(a.shape[0], b.shape[0]))\n        if n == 0:\n            continue\n        a = a[:n]\n        b = b[:n]\n        if n not in cache:\n            cache[n] = build(n)\n        weff, cascade = cache[n]\n\n        d = float(np.dot(weff, a - b))\n\n        winner = None\n        for j in cascade:\n            if a[j] > b[j]:\n                winner = 0\n                break\n            if b[j] > a[j]:\n                winner = 1\n                break\n        if winner is None:\n            continue\n        if abs(d) < 0.20:                    # ambiguous / near-cancelling for WADD-pos\n            continue\n        pos_choice = 0 if d > 0 else 1\n        if pos_choice == winner:             # models agree -> uninformative\n            continue\n\n        den += 1.0\n        try:\n            r = int(resp)\n        except Exception:\n            continue\n        num += 1.0 if r == pos_choice else 0.0\n\n    if den == 0:\n        return 0.5\n    return float(num / den)",
  "rationale": "The metric isolates the trials on which the two theories make OPPOSITE directional predictions with a large positional-WADD margin: families A (col0 .60 vs col3 .86), B (col0 vs col5 .78) and F (col3 vs the col0+col2 coalition). Take-The-Best follows the true validity cascade (col1 > col3 > col5 > col4 > col0 > col2) and therefore picks the option that positional WADD rejects on every one of these 30 trials; positional misattribution (weights re-sorted descending across display columns, giving col0 the largest subjective weight) picks the opposite option with |d| ~ .26-.39. The metric is simply the proportion of those conflict trials on which the subject chose the positionally-favoured option. Under TTB this is essentially the flip-noise floor ((1-eps)*sigmoid(-beta)+eps/2, roughly .05-.30), under the advocated model it is well above .5 (roughly .6-.95 depending on beta). Trials with |d| < .20 (families C, D, E) are excluded because the advocated model is near-indifferent there, which would only add between-subject noise; trials where both models agree (G, H, I) are excluded because they carry no directional contrast. This yields a large mean gap (~.5) while keeping each subject's value tightly bounded within its theory, since the sign of the prediction is invariant over both theories' whole parameter boxes."
}
```

## Usage

```json
{
  "input_tokens": 9720,
  "output_tokens": 6735
}
```
