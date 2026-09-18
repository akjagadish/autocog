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
**Validities (n_features=6):** [0.58, 0.97, 0.53, 0.8, 0.66, 0.9]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 3: A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 5: A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  trial 6: A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 7: A=[1, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  trial 8: A=[1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 9: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  trial 10: A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 11: A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  trial 12: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 13: A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  trial 14: A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 15: A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  trial 16: A=[1, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]

**Rationale:** GOAL: dissociate unit-weight Tallying (advocated) from Amplified-Validity Additive Integration (AVAI). Both models are permutation-invariant over display position, so the decisive manipulation is to hold the WIN COUNT constant while sweeping the validity-weighted margin across its whole range, and then to break the rank-order between count margin and weighted margin.

VALIDITIES: six experts in deliberately scrambled display order, [0.58, 0.97, 0.53, 0.80, 0.66, 0.90]. Ranked: H1 = idx1 (.97), H2 = idx5 (.90), M = idx3 (.80), L1 = idx4 (.66), L2 = idx0 (.58), L3 = idx2 (.53). Scrambling removes any 'read the leftmost cue' artifact and prevents position from proxying validity. AVAI weights at gamma = 2: H1=.941, H2=.810, M=.640, L1=.436, L2=.336, L3=.281 (the compressed regime AVAI assumes). Cues that are irrelevant on a trial are set to 1 for BOTH options, so every display is dense and requires actual comparison rather than counting 1s.

8 base structures, each accompanied by its exact A/B mirror (16 unique pairs, 6 repetitions = 96 trials), so the predicted response of BOTH models is exactly key-balanced and side bias cannot mimic either model.

(1) CONSTANT-COUNT-MARGIN LADDER (bases 1-4, plus mirrors). All four are +1 win-count margins for A, so Tallying must produce the SAME p(A) = sigmoid(beta) on every one of them, for any beta and epsilon. AVAI's weighted difference, in contrast, sweeps the entire probability range:
  - Base 1: A wins {H1} only. d = +0.94 -> AVAI p(A) ~ .998.
  - Base 2: A wins {H1,H2}, B wins {L2}. d = +1.42 -> AVAI p(A) ~ 1.00.
  - Base 3: A wins {L2,L3}, B wins {H1}. d = -0.32 -> AVAI p(A) ~ .11 (DIRECTIONAL REVERSAL relative to Tallying).
  - Base 4: A wins {L1,L2,L3}, B wins {H1,H2}. d = -0.70 -> AVAI p(A) ~ .01 (near-deterministic reversal).
A flat profile across these four (all equal, all above .5) is a signature Tallying alone can produce; a monotone sweep from ~1.00 down to ~.01 at constant count margin is impossible for Tallying under any parameters. This is the crown-jewel contrast: flat vs. full-range sweep.

(2) RANK-ORDER VIOLATION (bases 5-6, plus mirrors). Both are +2 count margins for A, so Tallying predicts them equal AND strictly more extreme than every margin-1 pair in block (1).
  - Base 5: A wins {M,L1,L2,L3}, B wins {H1,H2}. d = -0.06 -> AVAI p(A) ~ .41, i.e. essentially chance despite the LARGEST count margin in the design.
  - Base 6: A wins {H1,H2}, B wins nothing. d = +1.75 -> AVAI p(A) ~ 1.00.
AVAI therefore predicts that a margin-2 trial (base 5) is noisier than a margin-1 trial (base 1 or 2) — a rank-order violation Tallying strictly forbids. Base 6 doubles as an agreement baseline that identifies epsilon and the global beta scale free of the model contrast.

(3) TALLY TIES (bases 7-8, plus mirrors). Each option wins exactly the same number of cues, so Tallying is EXACTLY 50/50 for any beta and epsilon — a parameter-free prediction.
  - Base 7: A wins {H1}, B wins {L3}. d = +0.66 -> AVAI p(A) ~ .99.
  - Base 8: A wins {H1,L3}, B wins {H2,L1}. d = -0.02 -> AVAI p(A) ~ .46 (also near chance).
If subjects are at chance on base 7 as well as base 8, AVAI is falsified while Tallying is confirmed with no free parameters; if base 7 is near-deterministic while base 8 is at chance, that graded split is exactly AVAI's magnitude-sensitivity claim and cannot be produced by Tallying. Base 8 is additionally gamma-diagnostic: the sign of d for {H1,L3} vs {H2,L1} is near zero and flips with gamma, so it pins the amplification exponent if AVAI wins.

No pair is degenerate: on 5 of the 8 base structures the models disagree in direction or by more than 40 percentage points, the two tie trials are parameter-free tests of Tallying, and the remaining structures supply the noise/lapse baseline.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
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
**Description:** **Amplified-Validity Additive Integration (AVAI).**

People do not stop at one reason (Take The Best) and do not count cues as equal (Tallying). Instead, on every trial they read all expert ratings, convert each communicated validity v_j into a subjective *pull* w_j = v_j^gamma, and form a single continuous evidence difference d = sum_j w_j (a_j - b_j). Choice is a logistic function of that continuous difference, p(A) = (1-eps)*sigmoid(beta*d) + eps/2.

Three claims give the theory its empirical bite:

1. **No stopping rule.** Every cue enters the sum, so a single high-validity cue can be outvoted by a coalition of weaker cues. This is why, when the top cue points one way and three or four lesser cues point the other, people overwhelmingly follow the coalition (unlike TTB).

2. **Unequal but compressed weights.** Because all validities lie in (0.5, 1], raising them to a modest power gamma (~2) yields weights that are all of the same order of magnitude (roughly 0.2-0.9). Integration therefore *looks* like counting most of the time, but with a systematic tilt toward the option holding the more valid cues. Crucially, the exponent gamma is a *subjective amplification* of stated validity, not the normatively correct log-odds transform: people over-weight validity relative to plain counting yet massively under-weight it relative to Bayes (log(v/(1-v)) would make one 0.95 cue beat two mid cues; humans do not do this). gamma is the single psychological parameter that locates a person on the tallying-to-lexicographic continuum.

3. **Magnitude sensitivity.** Because the logistic acts on the raw weighted difference rather than on a binarised winner code, trials differ *gradedly* in confidence. Unit-tally ties are not forced to chance: they resolve toward whichever side owns the higher-validity cues, but only weakly when the weighted margin is small (e.g. 0.87+0.55 vs 0.78+0.70 stays near chance) and strongly when it is large (0.93 vs 0.60 is near-deterministic). Symmetrically, trials with a big count margin but a small weighted margin become noisy rather than deterministic.

Residual attentional lapses are captured by an epsilon-rate uniform guess. No learning occurs across trials (there is no feedback), so the rule is applied stationarily.

**Parameters:**
- gamma: [1.8, 2.7]
- beta: [4.5, 8.5]
- epsilon: [0.0, 0.04]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    # Amplified-Validity Additive Integration (AVAI).
    # d = sum_j (v_j ** gamma) * (a_j - b_j); p(A) = sigmoid(beta * d),
    # mixed with an epsilon lapse. All cues are consulted (no stopping
    # rule) and the logistic acts on the CONTINUOUS weighted difference,
    # so both tie-trials and large-margin trials are graded.
    import numpy as np

    # ---- parse the stimulus into two rating vectors -------------------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(state['option_a_ratings'], dtype=float).ravel()
            b = np.asarray(state['option_b_ratings'], dtype=float).ravel()
        else:
            vals = list(state.values())
            a = np.asarray(vals[0], dtype=float).ravel()
            b = np.asarray(vals[1], dtype=float).ravel()
    else:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 2:
            a, b = arr[0].ravel(), arr[1].ravel()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            a, b = arr[:, 0].ravel(), arr[:, 1].ravel()
        elif arr.ndim == 1:
            half = arr.shape[0] // 2
            a, b = arr[:half], arr[half:2 * half]
        else:
            flat = arr.reshape(2, -1)
            a, b = flat[0], flat[1]

    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.array([0.5, 0.5])
    a = a[:n]
    b = b[:n]

    # ---- validities -> subjective weights ----------------------------
    v_raw = parameters.get('validities', None)
    if v_raw is None:
        v = np.linspace(0.95, 0.55, n)
    else:
        v = np.asarray(v_raw, dtype=float).ravel()
        if v.shape[0] < n:
            pad = np.full(n - v.shape[0], 0.55)
            v = np.concatenate([v, pad])
        v = v[:n]
    # validities are probabilities of being right; keep them in (0.5, 1]
    v = np.clip(v, 0.5 + 1e-6, 1.0 - 1e-9)

    gamma = float(parameters['gamma'])
    gamma = max(gamma, 0.0)
    w = np.power(v, gamma)

    # ---- continuous weighted evidence difference ---------------------
    d = float(np.sum(w * (a - b)))

    beta = float(parameters['beta'])
    eps = float(parameters['epsilon'])
    eps = min(max(eps, 0.0), 1.0)

    # numerically stable two-option softmax on [beta*d, 0]
    z = np.array([beta * d, 0.0], dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / e.sum()

    p = (1.0 - eps) * p_core + eps * 0.5
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum()
    return p
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    v = np.array([0.58, 0.97, 0.53, 0.80, 0.66, 0.90], dtype=float)\n    w = v ** 2.0  # AVAI-style amplified weights (gamma=2)\n\n    num = 0.0\n    den = 0.0\n    for a_raw, b_raw, resp in zip(data['option_a_ratings'],\n                                  data['option_b_ratings'],\n                                  data['response']):\n        a = np.asarray(a_raw, dtype=float).ravel()\n        b = np.asarray(b_raw, dtype=float).ravel()\n        n = int(min(a.shape[0], b.shape[0], w.shape[0]))\n        if n == 0:\n            continue\n        a = a[:n]\n        b = b[:n]\n        ww = w[:n]\n        a_wins = float(np.sum(a > b))\n        b_wins = float(np.sum(b > a))\n        if a_wins != b_wins:\n            continue  # not a tally tie\n        d = float(np.sum(ww * (a - b)))\n        if abs(d) < 0.30:\n            continue  # tie trials whose weighted margin is negligible\n        den += 1.0\n        if d > 0:\n            num += 1.0 if int(resp) == 0 else 0.0\n        else:\n            num += 1.0 if int(resp) == 1 else 0.0\n\n    if den == 0:\n        return 0.5\n    return num / den",
  "rationale": "The design contains tally-TIE trials (each option wins exactly the same number of expert comparisons) that nevertheless carry a LARGE validity-weighted margin: base 7 / its mirror, where one option wins only on the .97 expert and the other only on the .53 expert. Tallying's softmax is exactly uniform whenever the two counts are equal, for ANY beta and epsilon, so the predicted proportion of choices for the option holding the high-validity cue is exactly 0.50 \\u2014 a parameter-free prediction. AVAI acts on the continuous weighted difference (d \\u2248 +0.66 with gamma=2), giving p \\u2248 0.95\\u20130.99 for that same option across its whole parameter range. I deliberately exclude the other tie structure (H1+L3 vs H2+L1, |d| \\u2248 0.02) via the 0.30 threshold, because both theories predict chance there and it would only dilute the contrast. Each subject contributes 12 such trials, so per-subject binomial noise is small (var of the proportion \\u2264 0.021 for Tallying, ~0.003 for AVAI), while the mean gap is ~0.47 \\u2014 large mean separation with tight per-subject spread. The metric is well defined on a single subject's slice and on the pooled frame."
}
```

## Usage

```json
{
  "input_tokens": 8525,
  "output_tokens": 2296
}
```
