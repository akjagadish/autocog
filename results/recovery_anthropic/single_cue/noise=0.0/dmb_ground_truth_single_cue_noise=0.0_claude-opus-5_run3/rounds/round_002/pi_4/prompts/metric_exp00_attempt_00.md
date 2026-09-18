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
**Validities (n_features=5):** [0.78, 0.74, 0.7, 0.58, 0.9]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 4: A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 6: A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 8: A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  trial 9: A=[0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  trial 10: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 11: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 12: A=[0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** GOAL: dissociate threshold-terminated compensatory integration with conclusion-graded commitment (weights fixed by validity, UNNORMALISED accumulation, early stop when no single unread expert could reverse the lead, amplified commitment after a stop vs. shallow readout of an unresolved residual) from read-order scanning (geometric weighting by SCREEN position, ~65% primacy / ~35% recency, evidence NORMALISED by the discriminating weight so one early cue is as confident as many).

DESIGN MOVE: five experts with validities by screen position [.78, .74, .70, .58, .90] \u2014 the single most valid expert (90%) is placed LAST, three mid-validity experts occupy the leading positions, and a near-useless expert (58%) sits at position 3. Under the advocated model the compressed weights are w \u2248 [.235, .191, .149, .043, .381] (stable across rho in [1.25,1.45]). Because the largest weight belongs to the last-read expert, the sufficiency bar for every earlier screen position is phi*w4 \u2248 .38-.44: NO single early cue can ever clear it, but a crowd of two-to-three leading mid cues can. The model therefore switches qualitatively as leading cues are added: 1 leading cue -> never stops, reads the 90% expert, shallow residual readout; 2 leading cues -> borderline/likely stop; 3 leading cues -> confident stop that never even reads the best expert (tallying-looking behaviour, choosing AGAINST the 90% expert). The competitor's weights are position-based, so its normalised evidence is near |S| = 1 on all of these, flat.

CORE LADDER (pairs 7/8, 3/4, 11/12, 5/6): the same target option is opposed by the last-listed 90% expert while it is supported by, respectively, (a) only pos0, (b) pos1+pos2, (c) pos0+pos1, (d) pos0+pos1+pos2. Advocated model predicts a steep MONOTONE gradient with a DIRECTION REVERSAL: P(support-side) \u2248 .25 (one cue, unresolved, best expert wins the residual), \u2248 .49 (two mid cues, unresolved and near chance), \u2248 .85 (pos0+pos1, stop typically triggered, amplified commitment despite a total weighted margin of only +.05), \u2248 .97 (three cues, stop certain). Read-order scanning predicts a FLAT ceiling: for a primacy reader |S| = .96-1.00 on all four (choose the leading-cue side every time), for a recency reader |S| = .78-1.00 in the opposite direction on all four. No (lam, gamma, beta, epsilon) can produce a monotone gradient plus a reversal across cells whose normalised evidence is essentially constant, and no accumulator with normalisation can put pair 3/4 near chance while pair 5/6 is at ceiling.

BOTH-ORIENTATIONS-AGREE CELL (pairs 1/2): d = [+1,-1,-1,+1,0] with the 90% expert tied. The earliest AND the latest discriminating cue favour the same option, so primacy AND recency scanners both choose it with S \u2248 +.45 to +.94 (P \u2248 .97-1.00) \u2014 no orientation escape hatch. The advocated model never clears the bar (max partial |E| = .235 < .38), so it exhausts the list and reads out a residual of only -.06, predicting a slight preference for the OPPOSITE option (P \u2248 .35-.43). This is the sharpest single cell: opposite direction, and the competitor is pinned near ceiling for its whole parameter box.

CASCADE-INVERSION CELL (pairs 9/10): d = [0,0,-1,0,+1] \u2014 the 70% expert at position 2 against the 90% expert at position 4. The advocated model cannot stop on the small pos2 cue, reads to the end and picks the 90% side (P \u2248 .76-.91); a primacy scanner picks the pos2 side robustly (pos2 keeps the lower inspection rank even at gamma = .25), a recency scanner picks the pos4 side. Direction disagreement with the 65% majority orientation, and a graded (not ceiling) confidence for the advocated model.

Every class is mirrored across sides (advocated model favours A on 6 pairs and B on 6; primacy scanning likewise 6/6), so side bias cannot mimic either model. 12 unique pairs x 8 repetitions = 96 trials. No pair is degenerate: every pair has at least one discriminating cue and the two theories differ in direction, in confidence level, or in the shape of the within-subject gradient across the ladder.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** **Threshold-terminated compensatory integration with CONCLUSION-GRADED commitment.**

People treat expert ratings as evidence to be added up, not as a lexicographic hierarchy. Five claims:

1. **Cue weights depend ONLY on the communicated validity, never on where the rating sits on the screen.** The weight of expert j is a compressed, monotone function of its stated validity, w_j ∝ (v_j − 0.5)^rho, normalised to sum to 1. Compression (rho ≈ 1.25–1.45) lets a crowd of medium-validity experts out-vote or tie a single .95 expert.

2. **Integration is compensatory and additive, carried out incrementally in the order the ratings are displayed, and it can terminate early.** After each rating that actually discriminates, the decision maker performs a *best-case reversal check*: "could any single expert I have not yet read still overturn what I now believe?" The sufficiency bar is theta_k = phi · max_{j after the current screen position} w_j, and the probability of stopping is sigmoid(s·(|E_k|/theta_k − 1)). With phi ≈ 1 this is the rational rule "stop when no single remaining expert can reverse my lead". Nothing is discounted by position; position matters *only* through this truncation.

3. **NEW — confidence is conclusion-graded, not merely margin-graded.** Two psychologically distinct states can end a trial. (a) The reversal check *passes*: the person has reached a categorical conclusion ("nothing still unread can overturn this"), and choice is committed sharply — softmax on the raw margin with an amplified sharpness beta·kappa (kappa > 1). Crucially, *which* cue triggered the stop is then almost irrelevant: a mid-validity cue that has already cleared the sufficiency bar is followed about as decisively as the top cue is. (b) The list is *exhausted* without the check ever passing: the person never reached a conclusion and is left holding a raw, ungrounded residual margin; choice is then a shallow softmax on that margin with the ordinary beta, so unresolved trials sit close to chance and are only weakly tilted by the residual. Two trials with identical final margins therefore differ in decisiveness depending on whether the stopping criterion was ever met — the theory's sharpest new prediction.

4. **No renormalisation, no side bias, plus a lapse.** Evidence is never divided by the total discriminating weight (that would make one-cue and many-cue decisions equally confident); a lapse epsilon mixes in a uniform guess.

5. **Individual differences are unimodal jitter** in compression (rho), conservatism (phi), stopping sharpness (s), commitment amplification (kappa), residual sensitivity (beta) and lapse — no discrete strategy types, no primacy/recency "reader types".

Signature predictions: in a validity-sorted display the leading cue clears the bar immediately, search stops, and behaviour looks like Take-The-Best in BOTH conflict and agreement cells — and because stopping commitment is amplified, cells whose first cues tie and which stop on a *middle* cue are followed just as decisively, so the TTB-vs-tally contrast collapses to ≈0 instead of going negative. When the best expert is listed last, the bar stays high until a mid-validity crowd piles past it, search stops before the best expert is read, and behaviour looks like tallying. When a strong cue sits mid-list with another strong cue still unread, the check fails, integration runs to the end, and the shallow residual softmax leaves the choice near chance.

**Parameters:**
- rho: [1.25, 1.45]
- phi: [1.00, 1.12]
- s: [12.0, 30.0]
- beta: [5.0, 10.0]
- kappa: [3.0, 5.0]
- epsilon: [0.0, 0.06]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ------------------------------------------------------------------
    # 1. parse the trial stimulus into two rating vectors (screen order)
    # ------------------------------------------------------------------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(list(state['option_a_ratings']), dtype=float)
            b = np.asarray(list(state['option_b_ratings']), dtype=float)
    if a is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(2, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(2, -1)
        if arr.shape[0] != 2:
            arr = arr.reshape(2, -1)
        a = arr[0].astype(float)
        b = arr[1].astype(float)

    a = np.ravel(a).astype(float)
    b = np.ravel(b).astype(float)
    n = int(min(a.shape[0], b.shape[0]))
    if n == 0:
        return np.array([0.5, 0.5])
    a = a[:n]
    b = b[:n]
    d = a - b

    # ------------------------------------------------------------------
    # 2. validity-only cue weights, compressed toward equality
    # ------------------------------------------------------------------
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        # fall back: assume display is validity-sorted, descending
        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])
    val = np.clip(val, 0.5 + 1e-6, 1.0)

    rho = float(parameters.get('rho', 1.35))
    rho = float(np.clip(rho, 0.1, 5.0))
    raw = np.clip(val - 0.5, 1e-9, None) ** rho
    tot = float(np.sum(raw))
    if not np.isfinite(tot) or tot <= 0:
        w = np.ones(n, dtype=float) / n
    else:
        w = raw / tot                      # weights sum to 1, position-free

    phi = float(np.clip(parameters.get('phi', 1.05), 0.05, 3.0))
    s = float(np.clip(parameters.get('s', 20.0), 0.1, 200.0))
    beta = float(np.clip(parameters.get('beta', 7.5), 0.0, 200.0))
    kappa = float(np.clip(parameters.get('kappa', 4.0), 1.0, 20.0))
    eps = float(np.clip(parameters.get('epsilon', 0.0), 0.0, 1.0))

    # ------------------------------------------------------------------
    # 3. incremental integration in screen order with graded stopping
    #    sufficiency bar = phi * (largest weight among cues NOT YET READ)
    #    i.e. "can any single remaining expert still overturn my lead?"
    #    Commitment is CONCLUSION-GRADED: a stop that passes the check is
    #    read out with sharpness beta*kappa; an exhausted list (no
    #    conclusion ever reached) is read out with the shallow beta.
    # ------------------------------------------------------------------
    disc = np.nonzero(np.abs(d) > 1e-12)[0]     # discriminating cues, in order
    if disc.size == 0:
        return np.array([0.5, 0.5])

    steps = w[disc] * np.sign(d[disc])
    cum = np.cumsum(steps)                      # margin after each new reason
    m = cum.shape[0]

    def _sig(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    p_a = 0.0
    remain = 1.0
    for k in range(m):
        E = float(cum[k])
        if k < m - 1:
            idx = int(disc[k])
            if idx + 1 < n:
                suffix_max = float(np.max(w[idx + 1:]))
            else:
                suffix_max = 0.0
            theta_k = max(phi * suffix_max, 1e-12)
            q = _sig(s * (abs(E) / theta_k - 1.0))   # graded termination
            p_choice = _sig(beta * kappa * E)        # committed conclusion
        else:
            q = 1.0                                  # nothing left to read
            p_choice = _sig(beta * E)                # unresolved residual
        p_a += remain * q * p_choice
        remain *= (1.0 - q)
        if remain <= 1e-12:
            break
    if remain > 1e-12:                           # numerical remainder
        p_a += remain * _sig(beta * float(cum[-1]))

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = (1.0 - eps) * p + eps * 0.5              # lapse, no side bias
    p = np.clip(p, 1e-12, None)
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
        probs = np.ones_like(probs) / len(probs)
    else:
        probs = probs / s
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** **Serial-position–driven, primacy/recency-graded evidence accumulation ("read-order scanning").**

People do not re-sort the expert ratings by the stated validities before deciding. They inspect the ratings in the order in which they appear on the screen (reading order) and their attention decays sharply across that inspection sequence, so the evidence carried by a cue is discounted geometrically by its *serial position*, not by its validity. Stated validities exert only a weak, partial pull on the inspection order (parameter gamma, small).

Two further claims:

1. **The gradient is steep but graded.** Weights fall off as w_j = exp(-lambda * rank_j) with |lambda| well above zero, so decisions are usually driven by the first cue(s) that discriminate — near-lexicographic but not strictly all-or-none: when the leading cues tie, the next-inspected cues take over, and when several later cues line up they can occasionally overturn a leading cue if the gradient is shallow.

2. **The gradient has a sign: most readers are primacy-oriented, a substantial minority are recency-oriented.** Reading a short list produces either an anchoring-on-the-first-item strategy (primacy, lambda > 0) or a last-impression-counts strategy (recency, lambda < 0). The population is bimodal: about one third of subjects anchor on the *last-read* cue. Near-zero gradients (i.e., genuine equal-weight tallying) are essentially absent — people do not integrate all six/five cues equally.

Evidence is evaluated *relatively*: the signed weighted difference is normalised by the total weight of the cues that actually discriminate, so a decision based on one early cue is as confident as a decision based on many, and only genuinely near-balanced weighted evidence produces hesitation. Choice is a softmax on this relative evidence with inverse temperature beta, plus a lapse epsilon.

Because validity order and screen order coincide in a validity-sorted display but dissociate in a scrambled display, this theory predicts TTB-looking behaviour in the former and tally-looking behaviour in the latter — without either heuristic being the actual mechanism. The causal driver is *where the cue sits on the screen*.

**Parameters:**
- lam: [1.2, 3.5]
- gamma: [0.0, 0.25]
- beta: [4.0, 12.0]
- epsilon: [0.0, 0.10]
- scan_orientation: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---- unpack the trial stimulus -> (2, n_features) ------------------
    if isinstance(state, dict):
        a = np.asarray(list(state.get('option_a_ratings', [])), dtype=float)
        b = np.asarray(list(state.get('option_b_ratings', [])), dtype=float)
        stim = np.vstack([a, b])
    else:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            stim = stim.reshape(2, -1)
        elif stim.ndim > 2:
            stim = stim.reshape(2, -1)
    if stim.shape[0] != 2:
        stim = stim.reshape(2, -1)

    a = stim[0].astype(float)
    b = stim[1].astype(float)
    n = a.shape[0]
    d = a - b

    # ---- inspection ranks: screen position, weakly pulled by validity --
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        # fallback: assume the display is already validity-sorted
        val = np.linspace(0.95, 0.55, n)

    order = np.argsort(-val, kind='stable')       # 0 = highest validity
    vrank = np.empty(n, dtype=float)
    vrank[order] = np.arange(n, dtype=float)
    pos = np.arange(n, dtype=float)               # reading-order index

    gamma = float(parameters.get('gamma', 0.0))
    gamma = min(max(gamma, 0.0), 1.0)
    rank = gamma * vrank + (1.0 - gamma) * pos

    # ---- signed attention gradient (primacy vs recency reader) --------
    lam_mag = float(parameters.get('lam', 2.0))
    orient = float(parameters.get('scan_orientation', 1.0))
    RECENCY_SHARE = 0.35                          # population constant
    lam = -lam_mag if orient < RECENCY_SHARE else lam_mag

    z = -lam * (rank - float(np.mean(rank)))      # centered for stability
    z = z - float(np.max(z))
    w = np.exp(z)

    # ---- relative (normalised) weighted evidence ----------------------
    num = float(np.sum(w * d))
    den = float(np.sum(w * np.abs(d)))
    if den <= 1e-12:
        S = 0.0                                   # nothing discriminates
    else:
        S = num / den                             # in [-1, 1]

    beta = float(parameters.get('beta', 6.0))
    eps = float(parameters.get('epsilon', 0.0))
    eps = min(max(eps, 0.0), 1.0)

    logits = np.array([beta * S, 0.0], dtype=float)
    logits = logits - np.max(logits)
    e = np.exp(logits)
    p = e / np.sum(e)

    p = (1.0 - eps) * p + eps * 0.5
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
        probs = np.ones_like(probs) / len(probs)
    else:
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    hits = 0\n    tot = 0\n    for _, row in data.iterrows():\n        try:\n            a = np.asarray([int(v) for v in row['option_a_ratings']], dtype=float)\n            b = np.asarray([int(v) for v in row['option_b_ratings']], dtype=float)\n        except Exception:\n            continue\n        if a.shape[0] != 5 or b.shape[0] != 5:\n            continue\n        d = a - b\n        # target cell: d = +-[1,-1,-1,1,0]  (pairs 1 and 2)\n        pattern_ok = (\n            abs(d[4]) < 1e-9\n            and abs(abs(d[0]) - 1.0) < 1e-9\n            and abs(abs(d[1]) - 1.0) < 1e-9\n            and abs(abs(d[2]) - 1.0) < 1e-9\n            and abs(abs(d[3]) - 1.0) < 1e-9\n            and abs(d[0] - d[3]) < 1e-9\n            and abs(d[1] - d[2]) < 1e-9\n            and abs(d[0] + d[1]) < 1e-9\n        )\n        if not pattern_ok:\n            continue\n        try:\n            resp = int(row['response'])\n        except Exception:\n            continue\n        if resp not in (0, 1):\n            continue\n        # side supported by expert positions 0 and 3 (the first- and last-\n        # discriminating cues); opposed by positions 1 and 2\n        favored_is_a = d[0] > 0\n        chose_favored = (resp == 0) if favored_is_a else (resp == 1)\n        hits += 1 if chose_favored else 0\n        tot += 1\n\n    if tot == 0:\n        return float('nan')\n    return float(hits) / float(tot)\n",
  "rationale": "This isolates the 'both-orientations-agree' cell (trial pairs 1 and 2: d = +-[1,-1,-1,1,0], with the 90% expert tied) and scores the proportion of those trials on which the subject chose the option supported by screen positions 0 and 3 rather than positions 1 and 2.\n\nWhy it discriminates: under read-order scanning, the earliest discriminating cue (position 0) AND the latest discriminating cue (position 3) both favour the SAME option, so primacy readers and recency readers alike produce normalised evidence S strongly favouring that side (|S| ~ .45-.94, P ~ .97-1.00). There is no orientation escape hatch, so this cell is tight both within and across the competitor's whole parameter box, including its bimodal lambda sign - the classic source of between-subject variance for that theory is neutralised here.\n\nUnder the advocated threshold-terminated integrator, no partial margin ever reaches the sufficiency bar (max partial |E| = .235 < phi * w4 ~ .38), so the list is exhausted, the trial ends unresolved, and choice is a shallow softmax on a residual margin of w0 - w1 - w2 + w3 ~ -.06, i.e. a mild preference for the OPPOSITE (positions 1-2) side: expected value ~ .35-.45.\n\nSo the predicted values are roughly .38 (advocated) vs .98 (competitor) - opposite sides of chance, with small per-subject binomial noise over 16 trials per subject. The direction of the effect (and not merely its size) is diagnostic, and the metric is well-defined on both the pooled frame and a single subject's slice."
}
```

## Usage

```json
{
  "input_tokens": 10323,
  "output_tokens": 2964
}
```
