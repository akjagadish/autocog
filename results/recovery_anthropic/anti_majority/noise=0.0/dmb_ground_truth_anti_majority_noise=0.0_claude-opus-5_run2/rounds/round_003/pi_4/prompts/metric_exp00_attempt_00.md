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
**Validities (n_features=6):** [0.92, 0.86, 0.78, 0.68, 0.6, 0.54]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 2: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 3: A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  trial 4: A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  trial 5: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  trial 6: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  trial 8: A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  trial 9: A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 1, 1, 0]
  trial 10: A=[1, 1, 0, 1, 1, 0]  B=[0, 0, 1, 0, 0, 1]
  trial 11: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 12: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 13: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 14: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 15: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 16: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]

**Rationale:** I advocate FC-LEX (sign-only, validity-blind flag counting with a lexicographic tiebreak) against WC-ST (Weber-scaled flag counting with a threshold-like Weibull psychometric plus discrete expert triage). The two models agree completely on POLARITY (a '1' is a warning flag, fewer flags wins) and on VALIDITY-BLINDNESS of the count, and they use the identical severity/lexicographic tiebreak on exact ties. So direction-only tests are useless here. The whole design is built on the ONE place they disagree: FC-LEX says only the SIGN of the count difference matters (one flat plateau accuracy on every unequal-count trial), while WC-ST says discriminability is ratio-scaled, s=(c_hi-c_lo)/sqrt(c_hi^2+c_lo^2+1), pushed through an accelerating Weibull, so accuracy must collapse to chance when the count ratio is small.

Six experts with well-spread validities (0.92, 0.86, 0.78, 0.68, 0.60, 0.54). Exactly two experts fall below the 0.65 'triage' threshold (experts 5 and 6), so WC-ST's minority triage mechanism is well defined and testable. All 8 base pairs appear in both A/B mirror orders (16 unique pairs, K=6 -> 96 trials), so side bias cancels exactly.

CLASS 1 - CONSTANT-DIFFERENCE MARGIN LADDER (P1 0-vs-1, P2 1-vs-2, P3 2-vs-3, P4 3-vs-4). Every pair has a count difference of exactly ONE flag in the same direction (A always carries fewer). FC-LEX is mathematically forced to the SAME accuracy (1-eps)*sigmoid(beta)+eps/2 on all four; no FC-LEX parameter can make them differ. WC-ST's ratio evidence is s = 0.707, 0.408, 0.267, 0.196, which through the Weibull (theta in [0.34,0.46], kappa in [4.5,7.5]) gives roughly 1.00, 0.84, 0.54, 0.51 before lapses: a steep monotone collapse from near-ceiling to essentially chance. A reliable within-subject accuracy drop from 0-vs-1 to 3-vs-4 falsifies FC-LEX; a flat plateau across the ladder cannot be produced by WC-ST at any theta/kappa, because its psychometric function is strictly increasing in s.

CLASS 2 - CONSTANT-SMALLER-COUNT CONTRAST (P3 2-vs-3 versus P5 2-vs-4). Same smaller count, different difference. FC-LEX: identical accuracy. WC-ST: s = 0.267 vs 0.436, i.e. near-chance versus ~0.90 core. This is a second, orthogonal magnitude test that shares direction between models, so no strategy-reversal confound can explain a gradient.

CLASS 3 - TRIAGE REVERSAL (P6: A=[1,1,0,0,0,0] two flags on the 92%/86% experts vs B=[0,0,0,1,1,1] three flags on the 68%/60%/54% experts). Counting subjects choose A (2<3); WC-ST triage subjects discard experts 5-6 and read 2-vs-1, choosing B, and n_drop=1 subjects read 2-vs-2 and fall to the severity tiebreak, which also pushes them toward B. FC-LEX must choose A at its usual flat rate for every subject. Bimodal between-subject dispersion, or a group mean pulled to/below chance on P6 while P1 stays near ceiling, is uniquely WC-ST.

CLASS 4 - EQUAL-COUNT PAIRS (P7: [0,0,0,0,1,1] vs [1,1,0,0,0,0], 2-vs-2 with the flags on opposite ends of the validity scale; P8: [1,0,0,0,0,0] vs [0,1,0,0,0,0], 1-vs-1 control). Both models are silent on counting and fall back on the top discriminating expert, so P8 cleanly identifies tie_bias for both. P7 is diagnostic only for WC-ST's triage subgroup (dropping experts 5-6 turns it into 0-vs-2, a decisive A choice), whereas FC-LEX must give exactly the same p(top-flag option)=tie_bias on P7 and P8. A systematically more extreme response on P7 than P8, in a subset of subjects, is a triage signature.

Ladder pairs P1-P3 carry no flags on the droppable experts (triage-immune), and P4-P5 are constructed so that triage preserves the direction of the count difference, so the magnitude ladder is not confounded by the triage mechanism.

PREDICTIONS. FC-LEX: one flat, above-chance accuracy toward the fewer-flag option on P1-P6 regardless of whether the comparison is 0-vs-1 or 3-vs-4, plus the same magnitude-invariant tie_bias on P7 and P8. WC-ST: a steep monotone ratio-driven gradient (P1 near ceiling > P5 > P2 >> P3 ~ P4 ~ chance), and a bimodal, partly reversed response on P6 with an accompanying P7-vs-P8 asymmetry.

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** **Flag Counting with Lexicographic Tiebreak (FC-LEX).**

Subjects in this task do not read a binary expert rating of '1' as an endorsement; they read it as a *warning flag*. And, crucially, they do not weight those flags by the validities they were told about in the instructions. On every trial the subject performs a single, extremely cheap operation: count how many 1s each option carries, and reject the option carrying more flags.

Three claims:

1. **Negative polarity.** A '1' is a strike against an option (a flag/complaint/warning), so an option's *badness* is monotone in its number of 1s. The option with the strictly smaller flag count is chosen.

2. **Validity-blind, magnitude-blind counting.** All experts are counted equally: the validities are acknowledged in the instructions but are not converted into cue weights. Furthermore, only the *sign* of the count difference matters. Choice probability is therefore a single flat accuracy h = (1-eps)*sigmoid(beta) + eps/2 on every trial where the counts differ, regardless of whether the difference is 1 flag or 5 flags, and regardless of which experts contributed the flags. There is no psychometric function over a validity-weighted margin: a 1-vs-0 comparison and a 1-vs-5 comparison produce the same accuracy.

3. **Lexicographic tiebreak on equal counts.** When the two options carry exactly the same number of flags the counting rule is silent. The subject then falls back to a one-reason rule: locate the highest-validity expert who discriminates and let that single expert decide. Because the subject is unsure whether that expert's '1' should attract or repel (the counting rule just told them 1s are bad, but the tiebreak is invoked precisely because badness is balanced), the direction of this tiebreak is a free subject-level bias tie_bias, which sits near chance and is mildly repelled-by-1 on average. Equal-count pairs therefore yield near-chance performance, not the strongly graded reversals a weighted-mass model predicts.

The theory is deliberately non-compensatory in *magnitude* while being fully compensatory in *counting*: it agrees with a weighted-evidence account on every design tested so far (because in those designs the option with the single top-validity 1 also happens to carry the fewest flags), but it diverges sharply on equal-count pairs such as [1,0,0,0,0] vs [0,0,0,0,1], on pairs where more 1s coincides with smaller weighted mass, and on margin ladders, where FC-LEX predicts a flat accuracy plateau conditional on the sign of the count difference.

**Parameters:**
- beta: [1.3, 2.7]
- epsilon: [0.05, 0.25]
- tie_bias: [0.10, 0.70]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---------- 1. parse the stimulus into two rating vectors ----------
    a = b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            if stim.size % 2 != 0:
                return np.ones(2) / 2.0
            half = stim.size // 2
            a, b = stim[:half], stim[half:]
        else:
            stim = stim.reshape(stim.shape[0], -1)
            if stim.shape[0] < 2:
                return np.ones(2) / 2.0
            a, b = stim[0], stim[1]

    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or a.size != b.size:
        return np.ones(2) / 2.0
    n_features = a.size

    beta = float(parameters["beta"])
    eps = float(parameters["epsilon"])
    tie_bias = float(parameters["tie_bias"])
    eps = min(max(eps, 0.0), 1.0)
    tie_bias = min(max(tie_bias, 0.0), 1.0)

    # ---------- 2. validity-blind flag count (negative polarity) ----------
    count_a = float(np.sum(a))
    count_b = float(np.sum(b))
    d = count_a - count_b   # positive => A carries MORE warning flags => reject A

    if abs(d) > 1e-9:
        # Sign-only rule: magnitude of the count difference is ignored.
        winner = 1 if d > 0 else 0   # fewer flags wins
        s = 1.0 / (1.0 + np.exp(-beta))   # sigmoid(beta * sign) for the winner
        p_core = np.zeros(2, dtype=float)
        p_core[winner] = s
        p_core[1 - winner] = 1.0 - s
    else:
        # ---------- 3. lexicographic tiebreak on equal flag counts ----------
        val = parameters.get("validities", None)
        try:
            val = np.asarray(val, dtype=float).ravel()
        except Exception:
            val = np.array([])
        if val.size != n_features:
            if n_features == 1:
                val = np.array([0.9])
            else:
                val = np.linspace(0.9, 0.55, n_features)

        order = np.argsort(-val, kind="stable")
        top1_owner = None   # index of the option carrying the '1' on the top discriminating cue
        for j in order:
            if a[j] > b[j]:
                top1_owner = 0
                break
            if b[j] > a[j]:
                top1_owner = 1
                break

        if top1_owner is None:
            p_core = np.ones(2) / 2.0
        else:
            p_core = np.zeros(2, dtype=float)
            p_core[top1_owner] = tie_bias
            p_core[1 - top1_owner] = 1.0 - tie_bias

    if not np.all(np.isfinite(p_core)):
        p_core = np.ones(2) / 2.0

    # ---------- 4. attention lapse ----------
    p = (1.0 - eps) * p_core + eps * (np.ones(2) / 2.0)
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
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        probs = np.ones(len(probs)) / len(probs)
    else:
        probs = probs / total
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** **Weber Counting with Severity Tiebreak and Expert Triage (WC-ST).**

A binary expert rating of '1' is read as a *warning flag*, not an endorsement, so an option's badness grows with the number of 1s it carries and the subject picks the option that seems to carry FEWER flags. Three mechanistic claims, plus one heterogeneity claim:

1. **Negative polarity, validity-blind numerosity.** The decision variable is the raw *count* of flags per option, not a validity-weighted mass. Which experts contributed the flags is irrelevant to the count; the stated validities are acknowledged but never converted into cue weights. This is what makes the model look anti-TTB on dominance/agreement pairs and pro-TTB on conflict pairs, with a single mechanism.

2. **Scalar variability (Weber noise) on the counts.** Perceived numerosity is noisy with noise that scales with magnitude, so discriminability depends on the RATIO-scaled difference, not the raw difference:  s = (c_hi - c_lo) / sqrt(c_hi^2 + c_lo^2 + 1). The small additive constant 1 keeps 0-vs-1 easy. Crucially, the psychometric function over s is *threshold-like* (Weibull with slope exponent kappa > 1), i.e. accelerating near zero: comparisons whose count ratio falls below the subject's Weber threshold theta stay essentially at chance, while comparisons above it jump quickly to a lapse-limited ceiling. This is what produces near-chance 2-vs-3 (~0.53) at the same time as high 1-vs-2 (~0.73), 2-vs-4 (~0.78) and near-ceiling 0-vs-1 / 1-vs-4 / 0-vs-5 (~0.85) — a pattern that neither a flat sign-only rule (FC-LEX) nor a linear/Gaussian margin rule can generate.

3. **Severity tiebreak on exact count ties.** When the two options carry exactly the same number of flags the counting mechanism is silent. The subject then asks which single flag is *worst*: the most reliable expert that discriminates decides, and the option carrying THAT expert's flag is (usually) rejected. The strength/direction of this fallback is a subject-level bias tie_bias = P(choose the option flagged by the top discriminating expert), sitting below 0.5 (mildly repelled). Because this rule keys on the top discriminating expert in exactly the same way for every equal-count pair, it predicts *no* differential across equal-count pairs that share the same top discriminating expert.

4. **Discrete expert triage (minority).** A minority of subjects do not count everyone: they apply an all-or-none trust filter, ignoring the one or two least-valid experts entirely before counting (never a graded weighting — gradedness is ruled out by the equal-count data). Triage subjects therefore *reverse* on pairs whose flag surplus lives entirely on the discarded experts (e.g. 2 high-validity flags vs 3 low-validity flags), which injects genuine bimodal between-subject dispersion exactly where the human data show it, and creates the small positive asymmetry among equal-count pairs whose tie is broken by triage.

WC-ST is a sharp competitor to sign-only flag counting: they agree on polarity and validity-blindness but disagree about magnitude invariance. WC-ST predicts a monotone *ratio-driven* accuracy gradient (0v1 > 1v2 > 2v3 > 3v4 at constant difference; 0v1 < 0v2 < 0v4 at constant smaller count), whereas the sign-only rule predicts one flat plateau.

**Parameters:**
- theta: [0.34, 0.46]
- kappa: [4.5, 7.5]
- epsilon: [0.12, 0.48]
- tie_bias: [0.05, 0.45]
- ignore_prone: [0, 1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np

    # ---------------- 1. parse stimulus ----------------
    a = b = None
    if isinstance(state, dict):
        if "option_a_ratings" in state and "option_b_ratings" in state:
            a = np.asarray(state["option_a_ratings"], dtype=float).ravel()
            b = np.asarray(state["option_b_ratings"], dtype=float).ravel()
    if a is None:
        stim = np.asarray(state, dtype=float)
        if stim.ndim == 1:
            if stim.size % 2 != 0:
                return np.ones(2) / 2.0
            half = stim.size // 2
            a, b = stim[:half], stim[half:]
        else:
            stim = stim.reshape(stim.shape[0], -1)
            if stim.shape[0] < 2:
                return np.ones(2) / 2.0
            a, b = stim[0], stim[1]
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 0 or a.size != b.size:
        return np.ones(2) / 2.0
    n = a.size

    # ---------------- 2. validities ----------------
    val = parameters.get("validities", None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = np.array([])
    if val.size != n:
        val = np.array([0.9]) if n == 1 else np.linspace(0.9, 0.55, n)

    # ---------------- 3. subject-level parameters ----------------
    theta = float(parameters["theta"])
    kappa = float(parameters["kappa"])
    eps = float(np.clip(float(parameters["epsilon"]), 0.0, 1.0))
    tie_bias = float(np.clip(float(parameters["tie_bias"]), 0.0, 1.0))
    ip = float(np.clip(float(parameters["ignore_prone"]), 0.0, 1.0))
    theta = max(theta, 1e-3)
    kappa = max(kappa, 0.5)

    # ---------------- 4. discrete expert triage (minority of subjects) ----
    # most subjects count every expert; a minority discard the 1 (or 2)
    # least-valid experts outright before counting.
    if ip < 0.84:
        n_drop = 0
    elif ip < 0.96:
        n_drop = 1
    else:
        n_drop = 2
    mask = np.ones(n, dtype=bool)
    if n_drop > 0 and n > 2:
        order_asc = np.argsort(val, kind="stable")   # ascending validity
        dropped = 0
        for idx in order_asc:
            if dropped >= n_drop or (n - dropped) <= 2:
                break
            if val[idx] < 0.65:                       # only near-chance experts
                mask[idx] = False
                dropped += 1
    af = a[mask]
    bf = b[mask]
    if af.size == 0:
        af, bf, mask = a, b, np.ones(n, dtype=bool)

    # ---------------- 5. Weber-noisy numerosity comparison --------------
    ca = float(np.sum(af))
    cb = float(np.sum(bf))

    if abs(ca - cb) > 1e-9:
        c_hi = max(ca, cb)
        c_lo = min(ca, cb)
        denom = np.sqrt(c_hi * c_hi + c_lo * c_lo + 1.0)
        s = (c_hi - c_lo) / denom               # ratio-scaled evidence
        x = (s / theta) ** kappa
        x = float(np.clip(x, 0.0, 50.0))
        core = 1.0 - 0.5 * np.exp(-x)           # Weibull psychometric fn
        core = float(np.clip(core, 0.5, 1.0))
        winner = 0 if ca < cb else 1            # fewer flags is chosen
        p_core = np.empty(2, dtype=float)
        p_core[winner] = core
        p_core[1 - winner] = 1.0 - core
    else:
        # ------------- 6. severity tiebreak on exact count ties ----------
        idxs = np.where(mask)[0]
        owner = None
        order = idxs[np.argsort(-val[idxs], kind="stable")]
        for j in order:
            if a[j] > b[j]:
                owner = 0
                break
            if b[j] > a[j]:
                owner = 1
                break
        if owner is None:                        # fall back to full cue set
            order_all = np.argsort(-val, kind="stable")
            for j in order_all:
                if a[j] > b[j]:
                    owner = 0
                    break
                if b[j] > a[j]:
                    owner = 1
                    break
        if owner is None:
            p_core = np.ones(2) / 2.0
        else:
            p_core = np.empty(2, dtype=float)
            p_core[owner] = tie_bias             # option carrying the worst flag
            p_core[1 - owner] = 1.0 - tie_bias

    if not np.all(np.isfinite(p_core)):
        p_core = np.ones(2) / 2.0

    # ---------------- 7. attention lapse ----------------
    p = (1.0 - eps) * p_core + eps * (np.ones(2) / 2.0)
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
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        probs = np.ones(len(probs)) / len(probs)
    else:
        probs = probs / total
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n\n    hits = 0\n    tot = 0\n    for _, row in data.iterrows():\n        try:\n            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()\n            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()\n        except Exception:\n            continue\n        if a.size == 0 or a.size != b.size:\n            continue\n        n = a.size\n        ca = float(a.sum())\n        cb = float(b.sum())\n        d = ca - cb\n        # keep only constant-difference-of-one comparisons with a LARGE base count\n        if abs(abs(d) - 1.0) > 1e-9:\n            continue\n        if min(ca, cb) < 2.0 - 1e-9:\n            continue\n        # drop triage-sensitive pairs: the sign of the count difference must be\n        # preserved when the two least-valid experts are discarded\n        k = n - 2\n        if k < 1:\n            continue\n        dh = float(a[:k].sum()) - float(b[:k].sum())\n        if abs(dh) < 1e-9:\n            continue\n        if np.sign(dh) != np.sign(d):\n            continue\n        winner = 0 if d < 0 else 1  # fewer flags wins\n        try:\n            resp = int(row['response'])\n        except Exception:\n            continue\n        if resp not in (0, 1):\n            continue\n        tot += 1\n        if resp == winner:\n            hits += 1\n\n    if tot == 0:\n        return 0.5\n    return float(hits) / float(tot)",
  "rationale": "The design's core disagreement is magnitude invariance. I restrict scoring to the two 'hard' constant-difference-of-one pairs where the base count is large (2-vs-3 in trials 5/6 and 3-vs-4 in trials 7/8), and I explicitly exclude the triage pair (2 high-validity flags vs 3 low-validity flags) by requiring that the sign of the count difference survives dropping the two least-valid experts. This leaves 24 clean trials per subject.\n\nFC-LEX (advocated) is mathematically forced to its single flat plateau accuracy h = (1-eps)*sigmoid(beta)+eps/2 on these trials, exactly as on 0-vs-1: with beta in [1.3,2.7] and eps in [0.05,0.25] this lands around 0.78-0.85 with small per-subject spread (binomial sd over 24 trials ~0.08).\n\nWC-ST predicts ratio-scaled evidence s = 0.267 (2v3) and 0.196 (3v4), which through its accelerating Weibull (theta in [0.34,0.46], kappa in [4.5,7.5]) gives cores of ~0.51-0.64, and its much larger lapse range (eps up to 0.48) compresses this further toward chance: roughly 0.50-0.58. Excluding the triage pair removes the bimodal reversal subgroup, so WC-ST's per-subject variance also stays tight (it is pinned near the chance floor).\n\nExpected contrast: ~0.81 (FC-LEX) vs ~0.53 (WC-ST), a ~0.28 gap with per-subject sd ~0.10 on both sides, easily significant at N=25. It is not trivially true for my theory: if humans show the ratio-driven collapse, this metric will read near chance and falsify FC-LEX."
}
```

## Usage

```json
{
  "input_tokens": 11217,
  "output_tokens": 3928
}
```
