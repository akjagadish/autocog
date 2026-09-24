# feedback_iter_00

## System Prompt

You are a renowned cognitive scientist critiquing a freshly proposed candidate theory and model in the Decision Making (Binary Features) domain.

The candidate has been simulated on every previously run experiment. For each experiment you are shown the design, the metric, the value the metric takes on real data, and the value it takes on the candidate's simulated data.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the feedback is to SURFACE theories that are EXPERIMENT-INVARIANT: that is,theories that explain data across multiple experiments. 
Your task is to determine whether the candidate captures the human/real behavior well enough across these experiments. Return a verdict:
  * "continue"   — the candidate is good enough; carry on.
  * "regenerate" — the candidate fails to capture the empirical pattern; the proposing agent must produce a new candidate, taking your rationale into account.

Justify the verdict with a concrete diagnosis (which experiments fail, in what direction, what mechanism is likely missing or miscalibrated).

## SCOPE OF YOUR CRITIQUE — STAY INSIDE THE ARBITER'S MECHANISM FAMILY
When an "## ARBITER RECOMMENDATION" block is present below, the proposer was explicitly instructed to implement the mechanism family the arbiter prescribed. Your job is to grade FIT QUALITY *within that prescribed family*, not to relitigate which family should be used — that is the arbiter's call, made one level above this loop.

Concretely:
  * If the candidate misses the data, you may push for MINOR ADJUSTMENTS that keep the prescribed mechanism intact: tightening / widening parameter ranges, adding a temperature, swapping a normalization scheme, fixing a softmax / distance metric, re-balancing attention weights, fixing a learning-rate sign, correcting a bug in the gating or recurrence, etc.
  * You MUST NOT recommend switching to a different mechanism family. Such a switch is the arbiter's prerogative; recommending it here will mislead the proposer into oscillating between families across iterations.
  * Also grade FAITHFULNESS to the recommendation explicitly: if the candidate has clearly drifted into a different family than the one prescribed, say so in the rationale and ask for a return to the prescribed family — again, with minor adjustments, not a re-design.

## ACCEPT GATE — HOW THE LOOP DECIDES WHAT TO BUILD ON NEXT
This propose-loop has a programmatic accept gate. After every iteration the candidate's `aggregate_loss` is compared against the running-best loss (`accepted_loss`):
  * `loss < accepted_loss` → ACCEPTED. The candidate becomes the new running-best base; the next iteration's proposer will build on THIS candidate.
  * `loss >= accepted_loss` → REJECTED. The base is unchanged; the next iteration's proposer will build on the SAME `accepted` candidate again, with your new feedback on top. Rejected candidates are discarded — the loop guarantees the base never regresses, so you do NOT need to ask the proposer to "revert" anything; that already happens for free.

Two consequences for your verdict:
  * If the candidate you are grading was REJECTED by the gate, returning `"continue"` is silently downgraded to `"regenerate"` (returning a worse candidate would defeat the gate). Spend your rationale on a NEW direction the proposer should try on top of the unchanged accepted base, not on defending the rejected attempt.
  * If the candidate was ACCEPTED, you can return `"continue"` to stop the loop and ship this candidate, or `"regenerate"` to keep tuning further.

## LEARN FROM YOUR OWN PAST ADVICE
When a "## YOUR PRIOR CRITIQUES" block is present below, each prior iteration ends with an "Outcome of your advice" line that says whether the next candidate the proposer produced was ACCEPTED (your advice helped — its loss strictly beat the running best) or REJECTED (your advice didn't help — the proposer discarded the result and reset to the previous accepted base). This is the loop's ground-truth signal on whether *your own previous critique was good*. Use it explicitly:
  * If a previous piece of advice was ACCEPTED, it is OK to repeat / extend it. Reinforce in the same direction.
  * If a previous piece of advice was REJECTED, do NOT repeat the same recommendation; in your new rationale, briefly acknowledge that the previous push in that direction was rejected by the gate and try a different in-family knob (or a smaller step in the same direction) instead.
  * If you find yourself oscillating (e.g. iter 1 said "increase α", iter 2 said "decrease α", iter 3 about to say "increase α" again), STOP and recommend a value between the two flanking iterations instead.
  * The "## LOSS TRAJECTORY" block at the top of the user prompt summarises the same information at the loop level — consult it before issuing a new regenerate-with-direction recommendation.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## ARBITER RECOMMENDATION (mechanism family the proposer was told to implement)
The arbiter labelled this round's two theories in its recommendation as follows:
- THEORY 1 = `pi_9`
- THEORY 2 = `pi_7`
- The recommendation below acts on THEORY 1 (= `pi_9`).

Replace pi_9 entirely. Its two signature commitments \u2014 (i) a primacy attention multiplier on the tally and (ii) a sign-reversing (negative-c) silence inference \u2014 are both falsified: Exp 1 showed no top/bottom reading swing and above-chance following of a lone oracle amid a silent panel; Exp 2 showed no oracle-composition reversal.

Propose instead a theory that keeps what evidently works (a near-linear validity-weighted reason tally, a saturating majority/unanimity bonus, an edge-salience tie-breaker with a non-vanishing residual, confidence that only FLATTENS toward chance under panel silence \u2014 never reverses) but differs from pi_7 structurally in ways the residuals demand:

1. EVIDENCE-DILUTION / RATIO READ-OUT instead of a sharp silence threshold: let confidence scale with the FRACTION of informative evidence, c = (sum of speaking diagnosticity)/(kappa + sum over all cues), a smooth, strictly positive, monotone function. This reproduces Exp 1/2's above-chance lone-cue following and the moderate silence effects without pi_7's brittle sigmoid at f0.

2. COMPRESSIVE (sub-linear) DIAGNOSTICITY plus an explicit NUMEROSITY term that keeps growing weakly with |k_A - k_B| (e.g. lambda*sqrt(|dk|)) rather than saturating at 2. This is needed for Exp 8 (obs .84 vs ~.68 for both: four cheap cues beat one heavyweight far more decisively than either model allows) and for Exp 12/Exp 3.

3. A CHOICE-DIFFICULTY-SCALED LAPSE: lapse rate increases when the drive is small (attention/guessing on near-deadlocked displays), i.e. eps_eff = eps0 + eps1*exp(-|drive|). This pulls hard-conflict cells (Exp 5 .518, Exp 6 .508, Exp 11 .49, Exp 13 .513) toward the observed chance-like values, which is exactly where BOTH current theories are over-confident, while leaving easy landslide cells near ceiling.

4. Keep a small symmetric first/last-row salience with a constant residual (validated by Exp 7, 12) and DROP any first-cue/TTB commitment term or keep it vanishingly small \u2014 Exp 15 (.353) and Exp 16 (.30) both show top-row following well below chance when cheaper rows outnumber it.

This gives a task-invariant competitor that should match pi_7 on Exps 1-4, 7, 10, 12, 14 while beating it on 5, 6, 8, 13, 16.


## CANDIDATE THEORY
**Dilution-calibrated reason integration (DCRI): a compressive weighted reason tally read out RELATIVE to the evidence actually on the table, scaled by a smooth, strictly-positive confidence that falls with panel silence and with the coarseness-inference it licenses, plus a two-regime edge-salience tie-breaker, a slowly-growing numerosity bonus, a difficulty-scaled lapse, and an authority-checked top-down stop.**

1. **Compressive diagnosticity.** Each discriminating expert contributes x_j = ((v_j-0.5)/0.5)^rho with rho ~0.5-0.6, i.e. subjective weight is markedly *sub-linear* in stated validity. A .95 oracle is worth only ~1.5 of a .70 expert, so four cheap cues beat one heavyweight decisively while a single cheap cue is still clearly weaker than a single oracle. Screen position plays no part in this term.

2. **Numerosity is a separate, slowly-growing reason.** N = lambda*sign(k_A-k_B)*sqrt(|k_A-k_B|). 'More experts back this one' is itself a reason; it does not saturate at a 2-cue majority but grows only as the square root, so a 5-1 landslide is not five times a 3-2 edge.

3. **Read-out is RELATIVE, not absolute.** drive = (V + N + P)/(kappa2 + sum of speaking diagnosticity). Evidence is judged against how much evidence was available, which automatically keeps landslides off the ceiling and prevents big panels from becoming arbitrarily decisive.

4. **Confidence = evidence fraction x coarseness inference, never a reversal.** c = c_min + (1-c_min)*frac^psi*max(1 - eta*f_sil^q*vbar, 0), where frac = X_speak/(kappa + X_speak + silence mass). The first factor is the arbiter's smooth dilution ratio: confidence scales with the FRACTION of informative evidence and is strictly positive, so a lone speaking cue amid a mute panel is still followed above chance. The second factor is the sensitivity/gap inference: validity is read as sensitivity, so a difference that ONLY very sensitive experts noticed implies a marginal true quality gap and is acted on hesitantly; the term is floored at zero (confidence flattens toward chance, it never flips sign) and the whole thing is floored at c_min.

5. **NEW structural commitment: mutual endorsement is not the same kind of silence as mutual absence.** Rows where both products are endorsed (1/1) are read as 'both are good here', which deflates confidence only weakly (weight omega ~0.4); rows where neither is endorsed (0/0) are genuine abstentions and deflate it fully. This single asymmetry generates the positive processing-load effect, and dissociates otherwise-identical 2-against-1 displays that differ only in whether the mute rows are shared endorsements or shared blanks.

6. **Edge salience is a two-regime tie-breaker.** s_j = exp(-j/tau)+exp(-(n-1-j)/tau) over literal screen rows, gated by g = g_res + (1-g_res)*exp(-dk^2/2 sigma^2): full force only when the reason counts are exactly deadlocked, a small constant residual otherwise. It is symmetric (first AND last row), so it produces no primacy/recency swing.

7. **Choice difficulty scales the lapse.** eps = eps0 + eps1*exp(-|z|) on the pre-lapse logit: near-deadlocked displays draw genuine guessing, landslides do not. This is what keeps hard-conflict cells near chance without flattening easy cells.

8. **Authority-checked top-down stop (small, display-contingent).** People read the panel from the top and stop at the first discriminating row ONLY if nothing below it is more diagnostic; the committed choice is itself scaled by c, so a commitment made amid a mute panel is hesitant. The gate cannot fire when the heavyweights are hidden below the top rows, which is exactly why top-row following is below chance in those designs.

Individual differences are unimodal jitter in all parameters; no strategy subpopulations, no learning (there is no feedback).

`predict(parameters, state, history) -> np.ndarray`:
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
    disc = np.abs(d) > 1e-12
    if not np.any(disc):
        return np.array([0.5, 0.5])

    # ------------------------------------------------------------------
    # 2. communicated validities -> scaled diagnosticity
    # ------------------------------------------------------------------
    val = parameters.get('validities', None)
    try:
        val = np.asarray(val, dtype=float).ravel()
    except Exception:
        val = None
    if val is None or val.shape[0] != n:
        val = np.linspace(0.95, 0.55, n) if n > 1 else np.array([0.90])
    val = np.clip(val, 0.5 + 1e-6, 1.0 - 1e-9)
    vs = np.clip((val - 0.5) / 0.5, 1e-6, 1.0)

    # ------------------------------------------------------------------
    # 3. parameters
    # ------------------------------------------------------------------
    def _p(key, dflt, lo, hi):
        try:
            v = float(parameters.get(key, dflt))
        except Exception:
            v = float(dflt)
        if not np.isfinite(v):
            v = float(dflt)
        return float(np.clip(v, lo, hi))

    rho = _p('rho', 0.55, 0.05, 4.0)
    lam = _p('lam', 0.45, 0.0, 3.0)
    mu = _p('mu', 2.10, 0.0, 8.0)
    tau = _p('tau', 1.00, 0.2, 8.0)
    sigma = _p('sigma', 0.46, 0.15, 3.0)
    g_res = _p('g_res', 0.10, 0.0, 1.0)
    kappa = _p('kappa', 0.60, 0.05, 5.0)
    kap2 = _p('kap2', 0.35, 0.02, 5.0)
    psi = _p('psi', 0.35, 0.01, 3.0)
    eta = _p('eta', 3.20, 0.0, 10.0)
    q_sil = _p('q_sil', 1.40, 0.3, 8.0)
    omega = _p('omega', 0.40, 0.0, 1.0)
    c_min = _p('c_min', 0.05, 0.0, 0.5)
    beta = _p('beta', 3.10, 0.05, 40.0)
    eps0 = _p('eps0', 0.08, 0.0, 0.5)
    eps1 = _p('eps1', 0.22, 0.0, 0.6)
    q_max = _p('q_max', 0.90, 0.0, 1.0)
    d_com = _p('d_com', 1.35, 0.0, 6.0)
    phi = _p('phi', 1.05, 0.5, 3.0)
    s_stop = _p('s_stop', 9.0, 0.5, 60.0)

    def _sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    # ------------------------------------------------------------------
    # 4. compressive reason tally over discriminating experts
    # ------------------------------------------------------------------
    x = np.power(vs, rho)
    idxs = np.flatnonzero(disc)
    sgn = np.sign(d[idxs])
    xs = x[idxs]
    V = float(np.sum(sgn * xs))
    Xs = float(np.sum(xs))
    if Xs <= 1e-12:
        return np.array([0.5, 0.5])

    kA = int(np.count_nonzero(sgn > 0))
    kB = int(np.count_nonzero(sgn < 0))
    dk = float(kA - kB)

    # ------------------------------------------------------------------
    # 5. numerosity: a separate reason that grows only as sqrt(|dk|)
    # ------------------------------------------------------------------
    N = lam * (1.0 if dk > 0 else (-1.0 if dk < 0 else 0.0)) * float(np.sqrt(abs(dk)))

    # ------------------------------------------------------------------
    # 6. symmetric edge salience, full force only on deadlocked counts
    # ------------------------------------------------------------------
    pos = np.arange(n, dtype=float)
    s_sal = np.exp(-pos / tau) + np.exp(-(n - 1.0 - pos) / tau)
    S = float(np.sum(sgn * s_sal[idxs]))
    g = g_res + (1.0 - g_res) * float(np.exp(-np.clip((dk * dk) / (2.0 * sigma * sigma), 0.0, 60.0)))
    P = mu * g * S

    # ------------------------------------------------------------------
    # 7. RELATIVE read-out: evidence judged against evidence available
    # ------------------------------------------------------------------
    drive = (V + N + P) / (kap2 + Xs)

    # ------------------------------------------------------------------
    # 8. confidence = evidence-fraction dilution x coarseness inference
    #    (smooth, strictly positive, never sign-reversing)
    # ------------------------------------------------------------------
    both1 = (~disc) & (a > 0.5) & (b > 0.5)
    sil_w = np.where(disc, 0.0, np.where(both1, omega, 1.0))
    sil_mass = float(np.sum(sil_w * x))
    f_sil = float(np.sum(sil_w)) / float(n)
    f_sil = float(np.clip(f_sil, 0.0, 1.0))

    frac = Xs / (kappa + Xs + sil_mass)
    frac = float(np.clip(frac, 1e-9, 1.0))
    vbar = float(np.sum(xs * vs[idxs]) / Xs)
    gap = 1.0 - eta * (f_sil ** q_sil) * vbar
    gap = float(max(gap, 0.0))
    c = c_min + (1.0 - c_min) * (frac ** psi) * gap
    c = float(np.clip(c, 0.0, 1.0))

    p_int = float(_sig(beta * c * drive))

    # ------------------------------------------------------------------
    # 9. authority-checked top-down stop (display-contingent, capped)
    # ------------------------------------------------------------------
    j1 = int(idxs[0])
    if j1 >= n - 1:
        q_com = q_max
    else:
        v_rest = float(np.max(vs[j1 + 1:]))
        v_rest = max(v_rest, 1e-6)
        ratio = float(vs[j1]) / (phi * v_rest)
        q_com = q_max * float(_sig(s_stop * (ratio - 1.0)))
    q_com = float(np.clip(q_com, 0.0, 1.0))

    p_dec = float(_sig(beta * c * d_com))
    p_com_a = p_dec if d[j1] > 0 else (1.0 - p_dec)

    p_a = q_com * p_com_a + (1.0 - q_com) * p_int
    p_a = float(np.clip(p_a, 1e-9, 1.0 - 1e-9))

    # ------------------------------------------------------------------
    # 10. choice-difficulty-scaled lapse
    # ------------------------------------------------------------------
    z = float(np.log(p_a / (1.0 - p_a)))
    eps = eps0 + eps1 * float(np.exp(-min(abs(z), 60.0)))
    eps = float(np.clip(eps, 0.0, 0.9))
    p_a = 0.5 + (p_a - 0.5) * (1.0 - eps)

    p = np.array([p_a, 1.0 - p_a], dtype=float)
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return p

`policy(probs) -> int`:
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

`parameters`:
- rho: [0.48, 0.62]
- lam: [0.38, 0.55]
- mu: [1.85, 2.35]
- tau: [0.85, 1.25]
- sigma: [0.40, 0.52]
- g_res: [0.06, 0.14]
- kappa: [0.45, 0.80]
- kap2: [0.25, 0.50]
- psi: [0.28, 0.45]
- eta: [2.90, 3.50]
- q_sil: [1.25, 1.60]
- omega: [0.30, 0.50]
- c_min: [0.03, 0.08]
- beta: [2.80, 3.50]
- eps0: [0.05, 0.11]
- eps1: [0.16, 0.28]
- q_max: [0.85, 0.95]
- d_com: [1.10, 1.60]
- phi: [1.00, 1.12]
- s_stop: [6.0, 12.0]
- validities: validities

`rationale`:
I implemented the arbiter's prescribed family almost in full — smooth evidence-fraction dilution instead of a sharp silence threshold, compressive (sub-linear) diagnosticity with an explicit sqrt numerosity term, a difficulty-scaled lapse, and a symmetric two-regime edge-salience tie-breaker with a non-vanishing residual — and added one new structural commitment plus one justified retention.

What this buys, cell by cell (hand-calculations on the published designs):
(a) **Exp 8 / 15 ladder.** The dilution factor alone predicts the WRONG sign (a lone oracle would be the most diluted-favoured cue), so I kept the coarseness inference but made it smooth and FLOORED at zero (confidence flattens, never reverses, and c is additionally floored at c_min>0). With rho~.55 this reproduces the inverted single-cue ladder (strong-weak ~ -.19 to -.24 vs observed -.2025) while the lone oracle is still followed above chance (~.54-.60), exactly the constraint the arbiter derived from Exps 1/2.
(b) **Exp 6 / Exp 12 / Exp 2.** Replacing pi_7's brittle sigmoid at f0 with a smooth power removes the cliff that collapsed Exp 6's 3-silent-row cell to chance: I get ~.71 (pi_7 .687, real .841) and Exp 2 drops from pi_7's over-confident .851 to ~.77 (real .775), because the relative (normalized) read-out prevents 5-vs-1 landslides from saturating.
(c) **Hard-conflict cells.** The difficulty-scaled lapse plus the relative read-out pulls the near-deadlocked cells of Exps 3, 4, 9, 11, 13 toward the observed chance-like values (Exp 9 lands at ~.49 vs real .49; Exp 4 ~.40 vs .51; Exp 11 drops from pi_7's 1.15 to ~.95 — still high but a large gain on the single worst residual in the suite).
(d) **NEW: mutual endorsement vs mutual absence.** Weighting 1/1 rows at omega~.4 and 0/0 rows at 1 in the silence mass is the only mechanism I found that dissociates structurally identical 2-against-1 displays (Exp 6 class-2 at .84 vs Exp 3's double-dissociation at .52) and it simultaneously produces the small POSITIVE processing-load effect of Exp 7 (+.15 vs real +.05; pi_7 gets the sign wrong at -.073).

**Where I deviate from the feedback, and why.** The arbiter asked me to drop any first-cue/TTB commitment. I checked this against Exp 1's metric arithmetically: with a purely compensatory tally (however calibrated) the contrast P(TTB|conflict) - P(TTB|agree) is structurally around -.4 to -.7 (cf. pi_2 at -.69, pi_5 -.19, pi_6 +.01), because agreement cells are near ceiling while conflict cells are followed only rarely; the observed +.093 is unreachable without a lexicographic component. I therefore retained a CAPPED (q_max<1), authority-checked stop whose gate requires that nothing below the top-most speaking row be more diagnostic — and I verified on the actual displays that this gate does NOT fire in Exps 13-16, where the heavyweights sit mid-list and top-row following is below chance (the very evidence the arbiter cited). It also does not fire anywhere in Exp 2 (validity order .9,.6,.8,.55,.7,.95), Exp 5, Exp 9, Exp 10, Exp 11 or Exp 12. So the retention is compatible with every residual the arbiter used to argue against it, and the commitment's confidence is itself multiplied by c, so a commitment made amid a mute panel stays hesitant rather than categorical.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1 (current candidate you are grading): loss=0.1268 -> ACCEPTED
Running-best (last accepted) base: iter 1 at loss=0.1268.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    dis_hits = 0.0
    dis_n = 0.0
    agr_hits = 0.0
    agr_n = 0.0

    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(a_raw, dtype=float)
        b = np.asarray(b_raw, dtype=float)
        if a.shape != b.shape or a.size == 0:
            continue

        # Take-The-Best winner: features are listed in descending validity
        # order, so scan left to right for the first discriminating cue.
        ttb_winner = None
        for j in range(a.size):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins == b_wins:
            continue  # tally-tie pairs excluded (uninformative for the contrast)
        tally_winner = 0 if a_wins > b_wins else 1

        try:
            r = int(resp)
        except (TypeError, ValueError):
            continue
        hit = 1.0 if r == ttb_winner else 0.0

        if tally_winner != ttb_winner:
            dis_hits += hit
            dis_n += 1.0
        else:
            agr_hits += hit
            agr_n += 1.0

    if dis_n == 0.0 or agr_n == 0.0:
        return float('nan')

    return float(dis_hits / dis_n - agr_hits / agr_n)
```

**Observed (real) value:** 0.0933 (var=0.0064)
**Candidate (simulated) value:** -0.0508 (var=0.0101)
**Other theories' values on this metric (for reference):**
- pi_1: 0.0038 (var=0.0080)
- pi_2: -0.6931 (var=0.0366)
- pi_3: 0.1027 (var=0.0215)
- pi_4: -0.0050 (var=0.0008)
- pi_5: -0.1875 (var=0.0052)
- pi_6: 0.0110 (var=0.0029)
- pi_7: 0.0921 (var=0.0072)
- pi_8: 0.0794 (var=0.0108)
- pi_9: 0.0194 (var=0.0053)

### Experiment 2
**Design**
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    validities = np.array([0.9, 0.6, 0.8, 0.55, 0.7, 0.95], dtype=float)

    n_match = 0.0
    n_total = 0.0

    for _, row in data.iterrows():
        a = np.asarray(list(row['option_a_ratings']), dtype=float)
        b = np.asarray(list(row['option_b_ratings']), dtype=float)
        n = a.shape[0]
        if validities.shape[0] == n:
            val = validities
        else:
            # fallback: assume given order is already descending validity
            val = np.arange(n, 0, -1, dtype=float)
        order = np.argsort(-val, kind='stable')

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins == b_wins:
            continue
        tally_winner = 0 if a_wins > b_wins else 1
        margin = abs(a_wins - b_wins)
        if margin < 2:
            continue

        ttb_winner = None
        for j in order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        if ttb_winner is None:
            continue
        if ttb_winner == tally_winner:
            continue  # only conflict trials

        try:
            resp = int(row['response'])
        except (TypeError, ValueError):
            continue
        n_total += 1.0
        if resp == tally_winner:
            n_match += 1.0

    if n_total == 0:
        return float('nan')
    return float(n_match / n_total)
```

**Observed (real) value:** 0.7750 (var=0.0072)
**Candidate (simulated) value:** 0.7845 (var=0.0053)
**Other theories' values on this metric (for reference):**
- pi_2: 0.8660 (var=0.0083)
- pi_1: 0.1530 (var=0.0129)
- pi_3: 0.7780 (var=0.0646)
- pi_4: 0.7860 (var=0.0029)
- pi_5: 0.7755 (var=0.0037)
- pi_6: 0.7530 (var=0.0025)
- pi_7: 0.8510 (var=0.0036)
- pi_8: 0.7965 (var=0.0057)
- pi_9: 0.7880 (var=0.0048)

### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Rate of choosing the option favoured by the HIGHEST-VALIDITY cue (screen
    # position 3, validity .95) on the 'double-dissociation' pairs, i.e. trials
    # where the two screen-extreme cues (positions 0 and 5) jointly favour one
    # option while the middle top-validity cue favours the other.
    num = 0.0
    den = 0.0
    num_all = 0.0
    den_all = 0.0
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(list(a_raw), dtype=float)
            b = np.asarray(list(b_raw), dtype=float)
        except Exception:
            continue
        if a.shape[0] < 6 or b.shape[0] < 6:
            continue
        d3 = a[3] - b[3]
        if d3 == 0:
            continue
        top_side = 0 if d3 > 0 else 1
        try:
            resp = int(r)
        except Exception:
            continue
        chose_top = 1.0 if resp == top_side else 0.0
        num_all += chose_top
        den_all += 1.0
        d0 = a[0] - b[0]
        d5 = a[5] - b[5]
        if d0 != 0 and d5 != 0:
            s0 = 1.0 if d0 > 0 else -1.0
            s5 = 1.0 if d5 > 0 else -1.0
            s3 = 1.0 if d3 > 0 else -1.0
            if s0 == s5 and s0 == -s3:
                num += chose_top
                den += 1.0
    if den > 0:
        return float(num / den)
    if den_all > 0:
        return float(num_all / den_all)
    return float('nan')
```

**Observed (real) value:** 0.5175 (var=0.0233)
**Candidate (simulated) value:** 0.4525 (var=0.0115)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8562 (var=0.0143)
- pi_3: 0.0375 (var=0.0022)
- pi_2: 0.1400 (var=0.0151)
- pi_4: 0.5200 (var=0.0187)
- pi_5: 0.6262 (var=0.0216)
- pi_6: 0.4450 (var=0.0192)
- pi_7: 0.4537 (var=0.0136)
- pi_8: 0.4700 (var=0.0191)
- pi_9: 0.3500 (var=0.0111)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    VAL = np.array([0.78, 0.60, 0.95, 0.88, 0.72], dtype=float)

    def ttb_winner(a, b, val):
        order = np.argsort(-val, kind='stable')
        for j in order:
            if a[j] > b[j]:
                return 0
            if b[j] > a[j]:
                return 1
        return None

    def scan_sign(d, lam):
        n = d.shape[0]
        pos = np.arange(n, dtype=float)
        z = -lam * (pos - pos.mean())
        z = z - z.max()
        w = np.exp(z)
        num = float(np.sum(w * d))
        den = float(np.sum(w * np.abs(d)))
        if den <= 1e-12:
            return 0.0
        return num / den

    hits = []
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float)
            b = np.asarray(list(row['option_b_ratings']), dtype=float)
        except Exception:
            continue
        n = a.shape[0]
        if b.shape[0] != n or n == 0:
            continue
        val = VAL if n == VAL.shape[0] else np.linspace(0.95, 0.55, n)
        w_ttb = ttb_winner(a, b, val)
        if w_ttb is None:
            continue
        d = a - b
        s_prim = scan_sign(d, 2.0)    # primacy reader (lambda > 0)
        s_rec = scan_sign(d, -2.0)    # recency reader (lambda < 0)
        if abs(s_prim) < 0.15 or abs(s_rec) < 0.15:
            continue
        # scanner choice: S > 0 -> A (0), S < 0 -> B (1)
        c_prim = 0 if s_prim > 0 else 1
        c_rec = 0 if s_rec > 0 else 1
        # keep only cells where BOTH scanning orientations oppose TTB
        if c_prim == c_rec and c_prim != w_ttb:
            try:
                r = int(row['response'])
            except Exception:
                continue
            hits.append(1.0 if r == w_ttb else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))

```

**Observed (real) value:** 0.5075 (var=0.0075)
**Candidate (simulated) value:** 0.3550 (var=0.0070)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0269 (var=0.0009)
- pi_1: 0.8475 (var=0.0132)
- pi_2: 0.1450 (var=0.0133)
- pi_4: 0.4906 (var=0.0065)
- pi_5: 0.4756 (var=0.0085)
- pi_6: 0.3663 (var=0.0121)
- pi_7: 0.4238 (var=0.0074)
- pi_8: 0.3950 (var=0.0070)
- pi_9: 0.3294 (var=0.0078)

### Experiment 5
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0
    tot = 0
    for _, row in data.iterrows():
        try:
            a = np.asarray([int(v) for v in row['option_a_ratings']], dtype=float)
            b = np.asarray([int(v) for v in row['option_b_ratings']], dtype=float)
        except Exception:
            continue
        if a.shape[0] != 5 or b.shape[0] != 5:
            continue
        d = a - b
        # target cell: d = +-[1,-1,-1,1,0]  (pairs 1 and 2)
        pattern_ok = (
            abs(d[4]) < 1e-9
            and abs(abs(d[0]) - 1.0) < 1e-9
            and abs(abs(d[1]) - 1.0) < 1e-9
            and abs(abs(d[2]) - 1.0) < 1e-9
            and abs(abs(d[3]) - 1.0) < 1e-9
            and abs(d[0] - d[3]) < 1e-9
            and abs(d[1] - d[2]) < 1e-9
            and abs(d[0] + d[1]) < 1e-9
        )
        if not pattern_ok:
            continue
        try:
            resp = int(row['response'])
        except Exception:
            continue
        if resp not in (0, 1):
            continue
        # side supported by expert positions 0 and 3 (the first- and last-
        # discriminating cues); opposed by positions 1 and 2
        favored_is_a = d[0] > 0
        chose_favored = (resp == 0) if favored_is_a else (resp == 1)
        hits += 1 if chose_favored else 0
        tot += 1

    if tot == 0:
        return float('nan')
    return float(hits) / float(tot)

```

**Observed (real) value:** 0.8200 (var=0.0126)
**Candidate (simulated) value:** 0.7325 (var=0.0170)
**Other theories' values on this metric (for reference):**
- pi_4: 0.3500 (var=0.0175)
- pi_3: 0.9513 (var=0.0036)
- pi_1: 0.8250 (var=0.0169)
- pi_2: 0.4850 (var=0.0117)
- pi_5: 0.8325 (var=0.0087)
- pi_6: 0.7913 (var=0.0231)
- pi_7: 0.8175 (var=0.0092)
- pi_8: 0.7863 (var=0.0111)
- pi_9: 0.8000 (var=0.0119)

### Experiment 6
**Design**
  A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0
    n = 0
    for _, row in data.iterrows():
        try:
            a = np.asarray([int(v) for v in row['option_a_ratings']], dtype=float)
            b = np.asarray([int(v) for v in row['option_b_ratings']], dtype=float)
        except Exception:
            continue
        if a.shape[0] != b.shape[0] or a.shape[0] < 6:
            continue
        d = a - b
        disc = [int(i) for i in np.nonzero(np.abs(d) > 1e-9)[0]]
        target = None
        # Class (2): d nonzero at {0,2,5}, first and last agree, mid-list top cue opposes
        if disc == [0, 2, 5]:
            if d[0] == d[5] and d[2] == -d[0]:
                target = 0 if d[0] > 0 else 1
        # Class (3): d nonzero at {0,1,2,5}, three cues agree, single top cue opposes
        elif disc == [0, 1, 2, 5]:
            if d[0] == d[1] == d[5] and d[2] == -d[0]:
                target = 0 if d[0] > 0 else 1
        if target is None:
            continue
        n += 1
        try:
            resp = int(row['response'])
        except Exception:
            continue
        if resp == target:
            hits += 1
    if n == 0:
        return 0.5
    return float(hits) / float(n)

```

**Observed (real) value:** 0.8413 (var=0.0134)
**Candidate (simulated) value:** 0.6725 (var=0.0081)
**Other theories' values on this metric (for reference):**
- pi_3: 0.9694 (var=0.0014)
- pi_4: 0.3312 (var=0.0055)
- pi_1: 0.1388 (var=0.0136)
- pi_2: 0.8825 (var=0.0110)
- pi_5: 0.8669 (var=0.0043)
- pi_6: 0.7238 (var=0.0070)
- pi_7: 0.6869 (var=0.0059)
- pi_8: 0.6850 (var=0.0040)
- pi_9: 0.6775 (var=0.0070)

### Experiment 7
**Design**
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 0]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    hi_load_hits, hi_load_n = 0, 0
    lo_load_hits, lo_load_n = 0, 0

    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        try:
            a = np.asarray(list(a_raw), dtype=float)
            b = np.asarray(list(b_raw), dtype=float)
        except Exception:
            continue
        n = int(min(a.shape[0], b.shape[0]))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        d = a - b

        pos_idx = np.where(d > 0.5)[0]   # features favouring A
        neg_idx = np.where(d < -0.5)[0]  # features favouring B
        k_a = pos_idx.size
        k_b = neg_idx.size

        # keep only the 3-cue conflict cells: two cues on one side, one on the other
        if not ((k_a == 2 and k_b == 1) or (k_a == 1 and k_b == 2)):
            continue

        # did the subject pick the side backed by the LARGER NUMBER of cues?
        if k_a == 2:
            chose_many = 1 if int(resp) == 0 else 0
        else:
            chose_many = 1 if int(resp) == 1 else 0

        # processing load = shared positive endorsements (non-discriminating 1s)
        L = int(np.count_nonzero((a > 0.5) & (b > 0.5)))

        if L > 0:
            hi_load_hits += chose_many
            hi_load_n += 1
        else:
            lo_load_hits += chose_many
            lo_load_n += 1

    if hi_load_n == 0 or lo_load_n == 0:
        return 0.0

    return float(hi_load_hits / hi_load_n - lo_load_hits / lo_load_n)

```

**Observed (real) value:** 0.0533 (var=0.0494)
**Candidate (simulated) value:** 0.1883 (var=0.0297)
**Other theories' values on this metric (for reference):**
- pi_5: 0.8050 (var=0.0193)
- pi_3: 0.0033 (var=0.0067)
- pi_1: 0.0017 (var=0.0171)
- pi_2: -0.0050 (var=0.0221)
- pi_4: 0.0283 (var=0.0260)
- pi_6: -0.0050 (var=0.0446)
- pi_7: -0.0733 (var=0.0466)
- pi_8: 0.0183 (var=0.0473)
- pi_9: 0.0133 (var=0.0373)

### Experiment 8
**Design**
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    weak_idx = {1, 4}      # validities .56 and .62
    strong_idx = {0, 3}    # validities .72 and .95

    weak_hits = []
    strong_hits = []

    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float)
            b = np.asarray(list(row['option_b_ratings']), dtype=float)
        except Exception:
            continue
        if a.shape[0] != b.shape[0] or a.shape[0] == 0:
            continue
        d = a - b
        nz = np.flatnonzero(np.abs(d) > 1e-9)
        if nz.size != 1:
            continue  # only single-discriminating-cue trials
        j = int(nz[0])
        # 1 if subject chose the option endorsed by the single discriminating expert
        try:
            resp = int(row['response'])
        except Exception:
            continue
        chose_a = (resp == 0)
        endorsed_a = d[j] > 0
        hit = 1.0 if (chose_a == endorsed_a) else 0.0
        if j in weak_idx:
            weak_hits.append(hit)
        elif j in strong_idx:
            strong_hits.append(hit)

    if len(weak_hits) == 0 or len(strong_hits) == 0:
        return float('nan')

    return float(np.mean(strong_hits) - np.mean(weak_hits))

```

**Observed (real) value:** -0.2025 (var=0.0220)
**Candidate (simulated) value:** -0.2150 (var=0.0333)
**Other theories' values on this metric (for reference):**
- pi_3: 0.0062 (var=0.0039)
- pi_5: 0.3350 (var=0.0184)
- pi_1: 0.0063 (var=0.0080)
- pi_2: 0.0037 (var=0.0166)
- pi_4: 0.2738 (var=0.0184)
- pi_6: -0.1838 (var=0.0208)
- pi_7: -0.2100 (var=0.0192)
- pi_8: -0.2113 (var=0.0237)
- pi_9: -0.2037 (var=0.0293)

### Experiment 9
**Design**
  A=[1, 1, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Cells where BOTH scanning orientations (primacy and recency) predict the
    # option favoured by the two screen-extreme cues (pos0 = .72, pos5 = .70),
    # while the advocated model predicts a coin flip (cell 5) or a reversal
    # toward the mid-list crowd (cell 8).
    #   cell 5: d = (+1, 0, -1, 0, 0, +1)  and its mirror
    #   cell 8: d = (+1, -1, -1, -1, 0, +1) and its mirror
    targets = {
        (1, 0, -1, 0, 0, 1),
        (-1, 0, 1, 0, 0, -1),
        (1, -1, -1, -1, 0, 1),
        (-1, 1, 1, 1, 0, -1),
    }

    hits = []
    for _, row in data.iterrows():
        try:
            a = [int(v) for v in row['option_a_ratings']]
            b = [int(v) for v in row['option_b_ratings']]
        except Exception:
            continue
        if len(a) != 6 or len(b) != 6:
            continue
        d = tuple(int(x) - int(y) for x, y in zip(a, b))
        if d not in targets:
            continue
        resp = row['response']
        try:
            resp = int(resp)
        except Exception:
            continue
        # option favoured by the screen-extreme cues (positions 0 and 5 agree)
        fav_a = (d[0] > 0)
        chose_a = (resp == 0)
        hits.append(1.0 if (chose_a == fav_a) else 0.0)

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4900 (var=0.0105)
**Candidate (simulated) value:** 0.5192 (var=0.0084)
**Other theories' values on this metric (for reference):**
- pi_6: 0.4325 (var=0.0102)
- pi_3: 0.9692 (var=0.0013)
- pi_1: 0.1575 (var=0.0113)
- pi_2: 0.4817 (var=0.0049)
- pi_4: 0.1583 (var=0.0049)
- pi_5: 0.4042 (var=0.0068)
- pi_7: 0.4583 (var=0.0105)
- pi_8: 0.7292 (var=0.0071)
- pi_9: 0.4625 (var=0.0097)

### Experiment 10
**Design**
  A=[1, 1, 1, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Cell-5 pairs (trials 9 and 10 of the design), mirrored across sides:
    #   d = a - b = [+1, 0, -1, -1, 0, +1]   -> extremes (pos0=.62, pos5=.58) favour A
    #   d = a - b = [-1, 0, +1, +1, 0, -1]   -> extremes favour B
    # Score = fraction of these trials on which the subject chose the option
    # endorsed by the two SCREEN-EXTREME (weakest) experts, i.e. against the
    # two mid-list heavyweight experts (.95 and .86).
    target_A = (1, 0, -1, -1, 0, 1)
    target_B = (-1, 0, 1, 1, 0, -1)

    hits = 0
    n = 0
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        try:
            a = [int(v) for v in a_raw]
            b = [int(v) for v in b_raw]
        except Exception:
            continue
        if len(a) != len(b) or len(a) != 6:
            continue
        d = tuple(int(x - y) for x, y in zip(a, b))
        if d == target_A:
            extreme_side = 0  # option A
        elif d == target_B:
            extreme_side = 1  # option B
        else:
            continue
        n += 1
        try:
            r = int(resp)
        except Exception:
            continue
        if r == extreme_side:
            hits += 1

    if n == 0:
        return 0.5
    return float(hits) / float(n)
```

**Observed (real) value:** 0.8500 (var=0.0111)
**Candidate (simulated) value:** 0.8450 (var=0.0108)
**Other theories' values on this metric (for reference):**
- pi_3: 0.9717 (var=0.0024)
- pi_6: 0.2300 (var=0.0174)
- pi_1: 0.1217 (var=0.0142)
- pi_2: 0.5083 (var=0.0237)
- pi_4: 0.0083 (var=0.0006)
- pi_5: 0.1233 (var=0.0134)
- pi_7: 0.8817 (var=0.0095)
- pi_8: 0.8183 (var=0.0177)
- pi_9: 0.8183 (var=0.0155)

### Experiment 11
**Design**
  A=[1, 0, 1, 0, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1]
  A=[1, 1, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 1, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def key(row):
        a = ''.join([str(int(round(float(v)))) for v in list(row['option_a_ratings'])])
        b = ''.join([str(int(round(float(v)))) for v in list(row['option_b_ratings'])])
        return a + '|' + b

    # cell -> {pair_key: target_response}  (target = the option predicted pro-A-side)
    cells = {
        'c4': {  # d = [+1,0,-1,-1,+1] : screen extremes vs two mid heavyweights
            '11001|01110': 0,
            '01110|11001': 1,
        },
        'c5': {  # d = [+1,+1,0,-1,0] : early cues favour, later heavyweight opposes
            '11101|00111': 0,
            '00111|11101': 1,
        },
        'c8': {  # d = [+1,+1,0,-1,+1] : three mid experts vs one oracle
            '11101|00110': 0,
            '00110|11101': 1,
        },
    }

    if len(data) == 0:
        return float('nan')

    k = data.apply(key, axis=1)
    resp = pd.to_numeric(data['response'], errors='coerce')

    rates = {}
    for cname, mapping in cells.items():
        hits = []
        for pk, tgt in mapping.items():
            sel = (k == pk)
            if sel.any():
                r = resp[sel]
                hits.append(float(np.mean((r.values == tgt).astype(float))))
        rates[cname] = float(np.mean(hits)) if len(hits) > 0 else 0.5

    # competitor (reason-counting + salience gate) : high on c4, high on c8, lowish on c5
    # advocated (read-until-sufficient, convex read-out) : low on c4, mid on c8, high on c5
    idx = rates['c4'] + rates['c8'] - rates['c5']
    return float(idx)

```

**Observed (real) value:** 0.5133 (var=0.0520)
**Candidate (simulated) value:** 0.9633 (var=0.0381)
**Other theories' values on this metric (for reference):**
- pi_6: 0.4233 (var=0.0611)
- pi_7: 1.1533 (var=0.0393)
- pi_1: 0.1667 (var=0.0469)
- pi_2: 0.5383 (var=0.0337)
- pi_3: 1.4033 (var=0.2137)
- pi_4: 0.2050 (var=0.0273)
- pi_5: 0.1667 (var=0.0303)
- pi_8: 0.7650 (var=0.0380)
- pi_9: 0.9550 (var=0.0428)

### Experiment 12
**Design**
  A=[1, 0, 1, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 0]  B=[0, 1, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0, 1]
  A=[1, 0, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Difference-vector signatures for the two full-direction-disagreement cells
    # (cell 4: deadlock/extremes cell, pairs 4 & 12;  cell 8: four-against-the-oracle, pairs 8 & 16)
    # value = index of the option favoured by the reason-counting (advocated) account
    targets = {
        (1, -1, -1, 0, 0, 0, 1): 0,     # pair 4  -> A
        (-1, 1, 1, 0, 0, 0, -1): 1,     # pair 12 -> B
        (-1, -1, 1, 1, 1, 0, 1): 0,     # pair 8  -> A
        (1, 1, -1, -1, -1, 0, -1): 1,   # pair 16 -> B
    }

    hits = []
    for a_r, b_r, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        try:
            a = [int(v) for v in a_r]
            b = [int(v) for v in b_r]
        except Exception:
            continue
        if len(a) != len(b):
            continue
        d = tuple(int(ai - bi) for ai, bi in zip(a, b))
        if d in targets:
            fav = targets[d]
            try:
                r = int(resp)
            except Exception:
                continue
            hits.append(1.0 if r == fav else 0.0)

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.6650 (var=0.0058)
**Candidate (simulated) value:** 0.7542 (var=0.0067)
**Other theories' values on this metric (for reference):**
- pi_7: 0.7775 (var=0.0103)
- pi_6: 0.1783 (var=0.0069)
- pi_1: 0.1342 (var=0.0079)
- pi_2: 0.6850 (var=0.0078)
- pi_3: 0.6025 (var=0.0521)
- pi_4: 0.0167 (var=0.0006)
- pi_5: 0.3217 (var=0.0130)
- pi_8: 0.4883 (var=0.0049)
- pi_9: 0.7208 (var=0.0140)

### Experiment 13
**Design**
  A=[1, 1, 0, 0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 0, 1, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # The six target displays (EXT, AUTH, TAIL cells and their mirrors).
    targets = {
        # EXT
        ("11000101", "01111110"),
        ("01111110", "11000101"),
        # AUTH
        ("10110100", "01111111"),
        ("01111111", "10110100"),
        # TAIL
        ("01110111", "11111000"),
        ("11111000", "01110111"),
    }

    def as_str(x):
        try:
            return "".join([str(int(round(float(v)))) for v in x])
        except Exception:
            return ""

    hits = 0
    tot = 0
    for _, row in data.iterrows():
        try:
            a = [int(round(float(v))) for v in row["option_a_ratings"]]
            b = [int(round(float(v))) for v in row["option_b_ratings"]]
        except Exception:
            continue
        ka = "".join([str(v) for v in a])
        kb = "".join([str(v) for v in b])
        if (ka, kb) not in targets:
            continue
        # first (top-most) screen row on which the two columns disagree
        j = -1
        for i in range(min(len(a), len(b))):
            if a[i] != b[i]:
                j = i
                break
        if j < 0:
            continue
        try:
            resp = int(row["response"])
        except Exception:
            continue
        if resp not in (0, 1):
            continue
        favors_a = a[j] > b[j]
        followed = (favors_a and resp == 0) or ((not favors_a) and resp == 1)
        hits += 1 if followed else 0
        tot += 1

    if tot == 0:
        return 0.5
    return float(hits) / float(tot)
```

**Observed (real) value:** 0.3533 (var=0.0101)
**Candidate (simulated) value:** 0.3183 (var=0.0049)
**Other theories' values on this metric (for reference):**
- pi_8: 0.9261 (var=0.0024)
- pi_7: 0.2983 (var=0.0068)
- pi_1: 0.6167 (var=0.0031)
- pi_2: 0.1467 (var=0.0094)
- pi_3: 0.7944 (var=0.0830)
- pi_4: 0.4167 (var=0.0016)
- pi_5: 0.3928 (var=0.0039)
- pi_6: 0.7717 (var=0.0096)
- pi_9: 0.3489 (var=0.0052)

### Experiment 14
**Design**
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Two diagnostic cells (each appears mirrored across sides, 6 reps each).
    # Cell V1  : focal = [1,0,1,0,1,0]  vs  [0,1,1,1,1,1]
    #            (two invisible mid-list heavyweights .95/.86 oppose the focal;
    #             screen extremes .74/.74 split)
    # Cell AUTH: focal = [1,1,0,1,0,0]  vs  [0,1,1,1,1,1]
    #            (pos1 .95 and pos3 .86 are mute mutual endorsements; the top
    #             speaking row .74 backs the focal, three cheaper rows oppose it)
    cells = [
        ((1, 0, 1, 0, 1, 0), (0, 1, 1, 1, 1, 1)),
        ((1, 1, 0, 1, 0, 0), (0, 1, 1, 1, 1, 1)),
    ]

    def _tup(v):
        try:
            return tuple(int(round(float(z))) for z in v)
        except Exception:
            return None

    if len(data) == 0:
        return float('nan')

    av = data['option_a_ratings'].apply(_tup).values
    bv = data['option_b_ratings'].apply(_tup).values
    try:
        rv = data['response'].astype(float).values
    except Exception:
        return float('nan')

    hits = []
    for i in range(len(av)):
        A = av[i]
        B = bv[i]
        if A is None or B is None:
            continue
        r = rv[i]
        if not np.isfinite(r):
            continue
        r = int(round(r))
        for focal, other in cells:
            if (A == focal and B == other) or (A == other and B == focal):
                chosen = A if r == 0 else B
                hits.append(1.0 if chosen == focal else 0.0)
                break

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.3000 (var=0.0112)
**Candidate (simulated) value:** 0.2225 (var=0.0101)
**Other theories' values on this metric (for reference):**
- pi_7: 0.1350 (var=0.0054)
- pi_8: 0.7658 (var=0.0055)
- pi_1: 0.5075 (var=0.0047)
- pi_2: 0.1275 (var=0.0067)
- pi_3: 0.7725 (var=0.1429)
- pi_4: 0.1692 (var=0.0046)
- pi_5: 0.2717 (var=0.0072)
- pi_6: 0.4683 (var=0.0115)
- pi_9: 0.1950 (var=0.0046)

### Experiment 15
**Design**
  A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0, 1]  B=[0, 0, 0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    oracle = []   # lone .95 expert (row 2) speaks, all other rows mutually ABSENT (0/0)
    weak = []     # lone .58 expert (row 4) speaks, all other rows mutually ABSENT (0/0)
    top = []      # top/bottom reading-swap cells: 6 discriminating rows, row 2 mutual endorsement

    for _, row in data.iterrows():
        try:
            a = np.asarray([int(v) for v in row['option_a_ratings']], dtype=int)
            b = np.asarray([int(v) for v in row['option_b_ratings']], dtype=int)
            resp = int(row['response'])
        except Exception:
            continue
        n = min(a.shape[0], b.shape[0])
        if n < 7:
            continue
        a = a[:n]
        b = b[:n]
        d = a - b
        disc = np.flatnonzero(d != 0)
        chosen = a if resp == 0 else b

        if disc.shape[0] == 1:
            j = int(disc[0])
            others = [k for k in range(n) if k != j]
            silent_zero = all((a[k] == 0 and b[k] == 0) for k in others)
            if silent_zero:
                follow = 1.0 if chosen[j] == 1 else 0.0
                if j == 2:
                    oracle.append(follow)
                elif j == 4:
                    weak.append(follow)
        elif disc.shape[0] == 6 and (2 not in set(disc.tolist())) and a[2] == 1 and b[2] == 1:
            top_is_A = (a[0] == 1 and a[1] == 1)
            top_is_B = (b[0] == 1 and b[1] == 1)
            if top_is_A != top_is_B:
                chose_A = (resp == 0)
                top.append(1.0 if (chose_A == top_is_A) else 0.0)

    m_oracle = float(np.mean(oracle)) if len(oracle) > 0 else 0.5
    m_weak = float(np.mean(weak)) if len(weak) > 0 else 0.5
    m_top = float(np.mean(top)) if len(top) > 0 else 0.5

    return float(m_oracle + m_weak - m_top)
```

**Observed (real) value:** 0.8550 (var=0.0485)
**Candidate (simulated) value:** 0.7642 (var=0.0322)
**Other theories' values on this metric (for reference):**
- pi_9: 0.2267 (var=0.0482)
- pi_7: 0.8933 (var=0.0347)
- pi_1: 0.8375 (var=0.0390)
- pi_2: 1.1908 (var=0.0665)
- pi_3: 1.3592 (var=0.2426)
- pi_4: 0.4658 (var=0.0219)
- pi_5: 0.6125 (var=0.0290)
- pi_6: 0.5250 (var=0.0263)
- pi_8: 0.5042 (var=0.0413)

### Experiment 16
**Design**
  A=[1, 0, 1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 1, 1]  B=[1, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    sums = {'O0': 0.0, 'O1': 0.0, 'SE': 0.0, 'SA': 0.0}
    cnts = {'O0': 0, 'O1': 0, 'SE': 0, 'SA': 0}

    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float)
            b = np.asarray(list(row['option_b_ratings']), dtype=float)
            resp = int(row['response'])
        except Exception:
            continue
        n = int(min(a.shape[0], b.shape[0]))
        if n < 7:
            continue
        a = a[:n]
        b = b[:n]
        d = a - b
        disc = [j for j in range(n) if abs(d[j]) > 1e-9]

        cell = None
        fav_a = None

        if len(disc) == 1 and disc[0] == 1:
            # lone near-oracle (position 1, validity .95) discriminates
            silent = [j for j in range(n) if j != 1]
            all_one = all(a[j] > 0.5 and b[j] > 0.5 for j in silent)
            all_zero = all(a[j] < 0.5 and b[j] < 0.5 for j in silent)
            if all_one:
                cell = 'O1'
            elif all_zero:
                cell = 'O0'
            if cell is not None:
                fav_a = d[1] > 0
        elif set(disc) == set([0, 1, 5, 6]):
            sil = [2, 3, 4]
            all_one = all(a[j] > 0.5 and b[j] > 0.5 for j in sil)
            all_zero = all(a[j] < 0.5 and b[j] < 0.5 for j in sil)
            if all_one:
                cell = 'SE'
            elif all_zero:
                cell = 'SA'
            if cell is not None:
                # majority (3 vs 1) side: position 0 sign marks it
                fav_a = d[0] > 0

        if cell is None or fav_a is None:
            continue

        chose_a = (resp == 0)
        chose_fav = 1.0 if (chose_a == bool(fav_a)) else 0.0
        sums[cell] += chose_fav
        cnts[cell] += 1

    def prop(k):
        if cnts[k] == 0:
            return 0.5
        return sums[k] / float(cnts[k])

    p_SE = prop('SE')
    p_SA = prop('SA')
    p_O1 = prop('O1')
    p_O0 = prop('O0')

    silence_cliff = p_SE - p_SA          # advocated: large (+.25), competitor: ~0
    oracle_reversal = p_O1 - p_O0        # advocated: ~0 (floored c), competitor: large (+.30)

    return float(silence_cliff - oracle_reversal)

```

**Observed (real) value:** 0.0967 (var=0.1015)
**Candidate (simulated) value:** -0.0583 (var=0.0773)
**Other theories' values on this metric (for reference):**
- pi_7: 0.2350 (var=0.0752)
- pi_9: -0.2500 (var=0.1053)
- pi_1: 0.0467 (var=0.0256)
- pi_2: -0.0183 (var=0.0395)
- pi_3: 0.0133 (var=0.0068)
- pi_4: -0.0167 (var=0.0267)
- pi_5: 0.2833 (var=0.0253)
- pi_6: -0.0500 (var=0.0847)
- pi_8: 0.3117 (var=0.0847)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate is clearly inside the arbiter's prescribed family (smooth evidence-fraction dilution, compressive diagnosticity + sqrt numerosity, difficulty-scaled lapse, symmetric two-regime edge salience with residual, floored non-reversing confidence) and it is now the running-best (loss 0.1268, ACCEPTED). Its wins are real and broad: Exp 2 (.784 vs .775), Exp 8 (-.215 vs -.203, the inverted single-cue ladder that most rivals get wrong in sign), Exp 9 (.519 vs .490), Exp 10 (.845 vs .850), Exp 13 (.318 vs .353), Exp 15 (.764 vs .855) and Exp 14 (.223 vs .300, better than pi_7's .135). That is a genuinely experiment-invariant core.\n\nThe residuals cluster into three coherent, diagnosable groups rather than random noise.\n\n(1) UNDER-WEIGHTING OF CHEAP-CUE COALITIONS AGAINST A LONE HEAVYWEIGHT. Exp 6 (.672 vs .841, the single largest miss after Exp 11), Exp 11's c5 sub-cell (index .963 vs .513 \u2014 the index is c4+c8-c5, and c4/c8 are already near ceiling, so the overshoot is almost entirely a too-low c5, i.e. two cheap early cues losing to the later .86 heavyweight), Exp 3 (.453 vs .518) and Exp 4 (.355 vs .508) all point the same way: in displays where a small coalition of cheap cues faces one high-validity cue, humans are at or above chance on the coalition while DCRI still lets the heavyweight win. Within the prescribed family this is exactly the rho/lam trade-off: compression is not yet strong enough and the sqrt-numerosity reason is too small.\n\n(2) EXP 1 SIGN FLIP. Observed +.093, candidate -.051 (pi_7 reached +.092). The authority-checked stop was retained specifically to buy this contrast, but as parameterised (q_max .85-.95, d_com 1.1-1.6, c multiplying the commitment) the committed branch is too hesitant, so agree-cells stay high while conflict-cells are dragged down by the integrator; the contrast goes negative. This is a calibration failure of a retained component, not a missing mechanism.\n\n(3) THE 1/1-vs-0/0 ASYMMETRY IS OVER-TUNED. Exp 7 overshoots badly (+.188 vs +.053; almost every rival sits near 0), which is the direct signature of omega being too far below 1. Yet Exp 16 (-.058 vs +.097) shows the same asymmetry is not buying the silence-cliff-minus-oracle-reversal contrast it was supposed to buy \u2014 currently the two terms cancel in the wrong order. So the new structural commitment is simultaneously too strong where it is measured directly (Exp 7) and mis-signed where it is contrasted (Exp 16). Exp 12 (.754 vs .665) is a mild over-confidence on landslides, consistent with numerosity/edge terms still pushing the relative read-out a bit high in big panels.",
  "verdict": "regenerate",
  "rationale": "The candidate is accepted and should be the base, but there is clear headroom from in-family recalibration only \\u2014 do NOT change mechanism family (keep compressive diagnosticity, sqrt numerosity, relative read-out, evidence-fraction confidence with a zero floor, two-regime symmetric salience, difficulty-scaled lapse). Four targeted knobs, in priority order:\n\n1. MORE COMPRESSION + MORE NUMEROSITY (fixes Exps 6, 11-c5, 3, 4 simultaneously). Shift rho down to roughly [0.35, 0.48] and lam up to roughly [0.55, 0.75]. A cheap-cue coalition must be able to beat a single .86/.95 cue at or slightly above chance (Exp 3 .518, Exp 4 .508) and a 3-vs-1 or extremes-vs-mid display must reach ~.84 (Exp 6). Check that Exp 10 (.850) and Exp 5 (.820, currently .733 \\u2014 also slightly low, and this change should lift it) do not blow past ceiling. If Exp 8 becomes more negative than -0.25 as a side effect of the lower rho, trim eta to ~[2.5, 3.1] rather than reverting rho.\n\n2. FIRM UP THE RETAINED AUTHORITY-CHECKED STOP (fixes Exp 1's sign). Raise q_max to ~[0.92, 0.99] and d_com to ~[1.7, 2.4], and/or make the commitment branch less attenuated by c (e.g. use sqrt(c) or a floor of ~0.5 on the multiplier for the commitment term only). Verify the gate still does NOT fire in Exps 13-16 (top-row following below chance is well captured now at .318/.223 \\u2014 do not lose that).\n\n3. TAME THE 1/1-vs-0/0 ASYMMETRY (fixes Exp 7, may help Exp 16). Move omega up to ~[0.55, 0.75]. Exp 7's observed effect is +.05, essentially a null; omega ~0.4 is producing +.19. If Exp 16 then still misses, adjust the oracle-reversal half instead by raising c_min slightly (~[0.06, 0.12]) so the lone-oracle O1/O0 cells flatten together, rather than by pushing omega back down.\n\n4. SMALL DE-CONFIDENCING OF LARGE PANELS (Exp 12 .754 vs .665): raise kap2 slightly (~[0.35, 0.65]) so the relative read-out divides by a larger baseline in 7-cue displays, or cap the numerosity term's contribution to the drive.\n\nDo not enlarge the difficulty-scaled lapse globally to fix Exps 3/4 \\u2014 Exp 5 (.820) and Exp 10 (.850) are high-confidence near-deadlock cells and a bigger eps0 would flatten them; fix 3/4 through compression (knob 1) instead."
}
```

## Usage

```json
{
  "input_tokens": 35951,
  "output_tokens": 5892
}
```
