# feedback_iter_07

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
**Dilution-calibrated reason integration with an ORACLE PREMIUM (DCRI-3b): a strongly compressive weighted reason tally whose diagnosticity map is sub-linear across the mid range but regains a categorical top-end premium near the stated-validity ceiling, read out RELATIVE to the evidence actually on the table, scaled by a smooth, strictly-positive confidence that falls with panel silence and with the coarseness inference it licenses, plus a two-regime edge-salience tie-breaker, a slowly-growing numerosity bonus, a difficulty-scaled lapse, and a firm authority-checked top-down stop.**

Identical mechanism family and identical equations to the accepted base; ONE calibration constant is moved (the mid-range compression exponent rho).

1. **Diagnosticity is STRONGLY compressive in the middle but has a TOP-END PREMIUM.** x_j = ((v_j-0.5)/0.5)^rho * (1 + alpha_v*sigmoid(kappa_v*(v_j - v_knee))) with rho ~0.40-0.50 and v_knee ~0.91. Across the .56-.88 band subjective weight is now markedly sub-linear — a .62 cue is worth roughly 70% of an .86 cue rather than 55% — so coalitions of cheap experts routinely outvote a single merely-good expert. Independently of that, a near-certain expert (.93-.95) is treated as CATEGORICALLY heavier, worth ~1.5x its compressed value, so the oracle still holds its own against two upper-middling cues. These are two separable claims about the same map: the *ordering* of everyday cues is read coarsely, while *near-certainty* is read as a different kind of reason.

2. **Numerosity is a separate, slowly-growing reason.** N = lambda*sign(k_A-k_B)*sqrt(|k_A-k_B|): 'more experts back this one' never saturates but grows only as a square root.

3. **Read-out is RELATIVE.** drive = (V + N + P)/(kappa2 + total speaking diagnosticity), keeping large panels and landslides off the ceiling.

4. **Confidence = evidence fraction x coarseness inference, never a reversal.** c = c_min + (1-c_min)*frac^psi*max(1 - eta*f_sil^q*vbar, 0). Confidence scales with the FRACTION of informative evidence and is strictly positive (a lone speaking cue amid a mute panel is still followed above chance), while the sensitivity/gap inference flattens confidence toward chance and is floored at zero so it can never flip the sign of the choice.

5. **Mutual endorsement is a weaker form of silence than mutual absence** (weight omega ~0.63 vs 1), so the processing-load effect is small and positive rather than large.

6. **Edge salience is a two-regime tie-breaker** over literal screen rows, at full force only on exactly deadlocked reason counts and a small constant residual otherwise. Symmetric, so no primacy/recency swing.

7. **Choice difficulty scales the lapse**, so near-deadlocked displays draw genuine guessing while landslides stay near ceiling.

8. **Authority-checked top-down stop**: people read from the top and commit at the first discriminating row only if nothing below it is more diagnostic; the committed choice is still multiplied by the same confidence c.

Individual differences are unimodal jitter in all parameters; no strategy subpopulations, no learning.

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

    rho = _p('rho', 0.45, 0.05, 4.0)
    alpha_v = _p('alpha_v', 0.62, 0.0, 3.0)
    v_knee = _p('v_knee', 0.910, 0.55, 0.999)
    kappa_v = _p('kappa_v', 35.0, 1.0, 200.0)
    lam = _p('lam', 0.56, 0.0, 3.0)
    mu = _p('mu', 2.40, 0.0, 8.0)
    tau = _p('tau', 1.00, 0.2, 8.0)
    sigma = _p('sigma', 0.41, 0.15, 3.0)
    g_res = _p('g_res', 0.10, 0.0, 1.0)
    kappa = _p('kappa', 0.60, 0.05, 5.0)
    kap2 = _p('kap2', 0.42, 0.02, 5.0)
    psi = _p('psi', 0.35, 0.01, 3.0)
    eta = _p('eta', 3.35, 0.0, 10.0)
    q_sil = _p('q_sil', 1.20, 0.3, 8.0)
    omega = _p('omega', 0.63, 0.0, 1.0)
    c_min = _p('c_min', 0.085, 0.0, 0.5)
    beta = _p('beta', 3.10, 0.05, 40.0)
    eps0 = _p('eps0', 0.08, 0.0, 0.5)
    eps1 = _p('eps1', 0.22, 0.0, 0.6)
    q_max = _p('q_max', 0.96, 0.0, 1.0)
    d_com = _p('d_com', 2.25, 0.0, 6.0)
    phi = _p('phi', 1.02, 0.5, 3.0)
    s_stop = _p('s_stop', 9.0, 0.5, 60.0)

    def _sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    # ------------------------------------------------------------------
    # 4. compressive reason tally WITH A TOP-END (ORACLE) PREMIUM
    #    mid-range validity is read coarsely (rho < 1), but a near-certain
    #    expert regains a categorical extra weight.
    # ------------------------------------------------------------------
    prem = 1.0 + alpha_v * _sig(kappa_v * (val - v_knee))
    x = np.power(vs, rho) * prem
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
- rho: [0.40, 0.50]
- alpha_v: [0.45, 0.85]
- v_knee: [0.895, 0.925]
- kappa_v: [25.0, 45.0]
- lam: [0.48, 0.64]
- mu: [2.15, 2.65]
- tau: [0.85, 1.25]
- sigma: [0.36, 0.46]
- g_res: [0.06, 0.14]
- kappa: [0.45, 0.80]
- kap2: [0.30, 0.55]
- psi: [0.28, 0.45]
- eta: [3.05, 3.65]
- q_sil: [1.05, 1.35]
- omega: [0.55, 0.72]
- c_min: [0.06, 0.11]
- beta: [2.80, 3.50]
- eps0: [0.05, 0.11]
- eps1: [0.16, 0.28]
- q_max: [0.92, 0.99]
- d_com: [2.00, 2.55]
- phi: [0.98, 1.06]
- s_stop: [6.0, 12.0]
- validities: validities

`rationale`:
MINIMAL DIFF: `predict` is re-emitted byte-for-byte except for ONE changed number (the fallback default of `rho`, 0.52 -> 0.45, which only matters if the sampler omits it), and ONE changed parameter range: `rho` [0.46, 0.58] -> [0.40, 0.50]. Every other parameter range, equation, gate, and constant is identical to the accepted iter-3 base, and the premium block (alpha_v, v_knee, kappa_v) and the whole authority-stop block (q_max, d_com, phi, s_stop) are reverted/held exactly at the iter-3 values, as the critic demanded after iters 6 and 7 falsified moves along those axes.

**Why this single step is the right one.** The critic's iteration-7 verdict is explicit and its own ledger supports it: four different axes have now been tested since iter 3 (salience width/residual, silence exponent + lapse, commitment strength, premium location/sharpness) and all four were rejected because they trade the under-confident cluster (Exps 5 .725, 6 .598, 9 .428, 10 .752, 15 .802 — all BELOW observed) against the over-confident cluster (Exps 3, 4, 11, 13, 14, 16) one-for-one. Those are *global gain* and *top-end shape* axes. The mid-range compression exponent rho is the one axis whose residual signs all point the same way and the one cell of the design that has never been tested with the top-end premium present.

**Mechanism of the expected gain.** Lowering rho from ~0.52 to ~0.45 raises the *relative* weight of cheap cues without touching global gain: at rho=.45 a .62 expert (vs=.24) gets x=.55 versus x=.51 at rho=.52, while a .86 expert (vs=.72) barely moves (.86 -> .85) and the .95 oracle is pinned by the premium term, which is a multiplicative factor on an already near-unity base. The relative effect is therefore concentrated exactly where the under-confident experiments live: Exp 6's two target classes are cheap-cue coalitions (disc {0,2,5} and {0,1,2,5}) opposing a single mid-list cue; Exp 10's cell-5 and Exp 5's target are screen-extreme *weak* experts (.62/.58 and analogues) opposing mid heavyweights; Exp 9's two cells are the same shape. All four need the cheap side relatively heavier, and all four are currently 5-24 points BELOW observed. Simultaneously the Exp-11 index should fall via c5 (two cheap early cues vs the .86 gain weight, so following the .86 drops), which is the only monotone route to that index that does not require blunting a gate.

**Why iter-1's rho push failed but this one should not.** In iter 1 the critic pushed rho down globally and then retracted it in iter 2 because Exps 3/4 demand that the .95 hold its own against two upper-middling cues. That retraction is no longer binding: the top-end premium term (added in iter 3, absent in iter 1) now discharges that job independently, multiplying the oracle by ~1.5 regardless of rho. So the two moves that were in conflict in iter 2 are now orthogonal — rho governs the .56-.88 ordering, the premium governs near-certainty. This is precisely the untried combination.

**Guardrails respected.** Exp 8 is the metric most exposed (strong-minus-weak single-cue following becomes more negative as weak cues gain weight), but the base sits at -.170 against an observed -.203, so this direction moves it *toward* the data with ~.06 of headroom before the -0.25 floor — and the step is deliberately small (rho range overlaps the base's low end) to keep that from overshooting as it did in iter 7. Exp 12 is protected because the relative read-out denominator (kap2 + Xs) rises with every x, so a uniform lift of cheap-cue mass partially cancels in the ratio in the big 7-cue panels. The confidence path (omega, eta, kappa, c_min, q_sil) that Exps 7/15/16 depend on is untouched, and nothing in the edit can make the authority-checked stop fire in Exps 13-16, so the hard-won below-chance top-row following is preserved. No bundling: this is a one-knob, one-line test, exactly as the critic asked after three consecutive rejected multi-knob or over-stepped edits.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.1268 -> ACCEPTED
- iter 2: loss=0.0883 -> ACCEPTED
- iter 3: loss=0.0872 -> ACCEPTED
- iter 4: loss=0.0880 -> REJECTED
- iter 5: loss=0.1150 -> REJECTED
- iter 6: loss=0.0990 -> REJECTED
- iter 7: loss=0.0978 -> REJECTED
- iter 8 (current candidate you are grading): loss=0.0883 -> REJECTED
Running-best (last accepted) base: iter 3 at loss=0.0872.

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
**Candidate trajectory (this loop):**
  - iter 1: -0.0508 (var=0.0101) (Δ vs real -0.1442)
  - iter 2: -0.0090 (var=0.0062) (Δ vs real -0.1023)
  - iter 3: 0.0423 (var=0.0063) (Δ vs real -0.0510)
  - iter 4: 0.0383 (var=0.0064) (Δ vs real -0.0550)
  - iter 5: -0.0537 (var=0.0039) (Δ vs real -0.1471)
  - iter 6: 0.0008 (var=0.0060) (Δ vs real -0.0925)
  - iter 7: 0.0481 (var=0.0065) (Δ vs real -0.0452)
  - iter 8 (current): 0.0298 (var=0.0057) (Δ vs real -0.0635)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.7845 (var=0.0053) (Δ vs real +0.0095)
  - iter 2: 0.7800 (var=0.0039) (Δ vs real +0.0050)
  - iter 3: 0.7690 (var=0.0037) (Δ vs real -0.0060)
  - iter 4: 0.7575 (var=0.0041) (Δ vs real -0.0175)
  - iter 5: 0.7990 (var=0.0033) (Δ vs real +0.0240)
  - iter 6: 0.7700 (var=0.0047) (Δ vs real -0.0050)
  - iter 7: 0.7475 (var=0.0048) (Δ vs real -0.0275)
  - iter 8 (current): 0.7625 (var=0.0043) (Δ vs real -0.0125)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.4525 (var=0.0115) (Δ vs real -0.0650)
  - iter 2: 0.4775 (var=0.0175) (Δ vs real -0.0400)
  - iter 3: 0.4763 (var=0.0173) (Δ vs real -0.0412)
  - iter 4: 0.4675 (var=0.0139) (Δ vs real -0.0500)
  - iter 5: 0.4275 (var=0.0221) (Δ vs real -0.0900)
  - iter 6: 0.4575 (var=0.0137) (Δ vs real -0.0600)
  - iter 7: 0.4938 (var=0.0130) (Δ vs real -0.0237)
  - iter 8 (current): 0.4700 (var=0.0125) (Δ vs real -0.0475)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.3550 (var=0.0070) (Δ vs real -0.1525)
  - iter 2: 0.3694 (var=0.0087) (Δ vs real -0.1381)
  - iter 3: 0.4238 (var=0.0073) (Δ vs real -0.0837)
  - iter 4: 0.3981 (var=0.0101) (Δ vs real -0.1094)
  - iter 5: 0.3175 (var=0.0048) (Δ vs real -0.1900)
  - iter 6: 0.4206 (var=0.0089) (Δ vs real -0.0869)
  - iter 7: 0.4119 (var=0.0077) (Δ vs real -0.0956)
  - iter 8 (current): 0.4012 (var=0.0094) (Δ vs real -0.1062)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.7325 (var=0.0170) (Δ vs real -0.0875)
  - iter 2: 0.7275 (var=0.0132) (Δ vs real -0.0925)
  - iter 3: 0.7250 (var=0.0116) (Δ vs real -0.0950)
  - iter 4: 0.7500 (var=0.0133) (Δ vs real -0.0700)
  - iter 5: 0.7850 (var=0.0121) (Δ vs real -0.0350)
  - iter 6: 0.7275 (var=0.0095) (Δ vs real -0.0925)
  - iter 7: 0.7588 (var=0.0147) (Δ vs real -0.0612)
  - iter 8 (current): 0.7338 (var=0.0117) (Δ vs real -0.0862)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.6725 (var=0.0081) (Δ vs real -0.1688)
  - iter 2: 0.6644 (var=0.0060) (Δ vs real -0.1769)
  - iter 3: 0.5981 (var=0.0102) (Δ vs real -0.2431)
  - iter 4: 0.6406 (var=0.0109) (Δ vs real -0.2006)
  - iter 5: 0.6681 (var=0.0085) (Δ vs real -0.1731)
  - iter 6: 0.5844 (var=0.0061) (Δ vs real -0.2569)
  - iter 7: 0.5713 (var=0.0079) (Δ vs real -0.2700)
  - iter 8 (current): 0.6056 (var=0.0068) (Δ vs real -0.2356)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.1883 (var=0.0297) (Δ vs real +0.1350)
  - iter 2: 0.0733 (var=0.0360) (Δ vs real +0.0200)
  - iter 3: 0.0583 (var=0.0290) (Δ vs real +0.0050)
  - iter 4: 0.1733 (var=0.0291) (Δ vs real +0.1200)
  - iter 5: 0.0550 (var=0.0463) (Δ vs real +0.0017)
  - iter 6: 0.0717 (var=0.0306) (Δ vs real +0.0183)
  - iter 7: 0.0800 (var=0.0286) (Δ vs real +0.0267)
  - iter 8 (current): 0.0583 (var=0.0340) (Δ vs real +0.0050)
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
**Candidate trajectory (this loop):**
  - iter 1: -0.2150 (var=0.0333) (Δ vs real -0.0125)
  - iter 2: -0.1813 (var=0.0210) (Δ vs real +0.0212)
  - iter 3: -0.1700 (var=0.0289) (Δ vs real +0.0325)
  - iter 4: -0.1825 (var=0.0143) (Δ vs real +0.0200)
  - iter 5: -0.1700 (var=0.0242) (Δ vs real +0.0325)
  - iter 6: -0.1437 (var=0.0254) (Δ vs real +0.0588)
  - iter 7: -0.2338 (var=0.0215) (Δ vs real -0.0312)
  - iter 8 (current): -0.2225 (var=0.0244) (Δ vs real -0.0200)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.5192 (var=0.0084) (Δ vs real +0.0292)
  - iter 2: 0.5058 (var=0.0106) (Δ vs real +0.0158)
  - iter 3: 0.4283 (var=0.0068) (Δ vs real -0.0617)
  - iter 4: 0.4825 (var=0.0089) (Δ vs real -0.0075)
  - iter 5: 0.4958 (var=0.0098) (Δ vs real +0.0058)
  - iter 6: 0.4308 (var=0.0084) (Δ vs real -0.0592)
  - iter 7: 0.4392 (var=0.0109) (Δ vs real -0.0508)
  - iter 8 (current): 0.4142 (var=0.0121) (Δ vs real -0.0758)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.8450 (var=0.0108) (Δ vs real -0.0050)
  - iter 2: 0.7983 (var=0.0161) (Δ vs real -0.0517)
  - iter 3: 0.7517 (var=0.0212) (Δ vs real -0.0983)
  - iter 4: 0.7567 (var=0.0180) (Δ vs real -0.0933)
  - iter 5: 0.7800 (var=0.0155) (Δ vs real -0.0700)
  - iter 6: 0.7417 (var=0.0137) (Δ vs real -0.1083)
  - iter 7: 0.7283 (var=0.0213) (Δ vs real -0.1217)
  - iter 8 (current): 0.7250 (var=0.0178) (Δ vs real -0.1250)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.9633 (var=0.0381) (Δ vs real +0.4500)
  - iter 2: 0.8750 (var=0.0551) (Δ vs real +0.3617)
  - iter 3: 0.8517 (var=0.0420) (Δ vs real +0.3383)
  - iter 4: 0.8283 (var=0.0571) (Δ vs real +0.3150)
  - iter 5: 0.8900 (var=0.0387) (Δ vs real +0.3767)
  - iter 6: 0.8767 (var=0.0390) (Δ vs real +0.3633)
  - iter 7: 0.8533 (var=0.0393) (Δ vs real +0.3400)
  - iter 8 (current): 0.8417 (var=0.0378) (Δ vs real +0.3283)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.7542 (var=0.0067) (Δ vs real +0.0892)
  - iter 2: 0.6983 (var=0.0084) (Δ vs real +0.0333)
  - iter 3: 0.6650 (var=0.0076) (Δ vs real +0.0000)
  - iter 4: 0.6658 (var=0.0132) (Δ vs real +0.0008)
  - iter 5: 0.7033 (var=0.0103) (Δ vs real +0.0383)
  - iter 6: 0.6642 (var=0.0079) (Δ vs real -0.0008)
  - iter 7: 0.6417 (var=0.0092) (Δ vs real -0.0233)
  - iter 8 (current): 0.6333 (var=0.0088) (Δ vs real -0.0317)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.3183 (var=0.0049) (Δ vs real -0.0350)
  - iter 2: 0.3483 (var=0.0062) (Δ vs real -0.0050)
  - iter 3: 0.3367 (var=0.0047) (Δ vs real -0.0167)
  - iter 4: 0.3478 (var=0.0051) (Δ vs real -0.0056)
  - iter 5: 0.3228 (var=0.0057) (Δ vs real -0.0306)
  - iter 6: 0.3450 (var=0.0065) (Δ vs real -0.0083)
  - iter 7: 0.3400 (var=0.0057) (Δ vs real -0.0133)
  - iter 8 (current): 0.3456 (var=0.0081) (Δ vs real -0.0078)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.2225 (var=0.0101) (Δ vs real -0.0775)
  - iter 2: 0.2483 (var=0.0085) (Δ vs real -0.0517)
  - iter 3: 0.2592 (var=0.0084) (Δ vs real -0.0408)
  - iter 4: 0.2283 (var=0.0065) (Δ vs real -0.0717)
  - iter 5: 0.1783 (var=0.0061) (Δ vs real -0.1217)
  - iter 6: 0.2392 (var=0.0080) (Δ vs real -0.0608)
  - iter 7: 0.2417 (var=0.0066) (Δ vs real -0.0583)
  - iter 8 (current): 0.2358 (var=0.0049) (Δ vs real -0.0642)
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
**Candidate trajectory (this loop):**
  - iter 1: 0.7642 (var=0.0322) (Δ vs real -0.0908)
  - iter 2: 0.8225 (var=0.0539) (Δ vs real -0.0325)
  - iter 3: 0.8017 (var=0.0373) (Δ vs real -0.0533)
  - iter 4: 0.7992 (var=0.0357) (Δ vs real -0.0558)
  - iter 5: 0.8200 (var=0.0563) (Δ vs real -0.0350)
  - iter 6: 0.8400 (var=0.0453) (Δ vs real -0.0150)
  - iter 7: 0.8167 (var=0.0456) (Δ vs real -0.0383)
  - iter 8 (current): 0.8392 (var=0.0545) (Δ vs real -0.0158)
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
**Candidate trajectory (this loop):**
  - iter 1: -0.0583 (var=0.0773) (Δ vs real -0.1550)
  - iter 2: 0.1567 (var=0.0699) (Δ vs real +0.0600)
  - iter 3: 0.1367 (var=0.0916) (Δ vs real +0.0400)
  - iter 4: 0.1300 (var=0.0648) (Δ vs real +0.0333)
  - iter 5: -0.0483 (var=0.0628) (Δ vs real -0.1450)
  - iter 6: 0.1483 (var=0.0954) (Δ vs real +0.0517)
  - iter 7: 0.1950 (var=0.1004) (Δ vs real +0.0983)
  - iter 8 (current): 0.0933 (var=0.0821) (Δ vs real -0.0033)
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

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate is clearly inside the arbiter's prescribed family (smooth evidence-fraction dilution, compressive diagnosticity + sqrt numerosity, difficulty-scaled lapse, symmetric two-regime edge salience with residual, floored non-reversing confidence) and it is now the running-best (loss 0.1268, ACCEPTED). Its wins are real and broad: Exp 2 (.784 vs .775), Exp 8 (-.215 vs -.203, the inverted single-cue ladder that most rivals get wrong in sign), Exp 9 (.519 vs .490), Exp 10 (.845 vs .850), Exp 13 (.318 vs .353), Exp 15 (.764 vs .855) and Exp 14 (.223 vs .300, better than pi_7's .135). That is a genuinely experiment-invariant core.

The residuals cluster into three coherent, diagnosable groups rather than random noise.

(1) UNDER-WEIGHTING OF CHEAP-CUE COALITIONS AGAINST A LONE HEAVYWEIGHT. Exp 6 (.672 vs .841, the single largest miss after Exp 11), Exp 11's c5 sub-cell (index .963 vs .513 — the index is c4+c8-c5, and c4/c8 are already near ceiling, so the overshoot is almost entirely a too-low c5, i.e. two cheap early cues losing to the later .86 heavyweight), Exp 3 (.453 vs .518) and Exp 4 (.355 vs .508) all point the same way: in displays where a small coalition of cheap cues faces one high-validity cue, humans are at or above chance on the coalition while DCRI still lets the heavyweight win. Within the prescribed family this is exactly the rho/lam trade-off: compression is not yet strong enough and the sqrt-numerosity reason is too small.

(2) EXP 1 SIGN FLIP. Observed +.093, candidate -.051 (pi_7 reached +.092). The authority-checked stop was retained specifically to buy this contrast, but as parameterised (q_max .85-.95, d_com 1.1-1.6, c multiplying the commitment) the committed branch is too hesitant, so agree-cells stay high while conflict-cells are dragged down by the integrator; the contrast goes negative. This is a calibration failure of a retained component, not a missing mechanism.

(3) THE 1/1-vs-0/0 ASYMMETRY IS OVER-TUNED. Exp 7 overshoots badly (+.188 vs +.053; almost every rival sits near 0), which is the direct signature of omega being too far below 1. Yet Exp 16 (-.058 vs +.097) shows the same asymmetry is not buying the silence-cliff-minus-oracle-reversal contrast it was supposed to buy — currently the two terms cancel in the wrong order. So the new structural commitment is simultaneously too strong where it is measured directly (Exp 7) and mis-signed where it is contrasted (Exp 16). Exp 12 (.754 vs .665) is a mild over-confidence on landslides, consistent with numerosity/edge terms still pushing the relative read-out a bit high in big panels.
Rationale: The candidate is accepted and should be the base, but there is clear headroom from in-family recalibration only \u2014 do NOT change mechanism family (keep compressive diagnosticity, sqrt numerosity, relative read-out, evidence-fraction confidence with a zero floor, two-regime symmetric salience, difficulty-scaled lapse). Four targeted knobs, in priority order:

1. MORE COMPRESSION + MORE NUMEROSITY (fixes Exps 6, 11-c5, 3, 4 simultaneously). Shift rho down to roughly [0.35, 0.48] and lam up to roughly [0.55, 0.75]. A cheap-cue coalition must be able to beat a single .86/.95 cue at or slightly above chance (Exp 3 .518, Exp 4 .508) and a 3-vs-1 or extremes-vs-mid display must reach ~.84 (Exp 6). Check that Exp 10 (.850) and Exp 5 (.820, currently .733 \u2014 also slightly low, and this change should lift it) do not blow past ceiling. If Exp 8 becomes more negative than -0.25 as a side effect of the lower rho, trim eta to ~[2.5, 3.1] rather than reverting rho.

2. FIRM UP THE RETAINED AUTHORITY-CHECKED STOP (fixes Exp 1's sign). Raise q_max to ~[0.92, 0.99] and d_com to ~[1.7, 2.4], and/or make the commitment branch less attenuated by c (e.g. use sqrt(c) or a floor of ~0.5 on the multiplier for the commitment term only). Verify the gate still does NOT fire in Exps 13-16 (top-row following below chance is well captured now at .318/.223 \u2014 do not lose that).

3. TAME THE 1/1-vs-0/0 ASYMMETRY (fixes Exp 7, may help Exp 16). Move omega up to ~[0.55, 0.75]. Exp 7's observed effect is +.05, essentially a null; omega ~0.4 is producing +.19. If Exp 16 then still misses, adjust the oracle-reversal half instead by raising c_min slightly (~[0.06, 0.12]) so the lone-oracle O1/O0 cells flatten together, rather than by pushing omega back down.

4. SMALL DE-CONFIDENCING OF LARGE PANELS (Exp 12 .754 vs .665): raise kap2 slightly (~[0.35, 0.65]) so the relative read-out divides by a larger baseline in 7-cue displays, or cap the numerosity term's contribution to the drive.

Do not enlarge the difficulty-scaled lapse globally to fix Exps 3/4 \u2014 Exp 5 (.820) and Exp 10 (.850) are high-confidence near-deadlock cells and a bigger eps0 would flatten them; fix 3/4 through compression (knob 1) instead.

**Outcome of your advice:** iter 1 candidate loss=0.1268 -> iter 2 candidate loss=0.0883 -> the gate marked it ACCEPTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate was ACCEPTED (0.1268 -> 0.0883) and it is clearly faithful to the arbiter's prescribed family: smooth evidence-fraction dilution (never sign-reversing), compressive diagnosticity + sqrt numerosity, relative read-out, symmetric two-regime edge salience with residual, difficulty-scaled lapse. The iteration-2 recalibration worked almost everywhere I asked for it: Exp 7 (+.188 -> +.073 vs obs +.053, the omega fix), Exp 16 (-.058 -> +.157 vs obs +.097, the c_min/omega fix), Exp 12 (.754 -> .698 vs obs .665, the kap2 fix), Exp 13 (.318 -> .348 vs obs .353), Exp 15 (.764 -> .823 vs obs .855), Exp 1 (-.051 -> -.009, sign still wrong but the gate firming moved it the right way), Exp 14 (.223 -> .248 vs obs .300). Exps 2, 8, 9, 13 are essentially matched. That is a broad, experiment-invariant core.

The residual mass is now concentrated and, importantly, its structure contradicts my own iteration-1 knob-1 advice. Exp 11 (index .875 vs .513) is by far the dominant term and Exp 6 (.664 vs .841) and Exp 4 (.369 vs .508) are next. Crucially Exps 3 (.478 vs .518) and 4 (.369 vs .508) both say the model lets cheap/moderate coalitions beat the lone high-validity cue TOO often, while Exp 6 says a coalition should win MORE often. That is not noise: it is a shape problem in the diagnosticity map, not a scale problem. Decomposing Exp 11's index (c4 + c8 - c5) against sister designs makes the same point: c4 is a 2-vs-2 deadlock of extremes vs mid heavyweights, whose analogue in Exp 10 humans score ~.85, and Exp 5's deadlock cell ~.82; if c4 is genuinely high in humans, then the index of .513 forces c8 (three mid cues vs ONE .95 oracle) to sit well BELOW c5 (two cues vs a .86 heavyweight) \u2014 i.e. humans treat a .95 oracle as categorically heavier than a .86 cue, whereas the current rho~.52 power map makes those two nearly interchangeable. The same top-end flattening explains Exp 4 (two .78/.72 cues beating the .95 63% of the time in the model vs 50% in humans) and the Exp 12 cell-8 overshoot, while leaving Exp 6 (whose opposing 'top' cue is not the .95 oracle) free to rise.

Secondary residuals: Exp 5 (.727 vs .820) and Exp 10 (.798 vs .850) drifted slightly low when kap2/omega were raised \u2014 both are dk=0 salience-gated cells, so they are controlled by mu/sigma, which unfortunately also control Exp 11's c4; that trade-off must be handled carefully rather than by a blunt mu increase. Exp 1's TTB contrast is still the wrong sign (-.009 vs +.093), though the firmer gate halved the gap.
Rationale: Accepted, but there is still ~0.36 of unexplained residual sitting in one experiment, so keep tuning \u2014 strictly inside the prescribed family (compressive tally + sqrt numerosity + relative read-out + evidence-fraction confidence with zero floor + symmetric two-regime salience + difficulty-scaled lapse + authority-checked stop). Four knobs, in priority order:

1. RESHAPE (do not merely rescale) THE DIAGNOSTICITY MAP \u2014 add a top-end premium. Replace x_j = ((v-0.5)/0.5)^rho with x_j = ((v-0.5)/0.5)^rho * (1 + alpha*sigmoid(kappa_v*(v - v_knee))), alpha ~[0.3, 0.9], v_knee ~0.90, or equivalently make rho validity-dependent (strong compression below ~.85, near-linear above). This is a normalization/shape change within the same compressive family, not a new mechanism. It is the single change that can simultaneously (a) pull Exp 4 up toward .50 and Exp 3 up toward .52 (the .95 must hold its own against two .78/.72 cues), (b) drop Exp 11's c8 (three mid cues vs the lone oracle) off its ceiling and hence the index toward .51, (c) trim Exp 12 (.698 -> ~.67), while (d) leaving Exp 6 alone or improving it, because Exp 6's opposing 'top' cue is not a .95 oracle so it keeps its compressed weight. IMPORTANT: I am explicitly retracting my iteration-1 push to lower rho globally \u2014 Exps 3/4 show the opposite sign, and the proposer was right to take only a small step. Do NOT lower rho further; put the extra coalition strength that Exp 6 needs into lam (numerosity, ~[0.55,0.70]) only if the oracle premium has first protected Exps 3/4/11-c8.

2. VERIFY EXP 11 CELL-BY-CELL BEFORE TUNING. Print the model's c4, c8, c5 rates on the actual design strings in the metric. The index is a difference, so a global confidence change can move it in either direction; only the decomposition tells you whether the overshoot is ceiling-c4, ceiling-c8, or floor-c5. Target whichever cell is off, and do NOT fix it by raising eps0 globally or lowering beta globally \u2014 that would drag Exps 5 (.727), 6 (.664), 10 (.798) and 15 (.823), all of which are already BELOW observed.

3. DEADLOCK CELLS ARE NOW SLIGHTLY UNDER-CONFIDENT. Exp 5 (.727 vs .820) and Exp 10 (.798 vs .850) both want a touch more salience drive at dk=0. Nudge mu up only modestly (~[2.2,2.7]) or tighten sigma (~[0.34,0.45]) so full-force salience is reserved for exact deadlocks; re-check Exp 11's c4 after any mu move, since it shares the dk=0 gate.

4. EXP 1 \u2014 KEEP FIRMING THE AUTHORITY-CHECKED STOP, SMALL STEP. My iteration-1 gate-firming advice was ACCEPTED and moved the contrast from -.051 to -.009, so continue in the same direction but do not overshoot: d_com ~[2.1,2.7] and/or phi -> 1.00-1.05 (gate fires a little more readily) while leaving q_max where it is. Do not adopt sqrt(c) for the commitment branch \u2014 the proposer's argument from Exps 15/8 is sound and those fits are good. Re-verify after the change that the gate still cannot fire in Exps 13-16 (the below-chance top-row following at .348/.248 is hard-won and must not be lost; Exp 14 in fact still needs to RISE toward .300, so any change that pushes it further below chance is a net loss).

**Outcome of your advice:** iter 2 candidate loss=0.0883 -> iter 3 candidate loss=0.0872 -> the gate marked it ACCEPTED.

### Iteration 3
Verdict: regenerate
Interpretation: The candidate was ACCEPTED, but only barely (0.0883 -> 0.0872), and the per-experiment decomposition shows why: the oracle-premium shape change I asked for in iteration 2 bought real but small gains on Exp 1 (-.009 -> +.042 vs obs +.093), Exp 4 (.369 -> .424 vs obs .508) and Exp 12 (.698 -> .665, now exact), while paying for them almost one-for-one elsewhere: Exp 6 got WORSE (.664 -> .598 vs obs .841), Exp 10 got worse (.798 -> .752 vs obs .850), Exp 9 got worse (.506 -> .428 vs obs .490), Exp 5 unchanged and still low (.725 vs .820). The premium plus the tightened sigma strengthened mid-list heavyweights and simultaneously stripped the residual edge-salience push out of every dk=1/dk=2 cell \u2014 which is exactly the cell class Exp 6 and Exp 9 are built on. So knob-1 was roughly neutral and knob-3 (tighter sigma) was net harmful.

The dominant residual is still Exp 11 (.852 vs .513, ~0.34 of the total), and it is now diagnosable analytically. The index is c4 + c8 - c5, and the c8 display ('11101|00110') is literally the c5 display ('11101|00111') plus ONE extra supporting cue at position 4. Any monotone reason-integrator must give c8 >= c5, hence index >= c4. The observed index of .513 therefore forces c4 <= ~.51 in humans: the Exp-11 extremes-vs-two-mid-heavyweights deadlock is at CHANCE, not near ceiling. The model puts c4 on the salience ceiling, which is the entire overshoot. Importantly this does NOT mean the edge-salience tie-breaker is wrong \u2014 the sister deadlocks in Exp 5 (.820) and Exp 10 (.850) are genuinely high. The variable that separates them is visible in the displays: Exp 5's target ('10010|01100') and Exp 10's cell-5 ('101100|001101') have their silent rows as mutual ABSENCES (0/0), whereas Exp 11's c4 ('11001|01110') has a silent row that is a mutual ENDORSEMENT (1/1) \u2014 and so do both Exp 9 target cells (row 3 = 1/1 in cell 5; row 4 = 1/1 in cell 8), which humans score at .490, i.e. also chance. That is a clean three-way replication: deadlocked extremes-vs-heavyweights displays are followed ~.82-.85 when the mute rows are mutual absences and ~.49-.51 when a mute row is a mutual endorsement. The current model has omega<1 (endorsement counts as LESS silence), which is the wrong sign for this particular read-out.

Everything else remains a strong, experiment-invariant core and should not be touched: Exp 2 (.769/.775), Exp 8 (-.170/-.203, right sign where most rivals fail), Exp 12 (.665 exact), Exp 13 (.337/.353), Exp 14 (.259/.300), Exp 15 (.802/.855), Exp 16 (.137/.097), Exp 7 (.058/.053).
Rationale: Accepted but with ~0.58 of residual concentrated in just two experiments (11 and 6), and the iteration-3 changes were close to a wash. Stay strictly inside the prescribed family (compressive tally + sqrt numerosity + relative read-out + evidence-fraction confidence with a zero floor and no sign reversal + symmetric two-regime edge salience + difficulty-scaled lapse + authority-checked stop). Three knobs, in priority order:

1. MAKE THE SALIENCE TIE-BREAKER ENDORSEMENT-SENSITIVE (this is the Exp-11 fix and it is a normalization tweak, not a new mechanism). Multiply the salience gain by a damping factor that grows with the mass of mute MUTUAL-ENDORSEMENT rows, e.g. P = mu * g * S * exp(-theta * f_end), where f_end = (share of non-discriminating rows with a=b=1), theta ~ [1.0, 2.5]. Rationale and target: Exp 11's index forces c4 <= ~.51 (because c8 is c5-plus-one-supporting-cue, so c8 >= c5 for any monotone tally), and c4 is the one deadlock cell among Exp 5 / Exp 10 / Exp 11 / Exp 9 whose mute row is a 1/1 endorsement. With theta in this range you should see Exp 11 drop from .852 toward ~.55-.65, Exp 9 stay near .49 (its cells also contain 1/1 mutes), and Exp 5 / Exp 10 (mutual-absence mutes only) rise back toward .82/.85. Print c4, c8, c5 separately before and after \u2014 do not tune the index blind. Note this is a damping of the SALIENCE term only; leave omega (the silence-mass weight in the confidence term) where it is, since Exp 7 (+.058 vs +.053) and Exp 16 (+.137 vs +.097) are currently well fit by it.

2. UNDO THE SIGMA TIGHTENING \u2014 it was my advice and it demonstrably backfired (Exp 6 .664 -> .598, Exp 9 .506 -> .428). Put sigma back to ~[0.45, 0.58] and raise the constant residual g_res to ~[0.14, 0.22] so that dk=1 and dk=2 displays whose two screen-extreme cues AGREE still get a real push. Exp 6's two target classes are exactly that (disc {0,2,5} with 0 and 5 agreeing, dk=1; and disc {0,1,2,5}, dk=2) and humans score them .841 while the model sits at .598 \u2014 by far the second-largest residual. Check that Exp 12 (currently exact at .665) does not overshoot: its cell-8 has disagreeing extremes so it should barely move, but its cell-4 will rise, so keep g_res at the low end if Exp 12 climbs past ~.69. Do NOT try to fix Exp 6 by raising lam again \u2014 the small lam step this round did not help and lam also loads on Exps 3/4/12.

3. FREEZE, DO NOT EXTEND, THE TOP-END PREMIUM. alpha_v bought Exps 1/3/4/12 and cost Exps 6/9/10 roughly equally; do not raise it further, and if after knobs 1-2 Exps 6/9/10 are still short, trim alpha_v to ~[0.30, 0.55] rather than compensating with mu. Likewise leave the authority-checked stop alone this round: Exp 1 has moved -.051 -> -.009 -> +.042 against an observed +.093 with diminishing returns, and further firming risks the hard-won below-chance top-row following in Exps 13/14 (.337/.259, where Exp 14 still needs to RISE toward .300). Any change that pushes Exp 13 or Exp 14 further below observed is a net loss.

Also re-verify after knob 1 that Exp 15's oracle/weak-follow terms and Exp 16's SE-SA cliff are unchanged \u2014 the damping should touch only the deadlock salience branch, not the confidence path.

**Outcome of your advice:** iter 3 candidate loss=0.0872 -> iter 4 candidate loss=0.0880 -> the gate marked it REJECTED.

### Iteration 4
Verdict: regenerate
Interpretation: The candidate was REJECTED by the gate (0.0880 vs the iter-3 base at 0.0872), and the per-experiment decomposition shows exactly why it was a wash rather than a win. The proposer implemented my knob-2 (widen sigma 0.36-0.46 -> 0.48-0.60, raise g_res 0.06-0.14 -> 0.13-0.19, small mu bump) and skipped knob-1. The knob-2 direction was directionally right where I predicted: Exp 6 .598 -> .641 (obs .841), Exp 9 .428 -> .483 (obs .490), Exp 5 .725 -> .750 (obs .820), Exp 11 .852 -> .828 (obs .513), Exp 10 .752 -> .757 (obs .850). But the same widening leaked into cells I did not anticipate, and the leak was expensive: Exp 7 blew up from +.058 to +.173 against an observed +.053 (a +.115 swing, the single largest new error), Exp 4 .424 -> .398 (obs .508), Exp 14 .259 -> .228 (obs .300), Exp 2 .769 -> .758, Exp 1 .042 -> .038. The mechanism of the Exp 7 leak is transparent: Exp 7 scores only 2-vs-1 cells, i.e. |dk| = 1, which is precisely where the enlarged residual/width now injects edge-salience force; because the high-load and low-load cells differ in WHICH rows discriminate, an edge-salience push at |dk|=1 manufactures a spurious load effect where humans show essentially none. So the knob-2 step was too large, not wrong in sign.

Second, and important for my own record: the proposer's refusal of my knob-1 (endorsement-damped salience) was correct and I retract that advice. They actually read the displays: Exp 10's scored pair A=[1,1,0,0,1,1] / B=[0,1,1,1,1,0] has BOTH mute rows as 1/1 mutual endorsements, as does Exp 6's class-2 display. My proposed discriminator therefore does not separate Exp 11-c4 (obs ~chance) from Exp 5/Exp 10 (obs .82/.85); it would have damaged the two largest residuals to buy the third. That diagnosis is dead and should not be resurrected.

That leaves Exp 11 (.828 vs .513, ~0.31 of the total loss) as the dominant residual with no working structural story, plus Exp 6 (.641 vs .841) second. Note the arithmetic of the Exp-11 index (c4 + c8 - c5): if all three cells sat near chance the index would be exactly .50, essentially the observed .513. The model's .828 therefore says its Exp-11 cells are collectively far too decisive, in the same direction as Exp 3 (.468 vs .518), Exp 4 (.398 vs .508) and Exp 14 (.228 vs .300) \u2014 all four are cases where humans are closer to chance than the model on heavily conflicted displays. The arbiter's prescribed choice-difficulty-scaled lapse is the one component built for exactly this and it is the least-tuned knob in the model (eps0 .05-.11, eps1 .16-.28, unchanged across all four iterations). The rest of the core remains strong and experiment-invariant: Exp 2, 8, 12, 13, 15, 16 are all well matched and must be protected.
Rationale: The candidate was rejected, so the base reverts to iter 3; build on that base with three in-family moves, in priority order. Stay inside the prescribed family (compressive tally with top-end premium + sqrt numerosity + relative read-out + evidence-fraction confidence with no sign reversal + symmetric edge salience with residual + difficulty-scaled lapse + authority-checked stop). No new mechanisms.

1. SPLIT THE DIFFERENCE ON THE SALIENCE GATE \u2014 I am flanking my own two previous recommendations, so take the midpoint, not another full step. iter-3 had sigma [0.36,0.46] / g_res [0.06,0.14] (Exp 6 .598, Exp 7 .058); iter-4 had [0.48,0.60] / [0.13,0.19] (Exp 6 .641, Exp 7 .173). Use sigma ~[0.42,0.50] and g_res ~[0.09,0.13], and REVERT mu to the iter-3 range (the mu bump bought little and loads on Exp 11). This should keep roughly half the Exp 6 / Exp 9 / Exp 5 gain at roughly a quarter of the Exp 7 cost. Treat Exp 7 as a hard guardrail: if the simulated value exceeds ~+0.10, g_res is still too large. Do not push g_res above 0.13 again \u2014 that direction has now been tested and rejected by the gate.

2. TUNE THE DIFFICULTY-SCALED LAPSE \u2014 this is the untouched knob and the one the arbiter explicitly prescribed for exactly this residual class. Raise eps1 to ~[0.24,0.34] (keep eps0 at ~[0.05,0.09]) so that only genuinely low-|drive| conflict displays flatten toward chance. Expected direction: Exp 11 .828 -> materially lower (its index goes to .50 when its three cells approach chance), Exp 3 .468 -> ~.49, Exp 4 .398 -> ~.44, Exp 14 .228 -> ~.25, Exp 13 stays near .35. Guardrails: Exps 5 (.750), 6 (.641), 10 (.757), 15 (.799) are ALL already below observed and a global flattening will hurt them \u2014 so the increase must come through eps1 (which decays as exp(-|z|)) and NOT through eps0 or a lower beta. If Exps 5/10 drop below ~.72 the step was too big; back eps1 down to ~[0.20,0.26].

3. REPORT EXP 11 CELL-BY-CELL BEFORE TUNING IT. I have asked twice and it has not been done; blind index-fitting is precisely why iterations 3 and 4 were a wash. Print the model's c4 ('11001|01110'), c5 ('11101|00111') and c8 ('11101|00110') rates on the accepted base and after each change. The index is a signed combination, so a global confidence change can move it either way. Note the added constraint: c8's display is c5's plus ONE extra supporting weak cue at the bottom row, so under a strictly monotone tally c8 >= c5 and the index is bounded below by c4; the ONLY way to reach .513 while keeping c4 high is for the relative (dilution) read-out to make c8 < c5 \u2014 i.e. adding a cheap bottom-row reason must slightly DILUTE confidence. That is already a property of the denominator (kap2 + Xs); check whether kap2 at the low end of its range makes that dilution strong enough, and if so tune kap2 rather than blunting the salience gate.

Explicitly DO NOT: (a) resurrect endorsement-damped salience \u2014 the display check killed it; (b) raise alpha_v / extend the top-end premium (it trades Exps 1/3/4/12 against 6/9/10 roughly one-for-one); (c) touch the authority-checked stop this round (Exp 1 is at .038 vs .093 with diminishing returns and further firming risks Exps 13/14); (d) change the confidence path constants (omega, eta, q_sil, kappa, c_min) \u2014 Exps 7, 15, 16 depend on them and are currently acceptable.

**Outcome of your advice:** iter 4 candidate loss=0.0880 -> iter 5 candidate loss=0.1150 -> the gate marked it REJECTED.

### Iteration 5
Verdict: regenerate
Interpretation: The candidate was REJECTED and badly so (0.1150 vs the 0.0872 base — the largest single-iteration regression of this loop). The ledger is unambiguous about which of the two edits caused it. The eps1 raise (0.16-0.28 -> 0.26-0.36) was benign-to-helpful in the direction I predicted: Exp 3 and Exp 14 moved only slightly, Exp 11 did not blow up further. The damage came entirely from the q_sil reshape (1.05-1.35 -> 1.90-2.50), i.e. from the confidence path I had explicitly listed as off-limits this round because Exps 1, 7, 15, 16 all load on it. Making partial silence 'nearly free' inflated confidence in exactly the displays whose observed behaviour is near chance or sign-flipped: Exp 16 collapsed from +.137 to -.048 against obs +.097 (a .18 swing; the SE-SA cliff and the oracle-reversal term now cancel in the wrong order again, exactly the iteration-1 pathology), Exp 1 flipped back from +.042 to -.054 against obs +.093 (undoing two iterations of hard-won gate firming), Exp 4 fell .424 -> .318 (obs .508), Exp 14 .259 -> .178 (obs .300), Exp 3 .476 -> .428, Exp 12 overshot .665 -> .703. Against that, the convex map bought real but smaller gains on the under-confident deadlock/partial-silence cells: Exp 5 .725 -> .785 (obs .820), Exp 6 .598 -> .668 (obs .841), Exp 9 .428 -> .496 (obs .490), Exp 10 .752 -> .780 (obs .850), Exp 7 held at +.055. So the silence exponent is a genuine lever on the Exp 5/6/9/10 under-confidence cluster, but at q~2.2 the step is far too large and it pays for those gains three times over in the oracle/silence-contrast experiments (1, 4, 14, 16).

Structurally the model is still faithful to the arbiter's prescribed family (compressive tally with top-end premium, sqrt numerosity, relative read-out, strictly positive non-reversing confidence, symmetric two-regime edge salience with residual, difficulty-scaled lapse, authority-checked stop) — no family drift. Exp 11 (.890 vs .513) remains the dominant residual and is still being fitted blind: the cell-by-cell decomposition (c4, c5, c8) was asserted in prose but the arithmetic constraint I flagged (c8 = c5 + one extra supporting weak cue, so index >= c4 under any monotone tally) still has only one in-family escape route — the relative read-out's denominator — and that route (kap2) has never actually been moved.
Rationale: The gate rejected this candidate, so the base reverts to iter 3; build on iter 3 with the following in-family moves. Stay in the prescribed family (compressive tally + top-end premium, sqrt numerosity, relative read-out, non-reversing evidence-fraction confidence, symmetric edge salience with residual, difficulty-scaled lapse, authority-checked stop). No new mechanisms, no family switch.

1. ISOLATE THE ONE EDIT THAT MAY HAVE HELPED: raise eps1 ALONE, modestly, with q_sil held at the iter-3 value. Set eps1 ~[0.20, 0.28] (not 0.26-0.36) and leave eps0, beta, sigma, g_res, mu, alpha_v, rho, lam untouched. This iteration bundled eps1 with q_sil and the bundle was rejected, so the lapse knob has still never been tested cleanly. Expected: Exp 3 -> ~.49, Exp 4 -> ~.44, Exp 11 index down a few points, Exps 13/14 roughly flat. Guardrail: if Exps 5 or 10 fall below ~.72, step eps1 back to ~[0.18,0.23].

2. TUNE kap2 (the relative read-out denominator) — still the only untouched, arithmetically motivated route to Exp 11. Under a monotone tally c8 >= c5, so the index is bounded below by c4 unless adding a cheap bottom-row reason DILUTES confidence, which is precisely what the (kap2 + Xs) denominator can do. Try kap2 at/below its low end (~[0.20, 0.35]) and report c4, c5, c8 numerically before and after. Guardrail: Exp 12 (currently exact at .665) and Exp 2 (.769) are sensitive to the denominator in large panels — if Exp 12 moves more than ~.03, back off.

3. IF you touch the silence exponent at all, take a TOKEN step, not the one just rejected. q_sil ~[1.25, 1.55] at most (the flank between the iter-3 base ~1.2 and the rejected ~2.2), and only after checking two hard guardrails on the simulated output: Exp 16 must stay within [+0.05, +0.20] and Exp 1 must stay >= +0.03. The Exp 5/6/9/10 gains are real but they are worth roughly a third of what the Exp 1/4/14/16 losses cost; a half-step is the most the ledger supports. Do not repeat the full convex reshape — that direction has now been tested and rejected.

EXPLICITLY DO NOT: (a) change omega, eta, kappa or c_min — Exps 7 (+.058), 15 (.802) and 16 (+.137) on the base depend on them and this round showed how fast that fit evaporates; (b) touch the authority-checked stop (Exp 1's +.042 on the base is hard-won and was destroyed this round by an indirect confidence change — protect it); (c) resurrect endorsement-damped salience (killed by the display check in iteration 4); (d) move sigma/g_res again — iter 3 tightened, iter 4 widened, both washed, so leave them at the iter-3 base; (e) raise alpha_v.

Treat Exp 11 as partially irreducible if knob 2 cannot reach it: it is worth more to protect the broad invariant core (Exps 2, 7, 8, 12, 13, 15, 16 all well fit on the base) than to buy .3 on one index at the cost of .15 on four other experiments, which is exactly the trade this iteration made.

**Outcome of your advice:** iter 5 candidate loss=0.1150 -> iter 6 candidate loss=0.0990 -> the gate marked it REJECTED.

### Iteration 6
Verdict: regenerate
Interpretation: The iter-6 candidate was REJECTED (0.0990 vs the iter-3 base at 0.0872), so the base is unchanged. The single edit (d_com 2.00-2.55 -> 2.40-2.95) did exactly what the proposer predicted on its target cell 
type 	Exp 15 rose .802 -> .840 (obs .855), the best value this loop has produced on that metric 	but the collateral was larger than forecast and went the wrong way almost everywhere the stop can fire even weakly: Exp 1 collapsed .042 -> .001 (obs .093), undoing two iterations of hard-won gate firming; Exp 8 degraded -.170 -> -.144 (obs -.203); Exp 6 .598 -> .584 (obs .841); Exp 10 .752 -> .742 (obs .850); Exp 11 .852 -> .877 (obs .513). The lesson is now empirically settled: the authority-checked stop is at its ceiling of usefulness 	it has been pushed three times (iters 2, 3, 6), the first two paid and the third did not. It should be frozen at the iter-3 range permanently.

Stepping back over the whole loop: iters 4, 5 and 6 were all single-knob moves along a GLOBAL-DECISIVENESS axis (salience width/residual, silence exponent, commitment strength). All three were washes or regressions, and the proposer's own sensitivity ledger explains why 	the model sits at a genuine saddle in overall gain: roughly half the experiments (5, 6, 10, 12, 15, 2, 8) are on the under-confident side and roughly half (11, 4, 3, 9, 13, 14, 16) are on the over-confident side, so any uniform gain change trades them one-for-one. I accept that diagnosis; it means further global-gain tuning is provably futile and my last three recommendations were aimed at the wrong axis.

The residual mass on the base is highly concentrated: Exp 11 (.852 vs .513) and Exp 6 (.598 vs .841) together are ~48% of the loss; everything else is <= .10. Exp 11 remains partly irreducible for the reason I established earlier (its c4 display type is near-identical to Exp 10's cell-5, yet humans score them .49 vs .85; and c8 = c5 + one extra supporting cue, so any monotone tally bounds the index below by c4). But crucially, the residuals are NOT all on one gain axis 	they have opposite signs along a VALIDITY-SHAPE axis, which no global knob can touch: Exps 3 (.458 vs .518), 4 (.424 vs .508), 11-c8 (three mids vs one oracle, model follows the mids far too often) and 11-c5 (two cheap cues vs a .86, model follows the .86 too often) all say the near-oracle cue needs MORE weight while the .80-.88 mid-heavyweights need LESS; Exp 6 says a cheap-cue coalition must beat its opposing top cue far more decisively than it does. That is a knee-location problem in the already-present top-end premium, not a scale problem.
Rationale: Rejected by the gate, so build on the unchanged iter-3 base. Stay strictly inside the prescribed family (compressive tally with top-end premium, sqrt numerosity, relative read-out, strictly positive non-reversing evidence-fraction confidence, symmetric two-regime edge salience with residual, difficulty-scaled lapse, authority-checked stop). No family drift 	the candidate is faithful and should stay that way.

0. FIRST, REVERT d_com to the iter-3 range [2.00, 2.55] and FREEZE the stop for the rest of the loop. Pushing it was my advice in iters 2-3 (accepted twice) but iter 6 shows it is now past its optimum; do not touch q_max, d_com, phi or s_stop again.

1. THE ONE AXIS NOT YET TRIED: RELOCATE AND SHARPEN THE TOP-END PREMIUM (v_knee / kappa_v have sat at [0.895,0.925] / [25,45] since iter 3 and have never been moved). Make the premium a genuinely CATEGORICAL near-oracle bonus rather than a smear over .89-.93: v_knee ~[0.925, 0.945], kappa_v ~[50, 90], alpha_v ~[0.60, 0.95]. This is the only in-family move whose residual signs all point the same way, because it changes RELATIVE weighting rather than gain: (a) a .95 cue gets heavier 	Exp 3 -> ~.50, Exp 4 -> ~.46-.48, Exp 11-c8 (three mids vs the oracle) drops off its ceiling; (b) a .86/.88 cue LOSES its current partial premium 	Exp 11-c5 rises, so the index c4+c8-c5 falls from both ends; (c) if Exp 6's opposing 'mid-list top cue' is below the new knee it also loses weight and Exp 6 should rise materially. Net predicted direction: Exp 11 down by .05-.12, Exp 3/4 up, Exp 6 up 	the two dominant residuals move together for the first time this loop.

2. VERIFY BEFORE TUNING (this has been asked for three times and still not done). Print parameters['validities'] for Exps 3, 4, 6, 10, 11 and the model's per-cell rates for Exp 6's two target classes and Exp 11's c4/c5/c8 on the base and after the change. The whole knob-1 argument hinges on whether Exp 6's position-2 cue sits above or below the new knee. If it turns out Exp 6's opposing cue IS a .95, knob 1 will hurt Exp 6 	in that case do NOT abandon knob 1 (it still buys 3/4/11), but pay for Exp 6 with a small lam increase (~[0.52, 0.68]) rather than by lowering v_knee back, and check Exp 12 stays within .03 of .665.

3. GUARDRAILS 	here is what must not regress (all currently good on the base): Exp 7 must stay <= +0.10 (it is +.058); Exp 12 within .03 of .665; Exp 13 in [.32,.37]; Exp 15 >= .79; Exp 16 in [+.05,+.20]; Exp 2 >= .76. Exp 8 is the one cell likely to pay for a stronger oracle premium (single-cue strong-minus-weak becomes less negative); if it rises above -.15, trim alpha_v to the low end rather than moving eta or rho.

EXPLICITLY DO NOT: (a) make any further GLOBAL decisiveness move 	beta, kap2, eps0/eps1, mu, sigma, g_res, q_sil have now each been tested and every one was a wash or a regression (iters 4, 5, 6); the proposer's saddle argument is correct and I am withdrawing my earlier eps1 and kap2 pushes; (b) touch the confidence-path constants omega, eta, kappa, c_min (Exps 7, 15, 16 depend on them and iter 5 showed how fast that fit evaporates); (c) resurrect endorsement-damped salience; (d) chase Exp 11's index directly 	treat any residual there after knob 1 as irreducible, since its c4 cell and Exp 10's cell-5 are the same display type with opposite human answers.

**Outcome of your advice:** iter 6 candidate loss=0.0990 -> iter 7 candidate loss=0.0978 -> the gate marked it REJECTED.

### Iteration 7 (most recent)
Verdict: regenerate
Interpretation: The iter-7 candidate was REJECTED (0.0978 vs the iter-3 base at 0.0872), so the base is unchanged and my iter-6 recommendation (relocate/sharpen the top-end premium: v_knee .920-.942, kappa_v 45-80, alpha_v .58-.92) is now empirically falsified as a net win. The per-experiment ledger shows why it was a wash-plus: it did what I predicted on a minority of cells 
- Exp 1 .042 -> .048 (obs .093), Exp 3 .476 -> .494 (obs .518), Exp 5 .725 -> .759 (obs .820), Exp 9 .428 -> .439 (obs .490), Exp 15 .802 -> .817 (obs .855) 
- but it paid for those on a larger set: Exp 6 .598 -> .571 (obs .841, now the worst it has ever been), Exp 10 .752 -> .728 (obs .850), Exp 4 .424 -> .412 (obs .508), Exp 2 .769 -> .748 (obs .775), Exp 12 .665 -> .642 (was exact), Exp 14 .259 -> .242 (obs .300), Exp 16 .137 -> .195 (obs .097), Exp 8 -.170 -> -.234 (obs -.203, now overshooting). Critically, the proposer's own verification note was right and my forecast was wrong: Exp 6's opposing mid-list cue behaves like a near-oracle, so de-premiumising the .86-.88 band could not lift Exp 6; and in Exp 10's cell-5 the .95's +5% gain outweighed the .86's -8% loss, so extremes-following FELL rather than rose. The validity-shape axis is therefore now tested in both directions (iter-3 smear vs iter-7 categorical knee) and neither location dominates.

Standing back: since iter 3 the loop has tested salience width/residual (iter 4), silence exponent + lapse (iter 5), commitment strength (iter 6) and premium location/sharpness (iter 7) 
- four different axes, all rejected. The model sits at a genuine saddle: Exps 5, 6, 9, 10 (and 15) are systematically UNDER-confident about cheap-cue coalitions and edge cues, while Exps 3, 4, 11, 13, 14, 16 are OVER-confident, so any pure gain move and any premium relocation trades them one-for-one. Exp 11 (.85 vs .51) and Exp 6 (.60 vs .84) remain ~half the loss. The candidate is faithful to the arbiter's prescribed family throughout (compressive tally, sqrt numerosity, relative read-out, strictly positive non-reversing evidence-fraction confidence, symmetric two-regime edge salience with residual, difficulty-scaled lapse, authority-checked stop) 
- there is no family drift to correct, only calibration.
Rationale: Rejected by the gate, so build on the UNCHANGED iter-3 base. Stay strictly inside the prescribed family; no new mechanisms, no family switch.

0. REVERT the premium to the iter-3 ranges (v_knee ~[0.895,0.925], kappa_v ~[25,45], alpha_v ~[0.45,0.85]) and FREEZE v_knee/kappa_v/alpha_v for the rest of the loop. I pushed this axis in iter 6, it was rejected in iter 7, and the display check shows the underlying premise (Exp 6's opposing cue is a mid-heavyweight) is false. Also keep the authority-checked stop frozen at the iter-3 range (that freeze was already established in iter 6).

1. ONE KNOB ONLY THIS ROUND: LOWER rho (mid-range compression). rho has sat at [0.46,0.58] since iter 2 and has never been moved downward while the top-end premium was in place 
- that specific COMBINATION is the one untried cell of the design. In iter 1 I pushed rho down and in iter 2 I retracted it, but the retraction rested on Exps 3/4 needing the .95 to hold its own; the premium term (absent in iter 1, present since iter 3) now discharges that job independently, so the two moves are no longer in conflict. Take a SMALL step: rho ~[0.40, 0.50] (overlapping the current low end), with lam, mu, sigma, g_res, kap2, beta, eps0/eps1, omega, eta, kappa, c_min, q_sil, tau and the whole stop block held EXACTLY at iter-3 values. Mechanism: lower rho raises the relative weight of the cheap cues that Exps 6, 10, 5 and 9 all require to win, without changing global gain, while the premium keeps the .95 heavy enough for Exps 3/4/11-c8.

Predicted direction and hard guardrails (report simulated values for all of these before concluding): Exp 6 up from .598 (target >= .64), Exp 10 up from .752 (target >= .78), Exp 5 up from .725 (target >= .75), Exp 9 toward .49, Exp 11 down a few points via c5. Guardrails 
- if ANY of these breaks, step rho back to ~[0.44,0.52] rather than compensating with another knob: Exp 3 >= .45, Exp 4 >= .40, Exp 12 within .03 of .665, Exp 8 not below -0.25 (it is -.170 on the base and has room, but iter 7 showed how fast it overshoots), Exp 2 >= .76, Exp 7 <= +0.10, Exp 13 in [.32,.37], Exp 15 >= .79, Exp 16 in [+.05,+.20], Exp 1 >= +0.03.

2. DO NOT BUNDLE. Iterations 5, 6 and 7 each bundled or over-stepped and all three were rejected; a single conservative rho step is the cleanest remaining test and the only one whose residual signs (Exps 5, 6, 9, 10 all below observed, all wanting cheap cues relatively heavier) point the same way.

3. EXPLICITLY DO NOT: (a) move the premium location/sharpness again (tested, rejected); (b) make any further global decisiveness move 
- beta, kap2, eps0/eps1, mu, sigma, g_res, q_sil, d_com have each been tested and each was a wash or a regression; (c) touch the confidence-path constants omega, eta, kappa, c_min (Exps 7, 15, 16 depend on them and iter 5 showed how fast that fit evaporates); (d) resurrect endorsement-damped salience (killed by the display check in iter 4); (e) chase Exp 11's index directly 
- treat the residual there as largely irreducible, since its c4 display type is near-identical to Exp 10's cell-5 yet humans answer them oppositely (.49 vs .85). If the rho step also fails the gate, the honest conclusion is that the iter-3 base is at the family's optimum and the loop should ship it rather than keep trading experiments against each other.

**Outcome of your advice:** iter 7 candidate loss=0.0978 -> CURRENT candidate loss=0.0883 -> the gate marked it REJECTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The iter-8 candidate (a single small downward step in the mid-range compression exponent rho, everything else held at the iter-3 base) was REJECTED by the gate (0.0883 vs 0.0872). It is fully faithful to the arbiter's prescribed family \ncompressive tally with a top-end premium, sqrt numerosity, relative read-out, strictly positive non-reversing evidence-fraction confidence, symmetric two-regime edge salience with residual, difficulty-scaled lapse, authority-checked stop \nso there is no family drift to correct; this is purely a calibration verdict. The important finding is that my own iter-7 forecast for this move was largely WRONG. I predicted rho-down would lift Exps 6, 10, 5, 9 together. In fact only Exp 6 (.598 -> .606) and Exp 5 (.725 -> .734) moved up trivially, while Exp 10 FELL (.752 -> .725, obs .850) and Exp 9 FELL (.428 -> .414, obs .490) \nthe opposite of the predicted sign. The gains were elsewhere and real but small: Exp 8 -.170 -> -.223 (obs -.203), Exp 16 .137 -> .093 (obs .097, essentially exact), Exp 11 .852 -> .842. The losses: Exp 12 .665 -> .633 (was exact), Exp 1 .042 -> .030, Exp 14 .259 -> .236, Exp 2 .769 -> .763, Exp 4 .424 -> .401. Net: a near-perfect wash that lands just above the base.\n\nThe cumulative ledger is now decisive. Since iter 3, six distinct axes have been tested and all six rejected: salience width/residual (iter 4), silence exponent + lapse (iter 5), commitment strength (iter 6), premium location/sharpness (iter 7), mid-range compression rho (iter 8). Every one traded the under-confident cluster (Exps 5 .725, 6 .598, 9 .428, 10 .752, 15 .802) against the over-confident cluster (Exps 3, 4, 11, 13, 14, 16) roughly one-for-one. This is the signature of a genuine saddle in both gain and validity-shape space. The remaining residual mass is concentrated almost entirely in Exp 11 (.852 vs .513) and Exp 6 (.598 vs .841), and I have already established that Exp 11 is at least partly irreducible for a monotone tally (its c4 display type is near-identical to Exp 10's cell-5 yet humans answer them oppositely, .49 vs .85; and c8 = c5 plus one extra supporting cue, bounding the index below by c4). The broad invariant core (Exps 2, 7, 8, 12, 13, 15, 16) is well fit on the base and must be protected at all costs.",
  "verdict": "regenerate",
  "rationale": "Rejected by the gate, so build on the UNCHANGED iter-3 base. Stay strictly inside the prescribed family (compressive tally with top-end premium, sqrt numerosity, relative read-out, non-reversing evidence-fraction confidence, symmetric two-regime edge salience with residual, difficulty-scaled lapse, authority-checked stop). No new mechanisms.\n\n0. REVERT rho to the iter-3 range [0.46, 0.58]. My iter-7 push on rho is now falsified: the two experiments it was supposed to rescue (10, 9) moved the WRONG way. rho joins the frozen list alongside v_knee/kappa_v/alpha_v, the authority-stop block (q_max, d_com, phi, s_stop), the confidence-path constants (omega, eta, kappa, c_min, q_sil), sigma/g_res, kap2, beta and eps0/eps1 \nevery one of which has now been tested and rejected.\n\n1. ONE KNOB, AND IT IS THE LAST UNTRIED ONE: tau, the salience DECAY WIDTH. tau has sat at [0.85, 1.25] since iteration 1 and has never been moved in either direction. Crucially it is a SHAPE knob, not a gain knob \nit changes the ratio of edge-row salience to near-edge/mid-row salience while leaving the total drive scale roughly intact \nand the only previous shape move I made (adding the top-end premium in iter 2) is also the only advice of mine that the gate accepted after iter 2. Take a SMALL step DOWNWARD: tau ~ [0.62, 0.88], with every other parameter byte-identical to iter 3. Mechanism: with tau ~1.0 the exponential leaks substantial salience onto rows 1 and n-2, so in the extremes-vs-mid-heavyweight displays that Exps 5, 9, 10 are built on, the mid-list opponents receive almost as much positional boost as the true screen extremes and the tie-breaker partly cancels itself. Sharpening the decay concentrates the push on literal rows 0 and n-1 only, which is exactly the contrast those three metrics score.\n\nPredicted direction, with the numbers to report before concluding: Exp 10 up from .752 (target >= .79), Exp 5 up from .725 (target >= .76), Exp 9 up toward .49 from .428, Exp 6 roughly flat or slightly up (its class-2 cells have agreeing rows 0 and 5, so they gain). HARD GUARDRAILS \nif any breaks, halve the step to tau ~[0.75, 0.95] rather than compensating with a second knob: Exp 7 <= +0.10 (this is the most exposed metric, since its 2-vs-1 cells differ in WHICH rows discriminate \nif it exceeds +0.10 the step is too big); Exp 11 must not exceed .88; Exp 12 within .03 of .665; Exp 3 >= .45; Exp 4 >= .40; Exp 8 in [-0.25, -0.15]; Exp 13 in [.32, .37]; Exp 14 >= .24; Exp 15 >= .79; Exp 16 in [+.05, +.20]; Exp 1 >= +0.03; Exp 2 >= .76.\n\n2. DO NOT BUNDLE, and do not repeat any previously rejected axis. Specifically do NOT: move rho again (both directions now tested); move the premium location/sharpness (tested, rejected); touch the stop block (frozen since iter 6); touch omega/eta/kappa/c_min/q_sil (Exps 7, 15, 16 depend on them and iter 5 showed how fast that fit evaporates); move sigma/g_res or kap2 or eps0/eps1 or beta (all washes); resurrect endorsement-damped salience (killed by the display check in iter 4); or chase Exp 11's index directly \ntreat it as largely irreducible.\n\n3. STOPPING RULE. If this single tau step also fails the gate, the honest conclusion is that the iter-3 base sits at this family's optimum given the current experiment set, and the loop should ship it rather than keep trading the under-confident cluster against the over-confident one. Six axes have now been exhausted; tau is the seventh and last, and it is the only one whose effect is on the relative geometry of the salience term rather than on global decisiveness."
}
```

## Usage

```json
{
  "input_tokens": 58146,
  "output_tokens": 3655
}
```
