# feedback_iter_03

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
- THEORY 1 = `pi_7`
- THEORY 2 = `pi_9`
- The recommendation below acts on THEORY 2 (= `pi_9`).

Replace pi_9 with a DOUBT-THRESHOLD-GATED MIXTURE theory (working name: DTG-2R, 'Doubt-Triggered One-Reason with compensatory fallback'). Core claims:

(1) ABSOLUTE doubt trigger, not relative argmax. An expert commands one-reason adjudication only if its PRINTED validity falls below an absolute threshold theta (fit range ~.56-.64), i.e. only experts that are essentially coin-flips are 'newsworthy'. Trigger strength should be a steep but smooth function of (theta - v), e.g. g = sigmoid((theta - v)/tau) with tau ~ .01-.04.

(2) When a sub-threshold expert discriminates, choice follows its verdict with very high probability, independent of how many higher-validity rows oppose it: target ~.85-.90 after lapses on 1-vs-1 through 1-vs-5 and on 2-vs-2 / 3-vs-3 ties (this is what Exps 1 and 2 demand: .862 and .826, flat across coalition size). The load lapse must be MILD in these cells — noticeably milder than pi_7's current eps0/rho, which drag these to .72-.76.

(3) When NO sub-threshold expert discriminates, the trigger is off and evidence is compensatory: a divisively-normalized, anti-validity ((1-v)^gamma) and recency-tilted (exp(lam*j)) weighted sum over discriminating rows, so that a lone mid-doubt dissenter (v~.62-.68) opposed by a coalition of 2-3 sits at .45-.56 (Exps 15, 16) and conflict-display metrics land near .42-.60 (Exps 12, 13).

(4) Silent sub-threshold experts do NOT need a special 'capture' mechanism: the near-chance behaviour in silent-anchor displays follows automatically from (3). Drop null-verdict capture and the doubt-gradient leak, or keep at most a small residual — they are now redundant and were the source of pi_7's over-prediction in Exps 15/16.

(5) Preserve the two features already validated across the corpus: the label-degeneracy collapse to an unweighted tally when two printed validities are within .002 (Exps 10, 11 fit at ~.77-.82 via recency among near-equal labels), and a load-dependent lapse that keeps unanimous 6-0 dominance at ~.74 (Exp 7) rather than ceiling.

The new theory differs from pi_7 by making the one-reason rule conditional on an ABSOLUTE near-chance validity rather than on being the relative minimum among discriminating rows, and by being compensatory when no near-chance expert speaks; it differs from pi_9 by denying that coalitions can ever overturn a genuinely near-chance dissenter. Parameter ranges should be chosen so that, at mid-box, the model reproduces: Exp1 ~.86, Exp2 ~.83, Exp15 ~.54, Exp16 ~.47, Exp7 ~.74, Exp13 ~.59, Exp12 ~.42, Exp14 ~.26, Exp6 ~.15.


## CANDIDATE THEORY
DTG-2R/OC-sharp — Doubt-Threshold-Gated one-reason choice with compensatory fallback, where the doubt gate is an ABSOLUTE and SHARP near-chance criterion and the one-reason limb is degraded by OPPOSITION rather than by raw display length. (1) One-reason adjudication is commanded only by an expert whose PRINTED validity is essentially a coin flip. The trigger is g_j = sigmoid((theta - v_j)/tau) evaluated only over rows that actually discriminate (P_trig = max_{j in disc} g_j), and the criterion is now sharp and low: theta ~ .558-.588 with tau ~ .006-.013. This means an expert at .52-.55 is 'newsworthy' (g >= .9) while an expert at .62-.68 is NOT (g <= .02): a merely mediocre expert is treated as ordinary evidence, not as a scandal demanding adjudication. Silent near-chance experts are irrelevant, so no null-verdict capture and no doubt-gradient leak are needed. (2) When a genuinely sub-threshold expert speaks, its verdict decides the trial essentially independently of how MANY higher-validity rows oppose it; among discriminating rows the consulted verdict is drawn from softmax(log g_j + mu*j), so the recency tilt mu only arbitrates BETWEEN near-chance rows (e.g. three ~.54 columns). (3) Because a single row is consulted, merely scanning extra rows costs the one-reason limb almost nothing (rho_t ~ .985-1.0), but holding a genuine RIVAL COALITION does cost it: each minority row beyond the first charges rho_tc (~.62-.86), keep_t = (1-eps_t)*rho_t^(m-2)*rho_tc^(k_conf-1) with k_conf the size of the minority side. Hence a near-chance dissenter standing ALONE against 1-5 opposing rows is followed at ~.82-.86 flat in coalition size (Exps 15, 16), whereas BALANCED splits (2-vs-2, 3-vs-3 tally ties) drift markedly toward chance: the adjudication, not the scan, is what is lost. (4) When no near-chance expert discriminates — including displays whose least valid dissenter is only mid-doubt (v ~ .58-.68) — evidence is compensatory: anti-validity, weakly recency-tilted weights w_j = (1-v_j)^gamma * exp(lam*j), divisively normalised to the loudest dissent, E = sum(s_j*what_j)/(sigma + sum(what_j)), p = sigmoid(beta*E), with its own separable load (rho_m) and opposition (rho_c) lapses, so unanimous 6-row displays stay confident (~.74) while a lone mid-doubt dissenter against a coalition sits near chance (~.46-.55). (5) If any two printed validities are within .002 the experts cannot be individuated, the doubt gate has no referent and choice collapses to a graded unweighted row tally (exact coin flip on 1-1 splits). No feedback is given, so the rule is stationary across the block.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    # DTG-2R/OC: Doubt-Threshold-Gated one-reason choice with compensatory fallback;
    # one-reason limb degraded by OPPOSITION (rho_tc) rather than by raw row count
    import numpy as np

    # ---------- parse the (2, n_features) stimulus robustly ----------
    a = None
    b = None
    if isinstance(state, dict):
        if 'option_a_ratings' in state and 'option_b_ratings' in state:
            a = np.asarray(state['option_a_ratings'], dtype=float).ravel()
            b = np.asarray(state['option_b_ratings'], dtype=float).ravel()
        else:
            vals = list(state.values())
            if len(vals) >= 2:
                a = np.asarray(vals[0], dtype=float).ravel()
                b = np.asarray(vals[1], dtype=float).ravel()
    if a is None or b is None:
        arr = np.asarray(state, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == 2:
            a, b = arr[0].ravel(), arr[1].ravel()
        elif arr.ndim == 2 and arr.shape[1] == 2:
            a, b = arr[:, 0].ravel(), arr[:, 1].ravel()
        elif arr.ndim == 1:
            h = arr.shape[0] // 2
            a, b = arr[:h], arr[h:2 * h]
        else:
            f = np.asarray(arr, dtype=float).reshape(2, -1)
            a, b = f[0].ravel(), f[1].ravel()

    n = int(min(np.size(a), np.size(b)))
    if n == 0:
        return np.array([0.5, 0.5])
    a = np.asarray(a, dtype=float)[:n]
    b = np.asarray(b, dtype=float)[:n]

    # ---------- printed validities ----------
    v_raw = parameters.get('validities', None) if hasattr(parameters, 'get') else None
    if v_raw is None:
        v = np.linspace(0.95, 0.55, n)
    else:
        v = np.asarray(v_raw, dtype=float).ravel()
        if v.size < n:
            v = np.concatenate([v, np.full(n - v.size, 0.55)])
        v = v[:n]
    v = np.clip(v, 0.5, 0.9995)

    # ---------- parameters ----------
    theta = float(parameters['theta'])
    tau = float(max(float(parameters['tau']), 1e-4))
    mu = float(parameters['mu'])
    gamma = float(parameters['gamma'])
    lam = float(parameters['lam'])
    sigma = float(max(float(parameters['sigma']), 1e-6))
    beta = float(parameters['beta'])
    eps_t = float(np.clip(float(parameters['eps_t']), 0.0, 1.0))
    rho_t = float(np.clip(float(parameters['rho_t']), 1e-6, 1.0))
    rho_tc = float(np.clip(float(parameters['rho_tc']), 1e-6, 1.0))
    eps0 = float(np.clip(float(parameters['eps0']), 0.0, 1.0))
    rho_m = float(np.clip(float(parameters['rho_m']), 1e-6, 1.0))
    rho_c = float(np.clip(float(parameters['rho_c']), 1e-6, 1.0))
    kappa = float(parameters['tally_kappa'])

    def sig(x):
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

    # ---------- discriminating rows ----------
    diff = a - b
    disc = np.nonzero(diff != 0.0)[0]
    m = int(disc.size)
    if m == 0:
        return np.array([0.5, 0.5])
    signs = np.sign(diff[disc])
    n_a = float(np.sum(signs > 0.0))
    n_b = float(m) - n_a
    k_conf = int(min(n_a, n_b))

    # ---------- lapses ----------
    expo = max(m - 2, 0)
    keep_c = (1.0 - eps0) * (rho_m ** expo) * (rho_c ** k_conf)
    keep_c = float(min(max(keep_c, 0.0), 1.0))
    # one-reason limb: scanning extra rows is nearly free, holding a RIVAL
    # coalition is not -> charge rho_tc per minority row beyond the first
    keep_t = (1.0 - eps_t) * (rho_t ** expo) * (rho_tc ** max(k_conf - 1, 0))
    keep_t = float(min(max(keep_t, 0.0), 1.0))

    # ---------- label-degeneracy gate ----------
    degenerate = False
    if n > 1:
        vs = np.sort(v)
        if bool(np.any(np.diff(vs) < 2.0e-3)):
            degenerate = True

    if degenerate:
        p_core = sig(kappa * (n_a - n_b))
        p_a = 0.5 + keep_c * (p_core - 0.5)
        p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
        p = np.array([p_a, 1.0 - p_a], dtype=float)
        return p / p.sum()

    # ---------- absolute doubt trigger over DISCRIMINATING rows ----------
    zz = np.clip((v - theta) / tau, -60.0, 60.0)
    gtrig = 1.0 / (1.0 + np.exp(zz))          # = sigmoid((theta - v)/tau)
    gd = gtrig[disc]
    P_trig = float(np.clip(np.max(gd), 0.0, 1.0))

    # verdict of the consulted (near-chance) expert; recency tilt among them
    logs = np.log(np.clip(gd, 1e-300, None)) + mu * disc.astype(float)
    logs = logs - np.max(logs)
    s = np.exp(np.clip(logs, -700.0, 0.0))
    ssum = float(np.sum(s))
    if (not np.isfinite(ssum)) or ssum <= 0.0:
        s = np.full(m, 1.0 / m)
    else:
        s = s / ssum
    p_or_a = float(np.sum(s[signs > 0.0]))
    p_t = 0.5 + keep_t * (p_or_a - 0.5)

    # ---------- compensatory fallback (divisively normalised anti-validity) ----------
    logw = gamma * np.log(np.clip(1.0 - v, 1e-12, None)) + lam * np.arange(n, dtype=float)
    ld = logw[disc]
    ld = ld - np.max(ld)
    w = np.exp(np.clip(ld, -700.0, 0.0))
    wsum = float(np.sum(w))
    if (not np.isfinite(wsum)) or wsum <= 0.0:
        w = np.ones(m, dtype=float)
        wsum = float(m)
    E = float(np.sum(signs * w)) / (sigma + wsum)
    p_comp = sig(beta * E)
    p_c = 0.5 + keep_c * (p_comp - 0.5)

    # ---------- doubt-threshold-gated mixture ----------
    p_a = P_trig * p_t + (1.0 - P_trig) * p_c
    p_a = float(min(max(p_a, 1e-9), 1.0 - 1e-9))
    p = np.array([p_a, 1.0 - p_a], dtype=float)
    return p / p.sum()

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()
    if not np.isfinite(s) or s <= 0.0:
        return int(np.random.randint(len(probs)))
    probs = probs / s
    return int(np.random.choice(len(probs), p=probs))

`parameters`:
- theta: [0.558, 0.588]
- tau: [0.006, 0.013]
- mu: [0.45, 0.95]
- gamma: [1.5, 2.3]
- lam: [0.0, 0.22]
- sigma: [0.40, 0.72]
- beta: [1.8, 2.6]
- eps_t: [0.20, 0.36]
- rho_t: [0.985, 1.00]
- rho_tc: [0.62, 0.86]
- eps0: [0.13, 0.27]
- rho_m: [0.89, 0.97]
- rho_c: [0.74, 0.90]
- tally_kappa: [0.6, 1.3]
- validities: validities

`rationale`:
MINIMAL DIFF on the ACCEPTED base (loss .0692): `predict` and `policy` are re-emitted byte-identical; only three parameter ranges change, exactly the two edits the critic prescribed after this base was accepted.

(1) PRIMARY — sharpen and lower the doubt gate: theta [.572,.605] -> [.558,.588], tau [.008,.020] -> [.006,.013]. The critic's decomposition of Exp14 (.581 = P_trig*.85 + (1-P_trig)*.454) shows the compensatory limb is ALREADY at the human value (.468) there and the entire +.114 residual is trigger leakage at v=.62. At the new mid-box (theta=.573, tau=.0095): g(.52)~1.00, g(.538)~1.00, g(.545)~.94, g(.55)~.90, g(.58)~.32, g(.62)~.007, g(.68)~0. So the .62 column in Exp14 is now essentially excluded from the one-reason limb (predicted .58 -> ~.46-.48, the second-largest residual eliminated), while every anchor the corpus depends on stays inside the gate: Exp15's .55, Exp16's .52, Exp8's .535/.540/.545, Exp9's .538/.542/.546, Exp11's .55, Exp12's .51, Exp1's .55, Exp4's .53. Exp13 is untouched (its lone dissenter is v=.68, g~0 before and after). theta is kept >= .558 and tau >= .006 precisely so the .545/.55 anchors keep g >= .9, as the critic required.

(2) SECONDARY — narrow rho_t [.96,1.00] -> [.985,1.00]. rho_t acts only through rho_t^(m-2), so this recovers the lone-anchor cells lost to row-scan decay in long displays: Exp15 (.785, real .862) and Exp16 (.789, real .826) gain ~+.02-.03 and ~+.02, and Exp9 (.791, real .819) gains too, while the m=2-3 displays of Exps 1/3/4 (currently well placed) are unmoved. I did NOT cut eps_t: a blanket eps_t cut was tested in iter 2 and the gate rejected it.

(3) EVERYTHING ELSE FROZEN, as instructed: rho_tc [.62,.86] (the edit the gate rewarded) stays exactly as implemented; eps0/rho_m/rho_c/mu/gamma/lam/sigma/beta/tally_kappa and the .002 degeneracy gate are untouched, protecting the best-in-corpus cells Exp10 (.416 vs .418), Exp7 (.433 vs .438), Exp6 (~.49 vs .507), Exp13 (.559 vs .546). Exp2 is deliberately not chased — it is a documented structural cost of the DTG family (all siblings overshoot it) and the only fixes would undo the accepted rho_tc edit or weaken the near-chance gate that Exps 15/16 depend on. Expected collateral is <= .02 on Exp3/Exp8/Exp11 from the sharper gate (Exp3's .58 column now routes ~2/3 of its mass through the compensatory limb, which is also anti-validity but more lapse-prone on 2-vs-2 ties, so it should move .239 slightly toward the real .286), against a ~.11 gain on Exp14 and ~.03 on Exp15/16/9 combined.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.0707 -> ACCEPTED
- iter 2: loss=0.0760 -> REJECTED
- iter 3: loss=0.0692 -> ACCEPTED
- iter 4 (current candidate you are grading): loss=0.0549 -> ACCEPTED
Running-best (last accepted) base: iter 4 at loss=0.0549.

## EXPERIMENTAL RESULTS (candidate vs real, per experiment)
### Experiment 1
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    """One-reason adherence on model-disagreement trials.

    For every trial we compute (a) the option favoured by the FIRST
    discriminating cue in validity order (features are listed in
    descending validity: 0.95, 0.85, 0.75, 0.65, 0.55) and (b) the option
    favoured by the unit-weight tally of feature-wise wins.  We keep only
    trials where these two disagree in direction OR where the tally is tied
    (so a counting rule must guess).  The metric is the proportion of such
    trials on which the subject picked the option favoured by the first
    discriminating cue.
    """
    hits = []
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = min(a.shape[0], b.shape[0])
        a = a[:n]
        b = b[:n]

        # first discriminating cue (features already in descending validity)
        first_winner = None
        for j in range(n):
            if a[j] > b[j]:
                first_winner = 0
                break
            if b[j] > a[j]:
                first_winner = 1
                break
        if first_winner is None:
            continue  # no cue discriminates: uninformative trial

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = None  # counting rule must guess

        if tally_winner is not None and tally_winner == first_winner:
            continue  # agreement control trial: not diagnostic

        try:
            r = int(resp)
        except (TypeError, ValueError):
            continue
        hits.append(1.0 if r == first_winner else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.2820 (var=0.0035)
**Candidate trajectory (this loop):**
  - iter 1: 0.2527 (var=0.0020) (Δ vs real -0.0292)
  - iter 2: 0.2335 (var=0.0029) (Δ vs real -0.0485)
  - iter 3: 0.2480 (var=0.0016) (Δ vs real -0.0340)
  - iter 4 (current): 0.2435 (var=0.0024) (Δ vs real -0.0385)
**Other theories' values on this metric (for reference):**
- pi_1: 0.8515 (var=0.0078)
- pi_2: 0.1903 (var=0.0042)
- pi_3: 0.2870 (var=0.0019)
- pi_4: 0.2843 (var=0.0034)
- pi_5: 0.2722 (var=0.0027)
- pi_6: 0.2560 (var=0.0021)
- pi_7: 0.2878 (var=0.0023)
- pi_8: 0.2730 (var=0.0019)
- pi_9: 0.3068 (var=0.0031)

### Experiment 2
**Design**
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.6, 0.93, 0.55, 0.87, 0.7, 0.78], dtype=float)

    def analyze(row):
        a = np.asarray(row['option_a_ratings'], dtype=float)
        b = np.asarray(row['option_b_ratings'], dtype=float)
        n = min(a.shape[0], b.shape[0])
        a = a[:n]
        b = b[:n]
        val = validities[:n] if validities.shape[0] >= n else np.arange(n, 0, -1, dtype=float)
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        margin = a_wins - b_wins
        if margin > 0:
            tally_winner = 0
        elif margin < 0:
            tally_winner = 1
        else:
            tally_winner = -1
        order = np.argsort(-val, kind='stable')
        ttb_winner = -1
        for j in order:
            if a[j] > b[j]:
                ttb_winner = 0
                break
            if b[j] > a[j]:
                ttb_winner = 1
                break
        return margin, tally_winner, ttb_winner

    info = [analyze(r) for _, r in data.iterrows()]
    margins = np.array([x[0] for x in info], dtype=float)
    tally_w = np.array([x[1] for x in info], dtype=int)
    ttb_w = np.array([x[2] for x in info], dtype=int)
    resp = data['response'].to_numpy().astype(int)

    # Group 1: tally-tie trials where TTB is decisive (parameter-free for Tallying)
    tie_mask = (tally_w == -1) & (ttb_w >= 0)
    # Group 2: trials where tally margin is >= 2 AND TTB points the other way
    dis_mask = (np.abs(margins) >= 2) & (tally_w >= 0) & (ttb_w >= 0) & (tally_w != ttb_w)

    if tie_mask.sum() == 0 and dis_mask.sum() == 0:
        return float('nan')

    if dis_mask.sum() > 0:
        p_tally = float(np.mean(resp[dis_mask] == tally_w[dis_mask]))
    else:
        p_tally = 0.5
    if tie_mask.sum() > 0:
        p_ttb_tie = float(np.mean(resp[tie_mask] == ttb_w[tie_mask]))
    else:
        p_ttb_tie = 0.5

    return p_tally - p_ttb_tie

```

**Observed (real) value:** 0.1744 (var=0.0127)
**Candidate trajectory (this loop):**
  - iter 1: 0.2897 (var=0.0103) (Δ vs real +0.1153)
  - iter 2: 0.2744 (var=0.0078) (Δ vs real +0.1000)
  - iter 3: 0.3622 (var=0.0155) (Δ vs real +0.1878)
  - iter 4 (current): 0.3214 (var=0.0119) (Δ vs real +0.1469)
**Other theories' values on this metric (for reference):**
- pi_2: 0.3828 (var=0.0220)
- pi_1: -0.7044 (var=0.0395)
- pi_3: 0.1722 (var=0.0132)
- pi_4: 0.1856 (var=0.0146)
- pi_5: 0.1956 (var=0.0184)
- pi_6: 0.1639 (var=0.0160)
- pi_7: 0.2300 (var=0.0149)
- pi_8: 0.2117 (var=0.0142)
- pi_9: 0.3256 (var=0.0184)

### Experiment 3
**Design**
  A=[0, 1, 0, 1, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 0, 0, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 0, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V = np.array([0.68, 0.96, 0.52, 0.88, 0.58, 0.78], dtype=float)

    hits = 0.0
    n = 0.0
    for _, row in data.iterrows():
        try:
            a = np.asarray([float(x) for x in row['option_a_ratings']], dtype=float)
            b = np.asarray([float(x) for x in row['option_b_ratings']], dtype=float)
        except Exception:
            continue
        k = int(min(a.shape[0], b.shape[0]))
        if k == 0:
            continue
        a = a[:k]
        b = b[:k]
        if V.shape[0] >= k:
            v = V[:k]
        else:
            v = np.concatenate([V, np.full(k - V.shape[0], 0.55)])[:k]

        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        # keep only unit-tally TIES with at least one discriminating cue
        if a_wins != b_wins or a_wins == 0:
            continue

        w = v ** 2.0
        d = float(np.sum(w * (a - b)))
        # discard tie trials whose validity-weighted margin is itself
        # near zero (no clear "higher-validity side" to predict)
        if abs(d) < 0.15:
            continue

        resp = row['response']
        try:
            resp = int(resp)
        except Exception:
            continue
        if resp not in (0, 1):
            continue

        chose_a = (resp == 0)
        picked_high_validity = chose_a if d > 0 else (not chose_a)
        hits += 1.0 if picked_high_validity else 0.0
        n += 1.0

    if n == 0:
        return 0.5
    return float(hits / n)
```

**Observed (real) value:** 0.2856 (var=0.0160)
**Candidate trajectory (this loop):**
  - iter 1: 0.2167 (var=0.0063) (Δ vs real -0.0689)
  - iter 2: 0.1583 (var=0.0049) (Δ vs real -0.1272)
  - iter 3: 0.2389 (var=0.0078) (Δ vs real -0.0467)
  - iter 4 (current): 0.2733 (var=0.0044) (Δ vs real -0.0122)
**Other theories' values on this metric (for reference):**
- pi_3: 0.9389 (var=0.0016)
- pi_2: 0.5100 (var=0.0066)
- pi_1: 0.8522 (var=0.0137)
- pi_4: 0.3056 (var=0.0052)
- pi_5: 0.2767 (var=0.0068)
- pi_6: 0.2928 (var=0.0057)
- pi_7: 0.2778 (var=0.0058)
- pi_8: 0.2794 (var=0.0061)
- pi_9: 0.2956 (var=0.0037)

### Experiment 4
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    v = np.array([0.58, 0.97, 0.53, 0.80, 0.66, 0.90], dtype=float)
    w = v ** 2.0  # AVAI-style amplified weights (gamma=2)

    num = 0.0
    den = 0.0
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = int(min(a.shape[0], b.shape[0], w.shape[0]))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        ww = w[:n]
        a_wins = float(np.sum(a > b))
        b_wins = float(np.sum(b > a))
        if a_wins != b_wins:
            continue  # not a tally tie
        d = float(np.sum(ww * (a - b)))
        if abs(d) < 0.30:
            continue  # tie trials whose weighted margin is negligible
        den += 1.0
        if d > 0:
            num += 1.0 if int(resp) == 0 else 0.0
        else:
            num += 1.0 if int(resp) == 1 else 0.0

    if den == 0:
        return 0.5
    return num / den
```

**Observed (real) value:** 0.1500 (var=0.0194)
**Candidate trajectory (this loop):**
  - iter 1: 0.1383 (var=0.0104) (Δ vs real -0.0117)
  - iter 2: 0.1050 (var=0.0066) (Δ vs real -0.0450)
  - iter 3: 0.1533 (var=0.0104) (Δ vs real +0.0033)
  - iter 4 (current): 0.1733 (var=0.0105) (Δ vs real +0.0233)
**Other theories' values on this metric (for reference):**
- pi_2: 0.4917 (var=0.0165)
- pi_3: 0.9850 (var=0.0010)
- pi_1: 0.8500 (var=0.0219)
- pi_4: 0.1583 (var=0.0128)
- pi_5: 0.1650 (var=0.0115)
- pi_6: 0.1450 (var=0.0086)
- pi_7: 0.1500 (var=0.0133)
- pi_8: 0.1217 (var=0.0081)
- pi_9: 0.1067 (var=0.0070)

### Experiment 5
**Design**
  A=[1, 1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 1]  B=[0, 0, 0, 1, 0, 1]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    correct = []
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()
            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        d = a[:n] - b[:n]
        nz = d[d != 0.0]
        if nz.size == 0:
            continue
        # dominance trial: every discriminating cue favours the same option
        if not (np.all(nz > 0) or np.all(nz < 0)):
            continue
        dominant_is_a = bool(np.all(nz > 0))
        try:
            resp = int(row['response'])
        except Exception:
            continue
        chose_a = (resp == 0)
        correct.append(1.0 if (chose_a == dominant_is_a) else 0.0)

    if len(correct) == 0:
        return 0.5
    return float(np.mean(correct))
```

**Observed (real) value:** 0.7427 (var=0.0173)
**Candidate trajectory (this loop):**
  - iter 1: 0.8013 (var=0.0070) (Δ vs real +0.0587)
  - iter 2: 0.7840 (var=0.0053) (Δ vs real +0.0413)
  - iter 3: 0.7987 (var=0.0041) (Δ vs real +0.0560)
  - iter 4 (current): 0.7847 (var=0.0051) (Δ vs real +0.0420)
**Other theories' values on this metric (for reference):**
- pi_3: 0.9847 (var=0.0005)
- pi_4: 0.7440 (var=0.0064)
- pi_1: 0.8340 (var=0.0138)
- pi_2: 0.8493 (var=0.0126)
- pi_5: 0.7427 (var=0.0074)
- pi_6: 0.7773 (var=0.0049)
- pi_7: 0.7467 (var=0.0048)
- pi_8: 0.7487 (var=0.0072)
- pi_9: 0.7473 (var=0.0050)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    # design validities (fixed by the experimental design)
    v_design = np.array([0.95, 0.70, 0.55, 0.95, 0.70, 0.55], dtype=float)
    hits = 0
    n = 0
    for a_raw, b_raw, r in zip(data['option_a_ratings'],
                               data['option_b_ratings'],
                               data['response']):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        k = int(min(a.size, b.size))
        if k == 0:
            continue
        if v_design.size >= k:
            vv = v_design[:k]
        else:
            vv = np.linspace(0.95, 0.55, k)
        d = a[:k] - b[:k]
        idx = np.nonzero(d != 0.0)[0]
        if idx.size != 2:
            continue
        j1 = int(idx[0])
        j2 = int(idx[1])
        # the two discriminating cues must point to OPPOSITE options
        if d[j1] * d[j2] > 0:
            continue
        # and the LATER cue must be the LESS valid one
        if not (vv[j2] < vv[j1]):
            continue
        late_choice = 0 if d[j2] > 0 else 1  # 0 = A, 1 = B
        try:
            resp = int(r)
        except Exception:
            continue
        n += 1
        if resp == late_choice:
            hits += 1
    if n == 0:
        return float('nan')
    return float(hits) / float(n)
```

**Observed (real) value:** 0.5067 (var=0.0112)
**Candidate trajectory (this loop):**
  - iter 1: 0.5175 (var=0.0075) (Δ vs real +0.0108)
  - iter 2: 0.4858 (var=0.0093) (Δ vs real -0.0208)
  - iter 3: 0.4850 (var=0.0092) (Δ vs real -0.0217)
  - iter 4 (current): 0.4983 (var=0.0111) (Δ vs real -0.0083)
**Other theories' values on this metric (for reference):**
- pi_4: 0.8367 (var=0.0052)
- pi_3: 0.0300 (var=0.0018)
- pi_1: 0.1550 (var=0.0169)
- pi_2: 0.4942 (var=0.0103)
- pi_5: 0.5083 (var=0.0072)
- pi_6: 0.5033 (var=0.0076)
- pi_7: 0.4817 (var=0.0130)
- pi_8: 0.4725 (var=0.0109)
- pi_9: 0.5208 (var=0.0068)

### Experiment 7
**Design**
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 0]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = []
    for _, row in data.iterrows():
        a = np.asarray(row['option_a_ratings'], dtype=float).ravel()
        b = np.asarray(row['option_b_ratings'], dtype=float).ravel()
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        d = a[:n] - b[:n]
        disc = np.nonzero(d != 0.0)[0]
        m = int(disc.size)
        if m < 2:
            continue
        fav_a = d[disc] > 0.0
        # skip unanimous (dominance) displays: both theories agree there
        if fav_a.all() or (~fav_a).all():
            continue
        # option favoured by the LAST-READ (highest-index) discriminating row
        last_idx = disc[-1]
        late_option = 0 if d[last_idx] > 0.0 else 1
        try:
            resp = int(row['response'])
        except Exception:
            continue
        hits.append(1.0 if resp == late_option else 0.0)

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4383 (var=0.0023)
**Candidate trajectory (this loop):**
  - iter 1: 0.4517 (var=0.0034) (Δ vs real +0.0133)
  - iter 2: 0.4294 (var=0.0033) (Δ vs real -0.0089)
  - iter 3: 0.4328 (var=0.0033) (Δ vs real -0.0056)
  - iter 4 (current): 0.4503 (var=0.0035) (Δ vs real +0.0119)
**Other theories' values on this metric (for reference):**
- pi_5: 0.4711 (var=0.0034)
- pi_4: 0.7642 (var=0.0020)
- pi_1: 0.2703 (var=0.0046)
- pi_2: 0.3842 (var=0.0037)
- pi_3: 0.3467 (var=0.0009)
- pi_6: 0.4419 (var=0.0028)
- pi_7: 0.4508 (var=0.0038)
- pi_8: 0.4408 (var=0.0032)
- pi_9: 0.4628 (var=0.0033)

### Experiment 8
**Design**
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[1, 1, 0, 0, 1, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design validities (fixed by the experiment); fallback handles other lengths.
    V_DESIGN = np.array([0.955, 0.545, 0.950, 0.540, 0.945, 0.535], dtype=float)
    LAM = 1.6
    DELTA = 0.9

    hits = []
    for _, row in data.iterrows():
        try:
            a = np.asarray(list(row['option_a_ratings']), dtype=float).ravel()
            b = np.asarray(list(row['option_b_ratings']), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]

        if V_DESIGN.size >= n:
            v = V_DESIGN[:n]
        else:
            v = np.concatenate([V_DESIGN, np.full(n - V_DESIGN.size, 0.545)])[:n]

        d = a - b
        disc = np.nonzero(d != 0.0)[0]
        m = int(disc.size)
        if m < 2:
            continue

        # salience of each discriminating row: recency x (1 - validity)
        j = disc.astype(float)
        logu = LAM * j + DELTA * np.log(np.clip(1.0 - v[disc], 1e-9, None))
        logu = logu - np.max(logu)
        u = np.exp(logu)
        top = disc[int(np.argmax(u))]
        top_owner_a = bool(d[top] > 0.0)  # True -> the salient row favours A

        n_a = int(np.sum(d[disc] > 0.0))
        n_b = m - n_a

        # include only trials where the row-count tally does NOT strictly
        # agree with the most salient (late / low-validity) discriminating row
        if n_a > n_b:
            majority_a = True
            tie = False
        elif n_b > n_a:
            majority_a = False
            tie = False
        else:
            majority_a = None
            tie = True
        if (not tie) and (majority_a == top_owner_a):
            continue

        try:
            resp = int(row['response'])
        except Exception:
            continue
        chose_a = (resp == 0)
        hits.append(1.0 if (chose_a == top_owner_a) else 0.0)

    if len(hits) == 0:
        return 0.5
    return float(np.mean(hits))
```

**Observed (real) value:** 0.7697 (var=0.0057)
**Candidate trajectory (this loop):**
  - iter 1: 0.8097 (var=0.0027) (Δ vs real +0.0400)
  - iter 2: 0.8600 (var=0.0021) (Δ vs real +0.0903)
  - iter 3: 0.8043 (var=0.0027) (Δ vs real +0.0346)
  - iter 4 (current): 0.7949 (var=0.0026) (Δ vs real +0.0251)
**Other theories' values on this metric (for reference):**
- pi_4: 0.7326 (var=0.0025)
- pi_5: 0.4900 (var=0.0045)
- pi_1: 0.1511 (var=0.0123)
- pi_2: 0.3354 (var=0.0031)
- pi_3: 0.0803 (var=0.0009)
- pi_6: 0.7540 (var=0.0026)
- pi_7: 0.7686 (var=0.0034)
- pi_8: 0.7611 (var=0.0025)
- pi_9: 0.7766 (var=0.0030)

### Experiment 9
**Design**
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V_DESIGN = np.array([0.958, 0.546, 0.954, 0.542, 0.950, 0.538], dtype=float)
    LAM = 1.5      # reading-order recency gain (shared by both accounts)
    GAM = 0.9      # oddness exponent on (1 - validity) (shared by both accounts)

    hits = 0.0
    tot = 0.0

    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, resp in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(a_raw, dtype=float).ravel()
            b = np.asarray(b_raw, dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]

        if n == V_DESIGN.size:
            v = V_DESIGN.copy()
        else:
            v = np.linspace(0.95, 0.55, n)
        v = np.clip(v, 0.5, 0.995)

        diff = a - b
        disc = np.nonzero(diff != 0.0)[0]
        m = int(disc.size)
        if m < 2:
            continue
        signs = np.sign(diff[disc])
        # keep only CONFLICT displays (both options win at least one row)
        if np.all(signs > 0) or np.all(signs < 0):
            continue

        logw = GAM * np.log(np.clip(1.0 - v[disc], 1e-9, None)) + LAM * disc.astype(float)
        k = int(np.argmax(logw))
        favored_a = signs[k] > 0.0

        try:
            r = int(resp)
        except Exception:
            continue
        chose_a = (r == 0)
        hits += 1.0 if (chose_a == favored_a) else 0.0
        tot += 1.0

    if tot == 0.0:
        return float('nan')
    return float(hits / tot)
```

**Observed (real) value:** 0.8189 (var=0.0074)
**Candidate trajectory (this loop):**
  - iter 1: 0.7989 (var=0.0017) (Δ vs real -0.0200)
  - iter 2: 0.8453 (var=0.0013) (Δ vs real +0.0264)
  - iter 3: 0.7911 (var=0.0030) (Δ vs real -0.0278)
  - iter 4 (current): 0.7969 (var=0.0025) (Δ vs real -0.0219)
**Other theories' values on this metric (for reference):**
- pi_5: 0.4964 (var=0.0034)
- pi_6: 0.7364 (var=0.0031)
- pi_1: 0.1528 (var=0.0126)
- pi_2: 0.3228 (var=0.0037)
- pi_3: 0.0178 (var=0.0002)
- pi_4: 0.7092 (var=0.0034)
- pi_7: 0.7714 (var=0.0022)
- pi_8: 0.7536 (var=0.0025)
- pi_9: 0.7783 (var=0.0025)

### Experiment 10
**Design**
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 1]
  A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # Design validities (fixed by the experiment)
    v_default = np.array([0.53, 0.72, 0.86, 0.93, 0.99], dtype=float)

    def weights(n):
        if n <= v_default.size:
            v = v_default[:n]
        else:
            v = np.concatenate([v_default, np.full(n - v_default.size, 0.99)])
        v = np.clip(v, 0.5, 0.9995)
        logw = 0.9 * np.log(1.0 - v) + 1.5 * np.arange(n, dtype=float)
        logw = logw - np.max(logw)
        return np.exp(logw)

    wcache = {}
    hits = []
    for a_raw, b_raw, resp in zip(data['option_a_ratings'].values,
                                  data['option_b_ratings'].values,
                                  data['response'].values):
        try:
            a = np.asarray(list(a_raw), dtype=float).ravel()
            b = np.asarray(list(b_raw), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]
        d = a - b
        nA = int(np.sum(d > 0))
        nB = int(np.sum(d < 0))
        # keep only CONFLICT displays (both options favoured by at least one expert)
        if nA == 0 or nB == 0:
            continue
        if n not in wcache:
            wcache[n] = weights(n)
        w = wcache[n]
        S = float(np.sum(np.sign(d) * w))
        if S == 0.0:
            continue
        fav = 0 if S > 0 else 1  # response coding: 0 = A, 1 = B
        try:
            r = int(resp)
        except Exception:
            continue
        hits.append(1.0 if r == fav else 0.0)

    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.4178 (var=0.0051)
**Candidate trajectory (this loop):**
  - iter 1: 0.4189 (var=0.0032) (Δ vs real +0.0011)
  - iter 2: 0.4192 (var=0.0023) (Δ vs real +0.0014)
  - iter 3: 0.4156 (var=0.0028) (Δ vs real -0.0022)
  - iter 4 (current): 0.4036 (var=0.0036) (Δ vs real -0.0142)
**Other theories' values on this metric (for reference):**
- pi_6: 0.7386 (var=0.0031)
- pi_5: 0.6297 (var=0.0048)
- pi_1: 0.6053 (var=0.0037)
- pi_2: 0.5439 (var=0.0033)
- pi_3: 0.7967 (var=0.0021)
- pi_4: 0.6303 (var=0.0046)
- pi_7: 0.3878 (var=0.0029)
- pi_8: 0.4492 (var=0.0019)
- pi_9: 0.4247 (var=0.0030)

### Experiment 11
**Design**
  A=[1, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 1]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    # Design-fixed printed validities for this experiment
    v = np.array([0.55, 0.90, 0.75, 0.95, 0.68, 0.85], dtype=float)
    n = v.size
    j = np.arange(n, dtype=float)

    # MDSC-NV salience: near-argmax doubt rule (delta=15, lam=0.115*15)
    logu = 15.0 * np.log(np.clip(1.0 - v, 1e-12, None)) + 1.725 * j
    # SSI-DS+ compensatory summation weights (gamma=0.9, lam=1.5)
    w_ssi = np.power(np.clip(1.0 - v, 1e-12, None), 0.9) * np.exp(1.5 * j)

    hits = 0
    tot = 0
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()

    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        try:
            a = np.asarray(a_raw, dtype=float).ravel()
            b = np.asarray(b_raw, dtype=float).ravel()
        except Exception:
            continue
        if a.size != n or b.size != n:
            continue
        d = a - b
        disc = np.nonzero(d != 0.0)[0]
        if disc.size == 0:
            continue

        # MDSC-NV predicted winner: verdict of the single most doubtful
        # discriminating expert
        k = disc[int(np.argmax(logu[disc]))]
        pred_mdsc = 0 if d[k] > 0 else 1

        # SSI-DS+ predicted winner: sign of the weighted sum over all
        # discriminating rows (coalitions add, late rows dominate)
        E = float(np.sum(np.sign(d[disc]) * w_ssi[disc]))
        if E == 0.0:
            continue
        pred_ssi = 0 if E > 0 else 1

        if pred_ssi == pred_mdsc:
            continue  # non-diagnostic trial type

        tot += 1
        try:
            resp = int(r)
        except Exception:
            continue
        if resp == pred_mdsc:
            hits += 1

    if tot == 0:
        return 0.5
    return float(hits) / float(tot)
```

**Observed (real) value:** 0.5925 (var=0.0026)
**Candidate trajectory (this loop):**
  - iter 1: 0.6142 (var=0.0034) (Δ vs real +0.0217)
  - iter 2: 0.6271 (var=0.0037) (Δ vs real +0.0346)
  - iter 3: 0.6125 (var=0.0028) (Δ vs real +0.0200)
  - iter 4 (current): 0.5946 (var=0.0052) (Δ vs real +0.0021)
**Other theories' values on this metric (for reference):**
- pi_7: 0.6833 (var=0.0038)
- pi_6: 0.2583 (var=0.0034)
- pi_1: 0.1550 (var=0.0102)
- pi_2: 0.4058 (var=0.0050)
- pi_3: 0.1054 (var=0.0016)
- pi_4: 0.3046 (var=0.0036)
- pi_5: 0.2854 (var=0.0033)
- pi_8: 0.6367 (var=0.0043)
- pi_9: 0.6033 (var=0.0045)

### Experiment 12
**Design**
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0, 1]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 0, 1]
  A=[0, 1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1, 1]
  A=[1, 0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[1, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    V_DESIGN = np.array([0.51, 0.72, 0.80, 0.90, 0.95, 0.985], dtype=float)

    GAMMA = 0.90
    LAM = 1.50
    DELTA = 15.0
    LAM_M = 1.725

    hits = 0.0
    tot = 0.0

    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        try:
            a = np.asarray(list(a_raw), dtype=float).ravel()
            b = np.asarray(list(b_raw), dtype=float).ravel()
        except Exception:
            continue
        n = int(min(a.size, b.size))
        if n == 0:
            continue
        a = a[:n]
        b = b[:n]

        if n <= V_DESIGN.size:
            v = V_DESIGN[:n]
        else:
            v = np.concatenate([V_DESIGN, np.full(n - V_DESIGN.size, 0.985)])
        v = np.clip(v, 0.5, 0.9995)

        diff = a - b
        disc = np.nonzero(diff != 0.0)[0]
        if disc.size == 0:
            continue
        signs = np.sign(diff[disc])

        # --- SSI-DS+ side: max-normalised weighted SUM, late/high-validity dominated
        logw = GAMMA * np.log(np.clip(1.0 - v[disc], 1e-12, None)) + LAM * disc.astype(float)
        logw = logw - np.max(logw)
        w = np.exp(logw)
        E = float(np.sum(signs * w))
        if abs(E) < 1e-9:
            continue
        ssi_side = 1.0 if E > 0 else -1.0

        # --- MDSC-NV side: single most-doubtful DISCRIMINATING row decides
        sal = DELTA * np.log(np.clip(1.0 - v[disc], 1e-12, None)) + LAM_M * disc.astype(float)
        k = int(np.argmax(sal))
        mdsc_side = float(signs[k])
        if mdsc_side == 0.0:
            continue

        # keep only trials where the two accounts point to OPPOSITE options
        if mdsc_side == ssi_side:
            continue

        try:
            r = int(resp)
        except Exception:
            continue
        chose_a = (r == 0)
        ssi_favours_a = (ssi_side > 0)
        hits += 1.0 if (chose_a == ssi_favours_a) else 0.0
        tot += 1.0

    if tot == 0.0:
        return 0.5
    return float(hits / tot)
```

**Observed (real) value:** 0.2608 (var=0.0079)
**Candidate trajectory (this loop):**
  - iter 1: 0.2108 (var=0.0034) (Δ vs real -0.0500)
  - iter 2: 0.2054 (var=0.0027) (Δ vs real -0.0554)
  - iter 3: 0.2317 (var=0.0046) (Δ vs real -0.0292)
  - iter 4 (current): 0.2204 (var=0.0043) (Δ vs real -0.0404)
**Other theories' values on this metric (for reference):**
- pi_6: 0.7412 (var=0.0040)
- pi_7: 0.2592 (var=0.0042)
- pi_1: 0.8400 (var=0.0123)
- pi_2: 0.5587 (var=0.0027)
- pi_3: 0.8158 (var=0.0005)
- pi_4: 0.7417 (var=0.0031)
- pi_5: 0.7421 (var=0.0057)
- pi_8: 0.2729 (var=0.0043)
- pi_9: 0.2617 (var=0.0055)

### Experiment 13
**Design**
  A=[1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[1, 1, 0, 1, 1]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    hits = []
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = int(min(a.size, b.size))
        if n < 5:
            continue
        a = a[:n]
        b = b[:n]
        d = a - b
        disc = [int(i) for i in np.nonzero(d != 0.0)[0]]
        # require feature 0 (validity .68) to discriminate
        if 0 not in disc:
            continue
        # require the global doubt anchor, feature 3 (validity .52), to be SILENT
        if 3 in disc:
            continue
        fav_a = [i for i in disc if d[i] > 0.0]
        fav_b = [i for i in disc if d[i] < 0.0]
        if d[0] > 0.0:
            lone, other = fav_a, fav_b
            lone_is_a = True
        else:
            lone, other = fav_b, fav_a
            lone_is_a = False
        # exactly one dissenter (feature 0) against a coalition of >=2 more valid experts
        if len(lone) != 1 or len(other) < 2:
            continue
        try:
            resp = int(r)
        except Exception:
            continue
        chose_lone = (resp == 0) if lone_is_a else (resp == 1)
        hits.append(1.0 if chose_lone else 0.0)
    if len(hits) == 0:
        return float('nan')
    return float(np.mean(hits))
```

**Observed (real) value:** 0.5460 (var=0.0126)
**Candidate trajectory (this loop):**
  - iter 1: 0.5060 (var=0.0123) (Δ vs real -0.0400)
  - iter 2: 0.5030 (var=0.0094) (Δ vs real -0.0430)
  - iter 3: 0.5590 (var=0.0150) (Δ vs real +0.0130)
  - iter 4 (current): 0.5310 (var=0.0134) (Δ vs real -0.0150)
**Other theories' values on this metric (for reference):**
- pi_7: 0.6410 (var=0.0093)
- pi_8: 0.2680 (var=0.0082)
- pi_1: 0.1620 (var=0.0145)
- pi_2: 0.1700 (var=0.0205)
- pi_3: 0.0110 (var=0.0005)
- pi_4: 0.2580 (var=0.0097)
- pi_5: 0.2380 (var=0.0124)
- pi_6: 0.2580 (var=0.0074)
- pi_9: 0.4660 (var=0.0150)

### Experiment 14
**Design**
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 1, 1]
  A=[0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1, 0]
  A=[0, 0, 0, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 0, 0, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0, 1]
  A=[0, 1, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[1, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 1]
  A=[0, 1, 1, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    V = np.array([0.62, 0.78, 0.86, 0.93, 0.53, 0.97], dtype=float)
    ANCHOR = 4  # the .53 expert (below DTV's doubt trigger)
    hits = 0.0
    tot = 0.0
    for a_raw, b_raw, resp in zip(data['option_a_ratings'].tolist(),
                                  data['option_b_ratings'].tolist(),
                                  data['response'].tolist()):
        a = np.asarray(a_raw, dtype=float).ravel()
        b = np.asarray(b_raw, dtype=float).ravel()
        n = int(min(a.size, b.size))
        if n != 6:
            continue
        diff = a[:6] - b[:6]
        disc = np.nonzero(diff != 0.0)[0]
        m = int(disc.size)
        if m < 3:
            continue
        # the globally most doubtful expert (.53) must be SILENT
        if ANCHOR in set(disc.tolist()):
            continue
        d = diff[disc]
        na = int(np.sum(d > 0.0))
        nb = m - na
        if min(na, nb) != 1 or max(na, nb) < 2:
            continue
        if na == 1:
            lone = int(disc[d > 0.0][0])
            fav = 0  # option A favoured by the lone dissenter
        else:
            lone = int(disc[d < 0.0][0])
            fav = 1
        # the lone dissenter must be the LEAST valid discriminating expert
        if abs(V[lone] - float(np.min(V[disc]))) > 1e-9:
            continue
        tot += 1.0
        if int(resp) == fav:
            hits += 1.0
    if tot < 1.0:
        return 0.5
    return float(hits / tot)
```

**Observed (real) value:** 0.4675 (var=0.0144)
**Candidate trajectory (this loop):**
  - iter 1: 0.5725 (var=0.0138) (Δ vs real +0.1050)
  - iter 2: 0.5763 (var=0.0118) (Δ vs real +0.1088)
  - iter 3: 0.5813 (var=0.0138) (Δ vs real +0.1138)
  - iter 4 (current): 0.5513 (var=0.0220) (Δ vs real +0.0837)
**Other theories' values on this metric (for reference):**
- pi_8: 0.2313 (var=0.0086)
- pi_7: 0.6388 (var=0.0149)
- pi_1: 0.1550 (var=0.0197)
- pi_2: 0.1313 (var=0.0107)
- pi_3: 0.0238 (var=0.0015)
- pi_4: 0.2812 (var=0.0111)
- pi_5: 0.2863 (var=0.0094)
- pi_6: 0.2313 (var=0.0138)
- pi_9: 0.5387 (var=0.0195)

### Experiment 15
**Design**
  A=[1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0, 1]
  A=[1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 0]  B=[0, 1, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 1, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[0, 0, 0, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  A=[1, 1, 1, 0, 1, 1]  B=[0, 0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    hits = 0.0
    tot = 0.0
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()
    ANCHOR = 3  # column with the lowest printed validity (0.55)
    for a_raw, b_raw, r in zip(a_col, b_col, r_col):
        a = [int(x) for x in list(a_raw)]
        b = [int(x) for x in list(b_raw)]
        k = min(len(a), len(b))
        if k <= ANCHOR:
            continue
        if a[ANCHOR] == b[ANCHOR]:
            continue  # anchor silent -> not a coalition test
        n_a = 0
        n_b = 0
        for i in range(k):
            if a[i] > b[i]:
                n_a += 1
            elif b[i] > a[i]:
                n_b += 1
        anchor_favors_a = a[ANCHOR] > b[ANCHOR]
        opposing = n_b if anchor_favors_a else n_a
        if opposing < 1:
            continue  # pure dominance for the anchor side -> no coalition
        try:
            resp = int(r)
        except Exception:
            continue
        chose_a = (resp == 0)
        if chose_a == anchor_favors_a:
            hits += 1.0
        tot += 1.0
    if tot == 0.0:
        return float('nan')
    return float(hits / tot)
```

**Observed (real) value:** 0.8620 (var=0.0155)
**Candidate trajectory (this loop):**
  - iter 1: 0.7773 (var=0.0035) (Δ vs real -0.0847)
  - iter 2: 0.8577 (var=0.0023) (Δ vs real -0.0043)
  - iter 3: 0.7853 (var=0.0034) (Δ vs real -0.0767)
  - iter 4 (current): 0.7977 (var=0.0046) (Δ vs real -0.0643)
**Other theories' values on this metric (for reference):**
- pi_7: 0.7610 (var=0.0035)
- pi_9: 0.4777 (var=0.0043)
- pi_1: 0.3003 (var=0.0058)
- pi_2: 0.2920 (var=0.0039)
- pi_3: 0.1703 (var=0.0013)
- pi_4: 0.2870 (var=0.0043)
- pi_5: 0.2827 (var=0.0026)
- pi_6: 0.2450 (var=0.0025)
- pi_8: 0.7143 (var=0.0022)

### Experiment 16
**Design**
  A=[1, 1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0, 1]
  A=[0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1]
  A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  A=[0, 1, 0, 1, 0, 0]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 0]  B=[0, 0, 1, 0, 1, 0]
  A=[0, 0, 1, 0, 1, 0]  B=[1, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 0]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0, 0]  B=[1, 0, 1, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    hits = 0.0
    n = 0.0
    for a_raw, b_raw, resp in zip(data['option_a_ratings'],
                                  data['option_b_ratings'],
                                  data['response']):
        a = np.asarray(list(a_raw), dtype=float).ravel()
        b = np.asarray(list(b_raw), dtype=float).ravel()
        k = int(min(a.size, b.size))
        if k < 6:
            continue
        # the near-chance anchor expert (column 0, validity .52) must discriminate
        if a[0] == b[0]:
            continue
        anchor_side = 0 if a[0] > b[0] else 1  # 0 == option A, 1 == option B
        # count late, mid-doubt rows (columns 3,4,5) that oppose the anchor
        opp = 0
        for j in (3, 4, 5):
            if a[j] == b[j]:
                continue
            fav = 0 if a[j] > b[j] else 1
            if fav != anchor_side:
                opp += 1
        if opp < 2:
            continue
        n += 1.0
        try:
            r = int(resp)
        except Exception:
            continue
        if r == anchor_side:
            hits += 1.0

    if n == 0.0:
        return 0.5
    return float(hits / n)

```

**Observed (real) value:** 0.8260 (var=0.0180)
**Candidate trajectory (this loop):**
  - iter 1: 0.8040 (var=0.0053) (Δ vs real -0.0220)
  - iter 2: 0.8770 (var=0.0026) (Δ vs real +0.0510)
  - iter 3: 0.7890 (var=0.0043) (Δ vs real -0.0370)
  - iter 4 (current): 0.8180 (var=0.0041) (Δ vs real -0.0080)
**Other theories' values on this metric (for reference):**
- pi_9: 0.4215 (var=0.0085)
- pi_7: 0.7245 (var=0.0054)
- pi_1: 0.4340 (var=0.0037)
- pi_2: 0.2900 (var=0.0065)
- pi_3: 0.1700 (var=0.0026)
- pi_4: 0.3045 (var=0.0052)
- pi_5: 0.3325 (var=0.0058)
- pi_6: 0.3115 (var=0.0039)
- pi_8: 0.6770 (var=0.0061)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1
Verdict: regenerate
Interpretation: The candidate (DTG-2R) is faithful to the arbiter's prescription: an absolute near-chance validity gate (sigmoid((theta-v)/tau), theta~.59, tau~.014) commanding a load-insensitive one-reason branch, a divisively-normalised anti-validity compensatory fallback when no sub-threshold expert speaks, the label-degeneracy tally collapse, and separate load/conflict lapses. No mechanism drift, and the accept gate confirms it is the best-fitting instantiation so far (loss .0707).

Per-experiment fit is good on 11 of 16 cells (|err| <= .05): Exp1 .253/.282, Exp4 .138/.150, Exp6 .518/.507, Exp7 .452/.438, Exp9 .799/.819, Exp10 .419/.418 (best of all theories), Exp11 .614/.593, Exp12 .211/.261, Exp13 .506/.546, Exp16 .804/.826, Exp8 .810/.770. The degeneracy gate is clearly doing real work (Exp6 at chance, Exp8/9 at ~.80 where every non-DTG theory is either at ceiling or floor), and the compensatory branch is what finally rescues Exp10 and Exp12, which pi_1..pi_6 miss by .3-.5.

The residual errors are systematic and point in one consistent direction rather than being random noise:
(a) The COMPENSATORY branch is a touch too decisive / too anti-validity. Exp2 overshoots badly (+.116: .290 vs .174 — the tally-minus-TTB gap is inflated), Exp3 undershoots (-.069, too anti-validity on tally-tie trials), Exp12 undershoots (-.050, same direction), Exp1 slightly under (-.029), and dominance displays are too confident (Exp5 .801 vs .743; Exp8 .810 vs .770). All six are consistent with beta too high / sigma too small / eps0 too small in the fallback.
(b) The recency+anti-validity tilt inside the fallback over-favours a lone EARLY least-valid dissenter. Exp14 (+.105: .573 vs .468) and Exp13 (-.040: .506 vs .546) both isolate a column-0 lone dissenter against a 2-3 coalition, with human values essentially at chance in both (.47, .55). The model separates them too much: it follows the v=.62 col-0 dissenter at .57 but the v=.68 col-0 dissenter at .51, i.e. the (1-v)^gamma gradient is steeper than humans show in this range.
(c) The TRIGGER branch is slightly too lapse-y: Exp15 .777 vs .862 (-.085) and Exp16 .804 vs .826 (-.022). The arbiter's target for these cells was ~.85-.86 flat; eps_t in [.20,.36] is eating ~5-8 points of anchor-following.
Rationale: Keep DTG-2R exactly as specified by the arbiter — absolute doubt gate + one-reason branch + divisively-normalised anti-validity fallback + degeneracy tally + split load/conflict lapse. Do NOT change the mechanism family; only re-tune three knobs, each of which pushes several cells in the same helpful direction. (1) Soften the compensatory branch: raise sigma to roughly [.60,.95] and/or lower beta to roughly [1.4,2.1], with eps0 nudged up to ~[.17,.30]. This pulls Exp2 down from .29 toward .17-.22, lifts Exp3 (.22->.26) and Exp12 (.21->.25), lifts Exp1 slightly toward .28, and brings the dominance cells Exp5 (.80->.76) and Exp8 (.81->.78) down to their human values. Watch Exp4 (.138) and Exp10 (.419) — both should stay put or improve slightly, so this is a low-risk global move. (2) Flatten the anti-validity gradient in the fallback only: lower gamma to ~[1.1,1.7] and/or trim lam to ~[.00,.12]. Exp13 and Exp14 both isolate a column-0 lone dissenter and both should land near .48-.55; the current gamma=1.9 makes the v=.62 dissenter far too compelling (Exp14 .573 vs .468) while barely helping the v=.68 case (Exp13 .506 vs .546). A shallower exponent compresses the pair toward the .47-.55 band the data demand. Make sure this does NOT change the trigger branch — Exps 15/16 must not be touched by gamma. (3) Make the one-reason branch slightly stiffer: drop eps_t to ~[.12,.24] and keep rho_t high (>=.95). Target Exp15 ~.84-.86 and Exp16 ~.82-.84, exactly the flat-across-coalition-size following the arbiter asked for; this is the single largest remaining residual (-.085 on Exp15). Expect a small cost on Exp1 (first-cue adherence may fall .01-.02 below .282) — that cost is smaller than the Exp15/16 gain, but if it bites, compensate by nudging theta down to the bottom of its range (~.572) so the v=.60 column in Exp2 and the v=.62 column in Exp14 leak less trigger mass into the one-reason branch. Do not widen theta above .605 or shrink tau below .008: the current gate cleanly separates .52-.55 anchors from .62-.68 dissenters and that separation is the theory's main empirical win.

**Outcome of your advice:** iter 1 candidate loss=0.0707 -> iter 2 candidate loss=0.0760 -> the gate marked it REJECTED.

### Iteration 2
Verdict: regenerate
Interpretation: The candidate is a verbatim re-emission of the accepted base's `predict`/`policy` with only parameter-range edits, so it remains fully faithful to the arbiter's DTG-2R family (absolute near-chance gate -> load-free one-reason limb; divisively-normalised anti-validity compensatory fallback; degeneracy tally collapse; split load/conflict lapse). No mechanism drift. But the gate rejected it (.0760 vs .0707), and the per-cell trajectory cleanly separates which of my three previous knobs helped and which hurt.

WHAT HELPED (knob 3, stiffening the one-reason limb: eps_t .20-.36 -> .13-.23, rho_t >= .95): Exp15 .777 -> .858 (real .862; residual -.085 -> -.004, the single biggest win in the whole table) and Exp16 .804 -> .877 (real .826; went from -.022 under to +.051 over). Exp9 .799 -> .845 (real .819) also flipped from under to over. So the direction was right but the step was too large: the trigger limb is now slightly TOO decisive, and it dragged Exp8 from .810 to .860 (real .770; residual +.040 -> +.090), which is now the second-largest error in the set.

WHAT HURT (knobs 1 and 2). The compensatory softening (sigma up, beta down, eps0 up) did NOT move the tally-tie cells toward chance as I predicted; they moved further from the data: Exp3 .217 -> .158 (real .286), Exp1 .253 -> .234 (real .282), Exp4 .138 -> .105 (real .150), Exp12 .211 -> .205 (real .261). Only Exp5 (.801 -> .784, real .743) and Exp2 (.290 -> .274, real .174) improved, and Exp2 is still +.100 off. The gamma/lam flattening was simply inert where it was supposed to act: Exp14 .573 -> .576 (real .468) and Exp13 .506 -> .503 (real .546) are unchanged. So the (1-v)^gamma exponent is not the lever controlling the lone-early-dissenter cells; the sign/magnitude of the lapse on conflicting displays is.

DIAGNOSIS OF THE REMAINING RESIDUAL STRUCTURE: nearly every remaining compensatory-branch error is an error of EXCESS DECISIVENESS ON CONFLICT DISPLAYS, in whichever direction the fallback happens to point. Exp1 (.234 vs .282), Exp3 (.158 vs .286), Exp4 (.105 vs .150), Exp12 (.205 vs .261) are all too far BELOW .5; Exp14 (.576 vs .468) and Exp2 (.274 vs .174) are too far ABOVE. Six of seven compensatory conflict cells would improve simultaneously if choices on displays with a genuine minority coalition were pulled harder toward chance. That is exactly what rho_c (the per-minority-row lapse) is for, and it is currently far too mild (.74-.90). It is also the one knob I have not yet pushed.
Rationale: Stay inside DTG-2R exactly as written (no code changes needed again; ranges only). Three concrete instructions, built on the UNCHANGED accepted base (loss .0707), not on this rejected candidate:

(1) KEEP a reduced-but-smaller eps_t. My iter-1 push here was the one component that clearly worked (Exp15 -.085 -> -.004) but overshot Exp16/Exp8/Exp9. Do not oscillate back to .20-.36 and do not stay at .13-.23; take the midpoint: eps_t in [.17, .27], rho_t in [.95, .99]. Target Exp15 ~.84, Exp16 ~.84, Exp8 ~.80-.82, Exp9 ~.82.

(2) REVERT the compensatory softening, which the gate punished. Put sigma back to ~[.42, .72], beta back to ~[1.8, 2.6], gamma back to ~[1.5, 2.3], lam back to ~[.00, .20], eps0 back to ~[.14, .26]. These edits moved Exp1/3/4/12 the wrong way and left Exp13/14 untouched; there is no evidence they buy anything.

(3) THE NEW KNOB — sharpen the CONFLICT lapse, not the display lapse. Lower rho_c from [.74, .90] to roughly [.60, .78]. rho_c is raised to the power k_conf = size of the minority coalition, so it acts only on displays where both options win rows; it leaves dominance (Exp5, Exp7 6-0 cells) and the trigger limb (Exps 15/16, and Exps 8/9 whose sub-.59 rows fire the gate) untouched. Predicted joint effect, all in the right direction at once: Exp14 .576 -> ~.51 (real .468), Exp2 .274 -> ~.20 (real .174), Exp3 .158 -> ~.22 (real .286), Exp1 .234 -> ~.27 (real .282), Exp4 .105 -> ~.14 (real .150), Exp12 .205 -> ~.24 (real .261). The only cost is Exp13 (.503, real .546) drifting a hair closer to .5, which is <=.01 of loss against six cells each gaining .03-.07.

(4) Leave theta/tau at the accepted base ([.572,.605] / [.008,.020]). The tightening in this iteration produced no isolable gain, and the .52-.55 vs .62-.68 separation is the theory's main empirical win — do not risk it. Also leave the degeneracy gate and tally_kappa alone: Exp10 (.419 vs .418) and Exp6 (~.50) are already the best in the corpus and should not move.

In short: keep one third of the last edit (a milder eps_t cut), discard the other two thirds, and spend the freed budget on a conflict-specific lapse that pushes six compensatory cells toward chance simultaneously.

**Outcome of your advice:** iter 2 candidate loss=0.0760 -> iter 3 candidate loss=0.0692 -> the gate marked it ACCEPTED.

### Iteration 3 (most recent)
Verdict: regenerate
Interpretation: The candidate is faithful to the arbiter's DTG-2R family: absolute near-chance validity gate (sigmoid((theta-v)/tau)) -> one-reason limb; divisively-normalised anti-validity compensatory fallback when no near-chance expert speaks; degeneracy tally collapse; separate load/conflict lapses. The only change from the accepted base is the opposition-scaled lapse on the trigger limb (rho_tc^(k_conf-1)), and it did what the proposer argued: Exp3 .158/.217 -> .239 (real .286), Exp1 -> .248 (real .282), Exp12 -> .232 (real .261), Exp4 .153 (real .150), Exp13 .559 (real .546), with Exp10 (.416 vs .418), Exp7 (.433 vs .438), Exp6 (.485 vs .507) essentially exact. The gate accepted it (.0692, new best). My iter-2 advice mix (keep a mild eps_t, revert the compensatory softening, add a conflict-specific lapse) was vindicated in spirit \u2014 the proposer implemented the conflict lapse on the trigger limb rather than on rho_c and it paid off.

The remaining loss is now extremely concentrated. Two cells carry ~70% of it:
(a) Exp14: .581 vs .468 (+.114). This cell is supposed to be pure compensatory (the .53 anchor is SILENT by construction; the lone dissenter is col0 at v=.62). But with theta up to .605 and tau up to .020, g(v=.62) = sigmoid((.605-.62)/.020) \u2248 .32 \u2014 a third of the trial mass is still being adjudicated by the one-reason limb, which follows the lone dissenter at ~.85. Decomposing the observed .581 = .32*.85 + .68*x gives x \u2248 .454 for the compensatory branch alone \u2014 i.e. the fallback is ALREADY at the human value (.468) and the entire residual is trigger leakage at v=.62. This also explains why my iter-1 gamma/lam flattening was inert on Exp13/14: the trigger mass, not the weight profile, was driving that cell.
(b) Exp2: .362 vs .174 (+.188), and it got WORSE this iteration (.290 -> .362) because rho_tc pulls the 2-vs-2 tally-tie displays (where the .55 column agrees with the TTB winner) down from ~.81 to ~.75, inflating p_tally - p_ttb_tie. Note that the two DTG siblings (pi_7 .230, pi_9 .326) also overshoot here, and the only theories that hit Exp2 (pi_3, pi_4, pi_6) fail catastrophically on Exps 12-16. This looks like a largely structural cost of the family and should NOT be chased hard.
Smaller, coherent residuals: Exp15 .785 vs .862 (-.077) and Exp16 .789 vs .826 (-.037) \u2014 the lone-anchor cells are still slightly too lapse-y in long (m=5,6) displays; Exp5 .799 vs .743 (+.056) and Exp8 .804 vs .770 (+.035) are mildly over-confident; Exp9 -.028, Exp11 +.020 are fine.
Rationale: The candidate was ACCEPTED and should be the base, but there is one quantified, low-risk in-family improvement left that is worth ~.013 of loss on its own, so run one more iteration on THIS candidate (no code changes needed beyond parameter ranges; keep rho_tc exactly as implemented \u2014 that edit was the one the gate rewarded).

(1) PRIMARY: sharpen and slightly lower the doubt gate so that v=.62 no longer leaks into the one-reason limb. Set theta to ~[.555, .585] and tau to ~[.006, .013]. At theta=.575/tau=.010 this gives g(.52)\u22481, g(.545)\u2248.95, g(.55)\u2248.92, g(.58)\u2248.31, g(.60)\u2248.08, g(.62)\u2248.01. Predicted: Exp14 .581 -> ~.46-.48 (real .468) \u2014 the second-largest residual essentially eliminated \u2014 with Exp13 untouched (its lone dissenter is v=.68, already g\u22480) and Exp15/16 anchors (.55/.52) still firmly inside the gate. I flagged in iters 1-2 that theta/tau should be left alone; I am reversing that specifically because the .581 = .32*.85 + .68*.454 decomposition shows the compensatory limb is already correct on Exp14 and only the leak is wrong. Do not push theta below ~.55 or tau below ~.006: the .545/.55 anchors in Exps 8/9/11/15 must keep g>=.9.

(2) SECONDARY, small: recover the lone-anchor cells lost to row-scan decay by narrowing rho_t to [.985, 1.00] (leave eps_t at [.20,.36] \u2014 a blanket eps_t cut was rejected by the gate in iter 2 and I will not repeat it). rho_t only acts through rho_t^(m-2), so this lifts the long-display cells Exp15 (~+.02-.03) and Exp16 (~+.02) while leaving the m=2-3 displays of Exps 1/3/4 \u2014 which are currently well placed \u2014 unmoved.

(3) DO NOT chase Exp2. Its residual is a structural property of any theory that both (i) follows a .55 expert that sides with a 5-row majority and (ii) discounts 2-vs-2 ties via rho_tc; every DTG sibling overshoots it, and the only fixes would either undo the accepted rho_tc edit (which would re-break Exp3 at .286) or weaken the near-chance gate that Exps 15/16 depend on. Leave it as a documented cost of the family.

(4) Freeze everything else: rho_tc [.62,.86], eps0/rho_m/rho_c at current values (Exp10 .416 vs .418, Exp7 .433 vs .438, Exp13 .559 vs .546 are at or near best-in-corpus \u2014 my iter-2 rho_c push was never tested and is now unnecessary given the theta fix does the Exp14 work), tally_kappa and the degeneracy gate untouched (Exp6 at chance is correct). Accept small collateral drift of <=.02 on Exp8/Exp9/Exp11 from the sharper gate; if Exp8 (+.035) worsens, take theta at the top of the new range (.585) rather than re-tuning eps_t.

**Outcome of your advice:** iter 3 candidate loss=0.0692 -> CURRENT candidate loss=0.0549 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "The candidate is fully faithful to the arbiter's DTG-2R prescription \u2014 absolute near-chance validity gate g=sigmoid((theta-v)/tau) commanding a one-reason limb, a divisively-normalised anti-validity + recency compensatory fallback when no sub-threshold expert speaks, the .002 label-degeneracy tally collapse, and separate load/opposition lapses. No mechanism drift, and the gate ACCEPTED it as the new running best (.0549, a .014 improvement over the previous base). My iter-3 advice (sharpen/lower theta-tau, narrow rho_t) is vindicated: Exp14 .581 -> .551, Exp15 .785 -> .798, Exp16 .789 -> .818 (real .826, now essentially exact), Exp11 .613 -> .595 (real .593), Exp3 .239 -> .273 (real .286), Exp4 .153 -> .173 (real .150), and the corpus anchors stayed put (Exp10 .404/.418, Exp7 .450/.438, Exp6 .498/.507, Exp13 .531/.546, Exp9 .797/.819).\n\n13 of 16 cells are now within .05 of the human value, and the model remains the only family that simultaneously handles the one-reason cells (Exps 15/16 at .80-.82 where pi_1..pi_6 are all at .17-.43) and the compensatory cells (Exps 10/12 at .40/.22 where pi_1..pi_6 are at .60-.84).\n\nThe residual is now concentrated in three places, and the diagnosis has CHANGED in one important respect:\n(a) Exp2 (+.147) is still the largest error and remains, as I argued, a structural cost of the family: p_tally (5-vs-1 displays where the .55 anchor sides with the tally) is kept high by the trigger limb while rho_tc^(k_conf-1) discounts the 2-vs-2 tie displays that set p_ttb_tie, inflating the difference. Both DTG siblings (pi_7 .230, pi_9 .326) overshoot too, and the only theories that hit it (pi_3, pi_4, pi_6) are catastrophic on Exps 12-16.\n(b) Exp14 (+.084) is now the informative one. With theta~.573 / tau~.0095, g(.62) is ~.007, so the trigger leak I blamed in iter 3 is CLOSED \u2014 and the cell still sits at .551. My earlier decomposition (compensatory limb already at .454) was therefore wrong: the compensatory limb itself follows the lone col-0 v=.62 dissenter at ~.55 against a 2-3 row coalition. That is over-decisive in the pro-dissenter direction. Note that Exp13 (lone col-0 dissenter at v=.68, real .546, model .531) is fitted well, so humans show essentially NO anti-validity gradient between v=.62 and v=.68 in these lone-dissenter displays, whereas (1-v)^gamma with gamma~1.9 produces one.\n(c) Exp15 (-.064) and the mildly over-confident Exp5 (+.042) / Exp8 (+.025) round out the list; Exp1 (-.039) and Exp12 (-.040) are still slightly too far below chance, i.e. the compensatory limb is a touch too extreme in whichever direction it points.",
  "verdict": "regenerate",
  "rationale": "The candidate is the new best base and should NOT be reverted \u2014 the next iteration must build on it. But there is one remaining, now properly diagnosed, low-risk in-family knob worth ~.01-.015 of loss, so run one more round on THIS candidate. Code and mechanism stay exactly as written (no family change, no structural edits); ranges only, and ONE knob at a time this round, because the gate punished my iter-2 multi-knob compensatory softening.\n\n(1) PRIMARY, single knob: raise sigma (the divisive-normalisation constant in the COMPENSATORY limb only) from [.40,.72] to ~[.55,.88], leaving beta, gamma, lam, eps0, rho_m, rho_c untouched. This scales the whole compensatory branch toward chance without touching the trigger limb (so Exps 15/16 are unaffected). Six cells move the right way at once: Exp14 .551 -> ~.51-.53 (real .468), Exp1 .244 -> ~.26 (real .282), Exp12 .220 -> ~.235 (real .261), Exp3 .273 -> ~.285 (real .286), Exp5 .785 -> ~.77 (real .743), Exp8 .795 -> ~.785 (real .770). The only costs are small and bounded: Exp4 .173 -> ~.185 (real .150, -.01) and Exp13 .531 -> ~.522 (real .546, -.01). Keep the step modest \u2014 do not push sigma past ~.9, or Exp4/Exp13/Exp10 start to flatten out.\n\n(2) DO NOT repeat the two things the gate has already punished. Do not lower beta and do not raise eps0 (iter-2 REJECTED); do not cut eps_t across the board (iter-2 REJECTED). Leave eps_t [.20,.36] and rho_t [.985,1.00] exactly as they are \u2014 Exp16 is now nearly exact and Exp8 is already .025 over, so a blanket trigger-limb stiffening would trade Exp15 against Exp8/Exp16 for no net gain.\n\n(3) gamma is the ONLY alternative lever on Exp14, and it is now genuinely operative (unlike in iter 1, when trigger leakage masked it) \u2014 but it is second choice, not first. If sigma alone leaves Exp14 above ~.52, a small gamma trim to ~[1.3,1.9] is permissible, but note the human data show NO anti-validity gradient between v=.62 (Exp14, .468) and v=.68 (Exp13, .546), so any gamma cut that improves Exp14 by .05 will cost Exp13 ~.04. Only take it if the net is clearly positive.\n\n(4) FREEZE the rest: theta [.558,.588] and tau [.006,.013] (the edit the gate just rewarded \u2014 the .52/.545/.55 anchors must keep g>=.9 and the .62/.68 dissenters g<=.02), rho_tc [.62,.86], tally_kappa and the .002 degeneracy gate (Exp10 .404/.418, Exp6 .498/.507, Exp7 .450/.438 are best-in-corpus).\n\n(5) Continue to NOT chase Exp2. Any fix there would require either undoing rho_tc (re-breaking Exp3/Exp1) or weakening the near-chance gate that Exps 15/16 depend on; it is a documented family-level cost and the arbiter accepted it when prescribing DTG-2R."
}
```

## Usage

```json
{
  "input_tokens": 45662,
  "output_tokens": 5825
}
```
