# feedback_iter_01

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
- THEORY 1 = `pi_10`
- THEORY 2 = `pi_11`
- The recommendation below acts on THEORY 2 (= `pi_11`).

Replace pi_11 with a NUMEROSITY-DOMINANT, NEAR-VALIDITY-FLAT tally theory (call it 'Democratic Reason Counting with Rarity-Scaled Confidence and a Discriminating-Edge Tie-Break'), explicitly built to hit the 0.84-0.85 count-following rates that both current theories miss:

1. CORE DRIVE = COUNT, not weighted vote. drive = [lam*(k_A - k_B) + small weighted-vote correction] / (k2 + n_speak)^~1. The validity map must be nearly FLAT: w_j = 1 + gamma*((v_j-0.5)/0.5)^rho with gamma ~0.2-0.45 (so a .95 oracle is worth at most ~1.3-1.5 cheap cues, NEVER 3-10x). This is the decisive break from both incumbents: pi_11 had oracle/cheap ~9-11, pi_10 ~3; the data demand ~1.2-1.5. Two cheap cues must beat one oracle comfortably (P ~0.8), three cues must beat oracle+helper (P ~0.85), and a 1-vs-3 minority must land near 0.20-0.30 (Exp 16).

2. MARGIN-GRADED, NOT CEILING-CAPPED ACCURACY. No hard lapse cap: P should run roughly 0.65-0.72 at |dk|=1, 0.82-0.88 at |dk|=2-3, and ~0.90-0.95 on unanimous landslides, with a difficulty-scaled lapse. Calibrate so the Exp1/Exp2 coalition metrics land near 0.84 and Exp 14's mixed cells near 0.66.

3. NO TAKE-THE-BEST COMMITMENT STAGE. Drop the authority-coincidence lead entirely (q_max = 0). Exp 15 shows first-row following at 0.353 and Exp 3 shows only a +0.09 TTB-conflict advantage; any strong first-cue stop over-predicts these. The small positive Exp 3 effect should instead fall out of the edge tie-break (the first discriminating row is often also a screen extreme).

4. EDGE TIE-BREAK ON DISCRIMINATING ROWS, gated by near-deadlock. When |dk| = 0 (and weakly at |dk| = 1) let the option endorsed by the FIRST and LAST DISCRIMINATING rows win with P ~0.78-0.85 (Exp 7 = 0.820, Exp 12 = 0.850), but keep it a tie-break only: it must be ~0 by |dk| >= 2 so Exp 11 (pooled 0.49) and Exp 13 (index 0.513) are not over-shot. Note Exp 13's c4 was near chance, so allow a moderate rather than maximal tie-break amplitude and let parameter jitter cover the spread.

5. DIAGNOSTICITY-SCALED SILENCE/RARITY CONFIDENCE, floored and never sign-reversing. Keep the one mechanism both incumbents got right: a lone speaking cue amid a mute panel is discounted MORE the more diagnostic it is, reproducing Exp 10's -0.20 (strong-minus-weak single-cue following) and Exp 17's 0.855. Silence COMPOSITION (mutual 1s vs mutual 0s) must be nearly inert: Exp 18 observed +0.097 and Exp 19 observed -0.001, so the dense/sparse contrast in Exp 19 should be ~0, not the +0.12 both incumbents predicted.

6. No learning, no subpopulations; unimodal parameter jitter. Parameter ranges should be chosen so the simulated values for the already-run experiments land near: Exp1 ~0.84, Exp2 ~0.85, Exp3 ~+0.09, Exp4 ~0.78, Exp8 ~0.84, Exp10 ~-0.20, Exp12 ~0.85, Exp13 ~0.51, Exp14 ~0.66, Exp15 ~0.35, Exp16 ~0.30, Exp17 ~0.85, Exp19 ~0.00, Exp20 ~-0.03.


## CANDIDATE THEORY
**Democratic Reason Counting with Rarity-Scaled Confidence, a Discriminating-Edge Tie-Break, a Narrow Authority-Coincidence Stop, and a SPARSE-PANEL AUTHORITY OFFSET (DRC-RE/A+SO).**

Unchanged core (the part that finally made the cheap-coalition experiments work): people *count reasons* with an almost flat validity credit, read the count out compressively, break (near-)deadlocks with the first and last *discriminating* rows, discount a lone speaker amid a nearly mute panel in proportion to his sensitivity, and commit early only when the top discriminating row is also the most diagnostic thing on the screen.

1. **NUMEROSITY-DOMINANT, NEAR-FLAT VALIDITY MAP.** w_j = 1 + gamma*((v_j-0.5)/0.5)^rho with gamma ~0.45: a .95 oracle is worth ~1.3 cheap cues, never 3-10x. T = sum_j sgn_j w_j.

2. **MARGIN-GRADED COMPRESSIVE READ-OUT + DIFFICULTY-SCALED LAPSE.** D = lam*sign(T)*|T|^alpha, alpha ~0.6; eps = eps0 + eps1*exp(-|logit p|).

3. **EDGE TIE-BREAK ON THE FIRST/LAST DISCRIMINATING ROWS, GATED BY NEAR-DEADLOCK** (full force at dk = 0, ~8% at |dk| = 1, nil beyond).

4. **DIAGNOSTICITY-SCALED RARITY DISCOUNT** with a high mute exponent, concentrated on lone-speaker panels; silence composition is inert for *confidence*.

5. **NARROW AUTHORITY-COINCIDENCE STOP**, scaled by the same confidence.

6. **NEW — SPARSE-PANEL AUTHORITY OFFSET (the one added claim).** A one-reason count advantage is *not* read the same way in an informative panel as in a mostly-blank one. When (i) the count margin is one, (ii) the dissenting camp is a *single* expert, (iii) that expert is the most diagnostic speaker on the screen, and (iv) a substantial share of the panel consists of MUTUAL ABSENCES (rows where neither product was endorsed), the reader treats the display as too thin to license 'more reasons wins': the count drive is shrunk toward zero, leaving genuine indifference between 'two cheap reasons' and 'one authoritative reason'. The offset never reverses the count, it only cancels it, and it is inert in dense panels (mutual endorsements, or many speakers), where the coalition keeps its full numerosity advantage. Psychologically: mutual-absence rows say 'nobody saw anything good in either product', so the two or three reasons that do exist are weak scraps and the single best-qualified judge regains parity; mutual-endorsement rows say 'both products are broadly fine', which does not undermine the reasons that did discriminate.

No learning, no strategy subpopulations; individual differences are unimodal jitter in all parameters.

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
    # 2. communicated validities -> diagnosticity scale
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

    rho = _p('rho', 1.20, 0.30, 4.0)
    gamma = _p('gamma', 0.45, 0.0, 2.0)
    lam = _p('lam', 1.25, 0.10, 6.0)
    alpha = _p('alpha', 0.60, 0.20, 1.50)
    mu_e = _p('mu_e', 1.60, 0.0, 6.0)
    sigma_e = _p('sigma_e', 0.45, 0.15, 2.0)
    eta = _p('eta', 3.80, 0.0, 15.0)
    q_mute = _p('q_mute', 6.0, 1.0, 20.0)
    p_u = _p('p_u', 1.20, 0.3, 4.0)
    c_min = _p('c_min', 0.14, 0.0, 0.6)
    eps0 = _p('eps0', 0.10, 0.0, 0.5)
    eps1 = _p('eps1', 0.20, 0.0, 0.6)
    q_max = _p('q_max', 0.95, 0.0, 1.0)
    phi = _p('phi', 1.02, 0.5, 3.0)
    s_stop = _p('s_stop', 8.0, 0.5, 60.0)
    d_com = _p('d_com', 2.55, 0.0, 8.0)
    # --- sparse-panel authority offset ---
    theta_max = _p('theta_max', 0.93, 0.0, 1.0)
    sigma_a = _p('sigma_a', 0.47, 0.15, 2.0)
    sigma_k = _p('sigma_k', 0.55, 0.15, 2.0)
    f_half = _p('f_half', 0.27, 0.05, 0.9)
    s_f = _p('s_f', 18.0, 1.0, 60.0)
    d0_u = _p('d0_u', 0.05, 0.0, 0.5)
    d_scale = _p('d_scale', 0.26, 0.05, 2.0)

    def _sig(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))

    # ------------------------------------------------------------------
    # 4. near-flat weights: a reason COUNT with a small validity credit
    # ------------------------------------------------------------------
    u = np.power(vs, rho)
    w = 1.0 + gamma * u

    idx = np.flatnonzero(disc)
    sgn = np.sign(d[idx])
    T = float(np.sum(sgn * w[idx]))

    kA = int(np.count_nonzero(sgn > 0))
    kB = int(np.count_nonzero(sgn < 0))
    dk = float(kA - kB)

    # compressive, margin-graded core drive
    if T > 0:
        D_core = lam * float(np.power(abs(T), alpha))
    elif T < 0:
        D_core = -lam * float(np.power(abs(T), alpha))
    else:
        D_core = 0.0

    # ------------------------------------------------------------------
    # 4b. SPARSE-PANEL AUTHORITY OFFSET.  A one-reason count advantage is
    #     cancelled (never reversed) when the lone dissenter is the most
    #     diagnostic speaker AND the panel is largely MUTUAL ABSENCE.
    # ------------------------------------------------------------------
    both0 = (~disc) & (a < 0.5) & (b < 0.5)
    f_abs = float(np.count_nonzero(both0)) / float(n)
    theta = 0.0
    if dk != 0.0:
        maj_mask = (sgn > 0) if dk > 0 else (sgn < 0)
        min_mask = ~maj_mask
        k_min = int(np.count_nonzero(min_mask))
        u_sp = u[idx]
        u_maj_max = float(np.max(u_sp[maj_mask])) if np.any(maj_mask) else 0.0
        u_min_max = float(np.max(u_sp[min_mask])) if k_min > 0 else 0.0
        g_dk = float(np.exp(-min(((abs(dk) - 1.0) ** 2) / (2.0 * sigma_a * sigma_a), 60.0)))
        g_km = float(np.exp(-min(((float(k_min) - 1.0) ** 2) / (2.0 * sigma_k * sigma_k), 60.0)))
        h_abs = float(_sig(s_f * (f_abs - f_half)))
        h_u = float(np.clip((u_min_max - u_maj_max - d0_u) / d_scale, 0.0, 1.0))
        theta = theta_max * g_dk * g_km * h_abs * h_u
    theta = float(np.clip(theta, 0.0, 1.0))
    D_core = D_core * (1.0 - theta)

    # ------------------------------------------------------------------
    # 5. edge tie-break over the FIRST and LAST DISCRIMINATING rows,
    #    licensed only by a (near-)deadlock in the reason count.
    # ------------------------------------------------------------------
    s_first = float(np.sign(d[int(idx[0])]))
    s_last = float(np.sign(d[int(idx[-1])]))
    e = 0.5 * (s_first + s_last)
    gate = float(np.exp(-min((dk * dk) / (2.0 * sigma_e * sigma_e), 60.0)))
    D_edge = mu_e * gate * e

    # ------------------------------------------------------------------
    # 6. rarity / sensitivity confidence (composition-blind silence)
    # ------------------------------------------------------------------
    n_speak = int(idx.shape[0])
    f_mute = float(n - n_speak) / float(n)
    f_mute = float(np.clip(f_mute, 0.0, 1.0))
    ubar = float(np.mean(u[idx]))
    ubar = float(np.clip(ubar, 0.0, 1.0))
    c = 1.0 - eta * float(np.power(f_mute, q_mute)) * float(np.power(ubar, p_u))
    c = float(np.clip(c, c_min, 1.0))

    p_int = float(_sig(c * (D_core + D_edge)))

    # ------------------------------------------------------------------
    # 7. narrow authority-coincidence stop (scaled by the same confidence)
    # ------------------------------------------------------------------
    j1 = int(idx[0])
    if j1 >= n - 1:
        q_com = q_max
    else:
        v_rest = float(np.max(u[j1 + 1:]))
        v_rest = max(v_rest, 1e-9)
        ratio = float(u[j1]) / (phi * v_rest)
        q_com = q_max * float(_sig(s_stop * (ratio - 1.0)))
    q_com = float(np.clip(q_com, 0.0, 1.0))

    p_dec = float(_sig(c * d_com))
    p_com_a = p_dec if d[j1] > 0 else (1.0 - p_dec)

    p_a = q_com * p_com_a + (1.0 - q_com) * p_int
    p_a = float(np.clip(p_a, 1e-9, 1.0 - 1e-9))

    # ------------------------------------------------------------------
    # 8. difficulty-scaled lapse (no hard ceiling)
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
- rho: [1.00, 1.40]
- gamma: [0.35, 0.55]
- lam: [1.12, 1.40]
- alpha: [0.52, 0.68]
- mu_e: [1.35, 1.85]
- sigma_e: [0.38, 0.52]
- eta: [3.30, 4.30]
- q_mute: [5.0, 7.0]
- p_u: [1.05, 1.35]
- c_min: [0.10, 0.18]
- eps0: [0.07, 0.13]
- eps1: [0.15, 0.25]
- q_max: [0.92, 0.98]
- phi: [0.99, 1.05]
- s_stop: [6.0, 10.0]
- d_com: [2.35, 2.85]
- theta_max: [0.86, 0.98]
- sigma_a: [0.40, 0.55]
- sigma_k: [0.48, 0.62]
- f_half: [0.24, 0.31]
- s_f: [14.0, 24.0]
- d0_u: [0.02, 0.09]
- d_scale: [0.20, 0.32]
- validities: validities

`rationale`:
MINIMAL DIFF on the accepted base (loss 0.1356): one new gated term plus a small nudge to two stop constants. Everything else — near-flat weights, compressive count read-out, deadlock-gated edge tie-break, rarity confidence, difficulty lapse — is byte-identical.

**Edit 1 (the critic's priority #1, implemented with a sparsity gate instead of a pure validity gate).** The critic asked for a margin-dependent authority sensitivity that drives the 2-cheap-vs-1-oracle cell to chance (Exp3 0.273 vs 0.518; Exp4 0.243 vs 0.508 — the two largest residuals). I implemented the shrink `D_core *= (1-theta)` (cancelling, never sign-reversing, so the coalition never loses) gated by four factors: |dk|≈1, a *single* dissenter, that dissenter being the most diagnostic speaker, and — the crucial addition — a substantial fraction of MUTUAL-ABSENCE mute rows. I added the sparsity gate because I checked the target cells cell-by-cell and a purely validity-based offset would have destroyed cells the base finally gets right: Exp6's class-2 display (2 extremes vs the mid-list top cue), Exp11's c5 (2 early cues vs the later heavyweight), Exp19's 2-vs-1 cell and Exp20's cell 5 are ALL structurally identical 'lone-authority-vs-coalition' displays where humans follow the count at ~0.84. What separates them from Exp3/Exp4/Exp9 is not the validity gap (unknowable to me a priori, and set per experiment) but the panel composition: Exp6/11/19/20's mute rows are mutual ENDORSEMENTS (f_abs ≈ 0-0.17), while Exp3 (f_abs = 0.50), Exp4 (0.40) and Exp9's extremes cell (0.33) sit in near-blank panels — and those three are exactly the cells where the real data sit at 0.518, 0.508 and 0.490, i.e. dead chance. The sparsity gate therefore turns three large residuals into ~0.06-0.13 errors while leaving the coalition experiments (the part the arbiter insisted must keep working) mathematically untouched: I verified g_dk/g_km/h_abs all suppress theta to <0.05 in every Exp6/10/11/12/13/14/16/17/18/19/20 target cell. The only cost is Exp7's load contrast (predicted ~+0.2 vs real 0.053) — an experiment whose between-subject variance (0.049) is the largest in the set and whose absolute levels are unconstrained; trading ~0.04 there for ~0.12 recovered on Exp3+Exp4+Exp9 is clearly net positive.

**Edit 2 (critic's priority #2, applied conservatively).** d_com 2.20 → 2.55 (range [2.35,2.85]), s_stop 9 → 8 (range [6,10]), phi 1.03 → 1.02 (range [0.99,1.05]). This raises first-row commitment where the top discriminating row is also the most diagnostic cue — lifting Exp1's conflict-trial TTB following toward its agree-trial level (Exp1 −0.083 → ≈ −0.04) and Exp13 (0.226 → ≈ 0.26) — while the softened gate leaks <5% into Exp3/Exp4-type displays (where the first row is a cheap cue sitting above the oracle), so the Edit-1 gains are preserved. Exp6/19/20 are unaffected because their first discriminating rows are *not* the most diagnostic, so the stop stays off there.

**Deliberately not done (critic's priority #3).** I left the deadlock edge term exactly as is. Exp10 (0.627 vs 0.850) and Exp11's index (0.762 vs 0.513) pull the same knob in opposite directions; I hand-computed that cancelling the tally residual at dk = 0 lifts Exp10 by ~0.2 but lifts Exp11's c4 by a comparable amount, giving a net squared-error increase. Likewise I left eps0 alone: the undershoots (Exp6/19/20) and overshoots (Exp2/11/12/13) are nearly balanced, so a lapse change is a wash with added regression risk.

## LOSS TRAJECTORY (this propose-loop)
Aggregate loss across iterations of THIS propose-loop (lower = better, 0 = perfect, `+inf` = unscorable). The ACCEPTED / REJECTED tag is the loop's programmatic accept-gate decision: `loss < accepted_loss` -> ACCEPTED (becomes new base), else REJECTED (base unchanged). Use this together with the per-experiment values below to grade fit-quality AND your own past advice (see `## YOUR PRIOR CRITIQUES` below).

- iter 1: loss=0.1356 -> ACCEPTED
- iter 2 (current candidate you are grading): loss=0.1268 -> ACCEPTED
Running-best (last accepted) base: iter 2 at loss=0.1268.

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
  - iter 1: -0.0827 (var=0.0057) (Δ vs real -0.1760)
  - iter 2 (current): -0.0735 (var=0.0063) (Δ vs real -0.1669)
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
- pi_10: 0.0354 (var=0.0104)
- pi_11: -0.0065 (var=0.0091)

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
  - iter 1: 0.8495 (var=0.0040) (Δ vs real +0.0745)
  - iter 2 (current): 0.8545 (var=0.0042) (Δ vs real +0.0795)
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
- pi_10: 0.7435 (var=0.0039)
- pi_11: 0.7190 (var=0.0030)

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
  - iter 1: 0.2725 (var=0.0079) (Δ vs real -0.2450)
  - iter 2 (current): 0.4025 (var=0.0153) (Δ vs real -0.1150)
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
- pi_10: 0.4625 (var=0.0178)
- pi_11: 0.4375 (var=0.0194)

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
  - iter 1: 0.2425 (var=0.0046) (Δ vs real -0.2650)
  - iter 2 (current): 0.3887 (var=0.0051) (Δ vs real -0.1187)
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
- pi_10: 0.4263 (var=0.0073)
- pi_11: 0.3312 (var=0.0102)

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
  - iter 1: 0.7788 (var=0.0111) (Δ vs real -0.0412)
  - iter 2 (current): 0.7500 (var=0.0123) (Δ vs real -0.0700)
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
- pi_10: 0.7650 (var=0.0146)
- pi_11: 0.8250 (var=0.0075)

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
  - iter 1: 0.7738 (var=0.0049) (Δ vs real -0.0675)
  - iter 2 (current): 0.7656 (var=0.0048) (Δ vs real -0.0756)
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
- pi_10: 0.5962 (var=0.0062)
- pi_11: 0.5544 (var=0.0089)

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
  - iter 1: 0.0233 (var=0.0320) (Δ vs real -0.0300)
  - iter 2 (current): 0.2100 (var=0.0403) (Δ vs real +0.1567)
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
- pi_10: 0.0433 (var=0.0476)
- pi_11: 0.1083 (var=0.0348)

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
  - iter 1: -0.1713 (var=0.0189) (Δ vs real +0.0312)
  - iter 2 (current): -0.1663 (var=0.0223) (Δ vs real +0.0363)
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
- pi_10: -0.1950 (var=0.0223)
- pi_11: -0.1738 (var=0.0294)

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
  - iter 1: 0.4725 (var=0.0069) (Δ vs real -0.0175)
  - iter 2 (current): 0.4058 (var=0.0100) (Δ vs real -0.0842)
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
- pi_10: 0.4683 (var=0.0093)
- pi_11: 0.4375 (var=0.0077)

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
  - iter 1: 0.6267 (var=0.0231) (Δ vs real -0.2233)
  - iter 2 (current): 0.6467 (var=0.0188) (Δ vs real -0.2033)
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
- pi_10: 0.7800 (var=0.0149)
- pi_11: 0.7217 (var=0.0285)

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
  - iter 1: 0.7617 (var=0.0378) (Δ vs real +0.2483)
  - iter 2 (current): 0.7467 (var=0.0558) (Δ vs real +0.2333)
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
- pi_10: 0.8983 (var=0.0498)
- pi_11: 0.8483 (var=0.0491)

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
  - iter 1: 0.7492 (var=0.0067) (Δ vs real +0.0842)
  - iter 2 (current): 0.7475 (var=0.0099) (Δ vs real +0.0825)
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
- pi_10: 0.6758 (var=0.0097)
- pi_11: 0.6608 (var=0.0081)

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
  - iter 1: 0.2256 (var=0.0055) (Δ vs real -0.1278)
  - iter 2 (current): 0.2200 (var=0.0059) (Δ vs real -0.1333)
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
- pi_10: 0.3556 (var=0.0076)
- pi_11: 0.3639 (var=0.0059)

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
  - iter 1: 0.1617 (var=0.0051) (Δ vs real -0.1383)
  - iter 2 (current): 0.1550 (var=0.0064) (Δ vs real -0.1450)
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
- pi_10: 0.2258 (var=0.0051)
- pi_11: 0.2125 (var=0.0070)

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
  - iter 1: 0.8317 (var=0.0389) (Δ vs real -0.0233)
  - iter 2 (current): 0.8092 (var=0.0575) (Δ vs real -0.0458)
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
- pi_10: 0.8383 (var=0.0491)
- pi_11: 0.6658 (var=0.0538)

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
  - iter 1: 0.0283 (var=0.0496) (Δ vs real -0.0683)
  - iter 2 (current): 0.0317 (var=0.0564) (Δ vs real -0.0650)
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
- pi_10: 0.1017 (var=0.0792)
- pi_11: 0.0933 (var=0.0685)

### Experiment 17
**Design**
  A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[1, 1, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]
  A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 1, 0, 0, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 0, 0, 1, 0]
  A=[0, 1, 0, 0, 1, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def key(a, b):
        sa = ''.join([str(int(round(float(v)))) for v in list(a)])
        sb = ''.join([str(int(round(float(v)))) for v in list(b)])
        return sa + '|' + sb

    # group 'D' = dense / high-evidence-mass cells (C, E, H)
    # group 'S' = sparse / silent-panel or cheap-coalition cells (A, B, G1, F)
    # value = (group, favored_response_code)  where 0 = option A, 1 = option B
    table = {
        # cell C : 4-1 dense majority
        '101111|010010': ('D', 0),
        '010010|101111': ('D', 1),
        # cell E : edge-deadlock (screen extremes vs deep heavyweights)
        '110011|011110': ('D', 0),
        '011110|110011': ('D', 1),
        # cell H : cheap coalition vs lone oracle, dk=2
        '110111|001011': ('D', 0),
        '001011|110111': ('D', 1),
        # cell A : lone oracle vs .66 expert, mute rows = mutual endorsements
        '111011|110111': ('S', 0),
        '110111|111011': ('S', 1),
        # cell B : same difference vector, mute rows = mutual abstentions
        '001000|000100': ('S', 0),
        '000100|001000': ('S', 1),
        # cell G1 : two .74 edge experts vs the .62 expert (dk = 1)
        '101111|011110': ('S', 0),
        '011110|101111': ('S', 1),
        # cell F : three cheap/mid reasons vs two better ones (dk = 1)
        '111100|001011': ('S', 0),
        '001011|111100': ('S', 1),
    }

    hitD = []
    hitS = []
    for a, b, r in zip(data['option_a_ratings'].values,
                       data['option_b_ratings'].values,
                       data['response'].values):
        try:
            k = key(a, b)
        except Exception:
            continue
        info = table.get(k)
        if info is None:
            continue
        grp, fav = info
        try:
            resp = int(round(float(r)))
        except Exception:
            continue
        hit = 1.0 if resp == fav else 0.0
        if grp == 'D':
            hitD.append(hit)
        else:
            hitS.append(hit)

    if len(hitD) == 0 or len(hitS) == 0:
        return 0.0
    return float(np.mean(hitD) - np.mean(hitS))
```

**Observed (real) value:** -0.0014 (var=0.0071)
**Candidate trajectory (this loop):**
  - iter 1: 0.0165 (var=0.0055) (Δ vs real +0.0179)
  - iter 2 (current): 0.0217 (var=0.0084) (Δ vs real +0.0231)
**Other theories' values on this metric (for reference):**
- pi_10: 0.1265 (var=0.0081)
- pi_7: 0.3296 (var=0.0047)
- pi_1: -0.2972 (var=0.0124)
- pi_2: 0.0676 (var=0.0083)
- pi_3: 0.2393 (var=0.1177)
- pi_4: -0.4179 (var=0.0059)
- pi_5: 0.0247 (var=0.0061)
- pi_6: -0.1332 (var=0.0107)
- pi_8: 0.2485 (var=0.0077)
- pi_9: 0.1007 (var=0.0068)
- pi_11: 0.1203 (var=0.0053)

### Experiment 18
**Design**
  A=[1, 1, 0, 1, 1, 1, 0]  B=[0, 0, 1, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 1, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 0, 1, 1]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0, 0, 1]  B=[1, 1, 0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0, 1, 0]  B=[1, 1, 1, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0, 1, 1]  B=[1, 1, 0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 1, 1]  B=[0, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 0, 1]
  A=[1, 1, 0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 1, 0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    K2 = "01110101111111"   # cell 2: 3-0 unanimous, 4 mutual-endorsement mute rows
    K3 = "01110101111101"   # cell 3: identical but the weak pos5 expert dissents (3-1)
    K6 = "10101011101011"   # cell 6: oracle+.86 (2 speakers) vs 3 cheap experts
    K7 = "10101001101011"   # cell 7: same, plus one more cheap dissenter at the bottom edge (2-4)

    def canon(a, b):
        sa = ''.join([str(int(v)) for v in a])
        sb = ''.join([str(int(v)) for v in b])
        return min(sa + sb, sb + sa)

    acc = {K2: [0.0, 0.0], K3: [0.0, 0.0], K6: [0.0, 0.0], K7: [0.0, 0.0]}

    for _, row in data.iterrows():
        try:
            a = [int(v) for v in row['option_a_ratings']]
            b = [int(v) for v in row['option_b_ratings']]
            r = int(row['response'])
        except Exception:
            continue
        if len(a) != len(b) or len(a) == 0:
            continue
        k = canon(a, b)
        if k not in acc:
            continue
        kA = sum(1 for i in range(len(a)) if a[i] > b[i])
        kB = sum(1 for i in range(len(a)) if b[i] > a[i])
        if kA == kB:
            continue
        if k in (K2, K3):
            # did the subject follow the numerically larger camp?
            hit = 1.0 if ((kA > kB and r == 0) or (kB > kA and r == 1)) else 0.0
        else:
            # did the subject follow the smaller, high-validity (oracle-backed) camp?
            hit = 1.0 if ((kA < kB and r == 0) or (kB < kA and r == 1)) else 0.0
        acc[k][0] += hit
        acc[k][1] += 1.0

    def rate(k):
        s, n = acc[k]
        if n <= 0:
            return None
        return s / n

    p2, p3, p6, p7 = rate(K2), rate(K3), rate(K6), rate(K7)

    comps = []
    if (p2 is not None) and (p3 is not None):
        comps.append(p3 - p2)          # dissenter-raises-confidence effect
    if (p6 is not None) and (p7 is not None):
        comps.append(p6 - p7)          # one-extra-cheap-dissenter reversal
    if len(comps) == 0:
        return 0.0
    return float(np.mean(comps))

```

**Observed (real) value:** -0.0333 (var=0.0253)
**Candidate trajectory (this loop):**
  - iter 1: 0.0433 (var=0.0126) (Δ vs real +0.0767)
  - iter 2 (current): 0.0025 (var=0.0158) (Δ vs real +0.0358)
**Other theories' values on this metric (for reference):**
- pi_7: 0.3100 (var=0.0160)
- pi_10: 0.0475 (var=0.0190)
- pi_1: 0.0042 (var=0.0109)
- pi_2: 0.0017 (var=0.0088)
- pi_3: 0.0075 (var=0.0038)
- pi_4: 0.0033 (var=0.0018)
- pi_5: 0.0108 (var=0.0128)
- pi_6: 0.0283 (var=0.0063)
- pi_8: 0.2692 (var=0.0118)
- pi_9: 0.1450 (var=0.0161)
- pi_11: 0.0717 (var=0.0127)

### Experiment 19
**Design**
  A=[1, 1, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 1, 1, 1]  B=[1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1]  B=[1, 0, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1]  B=[0, 0, 1, 0, 0, 1, 1]
  A=[0, 0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 0, 1]  B=[1, 0, 1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0, 1, 1]  B=[1, 1, 0, 1, 1, 0, 1]
  A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 1]  B=[1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 1]
  A=[1, 1, 0, 1, 1, 0, 1]  B=[0, 0, 1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0, 1, 0]  B=[1, 1, 0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    # Cheap-cue-coalition cells: the difference vectors (A - B) in which a
    # coalition of low/mid-validity experts (positions 1,3,4 and, in the dense
    # cell, 0 and 6) opposes the near-oracle expert at position 2 (v=.96)
    # (and, in two cells, the .84 expert at position 5).
    # Orientation: +1 entries mark the COALITION side.
    pats = [
        (0, 1, -1, 1, 0, 0, 0),    # pairs 3/4  : 2 cheap cues vs oracle
        (0, 1, -1, 1, 1, 0, 0),    # pairs 5/6  : 3 cheap cues vs oracle
        (0, 1, -1, 1, 1, -1, 0),   # pairs 9/10 : 3 cheap cues vs oracle + .84
        (1, 1, -1, 1, 1, -1, 1),   # pairs 15/16: dense 5-vs-2
    ]
    lookup = {}
    for p in pats:
        lookup[p] = 1
        lookup[tuple(-x for x in p)] = -1

    hits = 0
    tot = 0
    a_col = data['option_a_ratings'].tolist()
    b_col = data['option_b_ratings'].tolist()
    r_col = data['response'].tolist()
    for a, b, r in zip(a_col, b_col, r_col):
        try:
            av = [int(x) for x in a]
            bv = [int(x) for x in b]
        except Exception:
            continue
        if len(av) != 7 or len(bv) != 7:
            continue
        d = tuple(av[i] - bv[i] for i in range(7))
        s = lookup.get(d)
        if s is None:
            continue
        try:
            resp = int(r)
        except Exception:
            continue
        tot += 1
        chose_a = (resp == 0)
        if (s == 1 and chose_a) or (s == -1 and not chose_a):
            hits += 1
    if tot == 0:
        return float('nan')
    return float(hits) / float(tot)
```

**Observed (real) value:** 0.8442 (var=0.0123)
**Candidate trajectory (this loop):**
  - iter 1: 0.7654 (var=0.0036) (Δ vs real -0.0787)
  - iter 2 (current): 0.7558 (var=0.0053) (Δ vs real -0.0883)
**Other theories' values on this metric (for reference):**
- pi_10: 0.5404 (var=0.0041)
- pi_11: 0.3121 (var=0.0029)
- pi_1: 0.1554 (var=0.0075)
- pi_2: 0.8588 (var=0.0081)
- pi_3: 0.7854 (var=0.0343)
- pi_4: 0.0242 (var=0.0007)
- pi_5: 0.3962 (var=0.0048)
- pi_6: 0.3533 (var=0.0054)
- pi_7: 0.5642 (var=0.0041)
- pi_8: 0.6488 (var=0.0050)
- pi_9: 0.5108 (var=0.0056)

### Experiment 20
**Design**
  A=[1, 1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 0, 0, 1]
  A=[1, 0, 1, 1, 0, 0, 1]  B=[1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0, 0, 1]  B=[1, 1, 0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1, 0]
  A=[0, 0, 1, 1, 1, 1, 0]  B=[1, 1, 0, 0, 1, 1, 1]
  A=[0, 1, 0, 1, 1, 1, 1]  B=[1, 0, 1, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 1, 1, 0]  B=[0, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 1, 1, 1]  B=[0, 1, 0, 1, 1, 1, 0]
  A=[0, 1, 0, 1, 1, 1, 0]  B=[1, 1, 1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0]
  A=[1, 1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0, 0]  B=[1, 1, 0, 1, 1, 1, 1]
  A=[1, 1, 0, 1, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np

    # Six target displays (ordered A,B rating strings), all mirrored pairs of
    # three base cells:
    #   cell 2 (sparse cheap-coalition vs oracle):      trials 3 / 4
    #   cell 3 (near-dense cheap-coalition vs oracle):  trials 5 / 6
    #   cell 5 (edge-OFF |dk|=1 swap):                  trials 9 / 10
    sel_keys = {
        ("1101111", "1011001"), ("1011001", "1101111"),
        ("1101110", "1010001"), ("1010001", "1101110"),
        ("0101111", "1010110"), ("1010110", "0101111"),
    }

    def key(seq):
        return "".join([str(int(round(float(x)))) for x in list(seq)])

    hits = 0
    tot = 0
    for a, b, r in zip(data["option_a_ratings"], data["option_b_ratings"], data["response"]):
        try:
            av = [int(round(float(x))) for x in list(a)]
            bv = [int(round(float(x))) for x in list(b)]
        except Exception:
            continue
        if len(av) != len(bv):
            continue
        k = (key(av), key(bv))
        if k not in sel_keys:
            continue
        kA = sum(1 for i in range(len(av)) if av[i] > bv[i])
        kB = sum(1 for i in range(len(av)) if bv[i] > av[i])
        if kA == kB:
            continue
        target_is_A = kA > kB
        try:
            resp = int(r)
        except Exception:
            continue
        chose_A = (resp == 0)
        tot += 1
        if chose_A == target_is_A:
            hits += 1

    if tot == 0:
        return float("nan")
    return float(hits) / float(tot)

```

**Observed (real) value:** 0.8478 (var=0.0059)
**Candidate trajectory (this loop):**
  - iter 1: 0.7667 (var=0.0053) (Δ vs real -0.0811)
  - iter 2 (current): 0.7867 (var=0.0058) (Δ vs real -0.0611)
**Other theories' values on this metric (for reference):**
- pi_11: 0.3572 (var=0.0091)
- pi_10: 0.5567 (var=0.0063)
- pi_1: 0.1439 (var=0.0113)
- pi_2: 0.8633 (var=0.0117)
- pi_3: 0.5167 (var=0.0554)
- pi_4: 0.0133 (var=0.0005)
- pi_5: 0.4072 (var=0.0046)
- pi_6: 0.1833 (var=0.0064)
- pi_7: 0.5967 (var=0.0098)
- pi_8: 0.4333 (var=0.0059)
- pi_9: 0.5261 (var=0.0057)

## YOUR PRIOR CRITIQUES (for this propose-loop)
Each block is one of YOUR previous critique iterations: the verdict you returned, the interpretation and rationale you wrote, and an **Outcome of your advice** line that says whether the candidate the proposer produced AFTER following your advice was ACCEPTED (your advice helped — its loss strictly beat the running-best) or REJECTED (your advice didn't help — the gate discarded the candidate). Use this self-history to (a) reinforce advice that was ACCEPTED, (b) avoid repeating advice that was REJECTED, and (c) detect your own oscillation across iterations.

### Iteration 1 (most recent)
Verdict: regenerate
Interpretation: The candidate is a clear step forward on the cells that killed both incumbents: the cheap-coalition experiments now land at 0.774 (Exp6, real 0.841), 0.765 (Exp19, real 0.844) and 0.767 (Exp20, real 0.848) instead of pi_10/pi_11's 0.31-0.60, and the near-flat validity map plus the sharpened (q~6) mute discount also zeroed the dense/sparse artefacts (Exp17 +0.017 vs real -0.001; Exp16 0.028 vs 0.097; Exp15 0.832 vs 0.855; Exp8 -0.171 vs -0.203; Exp7 0.023 vs 0.053; Exp9 0.473 vs 0.490; Exp5 0.779 vs 0.820). Those are genuinely good, experiment-invariant fits and the gate accepted the candidate (loss 0.1356).

The residual error is concentrated in five cells, and it is systematic rather than noise. (a) Authority-vs-two-cheap cells: Exp3 0.273 vs 0.518 and Exp4 0.243 vs 0.508 \u2014 the model follows the 2-vs-1 numerical majority at ~0.75 where humans are exactly at chance, and both are ~0.19-0.26 WORSE than pi_10 (0.463 / 0.426). The real between-subject variances there (0.023, 0.0075) are below what a mixture of count-followers and oracle-followers would produce, so subjects are individually near-indifferent in the |dk|=1-against-the-oracle cell: the drive itself should be ~0 there, not +0.75. (b) Exp1 -0.083 vs +0.093 and Exp13 0.226 vs 0.353 and Exp2 0.850 vs 0.775 and Exp12 0.749 vs 0.665: all four move in the SAME direction \u2014 the model's first-discriminating-row commitment is currently too weak/too narrowly gated. Note these four are mutually consistent: a slightly stronger, slightly more broadly triggered top-row stop raises Exp1 and Exp13 while lowering Exp2 and Exp12, all toward the data. (c) The deadlock edge cells remain internally split: Exp10 undershoots (0.627 vs 0.850) while Exp11's index overshoots (0.762 vs 0.513); the edge term currently has to fight the gamma-weighted tally in Exp10 (extremes = .62/.58 vs mids .95/.86) but rides free in Exp11.

So the mechanism family is right and the fit is already the running best; what is left is calibration of three knobs, not a re-design.
Rationale: Keep the prescribed numerosity-dominant tally family exactly as is \u2014 core near-flat weighted count, compressive read-out, difficulty-scaled lapse, deadlock-gated edge tie-break, diagnosticity-scaled rarity discount, composition-blind silence. Three in-family adjustments, in priority order:

1. MARGIN-DEPENDENT AUTHORITY SENSITIVITY (fixes Exp3 -0.245 and Exp4 -0.265, the two largest errors). Do not raise gamma globally \u2014 that would undo the Exp6/19/20 wins. Instead let the weighted-vote correction scale with closeness of the count: drive = lam*sign(T)*|T|^alpha with w_j = 1 + gamma_eff*u_j and gamma_eff = gamma0 + gamma1*exp(-(dk^2-1)^2/...) \u2014 or more simply, add a term -kappa*(max_u in the minority camp - max_u in the majority camp) that is active only at |dk| = 1 and decays to ~0 by |dk| >= 2. Calibrate kappa so a 2-cheap-vs-1-oracle display sits near chance (Exp3/Exp4 metrics ~0.45-0.50) while 3-vs-1 and 3-vs-2 coalitions stay at ~0.80+. The projected cost on Exp19 (its 2-vs-1 subcell drops) is ~0.04-0.05, versus ~0.45 of squared-error recovered on Exp3+Exp4.

2. SLIGHTLY STRONGER / BROADER AUTHORITY-COINCIDENCE STOP. You kept the stop against the arbiter's item 3, and the numbers vindicate keeping it in principle \u2014 but it is currently under-powered, not over-powered: Exp1 (-0.083 vs +0.093) and Exp13 (0.226 vs 0.353) both want MORE first-row commitment, and Exp2 (0.850 vs 0.775) and Exp12 (0.749 vs 0.665) both want LESS pure-tally following, which the same change delivers. Raise d_com to ~2.6-3.0 and soften the gate (phi ~0.95-1.00, s_stop ~4-6) so the stop also fires when the first discriminating row is merely comparable to, not strictly above, the best cue below it. Check that Exp6/19/20 are untouched (their first discriminating rows sit above the oracle, so the stop must stay off there) \u2014 if any of them drops, tighten phi back rather than abandoning the change.

3. RE-BALANCE THE DEADLOCK TERM RATHER THAN SCALING IT. Leave mu_e near its current amplitude (dropping it would sink Exp5/Exp10 further, raising it would blow up Exp11). Instead make the edge tie-break add to the drive in a way that is insensitive to the validity contrast of the deadlocked camps \u2014 e.g. compute the gate on raw dk and let the edge term partially CANCEL the weighted-vote residual at dk = 0 (so the .95/.86-vs-.62/.58 deadlock in Exp10 is not fighting a -0.6 tally residual). That should lift Exp10 from 0.63 toward 0.78-0.82 without inflating Exp11's c4. Also shave eps0 to ~0.06-0.10 to buy back the uniform ~0.07 shortfall on Exp6/19/20, but only after checking Exp14 (0.162 vs 0.300) does not fall further \u2014 if it does, prefer a small increase in lam over a lapse reduction.

Do not change families, do not add subpopulations, and do not reduce gamma below the current near-flat band \u2014 the coalition experiments are the part that is finally working.

**Outcome of your advice:** iter 1 candidate loss=0.1356 -> CURRENT candidate loss=0.1268 -> the gate marked it ACCEPTED.

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of the model simulation results on existing experiments and how they do or do not support the proposed model/theory.
- verdict: Either 'regenerate' (regenerate a new model/theory) or 'continue' (keep the current model/theory).
- rationale: If verdict='regenerate', why the current predict / policy / parameter-ranges should be revised. If verdict='continue', a sketch of why the current theory should be kept.


## Response

```json
{
  "interpretation": "Iteration 2 is a genuine improvement (0.1356 -> 0.1268) and the gate accepted it, so the prescribed numerosity-dominant tally family is clearly the right frame: the cheap-coalition cells that killed pi_10/pi_11 are now in the right neighbourhood (Exp6 0.766 vs 0.841, Exp19 0.756 vs 0.844, Exp20 0.787 vs 0.848), the rarity/silence machinery is essentially correct (Exp8 -0.166 vs -0.203; Exp15 0.809 vs 0.855; Exp17 +0.022 vs -0.001; Exp18 +0.003 vs -0.033; Exp16 0.032 vs 0.097), and the sparse-panel authority offset did exactly what I asked for on the two worst cells (Exp3 0.273->0.403 vs 0.518; Exp4 0.243->0.389 vs 0.508). That was the ACCEPTED direction, so it should be kept and pushed slightly further rather than reverted.\n\nThe remaining errors are still systematic and now sort into three coherent bundles. (i) MARGIN CURVATURE: every small-margin coalition cell is UNDER-predicted (Exp6 -0.076, Exp19 -0.088, Exp20 -0.061, Exp5 -0.070, Exp3 -0.115, Exp4 -0.119) while every large-margin / heavy-tally cell is OVER-predicted (Exp2 +0.080, Exp12 +0.083, Exp11 index +0.233). That is a read-out shape problem, not a weight problem: the drive is too flat at |dk|=1-2 and too steep at |dk|>=3. (ii) TOP-ROW COMMITMENT is still under-powered: Exp1 -0.167, Exp13 -0.133, Exp14 -0.145 all demand more weight on the first/most-diagnostic discriminating row, and the same change pulls Exp2 and Exp12 down toward their (lower) real values. Notably the half-step taken this iteration (d_com 2.20->2.55) moved Exp1 in the right direction (-0.083 -> -0.074) and cost nothing, i.e. the knob is directionally right but far from exhausted. (iii) The pure deadlock cell Exp10 still undershoots badly (0.647 vs 0.850, real var only 0.011 so this is a reliable target) because at dk=0 the near-flat weighted residual (weak extremes vs .95/.86 mids) actively fights the edge tie-break.\n\nOne new regression was introduced: Exp7 went from +0.023 to +0.210 (real 0.053) because the mutual-absence gate fires on Exp7's low-load 2-vs-1 displays, which are structurally identical to Exp3/Exp4's. Exp7's between-subject variance is the largest in the set (0.049), so this is the cheapest of the residuals, but the gate is currently broader than it needs to be.",
  "verdict": "regenerate",
  "rationale": "Stay strictly inside the prescribed numerosity-dominant tally family (near-flat weights, compressive count read-out, deadlock-gated edge tie-break, diagnosticity-scaled rarity discount, difficulty lapse, no subpopulations). Keep the sparse-panel authority offset \u2014 that was my accepted recommendation and it bought ~0.14 on Exp3+Exp4. Three in-family calibration moves, in priority order:\n\n1. RESHAPE THE MARGIN CURVE (new, and the highest-yield move). Increase compression and compensate with gain: alpha ~0.42-0.50 (from 0.60) with lam re-scaled so that the large-margin drive is UNCHANGED or slightly reduced (e.g. lam ~1.50-1.65, chosen so lam*|T|^alpha at |T|~3.5 matches the current value while |T|~1.0-1.5 rises by ~20-25%). This lifts every |dk|=1-2 coalition cell (Exp6/19/20/5, and the post-offset Exp3/4 which currently undershoot from below) while leaving/pulling down the over-predicted heavy-tally cells (Exp2, Exp12, Exp11's c8). Do NOT do this with a global eps0 cut \u2014 the under- and over-shoots are on opposite sides of the margin axis, so a lapse change is a wash (as the proposer correctly noted).\n\n2. FINISH THE AUTHORITY-COINCIDENCE STOP (continue the direction that was just accepted, don't stop half-way). d_com ~2.9-3.1 and a softer trigger (phi ~0.95-0.99, s_stop ~4-6) so the stop also fires when the first discriminating row is merely comparable to \u2014 not strictly above \u2014 the best cue below it. Target Exp1 toward ~0.0 to +0.05, Exp13 toward ~0.28-0.32, Exp14 toward ~0.22-0.26; the same change drags Exp2 and Exp12 down toward 0.78/0.68. Guard-rail: re-check Exp6/19/20/17 after the change; if any of them drops more than ~0.02, tighten phi back rather than abandoning the stop.\n\n3. DAMP THE WEIGHTED RESIDUAL AT EXACT DEADLOCK (mild version of my untested item-3, not a repeat of anything the gate rejected). At dk = 0 only, shrink the gamma-driven tally residual by a factor beta ~0.4-0.5 before adding the edge term, so the Exp10 display (weak screen extremes vs .95/.86 mids) is not fighting a negative residual. Target Exp10 ~0.75-0.82. Because this term is active only at dk=0 it also touches Exp11's c4 \u2014 keep beta moderate and verify the Exp11 index does not exceed ~0.80; if it does, back beta off to ~0.3 rather than touching mu_e (lowering mu_e would sink Exp5/Exp10, raising it would blow up Exp11).\n\n4. (Second-order) NARROW THE MUTUAL-ABSENCE GATE so it stops leaking into Exp7's low-load 2-vs-1 cells: raise the required diagnosticity gap (d0_u ~0.10-0.15 and/or a steeper d_scale) and/or require a larger absence fraction (f_half ~0.33-0.38), so theta fires on the oracle-vs-two-cheap displays but is <0.2 when the lone dissenter's validity edge is only modest. Accept a small Exp3/Exp4 cost only if Exp7 falls below ~0.12; Exp7's real value has the largest between-subject variance in the set, so do not trade more than ~0.03 of Exp3/Exp4 accuracy for it.\n\nNo family switch, no subpopulations, no reduction of gamma below the current near-flat band."
}
```

## Usage

```json
{
  "input_tokens": 48757,
  "output_tokens": 7742
}
```
