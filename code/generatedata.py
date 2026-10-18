"""
Step 1 - Generate block generalized progressive hybrid censored (block GPHC)
data from the Kumaraswamy distribution.

Kumaraswamy(a, b) on 0 < x < 1:
    F(x) = 1 - (1 - x^a)^b
    S(x) = (1 - x^a)^b
    inverse CDF: x = [1 - (1 - u)^(1/b)]^(1/a)

Every block shares a; block i has its own b_i.

How one block is generated:
    1. Draw a progressive Type-II sample of all m failures with the
       Balakrishnan-Sandhu (1995) algorithm, using the planned removals R.
    2. Cut it with the GPHC stopping rule (Case I, II or III).
This is valid because the first j failures of a progressive sample depend
only on the removals made before failure j.

Run this file directly to run every Step 1 check:
    python step1_generate.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# Kumaraswamy basics
# ---------------------------------------------------------------------------

def kum_pdf(x, a, b):
    return a * b * x ** (a - 1) * (1 - x ** a) ** (b - 1)


def kum_cdf(x, a, b):
    return 1 - (1 - x ** a) ** b


def kum_inv_cdf(u, a, b):
    return (1 - (1 - u) ** (1 / b)) ** (1 / a)


# ---------------------------------------------------------------------------
# Removal plans (Burr-XII / IER papers)
# ---------------------------------------------------------------------------

def removal_plan(plan, n, m):
    """Planned removals R_1..R_m. They always add up to n - m."""
    R = np.zeros(m, dtype=int)
    if plan == "I":            # everyone left is removed at the m-th failure
        R[-1] = n - m
    elif plan == "II":         # everyone extra is removed at the first failure
        R[0] = n - m
    elif plan == "III":        # half at the first failure, half at the end
        R[0] = int(np.ceil((n - m) / 2))
        R[-1] = (n - m) - R[0]
    else:
        raise ValueError("plan must be 'I', 'II' or 'III'")
    return R


# ---------------------------------------------------------------------------
# One progressive Type-II sample (Balakrishnan & Sandhu, 1995)
# ---------------------------------------------------------------------------

def progressive_sample(R, a, b, rng):
    """All m progressive failure times for removal plan R, already sorted."""
    R = np.asarray(R)
    m = len(R)
    W = rng.uniform(size=m)
    # exponent for V_j is  j + R_m + R_{m-1} + ... + R_{m-j+1},  j = 1..m
    tail_sums = np.cumsum(R[::-1])              # R_m, R_m+R_{m-1}, ...
    j = np.arange(1, m + 1)
    V = W ** (1.0 / (j + tail_sums))
    # U_j = 1 - V_m * V_{m-1} * ... * V_{m-j+1}
    U = 1 - np.cumprod(V[::-1])
    return kum_inv_cdf(U, a, b)


# ---------------------------------------------------------------------------
# Generalized progressive hybrid stopping rule for one block
# ---------------------------------------------------------------------------

def apply_gphc(x_all, R, n, k, T):
    """
    Cut a full progressive sample with the GPHC rule.

    Returns a dict:
        x      observed failure times (length D)
        r      removals actually made at each observed failure (length D)
        rstar  units still running when the block stops at T (Case II only)
        D      number of observed failures
        case   "I", "II" or "III"
    """
    m = len(R)
    R = np.asarray(R)

    if x_all[m - 1] < T:                      # Case III: m-th failure before T
        case, D = "III", m
        x = x_all.copy()
        r = R.copy()
        rstar = 0
    elif x_all[k - 1] < T:                    # Case II: T between k-th and m-th
        case = "II"
        D = int(np.sum(x_all < T))            # d failures before T
        x = x_all[:D].copy()
        r = R[:D].copy()
        rstar = n - int(np.sum(r + 1))
    else:                                     # Case I: T before k-th failure
        case, D = "I", k
        x = x_all[:k].copy()
        r = R[:k].copy()
        r[-1] = n - k - int(np.sum(R[:k - 1]))   # remove everyone left
        rstar = 0

    # accounting check: failures + removals + survivors at T = n
    total = D + int(np.sum(r)) + rstar
    if total != n:
        raise RuntimeError(f"Units do not add up: {total} != {n} (case {case})")

    return {"x": x, "r": r, "rstar": rstar, "D": D, "case": case}


# ---------------------------------------------------------------------------
# Whole block experiment
# ---------------------------------------------------------------------------

def generate_block_gphc(a, b_list, n_list, m_list, k_list, T_list, plan, rng,
                        min_failures=2, max_tries=1000):
    """
    One block GPHC dataset: a list with one dict per block (see apply_gphc),
    plus the block's settings. Redraws a block if it has fewer than
    `min_failures` failures (needed by the pivotal method).
    """
    blocks = []
    for b, n, m, k, T in zip(b_list, n_list, m_list, k_list, T_list):
        R = removal_plan(plan, n, m)
        for _ in range(max_tries):
            x_all = progressive_sample(R, a, b, rng)
            blk = apply_gphc(x_all, R, n, k, T)
            if blk["D"] >= min_failures:
                break
        else:
            raise RuntimeError("Could not get enough failures; check T and k.")
        blk.update({"n": n, "m": m, "k": k, "T": T, "R_planned": R})
        blocks.append(blk)
    return blocks


# ---------------------------------------------------------------------------
# Delta_i(a): the "total exposure" used by every later step
# ---------------------------------------------------------------------------

def delta(a, blk):
    """
    Delta_i(a) = - sum_j (1 + r_ij) ln(1 - x_ij^a) - r_i* ln(1 - T_i^a)
    The second term only matters in Case II (otherwise r_i* = 0).
    """
    x, r = blk["x"], blk["r"]
    val = -np.sum((1 + r) * np.log1p(-x ** a))
    if blk["rstar"] > 0:
        val -= blk["rstar"] * np.log1p(-blk["T"] ** a)
    return val


# ---------------------------------------------------------------------------
# Checks (run this file directly)
# ---------------------------------------------------------------------------

def brute_force_block(a, b, n, R, rng):
    """Run the life test literally, unit by unit (used only to check the
    fast generator). No time limit: returns all m progressive failures."""
    alive = list(kum_inv_cdf(rng.uniform(size=n), a, b))
    out = []
    for Rj in R:
        alive.sort()
        out.append(alive.pop(0))                  # next failure
        if Rj > 0:                                # remove Rj survivors at random
            idx = rng.choice(len(alive), size=Rj, replace=False)
            alive = [v for i, v in enumerate(alive) if i not in set(idx)]
    return np.array(out)


def run_checks(out_dir="."):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    rng = np.random.default_rng(2026)
    a_true, b_true = 2.0, (1.5, 2.0, 2.5)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    # Check 0: the toy example from the guide, worked by the code
    say("CHECK 0 - toy example (n=10, m=6, k=4, T=0.5, Plan II)")
    toy_x = np.array([0.21, 0.33, 0.41, 0.47, 0.58, 0.66])
    toy = apply_gphc(toy_x, removal_plan("II", 10, 6), n=10, k=4, T=0.5)
    toy_r = [int(v) for v in toy["r"]]
    say(f"  case={toy['case']}  D={toy['D']}  removals={toy_r}  r*={toy['rstar']}")
    ok0 = toy["case"] == "II" and toy["D"] == 4 and toy_r == [4, 0, 0, 0] \
        and toy["rstar"] == 2
    say(f"  matches hand answer (Case II, D=4, (4,0,0,0), r*=2): {ok0}")
    say()

    # Check 1: uncensored values follow Kumaraswamy
    say("CHECK 1 - uncensored values vs Kumaraswamy(2, 2)")
    # One K-S test fails 5% of the time by pure chance, so repeat it:
    # with a correct generator about 5% of 400 tests should have p < 0.05.
    pvals = [stats.kstest(progressive_sample(np.zeros(2000, dtype=int), 2.0, 2.0, rng),
                          lambda x: kum_cdf(x, 2.0, 2.0)).pvalue for _ in range(400)]
    say(f"  400 K-S tests of 2000 values each: {np.mean(np.array(pvals) < 0.05):.1%}"
        f" had p < 0.05  (should be close to 5%)")
    big = progressive_sample(np.zeros(10000, dtype=int), 2.0, 2.0, rng)
    grid = np.linspace(0.001, 0.999, 400)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(big, bins=50, density=True, alpha=0.5, label="generated")
    ax.plot(grid, kum_pdf(grid, 2.0, 2.0), lw=2, label="Kumaraswamy(2, 2) density")
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title("Check 1: generator vs true density")
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "step1_check_histogram.png")
    fig.savefig(fig_path, dpi=150)
    say(f"  plot saved: {fig_path}")
    say()

    # Check 2: fast generator agrees with a literal unit-by-unit life test
    reps = 20000
    for plan in ("II", "III"):
        say(f"CHECK 2{plan} - fast generator vs literal life test "
            f"(n=20, m=8, Plan {plan}, {reps} reps)")
        R = removal_plan(plan, 20, 8)
        fast = np.array([progressive_sample(R, 2.0, 2.0, rng) for _ in range(reps)])
        slow = np.array([brute_force_block(2.0, 2.0, 20, R, rng) for _ in range(reps)])
        gap = fast.mean(0) - slow.mean(0)
        se = np.sqrt(fast.var(0) / reps + slow.var(0) / reps)
        say("  failure j : mean (fast)  mean (literal)  gap in std errors")
        for j in range(8):
            say(f"      {j+1}     :   {fast[:, j].mean():.4f}       "
                f"{slow[:, j].mean():.4f}          {gap[j] / se[j]:+.2f}")
        say("  passes if every gap is between -3 and +3 std errors: "
            f"{bool(np.all(np.abs(gap / se) < 3))}")
        say()

    # Check 3: accounting + case frequencies for the simulation settings
    say("CHECK 3 - case frequencies over 2000 datasets per setting")
    say("  a = 2, b = (1.5, 2.0, 2.5); every block's units add up (checked inside the code)")
    say("  setup  T    plan   Case I   Case II   Case III   (share of blocks)")
    setups = {"S1": (20, 16, 12), "S2": (30, 24, 18)}
    for name, (n, m, k) in setups.items():
        for T in (0.6, 0.9):
            for plan in ("I", "II", "III"):
                counts = {"I": 0, "II": 0, "III": 0}
                for _ in range(2000):
                    data = generate_block_gphc(a_true, b_true, [n] * 3, [m] * 3,
                                               [k] * 3, [T] * 3, plan, rng)
                    for blk in data:
                        counts[blk["case"]] += 1
                tot = sum(counts.values())
                say(f"   {name}   {T}   {plan:<4}   {counts['I']/tot:6.1%}   "
                    f"{counts['II']/tot:7.1%}   {counts['III']/tot:8.1%}")
    say()

    # Example dataset, printed so you can see what the output looks like
    say("EXAMPLE - one dataset (S1, T=0.6, Plan III)")
    data = generate_block_gphc(a_true, b_true, [20] * 3, [16] * 3, [12] * 3,
                               [0.6] * 3, "III", rng)
    for i, blk in enumerate(data, start=1):
        say(f"  Block {i}: case {blk['case']}, D={blk['D']}, r*={blk['rstar']},"
            f" Delta(a=2)={delta(2.0, blk):.4f}")
        say("    x = " + ", ".join(f"{v:.4f}" for v in blk["x"]))
        say("    r = " + ", ".join(str(v) for v in blk["r"]))

    txt_path = os.path.join(out_dir, "step1_check_results.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nresults saved: {txt_path}")


if __name__ == "__main__":
    run_checks()