"""
Steps 2-3 - Log-likelihood and MLE for block GPHC Kumaraswamy data.

Model: block i follows Kumaraswamy(a, b_i); a is shared by all blocks.

Log-likelihood (constants dropped):
    l(a, b_1..b_K) = sum_i [ D_i ln a + D_i ln b_i + (a-1) sum_j ln x_ij
                             - sum_j ln(1 - x_ij^a) - b_i * Delta_i(a) ]

For a fixed a, each b_i has a direct formula:
    b_i(a) = D_i / Delta_i(a)

Putting that back gives a function of a alone (profile log-likelihood),
which is maximised numerically:
    l_p(a) = (sum_i D_i) ln a + sum_i D_i ln(D_i / Delta_i(a))
             + (a-1) sum ln x - sum ln(1 - x^a) - sum_i D_i

Needs generatedata.py (Step 1) in the same folder.

Run this file directly to run every Step 2-3 check:
    python mle.py
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize

from generatedata import generate_block_gphc, delta


# ---------------------------------------------------------------------------
# Step 2 - log-likelihood
# ---------------------------------------------------------------------------

def loglik(a, b, blocks):
    """Full log-likelihood l(a, b_1..b_K). b is a list/array of length K."""
    if a <= 0 or np.any(np.asarray(b) <= 0):
        return -np.inf
    total = 0.0
    for b_i, blk in zip(b, blocks):
        x, D = blk["x"], blk["D"]
        total += (D * np.log(a) + D * np.log(b_i)
                  + (a - 1) * np.sum(np.log(x))
                  - np.sum(np.log1p(-x ** a))
                  - b_i * delta(a, blk))
    return total


def b_given_a(a, blocks):
    """b_i(a) = D_i / Delta_i(a) for every block."""
    return np.array([blk["D"] / delta(a, blk) for blk in blocks])


def profile_loglik(a, blocks):
    """l_p(a): the log-likelihood with every b_i replaced by b_i(a)."""
    return loglik(a, b_given_a(a, blocks), blocks)


# ---------------------------------------------------------------------------
# Step 3 - MLE
# ---------------------------------------------------------------------------

def mle(blocks, a_low=0.01, a_high=50.0):
    """
    Maximum likelihood estimates.
    Searches over log(a) between a_low and a_high (more stable than a itself).

    Returns a dict: a, b (array), loglik, converged (True/False)
    """
    res = minimize_scalar(lambda t: -profile_loglik(np.exp(t), blocks),
                          bounds=(np.log(a_low), np.log(a_high)),
                          method="bounded", options={"xatol": 1e-10})
    a_hat = float(np.exp(res.x))
    b_hat = b_given_a(a_hat, blocks)
    # if the answer sits on the edge of the search range, something is wrong
    on_edge = (a_hat < a_low * 1.01) or (a_hat > a_high * 0.99)
    return {"a": a_hat, "b": b_hat, "loglik": -res.fun,
            "converged": bool(res.success and not on_edge)}


# ---------------------------------------------------------------------------
# Checks (run this file directly)
# ---------------------------------------------------------------------------

def run_checks(out_dir="."):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(2026)
    a_true, b_true = 2.0, np.array([1.5, 2.0, 2.5])
    K = 3
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    def make(n, m, k, T, plan):
        return generate_block_gphc(a_true, b_true, [n] * K, [m] * K, [k] * K,
                                   [T] * K, plan, rng)

    # CHECK 1 - the likelihood matches the generator.
    # At the true values, the slope (score) of the log-likelihood averages
    # to zero over many datasets. A wrong likelihood (e.g. a missing Case II
    # term) makes the average slope clearly non-zero.
    say("CHECK 1 - average slope of log-likelihood at the true values (should be ~0)")
    theta0 = np.concatenate([[a_true], b_true])

    def ll_vec(th, blocks):
        return loglik(th[0], th[1:], blocks)

    def score(blocks, h=1e-6):
        g = np.zeros(len(theta0))
        for p in range(len(theta0)):
            up, dn = theta0.copy(), theta0.copy()
            up[p] += h
            dn[p] -= h
            g[p] = (ll_vec(up, blocks) - ll_vec(dn, blocks)) / (2 * h)
        return g

    names = ["a", "b1", "b2", "b3"]
    all_ok = True
    for T, label in [(0.6, "T=0.6 (mostly Case I/II)"), (0.9, "T=0.9 (mostly Case II/III)")]:
        S = np.array([score(make(20, 16, 12, T, "II")) for _ in range(3000)])
        z = S.mean(0) / (S.std(0) / np.sqrt(len(S)))
        ok = bool(np.all(np.abs(z) < 4))
        all_ok &= ok
        say(f"  {label}, Plan II, 3000 datasets")
        say("    average slope in std errors: "
            + "  ".join(f"{nm}={v:+.2f}" for nm, v in zip(names, z)))
        say(f"    passes (all between -4 and +4): {ok}")
    say("  (random noise gives values up to about 3; a real mistake in the")
    say("   likelihood, e.g. leaving out the Case II term, gives 15 to 60)")
    say()

    # CHECK 2 - one dataset: profile likelihood has a single peak, and the
    # 1-D answer equals a full 4-parameter search.
    say("CHECK 2 - one dataset (S1, T=0.6, Plan III)")
    data = make(20, 16, 12, 0.6, "III")
    est = mle(data)
    say(f"  cases: {[blk['case'] for blk in data]}   D: {[blk['D'] for blk in data]}")
    say(f"  MLE:  a = {est['a']:.4f}   b = "
        + ", ".join(f"{v:.4f}" for v in est["b"])
        + f"   (true a = 2, b = 1.5, 2.0, 2.5)")
    full = minimize(lambda th: -ll_vec(np.exp(th), data),
                    np.log(np.concatenate([[1.0], [1.0] * K])),
                    method="Nelder-Mead",
                    options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    full_est = np.exp(full.x)
    gap = np.max(np.abs(full_est - np.concatenate([[est["a"]], est["b"]])))
    say(f"  full 4-parameter search gives: " + ", ".join(f"{v:.4f}" for v in full_est))
    say(f"  largest difference = {gap:.2e}  (should be below 1e-3): {gap < 1e-3}")

    grid = np.linspace(0.3, 6, 400)
    prof = [profile_loglik(a, data) for a in grid]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, prof, lw=2)
    ax.axvline(est["a"], ls="--", color="grey", label=f"MLE a = {est['a']:.3f}")
    ax.axvline(a_true, ls=":", color="black", label="true a = 2")
    ax.set_xlabel("a")
    ax.set_ylabel("profile log-likelihood")
    ax.set_title("Profile log-likelihood of a (single peak)")
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "step3_profile_loglik.png")
    fig.savefig(fig_path, dpi=150)
    peaks = int(np.sum((np.diff(np.sign(np.diff(prof))) < 0)))
    say(f"  number of peaks on the grid: {peaks} (should be 1)")
    say(f"  plot saved: {fig_path}")
    say()

    # CHECK 3 - with lots of data the MLE lands on the true values
    say("CHECK 3 - one very large dataset (n_i = 2000, m_i = 1800, k_i = 1500, T = 0.95, Plan I)")
    big = make(2000, 1800, 1500, 0.95, "I")
    est = mle(big)
    say(f"  MLE:  a = {est['a']:.4f}   b = " + ", ".join(f"{v:.4f}" for v in est["b"]))
    say("  true: a = 2.0000   b = 1.5000, 2.0000, 2.5000   (should be very close)")
    say()

    # CHECK 4 - small repeat study: bias shrinks as samples grow
    say("CHECK 4 - average MLE over 1000 datasets (T = 0.6, Plan I)")
    say("  setup   avg a    avg b1   avg b2   avg b3   failed fits")
    for name, (n, m, k) in {"S1": (20, 16, 12), "S2": (30, 24, 18),
                            "n=100": (100, 80, 60)}.items():
        ests, fails = [], 0
        for _ in range(1000):
            e = mle(make(n, m, k, 0.6, "I"))
            if e["converged"]:
                ests.append(np.concatenate([[e["a"]], e["b"]]))
            else:
                fails += 1
        avg = np.mean(ests, axis=0)
        say(f"  {name:<6}  " + "   ".join(f"{v:.4f}" for v in avg) + f"   {fails}")
    say("  true:   2.0000   1.5000   2.0000   2.5000")
    say("  (small samples are a little off - normal for MLE; it should improve down the table)")

    txt_path = os.path.join(out_dir, "step3_check_results.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nresults saved: {txt_path}")


if __name__ == "__main__":
    run_checks()