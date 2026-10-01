#!/usr/bin/env python
"""
Second PySR pass for q_tot = f(Hm0, Tp, wave angle).

Changes from pysr_qtot.py:
  * Error-model weighting: each case is weighted by 1 / sigma^2, where
    sigma^2 = (r * typical|q|)^2 + sigma0^2 is estimated from the data (a relative error for
    large cases plus a noise floor for tiny ones). Unweighted, the largest cases dominated and
    formulas got the transport direction wrong for many small-wave cases; purely relative
    weights would instead let noise-dominated tiny cases dominate.
  * The dimensionless form gets the missing third group Hs/h0 (h0 = offshore depth).
  * Every formula is scored three ways on the held-out 20% of cases:
      R2_q      ordinary R^2 on q_tot (dominated by large-wave cases)
      R2_w      R^2 with each case scaled by its estimated error sigma (what the fit minimises)
      sign_ok   fraction of cases where the predicted transport direction is right
  * The argument of sin/cos/log/sqrt is limited in size, which rules out the rapidly
    oscillating cos(theta/s0^2)-type fits seen in the first run.
  * Stage 0 also shows where the data's transport runs opposite to sin(2 theta).

Run from the folder holding q_tot_summary_3d.csv:
  sbatch --cpus-per-task=16 --mem=16G --time=06:00:00 --output=logs/pysr_w_%j.out \
         --wrap="export PYTHON_JULIACALL_THREADS=16 && /hpc/home/acd99/.conda/envs/pysr/bin/python -u pysr_qtot_weighted.py"
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

CSV = "q_tot_summary_dx5.csv"
G = 9.81
H0_DEPTH = 20.0               # offshore water depth in the model (z0 in hy_setup), m
NITERATIONS = 300
TIMEOUT_S = 2 * 3600          # cap per PySR stage
rng = np.random.default_rng(0)  # same seed and draw as pysr_qtot.py -> same train/test split


def r2(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


# ----------------------------------------------------------------------------- data
df = pd.read_csv(CSV).dropna(subset=["q_tot"])
H, T, s0, ang, q = (df[c].to_numpy(float) for c in ["hm0", "tp", "s0", "mainang", "q_tot"])
test = rng.random(len(df)) < 0.2
train = ~test
print(f"{len(df)} cases, train {train.sum()}, test {test.sum()}")

# ----------------------------------------------------------------------------- 0. angle + reversals
def cerc(X, K, th_n):
    H_, _, ang_ = X
    return K * H_ ** 2.5 * np.sin(2 * np.deg2rad(ang_ - th_n))

K0 = np.linalg.lstsq((H ** 2.5 * np.sin(2 * np.deg2rad(ang - 270)))[train, None], q[train], rcond=None)[0][0]
p_cerc, _ = curve_fit(cerc, (H[train], T[train], ang[train]), q[train], p0=[K0, 270.0])
th_n = p_cerc[1]
theta = np.deg2rad(ang - th_n)
print(f"\n[0] Shore-normal direction: {th_n:.1f} deg")

expected = np.sign(p_cerc[0] * np.sin(2 * theta))       # direction the simple sin(2 theta) law predicts
reversed_ = (np.sign(q) != expected) & (q != 0)
print(f"    Transport opposite to the sin(2 theta) direction in {reversed_.mean():.0%} of cases")
print(f"    median |q| of those cases / others: {np.median(np.abs(q[reversed_])) / np.median(np.abs(q[~reversed_])):.2f}")
rev = pd.DataFrame({"reversed": reversed_, "Tp (s)": T, "Hm0 (m)": H, "|theta| (deg)": np.abs(np.rad2deg(theta))})
for col, bins in [("Tp (s)", [0, 5, 6, 7, 7.5, 8, 9, 11, 15]),
                  ("Hm0 (m)", [0.5, 1, 1.5, 2, 2.5, 3.01]),
                  ("|theta| (deg)", [0, 5, 10, 20, 40, 80])]:
    tab = rev.groupby(pd.cut(rev[col], bins), observed=True)["reversed"].agg(["mean", "size"])
    print(f"    fraction reversed by {col}: " +
          ", ".join(f"{iv}: {m:.0%} (n={n})" for iv, (m, n) in tab.iterrows()))

# ----------------------------------------------------------------------------- 1. baselines + weights
def kamphuis(X, K, a, b, c):
    H_, T_, th_ = X
    s = np.sin(2 * th_)
    return K * H_ ** a * T_ ** b * np.sign(s) * np.abs(s) ** c

p_k, _ = curve_fit(kamphuis, (H[train], T[train], theta[train]), q[train],
                   p0=[p_cerc[0], 2.0, 0.5, 0.6], maxfev=20000)
# typical transport size for each case's Hm0 and Tp, ignoring the angle term (which is ~0 near
# shore-normal and would give those cases huge weights)
mag = np.abs(p_k[0]) * H ** p_k[1] * T ** p_k[2]


# error model sigma^2 = (r * mag)^2 + sigma0^2, from the Kamphuis-like residuals in 10 bins of mag
from scipy.optimize import nnls
res2 = (q - kamphuis((H, T, theta), *p_k)) ** 2
b = pd.DataFrame({"m2": mag[train] ** 2, "res2": res2[train]}).groupby(
    pd.qcut(mag[train], 10, duplicates="drop"), observed=True).mean()
# each bin weighted equally (divide by its own res2), so the small-case noise floor isn't swamped
A = np.column_stack([b["m2"], np.ones(len(b))]) / b["res2"].to_numpy()[:, None]
(r_sq, sig0_sq), _ = nnls(A, np.ones(len(b)))
if r_sq == 0 and sig0_sq == 0:
    sig0_sq = res2[train].mean()
sigma = np.sqrt(r_sq * mag ** 2 + sig0_sq)
print(f"\n    error model: sigma = sqrt(({np.sqrt(r_sq):.3f} * typical|q|)^2 + {np.sqrt(sig0_sq):.3g}^2); "
      f"largest/smallest weight = {(sigma.max() / sigma.min()) ** 2:.0f}x")


def scores(qhat, idx=test):
    y, sg = q[idx], sigma[idx]
    return {"R2_q": r2(y, qhat), "R2_w": r2(y / sg, qhat / sg), "sign_ok": np.mean(np.sign(qhat) == np.sign(y))}


baselines = {
    "CERC-like  K H^2.5 sin(2theta)": cerc((H[test], T[test], ang[test]), *p_cerc),
    f"Kamphuis-like  K H^{p_k[1]:.2f} T^{p_k[2]:.2f} sgn(s)|s|^{p_k[3]:.2f}": kamphuis((H[test], T[test], theta[test]), *p_k),
    "first PySR run, complexity 14": -1.0106503e-4 * 5.8204546 ** H[test] * (T[test] - 7.513483) * np.sin(2.019526 * theta[test]),
}
print("\n[1] Baselines on the test cases")
print(f"    {'':50s} {'R2_q':>7s} {'R2_w':>7s} {'sign_ok':>8s}")
for name, qhat in baselines.items():
    s = scores(qhat)
    print(f"    {name:50s} {s['R2_q']:7.3f} {s['R2_w']:7.3f} {s['sign_ok']:8.0%}")


# ----------------------------------------------------------------------------- 2-3. weighted PySR
def run_pysr(tag, X, y, w, names, to_q):
    from pysr import PySRRegressor

    model = PySRRegressor(
        niterations=NITERATIONS,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sin", "cos", "sqrt", "log"],
        constraints={"^": (-1, 1), "sin": 5, "cos": 5, "log": 5, "sqrt": 5},
        nested_constraints={
            "sin": {"sin": 0, "cos": 0}, "cos": {"sin": 0, "cos": 0},
            "log": {"log": 0}, "sqrt": {"sqrt": 0},
        },
        maxsize=25,
        model_selection="best",
        timeout_in_seconds=TIMEOUT_S,
        random_state=0,
    )
    model.fit(X[train], y[train], weights=w[train] / w[train].mean(), variable_names=names)

    eqs = model.equations_.copy()
    rows = [scores(to_q(model.predict(X[test], index=i))) for i in range(len(eqs))]
    for k in ["R2_q", "R2_w", "sign_ok"]:
        eqs[k] = [r[k] for r in rows]
    cols = ["complexity", "loss", "R2_q", "R2_w", "sign_ok", "equation"]
    eqs[cols].to_csv(f"pysr_{tag}_equations.csv", index=False)
    print(f"\n[{tag}] Pareto front, scored on test cases:")
    with pd.option_context("display.max_colwidth", 90, "display.width", 180, "display.float_format", "{:.3f}".format):
        print(eqs[cols].to_string(index=False))
    return model


w_rel = 1.0 / sigma ** 2        # weighted loss = sum(((qhat - q) / sigma)^2)

# 2. dimensional: q_tot from Hs, Tp, theta
run_pysr("2w_dimensional", np.column_stack([H, T, theta]), q, w_rel,
         ["Hs", "Tp", "theta"], to_q=lambda yhat: yhat)

# 3. dimensionless: Q* = q / (sqrt(g) Hs^2.5) from s0, theta, Hs/h0 (same effective weighting)
scale = np.sqrt(G) * H ** 2.5
run_pysr("3w_dimensionless", np.column_stack([s0, theta, H / H0_DEPTH]), q / scale, w_rel * scale ** 2,
         ["s0", "theta", "H_h0"], to_q=lambda yhat: yhat * scale[test])
