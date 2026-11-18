"""Paper figures in out/figures: fig_weff (true vs CPL w(z)), fig_w0wa (posterior vs RVM track), fig_noise (noisy mocks)."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "python"))
from p1.models import theory  # noqa: E402

FIG = HERE / "out" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "legend.fontsize": 8, "figure.dpi": 150,
                     "mathtext.fontset": "cm", "font.family": "serif"})
real = json.loads((HERE / "results" / "real_fits.json").read_text())
mock = json.loads((HERE / "results" / "mock_test.json").read_text())
runs = mock["runs"]
base = mock["base"]


def weff_true(nu, zs):
    t = theory("rvm", list(base) + [nu])
    return np.array(t.cosmo.w_eff(list(zs), 1090.0))


# ---------- Fig 1: true vs apparent w(z) ----------
zs = np.linspace(0.0, 2.5, 200)
a = 1 / (1 + zs)
fig, ax = plt.subplots(figsize=(5.2, 3.7))
cols = {-0.002: "#b2182b", -0.001: "#ef8a62", 0.001: "#67a9cf", 0.002: "#2166ac"}
for nu, c in cols.items():
    r = next(x for x in runs if abs(x["nu"] - nu) < 1e-9)
    ax.plot(zs, weff_true(nu, zs), color=c, lw=1.8, label=fr"$\nu={nu:+.3f}$: true $w_{{\rm eff}}$")
    ax.plot(zs, r["w0"] + r["wa"] * (1 - a), color=c, lw=1.1, ls="--")
cpl = real["cpl"]
w0m, wam = cpl["mean"][3], cpl["mean"][4]
ch = np.load(HERE / "results" / "chains" / "cpl.npy").astype(float)
sel = ch[np.random.default_rng(0).choice(len(ch), 3000, replace=False)]
band = sel[:, 3][:, None] + sel[:, 4][:, None] * (1 - a)[None, :]
lo, hi = np.percentile(band, [16, 84], axis=0)
ax.fill_between(zs, lo, hi, color="0.75", alpha=0.6, lw=0, label=r"real data, $w_0w_a$CDM (68%)")
ax.axhline(-1, color="k", lw=0.6)
ax.set(xlabel=r"$z$", ylabel=r"$w(z)$", xlim=(0, 2.5), ylim=(-1.75, -0.55))
ax.plot([], [], color="0.3", lw=1.8, label="solid: true (CMB-calibrated)")
ax.plot([], [], color="0.3", lw=1.1, ls="--", label="dashed: CPL fit to RVM mocks")
ax.legend(loc="lower left", ncol=2, frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "fig_weff.pdf")
fig.savefig(FIG / "fig_weff.png", dpi=150)

# ---------- Fig 2: w0–wa plane ----------
fig, ax = plt.subplots(figsize=(5.0, 4.2))
H, xe, ye = np.histogram2d(ch[:, 3], ch[:, 4], bins=60)
Hs = H.ravel()
order = np.argsort(Hs)[::-1]
cum = np.cumsum(Hs[order]) / Hs.sum()
levels = [Hs[order][np.searchsorted(cum, q)] for q in (0.954, 0.683)]
xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
ax.contourf(xc, yc, H.T, levels=[levels[0], levels[1], H.max() + 1], colors=["#c6dbef", "#6baed6"])
ax.contour(xc, yc, H.T, levels=levels, colors="#2171b5", linewidths=0.8)
nus = np.array([r["nu"] for r in runs])
w0s = np.array([r["w0"] for r in runs])
was = np.array([r["wa"] for r in runs])
ax.plot(w0s, was, "-", color="#b2182b", lw=1.2, zorder=3)
ax.scatter(w0s, was, c="#b2182b", s=14, zorder=4)
for nu, x, y in zip(nus, w0s, was):
    if nu in (-0.003, -0.002, -0.001, 0.001, 0.002):
        ax.annotate(fr"$\nu={nu:+.3f}$", (x, y), textcoords="offset points", xytext=(5, -3), fontsize=7, color="#b2182b")
rv = np.load(HERE / "results" / "chains" / "rvm.npy")[:, 3].astype(float)
nlo, nhi = np.percentile(rv, [2.5, 97.5])
m = (nus >= nlo) & (nus <= nhi)
ax.plot(w0s[m], was[m], "-", color="#b2182b", lw=4, alpha=0.5, zorder=2,
        label=fr"RVM mocks, $\nu$ in real-data 95% range")
ax.plot([], [], "-o", color="#b2182b", ms=3, lw=1.2, label="RVM mock universes (CPL fit)")
ax.plot(-1, 0, "k+", ms=10, mew=1.5, label=r"$\Lambda$CDM")
ax.fill_between([], [], color="#6baed6", label=r"real data, $w_0w_a$CDM (68/95%)")
ax.set(xlabel=r"$w_0$", ylabel=r"$w_a$", xlim=(-1.2, -0.55), ylim=(-1.8, 1.0))
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig(FIG / "fig_w0wa.pdf")
fig.savefig(FIG / "fig_w0wa.png", dpi=150)

# ---------- Fig 3: noisy mocks ----------
recs = [json.loads(l) for f in glob.glob(str(HERE / "results" / "noisy_mocks_[abcd].jsonl")) for l in open(f)]
if recs:
    dl = np.array([r["chi2_lcdm"] - r["chi2_cpl"] for r in recs])
    dr = np.array([r["chi2_rvm"] - r["chi2_cpl"] for r in recs])
    obs_l = real["lcdm"]["chi2_min"] - real["cpl"]["chi2_min"]
    obs_r = real["rvm"]["chi2_min"] - real["cpl"]["chi2_min"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for axx, v, o, lab in ((axs[0], dl, obs_l, r"$\Delta\chi^2=\chi^2_{\Lambda{\rm CDM}}-\chi^2_{w_0w_a}$"),
                           (axs[1], dr, obs_r, r"$\Delta\chi^2=\chi^2_{\rm RVM}-\chi^2_{w_0w_a}$")):
        axx.hist(v, bins=25, color="0.6")
        axx.axvline(o, color="#b2182b", lw=1.5)
        p = (np.sum(v >= o) + 1) / (len(v) + 1)
        axx.set(xlabel=lab, ylabel="mock realisations")
        right = axx is axs[0]  # keep the label clear of the observed line
        axx.text(0.97 if right else 0.03, 0.92, f"observed {o:.1f}\n$p$ = {p:.3f} (N={len(v)})",
                 transform=axx.transAxes, ha="right" if right else "left", va="top", fontsize=8, color="#b2182b")
    fig.tight_layout()
    fig.savefig(FIG / "fig_noise.pdf")
    fig.savefig(FIG / "fig_noise.png", dpi=150)
print("figures written to", FIG)
