"""Repeat the compressed-CMB analysis with another SN sample (desy5, union3, dovekie or pantheon+).

Fits ΛCDM and w0waCDM (plus a short chain for w_a/(1+w_0)), profiles the RVM in ν, and fits w0waCDM to noiseless
RVM mocks about the profile minimum. Usage: python3 scripts/sn_robustness.py sn=desy5  →  results/sn_robustness_<sn>.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "python"))
from p1.inference import best_fit, load_data, profile_nu, sample  # noqa: E402
from p1.mocks import asimov  # noqa: E402
from p1.models import theory  # noqa: E402

args = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
SN = args.get("sn", "desy5")
NUS = [float(x) for x in args.get("nus", "-0.0015,-0.001,-0.0005,0,0.0005").split(",")]
MOCK_NUS = [-0.002, -0.001, -0.0005, 0.001]
t0 = time.time()
d = load_data(sn=SN)
out = {"sn": SN, "n_sn": int(len(d.sn.d.z_hd))}


def log(msg):
    print(f"[{SN} {time.time() - t0:5.0f}s] {msg}", flush=True)


# 1. ΛCDM and w0wa
xl, cl = best_fit("lcdm", d)
xc, cc = best_fit("cpl", d, start=list(xl) + [-0.85, -0.6], restarts=6)
chain, _, _ = sample("cpl", d, nwalkers=24, nsteps=1500, burn=500, start=xc)
slope = chain[:, 4] / (1 + chain[:, 3])
out.update({"lcdm": {"x": xl.tolist(), "chi2": cl}, "cpl": {"x": xc.tolist(), "chi2": cc},
            "cpl_w0": [float(chain[:, 3].mean()), float(chain[:, 3].std())],
            "cpl_wa": [float(chain[:, 4].mean()), float(chain[:, 4].std())],
            "slope_q": np.percentile(slope, [2.5, 16, 50, 84, 97.5]).tolist()})
ax = 1 + (1 + chain[:, 3]) / chain[:, 4]  # crossing scale factor of each sample
zx = 1 / ax - 1
ok = (ax > 1 / 6) & (ax < 1)
out["zcross_q"] = np.percentile(zx[ok], [16, 50, 84]).tolist()
log(f"ΛCDM χ²={cl:.2f}  w0wa χ²={cc:.2f}  Δχ²={cl - cc:.2f}  w0={xc[3]:.3f} wa={xc[4]:.3f}  "
    f"slope={np.median(slope):.2f} [{out['slope_q'][0]:.2f},{out['slope_q'][4]:.2f}]")

# 2. RVM profile, fitted with a parabola
prof = sorted(profile_nu(d, xl, sorted(NUS, key=abs)), key=lambda p: p["nu"])
x = 1e3 * np.array([p["nu"] for p in prof]); y = np.array([p["chi2"] for p in prof]) - cl
a2, a1, a0 = np.polyfit(x, y, 2)
nu0, sig, dmin = -a1 / (2 * a2), 1 / np.sqrt(a2), a0 - a1 ** 2 / (4 * a2)
out.update({"profile": prof, "nu_1e3": [nu0, sig], "dchi2_lcdm_minus_rvm": -dmin,
            "dchi2_rvm_minus_cpl": dmin + cl - cc})
log(f"RVM ν=({nu0:.2f}±{sig:.2f})e-3  Δχ²(ΛCDM−RVM)={-dmin:.2f}  Δχ²(RVM−w0wa)={dmin + cl - cc:.2f}")

# 3. noiseless mocks about the profile minimum
pmin = min(prof, key=lambda p: p["chi2"])["x"]
runs = []
for nu in MOCK_NUS:
    m = asimov(d, theory("rvm", list(pmin) + [nu]))
    ml, mcl = best_fit("lcdm", m, start=pmin)
    mc, mcc = best_fit("cpl", m, start=list(pmin) + [-1.0, 0.0], restarts=6)
    w0, wa = mc[3], mc[4]
    a_x = 1 + (1 + w0) / wa if wa else np.nan
    runs.append({"nu": nu, "w0": w0, "wa": wa, "slope": wa / (1 + w0), "z_cross": 1 / a_x - 1,
                 "dchi2_lcdm_minus_cpl": mcl - mcc, "chi2_cpl": mcc})
    log(f"mock ν={nu:+.4f}: w0={w0:.3f} wa={wa:.3f} slope={wa / (1 + w0):.2f} z×={1 / a_x - 1:.3f} "
        f"Δχ²={mcl - mcc:.2f}")
out["mocks"] = runs
lin = [r["slope"] for r in runs if abs(r["nu"]) <= 0.002]
out["mock_slope"] = [float(np.mean(lin)), float(min(lin)), float(max(lin))]
nus_m = np.array([r["nu"] for r in runs]); dch = np.array([r["dchi2_lcdm_minus_cpl"] for r in runs])
order = np.argsort(nus_m)
out["mock_dchi2_at_best_nu"] = float(np.interp(nu0 * 1e-3, nus_m[order], dch[order]))
(HERE / "results" / f"sn_robustness_{SN}.json").write_text(json.dumps(out, indent=1))
log(f"saved; expected Δχ²(ΛCDM−w0wa) at best ν = {out['mock_dchi2_at_best_nu']:.2f}")
