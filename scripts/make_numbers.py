"""Write out/numbers.tex and out/sn_table.tex: a LaTeX macro for every number in the paper, from results/.
Also out/numbers_chains.tex: the Nmc* macros of the full-Planck MCMC, from results/chains_summary_*.json."""
import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
real = json.loads((HERE / "results" / "real_fits.json").read_text())
mock = json.loads((HERE / "results" / "mock_test.json").read_text())
M = {}
chain = lambda model: np.load(HERE / "results" / "chains" / f"{model}.npy").astype(float)

prior = np.load(HERE / "data" / "cmb_prior_massless.npz")
SIG_WBC = float(np.sqrt(prior["cov"][2, 2]))
M["cmbtheta"], M["cmbwb"], M["cmbwbc"] = f"{prior['mean'][0]:.5f}", f"{prior['mean'][1]:.5f}", f"{prior['mean'][2]:.5f}"
M["sigwbc"] = f"{SIG_WBC:.5f}"
M["nsn"] = "1580"


def f(x, n=3):
    return f"{x:.{n}f}"


# real-data fits
for m in ("lcdm", "cpl", "rvm"):
    if m not in real:
        continue
    r = real[m]
    M[f"chi{m}"] = f(r["chi2_min"], 1)
    M[f"Om{m}"] = f"{r['Omega_m_mean']:.4f} \\pm {r['Omega_m_std']:.4f}"
    M[f"Hz{m}"] = f"{100 * r['mean'][0]:.2f} \\pm {100 * r['std'][0]:.2f}"
cpl = real["cpl"]
M["wzero"] = f"{cpl['mean'][3]:.3f} \\pm {cpl['std'][3]:.3f}"
M["wa"] = f"{cpl['mean'][4]:.2f} \\pm {cpl['std'][4]:.2f}"
M["dchiCPL"] = f(real["lcdm"]["chi2_min"] - cpl["chi2_min"], 1)
w0b, wab = cpl["best_fit"][3], cpl["best_fit"][4]
M["wzerobf"], M["wabf"] = f(w0b, 3), f(wab, 2)
M["chicmblcdm"] = f(real["lcdm"]["chi2_parts"]["cmb"], 1)
M["chisnlcdm"] = f(real["lcdm"]["chi2_parts"]["sn"], 0)
M["aicCPL"] = f(real["lcdm"]["chi2_min"] - cpl["chi2_min"] - 4, 1)
# direction and crossing of the real-data CPL posterior (from the chain)
chc = chain("cpl")
sl = chc[:, 4] / (1 + chc[:, 3])
zx = -(1 + chc[:, 3]) / (chc[:, 4] + 1 + chc[:, 3])
zx = zx[(zx > 0) & (zx < 5)]
M["slopereal"] = f(np.median(sl), 1)
M["sloperealerr"] = f"^{{+{np.percentile(sl, 84) - np.median(sl):.1f}}}_{{-{np.median(sl) - np.percentile(sl, 16):.1f}}}"
M["sloperealninetyfive"] = f"[{np.percentile(sl, 2.5):.1f},\\,{np.percentile(sl, 97.5):.1f}]"
M["zcrossreal"] = f(np.median(zx), 2)
M["zcrossrealrange"] = f"{np.percentile(zx, 16):.2f}--{np.percentile(zx, 84):.2f}"
if "rvm" in real:
    rv = real["rvm"]
    M["chicmbrvm"] = f(rv["chi2_parts"]["cmb"], 2)
    nu = chain("rvm")[:, 3]
    M["numean"] = f"{1e3 * nu.mean():.2f} \\pm {1e3 * nu.std():.2f}"
    lo, hi = np.percentile(nu, [2.5, 97.5])
    M["nulo"], M["nuhi"] = f(1e3 * lo, 2), f(1e3 * hi, 2)
    M["nubf"] = f(1e3 * rv["best_fit"][3], 2)
    M["dchiRVM"] = f(real["lcdm"]["chi2_min"] - rv["chi2_min"], 1)
    M["dchiCPLRVM"] = f(rv["chi2_min"] - cpl["chi2_min"], 1)
    M["aicRVM"] = f(real["lcdm"]["chi2_min"] - rv["chi2_min"] - 2, 1)

# mock test
runs = {round(r["nu"], 5): r for r in mock["runs"]}
for key, nu in (("m", -0.001), ("mm", -0.002), ("p", 0.001), ("h", -0.0005)):
    r = runs[nu]
    M[f"wzero{key}"] = f(r["w0"], 3)
    M[f"wa{key}"] = f(r["wa"], 2)
    M[f"zc{key}"] = f(r["z_cross"], 2)
    M[f"dchi{key}"] = f(r["dchi2_lcdm_minus_cpl"], 1)
    M[f"chicpl{key}"] = f(r["chi2_cpl"], 1)
    M[f"pull{key}"] = f((r["omega_bc_cpl"] - r["omega_bc_true_star"]) / SIG_WBC, 1)
neg = [r for r in mock["runs"] if r["nu"] < 0]
M["zcmin"] = f(min(r["z_cross"] for r in neg), 2)
M["zcmax"] = f(max(r["z_cross"] for r in mock["runs"] if r["z_cross"] and abs(r["nu"]) > 0), 2)
TRACK_NUS = (-0.002, -0.001, -0.0005, 0.001)  # same grid as scripts/sn_robustness.py
lin = [r["wa"] / (1 + r["w0"]) for r in mock["runs"] if any(abs(r["nu"] - t) < 1e-9 for t in TRACK_NUS)]
M["slopemock"] = f(np.mean(lin), 1)
M["slopemocklo"], M["slopemockhi"] = f(min(lin), 1), f(max(lin), 1)
# expected Δχ²(ΛCDM−CPL) at the edge of the real-data 95% ν range (linear interpolation on the noiseless-mock ν grid)
if "rvm" in real:
    grid = sorted(mock["runs"], key=lambda r: r["nu"])
    nus = np.array([r["nu"] for r in grid]); dch = np.array([r["dchi2_lcdm_minus_cpl"] for r in grid])
    nu_lo = 1e-3 * float(M["nulo"])
    M["dchiedge"] = f(float(np.interp(nu_lo, nus, dch)), 1)

# noisy mocks
recs = [json.loads(l) for fn in glob.glob(str(HERE / "results" / "noisy_mocks_[abcd].jsonl")) for l in open(fn)]
if recs:
    dl = np.array([r["chi2_lcdm"] - r["chi2_cpl"] for r in recs])
    dr = np.array([r["chi2_rvm"] - r["chi2_cpl"] for r in recs])
    obs_l = real["lcdm"]["chi2_min"] - cpl["chi2_min"]
    obs_r = real["rvm"]["chi2_min"] - cpl["chi2_min"]
    M["Nnoise"] = str(len(recs))
    M["pnoiseL"] = f((np.sum(dl >= obs_l) + 1) / (len(dl) + 1), 3)
    M["pnoiseR"] = f((np.sum(dr >= obs_r) + 1) / (len(dr) + 1), 3)
    M["noiseLmed"] = f(np.median(dl), 1)
    M["noiseRmed"] = f(np.median(dr), 1)
    M["noiseRnf"] = f(np.percentile(dr, 95), 1)
    w0n, wan = np.array([r["w0"] for r in recs]), np.array([r["wa"] for r in recs])
    M["noisefracDESI"] = f(np.mean((wan <= wab) & (w0n >= w0b)), 3)
    quad = (w0n > -1) & (wan < 0)
    obs_slope = wab / (1 + w0b)
    M["noisequadN"] = str(int(quad.sum()))
    M["noiseshallow"] = f"{100 * np.mean(wan[quad] / (1 + w0n[quad]) >= obs_slope):.0f}"
    M["noiseslopemed"] = f(np.median(wan[quad] / (1 + w0n[quad])), 1)
    from scipy.stats import norm
    pr = (np.sum(dr >= obs_r) + 1) / (len(dr) + 1)
    M["pnoiseRerr"] = f(np.sqrt(pr * (1 - pr) / len(dr)), 3)
    M["pnoiseRsigone"] = f(norm.isf(pr), 1)
    M["pnoiseRsigtwo"] = f(norm.isf(pr / 2), 1)
    pl = (np.sum(dl >= obs_l) + 1) / (len(dl) + 1)
    M["pnoiseLsigone"] = f(norm.isf(pl), 1)
    M["noisenustd"] = f(1e3 * np.std([r["nu"] for r in recs]), 2)
    # limitation (iii): the same p-value with mocks drawn from other truths (scripts/noisy_mocks.py truth=...)
    for key, tags in (("Tzero", ("l0a", "l0b")), ("Tmone", ("m1a", "m1b"))):
        files = [HERE / "results" / f"noisy_mocks_{t}.jsonl" for t in tags]
        if all(fn.exists() for fn in files):
            dt = np.array([r["chi2_rvm"] - r["chi2_cpl"] for fn in files for r in map(json.loads, open(fn))])
            pt = (np.sum(dt >= obs_r) + 1) / (len(dt) + 1)
            M[f"pnoise{key}"] = f(pt, 3)
            M[f"pnoise{key}Err"] = f(np.sqrt(pt * (1 - pt) / len(dt)), 3)
            M[f"Nnoise{key}"] = str(len(dt))
        else:
            M[f"pnoise{key}"] = M[f"pnoise{key}Err"] = M[f"Nnoise{key}"] = "\\textbf{??}"
else:  # visible placeholders until the noisy mocks exist
    for k in ("Nnoise", "pnoiseL", "pnoiseR", "noiseLmed", "noiseRmed", "noiseRnf", "noisefracDESI"):
        M[k] = "\\textbf{??}"

# full CMB (boltzmann/full_cmb.py): best (lowest) χ² per configuration; PR4 lensing is the main run, 2018 a cross-check
def parabola(points):
    """Fit Δχ²(ν) = a(ν−ν0)² + c through profile points; return ν0, 1σ, minimum, n points, max residual."""
    x = np.array([p[0] for p in points]); y = np.array([p[1] for p in points])
    a2, a1, a0 = np.polyfit(x, y, 2)
    return -a1 / (2 * a2), 1 / np.sqrt(a2), a0 - a1 ** 2 / (4 * a2), len(points), np.abs(y - np.polyval([a2, a1, a0], x)).max()


def best_records(path):
    """Lowest-χ² record per (model, ν, rad, switches)."""
    best = {}
    for l in open(path):
        r = json.loads(l)
        key = (r["model"], round(r["nu"], 6), r["rad"], r.get("z1"), r.get("z2"), r.get("free_z", False))
        if key not in best or r["chi2"] < best[key]["chi2"]:
            best[key] = r
    return best


def full_cmb(best, pre, numax):
    """Macros for the ΛCDM/w0wa/RVM full-CMB comparison, profile fitted over −numax ≤ 10³ν ≤ 0."""
    c_l, c_c = best[("lcdm", 0.0, 1, None, None, False)]["chi2"], best[("cpl", 0.0, 1, None, None, False)]["chi2"]
    cpl_r = best[("cpl", 0.0, 1, None, None, False)]
    M[pre + "dchiCPL"] = f(c_l - c_c, 1)
    M[pre + "aicCPL"] = f(c_l - c_c - 4, 1)
    M[pre + "wzero"], M[pre + "wa"] = f(cpl_r["slow"]["w0_fld"], 3), f(cpl_r["slow"]["wa_fld"], 2)
    M[pre + "Hzero"] = f(best[("lcdm", 0.0, 1, None, None, False)]["slow"]["H0"], 2)
    out = {}
    for rad, sfx in ((1, ""), (0, "R")):
        pts = [(0.0, 0.0)] + [(1e3 * k[1], v["chi2"] - c_l) for k, v in best.items()
                              if k[0] == "rvm" and k[2] == rad and -numax - 1e-9 <= 1e3 * k[1] <= 0]
        if len(pts) < 3:
            continue
        nu0, sig, dmin, n, res = parabola(pts)
        M[pre + "nu" + sfx] = f"{nu0:.2f} \\pm {sig:.2f}"
        M[pre + "dchiRVM" + sfx] = f(-dmin, 1)
        M[pre + "dchiCPLRVM" + sfx] = f(dmin + (c_l - c_c), 1)
        M[pre + "aicRVM" + sfx] = f(-dmin - 2, 1)
        M[pre + "npts" + sfx] = str(n)
        M[pre + "res" + sfx] = f(res, 2)
        out[rad] = pts
        print(f"{pre} rad={rad} profile:", [(round(a, 2), round(b, 2)) for a, b in sorted(pts)])
    return c_l, c_c, out


def flipped(ff, pre, c_l, c_c):
    """Flipped RVM, full Planck: profile in ν at fixed switches (prefix fc → fcfl…, fca → fcafl…)."""
    fb = best_records(ff)
    # A: our best switches; B: the w0wa-matching window; M: the posterior-mean switches of Solà et al.
    for (z1, z2), sfx, lo, hi in (((1.22, 2.16), "A", 0.03, 0.12), ((1.1, 1.8), "B", 0.06, 0.12),
                                  ((2.58, 6.1), "M", 0.0, 0.03)):
        pts = [(k[1], v["chi2"] - c_l) for k, v in fb.items()
               if k[3] == z1 and k[4] == z2 and not k[5] and lo - 1e-9 <= k[1] <= hi + 1e-9]
        if lo == 0.0:
            pts = [(0.0, 0.0)] + pts  # ν = 0 is ΛCDM whatever the switches
        if len(pts) < 3:
            continue
        nu0, sig, dmin, n, res = parabola(pts)
        M[pre + "fl" + sfx + "nu"] = f"{nu0:.3f} \\pm {sig:.3f}"
        M[pre + "fl" + sfx + "nuval"] = f(nu0, 2)
        M[pre + "fl" + sfx + "dchi"] = f(-dmin, 1)
        M[pre + "fl" + sfx + "gap"] = f(-(dmin + (c_l - c_c)), 1)  # >0: flipped beats w0wa
        M[pre + "fl" + sfx + "res"] = f(res, 2)
        if sfx == "M":
            M[pre + "flMcpl"] = f(dmin + (c_l - c_c), 1)  # how far w0wa is ahead at these switches
            a2, a1, a0 = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 2)
            M[pre + "flMSola"] = f(np.polyval([a2, a1, a0], 0.029) - dmin, 1)
        if sfx == "A":
            a2, a1, a0 = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 2)
            M[pre + "flSola"] = f(np.polyval([a2, a1, a0], 0.029) - dmin, 1)
            M[pre + "flSolasig"] = f(np.sqrt(np.polyval([a2, a1, a0], 0.029) - dmin), 1)
        print(f"flipped full CMB ({z1},{z2}):", [(round(a, 3), round(b, 2)) for a, b in sorted(pts)])
    far = [v["chi2"] - c_l for k, v in fb.items() if (k[3], k[4]) == (1.1, 1.8) and round(k[1], 3) == 0.18]
    if far:
        M[pre + "flBfar"] = f(far[0], 0)
    free = [v for k, v in fb.items() if k[5]]
    if free:
        M[pre + "flfreezone"], M[pre + "flfreeztwo"] = f(free[0]["slow"]["rvm_flip_z1"], 2), f(free[0]["slow"]["rvm_flip_z2"], 2)



fc_new, fc_old = HERE / "results" / "full_cmb_pr4.jsonl", HERE / "results" / "full_cmb.jsonl"
if fc_old.exists():
    full_cmb(best_records(fc_old), "fco", 1.5)  # 2018 lensing: four points, 10³ν = −1.5 … 0
if fc_new.exists():
    best = best_records(fc_new)
    c_l, c_c, prof = full_cmb(best, "fc", 1.0)  # PR4 lensing: the profile near its minimum
    M["fcnptsall"] = str(len([k for k in best if k[0] == "rvm" and k[2] == 1]) + 1)
    pos = [v["chi2"] - c_l for k, v in best.items() if k[0] == "rvm" and k[2] == 0 and k[1] > 0]
    if pos:
        M["fcdchiposR"] = f(pos[0], 0)
    far = {round(1e3 * k[1], 2): v["chi2"] - c_l for k, v in best.items() if k[0] == "rvm" and k[2] == 1}
    M["fcdchimtwo"] = f(far.get(-2.0, float("nan")), 0)
    allr = [json.loads(l) for l in open(fc_new)]
    runs = lambda m, nu, rad: sorted(r["chi2"] for r in allr if r["model"] == m and round(r["nu"], 6) == nu and r["rad"] == rad)
    keys = [("lcdm", 0.0, 1), ("cpl", 0.0, 1)] + [("rvm", round(1e-3 * a, 6), rad) for rad in (0, 1) for a, _ in prof.get(rad, []) if a != 0]
    M["fcrepeat"] = f(max(runs(*k)[1] - runs(*k)[0] for k in keys), 1)  # worst agreement of the two best runs
    M["fclcdmspread"] = f(runs("lcdm", 0.0, 1)[2] - runs("lcdm", 0.0, 1)[0], 2)
    M["fcdrop"] = f(max(r[-1] - r[0] for r in (runs(*k) for k in keys) if r[-1] - r[0] < 3), 1)  # excludes stalled first passes
    four = [(0.0, 0.0)] + [(1e3 * k[1], v["chi2"] - c_l) for k, v in best.items()
                           if k[0] == "rvm" and k[2] == 1 and round(1e3 * k[1], 2) in (-0.5, -1.0, -1.5)]
    M["fcgapfour"] = f(parabola(four)[2] + (c_l - c_c), 1)
    if recs:  # indicative: the compressed-prior mocks read at the full-CMB gap
        pf = (np.sum(dr >= float(M["fcdchiCPLRVM"])) + 1) / (len(dr) + 1)
        M["pnoiseRfc"] = f(pf, 2)
        M["pnoiseRfcsig"] = f(norm.isf(pf), 1)
    # ACT DR6 + Planck PR4 lensing (DESI DR2's CMB lensing combination): the same fits, prefix "fca"
    fc_act = HERE / "results" / "full_cmb_actplanck.jsonl"
    if fc_act.exists():
        ba_best = best_records(fc_act)
        ca_l, ca_c, aprof = full_cmb(ba_best, "fca", 1.0)
        ba = {k[0]: v for k, v in ba_best.items() if k[0] in ("lcdm", "cpl")}
        pa = lambda m, s: sum(v for k, v in ba[m]["parts"].items() if s in k)
        M["fcadlens"] = f(pa("lcdm", "lens") - pa("cpl", "lens"), 1)
        alla = [json.loads(l) for l in open(fc_act)]
        arun = lambda m, nu, rad: sorted(r["chi2"] for r in alla if r["model"] == m and round(r["nu"], 6) == nu and r["rad"] == rad)
        akeys = [("lcdm", 0.0, 1), ("cpl", 0.0, 1)] + [("rvm", round(1e-3 * a, 6), rad) for rad in (0, 1) for a, _ in aprof.get(rad, []) if a != 0]
        M["fcarepeat"] = f(max(arun(*k)[1] - arun(*k)[0] for k in akeys), 1)  # two best runs of each fit
        fl_act = HERE / "results" / "full_cmb_flipped_actplanck.jsonl"
        if fl_act.exists():  # the same for the flipped-RVM points (two runs each)
            grp = {}
            for l in open(fl_act):
                r = json.loads(l)
                grp.setdefault((round(r["nu"], 6), r.get("z1"), r.get("z2")), []).append(r["chi2"])
            M["fcaflrepeat"] = f(max(sorted(v)[1] - sorted(v)[0] for v in grp.values() if len(v) > 1), 1)
        if recs:
            pfa = (np.sum(dr >= float(M["fcadchiCPLRVM"])) + 1) / (len(dr) + 1)
            M["pnoiseRfca"] = f(pfa, 2)
            M["pnoiseRfcasig"] = f(norm.isf(pfa), 1)
        print("ACT+PR4 lensing: dchi2(LCDM-w0wa) =", M["fcadchiCPL"], "lensing part", M["fcadlens"],
              "RVM-w0wa", M["fcadchiCPLRVM"], "repeat", M["fcarepeat"])
    # same window from the compressed-CMB profile, for a like-for-like comparison
    cp = json.loads((HERE / "results" / "compressed_profile.json").read_text())
    used = {round(p[0], 3) for p in prof[1]}
    cpts = [(1e3 * q["nu"], q["chi2"] - cp["chi2_lcdm"]) for q in cp["profile"] if round(1e3 * q["nu"], 3) in used]
    cnu0, csig, cdmin, _, _ = parabola(cpts)
    M["cpnu"] = f"{cnu0:.2f} \\pm {csig:.2f}"
    M["cpdchiRVM"] = f(-cdmin, 1)
    M["cpdchiCPLRVM"] = f(cdmin + cp["chi2_lcdm"] - cp["chi2_cpl"], 1)
    # flipped RVM, full Planck, profile in ν at fixed switches
    if (HERE / "results" / "full_cmb_flipped.jsonl").exists():
        flipped(HERE / "results" / "full_cmb_flipped.jsonl", "fc", c_l, c_c)
    if fc_act.exists() and (HERE / "results" / "full_cmb_flipped_actplanck.jsonl").exists():
        flipped(HERE / "results" / "full_cmb_flipped_actplanck.jsonl", "fca", ca_l, ca_c)

# SN-sample robustness (scripts/sn_robustness.py) → out/sn_table.tex
SN_LABEL = {"pantheon+": "Pantheon+", "desy5": "DES-Y5", "union3": "Union3", "dovekie": "DES-Dovekie"}
rows = []
for key, label in SN_LABEL.items():
    fn = HERE / "results" / f"sn_robustness_{key}.json"
    if not fn.exists():
        continue
    r = json.loads(fn.read_text())
    q = np.percentile(sl, [2.5, 16, 50, 84, 97.5]).tolist() if key == "pantheon+" else r["slope_q"]  # main chain for Pantheon+, as in the text
    mn = np.array([0.0] + [m["nu"] for m in r["mocks"]]); md = np.array([0.0] + [m["dchi2_lcdm_minus_cpl"] for m in r["mocks"]])
    o = np.argsort(mn)
    r["mock_dchi2_at_best_nu"] = float(np.interp(1e-3 * r["nu_1e3"][0], mn[o], md[o]))
    rows.append(f"{label} & {r['lcdm']['chi2'] - r['cpl']['chi2']:.1f} & ${r['nu_1e3'][0]:.2f}\\pm{r['nu_1e3'][1]:.2f}$ & "
                f"{r['dchi2_lcdm_minus_rvm']:.1f} & {r['dchi2_rvm_minus_cpl']:.1f} & "
                f"${q[2]:.1f}$ $[{q[0]:.1f},{q[4]:.1f}]$ & ${r['mock_slope'][0]:.1f}$ & "
                f"{r['mock_dchi2_at_best_nu']:.1f} \\\\")
    if key != "pantheon+":
        M[f"sn{''.join(c for c in key if c.isalpha())}gap"] = f(r["dchi2_rvm_minus_cpl"], 1)
if rows:
    head = ("\\begin{tabular}{lccccccc}\n\\toprule\n"
            "SN sample & $\\Delta\\chi^2_{\\Lambda-w_0w_a}$ & $10^3\\nu$ & $\\Delta\\chi^2_{\\Lambda-{\\rm RVM}}$ &\n"
            "$\\Delta\\chi^2_{{\\rm RVM}-w_0w_a}$ & $w_a/(1+w_0)$ & RVM track & mock $\\Delta\\chi^2_{\\Lambda-w_0w_a}$ \\\\\n"
            "\\midrule\n")
    (OUT / "sn_table.tex").write_text("% generated by scripts/make_numbers.py\n" + head + "\n".join(rows)
                                                 + "\n\\bottomrule\n\\end{tabular}\n")
nm = [json.loads(l) for fn in glob.glob(str(HERE / "results" / "noisy_mocks_desy5_*.jsonl")) for l in open(fn)]
if nm:
    rob = json.loads((HERE / "results" / "sn_robustness_desy5.json").read_text())
    dr = np.array([r["chi2_rvm"] - r["chi2_cpl"] for r in nm])
    obs = rob["dchi2_rvm_minus_cpl"]
    M["Nnoisedes"] = str(len(nm))
    M["noisedesmax"] = f(dr.max(), 1)
    M["noisedesnf"] = f(np.percentile(dr, 95), 1)
    k = int(np.sum(dr >= obs))
    M["noisedesk"] = str(k)
    M["pnoisedes"] = f"\\le 1/{len(nm) + 1}" if k == 0 else "=" + f((k + 1) / (len(nm) + 1), 3)

s8f = HERE / "results" / "s8_bestfits.json"  # boltzmann/s8_at_bestfit.py: S8 and today's Ω_m at each best fit
if s8f.exists():
    s8 = json.loads(s8f.read_text())
    one = lambda row, lens: next(r for r in s8 if r["row"] == row and r["lens"] == lens)  # noqa: E731
    M["sEightLCDM"] = f(one("LCDM", "pr4")["S8"], 2)
    M["sEightRVM"] = f(one("RVM", "pr4")["S8"], 2)
    fl = [r for r in s8 if r["row"] in ("flipped (1.22, 2.16)", "flipped (1.1, 1.8)")]  # both lensing choices
    assert len(fl) == 4, fl
    M["sEightFlLo"], M["sEightFlHi"] = f(min(r["S8"] for r in fl), 2), f(max(r["S8"] for r in fl), 2)
    M["OmFlLo"], M["OmFlHi"] = f(min(r["Omega_m"] for r in fl), 2), f(max(r["Omega_m"] for r in fl), 2)

lines = ["% generated by scripts/make_numbers.py — do not edit"]
for k, v in M.items():
    name = "".join(c for c in k if c.isalpha())
    lines.append(f"\\newcommand{{\\N{name}}}{{{v}}}")
(OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
print(f"wrote {len(M)} macros")
for k, v in M.items():
    print(f"  {k:14s} {v}")

# Full-Planck MCMC (chains_v2): Nmc* macros in their own file, out/numbers_chains.tex, from
# results/chains_summary_{lcdm,rvm,flipped}.json. Folded in from make_numbers_chains.py (Fable, 18 Nov); the output,
# header line included, is byte-identical to that script's.
_cs = {m: HERE / "results" / f"chains_summary_{m}.json" for m in ("lcdm", "rvm", "flipped")}
if all(p.exists() for p in _cs.values()):
    S = {m: json.loads(p.read_text()) for m, p in _cs.items()}

    def _pm(p, d, k=1.0):
        return f"{k * p['mean']:.{d}f} \\pm {k * p['sd']:.{d}f}"

    def _rule(m):  # lcdm: the D-029 (0.03) assessment stored alongside its original 0.02 "rule"
        return S[m].get("rule_D029") or S[m]["rule"]

    MC = {}
    for m, t in (("lcdm", "L"), ("rvm", "R"), ("flipped", "F")):
        P = S[m]["params"]
        MC[f"mcSeight{t}"] = _pm(P["S8"], 3)
        MC[f"mcOm{t}"] = _pm(P["Omega_m"], 3)
        MC[f"mcHz{t}"] = _pm(P["H0"], 2)
    MC["mcNuR"] = _pm(S["rvm"]["params"]["rvm_nu"], 2, 1e3)
    MC["mcNuF"] = _pm(S["flipped"]["params"]["rvm_nu"], 3)
    MC["mcEdge"] = f"{100 * S['rvm']['frac_nu_near_edge']:.1f}"
    MC["mcRcosmo"] = f"{max(_rule(m)['R_minus_1_max_cosmo'] for m in S):.3f}"
    MC["mcRall"] = f"{max(_rule(m)['R_minus_1_max_sampled'] for m in S):.3f}"
    ess = [S[m]["params"][p]["ess"] for m in S for p in ("S8", "Omega_m", "H0") + (("rvm_nu",) if m != "lcdm" else ())]
    MC["mcEssMin"] = str(int(min(ess)))
    MC["mcRowsMin"] = str(min(min(S[m]["n_samples"]) for m in S))
    (OUT / "numbers_chains.tex").write_text(
        "% generated by make_numbers_chains.py from results/chains_summary_*.json - do not edit\n"
        + "".join(f"\\newcommand{{\\N{k}}}{{{v}}}\n" for k, v in MC.items()))
    print(f"wrote {len(MC)} chain macros to out/numbers_chains.tex")
    for k, v in MC.items():
        print(f"  {k:14s} {v}")
