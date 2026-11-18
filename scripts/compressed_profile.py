"""RVM profile χ²(ν) with the compressed CMB, for comparison with the full-CMB profile (boltzmann/full_cmb.py).

Usage: python3 scripts/compressed_profile.py [sn=pantheon+] [nus=...]  →  results/compressed_profile[_sn].json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "python"))
from p1.inference import best_fit, load_data, profile_nu  # noqa: E402

args = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
SN = args.get("sn", "pantheon+")
SUF = "" if SN == "pantheon+" else f"_{SN}"
NUS = [float(x) for x in args.get("nus", "-0.002,-0.0015,-0.001,-0.0005,0,0.0005").split(",")]

d = load_data(sn=SN)
xl, cl = best_fit("lcdm", d)
xc, cc = best_fit("cpl", d, start=list(xl) + [-0.85, -0.6], restarts=6)
prof = profile_nu(d, xl, NUS)
for p in prof:
    print(f"ν={p['nu']:+.4f}: χ²={p['chi2']:.3f}  Δχ² vs ΛCDM={p['chi2'] - cl:+.3f}", flush=True)
out = {"sn": SN, "chi2_lcdm": cl, "chi2_cpl": cc, "profile": [{"nu": p["nu"], "chi2": p["chi2"], "params": p["x"]}
                                                                for p in prof]}
(HERE / "results" / f"compressed_profile{SUF}.json").write_text(json.dumps(out, indent=1))
print(f"ΛCDM χ²={cl:.2f}  CPL χ²={cc:.2f}  saved")
