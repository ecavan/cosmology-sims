"""Real-data fits of ΛCDM, w0waCDM, the RVM and the flipped RVM (compressed CMB + DESI DR2 + SNe).

Usage: python3 scripts/flipped_fit.py [sn=pantheon+]  →  results/flipped_fit_<sn>.json
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from p1.flipped import fit_flipped, fit_simple  # noqa: E402
from p1.inference import chi2_parts, load_data  # noqa: E402
from p1.models import theory  # noqa: E402

args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
SN = args.get("sn", "pantheon+")
d = load_data(sn=SN)
out = {"sn": SN}
t0 = time.time()
for m in ("lcdm", "cpl", "rvm"):
    x, c = fit_simple(m, d)
    out[m] = {"x": x.tolist(), "chi2": c, "parts": chi2_parts(d, theory(m, x))}
    print(f"{m:8s} χ²={c:.2f}  x={x.round(5).tolist()}  ({time.time() - t0:.0f}s)", flush=True)
x, c, screen = fit_flipped(d, out["lcdm"]["x"])
out["flipped"] = {"x": x.tolist(), "chi2": c, "screen": screen}
print(f"flipped  χ²={c:.2f}  x={x.round(5).tolist()}  ({time.time() - t0:.0f}s)", flush=True)
L = out["lcdm"]["chi2"]
for m in ("cpl", "rvm", "flipped"):
    out[m]["dchi2_vs_lcdm"] = L - out[m]["chi2"]
    print(f"Δχ²(ΛCDM − {m}) = {L - out[m]['chi2']:.2f}   Δχ²({m} − CPL) = {out[m]['chi2'] - out['cpl']['chi2']:+.2f}")
(ROOT / "results" / f"flipped_fit_{SN}.json").write_text(json.dumps(out, indent=1))
