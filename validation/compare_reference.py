"""Compare cosmo-bg distances with astropy and CLASS for flat ΛCDM and CPL; exits non-zero on a mismatch."""
import io
import subprocess
import sys

import numpy as np
from astropy.cosmology import Flatw0waCDM
import astropy.units as u
from classy import Class

H, OMBH2, OMCH2, NEFF, TCMB = 0.6736, 0.02237, 0.1200, 3.044, 2.7255
ZS = [0.01, 0.1, 0.3, 0.51, 0.706, 0.934, 1.321, 1.484, 2.33, 5.0, 10.0, 100.0, 1089.9]
TOL_ASTROPY = 2e-6   # same physics, quadrature only
TOL_CLASS = 5e-5     # well below DESI errors (~0.3%)


def rust_table(**kw):
    args = [f"{k}={v}" for k, v in kw.items()] + ["zs=" + ",".join(map(str, ZS))]
    out = subprocess.run(["cargo", "run", "--release", "-q", "--bin", "bg_table", "--", *args],
                         check=True, capture_output=True, text=True).stdout
    header = out.splitlines()[0]
    meta = dict(kv.split("=") for kv in header.lstrip("# ").split())
    data = np.genfromtxt(io.StringIO("\n".join(out.splitlines()[1:])), delimiter=",", names=True)
    return data, {k: float(v) for k, v in meta.items()}


def run_case(name, w0, wa, mnu=0.0):
    model = "lcdm" if (w0, wa) == (-1.0, 0.0) else "cpl"
    data, meta = rust_table(model=model, h=H, ombh2=OMBH2, omch2=OMCH2, neff=NEFF, w0=w0, wa=wa, mnu=mnu)
    ok = True

    ap = Flatw0waCDM(H0=100 * H, Om0=(OMBH2 + OMCH2) / H**2, Ob0=OMBH2 / H**2, w0=w0, wa=wa,
                     Tcmb0=TCMB * u.K, Neff=NEFF, m_nu=0 * u.eV)
    z = np.array(ZS)
    ref = {"D_M": ap.comoving_transverse_distance(z).value, "D_H": (299792.458 / ap.H(z).value),
           "D_L": ap.luminosity_distance(z).value}

    cl = Class()
    pars = {"h": H, "omega_b": OMBH2, "omega_cdm": OMCH2, "N_ur": NEFF, "N_ncdm": 0, "T_cmb": TCMB,
            "output": ""}
    if mnu > 0:
        pars.update({"N_ur": NEFF - 1.01317, "N_ncdm": 1, "m_ncdm": mnu})
    if model == "cpl":
        pars.update({"Omega_Lambda": 0, "w0_fld": w0, "wa_fld": wa, "use_ppf": "yes"})
    cl.set(pars)
    cl.compute()
    cref = {"D_M": np.array([cl.angular_distance(zz) * (1 + zz) for zz in z]),
            "D_H": np.array([1 / cl.Hubble(zz) for zz in z]),
            "D_L": np.array([cl.luminosity_distance(zz) for zz in z])}
    bg = cl.get_current_derived_parameters(["rs_d", "z_d"])
    cl.struct_cleanup()

    print(f"\n== {name} ==")
    for key in ("D_M", "D_H", "D_L"):
        ea = np.max(np.abs(data[key] / ref[key] - 1)) if mnu == 0 else float("nan")  # astropy's massive ν differs
        ec = np.max(np.abs(data[key] / cref[key] - 1))
        flag = "OK " if ((mnu > 0 or ea < TOL_ASTROPY) and ec < TOL_CLASS) else "BAD"
        ok &= flag == "OK "
        print(f"  {flag} {key}: max |rel diff| astropy {ea:.2e}, CLASS {ec:.2e}")
    print(f"  r_d: CLASS {bg['rs_d']:.4f} Mpc (z_d {bg['z_d']:.2f}) | integral@EH98 {meta['rd_integral']:.4f}"
          f" (z_d {meta['zd_eh98']:.2f})")
    print(f"  calibration factor CLASS/integral = {bg['rs_d'] / meta['rd_integral']:.6f}")
    return ok


if __name__ == "__main__":
    good = run_case("LCDM (Planck18, massless nu)", -1.0, 0.0)
    good &= run_case("CPL w0=-0.75 wa=-0.85 (DESI-like)", -0.75, -0.85)
    good &= run_case("LCDM with one 0.06 eV neutrino (CLASS only)", -1.0, 0.0, mnu=0.06)
    good &= run_case("CPL w0=-0.75 wa=-0.85 with 0.06 eV neutrino (CLASS only)", -0.75, -0.85, mnu=0.06)
    print("\nALL OK" if good else "\nFAILURES")
    sys.exit(0 if good else 1)
