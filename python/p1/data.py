"""Download (pinned upstream versions) and load the public datasets: python3 -m p1.data (from python/)."""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# DESI DR2 BAO (Gaussian), as packaged for Cobaya.
DESI_COMMIT = "bb0c1c9009dc76d1391300e169e8df38fd1096db"
DESI_BASE = f"https://raw.githubusercontent.com/CobayaSampler/bao_data/{DESI_COMMIT}/desi_bao_dr2/"
DESI_FILES = ["desi_gaussian_bao_ALL_GCcomb_mean.txt", "desi_gaussian_bao_ALL_GCcomb_cov.txt"]

# Pantheon+ (Brout et al. 2022, arXiv:2202.04077).
PANTHEON_COMMIT = "c447f0fea703fcd0fff57de5000947b5ca81286b"
PANTHEON_BASE = (f"https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/{PANTHEON_COMMIT}/"
                 "Pantheon%2B_Data/4_DISTANCES_AND_COVAR/")
PANTHEON_FILES = {"Pantheon+SH0ES.dat": "Pantheon%2BSH0ES.dat",
                  "Pantheon+SH0ES_STAT+SYS.cov": "Pantheon%2BSH0ES_STAT%2BSYS.cov"}


# DES-Y5, Union3 and DES-Dovekie as packaged for Cobaya.
SNDATA_COMMIT = "61d96434cafc2770928322c38e5a750e686368ae"
SNDATA_BASE = f"https://raw.githubusercontent.com/CobayaSampler/sn_data/{SNDATA_COMMIT}/"
SNDATA_FILES = ["DESY5/DES-SN5YR_HD.csv", "DESY5/covsys_000.txt", "Union3/lcparam_full.txt",
                "Union3/mag_covmat.txt", "DES-Dovekie/DES-Dovekie_HD.csv", "DES-Dovekie/covtot_inv_000.npz"]


def _get(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {dest.name}")
    urllib.request.urlretrieve(url, dest)


def fetch_all() -> None:
    for f in DESI_FILES:
        _get(DESI_BASE + f, DATA_DIR / "desi_dr2" / f)
    for local, remote in PANTHEON_FILES.items():
        _get(PANTHEON_BASE + remote, DATA_DIR / "pantheon_plus" / local)
    for f in SNDATA_FILES:
        _get(SNDATA_BASE + f, DATA_DIR / "sn_data" / f)


@dataclass
class BaoData:
    z: np.ndarray
    value: np.ndarray
    kind: np.ndarray  # "DV_over_rs" | "DM_over_rs" | "DH_over_rs"
    cov: np.ndarray

    @property
    def inv_cov(self) -> np.ndarray:
        return np.linalg.inv(self.cov)


def load_desi_dr2() -> BaoData:
    d = DATA_DIR / "desi_dr2"
    rows = [ln.split() for ln in (d / DESI_FILES[0]).read_text().splitlines() if ln and not ln.startswith("#")]
    z = np.array([float(r[0]) for r in rows])
    val = np.array([float(r[1]) for r in rows])
    kind = np.array([r[2] for r in rows])
    cov = np.loadtxt(d / DESI_FILES[1])
    assert cov.shape == (len(z), len(z)), "DESI covariance shape mismatch"
    return BaoData(z, val, kind, cov)


@dataclass
class SnData:
    z_hd: np.ndarray       # CMB-frame redshift (distances)
    z_hel: np.ndarray      # heliocentric redshift (the (1+z) factor of D_L)
    mu_obs: np.ndarray     # distance modulus up to the nuisance M
    cov: np.ndarray | None = None
    inv_cov: np.ndarray | None = None  # set when a release ships only the inverse


def _cov_with_count(path: Path) -> np.ndarray:
    raw = np.loadtxt(path)
    n = int(raw[0])
    return raw[1:].reshape(n, n)


def load_desy5() -> SnData:
    """DES-SN5YR with C = C_sys + diag(MUERR_FINAL²), as in Cobaya/DESI."""
    d = DATA_DIR / "sn_data" / "DESY5"
    tab = np.genfromtxt(d / "DES-SN5YR_HD.csv", delimiter=",", names=True)
    cov = _cov_with_count(d / "covsys_000.txt") + np.diag(tab["MUERR_FINAL"] ** 2)
    return SnData(tab["zHD"], tab["zHEL"], tab["MU"], cov)


def load_union3() -> SnData:
    """Union3/UNITY1.5: 22 binned distance moduli with full covariance."""
    d = DATA_DIR / "sn_data" / "Union3"
    tab = np.loadtxt(d / "lcparam_full.txt", usecols=(1, 2, 4))
    return SnData(tab[:, 0], tab[:, 1], tab[:, 2], _cov_with_count(d / "mag_covmat.txt"))


def load_des_dovekie() -> SnData:
    """DES-SN5YR with the Dovekie recalibration; ships the inverse covariance."""
    d = DATA_DIR / "sn_data" / "DES-Dovekie"
    tab = np.genfromtxt(d / "DES-Dovekie_HD.csv", delimiter=",", names=True)
    z = np.load(d / "covtot_inv_000.npz")
    n = int(z["nsn"][0])
    inv = np.zeros((n, n))
    inv[np.triu_indices(n)] = z["cov"]
    inv = inv + np.triu(inv, 1).T
    assert n == len(tab["zHD"]) and tab["zHD"].min() > 0.01
    return SnData(tab["zHD"], tab["zHEL"], tab["MU"], None, inv)


def load_pantheon_plus(z_min: float = 0.01) -> SnData:
    """Pantheon+ without SH0ES calibrators, z_HD > z_min."""
    d = DATA_DIR / "pantheon_plus"
    tab = np.genfromtxt(d / "Pantheon+SH0ES.dat", names=True, dtype=None, encoding="utf-8")
    cov = _cov_with_count(d / "Pantheon+SH0ES_STAT+SYS.cov")
    idx = np.where((tab["zHD"] > z_min) & (tab["IS_CALIBRATOR"] == 0))[0]
    return SnData(tab["zHD"][idx], tab["zHEL"][idx], tab["m_b_corr"][idx], cov[np.ix_(idx, idx)])


SN_LOADERS = {"pantheon+": load_pantheon_plus, "desy5": load_desy5, "union3": load_union3, "dovekie": load_des_dovekie}


def load_sn(name: str) -> SnData:
    return SN_LOADERS[name]()


if __name__ == "__main__":
    fetch_all()
    b = load_desi_dr2()
    print(f"DESI DR2: {len(b.z)} BAO measurements, z = {b.z.min()}–{b.z.max()}")
    for name in SN_LOADERS:
        s = load_sn(name)
        print(f"{name}: {len(s.z_hd)} SNe, z = {s.z_hd.min():.3f}–{s.z_hd.max():.3f}")
