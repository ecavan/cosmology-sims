"""Run the full-CMB MCMC chains of several models with a fixed number of worker slots, check convergence, and stop
each model `post_hours` after it converges, or at the hard stop `stop_at`, whichever comes first. Restartable: rerun
the same command and it resumes every unfinished chain.

Priority order: lcdm, rvm, flipped (1.22, 2.16); 4 chains each (cpl dropped 15 Nov; still runnable with models=...,cpl), one process (one thread) per chain.
Every `check` seconds, for each model with chains on disk it computes R−1 across chains (boltzmann/mcmc_summary.py,
30% burn-in). Convergence rule (Fable, 16 Nov; threshold 0.03 since 17 Nov, D-029): max R−1 < `rtol` over the SAMPLED parameters (names from the chain's
learned covmat), at least `min_rows` rows per chain after burn-in, and getdist ESS >= `min_ess` for every quantity the
paper quotes (S8, Omega_m, and ν for rvm/flipped; w0, wa for cpl). A model is converged when the rule holds at a
check and again at a check at least `min_gap` hours later, with no failing check in between (the first passing time
is kept in <out>/<model>/rule_state.json, so it survives a queue restart). On convergence <out>/<model>/CONVERGED,
results/chains_summary_<model>.json and out/triangle_<model>.pdf are written and rewritten at every later check.
Stop (Fable, 16 Nov 12:10): `post_hours` (6) after convergence, or at `stop_at` (local time, default
2026-11-19 09:00) converged or not, the final summary and triangle are written from what exists (summary key
"final": reason and time; "rule" records whether it converged), the model's chains are stopped (SIGTERM) and
<out>/<model>/DONE is written. Freed slots stay empty. A model whose directory has a DONE file is never launched.
A chain that exits before its model is DONE is restarted (cobaya resumes it).
Usage: python3 boltzmann/mcmc_queue.py [jobs=8] [check=600] [rtol=0.03] [min_rows=500] [min_ess=400] [min_gap=3]
       [post_hours=6] [stop_at="2026-11-19 09:00"] [stop_label="hard stop"] [models=lcdm,rvm,flipped] [out=chains_v2]
Status: <out>/queue_status.json (rewritten every check) and <out>/queue.log, which also records every chain (re)start
with the number of rows already on disk. chains/ (the first, seeded run) is a record only and is not read.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "boltzmann"))
import mcmc_summary  # noqa: E402

args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
JOBS = int(args.get("jobs", 8))
CHECK = int(args.get("check", 600))
RTOL = float(args.get("rtol", 0.03))  # D-029 (17 Nov): 0.02 -> 0.03
MIN_ROWS = int(args.get("min_rows", 500))
MIN_ESS = float(args.get("min_ess", 400))
MIN_GAP = float(args.get("min_gap", 3)) * 3600
POST = float(args.get("post_hours", 6)) * 3600
STOP_AT_STR = args.get("stop_at", "2026-11-19 09:00")
STOP_LABEL = args.get("stop_label", "hard stop")  # wording of the final reason at stop_at
STOP_AT = time.mktime(time.strptime(STOP_AT_STR, "%Y-%m-%d %H:%M"))  # local time of the machine
COSMO = ["H0", "omega_b", "omega_cdm", "logA", "n_s", "tau_reio", "rvm_nu", "w0_fld", "wa_fld"]
ESS_PARAMS = {"lcdm": ["S8", "Omega_m"], "rvm": ["S8", "Omega_m", "rvm_nu"], "flipped": ["S8", "Omega_m", "rvm_nu"],
              "cpl": ["S8", "Omega_m", "w0_fld", "wa_fld"]}
NCHAIN = 4
ALL = {"lcdm": ("lcdm", []), "rvm": ("rvm", []), "flipped": ("flipped_1.22_2.16", ["z1=1.22", "z2=2.16"]),
       "cpl": ("cpl", [])}
MODELS = [m for m in args.get("models", "lcdm,rvm,flipped").split(",")]
OUT = args.get("out", "chains_v2")
CHAINS = HERE / OUT
CHAINS.mkdir(exist_ok=True)
LOG = CHAINS / "queue.log"
procs: dict[tuple[str, int], subprocess.Popen] = {}


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def done(m: str) -> bool:
    return (CHAINS / ALL[m][0] / "DONE").exists()


def launch(m: str, k: int) -> None:
    d = CHAINS / ALL[m][0]
    (d / "logs").mkdir(parents=True, exist_ok=True)
    out = open(d / "logs" / f"c{k}.log", "a")
    cmd = [sys.executable, str(HERE / "boltzmann" / "mcmc_full.py"), f"model={m}", f"chain={k}", f"out={OUT}", *ALL[m][1]]
    f = d / f"c{k}" / "chain.1.txt"
    rows = sum(1 for _ in open(f)) - 1 if f.exists() else 0
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    procs[(m, k)] = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=HERE,
                                     start_new_session=True)
    log(f"{'restarted' if rows else 'started'} {m} chain {k} (pid {procs[(m, k)].pid}), rows on disk {rows}")


def stop(m: str) -> None:
    for (mm, k), p in list(procs.items()):
        if mm == m:
            p.send_signal(signal.SIGTERM)
            p.wait()  # SIGTERM and wait; never SIGKILL (a truncated last chain line is skipped on reading)
            del procs[(mm, k)]


def check(m: str) -> dict | None:
    chains = mcmc_summary.load(CHAINS / ALL[m][0])
    if len(chains) < NCHAIN or min(a.shape[0] for _, a in chains) < 20:
        return None
    s, gd, params = mcmc_summary.summarise(chains, m)
    s["chain_dir"] = f"{OUT}/{ALL[m][0]}"
    return {"summary": s, "gd": gd, "params": params}


def sampled(m: str) -> list[str]:
    """Sampled parameter names: header of the chain's learned covmat (written by Cobaya, sampled parameters only)."""
    return open(CHAINS / ALL[m][0] / "c1" / "chain.covmat").readline().lstrip("#").split()


def rule_state(m: str) -> dict:
    f = CHAINS / ALL[m][0] / "rule_state.json"
    return json.loads(f.read_text()) if f.exists() else {"first_pass": None, "converged": None}


def write_results(m: str, res: dict) -> None:
    """Summary JSON + triangle plot."""
    s = res["summary"]
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"chains_summary_{m}.json").write_text(json.dumps(s, indent=1) + "\n")
    try:
        from getdist import plots

        show = [p for p in ("rvm_nu", "w0_fld", "wa_fld", "Omega_m", "H0", "S8") if p in res["params"]]
        g = plots.get_subplot_plotter()
        g.triangle_plot(res["gd"], show, filled=True)
        (HERE / "out").mkdir(exist_ok=True)
        g.export(str(HERE / "out" / f"triangle_{m}.pdf"))
    except Exception as e:  # noqa: BLE001
        log(f"{m}: triangle plot failed: {e}")
    finally:  # rewritten every check after convergence: free the figure and the plotter each time (it leaked ~70 MB/check)
        import gc

        import matplotlib.pyplot as plt

        plt.close("all")
        g = None
        gc.collect()


def finalize(m: str, res: dict | None, reason: str) -> None:
    """Final summary and triangle from what exists, stop the model's chains, write DONE. Slots stay empty."""
    if res is not None:
        res["summary"]["final"] = {"reason": reason, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        write_results(m, res)
    stop(m)
    (CHAINS / ALL[m][0] / "DONE").write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {reason}\n")
    log(f"{m}: FINAL ({reason}); summary {'written' if res is not None else 'NOT written (no usable chains)'}, "
        f"chains stopped")


def main() -> None:
    log(f"queue start: jobs={JOBS} check={CHECK}s rtol={RTOL} (sampled params) min_rows={MIN_ROWS} min_ess={MIN_ESS} "
        f"(on {ESS_PARAMS}) min_gap={MIN_GAP / 3600:g}h models={MODELS}; stop {POST / 3600:g} h after convergence, "
        f"hard stop {STOP_AT_STR}")

    def bye(*_):
        log("queue stopping (signal); chains stopped, rerun to resume")
        for m in MODELS:
            stop(m)
        sys.exit(0)

    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)
    last = 0.0
    while True:
        # restart crashed / exited chains of unfinished models, then fill free slots in priority order
        for (m, k), p in list(procs.items()):
            if p.poll() is not None:
                del procs[(m, k)]
                if not done(m):
                    log(f"{m} chain {k} exited (code {p.returncode}); restarting")
        for m in MODELS:
            if done(m):
                continue
            for k in range(1, NCHAIN + 1):
                if (m, k) not in procs and len(procs) < JOBS:
                    launch(m, k)
        if not procs and all(done(m) for m in MODELS):
            log("all models done")
            return
        if time.time() >= STOP_AT and not all(done(m) for m in MODELS):
            last = 0.0  # hard stop: check (and finalize) every unfinished model now
        if time.time() - last >= CHECK:
            last = time.time()
            status = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "running": sorted(f"{m}:{k}" for m, k in procs),
                      "models": {}}
            for m in MODELS:
                if done(m):
                    status["models"][m] = "done"
                    continue
                try:
                    res = check(m)
                except Exception as e:  # noqa: BLE001
                    log(f"{m}: check failed: {e}")
                    res = None
                    if time.time() < STOP_AT:
                        continue
                if res is None:
                    if time.time() >= STOP_AT:
                        finalize(m, None, f"{STOP_LABEL} {STOP_AT_STR}, not converged")
                        status["models"][m] = "done"
                    else:
                        status["models"][m] = "warming up"
                    continue
                s = res["summary"]
                samp = sampled(m)
                rs = {p: s["params"][p]["R_minus_1"] for p in samp if p in s["params"]}
                r_max, r_worst = max(rs.values()), max(rs, key=rs.get)
                ess = {p: s["params"][p].get("ess") or 0.0 for p in ESS_PARAMS[m]}
                ok = r_max < RTOL and min(s["n_samples"]) >= MIN_ROWS and all(v >= MIN_ESS for v in ess.values())
                st, now = rule_state(m), time.time()
                if not ok:
                    if st["converged"]:
                        log(f"{m}: WARNING rule fails after convergence was declared")
                    st["first_pass"] = None
                elif st["first_pass"] is None:
                    st["first_pass"] = now
                elif not st["converged"] and now - st["first_pass"] >= MIN_GAP:
                    st["converged"] = now
                    (CHAINS / ALL[m][0] / "CONVERGED").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
                    log(f"{m}: CONVERGED (rule held at checks {(now - st['first_pass']) / 3600:.1f} h apart); "
                        f"chains run {POST / 3600:g} h more, summary rewritten at every check")
                (CHAINS / ALL[m][0] / "rule_state.json").write_text(json.dumps(st) + "\n")
                rc = {p: s["params"][p]["R_minus_1"] for p in COSMO if p in s["params"]}
                s["rule"] = {"threshold": RTOL, "R_minus_1_max_cosmo": max(rc.values()), "worst_cosmo": max(rc, key=rc.get),
                             "rows_after_burn": s["n_samples"],
                             "R_minus_1_max_sampled": r_max, "worst_sampled": r_worst, "ess": ess, "passed": ok,
                             "first_pass": st["first_pass"], "converged": st["converged"]}
                if st["converged"]:
                    write_results(m, res)
                if st["converged"] and now - st["converged"] >= POST:
                    finalize(m, res, f"converged + {POST / 3600:g} h")
                elif now >= STOP_AT:
                    finalize(m, res, f"{STOP_LABEL} {STOP_AT_STR}, {'converged' if st['converged'] else 'not converged'}")
                status["models"][m] = {"R-1_max_sampled": r_max, "worst_sampled": r_worst,
                                       "R-1_max_all": s["R_minus_1_max"], "rows_after_burn": s["n_samples"],
                                       "chi2_best": s["chi2_best"], "ess": ess, "rule_passed": ok,
                                       "first_pass": st["first_pass"] and time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st["first_pass"])),
                                       "converged": bool(st["converged"])}
                if "frac_nu_near_edge" in s:
                    status["models"][m]["frac_nu_gt_-5e-5"] = s["frac_nu_near_edge"]
                log(f"{m}: max R-1 sampled = {r_max:.4f} ({r_worst}), all = {s['R_minus_1_max']:.4f}, "
                    f"rows {s['n_samples']}, ESS { {k: round(v) for k, v in ess.items()}}; "
                    f"rule {'passed' if ok else 'not passed'}{' (converged)' if st['converged'] else ''}")
            (CHAINS / "queue_status.json").write_text(json.dumps(status, indent=1) + "\n")
        time.sleep(30)


if __name__ == "__main__":
    main()
