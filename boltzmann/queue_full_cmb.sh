#!/usr/bin/env bash
# Full-Planck (PR4 lensing) profiles on two workers: minimal RVM, its RRVM-like variant and the flipped RVM.
# Restartable: a point already in its output file is skipped. Needs CLASSY_RVM, COBAYA_PACKAGES_PATH, planckpr4lensing.
# Usage: LOG_DIR=/some/dir bash boltzmann/queue_full_cmb.sh
set -u
cd "$(dirname "$0")/.."
LOG_DIR=${LOG_DIR:-boltzmann/logs}
mkdir -p "$LOG_DIR"
export OMP_NUM_THREADS=1
PR4=results/full_cmb_pr4.jsonl FLIP=results/full_cmb_flipped.jsonl
python3 -c "import planckpr4lensing" || { echo "planckpr4lensing is not installed"; exit 1; }

have() {  # have FILE COUNT key=value ...: whether FILE holds COUNT records matching every key=value
  python3 - "$@" <<'EOF'
import json, sys
f, n, kv = sys.argv[1], int(sys.argv[2]), dict(a.split("=", 1) for a in sys.argv[3:])
try:
    recs = [json.loads(l) for l in open(f) if l.strip()]
except FileNotFoundError:
    recs = []
cast = lambda v: {"True": True, "False": False}.get(v, float(v) if v.replace(".", "").replace("-", "").isdigit() else v)
hit = [r for r in recs if all(r.get(k) == cast(v) for k, v in kv.items())]
sys.exit(0 if len(hit) >= n else 1)
EOF
}

unsettled() {  # unsettled FILE key=value ...: whether the last matching run still improved by > 0.2 in its last round
  python3 - "$@" <<'EOF'
import json, sys
f, kv = sys.argv[1], dict(a.split("=", 1) for a in sys.argv[2:])
cast = lambda v: {"True": True, "False": False}.get(v, float(v) if v.replace(".", "").replace("-", "").isdigit() else v)
recs = [json.loads(l) for l in open(f) if l.strip()]
r = [r for r in recs if all(r.get(k) == cast(v) for k, v in kv.items())][-1]
h = [c for _, c in r["history"]]
sys.exit(0 if h[-2] - h[-1] > 0.2 else 1)
EOF
}

run() {  # run OUT TAG COUNT match... -- full_cmb args...: one point, then one warm continuation if it had not settled
  local out=$1 tag=$2 n=$3; shift 3
  local match=() ; while [ "$1" != "--" ]; do match+=("$1"); shift; done; shift
  if have "$out" "$n" "${match[@]}"; then echo "skip $tag"; return; fi
  echo "$(date +%T) start $tag"
  python3 boltzmann/full_cmb.py "$@" out="$out" >> "$LOG_DIR/$tag.log" 2>&1 || echo "FAIL $tag"
  if unsettled "$out" "${match[@]}" 2>/dev/null; then
    echo "$(date +%T) warm $tag"
    python3 boltzmann/full_cmb.py "$@" out="$out" warm=1 >> "$LOG_DIR/$tag.log" 2>&1 || echo "FAIL warm $tag"
  fi
  echo "$(date +%T) end $tag"
}

worker_a() {  # ΛCDM, the minimal-RVM profile outwards from its minimum, w0wa, then the RRVM-like variant
  run $PR4 lcdm_a 1 model=lcdm -- model=lcdm
  run $PR4 rvm_m0.5 1 model=rvm nu=-0.0005 rad=1 -- model=rvm nu=-0.0005 rad=1
  run $PR4 rvm_m1.0 1 model=rvm nu=-0.001 rad=1 -- model=rvm nu=-0.001 rad=1
  run $PR4 rvm_m1.5 1 model=rvm nu=-0.0015 rad=1 -- model=rvm nu=-0.0015 rad=1 extrap=-0.0005,-0.001
  run $PR4 rvm_m2.0 1 model=rvm nu=-0.002 rad=1 -- model=rvm nu=-0.002 rad=1 extrap=-0.001,-0.0015
  run $PR4 rvm_p0.5 1 model=rvm nu=0.0005 rad=1 -- model=rvm nu=0.0005 rad=1
  run $PR4 cpl 1 model=cpl -- model=cpl maxfun=220
  run $PR4 rrvm_m0.5 1 model=rvm nu=-0.0005 rad=0 -- model=rvm nu=-0.0005 rad=0
  run $PR4 rrvm_m1.0 1 model=rvm nu=-0.001 rad=0 -- model=rvm nu=-0.001 rad=0
  run $PR4 rrvm_p0.5 1 model=rvm nu=0.0005 rad=0 -- model=rvm nu=0.0005 rad=0
}

worker_b() {  # a second ΛCDM start, then the flipped RVM at Solà et al.'s switches, ours, and a free-switch polish
  run $PR4 lcdm_b 2 model=lcdm -- model=lcdm set=H0:67.6,omega_cdm:0.1200,n_s:0.9640,tau_reio:0.052
  local s="z1=1.22 z2=2.16"
  run $FLIP flip_0.015 1 model=flipped nu=0.015 $s free_z=False -- model=flipped nu=0.015 $s
  run $FLIP flip_0.03 1 model=flipped nu=0.03 $s free_z=False -- model=flipped nu=0.03 $s
  run $FLIP flip_0.06 1 model=flipped nu=0.06 $s free_z=False -- model=flipped nu=0.06 $s extrap=0.015,0.03
  run $FLIP flip_0.12 1 model=flipped nu=0.12 $s free_z=False -- model=flipped nu=0.12 $s extrap=0.03,0.06
  s="z1=1.1 z2=1.8"
  run $FLIP flipB_0.03 1 model=flipped nu=0.03 $s free_z=False -- model=flipped nu=0.03 $s
  run $FLIP flipB_0.12 1 model=flipped nu=0.12 $s free_z=False -- model=flipped nu=0.12 $s
  local best  # best ν of the (1.22, 2.16) profile, then free the switches there
  best=$(python3 -c "
import json; r=[json.loads(l) for l in open('$FLIP')]
r=[x for x in r if x['z1']==1.22 and x['z2']==2.16 and not x['free_z']]; print(min(r, key=lambda x: x['chi2'])['nu'])")
  run $FLIP flip_free 1 model=flipped free_z=True -- model=flipped nu="$best" z1=1.22 z2=2.16 free_z=1 maxfun=260
}

worker_a > "$LOG_DIR/worker_a.txt" 2>&1 &
worker_b > "$LOG_DIR/worker_b.txt" 2>&1 &
wait
echo "ALL DONE"
