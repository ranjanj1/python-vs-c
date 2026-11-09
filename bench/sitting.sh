#!/bin/bash
# One official timing sitting. Usage, from the experiment folder: bench/sitting.sh 2, then bench/sitting.sh 3 on a different day
cd "$(dirname "$0")/.." || exit 1
H=bench/kit/cpython/bin/python
K=/Users/Shared/pvc-kit
n=$1
case "$n" in 1) seed=20261106;; 2) seed=20261107;; 3) seed=20261109;; *) echo 'usage: bench/sitting.sh 1|2|3'; exit 1;; esac
[ -e "results/raw/sitting-$n.jsonl" ] && { echo "sitting $n already has data; not starting"; exit 1; }
$H bench/freeze.py --check || exit 1
if [ "$n" = 3 ]; then
  caffeinate -i $H bench/measure.py sitting-3 10 $seed --official "C0=ports/c0/run" "C1=ports/c1/run_c1" "CB=ports/cb/cb" "CB2=ports/cb2/cb2" "P3=$K/numpy/bin/python -I ports/p3/run.py" || exit 1
  echo "SITTING 3 FINISHED $(date)"; exit 0
fi
extra=(); [ "$n" = 2 ] && extra=("CB2=ports/cb2/cb2")
caffeinate -i $H bench/measure.py sitting-$n 10 $seed --official \
  "C0=ports/c0/run" "C1=ports/c1/run_c1" "CB=ports/cb/cb" "MJ=ports/mj/mj" \
  "P1=$K/stdlib/bin/python -I ports/p1/run.py" \
  "P2=$K/py/pypy-3.11.15-macos-aarch64-none/bin/pypy3 -I ports/p1/run.py" \
  "P3=$K/numpy/bin/python -I ports/p3/run.py" "P4=$K/numba/bin/python -I ports/p4/run.py" "${extra[@]}" || exit 1
if [ "$n" = 1 ]; then
  caffeinate -i $H bench/measure.py side-p2raw 3 20261108 --official "P2raw=$K/py/pypy-3.11.15-macos-aarch64-none/bin/pypy3 -I ports/p2raw/run.py" || exit 1
fi
echo "SITTING $n FINISHED $(date)"
