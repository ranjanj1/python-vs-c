#!/bin/bash
# Run from the experiment folder. Gate, then rebuild final snapshots into ports/, then hidden-prompt correctness on 110M.
H=bench/kit/cpython/bin/python
K=/Users/Shared/pvc-kit
$H bench/test_gate.py > validation/gate.log 2>&1; g=$?; echo "gate_exit $g"; [ $g = 0 ] || exit 1
$H bench/freeze.py; $H bench/freeze.py --check || exit 1
: > validation/build-times.txt
b() { s=$(python3 -c 'import time;print(time.time())'); ( cd "$2" && eval "$3" ) || { echo "build failed $1"; exit 1; }; e=$(python3 -c 'import time;print(time.time())'); echo "$1 $(python3 -c "print(round($e-$s,2))") s: $3" | tee -a validation/build-times.txt; }
for x in c1:optimize_2_redo2 p1:optimize_2 p4:optimize_2 p3:optimize_2 cb:optimize_2 mj:optimize_2; do
  i=${x%%:*}; rm -rf ports/$i; cp -R snapshots/$i/${x##*:} ports/$i; find ports/$i -name __pycache__ -prune -exec rm -rf {} +
done
rm -f ports/c1/run_c1 ports/cb/cb ports/mj/mj
b C1 ports/c1 './build.sh'
b CB ports/cb 'clang -O2 -Wall -Wextra -o cb cb.c -framework Accelerate'
b MJ ports/mj "$K/mojo/bin/mojo build mj.mojo -o mj"
v() { n=$1; shift; echo "START $n $(date +%H:%M)"; $H bench/validate.py --name $n --set eval --models 110M -- "$@"; c=$?; echo "END $n $(date +%H:%M) exit $c"; }
v C1 ports/c1/run_c1
v CB ports/cb/cb
v MJ ports/mj/mj
v P3 $K/numpy/bin/python -I ports/p3/run.py
v P4 $K/numba/bin/python -I ports/p4/run.py
v P2 $K/py/pypy-3.11.15-macos-aarch64-none/bin/pypy3 -I ports/p1/run.py
v P1 $K/stdlib/bin/python -I ports/p1/run.py
