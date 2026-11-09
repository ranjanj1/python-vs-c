#!/bin/bash
# Run from the experiment folder. Recheck after the P1 repair.
H=bench/kit/cpython/bin/python
K=/Users/Shared/pvc-kit
$H bench/freeze.py --check || exit 1
v() { n=$1; shift; echo "START $n $(date +%H:%M)"; $H bench/validate.py --name $n --set eval --models 110M -- "$@"; c=$?; echo "END $n $(date +%H:%M) exit $c"; }
v P2 $K/py/pypy-3.11.15-macos-aarch64-none/bin/pypy3 -I ports/p1/run.py
v P1 $K/stdlib/bin/python -I ports/p1/run.py
