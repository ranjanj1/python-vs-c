#!/bin/sh
# Build the C1 port. No -ffast-math, no -lm, no OpenMP.
#   ./build.sh          -> ./run_c1        (the submission binary)
#   ./build.sh profile  -> ./run_c1_prof   (same code + section timers, stderr report at QUIT)
set -e
cd "$(dirname "$0")"
CFLAGS="-O3 -mcpu=apple-m1 -std=c11 -D_DARWIN_C_SOURCE -Wall -Wextra"
if [ "$1" = profile ]; then
    cc $CFLAGS -DPROFILE -o run_c1_prof run_c1.c
else
    cc $CFLAGS -o run_c1 run_c1.c
fi
