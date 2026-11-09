#!/bin/sh
# Build the C1 port. No -ffast-math, no -lm, no OpenMP.
set -e
cd "$(dirname "$0")"
cc -O3 -mcpu=apple-m1 -std=c11 -D_DARWIN_C_SOURCE -Wall -Wextra -o run_c1 run_c1.c
