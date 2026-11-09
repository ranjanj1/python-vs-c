#!/bin/sh
set -e
cd "$(dirname "$0")"
cc -O3 -mcpu=native -Wall -Wextra -o cb2 cb2.c -framework Accelerate
