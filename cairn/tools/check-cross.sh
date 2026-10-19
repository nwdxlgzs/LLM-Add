#!/bin/sh
# Compile checks only. Example:
# CAIRN_CLANG=clang-18 CAIRN_ARM_SYSROOT=/usr/aarch64-linux-gnu \
# CAIRN_RISCV_SYSROOT=/usr/riscv64-linux-gnu sh tools/check-cross.sh
set -eu
: "${CAIRN_CLANG:=clang-18}"
: "${CAIRN_ARM_SYSROOT:=/usr/aarch64-linux-gnu}"
: "${CAIRN_RISCV_SYSROOT:=/usr/riscv64-linux-gnu}"
: "${CAIRN_CROSS_OUTPUT:=/tmp/cairn-cross-check}"
cd "$(dirname "$0")/.."
mkdir -p "$CAIRN_CROSS_OUTPUT/arm" "$CAIRN_CROSS_OUTPUT/riscv"
for task_src in src/cairn.c src/cpu.c src/cpu_neon.c src/quant.c src/tokenizer.c src/main.c tests/cpu_test.c; do
  "$CAIRN_CLANG" --target=aarch64-linux-gnu --sysroot="$CAIRN_ARM_SYSROOT" -DCN_BUILD_SVE=1 -std=c11 -O3 -Wall -Wextra -Wpedantic -Iinclude -c "$task_src" -o "$CAIRN_CROSS_OUTPUT/arm/$(basename "$task_src").o"
  "$CAIRN_CLANG" --target=riscv64-linux-gnu --sysroot="$CAIRN_RISCV_SYSROOT" -march=rv64gc -DCN_BUILD_RVV=1 -std=c11 -O3 -Wall -Wextra -Wpedantic -Iinclude -c "$task_src" -o "$CAIRN_CROSS_OUTPUT/riscv/$(basename "$task_src").o"
done
"$CAIRN_CLANG" --target=aarch64-linux-gnu --sysroot="$CAIRN_ARM_SYSROOT" -march=armv8-a+sve -DCN_BUILD_SVE=1 -std=c11 -O3 -Wall -Wextra -Wpedantic -Iinclude -c src/cpu_sve.c -o "$CAIRN_CROSS_OUTPUT/arm/sve.o"
"$CAIRN_CLANG" --target=riscv64-linux-gnu --sysroot="$CAIRN_RISCV_SYSROOT" -march=rv64gcv -DCN_BUILD_RVV=1 -std=c11 -O3 -Wall -Wextra -Wpedantic -Iinclude -c src/cpu_rvv.c -o "$CAIRN_CROSS_OUTPUT/riscv/rvv.o"
printf 'PASS aarch64 NEON/SVE and riscv64 RVV cross compilation\n'
