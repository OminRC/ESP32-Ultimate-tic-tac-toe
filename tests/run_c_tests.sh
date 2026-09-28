#!/usr/bin/env bash
# Compiles and runs the native (host, non-ESP32) C unit tests.
# game.c and net_infer.c are plain, portable C99 with no ESP-IDF
# dependencies, so they build fine with a normal host gcc/clang.
set -euo pipefail
cd "$(dirname "$0")"

MAIN_DIR="../main"
BUILD_DIR="/tmp/ultimate_ttt_c_tests"
mkdir -p "$BUILD_DIR"

echo "== building =="
gcc -std=c99 -Wall -Wextra -I "$MAIN_DIR" \
    -o "$BUILD_DIR/test_game" c/test_game.c "$MAIN_DIR/game.c"

gcc -std=c99 -Wall -Wextra -I "$MAIN_DIR" \
    -o "$BUILD_DIR/test_net_infer" c/test_net_infer.c "$MAIN_DIR/net_infer.c" "$MAIN_DIR/game.c" -lm

echo "== running =="
"$BUILD_DIR/test_game"
echo
"$BUILD_DIR/test_net_infer"
