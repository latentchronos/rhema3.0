#!/usr/bin/env bash
#
# Launch Rhema in dev with the on-device stack: local STT + comprehension LLM +
# neural VAD. This exists because those features need several environment
# variables that are easy to lose when pasting a multi-line command (a missing
# `\` continuation silently drops them, and the llama.cpp / bindgen build then
# fails with "libclang can't find <stdbool.h>" or the model just never loads).
#
# Every var below is overridable: export it yourself before running and this
# script leaves your value alone. Otherwise sane defaults for this machine are used.
#
#   ./scripts/dev-comprehension.sh
#
set -euo pipefail

# Run from the project root regardless of where the script was invoked.
cd "$(dirname "$0")/.."

# ── Build-time (llama.cpp via llama-cpp-sys-2 / bindgen) ──────────────────────
# The clang *driver* is absent on this box but libclang is present, so bindgen
# must be pointed at gcc's freestanding headers. Auto-detect the newest gcc so
# this keeps working across gcc version bumps (15 -> 16 -> ...).
export LIBCLANG_PATH="${LIBCLANG_PATH:-/usr/lib/x86_64-linux-gnu}"
if [ -z "${BINDGEN_EXTRA_CLANG_ARGS:-}" ]; then
  gcc_inc="$(ls -d /usr/lib/gcc/x86_64-linux-gnu/*/include 2>/dev/null | sort -V | tail -1)"
  if [ -n "${gcc_inc}" ]; then
    export BINDGEN_EXTRA_CLANG_ARGS="-I${gcc_inc}"
  fi
fi

# ── Runtime ──────────────────────────────────────────────────────────────────
# LD_LIBRARY_PATH is read by the dynamic linker before main(), so it must be in
# the environment at launch (a .env loaded inside the app is too late).
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/lib/x86_64-linux-gnu/blas}"
export RHEMA_STT_PROVIDER="${RHEMA_STT_PROVIDER:-local}"
export RHEMA_STT_STREAM="${RHEMA_STT_STREAM:-1}"
export RHEMA_COMPREHENSION_MODEL="${RHEMA_COMPREHENSION_MODEL:-$PWD/model/Qwen3-1.7B-Q4_K_M.gguf}"

echo "[dev-comprehension] LIBCLANG_PATH=$LIBCLANG_PATH"
echo "[dev-comprehension] BINDGEN_EXTRA_CLANG_ARGS=${BINDGEN_EXTRA_CLANG_ARGS:-<unset>}"
echo "[dev-comprehension] RHEMA_COMPREHENSION_MODEL=$RHEMA_COMPREHENSION_MODEL"
if [ ! -f "$RHEMA_COMPREHENSION_MODEL" ]; then
  echo "[dev-comprehension] WARNING: model file not found — comprehension will stay idle." >&2
fi

exec bun run tauri dev -- --features "local-stt,local-comprehension,neural-vad"
