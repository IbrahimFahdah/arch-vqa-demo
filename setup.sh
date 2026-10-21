#!/usr/bin/env bash
# One-time setup. Downloads into this folder only; re-running skips files that are already there.
#   bash setup.sh            # llama.cpp + base model (official Qwen files) + fpc adapter (66 MB)
#   bash setup.sh --full     # also fpcvis, the best door finder (+900 MB)
# Override the llama.cpp build with LLAMA_BUILD, e.g. LLAMA_BUILD=win-vulkan-x64 or ubuntu-vulkan-x64
# (names as on https://github.com/ggml-org/llama.cpp/releases/tag/b11606, without "llama-b11606-bin-").
set -euo pipefail
cd "$(dirname "$0")"

LLAMA_TAG=b11606                                  # the llama.cpp build the models were tested with
RELEASE="${RELEASE:-https://github.com/IbrahimFahdah/arch-vqa-demo/releases/download/v1.0}"
QWEN="https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main"
FULL=0
for a in "$@"; do
  case "$a" in
    --full) FULL=1 ;;
    *) echo "unknown option $a (use --full)"; exit 2 ;;
  esac
done

sha256() { if command -v sha256sum > /dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

# fetch URL FILE SHA256: download to models/FILE unless a verified copy is there
fetch() {
  local url=$1 out=models/$2 want=$3
  if [ -f "$out" ] && [ "$(sha256 "$out")" = "$want" ]; then echo "ok    $out"; return; fi
  echo "get   $out"
  if ! curl -fL --retry 3 -C - -o "$out.part" "$url"; then
    # private repo: release files need a logged-in GitHub CLI
    if [[ $url == "$RELEASE"* ]] && command -v gh > /dev/null; then
      rm -f "$out.part"
      gh release download v1.0 -R IbrahimFahdah/arch-vqa-demo -p "$2" -O "$out.part"
    else
      echo "download failed: $url"; exit 1
    fi
  fi
  [ "$(sha256 "$out.part")" = "$want" ] || { echo "checksum mismatch for $out; delete $out.part and retry"; exit 1; }
  mv "$out.part" "$out"
}

# 1. llama.cpp for this system
if [ -z "${LLAMA_BUILD:-}" ]; then
  gpu=cpu; command -v nvidia-smi > /dev/null && nvidia-smi > /dev/null 2>&1 && gpu=cuda
  case "$(uname -s)-$(uname -m)" in
    MINGW*|MSYS*|CYGWIN*) [ $gpu = cuda ] && LLAMA_BUILD=win-cuda-12.4-x64 || LLAMA_BUILD=win-cpu-x64 ;;
    Darwin-arm64)         LLAMA_BUILD=macos-arm64 ;;      # Apple GPU (Metal) built in
    Darwin-*)             LLAMA_BUILD=macos-x64 ;;
    Linux-x86_64)         [ $gpu = cuda ] && LLAMA_BUILD=ubuntu-cuda-12.8-x64 || LLAMA_BUILD=ubuntu-x64 ;;
    Linux-aarch64)        LLAMA_BUILD=ubuntu-arm64 ;;
    *) echo "unknown system $(uname -s)-$(uname -m); set LLAMA_BUILD"; exit 1 ;;
  esac
fi
BIN=third_party/llama.cpp
if [ -f "$BIN/.build" ] && [ "$(cat "$BIN/.build")" = "$LLAMA_TAG-$LLAMA_BUILD" ]; then
  echo "ok    llama.cpp $LLAMA_TAG ($LLAMA_BUILD)"
else
  echo "get   llama.cpp $LLAMA_TAG ($LLAMA_BUILD)"
  rm -rf "$BIN"; mkdir -p "$BIN"
  base="https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_TAG"
  case "$LLAMA_BUILD" in
    win-*)   files="llama-$LLAMA_TAG-bin-$LLAMA_BUILD.zip"
             [[ $LLAMA_BUILD == win-cuda-* ]] && files="$files cudart-llama-bin-$LLAMA_BUILD.zip" ;;
    *)       files="llama-$LLAMA_TAG-bin-$LLAMA_BUILD.tar.gz"
             [[ $LLAMA_BUILD == ubuntu-cuda-* ]] && files="$files cudart-llama-$LLAMA_TAG-bin-$LLAMA_BUILD.tar.gz" ;;
  esac
  for f in $files; do
    curl -fL --retry 3 -o "$BIN/$f" "$base/$f"
    case "$f" in
      *.zip) if command -v unzip > /dev/null; then unzip -oq "$BIN/$f" -d "$BIN"
             else python -m zipfile -e "$BIN/$f" "$BIN"; fi ;;
      *)     tar -xzf "$BIN/$f" -C "$BIN" ;;
    esac
    rm "$BIN/$f"
  done
  echo "$LLAMA_TAG-$LLAMA_BUILD" > "$BIN/.build"
fi
server=$(find "$BIN" -name 'llama-server' -o -name 'llama-server.exe' | head -1 || true)
[ -n "$server" ] || { echo "llama-server not found after unpacking"; exit 1; }
chmod +x "$server" 2> /dev/null || true
"$server" --version 2>&1 | tail -2

# 2. Base model, 4-bit, and its vision file: official Qwen files from Hugging Face (Apache 2.0)
mkdir -p models
fetch "$QWEN/Qwen3VL-4B-Instruct-Q4_K_M.gguf" Qwen3VL-4B-Instruct-Q4_K_M.gguf \
  66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a
fetch "$QWEN/mmproj-Qwen3VL-4B-Instruct-F16.gguf" mmproj-Qwen3VL-4B-Instruct-F16.gguf \
  256f3a43bd4205ffef48d6b92715e1e70b5b0e9aef06522584967513a9985331

# 3. Fine-tuned adapters from this repo's release (CC BY-NC-SA 4.0, see MODEL_LICENSE.md)
fetch "$RELEASE/qwen3vl4b_lora_fpc-F16.gguf" qwen3vl4b_lora_fpc-F16.gguf \
  e01fe1ac124974988b7fc95ee9f0c261a5b27742de41d89c37a39d159baceb2d
if [ $FULL = 1 ]; then
  fetch "$RELEASE/qwen3vl4b_lora_fpcvis-F16.gguf" qwen3vl4b_lora_fpcvis-F16.gguf \
    0a38b801dc9c391fb3f99f6590e92dfe3d720f154096a5809311a1c0b571019c
  fetch "$RELEASE/mmproj-fpcvis-F16.gguf" mmproj-fpcvis-F16.gguf \
    d6e4834dbf66fb21029b9d4b46b09501bba2f038d02af28e05b51ae1b7dbf960
fi

echo
echo "Setup done. Start the app with:  bash run.sh"
[ $FULL = 1 ] || echo "(For the best door finder, fpcvis, run:  bash setup.sh --full)"
