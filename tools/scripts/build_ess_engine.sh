#!/usr/bin/env bash
#
# Fetch NVIDIA's ESS stereo disparity model and build a TensorRT engine for the
# machine this runs on.
#
# A .engine is NOT a portable artifact. TensorRT serializes tactics chosen for
# one GPU and one TensorRT build, so a plan built on the desktop will not
# deserialize on an Orin and vice versa. This script is the portable thing:
# run it on each target and it works out what that target needs.
#
# Three things have to line up, and NGC ships combinations where they do not:
#
#   1. ONNX <-> TensorRT minor version. The 4.1.0 model is re-exported per TRT
#      minor line and they are NOT interchangeable -- the trt10.13 export fails
#      to build under 10.16 with a duplicate-tensor error out of the Myelin
#      optimizer. We require an exact match rather than guessing.
#
#   2. Plugin <-> GPU architecture. The graph carries five fused ops
#      (FusedConcatConv3x3 x2, FusedEinsum2Softmax, FusedConcat2*Conv3x3c64 x2)
#      that only exist as a prebuilt plugin. No PTX ships, only SASS, so a
#      plugin without your exact sm_ arch cannot run -- there is nothing to JIT.
#
#   3. Plugin <-> CUDA runtime <-> driver. The plugins link either libcudart.so.12
#      or .so.13, and a CUDA 13 plugin needs an r580+ driver. On an older driver
#      the engine still BUILDS and then fails at execute with an unhelpful
#      "Assertion pluginUtils::isSuccess(status) failed" -- which is really
#      cudaErrorInsufficientDriver surfacing three layers up. We check up front.
#
# Nothing here needs root: the matching trtexec is fetched with `apt-get
# download` and unpacked into the cache directory.

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly NGC_MODEL="https://api.ngc.nvidia.com/v2/models/nvidia/isaac/dnn_stereo_disparity"

MODEL="ess"
OUT=""
CACHE="${ESS_CACHE_DIR:-$HOME/.cache/perception_pipeline/ess}"
ARCH=""
NGC_VERSION=""
PRECISION="fp16"
BENCH=0
ALLOW_DRIVER_MISMATCH=0

die() { printf 'build_ess_engine: %s\n' "$*" >&2; exit 1; }
note() { printf '  %s\n' "$*" >&2; }
step() { printf '\n== %s\n' "$*" >&2; }

usage() {
  cat >&2 <<'EOF'
usage: build_ess_engine.sh [options]

  -m, --model ess|light_ess   Which export (default: ess, the 960x576 one the
                              pipeline's EssEngine requires; light_ess is
                              480x288 and is NOT supported by it)
  -o, --out PATH              Engine output (default: tensorrt/models/ess_full_<precision>.engine)
      --precision fp16|fp32   Build precision (default: fp16)
      --arch sm_XX            Override GPU arch detection
      --ngc-version ID        Override the NGC model version to fetch
      --cache DIR             Download/unpack cache (default: ~/.cache/perception_pipeline/ess)
      --bench                 Run trtexec inference timing after building
      --allow-driver-mismatch Build even if the driver is too old for the
                              plugin's CUDA runtime. The plan stays valid for
                              this machine, so this is how you pre-build ahead
                              of a driver upgrade -- it will not execute until
                              the driver is raised.
  -h, --help
EOF
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model) MODEL="${2:?}"; shift 2 ;;
    -o|--out) OUT="${2:?}"; shift 2 ;;
    --precision) PRECISION="${2:?}"; shift 2 ;;
    --arch) ARCH="${2:?}"; shift 2 ;;
    --ngc-version) NGC_VERSION="${2:?}"; shift 2 ;;
    --cache) CACHE="${2:?}"; shift 2 ;;
    --bench) BENCH=1; shift ;;
    --allow-driver-mismatch) ALLOW_DRIVER_MISMATCH=1; shift ;;
    -h|--help) usage 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage 1 ;;
  esac
done

[[ "$MODEL" == "ess" || "$MODEL" == "light_ess" ]] || die "--model must be ess or light_ess"
[[ "$PRECISION" == "fp16" || "$PRECISION" == "fp32" ]] || die "--precision must be fp16 or fp32"
if [[ "$MODEL" == "light_ess" ]]; then
  note "warning: EssEngine rejects light_ess -- it requires the 1x3x576x960 full export"
fi
[[ -n "$OUT" ]] || OUT="$REPO_ROOT/tensorrt/models/${MODEL/ess/ess_full}_${PRECISION}.engine"

for tool in curl tar python3 objdump dpkg-deb apt-get; do
  command -v "$tool" >/dev/null || die "missing required tool: $tool"
done
command -v cuobjdump >/dev/null || note "note: cuobjdump not found (CUDA toolkit); skipping plugin arch verification"

mkdir -p "$CACHE"

# --- 1. what TensorRT is installed ------------------------------------------
# The header, not the package: a tarball install has no dpkg entry, and the
# header is what the build actually compiles against.
step "TensorRT"
TRT_HEADER=""
for candidate in /usr/include/x86_64-linux-gnu/NvInferVersion.h \
                 /usr/include/aarch64-linux-gnu/NvInferVersion.h \
                 /usr/include/NvInferVersion.h; do
  [[ -f "$candidate" ]] && { TRT_HEADER="$candidate"; break; }
done
[[ -n "$TRT_HEADER" ]] || die "NvInferVersion.h not found -- install the TensorRT dev packages"

# TRT 10.7+ defines TRT_*_ENTERPRISE and aliases NV_TENSORRT_* to those names,
# so the alias expands to an identifier rather than a digit. Try it first.
read_ver() {
  local key="$1"
  local v
  v="$(grep -oP "define\s+TRT_${key}_ENTERPRISE\s+\K[0-9]+" "$TRT_HEADER" | head -1)"
  [[ -n "$v" ]] || v="$(grep -oP "define\s+NV_TENSORRT_${key}\s+\K[0-9]+" "$TRT_HEADER" | head -1)"
  printf '%s' "$v"
}
TRT_MAJOR="$(read_ver MAJOR)"; TRT_MINOR="$(read_ver MINOR)"
[[ -n "$TRT_MAJOR" && -n "$TRT_MINOR" ]] || die "could not parse a version out of $TRT_HEADER"
note "TensorRT ${TRT_MAJOR}.${TRT_MINOR} ($TRT_HEADER)"

# --- 2. GPU architecture and driver ------------------------------------------
step "GPU"
DRIVER_CUDA_MAJOR=""
if [[ -z "$ARCH" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
  [[ -n "$cc" ]] && ARCH="sm_${cc}"
fi
# Jetson has no nvidia-smi. Orin is the only Tegra this repo targets.
if [[ -z "$ARCH" && -f /etc/nv_tegra_release ]]; then
  ARCH="sm_87"
  note "Tegra detected, assuming Orin ($ARCH); override with --arch"
fi
[[ -n "$ARCH" ]] || die "could not detect the GPU architecture -- pass --arch sm_XX"

if command -v nvidia-smi >/dev/null 2>&1; then
  DRIVER_CUDA_MAJOR="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[0-9]+' | head -1)"
fi
note "arch $ARCH${DRIVER_CUDA_MAJOR:+, driver supports CUDA ${DRIVER_CUDA_MAJOR}.x}"

# --- 3. pick the NGC version matching this TensorRT --------------------------
step "NGC model version"
if [[ -z "$NGC_VERSION" ]]; then
  versions_json="$(curl -fsSL --max-time 60 "$NGC_MODEL/versions")" \
    || die "could not reach NGC to list model versions"
  NGC_VERSION="$(printf '%s' "$versions_json" | python3 -c '
import json, re, sys
want = sys.argv[1]
ids = [v["versionId"] for v in json.load(sys.stdin)["modelVersions"]]
# "..._onnx_trt10.16" or "..._onnx_trt10.16_r3" -- take the highest revision.
pat = re.compile(r"^(?P<base>.*_onnx_trt" + re.escape(want) + r")(?:_r(?P<rev>\d+))?$")
best, best_rev = None, -1
for i in ids:
    m = pat.match(i)
    if m:
        rev = int(m.group("rev") or 0)
        if rev > best_rev:
            best, best_rev = i, rev
if best:
    print(best)
    sys.exit(0)

# No per-minor export for this TensorRT. NVIDIA only started suffixing them at
# 10.13; before that there is a single un-suffixed ONNX export, which is what a
# JetPack 6 box (TensorRT 10.3) has to use. Fall back to it rather than giving
# up -- but say so, because a mismatch here fails at build time, loudly, in the
# Myelin optimizer rather than silently at runtime.
plain = [i for i in ids if i.endswith("_onnx")]
if plain:
    sys.stderr.write("  note: NGC has no ONNX export for TensorRT " + want + "; falling back to\n"
                     "        \x27" + plain[0] + "\x27, the un-suffixed export. If trtexec fails to\n"
                     "        build it, there is no compatible ESS release for this TensorRT.\n")
    print(plain[0])
    sys.exit(0)

sys.stderr.write("no ONNX export usable for TensorRT " + want + "; NGC has:\n  " + "\n  ".join(ids) + "\n")
sys.exit(1)
' "${TRT_MAJOR}.${TRT_MINOR}")" || die "no matching NGC version (see above). Pass --ngc-version to force one, but expect the build to fail: these exports are not interchangeable across TensorRT minor versions."
fi
note "$NGC_VERSION"

# --- 4. fetch and unpack ------------------------------------------------------
step "Model package"
PKG_DIR="$CACHE/dnn_stereo_disparity_v${NGC_VERSION}"
TARBALL="$CACHE/dnn_stereo_disparity_v${NGC_VERSION}.tar.gz"
if [[ ! -d "$PKG_DIR" ]]; then
  if [[ ! -s "$TARBALL" ]]; then
    note "downloading (~128 MB)"
    curl -fL --max-time 1800 --progress-bar -o "$TARBALL.part" \
      "$NGC_MODEL/versions/$NGC_VERSION/files/dnn_stereo_disparity_v${NGC_VERSION}.tar.gz" \
      || die "download failed"
    mv "$TARBALL.part" "$TARBALL"
  fi
  tar xzf "$TARBALL" -C "$CACHE"
fi
ONNX="$PKG_DIR/${MODEL}.onnx"
[[ -f "$ONNX" ]] || die "$ONNX missing from the package"
note "$ONNX"

# --- 5. pick and vet the plugin ----------------------------------------------
step "Plugin"
case "$(uname -m)" in
  x86_64)  PLUGIN_ARCH_DIR=x86_64 ;;
  aarch64) PLUGIN_ARCH_DIR=aarch64 ;;
  *) die "unsupported host architecture $(uname -m)" ;;
esac
PLUGIN="$PKG_DIR/plugins/$PLUGIN_ARCH_DIR/ess_plugins.so"
[[ -f "$PLUGIN" ]] || die "$PLUGIN missing from the package"

PLUGIN_CUDART="$(objdump -p "$PLUGIN" | grep -oE 'libcudart\.so\.[0-9]+' | grep -oE '[0-9]+$' | head -1)"
note "$PLUGIN (CUDA $PLUGIN_CUDART)"

if command -v cuobjdump >/dev/null 2>&1; then
  archs="$(cuobjdump -lelf "$PLUGIN" 2>/dev/null | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' ')"
  if [[ -n "$archs" ]] && ! grep -qw "$ARCH" <<<"$archs"; then
    die "this plugin has no $ARCH kernels (only: $archs) and ships no PTX to JIT from.
     Try a different --ngc-version whose plugin covers $ARCH."
  fi
  note "kernels: $archs"
fi

if [[ -n "$DRIVER_CUDA_MAJOR" && -n "$PLUGIN_CUDART" && "$PLUGIN_CUDART" -gt "$DRIVER_CUDA_MAJOR" ]]; then
  if [[ "$ALLOW_DRIVER_MISMATCH" -eq 1 ]]; then
    note "warning: plugin needs CUDA $PLUGIN_CUDART, driver supports ${DRIVER_CUDA_MAJOR}.x --"
    note "         building anyway; the engine will not execute until the driver is raised"
    BENCH=0
  else
  die "plugin needs CUDA $PLUGIN_CUDART but the driver only supports CUDA ${DRIVER_CUDA_MAJOR}.x.
     The engine would build and then fail at execute with
     'Assertion pluginUtils::isSuccess(status) failed', which is really
     cudaErrorInsufficientDriver. CUDA 13 needs an r580+ driver.
     Pass --allow-driver-mismatch to build the plan now and run it after the
     driver is upgraded."
  fi
fi

# The plugin's own libcudart has to be resolvable. It is not necessarily the one
# the rest of the build uses -- a CUDA 13 plugin alongside a CUDA 12 TensorRT is
# a combination NVIDIA ships, and the two runtimes coexist in one process.
PLUGIN_LD_PATH=""
if ! ldconfig -p | grep -q "libcudart\.so\.${PLUGIN_CUDART}"; then
  found="$(find /usr/local -name "libcudart.so.${PLUGIN_CUDART}" 2>/dev/null | head -1)"
  if [[ -z "$found" ]]; then
    note "libcudart.so.${PLUGIN_CUDART} not installed; fetching it into the cache"
    pkg="$(apt-cache pkgnames "cuda-cudart-${PLUGIN_CUDART}-" 2>/dev/null | grep -v -- '-dev' | sort -V | tail -1)"
    [[ -n "$pkg" ]] || die "no cuda-cudart-${PLUGIN_CUDART}-* package available"
    ( cd "$CACHE" && mkdir -p cudart && cd cudart \
      && apt-get download "$pkg" >/dev/null 2>&1 \
      && for d in *.deb; do dpkg-deb -x "$d" root; done )
    found="$(find "$CACHE/cudart/root" -name "libcudart.so.${PLUGIN_CUDART}" | head -1)"
    [[ -n "$found" ]] || die "could not obtain libcudart.so.${PLUGIN_CUDART}"
  fi
  PLUGIN_LD_PATH="$(dirname "$found")"
  note "using $found"
fi

# --- 6. a trtexec whose version matches libnvinfer ----------------------------
# A plan built by a different TensorRT than the one `acquire` links will not
# deserialize, so an approximately-right trtexec is worse than none.
step "trtexec"
trtexec_version() {
  "$1" --help 2>&1 | grep -oP '\[TensorRT v\K[0-9]+' | head -1
}
want_v="$(printf '%d%02d' "$TRT_MAJOR" "$TRT_MINOR")"
TRTEXEC=""
TRTEXEC_LD_PATH=""

# A trtexec unpacked from a .deb rather than installed carries its own
# libnvonnxparser next to it, and the loader will not find it on its own.
sibling_lib_dir() {
  local dir
  dir="$(cd "$(dirname "$1")/../lib/$(uname -m)-linux-gnu" 2>/dev/null && pwd)" || return 0
  [[ -e "$dir/libnvonnxparser.so.10" ]] && printf '%s' "$dir"
}

for candidate in ${TRTEXEC_BIN:-} "$(command -v trtexec 2>/dev/null || true)" \
                 /usr/src/tensorrt/bin/trtexec "$CACHE/trtexec/root/usr/bin/trtexec"; do
  [[ -n "$candidate" && -x "$candidate" ]] || continue
  lib="$(sibling_lib_dir "$candidate")"
  v="$(LD_LIBRARY_PATH="${lib}${lib:+:}${LD_LIBRARY_PATH:-}" trtexec_version "$candidate" 2>/dev/null || true)"
  if [[ "${v:0:${#want_v}}" == "$want_v" ]]; then
    TRTEXEC="$candidate"
    TRTEXEC_LD_PATH="$lib"
    break
  fi
done

if [[ -z "$TRTEXEC" ]]; then
  note "no trtexec matching TensorRT ${TRT_MAJOR}.${TRT_MINOR}; fetching one into the cache"
  apt_version="$(dpkg-query -W -f='${Version}' libnvinfer10 2>/dev/null || true)"
  [[ -n "$apt_version" ]] || die "no matching trtexec, and libnvinfer10 is not an apt package here.
     Point TRTEXEC_BIN at a trtexec from TensorRT ${TRT_MAJOR}.${TRT_MINOR}."
  ( cd "$CACHE" && mkdir -p trtexec && cd trtexec \
    && apt-get download "libnvinfer-bin=$apt_version" "libnvonnxparsers10=$apt_version" >/dev/null 2>&1 \
    && for d in *.deb; do dpkg-deb -x "$d" root; done ) \
    || die "could not fetch libnvinfer-bin=$apt_version"
  TRTEXEC="$CACHE/trtexec/root/usr/bin/trtexec"
  [[ -x "$TRTEXEC" ]] || die "trtexec not present in the downloaded package"
  TRTEXEC_LD_PATH="$(sibling_lib_dir "$TRTEXEC")"
fi
note "$TRTEXEC (v$(LD_LIBRARY_PATH="${TRTEXEC_LD_PATH}${TRTEXEC_LD_PATH:+:}${LD_LIBRARY_PATH:-}" trtexec_version "$TRTEXEC"))"

# --- 7. build -----------------------------------------------------------------
step "Building $PRECISION engine"
mkdir -p "$(dirname "$OUT")"
LD_PATH="$(printf '%s' "${TRTEXEC_LD_PATH}${TRTEXEC_LD_PATH:+:}${PLUGIN_LD_PATH}${PLUGIN_LD_PATH:+:}/usr/lib/$(uname -m)-linux-gnu")"

args=( --onnx="$ONNX" --saveEngine="$OUT" )
[[ "$PRECISION" == "fp16" ]] && args+=( --fp16 )
[[ "$BENCH" -eq 1 ]] || args+=( --skipInference )

log="$CACHE/build_${MODEL}_${PRECISION}.log"
if env LD_LIBRARY_PATH="$LD_PATH" LD_PRELOAD="$PLUGIN" "$TRTEXEC" "${args[@]}" >"$log" 2>&1; then
  printf '\n'
  grep -E 'Successfully created plugin|Engine built in' "$log" | sed 's/^/  /' >&2 || true
  [[ "$BENCH" -eq 1 ]] && grep -E 'Throughput|GPU Compute Time: (min|median|max)' "$log" | sed 's/^/  /' >&2
  printf '\nengine: %s (%s)\nlog:    %s\n' "$OUT" "$(du -h "$OUT" | cut -f1)" "$log" >&2
else
  printf '\n' >&2
  grep -E '^\[.*\] \[E\]' "$log" | tail -10 | sed 's/^/  /' >&2 || true
  die "trtexec failed -- full log at $log"
fi

cat >&2 <<EOF

Set this in app/config/acquire.yaml (paths are relative to the executable, and
tensorrt/models is symlinked next to it at build time):

  ess:
    enabled: true
    engine_path: "models/$(basename "$OUT")"
    plugin_path: "$PLUGIN"

plugin_path is not optional. The plugin is NOT baked into the plan: it registers
its creators through static initializers and exports no getCreators entry point,
so --setPluginsToSerialize cannot embed it, and whatever deserializes this engine
has to load the library first or the plan fails with "Cannot find plugin".
EssEngine dlopens plugin_path (RTLD_NOW|RTLD_GLOBAL) before building its runtime,
which is what makes the config key enough; anything else reading this engine has
to do the same, or LD_PRELOAD it.

The engine is specific to $ARCH and TensorRT ${TRT_MAJOR}.${TRT_MINOR}. Do not copy
it to another machine -- run this script there instead.
EOF
