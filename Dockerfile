# 12.8 is the floor for sm_120 (Blackwell); it also still supports sm_87 (Orin).
# Nothing here needs a GPU at build time -- nvcc cross-compiles fine on a laptop
# with no NVIDIA hardware. You only need `--gpus all` to *run* the binary.
# 22.04 also matches JetPack 6's userspace, which is what sm_87 ships against.
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG CUDA_ARCHS="87-real;120-real;120-virtual"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        pkg-config \
        ca-certificates \
        libusb-1.0-0 \
        libyaml-cpp-dev \
        libeigen3-dev \
        libgtest-dev \
        libglfw3-dev \
        libgl-dev \
        libglvnd-dev \
    && rm -rf /var/lib/apt/lists/*

# --- clangd ------------------------------------------------------------------
# jammy ships clangd 14, which predates C++20 consteval support good enough for
# libstdc++ 13's <chrono> (every file including it reports
# `_S_fractional_width is not a constant expression`) and does not know CUDA 12.
# 19 handles both. Indexing only -- nvcc still does the real build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget gnupg \
    && wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-19 main" \
        > /etc/apt/sources.list.d/llvm.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        clangd-19 \
    && update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-19 100 \
    && rm -rf /var/lib/apt/lists/*

# --- GCC 13 ------------------------------------------------------------------
# The tree is C++20 and includes <format>, which jammy's default GCC 11 does not
# ship. Same PPA the host dev machine uses. CUDA 12.8 accepts GCC 13 as the nvcc
# host compiler; do not jump to 14, which it rejects.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get install -y --no-install-recommends g++-13 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-13 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-13 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-13 100 \
    && rm -rf /var/lib/apt/lists/*

# --- TensorRT ----------------------------------------------------------------
# tensorrt/CMakeLists.txt pins major 10 + CUDA 12 so one source tree builds for
# both targets (sm_120 desktop needs >= 10.7, JetPack 6 ships 10.3). The base
# image already has NVIDIA's CUDA apt repo, which carries the +cuda12.9 debs.
# Override with --build-arg TRT_VERSION=... to match a deployment's engine file.
ARG TRT_VERSION=10.16.1.11-1+cuda12.9
# Every dependency is pinned too: apt otherwise resolves the unversioned deps to
# their newest +cuda13 build and declares the set unsatisfiable.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libnvinfer-dev=${TRT_VERSION} \
        libnvinfer-headers-dev=${TRT_VERSION} \
        libnvinfer-safe-headers-dev=${TRT_VERSION} \
        libnvinfer10=${TRT_VERSION} \
    && rm -rf /var/lib/apt/lists/*

# A user whose UID/GID match the host's, so bind-mounted files stay owned by you
# and the shell can actually resolve a name. Override with
# --build-arg UID=$(id -u) --build-arg GID=$(id -g) if yours aren't 1000.
ARG USERNAME=dev
ARG UID=1000
ARG GID=1000
RUN groupadd --gid ${GID} ${USERNAME} \
    && useradd --uid ${UID} --gid ${GID} --create-home --shell /bin/bash ${USERNAME}

# --- Spinnaker SDK -----------------------------------------------------------
# FLIR gates the SDK behind an account login, so it cannot be downloaded here.
# It arrives through a named build context pointing at a host install:
#
#   docker build --build-context spinnaker=/opt/spinnaker -t perception_pipeline:dev .
#
# Everything but libusb is bundled in lib/. The copy is architecture-specific:
# an Orin build needs an arm64 SDK behind that same flag. Match major.minor,
# not the full build number -- FLIR versions the arch packages independently
# (x86_64 ships 4.2.0.88 where ARM64 ships 4.2.0.21), so there is no arm64
# package carrying the exact build string of the amd64 one.
# Placed above the source COPY so editing smoke/ never invalidates this layer.
COPY --from=spinnaker / /opt/spinnaker
RUN echo /opt/spinnaker/lib > /etc/ld.so.conf.d/spinnaker.conf && ldconfig

# The GenTL producer is how Spinnaker discovers USB3 and GigE transports;
# without this the SDK loads but enumerates zero cameras.
ENV GENICAM_GENTL64_PATH=/opt/spinnaker/lib/spinnaker-gentl

WORKDIR /workspace
COPY CMakeLists.txt ./
COPY cmake ./cmake
COPY geometry ./geometry
COPY capture ./capture
COPY recording ./recording
COPY stereo ./stereo
COPY spinnaker ./spinnaker
COPY tools ./tools
COPY pipeline ./pipeline
COPY processing ./processing
COPY tensorrt ./tensorrt
COPY visualization ./visualization
COPY app ./app
COPY smoke ./smoke
COPY tests ./tests

RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    && cmake --build build \
    && chown -R ${UID}:${GID} /workspace

# Everything above needed root (apt, useradd). Drop privileges for the default
# process; add `--user root` to a docker run if you need to install packages.
USER ${USERNAME}

CMD ["ctest", "--test-dir", "build", "--output-on-failure"]
