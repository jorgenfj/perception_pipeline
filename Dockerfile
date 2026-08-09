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
        clangd \
        libusb-1.0-0 \
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
# an Orin build needs an arm64 SDK of the same version behind that same flag.
# Placed above the source COPY so editing smoke/ never invalidates this layer.
COPY --from=spinnaker / /opt/spinnaker
RUN echo /opt/spinnaker/lib > /etc/ld.so.conf.d/spinnaker.conf && ldconfig

# The GenTL producer is how Spinnaker discovers USB3 and GigE transports;
# without this the SDK loads but enumerates zero cameras.
ENV GENICAM_GENTL64_PATH=/opt/spinnaker/lib/spinnaker-gentl

WORKDIR /workspace
COPY CMakeLists.txt ./
COPY cmake ./cmake
COPY include ./include
COPY src ./src
COPY smoke ./smoke

RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    && cmake --build build \
    && chown -R ${UID}:${GID} /workspace

# Everything above needed root (apt, useradd). Drop privileges for the default
# process; add `--user root` to a docker run if you need to install packages.
USER ${USERNAME}

CMD ["ctest", "--test-dir", "build", "--output-on-failure"]
