# perception_pipeline

## GPU targets

| Arch | Hardware | Notes |
| --- | --- | --- |
| `sm_87` | Jetson AGX Orin / Orin NX / Orin Nano | JetPack 6 userspace is Ubuntu 22.04 |
| `sm_120` | Blackwell, RTX 50-series | Requires CUDA >= 12.8 |

`CMAKE_CUDA_ARCHITECTURES` defaults to `87-real;120-real;120-virtual`: native
machine code (SASS) for both targets so they load without JIT, plus PTX for
Blackwell so the binary still runs on future architectures. Overriding it is how
you trim build time while iterating — see below.


## Build the image

```bash
docker build -t perception_pipeline:dev .
```

The first run pulls ~2 GB of CUDA base image. Compilation smoke test happens during the
image build, so a successful `docker build` means the CUDA toolchain works.

Useful overrides:

```bash
# Iterate faster by building for one arch only
docker build --build-arg CUDA_ARCHS=87-real -t perception_pipeline:dev .

# Pin a different CUDA / Ubuntu base
docker build --build-arg CUDA_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu22.04 .

# See full compiler output instead of collapsed log lines
docker build --progress=plain -t perception_pipeline:dev .
```

## Run the smoke test

```bash
docker run --rm perception_pipeline:dev
```

On a machine without an NVIDIA GPU the expected output is:

```
compiled and linked OK, no CUDA device available (...)
```

That is a pass — it exits 0. On a machine with a GPU and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/)
installed, expose the device to actually execute the kernel:

```bash
docker run --rm --gpus all perception_pipeline:dev
```

which prints the device name, its compute capability, and the saxpy result.

## Development loop

Rebuilding the image for every source edit is slow. Bind-mount the working tree
into the container instead:

```bash
docker run --rm -it -h perception \
  -v "$PWD":/workspace -w /workspace perception_pipeline:dev bash
```

Then, inside the container:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_CUDA_ARCHITECTURES="87-real;120-real;120-virtual"
cmake --build build
./build/cuda_smoke
```