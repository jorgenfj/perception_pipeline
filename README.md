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



### Camera access from the container (GigE)

Three things must be set **on the host**; a container cannot change them, and
each shows up as dropped or incomplete frames rather than an obvious error:

```bash
# Receive buffers. The default ~200 KB is far too small for a GigE stream.
sudo sysctl -w net.core.rmem_max=10485760
sudo sysctl -w net.core.rmem_default=10485760

# Jumbo frames on the camera NIC (replace enp3s0). Without this you pay a
# per-packet overhead on ~1500-byte payloads and lose bandwidth headroom.
sudo ip link set enp3s0 mtu 9000
```

Make them permanent via `/etc/sysctl.d/` and your netplan/NetworkManager config once the rig is settled.

## Build the image

```bash
docker build --build-context spinnaker=/opt/spinnaker -t perception_pipeline:dev .
```

The first run pulls ~2 GB of CUDA base image. Compilation smoke test happens during the
image build, so a successful `docker build` means the CUDA toolchain works.

`--build-context spinnaker=...` points at a host install of the Spinnaker SDK,
which is baked into the image at `/opt/spinnaker`.

Useful overrides:

```bash
# Iterate faster by building for one arch only
docker build --build-arg CUDA_ARCHS=87-real ...

# Pin a different CUDA / Ubuntu base
docker build --build-arg CUDA_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu22.04 ...

# See full compiler output instead of collapsed log lines
docker build --progress=plain ...
```

## Run the smoke tests

`smoke/` holds two toolchain checks — they verify that CUDA and Spinnaker are
found, compile, link, and load at runtime. They test the *environment*, not
pipeline behaviour. The image's default command runs both through ctest:

```bash
docker run --rm --network host perception_pipeline:dev
```

```
1/2 Test #1: cuda_smoke .......................   Passed
2/2 Test #2: spinnaker_smoke ..................   Passed
```

Or run them individually:

```bash
docker run --rm --network host perception_pipeline:dev ./build/bin/spinnaker_smoke
docker run --rm perception_pipeline:dev ./build/bin/cuda_smoke
```



## Development loop

Rebuilding the image for every source edit is slow. Bind-mount the working tree
into the container instead:

```bash
docker run --rm -it -h perception --network host \
  -v "$PWD":/workspace -w /workspace perception_pipeline:dev bash
```

Then, inside the container:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_CUDA_ARCHITECTURES="87-real;120-real;120-virtual"
cmake --build build
ctest --test-dir build --output-on-failure
```