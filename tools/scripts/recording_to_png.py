#!/usr/bin/env python3
"""Turn a stereo recording into PNGs you can flip through.

    ./spinnaker/tools/recording_to_png.py recordings/recording-2026-08-21T13-22-04

Writes one side-by-side PNG per pair, left eye beside right, with the pair id
and the skew burned into a header strip. That layout is the point: to judge
whether the rig is synced you compare what the two eyes saw at the same
claimed instant, and anything that moved between them shows up as a
displacement across the seam.

The pairing here is the same merge the C++ side does (spinnaker/include/
frame_pairing.hpp): a two-pointer walk over the two streams' camera
timestamps at a tolerance you choose at read time. Nothing in the file says
what pairs with what, deliberately -- so --tolerance-us re-pairs the same
recording differently, and that is a feature, not a knob to leave alone.

What this CANNOT tell you: whether the timestamps are honest. Both halves of
every comparison come from the cameras' own clocks, so a constant PTP offset
error shifts the exposures without shifting the numbers and the images will
look perfectly paired. To catch that you need something in the scene whose
timing comes from neither camera -- a marked fan at a known rpm is the cheap
version, and 3000 rpm gives 18 degrees per millisecond. Use a short exposure,
or the error is smeared into the same blur in both eyes.

Needs: numpy, pyyaml, pillow.
"""

import argparse
import pathlib
import struct
import sys

import numpy as np
import yaml
from PIL import Image, ImageDraw

# timestamp_ns, host_recv_ns, offset, bytes, frame_id -- see recording_format.hpp
INDEX_RECORD = struct.Struct("<QQQII")

# Where R and B sit in the 2x2 quad, as (row, col). The two greens are always
# the other diagonal.
BAYER = {
    "BayerRG8": ((0, 0), (1, 1)),
    "BayerGR8": ((0, 1), (1, 0)),
    "BayerGB8": ((1, 0), (0, 1)),
    "BayerBG8": ((1, 1), (0, 0)),
}


def load_stream(directory, info):
    """Index records plus a memmap of the payloads."""
    raw = (directory / info["index"]).read_bytes()
    if len(raw) % INDEX_RECORD.size:
        sys.exit(f"{info['index']}: not a whole number of 32-byte records")
    index = [INDEX_RECORD.unpack_from(raw, i) for i in range(0, len(raw), INDEX_RECORD.size)]
    data = np.memmap(directory / info["data"], dtype=np.uint8, mode="r")
    return index, data


def _correlate3(plane, kernel):
    """3x3 correlation, numpy only, no scipy.

    Reflect, not edge-replicate. A mosaic has a phase, and the border has to
    keep it: reflection puts row 1 at position -1, whose parity is what row -1
    would have had, so the interpolation sees the pattern it expects. Replicating
    row 0 instead puts a red row where a green one belongs and the outermost
    pixels come out as much as 2x bright -- which a flat grey test frame catches
    immediately and a photograph of a calibration target hides completely, right
    where the corner detector is most sensitive.
    """
    padded = np.pad(plane, 1, mode="reflect").astype(np.float32)
    out = np.zeros(plane.shape, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            weight = kernel[dy][dx]
            if weight:
                out += weight * padded[dy:dy + plane.shape[0], dx:dx + plane.shape[1]]
    return out


def demosaic_full(plane, fmt):
    """Bilinear demosaic at FULL sensor resolution.

    This is the one calibration needs. Halving the resolution halves the focal
    length and principal point with it, so intrinsics fitted on a decimated
    image are silently wrong by exactly 2x when applied to a full-frame one --
    and nothing downstream can detect that, because the numbers are all
    self-consistent.

    Bilinear, via the standard masked-convolution form: scatter each colour
    into its own plane and interpolate the holes. R and B sit on a quincunx so
    they take the [[1,2,1],[2,4,2],[1,2,1]]/4 kernel; G is already half the
    pixels and takes the smaller cross. Good enough for corner and circle-grid
    detection, which is what this is for -- it is not trying to beat the
    pipeline's CUDA debayer on image quality.
    """
    if fmt == "Mono8":
        return np.repeat(plane[:, :, None], 3, axis=2)

    (r_row, r_col), (b_row, b_col) = BAYER[fmt]

    rb_kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    g_kernel = [[0, 1, 0], [1, 4, 1], [0, 1, 0]]

    def scatter(row, col):
        out = np.zeros(plane.shape, dtype=np.float32)
        out[row::2, col::2] = plane[row::2, col::2]
        return out

    red = _correlate3(scatter(r_row, r_col), rb_kernel) / 4.0
    blue = _correlate3(scatter(b_row, b_col), rb_kernel) / 4.0

    green_sites = np.zeros(plane.shape, dtype=np.float32)
    green_sites[r_row::2, b_col::2] = plane[r_row::2, b_col::2]
    green_sites[b_row::2, r_col::2] = plane[b_row::2, r_col::2]
    green = _correlate3(green_sites, g_kernel) / 4.0

    return np.clip(np.dstack([red, green, blue]), 0, 255).astype(np.uint8)


def demosaic_quad(plane, fmt):
    """One RGB pixel per Bayer quad: half resolution, no interpolation.

    The same method the live viewer uses, so what comes out matches what was on
    screen. Cheap and free of interpolation artefacts, which makes it the right
    choice for eyeballing a pair -- and the wrong choice for calibration, see
    demosaic_full().
    """
    quad = [[plane[0::2, 0::2], plane[0::2, 1::2]], [plane[1::2, 0::2], plane[1::2, 1::2]]]

    if fmt == "Mono8":
        grey = (quad[0][0].astype(np.uint16) + quad[0][1] + quad[1][0] + quad[1][1]) // 4
        return np.repeat(grey.astype(np.uint8)[:, :, None], 3, axis=2)

    (r_row, r_col), (b_row, b_col) = BAYER[fmt]
    red = quad[r_row][r_col]
    blue = quad[b_row][b_col]
    # Averaged rather than picked: the two greens are the only redundancy this
    # method has, and using both halves the green noise for free.
    green = (quad[r_row][b_col].astype(np.uint16) + quad[b_row][r_col]) // 2

    return np.dstack([red, green.astype(np.uint8), blue])


def read_rgb(index, data, i, info, full):
    _, _, offset, nbytes, _ = index[i]
    frame = np.asarray(data[offset:offset + nbytes])

    fmt = info["pixel_format"]
    if fmt != "Mono8" and fmt not in BAYER:
        sys.exit(f"unsupported pixel_format {fmt!r} (Mono8 and the four Bayer8 orders)")

    plane = frame.reshape(info["height"], info["stride_bytes"])[:, :info["width"]]
    return demosaic_full(plane, fmt) if full else demosaic_quad(plane, fmt)


def pair_by_timestamp(a, b, tolerance_ns):
    """Two-pointer merge. Yields (i, j, skew_ns); j is None when unpaired."""
    i = j = 0
    while i < len(a) and j < len(b):
        d = a[i][0] - b[j][0]
        if abs(d) <= tolerance_ns:
            yield i, j, -d
            i += 1
            j += 1
        elif d < 0:
            yield i, None, 0
            i += 1
        else:
            yield None, j, 0
            j += 1
    while i < len(a):
        yield i, None, 0
        i += 1
    while j < len(b):
        yield None, j, 0
        j += 1


def compose(left, right, caption, gutter=8):
    """Two eyes side by side under a caption strip."""
    panels = [p for p in (left, right) if p is not None]
    height = max(p.shape[0] for p in panels)
    width = sum(p.shape[1] for p in panels) + gutter * (len(panels) - 1)
    strip = 18

    canvas = Image.new("RGB", (width, height + strip), (12, 12, 16))
    x = 0
    for panel in panels:
        canvas.paste(Image.fromarray(panel), (x, strip))
        x += panel.shape[1] + gutter

    ImageDraw.Draw(canvas).text((4, 4), caption, fill=(220, 220, 200))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", type=pathlib.Path)
    parser.add_argument("-o", "--out", type=pathlib.Path,
                        help="output directory (default: <recording>/png)")
    parser.add_argument("--tolerance-us", type=int, default=500,
                        help="pairing tolerance, microseconds (default 500). Must stay under "
                             "half the frame period or two exposures qualify as one instant")
    parser.add_argument("--max", type=int, default=0, help="stop after N images (0 = all)")
    parser.add_argument("--every", type=int, default=1, help="keep every Nth pair")
    parser.add_argument("--calib", action="store_true",
                        help="calibration export: full resolution, one file per camera, named "
                             "by PAIR index so <role>_000123.png in each camera's folder are "
                             "the same instant. Paired frames only")
    parser.add_argument("--half", action="store_true",
                        help="half resolution (one pixel per Bayer quad), for quick previews. "
                             "Never use for calibration: it halves the intrinsics with it")
    parser.add_argument("--separate", action="store_true",
                        help="one PNG per frame per camera instead of side-by-side pairs, named "
                             "by each stream's own frame index")
    parser.add_argument("--unpaired", action="store_true",
                        help="also write frames that found no partner (they are the evidence "
                             "when the rig is not syncing)")
    args = parser.parse_args()

    directory = args.recording
    try:
        manifest = yaml.safe_load((directory / "manifest.yaml").read_text())
    except FileNotFoundError:
        # The manifest is written last, on purpose: it carries the frame counts
        # and the shared epoch, neither of which is known until the writers
        # stop. Missing means the run was killed mid-recording.
        sys.exit(f"{directory}/manifest.yaml not found -- the recording was killed before it "
                 f"was closed, so its frame counts and epoch were never written")
    streams = manifest["streams"]
    if len(streams) != 2:
        sys.exit(f"expected a two-stream recording, found {len(streams)}")

    out = args.out or directory / ("calib" if args.calib else "png")
    out.mkdir(parents=True, exist_ok=True)

    # Full resolution unless explicitly asked otherwise. Losing half the sensor
    # silently is the worse failure of the two, so it is the one that has to be
    # opted into.
    full = not args.half
    if args.calib and args.half:
        sys.exit("--calib and --half contradict: calibration on half-resolution images fits "
                 "intrinsics that are wrong by 2x on the frames the pipeline actually uses")

    # Calibration tools want per-camera directories of corresponding images.
    calib_dirs = {}
    if args.calib:
        for stream, info in enumerate(streams):
            role = info.get("role") or f"cam{stream}"
            calib_dirs[stream] = out / role
            calib_dirs[stream].mkdir(parents=True, exist_ok=True)

    loaded = [load_stream(directory, info) for info in streams]
    tolerance_ns = args.tolerance_us * 1000

    written = paired = unpaired = 0
    worst_skew = 0
    kept = 0

    for i, j, skew in pair_by_timestamp(loaded[0][0], loaded[1][0], tolerance_ns):
        complete = i is not None and j is not None
        if complete:
            paired += 1
            worst_skew = max(worst_skew, abs(skew))
        else:
            unpaired += 1
            # Calibration needs correspondence above all else: a lone frame has
            # no partner to be stereo-calibrated against, so it is never
            # exported here regardless of --unpaired.
            if args.calib or not args.unpaired:
                continue

        kept += 1
        if (kept - 1) % args.every:
            continue
        if args.max and written >= args.max:
            break

        panels = []
        for stream, idx in enumerate((i, j)):
            index, data = loaded[stream]
            panels.append(None if idx is None
                          else read_rgb(index, data, idx, streams[stream], full))

        if args.calib:
            # Numbered by PAIR index, not by each stream's own frame index.
            # Those differ the moment either camera drops a frame, and a
            # calibrator handed left_000042 / right_000042 from different
            # instants fits a baseline that includes whatever moved between
            # them -- with no way to tell from the images that it happened.
            pair_index = paired - 1
            for stream, panel in enumerate(panels):
                role = streams[stream].get("role") or f"cam{stream}"
                Image.fromarray(panel).save(calib_dirs[stream] / f"{role}_{pair_index:06d}.png")
                written += 1
            continue

        if args.separate:
            for stream, panel in enumerate(panels):
                if panel is None:
                    continue
                name = f"cam{stream}_{(i if stream == 0 else j):06d}.png"
                Image.fromarray(panel).save(out / name)
                written += 1
            continue

        tag = f"{skew / 1e3:+.1f}us" if complete else "UNPAIRED"
        caption = (f"pair {paired - 1:06d}  {tag}"
                   if complete else f"frame {(i if i is not None else j):06d}  {tag}")
        if complete:
            caption += f"  t={loaded[0][0][i][0]}ns"
        compose(panels[0], panels[1], caption).save(
            out / f"{'pair' if complete else 'single'}_{written:06d}_{tag}.png")
        written += 1

    total = len(loaded[0][0]) + len(loaded[1][0])
    geometry = f"{streams[0]['width']}x{streams[0]['height']}" if full else \
               f"{streams[0]['width'] // 2}x{streams[0]['height'] // 2}"
    print(f"{written} png -> {out}  ({geometry}{'' if full else ', HALF resolution'})")
    print(f"paired={paired} unpaired={unpaired} (of {total} frames) "
          f"max_skew={worst_skew / 1e3:.1f}us at tolerance {args.tolerance_us}us")
    if args.calib:
        print(f"Matching numbers across the two folders are the same instant, so "
              f"<role>_000042.png\npairs with <role>_000042.png. Worst skew inside a pair is "
              f"{worst_skew / 1e3:.1f}us -- for a static\ntarget that is irrelevant, and for a "
              f"handheld one it is the blur budget.")
    if paired:
        print("Reminder: skew is computed from the cameras' own clocks. It cannot see a\n"
              "constant PTP offset error -- for that, put something in the scene whose\n"
              "timing comes from neither camera (a marked fan: 3000 rpm = 18 deg/ms).")


if __name__ == "__main__":
    main()
