#!/usr/bin/env python3
"""Look at what the rectification map actually does to a frame.

    ./tools/scripts/rectify_preview.py --calib app/config/stereo_calibration.yaml --synthetic
    ./tools/scripts/rectify_preview.py --recording recordings/recording-2026-08-26T13-36-12 --pair 40

Writes PNGs: the raw frame, the undistorted frame, the rectified pair with
epipolar rulings ruled across both eyes, the grid overlay that shows where the
distortion is, and a heatmap of how far each pixel moves.

THE MAP HERE IS THE PIPELINE'S MAP. build_rectify_map() below is a line-by-line
numpy transcription of processing/src/transforms/rectification.cu, not a call to
cv2.initUndistortRectifyMap. That is the whole point: a preview built with
OpenCV would look correct no matter what the CUDA kernel does, and the failures
worth catching -- a transposed R, a dropped tangential term, the wrong half
pixel -- all still produce a plausible image. Use --check to diff this map
against OpenCV's, if you have cv2 installed; that is a separate question from
what this tool is for.

The one deliberate difference from the .cu: the maps here are in pixel-index
coordinates, because that is what a numpy gather wants. The kernel's map holds
texture coordinates, which are the same numbers plus half a texel. See the
u_scale/v_scale block in build_rectify_map() there.

What to look for
  undistort  Straight edges in the scene are straight. The grid overlay on the
             raw frame shows the curves those straight lines came from -- at
             this rig's -0.244 k1 the corners pull in by ~30px, which is
             obvious once drawn and invisible otherwise.
  rectify    The rulings are the test. Pick a feature, find it in both eyes,
             and check it sits between the same two rulings. If it does not,
             the rectification is not giving a scanline matcher what it needs,
             and no amount of tuning downstream recovers it.

What this CANNOT tell you: whether the calibration is right. Both halves of the
comparison come from the same numbers, so a rectification that is confidently
wrong produces a confidently wrong pair of images that look fine side by side.
For this rig the calibration's own report says the rectified row error is 5.9px
RMS (19.9px max) because the mount moves -- see calib/calibration/
rig_motion_findings.md in the recording. Expect the rulings to disagree.

Needs: numpy, pyyaml, pillow. cv2 only for --check.
"""

import argparse
import pathlib
import sys

import numpy as np
import yaml
from PIL import Image, ImageDraw

# Same directory: the recording reader and the demosaic already exist there and
# a second copy of either is a second thing to keep in step.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from recording_to_png import load_stream, pair_by_timestamp, read_rgb  # noqa: E402


# --- calibration --------------------------------------------------------------

class Camera:
    """One eye: K, D, and the R/P that rectify it. Row-major, OpenCV order."""

    def __init__(self, role, K, D, R, P):
        self.role = role
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(D, dtype=np.float64).reshape(-1)
        self.R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        self.P = np.asarray(P, dtype=np.float64).reshape(3, 4)
        if self.D.size > 5:
            sys.exit(f"{role}: {self.D.size} distortion coefficients; this preview implements "
                     f"plumb_bob [k1 k2 p1 p2 k3] only, same as the CUDA map. Truncating them "
                     f"would produce a map that is wrong everywhere without looking wrong.")
        self.D = np.pad(self.D, (0, 5 - self.D.size))


def _opencv_yaml(text):
    """OpenCV's own dump, which is YAML 1.0 with a tag pyyaml does not know."""
    yaml.add_multi_constructor(
        "tag:yaml.org,2002:opencv-matrix",
        lambda loader, suffix, node: loader.construct_mapping(node, deep=True),
        Loader=yaml.SafeLoader)
    # `%YAML:1.0` -- a colon where the spec wants a space, so the directive has
    # to go before pyyaml sees it.
    return yaml.safe_load("\n".join(
        line for line in text.splitlines() if not line.startswith("%YAML")))


def load_calibration(path):
    """(width, height, [Camera, Camera]) from any of the three shapes we write.

    The pipeline's own schema (app/config/stereo_calibration.yaml), the offline
    calibrator's calibration.yaml, and OpenCV's calibration_opencv.yml all
    describe the same rig in different words. Accepting all three means this
    tool can be pointed straight at a fresh solve, before anyone has copied the
    numbers into the config -- which is exactly when you want to look at it.
    """
    text = pathlib.Path(path).read_text()
    root = _opencv_yaml(text) if text.lstrip().startswith("%YAML") else yaml.safe_load(text)

    if "cameras" in root:  # app/config/stereo_calibration.yaml
        size = root["image_size"]
        cams = []
        for entry in root["cameras"]:
            rect = entry["rectification"]
            cams.append(Camera(entry["role"], entry["camera_matrix"],
                               entry["distortion"]["coefficients"],
                               rect["rotation"], rect["projection"]))
        return size["width"], size["height"], cams

    if "left" in root and "rectification" in root:  # the calibrator's own dump
        rect = root["rectification"]
        width, height = root["left"]["image_size"]
        return width, height, [
            Camera("left", root["left"]["camera_matrix"], root["left"]["distortion"],
                   rect["R1"], rect["P1"]),
            Camera("right", root["right"]["camera_matrix"], root["right"]["distortion"],
                   rect["R2"], rect["P2"])]

    if "K1" in root:  # OpenCV's dump
        def m(key):
            return np.asarray(root[key]["data"], dtype=np.float64)
        return root["image_width"], root["image_height"], [
            Camera("left", m("K1"), m("D1"), m("R1"), m("P1")),
            Camera("right", m("K2"), m("D2"), m("R2"), m("P2"))]

    sys.exit(f"{path}: not a calibration this understands (no 'cameras', 'left' or 'K1' key)")


# --- the map ------------------------------------------------------------------

def build_rectify_map(camera, width, height, rectify=True):
    """Per output pixel, the source pixel it samples. Mirrors rectification.cu.

    `rectify=False` drops the rectifying rotation and the new projection, so the
    output is the same camera with the lens distortion removed and nothing else
    moved. That separation is the reason this tool exists: undistortion and
    rectification fail differently and fixing one is not fixing the other, but
    in a single composed image their effects are indistinguishable.
    """
    R = camera.R if rectify else np.eye(3)
    P = camera.P[:, :3] if rectify else camera.K

    # Rectified pixel -> ray in the rectified frame -> ray in this camera's own
    # frame, in one matrix. R^-1 undoes the rectifying rotation, P^-1 the
    # rectified projection.
    inverse = np.linalg.inv(P @ R)

    u, v = np.meshgrid(np.arange(width, dtype=np.float64),
                       np.arange(height, dtype=np.float64))
    rx = inverse[0, 0] * u + inverse[0, 1] * v + inverse[0, 2]
    ry = inverse[1, 0] * u + inverse[1, 1] * v + inverse[1, 2]
    rw = inverse[2, 0] * u + inverse[2, 1] * v + inverse[2, 2]
    x, y = rx / rw, ry / rw

    # Forward plumb_bob: radial in r^2, then the two tangential terms. Forward,
    # not inverse -- the map answers "where do I read from", so it distorts an
    # already-undistorted coordinate rather than undistorting a measured one.
    k1, k2, p1, p2, k3 = camera.D
    r2 = x * x + y * y
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xy2 = 2.0 * x * y
    xd = x * radial + p1 * xy2 + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + p2 * xy2

    K = camera.K
    return (K[0, 0] * xd + K[0, 2]).astype(np.float32), \
           (K[1, 1] * yd + K[1, 2]).astype(np.float32)


def remap_bilinear(src, map_u, map_v, fill=(20, 20, 26)):
    """Gather with bilinear weights; out-of-frame samples get `fill`.

    The kernel does not do this -- it lets the texture's cudaAddressModeClamp
    replicate the edge, which is cheaper and produces a smeared border. Painting
    those pixels instead is a preview decision: smeared border and real image
    look alike, and the whole question here is how much of the output is which.
    """
    h, w = src.shape[:2]
    valid = (map_u >= 0) & (map_u <= w - 1) & (map_v >= 0) & (map_v <= h - 1)

    x0 = np.clip(np.floor(map_u), 0, w - 1)
    y0 = np.clip(np.floor(map_v), 0, h - 1)
    fx = (map_u - x0)[..., None]
    fy = (map_v - y0)[..., None]
    x0 = x0.astype(np.intp)
    y0 = y0.astype(np.intp)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    src = src.astype(np.float32)
    top = src[y0, x0] * (1 - fx) + src[y0, x1] * fx
    bottom = src[y1, x0] * (1 - fx) + src[y1, x1] * fx
    out = top * (1 - fy) + bottom * fy

    out = np.clip(out, 0, 255).astype(np.uint8)
    out[~valid] = fill
    return out, valid


# --- drawing ------------------------------------------------------------------

RULE_COLOURS = [(255, 92, 92), (92, 220, 255)]
STRIP = 20


def overlay_source_grid(image, map_u, map_v, step, colour=(80, 255, 140)):
    """Draw, on the SOURCE frame, the curves that straight output lines came from.

    This is the picture of the distortion. Every one of these polylines is a
    straight row or column of the rectified output; how far it bows is exactly
    what the map is undoing, in pixels, on the frame it is undoing it to.
    """
    canvas = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(canvas)
    h, w = map_u.shape

    for row in range(0, h, step):
        draw.line(list(zip(map_u[row, :].tolist(), map_v[row, :].tolist())), fill=colour, width=1)
    for col in range(0, w, step):
        draw.line(list(zip(map_u[:, col].tolist(), map_v[:, col].tolist())), fill=colour, width=1)
    return np.asarray(canvas)


def overlay_straight_grid(image, step, colour=(80, 255, 140)):
    """The same grid on the output, where it is straight by construction."""
    canvas = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(canvas)
    h, w = image.shape[:2]
    for row in range(0, h, step):
        draw.line([(0, row), (w - 1, row)], fill=colour, width=1)
    for col in range(0, w, step):
        draw.line([(col, 0), (col, h - 1)], fill=colour, width=1)
    return np.asarray(canvas)


def _contour_levels(peak):
    """Round pixel values, roughly six of them, spanning 0..peak."""
    if peak <= 0:
        return []
    for step in (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500):
        if peak / step <= 8:
            break
    return [step * i for i in range(1, int(peak / step) + 1)]


def displacement_heatmap(map_u, map_v):
    """How far each pixel moves, in pixels: a hot ramp with iso-pixel contours.

    Scaled to its own peak rather than a fixed range -- the peak differs by an
    order of magnitude between undistortion (tens of pixels, at the corners) and
    rectification (hundreds, once the rig needs rotating). That makes the colour
    alone unreadable as a quantity, which is what the contours are for: each one
    is a round number of pixels, so the picture can be measured and not just
    admired.
    """
    h, w = map_u.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    magnitude = np.hypot(map_u - u, map_v - v)
    peak = float(magnitude.max())

    t = magnitude / peak if peak > 0 else np.zeros_like(magnitude)
    # Black -> red -> yellow -> white. Three linear segments, no colormap
    # dependency, and monotone in luminance so it survives being printed.
    red = np.clip(t * 3.0, 0, 1)
    green = np.clip(t * 3.0 - 1.0, 0, 1)
    blue = np.clip(t * 3.0 - 2.0, 0, 1)
    image = (np.dstack([red, green, blue]) * 255).astype(np.uint8)

    # A level's contour is the boundary of the region above it: threshold, then
    # mark where the mask changes between neighbours. One pixel wide, and it
    # closes on itself without any marching-squares machinery.
    levels = _contour_levels(peak)
    for level in levels:
        above = magnitude >= level
        edge = np.zeros_like(above)
        edge[:, :-1] |= above[:, :-1] ^ above[:, 1:]
        edge[:-1, :] |= above[:-1, :] ^ above[1:, :]
        image[edge] = (0, 200, 255)

    return image, peak, levels


def compose(panels, caption, rulings=0, gutter=10):
    """Panels side by side under a caption, optionally ruled across the seam.

    The rulings run over both panels in one stroke on purpose. A ruling drawn
    per panel proves nothing -- it is the continuity across the gutter that
    says the two eyes agree on what row a thing is in.
    """
    panels = [p for p in panels if p is not None]
    height = max(p.shape[0] for p in panels)
    width = sum(p.shape[1] for p in panels) + gutter * (len(panels) - 1)

    canvas = Image.new("RGB", (width, height + STRIP), (12, 12, 16))
    x = 0
    for panel in panels:
        canvas.paste(Image.fromarray(panel), (x, STRIP))
        x += panel.shape[1] + gutter

    draw = ImageDraw.Draw(canvas)
    if rulings:
        for i, row in enumerate(range(0, height, rulings)):
            draw.line([(0, STRIP + row), (width - 1, STRIP + row)],
                      fill=RULE_COLOURS[i % len(RULE_COLOURS)], width=1)
    draw.text((4, 5), caption, fill=(220, 220, 200))
    return canvas


def synthetic_frame(width, height, square=120):
    """A checkerboard with a ruled border, for looking at a map with no capture.

    Distortion is a statement about the lens, not about the scene, so it is
    fully visible on a synthetic frame -- and a checkerboard shows it far better
    than a photograph does, because every edge is known to be straight.
    """
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    board = (((u // square) + (v // square)) % 2).astype(np.uint8)
    image = np.dstack([board * 200 + 30] * 3)

    # A few strong straight lines near the edges, where distortion is largest.
    for row in (2, height // 2, height - 3):
        image[row, :] = (255, 80, 80)
    for col in (2, width // 2, width - 3):
        image[:, col] = (80, 160, 255)
    return image


# --- frames -------------------------------------------------------------------

def frames_from_recording(directory, pair_index, tolerance_us):
    """The nth *paired* frame from each eye, demosaiced at full resolution."""
    manifest = yaml.safe_load((directory / "manifest.yaml").read_text())
    streams = manifest["streams"]
    if len(streams) != 2:
        sys.exit(f"expected a two-stream recording, found {len(streams)}")

    loaded = [load_stream(directory, info) for info in streams]
    paired = 0
    for i, j, skew in pair_by_timestamp(loaded[0][0], loaded[1][0], tolerance_us * 1000):
        if i is None or j is None:
            continue
        if paired == pair_index:
            return [read_rgb(loaded[s][0], loaded[s][1], idx, streams[s], True)
                    for s, idx in ((0, i), (1, j))], skew
        paired += 1
    sys.exit(f"--pair {pair_index}: the recording has only {paired} paired frames")


def check_against_opencv(camera, width, height, map_u, map_v, rectify):
    try:
        import cv2
    except ImportError:
        print("  --check: cv2 not installed, skipped")
        return
    R = camera.R if rectify else np.eye(3)
    P = camera.P[:, :3] if rectify else camera.K
    ref_u, ref_v = cv2.initUndistortRectifyMap(camera.K, camera.D, R, P,
                                               (width, height), cv2.CV_32FC1)
    worst = max(float(np.abs(map_u - ref_u).max()), float(np.abs(map_v - ref_v).max()))
    verdict = "matches" if worst < 1e-3 else "DIFFERS"
    print(f"  --check {camera.role} {'rectify' if rectify else 'undistort'}: "
          f"{verdict} cv2.initUndistortRectifyMap, worst {worst:.2e}px")


# --- main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calib", type=pathlib.Path,
                        default=pathlib.Path("app/config/stereo_calibration.yaml"),
                        help="calibration file: the pipeline's schema, the calibrator's "
                             "calibration.yaml, or OpenCV's calibration_opencv.yml "
                             "(default: app/config/stereo_calibration.yaml)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--recording", type=pathlib.Path,
                        help="a stereo recording directory; uses its nth paired frame")
    source.add_argument("--images", type=pathlib.Path, nargs=2, metavar=("LEFT", "RIGHT"),
                        help="two image files, in stream order")
    source.add_argument("--synthetic", action="store_true",
                        help="a checkerboard instead of a capture. Distortion is a property of "
                             "the lens, so this shows the map in full without needing a frame")
    parser.add_argument("--pair", type=int, default=0, help="which paired frame (default 0)")
    parser.add_argument("--tolerance-us", type=int, default=500,
                        help="pairing tolerance for --recording, microseconds (default 500)")
    parser.add_argument("-o", "--out", type=pathlib.Path, default=pathlib.Path("rectify_preview"),
                        help="output directory (default: ./rectify_preview)")
    parser.add_argument("--grid-step", type=int, default=80,
                        help="grid overlay spacing, pixels (default 80)")
    parser.add_argument("--rulings", type=int, default=60,
                        help="epipolar ruling spacing on the rectified pair, pixels "
                             "(default 60; 0 turns them off)")
    parser.add_argument("--check", action="store_true",
                        help="also diff every map against cv2.initUndistortRectifyMap")
    args = parser.parse_args()

    width, height, cameras = load_calibration(args.calib)
    print(f"calibration: {args.calib}  {width}x{height}  "
          f"{cameras[0].role}|{cameras[1].role}")

    skew = None
    if args.recording:
        frames, skew = frames_from_recording(args.recording, args.pair, args.tolerance_us)
        origin = f"{args.recording.name} pair {args.pair}"
    elif args.images:
        frames = [np.asarray(Image.open(p).convert("RGB")) for p in args.images]
        origin = " / ".join(p.name for p in args.images)
    else:
        frames = [synthetic_frame(width, height)] * 2
        origin = "synthetic checkerboard"

    for eye, frame in enumerate(frames):
        if frame.shape[1] != width or frame.shape[0] != height:
            sys.exit(f"{cameras[eye].role}: frame is {frame.shape[1]}x{frame.shape[0]} but the "
                     f"calibration was solved at {width}x{height}. Intrinsics are in pixels, so "
                     f"a frame at another size is not a scale factor away from correct.")

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"source: {origin}")

    rectified = []
    for eye, (camera, frame) in enumerate(zip(cameras, frames)):
        role = camera.role
        print(f"{role}:")

        for rectify, tag in ((False, "undistort"), (True, "rectify")):
            map_u, map_v = build_rectify_map(camera, width, height, rectify)
            out, valid = remap_bilinear(frame, map_u, map_v)
            outside = int((~valid).sum())
            heat, peak, levels = displacement_heatmap(map_u, map_v)

            print(f"  {tag:<9} max shift {peak:7.1f}px   "
                  f"outside frame {outside} px ({100.0 * outside / valid.size:.2f}%)")
            if args.check:
                check_against_opencv(camera, width, height, map_u, map_v, rectify)

            # Raw beside output, with the same grid drawn on both: curved where
            # it was read from, straight where it was written to.
            compose([overlay_source_grid(frame, map_u, map_v, args.grid_step),
                     overlay_straight_grid(out, args.grid_step)],
                    f"{role} {tag}: source grid (curved) -> output grid (straight), "
                    f"max shift {peak:.1f}px").save(args.out / f"{role}_{tag}_grid.png")

            Image.fromarray(out).save(args.out / f"{role}_{tag}.png")
            contours = ", ".join(f"{lv:g}" for lv in levels) or "none"
            compose([heat], f"{role} {tag}: pixel displacement, black=0 white={peak:.1f}px, "
                            f"contours at {contours}px") \
                .save(args.out / f"{role}_{tag}_field.png")

            if rectify:
                rectified.append(out)

        Image.fromarray(frame).save(args.out / f"{role}_raw.png")

    # The pair, ruled. This is the one to actually look at.
    caption = f"rectified pair -- {origin}"
    if skew is not None:
        caption += f"  skew {skew / 1e3:+.1f}us"
    if args.rulings:
        caption += f"  rulings every {args.rulings}px"
    compose(rectified, caption, rulings=args.rulings).save(args.out / "pair_rectified.png")

    compose(frames, f"raw pair -- {origin}", rulings=args.rulings) \
        .save(args.out / "pair_raw.png")

    print(f"-> {args.out}")
    print("Look at pair_rectified.png: pick a feature, find it in both eyes, and check it\n"
          "sits between the same two rulings. Rows that disagree are rows a scanline\n"
          "matcher cannot use, and nothing downstream recovers them.")


if __name__ == "__main__":
    main()
