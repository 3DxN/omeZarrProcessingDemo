#!/usr/bin/env python
"""Open OME-Zarr v0.5 image data, compute a binary threshold, and write the
result as a new label group -- using ome-zarr-py for the pyramid/metadata I/O.

Same behaviour as threshold_label.py, but the pyramid + multiscales +
image-label metadata + label registration are delegated to
``ome_zarr.writer.write_multiscale_labels`` instead of being written by hand.
The Otsu computation and the nearest-neighbour pyramid are kept explicit.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import zarr
from ome_zarr.writer import write_multiscale_labels


def otsu_threshold(data: np.ndarray, nbins: int = 256) -> float:
    """Compute Otsu's threshold (no scikit-image dependency)."""
    counts, edges = np.histogram(data.ravel(), bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    valid = (weight1[:-1] > 0) & (weight2[1:] > 0)
    mean1 = np.cumsum(counts * centers)[:-1] / np.where(weight1[:-1] > 0, weight1[:-1], 1)
    mean2 = (np.cumsum((counts * centers)[::-1])[::-1] / np.where(weight2 > 0, weight2, 1))[1:]
    variance = weight1[:-1] * weight2[1:] * (mean1 - mean2) ** 2
    variance = np.where(valid, variance, -1.0)
    return float(centers[:-1][np.argmax(variance)])


def downsample_yx(mask: np.ndarray, y: int, x: int) -> np.ndarray:
    """Halve y and x with nearest-neighbour subsampling (never average labels)."""
    slicer = [slice(None)] * mask.ndim
    slicer[y] = slice(None, None, 2)
    slicer[x] = slice(None, None, 2)
    return mask[tuple(slicer)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zarr_path", nargs="?", default="6001240_labels.zarr")
    ap.add_argument("--channel", type=int, default=1,
                    help="channel index to threshold (default 1 = Dapi)")
    ap.add_argument("--label-name", default="threshold",
                    help="name of the new label group under labels/")
    ap.add_argument("--threshold", type=float, default=None,
                    help="fixed threshold value; Otsu is used when omitted")
    ap.add_argument("--scale-path", default="0",
                    help="multiscale level of the source image to read")
    ap.add_argument("--levels", type=int, default=None,
                    help="number of pyramid levels (default: match the source image)")
    args = ap.parse_args()

    # --- open the source OME-Zarr v0.5 image -----------------------------
    root = zarr.open_group(args.zarr_path, mode="a")
    image_meta = root.attrs["ome"]["multiscales"][0]
    axes = image_meta["axes"]
    axis_names = [a["name"] for a in axes]
    y_ax, x_ax = axis_names.index("y"), axis_names.index("x")

    src_dataset = next(d for d in image_meta["datasets"] if d["path"] == args.scale_path)
    base_scale = next(t for t in src_dataset["coordinateTransformations"]
                      if t["type"] == "scale")["scale"]
    src_paths = [d["path"] for d in image_meta["datasets"]]
    n_levels = args.levels if args.levels is not None else len(src_paths)

    image = root[args.scale_path]
    chan = image[args.channel]
    print(f"Read channel {args.channel}: shape={chan.shape} dtype={chan.dtype}")

    # --- compute the binary threshold ------------------------------------
    if args.threshold is None:
        thr = otsu_threshold(chan)
        print(f"Otsu threshold = {thr:.2f}")
    else:
        thr = args.threshold
        print(f"Fixed threshold = {thr:.2f}")

    mask = (chan > thr).astype(np.int8)[np.newaxis, ...]   # keep (c, z, y, x)
    print(f"Foreground voxels: {int(mask.sum())} / {mask.size}")

    # --- build the nearest-neighbour pyramid + per-level I/O options -----
    pyramid, transforms, storage_options = [], [], []
    level = mask
    for i in range(n_levels):
        src = root[src_paths[min(i, len(src_paths) - 1)]]
        pyramid.append(level)
        scale = list(base_scale)
        scale[y_ax] *= 2 ** i
        scale[x_ax] *= 2 ** i
        transforms.append([{"type": "scale", "scale": scale}])
        # mirror the source chunk grid (inner chunks + shard shape) per level
        storage_options.append({"chunks": src.chunks, "shards": src.shards})
        print(f"  level {i}: shape={level.shape} chunks={src.chunks} shards={src.shards}")
        if i + 1 < n_levels:
            level = downsample_yx(level, y_ax, x_ax)

    # --- delete any stale label group, then delegate the write -----------
    if "labels" in root and args.label_name in root["labels"]:
        del root["labels"][args.label_name]

    with warnings.catch_warnings():
        # coordinate_transformations is deprecated but is the only way to pin
        # the source's exact x2 scales (the `scale=` path infers from shapes).
        warnings.simplefilter("ignore", DeprecationWarning)
        write_multiscale_labels(
            pyramid,
            root,
            name=args.label_name,
            axes=axes,
            coordinate_transformations=transforms,
            storage_options=storage_options,
            label_metadata={
                "colors": [{"label-value": 1, "rgba": [255, 0, 0, 128]}],
                "source": {"image": "../.."},
            },
        )

    labels = list(root["labels"].attrs["ome"]["labels"])
    print(f"Wrote label group: labels/{args.label_name}")
    print(f"labels/ now contains: {labels}")


if __name__ == "__main__":
    main()
