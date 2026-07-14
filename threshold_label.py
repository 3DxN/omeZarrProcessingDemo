#!/usr/bin/env python
"""Open OME-Zarr v0.5 image data, compute a binary threshold, and write the
result as a new label group inside the same OME-Zarr store.

Usage:
    python threshold_label.py [ZARR] [--channel N] [--label-name NAME]
                              [--threshold VALUE]

Defaults:
    ZARR         6001240_labels.zarr
    --channel    1              (0=LaminB1, 1=Dapi)
    --label-name threshold
    --threshold  Otsu (computed automatically when not given)
"""
from __future__ import annotations

import argparse

import numpy as np
import zarr


def otsu_threshold(data: np.ndarray, nbins: int = 256) -> float:
    """Compute Otsu's threshold (no scikit-image dependency)."""
    counts, edges = np.histogram(data.ravel(), bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    # Guard against divide-by-zero at the histogram ends.
    valid = (weight1[:-1] > 0) & (weight2[1:] > 0)
    mean1 = np.cumsum(counts * centers)[:-1] / np.where(weight1[:-1] > 0, weight1[:-1], 1)
    mean2 = (np.cumsum((counts * centers)[::-1])[::-1] / np.where(weight2 > 0, weight2, 1))[1:]
    variance = weight1[:-1] * weight2[1:] * (mean1 - mean2) ** 2
    variance = np.where(valid, variance, -1.0)
    return float(centers[:-1][np.argmax(variance)])


def downsample_yx(mask: np.ndarray, y: int, x: int) -> np.ndarray:
    """Halve the y and x axes with nearest-neighbour subsampling.

    Labels must never be averaged (that invents non-existent label values),
    so we take every other voxel along y and x -- matching the source image,
    which only downscales the spatial y/x axes and keeps z constant.
    """
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
                    help="number of pyramid levels to write "
                         "(default: match the source image)")
    args = ap.parse_args()

    # --- open the source OME-Zarr v0.5 image -----------------------------
    root = zarr.open_group(args.zarr_path, mode="a")
    image_meta = root.attrs["ome"]["multiscales"][0]
    axes = image_meta["axes"]
    # coordinateTransformations of the level we read, reused for the label.
    src_dataset = next(d for d in image_meta["datasets"] if d["path"] == args.scale_path)
    transforms = src_dataset["coordinateTransformations"]

    # Default: give the label the same number of pyramid levels as the source.
    n_levels = args.levels if args.levels is not None else len(image_meta["datasets"])

    image = root[args.scale_path]            # zarr array, axes (c, z, y, x)
    chan = image[args.channel]               # -> (z, y, x) as numpy array
    print(f"Read channel {args.channel}: shape={chan.shape} dtype={chan.dtype}")

    # --- compute the binary threshold ------------------------------------
    if args.threshold is None:
        thr = otsu_threshold(chan)
        print(f"Otsu threshold = {thr:.2f}")
    else:
        thr = args.threshold
        print(f"Fixed threshold = {thr:.2f}")

    mask = (chan > thr).astype(np.int8)       # binary 0/1 label image
    # keep the channel axis (size 1) to match the store's (c, z, y, x) layout
    mask = mask[np.newaxis, ...]
    print(f"Foreground voxels: {int(mask.sum())} / {mask.size}")

    # --- write the new label group ---------------------------------------
    labels_group = root.require_group("labels")
    label_group = labels_group.create_group(args.label_name, overwrite=True)

    axis_names = [a["name"] for a in axes]
    y_ax, x_ax = axis_names.index("y"), axis_names.index("x")
    base_scale = next(t for t in transforms if t["type"] == "scale")["scale"]
    dim_names = axis_names

    # Mirror the source image's chunk grid so the label chunking corresponds to
    # the source. The source shards a (1,10,512,512) shard into (1,1,256,256)
    # inner chunks; reuse the matching source level's chunks/shards per level
    # (falling back to the deepest source level for any extra label levels).
    src_paths = [d["path"] for d in image_meta["datasets"]]

    def source_chunking(i: int):
        src = root[src_paths[min(i, len(src_paths) - 1)]]
        return src.chunks, src.shards

    # Build the pyramid: level 0 is the full-resolution mask; each subsequent
    # level halves y/x (nearest-neighbour) and doubles the y/x scale factor.
    datasets = []
    level = mask
    for i in range(n_levels):
        chunks, shards = source_chunking(i)
        arr = label_group.create_array(
            name=str(i),
            shape=level.shape,
            chunks=chunks,
            shards=shards,
            dtype="int8",
            dimension_names=dim_names,
            fill_value=0,
        )
        arr[:] = level

        scale = list(base_scale)
        scale[y_ax] *= 2 ** i
        scale[x_ax] *= 2 ** i
        datasets.append({
            "path": str(i),
            "coordinateTransformations": [{"type": "scale", "scale": scale}],
        })
        print(f"  level {i}: shape={level.shape} chunks={chunks} shards={shards}")

        if i + 1 < n_levels:
            level = downsample_yx(level, y_ax, x_ax)

    # OME-Zarr v0.5 metadata for the label image
    label_group.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [{
            "name": args.label_name,
            "axes": axes,
            "datasets": datasets,
        }],
        "image-label": {
            "version": "0.5",
            "source": {"image": "../.."},
            "colors": [{"label-value": 1, "rgba": [255, 0, 0, 128]}],
        },
    }

    # register the new label in labels/ so viewers discover it
    existing = list(labels_group.attrs.get("ome", {}).get("labels", []))
    if args.label_name not in existing:
        existing.append(args.label_name)
    labels_group.attrs["ome"] = {"version": "0.5", "labels": existing}

    print(f"Wrote label group: labels/{args.label_name}")
    print(f"labels/ now contains: {existing}")


if __name__ == "__main__":
    main()
