#!/usr/bin/env python
"""Out-of-core binary-threshold labels for OME-Zarr v0.5 using dask.

Unlike threshold_label.py / threshold_label_omezarr.py -- which read the whole
channel into RAM -- this variant keeps the data lazy end-to-end:

  * the source is read with ome-zarr-py's high-level `OMEZarrMultiscale`, which
    yields one lazy dask array per pyramid level plus parsed axes / scale,
  * Otsu is computed from a streamed dask histogram (only 256 bins land in RAM),
  * the mask + pyramid stay lazy dask graphs, and
  * ome-zarr-py's `write_multiscale_labels` streams each chunk to disk via
    `da.to_zarr`, so peak memory is a few chunks, not the whole volume.

`OMEZarrMultiscale` handles read-side metadata, but `zarr` is still used to
WRITE (`write_multiscale_labels` needs a writable group) and to read each source
level's shard shape -- the high-level API only exposes inner-chunk granularity,
not the shard grid.

The Otsu maths is identical to the numpy version in threshold_label.py; here it
is fed a precomputed histogram instead of a numpy array.
"""
from __future__ import annotations

import argparse
import warnings

import dask.array as da
import numpy as np
import zarr
from ome_zarr import OMEZarrMultiscale
from ome_zarr.writer import write_multiscale_labels


def otsu_from_histogram(counts: np.ndarray, centers: np.ndarray) -> float:
    """Otsu's threshold from a histogram (same maths as threshold_label.py)."""
    counts = counts.astype(np.float64)
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    valid = (weight1[:-1] > 0) & (weight2[1:] > 0)
    mean1 = np.cumsum(counts * centers)[:-1] / np.where(weight1[:-1] > 0, weight1[:-1], 1)
    mean2 = (np.cumsum((counts * centers)[::-1])[::-1] / np.where(weight2 > 0, weight2, 1))[1:]
    variance = weight1[:-1] * weight2[1:] * (mean1 - mean2) ** 2
    variance = np.where(valid, variance, -1.0)
    return float(centers[:-1][np.argmax(variance)])


def otsu_threshold_dask(chan: da.Array, nbins: int = 256) -> float:
    """Streamed Otsu: one pass for min/max, one pass for the histogram.

    Matches np.histogram(data, bins=nbins)'s default [min, max] range, so the
    resulting threshold is identical to the in-memory implementation.
    """
    vmin, vmax = da.compute(chan.min(), chan.max())
    counts, edges = da.histogram(chan, bins=nbins, range=[float(vmin), float(vmax)])
    counts = counts.compute()
    centers = (edges[:-1] + edges[1:]) / 2.0
    return otsu_from_histogram(counts, centers)


def downsample_yx(mask: da.Array, y: int, x: int) -> da.Array:
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

    # --- read the source image via ome-zarr-py's high-level API ----------
    # OMEZarrMultiscale.from_ome_zarr gives one lazy dask array per pyramid
    # level (`.images[i].data`) plus parsed axes / scale -- no hand-parsing.
    ms = OMEZarrMultiscale.from_ome_zarr(args.zarr_path)
    axes = [a.model_dump(exclude_none=True) for a in ms.metadata.axes]  # full dicts (+units)
    axis_names = [a["name"] for a in axes]
    y_ax, x_ax = axis_names.index("y"), axis_names.index("x")

    # zarr is still needed to WRITE and to read each level's shard shape
    # (the high-level API exposes inner-chunk granularity, not the shard grid).
    root = zarr.open_group(args.zarr_path, mode="a")
    src_paths = [d["path"] for d in root.attrs["ome"]["multiscales"][0]["datasets"]]

    scale_index = src_paths.index(args.scale_path)
    img = ms.images[scale_index]
    base_scale = [img.scale[name] for name in axis_names]   # source's exact scales
    n_levels = args.levels if args.levels is not None else len(ms.images)

    chan = img.data[args.channel]                   # lazy (z, y, x)
    print(f"Channel {args.channel}: shape={chan.shape} dtype={chan.dtype} "
          f"chunks={chan.chunksize} (lazy dask array via OMEZarrMultiscale)")

    # --- compute the binary threshold (streamed) -------------------------
    if args.threshold is None:
        thr = otsu_threshold_dask(chan)
        print(f"Otsu threshold = {thr:.2f}")
    else:
        thr = args.threshold
        print(f"Fixed threshold = {thr:.2f}")

    # mask stays lazy; keep the (c, z, y, x) layout with a size-1 channel axis
    mask = (chan > thr).astype(np.int8)[np.newaxis, ...]

    # --- build the lazy nearest-neighbour pyramid ------------------------
    pyramid, transforms, storage_options = [], [], []
    level = mask
    for i in range(n_levels):
        src = root[src_paths[min(i, len(src_paths) - 1)]]
        # align each level's dask blocks to its shard shape for streamed writes
        shard = tuple(min(s, d) for s, d in zip(src.shards, level.shape))
        pyramid.append(level.rechunk(shard))
        scale = list(base_scale)
        scale[y_ax] *= 2 ** i
        scale[x_ax] *= 2 ** i
        transforms.append([{"type": "scale", "scale": scale}])
        storage_options.append({"chunks": src.chunks, "shards": src.shards})
        print(f"  level {i}: shape={level.shape} chunks={src.chunks} shards={src.shards}")
        if i + 1 < n_levels:
            level = downsample_yx(level, y_ax, x_ax)

    # --- delete any stale label group, then stream to disk ---------------
    if "labels" in root and args.label_name in root["labels"]:
        del root["labels"][args.label_name]

    with warnings.catch_warnings():
        # coordinate_transformations is deprecated but is the only way to pin
        # the source's exact x2 scales (the `scale=` path infers from shapes).
        warnings.simplefilter("ignore", DeprecationWarning)
        write_multiscale_labels(
            pyramid,                     # dask arrays -> da.to_zarr streams them
            root,
            name=args.label_name,
            axes=axes,
            coordinate_transformations=transforms,
            storage_options=storage_options,
            label_metadata={
                "colors": [{"label-value": 1, "rgba": [255, 0, 0, 128]}],
                "source": {"image": "../.."},
            },
            compute=True,
        )

    labels = list(root["labels"].attrs["ome"]["labels"])
    print(f"Wrote label group: labels/{args.label_name}")
    print(f"labels/ now contains: {labels}")


if __name__ == "__main__":
    main()
