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

`OMEZarrMultiscale` handles read-side metadata -- including each level's inner
chunk shape, exposed as `.images[i].data.chunksize` -- and the write goes through
`write_multiscale_labels` with a path string, so this script has **no direct
`zarr` dependency**. The inner CHUNK shape is mirrored from the source image; the
SHARD shape is that chunk times a user-defined factor per axis (`--shard-*`,
default c=1 z=4 y=4 x=4 t=1), or disabled with `--no-shard`. (Only the source's
*shard* shape is unreachable through the high-level API; its chunk shape is not.)

The Otsu maths is identical to the numpy version in threshold_label.py; here it
is fed a precomputed histogram instead of a numpy array.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import dask.array as da
import numpy as np
from ome_zarr import OMEZarrMultiscale
from ome_zarr.writer import write_multiscale_labels


def rmtree(path: Path) -> None:
    """Recursively delete a directory tree (pathlib-only, no shutil)."""
    for child in path.iterdir():
        rmtree(child) if child.is_dir() else child.unlink()
    path.rmdir()


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
    # Sharding factors: shard shape = source chunk shape x factor, per axis.
    ap.add_argument("--shard-c", type=int, default=1, help="chunks per shard along c")
    ap.add_argument("--shard-z", type=int, default=4, help="chunks per shard along z")
    ap.add_argument("--shard-y", type=int, default=4, help="chunks per shard along y")
    ap.add_argument("--shard-x", type=int, default=4, help="chunks per shard along x")
    ap.add_argument("--shard-t", type=int, default=1, help="chunks per shard along t")
    ap.add_argument("--no-shard", action="store_true",
                    help="write unsharded (chunks only, no shard grid)")
    args = ap.parse_args()

    shard_factor = {"c": args.shard_c, "z": args.shard_z, "y": args.shard_y,
                    "x": args.shard_x, "t": args.shard_t}

    # --- read the source image via ome-zarr-py's high-level API ----------
    # OMEZarrMultiscale.from_ome_zarr gives one lazy dask array per pyramid
    # level (`.images[i].data`) plus parsed axes / scale -- no hand-parsing.
    ms = OMEZarrMultiscale.from_ome_zarr(args.zarr_path)
    axes = [a.model_dump(exclude_none=True) for a in ms.metadata.axes]  # full dicts (+units)
    axis_names = [a["name"] for a in axes]
    y_ax, x_ax = axis_names.index("y"), axis_names.index("x")

    src_paths = [d.path for d in ms.metadata.datasets]
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
    n_src = len(ms.images)
    pyramid, transforms, storage_options = [], [], []
    level = mask
    for i in range(n_levels):
        # CHUNK: mirror the source level's inner chunk shape (high-level API).
        # `data.chunksize` is dask's effective chunk: at sub-levels where the
        # source's declared chunk (256) exceeds the axis, it reports the clamped
        # dim (e.g. 137) -- functionally the same single chunk, no zarr needed.
        chunk = tuple(ms.images[min(i, n_src - 1)].data.chunksize)
        opts = {"chunks": chunk}
        write_chunks = chunk
        if not args.no_shard:
            # SHARD: chunk x the user-defined per-axis factor.
            shard = tuple(c * shard_factor.get(name, 1)
                          for c, name in zip(chunk, axis_names))
            opts["shards"] = shard
            write_chunks = shard          # one dask block per shard for streamed writes
        pyramid.append(level.rechunk(write_chunks))
        storage_options.append(opts)

        scale = list(base_scale)
        scale[y_ax] *= 2 ** i
        scale[x_ax] *= 2 ** i
        transforms.append([{"type": "scale", "scale": scale}])
        print(f"  level {i}: shape={level.shape} chunks={chunk} shards={opts.get('shards')}")
        if i + 1 < n_levels:
            level = downsample_yx(level, y_ax, x_ax)

    # --- delete any stale label group, then stream to disk ---------------
    # write_multiscale_labels uses require_group (no overwrite), so remove any
    # prior label dir first to avoid leftover levels. Local-store assumption:
    # the write side targets a filesystem path.
    stale = Path(args.zarr_path) / "labels" / args.label_name
    if stale.is_dir():
        rmtree(stale)

    with warnings.catch_warnings():
        # coordinate_transformations is deprecated but is the only way to pin
        # the source's exact x2 scales (the `scale=` path infers from shapes).
        warnings.simplefilter("ignore", DeprecationWarning)
        write_multiscale_labels(
            pyramid,                     # dask arrays -> da.to_zarr streams them
            args.zarr_path,              # path string -> no zarr.Group needed
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

    written = OMEZarrMultiscale.from_ome_zarr(args.zarr_path)
    print(f"Wrote label group: labels/{args.label_name}")
    print(f"labels/ now contains: {sorted(written.labels)}")


if __name__ == "__main__":
    main()
