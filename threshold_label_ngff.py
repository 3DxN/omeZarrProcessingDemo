#!/usr/bin/env python
"""Out-of-core binary-threshold labels for OME-Zarr v0.5 using ngff-zarr.

Equivalent to threshold_label_dask.py, but built on the `ngff_zarr` library
instead of ome-zarr-py. Data stays lazy end-to-end:

  * `nz.from_ngff_zarr` reads the source as lazy dask arrays (one per level),
  * Otsu is computed from a streamed dask histogram (only 256 bins in RAM),
  * `nz.to_multiscales` builds a label-safe nearest pyramid (ITKWASM_LABEL_IMAGE,
    y/x-only) that stays lazy, and
  * `nz.to_ngff_zarr` streams each shard to disk.

ngff-zarr is image-focused: it writes the `multiscales` metadata but not the
label-specific `image-label` block, and it does not register the label in the
parent `labels/` group. Those two OME-Zarr specifics are patched afterwards by
editing the v3 `zarr.json` files directly (pathlib + json, no `zarr` import).

The Otsu maths is identical to threshold_label.py (fed a histogram here).
Note: ngff-zarr writes level arrays at nested paths `scaleN/<name>` (its native
layout), unlike the flat `0/1/2` used by the other scripts -- both are valid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import dask.array as da
import numpy as np
import ngff_zarr as nz


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
    """Streamed Otsu: one pass for min/max, one pass for the histogram."""
    vmin, vmax = da.compute(chan.min(), chan.max())
    counts, edges = da.histogram(chan, bins=nbins, range=[float(vmin), float(vmax)])
    counts = counts.compute()
    centers = (edges[:-1] + edges[1:]) / 2.0
    return otsu_from_histogram(counts, centers)


def patch_json(zarr_json: Path, update) -> None:
    """Load a v3 zarr.json, apply `update(attrs_ome_dict)`, write it back."""
    doc = json.loads(zarr_json.read_text())
    update(doc["attributes"]["ome"])
    zarr_json.write_text(json.dumps(doc, indent=2))


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
    ap.add_argument("--shard-c", type=int, default=1, help="chunks per shard along c")
    ap.add_argument("--shard-z", type=int, default=4, help="chunks per shard along z")
    ap.add_argument("--shard-y", type=int, default=4, help="chunks per shard along y")
    ap.add_argument("--shard-x", type=int, default=4, help="chunks per shard along x")
    ap.add_argument("--shard-t", type=int, default=1, help="chunks per shard along t")
    ap.add_argument("--no-shard", action="store_true",
                    help="write unsharded (chunks only, no shard grid)")
    args = ap.parse_args()

    # --- read the source image lazily via ngff-zarr ----------------------
    ms = nz.from_ngff_zarr(args.zarr_path)
    scale_index = int(args.scale_path)
    img = ms.images[scale_index]
    dims = img.dims                                   # ('c', 'z', 'y', 'x')
    n_levels = args.levels if args.levels is not None else len(ms.images)

    chan = img.data[args.channel]                     # lazy (z, y, x)
    print(f"Channel {args.channel}: shape={chan.shape} dtype={chan.dtype} "
          f"chunks={chan.chunksize} (lazy dask array via ngff-zarr)")

    # --- compute the binary threshold (streamed) -------------------------
    if args.threshold is None:
        thr = otsu_threshold_dask(chan)
        print(f"Otsu threshold = {thr:.2f}")
    else:
        thr = args.threshold
        print(f"Fixed threshold = {thr:.2f}")

    mask = (chan > thr).astype(np.int8)[np.newaxis, ...]   # lazy (c, z, y, x)

    # --- build the lazy label pyramid ------------------------------------
    # y/x-only nearest downsampling per level; z (and t/c) kept at factor 1.
    scale_factors = [
        {d: (2 ** i if d in ("y", "x") else 1) for d in dims if d != "c"}
        for i in range(1, n_levels)
    ]
    # mirror the source's inner chunk shape for the label
    chunks = dict(zip(dims, img.data.chunksize))
    label_img = nz.to_ngff_image(mask, dims=dims, scale=img.scale,
                                 axes_units=img.axes_units, name=args.label_name)
    label_ms = nz.to_multiscales(label_img, scale_factors=scale_factors,
                                 method=nz.Methods.ITKWASM_LABEL_IMAGE, chunks=chunks)
    for lvl in label_ms.images:
        print(f"  level: shape={lvl.data.shape} scale_yx={lvl.scale['y']:.4f}")

    # shard factor per axis (chunks_per_shard); None disables sharding
    shard_factor = {"c": args.shard_c, "z": args.shard_z, "y": args.shard_y,
                    "x": args.shard_x, "t": args.shard_t}
    chunks_per_shard = None if args.no_shard else {d: shard_factor[d] for d in dims}

    # --- write the label multiscale (overwrite makes re-runs idempotent) --
    label_group = f"{args.zarr_path}/labels/{args.label_name}"
    nz.to_ngff_zarr(label_group, label_ms, version="0.5",
                    chunks_per_shard=chunks_per_shard)

    # --- patch OME-Zarr label specifics ngff-zarr doesn't write ----------
    # (a) image-label metadata on the label group
    patch_json(Path(label_group) / "zarr.json", lambda ome: ome.__setitem__(
        "image-label", {"version": "0.5",
                        "colors": [{"label-value": 1, "rgba": [255, 0, 0, 128]}],
                        "source": {"image": "../.."}}))
    # (b) register the label in the parent labels/ group
    labels_json = Path(args.zarr_path) / "labels" / "zarr.json"

    def register(ome):
        names = ome.setdefault("labels", [])
        if args.label_name not in names:
            names.append(args.label_name)

    patch_json(labels_json, register)

    registered = json.loads(labels_json.read_text())["attributes"]["ome"]["labels"]
    print(f"Wrote label group: labels/{args.label_name}")
    print(f"labels/ now contains: {registered}")


if __name__ == "__main__":
    main()
