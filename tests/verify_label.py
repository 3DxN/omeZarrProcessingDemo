#!/usr/bin/env python
"""Verify a written label group. Usage: verify_label.py <store> <name> <thr>

Checks, layout-agnostically (levels are resolved through the multiscales
`datasets[].path`, so flat `0/1/2`, `s0/s1/s2` and nested `scaleN/<name>` all
work):
  * group is OME-Zarr v0.5 with `image-label` metadata (source.image == ../..),
  * every level array is int8 and binary ({0} or {0,1}),
  * the label is registered in the parent `labels/` group, and
  * level-0 pixels == (source channel > threshold).
Exits non-zero on the first failure.
"""
import sys

import numpy as np
import zarr

store, name, thr = sys.argv[1], sys.argv[2], float(sys.argv[3])
CHANNEL = 1  # matches every script's default


def check(cond: bool, msg: str) -> None:
    if not cond:
        print(f"    x {msg}")
        sys.exit(1)


g = zarr.open_group(f"{store}/labels/{name}", mode="r")
ome = g.attrs["ome"]
check(ome.get("version") == "0.5", f"version {ome.get('version')!r} != 0.5")
check("image-label" in ome, "missing image-label metadata")
check(ome["image-label"].get("source", {}).get("image") == "../..",
      "image-label source.image != ../..")

paths = [d["path"] for d in ome["multiscales"][0]["datasets"]]
check(len(paths) >= 1, "no datasets in multiscales")
for p in paths:
    a = zarr.open_array(f"{store}/labels/{name}/{p}", mode="r")
    check(str(a.dtype) == "int8", f"{p}: dtype {a.dtype} != int8")
    uniq = set(np.unique(a[:]).tolist())
    check(uniq <= {0, 1}, f"{p}: not binary, values {sorted(uniq)}")

registered = zarr.open_group(f"{store}/labels", mode="r").attrs["ome"].get("labels", [])
check(name in registered, f"{name} not registered in labels/ {registered}")

# level-0 correctness against the reference threshold
ref = (zarr.open_array(f"{store}/0", mode="r")[CHANNEL] > thr).astype(np.int8)
l0 = zarr.open_array(f"{store}/labels/{name}/{paths[0]}", mode="r")[0]  # drop c axis
check(l0.shape == ref.shape, f"L0 shape {l0.shape} != {ref.shape}")
check(np.array_equal(l0, ref), "L0 pixels != (channel > threshold)")

print(f"    ok: v0.5, image-label, {len(paths)} levels, binary int8, registered, L0 matches")
