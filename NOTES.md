# OME-Zarr binary-threshold labels — implementation notes

Four scripts write a binary-threshold **label** image into the existing
OME-Zarr **v0.5** store `6001240_labels.zarr`:

| Script | Approach | Memory model |
|---|---|---|
| `threshold_label.py` | Pyramid, metadata and chunking written **by hand** with `zarr` v3. | Eager — full channel in RAM. |
| `threshold_label_omezarr.py` | Same Otsu + pyramid, but writing delegated to **`ome-zarr-py`** (`write_multiscale_labels`). | Eager — full channel in RAM. |
| `threshold_label_dask.py` | Read via `OMEZarrMultiscale`, lazy **dask** end-to-end: streamed histogram Otsu + streamed `da.to_zarr` writes; **no `zarr` import**. | Out-of-core — a few chunks in RAM. |
| `threshold_label_ngff.py` | Same lazy pipeline built on **`ngff-zarr`** (`from_ngff_zarr` → `to_multiscales` → `to_ngff_zarr`); `image-label` + registration patched via json. | Out-of-core — a few chunks in RAM. |

The first three produce **byte-identical** label output (same v0.5 metadata,
scales, `int8` data, chunks `(1,1,256,256)`, shards `(1,10,512,512)`). The
ngff-zarr script writes the same pixels but with its own layout conventions
(see its section below). They differ mainly in *how* they read/write.

All read the source image (path `0`, shape `[2, 236, 275, 271]`, `uint16`,
axes `c,z,y,x`), threshold one channel (default `1` = Dapi), and write the
result under `labels/<name>` with full `multiscales` + `image-label` metadata.

## Environment

- The store is `zarr_format: 3` / OME-Zarr **0.5**. Use a venv:
  ```
  python -m venv .venv && source .venv/bin/activate
  pip install "zarr>=3" numpy            # hand-written script
  pip install "ome-zarr==0.18.0"         # adds the ome-zarr-py variants (pulls dask, ome-zarr-models)
  pip install ngff-zarr                  # adds the ngff-zarr variant (pulls itkwasm, wasmtime, rich)
  ```
- Verified with `zarr 3.2.1`, `numpy 2.5.1`, `ome-zarr 0.18.0`, `ngff-zarr 0.38.0`.

## Key insights / design decisions

### 1. Labels must be downsampled with nearest-neighbour, never averaging
Averaging a label image invents non-existent label values (e.g. `0.5` between
`0` and `1`). Each pyramid level is produced by subsampling y/x (`::2`).
The source image only downscales the spatial **y/x** axes and keeps **z**
constant, so the labels match that: z scale is constant across levels, y/x
scale doubles per level.

### 2. Default number of pyramid levels = the source image's level count
Derived at runtime from `len(multiscales[0].datasets)` of the source image
(3 levels here), not hardcoded. Override with `--levels N`.

### 3. Label chunking mirrors the source chunk grid
The source arrays are **sharded**: shard shape `(1, 10, 512, 512)` subdivided
into inner chunks `(1, 1, 256, 256)`. In `threshold_label.py` and
`threshold_label_omezarr.py`, each label level reads the matching source level's
`.chunks` and `.shards` (via `zarr`) and reuses both, so the label chunk grid
corresponds to the source's. `threshold_label_dask.py` diverges deliberately: it
mirrors only the **chunk** shape (from the high-level API, no `zarr`) and lets
the user pick the **shard** scheme via CLI — see "Chunking vs. sharding" below.

### 4. Label array keeps the channel axis (size 1)
The store's arrays are `(c, z, y, x)`; the existing `labels/0` array is
`(1, 236, 275, 271)` `int8`. New labels follow suit: `int8`, `c`-axis of size 1.

### 5. `image-label` metadata + registration
Each label group carries `image-label` with `source.image: "../.."` (relative
path back to the root image) and a `colors` entry for label-value 1. The label
name is registered in `labels/`'s `ome.labels` list so viewers discover it.

## `ome-zarr-py` vs. hand-written — findings

Both major OME-Zarr Python libraries recently **migrated to zarr v3**, so they
can now target a v0.5 store (`ome-zarr` 0.18 requires `zarr>=3`;
`ngff-zarr` 0.38 supports zarr v3). `ome_zarr.writer.write_multiscale_labels`
collapses the manual per-level array creation + `multiscales`/`image-label`
JSON + label registration into a single call.

Its output was verified **byte-equivalent** to the hand-written version on:
format `0.5`, axes, per-level scales, shapes, `int8` dtype, chunks
`(1,1,256,256)`, shards `(1,10,512,512)`, sharding codec, voxel data,
dimension_names, and the `image-label` block.

**Trade-offs of the `ome-zarr-py` variant:**
1. **Level paths become `s0/s1/s2`** instead of `0/1/2`. Spec-valid (dataset
   paths are arbitrary), but inconsistent with the source image / existing
   `labels/0`, which use `0/1/2`. No clean public-API override.
2. **`coordinate_transformations` is deprecated** in that call, but it is the
   only way to pin the source's exact ×2 scales. The non-deprecated `scale=`
   path infers each level's scale from array *shapes*, giving
   `0.3604 × 275/138 = 0.7182` instead of the source's `0.7208`. The script
   suppresses the deprecation warning.
3. **Extra dependencies**: `ome-zarr`, `ome-zarr-models`, `dask` (arrays are
   internally converted to dask).

**When to prefer which:**
- Hand-written (`threshold_label.py`): keeps `0/1/2` level naming consistent
  with the source, uses no deprecated arguments, minimal deps.
- `ome-zarr-py` (`threshold_label_omezarr.py`): fewer lines, spec-tracking
  metadata from the reference implementation; accept `sN` level naming.

## Scaling to very large datasets

The two eager scripts read the **entire channel into RAM** in one line
(`chan = image[args.channel]`), then build the mask and pyramid in memory. Fine
for this dataset (channel ≈ 236×275×271×2 B ≈ **35 MB**), but it **OOMs** on
light-sheet / whole-slide volumes.

**Is dask used?** Only incidentally in the eager path: `ome-zarr-py` calls
`da.from_array(level)` + `da.to_zarr` internally, so *writes* stream — but the
arrays handed to it are already fully materialised numpy, so peak residency is
still the whole channel.

`threshold_label_dask.py` fixes this by staying lazy end-to-end:

1. **Read lazily** — `OMEZarrMultiscale.from_ome_zarr(path)` yields one lazy
   dask array per level (`.images[i].data`); nothing is materialised.
2. **Streamed Otsu** — one pass for min/max, one for `da.histogram`; only the
   256-bin histogram lands in RAM. Identical maths to the numpy Otsu (factored
   into `otsu_from_histogram`), so the threshold is bit-for-bit the same.
3. **Lazy mask + pyramid** — `(chan > thr)` and `[..., ::2, ::2]` stay dask
   graphs; each level is `rechunk`-ed to the shard shape so writes align.
4. **Streamed write** — dask arrays go straight to `da.to_zarr`; peak memory is
   a few chunks, not the whole volume.

Cost: a global histogram is still one full read pass (unavoidable for Otsu),
and Otsu needs a value range (uses the channel's min/max — one extra pass).

### Reading: `OMEZarrMultiscale` vs. `Reader` vs. `zarr.open_group`

ome-zarr-py 0.18 has a high-level object API (`from ome_zarr import
OMEZarrMultiscale`) — the intended read interface (its docs use this very
dataset). `threshold_label_dask.py` uses it:

```python
ms = OMEZarrMultiscale.from_ome_zarr(path)
ms.images[i].data      # lazy dask array per pyramid level
ms.metadata.axes       # pydantic Axis objects -> full dicts incl. units
ms.images[i].scale     # {'c':1.0,'z':0.5002,'y':0.3604,'x':0.3604}
ms.labels              # {'0': ..., 'threshold': ...} existing labels
```

It replaces the older low-level `parse_url` + `Reader` + `node.metadata[...]`
path and is cleaner than hand-parsing `root.attrs["ome"]`. As a result
**`threshold_label_dask.py` has no `import zarr` at all** — it reads via
`OMEZarrMultiscale` and writes via `write_multiscale_labels(pyramid, "<path>",
...)` (a path string, not a `zarr.Group`).

The one asymmetry to know: the high-level API exposes each level's **inner
chunk** shape (`ms.images[i].data.chunksize`) but **not** its shard shape.
`from_ome_zarr` builds `da.from_zarr(group[path])` — dask sees inner chunks
only; `OMEZarrImage`'s fields are just `data/axes/scale/axes_units/name`; and
`OMEZarrMultiscale` keeps no store handle. So the source *chunk* grid can be
mirrored without `zarr`, but the source *shard* grid cannot be read at all.

### Chunking vs. sharding in `threshold_label_dask.py`

- **Chunking is mirrored from the source** via `ms.images[i].data.chunksize`.
  Caveat: `data.chunksize` is dask's *effective* chunk — at sub-levels where the
  source's declared chunk (`256`) exceeds the axis, dask reports the clamped dim
  (`137`, `68`). Functionally the same single chunk; just not literally `256`.
  Reading the declared `256` would require `zarr` (which we dropped).
- **Sharding is user-defined** via argparse: `--shard-c/-z/-y/-x/-t` give the
  per-axis factor (chunks per shard), default `c=1 z=4 y=4 x=4 t=1`, so
  `shard = chunk × factor` (e.g. x: `256 × 4 = 1024`). `--no-shard` writes
  unsharded. Verified: defaults, custom factors, and `--no-shard` all produce
  valid v0.5 labels; the pixel data is identical regardless of scheme.

### Does ome-zarr-py support sharding? Is the shard shape required?

`ome-zarr-py` 0.18 (the latest release) **fully supports sharded writes** via
`storage_options=[{"chunks": ..., "shards": ...}]` per level, and
`write_multiscale_labels(pyramid, "<path>", ...)` accepts a plain path string —
it opens the store, writes the sharded arrays, and registers the label. So the
write does not intrinsically need a `zarr.Group`.

**Is matching the source shard grid required? No.** Chunk/shard layout is a
per-array storage detail, independent of the multiscales / image-label metadata.
Mirroring it would be a consistency / co-access-performance choice, but it is not
an OME-Zarr correctness requirement — so the dask script instead lets the user
pick the shard scheme (defaulting to a sensible factor) and only mirrors the
chunk shape.

**Can it work with no sharding at all? Yes** (verified): `--no-shard` (i.e.
`write_multiscale_labels` with no `shards` in `storage_options`) produces a valid
v0.5 label — just **unsharded** (`shards=None`, `Bytes`+`Zstd` codecs instead of
a sharding codec) and still registered.

### Why NOT the high-level `OMEZarrLabels` writer

`OMEZarrLabels(img, method=NEAREST).to_ome_zarr(...)` looks ideal for labels
(defaults to label-safe NEAREST, auto-builds the pyramid, auto-writes
`image-label`). Tested against a store copy, it does emit v0.5 and mirrors
chunks/shards via `storage_options` — but for *adding one label to an existing
store* it has three verified gotchas, so the explicit `write_multiscale_labels`
path is kept instead:

1. **`store_exists` footgun.** Passing an existing `zarr.Group` sets
   `store_exists=True` → `write_image_data=False`, so it silently writes **only**
   `image-label` metadata and **no pixel data** — a broken empty label. You must
   pass `overwrite=True` to actually write arrays.
2. **No registration.** It does not add the label to `labels/`'s `ome.labels`
   list, so viewers won't discover it — a manual append is still required.
3. **Scales drift from the source.** It derives each level's scale from the
   array **shape ratio** (e.g. `y=0.7234, x=0.72347` at level 1, and `y != x`)
   instead of the source's clean `×2` (`0.7208`). `OMEZarrLabels` has no
   `coordinateTransformations` argument, so the source's exact scales can't be
   pinned. The explicit writer + hand-built transforms guarantee they match.

(Its NEAREST pyramid does round shapes like the source — `275→137→68` — vs the
scripts' `::2` `275→138→69`; both are valid nearest conventions.)

### `write_multiscale_labels` vs `write_labels`: who builds the pyramid?

Two writer functions, easy to confuse:

- **`write_multiscale_labels(pyramid, ...)`** (plural) takes a **pre-built
  pyramid** (list of arrays, largest first) and only *writes* it — it never
  downsamples. This is why the scripts carry `downsample_yx`: **it is required**
  to produce levels 1..N for this function.
- **`write_labels(labels, ...)`** (singular) takes a **single** array +
  `scale_factors` + `method=Methods.NEAREST` and **builds the pyramid itself**.
  With it (or the high-level `OMEZarrLabels`), `downsample_yx` is unnecessary.
  Verified: one `(1,236,275,271)` array → full pyramid, y/x-only (z stays 236),
  binary preserved.

**Why the scripts keep `downsample_yx` + `write_multiscale_labels` anyway** — the
lower-level writer costs the pyramid helper but buys three things the
auto-builders don't give:

1. **Out-of-core streaming.** `downsample_yx` uses lazy dask slicing
   (`[..., ::2, ::2]`) and each level is `rechunk`-ed to shard boundaries, so
   `da.to_zarr` streams. The auto-builders' internal scaler may materialise /
   re-chunk differently, undermining the dask variant's whole point.
2. **Exact scales.** `write_multiscale_labels` + hand-built `transforms` pin the
   source's clean `×2` scales; `write_labels` / `OMEZarrLabels` derive them from
   shape ratios (drift, `y≠x` — see the section above).
3. **Precise level control.** In testing, `write_labels`' deprecated default
   `scaler` (`max_layer=4`) **overrode** a 2-entry `scale_factors` and emitted
   **5** levels — the kind of surprise avoided by building the pyramid ourselves.

So `downsample_yx` is the deliberate price of a streaming, source-faithful
writer. Drop it only if you switch to `write_labels` and accept auto-derived
scales + less streaming control.

## The `ngff-zarr` variant (`threshold_label_ngff.py`)

A second out-of-core implementation built on **`ngff-zarr`** (0.38) instead of
ome-zarr-py. Same lazy pipeline shape; the whole read → pyramid → write path is
ngff-zarr idiom:

```python
ms = nz.from_ngff_zarr(path)                 # lazy: images[i].data are dask
label_img = nz.to_ngff_image(mask, dims=dims, scale=img.scale, axes_units=...)
label_ms  = nz.to_multiscales(label_img, scale_factors=[{"z":1,"y":2,"x":2}, ...],
                              method=nz.Methods.ITKWASM_LABEL_IMAGE, chunks=chunks)
nz.to_ngff_zarr(f"{path}/labels/{name}", label_ms, version="0.5",
                chunks_per_shard={"c":1,"z":4,"y":4,"x":4})
```

What it gets right out of the box (verified):

- **Lazy end-to-end** — `from_ngff_zarr` yields dask arrays, `to_multiscales`
  keeps them lazy, `to_ngff_zarr` streams shards. Out-of-core like the dask one.
- **Label-safe downsampling** — `Methods.ITKWASM_LABEL_IMAGE` (label-aware,
  y/x-only via dict `scale_factors`). `DASK_IMAGE_NEAREST` also exists but needs
  an extra `dask_image` package; ITKWASM ships with ngff-zarr (`itkwasm-downsample`).
- **Clean ×2 scales** — `to_multiscales` derives scales from the factors, not
  shape ratios, so they match the source exactly (`0.3604 → 0.7208 → 1.4416`).
  This is *better* than ome-zarr-py's `OMEZarrLabels`, which drifts (`y≠x`).
- **Sharding** — via `chunks_per_shard` (per-axis dict); `--no-shard` disables it.

What ngff-zarr does **not** do (it is image-focused), patched afterwards by
editing the v3 `zarr.json` directly (pathlib + json, no `zarr` import):

1. **No `image-label` metadata** — added to the label group's `zarr.json`.
2. **No label registration** — the name is appended to `labels/`'s `ome.labels`.

Cross-checked: the result is a valid v0.5 label, **readable back by ome-zarr-py**
(`OMEZarrMultiscale.from_ome_zarr`), with pixel data matching the reference
threshold. Extra deps vs the dask script: `ngff-zarr`, `itkwasm`,
`itkwasm-downsample`, `wasmtime`, `rich`.

### Data layout: ome-zarr-py vs ngff-zarr on disk

The two writers place the level arrays differently inside the label group. Both
are valid OME-Zarr — a reader resolves levels through the `datasets[].path`
strings in the `multiscales` metadata, not by directory convention — but the
directory trees and those path strings differ.

**ome-zarr-py** (`write_multiscale_labels`, used by `_omezarr.py` and
`_dask.py`) — each level is an array **directly** under the label group, named
`s0/s1/s2`:

```
labels/
├── zarr.json                 # group: ome.labels = [ …, "threshold" ]   (registration)
└── threshold/
    ├── zarr.json             # group: ome.multiscales + image-label
    │                         #        datasets[].path = "s0", "s1", "s2"
    ├── s0/
    │   ├── zarr.json         # ARRAY (1,236,275,271) int8, sharded
    │   └── c/…               # shard/chunk data
    ├── s1/  { zarr.json, c/… }   # ARRAY
    └── s2/  { zarr.json, c/… }   # ARRAY
```

**ngff-zarr** (`to_ngff_zarr`) — each level lives in a `scaleN/` subdirectory
that *wraps* an array named after the image, so arrays sit one level **deeper**
and the dataset paths are nested `scaleN/<name>`:

```
labels/
├── zarr.json                 # group: ome.labels = [ …, "threshold" ]   (patched in)
└── threshold/
    ├── zarr.json             # group: ome.multiscales + image-label (image-label patched in)
    │                         #        datasets[].path = "scale0/threshold", "scale1/threshold", …
    ├── scale0/
    │   └── threshold/
    │       ├── zarr.json     # ARRAY (1,236,275,271) int8, sharded
    │       └── c/…           # shard/chunk data
    ├── scale1/threshold/  { zarr.json, c/… }   # ARRAY
    └── scale2/threshold/  { zarr.json, c/… }   # ARRAY
```

Differences that matter:

- **Depth / paths.** ome-zarr-py: `datasets[].path = "s0"` (array one level down).
  ngff-zarr: `datasets[].path = "scale0/threshold"` (array two levels down, inside
  a per-scale `scaleN/` subdirectory). The hand-written `threshold_label.py` uses
  flat `0/1/2`.
- **Metadata provenance.** ome-zarr-py writes `multiscales` **and** `image-label`
  and registers the label itself. ngff-zarr writes only `multiscales`; this
  script **patches in** `image-label` and the `labels/` registration (marked
  above) by editing `zarr.json`.
- **Sub-level chunks** are dim-clamped (`137`, `68`) in both, like the dask script.
- **Interop.** Despite the layout gap, each is readable by the other library —
  verified `OMEZarrMultiscale.from_ome_zarr` reads the ngff-zarr output fine,
  because both honor the `datasets[].path` indirection.

### Benchmark caveat: dask is *slower* on small data

Measured on this 35 MB channel (peak RSS via `/usr/bin/time -v`):

| Script | Wall clock | Peak RSS |
|---|---|---|
| `threshold_label.py` (eager numpy) | ~0.9 s | ~177 MB |
| `threshold_label_dask.py` (dask)   | ~2.0 s | ~318 MB |

Dask's scheduler/graph overhead **loses** when the data fits comfortably in
RAM. Its win is strictly at scale — past the point where the eager version
would exhaust memory. **Rule of thumb: use the eager script until the channel
no longer fits in RAM, then switch to the dask script.**

### The Otsu threshold is intentionally hand-rolled
`otsu_threshold()` is a small numpy implementation, kept across **all** scripts
to avoid a scikit-image dependency (the lazy scripts feed the same maths a dask
histogram via `otsu_from_histogram`). `skimage.filters.threshold_otsu` is a
drop-in replacement if scikit-image is otherwise available. This was a
deliberate decision — do not replace it without cause.

## Usage

```
python threshold_label.py                       # Otsu on channel 1 -> labels/threshold
python threshold_label.py --channel 0           # LaminB1
python threshold_label.py --threshold 500       # fixed cutoff instead of Otsu
python threshold_label.py --label-name mymask   # custom label name
python threshold_label.py --levels 4            # override pyramid depth
```

`threshold_label_omezarr.py` takes the same flags. All scripts delete an
existing `labels/<name>` before writing, so re-runs are idempotent.

`threshold_label_dask.py` adds sharding controls (chunk shape is mirrored from
the source; only the shard factor is user-chosen):

```
python threshold_label_dask.py                        # shards = chunk x (c1 z4 y4 x4)
python threshold_label_dask.py --shard-z 2 --shard-y 8 # custom per-axis factors
python threshold_label_dask.py --no-shard             # unsharded (chunks only)
```

`threshold_label_ngff.py` (ngff-zarr) takes the same flags, including the
`--shard-*` / `--no-shard` sharding controls:

```
python threshold_label_ngff.py                        # ngff-zarr, shards c1 z4 y4 x4
python threshold_label_ngff.py --no-shard             # unsharded
```
