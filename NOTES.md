# OME-Zarr binary-threshold labels — implementation notes

Two scripts write a binary-threshold **label** image into the existing
OME-Zarr **v0.5** store `6001240_labels.zarr`:

| Script | Approach | Memory model |
|---|---|---|
| `threshold_label.py` | Pyramid, metadata and chunking written **by hand** with `zarr` v3. | Eager — full channel in RAM. |
| `threshold_label_omezarr.py` | Same Otsu + pyramid, but writing delegated to **`ome-zarr-py`** (`write_multiscale_labels`). | Eager — full channel in RAM. |
| `threshold_label_dask.py` | Read via ome-zarr-py Reader, lazy **dask** end-to-end: streamed histogram Otsu + streamed `da.to_zarr` writes. | Out-of-core — a few chunks in RAM. |

All three produce **byte-identical** label output (same v0.5 metadata, scales,
`int8` data, chunks `(1,1,256,256)`, shards `(1,10,512,512)`). They differ only
in *how* they read/write, not in *what* they write.

Both read the source image (path `0`, shape `[2, 236, 275, 271]`, `uint16`,
axes `c,z,y,x`), threshold one channel (default `1` = Dapi), and write the
result under `labels/<name>` with full `multiscales` + `image-label` metadata.

## Environment

- The store is `zarr_format: 3` / OME-Zarr **0.5**. Use a venv:
  ```
  python -m venv .venv && source .venv/bin/activate
  pip install "zarr>=3" numpy            # hand-written script
  pip install "ome-zarr==0.18.0"         # adds the ome-zarr-py variant (pulls dask, ome-zarr-models)
  ```
- Verified with `zarr 3.2.1`, `numpy 2.5.1`, `ome-zarr 0.18.0`.

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
into inner chunks `(1, 1, 256, 256)`. Each label level reads the matching
source level's `.chunks` and `.shards` and reuses them, so the label chunk
grid corresponds to the source's.

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

1. **Read lazily** — the ome-zarr-py Reader yields `node.data`, one dask array
   per level; nothing is materialised.
2. **Streamed Otsu** — one pass for min/max, one for `da.histogram`; only the
   256-bin histogram lands in RAM. Identical maths to the numpy Otsu (factored
   into `otsu_from_histogram`), so the threshold is bit-for-bit the same.
3. **Lazy mask + pyramid** — `(chan > thr)` and `[..., ::2, ::2]` stay dask
   graphs; each level is `rechunk`-ed to the shard shape so writes align.
4. **Streamed write** — dask arrays go straight to `da.to_zarr`; peak memory is
   a few chunks, not the whole volume.

Cost: a global histogram is still one full read pass (unavoidable for Otsu),
and Otsu needs a value range (uses the channel's min/max — one extra pass).

### Reading: ome-zarr-py Reader vs. `zarr.open_group`

`Reader(parse_url(path))` → `node.data` is a list of **lazy dask arrays** (one
per level) plus parsed `node.metadata["axes"]` and
`["coordinateTransformations"]`. Worth using in the dask script — it hands you
exactly the lazy arrays + parsed metadata the pipeline consumes.

But it does **not** replace `zarr`:
- It chunks the dask array at **inner-chunk** granularity `(1,1,256,256)`, not
  the shard shape `(1,10,512,512)`. Mirroring the source chunk grid on write
  still needs `zarr.open_array(...).shards`.
- The reader is **read-only**; `write_multiscale_labels` needs a writable
  `zarr.Group` (`zarr.open_group(mode="a")`).

So `threshold_label_dask.py` uses the Reader for the read side and keeps `zarr`
for the write side + the shard lookup. For the eager scripts the Reader is not
worth it — they materialise anyway and still need `zarr` to write.

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
`otsu_threshold()` is a small numpy implementation, kept in **both** scripts to
avoid a scikit-image dependency. `skimage.filters.threshold_otsu` is a drop-in
replacement if scikit-image is otherwise available. This was a deliberate
decision — do not replace it without cause.

## Usage

```
python threshold_label.py                       # Otsu on channel 1 -> labels/threshold
python threshold_label.py --channel 0           # LaminB1
python threshold_label.py --threshold 500       # fixed cutoff instead of Otsu
python threshold_label.py --label-name mymask   # custom label name
python threshold_label.py --levels 4            # override pyramid depth
```

`threshold_label_omezarr.py` takes the same flags. Both delete an existing
`labels/<name>` before writing, so re-runs are idempotent.
