# OME-ZARR (v0.5) processing demo

Repository to illustrate how to (lazy) load an OME-Zarr image (based on Zarr v3), compute a label image and store it within the original Zarr structure.

Script [threshold_label_dask.py]() is based on dependency https://pypi.org/project/ome-zarr/

Script [threshold_label_ngff.py]() is based on dependency https://pypi.org/project/ngff-zarr/

No low level API from https://pypi.org/project/zarr/ is used.

Implementation details can be found in [NOTES.md](NOTES.md).

Sample data `6001240_labels.zarr` (from https://livingobjects.ebi.ac.uk/idr/zarr/v0.5/idr0062A/6001240_labels.zarr) contained within this repository is consumed by default.

## Testing
- Clone repository
- Create virtual environment (or Conda env)
- Install dependencies `pip install -r requirements.txt`
- Run tests: `./tests/test_scripts.sh`
