# MeerKLASS Bent-Tail Radio Galaxy Catalog Workflow

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21760616.svg)](https://doi.org/10.5281/zenodo.21760616)

This repository contains the core scripts, configuration files, and
documentation used to construct and validate a MeerKLASS bent-tail radio
galaxy catalog.

## Scope

This repository includes:

- source cutout generation scripts;
- morphology-based candidate selection scripts;
- BT/non-BT classifier training and inference code;
- catalog assembly scripts;
- host-identification and AllWISE cross-matching scripts;
- opening-angle and spectral-index validation scripts;
- configuration files and workflow documentation.

The following large or generated products are not included:

- MeerKLASS FITS mosaics;
- HDF5 cutout and candidate files;
- output CSV and FITS catalog tables;
- PNG images and overlays;
- trained model checkpoints;
- exploratory notebooks.

These generated products are excluded from version control through
[`.gitignore`](.gitignore).

## Repository Layout

- [`cut_source.py`](cut_source.py)  
  Crops MeerKLASS FITS mosaics into island-centred HDF5 cutouts.

- [`make_candidates.py`](make_candidates.py)  
  Applies morphology-based preselection and creates the candidate HDF5 files.

- [`export_png.py`](export_png.py)  
  Exports candidate cutouts as PNG images for visual inspection and
  annotation.

- [`select_demo.py`](select_demo.py)  
  Calculates image features and prepares a subset for manual annotation.

- [`out_catalog.py`](out_catalog.py)  
  Converts selected HDF5 groups into a catalog table.

- [`wise-cross.py`](wise-cross.py)  
  Cross-matches host WISE identifiers with AllWISE and appends infrared
  photometry and quality information.

- [`bentcls/`](bentcls/)  
  Contains the BT/non-BT classifier training and inference code.

- [`configs/`](configs/)  
  Contains configuration files for classifier training and inference.

- [`val/`](val/)  
  Contains scripts for host identification, opening-angle measurement, and
  spectral-index analysis.

## Main Workflow

### 1. Source Cutout Generation

[`cut_source.py`](cut_source.py) reads the MeerKLASS source catalog and FITS
mosaics, and then writes island-centred cutouts to HDF5 files.

### 2. Candidate Preselection

[`make_candidates.py`](make_candidates.py) calculates connected-component,
skeleton, and morphology features, and retains extended candidates with
bent-tail-like structures.

### 3. Annotation Preparation

- [`export_png.py`](export_png.py) converts candidate cutouts into PNG images.
- [`select_demo.py`](select_demo.py) creates a feature table and selects a
  subset for manual annotation.

### 4. BT/non-BT Classification

- [`bentcls/train_compare.py`](bentcls/train_compare.py) trains and compares
  classifier backbones including ResNet-18, EfficientNet-B0, and
  ConvNeXt-Tiny.

- [`bentcls/predict_h5.py`](bentcls/predict_h5.py) performs single-model
  inference on candidate HDF5 groups.

- [`bentcls/predict_h5_topk.py`](bentcls/predict_h5_topk.py) retains a
  specified fraction of the highest-scoring candidates.

- [`bentcls/predict_multi_models.py`](bentcls/predict_multi_models.py)
  performs inference with multiple models and cross-matches their candidate
  outputs.

### 5. Catalog Assembly

[`out_catalog.py`](out_catalog.py) converts the final selected HDF5 groups
into a catalog table for subsequent analysis.

### 6. Host Identification and AllWISE Cross-Matching

- [`val/findhost_new_catalog.py`](val/findhost_new_catalog.py) identifies host
  candidates by comparing radio and WISE cutouts through WCS footprint
  checks.

- [`val/host.py`](val/host.py) produces example WISE and radio overlays for
  host-identification checks.

- [`val/wise_download.py`](val/wise_download.py) downloads WISE FITS cutouts
  used in host validation.

- [`wise-cross.py`](wise-cross.py) appends AllWISE photometry and quality
  flags to the host-matched catalog.

### 7. Morphological and Spectral Validation

- [`val/OA_auto.py`](val/OA_auto.py) performs automatic opening-angle
  measurements using emission masks, skeletons, and host-guided or
  bend-guided reference points.

- [`val/OA_plot.py`](val/OA_plot.py) creates opening-angle visualisations for
  quality control.

- [`val/crop_bent_300.py`](val/crop_bent_300.py) produces radio cutouts for
  morphology and opening-angle validation.

- [`val/crop_cube_sources.py`](val/crop_cube_sources.py) crops radio-image
  cubes for spectral-index analysis.

- [`val/spectral_index.py`](val/spectral_index.py) calculates two-band
  spectral-index maps.

## Installation

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

## Configuration and Data Paths

Some scripts contain input and output paths specific to the original
high-performance computing environment.

Before running the workflow on another system, users must update the relevant
paths for:

- MeerKLASS FITS mosaics;
- source catalogs;
- HDF5 cutout files;
- model checkpoints;
- output catalogs and validation products.

The original MeerKLASS images and other large intermediate data products are
not distributed with this software release.

## Reproducibility Notes

This repository is a cleaned software snapshot of the catalog-construction
workflow used in the associated study.

The repository preserves the main processing, classification, catalog
assembly, and validation scripts. Large input data, intermediate products,
trained model files, and generated catalog products remain outside version
control by design.

## Author

- Jie Cao, Guangzhou University

## License

This software is distributed under the MIT License. See the
[`LICENSE`](LICENSE) file for details.

## Citation

When using this software, please cite the archived release:

> Cao, J. (2026). *btrg_catalog: MeerKLASS Bent-Tail Radio Galaxy Catalog
> Construction Workflow* (Version 1.0.0). Zenodo.  
> https://doi.org/10.5281/zenodo.21760616

Machine-readable citation metadata are provided in
[`CITATION.cff`](CITATION.cff).
