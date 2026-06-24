# BT Radio Galaxy Catalog

This repository contains the core scripts, configs, and lightweight documentation used to build a MeerKLASS bent-tail radio galaxy catalog.

## Scope

This repository stores:

- source cutout generation scripts
- morphology-based candidate selection scripts
- BT/nonBT training and inference code
- validation scripts for host finding, opening angle, and spectral index
- configuration files and short workflow notes

This repository does not store:

- FITS mosaics
- HDF5 cutout or candidate files
- CSV result tables
- PNG overlays
- model checkpoints
- exploratory notebooks

These generated products are excluded in [.gitignore](/home/caojie/work/btrg_catalog/.gitignore).

## Repository Layout

- [cut_source.py](/home/caojie/work/btrg_catalog/cut_source.py)
  Crop MeerKLASS FITS mosaics into island-centered HDF5 cutouts.

- [make_candidates.py](/home/caojie/work/btrg_catalog/make_candidates.py)
  Apply morphology-based preselection and build the candidate HDF5.

- [export_png.py](/home/caojie/work/btrg_catalog/export_png.py)
  Export candidate cutouts to PNG for visual inspection and annotation.

- [select_demo.py](/home/caojie/work/btrg_catalog/select_demo.py)
  Compute PNG features and prepare an annotation subset.

- [out_catalog.py](/home/caojie/work/btrg_catalog/out_catalog.py)
  Convert selected HDF5 groups back into a FITS-style catalog table.

- [wise-cross.py](/home/caojie/work/btrg_catalog/wise-cross.py)
  Match host WISE identifiers to AllWISE and append infrared information.

- [bentcls](/home/caojie/work/btrg_catalog/bentcls)
  BT/nonBT classifier training and inference code.

- [configs](/home/caojie/work/btrg_catalog/configs)
  Default configuration for BT classifier training and inference.

- [val](/home/caojie/work/btrg_catalog/val)
  Validation scripts for host finding, opening angle, and spectral index.

## Main Workflow

### 1. Source Cropping

- [cut_source.py](/home/caojie/work/btrg_catalog/cut_source.py)
  Read the MeerKLASS source catalog and FITS mosaics, then write one HDF5 file per mosaic with island-centered cutouts.

### 2. Candidate Preselection

- [make_candidates.py](/home/caojie/work/btrg_catalog/make_candidates.py)
  Compute connected-component and skeleton-based morphology features, then keep extended bent-tail-like candidates.

### 3. Annotation Preparation

- [export_png.py](/home/caojie/work/btrg_catalog/export_png.py)
  Convert candidate cutouts into PNG images.
- [select_demo.py](/home/caojie/work/btrg_catalog/select_demo.py)
  Build a feature table and select a balanced annotation subset.

### 4. BT / nonBT Classification

- [bentcls/train_compare.py](/home/caojie/work/btrg_catalog/bentcls/train_compare.py)
  Train and compare multiple backbones such as ResNet18, EfficientNet-B0, and ConvNeXt-Tiny.
- [bentcls/predict_h5.py](/home/caojie/work/btrg_catalog/bentcls/predict_h5.py)
  Run single-model inference on candidate HDF5 groups.
- [bentcls/predict_h5_topk.py](/home/caojie/work/btrg_catalog/bentcls/predict_h5_topk.py)
  Keep the highest-scoring fraction of candidates.
- [bentcls/predict_multi_models.py](/home/caojie/work/btrg_catalog/bentcls/predict_multi_models.py)
  Run inference for multiple models and cross-match their candidate outputs.

The multi-model inference workflow is described in [predict_multi_models.md](/home/caojie/work/btrg_catalog/predict_multi_models.md).

### 5. Catalog Assembly

- [out_catalog.py](/home/caojie/work/btrg_catalog/out_catalog.py)
  Convert final HDF5 subsets into a FITS table for downstream catalog work.

### 6. Host Matching and AllWISE Enrichment

- [val/findhost_new_catalog.py](/home/caojie/work/btrg_catalog/val/findhost_new_catalog.py)
  Find host candidates from catalog RA/DEC by matching radio and WISE cutouts through WCS footprint checks.
- [val/host.py](/home/caojie/work/btrg_catalog/val/host.py)
  Create example WISE+radio host overlays.
- [val/wise_download.py](/home/caojie/work/btrg_catalog/val/wise_download.py)
  Download WISE FITS cutouts for host validation.
- [wise-cross.py](/home/caojie/work/btrg_catalog/wise-cross.py)
  Append AllWISE photometry and quality flags to the host-matched catalog.

### 7. Validation Products

- [val/OA_auto.py](/home/caojie/work/btrg_catalog/val/OA_auto.py)
  Automatic opening-angle measurement using masks, skeletons, and host-guided or bend-guided C points.
- [val/OA_plot.py](/home/caojie/work/btrg_catalog/val/OA_plot.py)
  Generate OA visualizations for quality control and presentation.
- [val/crop_bent_300.py](/home/caojie/work/btrg_catalog/val/crop_bent_300.py)
  Produce radio cutouts for morphology and OA validation.
- [val/crop_cube_sources.py](/home/caojie/work/btrg_catalog/val/crop_cube_sources.py)
  Crop radio cubes for spectral-index analysis.
- [val/spectral_index.py](/home/caojie/work/btrg_catalog/val/spectral_index.py)
  Compute two-band spectral-index maps.

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

Some scripts assume local HPC paths and MeerKLASS data locations. Before running them elsewhere, update the hard-coded input and output paths at the top of each script.

## Notes

- This repository is a cleaned code snapshot of the BT catalog construction workflow.
- Large data products and generated outputs remain outside version control by design.
