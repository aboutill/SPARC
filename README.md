[![Docker Pulls](https://img.shields.io/docker/pulls/aboutill/sparc)](https://hub.docker.com/r/aboutill/sparc)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](#license)

# SPARC: Slice-to-volume Pipeline for Automated Reconstruction of gated 3D+time fetal Cardiac cine

**An end-to-end, automated pipeline for reconstructing 3D+time fetal cardiac cine MRI from Doppler-gated 2D stacks.**

SPARC combines a physics-informed slice-to-volume reconstruction (SVR) algorithm with deep learning (segmentation and reorientation) models to turn raw, Doppler-gated MRI acquisitions into a reoriented, DICOM-ready 3D+time cine volume of the fetal heart, with automated quality control at every stage and a human-in-the-loop review workflow when it's needed.

<p align="center">
  <img src="docs/SPARC-demo.gif" width="700">
</p>

---
 	
## Preamble

The SPARC pipeline is a research software developed alongside the following: [10.48550/arXiv.2608.18616](https://doi.org/10.48550/arXiv.2608.18616). The pipeline is used to generate the results reported in that manuscript, and this repository is intended to support their reproduction.

The 3D+time SVR algorithm is a physics-based method (unlike this pipeline's deep-learning segmentation and reorientation components, it requires no domain-specific training) and is maintained as a separate, general-purpose project: [https://github.com/baby-MedIA/svr-lite](https://github.com/baby-MedIA/svr-lite).

The pipeline leverages a recently introduced, CE-marked Doppler ultrasound device [10.2463/mrms.mp.2017-0100](https://doi.org/10.2463/mrms.mp.2017-0100) used to gate the acquisiton of balanced steady-state free precession (bSSFP) sequence. This process produces stacks of doppler-ultrasound (DUS) gated 2D+time slices as seen below:

<p align="center">
  <img src="docs/SPARC-acq.png" width="700">
</p>

---

## Contents

- [Key results](#key-results)
- [Pipeline overview](#pipeline-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Operating modes](#operating-modes)
- [Configuration](#configuration)
- [Training and testing your own models](#training-and-testing-your-own-models)
  - [Downloading pretrained models](#downloading-pretrained-models)
- [Reproducibility](#reproducibility)
  - [Training experiments](#training-experiments)
  - [Statistical analysis](#statistical-analysis)
- [Repository structure](#repository-structure)
- [Citation](#citation)
- [Contact](#contact)

---

## Key results

| Metric | Result |
|---|---|
| Thoracic segmentation (DSC) | **84.7 ± 3.9%**, exceeds inter-rater agreement (81.4 ± 7.7%) |
| Reorientation success (with cardiac localisation) | **90.1%** |
| Fully automatic end-to-end success | **82.6%** on N = 121 clinical cases |
| Mean processing time | **7.1 ± 1.3 min** per case |
| Reconstruction speed | **~10× faster** than van Amerom *et al.* approach [10.1002/mrm.26686](https://doi.org/10.1002/mrm.26686) |

Full methodology, statistical testing, and quality-control analysis are described in the manuscript pre-print.

---

## Pipeline overview

<p align="center">
  <img src="docs/SPARC-pipeline.png" width="700">
</p>


SPARC runs as six stages, each with its own automated quality control:

1. **Preprocessing**: DICOM → NIfTI conversion, denoising, Gibbs-ringing and bias-field correction, RR-interval extraction from the Doppler-gating device, with an optional interactive review to exclude motion-corrupted stacks and align each stack's cardiac phase to a common diastolic reference frame.
2. **Chest segmentation**: ensemble deep-learning segmentation of the fetal thorax, used to select a reconstruction template stack.
3. **SVR reconstruction**: physics-informed slice-to-volume reconstruction producing a 3D+time cine volume.
4. **Heart segmentation**: ensemble deep-learning segmentation of the fetal heart on the reconstructed cine volume.
5. **Reorientation**: ensemble reorientation network predicts a rigid transform into a canonical fetal cardiac orientation.
6. **Postprocessing**: conversion of the final reoriented cine volume into a clinically viewable DICOM series.

Every automated stage produces a quality-control report (inter-model agreement metrics from the underlying ensemble and consistency metrics for super-resolution reconstruction). Depending on the selected [operating mode](#operating-modes), a case that fails QC is flagged for manual or semi-automatic review rather than silently proceeding.

---

## Requirements

- **[Docker Engine](https://docs.docker.com/engine/install/)**.
- **NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**, strongly recommended. The pipeline will run on CPU, but deep-learning inference/training will be substantially slower.
- **`make`** for installation.

No local Python environment is required to *run* the pipeline, everything executes inside the Docker image.

---

## Installation

```bash
git clone https://github.com/aboutill/SPARC.git
cd SPARC
make install
```

This adds the `SPARC` command to `~/.local/bin` and pulls the Docker image (`aboutill/sparc:v1.0.0`). If `~/.local/bin` isn't already on your `PATH`, `make install` will tell you, and you have to add it manually.

To install to a different location (e.g. system-wide):

```bash
sudo make install PREFIX=/usr/local/bin
```

Confirm the install worked:

```bash
SPARC -h
```

---

## Quick start

Run the pipeline on a single subject's raw DICOM directory:

```bash
SPARC pipeline --input /path/to/dicom --output /path/to/results --mode semi_auto
```

Run on a batch of subjects (one subdirectory per subject under `--input`):

```bash
SPARC pipeline --input /path/to/dicom_root --output /path/to/results --mode full_auto --batch
```

See every available option:

```bash
SPARC pipeline -h
```

---

## Operating modes

SPARC supports four operating modes, trading off automation against human oversight:

| Mode | Behaviour |
|---|---|
| `manual` | Every stage is manually performed and reviewed via ITK-SNAP. |
| `semi_auto` | Automatic segmentation/reorientation is run first, then always presented for review. |
| `monitored_auto` | Fully automatic, except a stage whose quality-control check fails is flagged for review. |
| `full_auto` | Fully automatic, no review (suited to large retrospective batch processing). |

For modes that involve review, `--gui_mode` selects how that review happens:

- `docker`: launches ITK-SNAP inside the container, forwarded to your display (default).
- `native`: pauses and prints the host file path to review, so you can use a native ITK-SNAP install on your own machine instead (useful on systems where GUI forwarding into Docker isn't available or permitted).

---

## Configuration

SPARC's algorithmic behaviour is controlled by `cfg/pipeline.yaml`, which is documented inline and covers, per stage:

- DICOM tag names used to identify magnitude/phase images, RR intervals, stack IDs, and acquisition parameters (vendor/site-dependent, check these first before deployment).
- Preprocessing corrections to enable (zero-filling removal, denoising, Gibbs correction, bias-field correction).
- Which pretrained model set to use for each deep-learning stage.
- SVR reconstruction parameters (resolution, iterations, regularisation).
- Quality-control acceptance thresholds for `monitored_auto` mode.

Provide a custom configuration file with `--cfg`:

```bash
SPARC pipeline --input ... --output ... --cfg my_config.yaml
```

Custom or fine-tuned model weights can be supplied per stage (e.g. `--models_chest_seg`, `--models_heart_seg`, `--models_reo`, each paired with a `*_cfg` architecture configuration file), see `SPARC pipeline -h` for the full list.

---

## Training and testing your own models

SPARC's deep-learning components (chest segmentation, heart segmentation, reorientation) can be retrained or fine-tuned on your own data, including multi-domain and transfer-learning setups:

```bash
# Train a segmentation model, optionally across multiple domains/input directories
SPARC train --input /data/domain_a --input /data/domain_b \
            --output /results/train --task segmentation --cfg training_config.yaml

# Fine-tune from a pretrained checkpoint
SPARC train --input /data/target_domain --output /results/train \
            --task segmentation --cfg training_config.yaml --models /path/to/pretrained

# Evaluate a trained ensemble
SPARC test --input /data/test --output /results/test \
           --task segmentation --cfg model_config.yaml --models /path/to/checkpoints
```

Run `SPARC train -h` / `SPARC test -h` for the full set of options.

### Downloading pretrained models

Pretrained models for all three components are baked into the Docker image by default, across all four training-domain variants (`philips`, `siemens`, `siemens_transfer`, `joint`).

For a host-side copy (e.g. as a starting checkpoint for `SPARC train --models`) or to browse the models weights directly, they're also hosted on Hugging Face Hub:

- [aboutill/sparc-chest-segmentation](https://huggingface.co/aboutill/sparc-chest-segmentation)
- [aboutill/sparc-heart-segmentation](https://huggingface.co/aboutill/sparc-heart-segmentation)
- [aboutill/sparc-reorientation](https://huggingface.co/aboutill/sparc-reorientation)

Each repository's four variants are available as separate branches. Download a host-side copy via:

```bash
SPARC download-models --output /path/to/pretrained
```

Run `SPARC download-models -h` for the full set of options.

---

## Reproducibility

The results reported in the accompanying manuscript were generated using the Docker image and configuration files in this repository.

### Training experiments

The training runs behind the paper's reported results — source-only, target-only, joint, and transfer-learning training for each of the three deep-learning components, across both domains — are invoked via [`experiments/experiments.bash`](experiments/experiments.bash).

### Statistical analysis

Jupyter notebooks reproducing the statistical analysis reported in the manuscript are provided under [`notebooks/`](notebooks/).

**Running the notebooks.** They run inside the same Docker image as the rest of the pipeline:

```bash
SPARC notebook --workdir ./notebooks
```

This launches Jupyter inside the container (forwarded to `http://localhost:8888`), with `./notebooks` mounted so any edits are saved back to the host.

---

## Repository structure

```
SPARC/
├── cfg/               # Pipeline and model configuration
├── conf/              # Invocation configuration
├── docs/              # Repository documentation
├── examples/          # Example wrappers
├── notebooks/         # Jupyter notebooks reproducing the statistical analysis
├── experiments/       # Experiments script behind the reported results
├── scripts/           # Bash entry points and the SPARC dispatcher
├── src/sparc/         # Python package (pipeline, DL models, CLI)
├── tests/             # Tests script for every major CLI usage pattern
├── ARCHITECTURE.md
├── Dockerfile
├── instructions.txt
├── LICENSE.md
├── Makefile
├── pyproject.toml
├── README.md
├── requirements.txt
└── setup.py
```

---

## Citation

If you use SPARC in your research, please cite:

1. A. Boutillon, N. Clarke *et al.*, "SPARC: Slice-to-volume Pipeline for Automated Reconstruction of gated 3D+time fetal Cardiac MRI," 2026. Preprint, DOI: [10.48550/arXiv.2608.18616](https://doi.org/10.48550/arXiv.2608.18616)
2. A. Boutillon, N. Clarke *et al.*, "Automated reconstruction of 3D+time fetal cardiac MRI from stacks of Doppler-gated slices," *ISMRM*, 2024. DOI: [10.58530/2025/3617](https://doi.org/10.58530/2025/3617)


---

## Contact

**Arnaud Boutillon** — arnaud.boutillon@kcl.ac.uk

**Naomi Clarke** — naomi.5.clarke@kcl.ac.uk
