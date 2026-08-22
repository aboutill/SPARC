# Notebooks

Reproduces the statistical analysis and figures reported in the manuscript, from de-identified, subject-level aggregate metrics.

## Running

```bash
SPARC notebook --workdir ./notebooks
```
Runs inside the same Docker image as the rest of the pipeline.


## What each notebook reproduces

| Notebook | Reproduces | Required data |
|---|---|---|
| `pipeline.ipynb` | Cohort/exclusion counts; thoracic segmentation, reconstruction, and reorientation QC acceptance-region figures; end-to-end automation rate and per-stage manual-intervention breakdown; processing time | `csv/pipeline.csv` |
| `segmentation.ipynb` | Chest and heart segmentation network comparison (source/joint/target/transfer), ensemble vs. individual prediction, inter-rater agreement | `csv/*_seg_*.csv`, `csv/cohort.csv` |
| `reorientation.ipynb` | Reorientation network comparison (source/joint/target/transfer), ensemble vs. individual prediction, rotation-averaging strategy comparison (chordal/quaternion/geodesic), inter-rater agreement, pose-sampling distribution validation | `csv/reo_*.csv`, `csv/cohort.csv` |
