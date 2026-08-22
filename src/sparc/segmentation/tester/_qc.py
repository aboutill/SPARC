import os
import json

import torch
import numpy as np
import pandas as pd
import nibabel as nib

from monai.metrics import (
    DiceMetric,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
)


def quality_control(
    self,
    ensemble_mask_path,
    indiv_mask_paths,
):
    """Compute pairwise inter-model DSC/HD95/ASSD and mask volume,
    independently per class.
    """
    if self.out_channels == 1:
        class_ids = [1]
    else:
        class_id = list(range(1, self.out_channels))

    dice_metric = DiceMetric()
    hd_metric = HausdorffDistanceMetric(percentile=95)
    assd_metric = SurfaceDistanceMetric(symmetric=True)

    n = len(indiv_mask_paths)
    label_volumes, max_err = [], None
    for path in indiv_mask_paths:
        label_volume, max_err = self.load_mask_as_tensor(path)
        label_volumes.append(label_volume)

    ensemble_label_volume, _ = self.load_mask_as_tensor(ensemble_mask_path)
    ensemble_nii = nib.load(ensemble_mask_path)
    voxel_volume = float(np.prod(ensemble_nii.header["pixdim"][1:4]))

    results = {}
    for class_id in class_ids:
        dices, hds, assds = [], [], []

        binary_masks = [(lv == class_id).unsqueeze(0).unsqueeze(0) for lv in label_volumes]

        for i in range(n):
            mask_i = binary_masks[i]
            for j in range(i + 1, n):
                mask_j = binary_masks[j]

                if torch.all(~mask_i) or torch.all(~mask_j):
                    dices.append(0.0)
                    hds.append(max_err)
                    assds.append(max_err)
                    continue

                dices.append(dice_metric(mask_i, mask_j).item())
                hds.append(hd_metric(mask_i, mask_j).item())
                assds.append(assd_metric(mask_i, mask_j).item())

        class_volume = float((ensemble_label_volume == class_id).sum().item()) * voxel_volume

        results[class_id] = {
            "dices": dices,
            "hds": hds,
            "assds": assds,
            "volume": class_volume,
        }

    return results


def _save_quality_control_dataframe(
    self,
    ensemble_mask_paths,
    indiv_mask_paths,
):
    """Save per-subject-per-class and summary-per-class inter-model
    agreement metrics to CSV, under self.qc_dir.

    quality_control_raw.csv has one row per (subject, class) pair;
    quality_control.csv has one row per class, averaged across
    subjects.
    """

    def _fmt(val):
        return f"{val:.4f}"

    per_subject_rows = []
    raw_by_class = {}

    for ensemble_path, indiv_paths in zip(ensemble_mask_paths, indiv_mask_paths):
        per_class_results = self.quality_control(
            ensemble_mask_path=ensemble_path,
            indiv_mask_paths=indiv_paths,
        )

        for class_id, result in per_class_results.items():
            dice_qc = np.mean(result["dices"])
            hd_qc = np.mean(result["hds"])
            assd_qc = np.mean(result["assds"])
            volume = result["volume"]

            per_subject_rows.append(
                {
                    "filename": ensemble_path,
                    "class": class_id,
                    "DICE_QC": _fmt(dice_qc),
                    "HD_QC": _fmt(hd_qc),
                    "ASSD_QC": _fmt(assd_qc),
                    "VOL_QC": _fmt(volume),
                }
            )

            bucket = raw_by_class.setdefault(
                class_id, {"DICE_QC": [], "HD_QC": [], "ASSD_QC": [], "VOL_QC": []}
            )
            bucket["DICE_QC"].append(dice_qc)
            bucket["HD_QC"].append(hd_qc)
            bucket["ASSD_QC"].append(assd_qc)
            bucket["VOL_QC"].append(volume)

    summary_rows = [
        {"class": class_id, **{k: _fmt(np.mean(v)) for k, v in raw.items()}}
        for class_id, raw in raw_by_class.items()
    ]

    os.makedirs(self.qc_dir, exist_ok=True)
    pd.DataFrame(per_subject_rows).to_csv(
        os.path.join(self.qc_dir, "quality_control_raw.csv"), index=False
    )
    pd.DataFrame(summary_rows).to_csv(os.path.join(self.qc_dir, "quality_control.csv"), index=False)


def save_quality_control_report(
    self,
    ensemble_mask_path,
    indiv_mask_paths,
    output_path,
):
    """Save a single-subject inter-model agreement report as JSON,
    keyed by class label.
    """

    def _fmt(vals):
        vals = list(vals)
        return {
            "mean": f"{np.mean(vals):.4f}",
            "std": f"{np.std(vals):.4f}",
            "vals": vals,
        }

    per_class_results = self.quality_control(
        ensemble_mask_path=ensemble_mask_path,
        indiv_mask_paths=indiv_mask_paths,
    )

    if self.out_channels == 1:
        result = per_class_results[1]
        report = {
            "DICE_QC": _fmt(result["dices"]),
            "HD_QC": _fmt(result["hds"]),
            "ASSD_QC": _fmt(result["assds"]),
            "VOL_QC": f"{result['volume']:.4f}",
        }
    else:
        report = {
            class_id: {
                "DICE_QC": _fmt(result["dices"]),
                "HD_QC": _fmt(result["hds"]),
                "ASSD_QC": _fmt(result["assds"]),
                "VOL_QC": f"{result['volume']:.4f}",
            }
            for class_id, result in per_class_results.items()
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
