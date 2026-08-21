import os
import json

import numpy as np


def _quality_control(self, affine_matrices):
    """Compute pairwise inter-model geodesic distance (GD) and
    centre/translation distance (CD) across ensemble predictions."""
    n = len(affine_matrices)
    model = self.models[0]

    gds, cds = [], []
    for i in range(n):
        mat_i = affine_matrices[i].unsqueeze(0)
        for j in range(i + 1, n):
            mat_j = affine_matrices[j].unsqueeze(0)
            gd_ij = model.geodesic_distance(mat_i[:, :3, :3], mat_j[:, :3, :3])
            cd_ij = model.translation_distance(mat_i[:, :3, 3], mat_j[:, :3, 3])
            gds.append(gd_ij.item())
            cds.append(cd_ij.item())

    return gds, cds


def _save_quality_control_report(self, affine_matrices, output_path):
    """Save a single-subject inter-model agreement report as JSON."""
    def _fmt(vals):
        vals = list(vals)
        return {
            "mean": f"{np.mean(vals):.4f}",
            "std": f"{np.std(vals):.4f}",
            "vals": vals,
        }

    gds, cds = self._quality_control(affine_matrices=affine_matrices)
    report = {"GD_QC": _fmt(gds), "CD_QC": _fmt(cds)}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

