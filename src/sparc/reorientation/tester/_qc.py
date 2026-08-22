import os
import json

import numpy as np


def quality_control(self, affine_matrices):
    """Compute pairwise inter-model geodesic distance (GD) and
    centre distance (CD) across ensemble predictions,
    independently per sample.
    """
    n_models, batch_size = affine_matrices.shape[0], affine_matrices.shape[1]
    model = self.models[0]

    all_gds, all_cds = [], []
    for b in range(batch_size):
        gds, cds = [], []
        for i in range(n_models):
            mat_i = affine_matrices[i, b].unsqueeze(0)
            for j in range(i + 1, n_models):
                mat_j = affine_matrices[j, b].unsqueeze(0)
                gd_ij = model.geodesic_distance(mat_i[:, :3, :3], mat_j[:, :3, :3])
                cd_ij = model.centre_distance(mat_i[:, :3, 3], mat_j[:, :3, 3])
                gds.append(gd_ij.item())
                cds.append(cd_ij.item())
        all_gds.append(gds)
        all_cds.append(cds)

    return all_gds, all_cds


def save_quality_control_report(self, affine_matrices, output_path):
    """Save a single-subject inter-model agreement report as JSON."""

    def _fmt(vals):
        vals = list(vals)
        return {
            "mean": f"{np.mean(vals):.4f}",
            "std": f"{np.std(vals):.4f}",
            "vals": vals,
        }

    gds, cds = self.quality_control(affine_matrices=affine_matrices)
    report = {"GD_QC": _fmt(gds[0]), "CD_QC": _fmt(cds[0])}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
