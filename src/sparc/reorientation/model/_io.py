import os

import nibabel as nib
import numpy as np


def init_output_dir(self, output_dir, data_root_dir, fold=None):
    """Set up model/log/metrics/prediction output paths for one
    CV fold (or a single non-CV run if fold is None)."""

    self.data_root_dir = data_root_dir
    self.fold = fold
    if self.fold is not None:
        self.models_dir = os.path.join(output_dir, "models")
        self.checkpoint_filename = f"model_{self.fold}.pth"
        self.log_dir = os.path.join(output_dir, "logs", f"model_{self.fold}")
        self.metrics_dir = os.path.join(output_dir, "metrics", f"model_{self.fold}")
        self.model_path = os.path.join(self.models_dir, self.checkpoint_filename)
    else:
        self.checkpoint_filename = "model.pth"
        self.log_dir = os.path.join(output_dir, "logs")
        self.metrics_dir = os.path.join(output_dir, "metrics")
        self.model_path = os.path.join(output_dir, self.checkpoint_filename)
    self.prediction_dir = os.path.join(output_dir, "prediction")


def save_img(self, img, path):
    """Save a 3D array as a NIfTI file using the configured voxel spacing."""

    # Create affine matrix
    affine = np.eye(4)
    affine[0, 0] = self.pixdim[0]
    affine[1, 1] = self.pixdim[1]
    affine[2, 2] = self.pixdim[2]
    affine[:3, -1] = np.zeros(3)

    img = nib.Nifti1Image(img, affine)
    img.header.set_xyzt_units(2)
    img.header.set_qform(affine, code="aligned")
    img.header.set_sform(affine, code="scanner")

    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    nib.save(img, path)
