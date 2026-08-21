import os

from monai.data import FolderLayout


def init_output_dir_layout(self, output_dir, fold=None, data_root_dir=None):
    """Set up model/log/metrics/prediction output paths for one
    CV fold (or a single non-CV run)."""
            
    # Set output folders
    if fold is not None:
        self.models_dir = os.path.join(output_dir, "models")
        self.checkpoint_filename = f"model_{fold}.pth"
        self.log_dir = os.path.join(output_dir, "logs", f"model_{fold}")
        self.metrics_dir = os.path.join(output_dir, "metrics", f"model_{fold}")
        self.model_path = os.path.join(self.models_dir, self.checkpoint_filename)
    else:
        self.checkpoint_filename = "model.pth"
        self.log_dir = os.path.join(output_dir, "logs")
        self.metrics_dir = os.path.join(output_dir, "metrics")
        self.model_path = os.path.join(output_dir, self.checkpoint_filename)
    self.prediction_dir = os.path.join(output_dir, "prediction")
    
    # Prediction folder layout
    self.prediction_layout = FolderLayout(
        output_dir=self.prediction_dir,
        postfix="pred",
        extension="nii.gz",
        makedirs=True,
        data_root_dir=data_root_dir,
    )