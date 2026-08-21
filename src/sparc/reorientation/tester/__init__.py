import torch
import numpy as np

from scipy.spatial.transform import Rotation

from sparc.reorientation.so3 import geodesic_mean_so3


class EnsembleTester:
    """Loads N independently trained reorientation networks and runs
    ensemble inference, with optional per-model inter-agreement QC and
    validation metrics."""
    
    from ._io import (
        setup_logging,
        print_model_info,
        save_model_info,
        init_output_dir_layout,
        load_ensemble,
        save_arrs,
        save_metrics_csv,
    )
    from ._qc import (
        _quality_control,
        _save_quality_control_report,
    )
    from ._run import (
        _ensemble_prediction,
        _ensemble_prediction_batch_with_sampling,
        _ensemble_prediction_step_with_sampling,
        _ensemble_prediction_with_sampling,
        run,
        run_from_file,
    )

    def __init__(
            self, 
            transforms_cfg,
            vit_cfg, 
            data_cfg=None,
            test_cfg=None,
        ):
        """Store config and select device."""
        self.data_cfg = data_cfg
        self.transforms_cfg = transforms_cfg
        self.vit_cfg = vit_cfg
        self.test_cfg = test_cfg
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    
    @staticmethod
    def average_rotations(rotations, method="chordal"):
        """Average a batch of (N,3,3) rotation matrices via the
        requested strategy and return a single (3,3) numpy array.
        """
        rotations = np.asarray(rotations)
        R_obj = Rotation.from_matrix(rotations)

        if method == "quaternion":
            quat = R_obj.as_quat()
            avg_quat = quat.mean(axis=0)
            avg_quat /= np.linalg.norm(avg_quat)
            return Rotation.from_quat(avg_quat).as_matrix()
        elif method == "chordal":
            return R_obj.mean().as_matrix()
        elif method == "geodesic":
            return geodesic_mean_so3(
                [R_obj[i].as_matrix() for i in range(len(R_obj))]
            )
        else:
            raise ValueError(f"Unknown rotation_avg '{method}'.")
            
    
    def _predict_ensemble(self, img, rotation_avg="chordal", t_max=1/3):
        """Run every ensemble member's forward pass once on `img` and
        aggregate. Returns (avg_affine_matrix [1,4,4], per_model_affine
        [n_models,4,4]), both as tensors on self.device.
        """
       
        affine_matrix_preds = []
        for model in self.models:
            model.net.eval()
            with torch.no_grad():
                points_pred, _ = model.net(img)
                affine_matrix_preds.append(model.points_to_matrix(points_pred))
        affine_matrix_preds = torch.cat(affine_matrix_preds, dim=0)

        roi_size = self.transforms_cfg["roi_size"]
        t_min = [-t_max*s for s in roi_size]
        t_max = [t_max*s for s in roi_size]
        t_preds = affine_matrix_preds[:, :3, 3]
        avg_t = t_preds.mean(axis=0).cpu().numpy()
        avg_t = np.clip(avg_t, t_min, t_max)

        R_preds = affine_matrix_preds[:, :3, :3].cpu().numpy()
        avg_R = self.average_rotations(rotations=R_preds, method=rotation_avg)

        avg_affine_matrix = np.eye(4)
        avg_affine_matrix[:3, :3] = avg_R
        avg_affine_matrix[:3, 3] = avg_t
        avg_affine_matrix = (
            torch.from_numpy(avg_affine_matrix)
            .to(dtype=torch.float32, device=self.device)
            .unsqueeze(0)
        )

        return avg_affine_matrix, affine_matrix_preds