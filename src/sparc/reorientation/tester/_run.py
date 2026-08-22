import os
import logging

import numpy as np

from monai.data import CacheDataset, DataLoader

from sparc.utils.io import init_datalist
from sparc.reorientation.transforms import init_test_transforms


def _ensemble_prediction(
        self,
        test_ds,
        workers=8, 
        rotation_avg="chordal", 
        qc_report_path=None,
    ):
    """Run ensemble pose prediction on a single-item test_ds.

    Returns (avg_affine_matrix, affine_matrices) as (4,4) and
    (n_models,4,4) numpy arrays."""
    assert len(test_ds) == 1, (
        f"_ensemble_prediction expects a single-item test_ds, got {len(test_ds)} items."
    )

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
    )

    for batch in test_loader:
        img = batch["image"].to(self.device, non_blocking=True)
        avg_affine_matrix, affine_matrices = self._predict_ensemble(
            img=img,
            rotation_avg=rotation_avg,
        )
        # avg_affine_matrix: (1, 4, 4); affine_matrices: (n_models, 1, 4, 4)

    if qc_report_path:
        self.save_quality_control_report(
            affine_matrices=affine_matrices,
            output_path=qc_report_path,
        )
        
    avg_affine_matrix = avg_affine_matrix.squeeze(0).cpu().numpy()  # (4, 4)
    affine_matrices = affine_matrices.squeeze(1).cpu().numpy()

    return avg_affine_matrix, affine_matrices


def _ensemble_prediction_batch_with_sampling(
        self,
        img,
        file_paths,
        test_step,
        metrics_raw_df,
        indiv_metrics_raw_df=None,
        qc_raw_df=None,
        rotate_range=None,
        translate_range=None,
        rotation_avg="chordal",
    ):
    """Simulate an independent random ground-truth pose per sample in
    the batch, run ensemble inference, record metrics, and return a
    list of per-sample dicts (and, if requested, per-sample lists of
    per-model dicts) for saving.
    """
    model = self.models[0]
    batch_size = img.shape[0]

    # Each sample now gets its own independent pose.
    affine_matrix, (angle, t) = model.init_affine_matrix(
        n=batch_size, r_range=rotate_range, t_range=translate_range,
    )
    inv_affine_matrix = model.invert_affine_matrix(affine_matrix)

    img_def, _ = model.apply_transform(img, affine_matrix)

    avg_affine_matrix, affine_matrix_preds = self._predict_ensemble(
        img=img_def,
        rotation_avg=rotation_avg,
    )
    # avg_affine_matrix: (batch, 4, 4); affine_matrix_preds: (n_models, batch, 4, 4)

    img_pred, _ = model.apply_transform(img_def, avg_affine_matrix)

    R_gt, R_pred = inv_affine_matrix[:, :3, :3], avg_affine_matrix[:, :3, :3]
    t_gt, t_pred = inv_affine_matrix[:, :3, 3], avg_affine_matrix[:, :3, 3]
    points_pred = model.matrix_to_points(avg_affine_matrix)

    img_gt_np = img.detach().cpu().numpy().squeeze(1)
    img_pred_np = img_pred.detach().cpu().numpy().squeeze(1)
    img_def_np = img_def.detach().cpu().numpy().squeeze(1)

    gd = model.geodesic_distance(R_gt, R_pred, reduction="none")
    cd = model.centre_distance(t_gt, t_pred, reduction="none")

    avg_affine_matrix_np = avg_affine_matrix.detach().cpu().numpy()  # (batch, 4, 4)
    points_pred_np = points_pred.detach().cpu().numpy()              # (batch, ...)

    per_sample_arrs = []
    for b in range(batch_size):
        metrics_raw_df["filename"].append(file_paths[b])
        metrics_raw_df["angle"].append(angle[b])
        metrics_raw_df["translation"].append(np.linalg.norm(t[b]))
        metrics_raw_df["geodesic-distance"].append(gd[b].item())
        metrics_raw_df["center-distance"].append(cd[b].item())
        metrics_raw_df["step"].append(test_step)

        per_sample_arrs.append({
            "img_gt": img_gt_np[b],
            "img_def": img_def_np[b],
            "img_pred": img_pred_np[b],
            "affine_matrix": avg_affine_matrix_np[b],
            "anchor_points": points_pred_np[b],
        })

    per_sample_indiv_arrs = [[] for _ in range(batch_size)] if indiv_metrics_raw_df is not None else None

    if indiv_metrics_raw_df is not None or qc_raw_df is not None:
        if indiv_metrics_raw_df is not None:
            for i, model_i in enumerate(self.models):
                affine_matrix_pred_i = affine_matrix_preds[i]

                img_pred_i, _ = model_i.apply_transform(img_def, affine_matrix_pred_i)

                R_gt_i, R_pred_i = inv_affine_matrix[:, :3, :3], affine_matrix_pred_i[:, :3, :3]
                t_gt_i, t_pred_i = inv_affine_matrix[:, :3, 3], affine_matrix_pred_i[:, :3, 3]

                gd_i = model_i.geodesic_distance(R_gt_i, R_pred_i, reduction="none")
                cd_i = model_i.centre_distance(t_gt_i, t_pred_i, reduction="none")

                img_pred_i_np = img_pred_i.detach().cpu().numpy().squeeze(1)
                affine_matrix_pred_i_np = affine_matrix_pred_i.detach().cpu().numpy()
                points_pred_i_np = model_i.matrix_to_points(affine_matrix_pred_i).detach().cpu().numpy()

                for b in range(batch_size):
                    indiv_metrics_raw_df[i]["filename"].append(file_paths[b])
                    indiv_metrics_raw_df[i]["angle"].append(angle[b])
                    indiv_metrics_raw_df[i]["translation"].append(np.linalg.norm(t[b]))
                    indiv_metrics_raw_df[i]["geodesic-distance"].append(gd_i[b].item())
                    indiv_metrics_raw_df[i]["center-distance"].append(cd_i[b].item())
                    indiv_metrics_raw_df[i]["step"].append(test_step)

                    per_sample_indiv_arrs[b].append({
                        "img_gt": img_gt_np[b],
                        "img_def": img_def_np[b],
                        "img_pred": img_pred_i_np[b],
                        "affine_matrix": affine_matrix_pred_i_np[b],
                        "anchor_points": points_pred_i_np[b],
                    })

        if qc_raw_df is not None:
            gds, cds = self.quality_control(affine_matrix_preds)
            for b in range(batch_size):
                qc_raw_df["filename"].append(file_paths[b])
                qc_raw_df["angle"].append(angle[b])
                qc_raw_df["translation"].append(np.linalg.norm(t[b]))
                qc_raw_df["GD_QC"].append(np.mean(gds[b]))
                qc_raw_df["CD_QC"].append(np.mean(cds[b]))
                qc_raw_df["step"].append(test_step)

    return per_sample_arrs, per_sample_indiv_arrs


def _ensemble_prediction_step_with_sampling(
        self,
        test_loader,
        test_step,
        metrics_raw_df,
        indiv_metrics_raw_df=None,
        qc_raw_df=None,
        rotation_avg="chordal",
        rotate_range=None,
        translate_range=None,
    ):
    """Run one simulated-pose evaluation step over every subject in
    test_loader, saving predictions and recording metrics."""
    n_models = len(self.models)
    model = self.models[0]

    for batch in test_loader:
        img = batch["image"].to(model.device, non_blocking=True)
        file_paths = batch["image_meta_dict"]["filename_or_obj"]

        per_sample_arrs, per_sample_indiv_arrs = self._ensemble_prediction_batch_with_sampling(
            img=img, file_paths=file_paths, test_step=test_step,
            metrics_raw_df=metrics_raw_df, indiv_metrics_raw_df=indiv_metrics_raw_df,
            qc_raw_df=qc_raw_df, rotation_avg=rotation_avg,
            rotate_range=rotate_range, translate_range=translate_range,
        )

        for b, (file_path, arrs) in enumerate(zip(file_paths, per_sample_arrs)):
            dirname = os.path.dirname(file_path)
            folder = dirname.replace(self.data_root_dir, self.prediction_dir)
            os.makedirs(folder, exist_ok=True)

            self.save_arrs(arrs, folder, test_step, model)

            if self.indiv_prediction_dirs is not None:
                for i in range(n_models):
                    folder_i = dirname.replace(self.data_root_dir, self.indiv_prediction_dirs[i])
                    os.makedirs(folder_i, exist_ok=True)
                    self.save_arrs(per_sample_indiv_arrs[b][i], folder_i, test_step, model)


def _ensemble_prediction_with_sampling(
        self,
        test_ds,
        save_qc=False,
        save_indiv=False,
        num_test=5,
        rotation_avg="chordal",
        rotate_range=None,
        translate_range=None,
        workers=8,
    ):
    """Run repeated simulated-pose evaluation passes over test_ds,
    saving predictions, metrics, and (optionally) per-model outputs
    and inter-model QC.
    """
    n_models = len(self.models)

    if save_indiv:
        self.indiv_prediction_dirs, self.indiv_metrics_dirs = [], []
        for i in range(n_models):
            indiv_output_dir = os.path.join(self.output_dir, f"model{i}")
            self.indiv_prediction_dirs.append(os.path.join(indiv_output_dir, "prediction"))
            self.indiv_metrics_dirs.append(os.path.join(indiv_output_dir, "metrics"))
    else:
        self.indiv_prediction_dirs = None
        self.indiv_metrics_dirs = None

    test_loader = DataLoader(
        dataset=test_ds, 
        batch_size=len(test_ds), 
        shuffle=False, 
        num_workers=workers,
    )

    metrics_str = ["geodesic-distance", "center-distance"]
    headers_str = ["filename", "angle", "translation", "step"]
    qc_str = ["GD_QC", "CD_QC"]

    metrics_raw_df = {s: [] for s in headers_str + metrics_str}
    indiv_metrics_raw_df = (
        [{s: [] for s in headers_str + metrics_str} for _ in range(n_models)]
        if save_indiv else None
    )
    qc_raw_df = {s: [] for s in headers_str + qc_str} if save_qc else None

    for test_step in range(num_test):
        self._ensemble_prediction_step_with_sampling(
            test_loader=test_loader, test_step=test_step,
            metrics_raw_df=metrics_raw_df, indiv_metrics_raw_df=indiv_metrics_raw_df,
            qc_raw_df=qc_raw_df, rotation_avg=rotation_avg,
            rotate_range=rotate_range, translate_range=translate_range,
        )

    self.save_metrics_csv(metrics_raw_df, metrics_str, self.metrics_dir)

    if save_indiv:
        for i in range(n_models):
            self.save_metrics_csv(indiv_metrics_raw_df[i], metrics_str, self.indiv_metrics_dirs[i])

    if save_qc:
        self.save_metrics_csv(
            qc_raw_df, qc_str, self.qc_dir,
            raw_name="quality_control_raw.csv",
            summary_name="quality_control.csv",
        )


def run(
        self,
        input_dir,
        output_dir,
        models_dir, 
        save_qc=False,
        save_indiv=False,
        workers=8, 
        verbose=False,
        log=False,
    ):
    """Run simulated-pose ensemble evaluation over every subject in input_dir."""
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    models_dir = os.path.abspath(models_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "sparc_reorientation_test.log") if log else None
    self.setup_logging(log_path=log_path, verbose=verbose)
    logging.info(f"Device: {self.device}")
    self.print_model_info()

    datalist = init_datalist(input_dir=input_dir, img=self.data_cfg["img"])
    
    report_path = os.path.join(output_dir, "sparc_reorientation_test.json")
    self.save_model_info(datalist=datalist, output_path=report_path)
    
    subjects = sorted(datalist.keys())
    test_set = [datalist[s][j] for s in subjects for j in range(len(datalist[s]))]

    self.load_ensemble(models_dir)

    test_transforms = init_test_transforms(
        roi_size=self.transforms_cfg["roi_size"],
        pixdim=self.transforms_cfg["pixdim"],
    )
    test_ds = CacheDataset(
        data=test_set, 
        transform=test_transforms,
        num_workers=workers,
        progress=verbose,
    )

    self.init_output_dir_layout(output_dir=output_dir, data_root_dir=input_dir)

    logging.info("Ensemble prediction.")
    self._ensemble_prediction_with_sampling(
        test_ds=test_ds, 
        workers=workers, 
        save_qc=save_qc, 
        save_indiv=save_indiv, 
        **self.test_cfg,
    )
    
    
def run_from_file(
        self,
        input_path,
        output_path,
        models_dir,
        qc_report_path=None,
        indiv_pred_dir=None,
        rotation_avg="chordal",
        workers=8,
        verbose=False,
        log=False,
    ):
    """Run ensemble inference on a single input file"""
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    models_dir = os.path.abspath(models_dir)

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "sparc_reorientation_test.log") if log else None
    self.setup_logging(log_path=log_path, verbose=verbose)
    
    logging.info(f"Device: {self.device}")

    test_set = [{"image": input_path}]

    self.load_ensemble(models_dir)

    test_transforms = init_test_transforms(
        roi_size=self.transforms_cfg["roi_size"],
        pixdim=self.transforms_cfg["pixdim"],
    )
    test_ds = CacheDataset(
        data=test_set, 
        transform=test_transforms,
        num_workers=workers,
        progress=verbose,
    )

    self.output_dir = output_dir
    self.prediction_dir = output_dir
    self.qc_dir = os.path.join(output_dir, "quality_control")

    logging.info("Ensemble prediction.")
    avg_affine_matrix, affine_matrices = self._ensemble_prediction(
        test_ds=test_ds,
        workers=workers,
        rotation_avg=rotation_avg,
        qc_report_path=qc_report_path,
    )

    np.savetxt(output_path, avg_affine_matrix)

    if indiv_pred_dir is not None:
        filename = os.path.basename(output_path)
        os.makedirs(indiv_pred_dir, exist_ok=True)
        for i, affine_matrix in enumerate(affine_matrices):
            path_i = os.path.join(indiv_pred_dir, filename.replace(".txt", f"_model{i}.txt"))
            np.savetxt(path_i, affine_matrix)