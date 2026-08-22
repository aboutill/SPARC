import os
import shutil
import glob
import tempfile
import logging

from monai.transforms import (
    Compose,
    Activationsd,
    AsDiscreted,
    SaveImaged,
    KeepLargestConnectedComponentd,
    FillHolesd,
    Invertd,
    VoteEnsembled,
)
from monai.data import CacheDataset, DataLoader, FolderLayout
from monai.engines import EnsembleEvaluator
from monai.handlers import (
    MetricsSaver,
    MeanDice,
    HausdorffDistance,
    SurfaceDistance,
)
from monai.handlers.utils import from_engine

from sparc.utils.io import init_datalist
from sparc.segmentation.transforms import (
    init_val_org_transforms,
    init_test_transforms,
)


def _predict_individual_models_to_dirs(
    self,
    test_ds,
    output_dir,
    data_root_dir,
    labels,
    workers,
):
    """Run each ensemble member's own prediction into its own subdirectory.
    Returns the per-model output directories, in ensemble order.
    """
    model_output_dirs = []
    n_models = len(self.models)
    for i, model in enumerate(self.models):
        logging.info(f"Individual model prediction {i+1}/{n_models}.")
        model_output_dir = os.path.join(output_dir, f"model{i}")
        model_output_dirs.append(model_output_dir)
        model.init_output_dir_layout(
            output_dir=model_output_dir,
            data_root_dir=data_root_dir,
        )
        model.prediction(
            test_ds=test_ds,
            workers=workers,
            labels=labels,
            **self.post_processing_cfg,
        )
    return model_output_dirs


def _ensemble_prediction(
    self,
    test_ds,
    labels=False,
    num_components=1,
    connectivity=None,
    workers=8,
):
    """Run majority-vote ensemble inference and optional label-based metrics."""

    # Initialize keys
    n_models = len(self.models)
    keys = [f"pred{i}" for i in range(n_models)]
    networks = [model.net for model in self.models]

    # Initialize data loaders
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
    )

    # Initialise multi-class options
    if self.out_channels == 1:
        sigmoid = True
        softmax = False
        threshold = 0.5
        argmax = False
        to_onehot = None
        include_background = True
        is_onehot = False
    else:
        sigmoid = False
        softmax = True
        threshold = None
        argmax = True
        to_onehot = self.out_channels
        include_background = False
        is_onehot = True

    # Initialize post prediction transformation
    post_transforms_list = [
        Invertd(
            keys=keys,
            transform=test_ds.transform,
            orig_keys=["image"] * n_models,
            meta_keys=["pred_meta_dict"] * n_models,
            orig_meta_keys=["image_meta_dict"] * n_models,
            meta_key_postfix=["meta_dict"] * n_models,
            nearest_interp=False,
            to_tensor=True,
        ),
        Activationsd(keys=keys, softmax=softmax, sigmoid=sigmoid),
        AsDiscreted(keys=keys, threshold=threshold, argmax=argmax, to_onehot=to_onehot),
    ]
    if labels:
        post_transforms_list.append(AsDiscreted(keys="label", to_onehot=to_onehot))
    post_transforms_list += [
        VoteEnsembled(
            keys=keys,
            output_key="pred",
        ),
        KeepLargestConnectedComponentd(
            keys="pred",
            num_components=num_components,
            connectivity=connectivity,
            is_onehot=is_onehot,
        ),
        FillHolesd(
            keys="pred",
            connectivity=connectivity,
        ),
        SaveImaged(
            keys="pred",
            meta_keys="pred_meta_dict",
            folder_layout=self.prediction_layout,
        ),
    ]
    post_transforms = Compose(post_transforms_list)

    # Validation handler
    if labels:
        val_handlers = [
            MetricsSaver(
                save_dir=self.metrics_dir,
                metrics=[
                    "val_mean_dice",
                    "val_hausdorff_distance",
                    "val_surface_distance",
                ],
                metric_details="*",
                batch_transform=from_engine("image_meta_dict"),
                summary_ops="*",
            ),
        ]
        key_val_metric = {
            "val_mean_dice": MeanDice(
                output_transform=from_engine(["pred", "label"]),
                include_background=include_background,
            ),
        }
        additional_metrics = {
            "val_hausdorff_distance": HausdorffDistance(
                output_transform=from_engine(["pred", "label"]),
                percentile=95,
            ),
            "val_surface_distance": SurfaceDistance(
                output_transform=from_engine(["pred", "label"]),
                symmetric=True,
            ),
        }
    else:
        val_handlers = None
        key_val_metric = None
        additional_metrics = None

    # Validation engine
    evaluator = EnsembleEvaluator(
        device=self.device,
        val_data_loader=test_loader,
        pred_keys=keys,
        networks=networks,
        inferer=self.inferer,
        postprocessing=post_transforms,
        key_val_metric=key_val_metric,
        additional_metrics=additional_metrics,
        val_handlers=val_handlers,
    )

    # Perform evaluation
    evaluator.run()


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
    """Run ensemble inference over every subject in `input_dir`."""

    # Get absolute path
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    models_dir = os.path.abspath(models_dir)

    # Set output dir
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_path = os.path.join(output_dir, "sparc_segmentation_test.log") if log else None
    self.setup_logging(log_path=log_path, verbose=verbose)
    logging.info(f"Device: {self.device}")
    self.print_model_info()

    # Init datalist
    datalist = init_datalist(input_dir=input_dir, **self.data_cfg)

    report_path = os.path.join(output_dir, "sparc_segmentation_test.json")
    self.save_model_info(
        datalist=datalist,
        output_path=report_path,
    )

    subjects = sorted(list(datalist.keys()))
    test_set = [datalist[s][j] for s in subjects for j in range(len(datalist[s]))]

    labels = self.data_cfg.get("mask") is not None

    # Initialize models
    reg_ex = os.path.join(models_dir, "*.pth")
    models_paths = sorted(glob.glob(reg_ex))
    if len(models_paths) < 2:
        raise ValueError("Ensemble inference requires at least two models.")

    # Load models
    self.load_ensemble(models_dir)

    # Initialize test transform
    test_transforms = init_val_org_transforms(
        pixdim=self.transforms_cfg["pixdim"],
    )

    # Initialize test dataset
    test_ds = CacheDataset(
        data=test_set,
        transform=test_transforms,
        num_workers=workers,
        progress=verbose,
    )

    # Set output path
    ensemble_output_dir = os.path.join(output_dir, "ensemble")
    self.init_output_dir_layout(
        output_dir=ensemble_output_dir,
        data_root_dir=input_dir,
    )

    # Ensemble prediction
    logging.info("Ensemble prediction.")
    self._ensemble_prediction(
        test_ds=test_ds,
        workers=workers,
        labels=labels,
        **self.post_processing_cfg,
    )

    # Individual predictions
    model_output_dirs = []
    if save_qc or save_indiv:
        model_output_dirs = self._predict_individual_models_to_dirs(
            test_ds=test_ds,
            output_dir=output_dir,
            data_root_dir=input_dir,
            labels=labels,
            workers=workers,
        )

    if save_qc:
        logging.info("Quality control.")

        p = os.path.join(self.prediction_dir, "**", "*pred.nii.gz")
        ensemble_mask_paths = sorted(glob.glob(p, recursive=True))
        indiv_mask_paths = [
            [p.replace(ensemble_output_dir, model_dir) for model_dir in model_output_dirs]
            for p in ensemble_mask_paths
        ]
        self._save_quality_control_dataframe(
            ensemble_mask_paths=ensemble_mask_paths,
            indiv_mask_paths=indiv_mask_paths,
        )

    if save_qc and not save_indiv:
        for model_dir in model_output_dirs:
            shutil.rmtree(model_dir, ignore_errors=True)


def run_from_file(
    self,
    input_path,
    output_path,
    models_dir,
    qc_report_path=None,
    indiv_pred_dir=None,
    workers=8,
    verbose=False,
    log=False,
):
    """Run ensemble inference on a single input file."""
    # Get absolute path
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    models_dir = os.path.abspath(models_dir)

    # Set output dir
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_path = os.path.join(output_dir, "sparc_segmentation_test.log") if log else None
    self.setup_logging(log_path=log_path, verbose=verbose)

    logging.info(f"Device: {self.device}")

    # Init datalist
    test_set = [{"image": input_path}]

    # Initialize models
    self.load_ensemble(models_dir)

    # Initialize test transform
    test_transforms = init_test_transforms(
        pixdim=self.transforms_cfg["pixdim"],
    )

    # Initialize test dataset
    test_ds = CacheDataset(
        data=test_set,
        transform=test_transforms,
        num_workers=workers,
        progress=verbose,
    )

    # Prediction folder layout
    self.prediction_layout = FolderLayout(
        output_dir=output_dir,
        postfix="pred",
        extension="nii.gz",
        makedirs=True,
    )

    logging.info("Ensemble prediction.")

    # Ensemble prediction
    self._ensemble_prediction(
        test_ds=test_ds,
        workers=workers,
        **self.post_processing_cfg,
    )

    # Rename file; prediction file name set by MONAI Folder Layout
    os.rename(input_path.replace(".nii.gz", "_pred.nii.gz"), output_path)

    # Individual predictions
    indiv_pred = indiv_pred_dir is not None
    qc = qc_report_path is not None
    if qc or indiv_pred:
        if indiv_pred_dir is None:
            indiv_pred_dir = tempfile.mkdtemp()

        n_models = len(self.models)
        for i in range(n_models):

            logging.info(f"Individual model prediction {i+1}/{n_models}.")

            # Set output path
            self.models[i].prediction_layout = FolderLayout(
                output_dir=indiv_pred_dir,
                postfix=f"pred_model{i}",
                extension="nii.gz",
                makedirs=True,
            )

            # Individual model prediction
            self.models[i].prediction(
                test_ds=test_ds,
                workers=workers,
                **self.post_processing_cfg,
            )

    if qc:
        logging.info("Quality control.")

        p = os.path.join(indiv_pred_dir, "*pred_model*.nii.gz")
        indiv_mask_paths = sorted(glob.glob(p))

        self.save_quality_control_report(
            ensemble_mask_path=output_path,
            indiv_mask_paths=indiv_mask_paths,
            output_path=qc_report_path,
        )

    if qc and not indiv_pred:

        shutil.rmtree(indiv_pred_dir, ignore_errors=True)
