import os
import json
import logging

from sparc.utils.logging import setup_logging_config


@staticmethod
def setup_logging(log_path=None, verbose=False):
    """Configure logging for this pipeline run."""
    setup_logging_config(
        log_path=log_path,
        verbose=verbose,
    )


def print_pipeline_info(self):
    """Log every stage's resolved configuration."""

    logging.info("Pipeline settings:")
    self.pre_processor.print_model_info()
    self.chest_segmentator.print_model_info()
    self.svr_reconstructor.print_model_info()
    self.heart_segmentator.print_model_info()
    self.reorientor.print_model_info()
    self.post_processor.print_model_info()


def save_pipeline_info(
    self,
    input_dir,
    output_path,
):
    """Save inputs and every stage's resolved configuration to a
    JSON report."""

    # Pipeline configuration
    cfgs = {
        "preprocessing": self.preprocessing_cfg,
        "chest_segmentation": self.chest_segmentation_cfg,
        "svr": self.svr_cfg,
        "heart_segmentation": self.heart_segmentation_cfg,
        "reorientation": self.reorientation_cfg,
        "postprocessing": self.postprocessing_cfg,
    }
    report = {
        "input": str(input_dir),
        "mode": self.mode,
    }
    for k, v in cfgs.items():
        if v is not None:
            report[k] = v

    # Chest segmentation model configuration
    report["chest_segmentation"]["models"] = self.chest_segmentator.models
    report["chest_segmentation"]["models_dir"] = str(self.chest_segmentator.models_dir)
    report["chest_segmentation"]["unet"] = self.chest_segmentator.unet_cfg
    report["chest_segmentation"]["inferer"] = self.chest_segmentator.inferer_cfg
    report["chest_segmentation"]["transforms"] = self.chest_segmentator.transforms_cfg

    # Heart segmentation model configuration
    report["heart_segmentation"]["models"] = self.heart_segmentator.models
    report["heart_segmentation"]["models_dir"] = str(self.heart_segmentator.models_dir)
    report["heart_segmentation"]["unet"] = self.heart_segmentator.unet_cfg
    report["heart_segmentation"]["inferer"] = self.heart_segmentator.inferer_cfg
    report["heart_segmentation"]["transforms"] = self.heart_segmentator.transforms_cfg

    # Reorientation model configuration
    report["reorientation"]["models"] = self.reorientor.models
    report["reorientation"]["models_dir"] = str(self.reorientor.models_dir)
    report["reorientation"]["vit"] = self.reorientor.vit_cfg
    report["reorientation"]["transforms"] = self.reorientor.transforms_cfg

    # Save report
    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)


@staticmethod
def save_qc_report(
    output_path,
    elapsed_seconds,
    processing_seconds,
    chest_mask_qc_path=None,
    cine_qc_report_path=None,
    heart_mask_qc_path=None,
    reo_qc_report_path=None,
):
    """Save pipeline QC to a JSON report."""

    report = {
        "pipeline_time": round(elapsed_seconds, 2),
        "processing_time": round(processing_seconds, 2),
    }

    if chest_mask_qc_path is not None and os.path.exists(chest_mask_qc_path):
        with open(chest_mask_qc_path) as f:
            chest_mask_report = json.load(f)
        data = {}
        data["DICE_QC"] = chest_mask_report["DICE_QC"]["mean"]
        data["HD_QC"] = chest_mask_report["HD_QC"]["mean"]
        data["ASSD_QC"] = chest_mask_report["ASSD_QC"]["mean"]
        data["VOL_QC"] = chest_mask_report["VOL_QC"]
        report["chest_segmentation"] = data

    if cine_qc_report_path is not None and os.path.exists(cine_qc_report_path):
        with open(cine_qc_report_path) as f:
            cine_report = json.load(f)
        report["reconstruction"] = cine_report["evaluation"]

    if heart_mask_qc_path is not None and os.path.exists(heart_mask_qc_path):
        with open(heart_mask_qc_path) as f:
            heart_mask_report = json.load(f)
        data = {}
        data["DICE_QC"] = heart_mask_report["DICE_QC"]["mean"]
        data["HD_QC"] = heart_mask_report["HD_QC"]["mean"]
        data["ASSD_QC"] = heart_mask_report["ASSD_QC"]["mean"]
        data["VOL_QC"] = heart_mask_report["VOL_QC"]
        report["heart_segmentation"] = data

    if reo_qc_report_path is not None and os.path.exists(reo_qc_report_path):
        with open(reo_qc_report_path) as f:
            reo_report = json.load(f)
        data = {}
        data["GD_QC"] = reo_report["GD_QC"]["mean"]
        data["CD_QC"] = reo_report["CD_QC"]["mean"]
        report["reorientation"] = data

    # Save report
    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)
