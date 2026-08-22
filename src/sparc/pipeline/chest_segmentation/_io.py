import json
import logging


def print_model_info(self):
    """Log the configured chest segmentation model and post-processing
    parameters.
    """

    logging.info("Chest segmentation:")
    if self.target_stack_orientation is not None:
        logging.info(f"  Target stack orientation '{self.target_stack_orientation}'")
    else:
        logging.info("  No target stack orientation specified")

    logging.info(f"Chest segmentation models: {self.models}")
    logging.info(f"  Directory: {self.models_dir}")

    logging.info("Chest segmentation postprocessing:")
    logging.info(f"  Dilation radius {self.dil_rad} [mm]")


@staticmethod
def print_qc(qc_report_path):
    """Log the mean Dice, HD, ASSD and volume QC metrics from a saved
    segmentation QC report."""

    with open(qc_report_path) as f:
        report = json.load(f)

    logging.info("Chest segmentation Quality Control:")
    logging.info(f"  Dice: {report['DICE_QC']['mean']}")
    logging.info(f"  Hausdorff Distance: {report['HD_QC']['mean']} [mm]")
    logging.info(f"  Average Surface Distance: {report['ASSD_QC']['mean']} [mm]")
    logging.info(f"  Volume: {report['VOL_QC']} [mm3]")
