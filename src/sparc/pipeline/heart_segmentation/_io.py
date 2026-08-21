import json
import logging


def print_model_info(self):
    """Log the configured heart segmentation model."""
    
    if self.activate:
        logging.info(f"Heart segmentation models: {self.models}")
        logging.info(f"  Directory: {self.models_dir}")
    

@staticmethod
def print_qc(qc_report_path):
    """Log the mean Dice, HD, ASSD and volume QC metrics from a saved
    segmentation QC report."""

    with open(qc_report_path) as f:
        report = json.load(f)

    logging.info("Heart segmentation Quality Control:")
    logging.info(f"  Dice: {report['DICE_QC']['mean']}")
    logging.info(f"  Hausdorff Distance: {report['HD_QC']['mean']} [mm]")
    logging.info(f"  Average Surface Distance: {report['ASSD_QC']['mean']} [mm]")
    logging.info(f"  Volume: {report['VOL_QC']} [mm3]")  