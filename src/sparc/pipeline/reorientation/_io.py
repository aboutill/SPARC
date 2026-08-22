import json
import logging

import numpy as np


def print_model_info(self):
    """Log the configured reorientation model."""

    logging.info("Reorientation parameters:")
    logging.info(f"  Rotation averaging: {self.rotation_avg}")
    logging.info(f"Reorientation models: {self.models}")
    logging.info(f"  Directory: {self.models_dir}")
    if self.reo_stacks:
        logging.info("Input stacks reorientation activated")


@staticmethod
def print_qc(qc_report_path):
    """Log the mean geodesic/centre distance QC metrics from a saved
    reorientation QC report."""

    with open(qc_report_path) as f:
        report = json.load(f)
    to_deg = lambda s: float(s) * 180 / np.pi

    logging.info("Reorientation Quality Control:")
    logging.info(f"  Geodesic Distance: {to_deg(report['GD_QC']['mean']):.4f} [deg]")
    logging.info(f"  Centre Distance: {report['CD_QC']['mean']} [mm]")
