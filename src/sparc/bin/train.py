#!/usr/bin/env python
import os
import yaml
import pathlib
import argparse

from sparc.segmentation.trainer import MultiDomainTrainer as SegmentationMultiDomainTrainer
from sparc.reorientation.trainer import MultiDomainTrainer as ReorientationMultiDomainTrainer


def parse_args():

    # Initialize parser
    parser = argparse.ArgumentParser(
        prog="train",
        description=(
            "Train SPARC deep learning models via a cross-validation (CV) setup.\n"
            "Training can be performed across multiple-domain defined by separate\n"
            "input directories. Transfer learning can be activated by providing\n"
            "initialisation model weights.\n"
            "\n"
            "Required arguments:\n"
            " - Input directory(ies).\n"
            " - Output directory.\n"
            " - Training task.\n"
            " - Configuration file.\n"
        ),
        epilog="Arnaud Boutillon (arnaud.boutillon@kcl.ac.uk)",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=6),
    )

    # Initialize arguments
    # Required arguments
    parser.add_argument(
        "-i",
        "--input",
        type=pathlib.Path,
        action="append",
        help="Input directory(ies); repeat -i/--input once per domain.",
        required=True,
        metavar="\b",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="Output directory.",
        required=True,
        metavar="\b",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["segmentation", "reorientation"],
        help="Training task ['segmentation', 'reorientation'].",
        required=True,
        metavar="\b",
    )
    parser.add_argument(
        "--cfg",
        type=pathlib.Path,
        help="Configuration file.",
        required=True,
        metavar="\b",
    )

    # Optional arguments
    # Cross-validation folds
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Cross-validation folds. [Default: 5]",
        metavar="\b",
    )
    # Transfer learning weights
    parser.add_argument(
        "--models",
        type=pathlib.Path,
        default=None,
        help="Models directory for transfer learning.",
        metavar="\b",
    )
    # Behaviour
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of CPU workers. [Default: 8]",
        metavar="\b",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase verbosity.")
    parser.add_argument("--log", action="store_true", help="Activate logging.")

    # Parse arguments
    args = parser.parse_args()

    return args


def check_args(args):

    # Number of folds
    if args.folds < 3:
        raise ValueError(f"Number of folds {args.folds} must be >= 3.")

    # Check input directory exists
    for in_dir in args.input:
        if not in_dir.is_dir():
            raise ValueError(f"Input directory not found: {in_dir}")
        if not os.listdir(in_dir):
            raise ValueError(f"Input directory {in_dir} must be non-empty.")

    # Check configuration file extension
    yml_exts = [".yml", ".yaml"]
    if args.cfg is not None and args.cfg.suffix not in yml_exts:
        raise ValueError(f"Configuration file must have {'/'.join(yml_exts)} extension.")
    if not args.cfg.is_file():
        raise ValueError(f"Configuration file not found: {args.cfg}")

    # Load configuration
    cfg_path = args.cfg
    with open(cfg_path) as f:
        args.cfg = yaml.safe_load(f)

    # Check configuration fields
    cfg_fields = {
        "segmentation": [
            "data",
            "transforms",
            "unet",
            "train",
            "inferer",
            "post_processing",
        ],
        "reorientation": [
            "data",
            "transforms",
            "vit",
            "train",
            "val",
        ],
    }
    for cfg_field in cfg_fields[args.task]:
        if cfg_field not in args.cfg:
            raise ValueError(f"'{cfg_field}' must be a field in configuration file.")
        if not isinstance(args.cfg[cfg_field], dict):
            raise ValueError(f"'{cfg_field}' field must be a dictionary.")

    if args.models is not None:
        if not os.listdir(args.models):
            raise ValueError("Input models directory must be non empty.")

    return args


def main():

    # Parse and check input arguments
    args = parse_args()
    args = check_args(args)

    # Instantiate trainer
    if args.task == "segmentation":
        trainer = SegmentationMultiDomainTrainer(
            data_cfg=args.cfg["data"],
            transforms_cfg=args.cfg["transforms"],
            unet_cfg=args.cfg["unet"],
            train_cfg=args.cfg["train"],
            inferer_cfg=args.cfg["inferer"],
            post_processing_cfg=args.cfg["post_processing"],
        )
    elif args.task == "reorientation":
        trainer = ReorientationMultiDomainTrainer(
            data_cfg=args.cfg["data"],
            transforms_cfg=args.cfg["transforms"],
            vit_cfg=args.cfg["vit"],
            train_cfg=args.cfg["train"],
            val_cfg=args.cfg["val"],
        )

    # Run trainer
    trainer.run(
        input_dirs=args.input,
        output_dir=args.output,
        models_dir=args.models,
        folds=args.folds,
        workers=args.workers,
        verbose=args.verbose,
        log=args.log,
    )


if __name__ == "__main__":

    main()
