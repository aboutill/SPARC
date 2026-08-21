#!/usr/bin/env python
import os
import yaml
import pathlib
import argparse

from sparc.segmentation.tester import EnsembleTester as SegmentationEnsembleTester
from sparc.reorientation.tester import EnsembleTester as ReorientationEnsembleTester


def parse_args():
    
    # Initialize parser
    parser = argparse.ArgumentParser(
        prog="test",
        description=(
            "Test SPARC deep learning models via an ensemble setup.\n"
            "\n"
            "Required arguments:\n"
            " - Input directory.\n"
            " - Output directory.\n"
            " - Test task.\n"
            " - Configuration file.\n"
            " - Models directory.\n"
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
        help="Input directory.",
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
        help="Test task ['sesegmentationg', 'reorientation'].",
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
    parser.add_argument(
        "--models",
        type=pathlib.Path,
        required=True,
        help="Models directory.",
        metavar="\b",
    )
    
    # Optional arguments
    # Behaviour
    parser.add_argument(
        "--save_qc", 
        action="store_true",
        help="Save quality control metrics."
    ) 
    parser.add_argument(
        "--save_indiv", 
        action="store_true",
        help="Save individual model outputs."
    ) 
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of CPU workers. [Default: 8]",
        metavar="\b",
    )
    parser.add_argument(
        "-v", 
        "--verbose",
        action="store_true",
        help="Increase verbosity."
    ) 
    parser.add_argument(
        "--log",
        action="store_true",
        help="Activate logging."
    ) 
    
    # Parse arguments
    args = parser.parse_args()

    return args


def check_args(args):
    
    # Check input directory exists
    if not args.input.is_dir():
        raise ValueError(f"Input directory not found: {args.input}")  
    if not os.listdir(args.input):
        raise ValueError(f"Input directory {args.input} must be non empty.")
            
    # Check configuration file extension
    yml_exts = [".yml" , ".yaml"]
    if args.cfg is not None and not args.cfg.suffix in yml_exts:
       raise ValueError(f"Configuration file must have {'/'.join(yml_exts)} extension.")
    if not args.cfg.is_file():
       raise ValueError(f"Configuration file not found: {args.cfg}")
            
    # Load configuration
    cfg_path = args.cfg
    with open(cfg_path) as f:
        args.cfg = yaml.safe_load(f)
            
    # Check configuration fields
    cfg_fields = {
        "segmentation":
            ["data",
            "transforms",
            "unet",
            "inferer",
            "post_processing",],
        "reorientation":
            ["data",
            "transforms",
            "vit",
            "test",],
    }
    for cfg_field in cfg_fields[args.task]:
        if cfg_field not in args.cfg:
            raise ValueError(f"'{cfg_field}' must be a field in configuration file.")   
        if not isinstance(args.cfg[cfg_field], dict):
            raise ValueError(f"'{cfg_field}' field must be a dictionary.") 
            
    if not os.listdir(args.models):
        raise ValueError("Input models directory must be non empty.")
        
    return args


def main():
    
    #  Parse and check input arguments
    args = parse_args()
    args = check_args(args)
    
    # Instantiate trainer
    if args.task == "segmentation":
        tester = SegmentationEnsembleTester(
            data_cfg=args.cfg["data"],
            transforms_cfg=args.cfg["transforms"],
            unet_cfg=args.cfg["unet"],
            inferer_cfg=args.cfg["inferer"],
            post_processing_cfg=args.cfg["post_processing"],
        )
    elif args.task == "reorientation":
        tester = ReorientationEnsembleTester(
            data_cfg=args.cfg["data"],
            transforms_cfg=args.cfg["transforms"],
            vit_cfg=args.cfg["vit"],
            test_cfg=args.cfg["test"],
        )
        
    # Run trainer
    tester.run(
        input_dir=args.input,
        output_dir=args.output,
        models_dir=args.models,
        save_qc=args.save_qc,
        save_indiv=args.save_indiv,
        workers=args.workers,
        verbose=args.verbose,
        log=args.log,
    )


if __name__ == "__main__":
    
    main()
