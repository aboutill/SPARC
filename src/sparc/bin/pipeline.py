#!/usr/bin/env python3
import os 
import yaml
import argparse
import pathlib
import logging

from sparc.pipeline import Pipeline
from sparc.cfg import PIPELINE_DEFAULT_CFG_PATH


def parse_args():
    
    # Initialize parser
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description=(
            "Run the Slice-to-volume Pipeline for Automated Reconstruction of\n" 
            "gated Cardiac cine (SPARC).\n"
            "\n"    
            "Required arguments:\n" 
            " - Input directory containing cine stacks in DICOM format.\n"
            " - Output directory.\n"
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
    
    # Optional arguments
    parser.add_argument(
        "--file_prefix",
        type=str,
        help="Output files prefix.",
        default=None,
        metavar="\b",
    )
    parser.add_argument(
        "--cfg",
        type=pathlib.Path,
        help="Configuration file.",
        default=None,
        metavar="\b",
    ) 
    parser.add_argument(
        "--models_chest_seg",
        type=pathlib.Path,
        help="Chest segmentation models directory.",
        default=None,
        metavar="\b",
    ) 
    parser.add_argument(
        "--models_chest_seg_cfg",
        type=pathlib.Path,
        help="Chest segmentation models configuration file.",
        default=None,
        metavar="\b",
    ) 
    parser.add_argument(
        "--models_heart_seg",
        type=pathlib.Path,
        help="Heart segmentation models directory.",
        default=None,
        metavar="\b",
    ) 
    parser.add_argument(
        "--models_heart_seg_cfg",
        type=pathlib.Path,
        help="Heart segmentation models configuration file.",
        default=None,
        metavar="\b",
    ) 
    parser.add_argument(
        "--models_reo",
        type=pathlib.Path,
        help="Reorientation models directory.",
        default=None,
        metavar="\b",
    ) 
    parser.add_argument(
        "--models_reo_cfg",
        type=pathlib.Path,
        help="Reorientation models configuration file.",
        default=None,
        metavar="\b",
    ) 
    
    # Behaviour
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Performs batch processing.",
    ) 
    parser.add_argument(
        "--mode",
        default="semi_auto",
        choices=["manual", "semi_auto", "monitored_auto", "full_auto"],
        help="Pipeline mode ['manual', 'semi_auto', 'monitored_auto', 'full_auto'].",
        metavar="\b",
    ) 
    parser.add_argument(
        "--gui_mode",
        default="docker",
        choices=["docker", "native"],
        help="Pipeline GUI mode ['docker', 'native'].",
        metavar="\b",
    ) 
    parser.add_argument(
        "--manual_stack_review",
        action="store_true",
        help="Activate manual stack review [disabled if --mode='full_auto'].",
    ) 
    parser.add_argument(
        "-v", 
        "--verbose",
        action="store_true",
        help="Increase verbosity.",
    ) 
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Show timing.",
    ) 
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save intermediate outputs.",
    ) 
    parser.add_argument(
        "--log",
        action="store_true",
        help="Activate logging.",
    ) 
    
    # Parse arguments
    args = parser.parse_args()

    return args


def check_args(args):
    
    # Check input directory exists
    if not args.input.is_dir():
        raise ValueError(f"Input directory not found: {args.input}")
    if not os.listdir(args.input):
        raise ValueError("Input directory must be non-empty.")
        
    # Check configuration file
    yml_exts = [".yml" , ".yaml"]
    if args.cfg is not None:
        if not args.cfg.is_file():
            raise ValueError(f"Configuration file not found: {args.cfg}")
        if args.cfg.suffix not in yml_exts:
           raise ValueError(f"Configuration file must have {'/'.join(yml_exts)} extension.")
            
    # Load configuration  
    cfg_path = args.cfg if args.cfg is not None else pathlib.Path(PIPELINE_DEFAULT_CFG_PATH)
    if not cfg_path.is_file():
        raise ValueError(f"Configuration file not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    args.cfg_path = cfg_path
    args.cfg = cfg
            
    # Check configuration fields
    cfg_fields = [
        "preprocessing",
        "chest_segmentation",
        "svr",
        "heart_segmentation",
        "reorientation",
        "postprocessing",
    ]
    for cfg_field in cfg_fields:
        if cfg_field not in args.cfg:
            raise ValueError(f"'{cfg_field}' must be a field in configuration file.")   
        if not isinstance(args.cfg[cfg_field], dict):
            raise ValueError(f"'{cfg_field}' field must be a dictionary.") 
            
    # Check models directories and configuration file
    if (args.models_chest_seg is None) != (args.models_chest_seg_cfg is None):
       raise ValueError("Invalid chest segmentation models combination.")
    if (args.models_heart_seg is None) != (args.models_heart_seg_cfg is None):
        raise ValueError("Invalid heart segmentation models combination.")
    if (args.models_reo is None) != (args.models_reo_cfg is None):
        raise ValueError("Invalid reorientation models combination.")
    
    # Chest segmentation
    if args.models_chest_seg is not None:
        if not args.models_chest_seg.is_dir():
            raise ValueError(f"Directory not found: {args.models_chest_seg}")
    if args.models_chest_seg_cfg is not None:
        if not args.models_chest_seg_cfg.is_file():
            raise ValueError(f"Configuration file not found: {args.models_chest_seg_cfg}")
        if args.models_chest_seg_cfg.suffix not in yml_exts:
           raise ValueError(f"Configuration file must have {'/'.join(yml_exts)} extension.")
        
    # Heart segmentation
    if args.models_heart_seg is not None:
        if not args.models_heart_seg.is_dir():
            raise ValueError(f"Directory not found: {args.models_heart_seg}")
    if args.models_heart_seg_cfg is not None:
        if not args.models_heart_seg_cfg.is_file():
            raise ValueError(f"Configuration file not found: {args.models_heart_seg_cfg}")
        if args.models_heart_seg_cfg.suffix not in yml_exts:
           raise ValueError(f"Configuration file must have {'/'.join(yml_exts)} extension.")
    
    # Reorientation
    if args.models_reo is not None:
        if not args.models_reo.is_dir():
            raise ValueError(f"Directory not found: {args.models_reo}")
    if args.models_reo_cfg is not None:
        if not args.models_reo_cfg.is_file():
            raise ValueError(f"Configuration file not found: {args.models_reo_cfg}")
        if args.models_reo_cfg.suffix not in yml_exts:
           raise ValueError(f"Configuration file must have {'/'.join(yml_exts)} extension.")
        
    return args
        

def main():
    
    # Parse and check input arguments
    args = parse_args()
    args = check_args(args)
    
    # Instantiate pipeline
    pipeline = Pipeline(
        preprocessing_cfg=args.cfg["preprocessing"],
        chest_segmentation_cfg=args.cfg["chest_segmentation"],
        svr_cfg=args.cfg["svr"],
        heart_segmentation_cfg=args.cfg["heart_segmentation"],
        reorientation_cfg=args.cfg["reorientation"],
        postprocessing_cfg=args.cfg["postprocessing"],
        models_chest_seg_dir=args.models_chest_seg,
        models_chest_seg_cfg_path=args.models_chest_seg_cfg,
        models_heart_seg_dir=args.models_heart_seg,
        models_heart_seg_cfg_path=args.models_heart_seg_cfg,
        models_reo_dir=args.models_reo,
        models_reo_cfg_path=args.models_reo_cfg,
    )
    
    
    # Batch processing
    if args.batch:
        
        # Build subject list
        sub_ids = [dirname for dirname in os.listdir(args.input) 
                   if os.path.isdir(os.path.join(args.input, dirname))]
        
        if not sub_ids:
            raise ValueError(f"No subject subdirectories found in {args.input}"
                             " for batch processing.")
        
        failed_sub_ids = []
        
        for sub_id in sub_ids:
            
            # Set directories
            sub_input_dir = os.path.join(args.input, sub_id)
            sub_output_dir = os.path.join(args.output, sub_id)
            file_prefix = sub_id
            
            # Run gated pipeline
            try:
                pipeline.run(
                    input_dir=sub_input_dir,
                    output_dir=sub_output_dir,
                    file_prefix=file_prefix,
                    mode=args.mode,
                    gui_mode=args.gui_mode,
                    manual_stack_review=args.manual_stack_review,
                    verbose=args.verbose,
                    profile=args.profile,
                    debug=args.debug,
                    log=args.log,
                )
            except Exception:
                logging.exception(
                    f"Subject {sub_id} failed with an unhandled exception, "
                    "continuing with the remaining subjects."
                )
                failed_sub_ids.append(sub_id)
        
        # Summary
        n_total = len(sub_ids)
        n_failed = len(failed_sub_ids)
        logging.info(
            f"Batch complete: {n_total - n_failed}/{n_total} subjects "
            f"finished without raising an exception."
        )
        if failed_sub_ids:
            logging.info(f"Subjects that raised an exception: {failed_sub_ids}")
    
    # Subject processing
    else:
        
        # Run pipeline
        pipeline.run(
            input_dir=args.input,
            output_dir=args.output,
            file_prefix=args.file_prefix,
            mode=args.mode,
            gui_mode=args.gui_mode,
            manual_stack_review=args.manual_stack_review,
            verbose=args.verbose,
            profile=args.profile,
            debug=args.debug,
            log=args.log,
        )


if __name__ == "__main__":
    
    main()