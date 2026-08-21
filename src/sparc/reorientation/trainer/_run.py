import os
import logging
import torch

from monai.data import CacheDataset

from sparc.utils.io import init_datalist
from sparc.reorientation.transforms import (
    init_train_transforms, 
    init_val_transforms,
)
from sparc.reorientation.model import Model


def run(
        self,
        input_dirs,
        output_dir,
        models_dir=None,
        folds=5,
        workers=8,
        verbose=False,
        log=False,
    ): 
    """Run k-fold CV training across one or more input domains,
    optionally initialising each fold from pretrained weights."""
    
    # Get absolute path        
    input_dirs = [os.path.abspath(_input_dir) for _input_dir in input_dirs]
    output_dir = os.path.abspath(output_dir)
    if models_dir is not None:
        models_dir = os.path.abspath(models_dir)
        
    # Set output dir
    os.makedirs(output_dir, exist_ok=True)
        
    # Set data root dir
    if len(input_dirs) == 1:
        data_root_dir = input_dirs[0]
    else:
        data_root_dir = os.path.commonpath(input_dirs)
        
    # Setup logging
    log_path = os.path.join(output_dir, "sparc_reorientation_train.log") if log else None
    self.setup_logging(log_path=log_path, verbose=verbose)
    logging.info(f"Device: {self.device}")
    self.print_model_info()
    
    # Initialize multi-domain datalist
    datalists = [init_datalist(input_dir=input_dir, **self.data_cfg)
                 for input_dir in input_dirs]
        
    report_path = os.path.join(output_dir, "sparc_reorientation_train.json")
    self.save_model_info(
        datalists=datalists,
        output_path=report_path,
    )
    
    # Display step
    logging.info(f"{folds}-fold cross validation.")
    
    # Initialize path to transfer learning models
    models_paths = self.get_models_paths(models_dir, folds)
    
    # Initialize multi-domain cross-validation dataset
    train_sets, val_sets = self.init_CV_folds(datalists, folds)
   
    # 
    labels = self.data_cfg.get("mask") is not None
    
    # Cross-validation loop
    for fold in range(folds):
        
        # Display step
        logging.info(f"Fold {fold+1}/{folds}: data loading...")
        
        # Initialize train and val transforms
        train_transforms = init_train_transforms(
            labels=labels, 
            **self.transforms_cfg,
        )
        val_transforms = init_val_transforms(
            labels=labels,
            roi_size=self.transforms_cfg["roi_size"],
            pixdim=self.transforms_cfg["pixdim"],
        )
        
        # Initialize multi-domain train and val datasets
        train_ds = CacheDataset(
            data=train_sets[fold], 
            transform=train_transforms,
            num_workers=workers,
        )
        val_ds = CacheDataset(
            data=val_sets[fold], 
            transform=val_transforms,
            num_workers=workers,
        )
        
        # Initialize model
        model = Model(
            roi_size=self.transforms_cfg["roi_size"],
            pixdim=self.transforms_cfg["pixdim"],
            **self.vit_cfg,
        )   
        model.init_output_dir(
            output_dir=output_dir,
            fold=fold, 
            data_root_dir=data_root_dir,
        )
        model.to_device(device=self.device)
        
        # Transfer learning
        if models_paths is not None:
            state = torch.load(models_paths[fold])
            model.net.load_state_dict(state, strict=False)
                          
        # Display step
        logging.info(f"Fold {fold+1}/{folds}: network training...")     
        
        # Train model
        model.train(
            train_ds=train_ds, 
            val_ds=val_ds, 
            workers=workers,
            **self.train_cfg,
        )
        
        # Display step
        logging.info(f"Fold {fold+1}/{folds}: network validation...") 
        
        # Evaluate model on validation set
        model.validate(
            val_ds=val_ds, 
            workers=workers,
            **self.val_cfg,
        )