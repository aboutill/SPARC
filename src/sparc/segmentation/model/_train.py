import torch.optim as optim

from monai.data import DataLoader
from monai.handlers import (
    MeanDice,
    StatsHandler, 
    ValidationHandler, 
    CheckpointSaver, 
    TensorBoardStatsHandler, 
    TensorBoardImageHandler, 
    from_engine,
)
from monai.transforms import Compose, EnsureTyped, Activationsd, AsDiscreted
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.inferers import SimpleInferer
from monai.losses import DiceCELoss


def train(
        self, 
        train_ds, 
        val_ds, 
        learning_rate=1e-3, 
        batch_size=64, 
        epochs=200, 
        workers=8,
    ):   
    """Supervised training loop for one segmentation network/fold, using
    MONAI engines with TensorBoard logging and best-checkpoint saving."""
    
    # Initialize data loaders
    train_loader = DataLoader(
        dataset=train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=workers,
    )
    val_loader = DataLoader(
        dataset=val_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=workers,
    )
    
    # Initialise multi-class options
    if self.out_channels == 1:
        sigmoid = True
        softmax = False
        to_onehot_y = False
        threshold = 0.5
        argmax = False
        to_onehot = None
        include_background = True
    else:
        sigmoid = False
        softmax = True
        to_onehot_y = True
        threshold = None
        argmax = True
        to_onehot = self.out_channels
        include_background = False
        
    # Initialize optimizer and loss function
    optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
    loss_function = DiceCELoss(sigmoid=sigmoid, softmax=softmax, to_onehot_y=to_onehot_y)
        
    # Initialize post prediction transforms
    val_post_transforms = Compose(
       [
            EnsureTyped(keys="pred"), 
            Activationsd(keys="pred", softmax=softmax, sigmoid=sigmoid),
            AsDiscreted(keys="pred", threshold=threshold, argmax=argmax, to_onehot=to_onehot),
            AsDiscreted(keys="label", to_onehot=to_onehot),
        ]
    )
        
    # Validation handlers
    val_handlers = [
        CheckpointSaver(
            save_dir=self.models_dir, 
            save_dict={"network": self.net},
            save_key_metric=True,
            key_metric_name="val_mean_dice",
            key_metric_filename=self.checkpoint_filename,   
        ),
        TensorBoardStatsHandler(
            log_dir=self.log_dir, 
            tag_name="val_mean_dice",
            output_transform=lambda x: None,
            global_epoch_transform=lambda x: trainer.state.epoch,
        ),
    ]
    
    # Validation engine
    evaluator = SupervisedEvaluator(
        device=self.device,
        val_data_loader=val_loader,
        network=self.net,
        inferer=self.inferer,
        postprocessing=val_post_transforms,
        key_val_metric={
            "val_mean_dice": MeanDice(
                output_transform=from_engine(["pred", "label"]),
                include_background=include_background,
            ),
        },
        val_handlers=val_handlers   
    )
    
    # Train handler
    train_handlers = [
        ValidationHandler(
            validator=evaluator, 
            interval=1
        ),
        StatsHandler(
            tag_name="train_loss", 
            output_transform=from_engine(["loss"], first=True),
        ),
        TensorBoardStatsHandler(
            log_dir=self.log_dir, 
            tag_name="train_loss", 
            output_transform=from_engine(["loss"], first=True),
        ),
        TensorBoardImageHandler(
            log_dir=self.log_dir,
            interval=10,
            batch_transform=from_engine(["image", "label"]),
            output_transform=from_engine(["pred"]),
            frame_dim=-1,
        )
    ]
 
    # Trainer engine
    trainer = SupervisedTrainer(
        device=self.device,
        max_epochs=epochs,
        train_data_loader=train_loader,
        network=self.net,
        optimizer=optimizer,
        loss_function=loss_function,
        inferer=SimpleInferer(),
        train_handlers=train_handlers,
    )
    
    # Perform training and validation
    trainer.run()