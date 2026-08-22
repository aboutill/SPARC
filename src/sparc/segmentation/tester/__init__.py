import torch

from monai.inferers import SlidingWindowInferer


class EnsembleTester:
    """Loads N independently trained segmentation networks and runs
    majority-vote ensemble inference, with optional per-model
    inter-agreement QC and label-based validation metrics."""

    from ._io import (
        setup_logging,
        print_model_info,
        save_model_info,
        init_output_dir_layout,
        load_ensemble,
        load_mask_as_tensor,
    )
    from ._qc import (
        quality_control,
        _save_quality_control_dataframe,
        save_quality_control_report,
    )
    from ._run import (
        _predict_individual_models_to_dirs,
        _ensemble_prediction,
        run,
        run_from_file,
    )

    def __init__(
        self,
        transforms_cfg,
        unet_cfg,
        inferer_cfg,
        post_processing_cfg=None,
        data_cfg=None,
    ):
        """Store config, select device, build the shared inferer."""

        self.data_cfg = data_cfg
        self.transforms_cfg = transforms_cfg
        self.unet_cfg = unet_cfg
        self.inferer_cfg = inferer_cfg
        self.post_processing_cfg = post_processing_cfg

        # Multi-class setup
        self.out_channels = unet_cfg["out_channels"]

        # Setup CUDA device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Init inferer
        self.init_inferer(**self.inferer_cfg)

    def init_inferer(
        self,
        roi_size,
        sw_batch_size=1,
        overlap=0.25,
        padding_mode="reflect",
    ):
        """Build the sliding-window inferer used at test-time inference."""
        # Initialize inferer
        self.inferer = SlidingWindowInferer(
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            padding_mode=padding_mode,
        )
