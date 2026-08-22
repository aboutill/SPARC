import torch

from monai.networks.nets import UNet
from monai.inferers import SlidingWindowInferer

torch.multiprocessing.set_sharing_strategy("file_system")


class Model:
    """A single segmentation network: architecture, inferer, device,
    and output-path bookkeeping."""

    # Import class methods
    from ._train import train
    from ._validate import validate
    from ._test import prediction
    from ._io import init_output_dir_layout

    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        channels,
        strides,
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=0,
        act="RELU",
        norm="BATCH",
        bias=False,
    ):
        """Build the MONAI UNet from architecture hyperparameters."""

        # Initialize network architecture
        self.net = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size,
            num_res_units=num_res_units,
            act=act,
            norm=norm,
            bias=bias,
        )
        self.out_channels = out_channels

        # Setup CUDA device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def init_inferer(
        self,
        roi_size,
        sw_batch_size=1,
        overlap=0.25,
        padding_mode="reflect",
    ):
        """Build the sliding-window inferer used for validation and
        test-time inference."""

        # Initialize inferer
        self.inferer = SlidingWindowInferer(
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            padding_mode=padding_mode,
        )

    def to_device(self, device):
        """Move the network to the given device (CPU/GPU)."""

        # Setup CUDA device
        self.device = device
        self.net = self.net.to(self.device)
