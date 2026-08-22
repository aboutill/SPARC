import numpy as np


class SliceVolumeReconstructor:
    """Configures and runs gated slice-to-volume cine reconstruction."""

    from ._io import print_model_info
    from ._run import (
        cine_post_processing,
        N4_corr_funct,
        smooth_roi_mask_funct,
        run,
    )

    def __init__(
        self,
        resolution,
        iterations,
        sr_iterations,
        lambda_smooth,
        delta_edge,
        no_stack_zcrop,
        mask_slices_not_svr,
        no_robust_statistics,
        no_intensity_matching,
        N4_corr,
        smooth_roi_mask,
    ):
        """Store SVR reconstruction parameters."""
        self.resolution = resolution
        self.iterations = iterations
        self.sr_iterations = sr_iterations
        self.lambda_smooth = lambda_smooth
        self.delta_edge = delta_edge
        self.no_stack_zcrop = no_stack_zcrop
        self.mask_slices_not_svr = mask_slices_not_svr
        self.no_robust_statistics = no_robust_statistics
        self.no_intensity_matching = no_intensity_matching
        self.N4_corr = N4_corr
        self.smooth_roi_mask = smooth_roi_mask

    def init_svr_cfg(
        self,
        stack_infos,
    ):
        """Build the svr_reconstruct configuration dict from per-stack
        metadata.
        """
        # Init cfg
        cfg = {
            "resolution": self.resolution,
            "iterations": self.iterations,
            "sr_iterations": self.sr_iterations,
            "lambda_smooth": self.lambda_smooth,
            "delta_edge": self.delta_edge,
            "no_stack_zcrop": self.no_stack_zcrop,
            "mask_slices_not_svr": self.mask_slices_not_svr,
            "no_robust_statistics": self.no_robust_statistics,
            "no_intensity_matching": self.no_intensity_matching,
        }

        # Slice thickness
        cfg["slice_thickness"] = [
            (
                stack_info["slice_thickness"]
                if stack_info["slice_thickness"] is not None
                else stack_info["pixdim"][2]
            )
            for stack_info in stack_infos
        ]

        # Time resolution of cine vol
        num_card_phase = stack_infos[0]["dim"][3]
        rr_interval = np.median(
            [rr for stack_info in stack_infos for rr in stack_info["rr_interval_slices"]]
        )
        cfg["time_resolution"] = rr_interval / num_card_phase

        target_indices = [
            i for i, stack_info in enumerate(stack_infos) if "chest_mask" in stack_info
        ]
        if len(target_indices) == 0:
            raise ValueError("No stack in stack_infos has a 'chest_mask' entry.")
        if len(target_indices) > 1:
            raise ValueError(
                f"Multiple stacks ({len(target_indices)}) carry a 'chest_mask' "
                "entry; expected exactly one template stack."
            )
        target_idx = target_indices[0]
        cfg["target_mask_path"] = stack_infos[target_idx]["chest_mask"]
        cfg["target_stack_idx"] = target_idx + 1

        return cfg
