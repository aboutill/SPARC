import os
import shutil
import logging
import datetime

from sparc.utils.io import check_file
from sparc.tools.itksnap import itksnap_subprocess


def gui(
    self,
    stack_nii_paths,
    stack_infos,
    output_dir,
    mode,
    gui_mode,
    chest_mask_path=None,
    profile=False,
    debug=False,
):
    """Interactive chest-mask review/refinement loop.

    Candidates are tried in one combined order: target-orientation
    stacks first, then (if those are exhausted) every remaining
    stack, best-to-worst by z_smooth_raw. Stacks flagged unet_too_small
    are skipped in either group, including as the very first candidate shown.
    """

    candidate_order = list(range(len(stack_infos)))

    def _next_candidate(order_pos):
        """First position >= order_pos whose stack isn't
        unet_too_small, or None if none remain."""
        while order_pos < len(candidate_order):
            if not stack_infos[candidate_order[order_pos]]["unet_too_small"]:
                return order_pos
            order_pos += 1
        return None

    order_pos = _next_candidate(0)
    if order_pos is None:
        logging.info(
            f"No candidate stack left "
            f"(target_orientation={self.chest_segmentator.target_stack_orientation})"
        )
        return None, None
    idx = candidate_order[order_pos]
    img_path = stack_nii_paths[idx]
    mask_path = chest_mask_path

    user_val = False
    tgt_mask = None
    tgt_idx = None
    copy_time = None

    while not user_val:

        filename = os.path.basename(img_path).split(".nii.gz")[0]
        if mode == "manual":
            manual_mask_path = os.path.join(output_dir, f"{filename}_chest_mask_manual.nii.gz")
        else:
            manual_mask_path = mask_path.replace("chest_mask_auto", "chest_mask_semi_auto")
            shutil.copy2(mask_path, manual_mask_path)
            copy_time = datetime.datetime.now()

        parent, filename = os.path.split(manual_mask_path)
        mask_display_path = os.path.join(os.path.basename(parent), filename)

        parent, filename = os.path.split(img_path)
        img_display_path = os.path.join(os.path.basename(parent), filename)

        if gui_mode == "docker":
            itksnap_mask_path = manual_mask_path if mode != "manual" else None
            p = itksnap_subprocess(
                img_path=img_path,
                mask_path=itksnap_mask_path,
            )
        else:
            msg = f"Native ITKSNAP: Please open stack {img_display_path}"
            if mode != "manual":
                msg += f" with segmentation {mask_display_path}"
            print(msg)

        z_smooth = stack_infos[idx]["z_smooth_raw"]
        choices = ["y", "n", "q"]
        msg = (
            f"Stack z-smoothness score: {z_smooth:.2f}\n"
            f"Please save segmentation as: {mask_display_path}\n"
            f"Validate segmentation? [{'/'.join(choices)}]"
        )
        if mode == "manual":
            loop = True
            while loop:
                user_input = input(msg)
                user_input = user_input.lower().strip()
                mask_exist = os.path.exists(manual_mask_path)

                if user_input == "y" and mask_exist:
                    loop = False
                elif user_input in ("q", "n"):
                    loop = False
        else:
            user_input = None
            while user_input not in choices:
                user_input = input(msg)
                user_input = user_input.lower().strip()

        if gui_mode == "docker":
            p.kill()

        if user_input == "q":
            if mode != "manual":
                check_file(manual_mask_path, copy_time)
            return None, None

        user_val = user_input == "y"

        if user_val:
            if mode != "manual":
                user_ref = check_file(manual_mask_path, copy_time)
                tgt_mask = manual_mask_path if user_ref else mask_path
            else:
                tgt_mask = manual_mask_path
            tgt_idx = idx

        else:
            if mode != "manual":
                check_file(manual_mask_path, copy_time)

            order_pos = _next_candidate(order_pos + 1)
            if order_pos is None:
                logging.info(
                    f"No candidate stack left "
                    f"(target_orientation={self.chest_segmentator.target_stack_orientation})"
                )
                return None, None
            idx = candidate_order[order_pos]

            img_path = stack_nii_paths[idx]

            if mode != "manual":
                mask_path, qc_path = self.run(
                    stack_nii_path=img_path,
                    output_dir=output_dir,
                    profile=profile,
                    debug=debug,
                )

                mask_valid = self.validate_mask(
                    qc_report_path=qc_path,
                )
                if mode == "monitored_auto" and mask_valid:
                    tgt_mask = mask_path
                    tgt_idx = idx
                    return tgt_mask, tgt_idx

    return tgt_mask, tgt_idx
