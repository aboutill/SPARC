import os
import shutil
import logging
import subprocess
import datetime

from sparc.utils.nii import roll_nii


@staticmethod
def _check_file(file_path, copy_time):
    """Return True if file_path was modified after copy_time (i.e.
    the user actually edited it in ITK-SNAP); if not, delete the
    unmodified temporary copy."""
    
    # Check file modification time
    modified_time = os.path.getmtime(file_path)
    modified_time = datetime.datetime.fromtimestamp(modified_time)
    
    # No manual refinement
    user_ref = modified_time > copy_time
    if not user_ref:
        # Remove temporary file
        os.remove(file_path)
        
    return user_ref


def stack_review_gui(
        self,
        stack_mag_nii_paths,
        stack_pha_nii_paths,
        stack_infos,
    ):
    """Interactive review of preprocessed input stacks: optionally
    exclude motion-corrupted stacks, and record each remaining
    stack's diastole frame index, then circularly shift that stack's
    temporal axis so that frame becomes frame 0.
    """

    user_input = None
    choices = ["y", "n", "q"]
    while user_input not in choices:
        user_input = input(
            f"Perform manual review of input stacks? [{'/'.join(choices)}]"
        )
        user_input = user_input.lower().strip()

    if user_input == "q":
        return None, None, None, None, None, None

    if user_input == "n":
        return (stack_mag_nii_paths, stack_pha_nii_paths, stack_infos,
                None, None, None)

    n = len(stack_mag_nii_paths)
    excluded_index = []
    t_starts = []

    for i in range(n):

        parent, filename = os.path.split(stack_mag_nii_paths[i])
        display_path = os.path.join(os.path.basename(parent), filename)

        p = None
        if self.gui_mode == "docker":
            itksnap_args = ["itksnap", "-g", stack_mag_nii_paths[i]]
            p = subprocess.Popen(
                args=itksnap_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            print(f"Native ITKSNAP: Please open stack {display_path}")

        try:
            user_input = None
            choices = ["y", "n", "q"]
            z_smooth = stack_infos[i]["z_smooth"]
            while user_input not in choices:
                user_input = input(
                    f"Stack z-smoothness score: {z_smooth:.2f}\n"
                    f"Include stack: {display_path}? [{'/'.join(choices)}]"
                )
                user_input = user_input.lower().strip()

            if user_input == "q":
                return None, None, None, None, None, None

            if user_input == "n":
                excluded_index.append(i)
                continue

            t_start = None
            loop = True
            while loop:
                user_input = input(
                    "Diastole frame index (0-based)? [0,1,2,.../n/q]"
                )
                user_input = user_input.lower().strip()

                if user_input == "n":
                    t_start = 0
                    loop = False
                    continue

                if user_input == "q":
                    return None, None, None, None, None, None

                if not user_input.isdigit():
                    print("Please enter an integer, 'n', or 'q'.")
                    continue

                candidate = int(user_input)
                t_dim = stack_infos[i]["dim"][3]
                if not (0 <= candidate < t_dim):
                    print(f"Index must be between 0 and {t_dim - 1}.")
                    continue

                confirm = input(
                    f"Confirm diastole frame index {candidate}? [y/n/q]"
                ).lower().strip()
                if confirm == "y":
                    t_start = candidate
                    loop = False
                elif confirm == "q":
                    return None, None, None, None, None, None

            t_starts.append(t_start)

        finally:
            if p is not None:
                p.kill()

    excluded_stack_mag_nii_paths = [stack_mag_nii_paths[i] for i in excluded_index]
    excluded_stack_pha_nii_paths = [stack_pha_nii_paths[i] for i in excluded_index]
    excluded_stack_infos = [stack_infos[i] for i in excluded_index]
    stack_mag_nii_paths = [stack_mag_nii_paths[i] for i in range(n) if i not in excluded_index]
    stack_pha_nii_paths = [stack_pha_nii_paths[i] for i in range(n) if i not in excluded_index]
    stack_infos = [stack_infos[i] for i in range(n) if i not in excluded_index]

    for stack_info, t_start in zip(stack_infos, t_starts):
        stack_info["raw_diastole_idx"] = t_start

    for mag_nii_path, pha_nii_path, t_start in zip(
            stack_mag_nii_paths, stack_pha_nii_paths, t_starts):
        roll_nii(mag_nii_path, t_start)
        if pha_nii_path is not None:
            roll_nii(pha_nii_path, t_start)

    return (stack_mag_nii_paths, stack_pha_nii_paths, stack_infos,
            excluded_stack_mag_nii_paths, excluded_stack_pha_nii_paths, excluded_stack_infos)


def chest_segmentation_gui(
        self, 
        stack_nii_paths,
        stack_infos,
        output_dir,
        chest_mask_path=None,
        profile=False,
        debug=False,
    ):
    """Interactive chest-mask review/refinement loop: launch ITK-SNAP
    on the current candidate stack, ask the user to validate; on
    rejection, try automatic segmentation on the next candidate stack
    (if any remain).
    """
    
    idx = 0
    img_path = stack_nii_paths[idx]
    mask_path = chest_mask_path
    n = sum(info["z_smooth"] < float("inf") for info in stack_infos)
 
    user_val = False
    tgt_mask = None
    tgt_idx = None
    copy_time = None
 
    while not user_val:
 
        filename = os.path.basename(img_path).split(".nii.gz")[0]
        if self.mode == "manual":
            manual_mask_path = os.path.join(output_dir, f"{filename}_chest_mask_manual.nii.gz")
        else:
            manual_mask_path = mask_path.replace("chest_mask_auto", "chest_mask_semi_auto")
            shutil.copy2(mask_path, manual_mask_path)
            copy_time = datetime.datetime.now()
            
        parent, filename = os.path.split(manual_mask_path)
        mask_display_path = os.path.join(os.path.basename(parent), filename)
        
        parent, filename = os.path.split(img_path)
        img_display_path = os.path.join(os.path.basename(parent), filename)
            
        if self.gui_mode == "docker":
            itksnap_args = ["itksnap", "-g", img_path]
            if self.mode != "manual":
                itksnap_args += ["-s", manual_mask_path]
            p = subprocess.Popen(
                args=itksnap_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            msg = f"Native ITKSNAP: Please open stack {img_display_path}"
            if self.mode != "manual":
                msg += f" with segmentation {mask_display_path}"
            print(msg)
     
        z_smooth = stack_infos[idx]["z_smooth_raw"]
        if self.mode == "manual":
            loop = True
            choices = ["y", "n", "q"]
            while loop:
                user_input = input(
                    f"Stack z-smoothness score: {z_smooth:.2f}\n"
                    f"Please save segmentation as: {mask_display_path}\n"
                    f"Validate segmentation? [{'/'.join(choices)}]"
                )
                user_input = user_input.lower().strip()
                mask_exist = os.path.exists(manual_mask_path)
 
                if user_input == "y" and mask_exist:
                    loop = False
                elif user_input in ("q", "n"):
                    loop = False
        else:
            user_input = None
            choices = ["y", "n", "q"]
            while user_input not in choices:
                user_input = input(
                    f"Stack z-smoothness score: {z_smooth:.2f}\n"
                    f"Please save segmentation: {mask_display_path}\n"
                    f"Validate segmentation? [{'/'.join(choices)}]"
                )
                user_input = user_input.lower().strip()
 
        if self.gui_mode == "docker":
            p.kill()
 
        if user_input == "q":
            if self.mode != "manual":
                self._check_file(manual_mask_path, copy_time)
            return None, None
 
        user_val = user_input == "y"
 
        if user_val:
            if self.mode != "manual":
                user_ref = self._check_file(manual_mask_path, copy_time)
                tgt_mask = manual_mask_path if user_ref else mask_path
            else:
                tgt_mask = manual_mask_path
            tgt_idx = idx
 
        else:
            if self.mode != "manual":
                self._check_file(manual_mask_path, copy_time)
 
            idx += 1
            if idx == n:
                logging.info(
                    f"No ornt={self.chest_segmentator.target_stack_orientation} "
                    f"candidate stack left"
                )
            if idx == len(stack_nii_paths):
                logging.info("No candidate stack left")
                return None, None
 
            img_path = stack_nii_paths[idx]
 
            if self.mode != "manual":
                mask_path, qc_path = self.chest_segmentator.run(
                    stack_nii_path=img_path,
                    output_dir=output_dir,
                    profile=profile,
                    debug=debug,
                )
 
                mask_valid = self.chest_segmentator.validate_mask(
                    qc_report_path=qc_path,
                )
                if self.mode == "monitored_auto" and mask_valid:
                    tgt_mask = mask_path
                    tgt_idx = idx
                    return tgt_mask, tgt_idx
 
    return tgt_mask, tgt_idx


def heart_segmentation_gui(
        self, 
        cine_nii_path,
        output_dir,
        heart_mask_path=None,
    ):
    """Interactive heart-mask review/refinement: launch ITK-SNAP
    ask the user to validate.
    """
    
    tgt_mask = None
    copy_time = None
 
    filename = os.path.basename(cine_nii_path).split(".nii.gz")[0]
    if self.mode == "manual":
        manual_mask_path = os.path.join(output_dir, f"{filename}_heart_mask_manual.nii.gz")
    else:
        manual_mask_path = heart_mask_path.replace("heart_mask_auto", "heart_mask_semi_auto")
        shutil.copy2(heart_mask_path, manual_mask_path)
        copy_time = datetime.datetime.now()
        
    parent, filename = os.path.split(manual_mask_path)
    mask_display_path = os.path.join(os.path.basename(parent), filename)
    
    parent, filename = os.path.split(cine_nii_path)
    cine_display_path = os.path.join(os.path.basename(parent), filename)
        
    if self.gui_mode == "docker":
        itksnap_args = ["itksnap", "-g", cine_nii_path]
        if self.mode != "manual":
            itksnap_args += ["-s", manual_mask_path]
        p = subprocess.Popen(
            args=itksnap_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    else:
        msg = f"Native ITKSNAP: Please open cine {cine_display_path}"
        if self.mode != "manual":
            msg += f" with segmentation {mask_display_path}"
        print(msg)
 
    if self.mode == "manual":
        loop = True
        choices = ["y", "q"]
        while loop:
            user_input = input(
                f"Please save segmentation as: {mask_display_path}\n"
                f"Validate segmentation? [{'/'.join(choices)}]"
            )
            user_input = user_input.lower().strip()
            mask_exist = os.path.exists(manual_mask_path)
 
            if user_input == "y" and mask_exist:
                loop = False
            elif user_input == "q":
                loop = False
    else:
        user_input = None
        choices = ["y", "q"]
        while user_input not in choices:
            user_input = input(
                f"Please save segmentation: {mask_display_path}\n"
                f"Validate segmentation? [{'/'.join(choices)}]"
            )
            user_input = user_input.lower().strip()
 
    if self.gui_mode == "docker":
        p.kill()
        
    user_val = user_input == "y"
    if user_val:
        if self.mode != "manual":
            user_ref = self._check_file(manual_mask_path, copy_time)
            tgt_mask = manual_mask_path if user_ref else heart_mask_path
        else:
            tgt_mask = manual_mask_path
 
    else:
        if self.mode != "manual":
            self._check_file(manual_mask_path, copy_time)
 
    return tgt_mask


def reorientation_gui(
        self, 
        ctr_cine_nii_path,
        stack_nii_paths,
        output_dir,
        reo_cine_nii_path=None,
        ctr_heart_mask_path=None,
        aff_path=None,
    ):
    """Interactive reorientation review/refinement: launch ITK-SNAP
    ask the user to validate.
    """
    tgt_img = None
        
    if self.mode == "manual":
        reo_cine_nii_path = ctr_cine_nii_path
        filename = os.path.basename(reo_cine_nii_path).split("_ctr.nii.gz")[0]
        itksnap_aff_path = os.path.join(output_dir, f"{filename}_reo_aff_manual.txt")
        manual_reo_path = os.path.join(output_dir, f"{filename}_reo_manual.nii.gz")
    else:
        filename = os.path.basename(reo_cine_nii_path).split("_reo_auto.nii.gz")[0]
        itksnap_aff_path = os.path.join(output_dir, f"{filename}_reo_aff_semi_auto.txt")
        manual_reo_path =  os.path.join(output_dir, f"{filename}_reo_semi_auto.nii.gz")
    
    manual_reo_heart_mask_path = ctr_heart_mask_path.replace("_ctr.nii.gz", "_reo.nii.gz") if ctr_heart_mask_path is not None else None
    with open(itksnap_aff_path, "w") as file:
        file.write("")
    creation_time = datetime.datetime.now()
    
    parent, filename = os.path.split(itksnap_aff_path)
    aff_display_path  = os.path.join(os.path.basename(parent), filename)
    
    parent, filename = os.path.split(reo_cine_nii_path)
    cine_display_path  = os.path.join(os.path.basename(parent), filename)
    
    if self.gui_mode == "docker":
        p = subprocess.Popen(
            args=["itksnap", 
                  "-g", reo_cine_nii_path, 
                  "-o", reo_cine_nii_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    else:
        print(f"Native ITKSNAP: Please open cine {cine_display_path}")
        
    user_input = None
    if self.mode == "manual":
        loop = True
        choices = ["y", "q"]
        while loop:
            user_input = input(
                f"Please save transformation affine as: {aff_display_path}\n"
                "Pure rotation correction only. Translation not supported!\n"
                f"Validate reorientation? [{'/'.join(choices)}]"
            )
            user_input = user_input.lower().strip()
            aff_written = os.path.exists(itksnap_aff_path) and os.path.getsize(itksnap_aff_path) > 0
            if user_input == "y" and aff_written:
                loop = False
            elif user_input == "q":
                os.remove(itksnap_aff_path)
                loop = False
    else:
        user_input = None
        choices = ["y", "q"]
        while user_input not in choices:
            user_input = input(
                f"Please save transformation affine as: {aff_display_path}\n"
                "Pure rotation correction only. Translation not supported!\n"
                f"Validate reorientation? [{'/'.join(choices)}]"
            )
            user_input = user_input.lower().strip()
 
    if self.gui_mode == "docker":
        p.kill()
        
    user_val = user_input == "y"
    if user_val:
        if self.mode != "manual":
            user_ref = self._check_file(itksnap_aff_path, creation_time)
            if user_ref:
                tgt_img = manual_reo_path
                self.reorientor.manual_run(
                    input_cine_nii_path=ctr_cine_nii_path, 
                    input_heart_mask_path=ctr_heart_mask_path,
                    output_cine_nii_path=manual_reo_path,
                    output_heart_mask_path=manual_reo_heart_mask_path,
                    stack_nii_paths=stack_nii_paths,
                    itksnap_aff_path=itksnap_aff_path,
                    monai_aff_path=aff_path,
                )
            else:
                tgt_img = reo_cine_nii_path
        else:
            tgt_img = manual_reo_path
            self.reorientor.manual_run(
                input_cine_nii_path=ctr_cine_nii_path, 
                input_heart_mask_path=ctr_heart_mask_path,
                output_cine_nii_path=manual_reo_path,
                output_heart_mask_path=manual_reo_heart_mask_path,
                stack_nii_paths=stack_nii_paths,
                itksnap_aff_path=itksnap_aff_path,
            )
            
    else:
        if self.mode != "manual":
            self._check_file(itksnap_aff_path, creation_time)
            
    return tgt_img