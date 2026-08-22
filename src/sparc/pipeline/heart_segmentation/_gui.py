import os
import shutil
import subprocess
import datetime

from sparc.utils.io import check_file


@staticmethod
def gui(
        cine_nii_path,
        output_dir,
        mode,
        gui_mode,
        heart_mask_path=None,
    ):
    """Interactive heart-mask review/refinement: launch ITK-SNAP
    ask the user to validate.
    """
    
    tgt_mask = None
    copy_time = None
 
    filename = os.path.basename(cine_nii_path).split(".nii.gz")[0]
    if mode == "manual":
        manual_mask_path = os.path.join(output_dir, f"{filename}_heart_mask_manual.nii.gz")
    else:
        manual_mask_path = heart_mask_path.replace("heart_mask_auto", "heart_mask_semi_auto")
        shutil.copy2(heart_mask_path, manual_mask_path)
        copy_time = datetime.datetime.now()
        
    parent, filename = os.path.split(manual_mask_path)
    mask_display_path = os.path.join(os.path.basename(parent), filename)
    
    parent, filename = os.path.split(cine_nii_path)
    cine_display_path = os.path.join(os.path.basename(parent), filename)
        
    if gui_mode == "docker":
        itksnap_args = ["itksnap", "-g", cine_nii_path]
        if mode != "manual":
            itksnap_args += ["-s", manual_mask_path]
        p = subprocess.Popen(
            args=itksnap_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    else:
        msg = f"Native ITKSNAP: Please open cine {cine_display_path}"
        if mode != "manual":
            msg += f" with segmentation {mask_display_path}"
        print(msg)
 
    if mode == "manual":
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
 
    if gui_mode == "docker":
        p.kill()
        
    user_val = user_input == "y"
    if user_val:
        if mode != "manual":
            user_ref = check_file(manual_mask_path, copy_time)
            tgt_mask = manual_mask_path if user_ref else heart_mask_path
        else:
            tgt_mask = manual_mask_path
 
    else:
        if mode != "manual":
            check_file(manual_mask_path, copy_time)
 
    return tgt_mask