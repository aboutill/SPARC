import os
import subprocess
import datetime

from sparc.utils.io import check_file


def gui(
        self, 
        ctr_cine_nii_path,
        stack_nii_paths,
        output_dir,
        mode,
        gui_mode,
        reo_cine_nii_path=None,
        ctr_heart_mask_path=None,
        aff_path=None,
    ):
    """Interactive reorientation review/refinement: launch ITK-SNAP
    ask the user to validate.
    """
    tgt_img = None
        
    if mode == "manual":
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
    
    if gui_mode == "docker":
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
    if mode == "manual":
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
 
    if gui_mode == "docker":
        p.kill()
        
    user_val = user_input == "y"
    if user_val:
        if mode != "manual":
            user_ref = check_file(itksnap_aff_path, creation_time)
            if user_ref:
                tgt_img = manual_reo_path
                self.manual_run(
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
            self.manual_run(
                input_cine_nii_path=ctr_cine_nii_path, 
                input_heart_mask_path=ctr_heart_mask_path,
                output_cine_nii_path=manual_reo_path,
                output_heart_mask_path=manual_reo_heart_mask_path,
                stack_nii_paths=stack_nii_paths,
                itksnap_aff_path=itksnap_aff_path,
            )
            
    else:
        if mode != "manual":
            check_file(itksnap_aff_path, creation_time)
            
    return tgt_img