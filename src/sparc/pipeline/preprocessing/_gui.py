import os
import subprocess

from sparc.utils.nii import roll_nii


@staticmethod
def stack_review_gui(
        stack_mag_nii_paths,
        stack_pha_nii_paths,
        stack_infos,
        gui_mode,
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
        if gui_mode == "docker":
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

    return (stack_mag_nii_paths, 
            stack_pha_nii_paths, 
            stack_infos,
            excluded_stack_mag_nii_paths, 
            excluded_stack_pha_nii_paths,
            excluded_stack_infos)