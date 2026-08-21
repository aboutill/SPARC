import os
import glob
import shlex
import shutil
import logging
import subprocess


def svr_reconstruct(
        cine_nii_path,
        stack_nii_paths,
        slice_thickness,
        target_stack_idx,
        target_mask_path,
        iterations,
        sr_iterations,
        resolution,
        time_resolution,
        lambda_smooth,
        delta_edge,
        no_stack_zcrop=False,
        mask_slices_not_svr=False,
        no_robust_statistics=False,
        no_intensity_matching=False,
        profile=False,
        verbose=False,
        debug=False,
        log=False,
    ):
    """Run gated slice-to-volume reconstruction via the SVR-lite
    binary, producing a 3D+time cine volume.
    """

    cine_nii_path = os.path.abspath(cine_nii_path)
    stack_nii_paths = [os.path.abspath(p) for p in stack_nii_paths]
    target_mask_path = os.path.abspath(target_mask_path)

    output_dir = os.path.dirname(cine_nii_path)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "svr", "reconstruct",
        cine_nii_path,
    ]
    cmd += stack_nii_paths
    cmd += ["-thickness"] + [str(s) for s in slice_thickness]
    cmd += [
        "-target_stack", str(target_stack_idx),
        "-mask", target_mask_path,
        "-iterations", str(iterations),
        "-sr_iterations", str(sr_iterations),
        "-resolution", str(resolution),
        "-time_resolution", str(time_resolution),
        "-lambda", str(lambda_smooth),
        "-delta", str(delta_edge),
    ]
    if no_stack_zcrop:
        cmd += ["-no_stack_zcrop"]
    if mask_slices_not_svr:
        cmd += ["-mask_slices_not_svr"]
    if no_robust_statistics:
        cmd += ["-no_robust_statistics"]
    if no_intensity_matching:
        cmd += ["-no_intensity_matching"]
    if debug:
        cmd += ["-debug", "-info", "svr_info.tsv"]
    if verbose:
        cmd += ["-verbose"]
    if profile:
        cmd += ["-profile"]

    cwd = os.getcwd()
    os.chdir(output_dir)
    try:
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as e:
            raise RuntimeError(f"SVR lite svr reconstruct failed: executable not found ({e})") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"SVR lite svr reconstruct failed: {e}") from e

        if debug:
            intermediate_output_dir = os.path.join(output_dir, "intermediate-outputs")
            os.makedirs(intermediate_output_dir, exist_ok=True)

            nifti_files = glob.glob("*.nii.gz")
            dof_files = glob.glob("*.dof")
            for file in nifti_files + dof_files:
                if os.path.basename(cine_nii_path) != file:
                    shutil.move(file, intermediate_output_dir)

            cmd_log_path = "svr-lite-cmd.txt"
            with open(cmd_log_path, "w") as f:
                f.write(shlex.join(cmd))
        
        log_file = "main.log"
        if os.path.exists(log_file):
            if log:
                os.rename(log_file, "svr.log")
            else:
                os.remove(log_file)
        else:
            logging.warning(f"Expected log file not found: {log_file}")

        report_file = "recon-metadata.json"
        new_report_file = "svr_qc.json"
        if os.path.exists(report_file):
            os.rename(report_file, new_report_file)
        report_path = os.path.join(output_dir, new_report_file)

        if not os.path.exists(report_path):
            raise FileNotFoundError(
                f"SVR lite did not produce the expected metadata "
                f"report: {report_path}"
            )

        return report_path
    finally:
        os.chdir(cwd)