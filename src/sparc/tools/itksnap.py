import subprocess


def itksnap_subprocess(img_path, mask_path=None, add_img_path=None):
    
    args = ["itksnap", "-g", img_path]
    if mask_path is not None:
        args += ["-s", mask_path]
    if add_img_path is not None:
        args += ["-o", add_img_path]
        
    p = subprocess.Popen(
        args=args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    return p