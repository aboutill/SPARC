import os
import glob
import logging
import datetime


def init_datalist(input_dir, img, mask=None):
    """Build a {subject: [{"image": ..., ["label": ...]}, ...]} datalist
    from '*<img>' (and optionally paired '*<mask>') files found in each
    subdirectory of input_dir."""

    input_dir = os.path.abspath(input_dir)
    datalist = {}

    subfolders = [
        folder for folder in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, folder))
    ]

    for folder in subfolders:
        subfolder = os.path.join(input_dir, folder)
        imgs_path = sorted(glob.glob(os.path.join(subfolder, "*" + img)))

        for img_path in imgs_path:
            entry = {"image": os.path.abspath(img_path)}

            if mask is not None:
                img_dir, img_filename = os.path.split(img_path)
                mask_filename = img_filename[: -len(img)] + mask
                mask_path = os.path.join(img_dir, mask_filename)

                if not os.path.exists(mask_path):
                    logging.warning(
                        f"No matching mask for {img_path} "
                        f"(expected {mask_path}); skipping."
                    )
                    continue

                entry["label"] = os.path.abspath(mask_path)

            datalist.setdefault(folder, []).append(entry)

    if not datalist:
        if mask is None:
            raise FileNotFoundError(f"No {img} files in {input_dir}.")
        raise FileNotFoundError(f"No {img} and {mask} files in {input_dir}.")

    return datalist


def check_file(file_path, copy_time):
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