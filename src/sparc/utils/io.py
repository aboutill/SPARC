import os
import glob
import logging


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