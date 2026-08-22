import torch

from sklearn.model_selection import KFold


class MultiDomainTrainer:
    """Trains segmentation networks via k-fold CV, optionally across
    multiple domains and/or from pretrained (transfer learning) weights."""

    from ._io import (
        setup_logging,
        print_model_info,
        save_model_info,
        get_models_paths,
    )
    from ._run import run

    def __init__(
        self,
        data_cfg,
        transforms_cfg,
        unet_cfg,
        train_cfg,
        inferer_cfg,
        post_processing_cfg,
    ):
        """Store training configuration and select CUDA/CPU device."""

        self.data_cfg = data_cfg
        self.transforms_cfg = transforms_cfg
        self.unet_cfg = unet_cfg
        self.train_cfg = train_cfg
        self.inferer_cfg = inferer_cfg
        self.post_processing_cfg = post_processing_cfg

        # Setup CUDA device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []

    @staticmethod
    def init_CV_folds(datalists, folds):
        """Initialise multidomain CV folds."""

        train_sets = [[] for i in range(folds)]
        val_sets = [[] for i in range(folds)]
        for domain, datalist in enumerate(datalists):
            subjects = sorted(list(datalist.keys()))
            kf = KFold(n_splits=folds, shuffle=True, random_state=42)
            for fold, (train_index, val_index) in enumerate(kf.split(subjects)):
                train_sets[fold] += [
                    datalist[subjects[i]][j]
                    for i in train_index
                    for j in range(len(datalist[subjects[i]]))
                ]
                val_sets[fold] += [
                    datalist[subjects[i]][j]
                    for i in val_index
                    for j in range(len(datalist[subjects[i]]))
                ]

        return train_sets, val_sets
