import sys
import logging
import warnings


def setup_logging_config(log_path=None, verbose=False):
    """Setup logging."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger("ignite").setLevel(logging.WARNING)
    logging.getLogger("monai").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=FutureWarning)

    if log_path is None:
        handlers = [logging.StreamHandler(stream=sys.stdout)]
    else:
        handlers = [logging.FileHandler(log_path), logging.StreamHandler()]

    # Setup logging
    logging.basicConfig(
        encoding="utf-8",
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=level,
        force=True,
        handlers=handlers,
    )

    # Simpler console format
    handlers[-1].setFormatter(logging.Formatter("%(message)s"))
