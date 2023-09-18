from pathlib import Path
import numpy as np


def load_mot17_det(path: Path):
    """
    Load MOT detection data

    Args:
        path:

    Returns:

    """
    return np.genfromtxt(path.as_posix(), delimiter=',')



def write_mot17_det(path: Path, mot_data: np.ndarray):
    """
    Write MOT detection data
    Args:
        path:

    Returns:

    """
    np.savetxt("foo.csv", mot_data, delimiter=",")