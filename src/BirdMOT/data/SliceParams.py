from dataclasses import dataclass


@dataclass
class SliceParams:
    height: int
    width: int
    overlap_height_ratio: float
    overlap_width_ratio: float
    min_area_ratio: float
    ignore_negative_samples: bool = False
    verbose: bool = False