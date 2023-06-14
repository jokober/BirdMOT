from dataclasses import asdict
from pathlib import Path

from sahi.predict import predict
from sahi.prediction import PredictionResult

from BirdMOT.detection.SahiPredictionParams import SahiPredictionParams


def sliced_batch_predict(image_dir: Path, sahi_prediction_params: SahiPredictionParams) -> PredictionResult:
    sahi_prediction_params.source = image_dir.as_posix()
    predictions = predict(**asdict(sahi_prediction_params))

    return predictions
