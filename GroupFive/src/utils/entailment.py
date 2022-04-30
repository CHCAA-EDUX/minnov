from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/decomposable-attention-elmo-2020.04.09.tar.gz"
)


def entailment_label(premise: str, hypothesis: str):
    return predictor.predict(premise, hypothesis)["label"]  # type: ignore
