"""训练与提交共用网络定义。"""
from predict.runtime import CXRClassifier


def build_model(cfg, pretrained=None):
    model = cfg["model"]
    return CXRClassifier(model["backbone"], len(cfg["class_names"]),
        model["pretrained"] if pretrained is None else pretrained,
        model.get("drop_rate", 0.2), cfg["train"]["img_size"])
