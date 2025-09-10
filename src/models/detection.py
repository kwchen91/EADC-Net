from torchvision.models.detection import fasterrcnn_resnet50_fpn
try:
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
except Exception:
    from torchvision.models.detection import FastRCNNPredictor


def build_detector(num_classes: int):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model