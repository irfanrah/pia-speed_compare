import torch
from pia.ai.device import load_model_backend
from pia.ai.model import PiaTorchModel
from pia.ai.tasks.OD.base import ODConfig, TrackerConfig
from pia.vision.roi import BoundingBoxManager


def test_yolov8(img, model_path):
    device = load_model_backend("cuda", type="str")
    config = ODConfig(
        model_path=model_path,
        device=device,
    )
    model = PiaTorchModel(target_task=0, target_model=0, config=config)  # OD  # yolov8

    ret = model(img)
    ret = model.forward(img)
    ########## Tracking test ##########
    trk_config = TrackerConfig(tracker=0, max_age=3, min_hits=1, threshold=200)

    config = ODConfig(
        model_path=model_path,
        device=device,
        tracker_config=trk_config,
    )
    model = PiaTorchModel(target_task=0, target_model=0, config=config)  # OD  # yolov8

    ret = model(img)
    ret = model.forward(img)
    ########## bounding box merge test ##########
    config = ODConfig(
        model_path=model_path,
        device=device,
        do_merge_bboxes=True,
        distance_thres_for_merge=100,
        iou_thres_for_merge=0.5,
    )
    model = PiaTorchModel(target_task=0, target_model=0, config=config)  # OD  # yolov8

    ret = model(img)
    ret = model.forward(img)
    assert isinstance(ret, list)

    ########## half model load ############

    config = ODConfig(
        model_path=model_path,
        device=device,
        use_half_precision=True,
    )
    model = PiaTorchModel(target_task=0, target_model=0, config=config)  # OD  # yolov8


def test_yolov8_batch_inference(img, model_path):
    device = load_model_backend("cuda", type="str")

    trk_config = TrackerConfig(tracker=0, max_age=3, min_hits=1, threshold=200)

    config = ODConfig(
        model_path=model_path,
        tracker_config=trk_config,
        device=device,
        do_merge_bboxes=True,
        distance_thres_for_merge=100,
        iou_thres_for_merge=0.5,
    )
    model = PiaTorchModel(target_task=0, target_model=0, config=config)  # OD  # yolov8

    ########## batch inference test ############
    img = model.preprocess(img)
    imgs = torch.vstack((img, img))
    ret = model(imgs)
    assert len(ret) == 2


def test_draw_bbox(img):
    boxes = [[10, 10, 80, 80], [90, 90, 200, 200]]
    clss = [0, 1]
    names = ["person", "car"]
    annotated_img = BoundingBoxManager.draw_bbox(img, boxes, clss, names)

    assert annotated_img is not None
