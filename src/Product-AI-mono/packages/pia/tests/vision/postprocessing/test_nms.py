import pytest
import torch
from pia.vision.postprocessing.nms import torch_non_max_suppression


def generate_dummy_predictions(batch_size: int, num_classes: int, num_candidates: int, device: torch.device):
    """
    더미 prediction 생성: (B, 4+nc, N)
    """
    xywh = torch.rand((batch_size, 4, num_candidates), device=device) * 640
    cls_scores = torch.rand((batch_size, num_classes, num_candidates), device=device)
    pred = torch.cat([xywh, cls_scores], dim=1)  # (B, 4+nc, N)
    return pred


@pytest.mark.parametrize("batch_size,num_classes,num_candidates", [(2, 5, 200), (1, 3, 50)])
def test_torch_non_max_suppression_shapes(batch_size, num_classes, num_candidates):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = generate_dummy_predictions(batch_size, num_classes, num_candidates, device=device)

    outs = torch_non_max_suppression(
        prediction=preds,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        pre_topk=100,
        max_det_per_img=50,
        max_wh=4096,
    )

    # 배치 사이즈만큼 리스트 반환해야 함
    assert isinstance(outs, list)
    assert len(outs) == batch_size

    for det in outs:
        assert isinstance(det, torch.Tensor)
        assert det.ndim == 2
        # 열 개수는 6 (x1,y1,x2,y2,conf,cls)
        if det.numel() > 0:
            assert det.shape[1] == 6
            # confidence는 0~1 범위
            conf = det[:, 4]
            assert torch.all((conf >= 0) & (conf <= 1))


def test_torch_non_max_suppression_empty_input():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # (B, 4+nc, N) → N=0 인 경우
    preds = torch.empty((1, 4 + 3, 0), device=device)

    outs = torch_non_max_suppression(prediction=preds)
    assert isinstance(outs, list)
    assert len(outs) == 1
    assert outs[0].numel() == 0
