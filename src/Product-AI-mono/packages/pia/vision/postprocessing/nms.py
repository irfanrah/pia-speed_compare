import time

import numpy as np
import torch
import torchvision
from torchvision.ops import box_iou


def xywh2xyxy(x):
    """
    바운딩 박스 좌표 형식을 (x, y, 너비, 높이)에서 (x1, y1, x2, y2) 형식으로 변환하는 함수.

    (x1, y1): 좌측 상단 모서리 좌표
    (x2, y2): 우측 하단 모서리 좌표

    Args:
        x (np.ndarray | torch.Tensor): 입력 바운딩 박스 좌표 (x, y, 너비, 높이) 형식.

    Returns:
        y (np.ndarray | torch.Tensor): 변환된 바운딩 박스 좌표 (x1, y1, x2, y2) 형식.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
):
    """
    수행된 예측 결과에 대해 Non-Maximum Suppression (NMS)을 적용하여 불필요한 중복 바운딩 박스를 제거한다.

    Args:
        prediction (torch.Tensor):
            모델의 예측 결과로 (batch_size, num_boxes, num_classes + 4 + num_masks) 형태의 텐서.
        conf_thres (float):
            최소 confidence threshold (0.0 ~ 1.0).
        iou_thres (float):
            IoU threshold로, 해당 값보다 낮은 IoU를 가진 박스는 제거됨 (0.0 ~ 1.0).
        classes (List[int], optional):
            고려할 클래스 인덱스. 기본값은 None (모든 클래스 고려).
        agnostic (bool, optional):
            클래스에 관계없이 NMS를 수행할지 여부. 기본값 False.
        multi_label (bool, optional):
            각 박스에 여러 개의 클래스가 할당될 수 있는지 여부. 기본값 False.
        labels (List[List[Union[int, float, torch.Tensor]]], optional):
            주어진 apriori 라벨. 기본값은 빈 리스트.
        max_det (int, optional):
            유지할 최대 박스 개수. 기본값 300.
        nc (int, optional):
            클래스 개수. 기본값 0.
        max_time_img (float, optional):
            한 이미지당 최대 처리 시간 (초 단위). 기본값 0.05.
        max_nms (int, optional):
            torchvision.ops.nms()에 들어갈 최대 박스 개수. 기본값 30000.
        max_wh (int, optional):
            최대 박스 크기 (픽셀 단위). 기본값 7680.

    Returns:
        List[torch.Tensor]: 각 배치별로 NMS가 적용된 결과를 포함한 리스트.
        각 텐서는 (num_boxes, 6 + num_masks) 형태이며,
        (x1, y1, x2, y2, confidence, class, mask1, mask2, ...)을 포함한다.
    """

    # Checks
    # assert (
    #     0 <= conf_thres <= 1
    # ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    # assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    if isinstance(
        prediction, (list, tuple)
    ):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    if isinstance(conf_thres, int) or isinstance(conf_thres, float):
        torch_conf_thres = (
            torch.Tensor([conf_thres for i in range(len(prediction))]).view(-1, 1).to(device)
        )
    elif isinstance(conf_thres, list):
        torch_conf_thres = torch.Tensor(conf_thres).view(-1, 1).to(device)
    elif isinstance(conf_thres, torch.Tensor):
        torch_conf_thres = conf_thres.view(-1, 1).to(device)

    if isinstance(iou_thres, int) or isinstance(iou_thres, float):
        torch_iou_thres = (
            torch.Tensor([iou_thres for i in range(len(prediction))]).view(-1, 1).to(device)
        )
    elif isinstance(iou_thres, list):
        torch_iou_thres = torch.Tensor(iou_thres).view(-1, 1).to(device)
    elif isinstance(iou_thres, torch.Tensor):
        torch_iou_thres = iou_thres.view(-1, 1).to(device)


    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > torch_conf_thres  # candidates

    # Settings
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > torch_conf_thres[xi])
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > torch_conf_thres[xi]]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[
                x[:, 4].argsort(descending=True)[:max_nms]
            ]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, torch_iou_thres[xi])  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > torch_iou_thres[xi]  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


@torch.inference_mode()
def torch_fast_nms_half_single(
    boxes_xyxy: torch.Tensor,  # (K,4), half, cuda
    scores: torch.Tensor,      # (K,), half, cuda
    iou_thres: float = 0.45,
    pre_topk: int = 1024,
    max_det: int | None = 300,
) -> torch.Tensor:
    """
    YOLACT-style fast NMS:
      1) 점수 내림차순 정렬 후 상위 K만 유지
      2) (K,K) IoU 행렬에서 각 box의 '자기보다 점수 높은 박스들'과의 IoU 최대값 계산
      3) 그 최대 IoU <= iou_thres 인 박스만 keep
    return: keep 인덱스 (원본 boxes_xyxy 기준의 인덱스)
    """
    K_total = boxes_xyxy.shape[0]
    if K_total == 0:
        return boxes_xyxy.new_zeros((0,), dtype=torch.long)

    # 1) 상위 K만 유지
    if pre_topk > 0 and pre_topk < K_total:
        topk_idx = torch.topk(scores, pre_topk, sorted=True).indices  # (pre_topk,)
    else:
        # 이미 내림차순 정렬하려면 점수 정렬
        topk_idx = torch.argsort(scores, descending=True)
    boxes = boxes_xyxy[topk_idx].contiguous()   # (K,4)

    # 2) pairwise IoU (half 연산)
    x1, y1, x2, y2 = boxes[:, 0:1], boxes[:, 1:2], boxes[:, 2:3], boxes[:, 3:4]
    area = (x2 - x1).clamp_(0) * (y2 - y1).clamp_(0)  # (K,1)
    # 교집합
    inter_x1 = torch.maximum(x1, x1.T)
    inter_y1 = torch.maximum(y1, y1.T)
    inter_x2 = torch.minimum(x2, x2.T)
    inter_y2 = torch.minimum(y2, y2.T)
    inter_w  = (inter_x2 - inter_x1).clamp_(0)
    inter_h  = (inter_y2 - inter_y1).clamp_(0)
    inter    = inter_w * inter_h
    # 합집합
    union = area + area.T - inter
    iou_mat = inter / (union + 1e-7)            # (K,K), half

    # 3) 위쪽 삼각(자기보다 점수 '높은' 인덱스들)에서 col-wise max IoU
    iou_upper = torch.triu(iou_mat, diagonal=1)        # (K,K)
    iou_max_per_col, _ = iou_upper.max(dim=0)          # (K,)
    keep_mask = iou_max_per_col <= torch.tensor(iou_thres, device=boxes.device, dtype=boxes.dtype)

    keep_idx_sorted = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)  # sorted by score (because topk_idx sorted)
    if keep_idx_sorted.numel() == 0:
        return boxes_xyxy.new_zeros((0,), dtype=torch.long)

    if max_det is not None and max_det > 0 and keep_idx_sorted.numel() > max_det:
        keep_idx_sorted = keep_idx_sorted[:max_det]

    # 정렬 전 원본 인덱스로 환원
    return topk_idx[keep_idx_sorted]


@torch.inference_mode()
def torch_non_max_suppression(
    prediction: torch.Tensor,      # (B, 4+nc, N), xywh + class scores, CUDA tensor
    conf_thres: float = 0.25,
    iou_thres: float  = 0.45,
    classes: list[int] | torch.Tensor | None = None,  # 필터할 클래스 (옵션)
    agnostic: bool = False,
    pre_topk: int = 1024,          # N이 큰 경우 강력 추천 (e.g., 1024~4096)
    max_det_per_img: int | None = 300,
    max_wh: float = 7680.0,        # class offset trick용
) -> list[torch.Tensor]:
    """
    반환: list[ Tensor(M_i, 6) ]  (각 row = [x1,y1,x2,y2,conf,cls], dtype=float16)
    - 내부 모든 연산을 FP16으로 수행 (입력도 강제로 half)
    - 클래스 분리 없이 수행하되, agnostic=False면 class offset trick으로 서로 억제되지 않게 처리
    """
    assert prediction.is_cuda, "prediction must be CUDA tensor"
    device = prediction.device
    B, C, N = prediction.shape
    nc = C - 4
    assert nc > 0, "prediction shape must be (B, 4+nc, N)"

    # FP16 강제
    if prediction.dtype != torch.float16:
        prediction = prediction.half()

    # (B,N,C)
    pred = prediction.transpose(-1, -2).contiguous()  # (B,N,4+nc)
    # xywh -> xyxy (half)
    xy = pred[..., 0:2]
    wh = pred[..., 2:4]
    pred[..., 0:2] = xy - wh * 0.5
    pred[..., 2:4] = xy + wh # xy pred[..., 0:2]를 참조하여 overwrite되어서 그냥 2배 해버리기

    boxes_all = pred[..., :4]         # (B,N,4) half
    cls_all   = pred[..., 4:4 + nc]     # (B,N,nc) half

    # 클래스 최고 점수 + argmax (half)
    cls_max, cls_argmax = cls_all.max(dim=2)   # (B,N),(B,N)

    # confidence 프리필터
    keep_mask = cls_max > torch.tensor(conf_thres, device=device, dtype=cls_max.dtype)

    # 클래스 필터
    classes_tensor = None
    if classes is not None:
        classes_tensor = torch.as_tensor(classes, device=device, dtype=torch.int64)

    outs = []
    for b in range(B):
        km = keep_mask[b]
        if not km.any():
            outs.append(torch.empty((0,6), dtype=torch.float16, device=device))
            continue

        boxes = boxes_all[b][km].contiguous()      # (M,4) half
        scores = cls_max[b][km].contiguous()       # (M,)  half
        c_idx  = cls_argmax[b][km].to(torch.float16)  # (M,) half (저장용)

        if classes_tensor is not None:
            keep_cls = torch.isin(c_idx.to(torch.int64), classes_tensor)
            if not keep_cls.any():
                outs.append(torch.empty((0,6), dtype=torch.float16, device=device))
                continue
            boxes  = boxes[keep_cls]
            scores = scores[keep_cls]
            c_idx  = c_idx[keep_cls]

        if boxes.numel() == 0:
            outs.append(torch.empty((0,6), dtype=torch.float16, device=device))
            continue

        # class-agnostic이 아니면 클래스별 억제 방지: 좌표에 offset
        if not agnostic:
            # NOTE: half 정밀도라서 너무 큰 오프셋은 피하세요. 기본 max_wh=7680이면 안전.
            offset = (c_idx * max_wh).unsqueeze(1)  # (M,1)
            boxes = boxes.clone()
            boxes[:, 0:2] += offset
            boxes[:, 2:4] += offset

        # Fast-NMS (한 번에) - FP16
        keep_idx = torch_fast_nms_half_single(
            boxes_xyxy=boxes,
            scores=scores,
            iou_thres=iou_thres,
            pre_topk=pre_topk,
            max_det=max_det_per_img,
        )

        if keep_idx.numel() == 0:
            outs.append(torch.empty((0,6), dtype=torch.float16, device=device))
            continue

        kept_boxes  = boxes[keep_idx]
        kept_scores = scores[keep_idx]
        kept_cls    = c_idx[keep_idx]

        det_b = torch.stack(
            (kept_boxes[:,0], kept_boxes[:,1], kept_boxes[:,2], kept_boxes[:,3], kept_scores, kept_cls),
            dim=1
        ).contiguous()  # (Mb,6) half
        outs.append(det_b)

    return outs
