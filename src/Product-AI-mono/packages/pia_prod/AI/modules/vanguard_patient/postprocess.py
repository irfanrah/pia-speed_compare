import torch
import torchvision


@torch.inference_mode()
def torch_non_max_suppression(
    prediction: torch.Tensor,  # (B, 4+nc, N)
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: list[int] | torch.Tensor | None = None,
    agnostic: bool = False,
    pre_topk: int = 1024,
    max_det_per_img: int = 300,
    max_wh: float = 7680.0,
) -> list[torch.Tensor]:

    assert prediction.is_cuda, "prediction must be CUDA tensor"
    device = prediction.device

    # 1. 입력 shape 확인 및 전치 (B, 4+nc, N) -> (B, N, 4+nc)
    B, C, N = prediction.shape
    nc = C - 4

    # FP16 강제 (NMS 속도 향상)
    if prediction.dtype != torch.float16:
        prediction = prediction.half()

    # Transpose 및 복사 (Memory Layout 최적화)
    pred = prediction.transpose(-1, -2).contiguous()

    # 2. 좌표 변환 (xywh -> xyxy) [안전한 방식]
    # In-place 연산의 모호함을 없애기 위해 명시적으로 계산하여 할당
    xc = pred[..., 0]  # center x
    yc = pred[..., 1]  # center y
    w = pred[..., 2]  # width
    h = pred[..., 3]  # height

    x1 = xc - w * 0.5
    y1 = yc - h * 0.5
    x2 = xc + w * 0.5
    y2 = yc + h * 0.5

    # 변환된 좌표를 다시 pred에 저장
    pred[..., 0] = x1
    pred[..., 1] = y1
    pred[..., 2] = x2
    pred[..., 3] = y2

    # 3. Class Score 추출
    boxes_all = pred[..., :4]  # (B, N, 4)
    cls_all = pred[..., 4 : 4 + nc]  # (B, N, nc)

    # 각 앵커별 최고 점수 클래스 추출
    # conf(obj_conf)가 별도로 없는 구조(YOLOv8 등)라고 가정
    cls_max, cls_argmax = cls_all.max(dim=2)  # (B, N)

    # 4. Confidence Filtering
    keep_mask = cls_max > conf_thres

    # 클래스 필터 텐서 준비
    classes_tensor = None
    if classes is not None:
        classes_tensor = torch.as_tensor(classes, device=device)

    outs = []

    for b in range(B):
        # 배치별 유효한 박스 필터링
        km = keep_mask[b]
        if not km.any():
            outs.append(torch.empty((0, 6), device=device, dtype=prediction.dtype))
            continue

        # 유효한 박스만 추출
        boxes = boxes_all[b][km]
        scores = cls_max[b][km]
        cls_idxs = cls_argmax[b][km]

        # 5. 특정 클래스만 필터링 (옵션)
        if classes_tensor is not None:
            # 현재 박스의 클래스가 타겟 클래스에 포함되는지 확인
            cls_mask = torch.isin(cls_idxs, classes_tensor)
            if not cls_mask.any():
                outs.append(torch.empty((0, 6), device=device, dtype=prediction.dtype))
                continue

            boxes = boxes[cls_mask]
            scores = scores[cls_mask]
            cls_idxs = cls_idxs[cls_mask]

        # 6. Agnostic NMS 처리 (클래스별 독립적 NMS 방지용 오프셋)
        # agnostic=True면 오프셋을 주지 않아 모든 클래스가 서로 겹치면 제거됨 (사람 필터링 시 적합)
        nms_boxes = boxes
        if not agnostic:
            # 클래스별로 좌표를 멀리 떨어뜨려 서로 IOU가 0이 되게 함
            class_offset = cls_idxs.view(-1, 1) * max_wh
            nms_boxes = boxes + class_offset

        # 7. Pre-NMS Top-K (속도 최적화)
        if nms_boxes.shape[0] > pre_topk:
            scores, idx = scores.sort(descending=True)
            scores = scores[:pre_topk]
            nms_boxes = nms_boxes[idx[:pre_topk]]
            cls_idxs = cls_idxs[idx[:pre_topk]]
            boxes = boxes[idx[:pre_topk]]  # 원본 좌표도 순서 맞춤

        # 8. NMS 수행 (Torchvision 표준 함수 사용)
        # torchvision.ops.nms는 FP16 입력을 지원합니다.
        keep_indices = torchvision.ops.nms(nms_boxes, scores, iou_thres)

        # 9. Max Det 제한
        if keep_indices.shape[0] > max_det_per_img:
            keep_indices = keep_indices[:max_det_per_img]

        # 최종 결과 취합
        kept_boxes = boxes[keep_indices]
        kept_scores = scores[keep_indices]
        kept_cls = cls_idxs[keep_indices]

        # (M, 6) 형태: [x1, y1, x2, y2, score, class]
        final_det = torch.cat(
            [kept_boxes, kept_scores.unsqueeze(1), kept_cls.float().unsqueeze(1)], dim=1
        )

        outs.append(final_det)

    return outs
