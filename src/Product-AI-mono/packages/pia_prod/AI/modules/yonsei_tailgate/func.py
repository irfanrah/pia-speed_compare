from shapely import Point, Polygon
from pia.vision.postprocessing.bbox import clip


def xyxy2rhombus_for_topview(x1, y1, x2, y2, expand=0.98, image_size=(640, 640)):
    # 마름모꼴로 변경
    box_width_half = (x2 - x1) / 2
    box_height_half = (y2 - y1) / 2
    p1 = (x2 - box_width_half), (y1 + box_height_half * expand)
    p2 = (x1 + box_width_half * expand), (y2 - box_height_half)
    p3 = (x2 - box_width_half), (y2 - box_height_half * expand)
    p4 = (x2 - box_width_half * expand), (y2 - box_height_half)

    # Top bias. topview 전용 사람 위치에대한 스케일이 필요하여 도입된 상수값. 이상적인 결과값은 발쪽을 가리켜야함
    center_x = box_width_half + x1
    center_y = box_height_half + y1
    center_image_x = image_size[0] / 2
    center_image_y = image_size[1] / 2

    # 중심좌표와 사람의 중심좌표를 비교하여 이동값을 계산
    dx = center_image_x - center_x
    dy = center_image_y - center_y

    # 이동값을 적용하여 사람의 위치를 조정
    max_move_weight_x = 2.2
    max_move_weight_y = 3

    bias_x = (box_width_half * max_move_weight_x) * (dx / image_size[0])
    bias_y = (box_height_half * max_move_weight_y) * (dy / image_size[1])

    # 보정된 좌표를 이미지 내로 제한
    scaled_p1 = (clip(p1[0] + bias_x, 0, image_size[0]), clip(p1[1] + bias_y, 0, image_size[1]))
    scaled_p2 = (clip(p2[0] + bias_x, 0, image_size[0]), clip(p2[1] + bias_y, 0, image_size[1]))
    scaled_p3 = (clip(p3[0] + bias_x, 0, image_size[0]), clip(p3[1] + bias_y, 0, image_size[1]))
    scaled_p4 = (clip(p4[0] + bias_x, 0, image_size[0]), clip(p4[1] + bias_y, 0, image_size[1]))

    return [scaled_p1, scaled_p2, scaled_p3, scaled_p4]


def calc_intersect_for_topview(bbox, rois, result_func_rho):  # bbox : xmin,ymin,xmax,ymax
    """
    ROI 영역과 바운딩 박스의 교차 여부를 계산한다.

    Args:
        bbox (tuple): 바운딩 박스 좌표 (xmin, ymin, xmax, ymax).
        rois (list): ROI 다각형의 좌표 리스트.
        result_func_rho (list): 변환된 바운딩 박스의 마름모꼴 좌표 리스트.

    Returns:
        bool: 바운딩 박스가 ROI 내부에 완전히 포함되면 True, 그렇지 않으면 False.
    """

    # bbox_4_coord = xyxy2rhombus(*bbox, OD_BOX_EXPAND_VALUE)

    # bbox_4_coord = xyxy2rhombus_for_topview(*bbox, OD_BOX_EXPAND_VALUE)
    bbox_4_coord = result_func_rho
    total = 0
    poly = Polygon(rois)

    for b in bbox_4_coord:
        if poly.contains(Point(b)):
            total += 1
    if total == 4:
        return True  # 모든 점이 교차하면 해당 박스는 roi 내에 있다
    else:
        return False
