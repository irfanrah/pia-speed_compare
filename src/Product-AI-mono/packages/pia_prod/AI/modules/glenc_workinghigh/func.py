from typing import Sequence

from shapely.geometry import Point, Polygon


def any_bbox_corner_in_roi(bbox: Sequence[float], roi: Sequence[Sequence[float]]) -> bool:
    """bbox 의 4 corner 중 *하나라도* RoI polygon 안에 있으면 True.

    Daegu_intrusion 의 ``calc_intersect`` (bbox 의 모든 점이 RoI 안일 때만 True)
    와 다른 점: 고소작업 (workinghigh) 은 사람의 일부만 RoI 안에 들어와도
    이벤트로 카운트해야 한다.

    Args:
        bbox: ``(x1, y1, x2, y2)`` 직사각형 좌표.
        roi: polygon 좌표 리스트 ``[[x, y], [x, y], ...]``.

    Returns:
        4 모서리 ``(x1, y1), (x2, y1), (x1, y2), (x2, y2)`` 중 하나라도 RoI
        polygon 내부일 때 True.
    """
    # GPU torch.Tensor 가 들어와도 동작하도록 Python float 으로 변환.
    # shapely.Point 는 내부에서 np.asarray(...).squeeze() 를 호출하는데,
    # CUDA tensor 는 .numpy() 직접 호출이 불가능해 exception 이 난다.
    # (prod 모드에서는 try_except_only_in_prod_mode 가 silent 로 삼키며,
    # 이 frame 의 inference 결과가 통째로 누락되어 saved video 가 jumpy 해진다.)
    x1, y1, x2, y2 = (float(v) for v in bbox)
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    poly = Polygon(roi)
    return any(poly.contains(Point(c)) for c in corners)
