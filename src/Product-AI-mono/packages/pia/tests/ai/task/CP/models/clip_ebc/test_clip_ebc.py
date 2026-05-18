import numpy as np
from PIL import Image


def test_predict_with_file(model, image_path):
    # 1. 이미지를 Numpy 배열로 로드
    image_np = np.array(Image.open(image_path).convert("RGB"))

    # 2. 단일 이미지를 리스트에 담아 predict 함수 호출
    image_list = [image_np]
    result = model(image_list)

    # 3. 결과 검증
    assert isinstance(result, list), "결과는 리스트 타입이어야 합니다."
    assert len(result) == 1, "결과 리스트의 길이는 1이어야 합니다."
    assert isinstance(
        result[0], (float, np.floating)
    ), "리스트의 요소는 float 타입이어야 합니다."


def test_predict_with_multiple_different_images_in_list(
        model,
        image_path,
        image_path2):
    """Case 2: 크기가 다른 여러 이미지를 리스트로 전달하여 예측 테스트"""
    # 1. 크기가 다른 두 이미지를 각각 Numpy 배열로 로드
    image_np_1 = np.array(Image.open(image_path).convert("RGB"))
    image_np_2 = np.array(Image.open(image_path2).convert("RGB"))

    # 원본 이미지 크기가 다른지 확인 (테스트의 유효성 검사)
    assert (
        image_np_1.shape != image_np_2.shape
    ), "테스트를 위해 두 이미지의 크기가 달라야 합니다."

    # 2. 두 이미지를 리스트에 담아 predict 함수 호출
    image_list = [image_np_1, image_np_2]
    result = model(image_list)

    # 3. 결과 검증
    assert isinstance(result, list), "결과는 리스트 타입이어야 합니다."
    assert len(result) == 2, "결과 리스트의 길이는 2여야 합니다."
    assert all(
        isinstance(x, (float, np.floating)) for x in result
    ), "리스트의 모든 요소는 float 타입이어야 합니다."
    # 두 이미지가 다르므로 예측 결과도 달라야 함
    assert result[0] != result[1], "서로 다른 이미지에 대한 예측 결과는 달라야 합니다."


def test_predict_with_dense_dot_with_file(model, image_path):
    image_np = np.array(Image.open(image_path).convert("RGB"))

    image_list = [image_np]
    counts, heat_overlays, dot_overlays = model.predict_dense_dot(image_list)

    assert isinstance(counts, list), "counts는 리스트 타입이어야 합니다."
    assert len(counts) == 1, "counts 리스트의 길이는 1이어야 합니다."
    assert isinstance(
        counts[0], (float, np.floating)
    ), "counts의 요소는 float 타입이어야 합니다."

    assert isinstance(heat_overlays, list), "heat_overlays는 리스트 타입이어야 합니다."
    assert len(heat_overlays) == 1, "heat_overlays 리스트의 길이는 1이어야 합니다."
    assert isinstance(
        heat_overlays[0], np.ndarray
    ), "heat_overlays의 요소는 numpy 배열이어야 합니다."

    assert isinstance(dot_overlays, list), "dot_overlays는 리스트 타입이어야 합니다."
    assert len(dot_overlays) == 1, "dot_overlays 리스트의 길이는 1이어야 합니다."
    assert isinstance(
        dot_overlays[0], np.ndarray
    ), "dot_overlays의 요소는 numpy 배열이어야 합니다."
