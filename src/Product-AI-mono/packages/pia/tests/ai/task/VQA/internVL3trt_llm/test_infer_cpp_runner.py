"""InternVL3 TRT-LLM VQA 추론 기능 테스트."""

from pia.ai.tasks.VQA.base import VQAConfig
from pia.ai.tasks.VQA.models.internVL3trt_llm.main import InternVL3trt_llm

def test_infer_single_image(image_300, falldown_prompt):
    """단일 이미지 입력에서 문자열 응답이 반환되는지 확인한다."""
    cfg = VQAConfig()
    model = InternVL3trt_llm(cfg)
    prompt = falldown_prompt
    img = image_300
    result = model.forward(img, prompt)
    print("---------------------------------------------------")
    print("answer : ", result)
    assert isinstance(result, str)
    assert result.strip() != ""


def test_infer_batch_two_images(image_300, red_image_300):
    """2장 배치 입력에서 응답 리스트가 정상 반환되는지 확인한다."""
    cfg = VQAConfig()
    model = InternVL3trt_llm(cfg)
    prompt = "explain this image, only english"
    results = model.forward([image_300, red_image_300], [prompt, prompt])
    print("---------------------------------------------------")
    print("answer : " , results)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(x, str) and x.strip() != "" for x in results)
