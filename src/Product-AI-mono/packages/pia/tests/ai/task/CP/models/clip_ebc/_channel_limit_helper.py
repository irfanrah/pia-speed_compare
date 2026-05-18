"""
멀티채널 GPU 한계 탐색 공통 헬퍼

test_limit_channel.py / test_limit_channel_hd.py 에서 공유하는
측정 로직, 출력 포맷, 요약 보고를 하나로 통합한다.
"""

import gc
import os
import time

import numpy as np
import psutil
import pytest

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False
    _NVML_HANDLE = None

from pia.ai.model import PiaTorchModel
from pia.ai.tasks.CP.base import CPONNXConfig
from pia.tests.test_config import ASSETS_MODEL_SAVE_DIR

WINDOW_SIZE = 224
STRIDE = 224
MODEL_PATH = os.path.join(ASSETS_MODEL_SAVE_DIR, "CLIP_EBC_nwpu_rmse_onnx.onnx")
CHANNEL_SEQUENCE = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def calc_tiles(h: int, w: int, ws: int = WINDOW_SIZE, st: int = STRIDE) -> int:
    rows = max(1, int(np.ceil((h - ws) / st)) + 1)
    cols = max(1, int(np.ceil((w - ws) / st)) + 1)
    return rows * cols


def gpu_used_mb() -> float:
    if not _NVML_AVAILABLE:
        return -1.0
    info = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
    return info.used / 1024**2


def gpu_free_mb() -> float:
    if not _NVML_AVAILABLE:
        return -1.0
    info = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
    return info.free / 1024**2


def cpu_mem_mb() -> float:
    return psutil.Process().memory_info().rss / 1024**2


def sep(title: str = ""):
    line = "\u2500" * 70
    print(f"\n{line}")
    if title:
        print(f"  {title}")
        print(line)


def create_model_fixture(h: int, w: int):
    """module-scope fixture 용 모델 생성 + warm-up."""
    config = CPONNXConfig(model_path=MODEL_PATH)
    m = PiaTorchModel(target_task="CP", target_model="clip_ebc_onnx", config=config)
    warmup_img = [np.zeros((h, w, 3), dtype=np.uint8)]
    for _ in range(3):
        m(warmup_img)
    return m


def run_channel_limit_test(model, h: int, w: int, label: str):
    """
    멀티채널 GPU 한계 탐색 실행.

    Args:
        model: 추론 모델
        h, w: 이미지 해상도
        label: 출력 라벨 (예: "FHD(1920x1080)")
    """
    tiles_per_image = calc_tiles(h, w)

    sep(f"{label} 멀티채널 GPU 한계 탐색")

    if _NVML_AVAILABLE:
        gpu_name = pynvml.nvmlDeviceGetName(_NVML_HANDLE)
        total_gpu_mb = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE).total / 1024**2
        print(f"  GPU: {gpu_name}")
        print(f"  총 GPU 메모리: {total_gpu_mb / 1024:.1f} GB")
        print(f"  {label} 타일 수/채널: {tiles_per_image}개")
    print()

    header = (
        f"  {'채널':>6}  {'총타일':>7}  {'지연(ms)':>10}  "
        f"{'채널FPS':>8}  {'전체처리량':>10}  {'GPU사용(MB)':>12}  {'GPU증분(MB)':>12}  "
        f"{'CPU(MB)':>9}  {'상태':>6}"
    )
    print(header)
    print(
        f"  {'-' * 6}  {'-' * 7}  {'-' * 10}  {'-' * 8}  {'-' * 10}  "
        f"{'-' * 12}  {'-' * 12}  {'-' * 9}  {'-' * 6}"
    )

    results = []
    max_ok_channels = 0
    baseline_gpu_mb = gpu_used_mb()

    for n_ch in CHANNEL_SEQUENCE:
        images = [np.zeros((h, w, 3), dtype=np.uint8)] * n_ch
        total_tiles = tiles_per_image * n_ch

        try:
            t0 = time.perf_counter()
            result = model(images)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            gpu_after = gpu_used_mb()
            cpu_after = cpu_mem_mb()
            gpu_delta = gpu_after - baseline_gpu_mb
            ch_fps = 1000.0 / elapsed_ms
            total_tput = n_ch / (elapsed_ms / 1000.0)

            assert isinstance(result, list) and len(result) == n_ch

            row = {
                "channels": n_ch,
                "tiles": total_tiles,
                "latency_ms": elapsed_ms,
                "ch_fps": ch_fps,
                "total_tput": total_tput,
                "gpu_used_mb": gpu_after,
                "gpu_delta_mb": gpu_delta,
                "cpu_mb": cpu_after,
                "status": "OK",
                "error": None,
            }
            results.append(row)
            max_ok_channels = n_ch

            print(
                f"  {n_ch:>6}  {total_tiles:>7}  {elapsed_ms:>9.1f}ms  "
                f"{ch_fps:>7.2f}  {total_tput:>9.1f}  "
                f"{gpu_after:>11.1f}  {gpu_delta:>+11.1f}  "
                f"{cpu_after:>8.1f}  {'pass':>6}"
            )

        except Exception as e:
            err_type = type(e).__name__
            short_msg = str(e)[:60].replace("\n", " ")

            row = {
                "channels": n_ch,
                "tiles": total_tiles,
                "latency_ms": None,
                "ch_fps": None,
                "total_tput": None,
                "gpu_used_mb": gpu_used_mb(),
                "gpu_delta_mb": gpu_used_mb() - baseline_gpu_mb,
                "cpu_mb": cpu_mem_mb(),
                "status": "OOM/ERR",
                "error": f"{err_type}: {short_msg}",
            }
            results.append(row)

            print(
                f"  {n_ch:>6}  {total_tiles:>7}  {'---':>10}  "
                f"{'---':>8}  {'---':>10}  {gpu_used_mb():>11.1f}  {'---':>12}  "
                f"{cpu_mem_mb():>8.1f}  {'FAIL':>6}"
            )
            print(f"          -> {err_type}: {short_msg}")

            gc.collect()
            break

        finally:
            gc.collect()

    # 최종 요약
    sep("최종 요약")

    ok_rows = [r for r in results if r["status"] == "OK"]
    fail_rows = [r for r in results if r["status"] != "OK"]

    print(f"  최대 처리 가능 채널 수: {max_ok_channels} 채널")
    print(f"  최대 처리 시 총 타일 수: {tiles_per_image * max_ok_channels}개")

    if ok_rows:
        last = ok_rows[-1]
        print(
            f"  최대 채널 지연시간:        {last['latency_ms']:.1f} ms  (배치 1회 완료까지)"
        )
        print(
            f"  채널당 갱신 주기(채널FPS): {last['ch_fps']:.2f} FPS  <- 각 채널이 초당 결과 받는 횟수"
        )
        print(
            f"  전체 시스템 처리량:        {last['total_tput']:.1f} images/s  <- 초당 총 이미지 수"
        )
        print(
            f"  최대 채널 GPU 메모리: {last['gpu_used_mb']:.1f} MB ({last['gpu_used_mb'] / 1024:.2f} GB)"
        )
        print(f"  최대 채널 GPU 증분:   {last['gpu_delta_mb']:+.1f} MB")
        print(
            f"  최대 채널 CPU 메모리: {last['cpu_mb']:.1f} MB ({last['cpu_mb'] / 1024:.2f} GB)"
        )

    if fail_rows:
        f = fail_rows[0]
        print(f"\n  한계 초과 채널: {f['channels']} 채널 (타일 {f['tiles']}개)")
        print(f"  실패 원인: {f['error']}")

    if len(ok_rows) >= 2:
        ch_vals = np.array([r["channels"] for r in ok_rows], dtype=float)
        gpu_vals = np.array([r["gpu_delta_mb"] for r in ok_rows], dtype=float)
        if ch_vals[-1] > ch_vals[0]:
            slope = (gpu_vals[-1] - gpu_vals[0]) / (ch_vals[-1] - ch_vals[0])
            print(f"\n  GPU 메모리 증가율 추정: 채널당 약 {slope:.1f} MB")
            if _NVML_AVAILABLE:
                free_mb = gpu_free_mb() + (gpu_vals[-1] - gpu_vals[0])
                est_max = int(ch_vals[-1] + free_mb / max(slope, 1))
                print(
                    f"  이론적 최대 채널 추정: 약 {est_max} 채널 (현재 여유 메모리 기준)"
                )

    sep()
    print(f"\n  처리 가능 채널 수: {max_ok_channels}")

    assert max_ok_channels >= 1, "1채널조차 처리 실패"
