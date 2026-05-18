"""
FHD(1920×1080) 기준 ClipEBCOnnx 리소스 벤치마크 테스트

목적: CCTV 1채널 이미지 입력 기준의 추론 지연시간 및 메모리 사용량 측정
실행: pytest -s packages/pia/tests/ai/task/CP/models/clip_ebc/test_clip_ebc_fhd_benchmark.py
"""

import time

import numpy as np
import psutil
import pytest

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

# CCTV 기준 상수
FHD_H, FHD_W = 1080, 1920
TARGET_FPS = 30
TARGET_LATENCY_MS = 1000.0 / TARGET_FPS  # 33.3ms
N_WARMUP = 3
N_REPEAT = 10


# ──────────────────────────────────────────────
# 헬퍼 함수
# ──────────────────────────────────────────────

def _make_image() -> np.ndarray:
    """FHD 단색(검정) 이미지 생성 (RGB uint8)."""
    return np.zeros((FHD_H, FHD_W, 3), dtype=np.uint8)


def _cpu_mem_mb() -> float:
    """현재 프로세스 RSS CPU 메모리 (MB)."""
    return psutil.Process().memory_info().rss / 1024 ** 2


def _gpu_mem_mb() -> float:
    """현재 프로세스 GPU 메모리 사용량 (MB). pynvml 없으면 -1."""
    if not _NVML_AVAILABLE:
        return -1.0
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024 ** 2
    except Exception:
        return -1.0


def _print_separator(title: str = ""):
    line = "─" * 60
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(line)
    else:
        print(line)


def _run_benchmark(model, image: np.ndarray, label: str):
    """
    공통 벤치마크 실행기.
    warm-up → N_REPEAT회 측정 → 통계 출력
    Returns: latency_list (ms)
    """
    _print_separator(label)

    print(f"  [Warm-up] {N_WARMUP}회 실행 중...")
    for _ in range(N_WARMUP):
        model([image])
    print(f"  [Warm-up] 완료\n")

    latencies_ms: list[float] = []
    cpu_mems_mb: list[float] = []
    gpu_mems_mb: list[float] = []

    print(f"  {'Run':>4}  {'Latency(ms)':>12}  {'CPU Mem(MB)':>12}  {'GPU Mem(MB)':>12}")
    print(f"  {'----':>4}  {'------------':>12}  {'------------':>12}  {'------------':>12}")

    for i in range(N_REPEAT):
        cpu_before = _cpu_mem_mb()
        gpu_before = _gpu_mem_mb()

        t_start = time.perf_counter()
        result = model([image])
        t_end = time.perf_counter()

        elapsed_ms = (t_end - t_start) * 1000.0
        cpu_after = _cpu_mem_mb()
        gpu_after = _gpu_mem_mb()

        latencies_ms.append(elapsed_ms)
        cpu_mems_mb.append(cpu_after)
        gpu_mems_mb.append(gpu_after if gpu_after >= 0 else 0.0)

        gpu_str = f"{gpu_after:>11.1f}" if gpu_after >= 0 else f"{'N/A':>11}"
        print(
            f"  {i+1:>4}  {elapsed_ms:>11.1f}ms  {cpu_after:>11.1f}MB  {gpu_str}MB"
            f"  (count={result[0]:.2f})"
        )

    lat = np.array(latencies_ms)
    cpu = np.array(cpu_mems_mb)
    gpu = np.array(gpu_mems_mb)

    _print_separator()
    print(f"  [통계] {label}")
    print(f"  {'항목':<16} {'평균':>10} {'표준편차':>10} {'최소':>10} {'최대':>10} {'p95':>10}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(
        f"  {'Latency (ms)':<16}"
        f"  {lat.mean():>9.1f}"
        f"  {lat.std():>9.1f}"
        f"  {lat.min():>9.1f}"
        f"  {lat.max():>9.1f}"
        f"  {np.percentile(lat, 95):>9.1f}"
    )
    print(
        f"  {'CPU Mem (MB)':<16}"
        f"  {cpu.mean():>9.1f}"
        f"  {cpu.std():>9.1f}"
        f"  {cpu.min():>9.1f}"
        f"  {cpu.max():>9.1f}"
        f"  {np.percentile(cpu, 95):>9.1f}"
    )
    if _NVML_AVAILABLE:
        print(
            f"  {'GPU Mem (MB)':<16}"
            f"  {gpu.mean():>9.1f}"
            f"  {gpu.std():>9.1f}"
            f"  {gpu.min():>9.1f}"
            f"  {gpu.max():>9.1f}"
            f"  {np.percentile(gpu, 95):>9.1f}"
        )
    else:
        print(f"  {'GPU Mem (MB)':<16}  {'pynvml 미설치, 측정 불가':>10}")

    avg_fps = 1000.0 / lat.mean()
    meet = "✅ 충족" if lat.mean() <= TARGET_LATENCY_MS else "❌ 미충족"
    print(f"\n  처리량(Throughput): {avg_fps:.1f} FPS")
    print(f"  30fps 기준({TARGET_LATENCY_MS:.1f}ms) 충족 여부: {meet}")
    _print_separator()

    return latencies_ms


# ──────────────────────────────────────────────
# 테스트 케이스
# ──────────────────────────────────────────────

def test_fhd_single_channel_benchmark(model):
    """
    [TC-01] FHD 단일채널 이미지 × 10회 추론
    - CCTV 1채널 기준 지연시간 및 메모리 사용량 측정
    """
    image = _make_image()
    assert image.shape == (FHD_H, FHD_W, 3)

    _run_benchmark(model, image, "TC-01: FHD Single Channel (1920×1080)")

    result = model([image])
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], (float, np.floating))


def test_fhd_gpu_memory_stability(model):
    """
    [TC-02] GPU 메모리 누수 안정성 검사
    - 10회 반복 추론 동안 GPU 메모리가 지속적으로 증가하지 않는지 확인
    - CCTV 장시간 운영 시 메모리 누수 여부 판단 기준
    """
    if not _NVML_AVAILABLE:
        pytest.skip("pynvml 미설치 — GPU 메모리 안정성 테스트 건너뜀")

    image = _make_image()

    _print_separator("TC-02: GPU 메모리 누수 안정성 (10회)")
    gpu_snapshots = []
    for i in range(N_REPEAT):
        model([image])
        gpu_mb = _gpu_mem_mb()
        gpu_snapshots.append(gpu_mb)
        print(f"  run {i+1:>2}: GPU={gpu_mb:.1f}MB")

    first, last = gpu_snapshots[0], gpu_snapshots[-1]
    delta = last - first
    print(f"\n  첫 번째: {first:.1f}MB  →  마지막: {last:.1f}MB  (증가량: {delta:+.1f}MB)")

    # 10회 동안 50MB 이상 증가 시 누수 경고
    leak_threshold_mb = 50.0
    assert delta < leak_threshold_mb, (
        f"GPU 메모리가 {delta:.1f}MB 증가했습니다. 누수가 의심됩니다."
    )
    print(f"  GPU 메모리 안정성: ✅ 누수 없음 (임계값 {leak_threshold_mb:.0f}MB 이하)")
    _print_separator()


def test_fhd_streaming_simulation(model):
    """
    [TC-03] CCTV 스트리밍 시뮬레이션 (연속 10프레임)
    - 동일 이미지를 연속으로 처리할 때의 누적 지연시간 측정
    - 실제 30fps 스트림에서 각 프레임이 제시간에 처리 가능한지 확인
    """
    image = _make_image()

    _print_separator("TC-03: CCTV 스트리밍 시뮬레이션 (30fps 타겟)")

    # warm-up
    for _ in range(N_WARMUP):
        model([image])

    total_start = time.perf_counter()
    frame_times_ms = []

    print(f"  {'Frame':>6}  {'Latency(ms)':>12}  {'누적시간(s)':>12}  {'30fps 충족':>10}")
    print(f"  {'------':>6}  {'------------':>12}  {'------------':>12}  {'----------':>10}")

    for frame_idx in range(N_REPEAT):
        t0 = time.perf_counter()
        model([image])
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000.0
        cumulative_s = t1 - total_start
        frame_times_ms.append(elapsed_ms)
        ok = "✅" if elapsed_ms <= TARGET_LATENCY_MS else "❌"
        print(
            f"  {frame_idx+1:>6}  {elapsed_ms:>11.1f}ms"
            f"  {cumulative_s:>11.3f}s"
            f"  {ok:>10}"
        )

    lat = np.array(frame_times_ms)
    total_elapsed_s = time.perf_counter() - total_start

    print(f"\n  총 소요시간({N_REPEAT}프레임): {total_elapsed_s:.3f}s")
    print(f"  평균 FPS: {N_REPEAT / total_elapsed_s:.1f}")
    print(f"  평균 지연: {lat.mean():.1f}ms  최대 지연: {lat.max():.1f}ms")
    ok_count = sum(1 for t in frame_times_ms if t <= TARGET_LATENCY_MS)
    print(f"  30fps 충족 프레임: {ok_count}/{N_REPEAT}")
    _print_separator()


def test_fhd_vs_lower_resolution_comparison(model):
    """
    [TC-04] 해상도별 처리 시간 비교 (FHD vs HD vs SD)
    - 해상도가 추론 시간에 미치는 영향 측정
    - 타일 수: SD(≈4), HD(24), FHD(45)
    """
    resolutions = {
        "SD  ( 640× 360)": (360, 640),
        "HD  (1280× 720)": (720, 1280),
        "FHD (1920×1080)": (1080, 1920),
    }

    _print_separator("TC-04: 해상도별 처리시간 비교")
    print(f"  {'해상도':<20}  {'타일수':>6}  {'평균(ms)':>10}  {'FPS':>8}")
    print(f"  {'-'*20}  {'-'*6}  {'-'*10}  {'-'*8}")

    for label, (h, w) in resolutions.items():
        image = np.zeros((h, w, 3), dtype=np.uint8)

        # 타일 수 계산
        ws = 224
        rows = max(1, int(np.ceil((h - ws) / ws)) + 1)
        cols = max(1, int(np.ceil((w - ws) / ws)) + 1)
        n_tiles = rows * cols

        # warm-up
        for _ in range(2):
            model([image])

        # 5회 측정
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            model([image])
            times.append((time.perf_counter() - t0) * 1000.0)

        avg_ms = np.mean(times)
        fps = 1000.0 / avg_ms
        print(f"  {label:<20}  {n_tiles:>6}  {avg_ms:>9.1f}ms  {fps:>7.1f}")

    _print_separator()
