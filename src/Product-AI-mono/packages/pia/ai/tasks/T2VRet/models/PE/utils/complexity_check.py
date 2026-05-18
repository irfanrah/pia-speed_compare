import time
import numpy as np
import torch

from pia.ai.tasks.T2VRet.base import T2VRetConfig
from pia.ai.model import PiaTorchModel

# --- NVML (no nvidia-smi parsing) ---
try:
    import pynvml as nvml
    _NVML_OK = True
except Exception:
    _NVML_OK = False


# -----------------------------
# Utilities
def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def time_call(fn, *args, **kwargs):
    cuda_sync()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    cuda_sync()
    return out, time.perf_counter() - t0

def make_config(model_name: str, device: str, model_path=None, temporal_size=8):
    return T2VRetConfig(
        model_path=model_path,
        device=device,
        tile_config=None,
        model_name=model_name,
        temporal_size=temporal_size,
        img_size=[None, None],
    )

def load_model(config: T2VRetConfig):
    return PiaTorchModel(
        target_task="RET",
        target_model="PerceptionEncoder",
        config=config
    )

def warmup_and_infer(model, video):
    with torch.no_grad():
        _ = model(video=video)                  # warm-up
        _, infer_time = time_call(model, video=video)
    return infer_time

def cleanup(*objs):
    for obj in objs:
        del obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(5)


# -----------------------------
# NVML helpers
def nvml_start():
    if _NVML_OK:
        try:
            nvml.nvmlInit()
        except nvml.NVMLError:
            pass  # ignore if already initialized or unavailable

def nvml_stop():
    if _NVML_OK:
        try:
            nvml.nvmlShutdown()
        except nvml.NVMLError:
            pass

def _gpu_index_from_device(device: str) -> int:
    try:
        if device.startswith("cuda:"):
            return int(device.split(":")[1])
    except Exception:
        pass
    return 0

def get_gpu_stats_nvml(device: str):
    if not _NVML_OK:
        return None
    try:
        idx = _gpu_index_from_device(device)
        handle = nvml.nvmlDeviceGetHandleByIndex(idx)

        name = nvml.nvmlDeviceGetName(handle).decode("utf-8")
        pci = nvml.nvmlDeviceGetPciInfo(handle).busId.decode("utf-8")

        mem = nvml.nvmlDeviceGetMemoryInfo(handle)  # bytes
        util = nvml.nvmlDeviceGetUtilizationRates(handle)  # %
        temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)  # °C

        # power in milliwatts; may raise if unsupported
        try:
            power_mw = nvml.nvmlDeviceGetPowerUsage(handle)
            power_w = round(power_mw / 1000.0, 2)
        except nvml.NVMLError:
            power_w = None

        return {
            "gpu_index": idx,
            "name": name,
            "bus_id": pci,
            "mem_used_MiB": int(mem.used / (1024 * 1024)),
            "mem_total_MiB": int(mem.total / (1024 * 1024)),
            "util_gpu_pct": int(util.gpu),
            "util_mem_pct": int(util.memory),
            "temp_C": int(temp),
            "power_W": power_w,
        }
    except nvml.NVMLError as e:
        return {"error": str(e)}

# def print_gpu_stats(label: str, device: str):
#     stats = get_gpu_stats_nvml(device)
#     if not stats:
#         print(f"[{label}] NVML not available; install 'pynvml' (nvidia-ml-py3).")
#         return
#     if "error" in stats:
#         print(f"[{label}] GPU stats error: {stats['error']}")
#         return
#     used = stats["mem_used_MiB"]; total = stats["mem_total_MiB"]
#     util = stats["util_gpu_pct"]; umem = stats["util_mem_pct"]
#     temp = stats["temp_C"]; pwr = stats["power_W"]
#     name = stats["name"]; idx = stats["gpu_index"]; bus = stats["bus_id"]
#     pwr_str = f"{pwr} W" if pwr is not None else "N/A"
#     print(
#         f"[{label}] GPU {idx} ({name}, {bus}) | "
#         f"Mem: {used}/{total} MiB | Util: {util}% (gpu) / {umem}% (mem) | "
#         f"Temp: {temp}°C | Power: {pwr_str}"
#     )


def print_gpu_stats(label: str, device: str, average_result_count: int = 1, sample_interval_s: float = 0.1):
    """
    Print GPU stats averaged over `average_result_count` samples.
    - average_result_count=1 keeps previous behavior (single snapshot).
    - sample_interval_s controls delay between samples (e.g., 0.1 ~ 100 ms).
    """
    def _safe_mean(vals):
        vals = [v for v in vals if v is not None]
        return (sum(vals) / len(vals)) if vals else None

    # Collect samples
    names, idxs, buses = [], [], []
    used_list, total_list = [], []
    util_list, umem_list = [], []
    temp_list, pwr_list = [], []

    for i in range(max(1, int(average_result_count))):
        stats = get_gpu_stats_nvml(device)
        if not stats:
            print(f"[{label}] NVML not available; install 'pynvml' (nvidia-ml-py3).")
            return
        if "error" in stats:
            print(f"[{label}] GPU stats error: {stats['error']}")
            return

        # metadata (should be stable; keep latest non-empty)
        names.append(stats.get("name"))
        idxs.append(stats.get("gpu_index"))
        buses.append(stats.get("bus_id"))

        used_list.append(stats.get("mem_used_MiB"))
        total_list.append(stats.get("mem_total_MiB"))
        util_list.append(stats.get("util_gpu_pct"))
        umem_list.append(stats.get("util_mem_pct"))
        temp_list.append(stats.get("temp_C"))
        pwr_list.append(stats.get("power_W"))

        if sample_interval_s and i < average_result_count - 1:
            time.sleep(sample_interval_s)

    # Use last seen metadata (typically constant)
    name = next((n for n in reversed(names) if n is not None), "Unknown")
    idx  = next((x for x in reversed(idxs) if x is not None), "N/A")
    bus  = next((b for b in reversed(buses) if b is not None), "N/A")

    used  = _safe_mean(used_list)
    total = _safe_mean(total_list)
    util  = _safe_mean(util_list)
    umem  = _safe_mean(umem_list)
    temp  = _safe_mean(temp_list)
    pwr   = _safe_mean(pwr_list)

    # Pretty formatting
    used_str = f"{used:.0f}" if used is not None else "N/A"
    total_str = f"{total:.0f}" if total is not None else "N/A"
    util_str = f"{util:.0f}" if util is not None else "N/A"
    umem_str = f"{umem:.0f}" if umem is not None else "N/A"
    temp_str = f"{temp:.0f}" if temp is not None else "N/A"
    pwr_str  = f"{pwr:.0f} W" if pwr is not None else "N/A"

    suffix = "" if average_result_count == 1 else f" (avg of {average_result_count})"
    print(
        f"[{label}] GPU {idx} ({name}, {bus}){suffix} | "
        f"Mem: {used_str}/{total_str} MiB | Util: {util_str}% (gpu) / {umem_str}% (mem) | "
        f"Temp: {temp_str}°C | Power: {pwr_str}"
    )


# -----------------------------
# Bench helpers

def bench_online(model_name: str, device: str, video, average_result_count:int):
    print("=== Online load ===")
    load_start = time.perf_counter()
    cfg = make_config(model_name=model_name, device=device, model_path=None)
    model = load_model(cfg)
    load_time = time.perf_counter() - load_start
    print(f"Online model load time: {load_time:.3f} s")
    print_gpu_stats("After online init", device, average_result_count=average_result_count)

    # -------- AVG INFERENCE --------
    all_times = []
    for _ in range(average_result_count):
        t = warmup_and_infer(model, video)
        all_times.append(t)
    mean_t = np.mean(all_times)
    std_t  = np.std(all_times)
    print(f"Online model inference avg over {average_result_count}: {mean_t:.3f} ± {std_t:.3f} s")
    print_gpu_stats("After online inference", device, average_result_count=average_result_count)

    cleanup(model, cfg)


def bench_offline(model_name: str, model_path: str, device: str, video, average_result_count:int):
    print("\n=== Offline load (with weights) ===")
    load_start = time.perf_counter()
    cfg = make_config(model_name=model_name, device=device, model_path=model_path)
    model = load_model(cfg)
    load_time = time.perf_counter() - load_start
    print(f"Offline model load time: {load_time:.3f} s")
    print_gpu_stats("After offline init", device, average_result_count=average_result_count)

    # -------- AVG INFERENCE --------
    all_times = []
    for _ in range(average_result_count):
        t = warmup_and_infer(model, video)
        all_times.append(t)
    mean_t = np.mean(all_times)
    std_t  = np.std(all_times)
    print(f"Offline model inference avg over {average_result_count}: {mean_t:.3f} ± {std_t:.3f} s")
    print_gpu_stats("After offline inference", device, average_result_count=average_result_count)

    cleanup(model, cfg)


def bench_lora(
        model_name: str,
        model_path: str,
        adapter_path: str,
        device: str, 
        video, 
        text, 
        average_result_count: int):
    print("\n=== Offline LoRA ===")

    load_start = time.perf_counter()
    # Keep your T2VRetConfig usage so adapter_path is explicit
    cfg = T2VRetConfig(
        model_path   = model_path,
        device       = device,
        tile_config  = None,
        model_name   = model_name,
        temporal_size= 8,
        img_size     = [None, None],
        adapter_path = adapter_path
    )
    # Reuse your loader + warmup/infer utilities
    model = load_model(cfg)
    load_time = time.perf_counter() - load_start
    print(f"LoRA model load time: {load_time:.3f} s")
    print_gpu_stats("After LoRA init", device, average_result_count=average_result_count)

    # Average inference timing (same as others)
    all_times = []
    for _ in range(average_result_count):
        t = warmup_and_infer(model, video)
        all_times.append(t)
    mean_t = np.mean(all_times)
    std_t  = np.std(all_times)
    print(f"LoRA model inference avg over {average_result_count}: {mean_t:.3f} ± {std_t:.3f} s")
    print_gpu_stats("After LoRA inference", device, average_result_count=average_result_count)

    cleanup(model, cfg)


def bench_lora_head(
        model_name: str,
        model_path: str or None,
        adapter_path: str or None,
        head_type: str,
        device: str, 
        video, 
        text, 
        average_result_count: int):
    model_name = f"{model_name}_{head_type}"
    print(f"\n=== Offline LoRA+{head_type} ===")
    # Your shared paths
    load_start = time.perf_counter()
    cfg = T2VRetConfig(
        model_path   = model_path,
        device       = device,
        tile_config  = None,
        model_name   = model_name,
        temporal_size= 8,
        img_size     = [None, None],
        adapter_path = adapter_path
    )
    model = load_model(cfg)
    load_time = time.perf_counter() - load_start
    print(f"LoRA+{head_type} model load time: {load_time:.3f} s")
    print_gpu_stats(f"After LoRA+{head_type} init", device, average_result_count=average_result_count)

    all_times = []
    for _ in range(average_result_count):
        t = warmup_and_infer(model, video)
        all_times.append(t)
    mean_t = np.mean(all_times)
    std_t  = np.std(all_times)
    print(f"LoRA+{head_type} model inference avg over {average_result_count}: {mean_t:.3f} ± {std_t:.3f} s")
    print_gpu_stats(f"After LoRA+{head_type} inference", device, average_result_count=average_result_count)

    cleanup(model, cfg)

# -----------------------------
# Main (keeps your original shapes & settings)
if __name__ == "__main__":
    # Start NVML once
    nvml_start()
    AVERAGE_RESULT_COUNT = 10

    # Dummy inputs
    dummy_B_S_H_W_C_input = np.random.randint(
        0, 256, size=(1, 8, 336, 336, 3), dtype=np.uint8
    )

    # (Unused in timing, kept for compatibility)
    dummy_text = [
        "a person is playing a guitar",
        "a person is playing a piano"
    ]

    DEVICE = "cuda:0"

    # print("\n#################### Benchmarks PE-Core-L14-336 ####################")
    # bench_offline(model_name="PE-Core-L14-336", 
    #               model_path="/home/kurnianto/Downloads/PE-Core-L14-336.pt", 
    #               device=DEVICE, 
    #               video=dummy_B_S_H_W_C_input, 
    #               average_result_count=AVERAGE_RESULT_COUNT)
    
    # bench_online(model_name="PE-Core-L14-336", 
    #              device=DEVICE, 
    #              video=dummy_B_S_H_W_C_input, 
    #              average_result_count=AVERAGE_RESULT_COUNT)
    
    # # L-14 LoRa
    # MODEL_NAME   = "FT_PE-Core-L14-336_250804"
    # MODEL_PATH   = "/home/kurnianto/code/Package-Common-AI-pia_ai_package/assets/models/PIA-SPACE-LAB/FT_PE-Core-L14-336_250804/FT_PE-Core-L14-336_250804.pt"
    # ADAPTER_PATH = "/home/kurnianto/code/Package-Common-AI-pia_ai_package/assets/models/PIA-SPACE-LAB/FT_PE-Core-L14-336_250804/FT_PE-Core-L14-336_250804_adapter"
    # bench_lora(
    #     model_name=MODEL_NAME,
    #     model_path=MODEL_PATH,
    #     adapter_path=ADAPTER_PATH,
    #     device=DEVICE,
    #     video=dummy_B_S_H_W_C_input,
    #     text=dummy_text,
    #     average_result_count=AVERAGE_RESULT_COUNT
    #     )

    # # L-14 MHCA
    # MODEL_NAME   = "FT_PE-Core-L14-336_MHCA_250915"
    # MODEL_PATH   = "/home/kurnianto/code/Package-Common-AI-pia_ai_package/assets/models/PIA-SPACE-LAB/FT_PE-Core-L14-336_MHCA_250915/FT_PE-Core-L14-336_MHCA_250915.pt"
    # ADAPTER_PATH = "/home/kurnianto/code/Package-Common-AI-pia_ai_package/assets/models/PIA-SPACE-LAB/FT_PE-Core-L14-336_MHCA_250915/FT_PE-Core-L14-336_MHCA_250915_adapter"

    # bench_lora_mhca(
    #     model_name=MODEL_NAME,
    #     model_path=MODEL_PATH,
    #     adapter_path=ADAPTER_PATH,
    #     device=DEVICE,
    #     video=dummy_B_S_H_W_C_input,
    #     text=dummy_text,
    #     average_result_count=AVERAGE_RESULT_COUNT
    #     )



    print("\n #################### Benchmarks PE-Core-S16-384 ####################")
    MODEL_NAME = "PE-Core-S16-384"
    dummy_B_S_H_W_C_input = np.random.randint(
        0, 256, size=(1, 8, 384, 384, 3), dtype=np.uint8
    )

    bench_offline(model_name=MODEL_NAME, 
                  model_path="/home/kurnianto/Downloads/PE-Core-S16-384.pt", 
                  device=DEVICE, 
                  video=dummy_B_S_H_W_C_input, 
                  average_result_count=AVERAGE_RESULT_COUNT)
    
    bench_online(model_name=MODEL_NAME, 
                 device=DEVICE, 
                 video=dummy_B_S_H_W_C_input, 
                 average_result_count=AVERAGE_RESULT_COUNT)

    MODEL_NAME   = "FT_PE-Core-S16-384_251115"
    MODEL_PATH   = "/home/kurnianto/code/Package-Common-AI-pia_ai_package/assets/models/PIA-SPACE-LAB/FT_PE-Core-S16-384_251115/FT_PE-Core-S16-384_251115.pt"
    ADAPTER_PATH = "/home/kurnianto/code/Package-Common-AI-pia_ai_package/assets/models/PIA-SPACE-LAB/FT_PE-Core-S16-384_251115/FT_PE-Core-S16-384_251115_adapter"
    bench_lora(
        model_name=MODEL_NAME,
        model_path=MODEL_PATH,
        adapter_path=ADAPTER_PATH,
        device=DEVICE,
        video=dummy_B_S_H_W_C_input,
        text=dummy_text,
        average_result_count=AVERAGE_RESULT_COUNT
        )
    

    MODEL_NAME = "FT_PE-Core-S16-384"
    bench_lora_head(
        model_name=MODEL_NAME,
        model_path=None,
        adapter_path=None,
        head_type="Linear_1",
        device=DEVICE,
        video=dummy_B_S_H_W_C_input,
        text=dummy_text,
        average_result_count=AVERAGE_RESULT_COUNT
        )
    
    bench_lora_head(
        model_name=MODEL_NAME,
        model_path=None,
        adapter_path=None,
        head_type="MHCA_1",
        device=DEVICE,
        video=dummy_B_S_H_W_C_input,
        text=dummy_text,
        average_result_count=AVERAGE_RESULT_COUNT
        )

    bench_lora_head(
        model_name=MODEL_NAME,
        model_path=None,
        adapter_path=None,
        head_type="vid_efficient_1",
        device=DEVICE,
        video=dummy_B_S_H_W_C_input,
        text=dummy_text,
        average_result_count=AVERAGE_RESULT_COUNT
        )

    bench_lora_head(
        model_name=MODEL_NAME,
        model_path=None,
        adapter_path=None,
        head_type="perciever_1",
        device=DEVICE,
        video=dummy_B_S_H_W_C_input,
        text=dummy_text,
        average_result_count=AVERAGE_RESULT_COUNT
        )

    bench_lora_head(
        model_name=MODEL_NAME,
        model_path=None,
        adapter_path=None,
        head_type="perciever_2",
        device=DEVICE,
        video=dummy_B_S_H_W_C_input,
        text=dummy_text,
        average_result_count=AVERAGE_RESULT_COUNT
        )


    # Stop NVML
    nvml_stop()
