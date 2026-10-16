"""Milestone 1 environment check: CUDA build, sm_120 support, device capability, matmul benchmark."""

import time

import torch

REQUIRED_ARCHITECTURE = "sm_120"
MATRIX_SIZE = 8192
BENCHMARK_REPEATS = 10


def matmul_teraflops(compute_device: torch.device) -> float:
    """Measure bf16 matrix-multiplication throughput.

    Args:
        compute_device: CUDA device.

    Returns:
        Achieved TFLOP/s.
    """
    left_matrix = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=compute_device, dtype=torch.bfloat16)
    right_matrix = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=compute_device, dtype=torch.bfloat16)
    torch.matmul(left_matrix, right_matrix)
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(BENCHMARK_REPEATS):
        torch.matmul(left_matrix, right_matrix)
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start_time
    return 2 * MATRIX_SIZE**3 * BENCHMARK_REPEATS / elapsed_seconds / 1e12


def describe_environment() -> dict:
    """Collect the environment facts required by STAGE_CLEAN.md milestone 1.

    Usage:
        ``python -m readermarks env-check``.

    Returns:
        Dict with torch/CUDA versions, architectures, device name/capability/memory and TFLOP/s.
    """
    architectures = torch.cuda.get_arch_list()
    environment_facts = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "architectures": architectures,
        "sm_120_supported": REQUIRED_ARCHITECTURE in architectures,
    }
    if torch.cuda.is_available():
        device_properties = torch.cuda.get_device_properties(0)
        environment_facts.update(
            {
                "device": device_properties.name,
                "capability": f"{device_properties.major}.{device_properties.minor}",
                "memory_gb": round(device_properties.total_memory / 1e9, 2),
                "bf16_matmul_tflops": round(matmul_teraflops(torch.device("cuda")), 1),
            }
        )
    return environment_facts
