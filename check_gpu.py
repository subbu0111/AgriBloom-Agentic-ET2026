#!/usr/bin/env python3
"""
GPU/CUDA Availability Checker for AgriBloom Agentic
Optimized for NVIDIA RTX 4060 with CUDA 12.x
"""
from __future__ import annotations

import sys


def check_pytorch_gpu() -> dict:
    """Check PyTorch CUDA availability."""
    try:
        import torch

        info = {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": None,
            "cudnn_version": None,
            "device_count": 0,
            "devices": [],
            "recommended_device": "cpu",
        }

        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
            info["device_count"] = torch.cuda.device_count()

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                })

            info["recommended_device"] = "cuda:0"

            # Memory allocation test
            try:
                test_tensor = torch.zeros(1000, 1000, device="cuda:0")
                del test_tensor
                torch.cuda.empty_cache()
                info["gpu_memory_test"] = "PASSED"
            except Exception as e:
                info["gpu_memory_test"] = f"FAILED: {e}"

        return info
    except ImportError:
        return {"error": "PyTorch not installed"}


def check_onnx_gpu() -> dict:
    """Check ONNX Runtime CUDA availability."""
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        info = {
            "onnxruntime_version": ort.__version__,
            "available_providers": providers,
            "cuda_available": "CUDAExecutionProvider" in providers,
            "tensorrt_available": "TensorRTExecutionProvider" in providers,
            "recommended_providers": [],
        }

        # Prioritize providers
        if "CUDAExecutionProvider" in providers:
            info["recommended_providers"].append("CUDAExecutionProvider")
        if "TensorRTExecutionProvider" in providers:
            info["recommended_providers"].insert(0, "TensorRTExecutionProvider")
        info["recommended_providers"].append("CPUExecutionProvider")

        return info
    except ImportError:
        return {"error": "ONNX Runtime not installed"}


def check_transformers_gpu() -> dict:
    """Check Transformers accelerate support."""
    try:
        import torch
        from transformers import AutoConfig

        info = {
            "transformers_installed": True,
            "accelerate_installed": False,
            "device_map_support": False,
        }

        try:
            import accelerate
            info["accelerate_installed"] = True
            info["accelerate_version"] = accelerate.__version__
            info["device_map_support"] = True
        except ImportError:
            pass

        return info
    except ImportError:
        return {"error": "Transformers not installed"}


def get_optimal_device() -> str:
    """Get the optimal device for inference."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except ImportError:
        pass
    return "cpu"


def get_optimal_onnx_providers() -> list[str]:
    """Get optimal ONNX execution providers."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()

        result = []
        if "CUDAExecutionProvider" in providers:
            result.append("CUDAExecutionProvider")
        result.append("CPUExecutionProvider")
        return result
    except ImportError:
        return ["CPUExecutionProvider"]


def print_gpu_report() -> None:
    """Print comprehensive GPU report."""
    print("=" * 70)
    print("AgriBloom Agentic - GPU/CUDA Availability Report")
    print("=" * 70)

    # PyTorch
    print("\n[PyTorch GPU Status]")
    pytorch_info = check_pytorch_gpu()
    if "error" in pytorch_info:
        print(f"  ERROR: {pytorch_info['error']}")
    else:
        print(f"  Version: {pytorch_info['pytorch_version']}")
        print(f"  CUDA Available: {pytorch_info['cuda_available']}")
        if pytorch_info["cuda_available"]:
            print(f"  CUDA Version: {pytorch_info['cuda_version']}")
            print(f"  cuDNN Version: {pytorch_info['cudnn_version']}")
            print(f"  Device Count: {pytorch_info['device_count']}")
            for dev in pytorch_info["devices"]:
                print(f"    [{dev['index']}] {dev['name']}")
                print(f"        Memory: {dev['total_memory_gb']} GB")
                print(f"        Compute: {dev['compute_capability']}")
                print(f"        SM Count: {dev['multi_processor_count']}")
            print(f"  Memory Test: {pytorch_info.get('gpu_memory_test', 'N/A')}")
        print(f"  Recommended Device: {pytorch_info['recommended_device']}")

    # ONNX Runtime
    print("\n[ONNX Runtime GPU Status]")
    onnx_info = check_onnx_gpu()
    if "error" in onnx_info:
        print(f"  ERROR: {onnx_info['error']}")
    else:
        print(f"  Version: {onnx_info['onnxruntime_version']}")
        print(f"  Available Providers: {', '.join(onnx_info['available_providers'])}")
        print(f"  CUDA Provider: {onnx_info['cuda_available']}")
        print(f"  TensorRT Provider: {onnx_info['tensorrt_available']}")
        print(f"  Recommended: {onnx_info['recommended_providers']}")

    # Transformers
    print("\n[Transformers Status]")
    tf_info = check_transformers_gpu()
    if "error" in tf_info:
        print(f"  ERROR: {tf_info['error']}")
    else:
        print(f"  Installed: {tf_info['transformers_installed']}")
        print(f"  Accelerate: {tf_info['accelerate_installed']}")
        if tf_info.get("accelerate_version"):
            print(f"  Accelerate Version: {tf_info['accelerate_version']}")
        print(f"  Device Map Support: {tf_info['device_map_support']}")

    # Summary
    print("\n" + "=" * 70)
    print("OPTIMAL CONFIGURATION FOR TRAINING:")
    print(f"  PyTorch Device: {get_optimal_device()}")
    print(f"  ONNX Providers: {get_optimal_onnx_providers()}")
    print("=" * 70)


if __name__ == "__main__":
    print_gpu_report()
