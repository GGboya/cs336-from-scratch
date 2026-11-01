import torch


def main() -> None:
    assert torch.cuda.is_available(), "CUDA is not available"
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    result = (x @ x).float()
    assert torch.isfinite(result).all()
    torch.cuda.synchronize()
    print(f"CUDA matmul passed; allocated memory: {torch.cuda.memory_allocated() / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
