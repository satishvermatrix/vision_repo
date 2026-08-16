"""Sanity-check that PyTorch sees and can use the GB10 (Blackwell) GPU."""

import torch


def main() -> None:
    print(f"torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA runtime: {torch.version.cuda}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available - check driver / wheel build.")

    dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device: {name} (sm_{cap[0]}{cap[1]})")

    a = torch.randn(4096, 4096, device=dev)
    b = torch.randn(4096, 4096, device=dev)
    c = a @ b
    torch.cuda.synchronize()
    print(f"matmul ok: {tuple(c.shape)} sum={c.float().sum().item():.2f}")
    print("GPU verification passed.")


if __name__ == "__main__":
    main()
