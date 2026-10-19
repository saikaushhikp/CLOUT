"""Optional CUDA parity smoke; core correctness is deliberately CPU-only."""
import unittest


class OptionalCudaTests(unittest.TestCase):
    def test_torch_cuda_is_usable_when_driver_is_available(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("CUDA driver/device is unavailable")
        value = torch.tensor([1.0], device="cuda") * 2
        self.assertEqual(value.cpu().item(), 2.0)
