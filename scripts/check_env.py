"""Step-1 acceptance check: torch/CUDA/Blackwell, BF16 autocast, timm pretrained forward."""
import sys

import torch

print("python        ", sys.version.split()[0])
print("torch         ", torch.__version__, "| built for CUDA", torch.version.cuda)
print("arch list     ", torch.cuda.get_arch_list())
ok = torch.cuda.is_available()
print("cuda available", ok)
if not ok:
    sys.exit("FAIL: CUDA not available (driver/GPU state). Fix before training.")
print("device        ", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
assert torch.cuda.get_device_capability(0) == (12, 0), "expected Blackwell sm_120"

a = torch.randn(1024, 1024, device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    c = a @ a
print("bf16 matmul   ", c.dtype, tuple(c.shape))

import timm, albumentations, cv2
m = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1).cuda().eval()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    y = m(torch.randn(2, 3, 224, 224, device="cuda"))
print("timm b0 out   ", tuple(y.shape), "| timm", timm.__version__, "| albumentations", albumentations.__version__, "| cv2", cv2.__version__)
assert tuple(y.shape) == (2, 1)
print("ALL CHECKS PASSED")
