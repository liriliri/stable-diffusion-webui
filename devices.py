import json
import sys
import torch
from packaging import version

def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    if version.parse(torch.__version__) <= version.parse("2.0.1"):
        if not getattr(torch, 'has_mps', False):
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False
    else:
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()

devices = []

if has_mps():
    devices.append("mps")

devices.append("cpu")

result = {
    "devices": devices
}

print(json.dumps(result))
