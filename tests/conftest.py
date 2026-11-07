import pytest
import torch

# Shared, heavily loaded host: small CPU thread pools are far faster than torch's default.
torch.set_num_threads(4)


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="CUDA unavailable")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
