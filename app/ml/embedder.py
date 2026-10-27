"""Lightweight CNN feature extractor (pretrained MobileNetV3-Small, frozen)."""
from typing import Protocol

import numpy as np

EMBEDDING_DIM = 576


class Embedder(Protocol):
    def embed(self, rgb: np.ndarray) -> np.ndarray:
        """RGB uint8 HxWx3 -> L2-normalized float32 vector of size EMBEDDING_DIM."""
        ...


class MobileNetEmbedder:
    def __init__(self, threads: int = 2) -> None:
        import torch
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        torch.set_num_threads(max(1, threads))
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self._features = torch.nn.Sequential(model.features, model.avgpool).eval()
        self._torch = torch
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def embed(self, rgb: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = (x - self._mean) / self._std
        with torch.inference_mode():
            v = self._features(x).flatten().numpy().astype(np.float32)
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else v
