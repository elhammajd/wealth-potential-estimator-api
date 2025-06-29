import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from functools import lru_cache
import io
import os


class EmbeddingModel:
    """ResNet-50 wrapper to get image embeddings."""

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _load_model(self):

        use_pretrained = os.getenv("WEALTHAPI_PRETRAINED", "1") == "1"
        
        if use_pretrained:
            print("Loading pretrained ResNet-50 (this is what you want)")
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            print("WARNING: Using untrained ResNet-50 - this won't work well!")
            print("   Set WEALTHAPI_PRETRAINED=1 for actual results")
            model = models.resnet50(weights=None)
        modules = list(model.children())[:-1]
        backbone = torch.nn.Sequential(*modules)
        backbone.eval()
        backbone.to(self.device)
        # Turn off gradients since we're not training
        for p in backbone.parameters():
            p.requires_grad = False
        return backbone

    def embed_pil(self, pil_img):
        """Turn a PIL image into a feature vector."""
        tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(tensor)
        # Flatten to 1D and normalize for cosine similarity
        embedding = features.squeeze().cpu().numpy()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def embed_bytes(self, image_bytes):
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.embed_pil(pil_img) 