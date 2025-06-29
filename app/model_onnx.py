import numpy as np
from PIL import Image
import cv2
import io
import os
from functools import lru_cache
import insightface
from insightface.app import FaceAnalysis


class EmbeddingModel:
    """ArcFace wrapper using ONNX runtime only - no PyTorch needed."""

    def __init__(self, device=None):
        # ArcFace with ONNX runtime works well on CPU
        self.device = "cpu"
        self.app = self._load_model()

    def _load_model(self):
        """Load InsightFace ArcFace model with ONNX runtime only."""
        print("Loading ArcFace model for face recognition (ONNX runtime, no PyTorch)...")
        
        try:
            # Initialize face analysis app with CPU provider only
            app = FaceAnalysis(providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
            
            print("ArcFace model loaded successfully")
            return app
        except Exception as e:
            print(f"Failed to load ArcFace: {e}")
            print("   Make sure InsightFace and ONNX runtime are installed")
            raise e

    def _preprocess_image(self, pil_img):
        """Convert PIL image to OpenCV format."""
        # Convert PIL to OpenCV format (RGB -> BGR)
        img_array = np.array(pil_img)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:  # RGB
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = img_array
        return img_bgr

    def embed_pil(self, pil_img):
        """Extract face embedding from PIL image."""
        try:
            # Convert to OpenCV format
            img_bgr = self._preprocess_image(pil_img)
            
            # Detect faces and extract embeddings
            faces = self.app.get(img_bgr)
            
            if not faces:
                print("Warning: No face detected in image - using fallback")
                # Return a consistent fallback vector
                np.random.seed(42)
                return np.random.normal(0, 0.1, 512).astype(np.float32)
            
            # Use the largest face (assuming it's the main subject)
            largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            
            # Get normalized embedding
            embedding = largest_face.embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            print(f"Face detected and processed, embedding shape: {embedding.shape}")
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"Error processing image: {e}")
            # Return fallback vector
            np.random.seed(42)
            return np.random.normal(0, 0.1, 512).astype(np.float32)

    def embed_bytes(self, image_bytes):
        """Extract embedding from image bytes."""
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return self.embed_pil(pil_img)
        except Exception as e:
            print(f"Error loading image: {e}")
            np.random.seed(42)
            return np.random.normal(0, 0.1, 512).astype(np.float32)
