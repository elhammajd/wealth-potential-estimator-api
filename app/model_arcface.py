import numpy as np
from PIL import Image
import cv2
import io
import os
from functools import lru_cache
import insightface
from insightface.app import FaceAnalysis


class ArcFaceEmbeddingModel:
    """ArcFace wrapper for high-quality face embeddings."""

    def __init__(self, device=None):
        # ArcFace works well on CPU, so default to CPU for simplicity
        self.device = device or "cpu"
        self.app = self._load_model()

    def _load_model(self):
        """Load InsightFace ArcFace model."""
        print("Loading ArcFace model for face recognition...")
        
        # Initialize face analysis app
        app = FaceAnalysis(providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        print("ArcFace model loaded successfully")
        return app

    def _preprocess_image(self, pil_img):
        """Convert PIL image to OpenCV format."""
        # Convert PIL to OpenCV format (RGB -> BGR)
        img_array = np.array(pil_img)
        if img_array.shape[2] == 3:  # RGB
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:  # RGBA
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        return img_bgr

    def embed_pil(self, pil_img):
        """Extract face embedding from PIL image."""
        try:
            # Convert to OpenCV format
            img_bgr = self._preprocess_image(pil_img)
            
            # Detect faces and extract embeddings
            faces = self.app.get(img_bgr)
            
            if not faces:
                print("Warning: No face detected in image")
                # Fallback: return zero vector
                return np.zeros(512, dtype=np.float32)
            
            # Use the largest face (assuming it's the main subject)
            largest_face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
            
            # Get normalized embedding
            embedding = largest_face.embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            print(f"Face detected, embedding shape: {embedding.shape}")
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"Error processing image: {e}")
            # Return zero vector as fallback
            return np.zeros(512, dtype=np.float32)

    def embed_bytes(self, image_bytes):
        """Extract embedding from image bytes."""
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return self.embed_pil(pil_img)
        except Exception as e:
            print(f"Error loading image: {e}")
            return np.zeros(512, dtype=np.float32)


# For backward compatibility, create an alias
EmbeddingModel = ArcFaceEmbeddingModel
