import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two unit vectors."""
    return float(a @ b) 