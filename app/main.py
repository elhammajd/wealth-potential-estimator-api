import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from .model import EmbeddingModel
from .data import WealthyProfileDB
from .schemas import PredictResponse, Match

import numpy as np
from typing import List, Tuple, Dict, Any

app = FastAPI(title="Wealth Potential Estimator API")

@app.get("/health")
async def health_check():
    """Basic health check - returns OK if the service is running."""
    return {"status": "healthy", "service": "wealth-estimator-api"}

# CORS setup for browser testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load everything when the app starts
embedding_model = EmbeddingModel()
profile_db = WealthyProfileDB()

# Try to load calibrator
try:
    from .calibrator import load_calibrator
    wealth_calibrator = load_calibrator()
    print("Wealth calibrator loaded")
except Exception as e:
    print(f"Could not load calibrator: {e}")
    wealth_calibrator = None


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """Main endpoint - upload a photo and get back a wealth estimate."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Need an image file, not whatever this is.")

    try:
        image_bytes = await file.read()
        embedding = embedding_model.embed_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Couldn't process that image: {e}")

    # Get top 5 matches to work with
    top_matches_raw = profile_db.get_top_k_similar(embedding, k=5)
    
    # Figure out how confident we should be in this prediction
    similarities = [sim for _, sim in top_matches_raw]
    max_similarity = max(similarities) if similarities else 0
    avg_similarity = np.mean(similarities) if similarities else 0
    
    # Basic confidence levels
    if max_similarity < 0.1:
        confidence = "very_low"
        print(f"Really low match scores (max: {max_similarity:.3f}) - this is basically a guess")
    elif max_similarity < 0.2:
        confidence = "low"
        print(f"Low match scores (max: {max_similarity:.3f}) - take this with a grain of salt")
    elif max_similarity < 0.4:
        confidence = "medium"
        print(f"Decent match scores (max: {max_similarity:.3f}) - probably reasonable")
    else:
        confidence = "high"
        print(f"Good match scores (max: {max_similarity:.3f}) - pretty confident here")
    
    # Calculate the wealth estimate using calibrated prediction if available
    if wealth_calibrator and top_matches_raw:
        # Use the highest similarity score for calibrated prediction
        max_similarity = max([sim for _, sim in top_matches_raw])
        estimated = wealth_calibrator.predict_wealth(max_similarity)
        print(f"Using calibrated prediction based on max similarity: {max_similarity:.3f}")
    else:
        # Fallback to original method
        estimated = predict_wealth(embedding, top_matches_raw[:3])
    
    # Return top 3 matches with full info
    top_matches = [
        Match(
            name=profile_dict["name"],
            net_worth=profile_dict["net_worth"],
            age=profile_dict["age"],
            source=profile_dict["source"],
            similarity_score=round(sim, 3)
        )
        for profile_dict, sim in top_matches_raw[:3]
    ]

    return PredictResponse(estimated_net_worth=estimated, top_matches=top_matches)

def predict_wealth(query_embedding: np.ndarray, top_matches: List[Tuple[Dict[str, Any], float]]) -> float:
    """
    Predict wealth based on similarity to database people.
    
    Now that embeddings actually correlate with wealth, we can be smarter:
    - Higher similarity = more confident prediction
    - Weight by both similarity and wealth class consistency
    """
    if not top_matches:
        return 50_000_000  # Default fallback
    
    # Extract similarities and net worths
    similarities = [match[1] for match in top_matches]
    net_worths = [match[0]["net_worth"] for match in top_matches]
    
    # Check if we have good similarity scores (meaningful matches)
    max_similarity = max(similarities)
    avg_similarity = sum(similarities) / len(similarities)
    
    print(f"Prediction confidence: max={max_similarity:.3f}, avg={avg_similarity:.3f}")
    
    if max_similarity < 0.02:
        # Extremely low similarity - probably doesn't match anyone well
        print("   Extremely low similarity - using conservative estimate")
        # Use median of database as fallback
        all_worths = [p["net_worth"] for p in profile_db.profiles]
        return float(np.median(all_worths))
    
    elif max_similarity > 0.06:
        # High similarity for this domain - trust the top match more
        print("   High similarity - confident prediction")
        weights = [s**2 for s in similarities]  # Square to emphasize high similarities
        
    elif avg_similarity > 0.04:
        # Decent similarity across matches
        print("   Good similarity - balanced prediction")
        weights = similarities
        
    else:
        # Medium similarity - be more conservative
        print("   Medium similarity - conservative prediction")
        # Reduce extreme values
        conservative_worths = []
        for worth in net_worths:
            if worth > 10_000_000_000:  # > $10B
                conservative_worths.append(worth * 0.3)  # Reduce by 70%
            elif worth > 1_000_000_000:  # > $1B
                conservative_worths.append(worth * 0.5)  # Reduce by 50%
            else:
                conservative_worths.append(worth)
        net_worths = conservative_worths
        weights = similarities
    
    # Calculate weighted average
    if sum(weights) > 0:
        prediction = sum(w * nw for w, nw in zip(weights, net_worths)) / sum(weights)
    else:
        prediction = sum(net_worths) / len(net_worths)
    
    # Apply final bounds checking
    prediction = max(prediction, 10_000)      # Min $10K
    prediction = min(prediction, 500_000_000_000)  # Max $500B
    
    return prediction 