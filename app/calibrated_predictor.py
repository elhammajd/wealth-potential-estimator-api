#!/usr/bin/env python3
"""
Updated wealth prediction using calibrated regression models
"""

import numpy as np
import pickle
import os
from typing import List, Tuple, Dict, Any


class CalibratedWealthPredictor:
    """
    Enhanced wealth predictor using calibrated regression models
    to map similarity scores to net worth estimates.
    """
    
    def __init__(self):
        self.calibrator = None
        self.load_calibrator()
    
    def load_calibrator(self):
        """Load the pre-trained calibrator."""
        calibrator_path = 'wealth_calibrator.pkl'
        if os.path.exists(calibrator_path):
            try:
                with open(calibrator_path, 'rb') as f:
                    self.calibrator = pickle.load(f)
                print("✅ Loaded calibrated wealth predictor")
            except Exception as e:
                print(f"⚠️  Could not load calibrator: {e}")
                self.calibrator = None
        else:
            print("⚠️  No calibrator found, using fallback method")
            self.calibrator = None
    
    def predict_wealth_calibrated(self, top_matches: List[Tuple[Dict[str, Any], float]]) -> float:
        """
        Predict wealth using calibrated regression models.
        """
        if not top_matches:
            return 50_000_000  # Default fallback
        
        if self.calibrator is None:
            return self._predict_wealth_fallback(top_matches)
        
        # Extract similarities and net worths
        similarities = [match[1] for match in top_matches]
        net_worths = [match[0]["net_worth"] for match in top_matches]
        
        max_similarity = max(similarities)
        avg_similarity = sum(similarities) / len(similarities)
        
        print(f"Calibrated prediction - max_sim: {max_similarity:.3f}, avg_sim: {avg_similarity:.3f}")
        
        try:
            # Use the ensemble model for best results
            if max_similarity > 0.08:
                # High confidence - use the top similarity score
                predicted_wealth = self.calibrator.predict_ensemble([max_similarity])[0]
                print(f"   High confidence: using max similarity {max_similarity:.3f}")
            elif avg_similarity > 0.05:
                # Medium confidence - use average similarity
                predicted_wealth = self.calibrator.predict_ensemble([avg_similarity])[0]
                print(f"   Medium confidence: using avg similarity {avg_similarity:.3f}")
            else:
                # Low confidence - use weighted average of similarities
                weighted_sim = sum(s * (i + 1) for i, s in enumerate(similarities)) / sum(range(1, len(similarities) + 1))
                predicted_wealth = self.calibrator.predict_ensemble([weighted_sim])[0]
                print(f"   Low confidence: using weighted similarity {weighted_sim:.3f}")
            
            # Apply bounds checking
            predicted_wealth = max(predicted_wealth, 1_000)      # Min $1K
            predicted_wealth = min(predicted_wealth, 500_000_000_000)  # Max $500B
            
            print(f"   Calibrated prediction: ${predicted_wealth:,.0f}")
            
            return float(predicted_wealth)
            
        except Exception as e:
            print(f"❌ Calibration failed: {e}, using fallback")
            return self._predict_wealth_fallback(top_matches)
    
    def _predict_wealth_fallback(self, top_matches: List[Tuple[Dict[str, Any], float]]) -> float:
        """
        Fallback wealth prediction method (original approach).
        """
        print("   Using fallback prediction method")
        
        similarities = [match[1] for match in top_matches]
        net_worths = [match[0]["net_worth"] for match in top_matches]
        
        # Simple weighted average
        if similarities and sum(similarities) > 0:
            weights = [s**2 for s in similarities]  # Square similarities for emphasis
            prediction = sum(w * nw for w, nw in zip(weights, net_worths)) / sum(weights)
        else:
            prediction = sum(net_worths) / len(net_worths)
        
        # Apply bounds
        prediction = max(prediction, 10_000)      # Min $10K
        prediction = min(prediction, 500_000_000_000)  # Max $500B
        
        return prediction


# Create global instance
calibrated_predictor = CalibratedWealthPredictor()


def predict_wealth_calibrated(query_embedding: np.ndarray, top_matches: List[Tuple[Dict[str, Any], float]]) -> float:
    """
    Main function to predict wealth using calibrated models.
    This replaces the original predict_wealth function.
    """
    return calibrated_predictor.predict_wealth_calibrated(top_matches)
