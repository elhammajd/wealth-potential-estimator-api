#!/usr/bin/env python3
"""
Wealth Calibrator for converting similarity scores to net worth predictions
"""

import numpy as np
import pickle
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression


class WealthCalibrator:
    """
    Calibrates cosine similarity scores to net worth predictions
    using regression models trained on the mock profile database.
    """
    
    def __init__(self):
        self.linear_model = None
        self.isotonic_model = None
        self.scaler = None
        self.is_trained = False
    
    def predict_wealth(self, similarity_score):
        """
        Predict net worth from similarity score using ensemble of calibrated models.
        
        Args:
            similarity_score: Cosine similarity score (float)
            
        Returns:
            Predicted net worth (float)
        """
        if not self.is_trained:
            # Fallback to simple heuristic if not trained
            return self._fallback_prediction(similarity_score)
        
        try:
            # Check if models are loaded
            if self.linear_model is None or self.isotonic_model is None:
                print("Warning: Calibration models not loaded, using fallback")
                return self._fallback_prediction(similarity_score)
            
            # Prepare input
            X = np.array([[similarity_score]])
            
            # Get predictions from both models
            linear_pred = self.linear_model.predict(X)[0]
            isotonic_pred = self.isotonic_model.predict(X)[0]
            
            # Ensemble prediction (weighted average)
            # Isotonic gets higher weight due to better R² score
            ensemble_pred = 0.3 * linear_pred + 0.7 * isotonic_pred
            
            # Apply bounds
            ensemble_pred = max(ensemble_pred, 1000)  # Min $1K
            ensemble_pred = min(ensemble_pred, 500_000_000_000)  # Max $500B
            
            return float(ensemble_pred)
            
        except Exception as e:
            print(f"Warning: Calibration prediction failed: {e}")
            return self._fallback_prediction(similarity_score)
    
    def _fallback_prediction(self, similarity_score):
        """
        Simple fallback prediction when calibrated models aren't available.
        """
        # Ensure similarity_score is a valid number
        if similarity_score is None or not isinstance(similarity_score, (int, float)):
            similarity_score = 0.05  # Default fallback
        
        # Convert similarity to wealth using exponential scaling
        base_wealth = 50_000  # Base wealth
        max_wealth = 100_000_000  # Max wealth for fallback
        
        # Exponential mapping - higher similarity = higher wealth
        # Clamp similarity score to reasonable range
        sim_clamped = max(0, min(1, similarity_score))
        wealth = base_wealth * (1 + sim_clamped * 20) ** 3
        
        # Apply bounds
        wealth = max(wealth, 1000)
        wealth = min(wealth, max_wealth)
        
        print(f"Fallback prediction: similarity={similarity_score:.3f} -> wealth=${wealth:,.2f}")
        return float(wealth)
    
    def load_from_pickle(self, filepath):
        """
        Load trained calibrator from pickle file using joblib for better compatibility.
        """
        try:
            if os.path.exists(filepath):
                print(f"Loading calibrator from: {filepath}")
                
                # Try joblib first (more robust for sklearn models)
                try:
                    data = joblib.load(filepath)
                    print("Loaded with joblib successfully")
                except:
                    # Fallback to pickle
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                    print("Loaded with pickle successfully")
                
                # Extract models
                if isinstance(data, dict):
                    self.linear_model = data.get('linear_model')
                    self.isotonic_model = data.get('isotonic_model') 
                    self.scaler = data.get('scaler', None)
                else:
                    # If data is the calibrator object itself
                    self.linear_model = getattr(data, 'linear_model', None)
                    self.isotonic_model = getattr(data, 'isotonic_model', None)
                    self.scaler = getattr(data, 'scaler', None)
                
                # Verify models loaded
                if self.linear_model is not None and self.isotonic_model is not None:
                    self.is_trained = True
                    print("Calibrator models loaded and verified successfully")
                    return True
                else:
                    print("Warning: Models not found in calibrator file")
                    return False
                    
            else:
                print(f"Calibrator file not found: {filepath}")
                return False
                
        except Exception as e:
            print(f"Warning: Could not load calibrator: {e}")
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            return False


def load_calibrator():
    """
    Factory function to load a trained calibrator.
    """
    calibrator = WealthCalibrator()
    
    # Try to load from pickle file
    calibrator_path = os.path.join(os.path.dirname(__file__), 'wealth_calibrator.pkl')
    if calibrator.load_from_pickle(calibrator_path):
        return calibrator
    
    # If loading fails, return uncalibrated version with fallback
    print("Using fallback prediction method")
    return calibrator
