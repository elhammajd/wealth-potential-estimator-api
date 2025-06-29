#!/usr/bin/env python3
"""
Metric Calibration for Wealth Estimation
Fits regression models to map cosine similarities to net worth scale
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import math
import sys
import os
import joblib
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.data import WealthyProfileDB
from app.model import EmbeddingModel


class WealthCalibrator:
    """
    Calibrates cosine similarity scores to net worth predictions
    using regression models trained on the mock profile database.
    """
    
    def __init__(self):
        self.linear_model = LinearRegression()
        self.isotonic_model = IsotonicRegression(out_of_bounds='clip')
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def generate_calibration_data(self, profile_db, embedding_model, n_samples=1000):
        """
        Generate training data by computing all pairwise similarities
        between profiles in the database.
        """
        print("Generating calibration data from profile database...")
        
        profiles = profile_db.profiles
        embeddings = profile_db.embeddings
        
        similarities = []
        net_worths = []
        
        # Use all pairwise similarities as training data
        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                # Compute cosine similarity
                sim = np.dot(embeddings[i], embeddings[j])
                similarities.append(sim)
                
                # Use geometric mean of net worths as target
                # This captures the "wealth level" of the pair
                net_worth_target = math.sqrt(profiles[i]['net_worth'] * profiles[j]['net_worth'])
                net_worths.append(net_worth_target)
        
        # Add some self-similarities (perfect matches)
        for i in range(0, len(profiles), 5):  # Sample every 5th profile
            similarities.append(1.0)  # Perfect similarity
            net_worths.append(profiles[i]['net_worth'])
        
        similarities = np.array(similarities)
        net_worths = np.array(net_worths)
        
        print(f"Generated {len(similarities)} calibration samples")
        print(f"Similarity range: {similarities.min():.3f} to {similarities.max():.3f}")
        print(f"Net worth range: ${net_worths.min():,.0f} to ${net_worths.max():,.0f}")
        
        return similarities, net_worths
    
    def fit(self, similarities, net_worths):
        """
        Fit both linear and isotonic regression models.
        """
        print("Fitting calibration models...")
        
        # Reshape for sklearn
        X = similarities.reshape(-1, 1)
        y = np.log10(net_worths + 1)  # Log transform for better regression
        
        # Fit linear regression
        self.linear_model.fit(X, y)
        
        # Fit isotonic regression (preserves monotonicity)
        self.isotonic_model.fit(similarities, y)
        
        # Evaluate models
        linear_pred = self.linear_model.predict(X)
        isotonic_pred = self.isotonic_model.predict(similarities)
        
        linear_r2 = r2_score(y, linear_pred)
        isotonic_r2 = r2_score(y, isotonic_pred)
        
        print(f"Linear Regression R²: {linear_r2:.3f}")
        print(f"Isotonic Regression R²: {isotonic_r2:.3f}")
        
        self.is_fitted = True
        return linear_r2, isotonic_r2
    
    def predict_linear(self, similarities):
        """Predict net worth using linear regression."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        X = np.array(similarities).reshape(-1, 1)
        log_pred = self.linear_model.predict(X)
        return 10**log_pred - 1  # Inverse log transform
    
    def predict_isotonic(self, similarities):
        """Predict net worth using isotonic regression."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        log_pred = self.isotonic_model.predict(similarities)
        return 10**log_pred - 1  # Inverse log transform
    
    def predict_ensemble(self, similarities, weights=(0.3, 0.7)):
        """Ensemble prediction combining both models."""
        linear_pred = self.predict_linear(similarities)
        isotonic_pred = self.predict_isotonic(similarities)
        
        return weights[0] * linear_pred + weights[1] * isotonic_pred
    
    def save(self, filepath):
        """Save the fitted calibrator using joblib."""
        # Save as dictionary for better compatibility
        data = {
            'linear_model': self.linear_model,
            'isotonic_model': self.isotonic_model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted
        }
        joblib.dump(data, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load a fitted calibrator using joblib."""
        data = joblib.load(filepath)
        calibrator = cls()
        calibrator.linear_model = data['linear_model']
        calibrator.isotonic_model = data['isotonic_model']
        calibrator.scaler = data['scaler']
        calibrator.is_fitted = data['is_fitted']
        return calibrator
    
    def plot_calibration(self, similarities, net_worths, save_path=None):
        """Plot calibration curves."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Sort by similarity for plotting
        sorted_indices = np.argsort(similarities)
        sim_sorted = similarities[sorted_indices]
        nw_sorted = net_worths[sorted_indices]
        
        # Predictions
        linear_pred = self.predict_linear(sim_sorted)
        isotonic_pred = self.predict_isotonic(sim_sorted)
        ensemble_pred = self.predict_ensemble(sim_sorted)
        
        plt.figure(figsize=(12, 8))
        
        # Scatter plot of actual data
        plt.scatter(sim_sorted, nw_sorted, alpha=0.5, s=20, label='Training Data')
        
        # Plot regression lines
        plt.plot(sim_sorted, linear_pred, 'r-', linewidth=2, label='Linear Regression')
        plt.plot(sim_sorted, isotonic_pred, 'g-', linewidth=2, label='Isotonic Regression')
        plt.plot(sim_sorted, ensemble_pred, 'b-', linewidth=2, label='Ensemble')
        
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Net Worth ($)')
        plt.title('Wealth Calibration: Similarity → Net Worth Mapping')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Train and evaluate the wealth calibrator."""
    print("=== Wealth Estimation Calibration ===\n")
    
    # Load database and model
    print("1. Loading profile database and embedding model...")
    profile_db = WealthyProfileDB()
    embedding_model = EmbeddingModel()
    
    # Generate calibration data
    print("\n2. Generating calibration training data...")
    calibrator = WealthCalibrator()
    similarities, net_worths = calibrator.generate_calibration_data(profile_db, embedding_model)
    
    # Fit calibration models
    print("\n3. Fitting calibration models...")
    linear_r2, isotonic_r2 = calibrator.fit(similarities, net_worths)
    
    # Save the calibrator
    calibrator.save('wealth_calibrator.pkl')
    print("Calibrator saved to 'wealth_calibrator.pkl'")
    
    # Test with some example similarities
    print("\n4. Testing calibrated predictions...")
    test_similarities = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 0.95]
    
    print("Similarity -> Predicted Net Worth:")
    for sim in test_similarities:
        linear_pred = calibrator.predict_linear([sim])[0]
        isotonic_pred = calibrator.predict_isotonic([sim])[0]
        ensemble_pred = calibrator.predict_ensemble([sim])[0]
        
        print(f"  {sim:.2f} -> Linear: ${linear_pred:,.0f}, "
              f"Isotonic: ${isotonic_pred:,.0f}, "
              f"Ensemble: ${ensemble_pred:,.0f}")
    
    # Plot calibration curves
    print("\n5. Generating calibration plot...")
    try:
        calibrator.plot_calibration(similarities, net_worths, 'calibration_curves.png')
        print("Calibration plot saved as 'calibration_curves.png'")
    except Exception as e:
        print(f"Could not generate plot: {e}")
    
    print("\n✅ Calibration complete!")
    print(f"Linear R²: {linear_r2:.3f}")
    print(f"Isotonic R²: {isotonic_r2:.3f}")
    print("Use the ensemble model for best results.")


if __name__ == "__main__":
    main()
