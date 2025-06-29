#!/usr/bin/env python3
"""
Comprehensive Evaluation Framework for Wealth Potential Estimator API
Implements regression and retrieval metrics for model performance assessment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.data import WealthyProfileDB
from app.model import EmbeddingModel
from app.calibrator import WealthCalibrator
import joblib
from typing import List, Tuple, Dict, Any
import json


class RegressionMetrics:
    """Metrics for net worth prediction evaluation"""
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error in dollars"""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error in dollars"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        # Avoid division by zero
        y_true_safe = np.where(y_true == 0, 1, y_true)
        return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    
    @staticmethod
    def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared coefficient"""
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Median Absolute Error (robust to outliers)"""
        return np.median(np.abs(y_true - y_pred))
    
    @staticmethod
    def wealth_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Accuracy of wealth class prediction (log10 bins)"""
        def wealth_to_class(wealth):
            """Convert wealth to discrete class"""
            if wealth < 1000:
                return 0  # Very low
            elif wealth < 50000:
                return 1  # Low
            elif wealth < 500000:
                return 2  # Middle
            elif wealth < 5000000:
                return 3  # Upper-middle
            elif wealth < 100000000:
                return 4  # Wealthy
            else:
                return 5  # Ultra-wealthy
        
        true_classes = [wealth_to_class(w) for w in y_true]
        pred_classes = [wealth_to_class(w) for w in y_pred]
        
        return np.mean(np.array(true_classes) == np.array(pred_classes))


class RetrievalMetrics:
    """Metrics for top-k similar profile retrieval evaluation"""
    
    @staticmethod
    def hit_at_k(true_relevant: List[List[int]], predicted_top_k: List[List[int]], k: int = 3) -> float:
        """
        Hit@K: Fraction of queries where at least one relevant item appears in top-k
        
        Args:
            true_relevant: List of lists, each containing indices of truly relevant items
            predicted_top_k: List of lists, each containing top-k predicted item indices
            k: Number of top items to consider
        
        Returns:
            Hit@K score (0-1)
        """
        hits = 0
        total = len(true_relevant)
        
        for true_set, pred_list in zip(true_relevant, predicted_top_k):
            if any(item in true_set for item in pred_list[:k]):
                hits += 1
                
        return hits / total if total > 0 else 0.0
    
    @staticmethod
    def recall_at_k(true_relevant: List[List[int]], predicted_top_k: List[List[int]], k: int = 3) -> float:
        """
        Recall@K: Average fraction of relevant items found in top-k
        
        Args:
            true_relevant: List of lists, each containing indices of truly relevant items
            predicted_top_k: List of lists, each containing top-k predicted item indices
            k: Number of top items to consider
        
        Returns:
            Recall@K score (0-1)
        """
        total_recall = 0
        total_queries = len(true_relevant)
        
        for true_set, pred_list in zip(true_relevant, predicted_top_k):
            if len(true_set) == 0:
                continue
                
            relevant_found = len(set(true_set) & set(pred_list[:k]))
            recall = relevant_found / len(true_set)
            total_recall += recall
            
        return total_recall / total_queries if total_queries > 0 else 0.0
    
    @staticmethod
    def precision_at_k(true_relevant: List[List[int]], predicted_top_k: List[List[int]], k: int = 3) -> float:
        """
        Precision@K: Average fraction of retrieved items that are relevant
        
        Args:
            true_relevant: List of lists, each containing indices of truly relevant items
            predicted_top_k: List of lists, each containing top-k predicted item indices
            k: Number of top items to consider
        
        Returns:
            Precision@K score (0-1)
        """
        total_precision = 0
        total_queries = len(true_relevant)
        
        for true_set, pred_list in zip(true_relevant, predicted_top_k):
            if len(pred_list) == 0:
                continue
                
            relevant_found = len(set(true_set) & set(pred_list[:k]))
            precision = relevant_found / min(len(pred_list), k)
            total_precision += precision
            
        return total_precision / total_queries if total_queries > 0 else 0.0
    
    @staticmethod
    def average_precision(true_relevant: List[int], predicted_ranking: List[int]) -> float:
        """
        Average Precision for a single query
        
        Args:
            true_relevant: List of relevant item indices
            predicted_ranking: List of all items in predicted ranking order
        
        Returns:
            Average precision score
        """
        if len(true_relevant) == 0:
            return 0.0
            
        score = 0.0
        num_hits = 0.0
        
        for i, item in enumerate(predicted_ranking):
            if item in true_relevant:
                num_hits += 1
                precision_at_i = num_hits / (i + 1)
                score += precision_at_i
                
        return score / len(true_relevant)
    
    @staticmethod
    def mean_average_precision(true_relevant: List[List[int]], predicted_rankings: List[List[int]]) -> float:
        """
        Mean Average Precision across all queries
        
        Args:
            true_relevant: List of lists, each containing indices of truly relevant items
            predicted_rankings: List of lists, each containing all items in ranking order
        
        Returns:
            Mean Average Precision score
        """
        ap_scores = []
        
        for true_set, pred_ranking in zip(true_relevant, predicted_rankings):
            ap = RetrievalMetrics.average_precision(true_set, pred_ranking)
            ap_scores.append(ap)
            
        return np.mean(ap_scores) if ap_scores else 0.0
    
    @staticmethod
    def ndcg_at_k(true_relevant: List[int], predicted_ranking: List[int], k: int = 3) -> float:
        """
        Normalized Discounted Cumulative Gain at K
        
        Args:
            true_relevant: List of relevant item indices
            predicted_ranking: List of items in predicted ranking order
            k: Number of top items to consider
        
        Returns:
            NDCG@K score
        """
        def dcg_at_k(relevance_scores, k):
            """Calculate DCG@K"""
            relevance_scores = np.array(relevance_scores)[:k]
            if relevance_scores.size:
                return np.sum(relevance_scores / np.log2(np.arange(2, relevance_scores.size + 2)))
            return 0.0
        
        # Calculate relevance scores (1 if relevant, 0 if not)
        predicted_relevance = [1 if item in true_relevant else 0 for item in predicted_ranking[:k]]
        
        # Calculate ideal ranking (all relevant items first)
        ideal_relevance = [1] * len(true_relevant) + [0] * (k - len(true_relevant))
        ideal_relevance = ideal_relevance[:k]
        
        # Calculate DCG and IDCG
        dcg = dcg_at_k(predicted_relevance, k)
        idcg = dcg_at_k(ideal_relevance, k)
        
        return dcg / idcg if idcg > 0 else 0.0


class WealthEstimatorEvaluator:
    """Complete evaluation framework for the Wealth Estimator API"""
    
    def __init__(self, profile_db: WealthyProfileDB, embedding_model: EmbeddingModel, calibrator: WealthCalibrator):
        self.profile_db = profile_db
        self.embedding_model = embedding_model
        self.calibrator = calibrator
        self.regression_metrics = RegressionMetrics()
        self.retrieval_metrics = RetrievalMetrics()
    
    def generate_ground_truth_relevance(self, similarity_threshold: float = 0.7) -> Dict[int, List[int]]:
        """
        Generate ground truth relevance based on similarity threshold
        
        Args:
            similarity_threshold: Minimum similarity to be considered relevant
        
        Returns:
            Dictionary mapping profile index to list of relevant profile indices
        """
        embeddings = self.profile_db.embeddings
        profiles = self.profile_db.profiles
        ground_truth = {}
        
        for i, emb_i in enumerate(embeddings):
            relevant_indices = []
            
            for j, emb_j in enumerate(embeddings):
                if i != j:
                    # Calculate cosine similarity
                    similarity = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
                    
                    # Also consider same wealth class as relevant
                    same_wealth_class = (
                        abs(np.log10(profiles[i]["net_worth"] + 1) - np.log10(profiles[j]["net_worth"] + 1)) < 0.5
                    )
                    
                    if similarity > similarity_threshold or same_wealth_class:
                        relevant_indices.append(j)
            
            ground_truth[i] = relevant_indices
        
        return ground_truth
    
    def evaluate_regression_performance(self, n_splits: int = 5) -> Dict[str, float]:
        """
        Evaluate regression performance using cross-validation
        
        Args:
            n_splits: Number of cross-validation splits
        
        Returns:
            Dictionary of regression metrics
        """
        # Generate training data
        similarities = []
        net_worths = []
        
        embeddings = self.profile_db.embeddings
        profiles = self.profile_db.profiles
        
        for i in range(len(profiles)):
            for j in range(i + 1, len(profiles)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                wealth = (profiles[i]["net_worth"] + profiles[j]["net_worth"]) / 2
                similarities.append(sim)
                net_worths.append(wealth)
        
        similarities = np.array(similarities)
        net_worths = np.array(net_worths)
        
        # Cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        mae_scores = []
        rmse_scores = []
        mape_scores = []
        r2_scores = []
        medae_scores = []
        class_acc_scores = []
        
        for train_idx, test_idx in kf.split(similarities):
            X_train, X_test = similarities[train_idx], similarities[test_idx]
            y_train, y_test = net_worths[train_idx], net_worths[test_idx]
            
            # Train a simple calibrator for this fold
            temp_calibrator = WealthCalibrator()
            
            # Initialize models
            from sklearn.linear_model import LinearRegression
            from sklearn.isotonic import IsotonicRegression
            temp_calibrator.linear_model = LinearRegression()
            temp_calibrator.isotonic_model = IsotonicRegression(out_of_bounds='clip')
            
            # Fit models
            temp_calibrator.linear_model.fit(X_train.reshape(-1, 1), y_train)
            temp_calibrator.isotonic_model.fit(X_train, y_train)
            temp_calibrator.is_trained = True
            
            # Predict
            y_pred = []
            for sim in X_test:
                pred = temp_calibrator.predict_wealth(sim)
                y_pred.append(pred)
            
            y_pred = np.array(y_pred)
            
            # Calculate metrics
            mae_scores.append(self.regression_metrics.mean_absolute_error(y_test, y_pred))
            rmse_scores.append(self.regression_metrics.root_mean_squared_error(y_test, y_pred))
            mape_scores.append(self.regression_metrics.mean_absolute_percentage_error(y_test, y_pred))
            r2_scores.append(self.regression_metrics.r_squared(y_test, y_pred))
            medae_scores.append(self.regression_metrics.median_absolute_error(y_test, y_pred))
            class_acc_scores.append(self.regression_metrics.wealth_class_accuracy(y_test, y_pred))
        
        return {
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'mape_mean': np.mean(mape_scores),
            'mape_std': np.std(mape_scores),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'median_ae_mean': np.mean(medae_scores),
            'median_ae_std': np.std(medae_scores),
            'wealth_class_acc_mean': np.mean(class_acc_scores),
            'wealth_class_acc_std': np.std(class_acc_scores)
        }
    
    def evaluate_retrieval_performance(self, k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Dict[int, float]]:
        """
        Evaluate retrieval performance for top-k similar profiles
        
        Args:
            k_values: List of k values to evaluate
        
        Returns:
            Dictionary of retrieval metrics for each k value
        """
        # Generate ground truth relevance
        ground_truth = self.generate_ground_truth_relevance()
        
        # Get predictions for all profiles
        embeddings = self.profile_db.embeddings
        
        true_relevant_lists = []
        predicted_rankings = []
        
        for query_idx in range(len(embeddings)):
            query_embedding = embeddings[query_idx]
            
            # Calculate similarities with all other profiles
            similarities = []
            for target_idx, target_embedding in enumerate(embeddings):
                if query_idx != target_idx:
                    sim = np.dot(query_embedding, target_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(target_embedding)
                    )
                    similarities.append((target_idx, sim))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            predicted_ranking = [idx for idx, _ in similarities]
            
            true_relevant_lists.append(ground_truth.get(query_idx, []))
            predicted_rankings.append(predicted_ranking)
        
        # Calculate metrics for different k values
        results = {}
        
        for k in k_values:
            predicted_top_k = [ranking[:k] for ranking in predicted_rankings]
            
            hit_k = self.retrieval_metrics.hit_at_k(true_relevant_lists, predicted_top_k, k)
            recall_k = self.retrieval_metrics.recall_at_k(true_relevant_lists, predicted_top_k, k)
            precision_k = self.retrieval_metrics.precision_at_k(true_relevant_lists, predicted_top_k, k)
            
            # Calculate NDCG@K for each query
            ndcg_scores = []
            for true_rel, pred_rank in zip(true_relevant_lists, predicted_rankings):
                ndcg = self.retrieval_metrics.ndcg_at_k(true_rel, pred_rank, k)
                ndcg_scores.append(ndcg)
            
            results[k] = {
                'hit_at_k': hit_k,
                'recall_at_k': recall_k,
                'precision_at_k': precision_k,
                'ndcg_at_k': np.mean(ndcg_scores)
            }
        
        # Calculate MAP
        map_score = self.retrieval_metrics.mean_average_precision(true_relevant_lists, predicted_rankings)
        results['map'] = map_score
        
        return results
    
    def generate_evaluation_report(self, save_plots: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        
        Args:
            save_plots: Whether to save visualization plots
        
        Returns:
            Complete evaluation results
        """
        print("=== Wealth Estimator API Evaluation Report ===\n")
        
        # Evaluate regression performance
        print("1. Evaluating Regression Performance...")
        regression_results = self.evaluate_regression_performance()
        
        print("Regression Metrics (5-fold CV):")
        print(f"  MAE: ${regression_results['mae_mean']:,.2f} ± ${regression_results['mae_std']:,.2f}")
        print(f"  RMSE: ${regression_results['rmse_mean']:,.2f} ± ${regression_results['rmse_std']:,.2f}")
        print(f"  MAPE: {regression_results['mape_mean']:.2f}% ± {regression_results['mape_std']:.2f}%")
        print(f"  R²: {regression_results['r2_mean']:.4f} ± {regression_results['r2_std']:.4f}")
        print(f"  Median AE: ${regression_results['median_ae_mean']:,.2f} ± ${regression_results['median_ae_std']:,.2f}")
        print(f"  Wealth Class Accuracy: {regression_results['wealth_class_acc_mean']:.4f} ± {regression_results['wealth_class_acc_std']:.4f}")
        
        # Evaluate retrieval performance
        print("\n2. Evaluating Retrieval Performance...")
        retrieval_results = self.evaluate_retrieval_performance()
        
        print("Retrieval Metrics:")
        for k in [1, 3, 5, 10]:
            if k in retrieval_results:
                results_k = retrieval_results[k]
                print(f"  Top-{k}:")
                print(f"    Hit@{k}: {results_k['hit_at_k']:.4f}")
                print(f"    Recall@{k}: {results_k['recall_at_k']:.4f}")
                print(f"    Precision@{k}: {results_k['precision_at_k']:.4f}")
                print(f"    NDCG@{k}: {results_k['ndcg_at_k']:.4f}")
        
        print(f"  MAP: {retrieval_results['map']:.4f}")
        
        # Generate visualizations
        if save_plots:
            self._generate_visualizations(regression_results, retrieval_results)
        
        # Combine results
        full_results = {
            'regression_metrics': regression_results,
            'retrieval_metrics': retrieval_results,                'dataset_stats': {
                    'num_profiles': len(self.profile_db.profiles),
                    'wealth_range': {
                        'min': min(p["net_worth"] for p in self.profile_db.profiles),
                        'max': max(p["net_worth"] for p in self.profile_db.profiles),
                        'mean': np.mean([p["net_worth"] for p in self.profile_db.profiles]),
                        'median': np.median([p["net_worth"] for p in self.profile_db.profiles])
                    }
                }
        }
        
        # Save results to JSON
        with open('evaluation_results.json', 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        
        print(f"\n3. Results saved to 'evaluation_results.json'")
        if save_plots:
            print("   Visualizations saved as PNG files")
        
        return full_results
    
    def _generate_visualizations(self, regression_results: Dict, retrieval_results: Dict):
        """Generate and save evaluation visualizations"""
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Regression metrics visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Regression Performance Metrics', fontsize=16, fontweight='bold')
        
        metrics = ['mae', 'rmse', 'mape', 'r2', 'median_ae', 'wealth_class_acc']
        metric_names = ['MAE ($)', 'RMSE ($)', 'MAPE (%)', 'R²', 'Median AE ($)', 'Wealth Class Acc']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i // 3, i % 3]
            mean_val = regression_results[f'{metric}_mean']
            std_val = regression_results[f'{metric}_std']
            
            ax.bar(['Mean'], [mean_val], yerr=[std_val], capsize=5, alpha=0.7)
            ax.set_title(name, fontweight='bold')
            ax.set_ylabel(name)
            
            # Format y-axis for monetary values
            if metric in ['mae', 'rmse', 'median_ae']:
                from matplotlib.ticker import FuncFormatter
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig('regression_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Retrieval metrics visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Retrieval Performance Metrics', fontsize=16, fontweight='bold')
        
        k_values = [1, 3, 5, 10]
        metrics = ['hit_at_k', 'recall_at_k', 'precision_at_k', 'ndcg_at_k']
        metric_names = ['Hit@K', 'Recall@K', 'Precision@K', 'NDCG@K']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i // 2, i % 2]
            
            values = [retrieval_results[k][metric] for k in k_values if k in retrieval_results]
            
            ax.plot(k_values[:len(values)], values, marker='o', linewidth=2, markersize=8)
            ax.set_title(name, fontweight='bold')
            ax.set_xlabel('K (Top-K)')
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(k_values[:len(values)])
        
        plt.tight_layout()
        plt.savefig('retrieval_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Wealth distribution visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Profile Database Analysis', fontsize=16, fontweight='bold')
        
        # Wealth distribution histogram
        net_worths = [p["net_worth"] for p in self.profile_db.profiles]
        ax1.hist(np.log10(np.array(net_worths) + 1), bins=20, alpha=0.7, edgecolor='black')
        ax1.set_title('Wealth Distribution (Log Scale)', fontweight='bold')
        ax1.set_xlabel('Log10(Net Worth + 1)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Wealth class distribution
        wealth_classes = []
        for worth in net_worths:
            if worth < 50000:
                wealth_classes.append('Low')
            elif worth < 500000:
                wealth_classes.append('Middle')
            elif worth < 5000000:
                wealth_classes.append('Upper-Middle')
            elif worth < 100000000:
                wealth_classes.append('Wealthy')
            else:
                wealth_classes.append('Ultra-Wealthy')
        
        class_counts = pd.Series(wealth_classes).value_counts()
        ax2.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Wealth Class Distribution', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Run comprehensive evaluation"""
    print("Loading models and data...")
    
    # Load components
    profile_db = WealthyProfileDB()
    embedding_model = EmbeddingModel()
    
    # Load calibrator
    calibrator = WealthCalibrator()
    calibrator_path = 'app/wealth_calibrator.pkl'
    if os.path.exists(calibrator_path):
        calibrator.load_from_pickle(calibrator_path)
    
    # Create evaluator
    evaluator = WealthEstimatorEvaluator(profile_db, embedding_model, calibrator)
    
    # Run evaluation
    results = evaluator.generate_evaluation_report(save_plots=True)
    
    print("\n=== Evaluation Complete ===")
    print("Check the following files for detailed results:")
    print("- evaluation_results.json: Complete numerical results")
    print("- regression_metrics.png: Regression performance visualization")
    print("- retrieval_metrics.png: Retrieval performance visualization")
    print("- dataset_analysis.png: Dataset statistics and distribution")


if __name__ == "__main__":
    main()
