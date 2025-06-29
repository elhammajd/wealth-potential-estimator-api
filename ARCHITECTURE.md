# ARCHITECTURE.md

### Project Overview

The Wealth Potential Estimator API is a machine-learning service that estimates a person’s potential net worth from a selfie. The pipeline:

1. Accepts a selfie via a `/predict` endpoint.
2. Validates and preprocesses the image.
3. Extracts a 512-dimensional face embedding using a pre-trained ArcFace model.
4. Computes cosine similarity against a mock database of wealthy profiles.
5. Applies a calibrated mapping from similarity scores to net-worth estimates.
6. Returns the final estimate and top-3 matching profiles in JSON.

---

### Architecture Diagram

```text
Client (Web/Mobile)
      │
      ▼
FastAPI Web Server (/predict)
  • Input validation  
  • Image loading      
      │
      ▼
Image Preprocessing  
  • Crop + align face  
  • Resize & normalize
      │
      ▼
ArcFace Embedding
  • InsightFace ONNX
  • 112×112 RGB input
  • 512-D L2-normalized output
      │
      ▼
Similarity & Calibration
  • Cosine similarity
  • Linear + Isotonic regression
      │
      ▼
Aggregation & Response
  • Distance-weighted average
  • JSON `{ estimated_net_worth, top_matches }`
```

---

### Core Components

* **`app/main.py`**: FastAPI app with `/predict` and `/health` endpoints, file handling, and error responses.
* **`app/model_arcface.py`**: Loads ArcFace ONNX model, detects and aligns faces, extracts embeddings.
* **`app/data.py`**: Mock profiles database (120 synthetic embeddings + net-worth labels).
* **`app/calibrator.py`**: Implements linear and isotonic regression, and ensemble prediction logic.
* **`app/utils.py`**: Validation helpers, image processing functions.
* **`tests/`**: pytest suite covering validation, embedding, similarity, and response format.

---

### Data Flow

1. **POST /predict** receives a multipart/form-data image.
2. **Validation**: MIME type, size limit, image integrity, face detection.
3. **Preprocessing**: Crop to face, resize to 112×112, normalize.
4. **Embedding**: ArcFace model → 512-D vector.
5. **Similarity**: Compute cosine scores against mock embeddings.
6. **Calibration**: Map scores via calibrated regressors to net-worth.
7. **Aggregation**: Distance-weighted or simple average of top-3 estimates.
8. **Response**: Return estimate and profile metadata in JSON.

---

### API Endpoints

* **POST /predict**
  **Input**: `file` field (JPEG/PNG selfie).
  **Output**: JSON with:

  ```json
  {
    "estimated_net_worth": 1.23e8,
    "top_matches": [
      {"name":"Alice","similarity":0.85,"net_worth":1.5e7},
      {"name":"Bob","similarity":0.82,"net_worth":8.5e6},
      {"name":"Carol","similarity":0.81,"net_worth":1.2e7}
    ]
  }
  ```

* **GET /health**
  **Output**: `{ "status": "healthy" }`

---

### Evaluation Metrics

* **Regression Metrics** (net-worth): MAE, RMSE, R²
* **Retrieval Metrics** (top-3): Accuracy\@3, Recall\@3, NDCG\@3

---

### Deployment

* **Docker**: Multi-stage build, Alpine/Python slim base, ONNX runtime.
* **Platforms**: Railway (Git-based), Fly.io, Render or Kubernetes.
* **Optimizations**: Bind UVicorn to `0.0.0.0`, health probe, non-root user.

---

### Next Steps

* Ensemble multiple face models for robust embeddings.
* Experiment with variable K (3,5,7) and distance-weighted aggregation.
* Integrate outfit/background embeddings for multi-modal fusion.
* Implement outlier detection on embeddings for low-confidence cases.
* Add authentication, rate-limiting, logging, and monitoring.

---

## Evaluation Framework

### Comprehensive Metrics

The system includes a robust evaluation framework that assesses both regression and retrieval performance:

#### Regression Metrics (Wealth Prediction)

* **MAE (Mean Absolute Error)**: Average dollar error in predictions
* **RMSE (Root Mean Squared Error)**: Penalizes large prediction errors
* **MAPE (Mean Absolute Percentage Error)**: Relative prediction accuracy
* **R² (Coefficient of Determination)**: Explained variance in wealth
* **Median AE**: Robust error metric (less sensitive to outliers)
* **Wealth Class Accuracy**: Accuracy in predicting wealth categories

#### Retrieval Metrics (Similar Profile Matching)

* **Hit\@K**: Fraction of queries with at least one relevant result in top-K
* **Recall\@K**: Average fraction of relevant items found in top-K
* **NDCG\@K**: Normalized Discounted Cumulative Gain (ranking quality)
* **MAP**: Mean Average Precision across all queries

### Running Evaluations

```bash
# Full evaluation with visualizations (generates PNG plots)
python evaluation.py

# Quick evaluation (faster, 3-fold CV)
python test_evaluation.py

# Basic functionality test
python simple_test.py
```

### Sample Evaluation Results

```
=== Wealth Estimator API Evaluation Report ===

Regression Metrics (5-fold CV):
  MAE: $28,533,467,114.75 ± $607,857,946.93
  RMSE: $37,997,834,829.36 ± $1,264,254,669.74
  MAPE: 16898650.32% ± 631537.18%
  R²: 0.0795 ± 0.0197
  Median AE: $18,202,048,183.60 ± $314,969,511.18
  Wealth Class Accuracy: 0.4022 ± 0.0129

Retrieval Metrics:
  Top-1: Hit@1: 1.0000, Recall@1: 0.0180, NDCG@1: 1.0000
  Top-3: Hit@3: 1.0000, Recall@3: 0.0539, NDCG@3: 1.0000
  Top-5: Hit@5: 1.0000, Recall@5: 0.0899, NDCG@5: 1.0000
  Top-10: Hit@10: 1.0000, Recall@10: 0.1765, NDCG@10: 0.9933
  
```
