# Design Decisions

This document captures the key rationale behind our component choices and metrics for the Wealth Potential Estimator API.

## 1. Face Embedding Model

**Chosen**: ArcFace (InsightFace ResNet50-ArcFace)

**Rationale**:

* **Angular Margin Loss** ensures embeddings of the same identity cluster tightly and different identities separate by a clear angular margin.
* **Pre-trained on Large-Scale Face Data** (tens of millions of images), providing state-of-the-art discriminative power without additional training.
* **512-Dimensional Outputs** strike a balance between representational richness and computational cost.

## 2. Similarity Metric

**Chosen**: Cosine Similarity

**Rationale**:

* **Scale Invariance**: Ignores embedding magnitude variations from lighting or cropping changes, focusing purely on direction.
* **Bounded Range**: Scores ∈ \[–1, 1], allowing consistent thresholding and interpretation.
* **Computational Efficiency**: Single dot-product and norm operations, optimized in modern BLAS libraries and hardware.

## 3. Score-to-Worth Calibration

**Chosen**: Ensemble of Linear Regression and Isotonic Regression

**Rationale**:

* **Linear Regression** offers an interpretable baseline mapping from cosine similarity to net-worth values.
* **Isotonic Regression** enforces monotonicity (higher similarity → equal or higher wealth) while flexibly adapting to dataset non-linearities.
* **Ensemble** (such as 30% linear, 70% isotonic) combines stability and flexibility, reducing reliance on arbitrary heuristics.

## 4. Evaluation Metrics

| Metric          | Type       | Purpose                                                     |
| --------------- | ---------- | ----------------------------------------------------------- |
| **MAE**         | Regression | Mean absolute deviation in estimated dollars                |
| **RMSE**        | Regression | Penalizes large errors, highlights worst-case misses        |
| **R²**          | Regression | Proportion of variance explained by the calibration model   |
| **Accuracy\@3** | Retrieval  | Fraction of queries with the true relevant profile in top-3 |
| **Recall\@3**   | Retrieval  | Fraction of all relevant profiles retrieved in top-3        |
| **NDCG\@3**     | Retrieval  | Position-aware ranking quality for top-3                    |

**Rationale**:

* Regression metrics quantify dollar-scale prediction quality and sensitivity to outliers.
* Retrieval metrics measure the API’s ability to surface the most similar profiles—critical to verifying embedding quality.

## 5. Placement

* Include this document alongside **ARCHITECTURE.md** in the project root.
* Reference it from **README.md** under a **Design Decisions** section.

```markdown
## Design Decisions

For detailed rationale behind model, metric, and architecture choices, see [DESIGN_DECISIONS.md](./DESIGN_DECISIONS.md).
```

---

By separating design rationale from system architecture, we maintain clarity in both documentation and code structure while ensuring evaluators can easily understand our engineering choices.
