# Wealth Potential Estimator API

A demonstration machine-learning API that estimates a person’s potential net worth from a selfie. Upload a photo to the `/predict` endpoint and receive:

* **Estimated Net Worth**: Calibrated prediction in USD
* **Top-3 Similar Profiles**: Names, similarity scores, and mock net-worths

Built with FastAPI, ArcFace face recognition, statistical calibration, and containerized via Docker. 

---

## Features

* **State-of-the-Art Face Embeddings**: Uses ArcFace (ONNX) for 512-dimensional, L2‑normalized vectors
* **Cosine Similarity Lookup**: Efficient nearest‑neighbor search against 120 synthetic wealthy profiles
* **Calibration Ensemble**: Linear + Isotonic regression models map similarity scores to realistic net‑worth estimates
* **Robust Validation**: MIME/type, file size, image integrity, and face detection checks
* **Fast & Lightweight**: \~200–400 ms per request, \~300 MB memory footprint
* **Dockerized**: Multi‑stage build with Alpine‑slim base and ONNX Runtime

---

## Quick Start

### Live Demo

```bash
# Predict from terminal
curl -X POST "https://wealth-potential-estimator-api-production.up.railway.app/predict" \
     -F "file=@your_photo.jpg"

# Health check
curl "https://wealth-potential-estimator-api-production.up.railway.app/health"
```

### Local Development

```bash
# Clone repository
git clone <repo-url>
cd wealth-potential-estimator-api

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive API docs.

### Docker Deployment

```bash
docker build -t wealth-estimator-api .
docker run -p 8000:8000 wealth-estimator-api
```

---

## Project Structure

```text
wealth-potential-estimator-api/
├── app/
│   ├── main.py           # FastAPI endpoints
│   ├── model_arcface.py  # ArcFace ONNX embedding
│   ├── data.py           # Mock profile database
│   ├── calibrator.py     # Calibration regressors
│   └── utils.py          # Validation & preprocessing
├── tests/                # pytest suite for validation, API, and calibration
├── calibrate_wealth.py   # Script to retrain calibration models
├── Dockerfile            # Multi-stage Docker build
├── ARCHITECTURE.md       # System architecture
├── DESIGN_DECISIONS.md   # Rationale for model and metric choices
├── README.md             # This file
└── requirements.txt
```

---

## Design Decisions

For detailed rationale on model selection, similarity metric, calibration, and evaluation metrics, see [DESIGN\_DECISIONS.md](./DESIGN_DECISIONS.md).

---

## System Architecture

Technical details and data flow are documented in [ARCHITECTURE.md](./ARCHITECTURE.md).

---

## API Reference

### POST /predict

* **Input**: `multipart/form-data` with field `file` (JPEG or PNG)
* **Output**:

  ```json
  {
    "estimated_net_worth": 1234567.89,
    "top_matches": [
      { "name": "Alice",  "similarity": 0.85, "net_worth": 15000000 },
      { "name": "Bob",    "similarity": 0.82, "net_worth":  8500000 },
      { "name": "Carol",  "similarity": 0.81, "net_worth": 12000000 }
    ]
  }
  ```
* **Errors**:

  * `400 Bad Request`: invalid file or no face detected
  * `415 Unsupported Media Type`: wrong MIME type
  * `413 Payload Too Large`: file exceeds size limit
  * `500 Internal Server Error`: unexpected failure

### GET /health

* **Output**:

  ```json
  { "status": "healthy" }
  ```

---

## Testing

Run the pytest suite:

```bash
pytest --disable-warnings -q
```

Sample scripts:

* `python test_api.py` (API endpoint)
* `python debug_calibrator.py` (calibration diagnostics)

---

## Configuration

Environment variables:

* `MAX_UPLOAD_SIZE` (bytes): file size limit
* `WEALTHAPI_PRETRAINED` (0 or 1): enable/disable pretrained ArcFace



