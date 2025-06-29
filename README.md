# Wealth Potential Estimator API

A simple web API that tries to guess someone's net worth from their photo. Upload a selfie and get back an estimate plus the 3 most similar wealthy people from our database.

Built with FastAPI and can run anywhere Docker works 

---

## How It Works

```
               +---------------------------+
               |  Database of Rich People  |
               +-------------+-------------+
                             |
                             | (embeddings)
               +-------------v-------------+
               |  FastAPI web server       |
    ┌──────────>  • /predict endpoint      |
    |          |  • ResNet50 vision model  |
    | photo    |  • Similarity matching    |
    | upload   +-------------+-------------+
    |                        |
    | JSON                   | cosine similarity
    | response               |
    |                        v
+---+----+           +--------------+
| You    |           | Pretrained   |
| (web/  |           | ResNet-50    |
|  app)  |           +--------------+
+--------+
```

### What We Used

* **FastAPI** - Easy to use, auto-generates docs
* **ResNet-50** - Pretrained vision model that turns images into 2,048 numbers. **Important**: Must use pretrained weights or everything looks the same!
* **120 People Database** - Mix of billionaires, millionaires, and regular folks across all income levels
* **Cosine Similarity** - Math to find who looks most similar
* **Docker** - Runs the same everywhere

---

## Running It Yourself

### Quick Start

```bash
# Get the code
git clone <repo-url> && cd wealth

# Install stuff
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start it up
uvicorn app.main:app --reload
```

Go to http://127.0.0.1:8000/docs to try it out.

### With Docker

```bash
# Build it
docker build -t wealth-estimator .

# Run it
docker run -p 8000:8000 wealth-estimator
```

---

## Important Settings

| Setting | Default | What It Does |
|---------|---------|--------------|
| `WEALTHAPI_PRETRAINED` | `1` | Use pretrained ResNet weights (you want this!) Set to `0` for random results |

⚠️ **Don't set WEALTHAPI_PRETRAINED=0** unless you want to see what broken looks like. Without pretrained weights, all photos look identical to the model and you get garbage results.

```bash
# Good (default)
export WEALTHAPI_PRETRAINED=1
uvicorn app.main:app --reload

# Bad (don't do this)
export WEALTHAPI_PRETRAINED=0
uvicorn app.main:app --reload
```

---

## How To Use It

```bash
curl -X POST \
     -F "file=@/path/to/your/photo.jpg" \
     http://localhost:8000/predict | jq
```

You get back:

```json
{
  "estimated_net_worth": 123456789.0,
  "top_matches": [
    {"name": "Some Rich Person", "similarity": 0.92},
    {"name": "Another Rich Person", "similarity": 0.88},
    {"name": "Third Rich Person", "similarity": 0.85}
  ]
}
```

---

## Limitations & Disclaimers

* **Fake Dataset**: Our "database" is mostly made up. Real performance would need real photos.
* **Just For Fun**: Guessing wealth from looks is obviously not scientific and probably biased.
* **Demo Only**: This is a proof of concept, not something you'd actually use.
* **Scale Issues**: For thousands of people, you'd want a proper vector database.
* **Security**: No rate limiting, auth, or input validation - don't put this on the internet as-is.

---

## API Reference

### `POST /predict`

Upload a photo, get wealth estimate.

**Input**: Image file (multipart/form-data)

**Output**:
```json
{
  "estimated_net_worth": 1.23e8,
  "top_matches": [
    {"name": "Rich Person", "similarity": 0.95},
    {"name": "Another Rich Person", "similarity": 0.90},
    {"name": "Third Rich Person", "similarity": 0.85}
  ]
}
```

**Errors**: `400` if you upload something that's not an image

---

## License

MIT - do whatever you want with it.

### Environment Variables

- `WEALTHAPI_PRETRAINED` (default: "1")
  - Set to "1" to use pretrained ResNet-50 weights (recommended)
  - Set to "0" to use untrained weights (for testing bias scenarios)

**WARNING:** Don't set WEALTHAPI_PRETRAINED=0 unless you want to see what broken looks like. Without pretrained weights, all photos look identical to the model and you get garbage results. 