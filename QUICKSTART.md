# Quick Start Guide

## Setup and Run in 3 Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Data and Train Model
```bash
# Generate sample dataset
python generate_data.py

# Train the model
python src/train_model.py
```

### 3. Run the Application

#### Option A: Local Flask API
```bash
python src/app.py
```

#### Option B: Docker Container
```bash
# Build the Docker image
docker build -t fertilizer-recommender .

# Run the container
docker run -p 5000:5000 fertilizer-recommender
```

## Test the API

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 25.0,
    "humidity": 60.0,
    "moisture": 50.0,
    "soil_type": "Loamy",
    "crop_type": "Wheat",
    "nitrogen": 40.0,
    "potassium": 10.0,
    "phosphorous": 15.0
  }'
```

## Explore the Data

```bash
jupyter notebook notebooks/eda_fertilizer.ipynb
```
