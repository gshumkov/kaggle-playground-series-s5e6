# Kaggle Playground Series S5E6 - Fertilizer Recommendation

A machine learning project that predicts the optimal fertilizer based on environmental conditions, soil properties, and crop requirements.

## Project Overview

This project develops a predictive model to recommend appropriate fertilizers based on:
- **Environmental factors**: Temperature, Humidity, Moisture
- **Soil properties**: Soil Type
- **Crop requirements**: Crop Type
- **Nutrient levels**: Nitrogen, Potassium, Phosphorous

## Project Structure

```
.
├── data/                          # Dataset directory
│   ├── train.csv                  # Training data
│   └── test.csv                   # Test data
├── notebooks/                     # Jupyter notebooks
│   └── eda_fertilizer.ipynb      # Exploratory Data Analysis
├── src/                           # Source code
│   ├── train_model.py            # Model training script
│   ├── predict.py                # Prediction service
│   └── app.py                    # Flask API
├── model/                         # Trained model artifacts (generated)
│   ├── fertilizer_model.pkl
│   ├── soil_encoder.pkl
│   ├── crop_encoder.pkl
│   ├── fertilizer_encoder.pkl
│   └── model_info.json
├── Dockerfile                     # Docker configuration
├── .dockerignore                  # Docker ignore file
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Features

### Input Features
- **Temperature** (°C): Environmental temperature
- **Humidity** (%): Air humidity percentage
- **Moisture** (%): Soil moisture level
- **Soil Type**: Sandy, Loamy, Black, Red, Clayey
- **Crop Type**: Wheat, Rice, Maize, Cotton, Sugarcane, Barley, Soybean, Groundnut, Pulses, Tobacco
- **Nitrogen**: Nitrogen content in soil
- **Potassium**: Potassium content in soil
- **Phosphorous**: Phosphorous content in soil

### Target Variable
- **Fertilizer Name**: Recommended fertilizer type
  - Urea
  - DAP
  - NPK 10-26-26
  - NPK 20-20-20
  - Ammonium Sulphate
  - Potassium Chloride
  - SSP (Single Super Phosphate)

## Getting Started

### Prerequisites
- Python 3.12+
- Docker (for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/gshumkov/kaggle-playground-series-s5e6.git
cd kaggle-playground-series-s5e6
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Generate training data (if not available):
```bash
python generate_data.py
```

## Usage

### 1. Exploratory Data Analysis

Open and run the Jupyter notebook for comprehensive EDA:
```bash
jupyter notebook notebooks/eda_fertilizer.ipynb
```

The notebook includes:
- Data loading and overview
- Statistical analysis
- Feature distributions
- Correlation analysis
- Visualizations of relationships between features and target
- Key insights and patterns

### 2. Train the Model

Train the machine learning model:
```bash
python src/train_model.py
```

This will:
- Load and preprocess the training data
- Train a Random Forest classifier
- Evaluate model performance
- Save the trained model and encoders to `model/` directory

### 3. Make Predictions

Use the prediction script:
```bash
python src/predict.py
```

Or import and use in your code:
```python
from src.predict import FertilizerPredictor

# Initialize predictor
predictor = FertilizerPredictor()

# Make prediction
fertilizer = predictor.predict_single(
    temperature=25.0,
    humidity=60.0,
    moisture=50.0,
    soil_type="Loamy",
    crop_type="Wheat",
    nitrogen=40.0,
    potassium=10.0,
    phosphorous=15.0
)

print(f"Recommended Fertilizer: {fertilizer}")
```

### 4. Run the Flask API

Start the API server:
```bash
python src/app.py
```

The API will be available at `http://localhost:5000`

#### API Endpoints

- **GET /** - API information
- **GET /health** - Health check
- **GET /model-info** - Model information
- **POST /predict** - Make predictions

Example prediction request:
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

## Docker Deployment

### Build the Docker Image

```bash
docker build -t fertilizer-recommender .
```

### Run the Docker Container

```bash
docker run -p 5000:5000 fertilizer-recommender
```

The API will be available at `http://localhost:5000`

### Docker Compose (Optional)

Create a `docker-compose.yml`:
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - PYTHONUNBUFFERED=1
```

Run with:
```bash
docker-compose up
```

## Model Details

### Algorithm
- **Random Forest Classifier**
  - 100 trees
  - Max depth: 20
  - Min samples split: 5
  - Min samples leaf: 2

### Performance Metrics
- Accuracy: ~95%+ on validation set
- Multi-class classification with 7 fertilizer types
- Feature importance: NPK values are most important

### Feature Engineering
- Label encoding for categorical variables (Soil Type, Crop Type)
- No scaling required for tree-based models

## Project Insights

Key findings from EDA:
1. Different fertilizers show distinct NPK (Nitrogen-Phosphorous-Potassium) patterns
2. Environmental factors (temperature, humidity) vary by fertilizer type
3. NPK values are the most distinguishing features for classification
4. Dataset is well-balanced across fertilizer types

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Kaggle Playground Series S5E6
- Scikit-learn for machine learning algorithms
- Flask for API framework