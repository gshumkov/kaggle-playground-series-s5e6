"""
Flask API for Fertilizer Recommendation
"""
from flask import Flask, request, jsonify
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(__file__))
from predict import FertilizerPredictor

app = Flask(__name__)

# Initialize predictor
predictor = None

@app.before_request
def initialize():
    """Initialize predictor before first request"""
    global predictor
    if predictor is None:
        try:
            predictor = FertilizerPredictor(model_path='model')
        except Exception as e:
            print(f"Error initializing predictor: {e}")

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Fertilizer Recommendation API',
        'version': '1.0',
        'endpoints': {
            'GET /': 'This help message',
            'GET /health': 'Health check',
            'POST /predict': 'Predict fertilizer',
            'GET /model-info': 'Get model information'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': predictor is not None})

@app.route('/model-info')
def model_info():
    """Get model information"""
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify(predictor.model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict fertilizer recommendation
    
    Expected JSON input:
    {
        "temperature": 25.0,
        "humidity": 60.0,
        "moisture": 50.0,
        "soil_type": "Loamy",
        "crop_type": "Wheat",
        "nitrogen": 40.0,
        "potassium": 10.0,
        "phosphorous": 15.0
    }
    """
    if predictor is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['temperature', 'humidity', 'moisture', 'soil_type', 
                          'crop_type', 'nitrogen', 'potassium', 'phosphorous']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Make prediction
        fertilizer = predictor.predict_single(
            temperature=float(data['temperature']),
            humidity=float(data['humidity']),
            moisture=float(data['moisture']),
            soil_type=data['soil_type'],
            crop_type=data['crop_type'],
            nitrogen=float(data['nitrogen']),
            potassium=float(data['potassium']),
            phosphorous=float(data['phosphorous'])
        )
        
        # Get probabilities
        probabilities = predictor.get_prediction_probabilities(
            temperature=float(data['temperature']),
            humidity=float(data['humidity']),
            moisture=float(data['moisture']),
            soil_type=data['soil_type'],
            crop_type=data['crop_type'],
            nitrogen=float(data['nitrogen']),
            potassium=float(data['potassium']),
            phosphorous=float(data['phosphorous'])
        )
        
        return jsonify({
            'recommended_fertilizer': fertilizer,
            'probabilities': probabilities,
            'input': data
        })
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input value: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
