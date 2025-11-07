"""
Fertilizer Recommendation Prediction Service
Load trained model and make predictions
"""
import pandas as pd
import pickle
import json
import os

class FertilizerPredictor:
    """Fertilizer prediction service"""
    
    def __init__(self, model_path='model'):
        """Initialize the predictor with trained model and encoders"""
        self.model_path = model_path
        self.model = None
        self.le_soil = None
        self.le_crop = None
        self.le_fertilizer = None
        self.model_info = None
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model and encoders"""
        try:
            # Load model
            with open(f'{self.model_path}/fertilizer_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            # Load encoders
            with open(f'{self.model_path}/soil_encoder.pkl', 'rb') as f:
                self.le_soil = pickle.load(f)
            
            with open(f'{self.model_path}/crop_encoder.pkl', 'rb') as f:
                self.le_crop = pickle.load(f)
            
            with open(f'{self.model_path}/fertilizer_encoder.pkl', 'rb') as f:
                self.le_fertilizer = pickle.load(f)
            
            # Load model info
            with open(f'{self.model_path}/model_info.json', 'r') as f:
                self.model_info = json.load(f)
            
            print("Model loaded successfully!")
            print(f"Available soil types: {self.model_info['soil_types']}")
            print(f"Available crop types: {self.model_info['crop_types']}")
            print(f"Fertilizer types: {self.model_info['fertilizer_types']}")
            
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            print("Please train the model first using train_model.py")
            raise
    
    def predict_single(self, temperature, humidity, moisture, soil_type, 
                      crop_type, nitrogen, potassium, phosphorous):
        """
        Predict fertilizer for a single sample
        
        Parameters:
        -----------
        temperature : float
            Temperature in Celsius
        humidity : float
            Humidity percentage
        moisture : float
            Soil moisture percentage
        soil_type : str
            Type of soil (Sandy, Loamy, Black, Red, Clayey)
        crop_type : str
            Type of crop
        nitrogen : float
            Nitrogen content
        potassium : float
            Potassium content
        phosphorous : float
            Phosphorous content
        
        Returns:
        --------
        str : Recommended fertilizer name
        """
        # Encode categorical variables
        soil_encoded = self.le_soil.transform([soil_type])[0]
        crop_encoded = self.le_crop.transform([crop_type])[0]
        
        # Create feature vector
        features = [[temperature, humidity, moisture, nitrogen, 
                    potassium, phosphorous, soil_encoded, crop_encoded]]
        
        # Predict
        prediction = self.model.predict(features)[0]
        fertilizer = self.le_fertilizer.inverse_transform([prediction])[0]
        
        return fertilizer
    
    def predict_batch(self, df):
        """
        Predict fertilizer for multiple samples
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with columns: Temperature, Humidity, Moisture, 
            Soil Type, Crop Type, Nitrogen, Potassium, Phosphorous
        
        Returns:
        --------
        list : List of recommended fertilizer names
        """
        df = df.copy()
        
        # Encode categorical variables
        df['Soil Type Encoded'] = self.le_soil.transform(df['Soil Type'])
        df['Crop Type Encoded'] = self.le_crop.transform(df['Crop Type'])
        
        # Create feature matrix
        feature_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 
                       'Potassium', 'Phosphorous', 'Soil Type Encoded', 'Crop Type Encoded']
        X = df[feature_cols]
        
        # Predict
        predictions = self.model.predict(X)
        fertilizers = self.le_fertilizer.inverse_transform(predictions)
        
        return fertilizers.tolist()
    
    def get_prediction_probabilities(self, temperature, humidity, moisture, 
                                    soil_type, crop_type, nitrogen, 
                                    potassium, phosphorous):
        """
        Get prediction probabilities for all fertilizer types
        
        Returns:
        --------
        dict : Dictionary with fertilizer names and their probabilities
        """
        # Encode categorical variables
        soil_encoded = self.le_soil.transform([soil_type])[0]
        crop_encoded = self.le_crop.transform([crop_type])[0]
        
        # Create feature vector
        features = [[temperature, humidity, moisture, nitrogen, 
                    potassium, phosphorous, soil_encoded, crop_encoded]]
        
        # Get probabilities
        probabilities = self.model.predict_proba(features)[0]
        
        # Create result dictionary
        result = {}
        for idx, prob in enumerate(probabilities):
            fertilizer = self.le_fertilizer.inverse_transform([idx])[0]
            result[fertilizer] = float(prob)
        
        # Sort by probability
        result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
        
        return result

def main():
    """Example usage"""
    print("="*50)
    print("FERTILIZER RECOMMENDATION PREDICTOR")
    print("="*50)
    
    # Initialize predictor
    predictor = FertilizerPredictor()
    
    # Example prediction
    print("\nExample Prediction:")
    temperature = 25.0
    humidity = 60.0
    moisture = 50.0
    soil_type = "Loamy"
    crop_type = "Wheat"
    nitrogen = 40.0
    potassium = 10.0
    phosphorous = 15.0
    
    print(f"\nInput:")
    print(f"  Temperature: {temperature}°C")
    print(f"  Humidity: {humidity}%")
    print(f"  Moisture: {moisture}%")
    print(f"  Soil Type: {soil_type}")
    print(f"  Crop Type: {crop_type}")
    print(f"  Nitrogen: {nitrogen}")
    print(f"  Potassium: {potassium}")
    print(f"  Phosphorous: {phosphorous}")
    
    fertilizer = predictor.predict_single(
        temperature, humidity, moisture, soil_type, crop_type,
        nitrogen, potassium, phosphorous
    )
    
    print(f"\nRecommended Fertilizer: {fertilizer}")
    
    # Get probabilities
    print("\nPrediction Probabilities:")
    probs = predictor.get_prediction_probabilities(
        temperature, humidity, moisture, soil_type, crop_type,
        nitrogen, potassium, phosphorous
    )
    
    for fert, prob in list(probs.items())[:3]:  # Top 3
        print(f"  {fert:30s}: {prob:.4f}")

if __name__ == "__main__":
    main()
