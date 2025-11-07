"""
Fertilizer Recommendation Model
Train and save a machine learning model for fertilizer prediction
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import json

def load_data(train_path='data/train.csv'):
    """Load training data"""
    df = pd.read_csv(train_path)
    print(f"Loaded {len(df)} training samples")
    return df

def preprocess_data(df):
    """Preprocess the data: encode categorical variables"""
    df = df.copy()
    
    # Initialize label encoders
    le_soil = LabelEncoder()
    le_crop = LabelEncoder()
    le_fertilizer = LabelEncoder()
    
    # Encode categorical features
    df['Soil Type Encoded'] = le_soil.fit_transform(df['Soil Type'])
    df['Crop Type Encoded'] = le_crop.fit_transform(df['Crop Type'])
    
    # Encode target variable
    y = le_fertilizer.fit_transform(df['Fertilizer Name'])
    
    # Select features
    feature_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 
                    'Potassium', 'Phosphorous', 'Soil Type Encoded', 'Crop Type Encoded']
    X = df[feature_cols]
    
    return X, y, le_soil, le_crop, le_fertilizer

def train_model(X, y):
    """Train a Random Forest classifier"""
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"\nValidation Accuracy: {accuracy:.4f}")
    
    return model, X_train, X_val, y_train, y_val

def evaluate_model(model, X_val, y_val, le_fertilizer):
    """Evaluate the trained model"""
    y_pred = model.predict(X_val)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Accuracy
    accuracy = accuracy_score(y_val, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    target_names = le_fertilizer.classes_
    print(classification_report(y_val, y_pred, target_names=target_names))
    
    # Feature importance
    feature_names = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 
                     'Potassium', 'Phosphorous', 'Soil Type', 'Crop Type']
    importances = model.feature_importances_
    
    print("\nFeature Importance:")
    for name, importance in sorted(zip(feature_names, importances), 
                                   key=lambda x: x[1], reverse=True):
        print(f"  {name:20s}: {importance:.4f}")
    
    return accuracy

def save_model(model, le_soil, le_crop, le_fertilizer, model_path='model'):
    """Save the trained model and encoders"""
    import os
    os.makedirs(model_path, exist_ok=True)
    
    # Save model
    with open(f'{model_path}/fertilizer_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save encoders
    with open(f'{model_path}/soil_encoder.pkl', 'wb') as f:
        pickle.dump(le_soil, f)
    
    with open(f'{model_path}/crop_encoder.pkl', 'wb') as f:
        pickle.dump(le_crop, f)
    
    with open(f'{model_path}/fertilizer_encoder.pkl', 'wb') as f:
        pickle.dump(le_fertilizer, f)
    
    # Save feature names and model info
    model_info = {
        'feature_names': ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 
                         'Potassium', 'Phosphorous', 'Soil Type', 'Crop Type'],
        'soil_types': le_soil.classes_.tolist(),
        'crop_types': le_crop.classes_.tolist(),
        'fertilizer_types': le_fertilizer.classes_.tolist()
    }
    
    with open(f'{model_path}/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\nModel and encoders saved to '{model_path}/' directory")

def main():
    """Main training pipeline"""
    print("="*50)
    print("FERTILIZER RECOMMENDATION MODEL TRAINING")
    print("="*50)
    
    # Load data
    df = load_data()
    
    # Preprocess
    print("\nPreprocessing data...")
    X, y, le_soil, le_crop, le_fertilizer = preprocess_data(df)
    
    # Train
    model, X_train, X_val, y_train, y_val = train_model(X, y)
    
    # Evaluate
    evaluate_model(model, X_val, y_val, le_fertilizer)
    
    # Save
    save_model(model, le_soil, le_crop, le_fertilizer)
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()
