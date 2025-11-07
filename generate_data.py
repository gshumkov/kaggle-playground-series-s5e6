"""
Generate sample fertilizer recommendation dataset
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Define fertilizers and their typical conditions
fertilizers = {
    'Urea': {'N': (30, 50), 'P': (10, 25), 'K': (5, 15), 'temp': (15, 30), 'humidity': (40, 70)},
    'DAP': {'N': (15, 25), 'P': (35, 50), 'K': (5, 15), 'temp': (20, 35), 'humidity': (50, 80)},
    'NPK 10-26-26': {'N': (8, 15), 'P': (20, 35), 'K': (20, 35), 'temp': (20, 32), 'humidity': (55, 85)},
    'NPK 20-20-20': {'N': (15, 25), 'P': (15, 25), 'K': (15, 25), 'temp': (18, 28), 'humidity': (45, 75)},
    'Ammonium Sulphate': {'N': (18, 25), 'P': (5, 12), 'K': (3, 8), 'temp': (10, 25), 'humidity': (30, 60)},
    'Potassium Chloride': {'N': (3, 8), 'P': (3, 8), 'K': (30, 50), 'temp': (22, 35), 'humidity': (60, 90)},
    'SSP (Single Super Phosphate)': {'N': (5, 12), 'P': (12, 20), 'K': (5, 12), 'temp': (20, 30), 'humidity': (50, 75)},
}

soil_types = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
crop_types = ['Wheat', 'Rice', 'Maize', 'Cotton', 'Sugarcane', 'Barley', 'Soybean', 'Groundnut', 'Pulses', 'Tobacco']

# Generate data
n_samples = 2000
data = []

for fertilizer_name, conditions in fertilizers.items():
    n_per_fertilizer = n_samples // len(fertilizers)
    
    for _ in range(n_per_fertilizer):
        # Generate values with some noise
        nitrogen = np.random.uniform(conditions['N'][0], conditions['N'][1])
        phosphorous = np.random.uniform(conditions['P'][0], conditions['P'][1])
        potassium = np.random.uniform(conditions['K'][0], conditions['K'][1])
        temperature = np.random.uniform(conditions['temp'][0], conditions['temp'][1])
        humidity = np.random.uniform(conditions['humidity'][0], conditions['humidity'][1])
        moisture = np.random.uniform(20, 80)  # General moisture range
        soil_type = np.random.choice(soil_types)
        crop_type = np.random.choice(crop_types)
        
        data.append({
            'Temperature': round(temperature, 2),
            'Humidity': round(humidity, 2),
            'Moisture': round(moisture, 2),
            'Soil Type': soil_type,
            'Crop Type': crop_type,
            'Nitrogen': round(nitrogen, 2),
            'Potassium': round(potassium, 2),
            'Phosphorous': round(phosphorous, 2),
            'Fertilizer Name': fertilizer_name
        })

# Create DataFrame
df = pd.DataFrame(data)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into train and test
train_size = int(0.8 * len(df))
train_df = df[:train_size]
test_df = df[train_size:]

# Save to CSV
train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

print(f"Generated {len(train_df)} training samples and {len(test_df)} test samples")
print(f"Features: {list(df.columns)}")
print(f"Fertilizer distribution:")
print(df['Fertilizer Name'].value_counts())
