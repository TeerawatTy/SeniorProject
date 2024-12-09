import joblib
import numpy as np  # Import numpy

# Load the model
model_data = joblib.load("C:/Users/USER/Documents/Party/01 - PSU/Project II/Models/Model - DCT.joblib")
clf = model_data['model']
signature_list = model_data['signature_list']

# Test prediction
sample_input = np.zeros((1, len(signature_list)))  # Just an example with zeros
prediction = clf.predict(sample_input)

print(f"Prediction: {prediction}")
