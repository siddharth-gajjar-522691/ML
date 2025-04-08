'''
train model By using opencv's LBPH (Local binary Pattern Histogram) 
face recognizer using your collected database
'''
import cv2
import os
import numpy as np

# Paths
dataset_path = "../data/faces"
model_save_path = "../model/trained_model.yml"
os.makedirs("../model", exist_ok=True)

# Initialize recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Storage for data and Labels
faces = []
labels = []

# Label extraction function
def extract_label(filename):
    # Expected filename format: user_<id>_<count>.jpg
    return int(filename.split('_')[1])

print(f"Loading dataset and extracting labels..")

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".jpg"):
            path = os.path.join(root, file)
            print(f"Opening file: {path}")
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            label = extract_label(file)
            faces.append(img)
            labels.append(label)
        
print(f"Training on {len(faces)} face images...")

# Train the recognizer
recognizer.train(faces, np.array(labels))

# Save the model
recognizer.save(model_save_path)

print(f"🧠 Training complete. Model saved to {model_save_path}")
        