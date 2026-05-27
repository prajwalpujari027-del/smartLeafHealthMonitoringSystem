import numpy as np
import cv2
import json
from tensorflow.keras.models import load_model
import os

# Load model and class names
print("🔄 Loading model...")
model = load_model("/content/drive/MyDrive/plant_disease_model.keras")

with open("/content/drive/MyDrive/class_names.json", "r") as f:
    class_names = json.load(f)

print("✅ Model loaded successfully!\n")

# Solutions dictionary
solutions = {
    "Tomato___healthy": "Plant is healthy. Maintain watering and sunlight.",
    "Tomato___Bacterial_spot": "Use copper fungicide. Avoid wet leaves.",
    "Tomato___Early_blight": "Remove infected leaves. Apply fungicide.",
    "Tomato___Late_blight": "Use resistant seeds. Apply fungicide.",
    "Tomato___Leaf_Mold": "Reduce humidity. Improve airflow.",
    "Tomato___Septoria_leaf_spot": "Remove affected leaves. Spray fungicide.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Use neem oil spray.",
    "Tomato___Target_Spot": "Avoid leaf wetness. Apply fungicide.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whiteflies.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants. Clean tools.",
    "Tomato__Random": "⚠️  This is not a tomato plant image. Please upload a tomato leaf image."
}

# Prediction function
def predict_image(image_path):
    """Predict disease from image"""
    img = cv2.imread(image_path)
 
    if img is None:
        return None, None
 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
 
    prediction = model.predict(img, verbose=0)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)
 
    return class_names[class_index], confidence

# Display prediction results
def display_results(image_path, disease, confidence):
    """Display prediction results in a nice format"""
    print("\n" + "=" * 80)
    print(f"📸 Image: {os.path.basename(image_path)}")
    print("=" * 80)
 
    if disease == "Tomato__Random":
        print(f"🚫 Result: NOT A TOMATO PLANT")
        print(f"📊 Confidence: {confidence*100:.2f}%")
        print(f"⚠️  This image does not contain a tomato leaf.")
    else:
        print(f"🌿 Disease Detected: {disease}")
        print(f"📊 Confidence: {confidence*100:.2f}%")
 
    advice = solutions.get(disease, "No suggestion available.")
    print(f"\n💡 Recommendation: {advice}")
    print("=" * 80 + "\n")

# Main testing loop
def main():
    """Main function to test images one by one"""
    print("🍅 Tomato Disease Detection System")
    print("-" * 80)
    print("Instructions:")
    print("  • Enter the full path to your image")
    print("  • Type 'quit' to exit")
    print("  • Type 'clear' to clear screen")
    print("-" * 80 + "\n")
 
    while True:
        # Get image path from user
        image_path = input("🖼️  Enter image path (or 'quit'): ").strip()
 
        # Handle quit
        if image_path.lower() == 'quit':
            print("\n👋 Thank you for using Tomato Disease Detection System!")
            break
 
        # Handle clear
        if image_path.lower() == 'clear':
            os.system('clear' if os.name == 'posix' else 'cls')
            print("🍅 Tomato Disease Detection System\n")
            continue
 
        # Check if path is empty
        if not image_path:
            print("❌ Please enter a valid path\n")
            continue
 
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}\n")
            continue
 
        # Predict
        print("\n🔍 Analyzing image...")
        disease, confidence = predict_image(image_path)
 
        if disease is None:
            print("❌ Could not read image. Make sure it's a valid image file (JPG, PNG, etc.)\n")
            continue
 
        # Display results
        display_results(image_path, disease, confidence)

# Run the program
if __name__ == "__main__":
    main()