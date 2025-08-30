import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Constants
IMG_SIZE = 64
CLASS_NAMES = ['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down']

def load_gesture_model():
    """Load the trained gesture recognition model"""
    try:
        model_path = 'models/hand_gesture_model.h5'
        print(f"Loading model from: {model_path}")
        model = load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_frame(frame):
    """Preprocess the frame for prediction"""
    if frame is None or frame.size == 0:
        return None
    # Convert to grayscale and resize
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    return np.expand_dims(normalized, axis=(0, -1))

def main():
    # Initialize webcam with DirectShow
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Load model
    print("Loading gesture recognition model...")
    model = load_gesture_model()
    if model is None:
        print("Failed to load model. Please check the model file.")
        cap.release()
        return
    
    print("\nStarting gesture recognition...")
    print("1. Place your hand in the green box")
    print("2. Make sure your hand is well-lit")
    print("3. Press 'q' to quit\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break
        
        # Create a copy for display
        display = frame.copy()
        
        # Get center of the frame for ROI
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        roi_size = min(w, h) // 3
        
        # Define ROI (Region of Interest)
        x1 = max(0, center_x - roi_size // 2)
        y1 = max(0, center_y - roi_size // 2)
        x2 = min(w, center_x + roi_size // 2)
        y2 = min(h, center_y + roi_size // 2)
        
        # Process ROI
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            try:
                # Preprocess and predict
                processed = preprocess_frame(roi)
                if processed is not None:
                    prediction = model.predict(processed, verbose=0)
                    predicted_class = np.argmax(prediction[0])
                    confidence = np.max(prediction[0]) * 100
                    
                    if confidence > 60:  # Only show if confident
                        gesture = CLASS_NAMES[predicted_class]
                        cv2.putText(display, f"{gesture} ({confidence:.1f}%)", 
                                  (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            except Exception as e:
                print(f"Prediction error: {e}")
        
        # Draw ROI rectangle and instructions
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, "Place hand here", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Hand Gesture Recognition', display)
        
        # Break on 'q' or window close
        if cv2.waitKey(1) & 0xFF == ord('q') or \
           cv2.getWindowProperty('Hand Gesture Recognition', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nGesture recognition stopped.")

if __name__ == "__main__":
    main()
