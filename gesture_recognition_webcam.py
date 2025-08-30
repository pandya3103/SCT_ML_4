import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import os

# Constants
IMG_SIZE = 64
CLASS_NAMES = [
    'palm', 'l', 'fist', 'fist_moved', 'thumb', 
    'index', 'ok', 'palm_moved', 'c', 'down'
]

def load_gesture_model():
    """Load the trained gesture recognition model"""
    try:
        model_path = 'models/hand_gesture_model.h5'
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {os.path.abspath(model_path)}")
            return None
            
        print(f"Loading model from: {os.path.abspath(model_path)}")
        model = load_model(model_path)
        print("Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def preprocess_frame(frame, debug=False):
    """Preprocess the frame for prediction"""
    if frame is None or frame.size == 0:
        print("Error: Empty frame received")
        return None
        
    try:
        # Convert to grayscale and resize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        normalized = resized / 255.0
        
        if debug:
            print(f"Preprocessed frame - Min: {np.min(normalized):.2f}, Max: {np.max(normalized):.2f}, Mean: {np.mean(normalized):.2f}")
            
        return np.expand_dims(normalized, axis=(0, -1))
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None

def main():
    # Load the model
    print("Loading gesture recognition model...")
    model = load_gesture_model()
    if model is None:
        print("Failed to load model. Please check the error messages above.")
        return
    
    # Initialize webcam with DirectShow
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Create window
    cv2.namedWindow('Hand Gesture Recognition', cv2.WINDOW_NORMAL)
    
    print("\nStarting gesture recognition...")
    print("1. Place your hand in the green box")
    print("2. Make sure your hand is well-lit")
    print("3. Press 'q' to quit\n")
    
    prev_time = 0
    fps = 0
    prediction_history = []
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break
        
        # Create a copy for display
        display = frame.copy()
        
        # Get center of the frame
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        roi_size = min(w, h) // 3
        
        # Draw ROI rectangle
        x1 = max(0, center_x - roi_size // 2)
        y1 = max(0, center_y - roi_size // 2)
        x2 = min(w, center_x + roi_size // 2)
        y2 = min(h, center_y + roi_size // 2)
        
        # Process ROI
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            try:
                # Preprocess and predict
                processed = preprocess_frame(roi, debug=False)
                if processed is not None:
                    prediction = model.predict(processed, verbose=0)
                    predicted_class = np.argmax(prediction[0])
                    confidence = np.max(prediction[0]) * 100
                    
                    # Add to prediction history (for stability)
                    prediction_history.append((predicted_class, confidence))
                    if len(prediction_history) > 5:  # Keep last 5 predictions
                        prediction_history.pop(0)
                    
                    # Get most common prediction in history
                    if prediction_history:
                        classes, confidences = zip(*prediction_history)
                        most_common = max(set(classes), key=classes.count)
                        avg_confidence = np.mean([c for c, conf in zip(classes, confidences) if c == most_common])
                        
                        if avg_confidence > 60:  # Only show if confident
                            gesture = CLASS_NAMES[most_common]
                            text = f"{gesture} ({avg_confidence:.1f}%)"
                            
                            # Draw prediction
                            cv2.putText(display, text, (20, 50), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            # Print debug info
                            print("\n" + "="*50)
                            print("Predictions:")
                            for i, (cls, conf) in enumerate(zip(CLASS_NAMES, prediction[0])):
                                print(f"{cls}: {conf*100:.1f}%")
            
            except Exception as e:
                print(f"Prediction error: {e}")
        
        # Draw ROI rectangle with instructions
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, "Place hand here", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Calculate FPS
        current_time = time.time()
        fps = 0.9 * fps + 0.1 * (1 / (current_time - prev_time + 1e-6))
        prev_time = current_time
        
        # Display FPS and status
        cv2.putText(display, f"FPS: {fps:.1f}", (20, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Hand Gesture Recognition', display)
        
        # Break on 'q' or window close
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty('Hand Gesture Recognition', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nGesture recognition stopped.")

if __name__ == "__main__":
    main()
