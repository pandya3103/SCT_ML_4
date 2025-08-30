import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Constants
IMG_SIZE = 64
CLASS_NAMES = [
    'palm', 'l', 'fist', 'fist_moved', 'thumb', 
    'index', 'ok', 'palm_moved', 'c', 'down'
]

def load_saved_model():
    """Load the trained model and class names"""
    try:
        model = load_model('models/hand_gesture_model.h5')
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_frame(frame):
    """Preprocess the frame for prediction"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize to match model's expected sizing
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    
    # Normalize the image
    normalized = resized / 255.0
    
    # Reshape for the model
    return np.expand_dims(normalized, axis=(0, -1))  # Add batch and channel dimensions

def draw_hand_roi(frame, x, y, size=200):
    """Draw the hand region of interest"""
    # Calculate ROI coordinates
    x1 = max(0, x - size // 2)
    y1 = max(0, y - size // 2)
    x2 = min(frame.shape[1], x + size // 2)
    y2 = min(frame.shape[0], y + size // 2)
    
    # Draw rectangle around ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return x1, y1, x2, y2

def main():
    # Load the trained model
    print("Loading model...")
    model = load_saved_model()
    if model is None:
        print("Failed to load model. Please ensure you've trained the model first.")
        return
    
    # Initialize webcam with different backends
    print("Initializing webcam...")
    
    # Try different backends in order of preference
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            print(f"Successfully opened webcam with backend: {backend}")
            break
    
    if not cap.isOpened():
        print("Error: Could not open webcam. Please ensure it's connected and not in use by another application.")
        return
    
    # Set webcam resolution - use smaller resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Verify resolution was set correctly
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam resolution set to: {actual_width}x{actual_height}")
    
    # Create a window and set it to be resizable
    cv2.namedWindow('Hand Gesture Recognition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hand Gesture Recognition', 800, 600)
    
    # Get screen resolution
    screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Center of the screen
    center_x, center_y = screen_width // 2, screen_height // 2
    roi_size = min(screen_width, screen_height) // 2  # Dynamic ROI size based on screen resolution
    
    # Variables for FPS calculation
    prev_time = 0
    fps = 0
    
    print("Starting real-time gesture recognition. Press 'q' to quit.")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Draw ROI and get coordinates
        x1, y1, x2, y2 = draw_hand_roi(frame, center_x, center_y, roi_size)
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        # Skip if ROI is too small
        if roi.size == 0 or roi.shape[0] < 32 or roi.shape[1] < 32:
            cv2.putText(frame, "ROI too small or invalid", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Hand Gesture Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        if roi.size > 0:
            try:
                # Preprocess and predict
                processed = preprocess_frame(roi)
                prediction = model.predict(processed, verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = np.max(prediction[0]) * 100
                
                # Only show prediction if confidence is high enough
                if confidence > 70:  # 70% confidence threshold
                    # Get the predicted class name
                    gesture = CLASS_NAMES[predicted_class]
                    
                    # Display prediction with confidence
                    cv2.putText(frame, f"Gesture: {gesture} ({confidence:.1f}%)", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Visual feedback - change ROI color based on confidence
                    color = (0, 255, 0)  # Green for high confidence
                    if confidence < 85:
                        color = (0, 255, 255)  # Yellow for medium confidence
                    
                    # Redraw ROI with confidence-based color
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                else:
                    cv2.putText(frame, "Low confidence - try again", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
            except Exception as e:
                print(f"Prediction error: {e}")
                cv2.putText(frame, "Error in prediction", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display instructions
        cv2.putText(frame, "Place your hand in the green box", (10, screen_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        try:
            # Show the frame
            cv2.imshow('Hand Gesture Recognition', frame)
            
            # Break the loop on 'q' key press or if window is closed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty('Hand Gesture Recognition', cv2.WND_PROP_VISIBLE) < 1:
                break
                
        except Exception as e:
            print(f"Display error: {e}")
            print("Trying to reinitialize display...")
            cv2.destroyAllWindows()
            cv2.namedWindow('Hand Gesture Recognition', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Hand Gesture Recognition', 800, 600)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
