import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Load the model
print("Loading model...")
model = load_model('models/hand_gesture_model.h5')
print("Model loaded!")

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Gesture labels
GESTURES = ['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down']

print("\nPress 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame")
        break
    
    # Create a copy for display
    display = frame.copy()
    
    # Get center of the frame
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    size = min(w, h) // 3
    
    # Define ROI
    x1 = max(0, center_x - size // 2)
    y1 = max(0, center_y - size // 2)
    x2 = min(w, center_x + size // 2)
    y2 = min(h, center_y + size // 2)
    
    # Process ROI
    roi = frame[y1:y2, x1:x2]
    if roi.size > 0:
        try:
            # Preprocess
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            normalized = resized / 255.0
            input_data = np.expand_dims(normalized, axis=(0, -1))
            
            # Debug: Show ROI
            debug_roi = cv2.resize(roi, (200, 200))
            cv2.imshow('ROI', debug_roi)
            
            # Predict
            start_time = time.time()
            prediction = model.predict(input_data, verbose=0)[0]
            predict_time = (time.time() - start_time) * 1000  # in ms
            
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class] * 100
            
            # Always show the top prediction for debugging
            gesture = GESTURES[predicted_class]
            text = f"{gesture} ({confidence:.1f}%)"
            cv2.putText(display, text, (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Print debug info to console
            print(f"\n--- Prediction ---")
            print(f"Top prediction: {text}")
            print(f"Prediction time: {predict_time:.1f}ms")
            print("All predictions:")
            for i, (g, p) in enumerate(zip(GESTURES, prediction)):
                print(f"  {g}: {p*100:.1f}%")
                
        except Exception as e:
            print(f"Error during prediction: {e}")
    
        # Draw ROI with instructions
    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(display, "Place hand here", (x1, y1 - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Add FPS counter
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cv2.putText(display, f"FPS: {fps}", (w - 120, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show frame
    cv2.imshow('Hand Gesture', display)
    
    # Exit on 'q' or window close
    if cv2.waitKey(1) & 0xFF == ord('q') or \
       cv2.getWindowProperty('Hand Gesture', cv2.WND_PROP_VISIBLE) < 1:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
