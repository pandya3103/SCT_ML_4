import cv2
import numpy as np

def test_webcam():
    print("Testing webcam...")
    
    # Try different backends
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Default")
    ]
    
    cap = None
    for backend, name in backends:
        print(f"Trying {name} backend...")
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            print(f"Successfully opened with {name}")
            break
    
    if not cap or not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set a reasonable resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nWebcam test started. Press 'q' to quit.")
    print("You should see the webcam feed in a new window.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame")
            break
            
        # Show the frame
        cv2.imshow('Webcam Test', frame)
        
        # Break the loop on 'q' key press or window close
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Webcam Test', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam test completed.")

if __name__ == "__main__":
    test_webcam()
