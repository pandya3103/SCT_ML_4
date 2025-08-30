import cv2

def test_webcam():
    print("Testing webcam feed...")
    
    # Try different backends
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Default")
    ]
    
    cap = None
    for backend, name in backends:
        print(f"\nTrying {name} backend...")
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            print(f"Success with {name} backend!")
            break
        else:
            print(f"Failed with {name} backend")
    
    if not cap or not cap.isOpened():
        print("\nError: Could not open webcam with any backend")
        print("Please check if your webcam is properly connected and not in use by other applications.")
        return
    
    # Set resolution
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
