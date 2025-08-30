import cv2

def test_camera():
    print("Testing camera access...")
    
    # Try different camera indices (0-4)
    for i in range(5):
        print(f"\nTrying camera index {i}...")
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print(f"  - Could not open camera {i}")
            cap.release()
            continue
            
        print(f"  - Successfully opened camera {i}")
        print("  - Press any key to continue to next camera...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("  - Could not read frame")
                break
                
            cv2.imshow(f'Camera {i}', frame)
            
            # Break the loop if any key is pressed
            if cv2.waitKey(1) != -1:
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    print("\nCamera test completed.")

if __name__ == "__main__":
    test_camera()
