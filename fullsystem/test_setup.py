#!/usr/bin/env python3
"""
Quick test script to verify camera access and basic functionality
"""

import cv2
import sys

def test_camera():
    """Test if camera is accessible"""
    print("Testing camera access...")
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot access camera. Please check:")
        print("1. Camera is connected and working")
        print("2. No other applications are using the camera")
        print("3. Camera permissions are granted")
        return False
    
    print("Camera access successful!")
    print("Press 'q' to close the test window")
    
    # Test camera feed
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read frame from camera")
            break
        
        # Display frame
        cv2.imshow('Camera Test - Press Q to quit', frame)
        
        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Camera test completed successfully!")
    return True

def test_imports():
    """Test if all required packages are available"""
    print("Testing package imports...")
    
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        return False
    
    print("All packages imported successfully!")
    return True

if __name__ == "__main__":
    print("="*50)
    print("DIGIT TRACKING SYSTEM - SETUP TEST")
    print("="*50)
    
    # Test imports
    if not test_imports():
        print("\nERROR: Some required packages are missing.")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n" + "-"*50)
    
    # Test camera
    if not test_camera():
        print("\nERROR: Camera test failed.")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("SETUP TEST COMPLETED SUCCESSFULLY!")
    print("You can now run the main application with: python main.py")
    print("="*50)
