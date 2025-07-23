#!/usr/bin/env python3
"""
Computer Vision Vault - A Raspberry Pi security system using digit recognition
"""

import argparse
import atexit
import sys
import time

import cv2
import RPi.GPIO as GPIO
from computer_vision.digit_tracker import DigitTracker


class ComputerVisionVault:
    """Computer Vision Vault system using MNIST digit recognition"""
    
    # GPIO Pin Configurations

    # TODO: Make sure variables are set to correct GPIO numbers based on your setup. 
    SEGMENTS = {
        'a': 13, 'b': 19, 'c': 7, 'd': 8,
        'e': 5, 'f': 6, 'g': 26,
    }
    
    # 7-segment display number patterns
    NUMBERS = {
        0: ['a', 'b', 'c', 'd', 'e', 'f'],
        1: ['b', 'c'],
        2: ['a', 'b', 'g', 'e', 'd'],
        3: ['a', 'b', 'g', 'c', 'd'],
        4: ['f', 'g', 'b', 'c'],
        5: ['a', 'f', 'g', 'c', 'd'],
        6: ['a', 'f', 'g', 'c', 'd', 'e'],
        7: ['a', 'b', 'c'],
        8: ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
        9: ['a', 'b', 'c', 'd', 'f', 'g'],
    }
    
    # Can adjust if needed
    CONFIDENCE_THRESHOLD = 0.80
    DISPLAY_DELAY = 1.0
    
    def __init__(self, model_path: str = "mnist_model.tflite"):
        """Initialize the Computer Vision Vault system"""
        self.model_path = model_path
        self.gpio_initialized = False
        
        self._initialize_gpio()
        
    def _initialize_gpio(self) -> None:
        """Initialize GPIO pins for 7-segment display"""
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        
        # Setup 7-segment display
        for segment_pin in self.SEGMENTS.values():
            GPIO.setup(segment_pin, GPIO.OUT)
            GPIO.output(segment_pin, False)

        self.gpio_initialized = True
        print("GPIO pins initialized successfully")
        
    def display_number_on_7seg(self, number: int) -> None:
        """Display a number on the 7-segment display"""
        if number in self.NUMBERS:
            segments_to_light = self.NUMBERS[number]
            for segment in segments_to_light:
                GPIO.output(self.SEGMENTS[segment], True)
                
    def clear_7seg_display(self) -> None:
        """Clear the 7-segment display"""
        for segment_pin in self.SEGMENTS.values():
            GPIO.output(segment_pin, False)
            
    def recognize_digit(self, target_digit: int) -> None:
        """Recognize a single digit"""
        print(f"Starting digit recognition for target: {target_digit}")
        print("Show the digit to the camera...")
        
        try:
            # Initialize digit tracker
            tracker = DigitTracker()
            tracker.initialize_camera()
            
            while True:
                # Get prediction from digit tracker
                prediction, confidence = tracker.get_single_prediction()

                # Display the frame with OpenCV (for monitoring)
                frame = tracker.capture_frame()[1]
                if frame is not None:
                    cv2.imshow("Pi Camera", frame)

                # Display prediction if confidence is above threshold and prediction is valid
                if prediction is not None and confidence > self.CONFIDENCE_THRESHOLD:
                    print(f"Detected digit: {prediction} (confidence: {confidence:.2f})")
                    
                    # Display predicted number on 7-segment display
                    self.display_number_on_7seg(prediction)
                    
                    # Check if it matches the target digit
                    if target_digit == prediction:
                        print("Access granted! Correct digit detected.")
                        break
                    else:
                        print(f"Incorrect. Expected: {target_digit}, Got: {prediction}")
                    
                    time.sleep(self.DISPLAY_DELAY)
                    # Clear 7-segment display
                    self.clear_7seg_display()
                    
        except KeyboardInterrupt:
            print("\nRecognition interrupted by user")
        except Exception as e:
            print(f"Error during recognition: {e}")
        finally:
            # Cleanup
            if 'tracker' in locals():
                tracker.__del__()
            self.cleanup()
            
    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            cv2.destroyAllWindows()
            if self.gpio_initialized:
                GPIO.cleanup()
            print("Cleanup completed")
        except Exception as e:
            print(f"Error during cleanup: {e}")


def main():
    """Main function"""
    def gpio_cleanup():
        try:
            if 'vault' in locals() and hasattr(vault, 'gpio_initialized') and vault.gpio_initialized:
                GPIO.cleanup()
        except Exception:
            # Ignore cleanup errors if GPIO wasn't properly initialized
            pass
    
    atexit.register(gpio_cleanup)
    
    parser = argparse.ArgumentParser(description="Computer Vision Vault - Raspberry Pi Security System")
    parser.add_argument('digit', type=int, help='Enter single digit (0-9)')
    args = parser.parse_args()
    
    # Validate digit
    if args.digit < 0 or args.digit > 9:
        print('ERROR: Enter a valid single digit (0-9)!')
        sys.exit(1)
    
    vault = None
    try:
        # Initialize the vault system
        vault = ComputerVisionVault()
        
        # Run the digit recognition
        vault.recognize_digit(args.digit)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        try:
            if vault and vault.gpio_initialized:
                GPIO.cleanup()
        except Exception:
            # Ignore cleanup errors if GPIO wasn't properly initialized
            pass


if __name__ == "__main__":
    main()
    
        
    