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

    # TODO: Make sure variables are set to correct pin numbers based on your setup. 
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
            
    def run_sequence(self, pin: int) -> None:
        """Run the main sequence for digit recognition"""
        num_correct = 0
        pin_digits = [int(d) for d in str(pin)]
        
        print(f"Starting vault sequence for PIN: {pin}")
        print("Show digits to the camera in sequence...")
        
        try:
            # Initialize digit tracker
            tracker = DigitTracker()
            tracker.initialize_camera()
            
            while num_correct < 3:
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
                    
                    # Check if it matches the expected digit
                    if pin_digits[num_correct] == prediction:
                        print(f"Correct! Digit {num_correct + 1}/3")
                        num_correct += 1
                        
                        # Check if all digits are correct
                        if num_correct >= 3:
                            print("Access granted! All digits correct.")
                            break
                    else:
                        print(f"Incorrect. Expected: {pin_digits[num_correct]}, Got: {prediction}")
                    
                    time.sleep(self.DISPLAY_DELAY)
                    # Clear 7-segment display
                    self.clear_7seg_display()
                    
        except KeyboardInterrupt:
            print("\nSequence interrupted by user")
        except Exception as e:
            print(f"Error during sequence: {e}")
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
    parser.add_argument('pin', type=int, help='Enter 3 digit PIN (100-999)')
    args = parser.parse_args()
    
    # Validate PIN
    if args.pin < 100 or args.pin > 999:
        print('ERROR: Enter a valid 3 digit PIN (100-999)!')
        sys.exit(1)
    
    vault = None
    try:
        # Initialize the vault system
        vault = ComputerVisionVault()
        
        # Run the sequence
        vault.run_sequence(args.pin)
        
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
    
        
    