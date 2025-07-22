#!/usr/bin/env python3
"""
Computer Vision Vault - A Raspberry Pi security system using digit recognition
Simple version for high school students learning programming
"""

import argparse
import atexit
import sys
import time

import cv2
import RPi.GPIO as GPIO
from computer_vision.digit_tracker import DigitTracker

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

# Settings you can adjust
CONFIDENCE_THRESHOLD = 0.80
DISPLAY_DELAY = 1.0

def setup_gpio():
    """Setup the GPIO pins for the 7-segment display"""
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    
    # Setup each segment pin
    for segment_pin in SEGMENTS.values():
        GPIO.setup(segment_pin, GPIO.OUT)
        GPIO.output(segment_pin, False)
    
    print("GPIO pins setup successfully")

def display_number(number):
    """Show a number on the 7-segment display"""
    if number in NUMBERS:
        segments_to_light = NUMBERS[number]
        for segment in segments_to_light:
            GPIO.output(SEGMENTS[segment], True)

def clear_display():
    """Turn off all segments on the display"""
    for segment_pin in SEGMENTS.values():
        GPIO.output(segment_pin, False)

def cleanup():
    """Clean up resources when done"""
    try:
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Cleanup completed")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def recognize_digit(target_digit):
    """Main function to recognize a digit from the camera"""
    print(f"Looking for digit: {target_digit}")
    print("Show the digit to the camera...")
    
    try:
        # Start the camera and digit tracker
        tracker = DigitTracker()
        tracker.initialize_camera()
        
        while True:
            # Get what the camera thinks it sees
            prediction, confidence = tracker.get_single_prediction()

            # Show the camera view
            frame = tracker.capture_frame()[1]
            if frame is not None:
                cv2.imshow("Pi Camera", frame)

            # If we detected a digit with good confidence
            if prediction is not None and confidence > CONFIDENCE_THRESHOLD:
                print(f"I see digit: {prediction} (confidence: {confidence:.2f})")
                
                # Show the detected number on the display
                display_number(prediction)
                
                # Check if it's the right digit
                if target_digit == prediction:
                    print("🎉 ACCESS GRANTED! Correct digit detected! 🎉")
                    break
                else:
                    print(f"❌ Wrong digit. Expected: {target_digit}, Got: {prediction}")
                
                time.sleep(DISPLAY_DELAY)
                clear_display()
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Always clean up when done
        if 'tracker' in locals():
            tracker.__del__()
        cleanup()


def main():
    """Main function"""
    # Setup cleanup function
    atexit.register(GPIO.cleanup)
    
    parser = argparse.ArgumentParser(description="Computer Vision Vault - Raspberry Pi Security System")
    parser.add_argument('digit', type=int, help='Enter single digit (0-9)')
    args = parser.parse_args()
    
    # Validate digit
    if args.digit < 0 or args.digit > 9:
        print('ERROR: Enter a valid single digit (0-9)!')
        sys.exit(1)
    
    try:
        # Setup the GPIO pins
        setup_gpio()
        
        # Start looking for the digit
        recognize_digit(args.digit)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        # Always clean up
        cleanup()


if __name__ == "__main__":
    main()
    
        
    