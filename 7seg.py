import RPi.GPIO as GPIO

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

def display_number_on_7seg(number: int) -> None:
    """Display a number on the 7-segment display"""
    if number in NUMBERS:
        segments_to_light = NUMBERS[number]
        for segment in segments_to_light:
            GPIO.output(SEGMENTS[segment], True)

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
# Setup 7-segment display
for segment_pin in SEGMENTS.values():
    GPIO.setup(segment_pin, GPIO.OUT)
    GPIO.output(segment_pin, False)

display_number_on_7seg(5)

