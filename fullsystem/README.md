# Computer Vision Vault - Raspberry Pi Security System

This project implements a computer vision-based security vault system using Python, OpenCV, and TensorFlow on a Raspberry Pi. The system uses a camera to detect and recognize hand-written digits (0-9) for PIN-based access control with physical hardware feedback.

## Features

- **Real-time digit recognition**: Uses OpenCV and TensorFlow for computer vision processing
- **PIN-based security**: 3-digit PIN authentication system
- **Hardware feedback**: 7-segment display, LEDs, and optional buzzer/motor control
- **Raspberry Pi camera**: Uses picamera2 for video capture
- **MNIST-trained model**: Neural network trained on MNIST dataset for digit recognition
- **GPIO integration**: Controls physical components through Raspberry Pi GPIO pins

## Hardware Requirements

- **Raspberry Pi** (with GPIO pins)
- **Pi Camera** or compatible camera module
- **7-segment display** (for showing detected digits)
- **3 LEDs** (for PIN progress indication)
- **Push button** (for reset functionality)
- **Breadboard and jumper wires**
- **Optional**: Buzzer and servo motor for additional feedback

## Software Requirements

- **Python 3.7 or higher**
- **Raspberry Pi OS** with camera support enabled
- **Good lighting conditions**
- **White paper and dark pen/pencil** for writing digits

## Installation

1. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Test your setup**:
   ```bash
   python test_setup.py
   ```
   This will verify that:
   - All required packages are installed correctly
   - Camera is accessible and working
   - Basic functionality is operational

3. **Configure GPIO pins**: Edit the pin assignments in [`main.py`](main.py) to match your hardware setup:
   ```python
   BUTTON_PIN = 0  # Set to your button pin
   LED_PINS = [0,0,0]  # Set to your LED pins
   SEGMENTS = {  # Set to your 7-segment display pins
       'a': 0, 'b': 0, 'c': 0, 'd': 0,
       'e': 0, 'f': 0, 'g': 0,
   }
   ```

## Usage

1. **Run the vault system** with a 3-digit PIN:
   ```bash
   python main.py 123
   ```

2. **Show digits to camera**: Write digits clearly on white paper with dark ink

3. **Enter PIN sequence**: Show each digit of your PIN in order to the camera

4. **Visual feedback**:
   - **7-segment display**: Shows the currently detected digit
   - **LEDs**: Light up progressively as correct digits are entered
   - **Reset button**: Press to restart the sequence

5. **Access granted**: When all 3 digits are correctly entered in sequence

## Command Line Options

```bash
python main.py <PIN>        # Enter 3-digit PIN (100-999)
python main.py --help       # Show help
```

Example:
```bash
python main.py 456          # Set PIN to 456
```

## GPIO Pin Configuration

Update these variables in [`main.py`](main.py) according to your wiring:

```python
# Required pins
BUTTON_PIN = 18              # Button for reset
LED_PINS = [16, 20, 21]      # 3 LEDs for progress
SEGMENTS = {                 # 7-segment display
    'a': 2, 'b': 3, 'c': 4, 'd': 5,
    'e': 6, 'f': 7, 'g': 8,
}

# Optional pins (uncomment to enable)
# BUZZER_PIN = 12            # Buzzer for success sound
# MOTOR_PIN = 13             # Servo motor for lock mechanism
```

## How It Works

1. **Camera Capture**: [`DigitTracker`](computer_vision/digit_tracker.py) captures frames using picamera2
2. **Image Processing**: Converts to grayscale, applies thresholding, and finds contours
3. **Digit Detection**: Locates potential digit regions in the center of the frame
4. **Recognition**: [`DigitRecognizer`](computer_vision/digit_recognizer.py) uses a trained neural network to classify digits
5. **PIN Validation**: Compares detected digits against the expected PIN sequence
6. **Hardware Control**: Updates LEDs, 7-segment display, and optional components

## Model Details

- **Architecture**: Simple Neural Network (Dense layers)
- **Training Data**: MNIST dataset (60,000 training images)
- **Input Size**: 28x28 pixels, grayscale, flattened to 784 features
- **Output**: 10 classes (digits 0-9)
- **Accuracy**: ~98% on MNIST test set
- **Model File**: Automatically saved as `digit_model.h5`

## Project Structure

```
upward-bound/
├── main.py                           # Main vault application
├── test_setup.py                     # Setup verification script
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── computer_vision/
    ├── __init__.py                   # Package initialization
    ├── digit_recognizer.py           # Neural network model
    └── digit_tracker.py              # Computer vision processing
```

## Tips for Better Recognition

1. **Lighting**: Ensure bright, even lighting on the paper
2. **Contrast**: Use dark ink/pencil on white paper
3. **Size**: Write digits large enough (at least 2-3cm tall)
4. **Clarity**: Write digits clearly and legibly
5. **Positioning**: Center digits in the camera view
6. **Stability**: Hold paper steady to avoid motion blur
7. **Distance**: Maintain consistent distance from camera

## Troubleshooting

### Setup Issues
- **Run test script**: `python test_setup.py` to diagnose problems
- **Package installation**: If packages fail to install, try `pip install --upgrade pip` first
- **Permission errors**: On some systems, use `pip install --user -r requirements.txt`

### Camera Issues
- **Enable camera**: `sudo raspi-config` → Interface Options → Camera
- **Check permissions**: Ensure user is in `video` group
- **Test camera**: The test script will verify camera access

### Recognition Issues
- **Improve lighting**: Add more light sources
- **Check contrast**: Ensure dark writing on white background
- **Adjust parameters**: Modify confidence threshold in [`main.py`](main.py)
- **Retrain model**: Delete `digit_model.h5` to retrain

### Hardware Issues
- **LED not lighting**: Check wiring and pin assignments
- **7-segment display**: Verify all segment connections
- **Button not working**: Check pull-up resistor configuration

## Future Enhancements

- **Multi-digit recognition**: Support for longer PINs
- **User management**: Multiple user PIN storage
- **Logging system**: Access attempt tracking
- **Web interface**: Remote monitoring and control
- **Encrypted storage**: Secure PIN storage
- **Backup authentication**: Alternative access methods