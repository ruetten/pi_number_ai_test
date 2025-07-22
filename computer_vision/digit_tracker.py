import cv2
import numpy as np
import time
import os
from computer_vision.digit_recognizer import DigitRecognizer

# Set environment variable to use pure Python protobuf implementation (slower but more compatible)
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Try to import digit_recognizer with error handling
try:
    DIGIT_RECOGNIZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import DigitRecognizer: {e}")
    print("Running in camera-only mode without digit recognition")
    DIGIT_RECOGNIZER_AVAILABLE = False

# Try to import picamera2 for Raspberry Pi
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    print("Warning: picamera2 not available. This code is designed for Raspberry Pi with picamera2.")
    PICAMERA2_AVAILABLE = False

class DigitTracker:
    def __init__(self):
        if not PICAMERA2_AVAILABLE:
            raise ImportError("picamera2 is required but not available. Please install it with: pip install picamera2")
        
        if DIGIT_RECOGNIZER_AVAILABLE:
            self.recognizer = DigitRecognizer()
        else:
            self.recognizer = None
            print("Running without digit recognition - will only show camera feed")
        
        self.picam2 = None
        self.min_contour_area = 200  # Reduced from 500 to catch smaller 1s
        self.max_contour_area = 10000
        
        # Flickering reduction parameters
        self.tracked_digits = {}  # Store tracked digits with their history
        self.prediction_history = {}  # Store prediction history for smoothing
        self.max_history_length = 5  # Number of frames to keep in history
        self.confidence_threshold = 0.6  # Higher threshold for initial detection
        self.stable_confidence_threshold = 0.4  # Lower threshold for stable tracking
        self.spatial_tolerance = 50  # Pixels tolerance for spatial tracking
        
    def initialize_camera(self):
        """Initialize camera capture using picamera2"""
        try:
            # Initialize Picamera2
            self.picam2 = Picamera2()
            
            # Configure camera for video capture
            config = self.picam2.create_video_configuration(
                main={"size": (640, 480), "format": "RGB888"},
                controls={"FrameRate": 30}
            )
            self.picam2.configure(config)
            
            # Start the camera
            self.picam2.start()
            
            # Allow camera to warm up
            time.sleep(0.5)
            
            print("Picamera2 initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize Picamera2: {e}")
            raise ValueError(f"Cannot initialize camera: {e}")
    
    def preprocess_frame(self, frame):
        """Preprocess frame for digit detection with improved stability"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply stronger Gaussian blur to reduce noise and improve stability
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Apply adaptive thresholding with more conservative parameters
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 15, 4)
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Additional noise reduction
        kernel_large = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_large)
        
        return gray, thresh
    
    def find_digit_contours(self, thresh):
        """Find contours that might contain digits, limited to 3 with priority for middle positions"""
        # Handle both OpenCV 3.x and 4.x versions
        result = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(result) == 3:
            # OpenCV 3.x returns (image, contours, hierarchy)
            contours = result[1]
        else:
            # OpenCV 4.x returns (contours, hierarchy)
            contours = result[0]
        
        digit_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area < area < self.max_contour_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (digits are usually taller than wide, but 1s can be very narrow)
                aspect_ratio = h / w
                if 0.3 < aspect_ratio < 8.0:  # Expanded range to catch narrow 1s (was 0.5 to 3.0)
                    digit_contours.append((contour, (x, y, w, h)))
        
        # Sort by x-coordinate (left to right order)
        digit_contours.sort(key=lambda item: item[1][0])
        
        # Select only the center digit
        digit_contours = self.select_center_digit(digit_contours, thresh.shape[1])
        
        return digit_contours
    
    def select_center_digit(self, digit_contours, frame_width):
        """Select the single digit closest to the center of the screen"""
        if not digit_contours:
            return []
        
        # Calculate center positions and distances from frame center
        frame_center = frame_width // 2
        
        best_digit = None
        min_distance = float('inf')
        
        for contour, bbox in digit_contours:
            x, y, w, h = bbox
            digit_center = x + w // 2
            distance_from_center = abs(digit_center - frame_center)
            
            if distance_from_center < min_distance:
                min_distance = distance_from_center
                best_digit = (contour, bbox)
        
        # Return only the center-most digit
        return [best_digit] if best_digit else []
    
    def find_closest_tracked_digit(self, bbox):
        """Find the closest tracked digit to the current bounding box"""
        x, y, w, h = bbox
        center_x, center_y = x + w // 2, y + h // 2
        
        closest_id = None
        min_distance = float('inf')
        
        for digit_id, digit_info in self.tracked_digits.items():
            tracked_x, tracked_y = digit_info['center']
            distance = np.sqrt((center_x - tracked_x)**2 + (center_y - tracked_y)**2)
            
            if distance < min_distance and distance < self.spatial_tolerance:
                min_distance = distance
                closest_id = digit_id
        
        return closest_id
    
    def update_tracked_digit(self, digit_id, bbox, digit, confidence):
        """Update tracked digit information"""
        x, y, w, h = bbox
        center = (x + w // 2, y + h // 2)
        
        if digit_id not in self.tracked_digits:
            self.tracked_digits[digit_id] = {
                'center': center,
                'bbox': bbox,
                'predictions': [],
                'confidences': [],
                'stable_digit': None,
                'stable_confidence': 0.0,
                'frames_stable': 0
            }
        
        # Update position
        self.tracked_digits[digit_id]['center'] = center
        self.tracked_digits[digit_id]['bbox'] = bbox
        
        # Add new prediction to history
        predictions = self.tracked_digits[digit_id]['predictions']
        confidences = self.tracked_digits[digit_id]['confidences']
        
        predictions.append(digit)
        confidences.append(confidence)
        
        # Keep only recent history
        if len(predictions) > self.max_history_length:
            predictions.pop(0)
            confidences.pop(0)
        
        # Calculate stable prediction using weighted voting
        self.calculate_stable_prediction(digit_id)
    
    def calculate_stable_prediction(self, digit_id):
        """Calculate stable prediction using weighted voting"""
        digit_info = self.tracked_digits[digit_id]
        predictions = digit_info['predictions']
        confidences = digit_info['confidences']
        
        if not predictions:
            return
        
        # Count votes for each digit, weighted by confidence
        vote_weights = {}
        for i, (digit, conf) in enumerate(zip(predictions, confidences)):
            # Give more weight to recent predictions
            recency_weight = (i + 1) / len(predictions)
            weight = conf * recency_weight
            
            if digit not in vote_weights:
                vote_weights[digit] = 0
            vote_weights[digit] += weight
        
        # Find the digit with highest weighted vote
        best_digit = max(vote_weights.keys(), key=lambda d: vote_weights[d])
        best_weight = vote_weights[best_digit]
        
        # Calculate normalized confidence
        total_weight = sum(vote_weights.values())
        normalized_confidence = best_weight / total_weight if total_weight > 0 else 0
        
        # Check if prediction is stable (same digit for multiple frames)
        recent_predictions = predictions[-3:] if len(predictions) >= 3 else predictions
        is_consistent = len(set(recent_predictions)) == 1 and recent_predictions[0] == best_digit
        
        if is_consistent:
            digit_info['frames_stable'] += 1
        else:
            digit_info['frames_stable'] = 0
        
        digit_info['stable_digit'] = best_digit
        digit_info['stable_confidence'] = normalized_confidence
    
    def capture_frame(self):
        """Capture frame from picamera2"""
        if self.picam2:
            try:
                # Capture frame from picamera2
                frame = self.picam2.capture_array()
                # Convert RGB to BGR for OpenCV compatibility
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return True, frame_bgr
                
            except Exception as e:
                print(f"Error capturing from picamera2: {e}")
                return False, None
        else:
            return False, None
        
    def extract_digit_roi(self, gray, bbox):
        """Extract region of interest containing the digit with special handling for narrow digits like 1"""
        x, y, w, h = bbox
        
        # Add padding around the digit, with extra horizontal padding for narrow digits like 1
        aspect_ratio = h / w
        if aspect_ratio > 4.0:  # Very narrow digit, likely a 1
            padding_horizontal = max(15, w // 2)  # Extra horizontal padding for 1s
            padding_vertical = 15
        else:
            padding_horizontal = 15
            padding_vertical = 15
        
        x = max(0, x - padding_horizontal)
        y = max(0, y - padding_vertical)
        w = min(gray.shape[1] - x, w + 2 * padding_horizontal)
        h = min(gray.shape[0] - y, h + 2 * padding_vertical)
        
        # Extract ROI
        roi = gray[y:y+h, x:x+w]
        
        # Apply additional preprocessing for better digit recognition
        # Apply Gaussian blur to smooth the image
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        
        # Apply binary thresholding
        _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if necessary (digits should be white on black background for MNIST)
        if np.mean(roi) > 127:
            roi = 255 - roi
        
        # Apply morphological operations to clean up the digit
        # Use smaller kernel for narrow digits to preserve thin lines
        if aspect_ratio > 4.0:  # Very narrow digit
            kernel = np.ones((1, 1), np.uint8)  # Smaller kernel for 1s
        else:
            kernel = np.ones((2, 2), np.uint8)
        
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
        
        return roi
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            if hasattr(self, 'picam2') and self.picam2 is not None:
                self.picam2.stop()
                self.picam2.close()
            cv2.destroyAllWindows()
        except Exception:
            # Ignore exceptions during cleanup
            pass
    
    def get_single_prediction(self):
        """Capture a single frame and return the best digit prediction with confidence"""
        if self.picam2 is None:
            raise ValueError("Camera not initialized. Call initialize_camera() first.")
        
        if not DIGIT_RECOGNIZER_AVAILABLE or not self.recognizer:
            return None, 0.0
        
        ret, frame = self.capture_frame()
        if not ret:
            return None, 0.0
        
        # Preprocess frame
        gray, thresh = self.preprocess_frame(frame)
        
        # Find digit contours
        digit_contours = self.find_digit_contours(thresh)
        
        best_prediction = None
        best_confidence = 0.0
        
        # Process each potential digit and find the best one
        for i, (contour, bbox) in enumerate(digit_contours):
            x, y, w, h = bbox
            
            # Extract digit ROI
            digit_roi = self.extract_digit_roi(gray, bbox)
            
            # Recognize digit
            try:
                digit, confidence, debug_image = self.recognizer.predict_digit(digit_roi)
                
                # Keep track of the best prediction
                if confidence > best_confidence:
                    best_prediction = digit
                    best_confidence = confidence
                    
            except Exception as e:
                print(f"Error recognizing digit: {e}")
                continue
        
        return best_prediction, best_confidence
