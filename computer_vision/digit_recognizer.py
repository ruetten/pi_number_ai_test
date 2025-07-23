import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os

class DigitRecognizer:
    def __init__(self):
        self.model = None
        self.model_path = 'digit_model.h5'
        self.load_or_create_model()
        
    def create_model(self):
        """Create and train a simple NN model for digit recognition using MNIST dataset"""
        print("Creating and training digit recognition model...")
        
        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalize pixel values
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Flatten images for simple NN (28*28 = 784 features)
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)
        
        # Convert labels to categorical
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        # Create simple NN model
        # TODO: Copy Neural Network from slides
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(784,)),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10, activation='softmax')
        ])
        
        # Compile model
        model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])        # Train model with slightly more epochs since simple NN might need more training
        print("Training model (this may take a few minutes)...")
        model.fit(x_train, y_train, epochs=10, batch_size=128, 
                 validation_split=0.1, verbose=1)
        
        # Evaluate model
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Save model
        model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        
        return model
    
    def load_or_create_model(self):
        """Load existing model or create new one"""
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
        else:
            self.model = self.create_model()
    
    def preprocess_image(self, image):
        """Preprocess image for digit recognition"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Ensure the image is square by padding with zeros
        h, w = gray.shape
        if h != w:
            # Make square by padding with zeros
            size = max(h, w)
            square = np.zeros((size, size), dtype=gray.dtype)
            start_y = (size - h) // 2
            start_x = (size - w) // 2
            square[start_y:start_y + h, start_x:start_x + w] = gray
            gray = square
        
        # Resize to 28x28 (MNIST size)
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Ensure proper contrast - make sure digit is white on black background
        if np.mean(resized) > 127:
            resized = 255 - resized
        
        # Apply slight Gaussian blur to match MNIST style
        resized = cv2.GaussianBlur(resized, (1, 1), 0)
        # Normalize pixel values to [0, 1] range
        normalized = resized.astype('float32') / 255.0
        
        # Flatten for simple NN input (784 features)
        preprocessed = normalized.reshape(1, 784)
        
        return preprocessed
    def predict_digit(self, image):
        """Predict digit from preprocessed image with optimized inference"""
        preprocessed = self.preprocess_image(image)
        
        # Use predict with minimal verbosity and single sample optimization
        prediction = self.model.predict(preprocessed, verbose=0, batch_size=1)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # Return the preprocessed image for debugging (reshape from flattened)
        debug_image = (preprocessed[0] * 255).astype(np.uint8).reshape(28, 28)
        return digit, confidence, debug_image
