from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def create_neural_network():
    """
    Create a very simple neural network for digit recognition
    """
    print("Creating neural network...")
    
    # Load MNIST dataset (handwritten digits 0-9)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize data (0-255 becomes 0-1)
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Flatten images (28x28 becomes 784)
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    
    # Create simple neural network
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),  # Hidden layer
        layers.Dense(10, activation='softmax')                     # Output layer
    ])
    
    # Configure model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train model
    print("Training...")
    model.fit(x_train, y_train, epochs=3, verbose=1)
    
    # Test model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Accuracy: {test_accuracy:.3f}")
    
    return model

def show_predictions(model, x_test, y_test, num_images=6):
    """
    Show text-based predictions for high school students (no GUI needed)
    """
    print("\n=== Visual Predictions ===")
    
    # Get predictions for first few test images
    predictions = model.predict(x_test[:num_images].reshape(-1, 784), verbose=0)
    predicted_digits = np.argmax(predictions, axis=1)
    
    def image_to_ascii(image, width=14):
        """Convert 28x28 image to ASCII art"""
        # Resize to smaller for display
        resized = image[::2, ::2]  # Take every 2nd pixel (14x14)
        ascii_chars = " .:-=+*#%@"
        ascii_img = ""
        for row in resized:
            for pixel in row:
                char_index = int(pixel * (len(ascii_chars) - 1))
                ascii_img += ascii_chars[char_index] + " "
            ascii_img += "\n"
        return ascii_img
    
    # Show each prediction
    for i in range(num_images):
        print(f"\n--- Image {i+1} ---")
        print(f"Actual digit: {y_test[i]}")
        print(f"Predicted digit: {predicted_digits[i]}")
        
        # Show correctness
        if y_test[i] == predicted_digits[i]:
            print("✓ CORRECT!")
        else:
            print("✗ WRONG")
        
        # Show ASCII art of the digit
        print("\nHow the digit looks:")
        print(image_to_ascii(x_test[i]))
    
    # Print accuracy summary
    correct = sum(1 for i in range(num_images) if y_test[i] == predicted_digits[i])
    print(f"\nOverall: {correct}/{num_images} correct predictions ({correct/num_images*100:.1f}%)")

def main():
    """
    Run the neural network
    """
    print("=== Simple Neural Network ===")
    
    # Create and train model
    model = create_neural_network()
    
    # Load test data for visualization
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test / 255.0  # Normalize for display
    
    # Show visual predictions
    show_predictions(model, x_test, y_test)
    
    print("Done!")

if __name__ == "__main__":
    main()
