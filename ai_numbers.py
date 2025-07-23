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
    
    # Get predictions for all test images to find correct and incorrect ones
    all_predictions = model.predict(x_test.reshape(-1, 784), verbose=0)
    all_predicted_digits = np.argmax(all_predictions, axis=1)
    
    # Find indices of correct and incorrect predictions
    correct_indices = np.where(all_predicted_digits == y_test)[0]
    incorrect_indices = np.where(all_predicted_digits != y_test)[0]
    
    # Select images to show: include at least one wrong prediction if available
    selected_indices = []
    
    # Add at least one incorrect prediction if any exist
    if len(incorrect_indices) > 0:
        selected_indices.extend(incorrect_indices[:2])  # Show up to 2 wrong ones
    
    # Fill the rest with correct predictions
    remaining_slots = num_images - len(selected_indices)
    if remaining_slots > 0 and len(correct_indices) > 0:
        selected_indices.extend(correct_indices[:remaining_slots])
    
    # If we still need more images, just use the first ones
    while len(selected_indices) < num_images:
        selected_indices.append(len(selected_indices))
    
    # Limit to available images
    selected_indices = selected_indices[:min(num_images, len(x_test))]
    
    # Get predictions for selected images
    predictions = all_predictions[selected_indices]
    predicted_digits = all_predicted_digits[selected_indices]
    
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
    for i, idx in enumerate(selected_indices):
        print(f"\n--- Image {i+1} ---")
        print(f"Actual digit: {y_test[idx]}")
        print(f"Predicted digit: {predicted_digits[i]}")
        
        # Show correctness
        if y_test[idx] == predicted_digits[i]:
            print("✓ CORRECT!")
        else:
            print("✗ WRONG")
        
        # Show ASCII art of the digit
        print("\nHow the digit looks:")
        print(image_to_ascii(x_test[idx]))
    
    # Print accuracy summary
    correct = sum(1 for i, idx in enumerate(selected_indices) if y_test[idx] == predicted_digits[i])
    print(f"\nOverall: {correct}/{len(selected_indices)} correct predictions ({correct/len(selected_indices)*100:.1f}%)")

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
