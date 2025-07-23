from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
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
    Show visual predictions for high school students
    """
    print("\n=== Visual Predictions ===")
    
    # Get predictions for first few test images
    predictions = model.predict(x_test[:num_images].reshape(-1, 784), verbose=0)
    predicted_digits = np.argmax(predictions, axis=1)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    fig.suptitle('Neural Network Predictions vs Actual Digits', fontsize=16)
    
    for i in range(num_images):
        row = i // 3
        col = i % 3
        
        # Show the original 28x28 image
        axes[row, col].imshow(x_test[i], cmap='gray')
        axes[row, col].set_title(f'Actual: {y_test[i]}\nPredicted: {predicted_digits[i]}', 
                                fontsize=12)
        axes[row, col].axis('off')
        
        # Color the title based on correctness
        if y_test[i] == predicted_digits[i]:
            axes[row, col].title.set_color('green')  # Correct prediction
        else:
            axes[row, col].title.set_color('red')    # Wrong prediction
    
    plt.tight_layout()
    plt.show()
    
    # Print accuracy summary
    correct = sum(1 for i in range(num_images) if y_test[i] == predicted_digits[i])
    print(f"Correct predictions: {correct}/{num_images}")

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
