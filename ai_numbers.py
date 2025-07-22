from tensorflow import keras
from tensorflow.keras import layers

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

def main():
    """
    Run the neural network
    """
    print("=== Simple Neural Network ===")
    
    # Create and train model
    model = create_neural_network()
    
    print("Done!")

if __name__ == "__main__":
    main()
