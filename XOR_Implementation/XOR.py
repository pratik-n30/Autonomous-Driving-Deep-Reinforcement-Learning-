import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

class NeuralNetwork:
    """
    A general-purpose neural network class that can be configured with a
    flexible number of layers and neurons.
    """

    def __init__(self, layer_dims, learning_rate=0.1):
        """
        Initialize the neural network with a given architecture.

        Args:
            layer_dims (list): A list of integers representing the number of neurons
                               in each layer, starting with the input layer.
                               Example: [2, 4, 1] for a 2-4-1 network.
            learning_rate (float): The learning rate for gradient descent.
        """
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.parameters = {}
        
        # History tracking for metrics
        self.train_costs = []
        self.test_costs = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.epoch_history = []

        self._initialize_parameters()

    def _initialize_parameters(self):
        """
        Initializes the weights (W) and biases (b) for each layer.
        """
        for l in range(1, len(self.layer_dims)):
            self.parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
            self.parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))

    # --- Activation Functions and their Derivatives ---
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _relu(self, z):
        return np.maximum(0, z)

    def _sigmoid_backward(self, dA, activation_cache):
        s = self._sigmoid(activation_cache)
        dZ = dA * s * (1 - s)
        return dZ

    def _relu_backward(self, dA, activation_cache):
        dZ = np.array(dA, copy=True)
        dZ[activation_cache <= 0] = 0
        return dZ
    
    

    # --- Core Network Logic ---
    def forward_propagation(self, X):
        """
        Implements forward propagation for the entire network.
        """
        caches = []
        A = X
        L = len(self.parameters) // 2

        for l in range(1, L):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(W, A_prev) + b
            A = self._relu(Z)
            cache = (A_prev, W, b, Z)
            caches.append(cache)

        A_prev = A
        W = self.parameters[f'W{L}']
        b = self.parameters[f'b{L}']
        Z = np.dot(W, A_prev) + b
        AL = self._sigmoid(Z)
        cache = (A_prev, W, b, Z)
        caches.append(cache)

        return AL, caches

    def compute_cost(self, AL, Y):
        """
        Computes the binary cross-entropy cost.
        """
        m = Y.shape[1]
        epsilon = 1e-8
        AL = np.clip(AL, epsilon, 1 - epsilon)
        cost = -(1/m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
        return np.squeeze(cost)

    def backward_propagation(self, AL, Y, caches):
        """
        Implements backward propagation for the entire network.
        """
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[L-1]
        A_prev, W, b, Z = current_cache
        dZ = self._sigmoid_backward(dAL, Z)
        grads[f'dW{L}'] = (1/m) * np.dot(dZ, A_prev.T)
        grads[f'db{L}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            A_prev, W, b, Z = current_cache
            dZ = self._relu_backward(dA_prev, Z)
            grads[f'dW{l+1}'] = (1/m) * np.dot(dZ, A_prev.T)
            grads[f'db{l+1}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = np.dot(W.T, dZ)

        return grads

    def update_parameters(self, grads):
        """
        Updates the parameters using gradient descent.
        """
        L = len(self.parameters) // 2
        for l in range(L):
            self.parameters[f'W{l+1}'] -= self.learning_rate * grads[f'dW{l+1}']
            self.parameters[f'b{l+1}'] -= self.learning_rate * grads[f'db{l+1}']

    def train(self, X_train, Y_train, X_test, Y_test, epochs, print_cost=True, print_interval=1000):
        """
        Trains the neural network and tracks train/test metrics.
        """
        for i in range(epochs):
            # Forward, cost, backward, update
            AL, caches = self.forward_propagation(X_train)
            cost = self.compute_cost(AL, Y_train)
            grads = self.backward_propagation(AL, Y_train, caches)
            self.update_parameters(grads)

            # Record and print metrics
            if i % print_interval == 0 or i == epochs - 1:
                # Training metrics
                train_cost = cost
                train_accuracy = self.compute_accuracy(X_train, Y_train)
                
                # Testing metrics
                AL_test, _ = self.forward_propagation(X_test)
                test_cost = self.compute_cost(AL_test, Y_test)
                test_accuracy = self.compute_accuracy(X_test, Y_test)

                # Store history
                self.epoch_history.append(i)
                self.train_costs.append(train_cost)
                self.test_costs.append(test_cost)
                self.train_accuracies.append(train_accuracy)
                self.test_accuracies.append(test_accuracy)

                if print_cost:
                    print(f"Epoch {i:5d} | Train Cost: {train_cost:.6f} | Train Acc: {train_accuracy:6.2f}% | Test Cost: {test_cost:.6f} | Test Acc: {test_accuracy:6.2f}%")

    def predict(self, X):
        """
        Makes predictions on new data.
        """
        AL, _ = self.forward_propagation(X)
        predictions = (AL > 0.5).astype(int)
        return predictions

    def compute_accuracy(self, X, Y):
        """
        Computes the accuracy of the model.
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == Y) * 100
        return accuracy

    def plot_training_curves(self):
        """
        Plots the training and test loss and accuracy curves.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot loss curves
        ax1.plot(self.epoch_history, self.train_costs, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(self.epoch_history, self.test_costs, 'r-', label='Test Loss', linewidth=2)
        ax1.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epochs', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Plot accuracy curves
        ax2.plot(self.epoch_history, self.train_accuracies, 'g-', label='Training Accuracy', linewidth=2)
        ax2.plot(self.epoch_history, self.test_accuracies, 'orange', linestyle='-', label='Test Accuracy', linewidth=2)
        ax2.set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epochs', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_ylim(0, 105)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# --- Example Usage: Solving the XOR Problem ---
def create_xor_dataset():
    """Creates the XOR dataset."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
    Y = np.array([[0, 1, 1, 0]])
    return X, Y

def main():
    """Main function to demonstrate the general-purpose neural network."""
    print("--- General-Purpose Neural Network ---")
    
    # 1. Create the dataset
    X_train, Y_train = create_xor_dataset()
    # For this simple problem, we'll use the same data for testing
    X_test, Y_test = X_train, Y_train

    # 2. Define the network architecture
    layer_architecture = [2, 4, 1]
    print(f"\nNetwork Architecture: {layer_architecture}")

    # 3. Initialize and train the network
    nn = NeuralNetwork(layer_dims=layer_architecture, learning_rate=1.0)
    print("\nTraining the network...")
    nn.train(X_train, Y_train, X_test, Y_test, epochs=5000, print_interval=100)

    # 4. Evaluate the model
    print("\n--- Training Complete ---")
    final_accuracy = nn.compute_accuracy(X_test, Y_test)
    print(f"Final Test Accuracy on XOR data: {final_accuracy:.2f}%")

    # 5. Make predictions
    print("\nPredictions:")
    predictions = nn.predict(X_train)
    for i in range(X_train.shape[1]):
        print(f"Input: {X_train[:, i]} -> Prediction: {predictions[0, i]}")
        
    # 6. Plot the training curves
    print("\nPlotting training curves...")
    nn.plot_training_curves()

if __name__ == "__main__":
    main()
