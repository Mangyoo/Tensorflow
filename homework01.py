import math
import numpy as np
import sklearn
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

def sigmoid(x):
    return 1/(1 + math.e ** (-x))
def sigmoid_derivative(x):
    return x * (1-x)
def softmax(x):
    #assert len(x.shape) == 2

    #print(x.shape)
    #exit()
    #s = np.max(x, axis=0)
    #s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(x) # - s
    #print(e_x)
    #exit()
    div = np.sum(e_x, axis=0)
    #div = div[:, np.newaxis] # dito
    return e_x / div
import numpy as np

def cross_entropy(input, target):
    # Ensure that the input and target have the same length
    #assert len(input) == len(target)
    n = len(input)
    loss = - np.sum(target * np.log(input))/n
    return loss
 


class Perceptron():

    def __init__(self,number_inputs, use_bias=True):
        self.number_inputs = number_inputs
        self.use_bias = use_bias
        self.weights = [0.] * self.number_inputs
        self.bias = 0.

    
    def set_weights(self,weights, bias):
        self.weights = weights
        self.bias = bias 
        return None

    def call(self,x):
        y = 0.
        for i, w in zip(x, self.weights):
            y = y + i*w
        if self.use_bias:
            y = y + self.bias
        y = sigmoid(y)
        return y
    def forward(x):
        x.shape =(64,10)

def sigmoid_np(x):
    return 1/(1 + np.exp(-x))

class MLP_layer():
    def __init__(self, num_inputs, num_units, activation_function):
        self.num_inputs = num_inputs
        self.num_units = num_units
        #self.weights = np.zeros(self.num_inputs + self.num_units)
        self.weights = np.random.normal(0, 0.2, size=(self.num_units, self.num_inputs))
        self.bias = np.zeros((self.num_units,))
        self.activation_function = activation_function
        

    def set_weights(self,weights, bias):
        self.weights = weights
        self.bias = bias

    def call(self,x):
        #x.shape: (num_inputs, 1)
       # print(x)
       # print(self.weights)
        #exit()
        pre_activation = self.weights @ x 
        activations = self.activation_function(pre_activation) + np.transpose(self.bias)

        return activations
    
    def backward(self, error_signal, preactivation):
     # Step 1: Calculate the gradient with respect to the pre-activation
     delta_activation = error_signal * sigmoid_derivative(self.activation_function(preactivation))

     # Step 2: Calculate the gradient with respect to the weights
     delta_weights = error_signal.T @ preactivation
     return delta_activation, delta_weights

class MLP():
    def __init__(self, layer_sizes, activation_functions, learning_rate=0.01):
        assert len(layer_sizes) == len(activation_functions) + 1  # Check that there is one activation function per layer

        self.layers = []
        self.learning_rate = learning_rate

        for i in range(len(layer_sizes) - 1):
            layer = MLP_layer(layer_sizes[i], layer_sizes[i + 1], activation_functions[i])
            self.layers.append(layer)

    #def set_weights(self, weights_list, biases_list):
        #for i, layer in enumerate(self.layers):
            #layer.set_weights(weights_list[i], biases_list[i])

    def forward(self, x):
        for layer in self.layers:
            x = layer.call(x)
        return x
    
    def backward(self, x, target):
        # Step 1: Initialize empty dictionaries to store activations, pre-activations, and weight gradients.
        activations = {}
        preactivations = {}
        weight_gradients = [{} for _ in range(len(self.layers))]

        # Step 2: Forward pass and populate dictionaries with activations and pre-activations.
        activations[0] = x  # Initial activation
        for layer_index, layer in enumerate(self.layers):
            preactivations[layer_index] = layer.weights @ activations[layer_index]
            activations[layer_index + 1] = layer.call(activations[layer_index])

        # Step 3: Calculate the error signal for the output layer (CCE Loss backward).
        output_layer_index = len(self.layers) - 1
        error_signal = activations[output_layer_index + 1] - target

        # Step 4: Backpropagate through MLP layers and calculate gradients.
        for layer_index in range(output_layer_index, -1, -1):
            layer = self.layers[layer_index]
            delta_activation, delta_weights = layer.backward(error_signal, preactivations[layer_index])

            weight_gradients[layer_index] = delta_weights
            error_signal = delta_activation

        # Step 5: Update weights of each MLP layer.
        for layer_index, layer in enumerate(self.layers):
            layer_weights = layer.weights
            layer_weights -= self.learning_rate * weight_gradients[layer_index]

        # You may also return the error signal for further backpropagation if needed.
        return error_signal
    
    def train_mlp(mlp, data, labels, learning_rate, batch_size, num_epochs):
    # Initialize lists to store loss and accuracy for each epoch.
     losses = []
     accuracies = []

     for epoch in range(num_epochs):
        # Shuffle the training data and labels for each epoch.
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]

        cumulative_loss = 0
        correct_predictions = 0

        for i in range(0, len(data), batch_size):
            # Create mini-batches.
            batch_data = data[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]

            for j in range(len(batch_data)):
                # Forward pass.
                predictions = mlp.forward(batch_data[j])

                # Compute CCE loss.
                loss = cross_entropy(predictions, batch_labels[j])
                cumulative_loss += loss

                # Backward pass to update weights with the learning rate.
                error_signal = predictions - batch_labels[j]
                mlp.backward(batch_data[j], batch_labels[j])

                # Check for correct predictions.
                if np.argmax(predictions) == np.argmax(batch_labels[j]):
                    correct_predictions += 1

        # Calculate average loss and accuracy for this epoch.
        average_loss = cumulative_loss / len(data)
        accuracy = correct_predictions / len(data) * 100.0

        # Store loss and accuracy for plotting.
        losses.append(average_loss)
        accuracies.append(accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")

     # Plot average loss and accuracy.
     plt.figure(figsize=(10, 5))
     plt.subplot(1, 2, 1)
     plt.plot(range(1, num_epochs + 1), losses)
     plt.xlabel('Epoch')
     plt.ylabel('Average Loss')
     plt.title('Average Loss vs. Epoch')

     plt.subplot(1, 2, 2)
     plt.plot(range(1, num_epochs + 1), accuracies)
     plt.xlabel('Epoch')
     plt.ylabel('Accuracy (%)')
     plt.title('Accuracy vs. Epoch')

     plt.show()
    
    

def one_hot_encode(target_digit, num_classes=10):
    # Create a one-hot encoded vector with all zeros
    one_hot_vector = np.zeros(num_classes)
    # Set the element corresponding to the target digit to 1
    one_hot_vector[target_digit] = 1
    return one_hot_vector

def generate_samples(inputs, targets, sample_size):
    sample = []
    num_samples = len(inputs)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for index in indices[:sample_size]:
        sample.append((inputs[index], targets[index]))

    sample_array = np.array(sample, dtype=object)
    print(sample_array.shape)
    return sample_array




def dataprep():
    digits = load_digits()
    data, target = load_digits(return_X_y=True)
    images = digits.images
    first_image = images[0]
    second_image = images[1]
    
    n_rows = 2
    n_cols = 5  # You can adjust the number of columns as needed

    # Plot the input images
    plt.figure(figsize=(12, 6))

    for i in range(n_rows * n_cols):
     plt.subplot(n_rows, n_cols, i + 1)
     plt.imshow(data[i].reshape(8, 8), cmap="gray")
     plt.title(f"Digit: {target[i]}")
     plt.axis("off")
    #plt.show()
    
    data = data / 16.0
    print(data[1,:])
    print(target)
    encoder = OneHotEncoder(handle_unknown= 'ignore')
    target_encoded = []
    #target_encoded = encoder.fit_transform(target.reshape(-1,1))
    #print(target_encoded)
    for i in target-1:
        target_encoded.append(one_hot_encode(target[i]))
    #print(target_encoded)
    data_generator = generate_samples(data, target_encoded, 10)
    #print(data_generator)
    #exit()
    #for input_data, target_data in data_generator:
     #print(f'Input: {input_data}, Target: {target_data}')
    #sample = softmax(data_generator[0][0])
    #print(sample) 
    return data_generator
    


def main():
    data = dataprep()

    weights_list = [0.1,0.2,0.3,0.5,0.2,0.2, 0.8, 0.187, 0.1,1, 0.1,0.2,0.3,0.5,0.2,0.2, 0.8, 0.187, 0.1,1, 0.1,0.2,0.3,0.5,0.2,0.2, 0.8, 0.187, 0.1,1, 0.1,0.2,0.3,0.5,0.2,0.2, 0.8, 0.187, 0.1,1, 0.1,0.2,0.3,0.5,0.2,0.2, 0.8, 0.187, 0.1,1, 0.1,0.2,0.3,0.5,0.2,0.2, 0.8, 0.187, 0.1,1, 1,1,1,1]
    bias_list = [1,2,3,4,5,6,7,8,9,0]
    layer_sizes = [64,10]
    activation_functions = [softmax]
    test = MLP(layer_sizes, activation_functions)
    #test.set_weights(weights_list, bias_list)
    input = test.forward(data[0][0])
    print(test.forward(data[0][0]))
    print(cross_entropy(input, data[0][1]))
    #activation - targets, error für backpropagation = initial error signal

    all_predictions = []
    all_error_signals = []

    for input_data, target_data in data:
     # Forward pass to obtain predictions
     predictions = test.forward(input_data)
     all_predictions.append(predictions)

     # Calculate the error signal (assuming you have the true target_data)
     error_signal = predictions - target_data
     all_error_signals.append(error_signal)
    print(all_predictions[0])


    train_data = dataprep()
    train_inputs = np.array([sample[0] for sample in train_data])
    train_targets = np.array([sample[1] for sample in train_data])

    versuch = MLP(layer_sizes, activation_functions)
    versuch.train_mlp(train_inputs, train_targets, 0.01, 32, 10)
    test_data = generate_samples(train_inputs, train_targets, 5)  # Change 5 to the number of test samples you want
    for test_sample in test_data:
        test_input, test_target = test_sample
        test_input = test_input.reshape(-1, 1)  # Ensure correct input shape
        prediction = versuch.forward(test_input)
        print(f"Target: {test_target}, Prediction: {prediction}")
    


if __name__ == "__main__":
    main()



def backward_sigmoid(preactivation, activation, error_signal):
    delta = error_signal * sigmoid_derivative(activation)
    return delta 

def weights_backward(error_signal, preactivation):
    delta_weights = error_signal.T @ preactivation
    return delta_weights

def input_backward(weights, error_signal):
    delta_input = error_signal @ weights
    return delta_input 

