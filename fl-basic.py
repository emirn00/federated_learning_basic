import numpy as np
import tensorflow as tf
import sklearn
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split

print(tf.__version__)
print(sklearn.__version__)
print("imports done ")

def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

def split_data_among_devices(x, y, num_devices=5):
    device_data = []
    data_per_device = len(x) // num_devices
    for i in range(num_devices):
        start = i * data_per_device
        end = (i + 1) * data_per_device
        device_data.append((x[start:end], y[start:end]))
    return device_data

def create_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_on_device(model, x, y, epochs=1):
    model.fit(x, y, epochs=epochs, verbose=0)
    return model.get_weights()

def federated_averaging(weights_list):
    average_weights = []
    for weights in zip(*weights_list):
        average_weights.append(np.mean(weights, axis=0))
    return average_weights

def federated_learning_simulation():
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    device_data = split_data_among_devices(x_train, y_train, num_devices=5)

    global_model = create_model()
    num_rounds = 5

    for round_num in range(num_rounds):
        print(f"\nRound {round_num + 1}/{num_rounds}")
        local_weights = []

        for device_num, (x_device, y_device) in enumerate(device_data):
            print(f"  Training on device {device_num + 1}...")
            local_model = create_model()
            local_model.set_weights(global_model.get_weights())
            local_weights.append(train_on_device(local_model, x_device, y_device))

        new_weights = federated_averaging(local_weights)
        global_model.set_weights(new_weights)

        loss, accuracy = global_model.evaluate(x_test, y_test, verbose=0)
        print(f"  Global model accuracy: {accuracy:.4f}")


#test simulation
federated_learning_simulation()