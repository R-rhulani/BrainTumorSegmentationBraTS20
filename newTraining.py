import os
import numpy as np
from unetClass import SimpleUNetLayer
from customDataGenerator import imageLoader
import json
import h5py

# Assuming you've already defined the model, loss function, and other components as shown earlier
# Define the number of classes and other parameters
NUM_CLASSES = 4
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_DEPTH = 128
IMG_CHANNELS = 3
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

train_img_dir = "BraTS2020_TrainingData/input_data_128/train/images/"
train_mask_dir = "BraTS2020_TrainingData/input_data_128/train/masks/"
train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)
train_img_list = sorted(train_img_list)
train_mask_list = sorted(train_mask_list)

# Create instances of SimpleUNetLayer for each layer
layer1 = SimpleUNetLayer()
layer2 = SimpleUNetLayer()
layer3 = SimpleUNetLayer()

# Set previous layer connections
layer1.set_prev_layer(None)  # The first layer has no previous layer
layer2.set_prev_layer(layer1)
layer3.set_prev_layer(layer2)

# Define the number of training samples and batch size
num_train_samples = len(train_img_list)
batch_size = 1  # Adjust this according to your needs

train_data_generator = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)

# Training loop
epochs = 1
learning_rate = 0.001

def categorical_crossentropy(y_true, y_pred):
    epsilon = 1e-10  # Small constant to avoid division by zero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to prevent log(0)
    return -np.sum(y_true * np.log(y_pred))

for epoch in range(epochs):
    total_loss = 0
    batches_per_epoch = 1 # num_train_samples // batch_size

    for _ in range(batches_per_epoch):
        batch_images, batch_masks = next(train_data_generator)

        for i in range(batch_size):
            input_data = batch_images[i]
            target_output = batch_masks[i]

            # Execute the forward pass through the layers
            output = layer3.forward(input_data)

            # Compute loss for the current sample
            loss = categorical_crossentropy(target_output, output)

            # Backpropagation
            gradient = output - target_output

            # Backpropagate through the model layers
            layer3.backward(gradient)

            # Accumulate gradients

        # Update weights using accumulated gradients and learning rate
        layer3.update_weights(learning_rate)

        # Reset gradients for the next batch
        layer1.reset_gradients()
        layer2.reset_gradients()
        layer3.reset_gradients()

        avg_batch_loss = loss / batch_size
        total_loss += avg_batch_loss

    avg_epoch_loss = total_loss / batches_per_epoch
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss}")

    # After training loop
    trained_weights = {
        "layer1_weights": layer1.weights,
        "layer2_weights": layer2.weights,
        "layer3_weights": layer3.weights
    }

    # Save the trained weights to a numpy array file
    np.savez("trained_weights.npz", **trained_weights)

print("Training completed.")
