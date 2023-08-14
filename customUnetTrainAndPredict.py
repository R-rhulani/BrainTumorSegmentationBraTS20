import os
import numpy as np
# from convolutionLayer2 import simple_unet_model
from unetClass import SimpleUNetLayer, simple_unet_model
from customDataGenerator import imageLoader
import random
import h5py

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, model):
        # Update model parameters using gradients and learning rate
        for layer in model.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'gradients'):
                for i in range(len(layer.weights)):
                    layer.weights[i] -= self.learning_rate * layer.gradients[i]

# Define a categorical cross-entropy loss function
def categorical_crossentropy(y_true, y_pred):
    epsilon = 1e-10  # Small constant to avoid division by zero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to prevent log(0)
    return -np.sum(y_true * np.log(y_pred))

# Training parameters
learning_rate = 0.001
num_epochs = 10
batch_size = 1

# Create an instance of the model
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes = 128, 128, 128, 3, 4
kernel_initializer = np.random.normal(loc=0.0, scale=np.sqrt(2.0), size=(3, 3, 3, 32)).astype(np.float32)
model_output = simple_unet_model(np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS), dtype=np.float32),
                                               kernel_initializer, num_classes)

# Define data directories and lists
train_img_dir = "BraTS2020_TrainingData/input_data_128/train/images/"
train_mask_dir = "BraTS2020_TrainingData/input_data_128/train/masks/"
train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)

# Initialize data generator
train_data_generator = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)

# Define an optimizer (for example, stochastic gradient descent)
optimizer = SGD(learning_rate=learning_rate)

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Generate and process batches of training data
    total_loss = 0
    num_batches = len(train_img_list) // batch_size
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = (batch_idx + 1) * batch_size

        batch_imgs = []
        batch_masks = []
        for idx in range(batch_start, batch_end):
            batch_imgs, batch_masks = next(train_data_generator)  # Get a batch from the generator

            # Forward pass
            predictions = simple_unet_model(batch_imgs, kernel_initializer, num_classes)

            # Compute loss
            loss = categorical_crossentropy(batch_masks, predictions)
            total_loss += loss

            # Backpropagation (Gradient descent update)
            gradient = -(batch_masks / predictions)  # Example gradient computation
        model_output.backward(gradient)  # Update gradients for each layer
        optimizer.step(model_output)  # Perform gradient descent update using optimizer

    # Print average loss for the epoch
    average_loss = total_loss / num_batches
    print(f"Average loss: {average_loss:.4f}")

print("Training complete")

# Save the trained model to an HDF5 file
with h5py.File('brats_3d.hdf5', 'w') as file:
    model_output.save_weights(file)

print("Model saved as 'brats_3d.hdf5'")
