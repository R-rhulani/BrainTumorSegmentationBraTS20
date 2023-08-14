import os
import numpy as np
# from convolutionLayer2 import simple_unet_model
from unetClass import SimpleUNetLayer
from customDataGenerator import imageLoader
import random
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

# Initialize the model
# model = simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, NUM_CLASSES)

# model = simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, NUM_CLASSES)

model = SimpleUNetLayer()

# Define the number of training samples and batch size
num_train_samples = len(train_img_list)
batch_size = 16  # Adjust this according to your needs

train_data_generator = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)
kernel_initializer = np.random.normal(loc=0.0, scale=np.sqrt(2.0), size=(3, 3, 3, 32)).astype(np.float32)

# Training loop
epochs = 10
learning_rate = 0.001

def categorical_crossentropy(y_true, y_pred):
    epsilon = 1e-10  # Small constant to avoid division by zero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to prevent log(0)
    return -np.sum(y_true * np.log(y_pred))

for epoch in range(epochs):
    total_loss = 0
    batches_per_epoch = num_train_samples // batch_size

    for _ in range(batches_per_epoch):
        # Get a batch of data from the generator
        batch_images, batch_masks = next(train_data_generator)

        batch_loss = 0

        for i in range(batch_size):
            # Forward pass
            input_data = batch_images[i]
            target_output = batch_masks[i]

            # Forward pass through the model
            output = model.forward(input_data)

            # Compute loss for the current sample
            loss = categorical_crossentropy(target_output, output)
            batch_loss += loss

            # Backpropagation
            gradient = output - target_output

            # Backpropagate through the model layers
            model.backward(gradient, input_data)

            # Update weights using gradients and learning rate
            for layer in model.layers:
                for j in range(len(layer.weights)):
                    weight_gradient = np.dot(layer.input_data.T, layer.gradients[j])
                    layer.weights[j] -= learning_rate * weight_gradient

        avg_batch_loss = batch_loss / batch_size
        total_loss += avg_batch_loss

    avg_epoch_loss = total_loss / batches_per_epoch
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss}")

print("Training completed.")

# for epoch in range(epochs):
#     total_loss = 0
#     batches_per_epoch = num_train_samples // batch_size
#
#     for _ in range(batches_per_epoch):
#         # Get a batch of data from the generator
#         batch_images, batch_masks = next(train_data_generator)
#
#         batch_loss = 0
#
#         for i in range(batch_size):
#             # Forward pass
#             input_data = batch_images[i]
#             target_output = batch_masks[i]
#             output = simple_unet_model(input_data, NUM_CLASSES)
#
#             # Compute loss for the current sample
#             loss = categorical_crossentropy(target_output, output)
#             batch_loss += loss
#
#             # Backpropagation
#             gradient = output - target_output
#
#             # Update weights using gradients
#             for layer in model.layers:
#                 for j in range(len(layer.weights)):
#                     weight_gradient = np.dot(layer.input.T, gradient)
#                     layer.weights[j] -= learning_rate * weight_gradient
#                     gradient = np.dot(gradient, layer.weights[j].T)
#
#         avg_batch_loss = batch_loss / batch_size
#         total_loss += avg_batch_loss
#
#     avg_epoch_loss = total_loss / batches_per_epoch
#     print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss}")
#
# print("Training completed.")
