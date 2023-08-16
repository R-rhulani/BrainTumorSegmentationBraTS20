import numpy as np

# Assuming you have defined your SimpleUNetLayer class and functions
import os
import numpy as np
from unetClass import SimpleUNetLayer
from customDataGenerator import imageLoader

batch_size = 1

val_img_dir = "BraTS2020_TrainingData/input_data_128/val/images/"
val_mask_dir = "BraTS2020_TrainingData/input_data_128/val/masks/"

# Load the trained weights from the numpy array file
loaded_weights = np.load("trained_weights.npz")

# Create instances of SimpleUNetLayer
layer1 = SimpleUNetLayer()
layer2 = SimpleUNetLayer()
layer3 = SimpleUNetLayer()

# Set the loaded weights to the layers
layer1.weights = loaded_weights["layer1_weights"]
layer2.weights = loaded_weights["layer2_weights"]
layer3.weights = loaded_weights["layer3_weights"]

# Now you can use this new instance of the model for predictions
# Assuming you have some input data for prediction, replace this with your actual data
input_data_for_prediction = np.random.randn(128, 128, 128, 3)  # Example input data

img_num = 83

test_img = np.load("BraTS2020_TrainingData/input_data_128/val/images/image_" + str(img_num) + ".npy")

test_mask = np.load("BraTS2020_TrainingData/input_data_128/val/masks/mask_" + str(img_num) + ".npy")
test_mask_argmax = np.argmax(test_mask, axis=3)

# test_img_input = np.expand_dims(test_img, axis=0)

# Perform forward pass to get predictions
predictions = layer3.forward(test_img)

# Print or use the predictions as needed
print("Predictions shape:", predictions.shape)

test_prediction_argmax = np.argmax(predictions, axis=3)[0, :, :]


print(test_prediction_argmax.shape)
print(test_mask_argmax.shape)
print(np.unique(test_prediction_argmax))


# Plot individual slices from test predictions for verification
from matplotlib import pyplot as plt
import random

n_slice=random.randint(0, test_prediction_argmax.shape[1])
depth_slice = 55  # Replace with the appropriate depth slice index
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:, :, depth_slice, 1], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(test_mask_argmax[:, :, depth_slice])
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_prediction_argmax, cmap='gray')  # Display the prediction
plt.show()

