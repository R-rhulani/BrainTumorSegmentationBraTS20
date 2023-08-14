import numpy as np

def conv3d(input_volume, kernel):
    kernel_size = kernel.shape
    volume_size = input_volume.shape
    output_volume = np.zeros_like(input_volume)
    kernel_center = np.array([k // 2 for k in kernel_size])

    for z in range(kernel_center[0], volume_size[0] - kernel_center[0]):
        for y in range(kernel_center[1], volume_size[1] - kernel_center[1]):
            for x in range(kernel_center[2], volume_size[2] - kernel_center[2]):
                region = input_volume[z - kernel_center[0]:z + kernel_center[0] + 1,
                                      y - kernel_center[1]:y + kernel_center[1] + 1,
                                      x - kernel_center[2]:x + kernel_center[2] + 1]
                output_volume[z, y, x] = np.sum(region * kernel)

    return output_volume

def relu(volume):
    return np.maximum(0, volume)

def max_pooling3d(volume, pool_size):
    depth, height, width, channels = volume.shape
    pooled_depth = depth // pool_size[0]
    pooled_height = height // pool_size[1]
    pooled_width = width // pool_size[2]
    pooled_volume = np.zeros((pooled_depth, pooled_height, pooled_width, channels))
    for d in range(pooled_depth):
        for h in range(pooled_height):
            for w in range(pooled_width):
                pool_region = volume[d * pool_size[0] : (d + 1) * pool_size[0],
                                     h * pool_size[1] : (h + 1) * pool_size[1],
                                     w * pool_size[2] : (w + 1) * pool_size[2]]
                pooled_volume[d, h, w] = np.max(pool_region, axis=(0, 1, 2))
    return pooled_volume

def conv3d_transpose(input_volume, kernel):
    kernel_size = kernel.shape
    input_size = input_volume.shape
    output_size = (input_size[0] + kernel_size[0] - 1,
                   input_size[1] + kernel_size[1] - 1,
                   input_size[2] + kernel_size[2] - 1,
                   input_size[3])

    output_volume = np.zeros(output_size)

    for z in range(input_size[0]):
        for y in range(input_size[1]):
            for x in range(input_size[2]):
                region = input_volume[z, y, x]
                output_volume[z:z + kernel_size[0], y:y + kernel_size[1], x:x + kernel_size[2]] += region * kernel

    return output_volume

# def upsample_with_padding(volume, target_shape):
#     original_shape = volume.shape
#     upscaled_volume = np.zeros(target_shape)
#
#     for d in range(original_shape[0]):
#         for h in range(original_shape[1]):
#             for w in range(original_shape[2]):
#                 upscaled_volume[d, h, w] = volume[d // (target_shape[0] // original_shape[0]),
#                                                   h // (target_shape[1] // original_shape[1]),
#                                                   w // (target_shape[2] // original_shape[2])]
#
#     return upscaled_volume

def upsample_with_padding(x, kernel, target_shape, strides=(2, 2, 2), padding='same'):
    result_depth, result_height, result_width = target_shape

    padded_depth = x.shape[0] * strides[0] + kernel.shape[0] - strides[0]
    padded_height = x.shape[1] * strides[1] + kernel.shape[1] - strides[1]
    padded_width = x.shape[2] * strides[2] + kernel.shape[2] - strides[2]

    pad_depth = max(0, padded_depth - result_depth)
    pad_height = max(0, padded_height - result_height)
    pad_width = max(0, padded_width - result_width)

    padded_x = np.pad(x, ((0, pad_depth), (0, pad_height), (0, pad_width), (0, 0)), mode='constant')

    return padded_x

def conv3d_transpose(input_volume, kernel):
    kernel_size = kernel.shape
    input_size = input_volume.shape
    output_size = (input_size[0] + kernel_size[0] - 1,
                   input_size[1] + kernel_size[1] - 1,
                   input_size[2] + kernel_size[2] - 1,
                   input_size[3])

    output_volume = np.zeros(output_size)

    for z in range(input_size[0]):
        for y in range(input_size[1]):
            for x in range(input_size[2]):
                region = input_volume[z, y:y + kernel_size[1], x:x + kernel_size[2]]
                output_volume[z:z + kernel_size[0], y:y + kernel_size[1], x:x + kernel_size[2]] += region * kernel

    return output_volume


def simple_unet_model(input_volume, num_classes):
    # Contraction path
    kernel_initializer = np.random.normal(loc=0.0, scale=np.sqrt(2.0), size=(3, 3, 3, 32)).astype(np.float32)
    s = input_volume

    c1 = conv3d(s, kernel_initializer)
    c1 = relu(c1)
    p1 = max_pooling3d(c1, (2, 2, 2))

    kernel_initializer2 = np.random.normal(loc=0.0, scale=np.sqrt(2.0), size=(3, 3, 3, 64)).astype(np.float32)
    c2 = conv3d(p1, kernel_initializer2)
    c2 = relu(c2)
    p2 = max_pooling3d(c2, (2, 2, 2))

    kernel_initializer3 = np.random.normal(loc=0.0, scale=np.sqrt(2.0), size=(3, 3, 3, 128)).astype(np.float32)
    c3 = conv3d(p2, kernel_initializer3)
    c3 = relu(c3)
    p3 = max_pooling3d(c3, (2, 2, 2))

    kernel_initializer4 = np.random.normal(loc=0.0, scale=np.sqrt(2.0), size=(3, 3, 3, 256)).astype(np.float32)
    c4 = conv3d(p3, kernel_initializer4)
    c4 = relu(c4)
    p4 = max_pooling3d(c4, (2, 2, 2))

    kernel_initializer5 = np.random.normal(loc=0.0, scale=np.sqrt(2.0), size=(3, 3, 3, 512)).astype(np.float32)
    c5 = conv3d(p4, kernel_initializer5)
    c5 = relu(c5)
    p5 = max_pooling3d(c5, (2, 2, 2))

    # Expansive path
    # u6 = upsample_with_padding(c5, c4.shape)

    target_shape_u6 = (c4.shape[0], c4.shape[1], c4.shape[2])
    u6 = upsample_with_padding(c5, kernel_initializer, target_shape_u6)
    # Add padding to match the dimensions of c4
    pad_depth = c4.shape[0] - u6.shape[0]
    pad_height = c4.shape[1] - u6.shape[1]
    pad_width = c4.shape[2] - u6.shape[2]
    u6 = np.pad(u6, ((0, pad_depth), (0, pad_height), (0, pad_width), (0, 0)), mode='constant')

    c6 = conv3d_transpose(u6, kernel_initializer2)
    c6 = relu(c6)

    u7 = upsample_with_padding(c6, c3.shape)
    c7 = conv3d_transpose(u7, kernel_initializer3)
    c7 = relu(c7)

    u8 = upsample_with_padding(c7, c2.shape)
    c8 = conv3d_transpose(u8, kernel_initializer4)
    c8 = relu(c8)

    u9 = upsample_with_padding(c8, c1.shape)
    c9 = conv3d_transpose(u9, kernel_initializer5)
    c9 = relu(c9)

    outputs = conv3d(c9, np.ones((1, 1, 1, num_classes)))

    return outputs

# Example usage
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS = 128, 128, 64, 1
num_classes = 2

# Generate a dummy 3D volume for input
input_volume = np.random.rand(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS)

# Create the model and get the output
model_output = simple_unet_model(input_volume, num_classes)
print("Input volume shape:", input_volume.shape)
print("Model output shape:", model_output.shape)
