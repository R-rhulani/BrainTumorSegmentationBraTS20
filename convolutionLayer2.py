import numpy as np

def conv3d(x, kernel):
    _, _, _, in_channels = x.shape
    kernel_depth, kernel_height, kernel_width, out_channels = kernel.shape

    result_depth = x.shape[0] - kernel_depth + 1
    result_height = x.shape[1] - kernel_height + 1
    result_width = x.shape[2] - kernel_width + 1

    result = np.zeros((result_depth, result_height, result_width, out_channels))

    for d in range(result_depth):
        for h in range(result_height):
            for w in range(result_width):
                x_slice = x[d:d + kernel_depth, h:h + kernel_height, w:w + kernel_width, :]
                for c in range(out_channels):
                    result[d, h, w, c] = np.sum(x_slice * kernel[:, :, :, c, np.newaxis])

    return result


def conv3d_transpose(x, kernel, strides=(1, 1, 1)):
    kernel_depth, kernel_height, kernel_width, out_channels = kernel.shape
    stride_depth, stride_height, stride_width = strides

    result_depth = x.shape[0] * stride_depth
    result_height = x.shape[1] * stride_height
    result_width = x.shape[2] * stride_width

    result = np.zeros((result_depth, result_height, result_width, out_channels))

    for d in range(result_depth):
        for h in range(result_height):
            for w in range(result_width):
                d_start, d_end = d // stride_depth, d // stride_depth + kernel_depth
                h_start, h_end = h // stride_height, h // stride_height + kernel_height
                w_start, w_end = w // stride_width, w // stride_width + kernel_width

                if (
                    d_end <= x.shape[0]
                    and h_end <= x.shape[1]
                    and w_end <= x.shape[2]
                ):
                    x_slice = x[
                        d_start:d_end:1,
                        h_start:h_end:1,
                        w_start:w_end:1,
                        :32
                    ]

                    # x_slice.shape[-1] == kernel.shape[-2]
                    result[d, h, w, :] += np.sum(x_slice * kernel)

    return result



def concatenate(arrays, axis=-1):
    return np.concatenate(arrays, axis=axis)

def relu(x):
    return np.maximum(0, x)

def max_pooling3d(x, pool_size):
    depth_factor, height_factor, width_factor = pool_size
    d_out = x.shape[0] // depth_factor
    h_out = x.shape[1] // height_factor
    w_out = x.shape[2] // width_factor

    result = np.zeros((d_out, h_out, w_out, x.shape[3]))

    for d in range(d_out):
        for h in range(h_out):
            for w in range(w_out):
                d_start, d_end = d * depth_factor, (d + 1) * depth_factor
                h_start, h_end = h * height_factor, (h + 1) * height_factor
                w_start, w_end = w * width_factor, (w + 1) * width_factor
                result[d, h, w] = np.max(x[d_start:d_end, h_start:h_end, w_start:w_end], axis=(0, 1, 2))

    return result


def upsample(x, kernel, strides=(2, 2, 2), padding='same'):
    result_depth = x.shape[0] * strides[0]
    result_height = x.shape[1] * strides[1]
    result_width = x.shape[2] * strides[2]

    return conv3d_transpose(x, kernel, strides=strides, padding=padding), (result_depth, result_height, result_width)

def upsample_with_padding(x, kernel, target_shape, strides=(2, 2, 2)):
    result_depth, result_height, result_width = target_shape

    padded_depth = x.shape[0] * strides[0] + kernel.shape[0] - strides[0]
    padded_height = x.shape[1] * strides[1] + kernel.shape[1] - strides[1]
    padded_width = x.shape[2] * strides[2] + kernel.shape[2] - strides[2]

    pad_depth = max(0, padded_depth - result_depth)
    pad_height = max(0, padded_height - result_height)
    pad_width = max(0, padded_width - result_width)

    padded_x = np.pad(x, ((0, pad_depth), (0, pad_height), (0, pad_width), (0, 0)), mode='constant')

    return conv3d_transpose(padded_x, kernel, strides=strides)



def upscale(input_array, scale_factor):
    d, h, w, c = input_array.shape
    new_shape = ((d - 1) * scale_factor + 1, (h - 1) * scale_factor + 1, (w - 1) * scale_factor + 1, c)
    output_array = np.zeros(new_shape)
    for dd in range(d):
        for hh in range(h):
            for ww in range(w):
                output_array[dd * scale_factor: (dd + 1) * scale_factor + 1,
                hh * scale_factor: (hh + 1) * scale_factor + 1,
                ww * scale_factor: (ww + 1) * scale_factor + 1, :] = input_array[dd, hh, ww, :]
    return output_array


def dropout(x, rate):
    mask = np.random.binomial(1, rate, size=x.shape)
    return x * mask

# Model architecture
def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    # Build the model
    inputs = np.zeros((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    s = inputs

    # Contraction path
    kernel_initializer = np.random.normal(loc=0.0, scale=np.sqrt(2.0), size=(3, 3, 3, 32)).astype(np.float32)

    c1 = conv3d(s, kernel_initializer)
    c1 = relu(c1)
    c1 = dropout(c1, 0.1)
    c1 = conv3d(c1, kernel_initializer)
    c1 = relu(c1)
    p1 = max_pooling3d(c1, (2, 2, 2))

    c2 = conv3d(p1, kernel_initializer)
    c2 = relu(c2)
    c2 = dropout(c2, 0.1)
    c2 = conv3d(c2, kernel_initializer)
    c2 = relu(c2)
    p2 = max_pooling3d(c2, (2, 2, 2))

    c3 = conv3d(p2, kernel_initializer)
    c3 = relu(c3)
    c3 = dropout(c3, 0.2)
    c3 = conv3d(c3, kernel_initializer)
    c3 = relu(c3)
    p3 = max_pooling3d(c3, (2, 2, 2))

    c4 = conv3d(p3, kernel_initializer)
    c4 = relu(c4)
    c4 = dropout(c4, 0.2)
    c4 = conv3d(c4, kernel_initializer)
    c4 = relu(c4)
    p4 = max_pooling3d(c4, (2, 2, 2))

    c5 = conv3d(p4, kernel_initializer)
    c5 = relu(c5)
    c5 = dropout(c5, 0.3)

    # Expansive path

    target_shape_u6 = (c4.shape[0], c4.shape[1], c4.shape[2])
    u6 = upsample_with_padding(c5, kernel_initializer, target_shape_u6, strides=(2, 2, 2))
    # Add padding to match the dimensions of c4
    pad_depth = c4.shape[0] - u6.shape[0]
    pad_height = c4.shape[1] - u6.shape[1]
    pad_width = c4.shape[2] - u6.shape[2]
    u6 = np.pad(u6, ((0, pad_depth), (0, pad_height), (0, pad_width), (0, 0)), mode='constant')
    c6 = conv3d_transpose(u6, kernel_initializer)
    c6 = relu(c6)
    c6 = dropout(c6, 0.2)
    c6 = conv3d_transpose(c6, kernel_initializer)
    c6 = relu(c6)

    target_shape_u7 = (c3.shape[0], c3.shape[1], c3.shape[2])
    u7 = upsample_with_padding(c6, kernel_initializer, target_shape_u7, strides=(2, 2, 2))
    # Add padding to match the dimensions of c3
    pad_depth = c3.shape[0] - u7.shape[0]
    pad_height = c3.shape[1] - u7.shape[1]
    pad_width = c3.shape[2] - u7.shape[2]
    u7 = np.pad(u7, ((0, pad_depth), (0, pad_height), (0, pad_width), (0, 0)), mode='constant')
    c7 = conv3d_transpose(u7, kernel_initializer)
    c7 = relu(c7)
    c7 = dropout(c7, 0.2)
    c7 = conv3d_transpose(c7, kernel_initializer)
    c7 = relu(c7)

    target_shape_u8 = (c2.shape[0], c2.shape[1], c2.shape[2])
    u8 = upsample_with_padding(c7, kernel_initializer, target_shape_u8, strides=(2, 2, 2))
    # Add padding to match the dimensions of c2
    pad_depth = c2.shape[0] - u8.shape[0]
    pad_height = c2.shape[1] - u8.shape[1]
    pad_width = c2.shape[2] - u8.shape[2]
    u8 = np.pad(u8, ((0, pad_depth), (0, pad_height), (0, pad_width), (0, 0)), mode='constant')
    u8 = concatenate([u8, c2], axis=-1)
    c8 = conv3d_transpose(u8, kernel_initializer)
    c8 = relu(c8)
    c8 = dropout(c8, 0.1)
    c8 = conv3d_transpose(c8, kernel_initializer)
    c8 = relu(c8)

    target_shape_u9 = (c1.shape[0], c1.shape[1], c1.shape[2])
    u9 = upsample_with_padding(c8, kernel_initializer, target_shape_u9, strides=(2, 2, 2))
    # Add padding to match the dimensions of c1
    pad_depth = c1.shape[0] - u9.shape[0]
    pad_height = c1.shape[1] - u9.shape[1]
    pad_width = c1.shape[2] - u9.shape[2]
    u9 = np.pad(u9, ((0, pad_depth), (0, pad_height), (0, pad_width), (0, 0)), mode='constant')
    c9 = conv3d_transpose(u9, kernel_initializer)
    c9 = relu(c9)
    c9 = dropout(c9, 0.1)
    c9 = conv3d_transpose(c9, kernel_initializer)
    c9 = relu(c9)

    outputs = conv3d(c9, np.ones((1, 1, 1, num_classes)))

    return outputs

# Test if everything is working ok.
IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes = 128, 128, 128, 3, 4
model_output = simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes)
print("Model output shape:", model_output.shape)
