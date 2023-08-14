import numpy as np

# Define convolution functions here...
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


def conv3d_transpose(x, kernel, strides=(1, 1, 1), padding='same'):
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

    return conv3d_transpose(x, kernel, strides=strides, padding=padding), (
        result_depth, result_height, result_width)


def upsample_with_padding(x, kernel, target_shape, strides=(2, 2, 2), padding='same'):
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

###################################################################################################################

class UNetBlock3D:
    def __init__(self, in_channels, out_channels):
        self.conv1 = conv3d(in_channels, out_channels)
        self.conv2 = conv3d(out_channels, out_channels)
        self.pool = max_pooling3d

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.pool(x2)
        return x2, x3

class UNet3D:
    def __init__(self, num_classes):
        self.down1 = UNetBlock3D(in_channels=1, out_channels=32)
        self.down2 = UNetBlock3D(in_channels=32, out_channels=64)
        self.down3 = UNetBlock3D(in_channels=64, out_channels=128)
        self.bottom = conv3d(128, 256)  # Bottom layer
        self.up3 = conv3d_transpose(256, 128)
        self.up2 = conv3d_transpose(128, 64)
        self.up1 = conv3d_transpose(64, 32)
        self.output_layer = conv3d(32, num_classes)

    def forward(self, x):
        x1, x2 = self.down1.forward(x)
        x2, x3 = self.down2.forward(x2)
        x3, x4 = self.down3.forward(x3)
        x5 = self.bottom(x4)
        x6 = self.up3(x5)
        x7 = self.up2(x6 + x3)  # Skip connection
        x8 = self.up1(x7 + x2)  # Skip connection
        output = self.output_layer(x8)
        return output


def main():
    # Example input shape and number of classes
    input_shape = (128, 128, 128, 3, 4)
    num_classes = 2

    # Create the U-Net model
    model = UNet3D(num_classes)

    # Generate example input
    inputs = np.random.randn(*input_shape)

    # Forward Pass
    output = model.forward(inputs)

    # Backward Pass
    gradient = np.random.randn(*output.shape)  # Assuming you have the gradient from the loss
    model.backward(gradient)

    gradient_down3_skip_connection = np.zeros_like(gradient)

if __name__ == "__main__":
    main()
