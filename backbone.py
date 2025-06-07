import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# Helper function for Standard Convolution block
def _conv_block(inputs, filters, kernel, strides):
    """
    Tạo một khối tích chập (Convolution + BatchNorm + ReLU).
    """
    x = layers.Conv2D(filters, kernel, padding='same', strides=strides, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

# Helper function for Depthwise Separable Convolution block
def _depthwise_conv_block(inputs, pointwise_filters, alpha, strides=(1, 1)):
    """
    Tạo một khối tích chập Depthwise Separable.
    """
    x = layers.DepthwiseConv2D((3, 3), padding='same', strides=strides, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(int(pointwise_filters * alpha), (1, 1), padding='same', strides=(1, 1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def MobileNetV1_Backbone(input_shape, alpha):
    """
    Xây dựng kiến trúc backbone MobileNetV1.

    Args:
        input_shape (tuple): Hình dạng của ảnh đầu vào (height, width).
        alpha (float): Hệ số nhân độ rộng (width multiplier) để điều chỉnh số lượng bộ lọc.

    Returns:
        tf.keras.Model: Mô hình MobileNetV1 dùng làm backbone.
    """
    # Define the input layer with the channel dimension
    # Assuming RGB images, so channels = 3
    inputs = tf.keras.Input(shape=(input_shape[0], input_shape[1], 3)) # ADDED CHANNEL DIMENSION HERE

    # Lớp đầu tiên (Standard Convolution)
    x = _conv_block(inputs, int(32 * alpha), (3, 3), strides=(2, 2)) # 112x112 / 56x56

    # Các khối Depthwise Separable Convolution
    x = _depthwise_conv_block(x, 64, alpha) # 56x56
    x = _depthwise_conv_block(x, 128, alpha, strides=(2, 2)) # 28x28
    x = _depthwise_conv_block(x, 128, alpha) # 28x28
    x = _depthwise_conv_block(x, 256, alpha, strides=(2, 2)) # 14x14
    x = _depthwise_conv_block(x, 256, alpha) # 14x14
    x = _depthwise_conv_block(x, 512, alpha, strides=(2, 2)) # 7x7

    # 5 khối depthwise separable với 512 filters (không giảm kích thước)
    for _ in range(5):
        x = _depthwise_conv_block(x, 512, alpha) # 7x7

    x = _depthwise_conv_block(x, 1024, alpha, strides=(2, 2)) # 4x4 (Adjusted strides based on YOLOv1 architecture)
    x = _depthwise_conv_block(x, 1024, alpha) # 4x4 (Adjusted based on YOLOv1 architecture)

    # Output của backbone là tensor cuối cùng trước các lớp dự đoán
    backbone_output = x

    # Trả về mô hình
    return Model(inputs=inputs, outputs=backbone_output, name='mobilenetv1_backbone')