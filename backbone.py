import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Để đảm bảo tính tái lập
tf.random.set_seed(42)
np.random.seed(42)

def _conv_block(inputs, filters, kernel, strides):
    """
    Tạo một khối tích chập (Convolution + BatchNorm + ReLU).
    """
    x = layers.Conv2D(filters, kernel, padding='same', strides=strides, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def _depthwise_separable_conv_block(inputs, pointwise_filters, strides=1):
    """
    Tạo một khối Depthwise Separable Convolution (Depthwise + Pointwise + BatchNorm + ReLU).
    """
    # Depthwise Convolution
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Pointwise Convolution (1x1)
    x = layers.Conv2D(pointwise_filters, kernel_size=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def MobileNetV1_Backbone(input_shape=(224, 224, 3), alpha=1.0):
    """
    Xây dựng MobileNetV1 Backbone.
    Args:
        input_shape: Kích thước ảnh đầu vào (height, width, channels).
        alpha: Hệ số độ rộng (width multiplier) để scale số lượng bộ lọc.
               (ví dụ: 1.0, 0.75, 0.50, 0.25). Giá trị nhỏ hơn tạo mô hình nhỏ hơn.
    """
    inputs = keras.Input(shape=input_shape)

    # Lớp đầu tiên (Standard Convolution)
    x = _conv_block(inputs, int(32 * alpha), (3, 3), strides=(2, 2)) # 112x112 / 56x56

    # Các khối Depthwise Separable Convolution
    x = _depthwise_separable_conv_block(x, int(64 * alpha))

    x = _depthwise_separable_conv_block(x, int(128 * alpha), strides=(2, 2)) # 56x56 / 28x28
    x = _depthwise_separable_conv_block(x, int(128 * alpha))

    x = _depthwise_separable_conv_block(x, int(256 * alpha), strides=(2, 2)) # 28x28 / 14x14
    x = _depthwise_separable_conv_block(x, int(256 * alpha))

    x = _depthwise_separable_conv_block(x, int(512 * alpha), strides=(2, 2)) # 14x14 / 7x7

    for _ in range(5):
        x = _depthwise_separable_conv_block(x, int(512 * alpha))

    x = _depthwise_separable_conv_block(x, int(1024 * alpha), strides=(2, 2)) # 7x7 / 7x7 (hoặc 4x4 tùy kích thước)
    x = _depthwise_separable_conv_block(x, int(1024 * alpha))

    return keras.Model(inputs, x, name="MobileNetV1_Backbone")

if __name__ == '__main__':
    # Thử nghiệm với các kích thước input và alpha khác nhau
    print("--- MobileNetV1 Backbone (alpha=1.0, input 224x224) ---")
    backbone_model_full = MobileNetV1_Backbone(input_shape=(224, 224, 3), alpha=1.0)
    backbone_model_full.summary()

    print("\n--- MobileNetV1 Backbone (alpha=0.25, input 128x128) - Nhỏ gọn hơn ---")
    backbone_model_tiny = MobileNetV1_Backbone(input_shape=(128, 128, 3), alpha=0.25)
    backbone_model_tiny.summary()

    print("\n--- MobileNetV1 Backbone (alpha=0.25, input 64x64) - Rất nhỏ gọn ---")
    backbone_model_micro = MobileNetV1_Backbone(input_shape=(64, 64, 3), alpha=0.25)
    backbone_model_micro.summary()

    print(f"\nSize of MobileNetV1_Backbone (alpha=1.0, 224x224): {backbone_model_full.count_params() * 4 / (1024*1024):.2f} MB ")
    print(f"Size of MobileNetV1_Backbone (alpha=0.25, 128x128): {backbone_model_tiny.count_params() * 4 / (1024*1024):.2f} MB ")
    print(f"Size of MobileNetV1_Backbone (alpha=0.25, 64x64): {backbone_model_micro.count_params() * 4 / (1024*1024):.2f} MB ")

    # backbone_model_tiny.save("mobilenetv1_backbone_alpha0_25_128x128.h5")