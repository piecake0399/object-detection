import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from backbone import MobileNetV1_Backbone

def get_num_classes(classes_file="classes.txt"):
    """
    Đọc số lượng lớp từ tệp classes.txt.
    Mỗi dòng trong tệp là một tên lớp.
    """
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"File '{classes_file}' not found. Please create it with one class per line.")
    
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f if line.strip()] # Đọc và loại bỏ dòng trống
    
    print(f"Detected {len(classes)} classes from '{classes_file}': {classes}")
    return len(classes), classes

def model_build(input_shape=(224, 224, 3), alpha=0.25, B=2, classes_file="classes.txt"):
    """
    Xây dựng mô hình YOLOv1 với MobileNetV1 Backbone và Prediction Layer
    sử dụng chỉ các lớp Convolutional.

    Args:
        input_shape: Kích thước ảnh đầu vào (height, width, channels).
        alpha: Hệ số độ rộng cho MobileNetV1 Backbone.
        B: Số lượng bounding box được dự đoán cho mỗi ô lưới (grid cell).
        classes_file: Đường dẫn đến tệp chứa danh sách các lớp.
    """
    
    num_classes, class_names = get_num_classes(classes_file)

    # Khởi tạo Backbone
    backbone = MobileNetV1_Backbone(input_shape=input_shape, alpha=alpha)
    
    backbone_output = backbone.output
    
    # Xác định kích thước lưới S x S từ shape của output backbone
    S = backbone_output.shape[1] 
    if S != backbone_output.shape[2]:
        raise ValueError(f"Output feature map from backbone is not square ({S}x{backbone_output.shape[2]}). YOLOv1 expects a square grid.")
    
    print(f"Input image shape: {input_shape}")
    print(f"MobileNetV1 Backbone output shape (feature map): {backbone_output.shape}")
    print(f"Inferred Grid Size (S): {S}x{S}")

    output_channels_per_cell = B * 5 + num_classes
    
    x = layers.Conv2D(int(512 * alpha), kernel_size=(3, 3), padding='same', activation='relu')(backbone_output)
    x = layers.Conv2D(output_channels_per_cell, kernel_size=(1, 1), padding='same', activation='linear', name='yolo_output')(x)
    
    model = keras.Model(inputs=backbone.input, outputs=x, name="YOLOv1_MobileNetV1_ConvHead_Tiny")
    
    print(f"\nFinal YOLOv1 output shape: {model.output_shape}")
    print(f"Expected output channels per cell: {output_channels_per_cell}")

    total_params = model.count_params()
    print(f"\nNumber of params: {total_params}")

    return model, S, B, num_classes, class_names