import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

tf.random.set_seed(42)
np.random.seed(42)

class YoloV1Loss(tf.keras.losses.Loss):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5.0, lambda_noobj=0.5, name="yolo_v1_loss"):
        """
        Hàm loss tùy chỉnh cho YOLOv1.

        Args:
            S (int): Kích thước lưới (S x S).
            B (int): Số lượng bounding box được dự đoán cho mỗi ô lưới.
            C (int): Số lượng lớp đối tượng.
            lambda_coord (float): Trọng số cho lỗi tọa độ.
            lambda_noobj (float): Trọng số cho lỗi confidence khi không có đối tượng.
            name (str): Tên của hàm loss.
        """
        super().__init__(name=name)
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
        # Tổng số giá trị dự đoán trên mỗi ô lưới: B*5 + C
        self.output_dim = self.B * 5 + self.C

    def call(self, y_true, y_pred):
        """
        Tính toán hàm mất mát YOLOv1.

        Args:
            y_true: Tensor Ground truth.
                    Shape: (batch_size, S, S, 5 + C)
                    Cấu trúc (cho mỗi ô lưới): [obj_mask, x, y, w, h, c1, c2, ..., cC]
                    obj_mask (bool): 1 nếu có đối tượng, 0 nếu không.
                    x, y, w, h: Tọa độ box (đã chuẩn hóa về [0,1] và tương đối với ô lưới cho x,y)
                    c1..cC: One-hot encoding của lớp.
            y_pred: Tensor dự đoán của mô hình.
                    Shape: (batch_size, S, S, B*5 + C)
                    Cấu trúc (cho mỗi ô lưới):
                    Box 1: [x1, y1, w1, h1, conf1]
                    Box 2: [x2, y2, w2, h2, conf2]
                    ...
                    Probabilities: [p1, p2, ..., pC]
        """
        
        pred_boxes = tf.reshape(y_pred[..., :self.B * 5], (-1, self.S, self.S, self.B, 5))
        pred_x_y = pred_boxes[..., 0:2] # x, y
        pred_w_h = pred_boxes[..., 2:4] # w, h
        pred_conf = pred_boxes[..., 4:5] # confidence

        # Class probabilities predictions
        pred_class_prob = y_pred[..., self.B * 5:] # Class probabilities (C giá trị)

        true_obj_mask = y_true[..., 0:1]  # object mask (1 nếu có đối tượng, 0 nếu không)
        true_x_y = y_true[..., 1:3]     # x, y
        true_w_h = y_true[..., 3:5]     # w, h
        true_class_prob = y_true[..., 5:] # Class probabilities


        true_x_y_expanded = tf.expand_dims(true_x_y, axis=-2) # add a dimension for B
        true_w_h_expanded = tf.expand_dims(true_w_h, axis=-2) # add a dimension for B


        true_obj_mask_expanded = tf.expand_dims(true_obj_mask, axis=-1)

        # Tính toán IoU giữa tất cả các predicted boxes và ground truth box

        true_box_x1y1 = true_x_y - true_w_h / 2
        true_box_x2y2 = true_x_y + true_w_h / 2
        
        # Predicted boxes (có B box cho mỗi cell)
        pred_box_x1y1 = pred_x_y - pred_w_h / 2
        pred_box_x2y2 = pred_x_y + pred_w_h / 2
        
        # IoU calculation
        intersect_x1 = tf.maximum(pred_box_x1y1[..., 0], true_box_x1y1[..., 0])
        intersect_y1 = tf.maximum(pred_box_x1y1[..., 1], true_box_x1y1[..., 1])
        intersect_x2 = tf.minimum(pred_box_x2y2[..., 0], true_box_x2y2[..., 0])
        intersect_y2 = tf.minimum(pred_box_x2y2[..., 1], true_box_x2y2[..., 1])

        intersect_w = tf.maximum(0.0, intersect_x2 - intersect_x1)
        intersect_h = tf.maximum(0.0, intersect_y2 - intersect_y1)
        intersection = intersect_w * intersect_h

        # Tính diện tích các box
        pred_area = pred_w_h[..., 0] * pred_w_h[..., 1]
        true_area = true_w_h[..., 0] * true_w_h[..., 1]
        
        # Tính hợp (union)
        union = pred_area + true_area - intersection
        
        # IoU (tránh chia cho 0)
        iou = intersection / (union + tf.keras.backend.epsilon()) # Shape: (batch_size, S, S, B)

        # Xác định responsible box (box chịu trách nhiệm)

        best_box_mask = tf.cast(tf.equal(iou, tf.reduce_max(iou, axis=-1, keepdims=True)), dtype=tf.float32) # Shape: (batch_size, S, S, B)
        
        responsible_box_mask = true_obj_mask_expanded * best_box_mask # Shape: (batch_size, S, S, B)

        responsible_box_mask_5 = tf.expand_dims(responsible_box_mask, axis=-1) # Shape: (batch_size, S, S, B, 1)

        # Tính toán các thành phần Loss
        coord_loss_x_y = responsible_box_mask_5[..., 0:2] * tf.square(true_x_y_expanded - pred_x_y)
        coord_loss_w_h = responsible_box_mask_5[..., 2:4] * tf.square(tf.sqrt(true_w_h_expanded + tf.keras.backend.epsilon()) - tf.sqrt(pred_w_h + tf.keras.backend.epsilon()))
        
        coord_loss = tf.reduce_sum(coord_loss_x_y) + tf.reduce_sum(coord_loss_w_h)
        
        iou_score_for_responsible_box = tf.expand_dims(iou, axis=-1) # (batch_size, S, S, B, 1)
        
        obj_conf_loss = responsible_box_mask_5 * tf.square(iou_score_for_responsible_box - pred_conf)
        obj_conf_loss = tf.reduce_sum(obj_conf_loss)

        non_responsible_box_mask = true_obj_mask_expanded * (1 - responsible_box_mask_5)
        
        no_obj_cell_mask = (1 - true_obj_mask_expanded) # Áp dụng cho tất cả B boxes trong cell đó
        
        noobj_conf_loss = (non_responsible_box_mask + no_obj_cell_mask) * tf.square(0.0 - pred_conf)
        noobj_conf_loss = tf.reduce_sum(noobj_conf_loss)

        class_loss = true_obj_mask * tf.square(true_class_prob - pred_class_prob)
        class_loss = tf.reduce_sum(class_loss)

        # Tổng hợp Loss
        total_loss = (self.lambda_coord * coord_loss +
                      obj_conf_loss +
                      self.lambda_noobj * noobj_conf_loss +
                      class_loss)
        
        return total_loss