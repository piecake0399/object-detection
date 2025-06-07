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
        """
        super().__init__(name=name)
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
        self.output_dim = self.B * 5 + self.C

    def call(self, y_true, y_pred):
        """
        Tính toán hàm mất mát YOLOv1.
        """
        
        # 1. Tách các thành phần từ y_pred (dự đoán)
        # pred_boxes: (batch_size, S, S, B, 5)
        pred_boxes = tf.reshape(y_pred[..., :self.B * 5], (-1, self.S, self.S, self.B, 5))
        pred_x_y = pred_boxes[..., 0:2] 
        pred_w_h = pred_boxes[..., 2:4] 
        pred_conf = pred_boxes[..., 4:5] 

        # pred_class_prob: (batch_size, S, S, C)
        pred_class_prob = y_pred[..., self.B * 5:] 

        # 2. Tách các thành phần từ y_true (ground truth)
        # true_obj_mask: (batch_size, S, S, 1)
        true_obj_mask = y_true[..., 0:1]  
        # true_x_y: (batch_size, S, S, 2)
        true_x_y = y_true[..., 1:3]     
        # true_w_h: (batch_size, S, S, 2)
        true_w_h = y_true[..., 3:5]     
        # true_class_prob: (batch_size, S, S, C)
        true_class_prob = y_true[..., 5:] 

        # Mở rộng chiều của true_w_h, true_x_y, true_obj_mask để phù hợp với pred_boxes (thêm chiều B=1)
        # true_x_y_expanded: (batch_size, S, S, 1, 2)
        true_x_y_expanded = tf.expand_dims(true_x_y, axis=-2) 
        # true_w_h_expanded: (batch_size, S, S, 1, 2)
        true_w_h_expanded = tf.expand_dims(true_w_h, axis=-2) 
        # true_obj_mask_expanded: (batch_size, S, S, 1, 1)
        true_obj_mask_expanded = tf.expand_dims(true_obj_mask, axis=-1) 

        # 3. Tính toán IoU giữa tất cả các predicted boxes và ground truth box
        # Ground truth box (chỉ có 1 cho mỗi cell), đã được mở rộng để broadcast
        true_box_x1y1 = true_x_y_expanded - true_w_h_expanded / 2 # Shape: (batch_size, S, S, 1, 2)
        true_box_x2y2 = true_x_y_expanded + true_w_h_expanded / 2 # Shape: (batch_size, S, S, 1, 2)
        
        # Predicted boxes
        pred_box_x1y1 = pred_x_y - pred_w_h / 2 # Shape: (batch_size, S, S, B, 2)
        pred_box_x2y2 = pred_x_y + pred_w_h / 2 # Shape: (batch_size, S, S, B, 2)
        
        # IoU calculation
        # Lấy tọa độ x (index 0) và y (index 1) của góc trên bên trái/dưới bên phải
        intersect_x1 = tf.maximum(pred_box_x1y1[..., 0:1], true_box_x1y1[..., 0:1]) # Slice 0:1 to keep dim
        intersect_y1 = tf.maximum(pred_box_x1y1[..., 1:2], true_box_x1y1[..., 1:2]) # Slice 1:2 to keep dim
        intersect_x2 = tf.minimum(pred_box_x2y2[..., 0:1], true_box_x2y2[..., 0:1])
        intersect_y2 = tf.minimum(pred_box_x2y2[..., 1:2], true_box_x2y2[..., 1:2])

        # Width and height of intersection
        intersect_w = tf.maximum(0.0, intersect_x2 - intersect_x1)
        intersect_h = tf.maximum(0.0, intersect_y2 - intersect_y1)
        intersection = intersect_w * intersect_h # Shape: (batch_size, S, S, B, 1)

        # Area of predicted and true boxes
        pred_area = pred_w_h[..., 0:1] * pred_w_h[..., 1:2] # Shape: (batch_size, S, S, B, 1)
        true_area = true_w_h_expanded[..., 0:1] * true_w_h_expanded[..., 1:2] # Shape: (batch_size, S, S, 1, 1)
        
        # Union
        union = pred_area + true_area - intersection # Shape: (batch_size, S, S, B, 1)
        
        # IoU (tránh chia cho 0)
        iou = intersection / (union + tf.keras.backend.epsilon()) # Shape: (batch_size, S, S, B, 1)

        # 4. Xác định responsible box (box chịu trách nhiệm)
        # best_box_mask: (batch_size, S, S, B, 1) - 1 for best box, 0 otherwise
        best_box_mask = tf.cast(tf.equal(iou, tf.reduce_max(iou, axis=-2, keepdims=True)), dtype=tf.float32) 
        
        # responsible_box_mask: (batch_size, S, S, B, 1) - 1 if obj present AND is best box
        responsible_box_mask = true_obj_mask_expanded * best_box_mask 

        # 5. Tính toán các thành phần Loss

        # a. Lỗi tọa độ (Coordinate Loss)
        # tf.square(true_x_y_expanded - pred_x_y) gives (batch_size, S, S, B, 2)
        coord_loss_x_y = responsible_box_mask * tf.square(true_x_y_expanded - pred_x_y)
        # Using sqrt(w) and sqrt(h)
        coord_loss_w_h = responsible_box_mask * tf.square(tf.sqrt(true_w_h_expanded + tf.keras.backend.epsilon()) - tf.sqrt(pred_w_h + tf.keras.backend.epsilon()))
        
        coord_loss = tf.reduce_sum(coord_loss_x_y) + tf.reduce_sum(coord_loss_w_h)
        
        # b. Lỗi độ tin cậy khi có đối tượng (Confidence Loss for Objects)
        # Goal for confidence is IoU of the responsible box
        obj_conf_loss = responsible_box_mask * tf.square(iou - pred_conf) # iou is already (..., B, 1)
        obj_conf_loss = tf.reduce_sum(obj_conf_loss)

        # c. Lỗi độ tin cậy khi KHÔNG có đối tượng (Confidence Loss for No Objects)
        # Mask for non-responsible boxes in object-containing cells, AND
        # Mask for all boxes in no-object cells.
        non_responsible_box_mask = true_obj_mask_expanded * (1 - responsible_box_mask)
        no_obj_cell_mask = (1 - true_obj_mask_expanded) 
        
        noobj_conf_loss = (non_responsible_box_mask + no_obj_cell_mask) * tf.square(0.0 - pred_conf)
        noobj_conf_loss = tf.reduce_sum(noobj_conf_loss)

        # d. Lỗi phân loại (Classification Loss) - only for cells with objects
        # true_obj_mask: (batch_size, S, S, 1) -> will broadcast to (batch_size, S, S, C)
        class_loss = true_obj_mask * tf.square(true_class_prob - pred_class_prob)
        class_loss = tf.reduce_sum(class_loss)

        # 6. Tổng hợp Loss
        total_loss = (self.lambda_coord * coord_loss +
                      obj_conf_loss +
                      self.lambda_noobj * noobj_conf_loss +
                      class_loss)
        
        return total_loss