import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import pandas as pd
import numpy as np
import os
import time
from metrics import *

# Assuming 'ball_class_index' is the index of the 'Ball' class in your dataset
# Update this index based on your dataset's class indices
# COCO class names for torchvision models
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Find the index of the class
ball_class_index = COCO_INSTANCE_CATEGORY_NAMES.index('sports ball')

def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = F.to_tensor(image)
    return image

def parse_labels(label_path):
    labels = pd.read_csv(label_path, header=None)
    labels.columns = ['image_name', 'x_min', 'y_min', 'x_max', 'y_max', 'class']
    grouped_labels = labels.groupby('image_name')
    return {name: group[group['class'] == ball_class_index].iloc[:, 1:5].values for name, group in grouped_labels}

def evaluate_model(model, directory, label_path):
    ground_truths = parse_labels(label_path)
    total_precision, total_recall, total_f1, total_iou, total_map, total_inference_time, num_images = 0, 0, 0, 0, 0, 0, 0
    all_aps = []

    for filename in os.listdir(directory):
        if filename.endswith((".png", ".jpg", ".jpeg")) and filename in ground_truths:
            image_path = os.path.join(directory, filename)
            image = process_image(image_path)

            start_time = time.time()
            with torch.no_grad():
                prediction = model([image])
            end_time = time.time()

            # Filter out predictions for the 'Ball' class
            pred_boxes = [[box[0], box[1], box[2], box[3]] for box, label in zip(prediction[0]['boxes'].cpu().numpy(), prediction[0]['labels'].cpu().numpy()) if label == ball_class_index]
            true_boxes = ground_truths[filename]

            precision, recall, f1_score = calculate_precision_recall_f1(pred_boxes, true_boxes)
            total_precision += precision
            total_recall += recall
            total_f1 += f1_score

            ious = [calculate_iou(pred_box, true_box) for pred_box in pred_boxes for true_box in true_boxes]
            avg_iou = np.mean(ious) if ious else 0
            total_iou += avg_iou

            precisions, recalls = [], []
            for i, pred_box in enumerate(pred_boxes):
                precisions.append(precision)
                recalls.append(recall)

            ap = calculate_average_precision(precisions, recalls)
            all_aps.append(ap)

            total_inference_time += end_time - start_time
            num_images += 1

    avg_precision = total_precision / num_images
    avg_recall = total_recall / num_images
    avg_f1 = total_f1 / num_images
    avg_iou = total_iou / num_images
    mAP = np.mean(all_aps) if all_aps else 0
    avg_inference_time = total_inference_time / num_images

    return avg_precision, avg_recall, avg_f1, avg_iou, mAP, avg_inference_time

# Usage
image_directory = 'path_to_your_image_directory'
label_path = 'path_to_your_label_file.csv'
model = load_model()
metrics = evaluate_model(model, image_directory, label_path)
print(f"Precision: {metrics[0]}, Recall: {metrics[1]}, F1-score: {metrics[2]}, IoU: {metrics[3]}, mAP: {metrics[4]}, Average Inference Time per Frame: {metrics[5]}")
