import torch
from PIL import Image
import numpy as np
import os
import time
from pathlib import Path
import sys
import subprocess
import json
import cv2
# Import YOLO v5 model from the official repository or a compatible library
sys.path.append('./yolov5')  # Ensure this points to the folder containing the YOLO v5 code

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox
from torchvision.transforms import functional as F
from yolov5.models.yolo import Model

from metrics import *
from config import ball_class_index, image_directory, label_path, output_directory, model_name
from parse_groundtruth import *
from annotate_boxes import *
from yolov5.utils.torch_utils import select_device


ball_class_index = 32
def eval_yolov5():

    def load_model(device):
        model_sizes = {'yolov5s': 'yolov5s', 'yolov5m': 'yolov5m', 'yolov5l': 'yolov5l', 'yolov5x': 'yolov5x'}
        weights_path = model_sizes.get(model_name, 'yolov5s')  # Default to 's' if version is invalid
        model = torch.hub.load('ultralytics/yolov5', weights_path)
        return model
    
    def evaluate_model(model, directory, label_path, output_directory=''):
        
        ground_truths = parse_labels(label_path)
        total_precision, total_recall, total_f1, total_iou, total_inference_time, num_images = 0, 0, 0, 0, 0, 0
        all_aps = []

        if output_directory:
            create_dir_if_not_exists(output_directory)

        for filename in os.listdir(directory):
            if filename.endswith((".png", ".jpg", ".jpeg")) and filename in ground_truths:
                image_path = os.path.join(directory, filename)

                start_time = time.time()
                results = model(image_path)
                end_time = time.time()
                
                results = results.pandas().xyxy[0]
                # Filter detections for the class of interest
                filtered_df = results[results['class'] == ball_class_index]

                # Convert DataFrame to list of bounding boxes
                pred_boxes = filtered_df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()

                pil_image = Image.open(image_path).convert("RGB")

                true_boxes = ground_truths[filename]
                # # Calculate metrics if there are predicted boxes
                if pred_boxes:
                    precision, recall, f1_score = calculate_precision_recall_f1(pred_boxes, true_boxes)
                    ious = [calculate_iou(pred_box, true_box) for pred_box in pred_boxes for true_box in true_boxes]
                    avg_iou = np.mean(ious) if ious else 0
                else:
                    precision, recall, f1_score, avg_iou = 0, 0, 0, 0

                if output_directory:
                    annotated_image = pil_image.copy()
                    draw_boxes_on_image(annotated_image, pred_boxes, "red", "Pred")
                    draw_boxes_on_image(annotated_image, true_boxes, "green", "GT")
                    annotated_image.save(os.path.join(output_directory, filename))

                # Calculate metrics
                TP, FP = 0, 0
                precisions, recalls = [], []
                for pred_box in pred_boxes:
                    TP += any(calculate_iou(pred_box, tb) >= 0.5 for tb in true_boxes)
                    FP += not any(calculate_iou(pred_box, tb) >= 0.5 for tb in true_boxes)
                    precision = TP / (TP + FP) if TP + FP > 0 else 0
                    recall = TP / len(true_boxes) if true_boxes else 0
                    precisions.append(precision)
                    recalls.append(recall)

                ap = calculate_average_precision(precisions, recalls)
                all_aps.append(ap)

                total_precision += precision
                total_recall += recall
                total_f1 += 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

                ious = [calculate_iou(pred_box, true_box) for pred_box in pred_boxes for true_box in true_boxes]
                avg_iou = np.mean(ious) if ious else 0
                total_iou += avg_iou

                total_inference_time += end_time - start_time
                num_images += 1

        avg_precision = total_precision / num_images if num_images > 0 else 0
        avg_recall = total_recall / num_images if num_images > 0 else 0
        avg_f1 = total_f1 / num_images if num_images > 0 else 0
        avg_iou = total_iou / num_images if num_images > 0 else 0
        mAP = np.mean(all_aps) if all_aps else 0
        avg_inference_time = total_inference_time / num_images if num_images > 0 else 0

        return avg_precision, avg_recall, avg_f1, avg_iou, mAP, avg_inference_time
    

    # Usage
    device = select_device()
    print("Running Inference on Device: ",device)
    model = load_model(device)
    metrics = evaluate_model(model, image_directory, label_path,output_directory)

    return metrics