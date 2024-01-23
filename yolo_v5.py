import torch
from PIL import Image
import numpy as np
import os
import time
from pathlib import Path
import sys

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


def eval_yolov5():

    def load_model():
        model_sizes = {'yolov5s': 'yolov5s.pt', 'yolov5m': 'yolov5m.pt', 'yolov5l': 'yolov5l.pt', 'yolov5x': 'yolov5x.pt'}
        weights_path = model_sizes.get(model_name, 'yolov5s.pt')  # Default to 's' if version is invalid
        print("weights_path ",weights_path)
        model = DetectMultiBackend(weights_path, device='cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        return model
    

    # def load_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    #     # Define the path to the weights file
    #     weights_path = f'./yolov5/weights/{model_name}.pt'  # Adjust this path

    #     # Load the model architecture from YOLOv5 source code
    #     model = Model(cfg=f'./yolov5/models/{model_name}.yaml', ch=3, nc=80) # Specify the correct YAML config file and number of classes (nc)
        
    #     # Load the weights into the model
    #     checkpoint = torch.load(weights_path, map_location=device)
    #     model.load_state_dict(checkpoint['model'].state_dict())

    #     model.to(device).eval()
    #     return model

    def process_image(image_path, stride, img_size=640):
        image = Image.open(image_path).convert("RGB")
        # Convert PIL Image to NumPy array
        image_np = np.array(image)

        # Convert stride to a Python number if it's a tensor
        stride_value = stride.item() if isinstance(stride, torch.Tensor) else stride

        # Apply letterbox (resize and pad)
        image_np = letterbox(image_np, img_size, stride=stride_value)[0]

        # Convert NumPy array to PyTorch tensor
        tensor_image = F.to_tensor(image_np)

        # Normalize and add a batch dimension
        tensor_image = tensor_image.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu').float()
        tensor_image /= 255  # Normalize to [0, 1]

        return tensor_image

    def evaluate_model(model, directory, label_path, output_directory=''):
        stride = model.stride  # Model stride
        ground_truths = parse_labels(label_path)
        total_precision, total_recall, total_f1, total_iou, total_inference_time, num_images = 0, 0, 0, 0, 0, 0
        all_aps = []  # List to store average precision for each image

        if output_directory:
            create_dir_if_not_exists(output_directory)

        for filename in os.listdir(directory):
            if filename.endswith((".png", ".jpg", ".jpeg")) and filename in ground_truths:
                image_path = os.path.join(directory, filename)
                image = process_image(image_path, stride)
                pil_image = Image.open(image_path).convert("RGB")

                true_boxes = ground_truths[filename]

                start_time = time.time()
                with torch.no_grad():
                    prediction = model(image, augment=False, visualize=False)
                end_time = time.time()

                ball_class_indices = [ball_class_index] if isinstance(ball_class_index, int) else ball_class_index
                prediction = non_max_suppression(prediction, conf_thres=0.05, iou_thres=0.05, classes=ball_class_indices, max_det=300)

                pred_boxes = []
                for i, det in enumerate(prediction[0]):  # detections per image
                    if len(det):
                        det[:,:4] = scale_boxes(image.shape[2:], det[:, :4], pil_image.size).round()
                        
                    for detection in reversed(det):
                        *xyxy, conf, cls = detection[:6]  # Ensure only the first six elements are considered
                        # if cls.item() == ball_class_index:  # Filter for 'ball' class
                        x_min, y_min, x_max, y_max = map(lambda v: v.item(), xyxy)

                        # Ensure x_min <= x_max and y_min <= y_max
                        x_min, x_max = min(x_min, x_max), max(x_min, x_max)
                        y_min, y_max = min(y_min, y_max), max(y_min, y_max)

                        pred_boxes.append([x_min, y_min, x_max, y_max])

                # Calculate metrics if there are predicted boxes
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
    model = load_model()
    metrics = evaluate_model(model, image_directory, label_path,output_directory)

    return metrics
