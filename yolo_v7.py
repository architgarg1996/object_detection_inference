import torch
import numpy as np
import os
import time
from pathlib import Path
import sys
from PIL import Image
from torchvision.transforms import functional as F

# Import YOLO v7 model from the official repository
sys.path.append('./yolov7_inference')  # Ensure this points to the folder containing the YOLO v7 code
from yolov7_inference.models.experimental import attempt_load
from yolov7_inference.utils.general import non_max_suppression, scale_coords
from yolov7_inference.utils.torch_utils import select_device, time_synchronized

# Import your custom metric calculation functions and label parsing
from metrics import calculate_precision_recall_f1, calculate_iou
from parse_groundtruth import parse_labels
from annotate_boxes import *
from config import ball_class_index, image_directory, label_path, output_directory, model_name

ball_class_index = 32
def display_results(metrics):

    print("Model Used: {}".format(model_name))
    print(f"Precision: {metrics[0]}\n \
          Recall: {metrics[1]}\n \
          F1-score: {metrics[2]}\n \
          IoU: {metrics[3]}\n \
          mAP: {metrics[4]}\n \
          Average Inference Time per Frame: {metrics[5]} sec")


def eval_yolov7():
    
    def load_model(weights_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)
        model = attempt_load(weights_path, map_location=device)  # load FP32 model
        model.to(device).eval()

        # # model_sizes = {'yolov5s': 'yolov5s', 'yolov5m': 'yolov5m', 'yolov5l': 'yolov5l', 'yolov5x': 'yolov5x'}
        # # weights_path = model_sizes.get(model_name, 'yolov5s')  # Default to 's' if version is invalid
        # model = torch.hub.load('WongKinYiu/yolov7', 'yolov7.pt')
        
        return model, device

    def process_image(image_path, img_size=640):
        image = Image.open(image_path).convert("RGB")
        image = F.to_tensor(image)
        image = F.resize(image, [img_size, img_size])
        return image.unsqueeze(0)  # Add batch dimension

    # Load the model
    model, device = load_model(model_name)
    stride = int(model.stride.max())  # model stride

    ground_truths = parse_labels(label_path)
    total_precision, total_recall, total_f1, total_iou, total_inference_time, num_images = 0, 0, 0, 0, 0, 0
    all_aps = []

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(image_directory):
        if filename.endswith((".png", ".jpg", ".jpeg")) and filename in ground_truths:
            image_path = os.path.join(image_directory, filename)
            image = process_image(image_path)
            image = image.to(device)
            pil_image = Image.open(image_path).convert("RGB")

            start_time = time.time()
            with torch.no_grad():
                pred = model(image)[0]
            pred = non_max_suppression(pred, conf_thres=0.05, iou_thres=0.05, classes=[ball_class_index])
            end_time = time.time()

            pred_boxes = []
            for det in pred:
                if len(det):    
                    det[:, :4] = scale_coords(image.shape[2:], det[:, :4], image.shape[2:]).round()
                    for *xyxy, conf, cls in det:
                        pred_boxes.append([coord.item() for coord in xyxy])

            true_boxes = ground_truths[filename]
            precision, recall, f1_score = calculate_precision_recall_f1(pred_boxes, true_boxes)
            ious = [calculate_iou(torch.tensor(pred_box), torch.tensor(true_box)) for pred_box in pred_boxes for true_box in true_boxes]
            avg_iou = np.mean(ious) if ious else 0

            total_precision += precision
            total_recall += recall
            total_f1 += (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            total_iou += avg_iou
            total_inference_time += end_time - start_time
            num_images += 1

            if output_directory:
                annotated_image = pil_image.copy()
                draw_boxes_on_image(annotated_image, pred_boxes, "red", "Pred")
                draw_boxes_on_image(annotated_image, true_boxes, "green", "GT")
                annotated_image.save(os.path.join(output_directory, filename))

    avg_precision = total_precision / num_images if num_images > 0 else 0
    avg_recall = total_recall / num_images if num_images > 0 else 0
    avg_f1 = total_f1 / num_images if num_images > 0 else 0
    avg_iou = total_iou / num_images if num_images > 0 else 0
    mAP = np.mean(all_aps) if all_aps else 0
    avg_inference_time = total_inference_time / num_images if num_images > 0 else 0

    return avg_precision, avg_recall, avg_f1, avg_iou, mAP, avg_inference_time


metrics = eval_yolov7()
display_results(metrics)
