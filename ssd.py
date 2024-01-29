import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import os
import time

from metrics import *
from config import ball_class_index, image_directory, label_path, output_directory
from parse_groundtruth import *
from annotate_boxes import *


def evaluate_ssd():

    def load_ssd_model(pretrained=True, device='cpu'):
        model = ssd300_vgg16(pretrained=pretrained)
        model = model.to(device)
        model.eval()
        return model

    def process_image(image_path):
        image = Image.open(image_path).convert("RGB")
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return image

    def evaluate_model(model, directory, label_path, output_directory='', device='cpu'):
        ground_truths = parse_labels(label_path)
        total_precision, total_recall, total_f1, total_iou, total_map, total_inference_time, num_images = 0, 0, 0, 0, 0, 0, 0
        all_aps = []

        if output_directory:
            create_dir_if_not_exists(output_directory)

        for filename in os.listdir(directory):
            if filename.endswith((".png", ".jpg", ".jpeg")) and filename in ground_truths:
                image_path = os.path.join(directory, filename)
                image = process_image(image_path).unsqueeze(0).to(device)
                true_boxes = ground_truths[filename]

                start_time = time.time()
                with torch.no_grad():
                    prediction = model(image)
                end_time = time.time()

                pred_boxes = prediction[0]['boxes'].cpu().numpy()
                scores = prediction[0]['scores'].cpu().numpy()
                labels = prediction[0]['labels'].cpu().numpy()

                # # Filter out predictions for the 'Ball' class
                ball_indices = np.where(labels == ball_class_index)[0]
                pred_boxes = pred_boxes[ball_indices]
                scores = scores[ball_indices]

                # Sort by scores in descending order
                sorted_indices = np.argsort(-scores)
                sorted_pred_boxes = pred_boxes[sorted_indices]

                if output_directory:
                    # Draw boxes and save annotated image
                    annotated_image = Image.open(image_path).convert("RGB")
                    draw_boxes_on_image(annotated_image, sorted_pred_boxes, "red", "Pred")
                    draw_boxes_on_image(annotated_image, true_boxes, "green", "GT")
                    annotated_image.save(os.path.join(output_directory, filename))

                precision, recall, avg_iou = 0, 0, 0
                TP, FP = 0, 0
                precisions, recalls = [], []
                for pred_box in sorted_pred_boxes:
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

                ious = [calculate_iou(pred_box, true_box) for pred_box in sorted_pred_boxes for true_box in true_boxes]
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
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_ssd_model(device=device)
    metrics = evaluate_model(model, image_directory, label_path, output_directory, device)
    
    return metrics