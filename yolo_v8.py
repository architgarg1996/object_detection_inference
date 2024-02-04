import torch
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import os
import time

# Assuming YOLO, calculate_iou, calculate_average_precision, parse_labels, and draw_boxes_on_image are correctly implemented
from yolov8_inference.ultralytics import YOLO
from metrics import calculate_iou, calculate_average_precision, calculate_precision_recall_f1
from parse_groundtruth import parse_labels
from annotate_boxes import draw_boxes_on_image
from config import ball_class_index, image_directory, label_path
from tqdm import tqdm

ball_class_index = 32.0

def eval_yolov8(model_name, output_directory):
    def load_model():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO(f"{model_name}.pt").to(device)
        print("Model loading successful")
        return model, device

    def process_image(image_path, img_size=640):
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        image = F.to_tensor(image)
        image = F.resize(image, [img_size, img_size])
        return image.unsqueeze(0), original_size  # Add batch dimension
    
    def resize_bbox(bbox,original_size,resized_size):

        original_width, original_height = original_size
        resized_width, resized_height = resized_size

        # Calculate scaling factors
        x_scale = original_width / resized_width
        y_scale = original_height / resized_height

        # Resize boxes
        x_min, y_min, x_max, y_max = bbox
        x_min_resized = x_min * x_scale
        y_min_resized = y_min * y_scale
        x_max_resized = x_max * x_scale
        y_max_resized = y_max * y_scale

        return x_min_resized,y_min_resized,x_max_resized,y_max_resized


    model, device = load_model()
    ground_truths = parse_labels(label_path)
    all_aps = []
    all_ious = []
    total_tp, total_fp, total_fn = 0, 0, 0
    total_inference_time = 0
    num_images = 0

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    for filename in tqdm(os.listdir(image_directory)):
        if filename.endswith((".png", ".jpg", ".jpeg")) and filename in ground_truths:
            image_path = os.path.join(image_directory, filename)
            image, original_size = process_image(image_path)
            image = image.to(device)

            start_time = time.time()
            results = model.predict(image, conf=0.25)
            end_time = time.time()

            pred_boxes = []
            scores = []
            pred_boxes = []
            for result in results:
                for box in result.boxes:
                    if box.cls == ball_class_index:
                        bbox = box.xyxy.cpu().numpy()
                        bbox_resized = resize_bbox([bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3]],
                                                   original_size,[640,640])
                        pred_boxes.append(bbox_resized)
                        scores = box.conf.cpu().numpy()
            true_boxes = ground_truths[filename]
            pred_boxes_scores = [(box, score) for box, score in zip(pred_boxes, scores)]

            if pred_boxes_scores:
                ap = calculate_average_precision(pred_boxes_scores, true_boxes, iou_threshold=0.3)
                all_aps.append(ap)

                tp, fp, fn = calculate_precision_recall_f1(pred_boxes, true_boxes)
                total_tp += tp
                total_fp += fp
                total_fn += fn

                ious = [calculate_iou(np.array(pred_box), np.array(true_box)) for pred_box, _ in pred_boxes_scores for true_box in true_boxes]
                all_ious.extend(ious)

            total_inference_time += end_time - start_time
            num_images += 1

            if output_directory:
                pil_image = Image.open(image_path).convert("RGB")
                draw_boxes_on_image(pil_image, pred_boxes, "red", "Pred")
                draw_boxes_on_image(pil_image, true_boxes, "green", "GT")
                pil_image.save(os.path.join(output_directory, filename))

    avg_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    avg_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if avg_precision + avg_recall > 0 else 0
    mAP = np.mean(all_aps) if all_aps else 0
    avg_iou = np.mean(all_ious) if all_ious else 0
    avg_inference_time = total_inference_time / num_images if num_images > 0 else 0

    return avg_precision, avg_recall, avg_f1, avg_iou, mAP, avg_inference_time

# Example usage
# metrics = eval_yolov8('yolov8s', 'output_directory')
# print("mAP:", metrics[0], "Average IoU:", metrics[1], "Average Inference Time:", metrics[2], "sec")
