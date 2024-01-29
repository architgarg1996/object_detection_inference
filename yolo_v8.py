import cv2
import torch
from tqdm import tqdm
from yolov8_inference.ultralytics import YOLO
import time
import numpy as np
from torchvision.transforms import functional as F

# Import your custom metric calculation functions and label parsing
from metrics import calculate_precision_recall_f1, calculate_iou
from parse_groundtruth import parse_labels
from annotate_boxes import *
from config import  image_directory, label_path, output_directory

ball_class_index = 32
def eval_yolov8():

    def load_model():
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO("yolov8n.pt")
        model.to(device).eval()
        # model = torch.hub.load('ultralytics/yolov5', "yolov8n")
        print("Model loading succesful")
        return model
    
    def process_image(image_path, img_size=640):
        image = Image.open(image_path).convert("RGB")
        image = F.to_tensor(image)
        image = F.resize(image, [img_size, img_size])
        return image.unsqueeze(0)  # Add batch dimension

    model = load_model()
    ground_truths = parse_labels(label_path)
    total_precision, total_recall, total_f1, total_iou, total_inference_time, num_images = 0, 0, 0, 0, 0, 0

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(image_directory):
        if filename.endswith((".png", ".jpg", ".jpeg")) and filename in ground_truths:
            image_path = os.path.join(image_directory, filename)
            image = process_image(image_path)
            image = image.to('cuda' if torch.cuda.is_available() else 'cpu')

            start_time = time.time()
            results = model.predict(image, imgsz=1280)
            end_time = time.time()

            start_time = time.time()
            results_1 = model.predict(image, imgsz=1280)
            end_time = time.time()

            pred_boxes = []
            for prediction in results[0].boxes.cpu().numpy():
                bbox, category_id, score = (
                    prediction.xyxy,
                    prediction.cls,
                    prediction.conf,
                )
                
                # print("Category ID Index[0]: ",category_id[0])
                if int(category_id[0]) == ball_class_index:
                    print("ball_class_index:",ball_class_index)
                    print("Category ID: ",category_id)
                    print(type(category_id))
                    print("Length of Category ID: ",len(category_id))
                    print("Category ID Index[0]: ",category_id[0])
                    print("Bbox: ",bbox)

                    if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox[0]) == 4:
                        pred_boxes.append([coord.item() if hasattr(coord, 'item') else coord for coord in bbox[0]])
                    else:
                        print("Unexpected bbox format:", bbox)

            true_boxes = ground_truths[filename]
            print("True Boxes: ",true_boxes)
            precision, recall, f1_score = calculate_precision_recall_f1(pred_boxes, true_boxes)
            total_precision += precision
            total_recall += recall
            total_f1 += (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            total_iou += np.mean([calculate_iou(torch.tensor(pred_box), torch.tensor(true_box)) for pred_box in pred_boxes for true_box in true_boxes]) if pred_boxes else 0
            total_inference_time += end_time - start_time
            num_images += 1

            if output_directory:
                pil_image = Image.open(image_path).convert("RGB")
                draw_boxes_on_image(pil_image, pred_boxes, "red", "Pred")
                draw_boxes_on_image(pil_image, true_boxes, "green", "GT")
                pil_image.save(os.path.join(output_directory, filename))

    avg_precision = total_precision / num_images if num_images > 0 else 0
    avg_recall = total_recall / num_images if num_images > 0 else 0
    avg_f1 = total_f1 / num_images if num_images > 0 else 0
    avg_iou = total_iou / num_images if num_images > 0 else 0
    avg_inference_time = total_inference_time / num_images if num_images > 0 else 0

    return avg_precision, avg_recall, avg_f1, avg_iou, avg_inference_time

# Example usage
metrics = eval_yolov8()
print("Precision:", metrics[0], "Recall:", metrics[1], "F1-score:", metrics[2], "IoU:", metrics[3], "Average Inference Time:", metrics[4], "sec")