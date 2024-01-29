from faster_rcnn import eval_faster_rcnn
from yolo_v5 import eval_yolov5
from yolo_v8 import eval_yolov8
from ssd import evaluate_ssd
from config import model_name

def display_results(metrics):

    print("Model Used: {}".format(model_name))
    print(f"Precision: {metrics[0]}\n \
          Recall: {metrics[1]}\n \
          F1-score: {metrics[2]}\n \
          IoU: {metrics[3]}\n \
          mAP: {metrics[4]}\n \
          Average Inference Time per Frame: {metrics[5]} sec")
    
if model_name == 'faster_rcnn':
    metrics = eval_faster_rcnn() 
elif model_name in ('yolov5s','yolov5m','yolov5l','yolov5x'):
    metrics = eval_yolov5()
elif model_name == 'ssd':
    metrics = evaluate_ssd()
elif model_name == 'yolov8':
    metrics = eval_yolov8()

display_results(metrics)