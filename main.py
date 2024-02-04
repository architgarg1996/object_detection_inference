from faster_rcnn import eval_faster_rcnn
from yolo_v5 import eval_yolov5
from yolo_v8 import eval_yolov8
# from ssd import evaluate_ssd
from config import model_name, output_directory, label_path, sport, tag

def display_results(metrics, model_name_i):

    print("Model Used: {}".format(model_name_i))
    print("CSV: ", label_path.split("/")[-1])
    print(f"Precision: {metrics[0]}\n \
          Recall: {metrics[1]}\n \
          F1-score: {metrics[2]}\n \
          IoU: {metrics[3]}\n \
          mAP: {metrics[4]}\n \
          Average Inference Time per Frame: {metrics[5]} sec")
    print(f"{model_name_i}, {label_path.split('/')[-1]}, {metrics[0]}, {metrics[1]}, {metrics[2]}, {metrics[3]}, {metrics[4]}, {metrics[5]}")
    out = f"{sport}, {tag}, {model_name_i}, {metrics[0]}, {metrics[1]}, {metrics[2]}, {metrics[3]}, {metrics[4]}, {metrics[5]}"
    return out
    

results_all = []
for model_name_i in model_name:
    if model_name_i == 'faster_rcnn':
        metrics = eval_faster_rcnn(model_name_i, output_directory+model_name_i) 
    elif model_name_i in ('yolov5s','yolov5m','yolov5l','yolov5x'):
        metrics = eval_yolov5(model_name_i, output_directory+model_name_i)
    # elif model_name == 'ssd':
    #     metrics = evaluate_ssd(
    elif model_name_i in ('yolov8s','yolov8m','yolov8l','yolov8x'):
        metrics = eval_yolov8(model_name_i, output_directory+model_name_i)

    r = display_results(metrics, model_name_i)
    results_all.append(r)

for r in results_all:
    print(r)