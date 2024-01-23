import os

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

model_name = 'yolov7.pt' #yolov5s, yolov5m, yolov5l, yolov5x, faster_rcnn
# Find the index of the class
ball_class_index = COCO_INSTANCE_CATEGORY_NAMES.index('sports ball')


image_directory = "D:/Freelancing/2023-2024/object_detection_inference/tracking_data/basketball/images"
label_path = "D:/Freelancing/2023-2024/object_detection_inference/tracking_data/basketball/Basketball-Ball-Tracking-Testing-15-01-2024-13-56-07.csv"


# image_directory = "D:/Freelancing/2023-2024/object_detection_inference/tracking_data/american_football/images"
# label_path = "D:/Freelancing/2023-2024/object_detection_inference/tracking_data/american_football/American-Football-Tracking-Testing-15-01-2024-13-55-12.csv"

# image_directory = "D:/Freelancing/2023-2024/object_detection_inference/tracking_data/cricket_ball/images"
# label_path = "D:/Freelancing/2023-2024/object_detection_inference/tracking_data/cricket_ball/Cricket-Ball-Tracking-Testing-15-01-2024-14-11-25.csv"

# image_directory = "D:/Freelancing/2023-2024/object_detection_inference/tracking_data/tennis_ball/images"
# label_path = "D:/Freelancing/2023-2024/object_detection_inference/tracking_data/tennis_ball/Tennis-Ball-Tracking-Testing-15-01-2024-14-08-53.csv"

output_directory = os.path.join(os.path.split(image_directory)[0],"annotated_images_"+model_name)