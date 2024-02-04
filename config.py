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

model_name = ["faster_rcnn"]#["yolov5s", "yolov5m", "yolov5l", "yolov5x", "yolov8m", "yolov8x", "faster_rcnn"]
# Find the index of the class
ball_class_index = COCO_INSTANCE_CATEGORY_NAMES.index('sports ball')


image_directory1 = "/media/yobiai/hugeDrive2/arshad/object_detection_inference/tracking_data/full/american_football/images"
label_path1 = "/media/yobiai/hugeDrive2/arshad/object_detection_inference/tracking_data/full/american_football/American-Football-Tracking-Testing-19-01-2024-14-25-08.csv"


image_directory2 = "/media/yobiai/hugeDrive2/arshad/object_detection_inference/tracking_data/full/basketball/images"
label_path2 = "/media/yobiai/hugeDrive2/arshad/object_detection_inference/tracking_data/full/basketball/Basketball-Ball-Tracking-Testing-19-01-2024-14-35-20.csv"

image_directory3 = "/media/yobiai/hugeDrive2/arshad/object_detection_inference/tracking_data/full/football/images"
label_path3 = "/media/yobiai/hugeDrive2/arshad/object_detection_inference/tracking_data/full/football/Football-Ball-Tracking-Testing-19-01-2024-14-18-34.csv"

image_directory4 = "/media/yobiai/hugeDrive2/arshad/object_detection_inference/tracking_data/tennis_ball/images"
label_path4 = "/media/yobiai/hugeDrive2/arshad/object_detection_inference/tracking_data/tennis_ball/Tennis-Ball-Tracking-Testing-15-01-2024-14-08-53.csv"

image_directory5 = "/media/yobiai/hugeDrive2/arshad/object_detection_inference/tracking_data/cricket_ball/images"
label_path5 = "/media/yobiai/hugeDrive2/arshad/object_detection_inference/tracking_data/cricket_ball/Cricket-Ball-Tracking-Testing-15-01-2024-14-11-25.csv"


image_directory = image_directory4
label_path = label_path4.replace(".csv", "_static.csv")

sport = "Tennis"
tag = "Static"

print(label_path)
output_directory = os.path.join(os.path.split(image_directory)[0],"annotated_images_")