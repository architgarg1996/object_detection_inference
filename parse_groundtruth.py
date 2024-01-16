import json
import pandas as pd


def extract_image_name(url):
    return url.split('/')[-1]

def parse_annotations(annotation_col):
    annotations = json.loads(annotation_col)
    bboxes = []
    for rect in annotations.get('rect', []):
        x_min, y_min, width, height = rect[:4]
        x_max = x_min + width
        y_max = y_min + height
        bboxes.append([x_min, y_min, x_max, y_max])
    return bboxes

def aggregate_bboxes(bboxes_list):
    # Aggregate bounding boxes, handling empty lists
    aggregated_bboxes = []
    for bboxes in bboxes_list:
        aggregated_bboxes.extend(bboxes)
    return aggregated_bboxes

def parse_labels(label_path):
    labels = pd.read_csv(label_path)
    labels['image_name'] = labels['Source Url'].apply(extract_image_name)
    labels['bboxes'] = labels['annotation'].apply(parse_annotations)
    grouped_labels = labels.groupby('image_name')['bboxes'].apply(aggregate_bboxes)
    return grouped_labels.to_dict()