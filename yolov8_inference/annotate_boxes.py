import os
from PIL import Image, ImageDraw

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def draw_boxes_on_image(image, boxes, color, label):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        # Draw rectangle
        draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color, width=2)
        # Optionally, add label text next to the box
        draw.text((box[0], box[1]), label, fill=color)