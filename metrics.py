import numpy as np

def calculate_iou(boxA, boxB):
    """ Calculate Intersection over Union of two boxes """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_precision_recall_f1(pred_boxes, true_boxes, iou_threshold=0.5):
    """ Calculate Precision, Recall, and F1-Score """
    # Special handling for cases with no predictions or no ground truths
    if len(pred_boxes) == 0 and len(true_boxes) == 0:
        return 1.0, 1.0, 1.0  # Perfect score, nothing was supposed to be detected and nothing was detected
    if len(pred_boxes) == 0:
        return 0.0, 0.0, 0.0  # No detections result in zero precision, zero recall
    if len(true_boxes) == 0:
        return 0.0, 0.0, 0.0  # No ground truths means any detection is a false positive

    TP, FP, FN = 0, 0, len(true_boxes)

    for pred_box in pred_boxes:
        max_iou = max([calculate_iou(pred_box, tb) for tb in true_boxes]) if true_boxes.size > 0 else 0
        if max_iou >= iou_threshold:
            TP += 1
            FN -= 1
        else:
            FP += 1

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score



def calculate_average_precision(precision, recall):
    """ Calculate the average precision (AP) for one image """
    precision = np.array(precision)
    recall = np.array(recall)
    indices = np.argsort(recall)
    precision = precision[indices]
    recall = recall[indices]

    precision = np.concatenate(([0], precision, [0]))
    recall = np.concatenate(([0], recall, [1]))

    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
    return ap