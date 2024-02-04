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
        max_iou = max([calculate_iou(pred_box, tb) for tb in true_boxes]) if len(true_boxes) > 0 else 0
        if max_iou >= iou_threshold:
            TP += 1
            FN -= 1
        else:
            FP += 1

    return TP, FP, FN


def calculate_average_precision(predictions, ground_truths, iou_threshold=0.3):
    """
    Calculate the average precision (AP) for object detection.
    
    Parameters:
    - predictions: A list of tuples [(box, score)], where 'box' is [x1, y1, x2, y2] and 'score' is the confidence.
    - ground_truths: A list of ground truth bounding boxes in the format [x1, y1, x2, y2].
    - iou_threshold: The IoU threshold to consider a detection as a True Positive.
    
    Returns:
    - average_precision: The average precision for the detections.
    """
    predictions.sort(key=lambda x: x[1], reverse=True)  # Sort by confidence score in descending order
    TP = np.zeros(len(predictions))
    FP = np.zeros(len(predictions))
    used_ground_truths = []

    for i, pred in enumerate(predictions):
        pred_box, _ = pred
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(ground_truths):
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou and gt_idx not in used_ground_truths:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_iou >= iou_threshold:
            TP[i] = 1
            used_ground_truths.append(best_gt_idx)
        else:
            FP[i] = 1
    
    TP_cumsum = np.cumsum(TP)
    FP_cumsum = np.cumsum(FP)
    precisions = TP_cumsum / (TP_cumsum + FP_cumsum)
    recalls = TP_cumsum / len(ground_truths)

    # Add a point at (recall=0, precision=1) for AP calculation
    precisions = np.concatenate(([1], precisions))
    recalls = np.concatenate(([0], recalls))

    # Calculate AP as the area under the precision-recall curve
    average_precision = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
    
    return average_precision



# def calculate_average_precision(precision, recall):
#     """ Calculate the average precision (AP) for one image """
#     precision = np.array(precision)
#     recall = np.array(recall)
#     indices = np.argsort(recall)
#     precision = precision[indices]
#     recall = recall[indices]

#     precision = np.concatenate(([0], precision, [0]))
#     recall = np.concatenate(([0], recall, [1]))

#     for i in range(len(precision) - 1, 0, -1):
#         precision[i - 1] = max(precision[i - 1], precision[i])

#     ap = np.sum((recall[1:] - recall[:-1]) * precision[1:])
#     return ap