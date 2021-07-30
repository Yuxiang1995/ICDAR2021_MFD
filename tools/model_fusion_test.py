import os
import csv
import cv2
import numpy as np
from ensemble_boxes import *

img_dir = 'Your test image folder' # '/data4/dataset/formula_icadar2021/Ts10/img/'
# Your predict result folder
pred_dir1 = '/data4/model/formula/gfl/s101/v9/result'
pred_dir2 = '/data4/model/formula/gfl/s101/v10/result'
pred_list = [pred_dir1, pred_dir2]

nms_iou_thr = 0.4
filter_iou_thr = 0.1
first_stage_thr = 0.6
skip_box_thr = 0.05
sigma = 0.1
weights = [1, 1]
assert len(pred_list) == len(weights)

def bb_intersection_over_union(A, B) -> float:
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and GT rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def box2segm(box):
    return [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]


if __name__ == '__main__':
    rows = []
    for file in os.listdir(pred_list[0]):
        basename, _ = file.split('.')
        img = cv2.imread(os.path.join(img_dir, basename+'.jpg'))
        h, w, _ = img.shape
        boxes_list = []
        scores_list = []
        labels_list = []
        for dir in pred_list:
            boxes = []
            scores = []
            labels = []
            img = np.zeros((h, w, _), np.uint8)
            with open(os.path.join(dir, file), 'r') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    box = list(map(float, line[:4]))
                    box[0] /= w
                    box[1] /= h
                    box[2] /= w
                    box[3] /= h
                    score = float(line[-2])
                    label = int(line[-1])
                    flag = True
                    if score < first_stage_thr:
                        flag = False
                    if flag:
                        boxes.append(box)
                        scores.append(score)
                        labels.append(label)
            if not boxes:
                boxes.append([0.0, 0.0, 0.0, 0.0])
                scores.append(0.0)
                labels.append(0)
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)

        assert len(boxes_list) == len(scores_list) == len(labels_list)
        flag = False
        for scores in scores_list:
            if sum(scores):
                flag |= True
        if not flag:
            continue

        boxes_final, scores_final, labels_final = weighted_boxes_fusion(boxes_list, scores_list, labels_list,
                                                                        weights=weights, iou_thr=nms_iou_thr,
                                                                        skip_box_thr=skip_box_thr)

        boxes_final[:, 0::2] *= w
        boxes_final[:, 1::2] *= h
        for i in range(len(boxes_final)):
            row = [basename + '.jpg']
            row += boxes_final[i].tolist()
            row.append(scores_final[i])
            row.append(labels_final[i])
            rows.append(row)

    with open('/data4/dataset/formula_icdar2021/Ts10/result_fusion.csv', 'w') as g:
        g_csv = csv.writer(g)
        g_csv.writerows(rows)