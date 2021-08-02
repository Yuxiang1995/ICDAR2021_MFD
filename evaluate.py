
import cv2
import numpy as np
import sys
import os
import pandas as pd
import warnings
from tqdm import tqdm
import argparse


def IoU(boxA, boxB):
    """
    Computes the Intersection over Union (IoU) metric, given
    two bounding boxes.
    Input:
    "boxA": bounding box A
    "boxB": bounding box B
    Output:
    "score": IoU score
    """

    # Compute the intersection points of the two BBs
    xLeft = max(boxA[0], boxB[0])
    yLeft = max(boxA[1], boxB[1])
    xRight = min(boxA[2], boxB[2])
    yRight = min(boxA[3], boxB[3])

    # Compute the area of the intersection rectangle
    interArea = max(0, xRight - xLeft + 1) * max(0, yRight - yLeft + 1)    

    # Compute the area of both boxA and boxB rectangles
    boxA_area = (boxA[2]-boxA[0] + 1)*(boxA[3]-boxA[1] + 1)
    boxB_area = (boxB[2]-boxB[0] + 1)*(boxB[3]-boxB[1] + 1)

    # Compute the intersection over union
    score = interArea / float(boxA_area + boxB_area - interArea)

    return score

def load_predicted(path):
    """
    Loads the bounding boxes of csv file "path".
    Input:
    "path": csv file containing one bounding box per line. 
    Output:
    "pages": dictionary with key = page_name, value = array containing
    the predicted bounding boxes    
    """   
    rows = pd.read_csv(path, header=None).to_numpy()
    pages = {}

    with tqdm(desc="Processing '" + path + "':", total=len(rows), ascii=True, bar_format='{desc} |{bar:10}| {percentage:3.0f}%') as pbar:
        for r in rows:
            # Remove the bounding boxes that have a confidence score of less than 0.05
            if r[-2] >= 0.05:
                page_name = r[0]
                list_BBs = pages.get(page_name, {})

                # [xmin, ymin, xmax, ymax, class_id, confidence]
                bb = np.array([int(r[1]),int(r[2]),int(r[3]),int(r[4]),int(r[6]),r[5]])

                if bb[4] == 0: # label
                    emb = list_BBs.get("embedded", np.array([]))
                    emb = np.vstack([emb, bb]) if len(emb > 0) else np.array([bb])
                    list_BBs["embedded"] = emb
                else:
                    iso = list_BBs.get("isolated", np.array([]))
                    iso = np.vstack([iso, bb]) if len(iso > 0) else np.array([bb])
                    list_BBs["isolated"] = iso

                pages[page_name] = list_BBs
                pbar.update(1)
    
    return pages

def load_gt(path):
    """
    Loads the bounding boxes of directory "path".
    Input:
    "path": path of folder containing a grount-truth txt file for every page image
    Output:
    "pages": dictionary with key = page_name, value = array containing
    the ground truth bounding boxes
    """
    pages = {}

    with tqdm(desc="Processing GT:", total=len(os.listdir(path))/2, ascii=True, bar_format='{desc} |{bar:10}| {percentage:3.0f}%') as pbar:
        for f in os.listdir(path):
            if f.endswith("txt"):
                img_name = f.replace("color_","").replace("txt","jpg")
                full_path = os.path.join(path, img_name)
                img = cv2.imread(full_path)

                h_img, w_img = img.shape[:2]    
                
                # Read the lines of the gt file
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    line = np.genfromtxt(os.path.join(path,f), usecols=(0,1,2,3,4),comments='#',)
                    if len(line) == 0:
                        pages[img_name] = {"isolated":np.array([]),"embedded":np.array([])}
                    else:
                        if len(line.shape) == 1: line = [line]            
                        for l in line:     
                            x = int(l[0] * w_img / 100)
                            y = int(l[1] * h_img / 100)
                            w = int(l[2] * w_img / 100)
                            h = int(l[3] * h_img / 100)
                            label = int(l[4])

                            # Dictionary key = page img name
                            list_BBs = pages.get(img_name, {})

                            bb = np.array([x, y, x+w-1, y+h-1, label, 0, 0])

                            if bb[4] == 0: # label
                                emb = list_BBs.get("embedded", np.array([]))
                                emb = np.vstack([emb, bb]) if len(emb > 0) else np.array([bb])
                                list_BBs["embedded"] = emb
                            else:
                                iso = list_BBs.get("isolated", np.array([]))
                                iso = np.vstack([iso, bb]) if len(iso > 0) else np.array([bb])
                                list_BBs["isolated"] = iso                           

                            pages[img_name] = list_BBs
                pbar.update(1)
            
    return pages

def display_results(true_pos_IoU, total_pred, total_gt):
    """
    Display the F1 score based on both IoU metrics.
    Input:
    "true_pos_IoU": number of true positive detections given a IoU threshold
    "total_pred": total number of predicted bounding boxes
    "total_gt": total number of ground-truth bounding boxes
    """

    precision_emb_IoU = true_pos_IoU["embedded"] / total_pred["embedded"] if total_pred["embedded"] != 0 else 0
    recall_emb_IoU = true_pos_IoU["embedded"] / total_gt["embedded"] if total_gt["embedded"] != 0 else 0   

    precision_iso_IoU = true_pos_IoU["isolated"] / total_pred["isolated"] if total_pred["isolated"] != 0 else 0
    recall_iso_IoU = true_pos_IoU["isolated"] / total_gt["isolated"] if total_gt["isolated"] != 0 else 0

    #####################################################################################
    #######  F1 score - IoU metric ######################################################
    print("\nF1-score using IoU metric")
    # F1 score for embedded mathematical expressions using IoU metric
    f1_score_emb_IoU = round(2*precision_emb_IoU*recall_emb_IoU/(precision_emb_IoU+recall_emb_IoU),4)
    print("F1-score embedded:\t",f1_score_emb_IoU*100, "( p:", round(precision_emb_IoU,4)*100, ", r:", round(recall_emb_IoU,4)*100, ")")    

    # F1 score for isolated mathematical expressions using IoU metric
    f1_score_iso_IoU = round(2*precision_iso_IoU*recall_iso_IoU/(precision_iso_IoU+recall_iso_IoU),4)
    print("F1-score isolated:\t",f1_score_iso_IoU*100, "( p:", round(precision_iso_IoU,4)*100, ", r:", round(recall_iso_IoU,4)*100, ")")    

    # F1 score for whole sistem using IoU metric
    prec_IoU = (true_pos_IoU["embedded"]+true_pos_IoU["isolated"]) / (total_pred["embedded"]+total_pred["isolated"])
    recall_IoU = (true_pos_IoU["embedded"]+true_pos_IoU["isolated"]) / (total_gt["embedded"]+total_gt["isolated"])
    f1_score_IoU = round(2* prec_IoU*recall_IoU / (prec_IoU+recall_IoU),4)
    print("F1-score whole system:\t", f1_score_IoU*100, "( p:", round(prec_IoU,4)*100, ", r:", round(recall_IoU,4)*100, ")")
    

def compute_F1(pred, gt, img_path):    
    """
    Compute the number of true positive detections based on IoU and GIoU thresholds.
    Generate an image for every page of the ground-truth with the predicted bounding
    boxes marked in red color and the ground-truth bounding boxes marked in green color. 
    Input:
    "pred": dictionary with an entry for each ground-truth page containing the predicted
    bounding-boxes separated by class.
    "gt": dictionary with an entry for each ground-truth page containing the ground-truth
    bounding-boxes separated by class. 
    "img_path": path of the ground-truth folder.
    """
    
    true_pos_IoU = {"embedded":0, "isolated":0}
    total_pred = {"embedded":0, "isolated":0}
    total_gt = {"embedded":0, "isolated":0}

    if gen_output:
        results_path = "Results"
        os.makedirs(results_path,exist_ok=True)

    with tqdm(desc="Generating results:", total=len(gt.keys()), ascii=True, bar_format='{desc} |{bar:10}| {percentage:3.0f}%') as pbar:
        for key in gt.keys():
            gt_bbs = gt.get(key)
            pred_bbs = pred.get(key, {})

            # Ground-truth image
            img = cv2.imread(os.path.join(img_path, key))
            h, w = img.shape[:2]

            for i in ["embedded", "isolated"]:
                gt_exps = gt_bbs.get(i, np.array([]))
                pred_exps = pred_bbs.get(i, np.array([]))

                total_gt[i] += len(gt_exps)
                total_pred[i] += len(pred_exps)

                tagged_pred_IoU = set()

                simil_mat_IoU = []
                for bb in gt_exps:            
                    # Calculate the best IoU score for a given ground truth bounding box
                    IoU_scores = np.array([IoU(bb, pred) for pred in pred_exps])
                    best_IoU = None
                    score_IoU = 0
                    for j in range(len(IoU_scores)):
                        if j not in tagged_pred_IoU:
                            if IoU_scores[j] > score_IoU:
                                score_IoU = IoU_scores[j]
                                best_IoU = j
                                            
                    simil_mat_IoU.append(list(IoU_scores))

                    if score_IoU >= 0.7:
                        true_pos_IoU[i] += 1
                        tagged_pred_IoU.add(best_IoU) 

                    if gen_output:
                        # Draw the ground-truth bounding boxes
                        x_min = bb[0]
                        y_min = bb[1]
                        x_max = bb[2]
                        y_max = bb[3]
                        
                        cv2.rectangle(img, (x_min,y_min), (x_max,y_max), (0,255,0), 2)

                        # Write the corresponding IoU score
                        cv2.putText(img, "{:.4f}".format(score_IoU), (int((x_max - x_min)/2 + x_min - int(0.024*w)), y_min-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                
                if gen_output:
                    for bb in range(len(pred_exps)):
                        x_min = int(pred_exps[bb][0])
                        y_min = int(pred_exps[bb][1])
                        x_max = int(pred_exps[bb][2])
                        y_max = int(pred_exps[bb][3])

                        # Draw the predicted bounding boxes
                        cv2.rectangle(img, (x_min,y_min), (x_max,y_max), (0,0,255), 2)

                        if bb not in tagged_pred_IoU:
                            # Write the corresponding max IoU (predicted bounding box not paired with gt)
                            aux = [simil_mat_IoU[j][bb] for j in range(len(simil_mat_IoU))]
                            score = max(aux) if len(simil_mat_IoU) > 0 else 0
                            cv2.putText(img, "{:.4f}".format(score), (int((x_max - x_min)/2 + x_min - int(0.024*w)), y_min-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            
                    
            if gen_output: cv2.imwrite(os.path.join(results_path, key), img)   
            
            pbar.update(1)
    
    display_results(true_pos_IoU, total_pred, total_gt)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("GT_DIR", help="path of folder containing the ground-truth provided (txt + jpg).")
    parser.add_argument("PREDICT_FILE", help="path of csv file containing the predicted bounding boxes.")
    parser.add_argument("--output", action="store_true", help='generate an image for every page of the\
        ground-truth with the predicted bounding boxes marked in red color and the ground-truth bounding\
        boxes marked in green color. Save the output in the "Results" directory.')
    args = parser.parse_args()

    gt_path = args.GT_DIR
    pred_path = args.PREDICT_FILE
    gen_output = args.output

    # Check if the argument directories are valid
    if not os.path.isfile(pred_path):
        print(pred_path + " is not a valid file.")
        exit()    
    if not os.path.isdir(gt_path):
        print(gt_path + " is not a valid directory.")
        exit()

    # Dictionary: key = page_img_name, value = dictionary with lists
    # of predicted BBs separated by class
    pred_dict = load_predicted(pred_path)
       
    # Dictionary: key = page_img_name, value = dictionary with lists
    # of predicted BBs separated by class
    gt_dict = load_gt(gt_path)
    
    # Draw the predicted and ground-truth bounding boxes and the resulting
    # IoU and GIoU scores
    compute_F1(pred_dict, gt_dict, gt_path)
