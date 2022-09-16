import torch
import cv2
import numpy as np

from shapely.geometry import Polygon
from math import pi 

from parameters import GrParams

params = GrParams()

def bboxes_to_grasps(bboxes):
    # convert bbox to grasp representation -> tensor([x, y, theta, h, w])
    x = bboxes[:,0] + (bboxes[:,4] - bboxes[:,0])/2
    y = bboxes[:,1] + (bboxes[:,5] - bboxes[:,1])/2 
    theta = torch.atan((bboxes[:,3] -bboxes[:,1]) / (bboxes[:,2] -bboxes[:,0]))
    w = torch.sqrt(torch.pow((bboxes[:,2] -bboxes[:,0]), 2) + torch.pow((bboxes[:,3] -bboxes[:,1]), 2))
    h = torch.sqrt(torch.pow((bboxes[:,6] -bboxes[:,0]), 2) + torch.pow((bboxes[:,7] -bboxes[:,1]), 2))
    grasps = torch.stack((x, y, theta, h, w), 1)
    return grasps


def grasps_to_bboxes(grasps):
    # convert grasp representation to bbox
    x = grasps[:,0] * 1024
    y = grasps[:,1] * 1024
    theta = torch.deg2rad(grasps[:,2] * 180 - 90)
    w = grasps[:,3] * 1024
    h = grasps[:,4] * 100
    
    x1 = x -w/2*torch.cos(theta) +h/2*torch.sin(theta)
    y1 = y -w/2*torch.sin(theta) -h/2*torch.cos(theta)
    x2 = x +w/2*torch.cos(theta) +h/2*torch.sin(theta)
    y2 = y +w/2*torch.sin(theta) -h/2*torch.cos(theta)
    x3 = x +w/2*torch.cos(theta) -h/2*torch.sin(theta)
    y3 = y +w/2*torch.sin(theta) +h/2*torch.cos(theta)
    x4 = x -w/2*torch.cos(theta) -h/2*torch.sin(theta)
    y4 = y -w/2*torch.sin(theta) +h/2*torch.cos(theta)
    bboxes = torch.stack((x1, y1, x2, y2, x3, y3, x4, y4), 1)
    return bboxes


def box_iou(bbox_value, bbox_target):
    p1 = Polygon(bbox_value.view(-1,2).tolist())
    p2 = Polygon(bbox_target.view(-1,2).tolist())
    iou = p1.intersection(p2).area / (p1.area +p2.area -p1.intersection(p2).area) 
    return iou


def get_correct_grasp_preds(output, target):
    bbox_output = grasps_to_bboxes(output)
    correct = 0
    for i in range(len(target)):
        bbox_target = grasps_to_bboxes(target[i])
        #print(output[i], target[i])
        for j in range(len(bbox_target)):
            iou = box_iou(bbox_output[i], bbox_target[j])
            pre_theta = output[i][2] * 180 - 90
            target_theta = target[i][j][2] * 180 - 90
            angle_diff = torch.abs(pre_theta - target_theta)
            
            if angle_diff < 30 and iou > 0.25:
                correct += 1
                break

    return correct, len(target)


def visualize_grasp(model, img, label):
    output = model(img)
    output_bbox = grasps_to_bboxes(output)
    target_bboxes = grasps_to_bboxes(label)

    img_vis = np.array(img.cpu())
    img_r = np.clip((img_vis[:, 0, :, :] * 0.229 + 0.485) * 255, 0, 255)
    img_g = np.clip((img_vis[:, 1, :, :] * 0.224 + 0.456) * 255, 0, 255)
    img_b = np.clip((img_vis[:, 2, :, :] * 0.225 + 0.406) * 255, 0, 255)
    
    img_bgr = np.concatenate((img_b, img_g, img_r), axis=0)
    img_bgr = np.moveaxis(img_bgr, 0, -1)
    img_bgr = np.ascontiguousarray(img_bgr, dtype=np.uint8)
    
    draw_bbox(img_bgr, output_bbox[0], (255, 0, 0))
    draw_bbox(img_bgr, target_bboxes[0], (0, 255, 0))

    #cv2.imshow('img', img_bgr)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return img_bgr


def draw_bbox(img, bbox, color):
    x1 = int(bbox[0] / 1024 * params.OUTPUT_SIZE)
    y1 = int(bbox[1] / 1024 * params.OUTPUT_SIZE)
    x2 = int(bbox[2] / 1024 * params.OUTPUT_SIZE)
    y2 = int(bbox[3] / 1024 * params.OUTPUT_SIZE)
    x3 = int(bbox[4] / 1024 * params.OUTPUT_SIZE)
    y3 = int(bbox[5] / 1024 * params.OUTPUT_SIZE)
    x4 = int(bbox[6] / 1024 * params.OUTPUT_SIZE)
    y4 = int(bbox[7] / 1024 * params.OUTPUT_SIZE)
    cv2.line(img, (x1, y1), (x2, y2), color, 1)
    cv2.line(img, (x2, y2), (x3, y3), (0, 0, 255), 1)
    cv2.line(img, (x3, y3), (x4, y4), color, 1)
    cv2.line(img, (x4, y4), (x1, y1), (0, 0, 255), 1)