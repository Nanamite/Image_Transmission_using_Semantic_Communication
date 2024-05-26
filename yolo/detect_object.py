import argparse
from collections import Counter
import os

import cv2 as cv
import torch
from yolo.util import load_classes, write_results
from yolo.darknet import Darknet
from yolo.preprocess import prep_image

import numpy as np
from profile_pytorch import profile


ROOT_DIR = os.getcwd()

def draw_object_labels(output_tensor, img, classes):
    """
    Draw bounding box w/ class label for each detected object
    """
    bb_coordinates1 = tuple(output_tensor[1:3].astype(np.int32))
    bb_coordinates2 = tuple(output_tensor[3:5].astype(np.int32))
    class_label = int(output_tensor[-1])

    label = "{0}".format(classes[class_label])
    color = (0, 0, 255)
    cv.rectangle(img, bb_coordinates1, bb_coordinates2, color, 1)
    t_size = cv.getTextSize(label, cv.FONT_HERSHEY_PLAIN, 1, 1)[0]
    bb_coordinates2 = bb_coordinates1[0] + t_size[0] + 3, bb_coordinates1[1] + t_size[1] + 4

    cv.rectangle(img, bb_coordinates1, bb_coordinates2, color, -1)
    cv.putText(
        img, label, (bb_coordinates1[0], bb_coordinates1[1] + t_size[1] + 4),
        cv.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1
    )
    return img

def predict_crop(img):
    print("Loading network.....")
    model = Darknet('./yolo/cfg/yolov3.cfg')
    model.load_weights("yolo/yolov3.weights")
    print("Network successfully loaded")

    model.net_info["height"] = 704#args.reso
    inp_dim = int(model.net_info["height"])

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    model = model.to(device)

    model.eval()

    num_ops, num_params = profile(model, (1, img.shape[2], img.shape[0], img.shape[1]))

    img, orig_im, dim = prep_image(img, inp_dim)
    im_dim = torch.FloatTensor(dim).repeat(1,2)

    with torch.no_grad():
        output = model(torch.autograd.Variable(img).to(device), True if torch.cuda.is_available() else False)
    classes = load_classes('./yolo/labels/coco-full.names')

    output = write_results(
        output, confidence=0.5, num_classes=len(classes), nms=True, nms_conf=0.4
    )

    output_dummy = []

    for obj in output:
        if classes[int(obj[-1])] == 'person':
            output_dummy.append(obj)

    output_dummy = torch.stack(output_dummy)

    class_counter = Counter([classes[int(obj[-1])] for obj in output_dummy])
    print("Class counts: " + str(class_counter))

    tot_objects = output_dummy.size(0)

    im_dim = im_dim.repeat(tot_objects, 1)
    scaling_factor = torch.min(inp_dim/im_dim, 1)[0].view(-1, 1)

    output_dummy = output_dummy.cpu()

    output_dummy[:, [1, 3]] -= (inp_dim - scaling_factor*im_dim[:, 0].view(-1, 1))/2
    output_dummy[:, [2, 4]] -= (inp_dim - scaling_factor*im_dim[:, 1].view(-1, 1))/2

    output_dummy[:, 1:5] /= scaling_factor

    for i in range(output_dummy.shape[0]):
        output_dummy[i, [1, 3]] = torch.clamp(output_dummy[i, [1, 3]], 0.0, im_dim[i, 0])
        output_dummy[i, [2, 4]] = torch.clamp(output_dummy[i, [2, 4]], 0.0, im_dim[i, 1])

    output_dummy = output_dummy.numpy()

    x_max = 0
    x_min = orig_im.shape[1]
    y_max = 0
    y_min = orig_im.shape[0]

    for obj in output_dummy:
        x = int(obj[1])
        y = int(obj[2])
        x_ = int(obj[3])
        y_ = int(obj[4])

        if x < x_min:
            x_min = x
        if x_ > x_max:
            x_max = x_
        if y < y_min:
            y_min = y
        if y_ > y_max:
            y_max = y_

    w = x_max - x_min
    h = y_max - y_min

    crop = orig_im[y_min:y_max, x_min:x_max]
    return crop, [x_min, y_min, w, h], num_ops, num_params


if __name__ == '__main__':
    img = cv.imread(r'data\people\bg_1\person_1\0.png')

    crop, coords = predict_crop(img)

    cv.imshow('', crop)
    cv.waitKey(0)

