from fastseg import MobileV3Large
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import argparse
from profile_pytorch import profile

def colorize(label_map):
    colormap = [
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32]
    ]

    colorized = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype= np.uint8)

    for i, row in enumerate(label_map):
        for j, label in enumerate(row):
            if label != -1:
                colorized[i, j] = colormap[label]

    return colorized

def predict_map(img, num_filters= 128):
    cityscapes_label = [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle"  
    ]

    model = MobileV3Large.from_pretrained(num_filters= num_filters).cuda().eval()

    num_ops, num_params = profile(model, (1, img.shape[2], img.shape[0], img.shape[1]))

    if img.shape[0] < 400 or img.shape[1] < 400:
        small_dim = np.argmin([img.shape[0], img.shape[1]])
        ratio = 400/img.shape[small_dim]
        img = cv.resize(img, (int(ratio*img.shape[1]), int(ratio*img.shape[0])), interpolation= cv.INTER_AREA)

    label_map = model.predict_one(img)
    unique_labels = np.unique(label_map)

    print('classes present:')
    for label in unique_labels:
        print(cityscapes_label[label])

    colored = colorize(label_map)
    # plt.figure()
    # plt.imshow(colored)
    # plt.show()

    return label_map, colored, num_ops, num_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path')
    parser.add_argument('--save_path')
    args = parser.parse_args()

    img = cv.imread(args.img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    label_map, colored, _, _ = predict_map(img)

    cv.imwrite(args.save_path, label_map)
