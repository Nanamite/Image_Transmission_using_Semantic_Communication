import numpy as np
from PIL import Image
from predict import *
from yolo.detect_object import predict_crop

def crop(img, num_filter= 128, semantic_segment= True):
    #img is an ndarray
    print('extracting person')
    if semantic_segment:
        print('using semantic segmentation')
        label_map, _ = predict_map(img, num_filter)
        person_class = 11

        crop = np.copy(img)

        x_min = crop.shape[1]
        y_min = crop.shape[0]
        x_max = 0
        y_max = 0
        for i, row in enumerate(label_map):
            for j, label in enumerate(row):
                if label == person_class:
                    if j < x_min:
                        x_min = j
                    if i < y_min:
                        y_min = i
                    if j > x_max:
                        x_max = j
                    if i > y_max:
                        y_max = i
                    #img_copy[i, j] = img[i, j]
        if x_max > x_min:
            crop = crop[y_min:y_max, x_min:x_max]
            w = x_max - x_min
            h = y_max - y_min
    else:
        print('using object detection')
        crop, [x_min, y_min, w, h] = predict_crop(img)
        x_max = x_min + w
        y_max = y_min + h

    print('extraction done')

    if x_max <= x_min:
        x_max = x_min
        y_max = y_min
        x_min = 0
        y_min = 0
        crop = np.copy(img)
        w = 0
        h = 0

    return crop[:, :, 0], [x_min, y_min, w, h]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path')
    parser.add_argument('--save_path')
    args = parser.parse_args()

    img = cv.imread(args.img_path)
    #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    cropped, _  = crop(img, semantic_segment= False)

    plt.imsave(args.save_path, cropped)
