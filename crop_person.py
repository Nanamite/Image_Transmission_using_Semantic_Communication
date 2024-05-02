import numpy as np
from PIL import Image
from predict import *

def crop(img):
    #img is an ndarray
    print('extracting person')

    label_map, _ = predict_map(img)
    person_class = 11

    img_copy = np.zeros_like(img)

    x_min = img_copy.shape[1]
    y_min = img_copy.shape[0]
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
                img_copy[i, j] = img[i, j]

    print('extraction done')
    img_copy = img_copy[y_min:y_max, x_min:x_max]

    return img_copy[:, :, 0], [x_min, y_min, x_max - x_min, y_max - y_min]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path')
    parser.add_argument('--save_path')
    args = parser.parse_args()

    img = cv.imread(args.img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    cropped, _  = crop(img)

    plt.imsave(args.save_path, cropped)
