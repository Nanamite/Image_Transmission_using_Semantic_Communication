import cv2 as cv
import numpy as np

#places the RoI crop onto background
def place_in_bg(section, background, x, y):
    w = section.shape[1]
    h = section.shape[0]

    bg_copy = np.copy(background[:, :, 0])

    # cv.imshow('', section)
    # cv.waitKey(0)
    num_replacings = 0

    for i, row in enumerate(background[y:y + h, x:x + w]):
        for j, _, in enumerate(row):
            if section[i, j] != 0:
                bg_copy[y + i, x + j] = section[i, j]
                num_replacings += 1

    return bg_copy, num_replacings
