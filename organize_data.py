import os
import cv2 as cv
import numpy as np

def resize_togray(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.resize(img_gray, (512, 512), interpolation= cv.INTER_AREA)

    return img_gray

background_dir = r'dataset_whatsapp\background'
people_dir = r'dataset_whatsapp\background_people'

save_dir = 'data'
background_save_dir = os.path.join(save_dir, 'background')
people_save_dir = os.path.join(save_dir, 'people')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if not os.path.exists(background_save_dir):
    os.makedirs(background_save_dir)

if not os.path.exists(people_save_dir):
    os.makedirs(people_save_dir)


for background in os.listdir(background_dir):
    people_bg_dir = os.path.join(people_dir, background[:-4])
    people_bg_save_dir = os.path.join(people_save_dir, background[:-4])

    bg_img_path = os.path.join(background_dir, background)

    background_img = cv.imread(bg_img_path)
    background_gray = resize_togray(background_img)

    cv.imwrite(os.path.join(background_save_dir, f'{background[:-4]}.png'), background_gray)

    for people_num in os.listdir(people_bg_dir):
        people_num_save_dir = os.path.join(people_bg_save_dir, people_num)
        people_num_dir = os.path.join(people_bg_dir, people_num)

        if not os.path.exists(people_num_save_dir):
            os.makedirs(people_num_save_dir)

        i = 0
        for people in os.listdir(people_num_dir):
            img_path = os.path.join(people_num_dir, people)
            img = cv.imread(img_path)

            img_gray = resize_togray(img)

            cv.imwrite(os.path.join(people_num_save_dir, f'{i}.png'), img_gray)
            i += 1