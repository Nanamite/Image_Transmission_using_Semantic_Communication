import cv2 as cv
import os

vid = cv.VideoCapture('vid_with_person.mp4')

save_dir = 'person'
bg= 1

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

i = 0
while True:
    ret, frame = vid.read()
    if not ret:
        break

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cv.imwrite(os.path.join(save_dir, f'frame_{i}.png'), frame)

    if bg:
        break

    i += 1