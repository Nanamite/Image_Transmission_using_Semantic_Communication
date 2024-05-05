from main import main
import os
from ssim import *
import cv2 as cv

#setting up dataset paths
root = 'data'
background_dir = os.path.join(root, 'background')
people_dir = os.path.join(root, 'people')

#save and method (to get RoI) option
#toggle semantic_segment to False to use object detection method
#change save_dir to a save directory of your choice
save_dir = 'test_results_3_segmentation'
semantic_segment = True

num_backgrounds = len(os.listdir(background_dir))


avg_bg_mse = np.array([0, 0, 0], dtype= np.float32)
avg_bg_nmse = np.array([0, 0, 0], dtype= np.float32)
avg_bg_psnr= np.array([0, 0, 0], dtype= np.float32)
avg_bg_mssim = np.array([0, 0, 0], dtype= np.float32)
avg_bg_bits_saved = np.array([0, 0, 0], dtype= np.float32)

for background in os.listdir(background_dir):
    bg = background[:-4]

    if background[-3:] == 'txt':
        continue

    background_path = os.path.join(background_dir, background)
    bg_img = cv.imread(background_path)

    diff_bg_dir = os.path.join(people_dir, bg)

    bg_result_path = os.path.join(save_dir, bg)

    if not os.path.exists(bg_result_path):
        os.makedirs(bg_result_path)


    bg_log = open(os.path.join(save_dir, 'bg_log.txt'), 'w+')

    print(f'{background} started')

    for person_num in os.listdir(diff_bg_dir):
        person_num_dir = os.path.join(diff_bg_dir, person_num)
        person_num_save_dir = os.path.join(bg_result_path, person_num)

        if not os.path.exists(person_num_save_dir):
            os.makedirs(person_num_save_dir)
        
        log = open(os.path.join(person_num_save_dir, 'log.txt'), 'w+')

        avg_mse = 0
        avg_nmse = 0
        avg_psnr = 0
        avg_mssim = 0
        avg_bits_saved = 0

        print(f'{background}, {person_num} started')

        for people in os.listdir(person_num_dir):
            print(f'{people} started')
            img_path = os.path.join(person_num_dir, people)
            img = cv.imread(img_path)

            img_save_dir = os.path.join(person_num_save_dir, people[:-4])

            if not os.path.exists(img_save_dir):
                os.makedirs(img_save_dir)

            # cv.imshow('', img)
            # cv.waitKey(0)

            bg_section_placed, section_sent, section_received, bits_img, bits_sent, bits_saved = main(img, bg_img, save= False, num_filter= 256, semantic_segment= semantic_segment)

            cv.imwrite(os.path.join(img_save_dir, 'reconstruction.png'), bg_section_placed)
            cv.imwrite(os.path.join(img_save_dir, 'section_sent.png'), section_sent)
            cv.imwrite(os.path.join(img_save_dir, 'section_received.png'), section_received)

            mse, nmse, psnr = MSE(img[:, :, 0], bg_section_placed)
            _, mssim = SSIM(img[:, :, 0], bg_section_placed)

            avg_mse += mse
            avg_nmse += nmse
            avg_psnr += psnr
            avg_mssim += mssim
            avg_bits_saved += 100 * bits_saved/bits_img

            print(f'{people} finished')

            log.write(f'{people}:\n')
            log.write('---------\n')
            log.write(f'MSE loss: {mse}\n')
            log.write(f'NMSE loss: {nmse}\n')
            log.write(f'PSNR: {psnr}\n')
            log.write(f'MSSIM: {mssim}\n')
            log.write(f'bits for full image: {bits_img}\n')
            log.write(f'bits sent: {bits_sent}\n')
            log.write(f'bits saved: {bits_saved}, {100 * bits_saved/bits_img:.3f}%\n')
            log.write('---------\n')

        avg_mse /= len(os.listdir(person_num_dir))
        avg_nmse /= len(os.listdir(person_num_dir))
        avg_psnr /= len(os.listdir(person_num_dir))
        avg_mssim /= len(os.listdir(person_num_dir))
        avg_bits_saved /= len(os.listdir(person_num_dir))

        log.write(f'AVG MSE loss: {avg_mse}\n')
        log.write(f'AVG NMSE loss: {avg_nmse}\n')
        log.write(f'AVG PSNR: {avg_psnr}\n')
        log.write(f'AVG MSSIM: {avg_mssim}\n')
        log.write(f'AVG bits saved: {avg_bits_saved:.3f}%\n')
        log.write('---------\n')
        log.close()
    
        person_idx = int(person_num[-1]) - 1

        avg_bg_mse[person_idx] += avg_mse
        avg_bg_nmse[person_idx] += avg_nmse
        avg_bg_psnr[person_idx] += avg_psnr
        avg_bg_mssim[person_idx] += avg_mssim
        avg_bg_bits_saved[person_idx] += avg_bits_saved
        print(avg_mse)
        print(avg_bg_mse)

        print(f'{background}, {person_num} finished')
    
    print(f'{background} finished')

avg_bg_mse = avg_bg_mse/num_backgrounds
avg_bg_nmse = avg_bg_nmse/num_backgrounds
avg_bg_psnr = avg_bg_psnr/num_backgrounds
avg_bg_mssim = avg_bg_mssim/num_backgrounds
avg_bg_bits_saved = avg_bg_bits_saved/num_backgrounds

bg_log.write('Average statistics over all backgrounds:\n')
bg_log.write('-------------------\n')
for i in range(3):
    bg_log.write(f'# of persons = {i + 1}:\n')
    bg_log.write(f'AVG MSE loss: {avg_bg_mse[i]}\n')
    bg_log.write(f'AVG NMSE loss: {avg_bg_nmse[i]}\n')
    bg_log.write(f'AVG PSNR: {avg_bg_psnr[i]}\n')
    bg_log.write(f'AVG MSSIM: {avg_bg_mssim[i]}\n')
    bg_log.write(f'AVG bits saved: {avg_bg_bits_saved[i]:.3f}%\n')
    bg_log.write('-------------------\n')

bg_log.write('Net Average:\n')
bg_log.write('-------------------\n')
bg_log.write(f'AVG MSE loss: {np.mean(avg_bg_mse)}\n')
bg_log.write(f'AVG NMSE loss: {np.mean(avg_bg_nmse)}\n')
bg_log.write(f'AVG PSNR: {np.mean(avg_bg_psnr)}\n')
bg_log.write(f'AVG MSSIM: {np.mean(avg_bg_mssim)}\n')
bg_log.write(f'AVG bits saved: {np.mean(avg_bg_bits_saved):.3f}%\n')
bg_log.write('-------------------\n')
bg_log.close()


