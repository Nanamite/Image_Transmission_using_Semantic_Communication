from crop_person import crop
from replace_img import *
from transmit_receive import *
from ssim import *
import os

def main(img, background, save_dir= 'results', save= True, num_filter= 128, semantic_segment = True, snr= 10, K= 32):
    #Taking images to be of size 512 x 512
    #coords need 10 bits
    total_ops = 0
    total_params = 0

    person_segment, coords, num_ops_crop, num_params_crop = crop(img, num_filter, semantic_segment)

    total_ops += num_ops_crop
    total_params += num_params_crop
    # cv.imshow('', person_segment)
    # cv.waitKey(0)

    section_received, x_received, y_received = tx_rx(person_segment, coords, snr= snr, K= K)

    bits_img = img.shape[0]*img.shape[1]*8
    bits_sent = person_segment.shape[0]*person_segment.shape[1]*8
    bits_saved = bits_img - bits_sent

    # cv.imshow('', section_received)
    # cv.waitKey(0)
    bg_section_placed, num_replacings = place_in_bg(section_received, background, x_received, y_received)

    total_ops += num_replacings

    if save:
        cv.imwrite(os.path.join(save_dir, 'section_received.png'), section_received)
        cv.imwrite(os.path.join(save_dir, 'section_sent.png'), person_segment)
        cv.imwrite(os.path.join(save_dir, 'bg_section_replaced.png'), bg_section_placed)

        mse, nmse, psnr = MSE(img[:, :, 0], bg_section_placed)
        _, mssim = SSIM(img[:, :, 0], bg_section_placed)

        fp = open(os.path.join(save_dir, 'log.txt'), 'w+')
        fp.write(f'MSE loss: {mse}\n')
        fp.write(f'NMSE loss: {nmse}\n')
        fp.write(f'PSNR: {psnr}\n')
        fp.write(f'MSSIM: {mssim}\n')
        fp.write(f'bits for full image: {bits_img}\n')
        fp.write(f'bits sent: {bits_sent}\n')
        fp.write(f'bits saved: {bits_saved}, {100 * bits_saved/bits_img:.3f}%\n')
        fp.close()

    return bg_section_placed, person_segment, section_received, bits_img, bits_sent, bits_saved, total_ops, total_params

if __name__ == '__main__':
    img = cv.imread(r'data\people\bg_3\person_3\2.png')
    background = cv.imread(r'data\background\bg_3.png')

    #img = cv.resize(img, (500, 500))

    save_dir = 'results'
    save = False


    semantic = True
    replaced, _, _, _, _, _, total_ops, total_params = main(img, background, save= False, semantic_segment= semantic)

    print('semantic segmentation method: ', semantic)
    print('total number of ops= ', total_ops)
    print('total number of parameters= ', total_params)

    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv.imwrite(os.path.join(save_dir, 'result.png'), replaced)



