from crop_person import *
from replace_img import *
from transmit_receive import *
from ssim import *
import os

def main(img, background, save_dir= 'results'):
    #Taking images to be of size 500 x 500
    #coords need 10 bits
    person_segment, coords = crop(img)
    # print(coords)

    # cv.imshow('', person_segment)
    # cv.waitKey(0)

    section_received, x_received, y_received = tx_rx(person_segment, coords, snr= 5, K= 32)

    bits_img = img.shape[0]*img.shape[1]*8 + 18
    bits_sent = person_segment.shape[0]*person_segment.shape[1]*8 + 36
    bits_saved = bits_img - bits_sent

    # cv.imshow('', section_received)
    # cv.waitKey(0)

    cv.imwrite(os.path.join(save_dir, 'section_received.png'), section_received)
    cv.imwrite(os.path.join(save_dir, 'section_sent.png'), person_segment)

    bg_section_placed = place_in_bg(section_received, background, x_received, y_received)

    cv.imwrite(os.path.join(save_dir, 'bg_section_replaced.png'), bg_section_placed)

    mse, psnr = MSE(img[:, :, 0], bg_section_placed)
    _, mssim = SSIM(img[:, :, 0], bg_section_placed)

    fp = open(os.path.join(save_dir, 'log.txt'), 'w+')
    fp.write(f'MSE loss: {mse}\n')
    fp.write(f'PSNR: {psnr}\n')
    fp.write(f'MSSIM: {mssim}\n')
    fp.write(f'bits for full image: {bits_img}\n')
    fp.write(f'bits sent: {bits_sent}\n')
    fp.write(f'bits saved: {bits_saved}, {100 * bits_saved/bits_img:.3f}%\n')
    fp.close()

    return bg_section_placed

if __name__ == '__main__':
    img = cv.imread(r'person\frame_0.png')
    background = cv.imread(r'background\frame_0.png')

    if img.shape[0] < 400 or img.shape[1] < 400:
        small_dim = np.argmin([img.shape[0], img.shape[1]])
        ratio = 400/img.shape[small_dim]
        img = cv.resize(img, (int(ratio*img.shape[1]), int(ratio*img.shape[0])), interpolation= cv.INTER_AREA)

    #img = cv.resize(img, (500, 500))

    #background = np.copy(img)
    save_dir = 'results'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    replaced = main(img, background)

    #cv.imwrite(os.path.join(save_dir, 'result.png'), replaced)



