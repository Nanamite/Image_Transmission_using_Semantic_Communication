import numpy as np

def MSE(img1, img2):
    img1 = np.array(img1, np.int32)
    img2 = np.array(img2, np.int32)
    mse_loss = np.mean(np.square(img1 - img2))
    psnr = 10*np.log10((255**2)/mse_loss)

    return mse_loss, psnr

def pad(img, window_size):
    pad_size = window_size//2

    img_padded = np.copy(img)
    img_padded = np.append(img, np.zeros((img.shape[0], pad_size)), axis= 1)
    img_padded = np.append(np.zeros((img_padded.shape[0], pad_size)), img_padded, axis= 1)
    img_padded = np.append(img_padded, np.zeros((pad_size, img_padded.shape[1])), axis= 0)
    img_padded = np.append(np.zeros((pad_size, img_padded.shape[1])), img_padded, axis= 0)

    return img_padded

def correlate(patch, kernel):
    return np.sum(patch*kernel)

def SSIM(img1, img2):
    img1 = np.array(img1, np.int32)
    img2 = np.array(img2, np.int32)
    window_size = 11
    sigma = 1.2

    gauss = np.zeros((window_size, window_size))

    for i, row in enumerate(gauss):
        for j, _ in enumerate(row):
            i_ = i - window_size//2
            j_ = j - window_size//2
            gauss[i, j] = np.exp(-1*(i_**2 + j_**2)/(2*sigma**2))/(2*np.pi*sigma**2)
    
    gauss /= np.sum(gauss)

    img1_padded = pad(img1, window_size)
    img2_padded = pad(img2, window_size)

    ssim_map = np.zeros(img1.shape)
    c1 = (0.01*255)**2
    c2 = (0.03*255)**2
    c3 = c2/2

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            i_ = i + window_size//2
            j_ = j + window_size//2

            patch_x = img1_padded[i_ - window_size//2:i_ + window_size//2 + 1, j_ - window_size//2:j_ + window_size//2 + 1]
            patch_y = img2_padded[i_ - window_size//2:i_ + window_size//2 + 1, j_ - window_size//2:j_ + window_size//2 + 1]

            mu_x = correlate(patch_x, gauss)
            mu_y = correlate(patch_y, gauss)

            sigma_x = np.sqrt(correlate(np.square(patch_x - np.mean(patch_x)), gauss))
            sigma_y = np.sqrt(correlate(np.square(patch_y - np.mean(patch_y)), gauss))

            sigma_xy = correlate((patch_x - np.mean(patch_x))*(patch_y - np.mean(patch_y)), gauss)

            l = (2*mu_x*mu_y + c1)/(mu_x**2 + mu_y**2 + c1)
            c = (2*sigma_x*sigma_y + c2)/(sigma_x**2 + sigma_y**2 + c2)
            s = (sigma_xy + c3)/(sigma_x*sigma_y + c3)

            ssim_map[i, j] = l*c*s

    mssim = np.mean(ssim_map)

    return ssim_map, mssim