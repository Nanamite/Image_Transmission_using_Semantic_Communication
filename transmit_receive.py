import numpy as np
from channel_simulate import *

def tx_rx(section, coords, snr= 5, K= 1):
    x = coords[0]
    y = coords[1]
    w = coords[2]
    h = coords[3]

    #max_dim = np.max(section.shape[0], section.shape[1])

    x_bin = format(x, '09b')
    y_bin = format(y, '09b')
    w_bin = format(w, '09b')
    h_bin = format(h, '09b')

    # img_bin = np.unpackbits(img, axis=-1)
    # img_bin_vec = img_bin.flatten()
    section_bin_vector = ''
    section_flat = np.reshape(section, -1)

    for _, px in enumerate(section_flat):
            section_bin_vector += format(int(px), '08b')

    bitstream = x_bin + y_bin + w_bin + h_bin + section_bin_vector

    bitstream_received = channel_sim(bitstream, snr, K)

    x_bin_received = bitstream_received[0:9]
    y_bin_received = bitstream_received[9:18]
    w_bin_received = bitstream_received[18:27]
    h_bin_received = bitstream_received[27:36]
    section_bin_received = bitstream_received[36:]

    x_received = int(x_bin_received, 2)
    y_received = int(y_bin_received, 2)
    w_received = int(w_bin_received, 2)
    h_received = int(h_bin_received, 2)

    if w_received == 0:
        w_received = 512
    if h_received == 0:
        h_received = 512

    section_received = np.zeros((h_received*w_received), dtype= np.uint8)
    # print(section_flat.shape, section_received.shape)

    bit_num = 0
    for i in range(0, len(section_bin_received), 8):
        # print(bit_num, i, len(section_bin_received), h_received*w_received, w*h, h_bin_received, h_bin, w_bin_received, w_bin)
        # print('yea1')
        bits = section_bin_received[i:i + 8]
        # section_received[bit_num//w_received, bit_num%w_received] = int(bits, 2)
        section_received[bit_num] = int(bits, 2)
        bit_num += 1
        # print('yea2')

    # print(np.reshape(section, -1) == section_received)
    # print((section == section_received).all())

    section_received = np.reshape(section_received, (h_received, w_received))

    return section_received, x_received, y_received

if __name__ == '__main__':
    section = np.array([[1, 1, 0], [123, 231, 23], [123 ,42, 12]])
    coords = [10, 20, 3, 3]

    section_received, x, y, = tx_rx(section, coords)
    # print(section_received)
    # print(x, y)


    
