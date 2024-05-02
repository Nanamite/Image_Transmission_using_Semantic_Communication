import cv2
import numpy as np
import matplotlib.pyplot as plt
from polarcodes import *

def process_label(input_image):
    # Load the image
    # img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    
    label = input_image
    
    # Convert each pixel value to binary
    img_bin = np.unpackbits(label, axis=-1)

    # Flatten the image
    img_bin_vec = img_bin.flatten()

    # initialise polar code
    K = 100
    myPC = PolarCode(K*2, K) # PolarCode(N, K) K = Message length
    myPC.construction_type1 = 'bb'

    # mothercode construction
    design_SNR  = 3
    Construct(myPC, design_SNR)

    # calculate the number of chunks
    num_chunks = len(img_bin_vec) // myPC.K
    print(num_chunks)

    # initialize an empty list to hold the decoded chunks
    decoded_chunks = []

    for i in range(num_chunks):
        # get the current chunk
        chunk = img_bin_vec[i*myPC.K:(i+1)*myPC.K]

        # set the message
        myPC.set_message(chunk)

        # encode message
        Encode(myPC)

        # transmit the codeword
        AWGN(myPC, design_SNR)

        # decode the received codeword
        Decode(myPC)

        # append the decoded message to the list
        decoded_chunks.append(myPC.message_received)
        # if i*100/num_chunks % 10 < 1e-3:
        print(i*100/num_chunks) if i % 100 == 0 else None

    decoded_chunks = np.array(decoded_chunks)
    # concatenate the decoded chunks to form the output array
    output_array = np.reshape(decoded_chunks, -1)
    print(img_bin_vec.shape)
    print(decoded_chunks.shape)

    # reshape the output array to the original shape
    output_img_bin = output_array.reshape(img_bin.shape)

    # Check BER
    ber = np.sum(np.not_equal(img_bin, output_img_bin)) / img_bin.size
    print("Bit Error Rate: ", ber)
    
    # Convert the binary image to an integer image
    output_img = np.packbits(output_img_bin, axis=-1)

    # Save the output image
    return output_img

if __name__ == "__main__":
    label = np.random.randint(0, 36, (500, 1008), dtype= np.uint8)
    recon = process_label(label)
    print((label == recon).all())
    print(np.where(label != recon, 1, 0))
    print(np.where(recon > 35, 1, 0))