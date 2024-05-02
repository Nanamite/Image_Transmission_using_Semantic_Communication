from polarcodes import *

def channel_sim(bitstream, snr= 5, K = 100):
    print('transmitting')
    myPC = PolarCode(K*2, K) # PolarCode(N, K) K = Message length
    myPC.construction_type1 = 'bb'

    bitstream_copy = []

    for bit in bitstream:
        bitstream_copy.append(int(bit))

    Construct(myPC, snr)

    num_chunks = len(bitstream) // myPC.K

    decoded_chunks = []

    for i in range(num_chunks):
        # get the current chunk
        chunk = bitstream_copy[i*myPC.K:(i+1)*myPC.K]

        # set the message
        myPC.set_message(chunk)

        # encode message
        Encode(myPC)

        # transmit the codeword
        AWGN(myPC, snr)

        # decode the received codeword
        Decode(myPC)

        # append the decoded message to the list
        decoded_chunks.append(myPC.message_received)
        # if i*100/num_chunks % 10 < 1e-3:
        print(i*100/num_chunks, ' %') if (i % 100 == 0) else None
        # percentage = i*100/num_chunks
        # percentage_int = (i*100)//num_chunks

        # if abs(percentage - percentage_int) < 0.002:
        #     print(f'{percentage_int} %')

    decoded_chunks = np.array(decoded_chunks)
    output_array = ''

    for bit_chunk in decoded_chunks:
        for bit in bit_chunk:
            output_array += str(bit)

    return output_array

if __name__ == '__main__':
    bitstream = '1000101001'

    received = channel_sim(bitstream, 5, 2)
    print(received)