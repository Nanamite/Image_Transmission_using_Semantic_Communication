# Image_Transmission_using_Semantic_Communication
-----------------
This is the repository for the course project in Wireless Communication (EE3801)

K R Nandakishore

Rayani Venkat Sai Rithvik

-----------------

## Acknowledgements:
- We thank Dr. Zafar Ali Khan and the course TAs for their constant support and guidance throughout the course and project
- For the creation of the dataset, thanks to
  - Komaragiri Sai Pranav
  - A P Vaideeswaran
  - Harishankar M
  - Kartik Agrawal

-----------------

For testing over the dataset, make use of `test.py`.

file descriptions:

- `main.py`: the main pipeline. Reads an image (change path within code), crops the RoI, transmits it, reconstruct by placing the crop on background.
- `predict.py`: Does the Semantic Segmentation mapping and returns it
- yolo/`detect_object.py`: Does object detection, returns the crop, coordinates, width and height
- `transmit_receive.py`: Simulates the transmission and receiving of the bitstream as well as reconstruction
- `channel_simulate.py`: Polar encodes a bitstream, simulates AWGN channel, and transmit it
- `replace_img.py`: Places Region of Interest (RoI) crop on background
- `ssim.py`: contains the metrics used
