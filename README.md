# WaferDetector

Wafer Detector project for DIP 2023, Guangming Lu, HITSZ.


## Naive Wafer Detector

The key idea of Naive Wafer Detector is to extract the longest line in the image, and then use the line to extract the wafer. The algorithm is as below. We then introduce two-step-processing used in Naive method.

1. Preprocessing - Used for extract the wafer from the background in a coarse granularity.
    - Gaussian Blur
    - Sobel Edge Detection
    - Morphological Transformations for Edge Enhancement
    - Segment (Line) Detection
    - Select top K lines with an given error threshold (i.e., 100)
    - Draw the lines on the original image

2. Processing - Used for extract the wafer in a fine-tuned granularity.
    - Gaussian Blur
    - Sobel Edge Detection with a smaller threshold to obtain the coarse wafer boundary 
    - Segment (Line) Detection
    - Select top K lines with a smaller error threshold (i.e., 10)
    - Group the selected lines into two groups, one for the top and one for the bottom
    - Choose the top and bottom lines with the smallest center distance
    - Adjust two selected lines to be parallel
    - Calculate the saw line and its angle, and we're good

Naive Wafer Detector cannot handle the case there are multiple wafers in the image. We further design a detection algorithm as below.

