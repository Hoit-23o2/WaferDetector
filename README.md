# WaferDetector

Wafer Detector project for DIP 2023, Guangming Lu, HITSZ.

## Naive Wafer Detector

The key idea of Naive Wafer Detector is to extract the longest line in the image, and then use the line to extract the wafer. The challenge here is that the line tends to be noisy and discontinuous; thus, we need a way to connect these lines. To address the challenges, we introduce a three-step-processing algorithm, as described below.

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

3. Postprocessing - Used for detect if there are any other wafers.
    - Fill the wafer with white-black segment
    - Re-run Processing to try to find the second wafer
    - Compare the new wafer with the previous one, determine if they are the ``same'' wafer
    - If they are the same, then we're good 
    - Otherwise, we obtain a new wafer, and try to find the third one until we cannot find any more possible new wafers

Naive Wafer Detector is able to handle the case where two wafers are crossed. However, it is fragile when there are multiple ``horizontal'' and ``vertical'' wafers in the image since Naive Wafer Detector is primarily designed to detect the longest line in the image. We further design a detection algorithm as below.

## Point-based LineThrough Wafer Detector

The key idea of Point-based LineThrough Wafer Detector is to carefully determine the wafer sawing line based on the border points, enabling the more precise and robust detection of wafers sawing line. The challenges here are that the border points are noisy. To address the challenges, we introduce a clean-area based algorithm, as described below.

## Experimental Results

1. Naive Wafer Detector Result
    - Present the result

2. Point-based LineThrough Wafer Detector Result (Optimal)
    - Present the result
    
3. Comparison 
    - Subtraction
    - Present the result

## Discussion

    - Performance

    - Robustness

## Conclusion