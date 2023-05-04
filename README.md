# WaferDetector

Wafer Sawing Line Detector project for DIP 2023, Guangming Lu, HITSZ.

## 1. QuickStart

```bash
pip install numpy, opencv-python, matplotlib, scipy
```

### 1.1 Running Naive Detector

```bash
cd detectors && python detect.py
```

- The preprocessed results will be saved in the `detectors/pre-output` folder.
- The mid results will be saved in the `detectors/mid-output` folder.
- The final results will be saved in the `detectors/final-output` folder.

You can also run ND with notebook at `notebooks/naive_detector.ipynb`. Just click the `Run All` button and you will get the results.

### 1.2 Running Line Through Detector

```bash
cd detectors && python detectPB.py
```

Note that PB here denotes point-based.

- The mid results will be saved in the `detectors/pb-mid-output` folder.
- The final results will be saved in the `detectors/pb-final-output` folder.

You can cancel the showing image by assigning `showResult = False` in the `detectPB.py` file, line 107.

### 1.3 Comparing with Golden Results

```bash
cd detectors && python measurement.py
```

### 1.4 Performance Evaluation

```bash
cd detectors

echo "Naive Detector"
time python detect.py

echo "Line Through Detector"
time python detectPB.py
```

The *real* field in the output indicates the real time of processing the images. 

## 2. Introduction
### 2.1 Naive Detector (*ND*)

The key idea of Naive Detector is to extract the longest line in the image, and then use the line to extract the wafer sawing lines. The challenge here is that the line tends to be noisy and discontinuous; thus, we need a way to connect these lines. To address the challenges, we introduce a three-step-processing algorithm, as described below.

1. Preprocessing - Used for extract the sedges from the background in a coarse granularity.
    - Gaussian Blur
    - Sobel Edge Detection
    - Morphological Transformations for Edge Enhancement
    - Segment (Line) Detection
    - Select top K lines with an given error threshold (i.e., 100)
    - Draw the lines on the original image

2. Processing - Used for extract the sedges in a fine-tuned granularity.
    - Gaussian Blur
    - Sobel Edge Detection with a smaller threshold to obtain the coarse wafer boundary 
    - Segment (Line) Detection
    - Select top K lines with a smaller error threshold (i.e., 10)
    - Group the selected lines into two groups, one for the top and one for the bottom
    - Choose the top and bottom lines with the smallest center distance
    - Adjust two selected lines to be parallel
    - Calculate the sawing line and its angle, and we're good

3. Postprocessing - Used for detect if there are any other wafer sawing lines.
    - Fill the sawing path with white-black segment
    - Re-run Processing to try to find the second wafer sawing line
    - Compare the new wafer sawing line with the previous one, determine if they are the ``same''
    - If they are the same, then we're good 
    - Otherwise, we obtain the new wafer sawing line, and try to find the third one until we cannot find any more possible new wafer sawing lines.

Naive Detector is able to handle the case where two sawing paths are crossed. However, it is fragile when there are multiple ``horizontal'' and ``vertical'' sawing paths in the image since Naive Detector is primarily designed to detect the longest line in the image. We further design a detection algorithm as below.

### 2.2 Point-based Line Through Detector (*LTD*)

The key idea of Point-based LineThrough Detector is to carefully determine the wafer sawing line based on the border points, enabling the more precise and robust detection of wafers sawing line. The challenges here are that the border points are noisy. To address the challenges, we introduce a clean-area based algorithm, as described below.

### 2.3 Experimental Results

1. ND Result

    ![NaiveResults](/results/results.png "Results of Naive Detector")

2. LTD Result (Optimal)
    
    ![LDTResults](/results/results-ltd.png "Results of Line Through Detector")

3. Robustness

    ![Robustness](/results/robustness.png "Robustness")

4. Performance

    - **Testbed**. We run the experiments on a Windows 10 machine with an Intel i7-11800H CPU clocked at 2.30GHz and 16GB memory.

    - **Result**. We study the performance of ND and LTD in this section. Specifically, we use time command to measure the real time of processing Wafer 1-4 and the corrupted Wafer 4. ND spends around **3.459s** while LTD spends more than **30s** since LTD performs complex verifications to ensure the correction of detected sedges.

## 3. Hyper Parameters of LTD

