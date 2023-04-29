import cv2
import os

from WaferCuttingLineDetector import WaferCuttingLineDetector

if __name__ == "__main__":
    detector = WaferCuttingLineDetector()
    imageDir = "./dataset"
    imageNameList = os.listdir(imageDir)
    for imageName in imageNameList:
        imagePath = os.path.abspath(os.path.join(imageDir, imageName))
        image = cv2.imread(imagePath)

        result = detector.DetectWaferCuttingLine(image)
