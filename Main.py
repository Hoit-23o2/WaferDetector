import cv2
import os
import json

from WaferCuttingLineDetector import WaferCuttingLineDetector


def DrawLine(image, line, color):
    cv2.line(image,
             (int(line[0]), int(line[1])),
             (int(line[2]), int(line[3])),
             color, 2)


def DrawCombination(imageName, image, combinationInfo):
    for cuttingPath in combinationInfo["cutting_path_list"]:
        DrawLine(image, cuttingPath["cutting_path_up"], (0, 0, 255))
        DrawLine(image, cuttingPath["cutting_path_bottom"], (255, 0, 0))
        DrawLine(image, cuttingPath["cutting_line"], (0, 255, 0))
    print("==========================image: {} ========================\ncombination: {}\n".format(imageName,
                                                                                                   combinationInfo))


def ShowImage(imageName, image):
    cv2.imshow(imageName, image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def OutputResult(imageName, outputDir, image, combinationInfo):
    if os.path.isdir(outputDir) is False:
        os.makedirs(outputDir, exist_ok=True)

    imageNameNP = os.path.splitext(imageName)[0]
    suffix = os.path.splitext(imageName)[-1]

    outputImageName = "{}_result{}".format(imageNameNP, suffix)
    outputImagePath = os.path.abspath(os.path.join(outputDir, outputImageName))
    cv2.imwrite(outputImagePath, image)

    outputCombinationName = "{}_cutting_path.json".format(imageNameNP)
    outputCombinationPath = os.path.abspath(os.path.join(outputDir, outputCombinationName))
    jsonStr = json.dumps(combinationInfo, indent=4)
    with open(outputCombinationPath, 'w') as f:
        f.write(jsonStr)


if __name__ == "__main__":
    detector = WaferCuttingLineDetector()
    imageDir = "./dataset"
    outputDir = "./result"
    imageNameList = os.listdir(imageDir)
    for imageName in imageNameList:
        imagePath = os.path.abspath(os.path.join(imageDir, imageName))
        image = cv2.imread(imagePath)

        combinationInfo = detector.DetectWaferCuttingLine(image)

        DrawCombination(imageName, image, combinationInfo)

        # 可以选择是否可视化
        ShowImage(imageName, image)

        OutputResult(imageName, outputDir, image, combinationInfo)
