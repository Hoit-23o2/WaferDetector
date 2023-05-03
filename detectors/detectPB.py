import cv2
import os
import json
import numpy

from matplotlib import pyplot

from PointBasedDetector import WaferCuttingLineDetector


def DrawLine(image, line, color):
    cv2.line(image,
             (int(line[0]), int(line[1])),
             (int(line[2]), int(line[3])),
             color, 2, lineType=cv2.LINE_AA)


def MidOutputToPyplot(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        imageNew = numpy.zeros((image.shape[0], image.shape[1], 3), dtype=image.dtype)
        imageNew[:, :, 0] = image
        imageNew[:, :, 1] = image
        imageNew[:, :, 2] = image
        return imageNew


def MakeMidOutput(imageName, midOutput, showWindow, saveResult=False, outputDir=""):
    midOutputKeys = ["OriImage",
                     "PossibleCleanArea",
                     "DetectEdge",
                     "ThroughLinesCandidate",
                     "HoughLineP",
                     "FilterOutLineResult"]

    fig, ax = pyplot.subplots(1, len(midOutputKeys), figsize=(16, 12), dpi=300)
    for kIndex, key in enumerate(midOutputKeys):
        midOutputImage = MidOutputToPyplot(midOutput[key])
        ax[kIndex].imshow(midOutputImage)
        ax[kIndex].set_title(key)

    if showWindow:
        pyplot.show()

    if saveResult:
        if os.path.isdir(outputDir) is False:
            os.makedirs(outputDir, exist_ok=True)
        imageNameNP = os.path.splitext(imageName)[0]
        outputName = "{}_mid_output.png".format(imageNameNP)
        outputPath = os.path.abspath(os.path.join(outputDir, outputName))
        pyplot.savefig(outputPath, bbox_inches='tight')


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


def MakeResult(imageName, outputDir, image, combinationInfo, showResult, saveResult):
    DrawCombination(imageName, image, combinationInfo)
    # 可以选择是否可视化
    if showResult:
        ShowImage(imageName, image)
    if saveResult:
        OutputResult(imageName, outputDir, image, combinationInfo)


if __name__ == "__main__":
    detector = WaferCuttingLineDetector()
    # dataset
    imageDir = "../dataset"
    # mid output
    # 如果要可视化中间结果，速度会变慢
    saveMidOutput = False
    detector.SetMakeMidOutput(saveMidOutput)
    midOutputDir = "./pb-mid-output"
    # final result
    showResult = True
    saveResult = True
    ResultDir = "./pb-final-output"

    imageNameList = os.listdir(imageDir)
    for imageName in imageNameList:
        imagePath = os.path.join(imageDir, imageName)
        print(imagePath)
        image = cv2.imread(imagePath)

        combinationInfo = detector.DetectWaferCuttingLine(image)

        if combinationInfo is None:
            print("The combinationInfo is None. The detector did not detect any cutting lines.")
            input("continue...")
        else:
            if detector.makeMidOutput:
                midOutput = detector.GetMidOutput()
                midOutput["OriImage"] = image
                MakeMidOutput(imageName, midOutput, False, saveMidOutput, midOutputDir)
            MakeResult(imageName, ResultDir, image, combinationInfo, showResult, saveResult)

