import numpy as np
import cv2 as cv
import argparse

ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to input image")
# ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.6,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.2,
                help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())


class YoloDetect:

    def __init__(self, pic_path):
        self.configurationPath = 'Yolo/yolo/yolo-fastest.cfg'
        self.weightsPath = 'Yolo/yolo/yolo-fastest_last.weights'
        self.classesPath = 'Yolo/yolo/obj.names'
        self.size = (320, 320)
        self.inputImage = cv.imread(pic_path, cv.IMREAD_COLOR)
        cv.imshow("Input_Image", self.inputImage)
        cv.waitKey(2000)

    def TargetResult(self):
        net = cv.dnn.readNetFromDarknet(self.configurationPath, self.weightsPath)
        labels = open(self.classesPath).read().strip().split("\n")

        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        blob = self.ReshapeInputPicture(self.inputImage)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        return self.PostProcess(layerOutputs, self.inputImage, labels)

    def ReshapeInputPicture(self, input_pic):
        reshape_pic = cv.resize(input_pic, self.size)
        scale = 1.0 / 255.0
        blob = cv.dnn.blobFromImage(reshape_pic, scale, self.size, swapRB=True, crop=False)
        return blob

    def PostProcess(self, layerOutputs, image, labels):
        boxes = []
        confidences = []
        classIDs = []
        (H, W) = image.shape[:2]
        np.random.seed(42)
        # color= np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > args["confidence"]:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(1)

        idxs = cv.dnn.NMSBoxes(boxes,
                               confidences,
                               args["confidence"],
                               args["threshold"])

        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                cv.rectangle(image, (x, y), (x + w, y + h), 1, 2)
                text = ""
                cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX,
                           0.5, 1, 2)
                result = (classIDs[i], confidences[i])

        # show the output image
        print(result)
        cv.imshow("Image", image)
        cv.waitKey(2000)
        return result


def main():
    pic_path = 'sheep.jpg'
    td = YoloDetect(pic_path)

    result = td.TargetResult()
    # [(int, float)]

    window_name = 'OpenCV Test'
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    ##cv.imshow(window_name, src)
    ##cv.waitKey(300000)


if __name__ == '__main__':
    main()
