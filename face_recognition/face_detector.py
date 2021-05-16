import cv2 as cv
import numpy as np
import os

class FaceDetector:

    def __init__(self, weights_folder='.', faceProto="opencv_face_detector.pbtxt", faceModel="opencv_face_detector_uint8.pb"):
        super().__init__()

        faceProto = os.path.join(weights_folder, faceProto)
        faceModel = os.path.join(weights_folder, faceModel)

        # Load network
        self.faceNet = cv.dnn.readNet(faceModel, faceProto)

    def get_bigger_face(self, frame, conf_threshold=0.7):
        bboxes = self.get_faces(frame, conf_threshold)

        if bboxes is None:
            return None
        else:
            bboxes.sort(key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True)
            return bboxes[0]

    def get_faces(self, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [
                                    104, 117, 123], True, False)

        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
        if len(bboxes) == 0:
            return None

        return bboxes