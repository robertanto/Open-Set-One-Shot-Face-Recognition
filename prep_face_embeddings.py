from face_recognition.embed import embedd
from face_recognition.video import ReadImage
from face_recognition.face_detector import FaceDetector
from face_recognition.keras_vggface import get_senet
from keras_vggface import utils

import keras.backend as K
import os
import pickle
import cv2
import numpy as np

class SENetPreprocess:
    def __init__(self):
        self.detector = FaceDetector('weights')
        self.conf_threshold = 0.7
        self.padding = 20

    def __call__(self, frame):
        '''
            frame : opencv image (BGR)
        '''

        face_bbox = self.detector.get_bigger_face(
            frame, conf_threshold=self.conf_threshold)

        if face_bbox is None:
            return face_bbox

        face = frame[max(0, face_bbox[1]-self.padding):min(face_bbox[3]+self.padding, frame.shape[0]-1),
                     max(0, face_bbox[0]-self.padding):min(face_bbox[2]+self.padding, frame.shape[1]-1)]

        # RESHAPE
        face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_CUBIC)

        # BGR 2 RGB
        face_rgb = np.expand_dims(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), 0)

        # img - mean
        face_rgb = utils.preprocess_input(face_rgb.astype(np.float32))

        return face_rgb

if __name__ == "__main__":
    ids_folder = 'face_identities'
    out_path = 'embed.pk'

    read_fun = ReadImage(rgb=False)

    X = []
    y = []
    for root, dirs, files in os.walk("./face_identities"):
        for filename in files:
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                X.append(read_fun(os.path.join('face_identities',filename)))
                y.append(os.path.splitext(filename)[0])
    


    model = get_senet('weights/rcmalli_vggface_tf_senet50.h5')

    preprocess = SENetPreprocess()

    out_dict = embedd(X, y, model, preprocess_input=preprocess)

    with open(out_path, 'wb') as file:
        pickle.dump(out_dict, file)
