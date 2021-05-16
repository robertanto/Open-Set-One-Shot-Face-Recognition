import numpy as np
import cv2 as cv
import pickle

from keras_vggface import utils

from face_recognition.video import OpenCVStream
from face_recognition.keras_vggface import get_senet
from face_recognition.face_detector import FaceDetector
from face_recognition.utils import batch_cosine_similarity, dist2id


def process(face):
    '''
        frame : opencv image (BGR)
    '''
    # RESHAPE
    face = cv.resize(face, (224, 224), interpolation=cv.INTER_CUBIC)

    # BGR 2 RGB
    face_rgb = np.expand_dims(cv.cvtColor(face, cv.COLOR_BGR2RGB), 0)

    # img - mean
    face_rgb = utils.preprocess_input(face_rgb.astype(np.float32))

    return face_rgb


if __name__ == "__main__":
    video_stream = OpenCVStream()

    th = 0.55

    # Load models and embeddings
    detector = FaceDetector('weights')
    model = get_senet('weights/rcmalli_vggface_tf_senet50.h5')

    with open('embed.pk', 'rb') as file:
        emb_dict = pickle.load(file)

    n_embs = len(emb_dict['y'])
    X = emb_dict['embeddings']
    y = emb_dict['y']

    # Open a video file or an image file or a camera stream
    cap = cv.VideoCapture(0)

    padding = 20
    while True:
        frame = video_stream.get_color_frame()

        # DETECTION
        face_bboxs = detector.get_faces(frame, conf_threshold=0.7)

        if face_bboxs is None:
            cv.imshow("Image", frame.astype(np.uint8))
            cv.waitKey(10)
            continue
        
        frameFace = frame.copy()
        for face_bbox in face_bboxs:

            face = frame[max(0, face_bbox[1]-padding):min(face_bbox[3]+padding, frame.shape[0]-1),
                        max(0, face_bbox[0]-padding):min(face_bbox[2]+padding, frame.shape[1]-1)]

            # PREPROCESSING & PREDICTION
            prep_face = process(face)
            emb_face = model.predict(prep_face)

            # IDENTIFICATION
            emb_face = np.repeat(emb_face, n_embs, 0)
            cos_dist = batch_cosine_similarity(X, emb_face)
            id_label = dist2id(cos_dist,y, th)
            if id_label is None:
                id_label = "?"

            # PRINT ON THE IMAGE
            cv.rectangle(frameFace, (face_bbox[0], face_bbox[1]), (
                face_bbox[2], face_bbox[3]), (255, 0, 0), int(round(frame.shape[0]/150)), 8)

            label = "{}".format(id_label)
            cv.putText(frameFace, label, (face_bbox[0], face_bbox[1]-10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)

        # SHOW IMAGE
        cv.imshow("Image", frameFace.astype(np.uint8))

        # Wait command
        key = cv.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv.destroyAllWindows()
            break

    video_stream.stop()
