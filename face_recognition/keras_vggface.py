import cv2
from keras_vggface.vggface import VGGFace
from keras.models import Model
from keras.layers import Lambda
import keras.backend as K


def get_senet(weights_path):
    model = VGGFace(model='senet50', include_top=True, weights=None)
    model.load_weights(weights_path)

    # L2 Normalization for the cosine distance
    x = Lambda(lambda x: K.l2_normalize(x, axis=1))(
        model.layers[-2].output)

    model = Model(model.input, x)

    return model
