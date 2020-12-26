# from import preprocess_data
from ..ml_code import preprocess_data
from tensorflow.keras.models import load_model
import numpy as np


def embedded_data(trainX, testX):

    model = load_model('./facenet_keras.h5')

    emdTrainX = list()
    emdTestX = list()

    def get_embedding(model, face):
        face = face.astype('float32')

        mean, std = face.mean(), face.std()
        face = (face-mean)/std

        sample = np.expand_dims(face, axis=0)
        yhat = model.predict(sample)
        return yhat[0]

    for face in trainX:
        emd = get_embedding(model, face)
        emdTrainX.append(emd)

    for face in testX:
        emd = get_embedding(model, face)
        emdTestX.append(emd)

    return emdTrainX, emdTestX
