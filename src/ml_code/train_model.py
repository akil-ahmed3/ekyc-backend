from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC


in_encoder = Normalizer()
out_encoder = LabelEncoder()

model = SVC(kernel='linear', probability=True)


def train_model(emdTrainX, trainy):

    emdTrainX_norm = in_encoder.transform(emdTrainX)

    out_encoder.fit(trainy)
    trainy_enc = out_encoder.transform(trainy)

    model.fit(emdTrainX_norm, trainy_enc)


def test(emdTestX, trainy):
    emdTestX_norm = in_encoder.transform(emdTestX)

    yhat_class = model.predict(emdTestX_norm)
    # score_train = accuracy_score(trainy_enc, yhat_train)
    predict_names = out_encoder.inverse_transform(yhat_class)
    # print('Accuracy: train=%.3f' % (score_train*100))
    return predict_names
