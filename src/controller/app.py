import os
import PIL
import shutil
import numpy as np
from PIL import Image, ImageFile
from flask_cors import CORS
from flask import Flask, jsonify, request, render_template

from ..ml_code import preprocess_data
from ..ml_code import embedding_data
from ..ml_code import train_model
from ..ml_code import extract_frames

app = Flask(__name__)
CORS(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/upload_video", methods=["POST"])
def upload_video():
    name = request.form.get("name")
    email = request.form.get("email")
    video = request.files.get("video")
    document = request.files.get("pan")

    video_name = str(name) + ".webm"

    video.save(video_name)

    path = os.path.join("./frames/train/", name)
    os.mkdir(path)

    extract_frames.extract_frames(name)

    trainX, trainy = preprocess_data.load_dataset("./frames/train/")
    testX = np.asarray(preprocess_data.load_faces("./frames/test/"))

    emdTrainX, emdTestX = embedding_data.embedded_data(trainX, testX)

    train_model.train_model(emdTrainX, trainy)

    prediction = train_model.test(emdTestX, trainy)

    shutil.rmtree(path, ignore_errors=True)

    os.remove(video_name)

    print(prediction)

    return "True"


@app.route("/test_face", methods=["POST"])
def test_face():
    name = request.form.get("name")
    email = request.form.get("email")
    picture = request.files.get("picture")
    picture_name = 'img.jpg'

    path = os.path.join("./frames/test/", picture_name)

    picture.save("./frames/test/img.jpg")

    testX = np.asarray(preprocess_data.load_faces("./frames/test/"))
    trainX, trainy = preprocess_data.load_dataset("./frames/train/")

    emdTrainX, emdTestX = embedding_data.embedded_data(trainX, testX)

    prediction = train_model.test(emdTestX, trainy)

    print(prediction)
    os.remove(path)
    return "True"
