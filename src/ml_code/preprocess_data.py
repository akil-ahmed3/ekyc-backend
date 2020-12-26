from os import listdir
from mtcnn import MTCNN
from PIL import Image, ImageFile
import numpy as np
import os
import cv2 as cv


def extract_face(filename, required_size=(160, 160)):
    print(filename)
    image = cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB)
    pixels = np.asarray(image)
    print(pixels)

    detector = MTCNN()
    results = detector.detect_faces(pixels)

    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    face = pixels[y1:y2, x1:x2]

    image = Image.fromarray(face)
    image = image.resize((160, 160))
    face_array = np.asarray(image)

    print(face_array)

    return face_array


def load_faces(directory):
    faces = list()
    # enumerate files
    for filename in listdir(directory):
        # path
        path = directory + filename
        # get face
        face = extract_face(path)
        # store
        faces.append(face)
    return faces


def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not os.path.isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)


# trainX, trainy = load_dataset("./frames/train/")

# print(trainX)
# testX = np.asarray(load_faces("./frames/test/"))
