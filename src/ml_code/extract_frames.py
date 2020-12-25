import cv2


def extract_frames(name):
    vidcap = cv2.VideoCapture(str(name) + ".webm")
    success, image = vidcap.read()
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        cv2.imwrite("./frames/train/" + name + "/frame%d.jpg" %
                    count, image)
        if cv2.waitKey(10) == 27 or count == 25:
            break
        count += 1
