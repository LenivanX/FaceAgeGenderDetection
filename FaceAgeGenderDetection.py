import cv2
import numpy as np
from tkinter import filedialog

# load age and gender model
age_model = cv2.dnn.readNetFromCaffe('models/age.prototxt', 'models/dex_chalearn_iccv2015.caffemodel')
gender_model = cv2.dnn.readNetFromCaffe('models/gender.prototxt', 'models/gender.caffemodel')

# load image
img_path = filedialog.askopenfilename()
img_color = cv2.imread(img_path)

# load classifier
cascade_face = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

# detect faces
# change values of scaleFactor and minNeighbors for varying accuracy
# scaleFactor > 1, minNeighbors must be int
faces = cascade_face.detectMultiScale(img_color, scaleFactor=1.2, minNeighbors=5)

# operate for each face
for (x, y, w, h) in faces:
    # detected face
    detected_face = img_color[int(y):int(y + h), int(x):int(x + w)]

    # resize
    detected_face = cv2.resize(detected_face, (224, 224))
    detected_face_blob = cv2.dnn.blobFromImage(detected_face)

    # detect age
    age_model.setInput(detected_face_blob)
    age_result = age_model.forward()
    indices = np.array(range(101))
    age = int(np.sum(age_result * indices))

    # detect gender
    gender_model.setInput(detected_face_blob)
    gender_result = gender_model.forward()
    gender = np.argmax(gender_result)
    if gender:
        gender = 'M'
    else:
        gender = 'F'

    # label for face
    label = str(age) + ',' + gender

    # label on image
    label_box = np.array([[x, y], [x + 70, y], [x + 70, y - 30], [x, y - 30]], np.int32)
    cv2.rectangle(img_color, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.fillPoly(img_color, [label_box], (255, 0, 0))
    cv2.putText(img_color, label, (x + 5, y - 10), 1, 1.5, (255, 255, 255), 1)

# show image
cv2.imshow('Result', img_color)
cv2.waitKey()
cv2.destroyAllWindows()
