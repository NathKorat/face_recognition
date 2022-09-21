import cv2
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
import qimage2ndarray # for a memory leak,see gist
import sys
import pandas as pd
from datetime import datetime
import csv

face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')


# to record the data to a csv file name "tracker.csv"
def track(name, confidence): 
    with open('tracker/tracker.csv', 'a', encoding= 'UTF8') as f:
        track_rec = csv.writer(f)
        track_rec.writerow([name, datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%H:%M'), confidence])

        f.close()
# to verify the data before record to csv
def check_csv(filename) : 
    df = pd.read_csv(filename)
    
    today = datetime.now().strftime('%Y-%m-%d')
    today_df = df[df['Date'] == today]

    name = list(today_df['Name'].unique())

    return name


font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
data_user = pd.read_csv('tracker/user_data.csv')



def display():
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 2,
        minSize = (int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x - 5, y - 5), (x+w + 5, y+h + 5), (255,0,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        if (confidence < 50):
            id = list(data_user[data_user['face_id'] == id]['user_name'])[0]
            confidence = "{0}%".format(round(100 - confidence))

            check_name = check_csv('tracker/tracker.csv')

            #report user to csv file format
            if id not in check_name: 
                track(id, confidence)

        else:
            id = data_user.iloc[0, 1]
            confidence = ""

        cv2.putText(frame, (str(id) + " " + str(confidence)), (x, y-10), font, 1, (255,255,255), 2)
        print('>> {}: {}'.format(id, confidence))

    image = qimage2ndarray.array2qimage(frame)
    label.setPixmap(QPixmap.fromImage(image))

app = QApplication(sys.argv)
window = QWidget()
window.resize(1050, 780)


cam = cv2.VideoCapture(0)
cam.set(3, 1020)
cam.set(4, 1080)

minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

timer = QTimer()

timer.timeout.connect(display)
timer.start(60)

label = QLabel("No camera")

layout = QVBoxLayout()
button = QPushButton("Exit")
button.clicked.connect(sys.exit)
layout.addWidget(label)
layout.addWidget(button)
layout.setAlignment(Qt.AlignTop | Qt.AlignCenter)


# window.setWindowState(Qt.WindowMaximized)
window.setLayout(layout)
window.show()
sys.exit(app.exec_())
