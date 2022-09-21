import cv2
import os
import csv
import pandas as pd

#face_training.py using it to train new faces.
import face_training 

#sample of face detection of haar cascade classicfier
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# Number in paramenter is the location of the camera, it varies among devices
cam = cv2.VideoCapture(0) 
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# for each person, enter a numeric face id and name as text
face_id = input('\nEnter user id: ')
user_name = input('\nEnter user name: ')

#record user ID and their name before capture their face
pre_id = pd.read_csv('tracker/user_data.csv')
pre_id = [i for i in pre_id.face_id.astype('string')]

if face_id not in pre_id:
    with open('tracker/user_data.csv', 'a', encoding='utf8') as f:
        users = csv.writer(f)
        users.writerow([face_id, user_name])

        f.close()


print('\n[INFO] Initializing face capture....')

# Sample individual face count 
count = 0

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 4,
        minSize = (int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x - 5, y - 5), (x+w + 5, y+h + 5), (255,0,0), 2)
        count += 1

        cv2.imshow('Capturing', gray)
        # Save captured image into the dataset folder
        cv2.imwrite('dataset/User.' + str(face_id) + '.' + str(count) + '.jpg', gray[y: y+h, x: x+w])
        print('{} captured...'.format(count))

    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break
    elif count >= 100:
        break

print('\n [INFO] {} faces of user id "{}" have recorded...'.format(count, face_id))
face_training.train()
cam.release()
cv2.destroyAllWindows()