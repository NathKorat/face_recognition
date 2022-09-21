import cv2
import os 
import csv
from datetime import datetime
import pandas as pd

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascades/haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


font = cv2.FONT_HERSHEY_SIMPLEX

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

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
data_user = pd.read_csv('tracker/user_data.csv')



# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)

cam.set(3, 640) # set video widht 640
cam.set(4, 480) # set video height 480e

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img = cam.read()


    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 3,
        # minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x - 5,y -5), (x+w+5,y+h+5), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less than 56 ==> "0" is perfect match 
        if (confidence < 50): # 100 checks all faces as it has id in dataset and no one are unknown
            #get user name by user_id 
            id = list(data_user[data_user['face_id'] == id]['user_name'])[0]
            confidence = "{0}%".format(round(100 - confidence))

            check_name = check_csv('tracker/tracker.csv')

            #report user to csv file format
            if id not in check_name: 
                track(id, confidence)
        else:
            id = data_user.iloc[0, 1]
            confidence = ''
        
        cv2.putText(img, (str(id) + ' '+ str(confidence)), (x, y-10), font, 1, (255,255,255), 2)
        
        #print text to terminal
        print('>> {}: {}'.format(id, confidence))
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(60) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
# print("\n [INFO] Exiting Program and cleanup stuff")
print(check_csv('tracker/tracker.csv'))
cam.release()
cv2.destroyAllWindows()