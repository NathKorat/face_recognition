import os
import cv2

cam = cv2.VideoCapture(0)

if cam.isOpened() == False:
    print('Cannot Open Camera!')

f_w = int(cam.get(3))
f_h = int(cam.get(4))

out = cv2.VideoWriter('video2.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (f_w, f_h))

while True:
    ret, frame = cam.read()

    if ret == True:
        out.write(frame)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
out.release()
cv2.destroyAllWindows()