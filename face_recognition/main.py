from typing import List, Any

import cv2
import numpy as np
import face_recognition as fr
import os
from DateTime import DateTime
from datetime import datetime
import pickle
import pandas


path = r"C:\Users\demon\Downloads\face_recognition\face_recognition\images"
images = []
names = []
myList = os.listdir(path)
print(myList)

for imgNames in myList:
    curImg = cv2.imread(f"{path}/{imgNames}")
    images.append(curImg)
    names.append(os.path.splitext(imgNames)[0])


def findencodings(images):
    encodedlist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodedlist.append(encode)
    return encodedlist


encodedListKnown = findencodings(images)

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines(f'n{name}, {time}, {date}')


cap = cv2.VideoCapture(0)
while True:
    _, webcam = cap.read()

    imgResized = cv2.resize(webcam, (0, 0), None, 0.25, 0.25)
    imgResized = cv2.cvtColor(imgResized, cv2.COLOR_BGR2RGB)

    faceCurFrame = fr.face_locations(imgResized)
    encodeFaceCurFrame = fr.face_encodings(imgResized, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeFaceCurFrame, faceCurFrame):
        matches = fr.compare_faces(encodedListKnown, encodeFace)
        faceDis = fr.face_distance(encodedListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)


        if matches[matchIndex]:
            name = names[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(webcam, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(webcam, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(
                webcam,
                name,
                (x1 + 6, y2 - 6),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                1,
            )

            face_match_percentage = (1 - faceDis) * 100

            for i, face_distance in enumerate(faceDis):
                print(
                    "The test image has a distance of {:.2} from known image {} ".format(
                        face_distance, i
                    )
                )
                print(
                    "- comparing with a tolerance of 0.6 {}".format(face_distance < 0.6)
                )
                print(
                    "Face Match Percentage = ", np.round(face_match_percentage, 4)
                )
        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(webcam, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw red rectangle for unrecognized faces
            cv2.rectangle(webcam, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(
                webcam,
                'UNKNOWN',
                (x1 + 6, y2 - 6),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                1,
            )





    cv2.imshow("webcam", webcam)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
