import cv2
import numpy as np
import face_recognition as fr
import os

# Load images and names from the directory
path = r"C:\Users\demon\Downloads\face_recognition\face_recognition\images"
images = []
names = []
myList = os.listdir(path)
for imgName in myList:
    curImg = cv2.imread(f"{path}/{imgName}")
    images.append(curImg)
    names.append(os.path.splitext(imgName)[0])

# Find encodings for the known faces
def findencodings(images):
    encodedList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodedList.append(encode)
    return encodedList

encodedListKnown = findencodings(images)

# Initialize the webcam capture
cap = cv2.VideoCapture(0)
while True:
    _, webcam = cap.read()
    imgResized = cv2.resize(webcam, (0, 0), None, 0.25, 0.25)
    imgResized = cv2.cvtColor(imgResized, cv2.COLOR_BGR2RGB)

    # Find faces in the current frame
    faceCurFrame = fr.face_locations(imgResized)
    encodeFaceCurFrame = fr.face_encodings(imgResized, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeFaceCurFrame, faceCurFrame):
        # Compare the face to the known faces
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
        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(webcam, (x1, y1), (x2, y2), (0, 0, 255), 2)
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