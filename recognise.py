import cv2
import numpy as np
import xlwrite
import os
import time

base_path = os.path.abspath(os.path.dirname(__file__))
cascade_path = os.path.join(base_path, 'haarcascade_frontalface_default.xml')
trainer_path = os.path.join(base_path, 'trainer', 'trainer.yml')
attendance_file = 'attendance'

face_cascade = cv2.CascadeClassifier(cascade_path)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_path)

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
start_time = time.time()
period = 8
attendance_dict = {}
row_counter = 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        id, conf = recognizer.predict(roi_gray)
        if conf < 50:
            name = f'User {id}'
            if name not in attendance_dict:
                xlwrite.output(attendance_file, 'class1', row_counter, id, name, 'Present')
                attendance_dict[name] = True
                row_counter += 1
        else:
            name = 'Unknown'

        cv2.putText(frame, f"{name} {conf:.2f}", (x, y - 10), font, 0.75, (255, 255, 255), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or time.time() > start_time + period:
        break

cap.release()
cv2.destroyAllWindows()
print("Attendance process completed")
