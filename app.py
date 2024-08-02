from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from PIL import Image
import xlwrite

app = Flask(__name__)

# Paths
base_path = os.path.abspath(os.path.dirname(__file__))
cascade_path = os.path.join(base_path, 'haarcascade_frontalface_default.xml')
attendance_file = 'attendance'

# Initialize face recognizer and face detector
face_cascade = cv2.CascadeClassifier(cascade_path)
recognizer = cv2.face.LBPHFaceRecognizer_create()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    if request.method == 'POST':
        user_id = request.form['user_id']
        count = 0
        cap = cv2.VideoCapture(0)
        os.makedirs('dataset', exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                count += 1
                cv2.imwrite(f'dataset/user.{user_id}.{count}.jpg', gray[y:y+h, x:x+w])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {user_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Captured {count} images for user ID: {user_id}")
        return redirect(url_for('index'))

    return render_template('capture.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        def getImagesAndLabels(path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.png'))]
            faceSamples = []
            Ids = []

            for imagePath in imagePaths:
                pilImage = Image.open(imagePath).convert('L')
                imageNp = np.array(pilImage, 'uint8')
                Id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = face_cascade.detectMultiScale(imageNp)

                for (x, y, w, h) in faces:
                    faceSamples.append(imageNp[y:y+h, x:x+w])
                    Ids.append(Id)

            return faceSamples, Ids

        faces, Ids = getImagesAndLabels('dataset')
        recognizer.train(faces, np.array(Ids))

        os.makedirs('trainer', exist_ok=True)
        recognizer.write('trainer/trainer.yml')

        print("Model trained and saved as trainer/trainer.yml")
        return redirect(url_for('index'))

    return render_template('train.html')

@app.route('/recognise', methods=['GET', 'POST'])
def recognise():
    if request.method == 'POST':
        recognizer.read(os.path.join(base_path, 'trainer', 'trainer.yml'))
        cap = cv2.VideoCapture(0)
        attendance_dict = {}
        row_counter = 1
        result_message = "Attendance process completed."

        while True:
            ret, frame = cap.read()
            if not ret:
                result_message = "Failed to capture image"
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

                cv2.putText(frame, f"{name} {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(result_message)
        return redirect(url_for('index'))

    return render_template('recognise.html')

if __name__ == '__main__':
    app.run(debug=True)
