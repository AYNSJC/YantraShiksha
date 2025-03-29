import cv2
import numpy as np
import os

if not os.path.exists("faces"):
    os.makedirs("faces")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
face_id = input("Enter User ID (any number): ")

print("Look at the camera and move a bit here and there... Collecting data...")

count = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y + h, x:x + w]
        cv2.imwrite(f"faces/User.{face_id}.{count}.jpg", face)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Capture", frame)

    if count >= 1000:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Face data collected! Now training model...")

recognizer = cv2.face.LBPHFaceRecognizer_create()
faces = []
labels = []
file_list = os.listdir("faces")

for file in file_list:
    img = cv2.imread(f"faces/{file}", cv2.IMREAD_GRAYSCALE)
    faces.append(np.array(img, dtype="uint8"))
    labels.append(int(file.split(".")[1]))

recognizer.train(faces, np.array(labels))
recognizer.save("face_model.xml")
print("Model trained successfully! Now running the recognition code.")




recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.xml")

cap = cv2.VideoCapture(0)
print("Scanning for faces...")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face)

        if confidence < 50:
            name = f"Recognized (ID: {label})"
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()