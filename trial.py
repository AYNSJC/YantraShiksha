import Tanitra
import Parata
import cv2
import Pratirup
import os
import cupy as cp

id = input(int("Enter User ID (any number):"))
dataset_path = "C:/Users/rohit/OneDrive/Documents/GitHub/YantraShiksha/faces_dataset/"
os.makedirs(dataset_path, exist_ok=True)

labels_file_path = "C:/Users/rohit/OneDrive/Documents/GitHub/YantraShiksha/labels.txt"
print("Look at the camera and move a bit here and there... Collecting data...")

class FaceDataset:
    def __init__(self, img_dir, labels_file):
        self.img_dir = img_dir
        self.labels = self.load_labels(labels_file)

    def load_labels(self, labels_file):
        with open(labels_file, "r") as f:
            lines = f.readlines()
        labels = []
        for line in lines:
            parts = line.strip().split()
            img_name = parts[0]
            bbox = list(map(float, parts[1:5]))  # x, y, w, h
            labels.append((img_name, bbox))
        return labels

    def load_image_and_bbox(self, img_name, bbox):
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        img = Tanitra.Tanitra(img / 255.0)  # Normalize
        return img, Tanitra.Tanitra(bbox)

    def __getitem__(self, idx):
        img_name, bbox = self.labels[idx]
        img, bbox = self.load_image_and_bbox(img_name, bbox)
        return img, bbox

    def __len__(self):
        return len(self.labels)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

img_count = 0
labels = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        img_name = f"face_{img_count}.jpg"
        img_path = os.path.join(dataset_path, img_name)

        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (64, 64))
        cv2.imwrite(img_path, face_img)

        labels.append(f"{img_name} {x} {y} {w} {h}")

        img_count += 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q") or img_count >= 1:
        break

cap.release()
cv2.destroyAllWindows()


print("Face data collected! Now training model...")


with open(labels_file_path, "w") as f:
    for label in labels:
        f.write(label + "\n")


class FaceDetector(Pratirup.AnukramikPratirup):
    def __init__(self):
        super().__init__()
        self.add(Parata.ConvLayer2D(stride=1, filters=32, channels=3, kernel_size=3, activation="relu"))
        self.add(Parata.ConvLayer2D(stride=1, filters=64, channels=32, kernel_size=3, activation="relu"))
        self.add(Parata.MaxPoolingLayer2D(stride=2, pool_window=2,channels=64))
        self.add(Parata.GuptaParata(n_neurons=128, activation="relu"))
        self.add(Parata.NirgamParata(n_neurons=4, activation="linear"))  # x, y, w, h

dataset = FaceDataset(dataset_path, labels_file_path)
X_train, y_train = [], []

for i in range(len(dataset)):
    img, bbox = dataset[i]
    image = img.data
    image = cp.resize(image,(3,64,64))
    X_train.append(image)
    y_train.append(bbox.data)


X_train = Tanitra.Tanitra(X_train)
y_train = Tanitra.Tanitra(y_train)

model = FaceDetector()
model.learn(X_train, y_train, optimizer="Gradient Descent", epochs=1000, lr=0.001)

print("Model trained successfully! Now running the recognition code.")

cap = cv2.VideoCapture(0)
print("Scanning for faces...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (64, 64))
    img = Tanitra.Tanitra(img / 255.0)

    bbox = model.estimate(img).detach().numpy()
    x, y, w, h = bbox[0]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)
    cv2.putText(frame, 'recognised id {id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,255) , 2)
    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
