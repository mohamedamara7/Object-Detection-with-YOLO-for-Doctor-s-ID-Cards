import os

from ultralytics import YOLO
import cv2


IMAGES_DIR = 'images/'

image_path = os.path.join(IMAGES_DIR, 'true_dentist.jpg')
image_path_out = '{}_out.jpg'.format(image_path)

frame = cv2.imread(image_path)

model_path = 'model2/last.pt'

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

results = model(frame)[0]

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        slice = frame[int(x1):int(x2), int(y1):int(y2)]
        slice = 5

#cv2.imshow('Window Name', frame)
cv2.imshow('Window Name', slice)
cv2.waitKey(0)  # Use 0 to wait indefinitely until a key is pressed
