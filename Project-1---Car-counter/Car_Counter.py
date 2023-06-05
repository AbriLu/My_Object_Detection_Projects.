import cv2
from ultralytics import YOLO as yolo
import cv2 as cv
import cvzone
import math

cap = cv.VideoCapture("../videos/cars.mp4") # Selecting the video input method in this case a video from our hard drive

model = yolo("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:  # looping through images
        boxes = r.boxes
        for box in boxes:  # Looping through bounding boxes in the image 'r'
            # Using the CV2 package to draw bounding boxes
            # x1, y1, x2, y2 = box.xyxy[0]
            # x1, y1, x2, y2 = int(x1),int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Using the CVZONE package to draw bounding boxes
            x1, y1, x2, y2 = box.xyxy[0]  # The index '0' is meant to extract the data from the box.xyxy tensor object
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bbox = x1, y1, x2, y2
            print(x1, y1, x2-x1, y2-y1)
            # cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=10, t=2)

            # Displaying the object class and confidence value using CVZONE
            conf = math.ceil(box.conf*100)/100
            print(f"Confidence = {conf}")
            class_index = box.cls
            objectClass = classNames[int(class_index[0])]
            print(f"Class Name = {objectClass}")

            # Selecting which object to detect
            if objectClass == 'car' or objectClass == 'bus' or objectClass == 'motorbike' or objectClass == 'truck'\
                and conf > 0.3:
                cvzone.putTextRect(img,
                                   f'{objectClass} {conf}',
                                   (max(0, x1), max(35, y1)),
                                   scale=1, thickness=1,
                                   offset=3)   # This line prints the label with its confidence on the corner rectangle
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=10, t=2)  # Drawing the bounding box

    cv.imshow("Image", img)
    cv2.waitKey(1)

