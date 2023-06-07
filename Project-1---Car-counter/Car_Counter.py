import cv2
import numpy
from ultralytics import YOLO as yolo
import cv2 as cv
import cvzone
import math
from sort import *

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

mask = cv.imread("detection_area (Custom).png")

# Object tracking setup
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

while True:     # This while function loops through all the frames of the input video and places labeled bounding box around objects of interest
    success, img = cap.read()
    imgRegion = cv.bitwise_and(img, mask)  # Combining the mask and the video
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))   # Array of detections with attributes x1, y1, x2, y2 and the tracking ID

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

            # Selecting which object to detect (Object of interest for detection)
            if objectClass == 'car' or objectClass == 'bus' or objectClass == 'motorbike' or objectClass == 'truck'\
                and conf > 0.3:
                cvzone.putTextRect(img,
                                   f'{objectClass} {conf}',
                                   (max(0, x1), max(35, y1)),
                                   scale=1, thickness=1,
                                   offset=3)   # This line prints the label with its confidence on the corner rectangle
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=10, t=2)  # Drawing the bounding box
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = numpy.vstack((detections, currentArray))  # Appending arrays of detections with numpy

    resultTracker = tracker.update(detections)
    for result in resultTracker:
        x1, y1, x2, y2, Id = result
        print(result)

    cv.imshow("Image", img)
    # cv.imshow("ImageRegion", imgRegion)
    cv2.waitKey(0)
